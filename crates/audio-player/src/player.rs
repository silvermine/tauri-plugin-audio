use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use tracing::warn;

use crate::error::{Error, Result};
use crate::models::{AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate};
use crate::net::reject_private_host;
use crate::{OnChanged, OnTimeUpdate, transitions};

/// Maximum audio download size (100 MiB).
const MAX_DOWNLOAD_BYTES: u64 = 100 * 1024 * 1024;

/// HTTP request timeout (connect + read combined).
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Audio player backed by Rodio for cross-platform desktop playback.
///
/// Manages a dedicated audio output thread, a playback monitor for time updates
/// and end-of-track detection, and a state machine matching the plugin's
/// [`PlaybackStatus`] model.
pub struct RodioAudioPlayer {
   inner: Arc<Mutex<Inner>>,
   stream_handle: OutputStreamHandle,
   /// Dropping this sender signals the audio output thread to exit.
   _stream_keep_alive: std::sync::mpsc::Sender<()>,
   on_changed: OnChanged,
   on_time_update: OnTimeUpdate,
}

struct Inner {
   state: PlayerState,
   playback: Option<PlaybackContext>,
   monitor_stop: Arc<AtomicBool>,
}

struct PlaybackContext {
   sink: Sink,
   /// Raw audio bytes kept for looping re-append and replay from Ended.
   /// Wrapped in `Arc` so re-append clones are cheap reference count bumps
   /// instead of multi-megabyte copies.
   source_data: Arc<[u8]>,
   duration: f64,
}

impl RodioAudioPlayer {
   /// Creates a new Rodio-backed audio player.
   ///
   /// Opens the default audio output device on a dedicated thread. Returns an error
   /// if no audio device is available.
   pub fn new(on_changed: OnChanged, on_time_update: OnTimeUpdate) -> Result<Self> {
      let stream_handle = open_audio_output()?;

      Ok(Self {
         inner: Arc::new(Mutex::new(Inner {
            state: PlayerState::default(),
            playback: None,
            monitor_stop: Arc::new(AtomicBool::new(true)),
         })),
         stream_handle: stream_handle.handle,
         _stream_keep_alive: stream_handle.keep_alive,
         on_changed,
         on_time_update,
      })
   }

   /// Stops the monitor thread by setting the flag.
   fn stop_monitor(inner: &Inner) {
      inner.monitor_stop.store(true, Ordering::Relaxed);
   }

   /// Spawns a new monitor thread for time updates and end-of-track detection.
   ///
   /// The old monitor thread may briefly overlap (up to 250ms) until it
   /// observes the stop flag on its next poll. This is harmless — any
   /// duplicate time updates are benign, and the state is already updated
   /// under the mutex before the new monitor starts, so the old one cannot
   /// trigger a spurious Ended transition.
   fn start_monitor(&self, inner: &mut Inner) {
      let stop = Arc::new(AtomicBool::new(false));
      inner.monitor_stop = stop.clone();

      let inner_arc = Arc::clone(&self.inner);
      let on_changed = Arc::clone(&self.on_changed);
      let on_time_update = Arc::clone(&self.on_time_update);

      if let Err(e) = std::thread::Builder::new()
         .name("audio-monitor".into())
         .spawn(move || {
            monitor_loop(stop, inner_arc, on_changed, on_time_update);
         })
      {
         warn!("Failed to spawn audio monitor thread: {e}");
      }
   }

   pub fn get_state(&self) -> PlayerState {
      lock_inner(&self.inner).state.clone()
   }

   pub fn load(&self, src: &str, metadata: Option<AudioMetadata>) -> Result<AudioActionResponse> {
      let meta = metadata.unwrap_or_default();

      // Transition to Loading and notify the frontend before starting I/O.
      {
         let mut inner = lock_inner(&self.inner);
         transitions::begin_load(&mut inner.state, src, &meta)?;
         let snapshot = inner.state.clone();
         drop(inner);
         (self.on_changed)(&snapshot);
      }

      // Perform I/O, decoding, and sink creation. If any step fails,
      // transition to Error so the frontend can recover from the Loading state.
      let result = self.load_inner(src, &meta);

      match result {
         Ok(snapshot) => {
            (self.on_changed)(&snapshot);
            Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Ready))
         }
         Err(e) => {
            let mut inner = lock_inner(&self.inner);
            transitions::error(&mut inner.state, e.to_string());
            let snapshot = inner.state.clone();
            drop(inner);
            (self.on_changed)(&snapshot);
            Err(e)
         }
      }
   }

   /// Inner load logic that may fail. Separated so `load()` can catch errors
   /// and transition to the Error state before propagating.
   fn load_inner(&self, src: &str, meta: &AudioMetadata) -> Result<PlayerState> {
      // Fetch audio data (may block on file I/O or HTTP download).
      let data: Arc<[u8]> = load_source_data(src)?.into();

      // Decode audio and extract duration.
      let source = Decoder::new(Cursor::new(Arc::clone(&data)))
         .map_err(|e| Error::Audio(format!("Failed to decode audio: {e}")))?;
      let duration = source
         .total_duration()
         .map(|d| d.as_secs_f64())
         .unwrap_or_else(|| probe_duration(&data).unwrap_or(0.0));

      // Create a new sink, append the decoded source, and pause immediately
      // so playback waits for an explicit play() call.
      let sink = Sink::try_new(&self.stream_handle)
         .map_err(|e| Error::Audio(format!("Failed to create audio sink: {e}")))?;
      sink.pause();
      sink.append(source);

      // Commit the state transition under the lock.
      let mut inner = lock_inner(&self.inner);

      // Re-check after I/O — another thread may have changed the state.
      transitions::load(&mut inner.state, src, meta, duration)?;

      Self::stop_monitor(&inner);

      // Apply current user settings to the new sink.
      sink.set_volume(effective_volume(&inner.state));
      sink.set_speed(inner.state.playback_rate as f32);

      inner.playback = Some(PlaybackContext {
         sink,
         source_data: data,
         duration,
      });

      Ok(inner.state.clone())
   }

   pub fn play(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         let is_ended = inner.state.status == PlaybackStatus::Ended;

         // Re-append source for replay from Ended before the transition
         // mutates status.
         if is_ended
            && let Some(ctx) = &inner.playback
            && ctx.sink.empty()
            && let Some(source) = decode_arc(&ctx.source_data)
         {
            ctx.sink.append(source);
         }

         transitions::play(&mut inner.state)?;

         if is_ended {
            inner.state.current_time = 0.0;
         }

         if let Some(ctx) = &inner.playback {
            ctx.sink.play();
         }

         self.start_monitor(&mut inner);
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Playing))
   }

   pub fn pause(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);

         transitions::pause(&mut inner.state)?;

         if let Some(ctx) = &inner.playback {
            ctx.sink.pause();
         }

         Self::stop_monitor(&inner);
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Paused))
   }

   pub fn stop(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);

         transitions::stop(&mut inner.state)?;

         Self::stop_monitor(&inner);

         // Clear the sink's queue before dropping so Sink::drop returns
         // immediately instead of blocking until the audio drains.
         if let Some(ctx) = inner.playback.take() {
            ctx.sink.stop();
         }

         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Idle))
   }

   pub fn seek(&self, position: f64) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         let was_ended = inner.state.status == PlaybackStatus::Ended;

         transitions::seek(&mut inner.state, position)?;

         if let Some(ctx) = &inner.playback {
            // If ended, re-append the source so we have something to seek within.
            if was_ended {
               if let Some(source) = decode_arc(&ctx.source_data) {
                  ctx.sink.append(source);
               }
               ctx.sink.pause();
            }

            if let Err(e) = ctx
               .sink
               .try_seek(Duration::from_secs_f64(inner.state.current_time))
            {
               warn!("Seek failed: {e}");
            }
         }

         inner.state.clone()
      };

      let expected = snapshot.status;
      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, expected))
   }

   pub fn set_volume(&self, level: f64) -> Result<PlayerState> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_volume(&mut inner.state, level)?;
         if let Some(ctx) = &inner.playback {
            ctx.sink.set_volume(effective_volume(&inner.state));
         }
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(snapshot)
   }

   pub fn set_muted(&self, muted: bool) -> PlayerState {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_muted(&mut inner.state, muted);
         if let Some(ctx) = &inner.playback {
            ctx.sink.set_volume(effective_volume(&inner.state));
         }
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      snapshot
   }

   pub fn set_playback_rate(&self, rate: f64) -> Result<PlayerState> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_playback_rate(&mut inner.state, rate)?;
         if let Some(ctx) = &inner.playback {
            ctx.sink.set_speed(inner.state.playback_rate as f32);
         }
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(snapshot)
   }

   pub fn set_loop(&self, looping: bool) -> PlayerState {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_loop(&mut inner.state, looping);
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      snapshot
   }
}

// ---------------------------------------------------------------------------
// Audio output thread
// ---------------------------------------------------------------------------

struct AudioOutput {
   handle: OutputStreamHandle,
   keep_alive: std::sync::mpsc::Sender<()>,
}

/// Opens the default audio output device on a dedicated thread.
///
/// The [`OutputStream`] must remain on the thread that created it (platform
/// requirement on some backends). We keep it alive via a channel — dropping the
/// returned sender signals the thread to exit.
fn open_audio_output() -> Result<AudioOutput> {
   let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
   let (keep_alive_tx, keep_alive_rx) = std::sync::mpsc::channel::<()>();

   std::thread::Builder::new()
      .name("audio-output".into())
      .spawn(move || match OutputStream::try_default() {
         Ok((_stream, handle)) => {
            let _ = result_tx.send(Ok(handle));
            // Block until the keep_alive sender is dropped.
            let _ = keep_alive_rx.recv();
         }
         Err(e) => {
            let _ = result_tx.send(Err(e));
         }
      })
      .map_err(|e| Error::Audio(format!("Failed to spawn audio thread: {e}")))?;

   let handle = result_rx
      .recv()
      .map_err(|_| Error::Audio("Audio thread terminated unexpectedly".into()))?
      .map_err(|e| Error::Audio(format!("Failed to open audio device: {e}")))?;

   Ok(AudioOutput {
      handle,
      keep_alive: keep_alive_tx,
   })
}

// ---------------------------------------------------------------------------
// Playback monitor
// ---------------------------------------------------------------------------

/// Polls the sink every 250ms for position updates and end-of-track detection.
fn monitor_loop(
   stop: Arc<AtomicBool>,
   inner: Arc<Mutex<Inner>>,
   on_changed: OnChanged,
   on_time_update: OnTimeUpdate,
) {
   loop {
      std::thread::sleep(Duration::from_millis(250));

      if stop.load(Ordering::Relaxed) {
         break;
      }

      let mut guard = lock_inner(&inner);

      let (pos, duration, is_empty) = match &guard.playback {
         Some(ctx) => (
            ctx.sink.get_pos().as_secs_f64(),
            ctx.duration,
            ctx.sink.empty(),
         ),
         None => break,
      };

      if is_empty {
         if guard.state.looping {
            // Re-append source for seamless (best-effort) loop.
            if let Some(ctx) = &guard.playback
               && let Some(source) = decode_arc(&ctx.source_data)
            {
               ctx.sink.append(source);
            }
            guard.state.current_time = 0.0;
            drop(guard);
            on_time_update(&TimeUpdate {
               current_time: 0.0,
               duration,
            });
         } else {
            guard.state.status = PlaybackStatus::Ended;
            guard.state.current_time = duration;
            let snapshot = guard.state.clone();
            drop(guard);
            on_changed(&snapshot);
            break;
         }
      } else {
         guard.state.current_time = pos;
         drop(guard);
         on_time_update(&TimeUpdate {
            current_time: pos,
            duration,
         });
      }
   }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Acquires the mutex, recovering from poisoning instead of panicking.
///
/// A poisoned mutex means a thread panicked while holding the lock. The inner
/// data may be in an inconsistent state, but for an audio player the worst case
/// is a glitched playback state — far better than crashing the host application.
fn lock_inner(mutex: &Mutex<Inner>) -> MutexGuard<'_, Inner> {
   mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Creates a new decoder from shared audio data (cheap Arc clone, no byte copy).
fn decode_arc(data: &Arc<[u8]>) -> Option<Decoder<Cursor<Arc<[u8]>>>> {
   Decoder::new(Cursor::new(Arc::clone(data))).ok()
}

/// Resolves the effective sink volume, accounting for the mute flag.
fn effective_volume(state: &PlayerState) -> f32 {
   if state.muted {
      0.0
   } else {
      state.volume as f32
   }
}

/// Probes audio data with symphonia to determine duration from container metadata.
///
/// This succeeds for most common formats (MP3, FLAC, WAV, OGG, AAC) where
/// `rodio::Decoder::total_duration()` returns `None`.
fn probe_duration(data: &Arc<[u8]>) -> Option<f64> {
   use symphonia::core::formats::FormatOptions;
   use symphonia::core::io::MediaSourceStream;
   use symphonia::core::meta::MetadataOptions;
   use symphonia::core::probe::Hint;

   let cursor = Cursor::new(Arc::clone(data));
   let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

   let probed = symphonia::default::get_probe()
      .format(
         &Hint::new(),
         mss,
         &FormatOptions::default(),
         &MetadataOptions::default(),
      )
      .ok()?;

   let track = probed.format.default_track()?;
   let time_base = track.codec_params.time_base?;
   let n_frames = track.codec_params.n_frames?;
   let time = time_base.calc_time(n_frames);

   Some(time.seconds as f64 + time.frac)
}

/// Loads raw audio bytes from a file path or HTTP(S) URL.
fn load_source_data(src: &str) -> Result<Vec<u8>> {
   if src.starts_with("http://") || src.starts_with("https://") {
      reject_private_host(src)?;

      let resp = ureq::AgentBuilder::new()
         .timeout(HTTP_TIMEOUT)
         .redirects(0)
         .build()
         .get(src)
         .call()
         .map_err(|e| Error::Http(format!("Failed to fetch {src}: {e}")))?;

      // Reject early if Content-Length exceeds the limit.
      if let Some(len) = resp
         .header("content-length")
         .and_then(|v| v.parse::<u64>().ok())
         && len > MAX_DOWNLOAD_BYTES
      {
         return Err(Error::Http(format!(
            "Response too large ({len} bytes, max {MAX_DOWNLOAD_BYTES})"
         )));
      }

      // Enforce the limit regardless of Content-Length (it can be absent or spoofed).
      let mut bytes = Vec::new();
      resp
         .into_reader()
         .take(MAX_DOWNLOAD_BYTES + 1)
         .read_to_end(&mut bytes)
         .map_err(Error::Io)?;

      if bytes.len() as u64 > MAX_DOWNLOAD_BYTES {
         return Err(Error::Http(format!(
            "Response exceeded maximum size of {MAX_DOWNLOAD_BYTES} bytes"
         )));
      }

      Ok(bytes)
   } else {
      if src.contains("://") || src.starts_with("data:") {
         return Err(Error::Http(format!("Unsupported URL scheme: {src}")));
      }
      std::fs::read(src).map_err(Error::Io)
   }
}
