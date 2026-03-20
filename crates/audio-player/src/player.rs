use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use tracing::warn;

use crate::error::{Error, Result};
use crate::models::{AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate};
use crate::{OnChanged, OnTimeUpdate, transitions};

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
   source_data: Vec<u8>,
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
   fn start_monitor(&self, inner: &mut Inner) {
      let stop = Arc::new(AtomicBool::new(false));
      inner.monitor_stop = stop.clone();

      let inner_arc = Arc::clone(&self.inner);
      let on_changed = Arc::clone(&self.on_changed);
      let on_time_update = Arc::clone(&self.on_time_update);

      std::thread::Builder::new()
         .name("audio-monitor".into())
         .spawn(move || {
            monitor_loop(stop, inner_arc, on_changed, on_time_update);
         })
         .ok();
   }

   pub fn get_state(&self) -> PlayerState {
      self.inner.lock().unwrap().state.clone()
   }

   pub fn load(&self, src: &str, metadata: Option<AudioMetadata>) -> Result<AudioActionResponse> {
      // Fast-path state check before potentially slow I/O.
      {
         let inner = self.inner.lock().unwrap();
         transitions::can_load(inner.state.status)?;
      }

      let meta = metadata.unwrap_or_default();

      // Fetch audio data (may block on file I/O or HTTP download).
      let data = load_source_data(src)?;

      // Decode audio and extract duration.
      let source = Decoder::new(Cursor::new(data.clone()))
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
      let snapshot = {
         let mut inner = self.inner.lock().unwrap();

         // Re-check after I/O — another thread may have changed the state.
         transitions::load(&mut inner.state, src, &meta, duration)?;

         Self::stop_monitor(&inner);

         // Apply current user settings to the new sink.
         sink.set_volume(effective_volume(&inner.state));
         sink.set_speed(inner.state.playback_rate as f32);

         inner.playback = Some(PlaybackContext {
            sink,
            source_data: data,
            duration,
         });

         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Ready))
   }

   pub fn play(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = self.inner.lock().unwrap();
         let is_ended = inner.state.status == PlaybackStatus::Ended;

         // Re-append source for replay from Ended before the transition
         // mutates status.
         if is_ended && let Some(ctx) = &inner.playback {
            let data = ctx.source_data.clone();
            if let Ok(source) = Decoder::new(Cursor::new(data)) {
               ctx.sink.append(source);
            }
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
         let mut inner = self.inner.lock().unwrap();

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
         let mut inner = self.inner.lock().unwrap();

         transitions::stop(&mut inner.state)?;

         Self::stop_monitor(&inner);
         inner.playback = None;
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Idle))
   }

   pub fn seek(&self, position: f64) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = self.inner.lock().unwrap();
         let was_ended = inner.state.status == PlaybackStatus::Ended;

         transitions::seek(&mut inner.state, position)?;

         if let Some(ctx) = &inner.playback {
            // If ended, re-append the source so we have something to seek within.
            if was_ended {
               let data = ctx.source_data.clone();
               if let Ok(source) = Decoder::new(Cursor::new(data)) {
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
         let mut inner = self.inner.lock().unwrap();
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
         let mut inner = self.inner.lock().unwrap();
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
         let mut inner = self.inner.lock().unwrap();
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
         let mut inner = self.inner.lock().unwrap();
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

      let mut guard = inner.lock().unwrap();

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
            if let Some(ctx) = &guard.playback {
               let data = ctx.source_data.clone();
               if let Ok(source) = Decoder::new(Cursor::new(data)) {
                  ctx.sink.append(source);
               }
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
fn probe_duration(data: &[u8]) -> Option<f64> {
   use symphonia::core::formats::FormatOptions;
   use symphonia::core::io::MediaSourceStream;
   use symphonia::core::meta::MetadataOptions;
   use symphonia::core::probe::Hint;

   let cursor = Cursor::new(data.to_vec());
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
      let resp = ureq::get(src)
         .call()
         .map_err(|e| Error::Http(format!("Failed to fetch {src}: {e}")))?;
      let len = resp
         .header("content-length")
         .and_then(|v| v.parse::<usize>().ok())
         .unwrap_or(0);
      let mut bytes = Vec::with_capacity(len);
      resp
         .into_reader()
         .read_to_end(&mut bytes)
         .map_err(Error::Io)?;
      Ok(bytes)
   } else {
      std::fs::read(src).map_err(Error::Io)
   }
}
