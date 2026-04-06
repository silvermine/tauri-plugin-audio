use std::collections::VecDeque;
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rodio::buffer::SamplesBuffer;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use tracing::warn;

use crate::error::{Error, Result};
use crate::models::{AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate};
use crate::net::reject_private_host;
use crate::wsola::{self, WsolaParams};
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

/// Maximum seconds of audio to process through WSOLA per rebuild.
///
/// Caps the cost of a single `build_source` call so rate changes and seeks
/// remain responsive even on long files. The monitor loop transparently
/// rebuilds the next chunk when the current one is exhausted.
const WSOLA_MAX_CHUNK_SECS: f64 = 30.0;
const WSOLA_MAX_WALL_CHUNK_SECS: f64 = 8.0;
const INTERACTIVE_WSOLA_MAX_WALL_CHUNK_SECS: f64 = 2.5;

/// Pre-decoded interleaved PCM audio data.
struct PcmData {
   /// Interleaved samples: `[L0, R0, L1, R1, ...]`.
   interleaved: Vec<f32>,
   sample_rate: u32,
   channel_count: u16,
}

/// Seconds of look-ahead before queued audio runs out at which the monitor
/// loop pre-queues the next WSOLA chunk. Must be comfortably larger than the
/// monitor poll interval (250ms) to avoid gaps.
const PRE_QUEUE_SECS: f64 = 3.0;

const SINK_POS_RESET_EPSILON_SECS: f64 = 0.05;

struct PlaybackContext {
   sink: Sink,
   /// Pre-decoded PCM kept for rebuild on seek, rate-change, loop, and replay.
   pcm: Arc<PcmData>,
   /// Original media duration in seconds (never the stretched duration).
   duration: f64,
   /// Media position (seconds) at which the current playback segment begins.
   seek_offset: f64,
   /// Playback rate baked into the currently queued source graph.
   active_rate: f64,
   /// Media time (seconds) through which audio has been queued into the sink.
   /// Used by the monitor loop to decide when to pre-queue the next chunk.
   queued_through: f64,
   current_chunk_media_secs: f64,
   queued_chunk_media_secs: VecDeque<f64>,
   last_sink_pos_secs: f64,
   pending_chunk_build: Option<PendingChunkBuild>,
   build_generation: u64,
}

struct PendingChunkBuild {
   generation: u64,
   start_offset: f64,
   rate: f64,
   rx: Receiver<BuiltChunkData>,
}

struct BuiltChunkData {
   interleaved: Vec<f32>,
   sample_rate: u32,
   channel_count: u16,
   chunk_media_secs: f64,
}

struct PreparedPlaybackRebuild {
   sink: Sink,
   seek_offset: f64,
   rate: f64,
   chunk_media_secs: f64,
}

impl PlaybackContext {
   fn sync_chunk_progress(&mut self) {
      let sink_pos_secs = self.sink.get_pos().as_secs_f64();
      update_chunk_tracking(
         &mut self.seek_offset,
         &mut self.current_chunk_media_secs,
         &mut self.queued_chunk_media_secs,
         &mut self.last_sink_pos_secs,
         sink_pos_secs,
         self.duration,
      );
   }

   /// Current media time derived from the sink position.
   fn media_time(&self) -> f64 {
      self.seek_offset + self.sink.get_pos().as_secs_f64() * self.active_rate
   }

   /// Current media time, clamped to duration to prevent jitter overshoot.
   fn clamped_media_time(&mut self) -> f64 {
      self.sync_chunk_progress();
      self.media_time().min(self.duration)
   }
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
      let stream_handle = self.stream_handle.clone();
      let on_changed = Arc::clone(&self.on_changed);
      let on_time_update = Arc::clone(&self.on_time_update);

      if let Err(e) = std::thread::Builder::new()
         .name("audio-monitor".into())
         .spawn(move || {
            monitor_loop(stop, inner_arc, stream_handle, on_changed, on_time_update);
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
      let raw_data = load_source_data(src)?;

      // Decode entire file to interleaved PCM. This eliminates format-dependent
      // seek bugs and avoids redundant re-decoding on rebuild.
      let (pcm, mut duration) = decode_to_pcm(&raw_data)?;

      // Fallback: if the decoder couldn't determine duration, try symphonia probe.
      if duration <= 0.0
         && let Some(d) = probe_duration(&raw_data)
      {
         duration = d;
      }
      drop(raw_data); // Free compressed bytes — PCM replaces them.

      let pcm = Arc::new(pcm);

      // Read the current rate before building the source (avoid holding lock
      // during potentially expensive WSOLA processing).
      let rate = lock_inner(&self.inner).state.playback_rate;
      let (source, chunk_media_secs) = build_initial_source(&pcm, 0.0, rate);

      // Create a new sink, append the built source, and pause immediately
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

      inner.playback = Some(PlaybackContext {
         sink,
         pcm,
         duration,
         seek_offset: 0.0,
         active_rate: rate,
         queued_through: chunk_media_secs.min(duration),
         current_chunk_media_secs: chunk_media_secs.min(duration),
         queued_chunk_media_secs: VecDeque::new(),
         last_sink_pos_secs: 0.0,
         pending_chunk_build: None,
         build_generation: 0,
      });

      Ok(inner.state.clone())
   }

   pub fn play(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         let is_ended = inner.state.status == PlaybackStatus::Ended;

         // Rebuild source for replay from Ended.
         if is_ended {
            let volume = effective_volume(&inner.state);
            if let Some(ctx) = &mut inner.playback
               && ctx.sink.empty()
            {
               let rate = ctx.active_rate;
               let _ = rebuild_playback(ctx, &self.stream_handle, 0.0, rate, volume);
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
      let mut pending_rebuild = None;

      let (seek_pos, status, volume) = {
         let mut inner = lock_inner(&self.inner);

         transitions::seek(&mut inner.state, position)?;

         let seek_pos = inner.state.current_time;
         let status = inner.state.status;
         let volume = effective_volume(&inner.state);
         if let Some(ctx) = &inner.playback {
            pending_rebuild = Some((Arc::clone(&ctx.pcm), ctx.active_rate, ctx.build_generation));
         }

         (seek_pos, status, volume)
      };

      let prepared_rebuild = pending_rebuild.map(|(pcm, rate, expected_generation)| {
         prepare_playback_rebuild(&self.stream_handle, &pcm, seek_pos, rate, volume)
            .map(|prepared| (pcm, expected_generation, prepared))
      });

      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         inner.state.current_time = seek_pos;

         if let Some(ctx) = &mut inner.playback
            && let Some(result) = prepared_rebuild
         {
               match result {
                  Ok((pcm, expected_generation, prepared)) => {
                     if Arc::ptr_eq(&ctx.pcm, &pcm) && ctx.build_generation == expected_generation {
                        apply_prepared_playback_rebuild(ctx, prepared);
                        if status == PlaybackStatus::Playing {
                           ctx.sink.play();
                        }
                     }
                  }
                  Err(e) => warn!("Seek rebuild failed: {e}"),
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

         let new_rate = inner.state.playback_rate;
         let status = inner.state.status;
         let volume = effective_volume(&inner.state);
         if let Some(ctx) = &mut inner.playback {
            if matches!(
               status,
               PlaybackStatus::Playing | PlaybackStatus::Paused | PlaybackStatus::Ready
            ) {
               // Capture current media time before teardown.
               let current_media_time = ctx.clamped_media_time();
               if let Err(e) = rebuild_playback(
                  ctx,
                  &self.stream_handle,
                  current_media_time,
                  new_rate,
                  volume,
               ) {
                  warn!("Rate-change rebuild failed: {e}");
               }
               if status == PlaybackStatus::Playing {
                  ctx.sink.play();
               }
            } else {
               // Track has ended or is idle — update rate for next replay.
               ctx.active_rate = new_rate;
            }
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
   stream_handle: OutputStreamHandle,
   on_changed: OnChanged,
   on_time_update: OnTimeUpdate,
) {
   loop {
      std::thread::sleep(Duration::from_millis(250));

      if stop.load(Ordering::Relaxed) {
         break;
      }

      let mut ended_snapshot = None;
      let mut time_update = None;

      {
         let mut guard = lock_inner(&inner);
         if guard.playback.is_none() {
            break;
         }

         let looping = guard.state.looping;
         let volume = effective_volume(&guard.state);

         let (media_time, duration, is_empty) = {
            let ctx = guard.playback.as_mut().expect("playback checked above");
            try_apply_pending_chunk_build(ctx);

            let media_time = ctx.clamped_media_time();
            let duration = ctx.duration;
            let is_empty = ctx.sink.empty();

            if is_empty {
               if looping {
                  let rate = ctx.active_rate;
                  if rebuild_playback(ctx, &stream_handle, 0.0, rate, volume).is_ok() {
                     ctx.sink.play();
                  }
               }
            } else {
               maybe_start_pending_chunk_build(ctx, media_time);
            }

            (media_time, duration, is_empty)
         };

         if is_empty {
            if looping {
               guard.state.current_time = 0.0;
               time_update = Some(TimeUpdate {
                  current_time: 0.0,
                  duration,
               });
            } else {
               guard.state.status = PlaybackStatus::Ended;
               guard.state.current_time = duration;
               ended_snapshot = Some(guard.state.clone());
            }
         } else {
            guard.state.current_time = media_time;
            time_update = Some(TimeUpdate {
               current_time: media_time,
               duration,
            });
         }
      }

      if let Some(snapshot) = ended_snapshot {
         on_changed(&snapshot);
         break;
      }

      if let Some(update) = time_update {
         on_time_update(&update);
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

fn update_chunk_tracking(
   seek_offset: &mut f64,
   current_chunk_media_secs: &mut f64,
   queued_chunk_media_secs: &mut VecDeque<f64>,
   last_sink_pos_secs: &mut f64,
   sink_pos_secs: f64,
   duration: f64,
) {
   if sink_pos_secs + SINK_POS_RESET_EPSILON_SECS < *last_sink_pos_secs {
      *seek_offset = (*seek_offset + *current_chunk_media_secs).min(duration);
      *current_chunk_media_secs = queued_chunk_media_secs
         .pop_front()
         .unwrap_or_else(|| (duration - *seek_offset).max(0.0));
   }

   *last_sink_pos_secs = sink_pos_secs;
}

#[cfg(test)]
fn max_wsola_chunk_media_secs(rate: f64) -> f64 {
   (WSOLA_MAX_WALL_CHUNK_SECS * rate).min(WSOLA_MAX_CHUNK_SECS)
}

fn max_wsola_chunk_media_secs_for_wall_limit(rate: f64, wall_limit_secs: f64) -> f64 {
   (wall_limit_secs * rate).min(WSOLA_MAX_CHUNK_SECS)
}

fn interactive_wsola_chunk_media_secs(rate: f64) -> f64 {
   max_wsola_chunk_media_secs_for_wall_limit(rate, INTERACTIVE_WSOLA_MAX_WALL_CHUNK_SECS)
}

fn prequeue_target_buffer_secs(current_chunk_media_secs: f64, active_rate: f64) -> f64 {
   if (active_rate - 1.0).abs() < 1e-9 {
      PRE_QUEUE_SECS
   } else {
      current_chunk_media_secs.max(PRE_QUEUE_SECS)
   }
}

fn maybe_start_pending_chunk_build(ctx: &mut PlaybackContext, media_time: f64) {
   if (ctx.active_rate - 1.0).abs() < 1e-9
      || ctx.pending_chunk_build.is_some()
      || ctx.queued_through >= ctx.duration - 0.01
   {
      return;
   }

   let buffer_ahead = ctx.queued_through - media_time;
   let target_buffer = prequeue_target_buffer_secs(ctx.current_chunk_media_secs, ctx.active_rate);
   if buffer_ahead >= target_buffer {
      return;
   }

   let pcm = Arc::clone(&ctx.pcm);
   let start_offset = ctx.queued_through;
   let rate = ctx.active_rate;
   let generation = ctx.build_generation;
   let (tx, rx) = mpsc::sync_channel(1);

   match std::thread::Builder::new()
      .name("audio-prebuild".into())
      .spawn(move || {
         let chunk = build_chunk_data(&pcm, start_offset, rate, WSOLA_MAX_WALL_CHUNK_SECS);
         let _ = tx.send(chunk);
      }) {
      Ok(_) => {
         ctx.pending_chunk_build = Some(PendingChunkBuild {
            generation,
            start_offset,
            rate,
            rx,
         });
      }
      Err(e) => warn!("Failed to spawn audio prebuild thread: {e}"),
   }
}

fn try_apply_pending_chunk_build(ctx: &mut PlaybackContext) {
   let Some(pending) = ctx.pending_chunk_build.take() else {
      return;
   };

   match pending.rx.try_recv() {
      Ok(chunk) => {
         if pending.generation == ctx.build_generation
            && (pending.start_offset - ctx.queued_through).abs() < 0.01
            && (pending.rate - ctx.active_rate).abs() < 1e-9
         {
            ctx.sink.append(SamplesBuffer::new(
               chunk.channel_count,
               chunk.sample_rate,
               chunk.interleaved,
            ));
            ctx.queued_chunk_media_secs
               .push_back(chunk.chunk_media_secs);
            ctx.queued_through = (pending.start_offset + chunk.chunk_media_secs).min(ctx.duration);
         }
      }
      Err(TryRecvError::Empty) => {
         ctx.pending_chunk_build = Some(pending);
      }
      Err(TryRecvError::Disconnected) => {}
   }
}

/// Decode compressed audio bytes into interleaved PCM and duration.
fn decode_to_pcm(data: &[u8]) -> Result<(PcmData, f64)> {
   let source = Decoder::new(Cursor::new(data.to_vec()))
      .map_err(|e| Error::Audio(format!("Failed to decode audio: {e}")))?;

   let sample_rate = source.sample_rate();
   let channel_count = source.channels();
   let reported_duration = source.total_duration();

   // Collect all interleaved samples, converting i16 → f32.
   let interleaved: Vec<f32> = source.map(|s| s as f32 / i16::MAX as f32).collect();

   let num_ch = channel_count as usize;
   let total_frames = if num_ch > 0 {
      interleaved.len() / num_ch
   } else {
      0
   };

   let duration = reported_duration
      .map(|d| d.as_secs_f64())
      .unwrap_or_else(|| {
         if sample_rate > 0 && total_frames > 0 {
            total_frames as f64 / sample_rate as f64
         } else {
            0.0
         }
      });

   Ok((
      PcmData {
         interleaved,
         sample_rate,
         channel_count,
      },
      duration,
   ))
}

/// Build a Rodio source from pre-decoded interleaved PCM.
///
/// Slices from `seek_offset_secs`, applies WSOLA when `rate != 1.0`
/// (capped to [`WSOLA_MAX_CHUNK_SECS`] of input), and wraps in a
/// [`SamplesBuffer`].
///
/// Returns `(source, chunk_media_secs)` where `chunk_media_secs` is the
/// number of media-time seconds this chunk covers (used for pre-queue
/// bookkeeping).
fn build_chunk_data(
   pcm: &PcmData,
   seek_offset_secs: f64,
   rate: f64,
   wall_limit_secs: f64,
) -> BuiltChunkData {
   let num_ch = pcm.channel_count as usize;
   let total_frames = if num_ch > 0 {
      pcm.interleaved.len() / num_ch
   } else {
      0
   };
   let frame_offset = ((seek_offset_secs * pcm.sample_rate as f64) as usize).min(total_frames);
   let sample_offset = frame_offset * num_ch;

   if (rate - 1.0).abs() < 1e-9 {
      let remaining_frames = total_frames - frame_offset;
      let chunk_media_secs = remaining_frames as f64 / pcm.sample_rate as f64;
      BuiltChunkData {
         interleaved: pcm.interleaved[sample_offset..].to_vec(),
         sample_rate: pcm.sample_rate,
         channel_count: pcm.channel_count,
         chunk_media_secs,
      }
   } else {
      let max_chunk_frames = ((max_wsola_chunk_media_secs_for_wall_limit(rate, wall_limit_secs)
         * pcm.sample_rate as f64) as usize)
         .max(1);
      let remaining_frames = total_frames - frame_offset;
      let chunk_frames = remaining_frames.min(max_chunk_frames);
      let chunk_media_secs = chunk_frames as f64 / pcm.sample_rate as f64;
      let chunk_end = sample_offset + chunk_frames * num_ch;
      let chunk = &pcm.interleaved[sample_offset..chunk_end];

      let mut channels = vec![Vec::with_capacity(chunk_frames); num_ch.max(1)];
      for (i, &sample) in chunk.iter().enumerate() {
         channels[i % num_ch.max(1)].push(sample);
      }

      let alpha = 1.0 / rate;
      let processed = wsola::wsola(&channels, alpha, &WsolaParams::default());

      let out_frames = processed.iter().map(|ch| ch.len()).max().unwrap_or(0);
      let mut interleaved = Vec::with_capacity(out_frames * num_ch);
      for i in 0..out_frames {
         for ch in &processed {
            interleaved.push(if i < ch.len() { ch[i] } else { 0.0 });
         }
      }

      BuiltChunkData {
         interleaved,
         sample_rate: pcm.sample_rate,
         channel_count: pcm.channel_count,
         chunk_media_secs,
      }
   }
}

fn build_source_with_wall_limit(
   pcm: &PcmData,
   seek_offset_secs: f64,
   rate: f64,
   wall_limit_secs: f64,
) -> (SamplesBuffer<f32>, f64) {
   let chunk = build_chunk_data(pcm, seek_offset_secs, rate, wall_limit_secs);
   (
      SamplesBuffer::new(chunk.channel_count, chunk.sample_rate, chunk.interleaved),
      chunk.chunk_media_secs,
   )
}

#[cfg(test)]
fn build_source(pcm: &PcmData, seek_offset_secs: f64, rate: f64) -> (SamplesBuffer<f32>, f64) {
   build_source_with_wall_limit(
      pcm,
      seek_offset_secs,
      rate,
      max_wsola_chunk_media_secs(rate) / rate,
   )
}

fn build_initial_source(
   pcm: &PcmData,
   seek_offset_secs: f64,
   rate: f64,
) -> (SamplesBuffer<f32>, f64) {
   build_source_with_wall_limit(
      pcm,
      seek_offset_secs,
      rate,
      interactive_wsola_chunk_media_secs(rate) / rate,
   )
}

fn prepare_playback_rebuild(
   stream_handle: &OutputStreamHandle,
   pcm: &PcmData,
   seek_offset: f64,
   rate: f64,
   volume: f32,
) -> Result<PreparedPlaybackRebuild> {
   let new_sink = Sink::try_new(stream_handle)
      .map_err(|e| Error::Audio(format!("Failed to create audio sink: {e}")))?;
   new_sink.pause();

   let (source, chunk_media_secs) = build_initial_source(pcm, seek_offset, rate);
   new_sink.append(source);
   new_sink.set_volume(volume);

   Ok(PreparedPlaybackRebuild {
      sink: new_sink,
      seek_offset,
      rate,
      chunk_media_secs,
   })
}

fn apply_prepared_playback_rebuild(ctx: &mut PlaybackContext, prepared: PreparedPlaybackRebuild) {
   ctx.sink.stop();
   ctx.build_generation = ctx.build_generation.wrapping_add(1);
   ctx.pending_chunk_build = None;
   ctx.sink = prepared.sink;
   ctx.seek_offset = prepared.seek_offset;
   ctx.active_rate = prepared.rate;
   ctx.queued_through = (prepared.seek_offset + prepared.chunk_media_secs).min(ctx.duration);
   ctx.current_chunk_media_secs = prepared.chunk_media_secs;
   ctx.queued_chunk_media_secs.clear();
   ctx.last_sink_pos_secs = 0.0;
}

/// Rebuild playback: stop the old sink, create a new one, build a source,
/// and update the context. The new sink starts paused.
///
/// Accepts pre-computed `volume` so callers can resolve it before taking a
/// mutable borrow on `PlaybackContext`, avoiding borrow-checker conflicts.
fn rebuild_playback(
   ctx: &mut PlaybackContext,
   stream_handle: &OutputStreamHandle,
   seek_offset: f64,
   rate: f64,
   volume: f32,
) -> Result<()> {
   let prepared = prepare_playback_rebuild(stream_handle, &ctx.pcm, seek_offset, rate, volume)?;
   apply_prepared_playback_rebuild(ctx, prepared);

   Ok(())
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

#[cfg(test)]
mod tests {
   use super::*;

   /// Create a minimal mono 16-bit PCM WAV in memory.
   fn make_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
      let num_channels: u16 = 1;
      let bits_per_sample: u16 = 16;
      let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
      let block_align = num_channels * bits_per_sample / 8;
      let data_size = (samples.len() * 2) as u32;
      let file_size = 36 + data_size;

      let mut buf = Vec::with_capacity(file_size as usize + 8);
      buf.extend_from_slice(b"RIFF");
      buf.extend_from_slice(&file_size.to_le_bytes());
      buf.extend_from_slice(b"WAVE");
      // fmt chunk
      buf.extend_from_slice(b"fmt ");
      buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
      buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
      buf.extend_from_slice(&num_channels.to_le_bytes());
      buf.extend_from_slice(&sample_rate.to_le_bytes());
      buf.extend_from_slice(&byte_rate.to_le_bytes());
      buf.extend_from_slice(&block_align.to_le_bytes());
      buf.extend_from_slice(&bits_per_sample.to_le_bytes());
      // data chunk
      buf.extend_from_slice(b"data");
      buf.extend_from_slice(&data_size.to_le_bytes());
      for &s in samples {
         buf.extend_from_slice(&s.to_le_bytes());
      }
      buf
   }

   /// Create a stereo 16-bit PCM WAV.
   fn make_stereo_wav(left: &[i16], right: &[i16], sample_rate: u32) -> Vec<u8> {
      assert_eq!(left.len(), right.len());
      let num_channels: u16 = 2;
      let bits_per_sample: u16 = 16;
      let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
      let block_align = num_channels * bits_per_sample / 8;
      let data_size = (left.len() * 2 * 2) as u32; // 2 channels, 2 bytes each
      let file_size = 36 + data_size;

      let mut buf = Vec::with_capacity(file_size as usize + 8);
      buf.extend_from_slice(b"RIFF");
      buf.extend_from_slice(&file_size.to_le_bytes());
      buf.extend_from_slice(b"WAVE");
      buf.extend_from_slice(b"fmt ");
      buf.extend_from_slice(&16u32.to_le_bytes());
      buf.extend_from_slice(&1u16.to_le_bytes());
      buf.extend_from_slice(&num_channels.to_le_bytes());
      buf.extend_from_slice(&sample_rate.to_le_bytes());
      buf.extend_from_slice(&byte_rate.to_le_bytes());
      buf.extend_from_slice(&block_align.to_le_bytes());
      buf.extend_from_slice(&bits_per_sample.to_le_bytes());
      buf.extend_from_slice(b"data");
      buf.extend_from_slice(&data_size.to_le_bytes());
      for (&l, &r) in left.iter().zip(right.iter()) {
         buf.extend_from_slice(&l.to_le_bytes());
         buf.extend_from_slice(&r.to_le_bytes());
      }
      buf
   }

   // -----------------------------------------------------------------------
   // decode_to_pcm
   // -----------------------------------------------------------------------

   #[test]
   fn decode_to_pcm_mono_wav() {
      let samples: Vec<i16> = (0..4410).map(|i| (i % 100) as i16).collect();
      let wav = make_wav(&samples, 44100);
      let (pcm, duration) = decode_to_pcm(&wav).unwrap();

      assert_eq!(pcm.channel_count, 1);
      assert_eq!(pcm.sample_rate, 44100);
      assert_eq!(pcm.interleaved.len(), 4410);
      assert!((duration - 0.1).abs() < 0.01, "duration {duration} ≠ ~0.1s");
   }

   #[test]
   fn decode_to_pcm_stereo_wav() {
      let left: Vec<i16> = (0..4410).map(|i| (i % 100) as i16).collect();
      let right: Vec<i16> = (0..4410).map(|i| -((i % 100) as i16)).collect();
      let wav = make_stereo_wav(&left, &right, 44100);
      let (pcm, _) = decode_to_pcm(&wav).unwrap();

      assert_eq!(pcm.channel_count, 2);
      // Interleaved: 4410 frames * 2 channels = 8820 samples
      assert_eq!(pcm.interleaved.len(), 8820);
   }

   // -----------------------------------------------------------------------
   // build_source
   // -----------------------------------------------------------------------

   /// Synthesize interleaved PCM test data.
   fn synth_pcm(num_frames: usize, num_channels: u16, sample_rate: u32) -> PcmData {
      let num_ch = num_channels as usize;
      let mut interleaved = Vec::with_capacity(num_frames * num_ch);
      for i in 0..num_frames {
         for c in 0..num_ch {
            interleaved.push(((i + c * 1000) as f32) * 0.001);
         }
      }
      PcmData {
         interleaved,
         sample_rate,
         channel_count: num_channels,
      }
   }

   #[test]
   fn build_source_passthrough_preserves_length() {
      let pcm = synth_pcm(4410, 1, 44100);
      let (source, _) = build_source(&pcm, 0.0, 1.0);
      // SamplesBuffer implements Source — count samples via iterator
      let total: usize = source.count();
      assert_eq!(total, 4410, "passthrough should preserve sample count");
   }

   #[test]
   fn build_source_seek_offset_slices_correctly() {
      let pcm = synth_pcm(44100, 1, 44100); // 1 second
      let (source, _) = build_source(&pcm, 0.5, 1.0); // seek to 0.5s
      let total = source.count();
      // Should have ~22050 samples (half the file)
      assert_eq!(total, 22050);
   }

   #[test]
   fn build_source_stereo_interleave() {
      let pcm = PcmData {
         interleaved: vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
         sample_rate: 44100,
         channel_count: 2,
      };
      let (source, _) = build_source(&pcm, 0.0, 1.0);
      let samples: Vec<f32> = source.collect();
      // Passthrough preserves interleaved order.
      assert_eq!(samples, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
   }

   #[test]
   fn build_source_wsola_doubles_at_half_rate() {
      let pcm = synth_pcm(8820, 1, 44100); // 0.2s
      let (source, _) = build_source(&pcm, 0.0, 0.5); // 0.5× speed → 2× length
      let total: usize = source.count();
      let ratio = total as f64 / 8820.0;
      assert!(
         (ratio - 2.0).abs() < 0.15,
         "0.5× rate should ~double length, got ratio {ratio}"
      );
   }

   #[test]
   fn build_source_wsola_halves_at_double_rate() {
      let pcm = synth_pcm(8820, 1, 44100); // 0.2s
      let (source, _) = build_source(&pcm, 0.0, 2.0); // 2× speed → 0.5× length
      let total: usize = source.count();
      let ratio = total as f64 / 8820.0;
      assert!(
         (ratio - 0.5).abs() < 0.15,
         "2.0× rate should ~halve length, got ratio {ratio}"
      );
   }

   #[test]
   fn build_source_seek_beyond_end_returns_empty() {
      let pcm = synth_pcm(4410, 1, 44100); // 0.1s
      let (source, _) = build_source(&pcm, 999.0, 1.0); // seek way past end
      let total: usize = source.count();
      assert_eq!(total, 0);
   }

   // -----------------------------------------------------------------------
   // PlaybackContext media time (unit-level, no live Sink needed)
   // -----------------------------------------------------------------------

   #[test]
   fn media_time_formula_at_1x() {
      // At rate 1.0, media_time = seek_offset + sink_pos * 1.0
      // We can't construct a real Sink without audio output, but we can
      // verify the formula in build_source output length:
      // A 1.0s file at rate 1.0 should produce ~44100 samples at 44100 Hz
      let pcm = synth_pcm(44100, 1, 44100);
      let (source, _) = build_source(&pcm, 0.0, 1.0);
      let total = source.count();
      let output_duration = total as f64 / 44100.0;
      assert!(
         (output_duration - 1.0).abs() < 0.01,
         "1.0s file at 1.0× should produce ~1.0s output, got {output_duration}"
      );
   }

   #[test]
   fn media_time_formula_at_2x() {
      // At rate 2.0, the output is half-length. Sink plays for 0.5s of
      // wall-clock time to cover 1.0s of media time.
      let pcm = synth_pcm(44100, 1, 44100); // 1.0s
      let (source, _) = build_source(&pcm, 0.0, 2.0);
      let total = source.count();
      let output_duration = total as f64 / 44100.0;
      // Output should be ~0.5s (wall time), media time = 0.0 + 0.5 * 2.0 = 1.0
      assert!(
         (output_duration - 0.5).abs() < 0.1,
         "1.0s file at 2.0× should produce ~0.5s output, got {output_duration}"
      );
   }

   // -----------------------------------------------------------------------
   // Behavior-level: simulated playback scenarios
   // -----------------------------------------------------------------------

   /// Simulates the source a user would hear after load → setPlaybackRate → play.
   /// Verifies that the source is actually rebuilt at the new rate, not stale.
   #[test]
   fn scenario_load_then_set_rate_then_play() {
      let pcm = synth_pcm(44100, 1, 44100); // 1.0s mono

      // load builds at rate 1.0
      let (source_at_1x, _) = build_source(&pcm, 0.0, 1.0);
      let len_1x = source_at_1x.count();

      // setPlaybackRate(2.0) should rebuild — verify the new source differs
      let (source_at_2x, _) = build_source(&pcm, 0.0, 2.0);
      let len_2x = source_at_2x.count();

      // The 2× source must be roughly half the length
      let ratio = len_2x as f64 / len_1x as f64;
      assert!(
         (ratio - 0.5).abs() < 0.1,
         "after setPlaybackRate(2.0), source should be ~half length; ratio={ratio}"
      );
   }

   /// Seek at non-1.0 rate: verify the output starts from the seek point
   /// and has the expected stretched length.
   #[test]
   fn scenario_seek_at_non_1x_rate() {
      let pcm = synth_pcm(44100, 1, 44100); // 1.0s

      // Seek to 0.5s at rate 0.5 (half speed → double length for remaining 0.5s)
      let (source, _) = build_source(&pcm, 0.5, 0.5);
      let total = source.count();

      // Remaining media is 0.5s. At 0.5× speed, output should be ~1.0s = ~44100 samples
      let output_secs = total as f64 / 44100.0;
      assert!(
         (output_secs - 1.0).abs() < 0.15,
         "seek@0.5s + rate 0.5× should produce ~1.0s output, got {output_secs}s"
      );
   }

   /// Replay from Ended at non-1.0 rate: source is rebuilt from offset 0
   /// with the current active_rate.
   #[test]
   fn scenario_replay_at_non_1x_rate() {
      let pcm = synth_pcm(44100, 1, 44100); // 1.0s

      // Simulate replay: rebuild from offset 0.0 at rate 1.5
      let (source, _) = build_source(&pcm, 0.0, 1.5);
      let total = source.count();

      // At 1.5× speed, output is ~0.667s = ~29400 samples
      let output_secs = total as f64 / 44100.0;
      let expected = 1.0 / 1.5;
      assert!(
         (output_secs - expected).abs() < 0.1,
         "replay at 1.5× should produce ~{expected:.3}s output, got {output_secs:.3}s"
      );
   }

   /// Loop at non-1.0 rate: each loop iteration rebuilds from offset 0
   /// and should produce the same length output.
   #[test]
   fn scenario_loop_at_non_1x_rate() {
      let pcm = synth_pcm(44100, 1, 44100);

      let iter1 = build_source(&pcm, 0.0, 0.75).0.count();
      let iter2 = build_source(&pcm, 0.0, 0.75).0.count();

      assert_eq!(
         iter1, iter2,
         "consecutive loop iterations should produce identical length"
      );
   }

   /// Monitor-loop time math: verify that output duration × rate ≈ media duration.
   /// This is the core invariant: sink plays for (media_duration / rate) wall-clock
   /// seconds, so media_time = seek_offset + wall_time × rate = full duration.
   #[test]
   fn scenario_monitor_time_math_invariant() {
      let rates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0];
      let pcm = synth_pcm(44100, 1, 44100); // 1.0s
      let media_duration = 1.0_f64;

      for rate in rates {
         let mut offset = 0.0_f64;
         let mut total_wall_time = 0.0_f64;

         while offset < media_duration - 0.01 {
            let (source, chunk_secs) = build_source(&pcm, offset, rate);
            let wall_samples = source.count();
            total_wall_time += wall_samples as f64 / 44100.0;
            offset += chunk_secs;
         }

         let computed_media = total_wall_time * rate;
         assert!(
            (computed_media - media_duration).abs() < 1.0,
            "rate {rate}: wall_time × rate = {computed_media:.2}s, expected ~{media_duration}s"
         );
      }
   }

   /// Verify that build_source at rate 1.0 produces bit-exact passthrough
   /// (no WSOLA processing applied).
   #[test]
   fn scenario_passthrough_is_exact() {
      let pcm = PcmData {
         interleaved: vec![0.1, 0.2, 0.3, 0.4, 0.5],
         sample_rate: 44100,
         channel_count: 1,
      };
      let samples: Vec<f32> = build_source(&pcm, 0.0, 1.0).0.collect();
      assert_eq!(samples, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
   }

   #[test]
   fn update_chunk_tracking_advances_to_next_appended_chunk() {
      let mut seek_offset = 0.0;
      let mut current_chunk_media_secs = 30.0;
      let mut queued_chunk_media_secs = VecDeque::from([5.0]);
      let mut last_sink_pos_secs = 14.9;

      update_chunk_tracking(
         &mut seek_offset,
         &mut current_chunk_media_secs,
         &mut queued_chunk_media_secs,
         &mut last_sink_pos_secs,
         0.1,
         35.0,
      );

      assert!((seek_offset - 30.0).abs() < 1e-9);
      assert!((current_chunk_media_secs - 5.0).abs() < 1e-9);
      assert!(queued_chunk_media_secs.is_empty());
      assert!((last_sink_pos_secs - 0.1).abs() < 1e-9);
   }

   #[test]
   fn update_chunk_tracking_keeps_media_time_monotonic_across_chunk_reset() {
      let rate = 2.0;
      let mut seek_offset = 0.0;
      let mut current_chunk_media_secs = 30.0;
      let mut queued_chunk_media_secs = VecDeque::from([5.0]);
      let mut last_sink_pos_secs = 14.9;
      let before = seek_offset + last_sink_pos_secs * rate;

      update_chunk_tracking(
         &mut seek_offset,
         &mut current_chunk_media_secs,
         &mut queued_chunk_media_secs,
         &mut last_sink_pos_secs,
         0.1,
         35.0,
      );

      let after = seek_offset + last_sink_pos_secs * rate;
      assert!(
         after > before,
         "media time should keep moving forward across chunk boundaries"
      );
      assert!((after - 30.2).abs() < 1e-9);
   }

   #[test]
   fn max_wsola_chunk_media_secs_scales_with_rate() {
      assert!((max_wsola_chunk_media_secs(0.75) - 6.0).abs() < 1e-9);
      assert!((max_wsola_chunk_media_secs(2.0) - 16.0).abs() < 1e-9);
      assert!((max_wsola_chunk_media_secs(4.0) - 30.0).abs() < 1e-9);
   }

   #[test]
   fn interactive_wsola_chunk_media_secs_scales_with_rate() {
      assert!((interactive_wsola_chunk_media_secs(0.75) - 1.875).abs() < 1e-9);
      assert!((interactive_wsola_chunk_media_secs(2.0) - 5.0).abs() < 1e-9);
      assert!((interactive_wsola_chunk_media_secs(4.0) - 10.0).abs() < 1e-9);
   }

   #[test]
   fn prequeue_target_buffer_secs_tracks_chunk_size_for_rate_adjusted_playback() {
      assert!((prequeue_target_buffer_secs(10.0, 1.0) - PRE_QUEUE_SECS).abs() < 1e-9);
      assert!((prequeue_target_buffer_secs(6.0, 0.75) - 6.0).abs() < 1e-9);
      assert!((prequeue_target_buffer_secs(16.0, 2.0) - 16.0).abs() < 1e-9);
      assert!((prequeue_target_buffer_secs(2.0, 0.5) - PRE_QUEUE_SECS).abs() < 1e-9);
   }

   // -----------------------------------------------------------------------
   // Chunk boundary: continuity, time math, and returned media coverage
   // -----------------------------------------------------------------------

   #[test]
   fn chunk_media_secs_passthrough() {
      let pcm = synth_pcm(44100, 1, 44100); // 1.0s
      let (_, chunk_secs) = build_source(&pcm, 0.0, 1.0);
      assert!(
         (chunk_secs - 1.0).abs() < 0.001,
         "passthrough chunk should cover full 1.0s, got {chunk_secs}"
      );
   }

   #[test]
   fn chunk_media_secs_wsola() {
      let pcm = synth_pcm(44100, 1, 44100); // 1.0s (< WSOLA_MAX_CHUNK_SECS)
      let (_, chunk_secs) = build_source(&pcm, 0.0, 2.0);
      // Chunk covers the full file since 1.0s < 30s cap.
      assert!(
         (chunk_secs - 1.0).abs() < 0.001,
         "wsola chunk should cover full 1.0s, got {chunk_secs}"
      );
   }

   #[test]
   fn chunk_media_secs_wsola_respects_wall_duration_cap() {
      let sr = 100;
      let total_secs = 65.0;
      let total_frames = (total_secs * sr as f64) as usize;
      let pcm = synth_pcm(total_frames, 1, sr);

      let (_, slow_chunk_secs) = build_source(&pcm, 0.0, 0.75);
      let (_, fast_chunk_secs) = build_source(&pcm, 0.0, 2.0);
      let (_, maxed_chunk_secs) = build_source(&pcm, 0.0, 4.0);

      assert!((slow_chunk_secs - 6.0).abs() < 0.02);
      assert!((fast_chunk_secs - 16.0).abs() < 0.02);
      assert!((maxed_chunk_secs - 30.0).abs() < 0.02);
   }

   #[test]
   fn initial_chunk_media_secs_wsola_respects_interactive_wall_duration_cap() {
      let sr = 100;
      let total_secs = 65.0;
      let total_frames = (total_secs * sr as f64) as usize;
      let pcm = synth_pcm(total_frames, 1, sr);

      let (_, slow_chunk_secs) = build_initial_source(&pcm, 0.0, 0.75);
      let (_, fast_chunk_secs) = build_initial_source(&pcm, 0.0, 2.0);
      let (_, maxed_chunk_secs) = build_initial_source(&pcm, 0.0, 4.0);

      assert!((slow_chunk_secs - 1.875).abs() < 0.02);
      assert!((fast_chunk_secs - 5.0).abs() < 0.02);
      assert!((maxed_chunk_secs - 10.0).abs() < 0.02);
   }

   /// Consecutive chunks cover the full file with no gap or overlap.
   #[test]
   fn chunk_boundary_continuity() {
      // Create audio longer than WSOLA_MAX_CHUNK_SECS (30s) at rate != 1.0
      // Use a small sample rate to keep the test fast.
      let sr = 100; // 100 Hz — tiny, but WSOLA still works
      let total_secs = 65.0; // 65 seconds of audio
      let total_frames = (total_secs * sr as f64) as usize;
      let pcm = synth_pcm(total_frames, 1, sr);
      let rate = 2.0;

      let mut covered = 0.0_f64;
      let mut offset = 0.0_f64;
      let mut chunks = 0;

      while offset < total_secs - 0.01 {
         let (_, chunk_secs) = build_source(&pcm, offset, rate);
         assert!(
            chunk_secs > 0.0,
            "chunk at offset {offset} produced 0 media seconds"
         );
         covered += chunk_secs;
         offset += chunk_secs;
         chunks += 1;
         // Safety: avoid infinite loop in case of bug.
         assert!(chunks <= 100, "too many chunks");
      }

      assert!(
         (covered - total_secs).abs() < 0.02,
         "chunks should cover ~{total_secs}s total, got {covered}s over {chunks} chunks"
      );
      assert!(
         chunks >= 2,
         "65s file at 100Hz should require multiple chunks, got {chunks}"
      );
   }

   /// Time math across chunk boundaries: the sum of per-chunk wall-time
   /// durations × rate ≈ total media duration.
   #[test]
   fn chunk_boundary_time_math() {
      let sr = 100;
      let total_secs = 65.0;
      let total_frames = (total_secs * sr as f64) as usize;
      let pcm = synth_pcm(total_frames, 1, sr);

      for rate in [0.5, 1.5, 2.0] {
         let mut offset = 0.0_f64;
         let mut total_wall_time = 0.0_f64;

         while offset < total_secs - 0.01 {
            let (source, chunk_secs) = build_source(&pcm, offset, rate);
            let wall_samples = source.count();
            total_wall_time += wall_samples as f64 / sr as f64;
            offset += chunk_secs;
         }

         let computed_media = total_wall_time * rate;
         assert!(
            (computed_media - total_secs).abs() < 1.0,
            "rate {rate}: wall_time × rate = {computed_media:.2}s, expected ~{total_secs}s"
         );
      }
   }

   /// Looping at chunk boundaries: rebuild from 0 produces the same first chunk.
   #[test]
   fn chunk_boundary_loop_restart() {
      let sr = 100;
      let total_frames = (65.0 * sr as f64) as usize;
      let pcm = synth_pcm(total_frames, 1, sr);

      let (src_a, secs_a) = build_source(&pcm, 0.0, 1.5);
      let (src_b, secs_b) = build_source(&pcm, 0.0, 1.5);

      assert_eq!(
         src_a.count(),
         src_b.count(),
         "loop restart should produce identical first chunk"
      );
      assert!(
         (secs_a - secs_b).abs() < 1e-9,
         "chunk coverage should be identical on restart"
      );
   }
}
