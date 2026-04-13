mod http;
mod source;
mod symphonia;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rodio::stream::{DeviceSinkBuilder, MixerDeviceSink};
use rodio::Player;
use tracing::warn;

use self::source::{SourceDescriptor, load_source_descriptor, open_source, open_source_at};

use crate::error::{Error, Result};
use crate::models::{AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate};
use crate::{OnChanged, OnTimeUpdate, transitions};

/// Audio player backed by Rodio for cross-platform desktop playback.
///
/// Manages audio output, a playback monitor for time updates
/// and end-of-track detection, and a state machine matching the plugin's
/// [`PlaybackStatus`] model.
pub struct RodioAudioPlayer {
   inner: Arc<Mutex<Inner>>,
   output_sink: MixerDeviceSink,
   on_changed: OnChanged,
   on_time_update: OnTimeUpdate,
}

struct Inner {
   state: PlayerState,
   playback: Option<PlaybackContext>,
   monitor_stop: Arc<AtomicBool>,
   load_generation: u64,
}

struct PlaybackContext {
   sink: Player,
   source: SourceDescriptor,
   duration: f64,
   position_offset: f64,
}

impl RodioAudioPlayer {
   /// Creates a new Rodio-backed audio player.
   ///
   /// Opens the default audio output device. Returns an error
   /// if no audio device is available.
   pub fn new(on_changed: OnChanged, on_time_update: OnTimeUpdate) -> Result<Self> {
      let mut output_sink = open_audio_output()?;
      output_sink.log_on_drop(false);

      Ok(Self {
         inner: Arc::new(Mutex::new(Inner {
            state: PlayerState::default(),
            playback: None,
            monitor_stop: Arc::new(AtomicBool::new(true)),
            load_generation: 0,
         })),
         output_sink,
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

   pub fn prepare(
      &self,
      src: &str,
      metadata: Option<AudioMetadata>,
   ) -> Result<AudioActionResponse> {
      let meta = metadata.unwrap_or_default();

      let load_generation = {
         let mut inner = lock_inner(&self.inner);
         transitions::begin_load(&mut inner.state, src, &meta)?;
         inner.load_generation = inner.load_generation.wrapping_add(1);
         let load_generation = inner.load_generation;
         let snapshot = inner.state.clone();
         drop(inner);
         (self.on_changed)(&snapshot);
         load_generation
      };

      let result = self.prepare_inner(src, &meta, load_generation);

      match result {
         Ok(snapshot) => {
            (self.on_changed)(&snapshot);
            Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Ready))
         }
         Err(e) => {
            let mut inner = lock_inner(&self.inner);

            if inner.load_generation != load_generation {
               return Err(Error::InvalidState("Prepare request was canceled".into()));
            }

            transitions::error(&mut inner.state, e.to_string());
            let snapshot = inner.state.clone();
            drop(inner);
            (self.on_changed)(&snapshot);
            Err(e)
         }
      }
   }

   /// Inner prepare logic that may fail. Separated so `prepare()` can catch errors
   /// and transition to the Error state before propagating.
   fn prepare_inner(
      &self,
      src: &str,
      meta: &AudioMetadata,
      load_generation: u64,
   ) -> Result<PlayerState> {
      let descriptor = load_source_descriptor(src)?;
      let source = open_source(&descriptor)?;
      let duration = source
         .total_duration()
         .map(|d| d.as_secs_f64())
         .unwrap_or(0.0);

      // Create a new sink, append the decoded source, and pause immediately
      // so playback waits for an explicit play() call.
      let sink = Player::connect_new(self.output_sink.mixer());
      sink.pause();
      sink.append(source);

      let mut inner = lock_inner(&self.inner);

      if inner.load_generation != load_generation {
         return Err(Error::InvalidState("Prepare request was canceled".into()));
      }

      transitions::prepare(&mut inner.state, src, meta, duration)?;

      Self::stop_monitor(&inner);

      sink.set_volume(effective_volume(&inner.state));
      sink.set_speed(inner.state.playback_rate as f32);

      inner.playback = Some(PlaybackContext {
         sink,
         source: descriptor,
         duration,
         position_offset: 0.0,
      });

      Ok(inner.state.clone())
   }

   pub fn play(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         let is_ended = inner.state.status == PlaybackStatus::Ended;
         let mut replayed_from_start = false;

         // Re-append source for replay from Ended before the transition
         // mutates status.
         if is_ended
            && let Some(ctx) = &mut inner.playback
            && ctx.sink.empty()
         {
            let source = open_source(&ctx.source)?;
            ctx.sink.append(source);
            ctx.position_offset = 0.0;
            replayed_from_start = true;
         }

         transitions::play(&mut inner.state)?;

         if replayed_from_start {
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
         inner.load_generation = inner.load_generation.wrapping_add(1);

         Self::stop_monitor(&inner);

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
         let previous_time = inner.state.current_time;

         transitions::seek(&mut inner.state, position)?;

         if let Some((source_descriptor, duration)) = inner
            .playback
            .as_ref()
            .map(|ctx| (ctx.source.clone(), ctx.duration))
         {
            self.seek_playback(
               &mut inner,
               source_descriptor,
               duration,
               was_ended,
               previous_time,
            )?;
         }

         inner.state.clone()
      };

      let expected = snapshot.status;
      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, expected))
   }

   fn seek_playback(
      &self,
      inner: &mut Inner,
      source_descriptor: SourceDescriptor,
      duration: f64,
      was_ended: bool,
      previous_time: f64,
   ) -> Result<()> {
      match &source_descriptor {
         SourceDescriptor::Remote(_) => {
            self.seek_remote_playback(inner, source_descriptor, duration, previous_time)
         }
         SourceDescriptor::Local { .. } => {
            Self::seek_local_playback(inner, &source_descriptor, was_ended, previous_time)
         }
      }
   }

   fn seek_remote_playback(
      &self,
      inner: &mut Inner,
      source_descriptor: SourceDescriptor,
      duration: f64,
      previous_time: f64,
   ) -> Result<()> {
      let was_playing = inner.state.status == PlaybackStatus::Playing;
      let current_volume = effective_volume(&inner.state);
      let target_time = inner.state.current_time;

      if let Some(ctx) = &inner.playback {
         ctx.sink.set_volume(0.0);
         ctx.sink.pause();
      }

      let source = match open_source_at(&source_descriptor, target_time) {
         Ok(source) => source,
         Err(error) => {
            if let Some(ctx) = &inner.playback {
               ctx.sink.set_volume(current_volume);
               if was_playing {
                  ctx.sink.play();
               }
            }
            inner.state.current_time = previous_time;
            return Err(error);
         }
      };

      let sink = Player::connect_new(self.output_sink.mixer());
      sink.pause();
      sink.append(source);
      sink.set_volume(current_volume);
      sink.set_speed(inner.state.playback_rate as f32);

      if let Some(previous_playback) = inner.playback.replace(PlaybackContext {
         sink,
         source: source_descriptor,
         duration,
         position_offset: target_time,
      }) {
         previous_playback.sink.stop();
      }

      if was_playing && let Some(ctx) = &inner.playback {
         ctx.sink.play();
      }

      Ok(())
   }

   fn seek_local_playback(
      inner: &mut Inner,
      source_descriptor: &SourceDescriptor,
      was_ended: bool,
      previous_time: f64,
   ) -> Result<()> {
      let Some(ctx) = &inner.playback else {
         unreachable!("Playback context disappeared during local seek");
      };
      let mut reopened_source = false;

      if was_ended && ctx.sink.empty() {
         let source = match open_source(source_descriptor) {
            Ok(source) => source,
            Err(error) => {
               inner.state.current_time = previous_time;
               return Err(error);
            }
         };
         ctx.sink.append(source);
         ctx.sink.pause();
         reopened_source = true;
      }

      if let Err(e) = ctx
         .sink
         .try_seek(Duration::from_secs_f64(inner.state.current_time))
      {
         if reopened_source {
            ctx.sink.stop();
         }
         inner.state.current_time = previous_time;
         return Err(Error::Audio(format!("Failed to seek audio: {e}")));
      }

      Ok(())
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
// Audio output
// ---------------------------------------------------------------------------

/// Opens the default audio output device for playback.
fn open_audio_output() -> Result<MixerDeviceSink> {
   DeviceSinkBuilder::open_default_sink()
      .map_err(|e| Error::Audio(format!("Failed to open audio device: {e}")))
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
         Some(ctx) => {
            let pos = ctx.position_offset + ctx.sink.get_pos().as_secs_f64();
            (pos, ctx.duration, ctx.sink.empty())
         }
         None => break,
      };

      if is_empty {
         if guard.state.looping {
            // Re-append source for seamless (best-effort) loop.
            if let Some(ctx) = &mut guard.playback {
               match open_source(&ctx.source) {
                  Ok(source) => {
                     ctx.sink.append(source);
                     ctx.position_offset = 0.0;
                  }
                  Err(e) => warn!("Failed to reopen loop source: {e}"),
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
            guard.state.current_time = if duration > 0.0 { duration } else { pos };
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

/// Resolves the effective sink volume, accounting for the mute flag.
fn effective_volume(state: &PlayerState) -> f32 {
   if state.muted {
      0.0
   } else {
      state.volume as f32
   }
}
