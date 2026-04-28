mod http;
mod resume_fade;
mod source;
mod stretch;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use rodio::Player;
use rodio::stream::{DeviceSinkBuilder, MixerDeviceSink};
use tracing::warn;

use self::resume_fade::ResumeFadeHandle;
use self::source::{SeekStrategy, SourceDescriptor, load_source_descriptor, open_source_at};

use crate::error::{Error, Result};
use crate::models::{
   AudioActionResponse, LoopMode, PlaybackStatus, PlayerState, PlaylistItem, TimeUpdate,
};
use crate::transitions::NavTarget;
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
   /// Weak self-reference so the monitor thread can call back without
   /// keeping the player alive after its owner has dropped it.
   weak_self: Weak<Self>,
}

struct Inner {
   state: PlayerState,
   playback: Option<PlaybackContext>,
   monitor_stop: Arc<AtomicBool>,
   load_generation: u64,
   seek_generation: u64,
}

struct PlaybackContext {
   sink: Player,
   source: SourceDescriptor,
   duration: f64,
   position_offset: f64,
   position_latency: f64,
   seek_strategy: SeekStrategy,
   resume_fade: Option<ResumeFadeHandle>,
}

impl RodioAudioPlayer {
   /// Creates a new Rodio-backed audio player.
   ///
   /// Opens the default audio output device. Returns an error
   /// if no audio device is available.
   ///
   /// Returns an `Arc` so the monitor thread can hold a weak self-reference
   /// for auto-advance callbacks.
   pub fn new(on_changed: OnChanged, on_time_update: OnTimeUpdate) -> Result<Arc<Self>> {
      let mut output_sink = open_audio_output()?;
      output_sink.log_on_drop(false);

      Ok(Arc::new_cyclic(|weak_self| Self {
         inner: Arc::new(Mutex::new(Inner {
            state: PlayerState::default(),
            playback: None,
            monitor_stop: Arc::new(AtomicBool::new(true)),
            load_generation: 0,
            seek_generation: 0,
         })),
         output_sink,
         on_changed,
         on_time_update,
         weak_self: weak_self.clone(),
      }))
   }

   /// Stops the monitor thread by setting the flag.
   fn stop_monitor(inner: &Inner) {
      inner.monitor_stop.store(true, Ordering::Relaxed);
   }

   /// Emits a `state-changed` event with per-item `artwork` data stripped
   /// from the playlist payload.
   fn emit_state_changed(&self, snapshot: &PlayerState) {
      let mut stripped = snapshot.clone();

      for item in stripped.playlist.iter_mut() {
         if let Some(meta) = item.metadata.as_mut() {
            meta.artwork = None;
         }
      }
      (self.on_changed)(&stripped);
   }

   /// Spawns a new monitor thread for time updates and end-of-track detection.
   ///
   /// The old monitor thread may briefly overlap (up to 250ms) until it
   /// observes the stop flag on its next poll. This is harmless — any
   /// duplicate time updates are benign, and the state is already updated
   /// under the mutex before the new monitor starts, so the old one cannot
   /// trigger a spurious auto-advance.
   fn start_monitor(&self, inner: &mut Inner) {
      let stop = Arc::new(AtomicBool::new(false));
      inner.monitor_stop = stop.clone();

      let Some(player) = self.weak_self.upgrade() else {
         warn!("Cannot start monitor: player has been dropped");
         return;
      };

      if let Err(e) = std::thread::Builder::new()
         .name("audio-monitor".into())
         .spawn(move || {
            monitor_loop(stop, player);
         })
      {
         warn!("Failed to spawn audio monitor thread: {e}");
      }
   }

   pub fn get_state(&self) -> PlayerState {
      lock_inner(&self.inner).state.clone()
   }

   /// Loads a playlist and prepares the chosen (or first) item for playback.
   pub fn load(
      &self,
      playlist: Vec<PlaylistItem>,
      start_index: Option<usize>,
   ) -> Result<AudioActionResponse> {
      let start_index = start_index.unwrap_or(0);

      let (item_src, load_generation) = {
         let mut inner = lock_inner(&self.inner);
         transitions::begin_load(&mut inner.state, playlist, start_index)?;
         inner.load_generation = inner.load_generation.wrapping_add(1);
         inner.seek_generation = inner.seek_generation.wrapping_add(1);
         let load_generation = inner.load_generation;
         let src = inner
            .state
            .current()
            .map(|item| item.src.clone())
            .ok_or_else(|| Error::InvalidState("Missing current item after begin_load".into()))?;
         let snapshot = inner.state.clone();
         drop(inner);
         self.emit_state_changed(&snapshot);
         (src, load_generation)
      };

      let result = self.load_inner(&item_src, load_generation);

      match result {
         Ok(snapshot) => {
            self.emit_state_changed(&snapshot);
            Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Ready))
         }
         Err(error) => {
            let snapshot = {
               let mut inner = lock_inner(&self.inner);
               finish_load_as_error(&mut inner, load_generation, error.to_string())
            };

            if let Some(snapshot) = snapshot {
               self.emit_state_changed(&snapshot);
            }

            Err(error)
         }
      }
   }

   /// Inner load logic that performs I/O for an already-set current item and
   /// finalizes the state to `Ready`. Used by `load`, `next`, `prev`, and
   /// auto-advance.
   fn load_inner(&self, src: &str, load_generation: u64) -> Result<PlayerState> {
      let descriptor = load_source_descriptor(src)?;

      // Best-effort ID3 extraction. Local files only: we don't want to
      // refetch a remote stream just for tags.
      let extracted_metadata = match &descriptor {
         SourceDescriptor::Local { path } => std::fs::read(path)
            .ok()
            .and_then(|bytes| crate::metadata::extract(&bytes)),
         SourceDescriptor::Remote(_) => None,
      };

      let mut attempted_playback_rate = lock_inner(&self.inner).state.playback_rate;
      let mut playback_rate_retry_stage = LoadPlaybackRateRetryStage::NotRetried;

      loop {
         let opened_source = open_source_at(&descriptor, 0.0, attempted_playback_rate)?;
         let duration = opened_source.duration;

         // Create a new sink, append the decoded source, and pause immediately
         // so playback waits for an explicit play() call.
         let sink = Player::connect_new(self.output_sink.mixer());
         sink.pause();
         sink.append(opened_source.source);

         let mut inner = lock_inner(&self.inner);

         if inner.load_generation != load_generation {
            sink.stop();
            return Err(Error::InvalidState("Load request was canceled".into()));
         }

         let playback_rate_change_action = classify_load_playback_rate_change(
            inner.state.playback_rate,
            attempted_playback_rate,
            playback_rate_retry_stage,
         );

         match playback_rate_change_action {
            LoadPlaybackRateChangeAction::Apply => {}
            LoadPlaybackRateChangeAction::RetryDuringLoad {
               next_playback_rate,
               retry_stage,
            } => {
               attempted_playback_rate = next_playback_rate;
               playback_rate_retry_stage = retry_stage;
               sink.stop();
               continue;
            }
            LoadPlaybackRateChangeAction::Exhausted => {
               sink.stop();
               return Err(Error::InvalidState(
                  "Playback rate changed too many times during load".into(),
               ));
            }
         }

         // Enrich the active playlist item's metadata with anything we found
         // in the file (caller-supplied fields win per-field). Done before
         // transitions::load so the Ready state-changed event already carries
         // the merged metadata.
         if let Some(idx) = inner.state.current_index
            && let Some(item) = inner.state.playlist.get_mut(idx)
         {
            let merged = crate::metadata::merge(item.metadata.take(), extracted_metadata.clone());
            item.metadata = Some(merged);
         }

         transitions::load(&mut inner.state, duration)?;

         Self::stop_monitor(&inner);

         sink.set_volume(effective_volume(&inner.state));

         if let Some(prev) = inner.playback.take() {
            prev.sink.stop();
         }

         inner.playback = Some(PlaybackContext {
            sink,
            source: descriptor.clone(),
            duration,
            position_offset: 0.0,
            position_latency: opened_source.position_latency,
            seek_strategy: opened_source.seek_strategy,
            resume_fade: opened_source.resume_fade,
         });

         return Ok(inner.state.clone());
      }
   }

   pub fn play(&self) -> Result<AudioActionResponse> {
      enum PlayAction {
         Complete(PlayerState),
         Reopen {
            source_descriptor: SourceDescriptor,
            playback_rate: f64,
            load_generation: u64,
            seek_generation: u64,
            expected_status: PlaybackStatus,
         },
      }

      let action = {
         let mut inner = lock_inner(&self.inner);
         let was_paused = inner.state.status == PlaybackStatus::Paused;
         let status = inner.state.status;
         if let Some(source_descriptor) = inner
            .playback
            .as_ref()
            .filter(|ctx| playback_requires_restart_from_start(status, ctx.sink.empty()))
            .map(|ctx| ctx.source.clone())
         {
            PlayAction::Reopen {
               source_descriptor,
               playback_rate: inner.state.playback_rate,
               load_generation: inner.load_generation,
               seek_generation: inner.seek_generation,
               expected_status: status,
            }
         } else {
            transitions::play(&mut inner.state)?;

            if let Some(ctx) = &inner.playback {
               if was_paused && let Some(resume_fade) = &ctx.resume_fade {
                  resume_fade.request_fade_in();
               }

               ctx.sink.play();
            }

            self.start_monitor(&mut inner);
            PlayAction::Complete(inner.state.clone())
         }
      };

      let snapshot = match action {
         PlayAction::Complete(snapshot) => snapshot,
         PlayAction::Reopen {
            source_descriptor,
            playback_rate,
            load_generation,
            seek_generation,
            expected_status,
         } => self.reopen_ended_playback_from_start(
            source_descriptor,
            playback_rate,
            load_generation,
            seek_generation,
            expected_status,
         )?,
      };

      self.emit_state_changed(&snapshot);
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

      self.emit_state_changed(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Paused))
   }

   pub fn stop(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);

         transitions::stop(&mut inner.state)?;
         inner.load_generation = inner.load_generation.wrapping_add(1);
         inner.seek_generation = inner.seek_generation.wrapping_add(1);

         Self::stop_monitor(&inner);

         if let Some(ctx) = inner.playback.take() {
            ctx.sink.stop();
         }

         inner.state.clone()
      };

      self.emit_state_changed(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Idle))
   }

   pub fn seek(&self, position: f64) -> Result<AudioActionResponse> {
      enum SeekAction {
         Complete(PlayerState),
         Reopen {
            source_descriptor: SourceDescriptor,
            target_time: f64,
            previous_time: f64,
            seek_generation: u64,
         },
      }

      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         let previous_time = inner.state.current_time;
         let status = inner.state.status;

         transitions::seek(&mut inner.state, position)?;
         inner.seek_generation = inner.seek_generation.wrapping_add(1);
         let seek_generation = inner.seek_generation;

         let action =
            if let Some((source_descriptor, seek_strategy)) = inner
               .playback
               .as_ref()
               .map(|ctx| (ctx.source.clone(), ctx.seek_strategy))
            {
               // Check the current state playback rate as well as the stored seek strategy.
               // During a playback-rate reopen, inner.playback can still point at the old
               // 1.0x PlaybackContext while inner.state.playback_rate has already been updated
               // to the requested non-1.0x value. In that window, forcing reopen avoids
               // seeking the stale sink locally and leaving state ahead of actual playback.
               if seek_can_use_direct_strategy(seek_strategy, inner.state.playback_rate) {
                  if inner.playback.as_ref().is_some_and(|ctx| {
                     playback_requires_restart_from_start(status, ctx.sink.empty())
                  }) {
                     SeekAction::Reopen {
                        source_descriptor,
                        target_time: inner.state.current_time,
                        previous_time,
                        seek_generation,
                     }
                  } else {
                     Self::seek_local_playback(&mut inner, previous_time)?;
                     SeekAction::Complete(inner.state.clone())
                  }
               } else {
                  Self::stop_monitor(&inner);
                  if let Some(ctx) = &inner.playback {
                     ctx.sink.pause();
                  }

                  SeekAction::Reopen {
                     source_descriptor,
                     target_time: inner.state.current_time,
                     previous_time,
                     seek_generation,
                  }
               }
            } else {
               SeekAction::Complete(inner.state.clone())
            };

         match action {
            SeekAction::Complete(snapshot) => snapshot,
            SeekAction::Reopen {
               source_descriptor,
               target_time,
               previous_time,
               seek_generation,
            } => {
               drop(inner);
               self.reopen_playback_at(
                  source_descriptor,
                  target_time,
                  previous_time,
                  None,
                  seek_generation,
               )?
            }
         }
      };

      let expected = snapshot.status;
      self.emit_state_changed(&snapshot);
      Ok(AudioActionResponse::new(snapshot, expected))
   }

   fn reopen_playback_at(
      &self,
      source_descriptor: SourceDescriptor,
      target_time: f64,
      previous_time: f64,
      previous_playback_rate: Option<f64>,
      seek_generation: u64,
   ) -> Result<PlayerState> {
      let playback_rate = lock_inner(&self.inner).state.playback_rate;
      let source = open_source_at(&source_descriptor, target_time, playback_rate);

      let opened_source = match source {
         Ok(source) => source,
         Err(error) => {
            let mut inner = lock_inner(&self.inner);

            if inner.seek_generation == seek_generation {
               restore_reopen_failure_state(
                  &mut inner.state,
                  previous_time,
                  previous_playback_rate,
               );

               if let Some(ctx) = &inner.playback {
                  ctx.sink.set_volume(effective_volume(&inner.state));
                  if inner.state.status == PlaybackStatus::Playing {
                     ctx.sink.play();
                  }
               }

               if inner.state.status == PlaybackStatus::Playing {
                  self.start_monitor(&mut inner);
               }
            }

            return Err(error);
         }
      };

      let sink = Player::connect_new(self.output_sink.mixer());
      sink.pause();
      let duration = opened_source.duration;
      let position_latency = opened_source.position_latency;
      let seek_strategy = opened_source.seek_strategy;
      let resume_fade = opened_source.resume_fade;
      sink.append(opened_source.source);

      let mut inner = lock_inner(&self.inner);

      if inner.seek_generation != seek_generation {
         sink.stop();
         return Err(Error::InvalidState("Seek request was canceled".into()));
      }

      sink.set_volume(effective_volume(&inner.state));

      if let Some(previous_playback) = inner.playback.replace(PlaybackContext {
         sink,
         source: source_descriptor,
         duration,
         position_offset: target_time,
         position_latency,
         seek_strategy,
         resume_fade,
      }) {
         previous_playback.sink.stop();
      }

      if inner.state.status == PlaybackStatus::Playing {
         if let Some(ctx) = &inner.playback {
            ctx.sink.play();
         }
         self.start_monitor(&mut inner);
      }

      Ok(inner.state.clone())
   }

   /// Reopens an ended track from the beginning for play(), without holding the
   /// player mutex during I/O. Unlike reopen_playback_at(), this always resumes
   /// from 0.0 and re-applies the play transition instead of preserving a seek target.
   fn reopen_ended_playback_from_start(
      &self,
      source_descriptor: SourceDescriptor,
      playback_rate: f64,
      load_generation: u64,
      seek_generation: u64,
      expected_status: PlaybackStatus,
   ) -> Result<PlayerState> {
      let opened_source = open_source_at(&source_descriptor, 0.0, playback_rate)?;
      let mut inner = lock_inner(&self.inner);

      if inner.load_generation != load_generation || inner.seek_generation != seek_generation {
         return Err(Error::InvalidState("Play request was canceled".into()));
      }

      if inner.state.status != expected_status {
         return Err(Error::InvalidState("Play request was canceled".into()));
      }

      let position_latency = opened_source.position_latency;
      let seek_strategy = opened_source.seek_strategy;
      let resume_fade = opened_source.resume_fade;

      {
         let Some(ctx) = &mut inner.playback else {
            return Err(Error::InvalidState("Play request was canceled".into()));
         };

         if !ctx.sink.empty() {
            return Err(Error::InvalidState("Play request was canceled".into()));
         }

         ctx.sink.append(opened_source.source);
         ctx.sink.pause();
         ctx.position_offset = 0.0;
         ctx.position_latency = position_latency;
         ctx.seek_strategy = seek_strategy;
         ctx.resume_fade = resume_fade;
      }

      transitions::play(&mut inner.state)?;
      inner.state.current_time = 0.0;

      if let Some(ctx) = &inner.playback {
         if expected_status == PlaybackStatus::Paused
            && let Some(resume_fade) = &ctx.resume_fade
         {
            resume_fade.request_fade_in();
         }

         ctx.sink.play();
      }

      self.start_monitor(&mut inner);

      Ok(inner.state.clone())
   }

   fn seek_local_playback(inner: &mut Inner, previous_time: f64) -> Result<()> {
      let Some(ctx) = &mut inner.playback else {
         unreachable!("Playback context disappeared during local seek");
      };

      if let Err(e) = ctx
         .sink
         .try_seek(Duration::from_secs_f64(inner.state.current_time))
      {
         inner.state.current_time = previous_time;
         return Err(Error::Audio(format!("Failed to seek audio: {e}")));
      }

      ctx.position_offset = 0.0;

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

      self.emit_state_changed(&snapshot);
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

      self.emit_state_changed(&snapshot);
      snapshot
   }

   pub fn set_playback_rate(&self, rate: f64) -> Result<PlayerState> {
      enum PlaybackRateAction {
         Complete(PlayerState),
         Reopen {
            source_descriptor: SourceDescriptor,
            target_time: f64,
            previous_time: f64,
            previous_playback_rate: f64,
            seek_generation: u64,
         },
      }

      let action = {
         let mut inner = lock_inner(&self.inner);
         let previous_time = inner.state.current_time;
         let previous_playback_rate = inner.state.playback_rate;

         transitions::set_playback_rate(&mut inner.state, rate)?;

         if !playback_rate_change_requires_reopen(previous_playback_rate, inner.state.playback_rate)
         {
            PlaybackRateAction::Complete(inner.state.clone())
         } else if let Some(source_descriptor) =
            inner.playback.as_ref().map(|ctx| ctx.source.clone())
         {
            inner.seek_generation = inner.seek_generation.wrapping_add(1);
            let seek_generation = inner.seek_generation;

            Self::stop_monitor(&inner);
            if let Some(ctx) = &inner.playback {
               ctx.sink.pause();
            }

            PlaybackRateAction::Reopen {
               source_descriptor,
               target_time: inner.state.current_time,
               previous_time,
               previous_playback_rate,
               seek_generation,
            }
         } else {
            PlaybackRateAction::Complete(inner.state.clone())
         }
      };

      let snapshot = match action {
         PlaybackRateAction::Complete(snapshot) => snapshot,
         PlaybackRateAction::Reopen {
            source_descriptor,
            target_time,
            previous_time,
            previous_playback_rate,
            seek_generation,
         } => self.reopen_playback_at(
            source_descriptor,
            target_time,
            previous_time,
            Some(previous_playback_rate),
            seek_generation,
         )?,
      };

      self.emit_state_changed(&snapshot);
      Ok(snapshot)
   }

   pub fn set_loop_mode(&self, mode: LoopMode) -> PlayerState {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_loop_mode(&mut inner.state, mode);
         inner.state.clone()
      };

      self.emit_state_changed(&snapshot);
      snapshot
   }

   /// Advance to the next playlist item, with wrap-around if `loopMode` is `All`.
   /// Errors with `InvalidState` from `Idle` / `Loading`.
   pub fn next(&self) -> Result<AudioActionResponse> {
      self.navigate(Direction::Next)
   }

   /// Move to the previous item, or restart the current item if `currentTime > 3s`
   /// or we're at the start of a non-looping playlist.
   pub fn prev(&self) -> Result<AudioActionResponse> {
      self.navigate(Direction::Prev)
   }

   /// Jump to a specific item in the loaded playlist by index.
   ///
   /// Errors with `InvalidState` from `Idle` / `Loading` and with
   /// `InvalidValue` if the index is out of range. Jumping to the currently
   /// active index restarts that item from the beginning.
   pub fn jump_to(&self, index: usize) -> Result<AudioActionResponse> {
      let target = {
         let inner = lock_inner(&self.inner);
         transitions::jump_target(&inner.state, index)?
      };

      match target {
         NavTarget::Index(idx) => self.advance_to(idx),
         NavTarget::RestartCurrent => self.seek(0.0),
         NavTarget::End => self.transition_to_ended(),
      }
   }

   fn navigate(&self, direction: Direction) -> Result<AudioActionResponse> {
      let target = {
         let inner = lock_inner(&self.inner);
         match direction {
            Direction::Next => transitions::next_target(&inner.state)?,
            Direction::Prev => transitions::prev_target(&inner.state)?,
         }
      };

      match target {
         NavTarget::Index(index) => self.advance_to(index),
         NavTarget::RestartCurrent => self.seek(0.0),
         NavTarget::End => self.transition_to_ended(),
      }
   }

   /// Loads `playlist[index]` while preserving play intent — if the player was
   /// `Playing` before the call, it resumes playing the new item once loaded.
   ///
   /// Always emits a `Loading` state-changed event before fetching, even when
   /// the source bytes are already cached. The cache makes the Loading window
   /// effectively instantaneous, but the event is preserved so consumers see
   /// identical state-machine emissions on desktop and on native mobile
   /// platforms (iOS AVPlayer / Android ExoPlayer always emit Loading).
   fn advance_to(&self, index: usize) -> Result<AudioActionResponse> {
      let (was_playing, item_src, load_generation) = {
         let mut inner = lock_inner(&self.inner);
         let was_playing = inner.state.status == PlaybackStatus::Playing;

         transitions::begin_load_index(&mut inner.state, index)?;
         inner.load_generation = inner.load_generation.wrapping_add(1);
         inner.seek_generation = inner.seek_generation.wrapping_add(1);
         let load_generation = inner.load_generation;

         let src = inner
            .state
            .current()
            .map(|item| item.src.clone())
            .ok_or_else(|| Error::InvalidState("Missing current item after begin_load".into()))?;

         Self::stop_monitor(&inner);
         if let Some(ctx) = inner.playback.take() {
            ctx.sink.stop();
         }

         let snapshot = inner.state.clone();
         drop(inner);
         self.emit_state_changed(&snapshot);
         (was_playing, src, load_generation)
      };

      let ready_snapshot = match self.load_inner(&item_src, load_generation) {
         Ok(snapshot) => {
            self.emit_state_changed(&snapshot);
            snapshot
         }
         Err(error) => {
            let snapshot = {
               let mut inner = lock_inner(&self.inner);
               finish_load_as_error(&mut inner, load_generation, error.to_string())
            };
            if let Some(snapshot) = snapshot {
               self.emit_state_changed(&snapshot);
            }
            return Err(error);
         }
      };

      if was_playing {
         self.play()
      } else {
         Ok(AudioActionResponse::new(
            ready_snapshot,
            PlaybackStatus::Ready,
         ))
      }
   }

   /// Transitions to `Ended` when navigation falls off the end of a non-looping
   /// playlist.
   fn transition_to_ended(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);

         transitions::ended(&mut inner.state);

         Self::stop_monitor(&inner);
         if let Some(ctx) = &inner.playback {
            ctx.sink.pause();
         }

         inner.state.clone()
      };

      self.emit_state_changed(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Ended))
   }
}

#[derive(Copy, Clone)]
enum Direction {
   Next,
   Prev,
}

// ---------------------------------------------------------------------------
// Audio output
// ---------------------------------------------------------------------------

/// Opens the default audio output device for playback.
fn open_audio_output() -> Result<MixerDeviceSink> {
   DeviceSinkBuilder::open_default_sink()
      .map_err(|e| Error::Audio(format!("Failed to open audio device: {e}")))
}

fn finish_playback_as_ended(inner: &mut Inner, duration: f64, position: f64) -> PlayerState {
   inner.state.status = PlaybackStatus::Ended;
   inner.state.current_time = if duration > 0.0 { duration } else { position };
   inner.state.clone()
}

fn finish_load_as_error(
   inner: &mut Inner,
   load_generation: u64,
   message: String,
) -> Option<PlayerState> {
   if inner.load_generation != load_generation || inner.state.status != PlaybackStatus::Loading {
      return None;
   }

   transitions::error(&mut inner.state, message);
   inner.playback = None;

   Some(inner.state.clone())
}

// ---------------------------------------------------------------------------
// Playback monitor
// ---------------------------------------------------------------------------

/// Polls the sink every 250ms for position updates and end-of-track detection.
///
/// On end-of-track, consults [`transitions::auto_advance_target`] to decide
/// whether to restart the current item ([`LoopMode::One`]), advance to the
/// next playlist item (with [`LoopMode::All`] wrap-around), or transition to
/// `Ended`.
fn monitor_loop(stop: Arc<AtomicBool>, player: Arc<RodioAudioPlayer>) {
   loop {
      std::thread::sleep(Duration::from_millis(250));

      if stop.load(Ordering::Relaxed) {
         break;
      }

      let mut guard = lock_inner(&player.inner);

      let (pos, duration, is_empty) = match &guard.playback {
         Some(ctx) => {
            let sink_pos = ctx.sink.get_pos().as_secs_f64();
            let audible_sink_pos = (sink_pos - ctx.position_latency).max(0.0);
            let pos = ctx.position_offset + (audible_sink_pos * guard.state.playback_rate);
            (pos, ctx.duration, ctx.sink.empty())
         }
         None => break,
      };

      if is_empty {
         match transitions::auto_advance_target(&guard.state) {
            NavTarget::RestartCurrent => {
               // Re-append source for seamless (best-effort) loop.
               let playback_rate = guard.state.playback_rate;
               let load_generation = guard.load_generation;
               let seek_generation = guard.seek_generation;
               let Some(source_descriptor) = guard.playback.as_ref().map(|ctx| ctx.source.clone())
               else {
                  break;
               };
               drop(guard);

               let reopened_source = open_source_at(&source_descriptor, 0.0, playback_rate);

               let mut guard = lock_inner(&player.inner);

               match classify_loop_reopen_attempt(&guard, &stop, load_generation, seek_generation) {
                  LoopReopenAttempt::Apply => {}
                  LoopReopenAttempt::Retry => continue,
                  LoopReopenAttempt::Break => break,
               }

               // Discard stale reopen work if this monitor was stopped or playback
               // changed state while open_source_at() was running.
               if stop.load(Ordering::Relaxed) || guard.state.status != PlaybackStatus::Playing {
                  break;
               }

               // Loop mode may have changed during the reopen window.
               if !matches!(
                  transitions::auto_advance_target(&guard.state),
                  NavTarget::RestartCurrent
               ) {
                  let snapshot = finish_playback_as_ended(&mut guard, duration, pos);
                  drop(guard);
                  player.emit_state_changed(&snapshot);
                  break;
               }

               match reopened_source {
                  Ok(source) => {
                     let position_latency = source.position_latency;
                     let seek_strategy = source.seek_strategy;
                     let resume_fade = source.resume_fade;

                     if let Some(ctx) = &mut guard.playback {
                        ctx.sink.append(source.source);
                        ctx.position_offset = 0.0;
                        ctx.position_latency = position_latency;
                        ctx.seek_strategy = seek_strategy;
                        ctx.resume_fade = resume_fade;
                     } else {
                        break;
                     }

                     guard.state.current_time = 0.0;
                     drop(guard);
                     (player.on_time_update)(&TimeUpdate {
                        current_time: 0.0,
                        duration,
                     });
                  }
                  Err(e) => {
                     warn!("Failed to reopen loop source: {e}");
                     let snapshot = finish_playback_as_ended(&mut guard, duration, pos);
                     drop(guard);
                     player.emit_state_changed(&snapshot);
                     break;
                  }
               }
            }
            NavTarget::Index(idx) => {
               drop(guard);
               if let Err(e) = player.advance_to(idx) {
                  warn!("Auto-advance failed: {e}");
               }
               // `advance_to` (via `play`) starts a fresh monitor; this one exits.
               break;
            }
            NavTarget::End => {
               let snapshot = finish_playback_as_ended(&mut guard, duration, pos);
               drop(guard);
               player.emit_state_changed(&snapshot);
               break;
            }
         }
      } else {
         guard.state.current_time = pos;
         drop(guard);
         (player.on_time_update)(&TimeUpdate {
            current_time: pos,
            duration,
         });
      }
   }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum LoopReopenAttempt {
   Apply,
   Retry,
   Break,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadPlaybackRateRetryStage {
   NotRetried,
   RetriedOnce,
   RetriedTwice,
}

#[derive(Debug, PartialEq)]
enum LoadPlaybackRateChangeAction {
   Apply,
   RetryDuringLoad {
      next_playback_rate: f64,
      retry_stage: LoadPlaybackRateRetryStage,
   },
   Exhausted,
}

fn classify_loop_reopen_attempt(
   inner: &Inner,
   stop: &Arc<AtomicBool>,
   load_generation: u64,
   seek_generation: u64,
) -> LoopReopenAttempt {
   if inner.load_generation != load_generation || !Arc::ptr_eq(&inner.monitor_stop, stop) {
      LoopReopenAttempt::Break
   } else if inner.seek_generation != seek_generation {
      LoopReopenAttempt::Retry
   } else {
      LoopReopenAttempt::Apply
   }
}

fn classify_load_playback_rate_change(
   current_requested_playback_rate: f64,
   attempted_playback_rate: f64,
   retry_stage: LoadPlaybackRateRetryStage,
) -> LoadPlaybackRateChangeAction {
   if !playback_rate_change_requires_reopen(
      attempted_playback_rate,
      current_requested_playback_rate,
   ) {
      LoadPlaybackRateChangeAction::Apply
   } else {
      match retry_stage {
         LoadPlaybackRateRetryStage::NotRetried => LoadPlaybackRateChangeAction::RetryDuringLoad {
            next_playback_rate: current_requested_playback_rate,
            retry_stage: LoadPlaybackRateRetryStage::RetriedOnce,
         },
         LoadPlaybackRateRetryStage::RetriedOnce => LoadPlaybackRateChangeAction::RetryDuringLoad {
            next_playback_rate: current_requested_playback_rate,
            retry_stage: LoadPlaybackRateRetryStage::RetriedTwice,
         },
         LoadPlaybackRateRetryStage::RetriedTwice => LoadPlaybackRateChangeAction::Exhausted,
      }
   }
}

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

fn playback_requires_restart_from_start(status: PlaybackStatus, sink_empty: bool) -> bool {
   sink_empty && matches!(status, PlaybackStatus::Ended | PlaybackStatus::Paused)
}

fn restore_reopen_failure_state(
   state: &mut PlayerState,
   previous_time: f64,
   previous_playback_rate: Option<f64>,
) {
   state.current_time = previous_time;

   if let Some(previous_playback_rate) = previous_playback_rate {
      state.playback_rate = previous_playback_rate;
   }
}

fn playback_rate_change_requires_reopen(previous_playback_rate: f64, playback_rate: f64) -> bool {
   (playback_rate - previous_playback_rate).abs() > f64::EPSILON
}

fn seek_can_use_direct_strategy(seek_strategy: SeekStrategy, playback_rate: f64) -> bool {
   matches!(seek_strategy, SeekStrategy::Direct) && (playback_rate - 1.0).abs() <= f64::EPSILON
}

#[cfg(test)]
mod tests {
   use super::*;
   use std::sync::atomic::AtomicBool;
   use std::sync::{Arc, Mutex};

   fn inner_with_state(state: PlayerState) -> Inner {
      Inner {
         state,
         playback: None,
         monitor_stop: Arc::new(AtomicBool::new(true)),
         load_generation: 0,
         seek_generation: 0,
      }
   }

   #[test]
   fn playback_rate_change_requires_reopen_is_false_when_clamping_keeps_same_rate() {
      let mut state = PlayerState {
         playback_rate: 0.25,
         ..Default::default()
      };
      let previous_playback_rate = state.playback_rate;

      transitions::set_playback_rate(&mut state, 0.0).unwrap();

      assert!(!playback_rate_change_requires_reopen(
         previous_playback_rate,
         state.playback_rate,
      ));
   }

   #[test]
   fn restore_reopen_failure_state_keeps_playback_rate_for_seek_failures() {
      let mut state = PlayerState {
         status: PlaybackStatus::Playing,
         current_time: 12.0,
         playback_rate: 1.5,
         ..Default::default()
      };

      restore_reopen_failure_state(&mut state, 4.0, None);

      assert_eq!(state.status, PlaybackStatus::Playing);
      assert_eq!(state.current_time, 4.0);
      assert_eq!(state.playback_rate, 1.5);
   }

   #[test]
   fn restore_reopen_failure_state_restores_previous_playback_rate() {
      let mut state = PlayerState {
         status: PlaybackStatus::Playing,
         current_time: 12.0,
         playback_rate: 2.0,
         ..Default::default()
      };

      restore_reopen_failure_state(&mut state, 4.0, Some(1.25));

      assert_eq!(state.status, PlaybackStatus::Playing);
      assert_eq!(state.current_time, 4.0);
      assert_eq!(state.playback_rate, 1.25);
   }

   #[test]
   fn finish_playback_as_ended_uses_duration_when_known() {
      let mut inner = inner_with_state(PlayerState {
         status: PlaybackStatus::Playing,
         current_time: 42.0,
         ..Default::default()
      });

      let snapshot = finish_playback_as_ended(&mut inner, 120.0, 119.25);

      assert_eq!(snapshot.status, PlaybackStatus::Ended);
      assert_eq!(snapshot.current_time, 120.0);
      assert_eq!(inner.state.status, PlaybackStatus::Ended);
      assert_eq!(inner.state.current_time, 120.0);
   }

   #[test]
   fn finish_playback_as_ended_falls_back_to_position_when_duration_unknown() {
      let mut inner = inner_with_state(PlayerState {
         status: PlaybackStatus::Playing,
         current_time: 42.0,
         ..Default::default()
      });

      let snapshot = finish_playback_as_ended(&mut inner, 0.0, 119.25);

      assert_eq!(snapshot.status, PlaybackStatus::Ended);
      assert_eq!(snapshot.current_time, 119.25);
      assert_eq!(inner.state.status, PlaybackStatus::Ended);
      assert_eq!(inner.state.current_time, 119.25);
   }

   #[test]
   fn finish_load_as_error_updates_matching_loading_request() {
      let mut inner = inner_with_state(PlayerState {
         status: PlaybackStatus::Loading,
         playlist: vec![PlaylistItem {
            src: "fixture.mp3".to_string(),
            metadata: None,
         }],
         current_index: Some(0),
         ..Default::default()
      });
      inner.load_generation = 7;

      let snapshot = finish_load_as_error(&mut inner, 7, "decode failed".to_string())
         .expect("matching loading request should transition to error");

      assert_eq!(snapshot.status, PlaybackStatus::Error);
      assert_eq!(snapshot.error.as_deref(), Some("decode failed"));
      assert_eq!(inner.state.status, PlaybackStatus::Error);
      assert_eq!(inner.state.error.as_deref(), Some("decode failed"));
   }

   #[test]
   fn finish_load_as_error_ignores_stale_or_non_loading_requests() {
      let mut stale_inner = inner_with_state(PlayerState {
         status: PlaybackStatus::Loading,
         ..Default::default()
      });
      stale_inner.load_generation = 7;

      let stale_snapshot = finish_load_as_error(&mut stale_inner, 6, "decode failed".to_string());

      assert!(stale_snapshot.is_none());
      assert_eq!(stale_inner.state.status, PlaybackStatus::Loading);
      assert!(stale_inner.state.error.is_none());

      let mut ready_inner = inner_with_state(PlayerState {
         status: PlaybackStatus::Ready,
         ..Default::default()
      });
      ready_inner.load_generation = 7;

      let ready_snapshot = finish_load_as_error(&mut ready_inner, 7, "decode failed".to_string());

      assert!(ready_snapshot.is_none());
      assert_eq!(ready_inner.state.status, PlaybackStatus::Ready);
      assert!(ready_inner.state.error.is_none());
   }

   #[test]
   fn lock_inner_recovers_from_poisoned_mutex() {
      let mutex = Arc::new(Mutex::new(inner_with_state(PlayerState {
         status: PlaybackStatus::Ready,
         volume: 0.4,
         ..Default::default()
      })));
      let mutex_for_thread = Arc::clone(&mutex);

      let _ = std::thread::spawn(move || {
         let _guard = mutex_for_thread.lock().unwrap();
         panic!("poison test mutex");
      })
      .join();

      let guard = lock_inner(mutex.as_ref());

      assert_eq!(guard.state.status, PlaybackStatus::Ready);
      assert_eq!(guard.state.volume, 0.4);
   }
}
