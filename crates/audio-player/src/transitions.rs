//! State machine transition rules for the audio player.
//!
//! These functions are the single source of truth for which state transitions
//! are valid and how `PlayerState` fields are mutated. [`RodioAudioPlayer`]
//! delegates to them.

use crate::error::{Error, Result};
use crate::models::{AudioMetadata, PlaybackStatus, PlayerState};

// ---------------------------------------------------------------------------
// Transport actions
// ---------------------------------------------------------------------------

/// Transitions to [`PlaybackStatus::Loading`] and stores metadata.
///
/// Call this before starting I/O so the frontend can show a loading indicator.
/// After the I/O completes, call [`load`] to finalize the transition to `Ready`.
pub fn begin_load(state: &mut PlayerState, src: &str, meta: &AudioMetadata) -> Result<()> {
   match state.status {
      PlaybackStatus::Idle | PlaybackStatus::Ended | PlaybackStatus::Error => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot load in {:?} state",
            state.status
         )));
      }
   }

   state.status = PlaybackStatus::Loading;
   state.src = Some(src.to_string());
   state.title = meta.title.clone();
   state.artist = meta.artist.clone();
   state.artwork = meta.artwork.clone();
   state.current_time = 0.0;
   state.duration = 0.0;
   state.error = None;
   Ok(())
}

/// Finalizes a load by transitioning from `Loading` to `Ready` with the
/// decoded duration. Also accepts `Idle`, `Ended`, and `Error` in case
/// `begin_load` was skipped (e.g. instant local file loads).
pub fn load(state: &mut PlayerState, src: &str, meta: &AudioMetadata, duration: f64) -> Result<()> {
   match state.status {
      PlaybackStatus::Loading
      | PlaybackStatus::Idle
      | PlaybackStatus::Ended
      | PlaybackStatus::Error => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot load in {:?} state",
            state.status
         )));
      }
   }

   state.status = PlaybackStatus::Ready;
   state.src = Some(src.to_string());
   state.title = meta.title.clone();
   state.artist = meta.artist.clone();
   state.artwork = meta.artwork.clone();
   state.current_time = 0.0;
   state.duration = duration;
   state.error = None;
   Ok(())
}

/// Validates and applies the play transition.
pub fn play(state: &mut PlayerState) -> Result<()> {
   match state.status {
      PlaybackStatus::Ready | PlaybackStatus::Paused | PlaybackStatus::Ended => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot play in {:?} state",
            state.status
         )));
      }
   }
   state.status = PlaybackStatus::Playing;
   Ok(())
}

/// Validates and applies the pause transition.
pub fn pause(state: &mut PlayerState) -> Result<()> {
   match state.status {
      PlaybackStatus::Playing => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot pause in {:?} state",
            state.status
         )));
      }
   }
   state.status = PlaybackStatus::Paused;
   Ok(())
}

/// Validates and applies the stop transition, preserving user settings.
pub fn stop(state: &mut PlayerState) -> Result<()> {
   match state.status {
      PlaybackStatus::Loading
      | PlaybackStatus::Ready
      | PlaybackStatus::Playing
      | PlaybackStatus::Paused
      | PlaybackStatus::Ended => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot stop in {:?} state",
            state.status
         )));
      }
   }
   *state = PlayerState {
      volume: state.volume,
      muted: state.muted,
      playback_rate: state.playback_rate,
      looping: state.looping,
      ..Default::default()
   };
   Ok(())
}

/// Validates and applies the seek transition. Preserves the current status.
pub fn seek(state: &mut PlayerState, position: f64) -> Result<()> {
   if !position.is_finite() {
      return Err(Error::InvalidValue(format!(
         "Seek position must be finite, got {position}"
      )));
   }
   match state.status {
      PlaybackStatus::Ready
      | PlaybackStatus::Playing
      | PlaybackStatus::Paused
      | PlaybackStatus::Ended => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot seek in {:?} state",
            state.status
         )));
      }
   }
   state.current_time = if state.duration.is_finite() && state.duration > 0.0 {
      position.clamp(0.0, state.duration)
   } else {
      position.max(0.0)
   };
   Ok(())
}

/// Transitions to [`PlaybackStatus::Error`] with a message.
///
/// Valid from `Loading` (I/O or decode failure during load). Other statuses are
/// left unchanged — callers should only invoke this when a load operation fails
/// after `begin_load` has already moved the state to `Loading`.
pub fn error(state: &mut PlayerState, message: String) {
   if state.status == PlaybackStatus::Loading {
      state.status = PlaybackStatus::Error;
      state.error = Some(message);
   }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Validates and applies a volume change.
pub fn set_volume(state: &mut PlayerState, level: f64) -> Result<()> {
   if !level.is_finite() {
      return Err(Error::InvalidValue(format!(
         "Volume must be finite, got {level}"
      )));
   }
   state.volume = level.clamp(0.0, 1.0);
   Ok(())
}

/// Applies a mute toggle.
pub fn set_muted(state: &mut PlayerState, muted: bool) {
   state.muted = muted;
}

/// Validates and applies a playback rate change.
pub fn set_playback_rate(state: &mut PlayerState, rate: f64) -> Result<()> {
   if !rate.is_finite() {
      return Err(Error::InvalidValue(format!(
         "Playback rate must be finite, got {rate}"
      )));
   }
   state.playback_rate = rate.clamp(0.25, 4.0);
   Ok(())
}

/// Applies a loop toggle.
pub fn set_loop(state: &mut PlayerState, looping: bool) {
   state.looping = looping;
}

#[cfg(test)]
mod tests {
   use super::*;

   fn state_with_status(status: PlaybackStatus) -> PlayerState {
      PlayerState {
         status,
         ..Default::default()
      }
   }

   fn state_with_duration(status: PlaybackStatus, duration: f64) -> PlayerState {
      PlayerState {
         status,
         duration,
         ..Default::default()
      }
   }

   fn meta(title: &str) -> AudioMetadata {
      AudioMetadata {
         title: Some(title.to_string()),
         artist: None,
         artwork: None,
      }
   }

   // -- begin_load --

   #[test]
   fn begin_load_from_idle() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      begin_load(&mut s, "test.mp3", &meta("Song")).unwrap();

      assert_eq!(s.status, PlaybackStatus::Loading);
      assert_eq!(s.src.as_deref(), Some("test.mp3"));
      assert_eq!(s.title.as_deref(), Some("Song"));
      assert_eq!(s.duration, 0.0);
      assert_eq!(s.current_time, 0.0);
      assert!(s.error.is_none());
   }

   #[test]
   fn begin_load_from_ended() {
      let mut s = state_with_status(PlaybackStatus::Ended);
      assert!(begin_load(&mut s, "a.mp3", &AudioMetadata::default()).is_ok());
      assert_eq!(s.status, PlaybackStatus::Loading);
   }

   #[test]
   fn begin_load_from_error() {
      let mut s = state_with_status(PlaybackStatus::Error);
      assert!(begin_load(&mut s, "a.mp3", &AudioMetadata::default()).is_ok());
      assert_eq!(s.status, PlaybackStatus::Loading);
   }

   #[test]
   fn begin_load_rejected_from_loading() {
      let mut s = state_with_status(PlaybackStatus::Loading);
      assert!(begin_load(&mut s, "a.mp3", &AudioMetadata::default()).is_err());
      assert_eq!(s.status, PlaybackStatus::Loading);
   }

   #[test]
   fn begin_load_rejected_from_ready() {
      let mut s = state_with_status(PlaybackStatus::Ready);
      assert!(begin_load(&mut s, "a.mp3", &AudioMetadata::default()).is_err());
   }

   #[test]
   fn begin_load_rejected_from_playing() {
      let mut s = state_with_status(PlaybackStatus::Playing);
      assert!(begin_load(&mut s, "a.mp3", &AudioMetadata::default()).is_err());
   }

   #[test]
   fn begin_load_rejected_from_paused() {
      let mut s = state_with_status(PlaybackStatus::Paused);
      assert!(begin_load(&mut s, "a.mp3", &AudioMetadata::default()).is_err());
   }

   // -- load (finalize) --

   #[test]
   fn load_from_loading() {
      let mut s = state_with_status(PlaybackStatus::Loading);
      load(&mut s, "test.mp3", &meta("Song"), 120.0).unwrap();

      assert_eq!(s.status, PlaybackStatus::Ready);
      assert_eq!(s.src.as_deref(), Some("test.mp3"));
      assert_eq!(s.title.as_deref(), Some("Song"));
      assert_eq!(s.duration, 120.0);
      assert_eq!(s.current_time, 0.0);
      assert!(s.error.is_none());
   }

   #[test]
   fn load_from_idle() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      load(&mut s, "a.mp3", &AudioMetadata::default(), 0.0).unwrap();
      assert_eq!(s.status, PlaybackStatus::Ready);
   }

   #[test]
   fn load_from_ended() {
      let mut s = state_with_status(PlaybackStatus::Ended);
      assert!(load(&mut s, "a.mp3", &AudioMetadata::default(), 0.0).is_ok());
      assert_eq!(s.status, PlaybackStatus::Ready);
   }

   #[test]
   fn load_from_error() {
      let mut s = state_with_status(PlaybackStatus::Error);
      assert!(load(&mut s, "a.mp3", &AudioMetadata::default(), 0.0).is_ok());
      assert_eq!(s.status, PlaybackStatus::Ready);
   }

   #[test]
   fn load_rejected_from_ready() {
      let mut s = state_with_status(PlaybackStatus::Ready);
      assert!(load(&mut s, "a.mp3", &AudioMetadata::default(), 0.0).is_err());
   }

   #[test]
   fn load_rejected_from_playing() {
      let mut s = state_with_status(PlaybackStatus::Playing);
      assert!(load(&mut s, "a.mp3", &AudioMetadata::default(), 0.0).is_err());
   }

   #[test]
   fn load_rejected_from_paused() {
      let mut s = state_with_status(PlaybackStatus::Paused);
      assert!(load(&mut s, "a.mp3", &AudioMetadata::default(), 0.0).is_err());
   }

   // -- play --

   #[test]
   fn play_from_ready() {
      let mut s = state_with_status(PlaybackStatus::Ready);
      play(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   #[test]
   fn play_from_paused() {
      let mut s = state_with_status(PlaybackStatus::Paused);
      play(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   #[test]
   fn play_from_ended() {
      let mut s = state_with_status(PlaybackStatus::Ended);
      play(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   #[test]
   fn play_rejected_from_idle() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      assert!(play(&mut s).is_err());
      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn play_rejected_from_loading() {
      let mut s = state_with_status(PlaybackStatus::Loading);
      assert!(play(&mut s).is_err());
   }

   #[test]
   fn play_rejected_from_playing() {
      let mut s = state_with_status(PlaybackStatus::Playing);
      assert!(play(&mut s).is_err());
   }

   #[test]
   fn play_rejected_from_error() {
      let mut s = state_with_status(PlaybackStatus::Error);
      assert!(play(&mut s).is_err());
   }

   // -- pause --

   #[test]
   fn pause_from_playing() {
      let mut s = state_with_status(PlaybackStatus::Playing);
      pause(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Paused);
   }

   #[test]
   fn pause_rejected_from_idle() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      assert!(pause(&mut s).is_err());
   }

   #[test]
   fn pause_rejected_from_ready() {
      let mut s = state_with_status(PlaybackStatus::Ready);
      assert!(pause(&mut s).is_err());
   }

   #[test]
   fn pause_rejected_from_paused() {
      let mut s = state_with_status(PlaybackStatus::Paused);
      assert!(pause(&mut s).is_err());
   }

   #[test]
   fn pause_rejected_from_ended() {
      let mut s = state_with_status(PlaybackStatus::Ended);
      assert!(pause(&mut s).is_err());
   }

   #[test]
   fn pause_rejected_from_loading() {
      let mut s = state_with_status(PlaybackStatus::Loading);
      assert!(pause(&mut s).is_err());
   }

   #[test]
   fn pause_rejected_from_error() {
      let mut s = state_with_status(PlaybackStatus::Error);
      assert!(pause(&mut s).is_err());
   }

   // -- stop --

   #[test]
   fn stop_from_loading() {
      let mut s = state_with_status(PlaybackStatus::Loading);
      stop(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn stop_from_ready() {
      let mut s = state_with_status(PlaybackStatus::Ready);
      stop(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn stop_from_playing() {
      let mut s = state_with_status(PlaybackStatus::Playing);
      stop(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn stop_from_paused() {
      let mut s = state_with_status(PlaybackStatus::Paused);
      stop(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn stop_from_ended() {
      let mut s = state_with_status(PlaybackStatus::Ended);
      stop(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn stop_rejected_from_idle() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      assert!(stop(&mut s).is_err());
   }

   #[test]
   fn stop_rejected_from_error() {
      let mut s = state_with_status(PlaybackStatus::Error);
      assert!(stop(&mut s).is_err());
   }

   #[test]
   fn stop_preserves_settings() {
      let mut s = PlayerState {
         status: PlaybackStatus::Playing,
         volume: 0.5,
         muted: true,
         playback_rate: 1.5,
         looping: true,
         src: Some("test.mp3".to_string()),
         title: Some("Song".to_string()),
         current_time: 42.0,
         ..Default::default()
      };
      stop(&mut s).unwrap();

      assert_eq!(s.status, PlaybackStatus::Idle);
      assert_eq!(s.volume, 0.5);
      assert!(s.muted);
      assert_eq!(s.playback_rate, 1.5);
      assert!(s.looping);
      assert!(s.src.is_none());
      assert!(s.title.is_none());
      assert_eq!(s.current_time, 0.0);
   }

   // -- seek --

   #[test]
   fn seek_from_ready() {
      let mut s = state_with_duration(PlaybackStatus::Ready, 120.0);
      seek(&mut s, 30.0).unwrap();
      assert_eq!(s.current_time, 30.0);
      assert_eq!(s.status, PlaybackStatus::Ready);
   }

   #[test]
   fn seek_from_playing() {
      let mut s = state_with_duration(PlaybackStatus::Playing, 120.0);
      seek(&mut s, 30.0).unwrap();
      assert_eq!(s.current_time, 30.0);
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   #[test]
   fn seek_from_paused() {
      let mut s = state_with_duration(PlaybackStatus::Paused, 120.0);
      seek(&mut s, 15.0).unwrap();
      assert_eq!(s.current_time, 15.0);
      assert_eq!(s.status, PlaybackStatus::Paused);
   }

   #[test]
   fn seek_from_ended() {
      let mut s = state_with_duration(PlaybackStatus::Ended, 120.0);
      seek(&mut s, 10.0).unwrap();
      assert_eq!(s.current_time, 10.0);
      assert_eq!(s.status, PlaybackStatus::Ended);
   }

   #[test]
   fn seek_clamps_negative_to_zero() {
      let mut s = state_with_duration(PlaybackStatus::Ready, 120.0);
      seek(&mut s, -5.0).unwrap();
      assert_eq!(s.current_time, 0.0);
   }

   #[test]
   fn seek_clamps_beyond_duration() {
      let mut s = state_with_duration(PlaybackStatus::Playing, 120.0);
      seek(&mut s, 999.0).unwrap();
      assert_eq!(s.current_time, 120.0);
   }

   #[test]
   fn seek_does_not_clamp_when_duration_unknown() {
      let mut s = state_with_duration(PlaybackStatus::Ready, 0.0);
      seek(&mut s, 45.0).unwrap();
      assert_eq!(s.current_time, 45.0);
   }

   #[test]
   fn seek_rejected_from_idle() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      assert!(seek(&mut s, 10.0).is_err());
   }

   #[test]
   fn seek_rejected_from_loading() {
      let mut s = state_with_status(PlaybackStatus::Loading);
      assert!(seek(&mut s, 10.0).is_err());
   }

   #[test]
   fn seek_rejected_from_error() {
      let mut s = state_with_status(PlaybackStatus::Error);
      assert!(seek(&mut s, 10.0).is_err());
   }

   #[test]
   fn seek_rejects_nan() {
      let mut s = state_with_status(PlaybackStatus::Ready);
      assert!(seek(&mut s, f64::NAN).is_err());
      assert!(seek(&mut s, f64::INFINITY).is_err());
   }

   // -- set_volume --

   #[test]
   fn set_volume_clamps_to_range() {
      let mut s = PlayerState::default();

      set_volume(&mut s, 1.5).unwrap();
      assert_eq!(s.volume, 1.0);

      set_volume(&mut s, -0.5).unwrap();
      assert_eq!(s.volume, 0.0);

      set_volume(&mut s, 0.7).unwrap();
      assert_eq!(s.volume, 0.7);
   }

   #[test]
   fn set_volume_rejects_nan() {
      let mut s = PlayerState::default();
      assert!(set_volume(&mut s, f64::NAN).is_err());
      assert!(set_volume(&mut s, f64::INFINITY).is_err());
   }

   // -- set_muted --

   #[test]
   fn set_muted_updates_state() {
      let mut s = PlayerState::default();
      set_muted(&mut s, true);
      assert!(s.muted);
      set_muted(&mut s, false);
      assert!(!s.muted);
   }

   // -- set_playback_rate --

   #[test]
   fn set_playback_rate_clamps_to_range() {
      let mut s = PlayerState::default();

      set_playback_rate(&mut s, 2.0).unwrap();
      assert_eq!(s.playback_rate, 2.0);

      set_playback_rate(&mut s, 0.0).unwrap();
      assert_eq!(s.playback_rate, 0.25);

      set_playback_rate(&mut s, 10.0).unwrap();
      assert_eq!(s.playback_rate, 4.0);
   }

   #[test]
   fn set_playback_rate_rejects_nan() {
      let mut s = PlayerState::default();
      assert!(set_playback_rate(&mut s, f64::NAN).is_err());
      assert!(set_playback_rate(&mut s, f64::INFINITY).is_err());
   }

   // -- set_loop --

   #[test]
   fn set_loop_updates_state() {
      let mut s = PlayerState::default();
      set_loop(&mut s, true);
      assert!(s.looping);
      set_loop(&mut s, false);
      assert!(!s.looping);
   }

   // -- error --

   #[test]
   fn error_from_loading() {
      let mut s = state_with_status(PlaybackStatus::Loading);
      error(&mut s, "decode failed".into());
      assert_eq!(s.status, PlaybackStatus::Error);
      assert_eq!(s.error.as_deref(), Some("decode failed"));
   }

   #[test]
   fn error_ignored_from_idle() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      error(&mut s, "should not apply".into());
      assert_eq!(s.status, PlaybackStatus::Idle);
      assert!(s.error.is_none());
   }

   #[test]
   fn error_ignored_from_playing() {
      let mut s = state_with_status(PlaybackStatus::Playing);
      error(&mut s, "should not apply".into());
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   // -- rollback on error --

   #[test]
   fn failed_transition_leaves_state_unchanged() {
      let mut s = state_with_status(PlaybackStatus::Idle);
      let before = s.clone();
      let _ = play(&mut s);
      assert_eq!(s.status, before.status);

      let mut s = state_with_status(PlaybackStatus::Playing);
      let _ = load(&mut s, "a.mp3", &AudioMetadata::default(), 0.0);
      assert_eq!(s.status, PlaybackStatus::Playing);
   }
}
