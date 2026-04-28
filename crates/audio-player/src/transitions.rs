//! State machine transition rules for the audio player.
//!
//! These functions are the single source of truth for which state transitions
//! are valid and how `PlayerState` fields are mutated. [`RodioAudioPlayer`]
//! delegates to them.

use crate::error::{Error, Result};
use crate::models::{LoopMode, PlaybackStatus, PlayerState, PlaylistItem};

/// Where transport navigation should land.
///
/// Used by [`next_target`], [`prev_target`], and [`auto_advance_target`] to
/// describe the result of evaluating navigation rules without mutating state.
/// The caller (typically the audio player) acts on the target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavTarget {
   /// Restart the current item from the beginning. Used by `prev` within
   /// the first 3s of playback and by `auto_advance` under [`LoopMode::One`].
   RestartCurrent,
   /// Switch to a different playlist index.
   Index(usize),
   /// No more items; the player should transition to [`PlaybackStatus::Ended`].
   /// Returned by `auto_advance` at the end of the playlist with looping off.
   End,
}

// ---------------------------------------------------------------------------
// Transport actions
// ---------------------------------------------------------------------------

/// Transitions to [`PlaybackStatus::Loading`] for a new playlist.
///
/// Replaces any existing playlist and clears per-item state. Call this before
/// starting I/O for the first item so the frontend can show a loading
/// indicator. After the I/O completes, call [`load`] to finalize the
/// transition to `Ready`.
///
/// # Errors
/// * `InvalidState` if the player is not currently `Idle`, `Ended`, or `Error`.
/// * `InvalidValue` if the playlist is empty or `start_index` is out of range.
pub fn begin_load(
   state: &mut PlayerState,
   playlist: Vec<PlaylistItem>,
   start_index: usize,
) -> Result<()> {
   match state.status {
      PlaybackStatus::Idle | PlaybackStatus::Ended | PlaybackStatus::Error => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot load in {:?} state",
            state.status
         )));
      }
   }

   if playlist.is_empty() {
      return Err(Error::InvalidValue(
         "Playlist must contain at least one item".into(),
      ));
   }
   if start_index >= playlist.len() {
      return Err(Error::InvalidValue(format!(
         "startIndex {start_index} out of range (playlist has {} items)",
         playlist.len()
      )));
   }

   state.status = PlaybackStatus::Loading;
   state.playlist = playlist;
   state.current_index = Some(start_index);
   state.current_time = 0.0;
   state.duration = 0.0;
   state.error = None;
   Ok(())
}

/// Transitions to [`PlaybackStatus::Loading`] for a different item in the
/// existing playlist. Used by [`next_target`] / [`prev_target`] / auto-advance.
///
/// # Errors
/// * `InvalidValue` if `index` is out of range.
pub fn begin_load_index(state: &mut PlayerState, index: usize) -> Result<()> {
   if index >= state.playlist.len() {
      return Err(Error::InvalidValue(format!(
         "Index {index} out of range (playlist has {} items)",
         state.playlist.len()
      )));
   }

   state.status = PlaybackStatus::Loading;
   state.current_index = Some(index);
   state.current_time = 0.0;
   state.duration = 0.0;
   state.error = None;
   Ok(())
}

/// Finalizes a load by transitioning from `Loading` to `Ready` with the
/// decoded duration. Also accepts `Idle`, `Ended`, and `Error` so that
/// callers performing instant local-file loads can skip the explicit
/// `Loading` step.
///
/// The playlist and `current_index` are expected to already be set on
/// `state` (typically by [`begin_load`] or [`begin_load_index`]).
pub fn load(state: &mut PlayerState, duration: f64) -> Result<()> {
   match state.status {
      PlaybackStatus::Loading => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot load in {:?} state",
            state.status
         )));
      }
   }

   if state.current_index.is_none() || state.playlist.is_empty() {
      return Err(Error::InvalidState(
         "Cannot finalize load without a playlist".into(),
      ));
   }

   state.status = PlaybackStatus::Ready;
   state.duration = duration;
   state.current_time = 0.0;
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
///
/// Allowed from any non-`Idle` status, including `Error`, so users can
/// recover from a failed load without re-loading the entire playlist.
pub fn stop(state: &mut PlayerState) -> Result<()> {
   match state.status {
      PlaybackStatus::Loading
      | PlaybackStatus::Ready
      | PlaybackStatus::Playing
      | PlaybackStatus::Paused
      | PlaybackStatus::Ended
      | PlaybackStatus::Error => {}
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
      loop_mode: state.loop_mode,
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

/// Computes the target for a `jump_to(index)` request.
///
/// Allowed from any status that has a loaded playlist, including `Error` —
/// jumping is the recovery path when the current item failed to load.
///
/// # Errors
/// * `InvalidState` if the player is `Idle` or `Loading` (no playlist or
///   already mid-load).
/// * `InvalidValue` if `index` is out of range.
pub fn jump_target(state: &PlayerState, index: usize) -> Result<NavTarget> {
   match state.status {
      PlaybackStatus::Ready
      | PlaybackStatus::Playing
      | PlaybackStatus::Paused
      | PlaybackStatus::Ended
      | PlaybackStatus::Error => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot navigate in {:?} state",
            state.status
         )));
      }
   }

   if index >= state.playlist.len() {
      return Err(Error::InvalidValue(format!(
         "Index {index} out of range (playlist has {} items)",
         state.playlist.len()
      )));
   }

   if Some(index) == state.current_index {
      Ok(NavTarget::RestartCurrent)
   } else {
      Ok(NavTarget::Index(index))
   }
}

/// Computes the target for a `next()` request.
///
/// Returns the next index, wrapping to 0 when [`LoopMode::All`] is set and
/// the current item is the last. Returns [`NavTarget::End`] when there's no
/// next item and looping is off.
///
/// Allowed from `Error` so users can skip past a track that failed to load
/// without re-loading the entire playlist.
///
/// # Errors
/// * `InvalidState` if the player is `Idle` or `Loading`, or if no playlist
///   is loaded.
pub fn next_target(state: &PlayerState) -> Result<NavTarget> {
   match state.status {
      PlaybackStatus::Ready
      | PlaybackStatus::Playing
      | PlaybackStatus::Paused
      | PlaybackStatus::Ended
      | PlaybackStatus::Error => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot navigate in {:?} state",
            state.status
         )));
      }
   }

   let Some(idx) = state.current_index else {
      return Err(Error::InvalidState("No playlist loaded".into()));
   };

   if idx + 1 < state.playlist.len() {
      Ok(NavTarget::Index(idx + 1))
   } else if state.loop_mode == LoopMode::All && !state.playlist.is_empty() {
      Ok(NavTarget::Index(0))
   } else {
      Ok(NavTarget::End)
   }
}

/// Computes the target for a `prev()` request.
///
/// If the current item has played for more than 3 seconds, returns
/// [`NavTarget::RestartCurrent`]. Otherwise returns the previous index,
/// wrapping to the last item when [`LoopMode::All`] is set. Falls back to
/// `RestartCurrent` at the start of a non-looping playlist.
///
/// Allowed from `Error` so users can skip past a track that failed to load.
/// In `Error` the `currentTime > 3.0` rule still applies, but since the
/// item never played, `currentTime` is 0 and the result moves to the
/// previous item.
///
/// # Errors
/// * `InvalidState` if the player is `Idle` or `Loading`, or if no playlist
///   is loaded.
pub fn prev_target(state: &PlayerState) -> Result<NavTarget> {
   match state.status {
      PlaybackStatus::Ready
      | PlaybackStatus::Playing
      | PlaybackStatus::Paused
      | PlaybackStatus::Ended
      | PlaybackStatus::Error => {}
      _ => {
         return Err(Error::InvalidState(format!(
            "Cannot navigate in {:?} state",
            state.status
         )));
      }
   }

   let Some(idx) = state.current_index else {
      return Err(Error::InvalidState("No playlist loaded".into()));
   };

   if state.current_time > 3.0 {
      return Ok(NavTarget::RestartCurrent);
   }

   if idx > 0 {
      Ok(NavTarget::Index(idx - 1))
   } else if state.loop_mode == LoopMode::All && state.playlist.len() > 1 {
      Ok(NavTarget::Index(state.playlist.len() - 1))
   } else {
      Ok(NavTarget::RestartCurrent)
   }
}

/// Computes the target for natural end-of-track auto-advance.
///
/// * [`LoopMode::One`] → restart the current item.
/// * Has a next item, or [`LoopMode::All`] → advance (with wrap-around).
/// * Otherwise → [`NavTarget::End`].
///
/// Unlike user-initiated `next`/`prev`, this is not gated by status — the
/// player calls it from the monitor thread and is responsible for ensuring
/// the player was actually playing.
pub fn auto_advance_target(state: &PlayerState) -> NavTarget {
   if state.loop_mode == LoopMode::One {
      return NavTarget::RestartCurrent;
   }

   let Some(idx) = state.current_index else {
      return NavTarget::End;
   };

   if idx + 1 < state.playlist.len() {
      NavTarget::Index(idx + 1)
   } else if state.loop_mode == LoopMode::All && !state.playlist.is_empty() {
      NavTarget::Index(0)
   } else {
      NavTarget::End
   }
}

/// Transitions to [`PlaybackStatus::Error`] with a message.
///
/// Valid from `Loading` (I/O or decode failure during load) or active playback
/// (`Playing` / `Paused`) for runtime failures such as a network stream
/// dropping mid-playback. Other statuses are left unchanged.
pub fn error(state: &mut PlayerState, message: String) {
   match state.status {
      PlaybackStatus::Loading | PlaybackStatus::Playing | PlaybackStatus::Paused => {
         state.status = PlaybackStatus::Error;
         state.error = Some(message);
      }
      _ => {}
   }
}

/// Transitions to [`PlaybackStatus::Ended`].
///
/// Called by the player when there's nothing left to play — either the monitor
/// detecting end-of-track with no auto-advance target, or a `next` request
/// falling off the end of a non-looping playlist. Valid from any active
/// transport status (`Playing`, `Paused`, `Ready`); a no-op otherwise.
pub fn ended(state: &mut PlayerState) {
   match state.status {
      PlaybackStatus::Playing | PlaybackStatus::Paused | PlaybackStatus::Ready => {
         state.status = PlaybackStatus::Ended;
         state.current_time = state.duration;
      }
      _ => {}
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

/// Applies a loop-mode change.
pub fn set_loop_mode(state: &mut PlayerState, mode: LoopMode) {
   state.loop_mode = mode;
}

#[cfg(test)]
mod tests {
   use super::*;
   use crate::models::AudioMetadata;

   fn item(src: &str, title: &str) -> PlaylistItem {
      PlaylistItem {
         src: src.to_string(),
         metadata: Some(AudioMetadata {
            title: Some(title.to_string()),
            artist: None,
            artwork: None,
         }),
      }
   }

   fn loaded_state(status: PlaybackStatus, items: usize, current: usize) -> PlayerState {
      let playlist = (0..items)
         .map(|i| item(&format!("track-{i}.mp3"), &format!("Track {i}")))
         .collect();
      PlayerState {
         status,
         playlist,
         current_index: Some(current),
         duration: 60.0,
         ..Default::default()
      }
   }

   // -- begin_load --

   #[test]
   fn begin_load_from_idle() {
      let mut s = PlayerState::default();
      let playlist = vec![item("a.mp3", "A"), item("b.mp3", "B")];
      begin_load(&mut s, playlist, 0).unwrap();

      assert_eq!(s.status, PlaybackStatus::Loading);
      assert_eq!(s.playlist.len(), 2);
      assert_eq!(s.current_index, Some(0));
      assert_eq!(s.current_time, 0.0);
      assert_eq!(s.duration, 0.0);
      assert!(s.error.is_none());
   }

   #[test]
   fn begin_load_with_start_index() {
      let mut s = PlayerState::default();
      begin_load(
         &mut s,
         vec![item("a", "A"), item("b", "B"), item("c", "C")],
         2,
      )
      .unwrap();
      assert_eq!(s.current_index, Some(2));
   }

   #[test]
   fn begin_load_rejects_empty_playlist() {
      let mut s = PlayerState::default();
      assert!(begin_load(&mut s, vec![], 0).is_err());
      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn begin_load_rejects_out_of_range_start_index() {
      let mut s = PlayerState::default();
      assert!(begin_load(&mut s, vec![item("a", "A")], 5).is_err());
   }

   #[test]
   fn begin_load_from_ended() {
      let mut s = loaded_state(PlaybackStatus::Ended, 1, 0);
      assert!(begin_load(&mut s, vec![item("x", "X")], 0).is_ok());
      assert_eq!(s.status, PlaybackStatus::Loading);
   }

   #[test]
   fn begin_load_from_error() {
      let mut s = loaded_state(PlaybackStatus::Error, 1, 0);
      assert!(begin_load(&mut s, vec![item("x", "X")], 0).is_ok());
      assert_eq!(s.status, PlaybackStatus::Loading);
   }

   #[test]
   fn begin_load_rejected_from_loading() {
      let mut s = loaded_state(PlaybackStatus::Loading, 1, 0);
      assert!(begin_load(&mut s, vec![item("x", "X")], 0).is_err());
   }

   #[test]
   fn begin_load_rejected_from_playing() {
      let mut s = loaded_state(PlaybackStatus::Playing, 1, 0);
      assert!(begin_load(&mut s, vec![item("x", "X")], 0).is_err());
   }

   #[test]
   fn begin_load_rejected_from_paused() {
      let mut s = loaded_state(PlaybackStatus::Paused, 1, 0);
      assert!(begin_load(&mut s, vec![item("x", "X")], 0).is_err());
   }

   // -- begin_load_index --

   #[test]
   fn begin_load_index_in_range() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 0);
      begin_load_index(&mut s, 2).unwrap();
      assert_eq!(s.status, PlaybackStatus::Loading);
      assert_eq!(s.current_index, Some(2));
      assert_eq!(s.current_time, 0.0);
   }

   #[test]
   fn begin_load_index_out_of_range() {
      let mut s = loaded_state(PlaybackStatus::Playing, 2, 0);
      assert!(begin_load_index(&mut s, 5).is_err());
   }

   // -- load --

   #[test]
   fn load_finalizes_from_loading() {
      let mut s = loaded_state(PlaybackStatus::Loading, 2, 0);
      load(&mut s, 120.0).unwrap();
      assert_eq!(s.status, PlaybackStatus::Ready);
      assert_eq!(s.duration, 120.0);
      assert_eq!(s.current_time, 0.0);
   }

   #[test]
   fn load_rejects_without_playlist() {
      let mut s = PlayerState {
         status: PlaybackStatus::Loading,
         ..Default::default()
      };
      assert!(load(&mut s, 30.0).is_err());
   }

   #[test]
   fn load_rejected_from_playing() {
      let mut s = loaded_state(PlaybackStatus::Playing, 1, 0);
      assert!(load(&mut s, 30.0).is_err());
   }

   // -- play / pause / stop --

   #[test]
   fn play_from_ready() {
      let mut s = loaded_state(PlaybackStatus::Ready, 1, 0);
      play(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   #[test]
   fn play_from_paused() {
      let mut s = loaded_state(PlaybackStatus::Paused, 1, 0);
      play(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   #[test]
   fn play_from_ended() {
      let mut s = loaded_state(PlaybackStatus::Ended, 1, 0);
      play(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Playing);
   }

   #[test]
   fn play_rejected_from_idle() {
      let mut s = PlayerState::default();
      assert!(play(&mut s).is_err());
   }

   #[test]
   fn pause_from_playing() {
      let mut s = loaded_state(PlaybackStatus::Playing, 1, 0);
      pause(&mut s).unwrap();
      assert_eq!(s.status, PlaybackStatus::Paused);
   }

   #[test]
   fn pause_rejected_from_paused() {
      let mut s = loaded_state(PlaybackStatus::Paused, 1, 0);
      assert!(pause(&mut s).is_err());
   }

   #[test]
   fn stop_clears_playlist_preserves_settings() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 1);
      s.volume = 0.5;
      s.muted = true;
      s.playback_rate = 1.5;
      s.loop_mode = LoopMode::All;
      s.current_time = 12.0;

      stop(&mut s).unwrap();

      assert_eq!(s.status, PlaybackStatus::Idle);
      assert!(s.playlist.is_empty());
      assert_eq!(s.current_index, None);
      assert_eq!(s.current_time, 0.0);
      assert_eq!(s.volume, 0.5);
      assert!(s.muted);
      assert_eq!(s.playback_rate, 1.5);
      assert_eq!(s.loop_mode, LoopMode::All);
   }

   #[test]
   fn stop_rejected_from_idle() {
      let mut s = PlayerState::default();
      assert!(stop(&mut s).is_err());
   }

   #[test]
   fn stop_from_error_recovers() {
      let mut s = loaded_state(PlaybackStatus::Error, 3, 1);
      s.error = Some("network failure".into());

      stop(&mut s).unwrap();

      assert_eq!(s.status, PlaybackStatus::Idle);
      assert!(s.error.is_none());
      assert!(s.playlist.is_empty());
   }

   #[test]
   fn next_target_from_error_advances() {
      let s = loaded_state(PlaybackStatus::Error, 3, 1);

      assert_eq!(next_target(&s).unwrap(), NavTarget::Index(2));
   }

   #[test]
   fn prev_target_from_error_moves_back() {
      let s = loaded_state(PlaybackStatus::Error, 3, 2);

      assert_eq!(prev_target(&s).unwrap(), NavTarget::Index(1));
   }

   #[test]
   fn jump_target_from_error_advances() {
      let s = loaded_state(PlaybackStatus::Error, 3, 0);

      assert_eq!(jump_target(&s, 2).unwrap(), NavTarget::Index(2));
   }

   #[test]
   fn begin_load_index_from_error_recovers() {
      let mut s = loaded_state(PlaybackStatus::Error, 3, 0);
      s.error = Some("decode failed".into());

      begin_load_index(&mut s, 1).unwrap();

      assert_eq!(s.status, PlaybackStatus::Loading);
      assert_eq!(s.current_index, Some(1));
      assert!(s.error.is_none());
   }

   // -- seek --

   #[test]
   fn seek_clamps_to_duration() {
      let mut s = loaded_state(PlaybackStatus::Playing, 1, 0);
      s.duration = 60.0;
      seek(&mut s, 999.0).unwrap();
      assert_eq!(s.current_time, 60.0);
      seek(&mut s, -5.0).unwrap();
      assert_eq!(s.current_time, 0.0);
   }

   #[test]
   fn seek_does_not_clamp_when_duration_unknown() {
      let mut s = loaded_state(PlaybackStatus::Ready, 1, 0);
      s.duration = 0.0;
      seek(&mut s, 45.0).unwrap();
      assert_eq!(s.current_time, 45.0);

      seek(&mut s, -5.0).unwrap();
      assert_eq!(s.current_time, 0.0);
   }

   #[test]
   fn seek_rejected_from_idle() {
      let mut s = PlayerState::default();
      assert!(seek(&mut s, 10.0).is_err());
   }

   #[test]
   fn seek_rejected_from_loading() {
      let mut s = loaded_state(PlaybackStatus::Loading, 1, 0);
      assert!(seek(&mut s, 10.0).is_err());
   }

   #[test]
   fn seek_rejected_from_error() {
      let mut s = loaded_state(PlaybackStatus::Error, 1, 0);
      assert!(seek(&mut s, 10.0).is_err());
   }

   #[test]
   fn seek_rejects_nan() {
      let mut s = loaded_state(PlaybackStatus::Ready, 1, 0);
      assert!(seek(&mut s, f64::NAN).is_err());
      assert!(seek(&mut s, f64::INFINITY).is_err());
   }

   // -- next_target --

   #[test]
   fn next_target_advances_within_playlist() {
      let s = loaded_state(PlaybackStatus::Playing, 3, 0);
      assert_eq!(next_target(&s).unwrap(), NavTarget::Index(1));
   }

   #[test]
   fn next_target_at_end_returns_end_when_loop_off() {
      let s = loaded_state(PlaybackStatus::Playing, 3, 2);
      assert_eq!(next_target(&s).unwrap(), NavTarget::End);
   }

   #[test]
   fn next_target_at_end_wraps_when_loop_all() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 2);
      s.loop_mode = LoopMode::All;
      assert_eq!(next_target(&s).unwrap(), NavTarget::Index(0));
   }

   #[test]
   fn next_target_rejected_from_idle() {
      let s = PlayerState::default();
      assert!(next_target(&s).is_err());
   }

   #[test]
   fn next_target_rejected_from_loading() {
      let s = loaded_state(PlaybackStatus::Loading, 2, 0);
      assert!(next_target(&s).is_err());
   }

   // -- prev_target --

   #[test]
   fn prev_target_moves_back_when_within_3s() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 1);
      s.current_time = 1.0;
      assert_eq!(prev_target(&s).unwrap(), NavTarget::Index(0));
   }

   #[test]
   fn prev_target_restarts_current_when_past_3s() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 1);
      s.current_time = 10.0;
      assert_eq!(prev_target(&s).unwrap(), NavTarget::RestartCurrent);
   }

   #[test]
   fn prev_target_at_start_restarts_when_loop_off() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 0);
      s.current_time = 0.5;
      assert_eq!(prev_target(&s).unwrap(), NavTarget::RestartCurrent);
   }

   #[test]
   fn prev_target_at_start_wraps_when_loop_all() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 0);
      s.current_time = 0.5;
      s.loop_mode = LoopMode::All;
      assert_eq!(prev_target(&s).unwrap(), NavTarget::Index(2));
   }

   #[test]
   fn prev_target_rejected_from_idle() {
      let s = PlayerState::default();
      assert!(prev_target(&s).is_err());
   }

   // -- auto_advance_target --

   #[test]
   fn auto_advance_advances_to_next() {
      let s = loaded_state(PlaybackStatus::Playing, 3, 0);
      assert_eq!(auto_advance_target(&s), NavTarget::Index(1));
   }

   #[test]
   fn auto_advance_at_end_returns_end() {
      let s = loaded_state(PlaybackStatus::Playing, 3, 2);
      assert_eq!(auto_advance_target(&s), NavTarget::End);
   }

   #[test]
   fn auto_advance_with_loop_one_restarts() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 1);
      s.loop_mode = LoopMode::One;
      assert_eq!(auto_advance_target(&s), NavTarget::RestartCurrent);
   }

   #[test]
   fn auto_advance_with_loop_all_wraps() {
      let mut s = loaded_state(PlaybackStatus::Playing, 3, 2);
      s.loop_mode = LoopMode::All;
      assert_eq!(auto_advance_target(&s), NavTarget::Index(0));
   }

   // -- jump_target --

   #[test]
   fn jump_target_to_other_index() {
      let s = loaded_state(PlaybackStatus::Playing, 3, 0);

      assert_eq!(jump_target(&s, 2).unwrap(), NavTarget::Index(2));
   }

   #[test]
   fn jump_target_to_current_index_restarts() {
      let s = loaded_state(PlaybackStatus::Playing, 3, 1);

      assert_eq!(jump_target(&s, 1).unwrap(), NavTarget::RestartCurrent);
   }

   #[test]
   fn jump_target_out_of_range() {
      let s = loaded_state(PlaybackStatus::Playing, 3, 0);

      assert!(jump_target(&s, 5).is_err());
   }

   #[test]
   fn jump_target_rejected_from_loading() {
      let s = loaded_state(PlaybackStatus::Loading, 3, 0);

      assert!(jump_target(&s, 1).is_err());
   }

   #[test]
   fn auto_advance_with_loop_one_overrides_end() {
      let mut s = loaded_state(PlaybackStatus::Playing, 1, 0);
      s.loop_mode = LoopMode::One;
      assert_eq!(auto_advance_target(&s), NavTarget::RestartCurrent);
   }

   // -- error / ended --

   #[test]
   fn error_from_loading() {
      let mut s = loaded_state(PlaybackStatus::Loading, 1, 0);
      error(&mut s, "decode failed".into());
      assert_eq!(s.status, PlaybackStatus::Error);
      assert_eq!(s.error.as_deref(), Some("decode failed"));
   }

   #[test]
   fn error_ignored_from_idle() {
      let mut s = PlayerState::default();
      error(&mut s, "should not apply".into());
      assert_eq!(s.status, PlaybackStatus::Idle);
      assert!(s.error.is_none());
   }

   #[test]
   fn ended_from_playing_sets_current_time_to_duration() {
      let mut s = loaded_state(PlaybackStatus::Playing, 1, 0);
      s.duration = 100.0;
      s.current_time = 99.5;
      ended(&mut s);
      assert_eq!(s.status, PlaybackStatus::Ended);
      assert_eq!(s.current_time, 100.0);
   }

   #[test]
   fn ended_from_paused_sets_status() {
      let mut s = loaded_state(PlaybackStatus::Paused, 1, 0);

      ended(&mut s);

      assert_eq!(s.status, PlaybackStatus::Ended);
   }

   #[test]
   fn ended_from_ready_sets_status() {
      let mut s = loaded_state(PlaybackStatus::Ready, 1, 0);

      ended(&mut s);

      assert_eq!(s.status, PlaybackStatus::Ended);
   }

   #[test]
   fn ended_ignored_from_idle() {
      let mut s = PlayerState::default();

      ended(&mut s);

      assert_eq!(s.status, PlaybackStatus::Idle);
   }

   #[test]
   fn error_from_playing() {
      let mut s = loaded_state(PlaybackStatus::Playing, 1, 0);

      error(&mut s, "stream dropped".into());

      assert_eq!(s.status, PlaybackStatus::Error);
      assert_eq!(s.error.as_deref(), Some("stream dropped"));
   }

   #[test]
   fn error_from_paused() {
      let mut s = loaded_state(PlaybackStatus::Paused, 1, 0);

      error(&mut s, "stream dropped".into());

      assert_eq!(s.status, PlaybackStatus::Error);
   }

   // -- settings --

   #[test]
   fn set_volume_clamps_to_range() {
      let mut s = PlayerState::default();
      set_volume(&mut s, 1.5).unwrap();
      assert_eq!(s.volume, 1.0);
      set_volume(&mut s, -0.5).unwrap();
      assert_eq!(s.volume, 0.0);
   }

   #[test]
   fn set_volume_rejects_nan() {
      let mut s = PlayerState::default();
      assert!(set_volume(&mut s, f64::NAN).is_err());
   }

   #[test]
   fn set_playback_rate_clamps_to_range() {
      let mut s = PlayerState::default();
      set_playback_rate(&mut s, 0.0).unwrap();
      assert_eq!(s.playback_rate, 0.25);
      set_playback_rate(&mut s, 10.0).unwrap();
      assert_eq!(s.playback_rate, 4.0);
   }

   #[test]
   fn set_loop_mode_updates_state() {
      let mut s = PlayerState::default();
      set_loop_mode(&mut s, LoopMode::All);
      assert_eq!(s.loop_mode, LoopMode::All);
      set_loop_mode(&mut s, LoopMode::One);
      assert_eq!(s.loop_mode, LoopMode::One);
      set_loop_mode(&mut s, LoopMode::Off);
      assert_eq!(s.loop_mode, LoopMode::Off);
   }

   #[test]
   fn set_muted_updates_state() {
      let mut s = PlayerState::default();
      set_muted(&mut s, true);
      assert!(s.muted);
   }
}
