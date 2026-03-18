use std::sync::{Arc, Mutex};

use tauri::{
   Emitter, Manager, Runtime,
   plugin::{Builder, TauriPlugin},
};
use tracing::warn;

mod commands;
mod error;
mod models;

pub use error::{Error, Result};
pub use models::{AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate};

/// Mock audio player that manages state transitions without actual audio playback.
///
/// This is the desktop-only mock implementation. When real playback is added, this will
/// be replaced by a platform-specific implementation (e.g. AVAudioPlayer on iOS,
/// MediaPlayer on Android, or a native audio library on Windows).
pub struct AudioPlayer {
   state: Mutex<PlayerState>,
   on_changed: Arc<dyn Fn(&PlayerState) + Send + Sync>,
}

impl AudioPlayer {
   pub fn new(on_changed: Arc<dyn Fn(&PlayerState) + Send + Sync>) -> Self {
      Self {
         state: Mutex::new(PlayerState::default()),
         on_changed,
      }
   }

   pub fn get_state(&self) -> PlayerState {
      self.state.lock().unwrap().clone()
   }

   /// Applies a mutation to the player state, emits a change event, and returns
   /// the resulting state snapshot.
   fn update_state(&self, f: impl FnOnce(&mut PlayerState)) -> PlayerState {
      let mut state = self.state.lock().unwrap();
      f(&mut state);
      let snapshot = state.clone();
      drop(state);
      (self.on_changed)(&snapshot);
      snapshot
   }

   /// Like [`update_state`], but the closure may fail. The change event is only
   /// emitted on success; on failure the state is left unchanged.
   fn try_update_state(
      &self,
      f: impl FnOnce(&mut PlayerState) -> Result<()>,
   ) -> Result<PlayerState> {
      let mut state = self.state.lock().unwrap();
      f(&mut state)?;
      let snapshot = state.clone();
      drop(state);
      (self.on_changed)(&snapshot);
      Ok(snapshot)
   }

   pub fn load(&self, src: &str, metadata: Option<AudioMetadata>) -> Result<AudioActionResponse> {
      let meta = metadata.unwrap_or_default();
      let player = self.try_update_state(|s| {
         match s.status {
            PlaybackStatus::Idle | PlaybackStatus::Ended | PlaybackStatus::Error => {}
            _ => {
               return Err(Error::InvalidState(format!(
                  "Cannot load in {:?} state",
                  s.status
               )));
            }
         }

         s.status = PlaybackStatus::Ready;
         s.src = Some(src.to_string());
         s.title = meta.title.clone();
         s.artist = meta.artist.clone();
         s.artwork = meta.artwork.clone();
         s.current_time = 0.0;
         s.duration = 0.0;
         s.error = None;
         Ok(())
      })?;

      Ok(AudioActionResponse::new(player, PlaybackStatus::Ready))
   }

   pub fn play(&self) -> Result<AudioActionResponse> {
      let player = self.try_update_state(|s| {
         match s.status {
            PlaybackStatus::Ready | PlaybackStatus::Paused | PlaybackStatus::Ended => {}
            _ => {
               return Err(Error::InvalidState(format!(
                  "Cannot play in {:?} state",
                  s.status
               )));
            }
         }

         s.status = PlaybackStatus::Playing;
         Ok(())
      })?;

      Ok(AudioActionResponse::new(player, PlaybackStatus::Playing))
   }

   pub fn pause(&self) -> Result<AudioActionResponse> {
      let player = self.try_update_state(|s| {
         match s.status {
            PlaybackStatus::Playing => {}
            _ => {
               return Err(Error::InvalidState(format!(
                  "Cannot pause in {:?} state",
                  s.status
               )));
            }
         }

         s.status = PlaybackStatus::Paused;
         Ok(())
      })?;

      Ok(AudioActionResponse::new(player, PlaybackStatus::Paused))
   }

   pub fn stop(&self) -> Result<AudioActionResponse> {
      let player = self.try_update_state(|s| {
         match s.status {
            PlaybackStatus::Loading
            | PlaybackStatus::Ready
            | PlaybackStatus::Playing
            | PlaybackStatus::Paused
            | PlaybackStatus::Ended => {}
            _ => {
               return Err(Error::InvalidState(format!(
                  "Cannot stop in {:?} state",
                  s.status
               )));
            }
         }

         // Reset to idle but preserve user settings (volume, muted, rate, loop).
         *s = PlayerState {
            volume: s.volume,
            muted: s.muted,
            playback_rate: s.playback_rate,
            looping: s.looping,
            ..Default::default()
         };
         Ok(())
      })?;

      Ok(AudioActionResponse::new(player, PlaybackStatus::Idle))
   }

   pub fn seek(&self, position: f64) -> Result<AudioActionResponse> {
      if !position.is_finite() {
         return Err(Error::InvalidValue(format!(
            "Seek position must be finite, got {position}"
         )));
      }

      let player = self.try_update_state(|s| {
         match s.status {
            PlaybackStatus::Ready
            | PlaybackStatus::Playing
            | PlaybackStatus::Paused
            | PlaybackStatus::Ended => {}
            _ => {
               return Err(Error::InvalidState(format!(
                  "Cannot seek in {:?} state",
                  s.status
               )));
            }
         }

         s.current_time = position.max(0.0);
         Ok(())
      })?;

      // Seek preserves the current status.
      let expected = player.status;
      Ok(AudioActionResponse::new(player, expected))
   }

   pub fn set_volume(&self, level: f64) -> Result<PlayerState> {
      if !level.is_finite() {
         return Err(Error::InvalidValue(format!(
            "Volume must be finite, got {level}"
         )));
      }

      Ok(self.update_state(|s| {
         s.volume = level.clamp(0.0, 1.0);
      }))
   }

   pub fn set_muted(&self, muted: bool) -> PlayerState {
      self.update_state(|s| {
         s.muted = muted;
      })
   }

   pub fn set_playback_rate(&self, rate: f64) -> Result<PlayerState> {
      if !rate.is_finite() {
         return Err(Error::InvalidValue(format!(
            "Playback rate must be finite, got {rate}"
         )));
      }

      Ok(self.update_state(|s| {
         s.playback_rate = rate.clamp(0.25, 4.0);
      }))
   }

   pub fn set_loop(&self, looping: bool) -> PlayerState {
      self.update_state(|s| {
         s.looping = looping;
      })
   }
}

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access
/// the audio player APIs.
pub trait AudioExt<R: Runtime> {
   fn audio(&self) -> &AudioPlayer;
}

impl<R: Runtime, T: Manager<R>> AudioExt<R> for T {
   fn audio(&self) -> &AudioPlayer {
      self.state::<AudioPlayer>().inner()
   }
}

/// Initializes the audio plugin with mock playback support.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
   Builder::new("audio")
      .invoke_handler(tauri::generate_handler![
         commands::load,
         commands::play,
         commands::pause,
         commands::stop,
         commands::seek,
         commands::set_volume,
         commands::set_muted,
         commands::set_playback_rate,
         commands::set_loop,
         commands::get_state,
         commands::is_native,
      ])
      .setup(|app, _api| {
         let app_handle = app.app_handle().clone();
         let player = AudioPlayer::new(Arc::new(move |state| {
            if let Err(e) = app_handle.emit("tauri-plugin-audio:state-changed", state) {
               warn!("Failed to emit state-changed event: {}", e);
            }
         }));
         app.manage(player);
         Ok(())
      })
      .build()
}

#[cfg(test)]
mod tests {
   use super::*;

   fn test_player() -> AudioPlayer {
      AudioPlayer::new(Arc::new(|_| {}))
   }

   #[test]
   fn initial_state_is_idle() {
      let player = test_player();
      assert_eq!(player.get_state().status, PlaybackStatus::Idle);
      assert_eq!(player.get_state().volume, 1.0);
      assert!(!player.get_state().muted);
   }

   #[test]
   fn load_transitions_idle_to_ready() {
      let player = test_player();
      let resp = player
         .load(
            "test.mp3",
            Some(AudioMetadata {
               title: Some("Test Song".to_string()),
               artist: Some("Test Artist".to_string()),
               artwork: None,
            }),
         )
         .unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Ready);
      assert_eq!(resp.player.src.as_deref(), Some("test.mp3"));
      assert_eq!(resp.player.title.as_deref(), Some("Test Song"));
      assert_eq!(resp.player.artist.as_deref(), Some("Test Artist"));
      assert!(resp.is_expected_status);
   }

   #[test]
   fn play_transitions_ready_to_playing() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      let resp = player.play().unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Playing);
      assert!(resp.is_expected_status);
   }

   #[test]
   fn pause_transitions_playing_to_paused() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      player.play().unwrap();
      let resp = player.pause().unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Paused);
      assert!(resp.is_expected_status);
   }

   #[test]
   fn resume_transitions_paused_to_playing() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      player.play().unwrap();
      player.pause().unwrap();
      let resp = player.play().unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Playing);
      assert!(resp.is_expected_status);
   }

   #[test]
   fn stop_resets_to_idle_preserving_settings() {
      let player = test_player();
      player.set_volume(0.5).unwrap();
      player.set_muted(true);
      player.set_playback_rate(1.5).unwrap();
      player.set_loop(true);
      player.load("test.mp3", None).unwrap();
      player.play().unwrap();
      let resp = player.stop().unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Idle);
      assert_eq!(resp.player.volume, 0.5);
      assert!(resp.player.muted);
      assert_eq!(resp.player.playback_rate, 1.5);
      assert!(resp.player.looping);
      assert!(resp.player.src.is_none());
      assert!(resp.is_expected_status);
   }

   #[test]
   fn seek_preserves_current_status() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      player.play().unwrap();
      let resp = player.seek(30.0).unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Playing);
      assert_eq!(resp.player.current_time, 30.0);
      assert!(resp.is_expected_status);
   }

   #[test]
   fn seek_clamps_negative_to_zero() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      let resp = player.seek(-5.0).unwrap();

      assert_eq!(resp.player.current_time, 0.0);
   }

   #[test]
   fn cannot_play_in_idle_state() {
      let player = test_player();
      assert!(player.play().is_err());
   }

   #[test]
   fn cannot_pause_in_idle_state() {
      let player = test_player();
      assert!(player.pause().is_err());
   }

   #[test]
   fn cannot_stop_in_idle_state() {
      let player = test_player();
      assert!(player.stop().is_err());
   }

   #[test]
   fn cannot_seek_in_idle_state() {
      let player = test_player();
      assert!(player.seek(10.0).is_err());
   }

   #[test]
   fn cannot_load_while_playing() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      player.play().unwrap();
      assert!(player.load("other.mp3", None).is_err());
   }

   #[test]
   fn can_load_after_ended() {
      let player = test_player();
      // Simulate ended state by loading, playing, then manually setting ended.
      player.load("test.mp3", None).unwrap();
      player.state.lock().unwrap().status = PlaybackStatus::Ended;
      let resp = player.load("other.mp3", None).unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Ready);
      assert_eq!(resp.player.src.as_deref(), Some("other.mp3"));
   }

   #[test]
   fn can_load_after_error() {
      let player = test_player();
      player.state.lock().unwrap().status = PlaybackStatus::Error;
      let resp = player.load("test.mp3", None).unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Ready);
   }

   #[test]
   fn set_volume_clamps_to_range() {
      let player = test_player();

      let state = player.set_volume(1.5).unwrap();
      assert_eq!(state.volume, 1.0);

      let state = player.set_volume(-0.5).unwrap();
      assert_eq!(state.volume, 0.0);

      let state = player.set_volume(0.7).unwrap();
      assert_eq!(state.volume, 0.7);
   }

   #[test]
   fn set_volume_rejects_nan() {
      let player = test_player();
      assert!(player.set_volume(f64::NAN).is_err());
      assert!(player.set_volume(f64::INFINITY).is_err());
   }

   #[test]
   fn set_muted_updates_state() {
      let player = test_player();
      let state = player.set_muted(true);
      assert!(state.muted);
   }

   #[test]
   fn set_playback_rate_clamps_to_range() {
      let player = test_player();

      let state = player.set_playback_rate(2.0).unwrap();
      assert_eq!(state.playback_rate, 2.0);

      let state = player.set_playback_rate(0.0).unwrap();
      assert_eq!(state.playback_rate, 0.25);

      let state = player.set_playback_rate(-1.0).unwrap();
      assert_eq!(state.playback_rate, 0.25);

      let state = player.set_playback_rate(10.0).unwrap();
      assert_eq!(state.playback_rate, 4.0);
   }

   #[test]
   fn set_playback_rate_rejects_nan() {
      let player = test_player();
      assert!(player.set_playback_rate(f64::NAN).is_err());
      assert!(player.set_playback_rate(f64::INFINITY).is_err());
   }

   #[test]
   fn seek_rejects_nan() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      assert!(player.seek(f64::NAN).is_err());
      assert!(player.seek(f64::INFINITY).is_err());
   }

   #[test]
   fn cannot_stop_in_error_state() {
      let player = test_player();
      player.state.lock().unwrap().status = PlaybackStatus::Error;
      assert!(player.stop().is_err());
   }

   #[test]
   fn set_loop_updates_state() {
      let player = test_player();
      let state = player.set_loop(true);
      assert!(state.looping);
   }

   #[test]
   fn play_from_ended_state() {
      let player = test_player();
      player.load("test.mp3", None).unwrap();
      player.state.lock().unwrap().status = PlaybackStatus::Ended;
      let resp = player.play().unwrap();

      assert_eq!(resp.player.status, PlaybackStatus::Playing);
   }
}
