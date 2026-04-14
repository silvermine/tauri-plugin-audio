use std::sync::Arc;

use tauri::{
   Emitter, Manager, Runtime,
   plugin::{Builder, TauriPlugin},
};
use tracing::warn;

mod commands;

pub use audio_player::{
   AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate,
};
pub use audio_player::{Error, OnChanged, OnTimeUpdate, Result, RodioAudioPlayer};

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access
/// the audio player APIs.
pub trait AudioExt<R: Runtime> {
   fn audio(&self) -> &RodioAudioPlayer;
}

impl<R: Runtime, T: Manager<R>> AudioExt<R> for T {
   fn audio(&self) -> &RodioAudioPlayer {
      self.state::<RodioAudioPlayer>().inner()
   }
}

/// Initializes the audio plugin with Rodio-backed desktop audio playback.
///
/// Opens the default audio output device. Fails during plugin setup if no
/// audio device is available.
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
      .setup(|app_handle, _api| {
         let state_handle = app_handle.clone();
         let time_handle = app_handle.clone();

         let player = RodioAudioPlayer::new(
            Arc::new(move |state| {
               if let Err(e) = state_handle.emit("tauri-plugin-audio:state-changed", state) {
                  warn!("Failed to emit state-changed event: {}", e);
               }
            }),
            Arc::new(move |time| {
               if let Err(e) = time_handle.emit("tauri-plugin-audio:time-update", time) {
                  warn!("Failed to emit time-update event: {}", e);
               }
            }),
         )
         .map_err(|e| e.to_string())?;

         app_handle.manage(player);
         Ok(())
      })
      .build()
}
