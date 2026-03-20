use tauri::{AppHandle, Runtime, command};

use audio_player::Result;
use audio_player::models::{AudioActionResponse, AudioMetadata, PlayerState};

use crate::AudioExt;

#[command]
pub(crate) async fn load<R: Runtime>(
   app: AppHandle<R>,
   src: String,
   metadata: Option<AudioMetadata>,
) -> Result<AudioActionResponse> {
   app.audio().load(&src, metadata)
}

#[command]
pub(crate) async fn play<R: Runtime>(app: AppHandle<R>) -> Result<AudioActionResponse> {
   app.audio().play()
}

#[command]
pub(crate) async fn pause<R: Runtime>(app: AppHandle<R>) -> Result<AudioActionResponse> {
   app.audio().pause()
}

#[command]
pub(crate) async fn stop<R: Runtime>(app: AppHandle<R>) -> Result<AudioActionResponse> {
   app.audio().stop()
}

#[command]
pub(crate) async fn seek<R: Runtime>(
   app: AppHandle<R>,
   position: f64,
) -> Result<AudioActionResponse> {
   app.audio().seek(position)
}

#[command]
pub(crate) async fn set_volume<R: Runtime>(app: AppHandle<R>, level: f64) -> Result<PlayerState> {
   app.audio().set_volume(level)
}

#[command]
pub(crate) async fn set_muted<R: Runtime>(app: AppHandle<R>, muted: bool) -> Result<PlayerState> {
   Ok(app.audio().set_muted(muted))
}

#[command]
pub(crate) async fn set_playback_rate<R: Runtime>(
   app: AppHandle<R>,
   rate: f64,
) -> Result<PlayerState> {
   app.audio().set_playback_rate(rate)
}

#[command]
pub(crate) async fn set_loop<R: Runtime>(app: AppHandle<R>, looping: bool) -> Result<PlayerState> {
   Ok(app.audio().set_loop(looping))
}

#[command]
pub(crate) async fn get_state<R: Runtime>(app: AppHandle<R>) -> Result<PlayerState> {
   Ok(app.audio().get_state())
}

#[command]
pub(crate) async fn is_native<R: Runtime>(_app: AppHandle<R>) -> Result<bool> {
   // Desktop is not "native" in the mobile plugin sense. When mobile implementations
   // are added, this will return `true` on iOS/Android to switch the event transport.
   Ok(false)
}
