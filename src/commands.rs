use tauri::{AppHandle, Runtime, command};

use audio_player::Result;
use audio_player::models::{AudioActionResponse, LoopMode, PlayerState, PlaylistItem};

use crate::AudioExt;

#[command]
pub(crate) async fn load<R: Runtime>(
   app: AppHandle<R>,
   playlist: Vec<PlaylistItem>,
   start_index: Option<usize>,
) -> Result<AudioActionResponse> {
   app.audio().load(playlist, start_index)
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
pub(crate) async fn next<R: Runtime>(app: AppHandle<R>) -> Result<AudioActionResponse> {
   app.audio().next()
}

#[command]
pub(crate) async fn prev<R: Runtime>(app: AppHandle<R>) -> Result<AudioActionResponse> {
   app.audio().prev()
}

#[command]
pub(crate) async fn jump_to<R: Runtime>(
   app: AppHandle<R>,
   index: usize,
) -> Result<AudioActionResponse> {
   app.audio().jump_to(index)
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
pub(crate) async fn set_loop_mode<R: Runtime>(
   app: AppHandle<R>,
   mode: LoopMode,
) -> Result<PlayerState> {
   Ok(app.audio().set_loop_mode(mode))
}

#[command]
pub(crate) async fn get_state<R: Runtime>(app: AppHandle<R>) -> Result<PlayerState> {
   Ok(app.audio().get_state())
}

#[command]
pub(crate) async fn is_native<R: Runtime>(_app: AppHandle<R>) -> Result<bool> {
   Ok(false)
}
