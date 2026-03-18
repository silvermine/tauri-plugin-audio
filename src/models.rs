use serde::{Deserialize, Serialize};

/// Represents the current playback status of the audio player.
///
/// Modeled after common media player states (inspired by Vidstack's player state model),
/// adapted for a headless native audio context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PlaybackStatus {
   /// No audio source is loaded.
   #[default]
   Idle,

   /// An audio source is being loaded. Reserved for the real implementation
   /// where loading is asynchronous. The mock transitions directly from
   /// Idle to Ready.
   Loading,

   /// Audio source is loaded and ready to play.
   Ready,

   /// Audio is currently playing.
   Playing,

   /// Audio playback is paused.
   Paused,

   /// Audio playback has reached the end.
   Ended,

   /// An error occurred during loading or playback.
   Error,
}

/// Metadata for the audio source, used for OS transport control integration
/// (lock screen, notification shade, headphone controls, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioMetadata {
   pub title: Option<String>,
   pub artist: Option<String>,
   pub artwork: Option<String>,
}

/// The complete state of the audio player at a point in time.
///
/// Serialized to the TypeScript layer via Tauri's IPC bridge. Field names use camelCase
/// to match JavaScript conventions (e.g. `current_time` becomes `currentTime`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerState {
   pub status: PlaybackStatus,
   pub src: Option<String>,
   pub title: Option<String>,
   pub artist: Option<String>,
   pub artwork: Option<String>,
   pub current_time: f64,
   pub duration: f64,
   pub volume: f64,
   pub muted: bool,
   pub playback_rate: f64,
   /// Whether the audio should loop when it reaches the end.
   /// Named `looping` in Rust (since `loop` is a keyword), serialized as `"loop"` in JSON.
   #[serde(rename = "loop")]
   pub looping: bool,
   pub error: Option<String>,
}

impl Default for PlayerState {
   fn default() -> Self {
      Self {
         status: PlaybackStatus::Idle,
         src: None,
         title: None,
         artist: None,
         artwork: None,
         current_time: 0.0,
         duration: 0.0,
         volume: 1.0,
         muted: false,
         playback_rate: 1.0,
         looping: false,
         error: None,
      }
   }
}

/// Lightweight time update payload emitted at high frequency during playback.
///
/// Separated from [`PlayerState`] to avoid serializing the full state on every
/// tick (typically every 250ms). The real implementation emits this via the
/// `tauri-plugin-audio:time-update` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeUpdate {
   pub current_time: f64,
   pub duration: f64,
}

/// Response from a transport action (load, play, pause, stop, seek).
///
/// Wraps the resulting [`PlayerState`] with status-expectation metadata so the
/// TypeScript layer can detect unexpected state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioActionResponse {
   pub player: PlayerState,
   pub expected_status: PlaybackStatus,
   pub is_expected_status: bool,
}

impl AudioActionResponse {
   pub fn new(player: PlayerState, expected_status: PlaybackStatus) -> Self {
      let is_expected_status = player.status == expected_status;
      Self {
         player,
         expected_status,
         is_expected_status,
      }
   }
}
