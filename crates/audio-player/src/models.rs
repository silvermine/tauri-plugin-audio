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

   /// An audio source is being fetched or decoded. The player enters this
   /// state before starting I/O and transitions to `Ready` on success.
   Loading,

   /// Audio source is loaded and ready to play.
   Ready,

   /// Audio is currently playing.
   Playing,

   /// Audio playback is paused.
   Paused,

   /// Audio playback has reached the end (last item in playlist with looping disabled).
   Ended,

   /// An error occurred during loading or playback.
   Error,
}

/// How the player advances when the current item finishes.
///
/// * `Off` — stop after the last item; emit `Ended`.
/// * `One` — repeat the current item indefinitely.
/// * `All` — wrap from the last item back to the first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LoopMode {
   #[default]
   Off,
   One,
   All,
}

/// Metadata for an audio source, used for OS transport control integration
/// (lock screen, notification shade, headphone controls, etc.).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioMetadata {
   pub title: Option<String>,
   pub artist: Option<String>,
   pub artwork: Option<String>,
}

/// A single item in a playlist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaylistItem {
   /// URL or file path of the audio source.
   pub src: String,

   /// Optional metadata for OS transport controls.
   #[serde(default, skip_serializing_if = "Option::is_none")]
   pub metadata: Option<AudioMetadata>,
}

/// The complete state of the audio player at a point in time.
///
/// Serialized to the TypeScript layer via Tauri's IPC bridge. Field names use camelCase
/// to match JavaScript conventions (e.g. `current_time` becomes `currentTime`).
///
/// Per-item fields like `current_time`, `duration`, and (via `current()`) the active
/// item's metadata refer to whichever playlist item is at `current_index`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerState {
   pub status: PlaybackStatus,

   /// All items currently loaded. Empty when `status == Idle`.
   pub playlist: Vec<PlaylistItem>,

   /// Index into `playlist` of the active item, or `None` when no playlist is loaded.
   pub current_index: Option<usize>,

   pub current_time: f64,
   pub duration: f64,
   pub volume: f64,
   pub muted: bool,
   pub playback_rate: f64,
   pub loop_mode: LoopMode,
   pub error: Option<String>,
}

impl Default for PlayerState {
   fn default() -> Self {
      Self {
         status: PlaybackStatus::Idle,
         playlist: Vec::new(),
         current_index: None,
         current_time: 0.0,
         duration: 0.0,
         volume: 1.0,
         muted: false,
         playback_rate: 1.0,
         loop_mode: LoopMode::Off,
         error: None,
      }
   }
}

impl PlayerState {
   /// The currently active playlist item, if any.
   pub fn current(&self) -> Option<&PlaylistItem> {
      self.current_index.and_then(|i| self.playlist.get(i))
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

/// Response from a transport action (prepare, play, pause, stop, seek, next, prev).
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
