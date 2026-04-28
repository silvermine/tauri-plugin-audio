pub mod error;
pub mod metadata;
pub mod models;
mod net;
mod player;
pub mod transitions;

pub use error::{Error, Result};
pub use models::{
   AudioActionResponse, AudioMetadata, LoopMode, PlaybackStatus, PlayerState, PlaylistItem,
   SettingsChange, StateChange, TimeUpdate, TrackChange,
};
pub use player::RodioAudioPlayer;

use std::sync::Arc;

/// Callback invoked on state-machine transitions (`status` / `error` changes).
pub type OnStateChanged = Arc<dyn Fn(&StateChange) + Send + Sync>;

/// Callback invoked when the active playlist item changes (initial load,
/// navigation, or auto-advance) with the freshly enriched item.
pub type OnTrackChanged = Arc<dyn Fn(&TrackChange) + Send + Sync>;

/// Callback invoked when a setting (`volume`, `muted`, `playbackRate`,
/// `loopMode`) is mutated. Carries only the changed field.
pub type OnSettingsChanged = Arc<dyn Fn(&SettingsChange) + Send + Sync>;

/// Callback invoked at high frequency during playback (~250ms) and on
/// user-initiated seek with the current position.
pub type OnTimeUpdate = Arc<dyn Fn(&TimeUpdate) + Send + Sync>;
