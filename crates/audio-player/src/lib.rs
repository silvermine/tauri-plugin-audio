pub mod error;
pub mod models;
mod net;
mod player;
pub mod transitions;
mod wsola;

pub use error::{Error, Result};
pub use models::{AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate};
pub use player::RodioAudioPlayer;

use std::sync::Arc;

/// Callback invoked when the player state changes (status transitions, settings, errors).
pub type OnChanged = Arc<dyn Fn(&PlayerState) + Send + Sync>;

/// Callback invoked at high frequency during playback (~250ms) with position updates.
pub type OnTimeUpdate = Arc<dyn Fn(&TimeUpdate) + Send + Sync>;
