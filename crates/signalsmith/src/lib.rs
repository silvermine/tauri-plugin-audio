//! Pure-Rust partial port of Signalsmith Stretch.
//!
//! The source of truth for the DSP design is the upstream MIT-licensed C++
//! code:
//! - https://github.com/Signalsmith-Audio/signalsmith-stretch
//! - https://github.com/Signalsmith-Audio/linear
//!
//! This crate intentionally ports the streaming time-stretch path and the
//! modified half-bin STFT it depends on. Pitch shifting, custom frequency maps,
//! and formant processing are outside the current project scope.

// Many loops intentionally mirror the upstream C++ buffer indexing across
// circular buffers and paired spectra. Rewriting them as iterators often makes
// the correspondence to Signalsmith's source harder to audit.
#![allow(clippy::needless_range_loop)]

mod playback;
mod port;

pub use playback::{PlaybackRateController, PlaybackStream};
pub use port::Stretch;
