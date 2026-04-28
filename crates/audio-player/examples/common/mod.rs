use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use audio_player::{
   PlaybackStatus, PlaylistItem, RodioAudioPlayer, SettingsChange, StateChange, TimeUpdate,
   TrackChange,
};

const POLL_INTERVAL: Duration = Duration::from_millis(50);
const POSITION_WAIT_TIMEOUT: Duration = Duration::from_secs(15);

pub type ExampleResult<T = ()> = Result<T, Box<dyn Error>>;

fn fixture_path() -> ExampleResult<PathBuf> {
   Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("..")
      .join("..")
      .join("fixtures")
      .join("music.wav")
      .canonicalize()?)
}

fn no_op_on_state() -> Arc<dyn Fn(&StateChange) + Send + Sync> {
   Arc::new(|_: &StateChange| {})
}

fn no_op_on_track() -> Arc<dyn Fn(&TrackChange) + Send + Sync> {
   Arc::new(|_: &TrackChange| {})
}

fn no_op_on_settings() -> Arc<dyn Fn(&SettingsChange) + Send + Sync> {
   Arc::new(|_: &SettingsChange| {})
}

fn no_op_on_time_update() -> Arc<dyn Fn(&TimeUpdate) + Send + Sync> {
   Arc::new(|_: &TimeUpdate| {})
}

pub fn open_fixture_player(
   playback_rate: f64,
   minimum_duration_seconds: usize,
) -> ExampleResult<Arc<RodioAudioPlayer>> {
   let fixture_src = fixture_path()?.to_string_lossy().into_owned();
   let player = RodioAudioPlayer::new(
      no_op_on_state(),
      no_op_on_track(),
      no_op_on_settings(),
      no_op_on_time_update(),
   )?;

   player.load(
      vec![PlaylistItem {
         src: fixture_src,
         metadata: None,
      }],
      None,
   )?;
   player.set_playback_rate(playback_rate)?;

   let state = player.get_state();
   assert!(
      state.duration >= minimum_duration_seconds as f64,
      "fixture duration must be at least {}s, got {}s",
      minimum_duration_seconds,
      state.duration,
   );

   Ok(player)
}

pub fn wait_for_position(player: &Arc<RodioAudioPlayer>, target_time: f64) -> ExampleResult {
   let deadline = Instant::now() + POSITION_WAIT_TIMEOUT;

   loop {
      let state = player.get_state();

      if state.status == PlaybackStatus::Error {
         let message = state
            .error
            .unwrap_or_else(|| "audio player entered the error state".to_string());

         return Err(message.into());
      }

      if state.current_time >= target_time {
         return Ok(());
      }

      if state.status == PlaybackStatus::Ended {
         return Err(
            format!(
               "playback ended at {}s before reaching target {}s",
               state.current_time, target_time,
            )
            .into(),
         );
      }

      if Instant::now() >= deadline {
         return Err(
            format!(
               "timed out waiting for playback to reach {}s; current state: {:?} at {}s",
               target_time, state.status, state.current_time,
            )
            .into(),
         );
      }

      thread::sleep(POLL_INTERVAL);
   }
}
