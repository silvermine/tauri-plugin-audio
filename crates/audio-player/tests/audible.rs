use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use audio_player::{PlaybackStatus, PlayerState, RodioAudioPlayer, TimeUpdate};

const POLL_INTERVAL: Duration = Duration::from_millis(50);
const POSITION_WAIT_TIMEOUT: Duration = Duration::from_secs(15);

fn fixture_path() -> Result<PathBuf, Box<dyn Error>> {
   Ok(
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
         .join("..")
         .join("signalsmith")
         .join("tests")
         .join("fixtures")
         .join("music.wav")
         .canonicalize()?,
   )
}

fn wait_for_position(player: &RodioAudioPlayer, target_time: f64) -> Result<(), Box<dyn Error>> {
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
               state.current_time,
               target_time,
            )
            .into(),
         );
      }

      if Instant::now() >= deadline {
         return Err(
            format!(
               "timed out waiting for playback to reach {}s; current state: {:?} at {}s",
               target_time,
               state.status,
               state.current_time,
            )
            .into(),
         );
      }

      thread::sleep(POLL_INTERVAL);
   }
}

fn no_op_on_changed() -> Arc<dyn Fn(&PlayerState) + Send + Sync> {
   Arc::new(|_: &PlayerState| {})
}

fn no_op_on_time_update() -> Arc<dyn Fn(&TimeUpdate) + Send + Sync> {
   Arc::new(|_: &TimeUpdate| {})
}

fn play_fixture_at_rate(playback_rate: f64, end_seconds: usize) -> Result<(), Box<dyn Error>> {
   let fixture_path = fixture_path()?;
   let fixture_src = fixture_path.to_string_lossy().into_owned();
   let player = RodioAudioPlayer::new(no_op_on_changed(), no_op_on_time_update())?;

   player.load(&fixture_src, None)?;
   player.set_playback_rate(playback_rate)?;

   let state = player.get_state();
   assert!(
      state.duration >= end_seconds as f64,
      "fixture duration must be at least {}s, got {}s",
      end_seconds,
      state.duration,
   );

   player.play()?;
   wait_for_position(&player, end_seconds as f64)?;
   player.stop()?;

   Ok(())
}

fn play_fixture_at_rate_with_seek(
   playback_rate: f64,
   first_segment_end_seconds: usize,
   seek_seconds: usize,
   second_segment_end_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   let fixture_path = fixture_path()?;
   let fixture_src = fixture_path.to_string_lossy().into_owned();
   let player = RodioAudioPlayer::new(no_op_on_changed(), no_op_on_time_update())?;

   player.load(&fixture_src, None)?;
   player.set_playback_rate(playback_rate)?;

   let state = player.get_state();
   assert!(
      state.duration >= second_segment_end_seconds as f64,
      "fixture duration must be at least {}s, got {}s",
      second_segment_end_seconds,
      state.duration,
   );

   player.play()?;
   wait_for_position(&player, first_segment_end_seconds as f64)?;
   player.seek(seek_seconds as f64)?;
   wait_for_position(&player, second_segment_end_seconds as f64)?;
   player.stop()?;

   Ok(())
}

#[test]
#[ignore = "manual audible check; plays audio-player output through rodio"]
fn plays_fixture_at_one_point_two_five_x() -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate(1.25, 5)
}

#[test]
#[ignore = "manual audible check; plays audio-player output through rodio"]
fn plays_fixture_at_one_point_two_five_x_with_seek_from_five_seconds_to_two_seconds(
) -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate_with_seek(1.25, 5, 2, 5)
}
