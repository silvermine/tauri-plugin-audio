mod common;

use std::thread;
use std::time::Duration;

// Manual audible check for pausing 1.25x playback for 5 seconds and then resuming the same source.
fn main() -> common::ExampleResult {
   let playback_rate = 1.25;
   let played_seconds = 3;
   let paused_seconds = 5;
   let source_duration_seconds = common::SOURCE_DURATION_SECONDS;
   let fixture = common::fixture::FixtureSource::open()?;
   let source_frames = fixture.seconds_to_frames(source_duration_seconds);

   assert!(
      played_seconds < source_duration_seconds,
      "pause point must land before the source ends"
   );
   fixture.assert_length(source_frames, "resume preview length");

   let (_sink, player) = common::fixture::open_player_with_source(
      fixture.into_streaming_source(playback_rate, source_frames),
   )?;

   common::fixture::wait_for_played_source_seconds(playback_rate, played_seconds as f64);
   player.pause();
   thread::sleep(Duration::from_secs(paused_seconds as u64));
   player.play();

   // clicks
   player.sleep_until_end();

   Ok(())
}
