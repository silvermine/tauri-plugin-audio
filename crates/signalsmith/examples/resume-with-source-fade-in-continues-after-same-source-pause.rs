mod common;

use std::thread;
use std::time::Duration;

// Manual audible check for resuming 1.25x playback on the same source with a source fade-in.
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

   let (source, resume_fade_handle) = fixture
      .into_streaming_source(playback_rate, source_frames)
      .into_resume_fade_source();
   let (_sink, player) = common::fixture::open_player_with_source(source)?;

   common::fixture::wait_for_played_source_seconds(playback_rate, played_seconds as f64);
   player.pause();
   thread::sleep(Duration::from_secs(paused_seconds as u64));
   resume_fade_handle.request_fade_in();
   player.play();

   // no clicks when SEEK_FADE_FRAMES is around 10+
   player.sleep_until_end();

   Ok(())
}
