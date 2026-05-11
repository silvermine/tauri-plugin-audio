mod common;

use std::thread;
use std::time::Duration;

// Manual audible check for resuming 1.25x playback by reopening the source after a 5 second pause.
fn main() -> common::ExampleResult {
   let playback_rate = 1.25;
   let played_seconds = 3;
   let paused_seconds = 5;
   let source_duration_seconds = common::SOURCE_DURATION_SECONDS;
   let fixture = common::fixture::FixtureSource::open()?;
   let source_frames = fixture.seconds_to_frames(source_duration_seconds);
   let seek_frame = fixture.seconds_to_frames(played_seconds);

   assert!(
      played_seconds < source_duration_seconds,
      "pause point must land before the source ends"
   );
   fixture.assert_length(source_frames, "resume preview length");

   common::fixture::play_streaming_audio(fixture.into_streaming_source(playback_rate, seek_frame))?;

   thread::sleep(Duration::from_secs(paused_seconds as u64));

   let second_source =
      common::open_seeked_fixture_source(playback_rate, seek_frame, source_frames)?;

   // clicks
   common::fixture::play_streaming_audio(second_source)
}
