mod common;

use std::time::Duration;

use rodio::Player;
use rodio::Source;
use rodio::source::Zero;

// Manual audible check for resuming 1.25x playback after inserting a 5 second silence gap.
fn main() -> common::ExampleResult {
   let playback_rate = 1.25;
   let played_seconds = 3;
   let silence_seconds = 5;
   let source_duration_seconds = common::SOURCE_DURATION_SECONDS;
   let fixture = common::fixture::FixtureSource::open()?;
   let channels = fixture.channels();
   let sample_rate_hz = fixture.sample_rate_hz();
   let source_frames = fixture.seconds_to_frames(source_duration_seconds);

   assert!(
      played_seconds < source_duration_seconds,
      "resume point must land before the source ends"
   );
   fixture.assert_length(source_frames, "resume preview length");

   let (sink, player) = common::fixture::open_player_with_source(
      fixture.into_streaming_source(playback_rate, source_frames),
   )?;

   common::fixture::wait_for_played_source_seconds(playback_rate, played_seconds as f64);
   player.pause();

   let silence_player = Player::connect_new(sink.mixer());
   silence_player.append(
      Zero::new(channels, sample_rate_hz)
         .take_duration(Duration::from_secs(silence_seconds as u64)),
   );
   silence_player.sleep_until_end();

   player.play();

   // clicks
   player.sleep_until_end();

   Ok(())
}
