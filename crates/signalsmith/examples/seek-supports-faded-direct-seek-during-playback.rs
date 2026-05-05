mod common;

use std::time::Duration;

// Manual audible check for direct seek during 1.25x playback with source-managed fade handling.
fn main() -> common::ExampleResult {
   let playback_rate = 1.25;
   let first_segment_end_seconds = 4;
   let seek_seconds = 2;
   let second_segment_end_seconds = common::SOURCE_DURATION_SECONDS;
   let fixture = common::fixture::FixtureSource::open()?;
   let available_frames = fixture.available_frames_or_preview();
   let first_segment_end_frames = fixture.seconds_to_frames(first_segment_end_seconds);
   let second_segment_end_frame = fixture.seconds_to_frames(second_segment_end_seconds);

   assert!(
      first_segment_end_frames < available_frames,
      "direct seek must happen before the source ends"
   );
   assert!(
      seek_seconds < second_segment_end_seconds,
      "seek must land before the final segment end"
   );
   assert!(
      second_segment_end_frame <= available_frames,
      "second segment exceeds fixture length"
   );

   let (_sink, player) = common::fixture::open_player_with_source(
      fixture.into_streaming_source(playback_rate, available_frames),
   )?;

   common::fixture::wait_for_played_source_seconds(playback_rate, first_segment_end_seconds as f64);
   player.try_seek(Duration::from_secs_f64(seek_seconds as f64))?;
   common::fixture::wait_for_played_source_seconds(
      playback_rate,
      (second_segment_end_seconds - seek_seconds) as f64,
   );
   player.stop();

   // There are no clicks when SEEK_FADE_FRAMES is around 20+ (about 400 ms to fade
   // out and back in at 44 kHz), so this is probably because the fading process is
   // not only addressing discontinuities, but is also masking algorithm
   // reinitialization artifacts from the reset() function in the upstream
   // signalsmith seeking process.
   Ok(())
}
