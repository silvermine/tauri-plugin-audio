mod common;

// Manual audible check for resuming playback by reopening the source at 1.25x after a seek.
fn main() -> common::ExampleResult {
   let playback_rate = 1.25;
   let first_segment_end_seconds = 4;
   let seek_seconds = 2;
   let second_segment_end_seconds = common::SOURCE_DURATION_SECONDS;
   let fixture = common::fixture::FixtureSource::open()?;
   let source_frames = fixture.preview_frames();
   let first_segment_end_frames = fixture.seconds_to_frames(first_segment_end_seconds);
   let seek_frame = fixture.seconds_to_frames(seek_seconds);
   let second_segment_end_frame = fixture.seconds_to_frames(second_segment_end_seconds);

   fixture.assert_length(source_frames, "audible preview length");
   assert!(
      first_segment_end_frames <= source_frames,
      "first segment exceeds audible preview length"
   );
   assert!(
      seek_frame < second_segment_end_frame,
      "seek must land before the final segment end"
   );
   assert!(
      second_segment_end_frame <= source_frames,
      "second segment exceeds audible preview length"
   );

   common::fixture::play_streaming_audio(
      fixture.into_streaming_source(playback_rate, first_segment_end_frames),
   )?;

   let second_source =
      common::open_seeked_fixture_source(playback_rate, seek_frame, second_segment_end_frame)?;

   // no click
   common::fixture::play_streaming_audio(second_source)
}
