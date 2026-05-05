use super::ExampleResult;
use super::fixture::{
   FixtureSource,
   play_streaming_audio,
};
use super::streaming_source::StreamingPlaybackSource;

pub const SOURCE_DURATION_SECONDS: usize = 5;

pub fn open_seeked_fixture_source(
   playback_rate: f32,
   seek_frame: usize,
   current_segment_end_frame: usize,
) -> ExampleResult<StreamingPlaybackSource> {
   let fixture = FixtureSource::open()?;
   let mut source = fixture.into_streaming_source(playback_rate, current_segment_end_frame);

   source.seek_to_frame(seek_frame)?;

   Ok(source)
}

pub fn play_fixture_at_rate(playback_rate: f32) -> ExampleResult {
   let fixture = FixtureSource::open()?;
   let source_frames = fixture.preview_frames();

   fixture.assert_length(source_frames, "preview duration");

   play_streaming_audio(fixture.into_streaming_source(playback_rate, source_frames))
}
