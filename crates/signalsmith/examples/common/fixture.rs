use std::fs::File;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::PathBuf;
use std::time::Duration;

use rodio::stream::MixerDeviceSink;
use rodio::{Decoder, DeviceSinkBuilder, Player, Source};

use super::streaming_source::StreamingPlaybackSource;
use super::ExampleResult;

pub(crate) type BoxedSource = Box<dyn Source<Item = f32> + Send>;
type FixtureDecoder = (BoxedSource, NonZeroU16, NonZeroU32, Option<Duration>);

pub(crate) struct FixtureSource {
   input: BoxedSource,
   channels: NonZeroU16,
   sample_rate_hz: NonZeroU32,
   total_duration: Option<Duration>,
}

impl FixtureSource {
   pub(crate) fn open() -> ExampleResult<Self> {
      let (input, channels, sample_rate_hz, total_duration) = open_fixture_decoder()?;

      Ok(Self {
         input,
         channels,
         sample_rate_hz,
         total_duration,
      })
   }

   pub(crate) fn channels(&self) -> NonZeroU16 {
      self.channels
   }

   pub(crate) fn sample_rate_hz(&self) -> NonZeroU32 {
      self.sample_rate_hz
   }

   pub(crate) fn preview_frames(&self) -> usize {
      self.sample_rate_hz.get() as usize * super::scenarios::SOURCE_DURATION_SECONDS
   }

   pub(crate) fn available_frames_or_preview(&self) -> usize {
      self
         .total_duration
         .map(|duration| (duration.as_secs_f64() * self.sample_rate_hz.get() as f64).floor() as usize)
         .unwrap_or(self.preview_frames())
   }

   pub(crate) fn seconds_to_frames(&self, seconds: usize) -> usize {
      self.sample_rate_hz.get() as usize * seconds
   }

   pub(crate) fn assert_length(&self, required_frames: usize, label: &str) {
      if let Some(available_frames) = self.total_duration.map(|duration| {
         (duration.as_secs_f64() * self.sample_rate_hz.get() as f64).floor() as usize
      }) {
         assert!(
            required_frames <= available_frames,
            "{label} exceeds fixture length"
         );
      }
   }

   pub(crate) fn into_streaming_source(
      self,
      playback_rate: f32,
      current_segment_end_frame: usize,
   ) -> StreamingPlaybackSource {
      StreamingPlaybackSource::new(
         self.input,
         self.channels,
         self.sample_rate_hz,
         playback_rate,
         current_segment_end_frame,
      )
   }
}

fn fixture_path() -> PathBuf {
   PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("fixtures")
      .join("music.wav")
}

fn open_fixture_decoder() -> ExampleResult<FixtureDecoder> {
   let file = File::open(fixture_path())?;
   let decoder = Decoder::try_from(file)?;
   let channels = decoder.channels();
   let sample_rate_hz = decoder.sample_rate();
   let total_duration = decoder.total_duration();

   Ok((Box::new(decoder), channels, sample_rate_hz, total_duration))
}

pub(crate) fn open_player_with_source(
   source: StreamingPlaybackSource,
) -> ExampleResult<(MixerDeviceSink, Player)> {
   let sink = DeviceSinkBuilder::open_default_sink()?;
   let player = Player::connect_new(sink.mixer());

   player.append(source);

   Ok((sink, player))
}

pub(crate) fn play_streaming_audio(source: StreamingPlaybackSource) -> ExampleResult {
   let (_sink, player) = open_player_with_source(source)?;
   player.sleep_until_end();

   Ok(())
}

pub(crate) fn wait_for_played_source_seconds(playback_rate: f32, source_seconds: f64) {
   std::thread::sleep(Duration::from_secs_f64(
      source_seconds / playback_rate as f64,
   ));
}
