use std::error::Error;
use std::fs::File;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::PathBuf;
use std::time::Duration;

use rodio::source::SeekError as RodioSeekError;
use rodio::{Decoder, DeviceSinkBuilder, Player, Source};
use signalsmith::PlaybackStream;

const SOURCE_DURATION_SECONDS: usize = 5;
const RESTART_FADE_MILLISECONDS: u32 = 100;
const OUTPUT_BLOCK_FRAMES: usize = 512;
const FLUSH_FRAMES: usize = OUTPUT_BLOCK_FRAMES * 4;

type BoxedSource = Box<dyn Source<Item = f32> + Send>;
type FixtureDecoder = (BoxedSource, NonZeroU16, NonZeroU32, Option<Duration>);

fn fixture_path() -> PathBuf {
   PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("fixtures")
      .join("music.wav")
}

fn open_fixture_decoder() -> Result<FixtureDecoder, Box<dyn Error>> {
   let file = File::open(fixture_path())?;
   let decoder = Decoder::try_from(file)?;
   let channels = decoder.channels();
   let sample_rate_hz = decoder.sample_rate();
   let total_duration = decoder.total_duration();

   Ok((Box::new(decoder), channels, sample_rate_hz, total_duration))
}

fn preview_source_frames(sample_rate_hz: u32) -> usize {
   sample_rate_hz as usize * SOURCE_DURATION_SECONDS
}

fn restart_fade_frames(sample_rate_hz: NonZeroU32) -> usize {
   ((sample_rate_hz.get() as u64 * RESTART_FADE_MILLISECONDS as u64) / 1000) as usize
}

fn available_source_frames(total_duration: Option<Duration>, sample_rate_hz: u32) -> Option<usize> {
   total_duration.map(|duration| (duration.as_secs_f64() * sample_rate_hz as f64).floor() as usize)
}

fn assert_fixture_length(
   total_duration: Option<Duration>,
   sample_rate_hz: NonZeroU32,
   required_frames: usize,
   label: &str,
) {
   if let Some(available_frames) = available_source_frames(total_duration, sample_rate_hz.get()) {
      assert!(
         required_frames <= available_frames,
         "{label} exceeds fixture length"
      );
   }
}

struct StreamingPlaybackSource {
   input: BoxedSource,
   stream: PlaybackStream,
   channels: NonZeroU16,
   sample_rate_hz: NonZeroU32,
   input_buffer: Vec<f32>,
   output_buffer: Vec<f32>,
   output_index: usize,
   current_input_frame: usize,
   current_segment_end_frame: usize,
   fade_in_samples_remaining: usize,
   fade_in_samples_total: usize,
   flushed: bool,
   ended: bool,
}

impl StreamingPlaybackSource {
   fn new(
      input: BoxedSource,
      channels: NonZeroU16,
      sample_rate_hz: NonZeroU32,
      playback_rate: f32,
      current_segment_end_frame: usize,
   ) -> Self {
      let stream = PlaybackStream::with_rate(
         usize::from(channels.get()),
         sample_rate_hz.get() as f32,
         playback_rate,
      );

      Self {
         input,
         stream,
         channels,
         sample_rate_hz,
         input_buffer: Vec::new(),
         output_buffer: Vec::new(),
         output_index: 0,
         current_input_frame: 0,
         current_segment_end_frame,
         fade_in_samples_remaining: 0,
         fade_in_samples_total: 0,
         flushed: false,
         ended: false,
      }
   }

   fn with_fade_in_frames(mut self, frame_count: usize) -> Self {
      let fade_in_samples = frame_count * self.channel_count();

      self.fade_in_samples_remaining = fade_in_samples;
      self.fade_in_samples_total = fade_in_samples;
      self
   }

   fn channel_count(&self) -> usize {
      usize::from(self.channels.get())
   }

   fn clear_output_buffer(&mut self) {
      self.output_buffer.clear();
      self.output_index = 0;
   }

   fn read_exact_input_frames(&mut self, input_frames: usize, context: &str) {
      let input_samples = input_frames * self.channel_count();

      self.input_buffer.clear();
      self.input_buffer.reserve(input_samples);

      while self.input_buffer.len() < input_samples {
         let Some(sample) = self.input.next() else {
            panic!("audible fixture too short during {context}");
         };

         self.input_buffer.push(sample);
      }

      self.current_input_frame += input_frames;
   }

   fn seek_to_frame(&mut self, seek_frame: usize) -> Result<(), RodioSeekError> {
      self.input.try_seek(Duration::from_secs_f64(
         seek_frame as f64 / self.sample_rate_hz.get() as f64,
      ))?;

      self.current_input_frame = seek_frame;
      self.flushed = false;
      self.ended = false;
      self.clear_output_buffer();

      let seek_input_frames = self.stream.output_seek_length();
      self.read_exact_input_frames(seek_input_frames, "seek warmup");
      self.stream.output_seek_interleaved(&self.input_buffer);
      self.input_buffer.clear();

      Ok(())
   }

   fn flush_output(&mut self) -> bool {
      if self.flushed {
         return false;
      }

      self
         .output_buffer
         .resize(FLUSH_FRAMES * self.channel_count(), 0.0);
      self.stream.flush_interleaved(&mut self.output_buffer);
      self.flushed = true;

      !self.output_buffer.is_empty()
   }

   fn refill_output_buffer(&mut self) -> bool {
      if self.output_index < self.output_buffer.len() {
         return true;
      }

      self.clear_output_buffer();

      if self.current_input_frame >= self.current_segment_end_frame {
         return self.flush_output();
      }

      let remaining_frames = self.current_segment_end_frame - self.current_input_frame;
      let mut output_frames = OUTPUT_BLOCK_FRAMES.max(1);

      while self.stream.input_samples_for_output(output_frames) > remaining_frames {
         output_frames -= 1;
      }

      let input_frames = self.stream.input_samples_for_output(output_frames);

      self.read_exact_input_frames(input_frames, "streaming playback");
      self
         .output_buffer
         .resize(output_frames * self.channel_count(), 0.0);

      let consumed = self
         .stream
         .process_interleaved(&self.input_buffer, &mut self.output_buffer);

      assert_eq!(consumed, input_frames);
      true
   }
}

impl Iterator for StreamingPlaybackSource {
   type Item = f32;

   fn next(&mut self) -> Option<Self::Item> {
      while self.output_index >= self.output_buffer.len() {
         if !self.refill_output_buffer() {
            self.ended = true;
            return None;
         }

         if self.output_buffer.is_empty() {
            self.ended = true;
            return None;
         }
      }

      let sample = self.output_buffer.get(self.output_index).copied();
      self.output_index += 1;

      sample.map(|value| {
         if self.fade_in_samples_remaining == 0 || self.fade_in_samples_total == 0 {
            return value;
         }

         let emitted_samples = self.fade_in_samples_total - self.fade_in_samples_remaining;
         self.fade_in_samples_remaining -= 1;

         value * (emitted_samples as f32 / self.fade_in_samples_total as f32)
      })
   }
}

impl Source for StreamingPlaybackSource {
   fn current_span_len(&self) -> Option<usize> {
      if self.ended { Some(0) } else { None }
   }

   fn channels(&self) -> NonZeroU16 {
      self.channels
   }

   fn sample_rate(&self) -> NonZeroU32 {
      self.sample_rate_hz
   }

   fn total_duration(&self) -> Option<Duration> {
      None
   }
}

fn play_streaming_audio(source: StreamingPlaybackSource) -> Result<(), Box<dyn Error>> {
   let sink = DeviceSinkBuilder::open_default_sink()?;
   let player = Player::connect_new(sink.mixer());

   player.append(source);
   player.sleep_until_end();

   Ok(())
}

fn play_fixture_at_rate(playback_rate: f32) -> Result<(), Box<dyn Error>> {
   let (input, channels, sample_rate_hz, total_duration) = open_fixture_decoder()?;
   let source_frames = preview_source_frames(sample_rate_hz.get());

   assert!(
      channels.get() > 0,
      "fixture WAV must have at least one channel"
   );
   assert_fixture_length(
      total_duration,
      sample_rate_hz,
      source_frames,
      "preview duration",
   );

   play_streaming_audio(StreamingPlaybackSource::new(
      input,
      channels,
      sample_rate_hz,
      playback_rate,
      source_frames,
   ))
}

fn play_fixture_at_rate_with_seek(
   playback_rate: f32,
   first_segment_end_seconds: usize,
   seek_seconds: usize,
   second_segment_end_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   let (first_input, channels, sample_rate_hz, total_duration) = open_fixture_decoder()?;
   let source_frames = preview_source_frames(sample_rate_hz.get());
   let first_segment_end_frames = sample_rate_hz.get() as usize * first_segment_end_seconds;
   let seek_frame = sample_rate_hz.get() as usize * seek_seconds;
   let second_segment_end_frame = sample_rate_hz.get() as usize * second_segment_end_seconds;

   assert!(
      channels.get() > 0,
      "fixture WAV must have at least one channel"
   );
   assert_fixture_length(
      total_duration,
      sample_rate_hz,
      source_frames,
      "audible preview length",
   );
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

   play_streaming_audio(StreamingPlaybackSource::new(
      first_input,
      channels,
      sample_rate_hz,
      playback_rate,
      first_segment_end_frames,
   ))?;

   let (second_input, second_channels, second_sample_rate_hz, _) = open_fixture_decoder()?;
   let mut second_source = StreamingPlaybackSource::new(
      second_input,
      second_channels,
      second_sample_rate_hz,
      playback_rate,
      second_segment_end_frame,
   )
   .with_fade_in_frames(restart_fade_frames(second_sample_rate_hz));

   second_source.seek_to_frame(seek_frame)?;
   play_streaming_audio(second_source)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_zero_point_seven_five_x() -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate(0.75)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_one_point_two_five_x() -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate(1.25)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_one_point_two_five_x_with_seek_from_five_seconds_to_two_seconds()
-> Result<(), Box<dyn Error>> {
   play_fixture_at_rate_with_seek(1.25, 5, 2, 5)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_two_x() -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate(2.0)
}
