use std::sync::Arc;
use std::time::Duration;

use rodio::source::SeekError as RodioSeekError;
use rodio::{Sample, Source};
use signalsmith::PlaybackStream;

const OUTPUT_CHUNK_FRAMES: usize = 2048;

pub(crate) struct StretchSource {
   inner: Box<dyn Source<Item = Sample> + Send>,
   stretch: PlaybackStream,
   channels: rodio::ChannelCount,
   sample_rate: rodio::SampleRate,
   total_duration: Option<Duration>,
   output_latency_frames: usize,
   chunk_frames: usize,
   input_buffer: Vec<Sample>,
   output_buffer: Vec<Sample>,
   output_index: usize,
   input_exhausted: bool,
   flushed: bool,
}

impl StretchSource {
   pub(crate) fn new(inner: Box<dyn Source<Item = Sample> + Send>, playback_rate: f64) -> Self {
      let channels = inner.channels();
      let sample_rate = inner.sample_rate();
      let total_duration = inner.total_duration();
      let stretch = PlaybackStream::with_rate(
         channels.get() as usize,
         sample_rate.get() as f32,
         playback_rate as f32,
      );
      let input_latency_frames = stretch.input_latency().max(1);
      let output_latency_frames = stretch.output_latency().max(1);
      let chunk_frames = OUTPUT_CHUNK_FRAMES
         .max(input_latency_frames.saturating_add(output_latency_frames))
         .max(1);

      Self {
         inner,
         stretch,
         channels,
         sample_rate,
         total_duration,
         output_latency_frames,
         chunk_frames,
         input_buffer: Vec::new(),
         output_buffer: Vec::new(),
         output_index: 0,
         input_exhausted: false,
         flushed: false,
      }
   }

   fn channel_count(&self) -> usize {
      self.channels.get() as usize
   }

   fn output_chunk_frames(&self) -> usize {
      self.chunk_frames
   }

   fn prepare_output_buffer(&mut self, frame_count: usize) {
      let channel_count = self.channel_count();

      self.output_buffer.clear();
      self.output_buffer.resize(frame_count * channel_count, 0.0);
   }

   fn refill_output_buffer(&mut self) -> bool {
      if self.output_index < self.output_buffer.len() {
         return true;
      }

      self.output_buffer.clear();
      self.output_index = 0;

      if self.flushed {
         return false;
      }

      let requested_output_frames = self.output_chunk_frames();
      let requested_input_frames = self
         .stretch
         .input_samples_for_output(requested_output_frames);
      let input_frames = self.read_input_frames(requested_input_frames);

      if input_frames == 0 {
         if !self.input_exhausted {
            return false;
         }

         let flush_frames = self.output_latency_frames;
         self.prepare_output_buffer(flush_frames);
         self.stretch.flush_interleaved(&mut self.output_buffer);
         self.flushed = true;

         return !self.output_buffer.is_empty();
      }

      if input_frames < requested_input_frames {
         self
            .input_buffer
            .resize(requested_input_frames * self.channel_count(), 0.0);
      }

      self.prepare_output_buffer(requested_output_frames);
      self
         .stretch
         .process_interleaved(&self.input_buffer, &mut self.output_buffer);

      true
   }

   fn read_input_frames(&mut self, frame_count: usize) -> usize {
      let sample_count = frame_count * self.channel_count();

      self.input_buffer.clear();
      self.input_buffer.reserve(sample_count);

      while self.input_buffer.len() < sample_count {
         let Some(sample) = self.inner.next() else {
            self.input_exhausted = true;
            break;
         };

         self.input_buffer.push(sample);
      }

      let input_frames = self.input_buffer.len() / self.channel_count();
      self
         .input_buffer
         .truncate(input_frames * self.channel_count());

      input_frames
   }

   fn reset_pipeline(&mut self) {
      self.stretch.reset();
      self.input_buffer.clear();
      self.output_buffer.clear();
      self.output_index = 0;
      self.input_exhausted = false;
      self.flushed = false;
   }
}

impl Iterator for StretchSource {
   type Item = Sample;

   fn next(&mut self) -> Option<Self::Item> {
      while self.output_index >= self.output_buffer.len() {
         if !self.refill_output_buffer() {
            return None;
         }

         if self.output_buffer.is_empty() {
            return None;
         }
      }

      let sample = self.output_buffer.get(self.output_index).copied();
      self.output_index += 1;
      sample
   }
}

impl Source for StretchSource {
   fn current_span_len(&self) -> Option<usize> {
      Some(self.output_buffer.len().saturating_sub(self.output_index))
   }

   fn channels(&self) -> rodio::ChannelCount {
      self.channels
   }

   fn sample_rate(&self) -> rodio::SampleRate {
      self.sample_rate
   }

   fn total_duration(&self) -> Option<Duration> {
      self.total_duration
   }

   fn try_seek(&mut self, position: Duration) -> std::result::Result<(), RodioSeekError> {
      self.inner.try_seek(position).map_err(|error| {
         RodioSeekError::Other(Arc::new(std::io::Error::other(error.to_string())))
      })?;

      self.reset_pipeline();
      Ok(())
   }
}

#[cfg(test)]
mod tests {
   use std::num::{NonZeroU16, NonZeroU32};

   use super::*;

   struct TestSource {
      samples: Vec<Sample>,
      index: usize,
      channels: rodio::ChannelCount,
      sample_rate: rodio::SampleRate,
      total_duration: Option<Duration>,
   }

   impl TestSource {
      fn new(samples: Vec<Sample>, channels: usize, sample_rate: u32) -> Self {
         let frame_count = samples.len() / channels;

         Self {
            samples,
            index: 0,
            channels: NonZeroU16::new(channels as u16).unwrap(),
            sample_rate: NonZeroU32::new(sample_rate).unwrap(),
            total_duration: Some(Duration::from_secs_f64(
               frame_count as f64 / sample_rate as f64,
            )),
         }
      }
   }

   impl Iterator for TestSource {
      type Item = Sample;

      fn next(&mut self) -> Option<Self::Item> {
         let sample = self.samples.get(self.index).copied();

         if sample.is_some() {
            self.index += 1;
         }

         sample
      }
   }

   impl Source for TestSource {
      fn current_span_len(&self) -> Option<usize> {
         Some(self.samples.len().saturating_sub(self.index))
      }

      fn channels(&self) -> rodio::ChannelCount {
         self.channels
      }

      fn sample_rate(&self) -> rodio::SampleRate {
         self.sample_rate
      }

      fn total_duration(&self) -> Option<Duration> {
         self.total_duration
      }
   }

   fn generate_signal(channels: usize, sample_rate: u32, frame_count: usize) -> Vec<Sample> {
      let mut samples = Vec::with_capacity(frame_count * channels);

      for frame in 0..frame_count {
         let time = frame as f32 / sample_rate as f32;

         for channel in 0..channels {
            let frequency = match channel {
               0 => 220.0,
               1 => 330.0,
               _ => 440.0 + (channel as f32 * 55.0),
            };
            let harmonic = match channel {
               0 => 0.17,
               1 => 0.11,
               _ => 0.07,
            };
            let sample = ((2.0 * std::f32::consts::PI * frequency * time).sin() * 0.45)
               + ((2.0 * std::f32::consts::PI * frequency * 2.0 * time).sin() * harmonic);

            samples.push(sample);
         }
      }

      samples
   }

   fn render_reference(
      decoded_samples: &[Sample],
      channels: usize,
      sample_rate: u32,
      playback_rate: f64,
   ) -> Vec<Sample> {
      let mut stream =
         PlaybackStream::with_rate(channels, sample_rate as f32, playback_rate as f32);
      let input_latency_frames = stream.input_latency().max(1);
      let output_latency_frames = stream.output_latency().max(1);
      let chunk_frames = OUTPUT_CHUNK_FRAMES
         .max(input_latency_frames.saturating_add(output_latency_frames))
         .max(1);
      let total_frames = decoded_samples.len() / channels;
      let mut cursor_frames = 0;
      let mut output = Vec::new();

      while cursor_frames < total_frames {
         let requested_input_frames = stream.input_samples_for_output(chunk_frames);
         let remaining_frames = total_frames - cursor_frames;
         let actual_input_frames = remaining_frames.min(requested_input_frames);
         let start_sample = cursor_frames * channels;
         let end_sample = start_sample + (actual_input_frames * channels);
         let mut input_block = decoded_samples[start_sample..end_sample].to_vec();

         if actual_input_frames < requested_input_frames {
            input_block.resize(requested_input_frames * channels, 0.0);
         }

         let mut output_block = vec![0.0; chunk_frames * channels];
         let consumed = stream.process_interleaved(&input_block, &mut output_block);

         assert_eq!(consumed, requested_input_frames);

         output.extend_from_slice(&output_block);
         cursor_frames += actual_input_frames;
      }

      let mut flush_block = vec![0.0; output_latency_frames * channels];
      stream.flush_interleaved(&mut flush_block);
      output.extend_from_slice(&flush_block);

      output
   }

   fn assert_signal_matches_reference(channels: usize, playback_rate: f64) {
      let sample_rate = 48_000;
      let frame_count = 72_000;
      let decoded_samples = generate_signal(channels, sample_rate, frame_count);
      let expected = render_reference(&decoded_samples, channels, sample_rate, playback_rate);
      let source = TestSource::new(decoded_samples, channels, sample_rate);
      let actual = StretchSource::new(Box::new(source), playback_rate).collect::<Vec<Sample>>();
      let max_difference = actual
         .iter()
         .zip(expected.iter())
         .map(|(lhs, rhs)| (lhs - rhs).abs())
         .fold(0.0_f32, f32::max);

      assert_eq!(actual.len(), expected.len());
      assert!(max_difference <= 1.0e-6, "max_difference={max_difference}");
   }

   #[test]
   fn stretch_source_matches_reference_for_slow_playback() {
      assert_signal_matches_reference(1, 0.75);
   }

   #[test]
   fn stretch_source_matches_reference_for_fast_playback() {
      assert_signal_matches_reference(2, 1.25);
   }
}
