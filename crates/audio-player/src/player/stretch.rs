use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use rodio::source::SeekError as RodioSeekError;
use rodio::{Sample, Source};
use signalsmith::PlaybackStream;

const OUTPUT_BLOCK_FRAMES: usize = 512;
const FLUSH_FRAMES: usize = OUTPUT_BLOCK_FRAMES * 4;

pub(crate) struct StretchSource {
   input: Box<dyn Source<Item = Sample> + Send>,
   stream: PlaybackStream,
   channels: rodio::ChannelCount,
   sample_rate: rodio::SampleRate,
   total_duration: Option<Duration>,
   flush_frames: usize,
   pending_input_buffer: VecDeque<Sample>,
   input_buffer: Vec<Sample>,
   output_buffer: Vec<Sample>,
   output_index: usize,
   flushed: bool,
   ended: bool,
}

impl StretchSource {
   pub(crate) fn new(input: Box<dyn Source<Item = Sample> + Send>, playback_rate: f64) -> Self {
      let channels = input.channels();
      let sample_rate = input.sample_rate();
      let total_duration = input.total_duration();
      let stream = PlaybackStream::with_rate(
         channels.get() as usize,
         sample_rate.get() as f32,
         playback_rate as f32,
      );
      let flush_frames = FLUSH_FRAMES.max(stream.output_latency().max(1));

      Self {
         input,
         stream,
         channels,
         sample_rate,
         total_duration,
         flush_frames,
         pending_input_buffer: VecDeque::new(),
         input_buffer: Vec::new(),
         output_buffer: Vec::new(),
         output_index: 0,
         flushed: false,
         ended: false,
      }
   }

   fn channel_count(&self) -> usize {
      self.channels.get() as usize
   }

   fn clear_output_buffer(&mut self) {
      self.output_buffer.clear();
      self.output_index = 0;
   }

   fn read_up_to_input_frames(&mut self, frame_count: usize) -> usize {
      let sample_count = frame_count * self.channel_count();

      self.input_buffer.clear();
      self.input_buffer.reserve(sample_count);

      while self.input_buffer.len() < sample_count {
         if let Some(sample) = self.pending_input_buffer.pop_front() {
            self.input_buffer.push(sample);
            continue;
         }

         let Some(sample) = self.input.next() else {
            break;
         };

         self.input_buffer.push(sample);
      }

      self.input_buffer.len() / self.channel_count()
   }

   fn read_exact_input_frames(
      &mut self,
      frame_count: usize,
      context: &str,
   ) -> std::result::Result<(), RodioSeekError> {
      let input_frames = self.read_up_to_input_frames(frame_count);

      if input_frames == frame_count {
         return Ok(());
      }

      Err(RodioSeekError::Other(Arc::new(std::io::Error::new(
         std::io::ErrorKind::UnexpectedEof,
         format!("Audio source ended during {context}"),
      ))))
   }

   fn flush_output(&mut self) -> bool {
      if self.flushed {
         return false;
      }

      self
         .output_buffer
         .resize(self.flush_frames * self.channel_count(), 0.0);
      self.stream.flush_interleaved(&mut self.output_buffer);
      self.flushed = true;

      !self.output_buffer.is_empty()
   }

   fn refill_output_buffer(&mut self) -> bool {
      if self.output_index < self.output_buffer.len() {
         return true;
      }

      self.clear_output_buffer();

      if self.flushed {
         return false;
      }

      let mut output_frames = OUTPUT_BLOCK_FRAMES.max(1);
      let mut input_frames = self.stream.input_samples_for_output(output_frames);
      let available_input_frames = self.read_up_to_input_frames(input_frames);

      if available_input_frames == 0 {
         return self.flush_output();
      }

      if available_input_frames < input_frames {
         while output_frames > 0 && self.stream.input_samples_for_output(output_frames) > available_input_frames {
            output_frames -= 1;
         }

         if output_frames == 0 {
            return self.flush_output();
         }

         input_frames = self.stream.input_samples_for_output(output_frames);
         self.pending_input_buffer.extend(
            self.input_buffer[(input_frames * self.channel_count())..]
               .iter()
               .copied(),
         );
         self.input_buffer.truncate(input_frames * self.channel_count());
      }

      self
         .output_buffer
         .resize(output_frames * self.channel_count(), 0.0);

      let consumed = self
         .stream
         .process_interleaved(&self.input_buffer, &mut self.output_buffer);

      debug_assert_eq!(consumed, input_frames);

      true
   }
}

impl Iterator for StretchSource {
   type Item = Sample;

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
      sample
   }
}

impl Source for StretchSource {
   fn current_span_len(&self) -> Option<usize> {
      if self.ended { Some(0) } else { None }
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
      self.input.try_seek(position).map_err(|error| {
         RodioSeekError::Other(Arc::new(std::io::Error::other(error.to_string())))
      })?;

      self.stream.reset();
      self.flushed = false;
      self.ended = false;
      self.clear_output_buffer();
      self.pending_input_buffer.clear();
      self.input_buffer.clear();

      let seek_input_frames = self.stream.output_seek_length();

      if seek_input_frames > 0 {
         self.read_exact_input_frames(seek_input_frames, "seek warmup")?;
         self.stream.output_seek_interleaved(&self.input_buffer);
         self.input_buffer.clear();
      }

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
      let mut stream = PlaybackStream::with_rate(channels, sample_rate as f32, playback_rate as f32);
      let flush_frames = FLUSH_FRAMES.max(stream.output_latency().max(1));
      let total_frames = decoded_samples.len() / channels;
      let mut cursor_frames = 0;
      let mut output = Vec::new();

      while cursor_frames < total_frames {
         let mut output_frames = OUTPUT_BLOCK_FRAMES;
         let requested_input_frames = stream.input_samples_for_output(output_frames);
         let remaining_frames = total_frames - cursor_frames;
         let mut actual_input_frames = remaining_frames.min(requested_input_frames);

         if actual_input_frames < requested_input_frames {
            while output_frames > 0 && stream.input_samples_for_output(output_frames) > actual_input_frames {
               output_frames -= 1;
            }

            if output_frames == 0 {
               break;
            }

            actual_input_frames = stream.input_samples_for_output(output_frames);
         }

         let start_sample = cursor_frames * channels;
         let end_sample = start_sample + (actual_input_frames * channels);
         let input_block = decoded_samples[start_sample..end_sample].to_vec();

         let mut output_block = vec![0.0; output_frames * channels];
         let consumed = stream.process_interleaved(&input_block, &mut output_block);

         assert_eq!(consumed, actual_input_frames);

         output.extend_from_slice(&output_block);
         cursor_frames += actual_input_frames;
      }

      let mut flush_block = vec![0.0; flush_frames * channels];
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
