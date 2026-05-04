use std::collections::VecDeque;
use std::time::Duration;

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
         while output_frames > 0
            && self.stream.input_samples_for_output(output_frames) > available_input_frames
         {
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
         self
            .input_buffer
            .truncate(input_frames * self.channel_count());
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
}
