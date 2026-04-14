use std::sync::Arc;
use std::time::Duration;

use rodio::source::SeekError as RodioSeekError;
use rodio::{Sample, Source};
use signalsmith_stretch::Stretch;

pub(crate) struct StretchSource {
   inner: Box<dyn Source<Item = Sample> + Send>,
   stretch: Stretch,
   channels: rodio::ChannelCount,
   sample_rate: rodio::SampleRate,
   total_duration: Option<Duration>,
   playback_rate: f64,
   input_latency_frames: usize,
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
      let stretch = Stretch::preset_default(channels.get() as u32, sample_rate.get());
      let input_latency_frames = stretch.input_latency().max(1);
      let output_latency_frames = stretch.output_latency().max(1);
      let chunk_frames = input_latency_frames.saturating_add(output_latency_frames).max(1);

      Self {
         inner,
         stretch,
         channels,
         sample_rate,
         total_duration,
         playback_rate,
         input_latency_frames,
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

   fn input_chunk_frames(&self) -> usize {
      let scaled_frames = (self.output_chunk_frames() as f64 * self.playback_rate).ceil() as usize;
      scaled_frames.max(self.input_latency_frames).max(1)
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

      let input_frames = self.read_input_frames(self.input_chunk_frames());

      if input_frames == 0 {
         if !self.input_exhausted {
            return false;
         }

         let flush_frames = self.output_latency_frames;
         self.output_buffer.resize(flush_frames * self.channel_count(), 0.0);
         self.stretch.flush(&mut self.output_buffer);
         self.flushed = true;
         return true;
      }

      let output_frames = ((input_frames as f64) / self.playback_rate).round() as usize;
      let output_frames = output_frames.max(1);
      self.output_buffer.resize(output_frames * self.channel_count(), 0.0);
      self.stretch.process(&self.input_buffer, &mut self.output_buffer);

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

      self.input_buffer.len() / self.channel_count()
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
      None
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
      self
         .inner
         .try_seek(position)
         .map_err(|error| RodioSeekError::Other(Arc::new(std::io::Error::other(error.to_string()))))?;

      self.reset_pipeline();
      Ok(())
   }
}
