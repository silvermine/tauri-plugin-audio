use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;
use std::time::Duration;

use rodio::{Sample, Source};

use crate::error::{Error, Result};

const BUFFERED_CHUNK_SAMPLES: usize = 16_384;
const BUFFERED_CHUNK_CAPACITY: usize = 8;

enum WorkerMessage {
   Samples(Vec<Sample>),
   End,
}

pub(crate) struct PrebufferedSource {
   receiver: Receiver<WorkerMessage>,
   current_chunk: Vec<Sample>,
   current_index: usize,
   channels: rodio::ChannelCount,
   sample_rate: rodio::SampleRate,
   total_duration: Option<Duration>,
   ended: bool,
}

impl PrebufferedSource {
   pub(crate) fn new(inner: Box<dyn Source<Item = Sample> + Send>) -> Result<Self> {
      let channels = inner.channels();
      let sample_rate = inner.sample_rate();
      let total_duration = inner.total_duration();
      let (sender, receiver) = sync_channel(BUFFERED_CHUNK_CAPACITY);

      thread::Builder::new()
         .name("audio-prebuffer".into())
         .spawn(move || {
            worker_loop(inner, sender);
         })
         .map_err(|error| Error::Audio(format!("Failed to spawn audio prebuffer worker: {error}")))?;

      Ok(Self {
         receiver,
         current_chunk: Vec::new(),
         current_index: 0,
         channels,
         sample_rate,
         total_duration,
         ended: false,
      })
   }

   fn refill_chunk(&mut self) -> bool {
      while self.current_index >= self.current_chunk.len() {
         if self.ended {
            return false;
         }

         match self.receiver.recv() {
            Ok(WorkerMessage::Samples(samples)) => {
               self.current_chunk = samples;
               self.current_index = 0;
            }
            Ok(WorkerMessage::End) | Err(_) => {
               self.current_chunk.clear();
               self.current_index = 0;
               self.ended = true;
               return false;
            }
         }
      }

      true
   }
}

impl Iterator for PrebufferedSource {
   type Item = Sample;

   fn next(&mut self) -> Option<Self::Item> {
      if !self.refill_chunk() {
         return None;
      }

      let sample = self.current_chunk.get(self.current_index).copied();
      self.current_index += 1;
      sample
   }
}

impl Source for PrebufferedSource {
   fn current_span_len(&self) -> Option<usize> {
      Some(self.current_chunk.len().saturating_sub(self.current_index))
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

fn worker_loop(
   mut inner: Box<dyn Source<Item = Sample> + Send>,
   sender: SyncSender<WorkerMessage>,
) {
   loop {
      let mut samples = Vec::with_capacity(BUFFERED_CHUNK_SAMPLES);

      while samples.len() < BUFFERED_CHUNK_SAMPLES {
         let Some(sample) = inner.next() else {
            if !samples.is_empty() && sender.send(WorkerMessage::Samples(samples)).is_err() {
               return;
            }

            let _ = sender.send(WorkerMessage::End);
            return;
         };

         samples.push(sample);
      }

      if sender.send(WorkerMessage::Samples(samples)).is_err() {
         return;
      }
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
            total_duration: Some(Duration::from_secs_f64(frame_count as f64 / sample_rate as f64)),
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
            let frequency = 220.0 + (channel as f32 * 110.0);
            let sample = (2.0 * std::f32::consts::PI * frequency * time).sin() * 0.5;

            samples.push(sample);
         }
      }

      samples
   }

   #[test]
   fn prebuffered_source_preserves_mono_samples() {
      let expected = generate_signal(1, 48_000, 24_000);
      let source = TestSource::new(expected.clone(), 1, 48_000);
      let actual = PrebufferedSource::new(Box::new(source)).unwrap().collect::<Vec<Sample>>();

      assert_eq!(actual, expected);
   }

   #[test]
   fn prebuffered_source_preserves_stereo_samples() {
      let expected = generate_signal(2, 48_000, 24_000);
      let source = TestSource::new(expected.clone(), 2, 48_000);
      let actual = PrebufferedSource::new(Box::new(source)).unwrap().collect::<Vec<Sample>>();

      assert_eq!(actual, expected);
   }
}
