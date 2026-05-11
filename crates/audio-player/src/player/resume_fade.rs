use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rodio::source::SeekError;
use rodio::{Sample, Source};

// About 360 ms at 44.1 kHz, or about 331 ms at 48 kHz.
const RESUME_FADE_FRAMES: usize = 15872;

#[derive(Clone)]
pub(crate) struct ResumeFadeHandle {
   generation: Arc<AtomicU64>,
}

impl ResumeFadeHandle {
   pub(crate) fn request_fade_in(&self) {
      self.generation.fetch_add(1, Ordering::Relaxed);
   }
}

pub(crate) struct ResumeFadeSource<S>
where
   S: Source<Item = Sample> + Send,
{
   inner: S,
   generation: Arc<AtomicU64>,
   observed_generation: u64,
   fade_frames: usize,
   remaining_fade_frames: usize,
   frame_position: usize,
   channels: rodio::ChannelCount,
}

impl<S> ResumeFadeSource<S>
where
   S: Source<Item = Sample> + Send,
{
   pub(crate) fn new(inner: S) -> (Self, ResumeFadeHandle) {
      Self::with_fade_frames(inner, RESUME_FADE_FRAMES)
   }

   fn with_fade_frames(inner: S, fade_frames: usize) -> (Self, ResumeFadeHandle) {
      let generation = Arc::new(AtomicU64::new(0));
      let channels = inner.channels();
      let observed_generation = generation.load(Ordering::Relaxed);
      let fade_frames = fade_frames.max(2);

      (
         Self {
            inner,
            generation: Arc::clone(&generation),
            observed_generation,
            fade_frames,
            remaining_fade_frames: 0,
            frame_position: 0,
            channels,
         },
         ResumeFadeHandle { generation },
      )
   }

   fn refresh_fade_request(&mut self) {
      let requested_generation = self.generation.load(Ordering::Relaxed);

      if requested_generation != self.observed_generation {
         self.observed_generation = requested_generation;
         self.remaining_fade_frames = self.fade_frames;
      }
   }

   fn current_gain(&self) -> f32 {
      if self.remaining_fade_frames == 0 {
         return 1.0;
      }

      envelope_for_step(
         self.fade_frames - self.remaining_fade_frames,
         self.fade_frames,
      )
   }

   fn advance_frame(&mut self) {
      self.frame_position += 1;

      if self.frame_position >= self.channels.get() as usize {
         self.frame_position = 0;

         if self.remaining_fade_frames > 0 {
            self.remaining_fade_frames -= 1;
         }
      }
   }
}

impl<S> Iterator for ResumeFadeSource<S>
where
   S: Source<Item = Sample> + Send,
{
   type Item = Sample;

   fn next(&mut self) -> Option<Self::Item> {
      self.refresh_fade_request();

      let sample = self.inner.next()?;
      let gain = self.current_gain();
      self.advance_frame();

      Some(sample * gain)
   }
}

impl<S> Source for ResumeFadeSource<S>
where
   S: Source<Item = Sample> + Send,
{
   fn current_span_len(&self) -> Option<usize> {
      self.inner.current_span_len()
   }

   fn try_seek(&mut self, position: Duration) -> Result<(), SeekError> {
      self.inner.try_seek(position)?;
      self.observed_generation = self.generation.load(Ordering::Relaxed);
      self.remaining_fade_frames = 0;
      self.frame_position = 0;

      Ok(())
   }

   fn channels(&self) -> rodio::ChannelCount {
      self.channels
   }

   fn sample_rate(&self) -> rodio::SampleRate {
      self.inner.sample_rate()
   }

   fn total_duration(&self) -> Option<Duration> {
      self.inner.total_duration()
   }
}

fn envelope_for_step(step: usize, total_steps: usize) -> f32 {
   let denominator = total_steps.saturating_sub(1).max(1) as f32;
   let progress = step.min(total_steps.saturating_sub(1)) as f32 / denominator;

   0.5 * (1.0 - (PI * progress).cos())
}
