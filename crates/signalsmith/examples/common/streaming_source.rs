use std::f32::consts::PI;
use std::num::{NonZeroU16, NonZeroU32};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rodio::Source;
use rodio::source::SeekError as RodioSeekError;
use signalsmith::PlaybackStream;

use super::fixture::BoxedSource;

const OUTPUT_BLOCK_FRAMES: usize = 512;
const FLUSH_FRAMES: usize = OUTPUT_BLOCK_FRAMES * 4;
const SEEK_FADE_FRAMES: usize = OUTPUT_BLOCK_FRAMES * 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SeekTransition {
   Idle,
   FadingOut {
      target_frame: usize,
      remaining_frames: usize,
      total_frames: usize,
   },
   Pending {
      target_frame: usize,
   },
   FadingIn {
      remaining_frames: usize,
      total_frames: usize,
   },
}

#[derive(Clone, Default)]
pub(crate) struct ResumeFadeHandle {
   start_fade_in: Arc<AtomicBool>,
}

impl ResumeFadeHandle {
   pub(crate) fn request_fade_in(&self) {
      self.start_fade_in.store(true, Ordering::Release);
   }

   fn take_fade_in_request(&self) -> bool {
      self.start_fade_in.swap(false, Ordering::AcqRel)
   }
}

pub(crate) struct StreamingPlaybackSource {
   input: BoxedSource,
   stream: PlaybackStream,
   channels: NonZeroU16,
   sample_rate_hz: NonZeroU32,
   input_buffer: Vec<f32>,
   output_buffer: Vec<f32>,
   output_index: usize,
   current_input_frame: usize,
   current_segment_end_frame: usize,
   flushed: bool,
   ended: bool,
   frame_position: usize,
   current_envelope: f32,
   seek_transition: SeekTransition,
   resume_fade_handle: ResumeFadeHandle,
}

impl StreamingPlaybackSource {
   pub(crate) fn new(
      input: BoxedSource,
      channels: NonZeroU16,
      sample_rate_hz: NonZeroU32,
      playback_rate: f32,
      current_segment_end_frame: usize,
   ) -> Self {
      Self::new_with_resume_fade_handle(
         input,
         channels,
         sample_rate_hz,
         playback_rate,
         current_segment_end_frame,
         ResumeFadeHandle::default(),
      )
   }

   fn new_with_resume_fade_handle(
      input: BoxedSource,
      channels: NonZeroU16,
      sample_rate_hz: NonZeroU32,
      playback_rate: f32,
      current_segment_end_frame: usize,
      resume_fade_handle: ResumeFadeHandle,
   ) -> Self {
      let stream = PlaybackStream::with_rate(
         usize::from(channels.get()),
         sample_rate_hz.get() as f32,
         playback_rate,
      )
      .unwrap();

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
         flushed: false,
         ended: false,
         frame_position: 0,
         current_envelope: 1.0,
         seek_transition: SeekTransition::Idle,
         resume_fade_handle,
      }
   }

   pub(crate) fn into_resume_fade_source(self) -> (Self, ResumeFadeHandle) {
      let resume_fade_handle = self.resume_fade_handle.clone();

      (self, resume_fade_handle)
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

   pub(crate) fn seek_to_frame(&mut self, seek_frame: usize) -> Result<(), RodioSeekError> {
      self.input.try_seek(Duration::from_secs_f64(
         seek_frame as f64 / self.sample_rate_hz.get() as f64,
      ))?;

      self.current_input_frame = seek_frame;
      self.flushed = false;
      self.ended = false;
      self.clear_output_buffer();

      let seek_input_frames = self.stream.output_seek_length();
      self.read_exact_input_frames(seek_input_frames, "seek warmup");
      self
         .stream
         .output_seek_interleaved(&self.input_buffer)
         .unwrap();
      self.input_buffer.clear();

      Ok(())
   }

   fn schedule_seek_to_frame(&mut self, seek_frame: usize) -> Result<(), RodioSeekError> {
      self.ended = false;

      let fade_frames = SEEK_FADE_FRAMES.max(2);

      self.seek_transition = SeekTransition::FadingOut {
         target_frame: seek_frame,
         remaining_frames: fade_frames,
         total_frames: fade_frames,
      };

      Ok(())
   }

   fn start_seek_fade_in(&mut self) {
      let fade_frames = SEEK_FADE_FRAMES.max(2);
      self.seek_transition = SeekTransition::FadingIn {
         remaining_frames: fade_frames,
         total_frames: fade_frames,
      };
   }

   fn is_seek_fading_out(&self) -> bool {
      matches!(self.seek_transition, SeekTransition::FadingOut { .. })
   }

   fn begin_frame(&mut self) -> Result<(), RodioSeekError> {
      if let SeekTransition::Pending { target_frame } = self.seek_transition {
         self.seek_to_frame(target_frame)?;
         self.start_seek_fade_in();
      }

      if self.resume_fade_handle.take_fade_in_request() {
         self.start_seek_fade_in();
      }

      self.current_envelope = match self.seek_transition {
         SeekTransition::FadingOut {
            remaining_frames,
            total_frames,
            ..
         } => envelope_for_step(total_frames - remaining_frames, total_frames, true),
         SeekTransition::FadingIn {
            remaining_frames,
            total_frames,
         } => envelope_for_step(total_frames - remaining_frames, total_frames, false),
         SeekTransition::Idle | SeekTransition::Pending { .. } => 1.0,
      };

      self.seek_transition = match self.seek_transition {
         SeekTransition::FadingOut {
            target_frame,
            remaining_frames,
            total_frames,
         } if remaining_frames > 1 => SeekTransition::FadingOut {
            target_frame,
            remaining_frames: remaining_frames - 1,
            total_frames,
         },
         SeekTransition::FadingOut { target_frame, .. } => SeekTransition::Pending { target_frame },
         SeekTransition::FadingIn {
            remaining_frames,
            total_frames,
         } if remaining_frames > 1 => SeekTransition::FadingIn {
            remaining_frames: remaining_frames - 1,
            total_frames,
         },
         SeekTransition::FadingIn { .. } => SeekTransition::Idle,
         transition => transition,
      };

      Ok(())
   }

   fn flush_output(&mut self) -> bool {
      if self.flushed {
         return false;
      }

      self
         .output_buffer
         .resize(FLUSH_FRAMES * self.channel_count(), 0.0);
      self
         .stream
         .flush_interleaved(&mut self.output_buffer)
         .unwrap();
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
         .process_interleaved(&self.input_buffer, &mut self.output_buffer)
         .unwrap();

      assert_eq!(consumed, input_frames);
      true
   }
}

impl Iterator for StreamingPlaybackSource {
   type Item = f32;

   fn next(&mut self) -> Option<Self::Item> {
      if self.frame_position == 0 && self.begin_frame().is_err() {
         self.ended = true;
         return None;
      }

      while self.output_index >= self.output_buffer.len() {
         if !self.refill_output_buffer() {
            if self.is_seek_fading_out() {
               break;
            }

            self.ended = true;
            return None;
         }

         if self.output_buffer.is_empty() {
            if self.is_seek_fading_out() {
               break;
            }

            self.ended = true;
            return None;
         }
      }

      let sample = if self.output_index < self.output_buffer.len() {
         let value = self.output_buffer.get(self.output_index).copied()?;
         self.output_index += 1;
         value
      } else {
         0.0
      };

      self.frame_position += 1;
      if self.frame_position >= self.channel_count() {
         self.frame_position = 0;
      }

      Some(sample * self.current_envelope)
   }
}

impl Source for StreamingPlaybackSource {
   fn current_span_len(&self) -> Option<usize> {
      if self.ended { Some(0) } else { None }
   }

   fn try_seek(&mut self, position: Duration) -> Result<(), RodioSeekError> {
      let seek_frame = ((position.as_secs_f64() * self.sample_rate_hz.get() as f64).floor()
         as usize)
         .min(self.current_segment_end_frame);

      self.schedule_seek_to_frame(seek_frame)
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

fn envelope_for_step(step: usize, total_steps: usize, fade_out: bool) -> f32 {
   let denominator = total_steps.saturating_sub(1).max(1) as f32;
   let progress = (step.min(total_steps.saturating_sub(1)) as f32) / denominator;
   if fade_out {
      0.5 * (1.0 + (PI * progress).cos())
   } else {
      0.5 * (1.0 - (PI * progress).cos())
   }
}
