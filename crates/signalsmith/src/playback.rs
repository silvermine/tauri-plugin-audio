use crate::port::Stretch;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PlaybackStreamError {
   #[error("playback_rate must be finite and greater than zero, got {0}")]
   InvalidPlaybackRate(f32),

   #[error("channel count must be greater than zero, got {channels}")]
   InvalidChannelCount { channels: usize },

   #[error("sample_rate must be finite and greater than zero, got {sample_rate}")]
   InvalidSampleRate { sample_rate: f32 },

   #[error("block_samples must be at least {minimum_block_samples}, got {block_samples}")]
   InvalidBlockSamples {
      block_samples: usize,
      minimum_block_samples: usize,
   },

   #[error(
      "interval_samples must be at least {minimum_interval_samples}, got {interval_samples}"
   )]
   InvalidIntervalSamples {
      interval_samples: usize,
      minimum_interval_samples: usize,
   },

   #[error(
      "input channel count {actual_channels} must match configured channel count {expected_channels}"
   )]
   InvalidInputChannelCount {
      actual_channels: usize,
      expected_channels: usize,
   },

   #[error("sample count {samples} with channel count {channels} overflows interleaved length")]
   InterleavedLengthOverflow { samples: usize, channels: usize },

   #[error(
      "output channel count {actual_channels} must match configured channel count {expected_channels}"
   )]
   InvalidOutputChannelCount {
      actual_channels: usize,
      expected_channels: usize,
   },

   #[error(
      "input channel {channel_index} length {channel_len} shorter than required {required_len}"
   )]
   InputChannelTooShort {
      channel_index: usize,
      channel_len: usize,
      required_len: usize,
   },

   #[error(
      "output channel {channel_index} length {channel_len} shorter than required {required_len}"
   )]
   OutputChannelTooShort {
      channel_index: usize,
      channel_len: usize,
      required_len: usize,
   },

   #[error("interleaved output length {output_len} must be divisible by channel count {channels}")]
   InvalidInterleavedOutputLength { output_len: usize, channels: usize },

   #[error("interleaved input length {input_len} shorter than required {required_len}")]
   InputTooShort {
      input_len: usize,
      required_len: usize,
   },
}

/// Streaming-first playback helper around the Signalsmith-style `Stretch` port.
///
/// The source-tracking port still expresses speed exactly like upstream
/// Signalsmith Stretch: `input_samples / output_samples` on each `process()`
/// call. This wrapper owns the stretcher and keeps the fractional input timing
/// so callers can drive fixed-size output blocks without manually carrying that
/// state between callbacks.
#[derive(Debug)]
pub struct PlaybackStream {
   stretch: Stretch,
   playback_rate: f32,
   input_remainder: f64,
   channels: usize,
   input_scratch: Vec<Vec<f32>>,
   output_scratch: Vec<Vec<f32>>,
}

/// Backwards-compatible name for the streaming playback wrapper.
pub type PlaybackRateController = PlaybackStream;

const MIN_BLOCK_SAMPLES: usize = 4;
const MIN_INTERVAL_SAMPLES: usize = 1;

impl PlaybackStream {
   pub fn new(channels: usize, sample_rate: f32) -> Result<Self, PlaybackStreamError> {
      Self::with_rate(channels, sample_rate, 1.0)
   }

   pub fn with_rate(
      channels: usize,
      sample_rate: f32,
      playback_rate: f32,
   ) -> Result<Self, PlaybackStreamError> {
      validate_channels(channels)?;
      validate_sample_rate(sample_rate)?;
      validate_playback_rate(playback_rate)?;
      let (block_samples, interval_samples) = preset_default_sizes(sample_rate);
      validate_stretch_configuration(block_samples, interval_samples)?;
      let mut stretch = Stretch::new();
      // Streaming-first default: upstream recommends split computation for
      // stricter real-time situations. The source-tracking `Stretch` API
      // still exposes the exact Signalsmith default for callers who want it.
      stretch.configure(channels, block_samples, interval_samples, true);
      Self::from_stretch(stretch, playback_rate)
   }

   pub fn configured(
      channels: usize,
      block_samples: usize,
      interval_samples: usize,
      split_computation: bool,
      playback_rate: f32,
   ) -> Result<Self, PlaybackStreamError> {
      validate_channels(channels)?;
      validate_playback_rate(playback_rate)?;
      validate_stretch_configuration(block_samples, interval_samples)?;
      let mut stretch = Stretch::new();
      stretch.configure(channels, block_samples, interval_samples, split_computation);
      Self::from_stretch(stretch, playback_rate)
   }

   pub fn from_stretch(stretch: Stretch, playback_rate: f32) -> Result<Self, PlaybackStreamError> {
      validate_playback_rate(playback_rate)?;
      let channels = stretch.channels();
      validate_channels(channels)?;
      Ok(Self {
         stretch,
         playback_rate,
         input_remainder: 0.0,
         channels,
         input_scratch: vec![Vec::new(); channels],
         output_scratch: vec![Vec::new(); channels],
      })
   }

   pub fn stretch(&self) -> &Stretch {
      &self.stretch
   }

   pub fn stretch_mut(&mut self) -> &mut Stretch {
      &mut self.stretch
   }

   pub fn into_stretch(self) -> Stretch {
      self.stretch
   }

   pub fn playback_rate(&self) -> f32 {
      self.playback_rate
   }

   pub fn set_playback_rate(&mut self, playback_rate: f32) -> Result<(), PlaybackStreamError> {
      validate_playback_rate(playback_rate)?;
      self.playback_rate = playback_rate;
      Ok(())
   }

   pub fn reset(&mut self) {
      self.stretch.reset();
      self.reset_timing();
   }

   pub fn reset_timing(&mut self) {
      self.input_remainder = 0.0;
   }

   pub fn input_latency(&self) -> usize {
      self.stretch.input_latency()
   }

   pub fn channels(&self) -> usize {
      self.channels
   }

   pub fn output_latency(&self) -> usize {
      self.stretch.output_latency()
   }

   pub fn input_samples_for_output(&self, output_samples: usize) -> usize {
      Self::input_samples_from(self.input_remainder, output_samples, self.playback_rate).0
   }

   /// Process the next streaming output block.
   ///
   /// Returns the number of input samples consumed. Input channel slices must
   /// contain at least `input_samples_for_output(output_samples)` samples.
   pub fn process(
      &mut self,
      inputs: &[&[f32]],
      outputs: &mut [&mut [f32]],
      output_samples: usize,
   ) -> Result<usize, PlaybackStreamError> {
      let input_samples = self.input_samples_for_output(output_samples);
      validate_inputs(inputs, self.channels, input_samples)?;
      validate_outputs(outputs, self.channels, output_samples)?;
      let input_samples = self.advance_timing(output_samples);
      self
         .stretch
         .process(inputs, input_samples, outputs, output_samples);
      Ok(input_samples)
   }

   /// Process a full output buffer, using the first output channel length.
   pub fn process_buffer(
      &mut self,
      inputs: &[&[f32]],
      outputs: &mut [&mut [f32]],
   ) -> Result<usize, PlaybackStreamError> {
      let output_samples = outputs.first().map_or(0, |output| output.len());
      self.process(inputs, outputs, output_samples)
   }

   /// Process interleaved input/output buffers.
   ///
   /// This mirrors the public layout used by `signalsmith-stretch-rs`, while
   /// keeping the pure port internally channel-major. Scratch buffers are kept
   /// on the stream and reused across calls.
   pub fn process_interleaved(
      &mut self,
      input: &[f32],
      output: &mut [f32],
   ) -> Result<usize, PlaybackStreamError> {
      validate_interleaved_output(output.len(), self.channels)?;
      let output_samples = output.len() / self.channels;
      let input_samples = self.input_samples_for_output(output_samples);
      validate_interleaved_input(input.len(), interleaved_len(input_samples, self.channels)?)?;

      self.prepare_interleaved_input(input, input_samples);
      self.prepare_output_scratch(output_samples);

      let consumed = self.advance_timing(output_samples);
      self.stretch.process_vecs(
         &self.input_scratch,
         consumed,
         &mut self.output_scratch,
         output_samples,
      );

      for frame in 0..output_samples {
         for channel in 0..self.channels {
            output[frame * self.channels + channel] = self.output_scratch[channel][frame];
         }
      }
      Ok(consumed)
   }

   pub fn seek(
      &mut self,
      inputs: &[&[f32]],
      input_samples: usize,
   ) -> Result<(), PlaybackStreamError> {
      validate_inputs(inputs, self.channels, input_samples)?;
      self.stretch.seek(inputs, input_samples, self.playback_rate);
      self.reset_timing();
      Ok(())
   }

   pub fn seek_interleaved(
      &mut self,
      input: &[f32],
      input_samples: usize,
   ) -> Result<(), PlaybackStreamError> {
      validate_interleaved_input(input.len(), interleaved_len(input_samples, self.channels)?)?;
      self.prepare_interleaved_input(input, input_samples);
      self
         .stretch
         .seek_vecs(&self.input_scratch, input_samples, self.playback_rate);
      self.reset_timing();
      Ok(())
   }

   pub fn seek_length(&self) -> usize {
      self.stretch.seek_length()
   }

   pub fn output_seek_length(&self) -> usize {
      self.stretch.output_seek_length(self.playback_rate)
   }

   /// Start playback aligned to the next output sample.
   ///
   /// Returns the number of input samples required by the seek. Input channel
   /// slices must contain at least this many samples.
   pub fn output_seek(&mut self, inputs: &[&[f32]]) -> Result<usize, PlaybackStreamError> {
      let input_length = self.output_seek_length();
      validate_inputs(inputs, self.channels, input_length)?;
      self.stretch.output_seek(inputs, input_length);
      self.reset_timing();
      Ok(input_length)
   }

   pub fn output_seek_interleaved(&mut self, input: &[f32]) -> Result<usize, PlaybackStreamError> {
      let input_length = self.output_seek_length();
      validate_interleaved_input(input.len(), interleaved_len(input_length, self.channels)?)?;
      self.prepare_interleaved_input(input, input_length);
      self
         .stretch
         .output_seek_vecs(&self.input_scratch, input_length);
      self.reset_timing();
      Ok(input_length)
   }

   pub fn flush(
      &mut self,
      outputs: &mut [&mut [f32]],
      output_samples: usize,
   ) -> Result<(), PlaybackStreamError> {
      validate_outputs(outputs, self.channels, output_samples)?;
      self
         .stretch
         .flush(outputs, output_samples, self.playback_rate);
      self.reset_timing();
      Ok(())
   }

   pub fn flush_buffer(&mut self, outputs: &mut [&mut [f32]]) -> Result<(), PlaybackStreamError> {
      let output_samples = outputs.first().map_or(0, |output| output.len());
      self.flush(outputs, output_samples)
   }

   pub fn flush_interleaved(&mut self, output: &mut [f32]) -> Result<(), PlaybackStreamError> {
      validate_interleaved_output(output.len(), self.channels)?;
      let output_samples = output.len() / self.channels;
      self.prepare_output_scratch(output_samples);
      self
         .stretch
         .flush_vecs(&mut self.output_scratch, output_samples, self.playback_rate);
      self.reset_timing();
      for frame in 0..output_samples {
         for channel in 0..self.channels {
            output[frame * self.channels + channel] = self.output_scratch[channel][frame];
         }
      }
      Ok(())
   }

   fn advance_timing(&mut self, output_samples: usize) -> usize {
      let (input_samples, input_remainder) =
         Self::input_samples_from(self.input_remainder, output_samples, self.playback_rate);
      self.input_remainder = input_remainder;
      input_samples
   }

   fn input_samples_from(
      input_remainder: f64,
      output_samples: usize,
      playback_rate: f32,
   ) -> (usize, f64) {
      let exact_input = input_remainder + output_samples as f64 * playback_rate as f64;
      let input_samples = exact_input.round() as usize;
      (input_samples, exact_input - input_samples as f64)
   }

   fn prepare_interleaved_input(&mut self, input: &[f32], input_samples: usize) {
      for channel in &mut self.input_scratch {
         channel.resize(input_samples, 0.0);
      }
      for frame in 0..input_samples {
         for channel in 0..self.channels {
            self.input_scratch[channel][frame] = input[frame * self.channels + channel];
         }
      }
   }

   fn prepare_output_scratch(&mut self, output_samples: usize) {
      for channel in &mut self.output_scratch {
         channel.resize(output_samples, 0.0);
      }
   }

}

fn validate_channels(channels: usize) -> Result<(), PlaybackStreamError> {
   if channels > 0 {
      Ok(())
   } else {
      Err(PlaybackStreamError::InvalidChannelCount { channels })
   }
}

fn validate_sample_rate(sample_rate: f32) -> Result<(), PlaybackStreamError> {
   if sample_rate.is_finite() && sample_rate > 0.0 {
      Ok(())
   } else {
      Err(PlaybackStreamError::InvalidSampleRate { sample_rate })
   }
}

fn validate_playback_rate(playback_rate: f32) -> Result<(), PlaybackStreamError> {
   if playback_rate.is_finite() && playback_rate > 0.0 {
      Ok(())
   } else {
      Err(PlaybackStreamError::InvalidPlaybackRate(playback_rate))
   }
}

fn preset_default_sizes(sample_rate: f32) -> (usize, usize) {
   (
      (sample_rate * 0.12) as usize,
      (sample_rate * 0.03) as usize,
   )
}

fn validate_stretch_configuration(
   block_samples: usize,
   interval_samples: usize,
) -> Result<(), PlaybackStreamError> {
   validate_block_samples(block_samples)?;
   validate_interval_samples(interval_samples)
}

fn validate_block_samples(block_samples: usize) -> Result<(), PlaybackStreamError> {
   if block_samples >= MIN_BLOCK_SAMPLES {
      Ok(())
   } else {
      Err(PlaybackStreamError::InvalidBlockSamples {
         block_samples,
         minimum_block_samples: MIN_BLOCK_SAMPLES,
      })
   }
}

fn validate_interval_samples(interval_samples: usize) -> Result<(), PlaybackStreamError> {
   if interval_samples >= MIN_INTERVAL_SAMPLES {
      Ok(())
   } else {
      Err(PlaybackStreamError::InvalidIntervalSamples {
         interval_samples,
         minimum_interval_samples: MIN_INTERVAL_SAMPLES,
      })
   }
}

fn interleaved_len(samples: usize, channels: usize) -> Result<usize, PlaybackStreamError> {
   validate_channels(channels)?;
   samples
      .checked_mul(channels)
      .ok_or(PlaybackStreamError::InterleavedLengthOverflow { samples, channels })
}

fn validate_inputs(
   inputs: &[&[f32]],
   expected_channels: usize,
   required_len: usize,
) -> Result<(), PlaybackStreamError> {
   if inputs.len() != expected_channels {
      return Err(PlaybackStreamError::InvalidInputChannelCount {
         actual_channels: inputs.len(),
         expected_channels,
      });
   }

   for (channel_index, channel) in inputs.iter().enumerate() {
      if channel.len() < required_len {
         return Err(PlaybackStreamError::InputChannelTooShort {
            channel_index,
            channel_len: channel.len(),
            required_len,
         });
      }
   }

   Ok(())
}

fn validate_outputs(
   outputs: &[&mut [f32]],
   expected_channels: usize,
   required_len: usize,
) -> Result<(), PlaybackStreamError> {
   if outputs.len() != expected_channels {
      return Err(PlaybackStreamError::InvalidOutputChannelCount {
         actual_channels: outputs.len(),
         expected_channels,
      });
   }

   for (channel_index, channel) in outputs.iter().enumerate() {
      if channel.len() < required_len {
         return Err(PlaybackStreamError::OutputChannelTooShort {
            channel_index,
            channel_len: channel.len(),
            required_len,
         });
      }
   }

   Ok(())
}

fn validate_interleaved_output(
   output_len: usize,
   channels: usize,
) -> Result<(), PlaybackStreamError> {
   validate_channels(channels)?;
   if output_len % channels == 0 {
      Ok(())
   } else {
      Err(PlaybackStreamError::InvalidInterleavedOutputLength {
         output_len,
         channels,
      })
   }
}

fn validate_interleaved_input(
   input_len: usize,
   required_len: usize,
) -> Result<(), PlaybackStreamError> {
   if input_len >= required_len {
      Ok(())
   } else {
      Err(PlaybackStreamError::InputTooShort {
         input_len,
         required_len,
      })
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   const PI: f32 = core::f32::consts::PI;

   #[test]
   fn playback_stream_averages_fractional_blocks() {
      let mut stream = PlaybackStream::configured(1, 128, 32, false, 1.1).unwrap();
      let mut total = 0;

      for _ in 0..10 {
         let output_samples = 128;
         let input_samples = stream.input_samples_for_output(output_samples);
         let input = vec![0.0_f32; input_samples];
         let mut output = vec![0.0_f32; output_samples];
         let inputs = [&input[..]];
         let mut outputs = [&mut output[..]];
         total += stream
            .process(&inputs, &mut outputs, output_samples)
            .unwrap();
      }

      assert_eq!(total, 1408);

      stream.set_playback_rate(0.75).unwrap();
      stream.reset_timing();
      assert_eq!(stream.input_samples_for_output(128), 96);
      assert_eq!(stream.playback_rate(), 0.75);
   }

   #[test]
   fn interleaved_processing_matches_channel_major_path() {
      let mut stream = PlaybackStream::configured(2, 256, 64, false, 1.25).unwrap();
      let mut reference = PlaybackStream::configured(2, 256, 64, false, 1.25).unwrap();

      let output_samples = 512;
      let input_samples = stream.input_samples_for_output(output_samples);
      let mut interleaved_input = vec![0.0_f32; input_samples * 2];
      let mut left = vec![0.0_f32; input_samples];
      let mut right = vec![0.0_f32; input_samples];
      for i in 0..input_samples {
         left[i] = (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin() * 0.25;
         right[i] = (2.0 * PI * 660.0 * i as f32 / 48_000.0).sin() * 0.2;
         interleaved_input[2 * i] = left[i];
         interleaved_input[2 * i + 1] = right[i];
      }

      let mut interleaved_output = vec![0.0_f32; output_samples * 2];
      let consumed = stream
         .process_interleaved(&interleaved_input, &mut interleaved_output)
         .unwrap();
      assert_eq!(consumed, input_samples);

      let mut out_left = vec![0.0_f32; output_samples];
      let mut out_right = vec![0.0_f32; output_samples];
      let inputs = [&left[..], &right[..]];
      let mut outputs = [&mut out_left[..], &mut out_right[..]];
      assert_eq!(
         reference
            .process(&inputs, &mut outputs, output_samples)
            .unwrap(),
         input_samples
      );

      for i in 0..output_samples {
         assert!((interleaved_output[2 * i] - out_left[i]).abs() < 1.0e-6);
         assert!((interleaved_output[2 * i + 1] - out_right[i]).abs() < 1.0e-6);
      }
   }

   #[test]
   fn playback_stream_output_seek_tracks_updated_rate() {
      let mut stream = PlaybackStream::configured(1, 256, 64, false, 1.0).unwrap();
      let initial_input_length = stream.output_seek_length();

      stream.set_playback_rate(0.75).unwrap();
      let updated_input_length = stream.output_seek_length();

      assert_eq!(stream.playback_rate(), 0.75);
      assert_eq!(
         initial_input_length,
         stream.input_latency() + stream.output_latency(),
      );
      assert_eq!(
         updated_input_length,
         stream.input_latency() + (0.75 * stream.output_latency() as f32) as usize,
      );
      assert!(updated_input_length < initial_input_length);

      let input: Vec<f32> = (0..updated_input_length)
         .map(|i| (2.0 * PI * 220.0 * i as f32 / 48_000.0).sin() * 0.2)
         .collect();
      let inputs = [&input[..]];

      assert_eq!(stream.output_seek(&inputs).unwrap(), updated_input_length);
   }

   #[test]
   fn process_interleaved_rejects_non_divisible_output_length() {
      let mut stream = PlaybackStream::configured(2, 256, 64, false, 1.0).unwrap();
      let input = vec![0.0_f32; 8];
      let mut output = vec![0.0_f32; 3];

      let result = stream.process_interleaved(&input, &mut output);

      assert_eq!(
         result,
         Err(PlaybackStreamError::InvalidInterleavedOutputLength {
            output_len: 3,
            channels: 2,
         })
      );
   }

   #[test]
   fn process_interleaved_rejects_short_input() {
      let mut stream = PlaybackStream::configured(1, 256, 64, false, 1.25).unwrap();
      let mut output = vec![0.0_f32; 16];
      let input = vec![0.0_f32; 1];

      let result = stream.process_interleaved(&input, &mut output);

      assert_eq!(
         result,
         Err(PlaybackStreamError::InputTooShort {
            input_len: 1,
            required_len: stream.input_samples_for_output(16),
         })
      );
   }

   #[test]
   fn with_rate_rejects_invalid_playback_rate() {
      let result = PlaybackStream::with_rate(1, 48_000.0, f32::NAN);

      assert!(matches!(
         result,
         Err(PlaybackStreamError::InvalidPlaybackRate(value)) if value.is_nan()
      ));
   }

   #[test]
   fn process_rejects_wrong_input_channel_count() {
      let mut stream = PlaybackStream::configured(2, 256, 64, false, 1.0).unwrap();
      let left = [0.0_f32; 16];
      let mut out_left = [0.0_f32; 8];
      let mut out_right = [0.0_f32; 8];
      let inputs = [&left[..]];
      let mut outputs = [&mut out_left[..], &mut out_right[..]];

      let result = stream.process(&inputs, &mut outputs, 8);

      assert_eq!(
         result,
         Err(PlaybackStreamError::InvalidInputChannelCount {
            actual_channels: 1,
            expected_channels: 2,
         })
      );
   }

   #[test]
   fn flush_rejects_short_output_channel() {
      let mut stream = PlaybackStream::configured(2, 256, 64, false, 1.0).unwrap();
      let mut out_left = [0.0_f32; 8];
      let mut out_right = [0.0_f32; 7];
      let mut outputs = [&mut out_left[..], &mut out_right[..]];

      let result = stream.flush(&mut outputs, 8);

      assert_eq!(
         result,
         Err(PlaybackStreamError::OutputChannelTooShort {
            channel_index: 1,
            channel_len: 7,
            required_len: 8,
         })
      );
   }
}
