use crate::port::Stretch;

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

impl PlaybackStream {
    pub fn new(channels: usize, sample_rate: f32) -> Self {
        Self::with_rate(channels, sample_rate, 1.0)
    }

    pub fn with_rate(channels: usize, sample_rate: f32, playback_rate: f32) -> Self {
        validate_playback_rate(playback_rate);
        let mut stretch = Stretch::new();
        // Streaming-first default: upstream recommends split computation for
        // stricter real-time situations. The source-tracking `Stretch` API
        // still exposes the exact Signalsmith default for callers who want it.
        stretch.preset_default(channels, sample_rate, true);
        Self::from_stretch(stretch, playback_rate)
    }

    pub fn configured(
        channels: usize,
        block_samples: usize,
        interval_samples: usize,
        split_computation: bool,
        playback_rate: f32,
    ) -> Self {
        validate_playback_rate(playback_rate);
        let mut stretch = Stretch::new();
        stretch.configure(channels, block_samples, interval_samples, split_computation);
        Self::from_stretch(stretch, playback_rate)
    }

    pub fn from_stretch(stretch: Stretch, playback_rate: f32) -> Self {
        validate_playback_rate(playback_rate);
        let channels = stretch.channels();
        Self {
            stretch,
            playback_rate,
            input_remainder: 0.0,
            channels,
            input_scratch: vec![Vec::new(); channels],
            output_scratch: vec![Vec::new(); channels],
        }
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

    pub fn set_playback_rate(&mut self, playback_rate: f32) {
        validate_playback_rate(playback_rate);
        self.playback_rate = playback_rate;
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
    ) -> usize {
        let input_samples = self.advance_timing(output_samples);
        self.stretch
            .process(inputs, input_samples, outputs, output_samples);
        input_samples
    }

    /// Process a full output buffer, using the first output channel length.
    pub fn process_buffer(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> usize {
        let output_samples = outputs.first().map_or(0, |output| output.len());
        self.process(inputs, outputs, output_samples)
    }

    /// Process interleaved input/output buffers.
    ///
    /// This mirrors the public layout used by `signalsmith-stretch-rs`, while
    /// keeping the pure port internally channel-major. Scratch buffers are kept
    /// on the stream and reused across calls.
    pub fn process_interleaved(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        assert_eq!(
            output.len() % self.channels,
            0,
            "interleaved output length must be divisible by channel count"
        );
        let output_samples = output.len() / self.channels;
        let input_samples = self.input_samples_for_output(output_samples);
        assert!(
            input.len() >= input_samples * self.channels,
            "interleaved input shorter than required input frames"
        );

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
        consumed
    }

    pub fn seek(&mut self, inputs: &[&[f32]], input_samples: usize) {
        self.stretch.seek(inputs, input_samples, self.playback_rate);
        self.reset_timing();
    }

    pub fn seek_interleaved(&mut self, input: &[f32], input_samples: usize) {
        assert!(
            input.len() >= input_samples * self.channels,
            "interleaved input shorter than input_samples"
        );
        self.prepare_interleaved_input(input, input_samples);
        self.stretch
            .seek_vecs(&self.input_scratch, input_samples, self.playback_rate);
        self.reset_timing();
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
    pub fn output_seek(&mut self, inputs: &[&[f32]]) -> usize {
        let input_length = self.output_seek_length();
        self.stretch.output_seek(inputs, input_length);
        self.reset_timing();
        input_length
    }

    pub fn output_seek_interleaved(&mut self, input: &[f32]) -> usize {
        let input_length = self.output_seek_length();
        assert!(
            input.len() >= input_length * self.channels,
            "interleaved input shorter than output seek length"
        );
        self.prepare_interleaved_input(input, input_length);
        self.stretch
            .output_seek_vecs(&self.input_scratch, input_length);
        self.reset_timing();
        input_length
    }

    pub fn flush(&mut self, outputs: &mut [&mut [f32]], output_samples: usize) {
        self.stretch
            .flush(outputs, output_samples, self.playback_rate);
        self.reset_timing();
    }

    pub fn flush_buffer(&mut self, outputs: &mut [&mut [f32]]) {
        let output_samples = outputs.first().map_or(0, |output| output.len());
        self.flush(outputs, output_samples);
    }

    pub fn flush_interleaved(&mut self, output: &mut [f32]) {
        assert_eq!(
            output.len() % self.channels,
            0,
            "interleaved output length must be divisible by channel count"
        );
        let output_samples = output.len() / self.channels;
        self.prepare_output_scratch(output_samples);
        self.stretch
            .flush_vecs(&mut self.output_scratch, output_samples, self.playback_rate);
        self.reset_timing();
        for frame in 0..output_samples {
            for channel in 0..self.channels {
                output[frame * self.channels + channel] = self.output_scratch[channel][frame];
            }
        }
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

fn validate_playback_rate(playback_rate: f32) {
    assert!(
        playback_rate.is_finite() && playback_rate >= 0.0,
        "playback_rate must be finite and non-negative"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f32 = core::f32::consts::PI;

    #[test]
    fn playback_stream_averages_fractional_blocks() {
        let mut stream = PlaybackStream::configured(1, 128, 32, false, 1.1);
        let mut total = 0;

        for _ in 0..10 {
            let output_samples = 128;
            let input_samples = stream.input_samples_for_output(output_samples);
            let input = vec![0.0_f32; input_samples];
            let mut output = vec![0.0_f32; output_samples];
            let inputs = [&input[..]];
            let mut outputs = [&mut output[..]];
            total += stream.process(&inputs, &mut outputs, output_samples);
        }

        assert_eq!(total, 1408);

        stream.set_playback_rate(0.75);
        stream.reset_timing();
        assert_eq!(stream.input_samples_for_output(128), 96);
        assert_eq!(stream.playback_rate(), 0.75);
    }

    #[test]
    fn playback_stream_processes_fixed_output_blocks() {
        let mut stream = PlaybackStream::configured(1, 256, 64, false, 1.25);

        let output_samples = 512;
        let input_samples = stream.input_samples_for_output(output_samples);
        assert_eq!(input_samples, 640);

        let input: Vec<f32> = (0..input_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin() * 0.25)
            .collect();
        let mut output = vec![0.0_f32; output_samples];
        let inputs = [&input[..]];
        let mut outputs = [&mut output[..]];

        let consumed = stream.process(&inputs, &mut outputs, output_samples);
        assert_eq!(consumed, input_samples);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn interleaved_processing_matches_channel_major_path() {
        let mut stream = PlaybackStream::configured(2, 256, 64, false, 1.25);
        let mut reference = PlaybackStream::configured(2, 256, 64, false, 1.25);

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
        let consumed = stream.process_interleaved(&interleaved_input, &mut interleaved_output);
        assert_eq!(consumed, input_samples);

        let mut out_left = vec![0.0_f32; output_samples];
        let mut out_right = vec![0.0_f32; output_samples];
        let inputs = [&left[..], &right[..]];
        let mut outputs = [&mut out_left[..], &mut out_right[..]];
        assert_eq!(
            reference.process(&inputs, &mut outputs, output_samples),
            input_samples
        );

        for i in 0..output_samples {
            assert!((interleaved_output[2 * i] - out_left[i]).abs() < 1.0e-6);
            assert!((interleaved_output[2 * i + 1] - out_right[i]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn playback_stream_output_seek_uses_current_rate() {
        let mut stream = PlaybackStream::configured(1, 256, 64, false, 0.75);
        let input_length = stream.output_seek_length();
        assert_eq!(input_length, 224);

        let input: Vec<f32> = (0..input_length)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 48_000.0).sin() * 0.2)
            .collect();
        let inputs = [&input[..]];

        assert_eq!(stream.output_seek(&inputs), input_length);
    }
}
