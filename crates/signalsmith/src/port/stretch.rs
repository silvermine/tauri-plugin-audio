use super::complex::Complex;
use super::stft::{DynamicStft, InputState, OutputState, WindowShape};

const PI: f32 = core::f32::consts::PI;
const NOISE_FLOOR: f32 = 1.0e-15;
const MAX_CLEAN_STRETCH: f32 = 2.0;
const SPLIT_MAIN_PREDICTION: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
struct BlockProcess {
    samples_since_last: usize,
    steps: usize,
    step: usize,
    new_spectrum: bool,
    reanalyse_prev: bool,
    time_factor: f32,
}

#[derive(Clone, Copy, Debug)]
struct Band {
    input: Complex,
    prev_input: Complex,
    output: Complex,
    input_energy: f32,
}

impl Default for Band {
    fn default() -> Self {
        Self {
            input: Complex::ZERO,
            prev_input: Complex::ZERO,
            output: Complex::ZERO,
            input_energy: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PitchMapPoint {
    input_bin: f32,
    freq_grad: f32,
}

#[derive(Clone, Copy, Debug)]
struct Prediction {
    energy: f32,
    input: Complex,
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            energy: 0.0,
            input: Complex::ZERO,
        }
    }
}

impl Prediction {
    fn make_output(self, mut phase: Complex) -> Complex {
        let mut phase_norm = phase.norm();
        if phase_norm <= NOISE_FLOOR {
            phase = self.input;
            phase_norm = self.input.norm() + NOISE_FLOOR;
        }
        phase * (self.energy / phase_norm).sqrt()
    }
}

// Signalsmith's C++ API accepts any buffer object with `buffer[channel][index]`
// access. These internal traits keep that shape in Rust, so the pure port can
// process borrowed channel slices and the playback wrapper's owned scratch
// buffers without allocating adapter `Vec`s in the audio callback.
trait InputChannels {
    fn channel_count(&self) -> usize;
    fn channel_len(&self, channel: usize) -> usize;
    fn sample(&self, channel: usize, index: usize) -> f32;
}

trait OutputChannels {
    fn channel_count(&self) -> usize;
    fn channel_len(&self, channel: usize) -> usize;
    fn sample(&self, channel: usize, index: usize) -> f32;
    fn set_sample(&mut self, channel: usize, index: usize, value: f32);

    fn fill_channel(&mut self, channel: usize, length: usize, value: f32) {
        for i in 0..length {
            self.set_sample(channel, i, value);
        }
    }
}

impl InputChannels for [&[f32]] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel_len(&self, channel: usize) -> usize {
        self[channel].len()
    }

    fn sample(&self, channel: usize, index: usize) -> f32 {
        self[channel][index]
    }
}

impl InputChannels for [Vec<f32>] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel_len(&self, channel: usize) -> usize {
        self[channel].len()
    }

    fn sample(&self, channel: usize, index: usize) -> f32 {
        self[channel][index]
    }
}

impl OutputChannels for [&mut [f32]] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel_len(&self, channel: usize) -> usize {
        self[channel].len()
    }

    fn sample(&self, channel: usize, index: usize) -> f32 {
        self[channel][index]
    }

    fn set_sample(&mut self, channel: usize, index: usize, value: f32) {
        self[channel][index] = value;
    }

    fn fill_channel(&mut self, channel: usize, length: usize, value: f32) {
        self[channel][..length].fill(value);
    }
}

impl OutputChannels for [Vec<f32>] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel_len(&self, channel: usize) -> usize {
        self[channel].len()
    }

    fn sample(&self, channel: usize, index: usize) -> f32 {
        self[channel][index]
    }

    fn set_sample(&mut self, channel: usize, index: usize, value: f32) {
        self[channel][index] = value;
    }

    fn fill_channel(&mut self, channel: usize, length: usize, value: f32) {
        self[channel][..length].fill(value);
    }
}

struct OffsetInputs<'a, I: InputChannels + ?Sized> {
    inputs: &'a I,
    offset: usize,
}

impl<I: InputChannels + ?Sized> InputChannels for OffsetInputs<'_, I> {
    fn channel_count(&self) -> usize {
        self.inputs.channel_count()
    }

    fn channel_len(&self, channel: usize) -> usize {
        self.inputs.channel_len(channel).saturating_sub(self.offset)
    }

    fn sample(&self, channel: usize, index: usize) -> f32 {
        self.inputs.sample(channel, self.offset + index)
    }
}

struct ZeroInputs {
    channels: usize,
    length: usize,
}

impl InputChannels for ZeroInputs {
    fn channel_count(&self) -> usize {
        self.channels
    }

    fn channel_len(&self, _channel: usize) -> usize {
        self.length
    }

    fn sample(&self, _channel: usize, _index: usize) -> f32 {
        0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Rng32 {
    state: u64,
}

impl Default for Rng32 {
    fn default() -> Self {
        Self {
            state: 0x9e37_79b9_7f4a_7c15,
        }
    }
}

impl Rng32 {
    fn with_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 40) as u32;
        bits as f32 / (1_u32 << 24) as f32
    }

    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}

/// Streaming time-stretcher ported from Signalsmith Stretch.
///
/// The processing model mirrors `SignalsmithStretch<float>`: playback rate is
/// represented by the ratio between input frames supplied and output frames
/// requested on each call. Pitch shifting, custom frequency maps, and formant
/// compensation are intentionally not exposed in this partial port.
#[derive(Debug)]
pub struct Stretch {
    split_computation: bool,
    block_process: BlockProcess,
    stft: DynamicStft,
    stashed_input: InputState,
    stashed_output: OutputState,
    tmp_process_buffer: Vec<f32>,
    tmp_pre_roll_buffer: Vec<f32>,
    channel_bands: Vec<Band>,
    output_map: Vec<PitchMapPoint>,
    channel_predictions: Vec<Prediction>,
    channels: usize,
    bands: usize,
    prev_input_offset: isize,
    did_seek: bool,
    seek_time_factor: f32,
    silence_counter: usize,
    silence_first: bool,
    process_spectrum_steps: usize,
    rng: Rng32,
}

impl Default for Stretch {
    fn default() -> Self {
        Self {
            split_computation: false,
            block_process: BlockProcess {
                samples_since_last: usize::MAX,
                ..BlockProcess::default()
            },
            stft: DynamicStft::default(),
            stashed_input: InputState::default(),
            stashed_output: OutputState::default(),
            tmp_process_buffer: Vec::new(),
            tmp_pre_roll_buffer: Vec::new(),
            channel_bands: Vec::new(),
            output_map: Vec::new(),
            channel_predictions: Vec::new(),
            channels: 0,
            bands: 0,
            prev_input_offset: -1,
            did_seek: false,
            seek_time_factor: 1.0,
            silence_counter: 0,
            silence_first: true,
            process_spectrum_steps: 0,
            rng: Rng32::default(),
        }
    }
}

impl Stretch {
    pub const VERSION: [usize; 3] = [1, 3, 2];

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Rng32::with_seed(seed),
            ..Self::default()
        }
    }

    pub fn preset_default(&mut self, channels: usize, sample_rate: f32, split_computation: bool) {
        self.configure(
            channels,
            (sample_rate * 0.12) as usize,
            (sample_rate * 0.03) as usize,
            split_computation,
        );
    }

    pub fn preset_cheaper(&mut self, channels: usize, sample_rate: f32, split_computation: bool) {
        self.configure(
            channels,
            (sample_rate * 0.1) as usize,
            (sample_rate * 0.04) as usize,
            split_computation,
        );
    }

    pub fn configure(
        &mut self,
        channels: usize,
        block_samples: usize,
        interval_samples: usize,
        split_computation: bool,
    ) {
        self.split_computation = split_computation;
        self.channels = channels;
        self.stft
            .configure(channels, channels, block_samples, interval_samples + 1);
        self.stft
            .set_interval(interval_samples, WindowShape::Kaiser);
        self.stft.reset(0.1);
        self.stashed_input = self.stft.input.clone();
        self.stashed_output = self.stft.output.clone();

        self.bands = self.stft.bands();
        self.channel_bands
            .resize(self.bands * self.channels, Band::default());
        self.output_map.resize(self.bands, PitchMapPoint::default());
        self.channel_predictions
            .resize(self.bands * self.channels, Prediction::default());

        self.block_process = BlockProcess {
            samples_since_last: usize::MAX,
            ..BlockProcess::default()
        };
        self.tmp_process_buffer
            .resize(block_samples + interval_samples, 0.0);
        self.tmp_pre_roll_buffer
            .resize(self.output_latency() * channels, 0.0);
        self.prev_input_offset = -1;
        self.did_seek = false;
        self.silence_counter = 0;
        self.silence_first = true;
    }

    pub fn block_samples(&self) -> usize {
        self.stft.block_samples()
    }

    pub fn interval_samples(&self) -> usize {
        self.stft.default_interval()
    }

    pub fn input_latency(&self) -> usize {
        self.stft.analysis_latency()
    }

    pub fn output_latency(&self) -> usize {
        self.stft.synthesis_latency()
            + if self.split_computation {
                self.stft.default_interval()
            } else {
                0
            }
    }

    pub fn split_computation(&self) -> bool {
        self.split_computation
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn reset(&mut self) {
        self.stft.reset(0.1);
        self.stashed_input = self.stft.input.clone();
        self.stashed_output = self.stft.output.clone();
        self.prev_input_offset = -1;
        self.channel_bands.fill(Band::default());
        self.silence_counter = 0;
        self.silence_first = true;
        self.did_seek = false;
        self.block_process = BlockProcess {
            samples_since_last: usize::MAX,
            ..BlockProcess::default()
        };
    }

    pub fn seek(&mut self, inputs: &[&[f32]], input_samples: usize, playback_rate: f32) {
        self.seek_impl(inputs, input_samples, playback_rate);
    }

    pub(crate) fn seek_vecs(
        &mut self,
        inputs: &[Vec<f32>],
        input_samples: usize,
        playback_rate: f32,
    ) {
        self.seek_impl(inputs, input_samples, playback_rate);
    }

    fn seek_impl<I: InputChannels + ?Sized>(
        &mut self,
        inputs: &I,
        input_samples: usize,
        playback_rate: f32,
    ) {
        self.validate_inputs(inputs, input_samples);
        let history = self.stft.block_samples() + self.stft.default_interval();
        self.tmp_process_buffer.clear();
        self.tmp_process_buffer.resize(history, 0.0);

        let start_index = input_samples.saturating_sub(history);
        let pad_start = history + start_index - input_samples;
        let mut total_energy = 0.0;
        for c in 0..self.channels {
            for i in start_index..input_samples {
                let s = inputs.sample(c, i);
                total_energy += s * s;
                self.tmp_process_buffer[i - start_index + pad_start] = s;
            }
            self.stft
                .write_input(c, self.tmp_process_buffer.len(), &self.tmp_process_buffer);
        }
        self.stft.move_input(self.tmp_process_buffer.len());
        if total_energy >= NOISE_FLOOR {
            self.silence_counter = 0;
            self.silence_first = true;
        }
        self.did_seek = true;
        self.seek_time_factor = if playback_rate * self.stft.default_interval() as f32 > 1.0 {
            1.0 / playback_rate
        } else {
            self.stft.default_interval() as f32
        };
    }

    pub fn seek_length(&self) -> usize {
        self.stft.block_samples() + self.stft.default_interval()
    }

    pub fn output_seek_length(&self, playback_rate: f32) -> usize {
        self.input_latency() + (playback_rate * self.output_latency() as f32) as usize
    }

    pub fn output_seek(&mut self, inputs: &[&[f32]], input_length: usize) {
        self.output_seek_impl(inputs, input_length);
    }

    pub(crate) fn output_seek_vecs(&mut self, inputs: &[Vec<f32>], input_length: usize) {
        self.output_seek_impl(inputs, input_length);
    }

    fn output_seek_impl<I: InputChannels + ?Sized>(&mut self, inputs: &I, input_length: usize) {
        self.validate_inputs(inputs, input_length);
        self.reset();
        let surplus_input = input_length.saturating_sub(self.input_latency());
        let playback_rate = surplus_input as f32 / self.output_latency() as f32;
        let seek_samples = input_length - surplus_input;
        self.seek_impl(inputs, seek_samples, playback_rate);

        let output_latency = self.output_latency();
        self.tmp_pre_roll_buffer
            .resize(output_latency * self.channels, 0.0);
        let mut pre_roll = vec![vec![0.0; output_latency]; self.channels];
        let offset_inputs = OffsetInputs {
            inputs,
            offset: seek_samples,
        };
        self.process_impl(
            &offset_inputs,
            surplus_input,
            pre_roll.as_mut_slice(),
            output_latency,
        );

        for channel in &mut pre_roll {
            for v in channel.iter_mut() {
                *v = -*v;
            }
            channel.reverse();
        }
        for (c, channel) in pre_roll.iter().enumerate().take(self.channels) {
            self.stft.add_output(c, output_latency, channel);
        }
    }

    pub fn process(
        &mut self,
        inputs: &[&[f32]],
        input_samples: usize,
        outputs: &mut [&mut [f32]],
        output_samples: usize,
    ) {
        self.process_impl(inputs, input_samples, outputs, output_samples);
    }

    pub(crate) fn process_vecs(
        &mut self,
        inputs: &[Vec<f32>],
        input_samples: usize,
        outputs: &mut [Vec<f32>],
        output_samples: usize,
    ) {
        self.process_impl(inputs, input_samples, outputs, output_samples);
    }

    fn process_impl<I: InputChannels + ?Sized, O: OutputChannels + ?Sized>(
        &mut self,
        inputs: &I,
        input_samples: usize,
        outputs: &mut O,
        output_samples: usize,
    ) {
        self.validate_inputs(inputs, input_samples);
        self.validate_outputs(outputs, output_samples);
        if output_samples == 0 {
            let mut prev_copied_input = 0;
            self.copy_input(inputs, input_samples, &mut prev_copied_input, input_samples);
            self.prev_input_offset -= input_samples as isize;
            return;
        }

        let mut prev_copied_input = 0;
        let mut total_energy = 0.0;
        for c in 0..self.channels {
            for i in 0..input_samples {
                let s = inputs.sample(c, i);
                total_energy += s * s;
            }
        }

        if total_energy < NOISE_FLOOR {
            if self.silence_counter >= 2 * self.stft.block_samples() {
                if self.silence_first {
                    self.silence_first = false;
                    self.block_process = BlockProcess {
                        samples_since_last: usize::MAX,
                        ..BlockProcess::default()
                    };
                    for band in &mut self.channel_bands {
                        band.input = Complex::ZERO;
                        band.prev_input = Complex::ZERO;
                        band.output = Complex::ZERO;
                        band.input_energy = 0.0;
                    }
                }

                if input_samples > 0 {
                    for output_index in 0..output_samples {
                        let input_index = output_index % input_samples;
                        for c in 0..self.channels {
                            outputs.set_sample(c, output_index, inputs.sample(c, input_index));
                        }
                    }
                } else {
                    for c in 0..self.channels {
                        outputs.fill_channel(c, output_samples, 0.0);
                    }
                }

                self.copy_input(inputs, input_samples, &mut prev_copied_input, input_samples);
                return;
            }
            self.silence_counter += input_samples;
        } else {
            self.silence_counter = 0;
            self.silence_first = true;
        }

        for output_index in 0..output_samples {
            let new_block = self.block_process.samples_since_last >= self.stft.default_interval();
            if new_block {
                self.block_process.step = 0;
                self.block_process.steps = 0;
                self.block_process.samples_since_last = 0;

                let input_offset = ((output_index as f32 * input_samples as f32
                    / output_samples as f32)
                    .round()) as isize;
                let input_interval = input_offset - self.prev_input_offset;
                self.prev_input_offset = input_offset;

                self.copy_input(
                    inputs,
                    input_samples,
                    &mut prev_copied_input,
                    input_offset as usize,
                );
                self.stashed_input = self.stft.input.clone();
                if self.split_computation {
                    self.stashed_output = self.stft.output.clone();
                    self.stft.move_output(self.stft.default_interval());
                }

                self.block_process.new_spectrum = self.did_seek || input_interval > 0;
                if self.block_process.new_spectrum {
                    self.block_process.reanalyse_prev = self.did_seek
                        || (input_interval - self.stft.default_interval() as isize).abs() > 1;
                    if self.block_process.reanalyse_prev {
                        self.block_process.steps += self.stft.analyse_steps() + 1;
                    }
                    self.block_process.steps += self.stft.analyse_steps() + 1;
                }

                self.block_process.time_factor = if self.did_seek {
                    self.seek_time_factor
                } else {
                    self.stft.default_interval() as f32 / input_interval.max(1) as f32
                };
                self.did_seek = false;

                self.update_process_spectrum_steps();
                self.block_process.steps += self.process_spectrum_steps;
                self.block_process.steps += self.stft.synthesise_steps() + 1;
            }

            let mut process_to_step = if new_block {
                self.block_process.steps
            } else {
                0
            };
            if self.split_computation {
                let process_ratio = (self.block_process.samples_since_last + 1) as f32
                    / self.stft.default_interval() as f32;
                process_to_step = self
                    .block_process
                    .steps
                    .min(((self.block_process.steps as f32 + 0.999) * process_ratio) as usize);
            }

            while self.block_process.step < process_to_step {
                let mut step = self.block_process.step;
                self.block_process.step += 1;

                if self.block_process.new_spectrum {
                    if self.block_process.reanalyse_prev {
                        if step < self.stft.analyse_steps() {
                            core::mem::swap(&mut self.stashed_input, &mut self.stft.input);
                            self.stft.analyse_step(step, self.stft.default_interval());
                            core::mem::swap(&mut self.stashed_input, &mut self.stft.input);
                            continue;
                        }
                        step -= self.stft.analyse_steps();
                        if step < 1 {
                            for c in 0..self.channels {
                                let start = c * self.bands;
                                for b in 0..self.bands {
                                    let value = self.stft.spectrum(c)[b];
                                    self.channel_bands[start + b].prev_input = value;
                                }
                            }
                            continue;
                        }
                        step -= 1;
                    }

                    if step < self.stft.analyse_steps() {
                        core::mem::swap(&mut self.stashed_input, &mut self.stft.input);
                        self.stft.analyse_step(step, 0);
                        core::mem::swap(&mut self.stashed_input, &mut self.stft.input);
                        continue;
                    }
                    step -= self.stft.analyse_steps();
                    if step < 1 {
                        for c in 0..self.channels {
                            // Copying sample-by-sample avoids heap allocation in the
                            // audio callback while preserving Signalsmith's exact
                            // spectrum handoff point.
                            let start = c * self.bands;
                            for b in 0..self.bands {
                                let value = self.stft.spectrum(c)[b];
                                self.channel_bands[start + b].input = value;
                            }
                        }
                        continue;
                    }
                    step -= 1;
                }

                if step < self.process_spectrum_steps {
                    self.process_spectrum(step);
                    continue;
                }
                step -= self.process_spectrum_steps;

                if step < 1 {
                    for c in 0..self.channels {
                        let start = c * self.bands;
                        for b in 0..self.bands {
                            let output = self.channel_bands[start + b].output;
                            self.stft.spectrum_mut(c)[b] = output;
                        }
                    }
                    continue;
                }
                step -= 1;

                if step < self.stft.synthesise_steps() {
                    self.stft.synthesise_step(step);
                    continue;
                }
            }

            self.block_process.samples_since_last += 1;
            if self.split_computation {
                core::mem::swap(&mut self.stashed_output, &mut self.stft.output);
            }
            for c in 0..self.channels {
                let mut value = [0.0];
                self.stft.read_output(c, 0, &mut value);
                outputs.set_sample(c, output_index, value[0]);
            }
            self.stft.move_output(1);
            if self.split_computation {
                core::mem::swap(&mut self.stashed_output, &mut self.stft.output);
            }
        }

        self.copy_input(inputs, input_samples, &mut prev_copied_input, input_samples);
        self.prev_input_offset -= input_samples as isize;
    }

    pub fn flush(&mut self, outputs: &mut [&mut [f32]], output_samples: usize, playback_rate: f32) {
        self.flush_impl(outputs, output_samples, playback_rate);
    }

    pub(crate) fn flush_vecs(
        &mut self,
        outputs: &mut [Vec<f32>],
        output_samples: usize,
        playback_rate: f32,
    ) {
        self.flush_impl(outputs, output_samples, playback_rate);
    }

    fn flush_impl<O: OutputChannels + ?Sized>(
        &mut self,
        outputs: &mut O,
        output_samples: usize,
        playback_rate: f32,
    ) {
        self.validate_outputs(outputs, output_samples);
        let output_block = output_samples.saturating_sub(self.stft.default_interval());
        if output_block > 0 {
            let zero_len = (output_block as f32 * playback_rate) as usize;
            let zeros = ZeroInputs {
                channels: self.channels,
                length: zero_len,
            };
            self.process_impl(&zeros, zero_len, outputs, output_block);
        }

        let tail_samples = output_samples - output_block;
        self.tmp_process_buffer.resize(tail_samples, 0.0);
        self.stft.finish_output(1.0, 0);
        for c in 0..self.channels {
            self.stft
                .read_output(c, 0, &mut self.tmp_process_buffer[..tail_samples]);
            for i in 0..tail_samples {
                outputs.set_sample(c, output_block + i, self.tmp_process_buffer[i]);
            }
            self.stft.read_output(
                c,
                tail_samples,
                &mut self.tmp_process_buffer[..tail_samples],
            );
            for i in 0..tail_samples {
                let index = output_block + tail_samples - 1 - i;
                let value = outputs.sample(c, index) - self.tmp_process_buffer[i];
                outputs.set_sample(c, index, value);
            }
        }
        self.stft.reset(0.1);
        for c in 0..self.channels {
            let start = c * self.bands;
            for b in 0..self.bands {
                self.channel_bands[start + b].prev_input = Complex::ZERO;
                self.channel_bands[start + b].output = Complex::ZERO;
            }
        }
    }

    /// Process a finite buffer to an exact output length.
    ///
    /// This mirrors upstream Signalsmith Stretch's `exact()` convenience method:
    /// it performs an output-aligned seek, processes the middle, then flushes
    /// the tail. It returns `false` and clears the output if the input is too
    /// short for the required pre-roll.
    pub fn exact(
        &mut self,
        inputs: &[&[f32]],
        input_samples: usize,
        outputs: &mut [&mut [f32]],
        output_samples: usize,
    ) -> bool {
        self.validate_inputs(inputs, input_samples);
        self.validate_outputs(outputs, output_samples);

        if output_samples == 0 {
            return input_samples == 0;
        }

        let playback_rate = input_samples as f32 / output_samples as f32;
        let seek_length = self.output_seek_length(playback_rate);
        if input_samples < seek_length {
            for output in outputs.iter_mut().take(self.channels) {
                output[..output_samples].fill(0.0);
            }
            return false;
        }

        self.output_seek(inputs, seek_length);

        let output_index =
            (output_samples as f32 - seek_length as f32 / playback_rate).max(0.0) as usize;
        let output_index = output_index.min(output_samples);
        let offset_inputs: Vec<&[f32]> = inputs
            .iter()
            .map(|channel| &channel[seek_length..])
            .collect();

        self.process(
            &offset_inputs,
            input_samples - seek_length,
            outputs,
            output_index,
        );

        let mut offset_outputs: Vec<&mut [f32]> = outputs
            .iter_mut()
            .map(|channel| {
                let (_, tail) = (*channel).split_at_mut(output_index);
                tail
            })
            .collect();
        self.flush(
            &mut offset_outputs,
            output_samples - output_index,
            playback_rate,
        );
        true
    }

    fn validate_inputs<I: InputChannels + ?Sized>(&self, inputs: &I, input_samples: usize) {
        assert_eq!(
            inputs.channel_count(),
            self.channels,
            "input channel count must match configured Stretch channels"
        );
        for c in 0..self.channels {
            assert!(
                inputs.channel_len(c) >= input_samples,
                "input channel shorter than input_samples"
            );
        }
    }

    fn validate_outputs<O: OutputChannels + ?Sized>(&self, outputs: &O, output_samples: usize) {
        assert_eq!(
            outputs.channel_count(),
            self.channels,
            "output channel count must match configured Stretch channels"
        );
        for c in 0..self.channels {
            assert!(
                outputs.channel_len(c) >= output_samples,
                "output channel shorter than output_samples"
            );
        }
    }

    fn copy_input<I: InputChannels + ?Sized>(
        &mut self,
        inputs: &I,
        input_samples: usize,
        prev_copied_input: &mut usize,
        to_index: usize,
    ) {
        debug_assert!(to_index <= input_samples);
        let available = to_index.saturating_sub(*prev_copied_input);
        let length = (self.stft.block_samples() + self.stft.default_interval()).min(available);
        self.tmp_process_buffer.resize(length, 0.0);
        let offset = to_index - length;
        for c in 0..self.channels {
            for i in 0..length {
                self.tmp_process_buffer[i] = inputs.sample(c, i + offset);
            }
            self.stft
                .write_input(c, length, &self.tmp_process_buffer[..length]);
        }
        self.stft.move_input(length);
        *prev_copied_input = to_index;
    }

    fn update_process_spectrum_steps(&mut self) {
        self.process_spectrum_steps = 0;
        if self.block_process.new_spectrum {
            self.process_spectrum_steps += self.channels;
        }
        self.process_spectrum_steps += 1; // output map/input energy update
        self.process_spectrum_steps += self.channels; // preliminary phase-vocoder prediction
        self.process_spectrum_steps += SPLIT_MAIN_PREDICTION;
        if self.block_process.new_spectrum {
            self.process_spectrum_steps += 1; // input -> previous input
        }
    }

    fn process_spectrum(&mut self, mut step: usize) {
        let mut time_factor = self.block_process.time_factor;
        let smoothing_bins = self.stft.fft_samples() as f32 / self.stft.default_interval() as f32;
        let long_vertical_step = smoothing_bins.round() as usize;
        time_factor = time_factor.max(1.0 / MAX_CLEAN_STRETCH);
        let random_time_factor = time_factor > MAX_CLEAN_STRETCH;

        if self.block_process.new_spectrum {
            if step < self.channels {
                let channel = step;
                let interval = self.stft.default_interval() as f32;
                let mut rot = Complex::polar(1.0, self.band_to_freq(0.0) * interval * 2.0 * PI);
                let freq_step = self.band_to_freq(1.0) - self.band_to_freq(0.0);
                let rot_step = Complex::polar(1.0, freq_step * interval * 2.0 * PI);
                let start = channel * self.bands;
                for band in &mut self.channel_bands[start..start + self.bands] {
                    band.output *= rot;
                    band.prev_input *= rot;
                    rot *= rot_step;
                }
                return;
            }
            step -= self.channels;
        }

        if step == 0 {
            // Time-stretch-only port: Signalsmith's mapped-frequency path is
            // only needed for pitch/formant features. With a 1:1 frequency map
            // the output map is the identity and energy is per-channel input
            // magnitude, matching the upstream `else` branch.
            for c in 0..self.channels {
                let start = c * self.bands;
                for b in 0..self.bands {
                    let input = self.channel_bands[start + b].input;
                    self.channel_bands[start + b].input_energy = input.norm();
                }
            }
            for b in 0..self.bands {
                self.output_map[b] = PitchMapPoint {
                    input_bin: b as f32,
                    freq_grad: 1.0,
                };
            }
            return;
        }
        step -= 1;

        if step < self.channels {
            let c = step;
            let start = c * self.bands;
            for b in 0..self.bands {
                let map_point = self.output_map[b];
                let low_index = map_point.input_bin.floor() as isize;
                let frac_index = map_point.input_bin - low_index as f32;

                let pred_index = start + b;
                let prev_energy = self.channel_predictions[pred_index].energy;
                let mut energy = self.get_fractional_energy(c, low_index, frac_index);
                energy *= 0.0_f32.max(map_point.freq_grad);
                let input = self.get_fractional_input(c, low_index, frac_index);
                self.channel_predictions[pred_index] = Prediction { energy, input };

                let prev_input = self.get_fractional_prev_input(c, low_index, frac_index);
                let freq_twist = input.mul_conj_rhs(prev_input);
                let phase = self.channel_bands[start + b].output * freq_twist;
                self.channel_bands[start + b].output =
                    phase / (prev_energy.max(energy) + NOISE_FLOOR);
            }
            return;
        }
        step -= self.channels;

        if step < SPLIT_MAIN_PREDICTION {
            let chunk = step;
            let start_b = self.bands * chunk / SPLIT_MAIN_PREDICTION;
            let end_b = self.bands * (chunk + 1) / SPLIT_MAIN_PREDICTION;
            for b in start_b..end_b {
                let mut max_channel = 0;
                let mut max_energy = self.channel_predictions[b].energy;
                for c in 1..self.channels {
                    let energy = self.channel_predictions[c * self.bands + b].energy;
                    if energy > max_energy {
                        max_channel = c;
                        max_energy = energy;
                    }
                }

                let pred_index = max_channel * self.bands + b;
                let prediction = self.channel_predictions[pred_index];
                let map_point = self.output_map[b];
                let mut phase = Complex::ZERO;

                if b > 0 {
                    let bin_time_factor = if random_time_factor {
                        self.rng.uniform(4.0 - time_factor, time_factor)
                    } else {
                        time_factor
                    };
                    let down_input = self.get_fractional_input_float(
                        max_channel,
                        map_point.input_bin - bin_time_factor,
                    );
                    let short_vertical_twist = prediction.input.mul_conj_rhs(down_input);
                    let down_output = self.channel_bands[max_channel * self.bands + b - 1].output;
                    phase += down_output * short_vertical_twist;

                    if b >= long_vertical_step {
                        let long_down_input = self.get_fractional_input_float(
                            max_channel,
                            map_point.input_bin - long_vertical_step as f32 * bin_time_factor,
                        );
                        let long_vertical_twist = prediction.input.mul_conj_rhs(long_down_input);
                        let long_down_output = self.channel_bands
                            [max_channel * self.bands + b - long_vertical_step]
                            .output;
                        phase += long_down_output * long_vertical_twist;
                    }
                }

                if b + 1 < self.bands {
                    let up_prediction = self.channel_predictions[max_channel * self.bands + b + 1];
                    let up_map_point = self.output_map[b + 1];
                    let bin_time_factor = if random_time_factor {
                        self.rng.uniform(4.0 - time_factor, time_factor)
                    } else {
                        time_factor
                    };
                    let down_input = self.get_fractional_input_float(
                        max_channel,
                        up_map_point.input_bin - bin_time_factor,
                    );
                    let short_vertical_twist = up_prediction.input.mul_conj_rhs(down_input);
                    let up_output = self.channel_bands[max_channel * self.bands + b + 1].output;
                    phase += up_output.mul_conj_rhs(short_vertical_twist);

                    if b + long_vertical_step < self.bands {
                        let long_up_prediction = self.channel_predictions
                            [max_channel * self.bands + b + long_vertical_step];
                        let long_up_map_point = self.output_map[b + long_vertical_step];
                        let long_down_input = self.get_fractional_input_float(
                            max_channel,
                            long_up_map_point.input_bin
                                - long_vertical_step as f32 * bin_time_factor,
                        );
                        let long_vertical_twist =
                            long_up_prediction.input.mul_conj_rhs(long_down_input);
                        let long_up_output = self.channel_bands
                            [max_channel * self.bands + b + long_vertical_step]
                            .output;
                        phase += long_up_output.mul_conj_rhs(long_vertical_twist);
                    }
                }

                let output = prediction.make_output(phase);
                self.channel_bands[max_channel * self.bands + b].output = output;

                for c in 0..self.channels {
                    if c != max_channel {
                        let channel_prediction = self.channel_predictions[c * self.bands + b];
                        let channel_twist = channel_prediction.input.mul_conj_rhs(prediction.input);
                        let channel_phase = output * channel_twist;
                        self.channel_bands[c * self.bands + b].output =
                            channel_prediction.make_output(channel_phase);
                    }
                }
            }
            return;
        }
        step -= SPLIT_MAIN_PREDICTION;

        if self.block_process.new_spectrum && step == 0 {
            for band in &mut self.channel_bands {
                band.prev_input = band.input;
            }
        }
    }

    fn band_to_freq(&self, band: f32) -> f32 {
        self.stft.bin_to_freq(band)
    }

    #[allow(dead_code)]
    fn freq_to_band(&self, freq: f32) -> f32 {
        self.stft.freq_to_bin(freq)
    }

    fn band(&self, channel: usize, index: isize) -> Option<&Band> {
        if index < 0 || index as usize >= self.bands {
            return None;
        }
        Some(&self.channel_bands[channel * self.bands + index as usize])
    }

    fn get_fractional_input(&self, channel: usize, low_index: isize, fractional: f32) -> Complex {
        let low = self
            .band(channel, low_index)
            .map(|band| band.input)
            .unwrap_or(Complex::ZERO);
        let high = self
            .band(channel, low_index + 1)
            .map(|band| band.input)
            .unwrap_or(Complex::ZERO);
        low + (high - low) * fractional
    }

    fn get_fractional_prev_input(
        &self,
        channel: usize,
        low_index: isize,
        fractional: f32,
    ) -> Complex {
        let low = self
            .band(channel, low_index)
            .map(|band| band.prev_input)
            .unwrap_or(Complex::ZERO);
        let high = self
            .band(channel, low_index + 1)
            .map(|band| band.prev_input)
            .unwrap_or(Complex::ZERO);
        low + (high - low) * fractional
    }

    fn get_fractional_input_float(&self, channel: usize, input_index: f32) -> Complex {
        let low_index = input_index.floor() as isize;
        self.get_fractional_input(channel, low_index, input_index - low_index as f32)
    }

    fn get_fractional_energy(&self, channel: usize, low_index: isize, fractional: f32) -> f32 {
        let low = self
            .band(channel, low_index)
            .map(|band| band.input_energy)
            .unwrap_or(0.0);
        let high = self
            .band(channel, low_index + 1)
            .map(|band| band.input_energy)
            .unwrap_or(0.0);
        low + (high - low) * fractional
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_stays_finite_and_near_zero() {
        let mut stretch = Stretch::new();
        stretch.configure(1, 128, 32, false);
        let input = vec![0.0_f32; 256];
        let mut output = vec![1.0_f32; 256];
        let inputs = [&input[..]];
        let mut outputs = [&mut output[..]];

        stretch.process(&inputs, input.len(), &mut outputs, 256);

        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().all(|v| v.abs() < 1.0e-3));
    }

    #[test]
    fn sine_processes_to_finite_nonzero_output() {
        let mut stretch = Stretch::with_seed(1);
        stretch.configure(1, 256, 64, false);
        assert_eq!(stretch.input_latency(), 128);
        assert_eq!(stretch.output_latency(), 128);

        let input: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin() * 0.25)
            .collect();
        let mut output = vec![0.0_f32; 1024];
        let inputs = [&input[..]];
        let mut outputs = [&mut output[..]];

        stretch.process(&inputs, input.len(), &mut outputs, 1024);

        assert!(output.iter().all(|v| v.is_finite()));
        let peak = output.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        assert!(peak > 1.0e-4);
        assert!(peak < 2.0);

        // Smoke metric generated from the upstream C++ Signalsmith Stretch
        // implementation with the same config/input. This is intentionally a
        // loose aggregate check, not a claim of bit-identical output.
        let sum: f64 = output.iter().map(|&v| v as f64).sum();
        let sum2: f64 = output.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!((sum - 0.105_244).abs() < 1.0e-4);
        assert!((sum2 - 23.867_5).abs() < 1.0e-3);
        assert!((peak - 0.25).abs() < 1.0e-5);
    }

    #[test]
    fn split_computation_processes_finite_output() {
        let mut stretch = Stretch::with_seed(1);
        stretch.configure(1, 256, 64, true);
        assert_eq!(stretch.input_latency(), 128);
        assert_eq!(stretch.output_latency(), 192);

        let input: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 48_000.0).sin() * 0.2)
            .collect();
        let mut output = vec![0.0_f32; 1024];
        let inputs = [&input[..]];
        let mut outputs = [&mut output[..]];

        stretch.process(&inputs, input.len(), &mut outputs, 1024);

        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().any(|v| v.abs() > 1.0e-4));
    }

    #[test]
    fn exact_processes_fixed_buffer_to_requested_length() {
        let mut stretch = Stretch::with_seed(1);
        stretch.configure(1, 256, 64, false);
        let input: Vec<f32> = (0..1200)
            .map(|i| (2.0 * PI * 330.0 * i as f32 / 48_000.0).sin() * 0.2)
            .collect();
        let mut output = vec![0.0_f32; 1600];
        let inputs = [&input[..]];
        let mut outputs = [&mut output[..]];

        assert!(stretch.exact(&inputs, input.len(), &mut outputs, 1600));
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().any(|v| v.abs() > 1.0e-4));
    }

    #[test]
    fn exact_clears_output_when_input_is_too_short() {
        let mut stretch = Stretch::with_seed(1);
        stretch.configure(1, 256, 64, false);
        let input = [0.25_f32; 32];
        let mut output = vec![1.0_f32; 256];
        let inputs = [&input[..]];
        let mut outputs = [&mut output[..]];

        assert!(!stretch.exact(&inputs, input.len(), &mut outputs, 256));
        assert!(output.iter().all(|&v| v == 0.0));
    }
}
