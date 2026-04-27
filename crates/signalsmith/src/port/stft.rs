use super::complex::Complex;
use super::fft::RealFft;

const PI: f32 = core::f32::consts::PI;
const ALMOST_ZERO: f32 = 1.0e-30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WindowShape {
   Acg,
   Kaiser,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct InputState {
   pos: usize,
   buffer: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OutputState {
   pos: usize,
   buffer: Vec<f32>,
   window_products: Vec<f32>,
}

/// Rust port of `signalsmith::linear::DynamicSTFT<float, false,
/// STFT_SPECTRUM_MODIFIED>`.
///
/// The modified half-bin spectrum is important for matching Signalsmith Stretch;
/// this is why we port the STFT layer instead of using a generic STFT crate.
#[derive(Clone, Debug, Default)]
pub(crate) struct DynamicStft {
   fft: RealFft,
   analysis_channels: usize,
   synthesis_channels: usize,
   input_length_samples: usize,
   block_samples: usize,
   fft_samples: usize,
   fft_bins: usize,
   default_interval: usize,
   analysis_window: Vec<f32>,
   synthesis_window: Vec<f32>,
   analysis_offset: usize,
   synthesis_offset: usize,
   spectrum_buffer: Vec<Complex>,
   time_buffer: Vec<f32>,
   samples_since_synthesis: usize,
   samples_since_analysis: usize,
   pub input: InputState,
   pub output: OutputState,
}

impl DynamicStft {
   pub fn configure(
      &mut self,
      in_channels: usize,
      out_channels: usize,
      block_samples: usize,
      extra_input_history: usize,
   ) {
      self.analysis_channels = in_channels;
      self.synthesis_channels = out_channels;
      self.block_samples = block_samples;
      self.fft = RealFft::new_modified();
      self.fft_samples = RealFft::fast_size_above(block_samples.div_ceil(2)) * 2;
      self.fft.resize(self.fft_samples);
      self.fft_bins = self.fft_samples / 2;

      self.input_length_samples = self.block_samples + extra_input_history;
      self
         .input
         .buffer
         .resize(self.input_length_samples * self.analysis_channels, 0.0);
      self
         .output
         .buffer
         .resize(self.block_samples * self.synthesis_channels, 0.0);
      self.output.window_products.resize(self.block_samples, 0.0);
      self.spectrum_buffer.resize(
         self.fft_bins * self.analysis_channels.max(self.synthesis_channels),
         Complex::ZERO,
      );
      self.time_buffer.resize(self.fft_samples, 0.0);
      self.analysis_window.resize(self.block_samples, 0.0);
      self.synthesis_window.resize(self.block_samples, 0.0);
      self.set_interval(block_samples / 4, WindowShape::Acg);
      self.reset(1.0);
   }

   pub fn block_samples(&self) -> usize {
      self.block_samples
   }

   pub fn fft_samples(&self) -> usize {
      self.fft_samples
   }

   pub fn default_interval(&self) -> usize {
      self.default_interval
   }

   pub fn bands(&self) -> usize {
      self.fft_bins
   }

   pub fn analysis_latency(&self) -> usize {
      self.block_samples - self.analysis_offset
   }

   pub fn synthesis_latency(&self) -> usize {
      self.synthesis_offset
   }

   pub fn bin_to_freq(&self, b: f32) -> f32 {
      (b + 0.5) / self.fft_samples as f32
   }

   pub fn freq_to_bin(&self, f: f32) -> f32 {
      f * self.fft_samples as f32 - 0.5
   }

   pub fn reset(&mut self, product_weight: f32) {
      self.input.pos = self.block_samples;
      self.output.pos = 0;
      self.input.buffer.fill(0.0);
      self.output.buffer.fill(0.0);
      self.spectrum_buffer.fill(Complex::ZERO);
      self.output.window_products.fill(0.0);
      self.add_window_product();
      if self.default_interval > 0 {
         for i in (0..self.block_samples.saturating_sub(self.default_interval)).rev() {
            self.output.window_products[i] +=
               self.output.window_products[i + self.default_interval];
         }
      }
      for v in &mut self.output.window_products {
         *v = *v * product_weight + ALMOST_ZERO;
      }
      self.move_output(self.default_interval);
   }

   pub fn set_interval(&mut self, default_interval: usize, window_shape: WindowShape) {
      self.default_interval = default_interval;

      match window_shape {
         WindowShape::Acg => {
            let window = ApproximateConfinedGaussian::with_bandwidth(
               self.block_samples as f64 / default_interval as f64,
            );
            window.fill(&mut self.synthesis_window, 0.0, false);
         }
         WindowShape::Kaiser => {
            let window =
               Kaiser::with_bandwidth(self.block_samples as f64 / default_interval as f64, true);
            window.fill(&mut self.synthesis_window, 0.0, true);
         }
      }

      self.analysis_offset = self.block_samples / 2;
      self.synthesis_offset = self.block_samples / 2;
      if self.analysis_channels == 0 {
         self.analysis_window.fill(1.0);
      } else {
         force_perfect_reconstruction(
            &mut self.synthesis_window,
            self.block_samples,
            self.default_interval,
         );
         self.analysis_window.copy_from_slice(&self.synthesis_window);
      }

      for i in 0..self.block_samples {
         if self.analysis_window[i] > self.analysis_window[self.analysis_offset] {
            self.analysis_offset = i;
         }
         if self.synthesis_window[i] > self.synthesis_window[self.synthesis_offset] {
            self.synthesis_offset = i;
         }
      }
   }

   pub fn write_input(&mut self, channel: usize, length: usize, input: &[f32]) {
      let base = channel * self.input_length_samples;
      let offset_pos = self.input.pos % self.input_length_samples;
      let input_wrap_index = self.input_length_samples - offset_pos;
      let chunk1 = length.min(input_wrap_index);
      for i in 0..chunk1 {
         self.input.buffer[base + offset_pos + i] = input[i];
      }
      for i in chunk1..length {
         self.input.buffer[base + i + offset_pos - self.input_length_samples] = input[i];
      }
   }

   pub fn move_input(&mut self, samples: usize) {
      self.input.pos = (self.input.pos + samples) % self.input_length_samples;
      self.samples_since_analysis += samples;
   }

   pub fn finish_output(&mut self, strength: f32, offset: usize) {
      let mut max_window_product = 0.0_f32;
      let chunk1 = offset.max(self.block_samples.min(self.block_samples - self.output.pos));

      for i in offset..chunk1 {
         let i2 = self.output.pos + i;
         let wp = &mut self.output.window_products[i2];
         max_window_product = max_window_product.max(*wp);
         *wp += (max_window_product - *wp) * strength;
      }
      for i in chunk1..self.block_samples {
         let i2 = i + self.output.pos - self.block_samples;
         let wp = &mut self.output.window_products[i2];
         max_window_product = max_window_product.max(*wp);
         *wp += (max_window_product - *wp) * strength;
      }
   }

   pub fn read_output(&self, channel: usize, offset: usize, output: &mut [f32]) {
      let length = output.len();
      let base = channel * self.block_samples;
      let offset_pos = (self.output.pos + offset) % self.block_samples;
      let output_wrap_index = self.block_samples - offset_pos;
      let chunk1 = length.min(output_wrap_index);
      for i in 0..chunk1 {
         let i2 = offset_pos + i;
         output[i] = self.output.buffer[base + i2] / self.output.window_products[i2];
      }
      for i in chunk1..length {
         let i2 = i + offset_pos - self.block_samples;
         output[i] = self.output.buffer[base + i2] / self.output.window_products[i2];
      }
   }

   pub fn add_output(&mut self, channel: usize, length: usize, new_output: &[f32]) {
      let length = length.min(self.block_samples);
      let base = channel * self.block_samples;
      let offset_pos = self.output.pos;
      let output_wrap_index = self.block_samples - offset_pos;
      let chunk1 = length.min(output_wrap_index);
      for i in 0..chunk1 {
         let i2 = offset_pos + i;
         self.output.buffer[base + i2] += new_output[i] * self.output.window_products[i2];
      }
      for i in chunk1..length {
         let i2 = i + offset_pos - self.block_samples;
         self.output.buffer[base + i2] += new_output[i] * self.output.window_products[i2];
      }
   }

   pub fn move_output(&mut self, samples: usize) {
      if samples == 0 || self.block_samples == 0 {
         return;
      }
      if samples == 1 {
         for c in 0..self.synthesis_channels {
            self.output.buffer[self.output.pos + c * self.block_samples] = 0.0;
         }
         self.output.window_products[self.output.pos] = ALMOST_ZERO;
         self.output.pos += 1;
         if self.output.pos >= self.block_samples {
            self.output.pos = 0;
         }
         self.samples_since_synthesis += 1;
         return;
      }

      let output_wrap_index = self.block_samples - self.output.pos;
      let chunk1 = samples.min(output_wrap_index);
      for c in 0..self.synthesis_channels {
         let base = c * self.block_samples;
         for i in 0..chunk1 {
            self.output.buffer[base + self.output.pos + i] = 0.0;
         }
         for i in chunk1..samples {
            self.output.buffer[base + i + self.output.pos - self.block_samples] = 0.0;
         }
      }
      for i in 0..chunk1 {
         self.output.window_products[self.output.pos + i] = ALMOST_ZERO;
      }
      for i in chunk1..samples {
         self.output.window_products[i + self.output.pos - self.block_samples] = ALMOST_ZERO;
      }
      self.output.pos = (self.output.pos + samples) % self.block_samples;
      self.samples_since_synthesis += samples;
   }

   pub fn spectrum(&self, channel: usize) -> &[Complex] {
      let start = channel * self.fft_bins;
      &self.spectrum_buffer[start..start + self.fft_bins]
   }

   pub fn spectrum_mut(&mut self, channel: usize) -> &mut [Complex] {
      let start = channel * self.fft_bins;
      &mut self.spectrum_buffer[start..start + self.fft_bins]
   }

   pub fn analyse_steps(&self) -> usize {
      self.analysis_channels
   }

   pub fn analyse_step(&mut self, step: usize, samples_in_past: usize) {
      let channel = step;
      let offset_pos =
         (self.input_length_samples * 2 + self.input.pos - self.block_samples - samples_in_past)
            % self.input_length_samples;
      let input_wrap_index = self.input_length_samples - offset_pos;
      let chunk1 = self.analysis_offset.min(input_wrap_index);
      let chunk2 = self
         .analysis_offset
         .max(self.block_samples.min(input_wrap_index));

      self.samples_since_analysis = samples_in_past;
      let base = channel * self.input_length_samples;
      for i in 0..chunk1 {
         let w = -self.analysis_window[i];
         let ti = i + (self.fft_samples - self.analysis_offset);
         self.time_buffer[ti] = self.input.buffer[base + offset_pos + i] * w;
      }
      for i in chunk1..self.analysis_offset {
         let w = -self.analysis_window[i];
         let ti = i + (self.fft_samples - self.analysis_offset);
         self.time_buffer[ti] =
            self.input.buffer[base + i + offset_pos - self.input_length_samples] * w;
      }
      for i in self.analysis_offset..chunk2 {
         let w = self.analysis_window[i];
         let ti = i - self.analysis_offset;
         self.time_buffer[ti] = self.input.buffer[base + offset_pos + i] * w;
      }
      for i in chunk2..self.block_samples {
         let w = self.analysis_window[i];
         let ti = i - self.analysis_offset;
         self.time_buffer[ti] =
            self.input.buffer[base + i + offset_pos - self.input_length_samples] * w;
      }
      for i in
         (self.block_samples - self.analysis_offset)..(self.fft_samples - self.analysis_offset)
      {
         self.time_buffer[i] = 0.0;
      }

      let fft_bins = self.fft_bins;
      let start = channel * fft_bins;
      self.fft.fft(
         &self.time_buffer,
         &mut self.spectrum_buffer[start..start + fft_bins],
      );
   }

   pub fn synthesise_steps(&self) -> usize {
      self.synthesis_channels
   }

   pub fn synthesise_step(&mut self, step: usize) {
      if step == 0 {
         self.add_window_product();
      }

      let channel = step;
      let fft_bins = self.fft_bins;
      let start = channel * fft_bins;
      self.fft.ifft(
         &self.spectrum_buffer[start..start + fft_bins],
         &mut self.time_buffer,
      );

      let base = channel * self.block_samples;
      let output_wrap_index = self.block_samples - self.output.pos;
      let chunk1 = self.synthesis_offset.min(output_wrap_index);
      let chunk2 = self
         .block_samples
         .min(self.synthesis_offset.max(output_wrap_index));

      for i in 0..chunk1 {
         let w = -self.synthesis_window[i];
         let ti = i + (self.fft_samples - self.synthesis_offset);
         self.output.buffer[base + self.output.pos + i] += self.time_buffer[ti] * w;
      }
      for i in chunk1..self.synthesis_offset {
         let w = -self.synthesis_window[i];
         let ti = i + (self.fft_samples - self.synthesis_offset);
         self.output.buffer[base + i + self.output.pos - self.block_samples] +=
            self.time_buffer[ti] * w;
      }
      for i in self.synthesis_offset..chunk2 {
         let w = self.synthesis_window[i];
         let ti = i - self.synthesis_offset;
         self.output.buffer[base + self.output.pos + i] += self.time_buffer[ti] * w;
      }
      for i in chunk2..self.block_samples {
         let w = self.synthesis_window[i];
         let ti = i - self.synthesis_offset;
         self.output.buffer[base + i + self.output.pos - self.block_samples] +=
            self.time_buffer[ti] * w;
      }
   }

   fn add_window_product(&mut self) {
      self.samples_since_synthesis = 0;
      let window_shift = self.synthesis_offset as isize - self.analysis_offset as isize;
      let w_min = 0.max(window_shift) as usize;
      let w_max = self
         .block_samples
         .min((self.block_samples as isize + window_shift) as usize);
      let output_wrap_index = self.block_samples - self.output.pos;
      let chunk1 = w_max.min(w_min.max(output_wrap_index));
      for i in w_min..chunk1 {
         let wa = self.analysis_window[(i as isize - window_shift) as usize];
         let ws = self.synthesis_window[i];
         self.output.window_products[self.output.pos + i] += wa * ws * self.fft_samples as f32;
      }
      for i in chunk1..w_max {
         let wa = self.analysis_window[(i as isize - window_shift) as usize];
         let ws = self.synthesis_window[i];
         self.output.window_products[i + self.output.pos - self.block_samples] +=
            wa * ws * self.fft_samples as f32;
      }
   }
}

#[derive(Clone, Copy, Debug)]
struct Kaiser {
   beta: f64,
   inv_b0: f64,
}

impl Kaiser {
   fn bessel0(x: f64) -> f64 {
      let significance_limit = 1.0e-4;
      let mut result = 0.0;
      let mut term = 1.0;
      let mut m = 0.0;
      while term > significance_limit {
         result += term;
         m += 1.0;
         term *= (x * x) / (4.0 * m * m);
      }
      result
   }

   fn with_bandwidth(mut bandwidth: f64, heuristic_optimal: bool) -> Self {
      if heuristic_optimal {
         bandwidth = bandwidth
            + 8.0 / ((bandwidth + 3.0) * (bandwidth + 3.0))
            + 0.25 * (3.0 - bandwidth).max(0.0);
      }
      bandwidth = bandwidth.max(2.0);
      let alpha = (bandwidth * bandwidth * 0.25 - 1.0).sqrt();
      let beta = alpha * PI as f64;
      Self {
         beta,
         inv_b0: 1.0 / Self::bessel0(beta),
      }
   }

   fn fill(&self, data: &mut [f32], warp: f64, is_for_synthesis: bool) {
      let size = data.len();
      let inv_size = 1.0 / size as f64;
      let offset_i = if size & 1 != 0 {
         1
      } else if is_for_synthesis {
         0
      } else {
         2
      };
      for (i, value) in data.iter_mut().enumerate() {
         let mut r = (2 * i + offset_i) as f64 * inv_size - 1.0;
         r = (r + warp) / (1.0 + r * warp);
         let arg = (1.0 - r * r).sqrt();
         *value = (Self::bessel0(self.beta * arg) * self.inv_b0) as f32;
      }
   }
}

#[derive(Clone, Copy, Debug)]
struct ApproximateConfinedGaussian {
   gaussian_factor: f64,
}

impl ApproximateConfinedGaussian {
   fn with_bandwidth(bandwidth: f64) -> Self {
      let sigma = 0.3 / bandwidth.sqrt();
      Self {
         gaussian_factor: 0.0625 / (sigma * sigma),
      }
   }

   fn gaussian(&self, x: f64) -> f64 {
      (-x * x * self.gaussian_factor).exp()
   }

   fn fill(&self, data: &mut [f32], warp: f64, is_for_synthesis: bool) {
      let size = data.len();
      let inv_size = 1.0 / size as f64;
      let offset_scale = self.gaussian(1.0) / (self.gaussian(3.0) + self.gaussian(-1.0));
      let norm = 1.0 / (self.gaussian(0.0) - 2.0 * offset_scale * self.gaussian(2.0));
      let offset_i = if size & 1 != 0 {
         1
      } else if is_for_synthesis {
         0
      } else {
         2
      };
      for (i, value) in data.iter_mut().enumerate() {
         let mut r = (2 * i + offset_i) as f64 * inv_size - 1.0;
         r = (r + warp) / (1.0 + r * warp);
         *value = (norm
            * (self.gaussian(r) - offset_scale * (self.gaussian(r - 2.0) + self.gaussian(r + 2.0))))
            as f32;
      }
   }
}

fn force_perfect_reconstruction(data: &mut [f32], window_length: usize, interval: usize) {
   for i in 0..interval {
      let mut sum2 = 0.0_f64;
      let mut index = i;
      while index < window_length {
         sum2 += (data[index] as f64) * (data[index] as f64);
         index += interval;
      }
      let factor = (1.0 / sum2.sqrt()) as f32;
      let mut index = i;
      while index < window_length {
         data[index] *= factor;
         index += interval;
      }
   }
}
