use super::complex::Complex;

const PI: f32 = core::f32::consts::PI;

#[derive(Clone, Debug, Default)]
struct Pow2Fft {
   size: usize,
}

impl Pow2Fft {
   fn resize(&mut self, size: usize) {
      self.size = size;
   }

   fn fft(&self, input: &[Complex], output: &mut [Complex]) {
      self.transform(input, output, false);
   }

   fn ifft(&self, input: &[Complex], output: &mut [Complex]) {
      self.transform(input, output, true);
   }

   fn transform(&self, input: &[Complex], output: &mut [Complex], inverse: bool) {
      let n = self.size;
      if n == 0 {
         return;
      }
      debug_assert!(input.len() >= n);
      debug_assert!(output.len() >= n);
      if n == 1 {
         output[0] = input[0];
         return;
      }
      debug_assert!(n.is_power_of_two());

      let bits = n.trailing_zeros();
      for i in 0..n {
         let j = i.reverse_bits() >> (usize::BITS - bits);
         output[j] = input[i];
      }

      let mut len = 2;
      while len <= n {
         let angle = if inverse { 2.0 * PI } else { -2.0 * PI } / len as f32;
         let w_len = Complex::polar(1.0, angle);
         for start in (0..n).step_by(len) {
            let mut w = Complex::from(1.0);
            for j in 0..(len / 2) {
               let u = output[start + j];
               let v = output[start + j + len / 2] * w;
               output[start + j] = u + v;
               output[start + j + len / 2] = u - v;
               w *= w_len;
            }
         }
         len *= 2;
      }
   }
}

#[derive(Clone, Copy, Debug)]
enum StepType {
   Passthrough,
   InterleaveOrder2,
   InterleaveOrder3,
   InterleaveOrder4,
   InterleaveOrder5,
   InterleaveOrderN,
   FirstFft,
   MiddleFft,
   Twiddles,
   FinalOrder2,
   FinalOrder3,
   FinalOrder4,
   FinalOrder5,
   FinalOrderN,
}

#[derive(Clone, Copy, Debug)]
struct Step {
   step_type: StepType,
   offset: usize,
}

/// Port of Signalsmith Linear's `SplitFFT<float, false>` planning.
///
/// The upstream implementation supports fast platform backends and chunked
/// computation. This Rust version keeps the same mixed-radix decomposition and
/// unnormalised transform convention, which is what the STFT/stretcher rely on.
#[derive(Clone, Debug, Default)]
pub(crate) struct SplitFft {
   inner_fft: Pow2Fft,
   inner_size: usize,
   outer_size: usize,
   tmp_freq: Vec<Complex>,
   outer_twiddles: Vec<Complex>,
   dft_twists: Vec<Complex>,
   dft_tmp: Vec<Complex>,
   plan: Vec<Step>,
}

impl SplitFft {
   const MIN_INNER_SIZE: usize = 32;

   pub fn fast_size_above(size: usize) -> usize {
      let mut pow2 = 1;
      while pow2 < 16 && pow2 < size {
         pow2 *= 2;
      }
      while pow2 * 8 < size {
         pow2 *= 2;
      }
      let mut multiple = size.div_ceil(pow2);
      if multiple == 7 {
         multiple += 1;
      }
      multiple * pow2
   }

   pub fn resize(&mut self, size: usize) {
      self.inner_size = 1;
      self.outer_size = size;
      self.tmp_freq.clear();
      self.outer_twiddles.clear();
      self.dft_twists.clear();
      self.dft_tmp.clear();
      self.plan.clear();

      if size == 0 {
         return;
      }

      while self.outer_size & 1 == 0
         && (self.outer_size > 1 || self.inner_size < Self::MIN_INNER_SIZE)
      {
         self.inner_size *= 2;
         self.outer_size /= 2;
      }

      self.tmp_freq.resize(size, Complex::ZERO);
      self.inner_fft.resize(self.inner_size);

      self.outer_twiddles.resize(
         self.inner_size * self.outer_size.saturating_sub(1),
         Complex::ZERO,
      );
      for i in 0..self.inner_size {
         for s in 1..self.outer_size {
            let phase =
               -2.0 * PI * i as f32 / self.inner_size as f32 * s as f32 / self.outer_size as f32;
            self.outer_twiddles[i + (s - 1) * self.inner_size] = Complex::polar(1.0, phase);
         }
      }

      let (interleave_step, final_step) = match self.outer_size {
         2 => (StepType::InterleaveOrder2, StepType::FinalOrder2),
         3 => (StepType::InterleaveOrder3, StepType::FinalOrder3),
         4 => (StepType::InterleaveOrder4, StepType::FinalOrder4),
         5 => (StepType::InterleaveOrder5, StepType::FinalOrder5),
         _ => (StepType::InterleaveOrderN, StepType::FinalOrderN),
      };

      if self.outer_size <= 1 {
         self.plan.push(Step {
            step_type: StepType::Passthrough,
            offset: 0,
         });
      } else {
         self.plan.push(Step {
            step_type: interleave_step,
            offset: 0,
         });
         self.plan.push(Step {
            step_type: StepType::FirstFft,
            offset: 0,
         });
         for s in 1..self.outer_size {
            self.plan.push(Step {
               step_type: StepType::MiddleFft,
               offset: s * self.inner_size,
            });
         }
         self.plan.push(Step {
            step_type: StepType::Twiddles,
            offset: 0,
         });
         self.plan.push(Step {
            step_type: final_step,
            offset: 0,
         });

         if matches!(final_step, StepType::FinalOrderN) {
            self.dft_tmp.resize(self.outer_size, Complex::ZERO);
            self.dft_twists.resize(self.outer_size, Complex::ZERO);
            for s in 0..self.outer_size {
               self.dft_twists[s] =
                  Complex::polar(1.0, -2.0 * PI * s as f32 / self.outer_size as f32);
            }
         }
      }
   }

   pub fn size(&self) -> usize {
      self.inner_size * self.outer_size
   }

   pub fn fft(&mut self, time: &[Complex], freq: &mut [Complex]) {
      let plan = self.plan.clone();
      for step in plan {
         self.fft_step(step, time, freq, false);
      }
   }

   pub fn ifft(&mut self, freq: &[Complex], time: &mut [Complex]) {
      let plan = self.plan.clone();
      for step in plan {
         self.fft_step(step, freq, time, true);
      }
   }

   fn fft_step(&mut self, step: Step, input: &[Complex], output: &mut [Complex], inverse: bool) {
      match step.step_type {
         StepType::Passthrough => {
            if inverse {
               self.inner_fft.ifft(input, output);
            } else {
               self.inner_fft.fft(input, output);
            }
         }
         StepType::InterleaveOrder2 => self.interleave_copy(input, 2),
         StepType::InterleaveOrder3 => self.interleave_copy(input, 3),
         StepType::InterleaveOrder4 => self.interleave_copy(input, 4),
         StepType::InterleaveOrder5 => self.interleave_copy(input, 5),
         StepType::InterleaveOrderN => self.interleave_copy(input, self.outer_size),
         StepType::FirstFft => {
            if inverse {
               self.inner_fft.ifft(&self.tmp_freq, output);
            } else {
               self.inner_fft.fft(&self.tmp_freq, output);
            }
         }
         StepType::MiddleFft => {
            let offset = step.offset;
            if inverse {
               self.inner_fft.ifft(
                  &self.tmp_freq[offset..],
                  &mut output[offset..offset + self.inner_size],
               );
            } else {
               self.inner_fft.fft(
                  &self.tmp_freq[offset..],
                  &mut output[offset..offset + self.inner_size],
               );
            }
         }
         StepType::Twiddles => {
            let len = self.inner_size * self.outer_size.saturating_sub(1);
            for i in 0..len {
               let index = self.inner_size + i;
               output[index] = if inverse {
                  output[index].mul_conj_rhs(self.outer_twiddles[i])
               } else {
                  output[index] * self.outer_twiddles[i]
               };
            }
         }
         StepType::FinalOrder2 => self.final_pass2(output),
         StepType::FinalOrder3 => self.final_pass3(output, inverse),
         StepType::FinalOrder4 => self.final_pass4(output, inverse),
         StepType::FinalOrder5 => self.final_pass5(output, inverse),
         StepType::FinalOrderN => self.final_pass_n(output, inverse),
      }
   }

   fn interleave_copy(&mut self, input: &[Complex], stride: usize) {
      for bi in 0..self.inner_size {
         for ai in 0..stride {
            self.tmp_freq[ai * self.inner_size + bi] = input[bi * stride + ai];
         }
      }
   }

   fn final_pass2(&self, f: &mut [Complex]) {
      let (f0, f1) = f.split_at_mut(self.inner_size);
      for i in 0..self.inner_size {
         let a = f0[i];
         let b = f1[i];
         f0[i] = a + b;
         f1[i] = a - b;
      }
   }

   fn final_pass3(&self, f: &mut [Complex], inverse: bool) {
      let tw1 = Complex::new(-0.5, -0.75_f32.sqrt() * if inverse { -1.0 } else { 1.0 });
      for i in 0..self.inner_size {
         let a = f[i];
         let b = f[i + self.inner_size];
         let c = f[i + 2 * self.inner_size];
         let bc0 = b + c;
         let bc1 = b - c;
         f[i] = a + bc0;
         f[i + self.inner_size] = Complex::new(
            a.re + bc0.re * tw1.re - bc1.im * tw1.im,
            a.im + bc0.im * tw1.re + bc1.re * tw1.im,
         );
         f[i + 2 * self.inner_size] = Complex::new(
            a.re + bc0.re * tw1.re + bc1.im * tw1.im,
            a.im + bc0.im * tw1.re - bc1.re * tw1.im,
         );
      }
   }

   fn final_pass4(&self, f: &mut [Complex], inverse: bool) {
      for i in 0..self.inner_size {
         let a = f[i];
         let b = f[i + self.inner_size];
         let c = f[i + 2 * self.inner_size];
         let d = f[i + 3 * self.inner_size];
         let ac0 = a + c;
         let ac1 = a - c;
         let bd0 = b + d;
         let bd1 = if inverse { b - d } else { d - b };
         let bd1i = Complex::new(-bd1.im, bd1.re);
         f[i] = ac0 + bd0;
         f[i + self.inner_size] = ac1 + bd1i;
         f[i + 2 * self.inner_size] = ac0 - bd0;
         f[i + 3 * self.inner_size] = ac1 - bd1i;
      }
   }

   fn final_pass5(&self, f: &mut [Complex], inverse: bool) {
      let sign = if inverse { -1.0 } else { 1.0 };
      let tw1r = 0.309_016_97;
      let tw1i = -0.951_056_54 * sign;
      let tw2r = -0.809_017;
      let tw2i = -0.587_785_24 * sign;
      for i in 0..self.inner_size {
         let a = f[i];
         let b = f[i + self.inner_size];
         let c = f[i + 2 * self.inner_size];
         let d = f[i + 3 * self.inner_size];
         let e = f[i + 4 * self.inner_size];

         let be0 = b + e;
         let be1 = Complex::new(e.im - b.im, b.re - e.re);
         let cd0 = c + d;
         let cd1 = Complex::new(d.im - c.im, c.re - d.re);

         let bcde01 = be0 * tw1r + cd0 * tw2r;
         let bcde02 = be0 * tw2r + cd0 * tw1r;
         let bcde11 = be1 * tw1i + cd1 * tw2i;
         let bcde12 = be1 * tw2i - cd1 * tw1i;

         f[i] = a + be0 + cd0;
         f[i + self.inner_size] = a + bcde01 + bcde11;
         f[i + 2 * self.inner_size] = a + bcde02 + bcde12;
         f[i + 3 * self.inner_size] = a + bcde02 - bcde12;
         f[i + 4 * self.inner_size] = a + bcde01 - bcde11;
      }
   }

   fn final_pass_n(&mut self, f: &mut [Complex], inverse: bool) {
      for i in 0..self.inner_size {
         let mut sum = Complex::ZERO;
         for i2 in 0..self.outer_size {
            self.dft_tmp[i2] = f[i + i2 * self.inner_size];
            sum += self.dft_tmp[i2];
         }
         f[i] = sum;

         for out_bin in 1..self.outer_size {
            let mut sum = self.dft_tmp[0];
            for i2 in 1..self.outer_size {
               let twist_index = (i2 * out_bin) % self.outer_size;
               let twist = if inverse {
                  self.dft_twists[twist_index].conj()
               } else {
                  self.dft_twists[twist_index]
               };
               sum += self.dft_tmp[i2] * twist;
            }
            f[i + out_bin * self.inner_size] = sum;
         }
      }
   }
}

/// Real FFT with Signalsmith Linear's optional half-bin shift.
///
/// `half_bin_shift=true` is the key layout used by `DynamicSTFT<float, false,
/// STFT_SPECTRUM_MODIFIED>` in Signalsmith Stretch. It avoids a DC/Nyquist bin
/// and centres band 0 at half a bin, so the stretcher's phase predictors see
/// the same frequency grid as upstream.
#[derive(Clone, Debug, Default)]
pub(crate) struct RealFft {
   complex_fft: SplitFft,
   tmp_freq: Vec<Complex>,
   tmp_time: Vec<Complex>,
   twiddles: Vec<Complex>,
   half_bin_twists: Vec<Complex>,
   half_bin_shift: bool,
}

impl RealFft {
   pub fn new_modified() -> Self {
      Self {
         half_bin_shift: true,
         ..Self::default()
      }
   }

   pub fn fast_size_above(size: usize) -> usize {
      SplitFft::fast_size_above(size.div_ceil(2)) * 2
   }

   pub fn resize(&mut self, size: usize) {
      let h_size = size / 2;
      self.complex_fft.resize(h_size);
      self.tmp_freq.resize(h_size, Complex::ZERO);
      self.tmp_time.resize(h_size, Complex::ZERO);
      self.twiddles.resize(h_size / 2 + 1, Complex::ZERO);

      if !self.half_bin_shift {
         for i in 0..self.twiddles.len() {
            let phase = i as f32 * (-2.0 * PI / size as f32) - PI / 2.0;
            self.twiddles[i] = Complex::polar(1.0, phase);
         }
      } else {
         for i in 0..self.twiddles.len() {
            let phase = (i as f32 + 0.5) * (-2.0 * PI / size as f32) - PI / 2.0;
            self.twiddles[i] = Complex::polar(1.0, phase);
         }

         self.half_bin_twists.resize(h_size, Complex::ZERO);
         for i in 0..h_size {
            let phase = -2.0 * PI * i as f32 / size as f32;
            self.half_bin_twists[i] = Complex::polar(1.0, phase);
         }
      }
   }

   pub fn fft(&mut self, time: &[f32], freq: &mut [Complex]) {
      let h_size = self.complex_fft.size();
      if self.half_bin_shift {
         for i in 0..h_size {
            let tr = time[2 * i];
            let ti = time[2 * i + 1];
            let twist = self.half_bin_twists[i];
            self.tmp_time[i] =
               Complex::new(tr * twist.re - ti * twist.im, ti * twist.re + tr * twist.im);
         }
      } else {
         for i in 0..h_size {
            self.tmp_time[i] = Complex::new(time[2 * i], time[2 * i + 1]);
         }
      }

      self.complex_fft.fft(&self.tmp_time, &mut self.tmp_freq);

      if !self.half_bin_shift {
         let bin0 = self.tmp_freq[0];
         freq[0] = Complex::new(bin0.re + bin0.im, bin0.re - bin0.im);
      }

      let start_i = if self.half_bin_shift { 0 } else { 1 };
      let end_i = h_size / 2 + 1;
      for i in start_i..end_i {
         let conj_i = if self.half_bin_shift {
            h_size - 1 - i
         } else {
            h_size - i
         };
         let twiddle = self.twiddles[i];

         let odd = (self.tmp_freq[i] + self.tmp_freq[conj_i].conj()) * 0.5;
         let even_i = (self.tmp_freq[i] - self.tmp_freq[conj_i].conj()) * 0.5;
         let even_rot_minus_i = even_i * twiddle;

         freq[i] = odd + even_rot_minus_i;
         freq[conj_i] = Complex::new(odd.re - even_rot_minus_i.re, even_rot_minus_i.im - odd.im);
      }
   }

   pub fn ifft(&mut self, freq: &[Complex], time: &mut [f32]) {
      let h_size = self.complex_fft.size();
      if !self.half_bin_shift {
         let bin0 = freq[0];
         self.tmp_freq[0] = Complex::new(bin0.re + bin0.im, bin0.re - bin0.im);
      }

      let start_i = if self.half_bin_shift { 0 } else { 1 };
      let end_i = h_size / 2 + 1;
      for i in start_i..end_i {
         let conj_i = if self.half_bin_shift {
            h_size - 1 - i
         } else {
            h_size - i
         };
         let twiddle = self.twiddles[i];

         let odd = freq[i] + freq[conj_i].conj();
         let even_rot_minus_i = freq[i] - freq[conj_i].conj();
         let even_i = Complex::new(
            even_rot_minus_i.re * twiddle.re + even_rot_minus_i.im * twiddle.im,
            even_rot_minus_i.im * twiddle.re - even_rot_minus_i.re * twiddle.im,
         );

         self.tmp_freq[i] = odd + even_i;
         self.tmp_freq[conj_i] = Complex::new(odd.re - even_i.re, even_i.im - odd.im);
      }

      self.complex_fft.ifft(&self.tmp_freq, &mut self.tmp_time);

      if self.half_bin_shift {
         for i in 0..h_size {
            let t = self.tmp_time[i];
            let twist = self.half_bin_twists[i];
            time[2 * i] = t.re * twist.re + t.im * twist.im;
            time[2 * i + 1] = t.im * twist.re - t.re * twist.im;
         }
      } else {
         for i in 0..h_size {
            time[2 * i] = self.tmp_time[i].re;
            time[2 * i + 1] = self.tmp_time[i].im;
         }
      }
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn split_fft_round_trips_without_normalisation() {
      let n = SplitFft::fast_size_above(45);
      let mut fft = SplitFft::default();
      fft.resize(n);
      let input: Vec<Complex> = (0..n)
         .map(|i| Complex::new((i as f32 * 0.13).sin(), (i as f32 * 0.07).cos()))
         .collect();
      let mut freq = vec![Complex::ZERO; n];
      let mut output = vec![Complex::ZERO; n];

      fft.fft(&input, &mut freq);
      fft.ifft(&freq, &mut output);

      for (actual, expected) in output.iter().zip(input.iter()) {
         assert!((actual.re / n as f32 - expected.re).abs() < 1.0e-4);
         assert!((actual.im / n as f32 - expected.im).abs() < 1.0e-4);
      }
   }

   #[test]
   fn modified_real_fft_round_trips_without_normalisation() {
      let n = 96;
      let mut fft = RealFft::new_modified();
      fft.resize(n);
      let input: Vec<f32> = (0..n)
         .map(|i| (i as f32 * 0.11).sin() + 0.25 * (i as f32 * 0.37).cos())
         .collect();
      let mut freq = vec![Complex::ZERO; n / 2];
      let mut output = vec![0.0; n];

      fft.fft(&input, &mut freq);
      fft.ifft(&freq, &mut output);

      for (actual, expected) in output.iter().zip(input.iter()) {
         assert!((actual / n as f32 - expected).abs() < 1.0e-4);
      }
   }
}
