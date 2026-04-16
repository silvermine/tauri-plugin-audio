// Constant-rate WSOLA (Waveform Similarity Overlap-Add) time-scale modification.
//
// Adapted from the constant-alpha path of libtsm by Meinard Müller et al.
// (MIT license). This is not a full port of libtsm — it omits anchor-point /
// time-varying stretch support, which is not needed for fixed playback-rate
// changes.
//
// Reference: Driedger & Müller, "TSM Toolbox: MATLAB Implementations of
// Time-Scale Modification Algorithms", DAFx 2014.
//
// Default parameters assume 44.1–48 kHz sample rates (~21–23 ms windows).

use std::f64::consts::PI;

/// WSOLA parameters with sensible defaults from libtsm.
pub(crate) struct WsolaParams {
   /// Hop size of the synthesis window (samples).
   pub syn_hop: usize,
   /// Length of the analysis/synthesis window (samples).
   pub win_length: usize,
   /// Exponent for the sin^β window.
   pub win_beta: u32,
   /// Tolerance in samples for the cross-correlation search.
   pub tolerance: usize,
}

impl Default for WsolaParams {
   fn default() -> Self {
      Self {
         syn_hop: 512,
         win_length: 1024,
         win_beta: 2,
         tolerance: 512,
      }
   }
}

/// Generate a sin^β window of length `len`.
fn sin_beta_window(len: usize, beta: u32) -> Vec<f64> {
   (0..len)
      .map(|i| (PI * i as f64 / len as f64).sin().powi(beta as i32))
      .collect()
}

/// Sliding dot-product cross-correlation.
///
/// `x` has length `win_len + 2 * tol`, `y` has length `win_len`.
/// Returns a vector of length `2 * tol + 1`.
fn cross_corr(x: &[f64], y: &[f64], win_len: usize) -> Vec<f64> {
   let n = x.len() - win_len + 1;
   (0..n)
      .map(|offset| {
         x[offset..offset + win_len]
            .iter()
            .zip(y.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
      })
      .collect()
}

/// Find the index of the maximum value in a slice.
fn argmax(v: &[f64]) -> usize {
   v.iter()
      .enumerate()
      .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
      .map(|(i, _)| i)
      .unwrap_or(0)
}

fn delay_from_match_index(tol: usize, max_index: usize) -> isize {
   max_index as isize - tol as isize
}

/// Time-scale modify a single channel using WSOLA.
///
/// `samples` is one channel of f32 PCM. `alpha` is the stretch factor
/// (>1.0 = slower / longer output, <1.0 = faster / shorter output).
///
/// Returns the stretched channel as `Vec<f32>`.
pub(crate) fn wsola_single_channel(samples: &[f32], alpha: f64, params: &WsolaParams) -> Vec<f32> {
   assert!(alpha > 0.0, "alpha must be positive");
   let n = samples.len();
   if n == 0 {
      return Vec::new();
   }

   let win_len = params.win_length;
   let syn_hop = params.syn_hop;
   let tol = params.tolerance;
   let win_len_half = win_len / 2;

   // Generate window
   let w = sin_beta_window(win_len, params.win_beta);

   // Convert input to f64 for precision during processing
   let x_orig: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

   // Output length
   let output_length = ((alpha * n as f64).ceil() as usize).max(1);

   // Synthesis window positions
   let mut syn_win_pos: Vec<usize> = Vec::new();
   let mut pos = 0usize;
   while pos < output_length + win_len_half {
      syn_win_pos.push(pos);
      pos += syn_hop;
   }

   // Analysis window positions (constant alpha → linear mapping)
   let ana_win_pos: Vec<isize> = syn_win_pos
      .iter()
      .map(|&s| (s as f64 / alpha).round() as isize)
      .collect();

   // Analysis hops (for padding calculation)
   let ana_hop_sizes: Vec<isize> = {
      let mut h = Vec::with_capacity(ana_win_pos.len());
      h.push(0);
      for i in 1..ana_win_pos.len() {
         h.push(ana_win_pos[i] - ana_win_pos[i - 1]);
      }
      h
   };

   // Validate analysis hops are positive
   debug_assert!(
      ana_hop_sizes[1..].iter().all(|&h| h > 0),
      "analysis hops must be strictly positive"
   );

   // Minimal local stretching factor for padding calculation
   let min_fac: f64 = ana_hop_sizes[1..]
      .iter()
      .map(|&h| syn_hop as f64 / h as f64)
      .fold(f64::INFINITY, f64::min);

   let pad_end = ((1.0 / min_fac).ceil() as usize) * win_len + tol;
   let pad_start = win_len_half + tol;

   // Zero-pad the input
   let padded_len = pad_start + x_orig.len() + pad_end;
   let mut x = vec![0.0f64; padded_len];
   x[pad_start..pad_start + x_orig.len()].copy_from_slice(&x_orig);

   // Shift analysis positions to compensate for start padding
   let ana_win_pos: Vec<usize> = ana_win_pos
      .iter()
      .map(|&p| (p + tol as isize) as usize)
      .collect();

   // Output buffer and overlap weight buffer
   let buf_len = output_length + 2 * win_len;
   let mut y = vec![0.0f64; buf_len];
   let mut ow = vec![0.0f64; buf_len];
   let mut delay: isize = 0;

   let frame_count = ana_win_pos.len();

   for i in 0..frame_count.saturating_sub(1) {
      let syn_start = syn_win_pos[i];
      let ana_start = (ana_win_pos[i] as isize + delay) as usize;

      // Overlap-add
      for j in 0..win_len {
         if syn_start + j < buf_len && ana_start + j < x.len() {
            y[syn_start + j] += x[ana_start + j] * w[j];
            ow[syn_start + j] += w[j];
         }
      }

      // Natural progression of the last copied segment
      let nat_prog_start = ana_start + syn_hop;
      let nat_prog: Vec<f64> = (0..win_len)
         .map(|j| {
            if nat_prog_start + j < x.len() {
               x[nat_prog_start + j]
            } else {
               0.0
            }
         })
         .collect();

      // Next analysis window range including tolerance region
      let next_start = ana_win_pos[i + 1] as isize - tol as isize;
      let next_len = win_len + 2 * tol;
      let x_next: Vec<f64> = (0..next_len)
         .map(|j| {
            let idx = (next_start + j as isize) as usize;
            if idx < x.len() { x[idx] } else { 0.0 }
         })
         .collect();

      // Cross-correlation to find best overlap
      let cc = cross_corr(&x_next, &nat_prog, win_len);
      let max_index = argmax(&cc);
      delay = delay_from_match_index(tol, max_index);
   }

   // Process last frame.
   //
   // NOTE: intentional divergence from upstream libtsm. The Python source
   // reuses the loop variable `i` (left at frame_count - 2) for the last-
   // frame analysis position, which appears to be an off-by-one bug. We
   // use `ana_win_pos[last]` (the true last frame) instead, which gives
   // correct alignment of the final overlap-add window.
   if frame_count > 0 {
      let last = frame_count - 1;
      let syn_start = syn_win_pos[last];
      let ana_start = (ana_win_pos[last] as isize + delay) as usize;
      for j in 0..win_len {
         if syn_start + j < buf_len && ana_start + j < x.len() {
            y[syn_start + j] += x[ana_start + j] * w[j];
            ow[syn_start + j] += w[j];
         }
      }
   }

   // Normalize by overlapping windows
   for v in ow.iter_mut() {
      if *v < 1e-3 {
         *v = 1.0;
      }
   }
   for (y_val, ow_val) in y.iter_mut().zip(ow.iter()) {
      *y_val /= ow_val;
   }

   // Remove padding and trim to output length
   let start = win_len_half;
   let end = (start + output_length).min(y.len());
   y[start..end].iter().map(|&v| v as f32).collect()
}

/// Time-scale modify multi-channel audio using WSOLA.
///
/// `channels` is per-channel PCM data (`channels[c][sample]`).
/// `alpha` is the stretch factor (>1 = slower, <1 = faster).
///
/// Returns per-channel stretched PCM.
pub(crate) fn wsola(channels: &[Vec<f32>], alpha: f64, params: &WsolaParams) -> Vec<Vec<f32>> {
   channels
      .iter()
      .map(|ch| wsola_single_channel(ch, alpha, params))
      .collect()
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn sin_beta_window_sums_correctly() {
      let w = sin_beta_window(1024, 2);
      assert_eq!(w.len(), 1024);
      // First element is sin(0)^2 = 0
      assert!(w[0].abs() < 1e-10);
      // Last element is sin(π * 1023/1024)^2 ≈ small but not zero
      assert!(w[1023] < 0.01);
      // Peak at center: sin(π * 512/1024)^2 = sin(π/2)^2 = 1.0
      let mid = w[512];
      assert!((mid - 1.0).abs() < 1e-6);
   }

   #[test]
   fn cross_corr_identity() {
      let signal: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
      let pattern = signal[5..15].to_vec();
      let cc = cross_corr(&signal, &pattern, 10);
      // Best match should be at offset 5
      assert_eq!(cc.len(), 11);
      assert_eq!(argmax(&cc), 5);
   }

   #[test]
   fn delay_from_match_index_tracks_search_offset_direction() {
      assert_eq!(delay_from_match_index(512, 512), 0);
      assert_eq!(delay_from_match_index(512, 640), 128);
      assert_eq!(delay_from_match_index(512, 384), -128);
   }

   #[test]
   fn wsola_passthrough_at_1x() {
      // 1.0× stretch should return approximately the same length
      let input: Vec<f32> = (0..4410)
         .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32)
         .collect();
      let out = wsola_single_channel(&input, 1.0, &WsolaParams::default());
      let len_ratio = out.len() as f64 / input.len() as f64;
      assert!(
         (len_ratio - 1.0).abs() < 0.05,
         "1.0× stretch length ratio {len_ratio} too far from 1.0"
      );
   }

   #[test]
   fn wsola_doubles_length_at_2x() {
      let input: Vec<f32> = (0..4410)
         .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32)
         .collect();
      let out = wsola_single_channel(&input, 2.0, &WsolaParams::default());
      let len_ratio = out.len() as f64 / input.len() as f64;
      assert!(
         (len_ratio - 2.0).abs() < 0.1,
         "2.0× stretch length ratio {len_ratio} too far from 2.0"
      );
   }

   #[test]
   fn wsola_halves_length_at_0_5x() {
      let input: Vec<f32> = (0..8820)
         .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32)
         .collect();
      let out = wsola_single_channel(&input, 0.5, &WsolaParams::default());
      let len_ratio = out.len() as f64 / input.len() as f64;
      assert!(
         (len_ratio - 0.5).abs() < 0.1,
         "0.5× stretch length ratio {len_ratio} too far from 0.5"
      );
   }

   #[test]
   fn wsola_no_nan_or_inf() {
      let input: Vec<f32> = (0..4410)
         .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32)
         .collect();
      for &alpha in &[0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
         let out = wsola_single_channel(&input, alpha, &WsolaParams::default());
         assert!(
            out.iter().all(|v| v.is_finite()),
            "NaN/Inf found at alpha={alpha}"
         );
      }
   }

   #[test]
   fn wsola_empty_input() {
      let out = wsola_single_channel(&[], 1.5, &WsolaParams::default());
      assert!(out.is_empty());
   }

   #[test]
   fn wsola_multi_channel() {
      let ch0: Vec<f32> = (0..4410)
         .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32)
         .collect();
      let ch1: Vec<f32> = (0..4410)
         .map(|i| (2.0 * PI * 880.0 * i as f64 / 44100.0).sin() as f32)
         .collect();
      let channels = vec![ch0.clone(), ch1.clone()];
      let out = wsola(&channels, 1.5, &WsolaParams::default());
      assert_eq!(out.len(), 2);
      let expected_len = (4410.0_f64 * 1.5).ceil() as usize;
      for (c, ch) in out.iter().enumerate() {
         let ratio = ch.len() as f64 / expected_len as f64;
         assert!(
            (ratio - 1.0).abs() < 0.05,
            "channel {c} length ratio {ratio} too far from 1.0"
         );
         assert!(ch.iter().all(|v| v.is_finite()));
      }
   }

   #[test]
   fn wsola_energy_preservation() {
      let input: Vec<f32> = (0..8820)
         .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32)
         .collect();
      let input_rms =
         (input.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / input.len() as f64).sqrt();

      for &alpha in &[0.75, 1.25, 1.5] {
         let out = wsola_single_channel(&input, alpha, &WsolaParams::default());
         let out_rms =
            (out.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / out.len() as f64).sqrt();
         let ratio = out_rms / input_rms;
         assert!(
            (ratio - 1.0).abs() < 0.3,
            "RMS ratio {ratio} at alpha={alpha} deviates too much"
         );
      }
   }
}
