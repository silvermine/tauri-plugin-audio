use std::fs::File;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rodio::source::SeekError as RodioSeekError;
use rodio::{Sample, Source};
use symphonia::core::audio::SampleBuffer as SymphoniaSampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder as SymphoniaDecoderTrait, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{
   FormatOptions, FormatReader as SymphoniaFormatReader, SeekMode, SeekTo,
};
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::{Time, TimeBase};
use tracing::warn;

use super::http::{HttpAudioReader, RemoteSourceDescriptor};
use super::source::infer_hint_from_path;

use crate::error::{Error, Result};

type SymphoniaDecoder = Box<dyn SymphoniaDecoderTrait>;
type SymphoniaFormat = Box<dyn SymphoniaFormatReader>;

pub(crate) struct SymphoniaSource {
   format: SymphoniaFormat,
   decoder: SymphoniaDecoder,
   track_id: u32,
   channels: NonZeroU16,
   sample_rate: NonZeroU32,
   total_duration: Option<Duration>,
   time_base: TimeBase,
   sample_buffer: Option<SymphoniaSampleBuffer<f32>>,
   pending_samples: Vec<f32>,
   pending_index: usize,
   pending_seek_ts: Option<u64>,
   exhausted: bool,
}

impl SymphoniaSource {
   pub(crate) fn new_local(path: &Path, start_time: Duration) -> Result<Self> {
      let media_source = File::open(path).map_err(Error::Io)?;
      let hint = infer_hint_from_path(path);
      Self::new(Box::new(media_source), hint.as_deref(), None, start_time)
   }

   pub(crate) fn new_remote(remote: &RemoteSourceDescriptor, start_time: Duration) -> Result<Self> {
      let media_source = HttpAudioReader::new(remote.url.clone(), remote.byte_len);
      Self::new(
         Box::new(media_source),
         remote.hint.as_deref(),
         remote.mime_type.as_deref(),
         start_time,
      )
   }

   fn new(
      media_source: Box<dyn MediaSource>,
      extension_hint: Option<&str>,
      mime_type_hint: Option<&str>,
      start_time: Duration,
   ) -> Result<Self> {
      let mut hint = Hint::new();
      if let Some(extension) = extension_hint {
         hint.with_extension(extension);
      }
      if let Some(mime_type) = mime_type_hint {
         hint.mime_type(mime_type);
      }

      let stream = MediaSourceStream::new(media_source, MediaSourceStreamOptions::default());
      let format_options = FormatOptions {
         enable_gapless: true,
         ..FormatOptions::default()
      };
      let probed = symphonia::default::get_probe()
         .format(&hint, stream, &format_options, &MetadataOptions::default())
         .map_err(|error| Error::Audio(format!("Failed to probe audio source: {error}")))?;
      let format = probed.format;
      let track = format
         .default_track()
         .cloned()
         .or_else(|| format.tracks().first().cloned())
         .ok_or_else(|| Error::Audio("Audio source contained no playable tracks".into()))?;

      if track.codec_params.codec == CODEC_TYPE_NULL {
         return Err(Error::Audio("Audio track has no supported codec".into()));
      }

      let decoder = symphonia::default::get_codecs()
         .make(&track.codec_params, &DecoderOptions::default())
         .map_err(|error| Error::Audio(format!("Failed to open audio decoder: {error}")))?;
      let sample_rate = NonZeroU32::new(track.codec_params.sample_rate.unwrap_or(44_100))
         .unwrap_or(NonZeroU32::MIN);
      let channels = track
         .codec_params
         .channels
         .and_then(|value| NonZeroU16::new(value.count() as u16))
         .unwrap_or(NonZeroU16::MIN.saturating_add(1));
      let time_base = track
         .codec_params
         .time_base
         .unwrap_or_else(|| TimeBase::new(1, sample_rate.get()));
      let total_duration = track
         .codec_params
         .n_frames
         .map(|frames| Duration::from(time_base.calc_time(frames)));

      let mut source = Self {
         format,
         decoder,
         track_id: track.id,
         channels,
         sample_rate,
         total_duration,
         time_base,
         sample_buffer: None,
         pending_samples: Vec::new(),
         pending_index: 0,
         pending_seek_ts: None,
         exhausted: false,
      };

      if start_time > Duration::ZERO {
         source.seek_internal(start_time)?;
      }

      Ok(source)
   }

   fn fill_pending_samples(&mut self) -> Result<bool> {
      if self.pending_index < self.pending_samples.len() {
         return Ok(true);
      }

      self.pending_samples.clear();
      self.pending_index = 0;

      while !self.exhausted {
         let packet = match self.format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
               if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
               self.exhausted = true;
               return Ok(false);
            }
            Err(SymphoniaError::ResetRequired) => {
               self.exhausted = true;
               return Err(Error::Audio("Audio format changed unexpectedly".into()));
            }
            Err(error) => {
               self.exhausted = true;
               return Err(Error::Audio(format!(
                  "Failed to read audio packet: {error}"
               )));
            }
         };

         if packet.track_id() != self.track_id {
            continue;
         }

         let packet_ts = packet.ts();
         let packet_end_ts = packet_ts.saturating_add(packet.dur());
         let decoded = match self.decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
               self.decoder.reset();
               continue;
            }
            Err(error) => {
               self.exhausted = true;
               return Err(Error::Audio(format!(
                  "Failed to decode audio packet: {error}"
               )));
            }
         };

         let spec = *decoded.spec();
         if let Some(channels) = NonZeroU16::new(spec.channels.count() as u16) {
            self.channels = channels;
         }
         if let Some(sample_rate) = NonZeroU32::new(spec.rate) {
            self.sample_rate = sample_rate;
         }

         let mut trim_start_frames = packet.trim_start as usize;
         let trim_end_frames = packet.trim_end as usize;

         if let Some(target_ts) = self.pending_seek_ts {
            if packet_end_ts <= target_ts {
               continue;
            }

            if packet_ts < target_ts {
               let delta_duration = Duration::from(self.time_base.calc_time(target_ts - packet_ts));
               let delta_frames =
                  (delta_duration.as_secs_f64() * self.sample_rate.get() as f64).floor() as usize;
               trim_start_frames = trim_start_frames.max(delta_frames);
            }

            self.pending_seek_ts = None;
         }

         if trim_start_frames + trim_end_frames >= decoded.frames() {
            continue;
         }

         let required_capacity = decoded.frames() * spec.channels.count();
         if self
            .sample_buffer
            .as_ref()
            .map(|buffer| buffer.capacity() < required_capacity)
            .unwrap_or(true)
         {
            self.sample_buffer = Some(SymphoniaSampleBuffer::new(decoded.capacity() as u64, spec));
         }

         let Some(sample_buffer) = &mut self.sample_buffer else {
            continue;
         };

         sample_buffer.copy_interleaved_ref(decoded);
         let channel_count = spec.channels.count();
         let trim_start_samples = trim_start_frames * channel_count;
         let trim_end_samples = trim_end_frames * channel_count;
         let samples = sample_buffer.samples();
         let end_index = samples.len().saturating_sub(trim_end_samples);

         if trim_start_samples >= end_index {
            continue;
         }

         self
            .pending_samples
            .extend_from_slice(&samples[trim_start_samples..end_index]);
         return Ok(true);
      }

      Ok(false)
   }

   fn seek_internal(&mut self, position: Duration) -> Result<()> {
      let target_position = self
         .total_duration
         .map(|duration| position.min(duration))
         .unwrap_or(position);
      let seeked_to = self
         .format
         .seek(
            SeekMode::Accurate,
            SeekTo::Time {
               time: Time::from(target_position.as_secs_f64()),
               track_id: Some(self.track_id),
            },
         )
         .map_err(|error| Error::Audio(format!("Failed to seek audio source: {error}")))?;

      self.decoder.reset();
      self.pending_samples.clear();
      self.pending_index = 0;
      self.pending_seek_ts = Some(seeked_to.required_ts);
      self.exhausted = false;

      Ok(())
   }
}

impl Iterator for SymphoniaSource {
   type Item = Sample;

   fn next(&mut self) -> Option<Self::Item> {
      if self.pending_index >= self.pending_samples.len() {
         match self.fill_pending_samples() {
            Ok(true) => {}
            Ok(false) => return None,
            Err(error) => {
               warn!("Failed to stream audio via Symphonia: {error}");
               return None;
            }
         }
      }

      let sample = *self.pending_samples.get(self.pending_index)?;
      self.pending_index += 1;
      Some(sample)
   }
}

impl Source for SymphoniaSource {
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
         .seek_internal(position)
         .map_err(|error| RodioSeekError::Other(Arc::new(std::io::Error::other(error.to_string()))))
   }
}
