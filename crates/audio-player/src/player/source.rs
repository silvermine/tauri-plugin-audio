use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rodio::{Decoder, Sample, Source};

use super::http::{HttpAudioReader, RemoteSourceDescriptor, fetch_remote_source_descriptor};
use super::stretch::StretchSource;

use crate::error::{Error, Result};
use crate::net::reject_private_host;

#[derive(Clone)]
pub(crate) enum SourceDescriptor {
   Local { path: PathBuf },
   Remote(RemoteSourceDescriptor),
}

pub(crate) type BoxedSource = Box<dyn Source<Item = Sample> + Send>;

pub(crate) struct OpenedSource {
   pub(crate) source: BoxedSource,
   pub(crate) duration: f64,
   pub(crate) supports_direct_seek: bool,
}

pub(crate) fn load_source_descriptor(src: &str) -> Result<SourceDescriptor> {
   if src.starts_with("http://") || src.starts_with("https://") {
      reject_private_host(src)?;
      return Ok(SourceDescriptor::Remote(fetch_remote_source_descriptor(
         src,
      )?));
   }

   if src.contains("://") || src.starts_with("data:") {
      return Err(Error::Http(format!("Unsupported URL scheme: {src}")));
   }

   Ok(SourceDescriptor::Local {
      path: PathBuf::from(src),
   })
}

pub(crate) fn open_source_at(
   source: &SourceDescriptor,
   position: f64,
   playback_rate: f64,
) -> Result<OpenedSource> {
   let supports_direct_seek = matches!(source, SourceDescriptor::Local { .. })
      && (playback_rate - 1.0).abs() <= f64::EPSILON;
   let decoded_source = open_decoded_source(source, Duration::from_secs_f64(position.max(0.0)))?;
   let duration = decoded_source
      .total_duration()
      .map(|value| value.as_secs_f64())
      .unwrap_or(0.0);
   let source = if (playback_rate - 1.0).abs() > f64::EPSILON {
      Box::new(StretchSource::new(decoded_source, playback_rate)) as BoxedSource
   } else {
      decoded_source
   };

   Ok(OpenedSource {
      source,
      duration,
      supports_direct_seek,
   })
}

fn open_decoded_source(source: &SourceDescriptor, start_time: Duration) -> Result<BoxedSource> {
   match source {
      SourceDescriptor::Local { path } => open_local_source(path, start_time),
      SourceDescriptor::Remote(remote) => open_remote_source(remote, start_time),
   }
}

fn open_local_source(path: &Path, start_time: Duration) -> Result<BoxedSource> {
   let file = File::open(path).map_err(Error::Io)?;
   let byte_len = file.metadata().map_err(Error::Io)?.len();
   let hint = infer_hint_from_path(path);
   let decoder = build_decoder(
      BufReader::new(file),
      Some(byte_len),
      hint.as_deref(),
      None,
      start_time,
   )?;

   Ok(Box::new(decoder))
}

fn open_remote_source(
   remote: &RemoteSourceDescriptor,
   start_time: Duration,
) -> Result<BoxedSource> {
   let decoder = build_decoder(
      BufReader::new(HttpAudioReader::new(remote.url.clone(), remote.byte_len)),
      remote.byte_len,
      remote.hint.as_deref(),
      remote.mime_type.as_deref(),
      start_time,
   )?;

   Ok(Box::new(decoder))
}

fn build_decoder<R>(
   reader: R,
   byte_len: Option<u64>,
   hint: Option<&str>,
   mime_type: Option<&str>,
   start_time: Duration,
) -> Result<Decoder<R>>
where
   R: Read + Seek + Send + Sync + 'static,
{
   let mut builder = Decoder::builder()
      .with_data(reader)
      .with_gapless(true)
      .with_seekable(true);

   if let Some(byte_len) = byte_len {
      builder = builder.with_byte_len(byte_len);
   }

   if let Some(hint) = hint {
      builder = builder.with_hint(hint);
   }

   if let Some(mime_type) = mime_type {
      builder = builder.with_mime_type(mime_type);
   }

   let mut decoder = builder
      .build()
      .map_err(|error| Error::Audio(format!("Failed to open audio decoder: {error}")))?;

   if start_time > Duration::ZERO {
      decoder
         .try_seek(start_time)
         .map_err(|error| Error::Audio(format!("Failed to seek audio source: {error}")))?;
   }

   Ok(decoder)
}

pub(crate) fn infer_hint(src: &str) -> Option<String> {
   let path = src.split('?').next().unwrap_or(src);
   let path = path.split('#').next().unwrap_or(path);

   infer_hint_from_path(Path::new(path))
}

pub(crate) fn infer_hint_from_path(path: &Path) -> Option<String> {
   path
      .extension()
      .and_then(|ext| ext.to_str())
      .map(|ext| ext.to_ascii_lowercase())
}
