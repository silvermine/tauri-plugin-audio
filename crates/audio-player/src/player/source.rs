use std::path::{Path, PathBuf};
use std::time::Duration;

use rodio::{Sample, Source};

use super::http::{RemoteSourceDescriptor, fetch_remote_source_descriptor};
use super::symphonia::SymphoniaSource;

use crate::error::{Error, Result};
use crate::net::reject_private_host;

#[derive(Clone)]
pub(crate) enum SourceDescriptor {
   Local { path: PathBuf },
   Remote(RemoteSourceDescriptor),
}

type BoxedSource = Box<dyn Source<Item = Sample> + Send>;

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

pub(crate) fn open_source(source: &SourceDescriptor) -> Result<BoxedSource> {
   open_source_at(source, 0.0)
}

pub(crate) fn open_source_at(source: &SourceDescriptor, position: f64) -> Result<BoxedSource> {
   match source {
      SourceDescriptor::Local { path } => Ok(Box::new(SymphoniaSource::new_local(
         path,
         Duration::from_secs_f64(position.max(0.0)),
      )?)),
      SourceDescriptor::Remote(remote) => Ok(Box::new(SymphoniaSource::new_remote(
         remote,
         Duration::from_secs_f64(position.max(0.0)),
      )?)),
   }
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
