use serde::{Serialize, ser::Serializer};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
   #[error("Invalid State: {0}")]
   InvalidState(String),

   #[error("Invalid Value: {0}")]
   InvalidValue(String),

   #[error("Audio: {0}")]
   Audio(String),

   #[error("HTTP: {0}")]
   Http(String),

   #[error(transparent)]
   Io(#[from] std::io::Error),
}

/// Serialize errors as plain strings for the Tauri IPC bridge.
/// The TypeScript layer receives these as rejected promise messages.
impl Serialize for Error {
   fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
   where
      S: Serializer,
   {
      serializer.serialize_str(self.to_string().as_ref())
   }
}
