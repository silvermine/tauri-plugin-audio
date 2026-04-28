//! Best-effort ID3v2 metadata extraction for in-memory MP3 byte buffers.
//!
//! Used by [`crate::player::RodioAudioPlayer`] to enrich a playlist item's
//! metadata with embedded title / artist / artwork when the caller did not
//! supply them. Caller-supplied metadata always wins on a per-field basis;
//! ID3 only fills in gaps.
//!
//! Non-MP3 formats and MP3s without ID3v2 tags return [`None`]. The two
//! failure modes log at different levels so consumers can tell them apart:
//! `id3::ErrorKind::NoTag` is a normal expected outcome and logs at
//! `tracing::debug`, while a malformed ID3 header on otherwise-MP3-looking
//! bytes logs at `tracing::warn` because it points to a real file problem.

use std::io::Cursor;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use id3::{ErrorKind, Tag, TagLike};
use tracing::{debug, warn};

use crate::models::AudioMetadata;

/// Extracts an [`AudioMetadata`] view from the given audio bytes.
///
/// Returns `None` if the bytes don't carry a parseable ID3v2 tag (e.g. WAV,
/// FLAC, or an MP3 without ID3v2). Individual fields within the returned
/// metadata are also `None` when absent from the tag.
pub fn extract(bytes: &[u8]) -> Option<AudioMetadata> {
   let tag = match Tag::read_from2(&mut Cursor::new(bytes)) {
      Ok(tag) => tag,
      Err(e) => {
         match e.kind {
            ErrorKind::NoTag => debug!("No ID3 tag present: {e}"),
            _ => warn!("Malformed ID3 tag: {e}"),
         }
         return None;
      }
   };

   let title = tag.title().map(str::to_owned);
   let artist = tag.artist().map(str::to_owned);
   let artwork = first_picture_data_url(&tag);

   if title.is_none() && artist.is_none() && artwork.is_none() {
      return None;
   }

   Some(AudioMetadata {
      title,
      artist,
      artwork,
   })
}

/// Merges `extracted` into `existing` with caller-provided fields winning.
///
/// Used by the player to enrich `PlaylistItem.metadata` after fetch. If the
/// caller already supplied a title in their playlist, the extracted title is
/// ignored even when the ID3 tag carries a different value.
pub fn merge(existing: Option<AudioMetadata>, extracted: Option<AudioMetadata>) -> AudioMetadata {
   let existing = existing.unwrap_or_default();
   let extracted = extracted.unwrap_or_default();

   AudioMetadata {
      title: existing.title.or(extracted.title),
      artist: existing.artist.or(extracted.artist),
      artwork: existing.artwork.or(extracted.artwork),
   }
}

/// Encodes the first APIC (attached picture) frame as a `data:` URL suitable
/// for an `<img src>` or `MPMediaItemArtwork`. Prefers a "front cover"
/// picture when multiple are present.
fn first_picture_data_url(tag: &Tag) -> Option<String> {
   let mut pictures = tag.pictures();
   let first = pictures.next()?;

   // Look for a front-cover picture; otherwise fall back to whatever was first.
   let chosen = std::iter::once(first)
      .chain(tag.pictures())
      .find(|p| p.picture_type == id3::frame::PictureType::CoverFront)
      .unwrap_or(first);

   let mime = if chosen.mime_type.is_empty() {
      "image/jpeg"
   } else {
      chosen.mime_type.as_str()
   };
   let encoded = BASE64.encode(&chosen.data);

   Some(format!("data:{mime};base64,{encoded}"))
}

#[cfg(test)]
mod tests {
   use super::*;
   use id3::frame::{Picture, PictureType};
   use id3::{Tag, TagLike, Version};

   fn write_tag_to_bytes(tag: &Tag) -> Vec<u8> {
      // Minimum-viable container: write ID3v2 header + zero MP3 frames after.
      // The id3 crate writes only the tag itself; we append a dummy MP3 frame
      // header so the buffer doesn't get rejected by stricter parsers, but for
      // ID3 read tests the tag header alone is sufficient.
      let mut buf = Vec::new();
      tag.write_to(&mut buf, Version::Id3v24).unwrap();
      buf
   }

   #[test]
   fn returns_none_for_non_id3_bytes() {
      assert!(extract(b"\x00\x01\x02not an mp3").is_none());
   }

   #[test]
   fn extracts_title_and_artist() {
      let mut tag = Tag::new();

      tag.set_title("Test Title");
      tag.set_artist("Test Artist");

      let bytes = write_tag_to_bytes(&tag);
      let extracted = extract(&bytes).expect("metadata should parse");

      assert_eq!(extracted.title.as_deref(), Some("Test Title"));
      assert_eq!(extracted.artist.as_deref(), Some("Test Artist"));
      assert!(extracted.artwork.is_none());
   }

   #[test]
   fn extracts_artwork_as_data_url() {
      let mut tag = Tag::new();

      tag.add_frame(Picture {
         mime_type: "image/png".to_string(),
         picture_type: PictureType::CoverFront,
         description: String::new(),
         data: vec![0x89, 0x50, 0x4e, 0x47],
      });

      let bytes = write_tag_to_bytes(&tag);
      let extracted = extract(&bytes).expect("metadata should parse");
      let artwork = extracted.artwork.expect("artwork should be present");

      assert!(artwork.starts_with("data:image/png;base64,"));
      assert!(artwork.contains("iVBORw")); // base64 of 0x89 0x50 0x4e 0x47
   }

   #[test]
   fn returns_none_when_tag_has_no_known_fields() {
      let tag = Tag::new();

      let bytes = write_tag_to_bytes(&tag);

      assert!(extract(&bytes).is_none());
   }

   #[test]
   fn merge_keeps_existing_fields() {
      let existing = Some(AudioMetadata {
         title: Some("User Title".into()),
         artist: None,
         artwork: None,
      });
      let extracted = Some(AudioMetadata {
         title: Some("ID3 Title".into()),
         artist: Some("ID3 Artist".into()),
         artwork: Some("data:image/png;base64,abc".into()),
      });

      let merged = merge(existing, extracted);

      assert_eq!(merged.title.as_deref(), Some("User Title"));
      assert_eq!(merged.artist.as_deref(), Some("ID3 Artist"));
      assert_eq!(merged.artwork.as_deref(), Some("data:image/png;base64,abc"));
   }

   #[test]
   fn merge_with_no_extracted_returns_existing() {
      let existing = Some(AudioMetadata {
         title: Some("X".into()),
         artist: None,
         artwork: None,
      });

      let merged = merge(existing, None);

      assert_eq!(merged.title.as_deref(), Some("X"));
   }

   #[test]
   fn merge_with_neither_returns_default() {
      let merged = merge(None, None);

      assert!(merged.title.is_none());
      assert!(merged.artist.is_none());
      assert!(merged.artwork.is_none());
   }
}
