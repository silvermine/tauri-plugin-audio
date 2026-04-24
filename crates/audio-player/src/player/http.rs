use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;

use super::source::infer_hint;

use crate::error::{Error, Result};

/// HTTP request timeout (connect + read combined).
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub(crate) struct RemoteSourceDescriptor {
   pub(crate) url: String,
   pub(crate) byte_len: Option<u64>,
   pub(crate) mime_type: Option<String>,
   pub(crate) hint: Option<String>,
}

pub(crate) struct HttpAudioReader {
   url: String,
   position: u64,
   byte_len: Option<u64>,
   reader: Option<HttpResponseReader>,
   reached_eof: bool,
}

struct HttpResponseReader {
   inner: Box<dyn Read + Send + Sync>,
}

pub(crate) fn fetch_remote_source_descriptor(src: &str) -> Result<RemoteSourceDescriptor> {
   let resp = match descriptor_probe_request(src, true) {
      Ok(resp) => resp,
      Err(error) if matches!(error.as_ref(), ureq::Error::Status(_, _)) => {
         descriptor_probe_request(src, false)
            .map_err(|e| Error::Http(format!("Failed to fetch {src}: {e}")))?
      }
      Err(error) => return Err(Error::Http(format!("Failed to fetch {src}: {error}"))),
   };

   Ok(RemoteSourceDescriptor {
      url: src.to_string(),
      byte_len: parse_byte_len(&resp),
      mime_type: resp.header("content-type").map(str::to_string),
      hint: infer_hint(src),
   })
}

fn descriptor_probe_request(
   src: &str,
   use_range: bool,
) -> std::result::Result<ureq::Response, Box<ureq::Error>> {
   let request = descriptor_http_agent()
      .get(src)
      .set("Accept-Encoding", "identity");
   let request = if use_range {
      request.set("Range", "bytes=0-0")
   } else {
      request
   };

   request.call().map_err(Box::new)
}

fn descriptor_http_agent() -> ureq::Agent {
   ureq::AgentBuilder::new()
      .timeout(HTTP_TIMEOUT)
      .redirects(0)
      .build()
}

fn stream_http_agent() -> ureq::Agent {
   ureq::AgentBuilder::new()
      .timeout_connect(HTTP_TIMEOUT)
      .timeout_read(HTTP_TIMEOUT)
      .redirects(0)
      .build()
}

fn parse_byte_len(resp: &ureq::Response) -> Option<u64> {
   resp
      .header("content-range")
      .and_then(parse_content_range_len)
      .or_else(|| {
         resp
            .header("content-length")
            .and_then(|value| value.parse::<u64>().ok())
      })
}

fn parse_content_range_len(value: &str) -> Option<u64> {
   value.rsplit('/').next()?.parse::<u64>().ok()
}

fn open_http_stream(url: &str, position: u64) -> Result<(HttpResponseReader, Option<u64>)> {
   let request = stream_http_agent()
      .get(url)
      .set("Accept-Encoding", "identity");
   let request = if position > 0 {
      request.set("Range", &format!("bytes={position}-"))
   } else {
      request
   };

   let resp = request
      .call()
      .map_err(|e| Error::Http(format!("Failed to fetch {url}: {e}")))?;
   let status = resp.status();
   let byte_len = parse_byte_len(&resp);
   let mut reader = HttpResponseReader::new(resp);

   if position > 0 && status != 206 {
      skip_bytes(&mut reader, position).map_err(Error::Io)?;
   }

   Ok((reader, byte_len))
}

fn skip_bytes<R: Read>(reader: &mut R, mut remaining: u64) -> std::io::Result<()> {
   let mut buffer = [0_u8; 8192];

   while remaining > 0 {
      let chunk_len = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
      let read = reader.read(&mut buffer[..chunk_len])?;
      if read == 0 {
         return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Unexpected EOF while skipping remote stream",
         ));
      }
      remaining -= read as u64;
   }

   Ok(())
}

impl HttpResponseReader {
   fn new(response: ureq::Response) -> Self {
      Self {
         inner: response.into_reader(),
      }
   }
}

impl Read for HttpResponseReader {
   fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
      self.inner.read(buf)
   }
}

impl HttpAudioReader {
   pub(crate) fn new(url: String, byte_len: Option<u64>) -> Self {
      Self {
         url,
         position: 0,
         byte_len,
         reader: None,
         reached_eof: false,
      }
   }

   fn ensure_reader(&mut self) -> std::io::Result<()> {
      if self.reader.is_none() && !self.reached_eof {
         let (reader, byte_len) = open_http_stream(&self.url, self.position)
            .map_err(|error| std::io::Error::other(error.to_string()))?;

         if self.byte_len.is_none() {
            self.byte_len = byte_len;
         }

         self.reader = Some(reader);
      }

      Ok(())
   }
}

impl Read for HttpAudioReader {
   fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
      self.ensure_reader()?;

      let Some(reader) = &mut self.reader else {
         self.reached_eof = true;
         return Ok(0);
      };

      let read = reader.read(buf)?;
      self.position += read as u64;

      if read == 0 {
         self.reader = None;
         self.reached_eof = true;
      }

      Ok(read)
   }
}

impl Seek for HttpAudioReader {
   fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
      let next = match position {
         SeekFrom::Start(offset) => offset as i128,
         SeekFrom::Current(offset) => self.position as i128 + offset as i128,
         SeekFrom::End(offset) => match self.byte_len {
            Some(byte_len) => byte_len as i128 + offset as i128,
            None => {
               return Err(std::io::Error::new(
                  std::io::ErrorKind::Unsupported,
                  "Cannot seek from end without a known content length",
               ));
            }
         },
      };

      if next < 0 {
         return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Cannot seek before the start of the stream",
         ));
      }

      let next = next as u64;

      if next != self.position {
         self.position = next;
         self.reader = None;
         self.reached_eof = false;
      }

      Ok(self.position)
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   use std::io::{Read, Write};
   use std::net::TcpListener;
   use std::sync::mpsc;
   use std::thread;

   fn spawn_http_server(
      responses: Vec<(String, Vec<u8>)>,
   ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
      let listener = TcpListener::bind("127.0.0.1:0").unwrap();
      let base_url = format!("http://{}", listener.local_addr().unwrap());
      let (request_tx, request_rx) = mpsc::channel();

      let handle = thread::spawn(move || {
         for (head, body) in responses {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];

            loop {
               let read = stream.read(&mut buffer).unwrap();
               if read == 0 {
                  break;
               }
               request.extend_from_slice(&buffer[..read]);
               if request.windows(4).any(|chunk| chunk == b"\r\n\r\n") {
                  break;
               }
            }

            request_tx
               .send(String::from_utf8_lossy(&request).into_owned())
               .unwrap();

            let response = format!(
               "{head}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
               body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
         }
      });

      (base_url, request_rx, handle)
   }

   #[test]
   fn fetch_remote_source_descriptor_falls_back_to_plain_request() {
      let responses = vec![
         ("HTTP/1.1 416 Range Not Satisfiable".to_string(), Vec::new()),
         (
            "HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg".to_string(),
            b"abcde".to_vec(),
         ),
      ];
      let (url, request_rx, handle) = spawn_http_server(responses);

      let descriptor = fetch_remote_source_descriptor(&url).unwrap();
      let first_request = request_rx.recv().unwrap();
      let second_request = request_rx.recv().unwrap();
      handle.join().unwrap();

      assert!(first_request.contains("Range: bytes=0-0"));
      assert!(!second_request.contains("Range:"));
      assert_eq!(descriptor.byte_len, Some(5));
      assert_eq!(descriptor.mime_type.as_deref(), Some("audio/mpeg"));
   }

   #[test]
   fn open_http_stream_skips_bytes_when_server_ignores_range() {
      let responses = vec![("HTTP/1.1 200 OK".to_string(), b"abcdef".to_vec())];
      let (url, request_rx, handle) = spawn_http_server(responses);

      let (mut reader, byte_len) = open_http_stream(&url, 2).unwrap();
      let mut bytes = Vec::new();
      reader.read_to_end(&mut bytes).unwrap();
      let request = request_rx.recv().unwrap();
      handle.join().unwrap();

      assert!(request.contains("Range: bytes=2-"));
      assert_eq!(byte_len, Some(6));
      assert_eq!(bytes, b"cdef");
   }
}
