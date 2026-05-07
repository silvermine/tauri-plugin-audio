use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;

use url::Url;

use super::source::infer_hint;

use crate::error::{Error, Result};
use crate::net::reject_private_host;

/// HTTP request timeout (connect + read combined).
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_MAX_REDIRECTS: usize = 10;

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

enum RequestError {
   Ureq(Box<ureq::Error>),
   Http(Error),
}

pub(crate) fn fetch_remote_source_descriptor(src: &str) -> Result<RemoteSourceDescriptor> {
   let (url, resp) = match descriptor_probe_request(src, true) {
      Ok(result) => result,
      Err(RequestError::Ureq(error))
         if matches!(error.as_ref(), ureq::Error::Status(_, _)) =>
      {
         descriptor_probe_request(src, false).map_err(|error| map_request_error(src, error))?
      }
      Err(error) => return Err(map_request_error(src, error)),
   };

   Ok(RemoteSourceDescriptor {
      url: url.clone(),
      byte_len: parse_byte_len(&resp),
      mime_type: resp.header("content-type").map(str::to_string),
      hint: infer_hint(&url),
   })
}

fn descriptor_probe_request(
   src: &str,
   use_range: bool,
) -> std::result::Result<(String, ureq::Response), RequestError> {
   send_request_following_redirects(src, |url| {
      let request = descriptor_http_agent()
         .get(url)
         .set("Accept-Encoding", "identity");
      let request = if use_range {
         request.set("Range", "bytes=0-0")
      } else {
         request
      };

      request.call().map_err(Box::new)
   })
}

fn map_request_error(src: &str, error: RequestError) -> Error {
   match error {
      RequestError::Ureq(error) => Error::Http(format!("Failed to fetch {src}: {error}")),
      RequestError::Http(error) => error,
   }
}

fn send_request_following_redirects<F>(
   src: &str,
   mut send_request: F,
) -> std::result::Result<(String, ureq::Response), RequestError>
where
   F: FnMut(&str) -> std::result::Result<ureq::Response, Box<ureq::Error>>,
{
   let mut current_url = src.to_string();
   let mut redirect_count = 0;

   loop {
      match send_request(&current_url) {
         Ok(response) => return Ok((current_url, response)),
         Err(error) => match *error {
            ureq::Error::Status(status, response) if is_redirect_status(status) => {
               if redirect_count >= HTTP_MAX_REDIRECTS {
                  return Ok((current_url, response));
               }

               current_url =
                  resolve_redirect_url(&current_url, &response).map_err(RequestError::Http)?;
               redirect_count += 1;
            }
            error => return Err(RequestError::Ureq(Box::new(error))),
         },
      }
   }
}

fn is_redirect_status(status: u16) -> bool {
   matches!(status, 301 | 302 | 303 | 307 | 308)
}

fn resolve_redirect_url(current_url: &str, response: &ureq::Response) -> Result<String> {
   let location = response.header("location").ok_or_else(|| {
      Error::Http(format!(
         "Failed to fetch {current_url}: redirect response missing Location header",
      ))
   })?;

   resolve_redirect_location(current_url, location)
}

fn resolve_redirect_location(current_url: &str, location: &str) -> Result<String> {
   let base_url = Url::parse(current_url)
      .map_err(|error| Error::Http(format!("Failed to fetch {current_url}: {error}")))?;
   let redirect_url = base_url.join(location).map_err(|error| {
      Error::Http(format!(
         "Failed to fetch {current_url}: invalid redirect URL {location}: {error}",
      ))
   })?;
   let redirect_url = redirect_url.to_string();

   match reject_private_host(&redirect_url) {
      Ok(()) => Ok(redirect_url),
      Err(Error::Http(message)) => {
         Err(Error::Http(format!("Failed to fetch {current_url}: {message}")))
      }
      Err(error) => Err(Error::Http(format!("Failed to fetch {current_url}: {error}"))),
   }
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

fn open_http_stream(src: &str, position: u64) -> Result<(String, HttpResponseReader, Option<u64>)> {
   let (url, resp) = send_request_following_redirects(src, |next_url| {
      let request = stream_http_agent()
         .get(next_url)
         .set("Accept-Encoding", "identity");
      let request = if position > 0 {
         request.set("Range", &format!("bytes={position}-"))
      } else {
         request
      };

      request.call().map_err(Box::new)
   })
   .map_err(|error| map_request_error(src, error))?;
   let status = resp.status();
   let byte_len = parse_byte_len(&resp);
   let mut reader = HttpResponseReader::new(resp);

   if position > 0 && status != 206 {
      skip_bytes(&mut reader, position).map_err(Error::Io)?;
   }

   Ok((url, reader, byte_len))
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
         let (url, reader, byte_len) = open_http_stream(&self.url, self.position)
            .map_err(|error| std::io::Error::other(error.to_string()))?;

         self.url = url;

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
   fn resolve_redirect_location_supports_relative_urls() {
      let redirect_url = resolve_redirect_location(
         "https://cdn.example.com/audio/path/playlist.m3u8",
         "../media/file.mp3?token=abc",
      )
      .unwrap();

      assert_eq!(
         redirect_url,
         "https://cdn.example.com/audio/media/file.mp3?token=abc",
      );
   }

   #[test]
   fn resolve_redirect_location_rejects_private_hosts() {
      let error = resolve_redirect_location(
         "https://example.com/audio.mp3",
         "http://127.0.0.1/private.mp3",
      )
      .unwrap_err();

      assert!(
         error
            .to_string()
            .contains("Requests to private/reserved address 127.0.0.1 are not allowed"),
      );
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

      let (_, mut reader, byte_len) = open_http_stream(&url, 2).unwrap();
      let mut bytes = Vec::new();
      reader.read_to_end(&mut bytes).unwrap();
      let request = request_rx.recv().unwrap();
      handle.join().unwrap();

      assert!(request.contains("Range: bytes=2-"));
      assert_eq!(byte_len, Some(6));
      assert_eq!(bytes, b"cdef");
   }

   #[test]
   fn send_request_following_redirects_returns_last_redirect_response_at_limit() {
      let start_url = "https://example.com/redirect-0/audio.mp3";
      let mut request_count = 0;

      let (url, response) = send_request_following_redirects(start_url, |next_url| {
         request_count += 1;

         let hop = next_url
            .split("/redirect-")
            .nth(1)
            .and_then(|suffix| suffix.split('/').next())
            .and_then(|hop| hop.parse::<usize>().ok())
            .unwrap();
         let response = format!(
            "HTTP/1.1 302 Found\r\nLocation: /redirect-{}/audio.mp3\r\n\r\n",
            hop + 1,
         )
         .parse::<ureq::Response>()
         .unwrap();

         Err(Box::new(ureq::Error::Status(302, response)))
      })
      .unwrap();

      assert_eq!(response.status(), 302);
      assert_eq!(url, "https://example.com/redirect-10/audio.mp3");
      assert_eq!(request_count, HTTP_MAX_REDIRECTS + 1);
   }
}
