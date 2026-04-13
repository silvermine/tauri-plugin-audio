use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rodio::source::SeekError as RodioSeekError;
use rodio::stream::{DeviceSinkBuilder, MixerDeviceSink};
use rodio::{Player, Sample, Source};
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

use crate::error::{Error, Result};
use crate::models::{AudioActionResponse, AudioMetadata, PlaybackStatus, PlayerState, TimeUpdate};
use crate::net::reject_private_host;
use crate::{OnChanged, OnTimeUpdate, transitions};

/// HTTP request timeout (connect + read combined).
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Audio player backed by Rodio for cross-platform desktop playback.
///
/// Manages audio output, a playback monitor for time updates
/// and end-of-track detection, and a state machine matching the plugin's
/// [`PlaybackStatus`] model.
pub struct RodioAudioPlayer {
   inner: Arc<Mutex<Inner>>,
   output_sink: MixerDeviceSink,
   on_changed: OnChanged,
   on_time_update: OnTimeUpdate,
}

struct Inner {
   state: PlayerState,
   playback: Option<PlaybackContext>,
   monitor_stop: Arc<AtomicBool>,
   load_generation: u64,
}

struct PlaybackContext {
   sink: Player,
   source: SourceDescriptor,
   duration: f64,
   position_offset: f64,
}

#[derive(Clone)]
enum SourceDescriptor {
   Local { path: PathBuf },
   Remote(RemoteSourceDescriptor),
}

#[derive(Clone)]
struct RemoteSourceDescriptor {
   url: String,
   byte_len: Option<u64>,
   mime_type: Option<String>,
   hint: Option<String>,
}

struct HttpAudioReader {
   url: String,
   position: u64,
   byte_len: Option<u64>,
   reader: Option<HttpResponseReader>,
   reached_eof: bool,
}

struct HttpResponseReader {
   inner: Mutex<Box<dyn Read + Send>>,
}

type BoxedSource = Box<dyn Source<Item = Sample> + Send>;
type SymphoniaDecoder = Box<dyn SymphoniaDecoderTrait>;
type SymphoniaFormat = Box<dyn SymphoniaFormatReader>;

struct SymphoniaSource {
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

impl RodioAudioPlayer {
   /// Creates a new Rodio-backed audio player.
   ///
   /// Opens the default audio output device. Returns an error
   /// if no audio device is available.
   pub fn new(on_changed: OnChanged, on_time_update: OnTimeUpdate) -> Result<Self> {
      let mut output_sink = open_audio_output()?;
      output_sink.log_on_drop(false);

      Ok(Self {
         inner: Arc::new(Mutex::new(Inner {
            state: PlayerState::default(),
            playback: None,
            monitor_stop: Arc::new(AtomicBool::new(true)),
            load_generation: 0,
         })),
         output_sink,
         on_changed,
         on_time_update,
      })
   }

   /// Stops the monitor thread by setting the flag.
   fn stop_monitor(inner: &Inner) {
      inner.monitor_stop.store(true, Ordering::Relaxed);
   }

   /// Spawns a new monitor thread for time updates and end-of-track detection.
   ///
   /// The old monitor thread may briefly overlap (up to 250ms) until it
   /// observes the stop flag on its next poll. This is harmless — any
   /// duplicate time updates are benign, and the state is already updated
   /// under the mutex before the new monitor starts, so the old one cannot
   /// trigger a spurious Ended transition.
   fn start_monitor(&self, inner: &mut Inner) {
      let stop = Arc::new(AtomicBool::new(false));
      inner.monitor_stop = stop.clone();

      let inner_arc = Arc::clone(&self.inner);
      let on_changed = Arc::clone(&self.on_changed);
      let on_time_update = Arc::clone(&self.on_time_update);

      if let Err(e) = std::thread::Builder::new()
         .name("audio-monitor".into())
         .spawn(move || {
            monitor_loop(stop, inner_arc, on_changed, on_time_update);
         })
      {
         warn!("Failed to spawn audio monitor thread: {e}");
      }
   }

   pub fn get_state(&self) -> PlayerState {
      lock_inner(&self.inner).state.clone()
   }

   pub fn prepare(
      &self,
      src: &str,
      metadata: Option<AudioMetadata>,
   ) -> Result<AudioActionResponse> {
      let meta = metadata.unwrap_or_default();

      let load_generation = {
         let mut inner = lock_inner(&self.inner);
         transitions::begin_load(&mut inner.state, src, &meta)?;
         inner.load_generation = inner.load_generation.wrapping_add(1);
         let load_generation = inner.load_generation;
         let snapshot = inner.state.clone();
         drop(inner);
         (self.on_changed)(&snapshot);
         load_generation
      };

      let result = self.prepare_inner(src, &meta, load_generation);

      match result {
         Ok(snapshot) => {
            (self.on_changed)(&snapshot);
            Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Ready))
         }
         Err(e) => {
            let mut inner = lock_inner(&self.inner);

            if inner.load_generation != load_generation {
               return Err(Error::InvalidState("Prepare request was canceled".into()));
            }

            transitions::error(&mut inner.state, e.to_string());
            let snapshot = inner.state.clone();
            drop(inner);
            (self.on_changed)(&snapshot);
            Err(e)
         }
      }
   }

   /// Inner prepare logic that may fail. Separated so `prepare()` can catch errors
   /// and transition to the Error state before propagating.
   fn prepare_inner(
      &self,
      src: &str,
      meta: &AudioMetadata,
      load_generation: u64,
   ) -> Result<PlayerState> {
      let descriptor = load_source_descriptor(src)?;
      let source = open_source(&descriptor)?;
      let duration = source
         .total_duration()
         .map(|d| d.as_secs_f64())
         .unwrap_or(0.0);

      // Create a new sink, append the decoded source, and pause immediately
      // so playback waits for an explicit play() call.
      let sink = Player::connect_new(self.output_sink.mixer());
      sink.pause();
      sink.append(source);

      let mut inner = lock_inner(&self.inner);

      if inner.load_generation != load_generation {
         return Err(Error::InvalidState("Prepare request was canceled".into()));
      }

      transitions::prepare(&mut inner.state, src, meta, duration)?;

      Self::stop_monitor(&inner);

      sink.set_volume(effective_volume(&inner.state));
      sink.set_speed(inner.state.playback_rate as f32);

      inner.playback = Some(PlaybackContext {
         sink,
         source: descriptor,
         duration,
         position_offset: 0.0,
      });

      Ok(inner.state.clone())
   }

   pub fn play(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         let is_ended = inner.state.status == PlaybackStatus::Ended;
         let mut replayed_from_start = false;

         // Re-append source for replay from Ended before the transition
         // mutates status.
         if is_ended
            && let Some(ctx) = &mut inner.playback
            && ctx.sink.empty()
         {
            let source = open_source(&ctx.source)?;
            ctx.sink.append(source);
            ctx.position_offset = 0.0;
            replayed_from_start = true;
         }

         transitions::play(&mut inner.state)?;

         if replayed_from_start {
            inner.state.current_time = 0.0;
         }

         if let Some(ctx) = &inner.playback {
            ctx.sink.play();
         }

         self.start_monitor(&mut inner);
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Playing))
   }

   pub fn pause(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);

         transitions::pause(&mut inner.state)?;

         if let Some(ctx) = &inner.playback {
            ctx.sink.pause();
         }

         Self::stop_monitor(&inner);
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Paused))
   }

   pub fn stop(&self) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);

         transitions::stop(&mut inner.state)?;
         inner.load_generation = inner.load_generation.wrapping_add(1);

         Self::stop_monitor(&inner);

         if let Some(ctx) = inner.playback.take() {
            ctx.sink.stop();
         }

         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, PlaybackStatus::Idle))
   }

   pub fn seek(&self, position: f64) -> Result<AudioActionResponse> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         let was_ended = inner.state.status == PlaybackStatus::Ended;
         let previous_time = inner.state.current_time;

         transitions::seek(&mut inner.state, position)?;

         if let Some((source_descriptor, duration)) = inner
            .playback
            .as_ref()
            .map(|ctx| (ctx.source.clone(), ctx.duration))
         {
            self.seek_playback(
               &mut inner,
               source_descriptor,
               duration,
               was_ended,
               previous_time,
            )?;
         }

         inner.state.clone()
      };

      let expected = snapshot.status;
      (self.on_changed)(&snapshot);
      Ok(AudioActionResponse::new(snapshot, expected))
   }

   fn seek_playback(
      &self,
      inner: &mut Inner,
      source_descriptor: SourceDescriptor,
      duration: f64,
      was_ended: bool,
      previous_time: f64,
   ) -> Result<()> {
      match &source_descriptor {
         SourceDescriptor::Remote(_) => {
            self.seek_remote_playback(inner, source_descriptor, duration, previous_time)
         }
         SourceDescriptor::Local { .. } => {
            Self::seek_local_playback(inner, &source_descriptor, was_ended, previous_time)
         }
      }
   }

   fn seek_remote_playback(
      &self,
      inner: &mut Inner,
      source_descriptor: SourceDescriptor,
      duration: f64,
      previous_time: f64,
   ) -> Result<()> {
      let was_playing = inner.state.status == PlaybackStatus::Playing;
      let current_volume = effective_volume(&inner.state);
      let target_time = inner.state.current_time;

      if let Some(ctx) = &inner.playback {
         ctx.sink.set_volume(0.0);
         ctx.sink.pause();
      }

      let source = match open_source_at(&source_descriptor, target_time) {
         Ok(source) => source,
         Err(error) => {
            if let Some(ctx) = &inner.playback {
               ctx.sink.set_volume(current_volume);
               if was_playing {
                  ctx.sink.play();
               }
            }
            inner.state.current_time = previous_time;
            return Err(error);
         }
      };

      let sink = Player::connect_new(self.output_sink.mixer());
      sink.pause();
      sink.append(source);
      sink.set_volume(current_volume);
      sink.set_speed(inner.state.playback_rate as f32);

      if let Some(previous_playback) = inner.playback.replace(PlaybackContext {
         sink,
         source: source_descriptor,
         duration,
         position_offset: target_time,
      }) {
         previous_playback.sink.stop();
      }

      if was_playing && let Some(ctx) = &inner.playback {
         ctx.sink.play();
      }

      Ok(())
   }

   fn seek_local_playback(
      inner: &mut Inner,
      source_descriptor: &SourceDescriptor,
      was_ended: bool,
      previous_time: f64,
   ) -> Result<()> {
      let Some(ctx) = &inner.playback else {
         unreachable!("Playback context disappeared during local seek");
      };
      let mut reopened_source = false;

      if was_ended && ctx.sink.empty() {
         let source = match open_source(source_descriptor) {
            Ok(source) => source,
            Err(error) => {
               inner.state.current_time = previous_time;
               return Err(error);
            }
         };
         ctx.sink.append(source);
         ctx.sink.pause();
         reopened_source = true;
      }

      if let Err(e) = ctx
         .sink
         .try_seek(Duration::from_secs_f64(inner.state.current_time))
      {
         if reopened_source {
            ctx.sink.stop();
         }
         inner.state.current_time = previous_time;
         return Err(Error::Audio(format!("Failed to seek audio: {e}")));
      }

      Ok(())
   }

   pub fn set_volume(&self, level: f64) -> Result<PlayerState> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_volume(&mut inner.state, level)?;
         if let Some(ctx) = &inner.playback {
            ctx.sink.set_volume(effective_volume(&inner.state));
         }
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(snapshot)
   }

   pub fn set_muted(&self, muted: bool) -> PlayerState {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_muted(&mut inner.state, muted);
         if let Some(ctx) = &inner.playback {
            ctx.sink.set_volume(effective_volume(&inner.state));
         }
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      snapshot
   }

   pub fn set_playback_rate(&self, rate: f64) -> Result<PlayerState> {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_playback_rate(&mut inner.state, rate)?;
         if let Some(ctx) = &inner.playback {
            ctx.sink.set_speed(inner.state.playback_rate as f32);
         }
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      Ok(snapshot)
   }

   pub fn set_loop(&self, looping: bool) -> PlayerState {
      let snapshot = {
         let mut inner = lock_inner(&self.inner);
         transitions::set_loop(&mut inner.state, looping);
         inner.state.clone()
      };

      (self.on_changed)(&snapshot);
      snapshot
   }
}

// ---------------------------------------------------------------------------
// Audio output
// ---------------------------------------------------------------------------

/// Opens the default audio output device for playback.
fn open_audio_output() -> Result<MixerDeviceSink> {
   DeviceSinkBuilder::open_default_sink()
      .map_err(|e| Error::Audio(format!("Failed to open audio device: {e}")))
}

// ---------------------------------------------------------------------------
// Playback monitor
// ---------------------------------------------------------------------------

/// Polls the sink every 250ms for position updates and end-of-track detection.
fn monitor_loop(
   stop: Arc<AtomicBool>,
   inner: Arc<Mutex<Inner>>,
   on_changed: OnChanged,
   on_time_update: OnTimeUpdate,
) {
   loop {
      std::thread::sleep(Duration::from_millis(250));

      if stop.load(Ordering::Relaxed) {
         break;
      }

      let mut guard = lock_inner(&inner);

      let (pos, duration, is_empty) = match &guard.playback {
         Some(ctx) => {
            let pos =
               ctx.position_offset + (ctx.sink.get_pos().as_secs_f64() * guard.state.playback_rate);
            (pos, ctx.duration, ctx.sink.empty())
         }
         None => break,
      };

      if is_empty {
         if guard.state.looping {
            // Re-append source for seamless (best-effort) loop.
            if let Some(ctx) = &mut guard.playback {
               match open_source(&ctx.source) {
                  Ok(source) => {
                     ctx.sink.append(source);
                     ctx.position_offset = 0.0;
                  }
                  Err(e) => warn!("Failed to reopen loop source: {e}"),
               }
            }
            guard.state.current_time = 0.0;
            drop(guard);
            on_time_update(&TimeUpdate {
               current_time: 0.0,
               duration,
            });
         } else {
            guard.state.status = PlaybackStatus::Ended;
            guard.state.current_time = if duration > 0.0 { duration } else { pos };
            let snapshot = guard.state.clone();
            drop(guard);
            on_changed(&snapshot);
            break;
         }
      } else {
         guard.state.current_time = pos;
         drop(guard);
         on_time_update(&TimeUpdate {
            current_time: pos,
            duration,
         });
      }
   }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Acquires the mutex, recovering from poisoning instead of panicking.
///
/// A poisoned mutex means a thread panicked while holding the lock. The inner
/// data may be in an inconsistent state, but for an audio player the worst case
/// is a glitched playback state — far better than crashing the host application.
fn lock_inner(mutex: &Mutex<Inner>) -> MutexGuard<'_, Inner> {
   mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Resolves the effective sink volume, accounting for the mute flag.
fn effective_volume(state: &PlayerState) -> f32 {
   if state.muted {
      0.0
   } else {
      state.volume as f32
   }
}

fn load_source_descriptor(src: &str) -> Result<SourceDescriptor> {
   if src.starts_with("http://") || src.starts_with("https://") {
      reject_private_host(src)?;
      Ok(SourceDescriptor::Remote(fetch_remote_source_descriptor(
         src,
      )?))
   } else {
      if src.contains("://") || src.starts_with("data:") {
         return Err(Error::Http(format!("Unsupported URL scheme: {src}")));
      }

      Ok(SourceDescriptor::Local {
         path: PathBuf::from(src),
      })
   }
}

fn open_source(source: &SourceDescriptor) -> Result<BoxedSource> {
   open_source_at(source, 0.0)
}

fn open_source_at(source: &SourceDescriptor, position: f64) -> Result<BoxedSource> {
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

fn fetch_remote_source_descriptor(src: &str) -> Result<RemoteSourceDescriptor> {
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

fn infer_hint(src: &str) -> Option<String> {
   let path = src.split('?').next().unwrap_or(src);
   let path = path.split('#').next().unwrap_or(path);

   infer_hint_from_path(Path::new(path))
}

fn infer_hint_from_path(path: &Path) -> Option<String> {
   path
      .extension()
      .and_then(|ext| ext.to_str())
      .map(|ext| ext.to_ascii_lowercase())
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
         inner: Mutex::new(Box::new(response.into_reader())),
      }
   }
}

impl Read for HttpResponseReader {
   fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
      self
         .inner
         .lock()
         .unwrap_or_else(|e| e.into_inner())
         .read(buf)
   }
}

impl HttpAudioReader {
   fn new(url: String, byte_len: Option<u64>) -> Self {
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
         return Ok(0);
      };

      let read = reader.read(buf)?;
      if read == 0 {
         self.reader = None;
         self.reached_eof = true;
      } else {
         self.position += read as u64;
      }

      Ok(read)
   }
}

impl MediaSource for HttpAudioReader {
   fn is_seekable(&self) -> bool {
      true
   }

   fn byte_len(&self) -> Option<u64> {
      self.byte_len
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

      self.position = next as u64;
      self.reader = None;
      self.reached_eof = false;
      Ok(self.position)
   }
}

impl SymphoniaSource {
   fn new_local(path: &Path, start_time: Duration) -> Result<Self> {
      let media_source = File::open(path).map_err(Error::Io)?;
      let hint = infer_hint_from_path(path);
      Self::new(Box::new(media_source), hint.as_deref(), None, start_time)
   }

   fn new_remote(remote: &RemoteSourceDescriptor, start_time: Duration) -> Result<Self> {
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
