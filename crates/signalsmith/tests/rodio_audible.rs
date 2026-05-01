use std::error::Error;
use std::f32::consts::PI;
use std::fs::File;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use rodio::stream::MixerDeviceSink;
use rodio::source::Zero;
use rodio::source::SeekError as RodioSeekError;
use rodio::{Decoder, DeviceSinkBuilder, Player, Source};
use signalsmith::PlaybackStream;

const SOURCE_DURATION_SECONDS: usize = 5;
const OUTPUT_BLOCK_FRAMES: usize = 512;
const FLUSH_FRAMES: usize = OUTPUT_BLOCK_FRAMES * 4;
const SEEK_FADE_FRAMES: usize = OUTPUT_BLOCK_FRAMES * 6;
const PLAYER_SEEK_FADE_STEPS: usize = 10;
const PLAYER_SEEK_FADE_MS: u64 = 50;

type BoxedSource = Box<dyn Source<Item = f32> + Send>;
type FixtureDecoder = (BoxedSource, NonZeroU16, NonZeroU32, Option<Duration>);

struct FixtureSource {
   input: BoxedSource,
   channels: NonZeroU16,
   sample_rate_hz: NonZeroU32,
   total_duration: Option<Duration>,
}

impl FixtureSource {
   fn open() -> Result<Self, Box<dyn Error>> {
      let (input, channels, sample_rate_hz, total_duration) = open_fixture_decoder()?;

      Ok(Self {
         input,
         channels,
         sample_rate_hz,
         total_duration,
      })
   }

   fn preview_frames(&self) -> usize {
      preview_source_frames(self.sample_rate_hz.get())
   }

   fn available_frames(&self) -> Option<usize> {
      available_source_frames(self.total_duration, self.sample_rate_hz.get())
   }

   fn available_frames_or_preview(&self) -> usize {
      self.available_frames().unwrap_or(self.preview_frames())
   }

   fn seconds_to_frames(&self, seconds: usize) -> usize {
      self.sample_rate_hz.get() as usize * seconds
   }

   fn assert_has_channels(&self) {
      assert!(
         self.channels.get() > 0,
         "fixture WAV must have at least one channel"
      );
   }

   fn assert_length(&self, required_frames: usize, label: &str) {
      assert_fixture_length(self.total_duration, self.sample_rate_hz, required_frames, label);
   }

   fn into_streaming_source(
      self,
      playback_rate: f32,
      current_segment_end_frame: usize,
   ) -> StreamingPlaybackSource {
      StreamingPlaybackSource::new(
         self.input,
         self.channels,
         self.sample_rate_hz,
         playback_rate,
         current_segment_end_frame,
      )
   }
}

fn fixture_path() -> PathBuf {
   PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("fixtures")
      .join("music.wav")
}

fn open_fixture_decoder() -> Result<FixtureDecoder, Box<dyn Error>> {
   let file = File::open(fixture_path())?;
   let decoder = Decoder::try_from(file)?;
   let channels = decoder.channels();
   let sample_rate_hz = decoder.sample_rate();
   let total_duration = decoder.total_duration();

   Ok((Box::new(decoder), channels, sample_rate_hz, total_duration))
}

fn preview_source_frames(sample_rate_hz: u32) -> usize {
   sample_rate_hz as usize * SOURCE_DURATION_SECONDS
}

fn available_source_frames(total_duration: Option<Duration>, sample_rate_hz: u32) -> Option<usize> {
   total_duration.map(|duration| (duration.as_secs_f64() * sample_rate_hz as f64).floor() as usize)
}

fn assert_fixture_length(
   total_duration: Option<Duration>,
   sample_rate_hz: NonZeroU32,
   required_frames: usize,
   label: &str,
) {
   if let Some(available_frames) = available_source_frames(total_duration, sample_rate_hz.get()) {
      assert!(
         required_frames <= available_frames,
         "{label} exceeds fixture length"
      );
   }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SeekTransition {
   Idle,
   FadingOut {
      target_frame: usize,
      remaining_frames: usize,
      total_frames: usize,
   },
   Pending {
      target_frame: usize,
   },
   FadingIn {
      remaining_frames: usize,
      total_frames: usize,
   },
}

struct StreamingPlaybackSource {
   input: BoxedSource,
   stream: PlaybackStream,
   channels: NonZeroU16,
   sample_rate_hz: NonZeroU32,
   input_buffer: Vec<f32>,
   output_buffer: Vec<f32>,
   output_index: usize,
   current_input_frame: usize,
   current_segment_end_frame: usize,
   flushed: bool,
   ended: bool,
   frame_position: usize,
   current_envelope: f32,
   seek_transition: SeekTransition,
}

impl StreamingPlaybackSource {
   fn new(
      input: BoxedSource,
      channels: NonZeroU16,
      sample_rate_hz: NonZeroU32,
      playback_rate: f32,
      current_segment_end_frame: usize,
   ) -> Self {
      let stream = PlaybackStream::with_rate(
         usize::from(channels.get()),
         sample_rate_hz.get() as f32,
         playback_rate,
      );

      Self {
         input,
         stream,
         channels,
         sample_rate_hz,
         input_buffer: Vec::new(),
         output_buffer: Vec::new(),
         output_index: 0,
         current_input_frame: 0,
         current_segment_end_frame,
         flushed: false,
         ended: false,
         frame_position: 0,
         current_envelope: 1.0,
         seek_transition: SeekTransition::Idle,
      }
   }

   fn channel_count(&self) -> usize {
      usize::from(self.channels.get())
   }

   fn clear_output_buffer(&mut self) {
      self.output_buffer.clear();
      self.output_index = 0;
   }

   fn remaining_output_frames(&self) -> usize {
      self
         .output_buffer
         .len()
         .saturating_sub(self.output_index)
         / self.channel_count()
   }

   fn read_exact_input_frames(&mut self, input_frames: usize, context: &str) {
      let input_samples = input_frames * self.channel_count();

      self.input_buffer.clear();
      self.input_buffer.reserve(input_samples);

      while self.input_buffer.len() < input_samples {
         let Some(sample) = self.input.next() else {
            panic!("audible fixture too short during {context}");
         };

         self.input_buffer.push(sample);
      }

      self.current_input_frame += input_frames;
   }

   fn seek_to_frame(&mut self, seek_frame: usize) -> Result<(), RodioSeekError> {
      self.input.try_seek(Duration::from_secs_f64(
         seek_frame as f64 / self.sample_rate_hz.get() as f64,
      ))?;

      self.current_input_frame = seek_frame;
      self.flushed = false;
      self.ended = false;
      self.clear_output_buffer();

      let seek_input_frames = self.stream.output_seek_length();
      self.read_exact_input_frames(seek_input_frames, "seek warmup");
      self.stream.output_seek_interleaved(&self.input_buffer);
      self.input_buffer.clear();

      Ok(())
   }

   fn schedule_seek_to_frame(&mut self, seek_frame: usize) -> Result<(), RodioSeekError> {
      self.ended = false;

      if self.output_index >= self.output_buffer.len() && !self.refill_output_buffer() {
         return self.apply_or_defer_seek_without_fade(seek_frame);
      }

      let fade_frames = self.remaining_output_frames().min(SEEK_FADE_FRAMES);
      if fade_frames < 2 {
         return self.apply_or_defer_seek_without_fade(seek_frame);
      }

      self.seek_transition = SeekTransition::FadingOut {
         target_frame: seek_frame,
         remaining_frames: fade_frames,
         total_frames: fade_frames,
      };

      Ok(())
   }

   fn apply_or_defer_seek_without_fade(&mut self, seek_frame: usize) -> Result<(), RodioSeekError> {
      if self.frame_position == 0 {
         self.seek_to_frame(seek_frame)?;
         self.start_seek_fade_in();
      } else {
         self.seek_transition = SeekTransition::Pending {
            target_frame: seek_frame,
         };
      }

      Ok(())
   }

   fn start_seek_fade_in(&mut self) {
      let fade_frames = SEEK_FADE_FRAMES.max(2);
      self.seek_transition = SeekTransition::FadingIn {
         remaining_frames: fade_frames,
         total_frames: fade_frames,
      };
   }

   fn begin_frame(&mut self) -> Result<(), RodioSeekError> {
      if let SeekTransition::Pending { target_frame } = self.seek_transition {
         self.seek_to_frame(target_frame)?;
         self.start_seek_fade_in();
      }

      self.current_envelope = match self.seek_transition {
         SeekTransition::FadingOut {
            remaining_frames,
            total_frames,
            ..
         } => envelope_for_step(total_frames - remaining_frames, total_frames, true),
         SeekTransition::FadingIn {
            remaining_frames,
            total_frames,
         } => envelope_for_step(total_frames - remaining_frames, total_frames, false),
         SeekTransition::Idle | SeekTransition::Pending { .. } => 1.0,
      };

      self.seek_transition = match self.seek_transition {
         SeekTransition::FadingOut {
            target_frame,
            remaining_frames,
            total_frames,
         } if remaining_frames > 1 => SeekTransition::FadingOut {
            target_frame,
            remaining_frames: remaining_frames - 1,
            total_frames,
         },
         SeekTransition::FadingOut { target_frame, .. } => SeekTransition::Pending {
            target_frame,
         },
         SeekTransition::FadingIn {
            remaining_frames,
            total_frames,
         } if remaining_frames > 1 => SeekTransition::FadingIn {
            remaining_frames: remaining_frames - 1,
            total_frames,
         },
         SeekTransition::FadingIn { .. } => SeekTransition::Idle,
         transition => transition,
      };

      Ok(())
   }

   fn flush_output(&mut self) -> bool {
      if self.flushed {
         return false;
      }

      self
         .output_buffer
         .resize(FLUSH_FRAMES * self.channel_count(), 0.0);
      self.stream.flush_interleaved(&mut self.output_buffer);
      self.flushed = true;

      !self.output_buffer.is_empty()
   }

   fn refill_output_buffer(&mut self) -> bool {
      if self.output_index < self.output_buffer.len() {
         return true;
      }

      self.clear_output_buffer();

      if self.current_input_frame >= self.current_segment_end_frame {
         return self.flush_output();
      }

      let remaining_frames = self.current_segment_end_frame - self.current_input_frame;
      let mut output_frames = OUTPUT_BLOCK_FRAMES.max(1);

      while self.stream.input_samples_for_output(output_frames) > remaining_frames {
         output_frames -= 1;
      }

      let input_frames = self.stream.input_samples_for_output(output_frames);

      self.read_exact_input_frames(input_frames, "streaming playback");
      self
         .output_buffer
         .resize(output_frames * self.channel_count(), 0.0);

      let consumed = self
         .stream
         .process_interleaved(&self.input_buffer, &mut self.output_buffer);

      assert_eq!(consumed, input_frames);
      true
   }
}

impl Iterator for StreamingPlaybackSource {
   type Item = f32;

   fn next(&mut self) -> Option<Self::Item> {
      if self.frame_position == 0 && self.begin_frame().is_err() {
         self.ended = true;
         return None;
      }

      while self.output_index >= self.output_buffer.len() {
         if !self.refill_output_buffer() {
            self.ended = true;
            return None;
         }

         if self.output_buffer.is_empty() {
            self.ended = true;
            return None;
         }
      }

      let sample = self.output_buffer.get(self.output_index).copied()?;
      self.output_index += 1;

      self.frame_position += 1;
      if self.frame_position >= self.channel_count() {
         self.frame_position = 0;
      }

      Some(sample * self.current_envelope)
   }
}

impl Source for StreamingPlaybackSource {
   fn current_span_len(&self) -> Option<usize> {
      if self.ended { Some(0) } else { None }
   }

   fn try_seek(&mut self, position: Duration) -> Result<(), RodioSeekError> {
      let seek_frame = ((position.as_secs_f64() * self.sample_rate_hz.get() as f64).floor() as usize)
         .min(self.current_segment_end_frame);

      self.schedule_seek_to_frame(seek_frame)
   }

   fn channels(&self) -> NonZeroU16 {
      self.channels
   }

   fn sample_rate(&self) -> NonZeroU32 {
      self.sample_rate_hz
   }

   fn total_duration(&self) -> Option<Duration> {
      None
   }
}

fn open_player() -> Result<(MixerDeviceSink, Player), Box<dyn Error>> {
   let sink = DeviceSinkBuilder::open_default_sink()?;
   let player = Player::connect_new(sink.mixer());

   Ok((sink, player))
}

fn open_player_with_source(
   source: StreamingPlaybackSource,
) -> Result<(MixerDeviceSink, Player), Box<dyn Error>> {
   let (sink, player) = open_player()?;

   player.append(source);

   Ok((sink, player))
}

fn play_streaming_audio(source: StreamingPlaybackSource) -> Result<(), Box<dyn Error>> {
   let (_sink, player) = open_player_with_source(source)?;
   player.sleep_until_end();

   Ok(())
}

fn wait_for_played_source_seconds(playback_rate: f32, source_seconds: f64) {
   thread::sleep(Duration::from_secs_f64(
      source_seconds / playback_rate as f64,
   ));
}

fn envelope_for_step(step: usize, total_steps: usize, fade_out: bool) -> f32 {
   let denominator = total_steps.saturating_sub(1).max(1) as f32;
   let progress = (step.min(total_steps.saturating_sub(1)) as f32) / denominator;
   if fade_out {
      0.5 * (1.0 + (PI * progress).cos())
   } else {
      0.5 * (1.0 - (PI * progress).cos())
   }
}

fn fade_player_volume(player: &Player, fade_out: bool) {
   let step_duration = Duration::from_millis(
      (PLAYER_SEEK_FADE_MS / PLAYER_SEEK_FADE_STEPS.max(1) as u64).max(1),
   );

   for step in 0..PLAYER_SEEK_FADE_STEPS {
      player.set_volume(envelope_for_step(step, PLAYER_SEEK_FADE_STEPS, fade_out));
      thread::sleep(step_duration);
   }

   player.set_volume(if fade_out { 0.0 } else { 1.0 });
}

fn seek_player_with_fade(player: &Player, position: Duration) -> Result<(), RodioSeekError> {
   fade_player_volume(player, true);
   player.try_seek(position)?;
   fade_player_volume(player, false);

   Ok(())
}

fn open_seeked_fixture_source(
   playback_rate: f32,
   seek_frame: usize,
   current_segment_end_frame: usize,
) -> Result<StreamingPlaybackSource, Box<dyn Error>> {
   let fixture = FixtureSource::open()?;
   let mut source = fixture.into_streaming_source(playback_rate, current_segment_end_frame);

   source.seek_to_frame(seek_frame)?;

   Ok(source)
}

fn play_fixture_at_rate_with_playback_seek<F>(
   playback_rate: f32,
   first_segment_end_seconds: usize,
   seek_seconds: usize,
   second_segment_end_seconds: usize,
   seek_operation: F,
) -> Result<(), Box<dyn Error>>
where
   F: FnOnce(&Player, Duration) -> Result<(), RodioSeekError>,
{
   let fixture = FixtureSource::open()?;
   let available_frames = fixture.available_frames_or_preview();
   let first_segment_end_frames = fixture.seconds_to_frames(first_segment_end_seconds);
   let second_segment_end_frame = fixture.seconds_to_frames(second_segment_end_seconds);

   fixture.assert_has_channels();
   assert!(
      first_segment_end_frames < available_frames,
      "direct seek must happen before the source ends"
   );
   assert!(
      seek_seconds < second_segment_end_seconds,
      "seek must land before the final segment end"
   );
   assert!(
      second_segment_end_frame <= available_frames,
      "second segment exceeds fixture length"
   );

   let (_sink, player) = open_player_with_source(
      fixture.into_streaming_source(playback_rate, available_frames),
   )?;

   wait_for_played_source_seconds(playback_rate, first_segment_end_seconds as f64);
   seek_operation(&player, Duration::from_secs_f64(seek_seconds as f64))?;
   wait_for_played_source_seconds(
      playback_rate,
      (second_segment_end_seconds - seek_seconds) as f64,
   );
   player.stop();

   Ok(())
}

fn play_fixture_at_rate(playback_rate: f32) -> Result<(), Box<dyn Error>> {
   let fixture = FixtureSource::open()?;
   let source_frames = fixture.preview_frames();

   fixture.assert_has_channels();
   fixture.assert_length(source_frames, "preview duration");

   play_streaming_audio(fixture.into_streaming_source(playback_rate, source_frames))
}

fn play_fixture_at_rate_with_seek(
   playback_rate: f32,
   first_segment_end_seconds: usize,
   seek_seconds: usize,
   second_segment_end_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   let fixture = FixtureSource::open()?;
   let source_frames = fixture.preview_frames();
   let first_segment_end_frames = fixture.seconds_to_frames(first_segment_end_seconds);
   let seek_frame = fixture.seconds_to_frames(seek_seconds);
   let second_segment_end_frame = fixture.seconds_to_frames(second_segment_end_seconds);

   fixture.assert_has_channels();
   fixture.assert_length(source_frames, "audible preview length");
   assert!(
      first_segment_end_frames <= source_frames,
      "first segment exceeds audible preview length"
   );
   assert!(
      seek_frame < second_segment_end_frame,
      "seek must land before the final segment end"
   );
   assert!(
      second_segment_end_frame <= source_frames,
      "second segment exceeds audible preview length"
   );

   play_streaming_audio(fixture.into_streaming_source(
      playback_rate,
      first_segment_end_frames,
   ))?;

   let second_source = open_seeked_fixture_source(
      playback_rate,
      seek_frame,
      second_segment_end_frame,
   )?;
   play_streaming_audio(second_source)
}

fn play_fixture_at_rate_with_faded_direct_seek(
   playback_rate: f32,
   first_segment_end_seconds: usize,
   seek_seconds: usize,
   second_segment_end_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate_with_playback_seek(
      playback_rate,
      first_segment_end_seconds,
      seek_seconds,
      second_segment_end_seconds,
      seek_player_with_fade,
   )
}

fn play_fixture_at_rate_with_direct_seek(
   playback_rate: f32,
   first_segment_end_seconds: usize,
   seek_seconds: usize,
   second_segment_end_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate_with_playback_seek(
      playback_rate,
      first_segment_end_seconds,
      seek_seconds,
      second_segment_end_seconds,
      |player, position| player.try_seek(position),
   )
}

fn play_fixture_at_rate_with_pause_and_resume(
   playback_rate: f32,
   played_seconds: usize,
   paused_seconds: usize,
   source_duration_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   let fixture = FixtureSource::open()?;
   let source_frames = fixture.seconds_to_frames(source_duration_seconds);

   fixture.assert_has_channels();
   assert!(
      played_seconds < source_duration_seconds,
      "pause point must land before the source ends"
   );
   fixture.assert_length(source_frames, "resume preview length");

   let (_sink, player) = open_player_with_source(
      fixture.into_streaming_source(playback_rate, source_frames),
   )?;

   wait_for_played_source_seconds(playback_rate, played_seconds as f64);
   player.pause();
   thread::sleep(Duration::from_secs(paused_seconds as u64));
   player.play();
   player.sleep_until_end();

   Ok(())
}

fn play_fixture_at_rate_with_pause_and_reopened_resume(
   playback_rate: f32,
   played_seconds: usize,
   paused_seconds: usize,
   source_duration_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   let fixture = FixtureSource::open()?;
   let source_frames = fixture.seconds_to_frames(source_duration_seconds);
   let seek_frame = fixture.seconds_to_frames(played_seconds);

   fixture.assert_has_channels();
   assert!(
      played_seconds < source_duration_seconds,
      "pause point must land before the source ends"
   );
   fixture.assert_length(source_frames, "resume preview length");

   play_streaming_audio(fixture.into_streaming_source(playback_rate, seek_frame))?;

   thread::sleep(Duration::from_secs(paused_seconds as u64));

   let second_source = open_seeked_fixture_source(playback_rate, seek_frame, source_frames)?;
   play_streaming_audio(second_source)
}

fn play_fixture_at_rate_with_silence_and_resume(
   playback_rate: f32,
   played_seconds: usize,
   silence_seconds: usize,
   source_duration_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   let fixture = FixtureSource::open()?;
   let channels = fixture.channels;
   let sample_rate_hz = fixture.sample_rate_hz;
   let source_frames = fixture.seconds_to_frames(source_duration_seconds);

   fixture.assert_has_channels();
   assert!(
      played_seconds < source_duration_seconds,
      "resume point must land before the source ends"
   );
   fixture.assert_length(source_frames, "resume preview length");

   let (sink, player) = open_player_with_source(
      fixture.into_streaming_source(playback_rate, source_frames),
   )?;

   wait_for_played_source_seconds(playback_rate, played_seconds as f64);
   player.pause();

   let silence_player = Player::connect_new(sink.mixer());
   silence_player.append(
      Zero::new(channels, sample_rate_hz)
         .take_duration(Duration::from_secs(silence_seconds as u64)),
   );
   silence_player.sleep_until_end();

   player.play();
   player.sleep_until_end();

   Ok(())
}

fn play_fixture_at_rate_with_source_fade_only_reopened_resume(
   playback_rate: f32,
   played_seconds: usize,
   paused_seconds: usize,
   source_duration_seconds: usize,
) -> Result<(), Box<dyn Error>> {
   let fixture = FixtureSource::open()?;
   let source_frames = fixture.seconds_to_frames(source_duration_seconds);
   let seek_frame = fixture.seconds_to_frames(played_seconds);

   fixture.assert_has_channels();
   assert!(
      played_seconds < source_duration_seconds,
      "pause point must land before the source ends"
   );
   fixture.assert_length(source_frames, "resume preview length");

   let (_sink, player) = open_player_with_source(
      fixture.into_streaming_source(playback_rate, source_frames),
   )?;

   wait_for_played_source_seconds(playback_rate, played_seconds as f64);
   player.stop();
   thread::sleep(Duration::from_secs(paused_seconds as u64));

   let mut resumed_source = open_seeked_fixture_source(playback_rate, seek_frame, source_frames)?;
   resumed_source.start_seek_fade_in();

   let (_resume_sink, resume_player) = open_player()?;
   resume_player.append(resumed_source);
   resume_player.sleep_until_end();

   Ok(())
}

// Play - no clicks
#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn play_supports_slower_playback_rate() -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate(0.75)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn play_supports_fractional_faster_playback_rate() -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate(1.25)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn play_supports_significantly_faster_playback_rate() -> Result<(), Box<dyn Error>> {
   play_fixture_at_rate(2.0)
}

// Seek
#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn seek_supports_faded_direct_seek_during_playback()
-> Result<(), Box<dyn Error>> {
   // rare clicks
   play_fixture_at_rate_with_faded_direct_seek(1.25, 4, 2, SOURCE_DURATION_SECONDS)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn seek_supports_reopened_source_seek()
-> Result<(), Box<dyn Error>> {
   // no click
   play_fixture_at_rate_with_seek(1.25, 4, 2, SOURCE_DURATION_SECONDS)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn seek_supports_direct_seek_during_playback()
-> Result<(), Box<dyn Error>> {
   // clicks
   play_fixture_at_rate_with_direct_seek(1.25, 4, 2, SOURCE_DURATION_SECONDS)
}

// Resume
#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn resume_continues_after_pause()
-> Result<(), Box<dyn Error>> {
   // clicks
   play_fixture_at_rate_with_pause_and_resume(1.25, 3, 5, SOURCE_DURATION_SECONDS)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn resume_continues_after_reopening_source()
-> Result<(), Box<dyn Error>> {
   // clicks
   play_fixture_at_rate_with_pause_and_reopened_resume(1.25, 3, 5, SOURCE_DURATION_SECONDS)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn resume_continues_after_silence_gap()
-> Result<(), Box<dyn Error>> {
   // sometimes clicks
   play_fixture_at_rate_with_silence_and_resume(1.25, 3, 5, SOURCE_DURATION_SECONDS)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn resume_with_source_fade_in_continues_after_pause()
-> Result<(), Box<dyn Error>> {
   // no clicks
   play_fixture_at_rate_with_source_fade_only_reopened_resume(1.25, 3, 5, SOURCE_DURATION_SECONDS)
}
