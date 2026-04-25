use std::error::Error;
use std::fs::File;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::PathBuf;

use rodio::{Decoder, DeviceSinkBuilder, Player, Source, buffer::SamplesBuffer};
use signalsmith::PlaybackStream;

const SOURCE_DURATION_SECONDS: usize = 5;
const OUTPUT_BLOCK_FRAMES: usize = 512;
const FLUSH_FRAMES: usize = OUTPUT_BLOCK_FRAMES * 4;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("music.wav")
}

fn read_fixture_interleaved() -> Result<(NonZeroU16, NonZeroU32, Vec<f32>), Box<dyn Error>> {
    let file = File::open(fixture_path())?;
    let decoder = Decoder::try_from(file)?;
    let channels = decoder.channels();
    let sample_rate_hz = decoder.sample_rate();
    let samples = decoder.collect::<Vec<f32>>();

    Ok((channels, sample_rate_hz, samples))
}

fn preview_source_frames(sample_rate_hz: u32, available_frames: usize) -> usize {
    available_frames.min(sample_rate_hz as usize * SOURCE_DURATION_SECONDS)
}

fn render_playback_segment(
    stream: &mut PlaybackStream,
    input_interleaved: &[f32],
    channels: usize,
    source_frames: usize,
    playback_rate: f32,
    output: &mut Vec<f32>,
) {
    let mut cursor = 0_usize;
    let source_samples = source_frames * channels;

    assert!(
        source_samples <= input_interleaved.len(),
        "requested source segment exceeds fixture length"
    );

    while cursor < source_samples {
        let remaining_frames = source_frames - cursor / channels;
        let mut output_frames = OUTPUT_BLOCK_FRAMES
            .min(((remaining_frames as f64 / playback_rate as f64).ceil() as usize).max(1));

        while stream.input_samples_for_output(output_frames) > remaining_frames {
            output_frames -= 1;
        }

        let input_frames = stream.input_samples_for_output(output_frames);
        let next_cursor = cursor + input_frames * channels;
        assert!(
            next_cursor <= source_samples,
            "audible fixture too short for playback_rate={playback_rate}: need {next_cursor} samples"
        );

        let input_block = &input_interleaved[cursor..next_cursor];
        let mut output_block = vec![0.0_f32; output_frames * channels];
        let consumed = stream.process_interleaved(input_block, &mut output_block);

        assert_eq!(consumed, input_frames);
        output.extend_from_slice(&output_block);
        cursor = next_cursor;
    }
}

fn render_playback_rate(
    input_interleaved: &[f32],
    channels: NonZeroU16,
    sample_rate_hz: NonZeroU32,
    source_frames: usize,
    playback_rate: f32,
) -> Vec<f32> {
    let channels = usize::from(channels.get());
    let mut stream =
        PlaybackStream::with_rate(channels, sample_rate_hz.get() as f32, playback_rate);
    let estimated_output_frames = (source_frames as f64 / playback_rate as f64).ceil() as usize;
    let mut output = Vec::with_capacity((estimated_output_frames + FLUSH_FRAMES) * channels);

    assert!(channels > 0, "fixture WAV must have at least one channel");
    render_playback_segment(
        &mut stream,
        input_interleaved,
        channels,
        source_frames,
        playback_rate,
        &mut output,
    );

    let mut flush_block = vec![0.0_f32; FLUSH_FRAMES * channels];
    stream.flush_interleaved(&mut flush_block);
    output.extend_from_slice(&flush_block);

    let trim_start_frames = (stream.output_latency() * 2).min(output.len() / channels / 4);
    let trim_start = trim_start_frames * channels;

    output[trim_start..].to_vec()
}

fn render_playback_rate_with_seek_segments(
    input_interleaved: &[f32],
    channels: NonZeroU16,
    sample_rate_hz: NonZeroU32,
    playback_rate: f32,
    first_segment_end_frames: usize,
    seek_frame: usize,
    second_segment_end_frame: usize,
) -> Vec<f32> {
    let channels = usize::from(channels.get());
    let total_frames = input_interleaved.len() / channels;
    let first_segment_frames = first_segment_end_frames;
    let second_segment_frames = second_segment_end_frame - seek_frame;
    let mut stream =
        PlaybackStream::with_rate(channels, sample_rate_hz.get() as f32, playback_rate);
    let estimated_output_frames = (first_segment_frames as f64 / playback_rate as f64).ceil() as usize
        + (second_segment_frames as f64 / playback_rate as f64).ceil() as usize;
    let mut output = Vec::with_capacity((estimated_output_frames + FLUSH_FRAMES) * channels);

    assert!(channels > 0, "fixture WAV must have at least one channel");
    assert!(
        first_segment_end_frames <= total_frames,
        "first segment exceeds fixture length"
    );
    assert!(seek_frame < second_segment_end_frame, "seek frame must precede segment end");
    assert!(
        second_segment_end_frame <= total_frames,
        "second segment exceeds fixture length"
    );

    render_playback_segment(
        &mut stream,
        input_interleaved,
        channels,
        first_segment_frames,
        playback_rate,
        &mut output,
    );

    let seek_offset = seek_frame * channels;
    let seek_input = &input_interleaved[seek_offset..];
    let seek_input_frames = stream.output_seek_interleaved(seek_input);

    assert!(
        seek_input_frames <= second_segment_frames,
        "seek warmup exceeds remaining segment length"
    );

    render_playback_segment(
        &mut stream,
        &seek_input[seek_input_frames * channels..],
        channels,
        second_segment_frames - seek_input_frames,
        playback_rate,
        &mut output,
    );

    let mut flush_block = vec![0.0_f32; FLUSH_FRAMES * channels];
    stream.flush_interleaved(&mut flush_block);
    output.extend_from_slice(&flush_block);

    let trim_start_frames = (stream.output_latency() * 2).min(output.len() / channels / 4);
    let trim_start = trim_start_frames * channels;

    output[trim_start..].to_vec()
}

fn play_rendered_audio(
    channels: NonZeroU16,
    sample_rate_hz: NonZeroU32,
    samples: Vec<f32>,
) -> Result<(), Box<dyn Error>> {
    let sink = DeviceSinkBuilder::open_default_sink()?;
    let player = Player::connect_new(sink.mixer());

    player.append(SamplesBuffer::new(channels, sample_rate_hz, samples));
    player.sleep_until_end();

    Ok(())
}

fn play_fixture_at_rate(playback_rate: f32) -> Result<(), Box<dyn Error>> {
    let (channels, sample_rate_hz, source_input) = read_fixture_interleaved()?;
    let source_frames = preview_source_frames(
        sample_rate_hz.get(),
        source_input.len() / usize::from(channels.get()),
    );
    let rendered = render_playback_rate(
        &source_input,
        channels,
        sample_rate_hz,
        source_frames,
        playback_rate,
    );

    play_rendered_audio(channels, sample_rate_hz, rendered)
}

fn play_fixture_at_rate_with_seek(
    playback_rate: f32,
    first_segment_end_seconds: usize,
    seek_seconds: usize,
    second_segment_end_seconds: usize,
) -> Result<(), Box<dyn Error>> {
    let (channels, sample_rate_hz, source_input) = read_fixture_interleaved()?;
    let available_frames = source_input.len() / usize::from(channels.get());
    let source_frames = preview_source_frames(sample_rate_hz.get(), available_frames);
    let first_segment_end_frames = sample_rate_hz.get() as usize * first_segment_end_seconds;
    let seek_frame = sample_rate_hz.get() as usize * seek_seconds;
    let second_segment_end_frame = sample_rate_hz.get() as usize * second_segment_end_seconds;

    assert!(
        first_segment_end_frames <= source_frames,
        "first segment exceeds audible preview length"
    );
    assert!(seek_frame < second_segment_end_frame, "seek must land before the final segment end");
    assert!(
        second_segment_end_frame <= source_frames,
        "second segment exceeds audible preview length"
    );

    let rendered = render_playback_rate_with_seek_segments(
        &source_input,
        channels,
        sample_rate_hz,
        playback_rate,
        first_segment_end_frames,
        seek_frame,
        second_segment_end_frame,
    );

    play_rendered_audio(channels, sample_rate_hz, rendered)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_zero_point_seven_five_x() -> Result<(), Box<dyn Error>> {
    play_fixture_at_rate(0.75)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_one_point_two_five_x() -> Result<(), Box<dyn Error>> {
    play_fixture_at_rate(1.25)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_one_point_two_five_x_with_seek_from_five_seconds_to_two_seconds(
) -> Result<(), Box<dyn Error>> {
    play_fixture_at_rate_with_seek(1.25, 5, 2, 5)
}

#[test]
#[ignore = "manual audible check; plays rendered output through rodio"]
fn plays_fixture_at_two_x() -> Result<(), Box<dyn Error>> {
    play_fixture_at_rate(2.0)
}
