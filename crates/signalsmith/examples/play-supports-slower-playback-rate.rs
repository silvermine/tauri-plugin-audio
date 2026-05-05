mod common;

// Manual audible check for uninterrupted fixture playback at 0.75x speed.
fn main() -> common::ExampleResult {
   // no clicks
   common::play_fixture_at_rate(0.75)
}
