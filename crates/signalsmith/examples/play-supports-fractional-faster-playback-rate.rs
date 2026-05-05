mod common;

// Manual audible check for uninterrupted fixture playback at 1.25x speed.
fn main() -> common::ExampleResult {
   // no clicks
   common::play_fixture_at_rate(1.25)
}
