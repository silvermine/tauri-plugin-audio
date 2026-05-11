mod common;

// Manual audible check for uninterrupted fixture playback at 2.0x speed.
fn main() -> common::ExampleResult {
   // no clicks
   common::play_fixture_at_rate(2.0)
}
