mod common;

// Manual audible check for uninterrupted fixture playback at 1.25x speed.
fn main() -> common::ExampleResult {
   let player = common::open_fixture_player(1.25, 5)?;

   player.play()?;

   common::wait_for_position(&player, 5.0)?;
   player.stop()?;

   Ok(())
}
