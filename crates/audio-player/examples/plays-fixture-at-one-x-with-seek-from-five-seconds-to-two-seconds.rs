mod common;

// Manual audible check for seeking during 1.0x playback from 5 seconds back to 2 seconds.
fn main() -> common::ExampleResult {
   let player = common::open_fixture_player(1.0, 5)?;

   player.play()?;
   common::wait_for_position(&player, 5.0)?;
   player.seek(2.0)?;

   common::wait_for_position(&player, 5.0)?;
   player.stop()?;

   Ok(())
}
