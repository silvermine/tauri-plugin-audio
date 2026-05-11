mod common;

use std::thread;
use std::time::Duration;

// Manual audible check for pausing 1.0x playback for 5 seconds and then resuming.
fn main() -> common::ExampleResult {
   let player = common::open_fixture_player(1.0, 5)?;

   player.play()?;
   common::wait_for_position(&player, 3.0)?;
   player.pause()?;
   thread::sleep(Duration::from_secs(5));
   player.play()?;

   common::wait_for_position(&player, 5.0)?;
   player.stop()?;

   Ok(())
}
