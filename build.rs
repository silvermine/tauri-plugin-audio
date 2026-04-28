const COMMANDS: &[&str] = &[
   "load",
   "play",
   "pause",
   "stop",
   "seek",
   "next",
   "prev",
   "jump_to",
   "set_volume",
   "set_muted",
   "set_playback_rate",
   "set_loop_mode",
   "get_state",
   "is_native",
];

fn main() {
   tauri_plugin::Builder::new(COMMANDS).build();
}
