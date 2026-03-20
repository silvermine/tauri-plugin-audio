fn main() {
   // Watch the plugin crate sources so `cargo tauri dev` rebuilds on changes.
   println!("cargo::rerun-if-changed=../../../src");
   println!("cargo::rerun-if-changed=../../../crates");
   tauri_build::build()
}
