fn main() {
   // Oboe (used by cpal on Android) is a C++ library that requires the C++
   // standard library at runtime. The oboe-sys build script no longer emits
   // the link directive itself, so we must do it here.
   if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
      println!("cargo:rustc-link-lib=c++_shared");
   }
}
