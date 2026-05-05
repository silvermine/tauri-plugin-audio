use std::error::Error;

#[allow(dead_code)]
pub(crate) mod fixture;
#[allow(dead_code)]
mod scenarios;
#[allow(dead_code)]
pub(crate) mod streaming_source;

pub type ExampleResult<T = ()> = Result<T, Box<dyn Error>>;

#[allow(unused_imports)]
pub use scenarios::{open_seeked_fixture_source, play_fixture_at_rate, SOURCE_DURATION_SECONDS};
