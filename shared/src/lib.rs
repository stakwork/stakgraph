pub mod error;
#[cfg(feature = "codecov")]
pub mod codecov;

pub mod coverage;
pub use error::{Context, Error, Result};
