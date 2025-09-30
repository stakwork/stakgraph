#[cfg(feature = "codecov")]
pub mod codecov;
pub mod error;

pub use error::{Context, Error, Result};
