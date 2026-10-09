//! # botorch_core
//!
//! A Rust re-implementation of the Gaussian-process core of
//! [BoTorch](https://botorch.org), in double precision. See the repository
//! README for the roadmap.
//!
//! ## Conventions
//!
//! * [`Matrix`] / [`Vector`] are `nalgebra` `DMatrix<f64>` / `DVector<f64>`.
//!   Point sets are `n × d` matrices whose **rows** are points.

#![warn(missing_docs)]

pub mod error;

/// Dense double-precision matrix (`nalgebra::DMatrix<f64>`); point sets are
/// stored with one point per row.
pub type Matrix = nalgebra::DMatrix<f64>;

/// Dense double-precision column vector (`nalgebra::DVector<f64>`).
pub type Vector = nalgebra::DVector<f64>;

pub use error::{BotorchError, Result};
