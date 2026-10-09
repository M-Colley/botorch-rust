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
//! * GPyTorch materializes prior parameters, constraint bounds and initial
//!   values in float32 before casting the model to double precision. The
//!   BoTorch presets in this crate reproduce that rounding (see
//!   [`Prior::rounded_to_f32`]) so that results agree to ~1e-10.

#![warn(missing_docs)]

pub mod constraints;
pub mod error;
pub mod kernels;
pub mod linalg;
pub mod priors;

/// Dense double-precision matrix (`nalgebra::DMatrix<f64>`); point sets are
/// stored with one point per row.
pub type Matrix = nalgebra::DMatrix<f64>;

/// Dense double-precision column vector (`nalgebra::DVector<f64>`).
pub type Vector = nalgebra::DVector<f64>;

pub use constraints::{GreaterThan, ParameterTransform};
pub use error::{BotorchError, Result};
pub use kernels::{Kernel, KernelKind, KernelSpec, ScaleKernelSpec};
pub use linalg::{psd_safe_cholesky, CholeskyFactor};
pub use priors::Prior;
