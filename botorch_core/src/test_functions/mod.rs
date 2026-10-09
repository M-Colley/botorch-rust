//! Synthetic test functions for optimization benchmarks
//! (`botorch.test_functions.synthetic`, unconstrained functions).
//!
//! A [`SyntheticTestFunction`] has BoTorch's semantics:
//! [`SyntheticTestFunction::evaluate_true`] returns the noiseless function
//! value (never negated, and an error for points outside the bounds), while
//! [`SyntheticTestFunction::forward`] adds Gaussian observation noise with
//! standard deviation `noise_std` and then negates the result if `negate` is
//! set; [`SyntheticTestFunction::optimal_value`] accounts for `negate`.
//!
//! Constants that BoTorch stores in buffers created with torch's default
//! dtype (float32) — Hartmann's `ALPHA` and `A`, Shekel's `C` — are rounded
//! to float32 exactly as in BoTorch, so that values agree to round-off.

mod synthetic;

pub use synthetic::{SyntheticTestFunction, TestFunctionKind};
