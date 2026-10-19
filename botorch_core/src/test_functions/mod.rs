//! Test functions for optimization benchmarks: the unconstrained synthetic
//! functions of `botorch.test_functions.synthetic` and the multi-objective
//! problems of `botorch.test_functions.multi_objective` and
//! `multi_objective_multi_fidelity` ([`MultiObjectiveTestProblem`]).
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

pub mod multi_objective;
pub mod single_objective;
mod synthetic;

pub use multi_objective::{MultiObjectiveKind, MultiObjectiveTestProblem, NoiseStd};
pub use single_objective::{SingleObjectiveKind, SingleObjectiveProblem};
pub use synthetic::{SyntheticTestFunction, TestFunctionKind};
