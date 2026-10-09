//! # botorch_core
//!
//! A Rust re-implementation of the Gaussian-process core of
//! [BoTorch](https://botorch.org): `SingleTaskGP` with BoTorch's default
//! priors, constraints and transforms, the exact marginal log likelihood with
//! analytic gradients, and `fit_gpytorch_mll` (L-BFGS-B). All computations are
//! in double precision and are validated against BoTorch itself (see
//! `scripts/generate_fixtures.py` and `tests/parity.rs`).
//!
//! ## Conventions
//!
//! * [`Matrix`] / [`Vector`] are `nalgebra` `DMatrix<f64>` / `DVector<f64>`.
//!   Point sets are `n × d` matrices whose **rows** are points.
//! * Inputs passed to a model are in the *raw* (original) input space; the
//!   model applies its input transform. Posteriors are reported in the
//!   *original* outcome space (un-standardized), like BoTorch's `posterior`.
//! * Hyperparameters are stored in BoTorch's *raw* parameterization. With the
//!   `SingleTaskGP` defaults the constraints are not enforced through a
//!   transform, so raw values equal actual values and the bounds are imposed
//!   by the optimizer.
//! * GPyTorch materializes prior parameters, constraint bounds and initial
//!   values in float32 before casting the model to double precision. The
//!   BoTorch presets in this crate reproduce that rounding (see
//!   [`Prior::rounded_to_f32`]) so that results agree to ~1e-10.
//!
//! ## API overview
//!
//! | BoTorch / GPyTorch | botorch_core |
//! |---|---|
//! | `SingleTaskGP(train_X, train_Y, train_Yvar, covar_module, likelihood, outcome_transform, input_transform)` | [`SingleTaskGP::new`]`(train_x, train_y, `[`SingleTaskGPOptions`]`)` |
//! | `model.posterior(X, observation_noise)` | [`SingleTaskGP::posterior`]`(&x, observation_noise) -> `[`Posterior`]` { mean, covariance }` |
//! | mean/variance and their gradients for `q = 1` | [`SingleTaskGP::posterior_q1_with_grad`]`(&x) -> (mean, variance, dmean_dx, dvariance_dx)` |
//! | joint mean/covariance and their gradients | [`SingleTaskGP::joint_posterior_with_grad`]`(&x) -> `[`JointPosteriorGrad`] (index `j = point·d + dim`) |
//! | `model.named_parameters()` | [`SingleTaskGP::hyperparameters`], [`SingleTaskGP::set_hyperparameters`], [`SingleTaskGP::raw_parameters`], [`SingleTaskGP::parameter_names`] |
//! | `model.train_inputs`, `model.train_targets` | [`SingleTaskGP::train_inputs`], [`SingleTaskGP::train_targets`] (original space), [`SingleTaskGP::transformed_train_inputs`], [`SingleTaskGP::transformed_train_targets`] |
//! | `model.likelihood.noise` | [`SingleTaskGP::likelihood_noise`] |
//! | `ExactMarginalLogLikelihood(likelihood, model)` | [`ExactMarginalLogLikelihood::new`]`(&model)`, `.value()`, `.value_and_grad(&raw_params)` |
//! | `fit_gpytorch_mll(mll)` | [`fit_gpytorch_mll`]`(&mut model, `[`FitOptions::default()`](FitOptions)`) -> `[`FitResult`] |
//! | `get_covar_module_with_dim_scaled_prior`, `get_matern_kernel_with_gamma_prior` | [`kernels::get_covar_module_with_dim_scaled_prior`], [`kernels::get_matern_kernel_with_gamma_prior`] |
//! | `get_gaussian_likelihood_with_lognormal_prior`, `..._with_gamma_prior` | [`get_gaussian_likelihood_with_lognormal_prior`], [`get_gaussian_likelihood_with_gamma_prior`] |
//! | `Normalize`, `Standardize` | [`Normalize`], [`Standardize`] |
//! | `LogNormalPrior`, `GammaPrior` | [`Prior`] |
//! | `GreaterThan`, `Positive` | [`GreaterThan`] |
//! | `scipy.optimize.minimize(method="L-BFGS-B")` | [`lbfgsb::minimize`]`(fun, &x0, &lower, &upper, &`[`lbfgsb::LbfgsbOptions`]`)` |
//! | `linear_operator.utils.cholesky.psd_safe_cholesky` | [`psd_safe_cholesky`] |
//!
//! ## Example
//!
//! ```
//! use botorch_core::{fit_gpytorch_mll, FitOptions, Matrix, Normalize, SingleTaskGP, SingleTaskGPOptions};
//!
//! let train_x = Matrix::from_fn(12, 2, |i, j| ((i * 7 + j * 3) % 12) as f64 / 11.0);
//! let train_y = Matrix::from_fn(12, 1, |i, _| (6.0 * train_x[(i, 0)]).sin() + train_x[(i, 1)]);
//! let options = SingleTaskGPOptions {
//!     input_transform: Some(Normalize::new(2)),
//!     ..Default::default()
//! };
//! let mut model = SingleTaskGP::new(train_x, train_y, options)?;
//! let fit = fit_gpytorch_mll(&mut model, FitOptions::default())?;
//! assert!(fit.loss.is_finite());
//!
//! let x = Matrix::from_row_slice(2, 2, &[0.2, 0.4, 0.9, 0.1]);
//! let post = model.posterior(&x, false)?;
//! assert_eq!(post.covariance.shape(), (2, 2));
//! let (_mean, variance, dmean, _dvar) = model.posterior_q1_with_grad(&[0.2, 0.4])?;
//! assert!(variance > 0.0 && dmean.len() == 2);
//! # Ok::<(), botorch_core::BotorchError>(())
//! ```
//!
//! ## Validation
//!
//! `scripts/generate_fixtures.py` builds 17 BoTorch models (d ∈ {1, 2, 5},
//! n ∈ {1, 3, 5, 10, 20}, RBF / Matérn-5/2, `ScaleKernel`, fixed noise,
//! `Normalize` with learned or explicit bounds, with and without
//! `Standardize`, legacy Gamma priors, degenerate data) and records reference
//! values. `tests/parity.rs` requires agreement to a relative 1e-10 for
//! kernels and their derivatives, transforms, MLL values and gradients,
//! posterior means and covariances (with and without observation noise) and
//! their input gradients. The observed errors are at most 6e-14, except
//! 3e-12 for an MLL whose covariance has condition number ~3e6. Fitting from
//! BoTorch's default initialization reaches BoTorch's final loss (worst excess
//! 3.4e-7, typically ~1e-15) with the same number of L-BFGS-B iterations in 16
//! of 17 problems. `tests/gradients.rs` checks every analytic gradient against
//! central finite differences.
//!
//! ## Scope and known differences (phase 1)
//!
//! * Single-output `SingleTaskGP` only (an error is returned for `m > 1`);
//!   RBF and Matérn-5/2 ARD kernels with optional `ScaleKernel`; LogNormal and
//!   Gamma priors; homoskedastic and fixed-noise Gaussian likelihoods;
//!   `Normalize` (all dimensions, `center = 0.5`) and `Standardize`.
//! * `SingleTaskGPOptions::use_scale_kernel` wraps the default kernel in a
//!   `ScaleKernel` with BoTorch's legacy `GammaPrior(2.0, 0.15)` outputscale
//!   prior; use [`kernels::get_matern_kernel_with_gamma_prior`] for BoTorch's
//!   complete legacy kernel.
//! * BoTorch's input-scaling *warnings* are not emitted; NaN inputs and
//!   negative noise variances are errors, as in BoTorch.
//! * `fit_gpytorch_mll` retries failed attempts from prior samples like
//!   BoTorch, but with its own seeded random number generator.
//! * Acquisition functions and `optimize_acqf` follow in phase 2, Python
//!   bindings in phase 3.

#![warn(missing_docs)]

pub mod acquisition;
pub mod constraints;
pub mod error;
pub mod fit;
pub mod kernels;
pub mod lbfgsb;
pub mod linalg;
pub mod mll;
pub mod models;
pub mod posterior;
pub mod priors;
pub mod sampling;
pub mod transforms;
pub mod utils;

/// Dense double-precision matrix (`nalgebra::DMatrix<f64>`); point sets are
/// stored with one point per row.
pub type Matrix = nalgebra::DMatrix<f64>;

/// Dense double-precision column vector (`nalgebra::DVector<f64>`).
pub type Vector = nalgebra::DVector<f64>;

pub use acquisition::AcquisitionFunction;
pub use constraints::{GreaterThan, ParameterTransform};
pub use error::{BotorchError, Result};
pub use fit::{fit_gpytorch_mll, FitOptions, FitResult, FitStatus};
pub use kernels::{Kernel, KernelKind, KernelSpec, ScaleKernelSpec};
pub use linalg::{psd_safe_cholesky, CholeskyFactor};
pub use mll::ExactMarginalLogLikelihood;
pub use models::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    GaussianLikelihoodSpec, Hyperparameters, Likelihood, Model, ParameterLayout, SingleTaskGP,
    SingleTaskGPOptions,
};
pub use posterior::{JointPosteriorGrad, Posterior};
pub use priors::Prior;
pub use sampling::SobolQMCNormalSampler;
pub use transforms::{Normalize, OutcomeTransformChoice, Standardize};
