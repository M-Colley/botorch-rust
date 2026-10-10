//! # botorch_core
//!
//! A Rust re-implementation of [BoTorch](https://botorch.org): the
//! Gaussian-process core (`SingleTaskGP` with BoTorch's default priors,
//! constraints and transforms, the exact marginal log likelihood with analytic
//! gradients, `fit_gpytorch_mll` with a port of L-BFGS-B 3.0), analytic and
//! Monte-Carlo acquisition functions with exact gradients, `optimize_acqf`,
//! quasi-random sampling and synthetic test functions. All computations are in
//! double precision and are validated against BoTorch itself (see
//! `scripts/generate_fixtures.py` and the `tests/*parity*.rs` files).
//!
//! ## Conventions
//!
//! * [`Matrix`] / [`Vector`] are `nalgebra` `DMatrix<f64>` / `DVector<f64>`.
//!   Point sets are `n × d` matrices whose **rows** are points; a candidate set
//!   `X` of BoTorch's shape `1 × q × d` is a `q × d` matrix, a batch
//!   `b × q × d` is a slice of `b` such matrices, and box bounds are `2 × d`
//!   (lower bounds in row 0).
//! * Inputs passed to a model or an acquisition function are in the *raw*
//!   (original) input space; the model applies its input transform. Posteriors
//!   are reported in the *original* outcome space (un-standardized), like
//!   BoTorch's `posterior`.
//! * Hyperparameters are stored in BoTorch's *raw* parameterization. With the
//!   `SingleTaskGP` defaults the constraints are not enforced through a
//!   transform, so raw values equal actual values and the bounds are imposed
//!   by the optimizer.
//! * GPyTorch materializes prior parameters, constraint bounds and initial
//!   values in float32 before casting the model to double precision. The
//!   BoTorch presets in this crate reproduce that rounding (see
//!   [`Prior::rounded_to_f32`]) so that results agree to ~1e-10. The same holds
//!   for the float32 constants of BoTorch's test functions.
//! * Models and acquisition functions are `Send + Sync`; acquisition
//!   functions share their model through an `Arc<dyn Model>`.
//! * Seeds are `Option<u64>`: `Some(seed)` makes results reproducible, `None`
//!   draws a random seed (BoTorch draws from torch's global RNG).
//!
//! ## API overview
//!
//! | BoTorch / GPyTorch | botorch_core |
//! |---|---|
//! | `SingleTaskGP(train_X, train_Y, train_Yvar, covar_module, likelihood, outcome_transform, input_transform)` | [`SingleTaskGP::new`]`(train_x, train_y, `[`SingleTaskGPOptions`]`)` |
//! | `model.posterior(X, observation_noise)` | [`SingleTaskGP::posterior`]`(&x, observation_noise) -> `[`Posterior`]` { mean, covariance }` |
//! | mean/variance and their gradients for `q = 1` | [`SingleTaskGP::posterior_q1_with_grad`]`(&x) -> (mean, variance, dmean_dx, dvariance_dx)` |
//! | joint mean/covariance and their gradients | [`SingleTaskGP::joint_posterior_with_grad`]`(&x) -> `[`JointPosteriorGrad`] (index `j = point·d + dim`); reverse mode [`SingleTaskGP::joint_posterior_vjp`] |
//! | `botorch.models.model.Model` (what acquisition functions use) | the [`Model`] trait, implemented by [`SingleTaskGP`] |
//! | `model.named_parameters()` | [`SingleTaskGP::hyperparameters`], [`SingleTaskGP::set_hyperparameters`], [`SingleTaskGP::raw_parameters`], [`SingleTaskGP::parameter_names`] |
//! | `model.train_inputs`, `model.train_targets` | [`SingleTaskGP::train_inputs`], [`SingleTaskGP::train_targets`] (original space), [`SingleTaskGP::transformed_train_inputs`], [`SingleTaskGP::transformed_train_targets`] |
//! | `model.likelihood.noise` | [`SingleTaskGP::likelihood_noise`] |
//! | `ExactMarginalLogLikelihood(likelihood, model)` | [`ExactMarginalLogLikelihood::new`]`(&model)`, `.value()`, `.value_and_grad(&raw_params)` |
//! | `fit_gpytorch_mll(mll)` | [`fit_gpytorch_mll`]`(&mut model, `[`FitOptions::default()`](FitOptions)`) -> `[`FitResult`] |
//! | `ExpectedImprovement(model, best_f, maximize=...)`, `LogExpectedImprovement`, `ProbabilityOfImprovement`, `LogProbabilityOfImprovement` | [`acquisition::ExpectedImprovement::new`]`(model, best_f, maximize)`, [`acquisition::LogExpectedImprovement`], [`acquisition::ProbabilityOfImprovement`], [`acquisition::LogProbabilityOfImprovement`] |
//! | `UpperConfidenceBound(model, beta, maximize=...)`, `PosteriorMean`, `PosteriorStandardDeviation` | [`acquisition::UpperConfidenceBound::new`]`(model, beta, maximize)`, [`acquisition::PosteriorMean`], [`acquisition::PosteriorStandardDeviation`] |
//! | `qExpectedImprovement(model, best_f, sampler, X_pending=...)`, `qProbabilityOfImprovement(..., tau)`, `qUpperConfidenceBound(model, beta, ...)`, `qSimpleRegret` | [`acquisition::QExpectedImprovement::new`]`(model, best_f, `[`acquisition::McAcquisitionOptions`]` { sampler, x_pending })`, [`acquisition::QProbabilityOfImprovement`], [`acquisition::QUpperConfidenceBound`], [`acquisition::QSimpleRegret`] |
//! | `qLogExpectedImprovement(model, best_f, sampler, X_pending, fat, tau_max, tau_relu)` | [`acquisition::QLogExpectedImprovement::new`]`(model, best_f, `[`acquisition::QLogExpectedImprovementOptions`]`)` |
//! | `qLogNoisyExpectedImprovement(model, X_baseline, sampler, X_pending, prune_baseline, incremental, ...)` | [`acquisition::QLogNoisyExpectedImprovement::new`]`(model, &x_baseline, `[`acquisition::QLogNoisyExpectedImprovementOptions`]`)` |
//! | `acqf(X)`, `torch.autograd.grad(acqf(X), X)`, `acqf(X_batch)` | [`AcquisitionFunction::forward`]`(&x)`, [`AcquisitionFunction::forward_with_grad`]`(&x) -> (value, grad)`, [`AcquisitionFunction::forward_many`]`(&[x])` (parallel) |
//! | `acqf.set_X_pending(X_pending)` | [`AcquisitionFunction::set_x_pending`]`(Some(&x_pending))` |
//! | `prune_inferior_points(model, X, num_samples, max_frac, sampler)` | [`acquisition::prune_inferior_points`]`(&*model, &x, &`[`acquisition::PruneOptions`]`)` |
//! | `SobolQMCNormalSampler(sample_shape=[n], seed=...)` | [`sampling::SobolQMCNormalSampler::new`]`(n, seed)`; explicit base samples via [`sampling::SobolQMCNormalSampler::from_base_samples`] |
//! | `draw_sobol_samples(bounds, n, q, seed)`, `draw_sobol_normal_samples(d, n, seed)` | [`utils::sampling::draw_sobol_samples`], [`utils::sampling::draw_sobol_normal_samples`], [`utils::sampling::SobolEngine`] |
//! | `optimize_acqf(acq_function, bounds, q, num_restarts, raw_samples, options, fixed_features, sequential, ...)` | [`optim::optimize_acqf`]`(&acqf, &bounds, q, num_restarts, raw_samples, &`[`optim::OptimizeAcqfOptions`]`) -> (candidates, value)`; all restarts via [`optim::optimize_acqf_detailed`] |
//! | `gen_batch_initial_conditions`, `initialize_q_batch`, `initialize_q_batch_nonneg` | [`optim::gen_batch_initial_conditions`], [`optim::initialize_q_batch`], [`optim::initialize_q_batch_nonneg`] |
//! | `gen_candidates_scipy(initial_conditions, acqf, lower_bounds, upper_bounds)` | [`optim::gen_candidates_lbfgsb`] |
//! | `botorch.test_functions.Hartmann(dim=6, noise_std=..., negate=True)` etc. | [`test_functions::SyntheticTestFunction::hartmann`]`(6)?.with_negate(true)` etc. (20 functions) |
//! | `botorch.utils.safe_math`, `botorch.utils.probability.utils` | [`utils::safe_math`], [`utils::probability`] |
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
//! use std::sync::Arc;
//!
//! use botorch_core::acquisition::LogExpectedImprovement;
//! use botorch_core::optim::{optimize_acqf, OptimizeAcqfOptions};
//! use botorch_core::{fit_gpytorch_mll, FitOptions, Matrix, Model, SingleTaskGP, SingleTaskGPOptions};
//!
//! let train_x = Matrix::from_fn(12, 2, |i, j| ((i * 7 + j * 3) % 12) as f64 / 11.0);
//! let train_y = Matrix::from_fn(12, 1, |i, _| (6.0 * train_x[(i, 0)]).sin() + train_x[(i, 1)]);
//! let mut model = SingleTaskGP::new(train_x, train_y.clone(), SingleTaskGPOptions::default())?;
//! let fit = fit_gpytorch_mll(&mut model, FitOptions::default())?;
//! assert!(fit.loss.is_finite());
//! let (_mean, variance, dmean, _dvar) = model.posterior_q1_with_grad(&[0.2, 0.4])?;
//! assert!(variance > 0.0 && dmean.len() == 2);
//!
//! // Maximize LogEI over [0, 1]^2 (10 restarts, 512 raw samples, seed 0).
//! let model: Arc<dyn Model> = Arc::new(model);
//! let acqf = LogExpectedImprovement::new(model, train_y.max(), true)?;
//! let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
//! let options = OptimizeAcqfOptions { seed: Some(0), ..Default::default() };
//! let (candidate, value) = optimize_acqf(&acqf, &bounds, 1, 10, 512, &options)?;
//! assert_eq!(candidate.shape(), (1, 2));
//! assert!(value.is_finite());
//! # Ok::<(), botorch_core::BotorchError>(())
//! ```
//!
//! `examples/bo_loop.rs` runs a complete Bayesian optimization loop on
//! Hartmann-6.
//!
//! ## Validation
//!
//! `scripts/generate_fixtures.py` records reference values from BoTorch.
//!
//! *Gaussian processes* (17 models, d ∈ {1, 2, 5}, n ∈ {1, 3, 5, 10, 20}, RBF /
//! Matérn-5/2, `ScaleKernel`, fixed noise, `Normalize` with learned or explicit
//! bounds, with and without `Standardize`, legacy Gamma priors, degenerate
//! data): `tests/parity.rs` requires agreement to a relative 1e-10 for kernels
//! and their derivatives, transforms, MLL values and gradients, posterior means
//! and covariances (with and without observation noise) and their input
//! gradients. The observed errors are at most 6e-14, except 3e-12 for an MLL
//! whose covariance has condition number ~3e6. Fitting from BoTorch's default
//! initialization reaches BoTorch's final loss (worst excess 3.4e-7, typically
//! ~1e-15) with the same number of L-BFGS-B iterations in 16 of 17 problems.
//!
//! *Acquisition functions* (3 models, `tests/acquisition_parity.rs`): values and
//! autograd gradients of all analytic acquisition functions at 8 points
//! (`maximize` true and false, `best_f` up to 10⁹ above the data); MC
//! acquisition functions for `q` = 1, 2, 3 with and without pending points,
//! evaluated with BoTorch's recorded base samples; qLogNEI (incremental or not,
//! full or pruned baseline, and BoTorch's default `cache_root=True` with its
//! effective base samples); `prune_inferior_points` (identical selections).
//! All agree to a relative 1e-10 (observed: at most 6e-11 for EI/PI deep in
//! the tail, where the value's sensitivity to the posterior is about `u²`, and
//! at most 3e-11 for MC gradients; at most 1e-13 for most quantities).
//! `tests/acquisition_gradients.rs` checks every gradient against central
//! finite differences.
//!
//! *Optimization* (`tests/optimize_parity.rs`, a fitted Hartmann-6 model): from
//! BoTorch's own initial conditions, L-BFGS-B reaches BoTorch's final values to
//! ~1e-15 (same iterates and iteration counts); full `optimize_acqf` runs reach
//! BoTorch's best value in 10/10 seeds for LogEI and, for the multimodal q = 2
//! qLogEI problem, about as often as BoTorch does.
//!
//! *Test functions* (`tests/test_functions_parity.rs`, 28 configurations):
//! values, bounds, optima and optimizers agree to 3e-16.
//!
//! ## Scope and known differences
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
//!   negative noise variances are errors, as in BoTorch. `optimize_acqf`
//!   reports BoTorch's warnings as messages in
//!   [`optim::OptimizeAcqfResult::warnings`].
//! * `fit_gpytorch_mll` retries failed attempts from prior samples like
//!   BoTorch, but with its own seeded random number generator.
//! * Acquisition functions take `best_f`, `beta` and `tau` as exact `f64`
//!   values. BoTorch stores Python floats passed for these arguments as float32
//!   tensors (torch's default dtype), i.e. rounded to about 7 digits; the
//!   values here equal BoTorch's when float64 tensors are passed (as with the
//!   common `best_f=train_Y.max()`).
//! * No `objective`, `posterior_transform` or outcome `constraints` for the
//!   acquisition functions; qLogNEI implements the plain (`cache_root=False`)
//!   computation, which equals BoTorch's default `cache_root=True` given the
//!   same effective base samples (see
//!   [`acquisition::QLogNoisyExpectedImprovement`]).
//! * Quasi-random numbers come from an Owen-scrambled Sobol sequence
//!   (`sobol_burley`), not torch's `SobolEngine`, so raw samples and base
//!   samples differ from BoTorch's for the same seed (with the same
//!   distribution); explicit base samples reproduce BoTorch exactly. The
//!   random selection of initial conditions is seeded from `options.seed`
//!   (BoTorch uses torch's global RNG).
//! * The LogEI gradient is accurate to machine precision far in the tail,
//!   where BoTorch's autograd gradient has a relative error of about `ε u²`.
//! * `prune_inferior_points` with `max_frac < 1` keeps the most frequent
//!   maximizers (BoTorch's implementation indexes the wrong points there).
//! * The posterior sample root of a single point is `sqrt(max(v, 0))`
//!   (BoTorch yields NaN for a numerically negative variance); gradients
//!   through the eigendecomposition fallback for covariances that are not
//!   positive definite even with jitter are not available (an error; BoTorch's
//!   are NaN).
//! * `optimize_acqf` supports box bounds and fixed features (scalar values);
//!   no linear or nonlinear constraints, `post_processing_func`, timeouts or
//!   one-shot acquisition functions. Restarts are independent L-BFGS-B
//!   problems (BoTorch's default parallel mode), so `batch_limit` only chunks
//!   the work.
//! * Python bindings with BoTorch's API are provided by the `botorch_pyo3`
//!   crate (the `botorch_rs` package; see the repository README).

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
pub mod optim;
pub mod posterior;
pub mod priors;
pub mod sampling;
pub mod test_functions;
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
pub use fit::{fit_gpytorch_mll, BatchedFitResult, FitOptions, FitResult, FitStatus, Fittable};
pub use kernels::{Kernel, KernelKind, KernelSpec, ScaleKernelSpec};
pub use linalg::{psd_safe_cholesky, CholeskyFactor};
pub use mll::ExactMarginalLogLikelihood;
pub use models::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    BatchedInputTransform, BatchedMultiOutputGP, BatchedMultiOutputGPOptions,
    GaussianLikelihoodSpec, Hyperparameters, Likelihood, Model, ModelListGP, ParameterLayout,
    SingleTaskGP, SingleTaskGPOptions,
};
pub use optim::{optimize_acqf, OptimizeAcqfOptions};
pub use posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
pub use priors::Prior;
pub use sampling::SobolQMCNormalSampler;
pub use test_functions::SyntheticTestFunction;
pub use transforms::{Normalize, OutcomeTransformChoice, Standardize};
