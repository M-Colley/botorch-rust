//! Hyperparameter fitting mirroring `botorch.fit.fit_gpytorch_mll`.
//!
//! The negative MLL is minimized over the raw hyperparameters with L-BFGS-B
//! ([`crate::lbfgsb`], a port of L-BFGS-B 3.0 that reproduces scipy's
//! iterates), starting from the model's current values, with box bounds from
//! the non-enforced constraints (e.g. lengthscales `>= 2.5e-2`, noise
//! `>= 1e-4`; the constant mean is unbounded). The defaults are the scipy
//! L-BFGS-B settings BoTorch uses (`maxcor = 10`, `pgtol = 1e-5`,
//! `ftol = 2.22e-9` i.e. `factr = 1e7`, `maxiter = maxfun = 15000`,
//! `maxls = 20`).
//!
//! Like BoTorch's `_fit_fallback`, an attempt that terminates abnormally (a
//! line-search failure) or raises a numerical error (a covariance that is not
//! positive definite even with jitter, or non-finite values) is retried from
//! hyperparameters resampled from their priors (BoTorch's
//! `sample_all_priors`; parameters without priors keep their initial values),
//! up to `max_attempts` attempts in total. Hitting the iteration or evaluation
//! limit is not a failure (BoTorch's `OptimizationStatus.STOPPED`).
//!
//! L-BFGS-B iterates are sensitive to round-off: on flat or ill-conditioned
//! likelihood surfaces, long runs can stop at slightly different points than
//! BoTorch (BoTorch's own results change with the number of BLAS threads).
//! On the parity fixtures the final losses agree with BoTorch's to 4e-7 or
//! better, usually with identical iteration counts.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::error::{BotorchError, Result};
use crate::lbfgsb::{self, LbfgsbError, LbfgsbOptions, LbfgsbStatus};
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::{Hyperparameters, Likelihood, SingleTaskGP};

/// Options of [`fit_gpytorch_mll`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitOptions {
    /// Number of L-BFGS corrections kept (scipy `maxcor`, default 10).
    pub maxcor: usize,
    /// Projected-gradient tolerance (scipy `gtol`, default 1e-5).
    pub pgtol: f64,
    /// Relative function-reduction tolerance (scipy `ftol`, default
    /// `2.220446049250313e-9`, i.e. `factr = 1e7`).
    pub ftol: f64,
    /// Maximum number of iterations (scipy `maxiter`, default 15000).
    pub maxiter: usize,
    /// Maximum number of function evaluations (scipy `maxfun`, default 15000).
    pub maxfun: usize,
    /// Maximum number of line-search steps per iteration (scipy `maxls`, default 20).
    pub maxls: usize,
    /// Maximum number of fit attempts (BoTorch `max_attempts`, default 5).
    pub max_attempts: usize,
    /// Seed of the random number generator used to resample hyperparameters
    /// from their priors on retries.
    pub seed: u64,
}

impl Default for FitOptions {
    fn default() -> Self {
        let lb = LbfgsbOptions::default();
        Self {
            maxcor: lb.maxcor,
            pgtol: lb.gtol,
            ftol: lb.ftol,
            maxiter: lb.maxiter,
            maxfun: lb.maxfun,
            maxls: lb.maxls,
            max_attempts: 5,
            seed: 0,
        }
    }
}

/// Termination status (BoTorch's `OptimizationStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitStatus {
    /// Converged (projected gradient or relative reduction criterion).
    Success,
    /// Stopped by the iteration or function-evaluation limit.
    Stopped,
    /// Abnormal termination (line-search failure or numerical error).
    Failure,
}

/// Summary of a successful [`fit_gpytorch_mll`] call.
#[derive(Clone, Debug, PartialEq)]
pub struct FitResult {
    /// Final loss, i.e. the negative MLL at the fitted hyperparameters
    /// (BoTorch's `OptimizationResult.fval`).
    pub loss: f64,
    /// Number of L-BFGS-B iterations of the successful attempt (`nit`).
    pub iterations: usize,
    /// Number of loss/gradient evaluations of the successful attempt (`nfev`).
    pub evaluations: usize,
    /// Termination status (never [`FitStatus::Failure`]).
    pub status: FitStatus,
    /// Termination message (scipy's wording).
    pub message: String,
    /// Number of attempts used (1 if the first attempt succeeded).
    pub attempts: usize,
}

fn fit_status(status: LbfgsbStatus) -> FitStatus {
    match status {
        LbfgsbStatus::ConvergedProjectedGradient | LbfgsbStatus::ConvergedRelativeReduction => {
            FitStatus::Success
        }
        LbfgsbStatus::MaxIterations | LbfgsbStatus::MaxEvaluations | LbfgsbStatus::CallbackStop => {
            FitStatus::Stopped
        }
        LbfgsbStatus::AbnormalLineSearch => FitStatus::Failure,
    }
}

/// BoTorch's `sample_all_priors`: draws every hyperparameter that has a prior
/// from that prior (keeping all other parameters at `initial`) and returns the
/// flat raw-parameter vector.
fn sample_all_priors(
    model: &SingleTaskGP,
    initial: &Hyperparameters,
    rng: &mut StdRng,
) -> Vec<f64> {
    let mut h = initial.clone();
    if let Likelihood::Gaussian(spec) = model.likelihood() {
        if let Some(prior) = spec.noise_prior {
            h.raw_noise = Some(spec.noise_constraint.inverse_transform(prior.sample(rng)));
        }
    }
    let covar = model.covar_module();
    if let Some(scale) = covar.scale {
        if let Some(prior) = scale.outputscale_prior {
            h.raw_outputscale = Some(
                scale
                    .outputscale_constraint
                    .inverse_transform(prior.sample(rng)),
            );
        }
    }
    if let Some(prior) = covar.lengthscale_prior {
        for v in h.raw_lengthscale.iter_mut() {
            *v = covar
                .lengthscale_constraint
                .inverse_transform(prior.sample(rng));
        }
    }
    h.to_vector().as_slice().to_vec()
}

/// Fits the hyperparameters of `model` by maximizing the exact marginal log
/// likelihood (BoTorch's `fit_gpytorch_mll` with `fit_gpytorch_mll_scipy`).
///
/// On success the model's hyperparameters are set to the optimum. If every
/// attempt fails, the model keeps its initial hyperparameters (BoTorch rolls
/// the module back) and [`BotorchError::ModelFitting`] is returned.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for invalid options and
/// [`BotorchError::ModelFitting`] if all attempts fail.
pub fn fit_gpytorch_mll(model: &mut SingleTaskGP, options: FitOptions) -> Result<FitResult> {
    if options.max_attempts == 0 || options.maxcor == 0 {
        return Err(BotorchError::InvalidArgument(
            "max_attempts and maxcor must be at least 1".into(),
        ));
    }
    let lb_options = LbfgsbOptions {
        maxcor: options.maxcor,
        ftol: options.ftol,
        gtol: options.pgtol,
        maxiter: options.maxiter,
        maxfun: options.maxfun,
        maxls: options.maxls,
    };
    let initial = model.hyperparameters().clone();
    let x_init = initial.to_vector().as_slice().to_vec();
    let (lower, upper): (Vec<f64>, Vec<f64>) = model.raw_parameter_bounds().into_iter().unzip();
    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut messages = Vec::new();
    for attempt in 1..=options.max_attempts {
        let x0 = if attempt == 1 {
            x_init.clone()
        } else {
            sample_all_priors(model, &initial, &mut rng)
        };
        let outcome = {
            let mll = ExactMarginalLogLikelihood::new(model);
            let loss_and_grad = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
                let (value, grad) = mll.value_and_grad(x)?;
                if !value.is_finite() || grad.iter().any(|g| !g.is_finite()) {
                    return Err(BotorchError::Numerical(format!(
                        "non-finite marginal log likelihood {value} or gradient at raw parameters {x:?}"
                    )));
                }
                Ok((-value, grad.iter().map(|g| -g).collect()))
            };
            lbfgsb::minimize(loss_and_grad, &x0, &lower, &upper, &lb_options)
        };
        match outcome {
            Ok(res) => {
                let status = fit_status(res.status);
                if status != FitStatus::Failure {
                    model.set_raw_parameters(&res.x)?;
                    return Ok(FitResult {
                        loss: res.f,
                        iterations: res.iterations,
                        evaluations: res.evaluations,
                        status,
                        message: res.status.message().to_string(),
                        attempts: attempt,
                    });
                }
                messages.push(format!("attempt {attempt}: {}", res.status.message()));
            }
            Err(LbfgsbError::Objective(e)) => messages.push(format!("attempt {attempt}: {e}")),
            Err(LbfgsbError::InvalidInput(msg)) => return Err(BotorchError::InvalidArgument(msg)),
        }
    }
    Err(BotorchError::ModelFitting(format!(
        "All attempts to fit the model have failed ({}).",
        messages.join("; ")
    )))
}
