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
//! positive definite even with jitter) is retried from
//! hyperparameters resampled from their priors (BoTorch's
//! `sample_all_priors`; parameters without priors keep their initial values),
//! up to `max_attempts` attempts in total. An invalid argument (a
//! hyperparameter leaving the support of its prior, torch's `ValueError`)
//! ends the fit with that error, as in BoTorch. Hitting the iteration or
//! evaluation limit is not a failure (BoTorch's `OptimizationStatus.STOPPED`).
//!
//! Non-finite values and gradients go to L-BFGS-B exactly as BoTorch's
//! closure hands them to scipy (the L-BFGS-B port treats them like scipy,
//! whose line search does not raise on them): at raw hyperparameters where
//! GPyTorch's covariance is NaN (squared distances that overflow,
//! lengthscales that underflowed to zero; see [`crate::kernels::nonfinite`])
//! the loss and all gradients are NaN
//! (BoTorch's `_handle_numerical_errors` for linear_operator's `NanError`);
//! where a prior density vanishes (e.g. an outputscale that underflowed to
//! zero under a `GammaPrior`) the loss is `+inf` with the gradients that
//! GPyTorch computes (`inf · 0 = NaN` through the softplus). Lengthscales
//! and outputscales of zero are accepted during the fit; set by the user
//! (`initialize`, constructors) they still raise.
//!
//! L-BFGS-B iterates are sensitive to round-off: on flat or ill-conditioned
//! likelihood surfaces, long runs can stop at slightly different points than
//! BoTorch (BoTorch's own results change with the number of BLAS threads).
//! On the parity fixtures the final losses agree with BoTorch's to 4e-7 or
//! better, usually with identical iteration counts.
//!
//! [`fit_gpytorch_mll`] is generic over [`Fittable`] models:
//!
//! * [`SingleTaskGP`]: one L-BFGS-B problem (BoTorch's `fit_gpytorch_mll_scipy`);
//! * [`BatchedMultiOutputGP`]: one independent L-BFGS-B problem per batch
//!   element and output, run in parallel. This is BoTorch's batched
//!   independent fitting (`_fit_gpytorch_mll_scipy_independent` with
//!   `fmin_l_bfgs_b_batched`), which BoTorch uses for every model whose
//!   `_aug_batch_shape` has more than one element: every element follows
//!   exactly the iterates of its own L-BFGS-B problem, without a limit on the
//!   number of function evaluations (`maxfun` is not used, as in BoTorch).
//!   Failed elements are retried individually from prior samples (BoTorch
//!   accepts abnormal terminations of individual elements without retrying
//!   and restarts all elements after a numerical error). A NaN covariance of
//!   an element gives that element a NaN loss and NaN gradients, as in
//!   BoTorch's unbatched closure. This deviates from BoTorch's batched
//!   closure, which evaluates the elements' trial points in one batch: a NaN
//!   covariance of any element raises `NanError` for the whole batch and
//!   gives every element a NaN loss with zero gradients, after which an
//!   element can stop at that point with "converged" and a NaN loss (a
//!   BoTorch bug, see `docs/drafts/fit-nonfinite.md`);
//! * [`ModelListGP`]: every sub-model is fitted on its own, like BoTorch's
//!   `fit_gpytorch_mll` for a `SumMarginalLogLikelihood`.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::error::{BotorchError, Result};
use crate::kernels::CovarModule;
use crate::lbfgsb::{self, LbfgsbError, LbfgsbOptions, LbfgsbStatus};
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::{BatchedMultiOutputGP, Hyperparameters, Likelihood, ModelListGP, SingleTaskGP};

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
///
/// Kernel trees: every parameter with a univariate prior is resampled; the
/// task factor of a `PositiveIndexKernel` with a task prior is set from
/// sampled correlations (BoTorch's `_set_lower_triangle_corr`); an
/// `IndexKernel` task prior has no setting closure in GPyTorch (BoTorch's
/// `sample_all_priors` raises), so those parameters keep their values.
fn sample_all_priors(
    model: &SingleTaskGP,
    initial: &Hyperparameters,
    rng: &mut StdRng,
) -> Vec<f64> {
    let mut h = initial.clone();
    if let Some(prior) = model.mean_module().prior {
        // GPyTorch's `ConstantMean` setting closure (the raw constant is the
        // constant: the constraint is not enforced).
        h.raw_constant = prior.sample(rng);
    }
    if let Likelihood::Gaussian(spec) = model.likelihood() {
        if let Some(prior) = spec.noise_prior {
            h.raw_noise = Some(spec.noise_constraint.inverse_transform(prior.sample(rng)));
        }
    }
    match model.covar_module() {
        CovarModule::Standard(covar) => {
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
        }
        CovarModule::Tree(tree) => {
            crate::kernels::tree::sample_tree_priors(tree, h.raw_kernel.as_mut_slice(), rng);
        }
    }
    h.to_vector().as_slice().to_vec()
}

/// BoTorch's `sample_all_priors` on every element of a batched model:
/// every hyperparameter that has a prior is drawn from it (in place).
///
/// # Errors
/// The errors of [`SingleTaskGP::set_raw_parameters`].
pub fn sample_all_priors_batched(model: &mut BatchedMultiOutputGP, seed: u64) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(seed);
    for m in model.models_mut() {
        let h = m.hyperparameters().clone();
        let x = sample_all_priors(m, &h, &mut rng);
        m.set_raw_parameters(&x)?;
    }
    Ok(())
}

/// Models whose hyperparameters can be fitted by [`fit_gpytorch_mll`].
pub trait Fittable {
    /// The summary returned by a successful fit.
    type Output;

    /// Fits the hyperparameters by maximizing the exact marginal log
    /// likelihood (see [`fit_gpytorch_mll`]).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for invalid options or hyperparameters
    /// outside the support of their priors, and
    /// [`BotorchError::ModelFitting`] if all attempts fail.
    fn fit_mll(&mut self, options: FitOptions) -> Result<Self::Output>;
}

/// Fits the hyperparameters of `model` by maximizing the exact marginal log
/// likelihood (BoTorch's `fit_gpytorch_mll` with `fit_gpytorch_mll_scipy`).
///
/// On success the model's hyperparameters are set to the optimum. If every
/// attempt fails, the model keeps its initial hyperparameters (BoTorch rolls
/// the module back) and [`BotorchError::ModelFitting`] is returned. See the
/// [module documentation](self) for multi-output, batched and list models.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for invalid options or hyperparameters
/// outside the support of their priors, and
/// [`BotorchError::ModelFitting`] if all attempts fail.
pub fn fit_gpytorch_mll<M: Fittable + ?Sized>(
    model: &mut M,
    options: FitOptions,
) -> Result<M::Output> {
    model.fit_mll(options)
}

impl Fittable for SingleTaskGP {
    type Output = FitResult;

    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        fit_single_task_gp(self, options)
    }
}

/// The single-output fit of [`fit_gpytorch_mll`].
fn fit_single_task_gp(model: &mut SingleTaskGP, options: FitOptions) -> Result<FitResult> {
    let initial = model.hyperparameters().clone();
    let x_init = initial.to_vector().as_slice().to_vec();
    let bounds = model.raw_parameter_bounds();
    let (result, x) = {
        let mll = ExactMarginalLogLikelihood::new(model);
        let model_ref: &SingleTaskGP = model;
        minimize_with_retries(
            &x_init,
            &bounds,
            options,
            |x| {
                mll.value_and_grad(x)
                    .map(|(v, g)| (v, g.as_slice().to_vec()))
            },
            |rng| sample_all_priors(model_ref, &initial, rng),
        )?
    };
    model.set_raw_parameters(&x)?;
    Ok(result)
}

/// BoTorch's `_fit_fallback` with `fit_gpytorch_mll_scipy` for a model with
/// flat raw parameters: maximizes `mll` (value and gradient with respect to
/// the raw parameters) with L-BFGS-B from `x_init` within `bounds`, and
/// retries failed attempts from `resample` (BoTorch's `sample_all_priors`),
/// up to `options.max_attempts` attempts. Returns the summary and the fitted
/// raw parameters.
///
/// A NaN covariance ([`BotorchError::Nan`]) gives NaN values and
/// gradients, and other non-finite values and gradients are passed on
/// unchanged, as BoTorch's closure (which maps linear_operator's `NanError`
/// to NaN) passes them to scipy.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for invalid options or hyperparameters
/// outside the support of their priors, and
/// [`BotorchError::ModelFitting`] if all attempts fail.
pub(crate) fn minimize_with_retries<F, S>(
    x_init: &[f64],
    bounds: &[(f64, f64)],
    options: FitOptions,
    mll: F,
    mut resample: S,
) -> Result<(FitResult, Vec<f64>)>
where
    F: Fn(&[f64]) -> Result<(f64, Vec<f64>)>,
    S: FnMut(&mut StdRng) -> Vec<f64>,
{
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
    let (lower, upper): (Vec<f64>, Vec<f64>) = bounds.iter().copied().unzip();
    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut messages = Vec::new();
    for attempt in 1..=options.max_attempts {
        let x0 = if attempt == 1 {
            x_init.to_vec()
        } else {
            resample(&mut rng)
        };
        let loss_and_grad = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
            let (value, grad) = match mll(x) {
                Ok(vg) => vg,
                // BoTorch's `_handle_numerical_errors`: a NaN covariance
                // (linear_operator's `NanError`) gives NaN values and
                // gradients, which L-BFGS-B's line search handles like
                // scipy's.
                Err(BotorchError::Nan { .. }) => return Ok((f64::NAN, vec![f64::NAN; x.len()])),
                Err(e) => return Err(e),
            };
            Ok((-value, grad.iter().map(|g| -g).collect()))
        };
        match lbfgsb::minimize(loss_and_grad, &x0, &lower, &upper, &lb_options) {
            Ok(res) => {
                let status = fit_status(res.status);
                if status != FitStatus::Failure {
                    let result = FitResult {
                        loss: res.f,
                        iterations: res.iterations,
                        evaluations: res.evaluations,
                        status,
                        message: res.status.message().to_string(),
                        attempts: attempt,
                    };
                    return Ok((result, res.x));
                }
                messages.push(format!("attempt {attempt}: {}", res.status.message()));
            }
            // BoTorch retries only numerical failures (`NotPSDError`); an
            // invalid argument (e.g. a hyperparameter outside the support of
            // its prior, torch's `ValueError`) ends the fit.
            Err(LbfgsbError::Objective(e @ BotorchError::InvalidArgument(_))) => return Err(e),
            Err(LbfgsbError::Objective(e)) => messages.push(format!("attempt {attempt}: {e}")),
            Err(LbfgsbError::InvalidInput(msg)) => return Err(BotorchError::InvalidArgument(msg)),
        }
    }
    Err(BotorchError::ModelFitting(format!(
        "All attempts to fit the model have failed ({}).",
        messages.join("; ")
    )))
}

/// Summary of a successful fit of a [`BatchedMultiOutputGP`] (BoTorch's
/// `OptimizationResult` of the batched fit) with the results of every batch
/// element.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchedFitResult {
    /// Sum of the final losses of all elements (BoTorch's `fval`).
    pub loss: f64,
    /// Largest number of L-BFGS-B iterations of an element (BoTorch's `step`).
    pub iterations: usize,
    /// [`FitStatus::Success`] if every element converged, otherwise
    /// [`FitStatus::Stopped`] (an iteration limit was reached).
    pub status: FitStatus,
    /// BoTorch's message (`"Batched L-BFGS-B: k/E outputs converged."`), or
    /// the L-BFGS-B message for a model with a single element.
    pub message: String,
    /// The fits of the elements in row-major order of BoTorch's
    /// `_aug_batch_shape` (batch elements, then outputs).
    pub results: Vec<FitResult>,
}

impl Fittable for BatchedMultiOutputGP {
    type Output = BatchedFitResult;

    fn fit_mll(&mut self, options: FitOptions) -> Result<BatchedFitResult> {
        let n_elements = self.num_elements();
        if n_elements == 1 {
            // BoTorch's plain (non-batched) scipy fit.
            let r = fit_single_task_gp(&mut self.models_mut()[0], options)?;
            return Ok(BatchedFitResult {
                loss: r.loss,
                iterations: r.iterations,
                status: r.status,
                message: r.message.clone(),
                results: vec![r],
            });
        }
        // BoTorch's fmin_l_bfgs_b_batched does not limit function evaluations.
        let element_options = FitOptions {
            maxfun: usize::MAX,
            ..options
        };
        let initial: Vec<_> = self
            .models()
            .iter()
            .map(|m| m.hyperparameters().clone())
            .collect();
        let outcomes: Vec<Result<FitResult>> = self
            .models_mut()
            .par_iter_mut()
            .enumerate()
            .map(|(e, model)| {
                let o = FitOptions {
                    seed: options.seed.wrapping_add(e as u64),
                    ..element_options
                };
                fit_single_task_gp(model, o)
            })
            .collect();
        let mut results = Vec::with_capacity(n_elements);
        let mut failure = None;
        for (e, outcome) in outcomes.into_iter().enumerate() {
            match outcome {
                Ok(r) => results.push(r),
                Err(err) => {
                    failure.get_or_insert((e, err));
                }
            }
        }
        if let Some((e, err)) = failure {
            // Roll the whole model back, like BoTorch.
            for (model, h) in self.models_mut().iter_mut().zip(initial) {
                model.set_hyperparameters(h)?;
            }
            return Err(match err {
                BotorchError::ModelFitting(msg) => {
                    BotorchError::ModelFitting(format!("batch element {e}: {msg}"))
                }
                other => other,
            });
        }
        let converged = results
            .iter()
            .filter(|r| r.status == FitStatus::Success)
            .count();
        Ok(BatchedFitResult {
            loss: results.iter().map(|r| r.loss).sum(),
            iterations: results.iter().map(|r| r.iterations).max().unwrap_or(0),
            status: if converged == n_elements {
                FitStatus::Success
            } else {
                FitStatus::Stopped
            },
            message: format!("Batched L-BFGS-B: {converged}/{n_elements} outputs converged."),
            results,
        })
    }
}

impl Fittable for ModelListGP {
    type Output = Vec<BatchedFitResult>;

    /// Fits every sub-model on its own (in parallel). As with BoTorch's
    /// sequential loop over the sub-models, if a sub-model cannot be fitted,
    /// the preceding sub-models keep their fitted hyperparameters, the
    /// following ones are left unchanged and the error is returned.
    fn fit_mll(&mut self, options: FitOptions) -> Result<Vec<BatchedFitResult>> {
        let fitted: Vec<Result<(BatchedMultiOutputGP, BatchedFitResult)>> = self
            .models()
            .par_iter()
            .map(|m| {
                let mut model = (**m).clone();
                let r = fit_gpytorch_mll(&mut model, options)?;
                Ok((model, r))
            })
            .collect();
        let mut results = Vec::with_capacity(fitted.len());
        for (i, outcome) in fitted.into_iter().enumerate() {
            let (model, r) = outcome.map_err(|e| match e {
                BotorchError::ModelFitting(msg) => {
                    BotorchError::ModelFitting(format!("sub-model {i}: {msg}"))
                }
                other => other,
            })?;
            self.replace_model(i, model)?;
            results.push(r);
        }
        Ok(results)
    }
}
