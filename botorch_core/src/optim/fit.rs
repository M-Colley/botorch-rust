//! Model fitting with BoTorch's optimizer routines (`botorch.optim.fit`):
//! [`fit_gpytorch_mll_torch`] (Adam with a stopping criterion, BoTorch's
//! `fit_gpytorch_mll_torch`) and [`fit_gpytorch_mll_scipy`] (one L-BFGS-B
//! attempt without retries, BoTorch's `fit_gpytorch_mll_scipy`; batched and
//! multi-output models are fitted element by element). Both optimize the
//! raw hyperparameters, optionally a subset of them (`mask`), within their
//! bounds (optionally overridden).
//!
//! `crate::fit::fit_gpytorch_mll` adds BoTorch's retries from prior samples
//! on top of the L-BFGS-B fit.

use std::time::Instant;

use rayon::prelude::*;

use crate::error::{BotorchError, Result};
use crate::lbfgsb::{self, LbfgsbError, LbfgsbOptions, LbfgsbStatus};
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::BatchedMultiOutputGP;
use crate::optim::minimize::{
    torch_minimize, MinimizeCallback, OptimizationResult, OptimizationStatus, TorchMinimizeOptions,
};

/// A model whose marginal log likelihood can be maximized over its flat raw
/// hyperparameters.
pub trait MllModel {
    /// The raw hyperparameters in BoTorch's flat order.
    fn raw_parameter_vector(&self) -> Vec<f64>;
    /// Sets the raw hyperparameters.
    ///
    /// # Errors
    /// For a wrong length or invalid values.
    fn set_raw_parameter_vector(&mut self, values: &[f64]) -> Result<()>;
    /// The optimizer bounds of the raw hyperparameters.
    fn raw_bounds(&self) -> Vec<(f64, f64)>;
    /// The MLL values (one per element of the model's batch) and the gradient
    /// of their sum at the raw hyperparameters `x`.
    ///
    /// # Errors
    /// Numerical errors of the MLL.
    fn mll_values_and_gradient(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)>;
}

impl MllModel for BatchedMultiOutputGP {
    fn raw_parameter_vector(&self) -> Vec<f64> {
        self.raw_parameters().as_slice().to_vec()
    }

    fn set_raw_parameter_vector(&mut self, values: &[f64]) -> Result<()> {
        self.set_raw_parameters(values)
    }

    fn raw_bounds(&self) -> Vec<(f64, f64)> {
        self.raw_parameter_bounds()
    }

    fn mll_values_and_gradient(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let (v, g) = self.mll_values_and_grad(x)?;
        Ok((v, g.as_slice().to_vec()))
    }
}

/// The free entries and their bounds.
fn free_entries(
    n: usize,
    mask: Option<&[bool]>,
    default_bounds: &[(f64, f64)],
    bounds: Option<&[(f64, f64)]>,
) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>)> {
    if let Some(m) = mask {
        if m.len() != n {
            return Err(BotorchError::InvalidArgument(format!(
                "the parameter mask has {} entries for {n} parameters",
                m.len()
            )));
        }
    }
    let bounds = bounds.unwrap_or(default_bounds);
    if bounds.len() != n {
        return Err(BotorchError::InvalidArgument(format!(
            "{} bounds for {n} parameters",
            bounds.len()
        )));
    }
    let free: Vec<usize> = (0..n).filter(|&i| mask.is_none_or(|m| m[i])).collect();
    let lower = free.iter().map(|&i| bounds[i].0).collect();
    let upper = free.iter().map(|&i| bounds[i].1).collect();
    Ok((free, lower, upper))
}

/// Fits the hyperparameters with Adam (BoTorch's `fit_gpytorch_mll_torch`
/// with the default closure): `torch_minimize` on the negated sum of the
/// MLL values of all elements over the free raw hyperparameters (`mask`,
/// default all) within `bounds` (default the model's), starting from the
/// current values. The model keeps the final hyperparameters.
///
/// # Errors
/// The errors of [`torch_minimize`] and of the MLL.
pub fn fit_gpytorch_mll_torch<M: MllModel + ?Sized>(
    model: &mut M,
    mask: Option<&[bool]>,
    bounds: Option<&[(f64, f64)]>,
    options: &TorchMinimizeOptions,
    callback: Option<MinimizeCallback<'_>>,
) -> Result<OptimizationResult> {
    let x_full = model.raw_parameter_vector();
    let n = x_full.len();
    let (free, lower, upper) = free_entries(n, mask, &model.raw_bounds(), bounds)?;
    let mut params: Vec<f64> = free.iter().map(|&i| x_full[i]).collect();
    let result = {
        let model_ref: &M = model;
        let closure = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
            let mut full = x_full.clone();
            for (j, &i) in free.iter().enumerate() {
                full[i] = x[j];
            }
            let (v, g) = model_ref.mll_values_and_gradient(&full)?;
            let loss = -v.iter().sum::<f64>();
            Ok((loss, free.iter().map(|&i| -g[i]).collect()))
        };
        torch_minimize(closure, &mut params, &lower, &upper, options, callback)?
    };
    let mut full = x_full;
    for (j, &i) in free.iter().enumerate() {
        full[i] = params[j];
    }
    model.set_raw_parameter_vector(&full)?;
    Ok(result)
}

/// BoTorch's `OptimizationStatus` of an L-BFGS-B run.
fn lbfgsb_status(status: LbfgsbStatus) -> OptimizationStatus {
    match status {
        LbfgsbStatus::ConvergedProjectedGradient | LbfgsbStatus::ConvergedRelativeReduction => {
            OptimizationStatus::Success
        }
        LbfgsbStatus::MaxIterations | LbfgsbStatus::MaxEvaluations | LbfgsbStatus::CallbackStop => {
            OptimizationStatus::Stopped
        }
        LbfgsbStatus::AbnormalLineSearch => OptimizationStatus::Failure,
    }
}

/// One L-BFGS-B run of `f` from `x0` (with a timeout); numerical errors of
/// the MLL give NaN values and gradients, from which the line search
/// backtracks (BoTorch's `_handle_numerical_errors`), other errors abort.
fn lbfgsb_run<F>(
    f: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: &LbfgsbOptions,
    timeout_sec: Option<f64>,
) -> Result<(lbfgsb::LbfgsbResult, bool)>
where
    F: Fn(&[f64]) -> Result<(f64, Vec<f64>)>,
{
    let start = Instant::now();
    let mut timed_out = false;
    let objective = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
        match f(x) {
            Ok(r) => Ok(r),
            Err(BotorchError::Nan { .. }) => Ok((f64::NAN, vec![f64::NAN; x.len()])),
            Err(e) => Err(e),
        }
    };
    let callback = |_: usize, _: &[f64], _: f64| -> bool {
        let stop = timeout_sec.is_some_and(|t| start.elapsed().as_secs_f64() > t);
        if stop {
            timed_out = true;
        }
        stop
    };
    match lbfgsb::minimize_with_callback(objective, x0, lower, upper, options, callback) {
        Ok(r) => Ok((r, timed_out)),
        Err(LbfgsbError::Objective(e)) => Err(e),
        Err(LbfgsbError::InvalidInput(msg)) => Err(BotorchError::InvalidArgument(msg)),
    }
}

/// Fits the hyperparameters with one L-BFGS-B run (BoTorch's
/// `fit_gpytorch_mll_scipy` with the default closure, no retries): for a
/// model with one element, the negative MLL is minimized over the free raw
/// hyperparameters within the bounds; for batched or multi-output models,
/// every element is fitted independently (BoTorch's batched independent
/// fitting, without a limit on function evaluations; in parallel) and the
/// status is `SUCCESS` if all converged, `STOPPED` if one reached the
/// iteration limit, else `FAILURE`. The model keeps the final
/// hyperparameters.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for inconsistent masks or bounds, and
/// non-numerical errors of the MLL.
pub fn fit_gpytorch_mll_scipy(
    model: &mut BatchedMultiOutputGP,
    mask: Option<&[bool]>,
    bounds: Option<&[(f64, f64)]>,
    options: &LbfgsbOptions,
    timeout_sec: Option<f64>,
) -> Result<OptimizationResult> {
    let start = Instant::now();
    let x_full = model.raw_parameter_vector();
    let n = x_full.len();
    let (free, lower, upper) = free_entries(n, mask, &model.raw_bounds(), bounds)?;
    let n_elements = model.num_elements();
    if n_elements == 1 {
        let model_ref: &BatchedMultiOutputGP = model;
        let f = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
            let mut full = x_full.clone();
            for (j, &i) in free.iter().enumerate() {
                full[i] = x[j];
            }
            let (v, g) = model_ref.mll_values_and_gradient(&full)?;
            Ok((
                -v.iter().sum::<f64>(),
                free.iter().map(|&i| -g[i]).collect(),
            ))
        };
        let x0: Vec<f64> = free.iter().map(|&i| x_full[i]).collect();
        let (res, timed_out) = lbfgsb_run(f, &x0, &lower, &upper, options, timeout_sec)?;
        let mut full = x_full;
        for (j, &i) in free.iter().enumerate() {
            full[i] = res.x[j];
        }
        model.set_raw_parameter_vector(&full)?;
        let message = if timed_out {
            format!(
                "Optimization timed out after {} seconds.",
                start.elapsed().as_secs_f64()
            )
        } else {
            res.status.message().to_string()
        };
        return Ok(OptimizationResult {
            step: res.iterations,
            fval: res.f,
            status: lbfgsb_status(res.status),
            runtime: Some(start.elapsed().as_secs_f64()),
            message: Some(message),
        });
    }
    // Batched independent fitting: per-element masks and bounds.
    let mask_f: Vec<f64> = (0..n)
        .map(|i| if mask.is_none_or(|m| m[i]) { 1.0 } else { 0.0 })
        .collect();
    let bounds_all: Vec<(f64, f64)> = bounds.map_or_else(|| model.raw_bounds(), <[_]>::to_vec);
    let lower_f: Vec<f64> = bounds_all.iter().map(|b| b.0).collect();
    let upper_f: Vec<f64> = bounds_all.iter().map(|b| b.1).collect();
    let per_mask = model.split_flat(&mask_f)?;
    let per_lower = model.split_flat(&lower_f)?;
    let per_upper = model.split_flat(&upper_f)?;
    let element_options = LbfgsbOptions {
        maxfun: usize::MAX,
        ..*options
    };
    let outcomes = model
        .models_mut()
        .par_iter_mut()
        .enumerate()
        .map(|(e, m)| -> Result<lbfgsb::LbfgsbResult> {
            let x_e = m.raw_parameters().as_slice().to_vec();
            let free_e: Vec<usize> = (0..x_e.len()).filter(|&i| per_mask[e][i] != 0.0).collect();
            let lo: Vec<f64> = free_e.iter().map(|&i| per_lower[e][i]).collect();
            let hi: Vec<f64> = free_e.iter().map(|&i| per_upper[e][i]).collect();
            let x0: Vec<f64> = free_e.iter().map(|&i| x_e[i]).collect();
            let res = {
                let mll = ExactMarginalLogLikelihood::new(m);
                let f = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
                    let mut full = x_e.clone();
                    for (j, &i) in free_e.iter().enumerate() {
                        full[i] = x[j];
                    }
                    let (v, g) = mll.value_and_grad(&full)?;
                    Ok((-v, free_e.iter().map(|&i| -g[i]).collect()))
                };
                lbfgsb_run(f, &x0, &lo, &hi, &element_options, None)?.0
            };
            let mut full = x_e;
            for (j, &i) in free_e.iter().enumerate() {
                full[i] = res.x[j];
            }
            m.set_raw_parameters(&full)?;
            Ok(res)
        })
        .collect::<Result<Vec<_>>>()?;
    let converged = outcomes.iter().filter(|r| r.status.success()).count();
    let any_limit = outcomes.iter().any(|r| {
        matches!(
            r.status,
            LbfgsbStatus::MaxIterations | LbfgsbStatus::MaxEvaluations
        )
    });
    let status = if converged == n_elements {
        OptimizationStatus::Success
    } else if any_limit {
        OptimizationStatus::Stopped
    } else {
        OptimizationStatus::Failure
    };
    Ok(OptimizationResult {
        step: outcomes.iter().map(|r| r.iterations).max().unwrap_or(0),
        fval: outcomes.iter().map(|r| r.f).sum(),
        status,
        runtime: Some(start.elapsed().as_secs_f64()),
        message: Some(format!(
            "Batched L-BFGS-B: {converged}/{n_elements} outputs converged."
        )),
    })
}
