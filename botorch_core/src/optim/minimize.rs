//! Generic optimization routines (`botorch.optim.core`): the result types
//! [`OptimizationStatus`] and [`OptimizationResult`], and [`torch_minimize`],
//! BoTorch's `torch.optim`-based loop with Adam (the loop that
//! `fit_gpytorch_mll_torch` runs).

use std::time::Instant;

use crate::error::{BotorchError, Result};
use crate::optim::adam::{Adam, AdamOptions};
use crate::optim::stopping::{ExpMaStoppingCriterion, StoppingCriterion};

/// The status of an optimization (BoTorch's `OptimizationStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationStatus {
    /// Incomplete.
    Running,
    /// The optimizer converged.
    Success,
    /// Terminated abnormally.
    Failure,
    /// Stopped by a limit (iterations, evaluations, steps, timeout) or a
    /// stopping criterion.
    Stopped,
}

impl OptimizationStatus {
    /// BoTorch's enum value (`RUNNING = 1`, ..., `STOPPED = 4`).
    pub fn value(self) -> u8 {
        match self {
            OptimizationStatus::Running => 1,
            OptimizationStatus::Success => 2,
            OptimizationStatus::Failure => 3,
            OptimizationStatus::Stopped => 4,
        }
    }

    /// BoTorch's enum name.
    pub fn name(self) -> &'static str {
        match self {
            OptimizationStatus::Running => "RUNNING",
            OptimizationStatus::Success => "SUCCESS",
            OptimizationStatus::Failure => "FAILURE",
            OptimizationStatus::Stopped => "STOPPED",
        }
    }
}

/// The summary of an optimization (BoTorch's `OptimizationResult`).
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationResult {
    /// The number of steps (iterations).
    pub step: usize,
    /// The final function value.
    pub fval: f64,
    /// The termination status.
    pub status: OptimizationStatus,
    /// The run time in seconds.
    pub runtime: Option<f64>,
    /// The termination message.
    pub message: Option<String>,
}

/// Settings of [`torch_minimize`].
#[derive(Clone, Debug, PartialEq)]
pub struct TorchMinimizeOptions {
    /// Adam settings (torch's defaults, learning rate 1e-3).
    pub adam: AdamOptions,
    /// Maximum number of steps.
    pub step_limit: Option<usize>,
    /// Stop after this many seconds.
    pub timeout_sec: Option<f64>,
    /// The stopping criterion (reset before the run).
    pub stopping_criterion: Option<ExpMaStoppingCriterion>,
}

impl Default for TorchMinimizeOptions {
    fn default() -> Self {
        Self {
            adam: AdamOptions::default(),
            step_limit: None,
            timeout_sec: None,
            stopping_criterion: Some(ExpMaStoppingCriterion::default()),
        }
    }
}

/// A callback invoked after the evaluation of every step with the current
/// parameters and the running result (BoTorch's `callback(parameters,
/// result)`).
pub type MinimizeCallback<'a> = &'a mut dyn FnMut(&[f64], &OptimizationResult) -> Result<()>;

/// Minimizes `closure` (value and gradient of the parameters) with Adam,
/// BoTorch's `torch_minimize`: every step evaluates the closure, checks the
/// stopping criterion on the value and the timeout, calls the callback,
/// stops if a condition holds, else takes an Adam step and clamps the
/// parameters to `[lower, upper]` (`±inf` for no bound). After `step_limit`
/// steps the closure is evaluated once more and the result is `STOPPED`.
///
/// # Errors
/// [`BotorchError::Runtime`] without a termination condition ("No
/// termination conditions were given."), [`BotorchError::InvalidArgument`]
/// for inconsistent sizes or Adam settings, and errors of the closure and the
/// callback.
pub fn torch_minimize<F>(
    mut closure: F,
    params: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    options: &TorchMinimizeOptions,
    mut callback: Option<MinimizeCallback<'_>>,
) -> Result<OptimizationResult>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
{
    let start = Instant::now();
    let n = params.len();
    if lower.len() != n || upper.len() != n {
        return Err(BotorchError::InvalidArgument(format!(
            "bounds have lengths {} and {} for {n} parameters",
            lower.len(),
            upper.len()
        )));
    }
    let step_limit = match options.step_limit {
        Some(s) => s,
        None => {
            if options.stopping_criterion.is_none() && options.timeout_sec.is_none() {
                return Err(BotorchError::Runtime(
                    "No termination conditions were given.".into(),
                ));
            }
            usize::MAX
        }
    };
    let mut adam = Adam::new(n, options.adam)?;
    let mut criterion = options.stopping_criterion.clone();
    if let Some(c) = criterion.as_mut() {
        c.reset();
    }
    let mut last: Option<OptimizationResult> = None;
    let mut step = 0;
    while step < step_limit {
        step += 1;
        let (fval, grad) = closure(params)?;
        if grad.len() != n {
            return Err(BotorchError::DimensionMismatch(format!(
                "the closure returned {} gradient entries for {n} parameters",
                grad.len()
            )));
        }
        let runtime = start.elapsed().as_secs_f64();
        let mut result = OptimizationResult {
            step,
            fval,
            status: OptimizationStatus::Running,
            runtime: Some(runtime),
            message: None,
        };
        if let Some(c) = criterion.as_mut() {
            if c.evaluate(&[fval]) {
                result.status = OptimizationStatus::Stopped;
                result.message =
                    Some("`torch_minimize` stopped due to `stopping_criterion`.".into());
            }
        }
        if let Some(t) = options.timeout_sec {
            if runtime >= t {
                result.status = OptimizationStatus::Stopped;
                result.message = Some(format!(
                    "`torch_minimize` stopped due to timeout after {runtime} seconds."
                ));
            }
        }
        if let Some(cb) = callback.as_mut() {
            cb(params, &result)?;
        }
        if result.status != OptimizationStatus::Running {
            result.runtime = Some(start.elapsed().as_secs_f64());
            return Ok(result);
        }
        adam.step(params, &grad);
        for i in 0..n {
            // torch.clamp(min=lower, max=upper) (NaN stays NaN).
            if !params[i].is_nan() {
                params[i] = params[i].max(lower[i]).min(upper[i]);
            }
        }
        last = Some(result);
    }
    let _ = last;
    let (fval, _) = closure(params)?;
    Ok(OptimizationResult {
        step,
        fval,
        status: OptimizationStatus::Stopped,
        runtime: Some(start.elapsed().as_secs_f64()),
        message: Some(format!(
            "`torch_minimize` stopped after reaching step_limit={step_limit}."
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_with_bounds_and_limits() {
        let mut x = vec![2.0, -3.0];
        let opts = TorchMinimizeOptions {
            adam: AdamOptions {
                lr: 0.05,
                ..Default::default()
            },
            step_limit: Some(2000),
            stopping_criterion: None,
            ..Default::default()
        };
        let res = torch_minimize(
            |x: &[f64]| {
                Ok((
                    x[0] * x[0] + (x[1] + 1.0).powi(2),
                    vec![2.0 * x[0], 2.0 * (x[1] + 1.0)],
                ))
            },
            &mut x,
            &[-10.0, -0.5],
            &[10.0, 10.0],
            &opts,
            None,
        )
        .unwrap();
        assert_eq!(res.status, OptimizationStatus::Stopped);
        assert!(x[0].abs() < 1e-2, "{x:?}");
        assert_eq!(x[1], -0.5);
        let none = TorchMinimizeOptions {
            stopping_criterion: None,
            ..Default::default()
        };
        assert!(torch_minimize(
            |_: &[f64]| Ok((0.0, vec![0.0])),
            &mut [0.0],
            &[0.0],
            &[1.0],
            &none,
            None
        )
        .is_err());
        let limited = TorchMinimizeOptions {
            step_limit: Some(3),
            stopping_criterion: None,
            ..Default::default()
        };
        let res = torch_minimize(
            |x: &[f64]| Ok((x[0], vec![1.0])),
            &mut [0.0],
            &[-1.0],
            &[1.0],
            &limited,
            None,
        )
        .unwrap();
        assert_eq!(res.step, 3);
        assert!(res.message.unwrap().contains("step_limit=3"));
    }
}
