//! Monte-Carlo LogEI acquisition functions mirroring
//! `botorch.acquisition.logei` (Ament et al., *Unexpected Improvements to
//! Expected Improvement for Bayesian Optimization*, NeurIPS 2023).
//!
//! For posterior samples `y_{js}` of the candidate points, qLogEI computes
//!
//! ```text
//! α(X) = logmeanexp_s( fatmax_j( log_fatplus(y_js - best_f_s, τ_relu), τ_max ) )
//! ```
//!
//! (with `log_softplus` and `smooth_amax` instead when `fat = false`), i.e. a
//! smooth, numerically robust logarithm of qEI.

use std::sync::Arc;

use crate::acquisition::monte_carlo::{McAcquisitionOptions, McCore, Reduction};
use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::safe_math::{
    fatmax, fatmax_grad_into, log_fatplus, log_fatplus_with_grad, log_softplus,
    log_softplus_with_grad, logmeanexp, logmeanexp_with_grad, smooth_amax, smooth_amax_with_grad,
    ALPHA,
};
use crate::Matrix;

/// Default temperature of the smooth ReLU (`logei.TAU_RELU`).
pub const TAU_RELU: f64 = 1e-6;
/// Default temperature of the smooth maximum over the `q` points
/// (`logei.TAU_MAX`).
pub const TAU_MAX: f64 = 1e-2;

fn check_tau(tau: f64, name: &str) -> Result<f64> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(BotorchError::InvalidArgument(format!(
            "{name} must be positive and finite, got {tau}"
        )));
    }
    Ok(tau)
}

/// The log-improvement reduction of [`QLogExpectedImprovement`] for samples
/// `y` (`points × S`) and per-sample incumbents `best_f(s)`.
fn log_improvement_reduce<B>(
    y: &Matrix,
    best_f: B,
    fat: bool,
    tau_relu: f64,
    tau_max: f64,
    with_grad: bool,
) -> Reduction
where
    B: Fn(usize) -> f64,
{
    let (n, s) = y.shape();
    let mut r = vec![0.0; s];
    let mut l = vec![0.0; n];
    if !with_grad {
        for (c, rc) in r.iter_mut().enumerate() {
            let b = best_f(c);
            for (i, li) in l.iter_mut().enumerate() {
                let z = y[(i, c)] - b;
                *li = if fat {
                    log_fatplus(z, tau_relu)
                } else {
                    log_softplus(z, tau_relu)
                };
            }
            *rc = if fat {
                fatmax(&l, tau_max, ALPHA)
            } else {
                smooth_amax(&l, tau_max)
            };
        }
        return (logmeanexp(&r), None);
    }
    let mut g = Matrix::zeros(n, s);
    let mut dl = vec![0.0; n];
    let mut rg = vec![0.0; n];
    for (c, rc) in r.iter_mut().enumerate() {
        let b = best_f(c);
        for i in 0..n {
            let z = y[(i, c)] - b;
            let (v, d) = if fat {
                log_fatplus_with_grad(z, tau_relu)
            } else {
                log_softplus_with_grad(z, tau_relu)
            };
            l[i] = v;
            dl[i] = d;
        }
        *rc = if fat {
            fatmax_grad_into(&l, tau_max, ALPHA, &mut rg)
        } else {
            let (v, gr) = smooth_amax_with_grad(&l, tau_max);
            rg.copy_from_slice(&gr);
            v
        };
        for i in 0..n {
            g[(i, c)] = rg[i] * dl[i];
        }
    }
    let (value, w) = logmeanexp_with_grad(&r);
    for c in 0..s {
        for i in 0..n {
            g[(i, c)] *= w[c];
        }
    }
    (value, Some(g))
}

/// Options of [`QLogExpectedImprovement`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QLogExpectedImprovementOptions {
    /// The sampler of base samples (default: 512 Sobol samples, random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
    /// Use the fat-tailed `fatplus` / `fatmax` approximations (default `true`)
    /// instead of `softplus` / `logsumexp`.
    pub fat: bool,
    /// Temperature of the smooth maximum over the `q` points (default `1e-2`).
    pub tau_max: f64,
    /// Temperature of the smooth ReLU (default `1e-6`).
    pub tau_relu: f64,
}

impl Default for QLogExpectedImprovementOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            x_pending: None,
            fat: true,
            tau_max: TAU_MAX,
            tau_relu: TAU_RELU,
        }
    }
}

/// MC-based batch Log Expected Improvement, a smooth approximation of
/// `log qEI(X) = log E[max(max_j y_j - best_f, 0)]` (BoTorch's
/// `qLogExpectedImprovement`).
#[derive(Debug)]
pub struct QLogExpectedImprovement {
    core: McCore,
    best_f: f64,
    fat: bool,
    tau_max: f64,
    tau_relu: f64,
}

impl QLogExpectedImprovement {
    /// Creates `qLogExpectedImprovement(model, best_f, ...)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models,
    /// [`BotorchError::InvalidArgument`] for a NaN `best_f` or non-positive
    /// temperatures and [`BotorchError::DimensionMismatch`] for pending points
    /// of the wrong dimension.
    pub fn new(
        model: Arc<dyn Model>,
        best_f: f64,
        options: QLogExpectedImprovementOptions,
    ) -> Result<Self> {
        if best_f.is_nan() {
            return Err(BotorchError::InvalidArgument(
                "best_f must not be NaN".into(),
            ));
        }
        let tau_max = check_tau(options.tau_max, "tau_max")?;
        let tau_relu = check_tau(options.tau_relu, "tau_relu")?;
        let core = McCore::new(
            model,
            McAcquisitionOptions {
                sampler: options.sampler,
                x_pending: options.x_pending,
            },
        )?;
        Ok(Self {
            core,
            best_f,
            fat: options.fat,
            tau_max,
            tau_relu,
        })
    }

    /// The incumbent value `best_f`.
    pub fn best_f(&self) -> f64 {
        self.best_f
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.core.sampler
    }

    fn reduce(&self, y: &Matrix, with_grad: bool) -> Reduction {
        log_improvement_reduce(
            y,
            |_| self.best_f,
            self.fat,
            self.tau_relu,
            self.tau_max,
            with_grad,
        )
    }
}

impl AcquisitionFunction for QLogExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.core.evaluate(x, false, |y, g| self.reduce(y, g))?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.core.evaluate(x, true, |y, g| self.reduce(y, g))?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.core.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.core.set_x_pending(x_pending)
    }

    fn is_log(&self) -> bool {
        true
    }
}
