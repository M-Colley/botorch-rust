//! Active learning (`botorch.acquisition.active_learning`).
//!
//! [`QNegIntegratedPosteriorVariance`] (qNIPV) is the negative posterior
//! variance (without observation noise), averaged over a set of Monte-Carlo
//! integration points, of the model after observing the candidates (and
//! pending points). For exact GPs the variance after conditioning does not
//! depend on the observed values: with the current posterior's covariance
//! `Σ` and the noisy covariance `A = Σ(X, X) + N` of the candidates,
//!
//! ```text
//! σ²_new(z) = σ²(z) - Σ(z, X) A⁻¹ Σ(X, z),
//! ```
//!
//! which is what BoTorch computes with a fantasy model. The variances of the
//! integration points and their cross-covariances are computed from cached
//! quantities (see [`SingleTaskGP::cross_posterior`]); gradients are exact.

use std::sync::{Arc, RwLock};

use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock};
use crate::acquisition::objective::ScalarizedPosteriorTransform;
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::psd_safe_cholesky;
use crate::models::gp::BaselineCache;
use crate::models::{Model, SingleTaskGP};
use crate::posterior::MIN_VARIANCE;
use crate::{Matrix, Vector};

/// The integration points of one output GP: the cache of their
/// cross-covariances and their current variances.
#[derive(Debug)]
struct OutputPoints {
    cache: BaselineCache,
    variance: Vector,
}

/// BoTorch's `qNegIntegratedPosteriorVariance`: see the [module
/// documentation](self).
#[derive(Debug)]
pub struct QNegIntegratedPosteriorVariance {
    model: Arc<dyn Model>,
    mc_points: Matrix,
    transform: Option<ScalarizedPosteriorTransform>,
    outputs: Vec<OutputPoints>,
    x_pending: RwLock<Option<Matrix>>,
}

impl QNegIntegratedPosteriorVariance {
    /// `qNegIntegratedPosteriorVariance(model, mc_points,
    /// posterior_transform, X_pending)` for a model of exact GPs and the
    /// integration points `mc_points` (`N × d`). BoTorch's `sampler` of the
    /// fantasies does not change the value (the variance after conditioning
    /// does not depend on the fantasized values).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for models without independent exact GP
    /// outputs, batched models and multi-output models without a posterior
    /// transform; [`BotorchError::DimensionMismatch`] for inputs of the wrong
    /// dimension.
    pub fn new(
        model: Arc<dyn Model>,
        mc_points: &Matrix,
        posterior_transform: Option<ScalarizedPosteriorTransform>,
        x_pending: Option<Matrix>,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        let gps = model.output_gps().ok_or_else(|| {
            BotorchError::Unsupported(
                "qNegIntegratedPosteriorVariance requires a model of exact GPs (SingleTaskGP or \
                 ModelListGP)"
                    .into(),
            )
        })?;
        let d = model.dim();
        if mc_points.ncols() != d || mc_points.nrows() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "mc_points must be a non-empty `N x {d}` matrix, got {}x{}",
                mc_points.nrows(),
                mc_points.ncols()
            )));
        }
        match &posterior_transform {
            Some(t) if t.num_outputs() != gps.len() => {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Output shape not equal to that of weights. Output shape is {} and weights \
                     are [{}]",
                    gps.len(),
                    t.num_outputs()
                )))
            }
            None if gps.len() != 1 => {
                return Err(BotorchError::Unsupported(
                    "Must specify a posterior transform when using a multi-output model.".into(),
                ))
            }
            _ => {}
        }
        if let Some(p) = &x_pending {
            check_pending(p, d)?;
        }
        let outputs = gps
            .iter()
            .map(|g| {
                let pp = g.point_posteriors(mc_points)?;
                Ok(OutputPoints {
                    cache: g.baseline_cache(mc_points)?,
                    variance: Vector::from_fn(mc_points.nrows(), |i, _| pp.covariance[(i, 0)]),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            model,
            mc_points: mc_points.clone(),
            transform: posterior_transform,
            outputs,
            x_pending: RwLock::new(x_pending),
        })
    }

    /// The integration points.
    pub fn mc_points(&self) -> &Matrix {
        &self.mc_points
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let xf = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let gps: Vec<&SingleTaskGP> = self
            .model
            .output_gps()
            .ok_or_else(|| BotorchError::Unsupported("the model lost its output GPs".into()))?;
        let n = self.mc_points.nrows();
        // Per output: the cross-covariances with the integration points, the
        // factor of the noisy covariance of the candidates and the variances
        // after conditioning.
        let mut parts = Vec::with_capacity(gps.len());
        let mut var_total = Vector::zeros(n);
        for (t, (g, op)) in gps.iter().zip(&self.outputs).enumerate() {
            let cp = g.cross_posterior(&op.cache, &xf)?;
            let mut a = cp.covariance.clone();
            let noise = g.observation_noise();
            for i in 0..a.nrows() {
                a[(i, i)] += noise;
            }
            let chol = psd_safe_cholesky(&a)?;
            let mut w = cp.cross.clone();
            chol.solve_lower_mut(&mut w);
            let w2 = self
                .transform
                .as_ref()
                .map_or(1.0, |tr| tr.weights()[t].powi(2));
            for i in 0..n {
                let red: f64 = w.column(i).iter().map(|v| v * v).sum();
                var_total[i] += w2 * (op.variance[i] - red);
            }
            parts.push((cp, chol, w, w2));
        }
        // BoTorch's `posterior.variance` clamp and the mean.
        let clamped: Vec<bool> = var_total.iter().map(|&v| v <= MIN_VARIANCE).collect();
        let value = -var_total.iter().map(|&v| v.max(MIN_VARIANCE)).sum::<f64>() / n as f64;
        if !with_grad {
            return Ok((value, None));
        }
        // d value / d var_new_i = -1/N (unless clamped); var_new_i = v_i -
        // c_iᵀ A⁻¹ c_i.
        let mut grad = Matrix::zeros(q, d);
        for (((cp, chol, w, w2), g), op) in parts.iter().zip(&gps).zip(&self.outputs) {
            let nq = cp.covariance.nrows();
            // u = A⁻¹ C (nq × N).
            let mut u = w.clone();
            chol.solve_lower_transpose_mut(&mut u);
            let mut cross_bar = Matrix::zeros(nq, n);
            let mut cov_bar = Matrix::zeros(nq, nq);
            for i in 0..n {
                if clamped[i] {
                    continue;
                }
                let gi = -w2 / n as f64;
                let ui = u.column(i);
                for r in 0..nq {
                    cross_bar[(r, i)] = -2.0 * gi * ui[r];
                }
                cov_bar.ger(gi, &ui, &ui, 1.0);
            }
            let gx =
                g.cross_posterior_vjp(&op.cache, &xf, &Vector::zeros(nq), &cross_bar, &cov_bar)?;
            grad += gx.rows(0, q);
        }
        Ok((value, Some(grad)))
    }
}

impl AcquisitionFunction for QNegIntegratedPosteriorVariance {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.x_pending).clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }
}
