//! Posterior transforms (`botorch.acquisition.objective`):
//! [`ScalarizedPosteriorTransform`] and [`ScalarizedModel`], the
//! single-output view of a model through a posterior transform.
//!
//! BoTorch passes `posterior_transform` to `model.posterior(X, ...)`, which
//! scalarizes the Gaussian posterior *before* acquisition functions read its
//! moments or draw samples from it. [`ScalarizedModel`] implements exactly
//! that as a [`Model`] with one output, so every acquisition function
//! supports a posterior transform by being given the wrapped model:
//!
//! ```
//! use std::sync::Arc;
//!
//! use botorch_core::acquisition::objective::{ScalarizedModel, ScalarizedPosteriorTransform};
//! use botorch_core::acquisition::LogExpectedImprovement;
//! use botorch_core::{BatchedMultiOutputGP, Matrix, Model, SingleTaskGPOptions, Vector};
//!
//! let x = Matrix::from_fn(8, 2, |i, j| ((i * 3 + j * 5) % 8) as f64 / 8.0);
//! let y = Matrix::from_fn(8, 2, |i, t| (x[(i, 0)] * (2.0 + t as f64)).sin() + x[(i, 1)]);
//! let model: Arc<dyn Model> =
//!     Arc::new(BatchedMultiOutputGP::multi_output(x, y, SingleTaskGPOptions::default())?);
//! // ExpectedImprovement(model, best_f, posterior_transform=ScalarizedPosteriorTransform(w))
//! let transform = ScalarizedPosteriorTransform::new(Vector::from_vec(vec![0.5, 0.25]), 0.0)?;
//! let scalarized = ScalarizedModel::wrap(model, Some(transform))?;
//! let acqf = LogExpectedImprovement::new(scalarized, 1.0, true)?;
//! # let _ = acqf;
//! # Ok::<(), botorch_core::BotorchError>(())
//! ```
//!
//! Gradients with respect to the inputs are propagated through the (linear)
//! transform exactly.

use std::ops::Range;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::{Matrix, Vector};

/// An affine scalarization of a multi-output posterior (BoTorch's
/// `ScalarizedPosteriorTransform(weights, offset)`): the posterior of
/// `offset + Σ_t weights[t] f_t(x)`, i.e. mean `offset + wᵀμ` and covariance
/// `wᵀ Σ w` per pair of points.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarizedPosteriorTransform {
    weights: Vector,
    offset: f64,
}

impl ScalarizedPosteriorTransform {
    /// A transform with one weight per model output and an `offset` added to
    /// the posterior mean (BoTorch's default offset is 0).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for empty or non-finite weights or a
    /// non-finite offset.
    pub fn new(weights: Vector, offset: f64) -> Result<Self> {
        if weights.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "weights must contain at least one element".into(),
            ));
        }
        if weights.iter().any(|w| !w.is_finite()) || !offset.is_finite() {
            return Err(BotorchError::InvalidArgument(format!(
                "weights and offset must be finite, got {:?} and {offset}",
                weights.as_slice()
            )));
        }
        Ok(Self { weights, offset })
    }

    /// The weights (one per output).
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// The offset.
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Number of outputs `m` the transform applies to.
    pub fn num_outputs(&self) -> usize {
        self.weights.len()
    }

    fn check_outputs(&self, m: usize) -> Result<()> {
        if m != self.weights.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Output shape not equal to that of weights. Output shape is {m} and weights \
                 are [{}]",
                self.weights.len()
            )));
        }
        Ok(())
    }

    /// The transform applied to outcomes `y` (`q × m`): `offset + y w`
    /// (BoTorch's `evaluate`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `y` does not have `m` columns.
    pub fn evaluate(&self, y: &Matrix) -> Result<Vector> {
        self.check_outputs(y.ncols())?;
        Ok((y * &self.weights).add_scalar(self.offset))
    }

    /// `offset + Σ_t w_t v[t·q + p]` for `p < q` (the scalarized mean).
    fn scalarize_mean(&self, v: &Vector, q: usize, with_offset: bool) -> Vector {
        let w = &self.weights;
        let off = if with_offset { self.offset } else { 0.0 };
        if w.len() == 1 {
            return Vector::from_fn(q, |p, _| off + w[0] * v[p]);
        }
        Vector::from_fn(q, |p, _| {
            let mut s = 0.0;
            for (t, wt) in w.iter().enumerate() {
                s += v[t * q + p] * wt;
            }
            off + s
        })
    }

    /// `Σ_{t,t'} w_t w_t' c[(t,p), (t',p')]` (the scalarized covariance),
    /// evaluated like BoTorch's `scalarize_posterior_gpytorch`.
    fn scalarize_cov(&self, c: &Matrix, q: usize) -> Matrix {
        let w = &self.weights;
        let m = w.len();
        if m == 1 {
            let w2 = w[0] * w[0];
            return c.map(|v| w2 * v);
        }
        if q == 1 {
            // wᵀ (C w)
            let cw = c * w;
            return Matrix::from_element(1, 1, w.dot(&cw));
        }
        Matrix::from_fn(q, q, |p, pp| {
            let mut s = 0.0;
            for t in 0..m {
                for tt in 0..m {
                    let v = c[(t * q + p, tt * q + pp)];
                    s += if t == tt {
                        v * (w[t] * w[t])
                    } else {
                        w[t] * v * w[tt]
                    };
                }
            }
            s
        })
    }

    /// The scalarized (single-output) posterior (BoTorch's
    /// `scalarize_posterior`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the posterior does not have `m`
    /// outputs.
    pub fn transform_posterior(&self, posterior: &Posterior) -> Result<Posterior> {
        self.check_outputs(posterior.num_outputs())?;
        if posterior.is_transformed() {
            return Err(BotorchError::Unsupported(SCALARIZE_TRANSFORMED.into()));
        }
        let q = posterior.num_points();
        Ok(Posterior::single_output(
            self.scalarize_mean(&posterior.mean, q, true),
            self.scalarize_cov(&posterior.covariance, q),
        ))
    }

    /// The scalarized posteriors of individual points (unclamped).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of outputs.
    pub fn transform_point_posteriors(&self, pp: &PointPosteriors) -> Result<PointPosteriors> {
        self.check_outputs(pp.num_outputs())?;
        let b = pp.len();
        let m = self.weights.len();
        let mut mean = Vector::zeros(b);
        let mut var = Vector::zeros(b);
        for i in 0..b {
            let mu = Vector::from_fn(m, |t, _| pp.mean[(i, t)]);
            mean[i] = self.scalarize_mean(&mu, 1, true)[0];
            var[i] = self.scalarize_cov(&pp.covariance_of(i), 1)[(0, 0)];
        }
        Ok(PointPosteriors::single_output(&mean, &var))
    }

    /// The scalarized joint posterior with derivatives.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of outputs.
    pub fn transform_grad(&self, g: &JointPosteriorGrad) -> Result<JointPosteriorGrad> {
        self.check_outputs(g.num_outputs)?;
        let q = g.num_points();
        Ok(JointPosteriorGrad {
            mean: self.scalarize_mean(&g.mean, q, true),
            covariance: self.scalarize_cov(&g.covariance, q),
            dmean: g
                .dmean
                .iter()
                .map(|v| self.scalarize_mean(v, q, false))
                .collect(),
            dcov: g.dcov.iter().map(|c| self.scalarize_cov(c, q)).collect(),
            num_outputs: 1,
        })
    }

    /// Pulls adjoints of the scalarized posterior over `q` points back to the
    /// adjoints of the `m`-output posterior (the transpose of the linear map).
    pub(crate) fn pull_back(
        &self,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> (Vector, Matrix) {
        let w = &self.weights;
        let m = w.len();
        let q = mean_adjoint.len();
        let mean = Vector::from_fn(m * q, |i, _| w[i / q] * mean_adjoint[i % q]);
        let cov = Matrix::from_fn(m * q, m * q, |i, j| {
            w[i / q] * w[j / q] * covariance_adjoint[(i % q, j % q)]
        });
        (mean, cov)
    }
}

/// BoTorch's error for scalarizing a `TransformedPosterior` (it has no
/// `distribution`).
pub const SCALARIZE_TRANSFORMED_MESSAGE: &str = SCALARIZE_TRANSFORMED;

/// See [`SCALARIZE_TRANSFORMED_MESSAGE`].
pub(crate) const SCALARIZE_TRANSFORMED: &str = "scalarize_posterior only works with a posterior \
     that has an attribute `distribution`, such as a GPyTorchPosterior, or a posterior that \
     contains sub-posteriors in an attribute `posteriors`, as in a PosteriorList.";

/// A model seen through a [`ScalarizedPosteriorTransform`]: a single-output
/// [`Model`] whose posterior is the scalarized posterior of the wrapped
/// model (what BoTorch's acquisition functions evaluate with
/// `posterior_transform=...`).
///
/// For a model list of single-output models with nonlinear outcome
/// transforms (BoTorch's `PosteriorList` of `TransformedPosterior`s),
/// BoTorch's `scalarize_posterior` builds a Gaussian posterior from the
/// marginal moments of one point (`q = 1`): mean `offset + Σ w_t μ_t`,
/// variance `Σ w_t² σ_t²`; this model does the same.
#[derive(Clone, Debug)]
pub struct ScalarizedModel {
    model: Arc<dyn Model>,
    transform: ScalarizedPosteriorTransform,
    /// Whether the posterior is moment-matched (see above).
    moments: bool,
}

impl ScalarizedModel {
    /// Wraps `model`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] unless the transform has one weight
    /// per model output.
    pub fn new(model: Arc<dyn Model>, transform: ScalarizedPosteriorTransform) -> Result<Self> {
        transform.check_outputs(model.num_outputs())?;
        let transformed = model.transformed_outputs().iter().any(|&t| t);
        let moments = if transformed {
            match model.list_output_sizes() {
                Some(sizes) if sizes.iter().all(|&k| k == 1) => true,
                Some(_) => {
                    return Err(BotorchError::Unsupported(
                        "scalarize_posterior only works with a PosteriorList if each \
                         sub-posterior has one outcome."
                            .into(),
                    ))
                }
                None => return Err(BotorchError::Unsupported(SCALARIZE_TRANSFORMED.into())),
            }
        } else {
            false
        };
        Ok(Self {
            model,
            transform,
            moments,
        })
    }

    /// The moment-matched scalarized posterior at one point and its
    /// derivatives (see the type documentation).
    fn moment_posterior(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        let g = self.model.point_posterior_with_grad(x)?;
        let transformed = self.model.transformed_outputs();
        let w = &self.transform.weights;
        let d = x.len();
        let mut mean = 0.0;
        let mut var = 0.0;
        let mut dmean = vec![0.0; d];
        let mut dvar = vec![0.0; d];
        for (t, &wt) in w.iter().enumerate() {
            mean += g.mean[t] * wt;
            let v = g.covariance[(t, t)];
            // GPyTorchPosterior.variance is clamped; TransformedPosterior's is not.
            let clamped = !transformed[t] && v < crate::posterior::MIN_VARIANCE;
            var += if clamped {
                crate::posterior::MIN_VARIANCE
            } else {
                v
            } * (wt * wt);
            for k in 0..d {
                dmean[k] += g.dmean[k][t] * wt;
                if !clamped {
                    dvar[k] += g.dcov[k][(t, t)] * (wt * wt);
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: Vector::from_element(1, self.transform.offset + mean),
            covariance: Matrix::from_element(1, 1, var),
            dmean: dmean
                .into_iter()
                .map(|v| Vector::from_element(1, v))
                .collect(),
            dcov: dvar
                .into_iter()
                .map(|v| Matrix::from_element(1, 1, v))
                .collect(),
            num_outputs: 1,
        })
    }

    fn require_q1(&self, x: &Matrix, observation_noise: bool) -> Result<()> {
        if x.nrows() * self.model.points_per_input() != 1 {
            return Err(BotorchError::Unsupported(
                "scalarize_posterior only works with a PosteriorList if each sub-posterior has \
                 q=1."
                    .into(),
            ));
        }
        if observation_noise {
            return Err(BotorchError::Unsupported(
                "the moment-matched scalarization of a PosteriorList with observation noise is \
                 not supported"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The model seen through `transform`, or `model` itself without a
    /// transform (BoTorch's optional `posterior_transform` argument).
    ///
    /// # Errors
    /// See [`ScalarizedModel::new`].
    pub fn wrap(
        model: Arc<dyn Model>,
        transform: Option<ScalarizedPosteriorTransform>,
    ) -> Result<Arc<dyn Model>> {
        match transform {
            Some(t) => Ok(Arc::new(Self::new(model, t)?)),
            None => Ok(model),
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The posterior transform.
    pub fn transform(&self) -> &ScalarizedPosteriorTransform {
        &self.transform
    }
}

impl Model for ScalarizedModel {
    fn dim(&self) -> usize {
        self.model.dim()
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn batch_shape(&self) -> &[usize] {
        self.model.batch_shape()
    }

    fn points_per_input(&self) -> usize {
        self.model.points_per_input()
    }

    fn is_deterministic(&self) -> bool {
        self.model.is_deterministic()
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        if self.moments {
            self.require_q1(x, observation_noise)?;
            let g = self.moment_posterior(&x.row(0).iter().copied().collect::<Vec<_>>())?;
            return Ok(Posterior::single_output(g.mean, g.covariance));
        }
        let p = self.model.posterior(x, observation_noise)?;
        self.transform.transform_posterior(&p)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        if self.moments {
            self.require_q1(x, false)?;
            return self.moment_posterior(&x.row(0).iter().copied().collect::<Vec<_>>());
        }
        let g = self.model.joint_posterior_with_grad(x)?;
        self.transform.transform_grad(&g)
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        crate::models::model::check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            1,
            self.model.points_per_input(),
        )?;
        if self.moments {
            let g = self.joint_posterior_with_grad(x)?;
            let d = x.ncols();
            let mut out = Matrix::zeros(rows.len(), d);
            for k in 0..d {
                out[(0, k)] = mean_adjoint[0] * g.dmean[k][0]
                    + covariance_adjoint[(0, 0)] * g.dcov[k][(0, 0)];
            }
            return Ok(out);
        }
        let (ma, ca) = self.transform.pull_back(mean_adjoint, covariance_adjoint);
        self.model.joint_posterior_vjp(x, rows, &ma, &ca)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        if self.moments {
            let b = x.nrows();
            let mut mean = Vector::zeros(b);
            let mut var = Vector::zeros(b);
            for i in 0..b {
                let g = self.moment_posterior(&x.row(i).iter().copied().collect::<Vec<_>>())?;
                mean[i] = g.mean[0];
                var[i] = g.covariance[(0, 0)];
            }
            return Ok(PointPosteriors::single_output(&mean, &var));
        }
        let pp = self.model.point_posteriors(x)?;
        self.transform.transform_point_posteriors(&pp)
    }

    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        if self.moments {
            return self.moment_posterior(x);
        }
        let g = self.model.point_posterior_with_grad(x)?;
        self.transform.transform_grad(&g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BatchedMultiOutputGP, SingleTaskGPOptions};

    fn model() -> Arc<dyn Model> {
        let x = Matrix::from_fn(9, 2, |i, j| ((i * 4 + j * 7) % 9) as f64 / 9.0);
        let y = Matrix::from_fn(9, 3, |i, t| {
            (x[(i, 0)] * (1.5 + t as f64)).cos() + (t as f64) * x[(i, 1)]
        });
        Arc::new(BatchedMultiOutputGP::multi_output(x, y, SingleTaskGPOptions::default()).unwrap())
    }

    #[test]
    fn scalarization_matches_direct_formulas() {
        let m = model();
        let t =
            ScalarizedPosteriorTransform::new(Vector::from_vec(vec![0.5, -1.0, 2.0]), 0.3).unwrap();
        let s = ScalarizedModel::new(m.clone(), t.clone()).unwrap();
        let x = Matrix::from_row_slice(3, 2, &[0.1, 0.4, 0.6, 0.2, 0.9, 0.9]);
        let p = m.posterior(&x, false).unwrap();
        let ps = s.posterior(&x, false).unwrap();
        let w = [0.5, -1.0, 2.0];
        for a in 0..3 {
            let mu: f64 = 0.3 + (0..3).map(|k| w[k] * p.mean[k * 3 + a]).sum::<f64>();
            assert!((ps.mean[a] - mu).abs() < 1e-14);
            for b in 0..3 {
                let c: f64 = (0..3)
                    .map(|k| w[k] * w[k] * p.covariance[(k * 3 + a, k * 3 + b)])
                    .sum();
                assert!((ps.covariance[(a, b)] - c).abs() < 1e-14);
            }
        }
        // Point posteriors agree with the joint posterior's marginals.
        let pp = s.point_posteriors(&x).unwrap();
        for a in 0..3 {
            assert!((pp.mean[(a, 0)] - ps.mean[a]).abs() < 1e-14);
            assert!((pp.variance(a, 0) - ps.covariance[(a, a)]).abs() < 1e-14);
        }
        // evaluate: offset + Y w.
        let y = Matrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        assert!((t.evaluate(&y).unwrap()[0] - (0.3 + 0.5 - 2.0 + 6.0)).abs() < 1e-15);
        assert!(ScalarizedModel::new(
            m,
            ScalarizedPosteriorTransform::new(Vector::from_vec(vec![1.0]), 0.0).unwrap()
        )
        .is_err());
        assert!(ScalarizedPosteriorTransform::new(Vector::zeros(0), 0.0).is_err());
    }

    #[test]
    fn gradients_propagate_through_the_transform() {
        let m = model();
        let t = ScalarizedPosteriorTransform::new(Vector::from_vec(vec![1.5, 0.25, -0.75]), 0.0)
            .unwrap();
        let s = ScalarizedModel::new(m, t).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.7, 0.8, 0.15]);
        let g = s.joint_posterior_with_grad(&x).unwrap();
        // VJP against the forward-mode contraction.
        let ma = Vector::from_vec(vec![0.7, -0.2]);
        let ca = Matrix::from_row_slice(2, 2, &[0.3, 0.1, -0.4, 0.9]);
        let vjp = s.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap();
        for j in 0..4 {
            let expected = ma.dot(&g.dmean[j]) + ca.component_mul(&g.dcov[j]).sum();
            assert!((vjp[(j / 2, j % 2)] - expected).abs() < 1e-12 * expected.abs().max(1.0));
        }
        // Forward mode against central differences of the posterior.
        let h = 1e-6;
        for j in 0..4 {
            let (p, k) = (j / 2, j % 2);
            let mut xp = x.clone();
            xp[(p, k)] += h;
            let mut xm = x.clone();
            xm[(p, k)] -= h;
            let pp = s.posterior(&xp, false).unwrap();
            let pm = s.posterior(&xm, false).unwrap();
            let fd_mean = (&pp.mean - &pm.mean) / (2.0 * h);
            let fd_cov = (&pp.covariance - &pm.covariance) / (2.0 * h);
            assert!((fd_mean - &g.dmean[j]).amax() < 1e-6);
            assert!((fd_cov - &g.dcov[j]).amax() < 1e-6);
        }
    }
}
