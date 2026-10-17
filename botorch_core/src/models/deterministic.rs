//! Deterministic models (`botorch.models.deterministic`):
//! [`PosteriorMeanModel`], the posterior mean of another model as a model;
//! [`AffineDeterministicModel`], `b + X a`; [`FixedSingleSampleModel`],
//! `mean + sqrt(variance + jitter) * w` of another model's posterior.
//!
//! BoTorch's deterministic models return an `EnsemblePosterior` of a single
//! member: the mean is the model's value, the variance is exactly zero (not
//! clamped like a GPyTorch posterior's), every Monte-Carlo sample equals the
//! mean and `observation_noise=True` is ignored. The models of this module
//! report [`Model::is_deterministic`], which the acquisition functions use to
//! sample with a zero covariance root (so the sampler's base samples do not
//! matter) and to skip GPyTorch's variance clamp; gradients flow through the
//! mean only.

use std::ops::Range;
use std::sync::Arc;

use crate::error::Result;
use crate::models::model::{check_vjp_shapes_expanded, clamp_variance};
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::{Matrix, Vector};

/// A deterministic model whose value is the posterior mean of another model
/// (BoTorch's `PosteriorMeanModel(model)`, whose `forward(X)` is
/// `model.posterior(X).mean`).
///
/// It has the wrapped model's input dimension, outputs and batch shape; its
/// posterior has the wrapped model's mean and a zero covariance.
#[derive(Clone, Debug)]
pub struct PosteriorMeanModel {
    model: Arc<dyn Model>,
}

impl PosteriorMeanModel {
    /// The posterior mean of `model`.
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }

    /// The wrapped model.
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

/// `p` with a zero covariance.
fn point_mass(mut p: Posterior) -> Posterior {
    p.covariance.fill(0.0);
    p
}

/// `g` with zero covariances and covariance derivatives.
fn point_mass_grad(mut g: JointPosteriorGrad) -> JointPosteriorGrad {
    g.covariance.fill(0.0);
    for c in &mut g.dcov {
        c.fill(0.0);
    }
    g
}

impl Model for PosteriorMeanModel {
    fn dim(&self) -> usize {
        self.model.dim()
    }

    fn num_outputs(&self) -> usize {
        self.model.num_outputs()
    }

    fn batch_shape(&self) -> &[usize] {
        self.model.batch_shape()
    }

    fn points_per_input(&self) -> usize {
        self.model.points_per_input()
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    /// The wrapped model's posterior mean with a zero covariance;
    /// `observation_noise` is ignored, as by BoTorch's ensemble models.
    fn posterior(&self, x: &Matrix, _observation_noise: bool) -> Result<Posterior> {
        Ok(point_mass(self.model.posterior(x, false)?))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        Ok(point_mass_grad(self.model.joint_posterior_with_grad(x)?))
    }

    /// The vector-Jacobian product of the mean (the covariance is constant).
    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_outputs(),
            self.points_per_input(),
        )?;
        let zero = Matrix::zeros(covariance_adjoint.nrows(), covariance_adjoint.ncols());
        self.model.joint_posterior_vjp(x, rows, mean_adjoint, &zero)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        let mut pp = self.model.point_posteriors(x)?;
        pp.covariance.fill(0.0);
        Ok(pp)
    }

    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        Ok(point_mass_grad(self.model.point_posterior_with_grad(x)?))
    }
}

/// BoTorch's `AffineDeterministicModel(a, b)`: the deterministic model
/// `y[..., t] = b[t] + Σ_i a[i, t] X[..., i]` with weights `a` (`d × m`) and
/// offsets `b` (length `m`).
#[derive(Clone, Debug)]
pub struct AffineDeterministicModel {
    a: Matrix,
    b: Vector,
}

impl AffineDeterministicModel {
    /// The model with weights `a` (`d × m`) and offsets `b` (length `m`).
    ///
    /// # Errors
    /// [`crate::BotorchError::DimensionMismatch`] if `b` does not have one
    /// value per output or `a` is empty.
    pub fn new(a: Matrix, b: Vector) -> Result<Self> {
        if a.nrows() == 0 || a.ncols() == 0 || b.len() != a.ncols() {
            return Err(crate::BotorchError::DimensionMismatch(format!(
                "AffineDeterministicModel needs d x m weights and m offsets, got {}x{} and {}",
                a.nrows(),
                a.ncols(),
                b.len()
            )));
        }
        Ok(Self { a, b })
    }

    /// The weights (`d × m`).
    pub fn weights(&self) -> &Matrix {
        &self.a
    }

    /// The offsets (length `m`).
    pub fn offsets(&self) -> &Vector {
        &self.b
    }

    fn check(&self, x: &Matrix) -> Result<()> {
        if x.ncols() != self.a.nrows() {
            return Err(crate::BotorchError::DimensionMismatch(format!(
                "AffineDeterministicModel has {} inputs, got points with {} columns",
                self.a.nrows(),
                x.ncols()
            )));
        }
        Ok(())
    }

    /// The values at the rows of `x` in output-major order (`t·q + p`).
    fn values(&self, x: &Matrix) -> Result<Vector> {
        self.check(x)?;
        let (q, m) = (x.nrows(), self.a.ncols());
        let xa = x * &self.a;
        Ok(Vector::from_fn(q * m, |i, _| {
            let (t, p) = (i / q, i % q);
            self.b[t] + xa[(p, t)]
        }))
    }
}

impl Model for AffineDeterministicModel {
    fn dim(&self) -> usize {
        self.a.nrows()
    }

    fn num_outputs(&self) -> usize {
        self.a.ncols()
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn posterior(&self, x: &Matrix, _observation_noise: bool) -> Result<Posterior> {
        let mean = self.values(x)?;
        let n = mean.len();
        Ok(Posterior::multi_output(
            mean,
            Matrix::zeros(n, n),
            self.num_outputs(),
        ))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let mean = self.values(x)?;
        let (q, d, m) = (x.nrows(), x.ncols(), self.a.ncols());
        let n = q * m;
        let mut dmean = vec![Vector::zeros(n); q * d];
        for p in 0..q {
            for k in 0..d {
                for t in 0..m {
                    dmean[p * d + k][t * q + p] = self.a[(k, t)];
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean,
            covariance: Matrix::zeros(n, n),
            dmean,
            dcov: vec![Matrix::zeros(n, n); q * d],
            num_outputs: m,
        })
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        self.check(x)?;
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_outputs(),
            1,
        )?;
        let (q, d, m) = (x.nrows(), x.ncols(), self.a.ncols());
        let mut out = Matrix::zeros(rows.len(), d);
        for (r, p) in rows.enumerate() {
            for k in 0..d {
                out[(r, k)] = (0..m)
                    .map(|t| mean_adjoint[t * q + p] * self.a[(k, t)])
                    .sum();
            }
        }
        Ok(out)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.check(x)?;
        let m = self.a.ncols();
        let xa = x * &self.a;
        Ok(PointPosteriors {
            mean: Matrix::from_fn(x.nrows(), m, |p, t| self.b[t] + xa[(p, t)]),
            covariance: Matrix::zeros(x.nrows(), m * m),
        })
    }
}

/// BoTorch's `FixedSingleSampleModel(model, w, jitter)`: the deterministic
/// model `y = mean(X) + sqrt(variance(X) + jitter) * w` of another model's
/// posterior at `X` (the joint posterior of the points of `X`, its marginal
/// variances clamped from below like `posterior.variance`; the outputs
/// are treated as uncorrelated), for a fixed vector `w` with one entry per
/// output (or a single entry for all outputs).
#[derive(Clone, Debug)]
pub struct FixedSingleSampleModel {
    model: Arc<dyn Model>,
    w: Vector,
    jitter: f64,
}

impl FixedSingleSampleModel {
    /// The model of the sample `w` (length `m` or 1) of `model`'s
    /// posterior with `jitter` added to the variances.
    ///
    /// # Errors
    /// [`crate::BotorchError::DimensionMismatch`] if `w` does not broadcast
    /// over the outputs; [`crate::BotorchError::InvalidArgument`] for a
    /// negative or non-finite jitter.
    pub fn new(model: Arc<dyn Model>, w: Vector, jitter: f64) -> Result<Self> {
        let m = model.num_outputs();
        if !(w.len() == m || w.len() == 1) {
            return Err(crate::BotorchError::DimensionMismatch(format!(
                "w has {} entries, expected one per output ({m}) or one",
                w.len()
            )));
        }
        if !(jitter.is_finite() && jitter >= 0.0) {
            return Err(crate::BotorchError::InvalidArgument(format!(
                "jitter must be finite and non-negative, got {jitter}"
            )));
        }
        Ok(Self { model, w, jitter })
    }

    /// The wrapped model.
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The fixed sample `w`.
    pub fn w(&self) -> &Vector {
        &self.w
    }

    fn w_of(&self, t: usize) -> f64 {
        if self.w.len() == 1 {
            self.w[0]
        } else {
            self.w[t]
        }
    }

    /// The values and the derivatives of every value with respect to the
    /// (clamped) variance it depends on: `y_i = μ_i + s_i w_t`,
    /// `∂y_i/∂v_i = w_t / (2 s_i)` (zero where the variance is clamped).
    fn sample_values(&self, mean: &Vector, variance: &[f64], q: usize) -> (Vector, Vec<f64>) {
        let n = mean.len();
        let mut y = Vector::zeros(n);
        let mut dv = vec![0.0; n];
        for i in 0..n {
            let t = i / q;
            let v = variance[i];
            let clamped = clamp_variance(v);
            let s = (clamped + self.jitter).sqrt();
            y[i] = mean[i] + s * self.w_of(t);
            if v > crate::posterior::MIN_VARIANCE {
                dv[i] = self.w_of(t) / (2.0 * s);
            }
        }
        (y, dv)
    }
}

impl Model for FixedSingleSampleModel {
    fn dim(&self) -> usize {
        self.model.dim()
    }

    fn num_outputs(&self) -> usize {
        self.model.num_outputs()
    }

    fn batch_shape(&self) -> &[usize] {
        self.model.batch_shape()
    }

    fn points_per_input(&self) -> usize {
        self.model.points_per_input()
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn posterior(&self, x: &Matrix, _observation_noise: bool) -> Result<Posterior> {
        let p = self.model.posterior(x, false)?;
        let n = p.mean.len();
        let q = n / self.num_outputs().max(1);
        let var: Vec<f64> = (0..n).map(|i| p.covariance[(i, i)]).collect();
        let (y, _) = self.sample_values(&p.mean, &var, q);
        Ok(Posterior::multi_output(
            y,
            Matrix::zeros(n, n),
            self.num_outputs(),
        ))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let g = self.model.joint_posterior_with_grad(x)?;
        let n = g.mean.len();
        let q = n / self.num_outputs().max(1);
        let var: Vec<f64> = (0..n).map(|i| g.covariance[(i, i)]).collect();
        let (y, dv) = self.sample_values(&g.mean, &var, q);
        let dmean = g
            .dmean
            .iter()
            .zip(&g.dcov)
            .map(|(dm, dc)| Vector::from_fn(n, |i, _| dm[i] + dv[i] * dc[(i, i)]))
            .collect::<Vec<_>>();
        let k = dmean.len();
        Ok(JointPosteriorGrad {
            mean: y,
            covariance: Matrix::zeros(n, n),
            dmean,
            dcov: vec![Matrix::zeros(n, n); k],
            num_outputs: g.num_outputs,
        })
    }

    /// The vector-Jacobian product through the mean and the diagonal of the
    /// wrapped model's covariance.
    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_outputs(),
            self.points_per_input(),
        )?;
        let p = self.model.posterior(x, false)?;
        let n = p.mean.len();
        let q = n / self.num_outputs().max(1);
        let var: Vec<f64> = (0..n).map(|i| p.covariance[(i, i)]).collect();
        let (_, dv) = self.sample_values(&p.mean, &var, q);
        let cov_adjoint = Matrix::from_fn(
            n,
            n,
            |i, j| {
                if i == j {
                    mean_adjoint[i] * dv[i]
                } else {
                    0.0
                }
            },
        );
        self.model
            .joint_posterior_vjp(x, rows, mean_adjoint, &cov_adjoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BatchedMultiOutputGP, ModelListGP, SingleTaskGP, SingleTaskGPOptions};

    fn list() -> Arc<dyn Model> {
        let x = Matrix::from_fn(8, 2, |i, j| ((i * 5 + j * 3) % 8) as f64 / 8.0);
        let y0 = Matrix::from_fn(8, 1, |i, _| (4.0 * x[(i, 0)]).sin() + x[(i, 1)]);
        let y1 = Matrix::from_fn(8, 2, |i, t| (x[(i, 1)] * (2.0 + t as f64)).cos());
        let a = BatchedMultiOutputGP::multi_output(x.clone(), y0, SingleTaskGPOptions::default())
            .unwrap();
        let b = BatchedMultiOutputGP::multi_output(x, y1, SingleTaskGPOptions::default()).unwrap();
        Arc::new(ModelListGP::from_shared(vec![Arc::new(a), Arc::new(b)]).unwrap())
    }

    #[test]
    fn posterior_is_the_mean_with_zero_covariance() {
        let inner = list();
        let pm = PosteriorMeanModel::new(inner.clone());
        assert!(pm.is_deterministic() && !inner.is_deterministic());
        assert_eq!((pm.dim(), pm.num_outputs()), (2, 3));
        let x = Matrix::from_row_slice(3, 2, &[0.1, 0.4, 0.6, 0.2, 0.9, 0.95]);
        let p = inner.posterior(&x, false).unwrap();
        for noise in [false, true] {
            let q = pm.posterior(&x, noise).unwrap();
            assert_eq!(q.mean, p.mean);
            assert!(q.covariance.iter().all(|&v| v == 0.0));
        }
        let pp = pm.point_posteriors(&x).unwrap();
        assert_eq!(pp.mean, inner.point_posteriors(&x).unwrap().mean);
        assert!(pp.covariance.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradients_are_those_of_the_mean() {
        let inner = list();
        let pm = PosteriorMeanModel::new(inner.clone());
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.7, 0.8, 0.15]);
        let g = pm.joint_posterior_with_grad(&x).unwrap();
        let gi = inner.joint_posterior_with_grad(&x).unwrap();
        assert_eq!(g.dmean, gi.dmean);
        assert!(g.dcov.iter().all(|c| c.iter().all(|&v| v == 0.0)));
        let ma = Vector::from_fn(6, |i, _| 0.3 - 0.1 * i as f64);
        let ca = Matrix::from_fn(6, 6, |i, j| (i + 2 * j) as f64);
        let vjp = pm.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap();
        let expected = inner
            .joint_posterior_vjp(&x, 0..2, &ma, &Matrix::zeros(6, 6))
            .unwrap();
        assert_eq!(vjp, expected);
        for j in 0..4 {
            let fwd = ma.dot(&g.dmean[j]);
            assert!((vjp[(j / 2, j % 2)] - fwd).abs() < 1e-12 * fwd.abs().max(1.0));
        }
    }

    fn fd_check(model: &dyn Model, x: &Matrix) {
        let g = model.joint_posterior_with_grad(x).unwrap();
        let n = g.mean.len();
        let ma = Vector::from_fn(n, |i, _| 0.7 - 0.2 * i as f64);
        let zero = Matrix::zeros(n, n);
        let vjp = model
            .joint_posterior_vjp(x, 0..x.nrows(), &ma, &zero)
            .unwrap();
        let h = 1e-6;
        for p in 0..x.nrows() {
            for k in 0..x.ncols() {
                let mut xp = x.clone();
                xp[(p, k)] += h;
                let mut xm = x.clone();
                xm[(p, k)] -= h;
                let fp = model.posterior(&xp, false).unwrap().mean.dot(&ma);
                let fm = model.posterior(&xm, false).unwrap().mean.dot(&ma);
                let fd = (fp - fm) / (2.0 * h);
                let fwd = ma.dot(&g.dmean[p * x.ncols() + k]);
                assert!(
                    (fd - vjp[(p, k)]).abs() < 1e-6 * fd.abs().max(1.0),
                    "{fd} {}",
                    vjp[(p, k)]
                );
                assert!((fwd - vjp[(p, k)]).abs() < 1e-10 * fwd.abs().max(1.0));
            }
        }
    }

    #[test]
    fn affine_model_values_and_gradients() {
        let a = Matrix::from_row_slice(2, 3, &[1.0, -0.5, 0.25, 2.0, 0.0, -1.5]);
        let b = Vector::from_vec(vec![0.1, 0.2, -0.3]);
        let model = AffineDeterministicModel::new(a.clone(), b.clone()).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.7, -0.4, 1.2]);
        let p = model.posterior(&x, true).unwrap();
        for t in 0..3 {
            for i in 0..2 {
                let expected = b[t] + x[(i, 0)] * a[(0, t)] + x[(i, 1)] * a[(1, t)];
                assert!((p.mean[t * 2 + i] - expected).abs() < 1e-15);
            }
        }
        assert!(p.covariance.iter().all(|&v| v == 0.0));
        fd_check(&model, &x);
        assert!(AffineDeterministicModel::new(a, Vector::zeros(2)).is_err());
    }

    #[test]
    fn fixed_single_sample_values_and_gradients() {
        let inner = list();
        let w = Vector::from_vec(vec![0.5, -1.0, 2.0]);
        let model = FixedSingleSampleModel::new(inner.clone(), w.clone(), 1e-8).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.7, 0.8, 0.15]);
        let p = inner.posterior(&x, false).unwrap();
        let y = model.posterior(&x, false).unwrap();
        for i in 0..6 {
            let v = p.covariance[(i, i)].max(crate::posterior::MIN_VARIANCE);
            let expected = p.mean[i] + (v + 1e-8).sqrt() * w[i / 2];
            assert!((y.mean[i] - expected).abs() < 1e-14 * expected.abs().max(1.0));
        }
        fd_check(&model, &x);
        assert!(FixedSingleSampleModel::new(inner, Vector::zeros(2), 0.0).is_err());
    }

    #[test]
    fn marginals_are_not_clamped() {
        let x = Matrix::from_fn(6, 1, |i, _| i as f64 / 5.0);
        let y = Matrix::from_fn(6, 1, |i, _| (3.0 * x[(i, 0)]).sin());
        let gp = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
        let pm = PosteriorMeanModel::new(Arc::new(gp));
        let xt = Matrix::from_row_slice(2, 1, &[0.25, 0.5]);
        let (_, var) = pm.posterior_marginals(&xt).unwrap();
        assert!(var.iter().all(|&v| v == 0.0));
        let (_, var, _, dvar) = pm.posterior_q1_with_grad(&[0.4]).unwrap();
        assert_eq!((var, dvar[0]), (0.0, 0.0));
    }
}
