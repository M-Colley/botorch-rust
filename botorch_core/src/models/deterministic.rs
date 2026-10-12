//! Deterministic models (`botorch.models.deterministic`):
//! [`PosteriorMeanModel`], the posterior mean of another model as a model.
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
use crate::models::model::check_vjp_shapes_expanded;
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
