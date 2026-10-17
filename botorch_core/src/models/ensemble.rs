//! The Gaussian moments of an ensemble model (BoTorch's `EnsemblePosterior`
//! as analytic acquisition functions see it).
//!
//! An ensemble model (BoTorch's `EnsembleModel`, e.g. a deep ensemble) has
//! `s` deterministic members. BoTorch's analytic acquisition functions use
//! the `mean` and `variance` of its `EnsemblePosterior`: the weighted mean
//! `μ = Σ_k w_k y_k` and the unbiased weighted variance
//! `Σ_k w_k (y_k - μ)² / (1 - Σ_k w_k²)` of the members' values `y_k` at
//! every point (normalized weights `w_k`). [`EnsembleMomentsModel`] is that
//! Gaussian, with the values of all members given by a deterministic model
//! with `s·m` outputs (member `k`, output `t` at output `k·m + t`). Points are
//! independent (zero covariance between points and outputs); Monte-Carlo
//! acquisition functions sample the members instead (see
//! [`crate::acquisition::EnsembleAverage`]).

use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::{Matrix, Vector};

/// The mean and variance of an ensemble of deterministic members (see the
/// [module documentation](self)).
#[derive(Clone, Debug)]
pub struct EnsembleMomentsModel {
    members: Arc<dyn Model>,
    size: usize,
    weights: Vec<f64>,
}

impl EnsembleMomentsModel {
    /// The moments of the `size` members whose values are the `size·m`
    /// outputs of `members`, with the (unnormalized) `weights` (uniform if
    /// `None`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if the number of outputs is not a
    /// multiple of `size`, or for weights that are not one finite,
    /// non-negative value per member with a positive sum.
    pub fn new(members: Arc<dyn Model>, size: usize, weights: Option<Vec<f64>>) -> Result<Self> {
        if size == 0 || !members.num_outputs().is_multiple_of(size) {
            return Err(BotorchError::InvalidArgument(format!(
                "an ensemble of {size} members needs a multiple of {size} outputs, got {}",
                members.num_outputs()
            )));
        }
        let w = weights.unwrap_or_else(|| vec![1.0; size]);
        if w.len() != size || w.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(BotorchError::InvalidArgument(format!(
                "expected {size} finite, non-negative ensemble weights, got {w:?}"
            )));
        }
        let total: f64 = w.iter().sum();
        if total <= 0.0 {
            return Err(BotorchError::InvalidArgument(
                "the ensemble weights are all zero".into(),
            ));
        }
        Ok(Self {
            members,
            size,
            weights: w.iter().map(|v| v / total).collect(),
        })
    }

    /// The moments of every point and output from the members' values
    /// (output-major, `(k·m + t)·q + p`): `(μ, σ²)` in output-major order
    /// (`t·q + p`) and, per value, `∂μ/∂y` (= `w_k`) and `∂σ²/∂y`.
    fn moments(&self, y: &Vector, q: usize) -> (Vector, Vector, Vec<f64>) {
        let (s, m) = (self.size, self.num_outputs());
        let denom = 1.0 - self.weights.iter().map(|w| w * w).sum::<f64>();
        let mut mean = Vector::zeros(q * m);
        let mut var = Vector::zeros(q * m);
        let mut dvar_dy = vec![0.0; y.len()];
        for t in 0..m {
            for p in 0..q {
                let i = t * q + p;
                let mu: f64 = (0..s)
                    .map(|k| self.weights[k] * y[(k * m + t) * q + p])
                    .sum();
                mean[i] = mu;
                if s > 1 {
                    let mut v = 0.0;
                    for k in 0..s {
                        let dev = y[(k * m + t) * q + p] - mu;
                        v += self.weights[k] * dev * dev;
                        // d/dy_k: 2 w_k dev_k / denom (the mean's derivative
                        // cancels: Σ_k w_k dev_k = 0).
                        dvar_dy[(k * m + t) * q + p] = 2.0 * self.weights[k] * dev / denom;
                    }
                    var[i] = v / denom;
                }
            }
        }
        (mean, var, dvar_dy)
    }
}

impl Model for EnsembleMomentsModel {
    fn dim(&self) -> usize {
        self.members.dim()
    }

    fn num_outputs(&self) -> usize {
        self.members.num_outputs() / self.size
    }

    fn batch_shape(&self) -> &[usize] {
        self.members.batch_shape()
    }

    fn points_per_input(&self) -> usize {
        self.members.points_per_input()
    }

    /// BoTorch's `EnsemblePosterior.variance` is not clamped.
    fn clamps_variance(&self) -> bool {
        false
    }

    fn posterior(&self, x: &Matrix, _observation_noise: bool) -> Result<Posterior> {
        let y = self.members.posterior(x, false)?.mean;
        let q = x.nrows() * self.members.points_per_input();
        let (mean, var, _) = self.moments(&y, q);
        Ok(Posterior::multi_output(
            mean,
            Matrix::from_diagonal(&var),
            self.num_outputs(),
        ))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let g = self.members.joint_posterior_with_grad(x)?;
        let q = x.nrows() * self.members.points_per_input();
        let (mean, var, dvar_dy) = self.moments(&g.mean, q);
        let (s, m) = (self.size, self.num_outputs());
        let n = q * m;
        let mut dmean = Vec::with_capacity(g.dmean.len());
        let mut dcov = Vec::with_capacity(g.dmean.len());
        for dy in &g.dmean {
            let mut dm = Vector::zeros(n);
            let mut dc = Matrix::zeros(n, n);
            for t in 0..m {
                for p in 0..q {
                    let i = t * q + p;
                    for k in 0..s {
                        let j = (k * m + t) * q + p;
                        dm[i] += self.weights[k] * dy[j];
                        dc[(i, i)] += dvar_dy[j] * dy[j];
                    }
                }
            }
            dmean.push(dm);
            dcov.push(dc);
        }
        Ok(JointPosteriorGrad {
            mean,
            covariance: Matrix::from_diagonal(&var),
            dmean,
            dcov,
            num_outputs: m,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::AffineDeterministicModel;

    #[test]
    fn moments_and_gradients() {
        // Three members of an affine ensemble with one output.
        let a = Matrix::from_row_slice(2, 3, &[1.0, -0.5, 0.25, 2.0, 0.0, -1.5]);
        let b = Vector::from_vec(vec![0.1, 0.2, -0.3]);
        let members: Arc<dyn Model> = Arc::new(AffineDeterministicModel::new(a, b).unwrap());
        let model =
            EnsembleMomentsModel::new(members.clone(), 3, Some(vec![1.0, 2.0, 1.0])).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.7, -0.4, 1.2]);
        let p = model.posterior(&x, false).unwrap();
        let y = members.posterior(&x, false).unwrap().mean;
        let w = [0.25, 0.5, 0.25];
        for pt in 0..2 {
            let vals: Vec<f64> = (0..3).map(|k| y[k * 2 + pt]).collect();
            let mu: f64 = vals.iter().zip(&w).map(|(v, w)| v * w).sum();
            let var: f64 = vals
                .iter()
                .zip(&w)
                .map(|(v, w)| w * (v - mu) * (v - mu))
                .sum::<f64>()
                / (1.0 - 0.375);
            assert!((p.mean[pt] - mu).abs() < 1e-14);
            assert!((p.covariance[(pt, pt)] - var).abs() < 1e-14);
        }
        let g = model.joint_posterior_with_grad(&x).unwrap();
        let h = 1e-6;
        for j in 0..4 {
            let (pt, k) = (j / 2, j % 2);
            let mut xp = x.clone();
            xp[(pt, k)] += h;
            let mut xm = x.clone();
            xm[(pt, k)] -= h;
            let pp = model.posterior(&xp, false).unwrap();
            let pm = model.posterior(&xm, false).unwrap();
            let fd_v = (pp.covariance[(pt, pt)] - pm.covariance[(pt, pt)]) / (2.0 * h);
            let fd_m = (pp.mean[pt] - pm.mean[pt]) / (2.0 * h);
            assert!((g.dcov[j][(pt, pt)] - fd_v).abs() < 1e-6 * fd_v.abs().max(1.0));
            assert!((g.dmean[j][pt] - fd_m).abs() < 1e-6 * fd_m.abs().max(1.0));
        }
        assert!(EnsembleMomentsModel::new(members, 2, None).is_err());
    }
}
