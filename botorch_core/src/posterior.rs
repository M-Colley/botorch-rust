//! Gaussian posterior containers returned by the models.
//!
//! # Layout of multi-output posteriors
//!
//! BoTorch represents the joint posterior of a model with `m` independent
//! outputs (a multi-output `SingleTaskGP` or a `ModelListGP`) at `q` points as
//! a `MultitaskMultivariateNormal` with `interleaved=False`: `posterior.mean`
//! is `q × m` and `posterior.distribution.covariance_matrix` is the
//! `(q·m) × (q·m)` covariance of the *output-major* flattening, i.e. index
//! `t·q + p` refers to output `t` at point `p` (a block-diagonal matrix with
//! one `q × q` block per output for independent outputs). The containers of
//! this module use exactly this flattening, so for a single-output model the
//! mean and covariance are the familiar `q`-vector and `q × q` matrix.

use crate::error::{BotorchError, Result};
use crate::transforms::outcome::SampleTransform;
use crate::{Matrix, Vector};

/// Minimum variance returned by [`Posterior::variance`]
/// (`gpytorch.settings.min_variance` for double precision).
pub const MIN_VARIANCE: f64 = 1e-10;

/// A joint Gaussian posterior over `q` points and `m` outputs, in the original
/// outcome space.
///
/// This corresponds to BoTorch's `GPyTorchPosterior`: for a single-output
/// model `mean` is `posterior.mean.squeeze(-1)` and `covariance` is
/// `posterior.distribution.covariance_matrix`; for `m` outputs both are
/// flattened in BoTorch's output-major order (see the [module
/// documentation](self)), and [`Posterior::mean_matrix`] /
/// [`Posterior::variance_matrix`] return BoTorch's `q × m` `posterior.mean`
/// and `posterior.variance`.
#[derive(Clone, Debug, PartialEq)]
pub struct Posterior {
    /// Posterior mean (length `q·m`; entry `t·q + p` is output `t` at point `p`).
    pub mean: Vector,
    /// Posterior covariance (`q·m × q·m`, same order as `mean`).
    pub covariance: Matrix,
    /// Number of outputs `m` (1 for single-output models).
    pub num_outputs: usize,
    /// The elementwise maps of the samples of a nonlinear outcome transform
    /// (BoTorch's `TransformedPosterior`): `mean` and `covariance` are then
    /// those of the Gaussian *base* posterior, whose samples the transform
    /// maps (see [`crate::transforms::outcome`]). `None` for a Gaussian
    /// posterior.
    pub sample_transform: Option<SampleTransform>,
}

impl Posterior {
    /// A single-output posterior over `q = mean.len()` points.
    ///
    /// # Panics
    /// If `covariance` is not `q × q`.
    pub fn single_output(mean: Vector, covariance: Matrix) -> Self {
        Self::multi_output(mean, covariance, 1)
    }

    /// A posterior over `m = num_outputs` outputs with `mean` and `covariance`
    /// in output-major order.
    ///
    /// # Panics
    /// If `num_outputs` is zero, does not divide `mean.len()`, or the
    /// covariance does not match the mean.
    pub fn multi_output(mean: Vector, covariance: Matrix, num_outputs: usize) -> Self {
        let n = mean.len();
        assert!(
            num_outputs > 0 && n.is_multiple_of(num_outputs) && covariance.shape() == (n, n),
            "inconsistent posterior: mean of length {n}, covariance {:?}, {num_outputs} outputs",
            covariance.shape()
        );
        Self {
            mean,
            covariance,
            num_outputs,
            sample_transform: None,
        }
    }

    /// The posterior with the elementwise sample maps `transform` (a
    /// `TransformedPosterior` of this Gaussian posterior; `None` keeps it
    /// Gaussian).
    ///
    /// # Panics
    /// If the transform does not match the numbers of points and outputs.
    pub fn with_sample_transform(mut self, transform: Option<SampleTransform>) -> Self {
        if let Some(t) = &transform {
            assert!(
                t.num_outputs() == self.num_outputs && t.num_points() == self.num_points(),
                "sample transform over {} outputs and {} points for a posterior over {} and {}",
                t.num_outputs(),
                t.num_points(),
                self.num_outputs,
                self.num_points()
            );
        }
        self.sample_transform = transform;
        self
    }

    /// Whether the posterior is a `TransformedPosterior` (see
    /// [`Posterior::sample_transform`]).
    pub fn is_transformed(&self) -> bool {
        self.sample_transform.is_some()
    }

    /// Errors for a `TransformedPosterior`, for computations that need a
    /// Gaussian posterior (`what` names the computation).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] if the posterior is transformed.
    pub fn require_gaussian(&self, what: &str) -> Result<()> {
        if self.is_transformed() {
            return Err(BotorchError::Unsupported(format!(
                "{what} requires a Gaussian posterior, but the model's nonlinear outcome \
                 transform (e.g. Log, Power or Bilog) makes it a TransformedPosterior"
            )));
        }
        Ok(())
    }

    /// The marginal means and variances (flat, output-major) as BoTorch's
    /// `posterior.mean` and `posterior.variance` report them: those of the
    /// Gaussian posterior (variances clamped at [`MIN_VARIANCE`]), or the
    /// moments of a `TransformedPosterior` (unclamped, from the clamped base
    /// variances).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] (`NotImplementedError`) for transformed
    /// outputs without moments (`Power`, `Bilog`).
    pub fn moments(&self) -> Result<(Vector, Vector)> {
        let q = self.num_points();
        let mut mean = self.mean.clone();
        let mut var = self.variance();
        if let Some(st) = &self.sample_transform {
            for t in 0..self.num_outputs {
                if !st.is_transformed(t) {
                    continue;
                }
                for p in 0..q {
                    let i = t * q + p;
                    let (m, v, ..) = st.moments(t, p, self.mean[i], self.covariance[(i, i)])?;
                    mean[i] = m;
                    var[i] = v;
                }
            }
        }
        Ok((mean, var))
    }

    /// Number of jointly distributed values `q·m`.
    pub fn len(&self) -> usize {
        self.mean.len()
    }

    /// Whether the posterior is over zero points.
    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Number of outputs `m`.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Number of points `q`.
    pub fn num_points(&self) -> usize {
        self.mean.len() / self.num_outputs.max(1)
    }

    /// The flat index of output `output` at point `point` (`output·q + point`).
    pub fn index(&self, point: usize, output: usize) -> usize {
        output * self.num_points() + point
    }

    /// The posterior mean as a `q × m` matrix (BoTorch's `posterior.mean`).
    pub fn mean_matrix(&self) -> Matrix {
        let q = self.num_points();
        Matrix::from_fn(q, self.num_outputs, |p, t| self.mean[t * q + p])
    }

    /// Marginal variances (flat, output-major), clamped from below at
    /// [`MIN_VARIANCE`] exactly like GPyTorch's `MultivariateNormal.variance`
    /// (which BoTorch's `posterior.variance` returns).
    pub fn variance(&self) -> Vector {
        self.covariance.diagonal().map(|v| v.max(MIN_VARIANCE))
    }

    /// Marginal variances as a `q × m` matrix, clamped like
    /// [`Posterior::variance`] (BoTorch's `posterior.variance`).
    pub fn variance_matrix(&self) -> Matrix {
        let q = self.num_points();
        Matrix::from_fn(q, self.num_outputs, |p, t| {
            self.covariance[(t * q + p, t * q + p)].max(MIN_VARIANCE)
        })
    }

    /// Unclamped diagonal of the covariance matrix (flat, output-major).
    pub fn raw_variance(&self) -> Vector {
        self.covariance.diagonal()
    }

    /// The joint posterior of the outputs `outputs` (in that order, repetitions
    /// allowed), e.g. BoTorch's `output_indices`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty selection or an index
    /// out of range.
    pub fn select_outputs(&self, outputs: &[usize]) -> Result<Posterior> {
        check_output_selection(outputs, self.num_outputs)?;
        let q = self.num_points();
        let k = outputs.len();
        let idx = |i: usize| outputs[i / q] * q + i % q;
        let mean = Vector::from_fn(k * q, |i, _| self.mean[idx(i)]);
        let covariance = Matrix::from_fn(k * q, k * q, |i, j| self.covariance[(idx(i), idx(j))]);
        let transform = self
            .sample_transform
            .as_ref()
            .and_then(|t| t.select_outputs(outputs));
        Ok(Posterior::multi_output(mean, covariance, k).with_sample_transform(transform))
    }
}

/// Validates output indices for [`Posterior::select_outputs`] and
/// [`PointPosteriors::select_outputs`].
pub(crate) fn check_output_selection(outputs: &[usize], m: usize) -> Result<()> {
    if outputs.is_empty() {
        return Err(BotorchError::InvalidArgument(
            "at least one output must be selected".into(),
        ));
    }
    if let Some(&t) = outputs.iter().find(|&&t| t >= m) {
        return Err(BotorchError::InvalidArgument(format!(
            "output index {t} is out of range for a model with {m} outputs"
        )));
    }
    Ok(())
}

/// Posteriors of a batch of `b` points, each evaluated on its own (`q = 1`,
/// like a `b × 1 × d` input in BoTorch), over all `m` outputs.
///
/// The covariances among the outputs are *not* clamped; models with
/// independent outputs have diagonal covariances.
#[derive(Clone, Debug, PartialEq)]
pub struct PointPosteriors {
    /// Posterior means (`b × m`).
    pub mean: Matrix,
    /// Output covariances (`b × m²`): row `i` is the row-major `m × m`
    /// covariance matrix of point `i`.
    pub covariance: Matrix,
}

impl PointPosteriors {
    /// Point posteriors of a single-output model from means and (unclamped)
    /// variances.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn single_output(mean: &Vector, variance: &Vector) -> Self {
        assert_eq!(
            mean.len(),
            variance.len(),
            "mean / variance length mismatch"
        );
        Self {
            mean: Matrix::from_column_slice(mean.len(), 1, mean.as_slice()),
            covariance: Matrix::from_column_slice(variance.len(), 1, variance.as_slice()),
        }
    }

    /// Point posteriors of independent outputs from per-output means and
    /// (unclamped) variances (`b × m` each).
    ///
    /// # Panics
    /// If the shapes differ.
    pub fn independent(mean: Matrix, variance: &Matrix) -> Self {
        assert_eq!(
            mean.shape(),
            variance.shape(),
            "mean / variance shape mismatch"
        );
        let (b, m) = mean.shape();
        let mut covariance = Matrix::zeros(b, m * m);
        for i in 0..b {
            for t in 0..m {
                covariance[(i, t * m + t)] = variance[(i, t)];
            }
        }
        Self { mean, covariance }
    }

    /// Number of points `b`.
    pub fn len(&self) -> usize {
        self.mean.nrows()
    }

    /// Whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.mean.nrows() == 0
    }

    /// Number of outputs `m`.
    pub fn num_outputs(&self) -> usize {
        self.mean.ncols()
    }

    /// The `m × m` output covariance of point `i` (unclamped).
    pub fn covariance_of(&self, i: usize) -> Matrix {
        let m = self.num_outputs();
        Matrix::from_fn(m, m, |a, b| self.covariance[(i, a * m + b)])
    }

    /// The unclamped variance of output `t` at point `i`.
    pub fn variance(&self, i: usize, t: usize) -> f64 {
        let m = self.num_outputs();
        self.covariance[(i, t * m + t)]
    }

    /// The point posteriors of the outputs `outputs` (in that order).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty selection or an index
    /// out of range.
    pub fn select_outputs(&self, outputs: &[usize]) -> Result<PointPosteriors> {
        let m = self.num_outputs();
        check_output_selection(outputs, m)?;
        let k = outputs.len();
        let b = self.len();
        Ok(Self {
            mean: Matrix::from_fn(b, k, |i, a| self.mean[(i, outputs[a])]),
            covariance: Matrix::from_fn(b, k * k, |i, j| {
                self.covariance[(i, outputs[j / k] * m + outputs[j % k])]
            }),
        })
    }
}

/// Joint posterior over `q` points and `m` outputs together with derivatives
/// with respect to every input coordinate, all in the original (untransformed)
/// input and outcome spaces, without observation noise.
///
/// The mean and covariance use the output-major flattening of [`Posterior`]
/// (length `n = q·m`). Derivatives are indexed by `j = p·d + k`, the `k`-th
/// coordinate of the `p`-th *input point*: `dmean[j][a] = ∂mean[a]/∂x[p,k]`
/// and `dcov[j][(a, b)] = ∂cov[a,b]/∂x[p,k]`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPosteriorGrad {
    /// Posterior mean (length `q·m`).
    pub mean: Vector,
    /// Posterior covariance (`q·m × q·m`).
    pub covariance: Matrix,
    /// Mean derivatives (`q·d` vectors of length `q·m`).
    pub dmean: Vec<Vector>,
    /// Covariance derivatives (`q·d` matrices of size `q·m × q·m`).
    pub dcov: Vec<Matrix>,
    /// Number of outputs `m`.
    pub num_outputs: usize,
}

impl JointPosteriorGrad {
    /// Number of points `q`.
    pub fn num_points(&self) -> usize {
        self.mean.len() / self.num_outputs.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_output_posterior() -> Posterior {
        // q = 2 points, m = 2 outputs (output-major: [o0p0, o0p1, o1p0, o1p1]).
        let mean = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut cov = Matrix::zeros(4, 4);
        cov[(0, 0)] = 0.5;
        cov[(0, 1)] = 0.1;
        cov[(1, 0)] = 0.1;
        cov[(1, 1)] = 0.6;
        cov[(2, 2)] = 1e-14;
        cov[(3, 3)] = 0.9;
        Posterior::multi_output(mean, cov, 2)
    }

    #[test]
    fn layout_helpers() {
        let p = two_output_posterior();
        assert_eq!((p.len(), p.num_points(), p.num_outputs()), (4, 2, 2));
        assert_eq!(p.index(1, 1), 3);
        assert_eq!(
            p.mean_matrix(),
            Matrix::from_row_slice(2, 2, &[1.0, 3.0, 2.0, 4.0])
        );
        let v = p.variance_matrix();
        assert_eq!(v[(0, 1)], MIN_VARIANCE);
        assert_eq!(v[(1, 0)], 0.6);
        let s = p.select_outputs(&[1]).unwrap();
        assert_eq!(s.num_outputs(), 1);
        assert_eq!(s.mean.as_slice(), &[3.0, 4.0]);
        let r = p.select_outputs(&[1, 0]).unwrap();
        assert_eq!(r.mean.as_slice(), &[3.0, 4.0, 1.0, 2.0]);
        assert_eq!(r.covariance[(2, 3)], 0.1);
        assert!(p.select_outputs(&[2]).is_err());
        assert!(p.select_outputs(&[]).is_err());
    }

    #[test]
    fn point_posteriors() {
        let mean = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let var = Matrix::from_row_slice(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let pp = PointPosteriors::independent(mean, &var);
        assert_eq!(pp.variance(1, 1), 0.4);
        assert_eq!(pp.covariance_of(0)[(0, 1)], 0.0);
        let s = pp.select_outputs(&[1, 0]).unwrap();
        assert_eq!(s.mean[(1, 0)], 4.0);
        assert_eq!(s.variance(1, 1), 0.3);
        let single = PointPosteriors::single_output(
            &Vector::from_vec(vec![1.0]),
            &Vector::from_vec(vec![2.0]),
        );
        assert_eq!((single.len(), single.num_outputs()), (1, 1));
    }
}
