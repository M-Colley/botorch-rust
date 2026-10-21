//! Gaussian KL divergence and parametric fits to empirical Gaussian priors.

use crate::constraints::Constraint;
use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult};
use crate::kernels::tree::sample_tree_priors;
use crate::kernels::{KernelNode, TreeKernel};
use crate::linalg::psd_safe_cholesky;
use crate::{Matrix, Vector};

/// A fixed Gaussian target, with its log determinant cached for optimization.
#[derive(Clone, Debug)]
pub struct GaussianKlTarget {
    mean: Vector,
    covariance: Matrix,
    logdet: f64,
}

/// KL value and derivatives with respect to both means and covariances.
#[derive(Clone, Debug)]
pub struct GaussianKlGrad {
    /// `KL(p || q)`.
    pub value: f64,
    /// Derivative with respect to the mean of p.
    pub mean_p: Vector,
    /// Derivative with respect to the covariance of p.
    pub covariance_p: Matrix,
    /// Derivative with respect to the mean of q.
    pub mean_q: Vector,
    /// Derivative with respect to the covariance of q.
    pub covariance_q: Matrix,
}

fn check_gaussian(mean: &Vector, covariance: &Matrix) -> Result<()> {
    if covariance.shape() != (mean.len(), mean.len()) {
        return Err(BotorchError::DimensionMismatch(
            "Gaussian covariance must be square and match its mean".into(),
        ));
    }
    if mean.iter().chain(covariance.iter()).any(|v| !v.is_finite()) {
        return Err(BotorchError::InvalidArgument(
            "Gaussian means and covariances must be finite".into(),
        ));
    }
    Ok(())
}

impl GaussianKlTarget {
    /// Capture the target distribution p, using the standard PSD-safe jitter.
    pub fn new(mean: Vector, covariance: Matrix) -> Result<Self> {
        check_gaussian(&mean, &covariance)?;
        let logdet = if mean.is_empty() {
            0.0
        } else {
            psd_safe_cholesky(&covariance)?.logdet()
        };
        Ok(Self {
            mean,
            covariance,
            logdet,
        })
    }

    /// Value and derivatives with respect to the candidate distribution q.
    pub fn value_q_gradient(
        &self,
        mean: &Vector,
        covariance: &Matrix,
    ) -> Result<(f64, Vector, Matrix)> {
        check_gaussian(mean, covariance)?;
        if mean.len() != self.mean.len() {
            return Err(BotorchError::DimensionMismatch(
                "the two Gaussian dimensions differ".into(),
            ));
        }
        let n = mean.len();
        if n == 0 {
            return Ok((0.0, Vector::zeros(0), Matrix::zeros(0, 0)));
        }
        let chol = psd_safe_cholesky(covariance)?;
        let inverse = chol.inverse();
        let delta = mean - &self.mean;
        let w = &inverse * &delta;
        let trace = inverse.component_mul(&self.covariance.transpose()).sum();
        let value = 0.5 * (chol.logdet() - self.logdet + trace + delta.dot(&w) - n as f64);
        let gradient =
            (&inverse - &inverse * &self.covariance * &inverse - &w * w.transpose()) * 0.5;
        Ok((value, w, gradient))
    }
}

/// Stable `KL(N(mean_p,covariance_p) || N(mean_q,covariance_q))`, including
/// exact first derivatives in all four arguments, with jitter treated as fixed.
pub fn kl_divergence_mvn(
    mean_p: &Vector,
    covariance_p: &Matrix,
    mean_q: &Vector,
    covariance_q: &Matrix,
) -> Result<GaussianKlGrad> {
    let target = GaussianKlTarget::new(mean_p.clone(), covariance_p.clone())?;
    let (value, mean_q_gradient, covariance_q_gradient) =
        target.value_q_gradient(mean_q, covariance_q)?;
    let n = mean_p.len();
    let covariance_p_gradient = if n == 0 {
        Matrix::zeros(0, 0)
    } else {
        (psd_safe_cholesky(covariance_q)?.inverse() - psd_safe_cholesky(covariance_p)?.inverse())
            * 0.5
    };
    Ok(GaussianKlGrad {
        value,
        mean_p: -&mean_q_gradient,
        covariance_p: covariance_p_gradient,
        mean_q: mean_q_gradient,
        covariance_q: covariance_q_gradient,
    })
}

/// Kernel and optional constant-mean objective, with no likelihood or prior terms.
#[derive(Clone, Debug)]
pub struct KernelPriorKl {
    node: KernelNode,
    points: Matrix,
    target: GaussianKlTarget,
    learn_mean: bool,
    mean_constraint: Option<Constraint>,
}

impl KernelPriorKl {
    /// Build a KL minimization problem at fixed reference locations.
    pub fn new(
        node: KernelNode,
        points: Matrix,
        mean: Vector,
        covariance: Matrix,
        learn_mean: bool,
        mean_constraint: Option<Constraint>,
    ) -> Result<Self> {
        if points.nrows() != mean.len()
            || points.nrows() == 0
            || points.ncols() == 0
            || points.iter().any(|v| !v.is_finite())
        {
            return Err(BotorchError::DimensionMismatch(
                "nonempty finite reference points must match the target mean".into(),
            ));
        }
        node.validate(points.ncols())?;
        let target = GaussianKlTarget::new(mean, covariance)?;
        Ok(Self {
            node,
            points,
            target,
            learn_mean,
            mean_constraint,
        })
    }

    /// Negative KL and its raw parameter gradient (mean, then kernel).
    pub fn value_and_grad(&self, raw: &[f64]) -> Result<(f64, Vec<f64>)> {
        let offset = usize::from(self.learn_mean);
        if raw.len() != offset + self.node.num_raw() || raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::DimensionMismatch(
                "wrong number of finite mean/kernel parameters".into(),
            ));
        }
        let constant = if self.learn_mean {
            self.mean_constraint.map_or(raw[0], |c| c.transform(raw[0]))
        } else {
            0.0
        };
        let tree = TreeKernel::build(&self.node, &raw[offset..], self.points.ncols())?;
        let prepared = tree.prepare(&self.points)?;
        let covariance = tree.symmetric(&prepared);
        let (kl, mean_gradient, covariance_gradient) = self.target.value_q_gradient(
            &Vector::from_element(self.points.nrows(), constant),
            &covariance,
        )?;
        let kernel_gradient = tree.contract(&prepared, &covariance_gradient);
        let mut gradient = Vec::with_capacity(raw.len());
        if self.learn_mean {
            gradient.push(
                -mean_gradient.sum()
                    * self
                        .mean_constraint
                        .map_or(1.0, |c| c.transform_derivative(raw[0])),
            );
        }
        gradient.extend(
            kernel_gradient
                .into_iter()
                .zip(self.node.transform_derivatives(&raw[offset..]))
                .map(|(g, d)| -g * d),
        );
        Ok((-kl, gradient))
    }

    /// Fit with the common native L-BFGS-B retry and bound handling.
    pub fn fit(&self, initial: &[f64], options: FitOptions) -> Result<(FitResult, Vec<f64>)> {
        self.value_and_grad(initial)?;
        let offset = usize::from(self.learn_mean);
        let mut bounds = Vec::new();
        if self.learn_mean {
            bounds.push(
                self.mean_constraint
                    .map_or((f64::NEG_INFINITY, f64::INFINITY), |c| c.raw_bounds()),
            );
        }
        bounds.extend(self.node.raw_bounds());
        minimize_with_retries(
            initial,
            &bounds,
            options,
            |raw| self.value_and_grad(raw),
            |rng| {
                let mut raw = initial.to_vec();
                sample_tree_priors(&self.node, &mut raw[offset..], rng);
                raw
            },
        )
    }
}
