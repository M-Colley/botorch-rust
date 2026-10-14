//! Composite acquisition functions (a botorch-rust extension): the weighted
//! blend of log-scale acquisition functions that BoTorch users write as a
//! `torch.nn.Module`, e.g. the transfer acquisition functions of TAF-style
//! meta-learning (Wistuba, Schilling and Schmidt-Thieme, *Machine Learning*
//! 2018), which average a target acquisition function with source
//! acquisition functions.
//!
//! [`WeightedLogMeanExp`] evaluates
//!
//! ```text
//! α(X) = log( Σ_i w_i exp(α_i(X)) / Σ_i w_i )
//!      = logsumexp_i(log w_i + α_i(X)) - log Σ_i w_i,
//! ```
//!
//! the logarithm of the weighted mean of the exponentiated terms, i.e. of the
//! weighted mean of the linear-scale acquisition values when the terms are
//! log-scale (qLogEI, qLogNEHVI, ...). It is computed exactly like
//! `torch.logsumexp(torch.stack([log(w_i) + α_i(X)], -1), -1) - log(Σ w_i)`,
//! with the gradient `Σ_i softmax_i ∇α_i(X)`. Terms with weight zero are left
//! out of both sums (and are not evaluated).

use std::sync::Arc;

use rayon::prelude::*;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::Matrix;

/// `log(Σ_i w_i exp(α_i(X)) / Σ_i w_i)` over acquisition functions `α_i`
/// with non-negative weights `w_i` (see the [module documentation](self)).
///
/// All terms must have the same input dimension. Pending points are those of
/// the first term; [`AcquisitionFunction::set_x_pending`] sets them on every
/// term. The model of the composite is the first term's model.
#[derive(Debug)]
pub struct WeightedLogMeanExp {
    terms: Vec<Arc<dyn AcquisitionFunction>>,
    weights: Vec<f64>,
    /// `(index, log w_i)` of the terms with a positive weight.
    active: Vec<(usize, f64)>,
    /// `log Σ_i w_i`.
    log_total: f64,
}

impl WeightedLogMeanExp {
    /// The composite of `terms` with `weights` (one per term).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for no terms, a number of weights
    /// that differs from the number of terms, negative or non-finite weights,
    /// or weights that are all zero; [`BotorchError::DimensionMismatch`] for
    /// terms of different input dimensions.
    pub fn new(terms: Vec<Arc<dyn AcquisitionFunction>>, weights: Vec<f64>) -> Result<Self> {
        if terms.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "WeightedLogMeanExp requires at least one acquisition function".into(),
            ));
        }
        if weights.len() != terms.len() {
            return Err(BotorchError::InvalidArgument(format!(
                "WeightedLogMeanExp got {} weights for {} acquisition functions",
                weights.len(),
                terms.len()
            )));
        }
        if let Some(w) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(BotorchError::InvalidArgument(format!(
                "the weights of WeightedLogMeanExp must be finite and non-negative, got {w}"
            )));
        }
        let d = terms[0].dim();
        if let Some(t) = terms.iter().find(|t| t.dim() != d) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the acquisition functions of WeightedLogMeanExp must have the same input \
                 dimension, got {d} and {}",
                t.dim()
            )));
        }
        let mut active = Vec::new();
        let mut total = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0.0 {
                active.push((i, w.ln()));
                total += w;
            }
        }
        if active.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "WeightedLogMeanExp has no active terms: every weight is zero".into(),
            ));
        }
        Ok(Self {
            terms,
            weights,
            active,
            log_total: total.ln(),
        })
    }

    /// The acquisition functions.
    pub fn terms(&self) -> &[Arc<dyn AcquisitionFunction>] {
        &self.terms
    }

    /// The weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `logsumexp(z) - log Σ w` and the softmax weights of `z`
    /// (`z_i = log w_i + α_i`), as `torch.logsumexp` and its derivative
    /// compute them. An all `-inf` input gives `-inf` and zero weights.
    fn reduce(&self, z: &[f64]) -> (f64, Vec<f64>) {
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            // torch.logsumexp: the maximum itself for infinite maxima (NaN
            // inputs propagate through the sum below otherwise).
            let lse = if z.iter().any(|v| v.is_nan()) {
                f64::NAN
            } else {
                max
            };
            let p = z
                .iter()
                .map(|&v| if v == f64::INFINITY { 1.0 } else { 0.0 })
                .collect();
            return (lse - self.log_total, p);
        }
        let sum: f64 = z.iter().map(|&v| (v - max).exp()).sum();
        let lse = max + sum.ln();
        let p = z.iter().map(|&v| (v - lse).exp()).collect();
        (lse - self.log_total, p)
    }
}

impl AcquisitionFunction for WeightedLogMeanExp {
    fn model(&self) -> &Arc<dyn Model> {
        self.terms[0].model()
    }

    fn dim(&self) -> usize {
        self.terms[0].dim()
    }

    fn expected_q(&self) -> Option<usize> {
        self.active
            .iter()
            .find_map(|&(i, _)| self.terms[i].expected_q())
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        let z = self
            .active
            .iter()
            .map(|&(i, lw)| Ok(lw + self.terms[i].forward(x)?))
            .collect::<Result<Vec<f64>>>()?;
        Ok(self.reduce(&z).0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let mut z = Vec::with_capacity(self.active.len());
        let mut grads = Vec::with_capacity(self.active.len());
        for &(i, lw) in &self.active {
            let (v, g) = self.terms[i].forward_with_grad(x)?;
            z.push(lw + v);
            grads.push(g);
        }
        let (value, p) = self.reduce(&z);
        let mut grad = Matrix::zeros(x.nrows(), x.ncols());
        for (pi, g) in p.iter().zip(&grads) {
            if *pi != 0.0 {
                grad += g * *pi;
            }
        }
        Ok((value, grad))
    }

    /// Every term evaluates the whole batch (in parallel), then the values
    /// are combined per candidate set.
    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        let per_term = self
            .active
            .iter()
            .map(|&(i, _)| self.terms[i].forward_many(xs))
            .collect::<Result<Vec<_>>>()?;
        Ok((0..xs.len())
            .into_par_iter()
            .map(|b| {
                let z: Vec<f64> = self
                    .active
                    .iter()
                    .zip(&per_term)
                    .map(|(&(_, lw), v)| lw + v[b])
                    .collect();
                self.reduce(&z).0
            })
            .collect())
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.terms[0].x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        for t in &self.terms {
            t.set_x_pending(x_pending)?;
        }
        Ok(())
    }

    fn is_log(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition::{LogExpectedImprovement, UpperConfidenceBound};
    use crate::models::{SingleTaskGP, SingleTaskGPOptions};

    fn terms() -> Vec<Arc<dyn AcquisitionFunction>> {
        let x = Matrix::from_fn(7, 2, |i, j| ((i * 3 + j * 5) % 7) as f64 / 7.0);
        let y = Matrix::from_fn(7, 1, |i, _| (5.0 * x[(i, 0)]).sin() - x[(i, 1)]);
        let model: Arc<dyn Model> =
            Arc::new(SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap());
        vec![
            Arc::new(LogExpectedImprovement::new(model.clone(), 0.3, true).unwrap()),
            Arc::new(UpperConfidenceBound::new(model.clone(), 0.5, true).unwrap()),
            Arc::new(LogExpectedImprovement::new(model, -0.2, false).unwrap()),
        ]
    }

    #[test]
    fn values_follow_the_formula() {
        let t = terms();
        let w = vec![0.7, 0.0, 0.2];
        let c = WeightedLogMeanExp::new(t.clone(), w.clone()).unwrap();
        let x = Matrix::from_row_slice(1, 2, &[0.35, 0.6]);
        let a: Vec<f64> = t.iter().map(|t| t.forward(&x).unwrap()).collect();
        let expected = ((w[0] * a[0].exp() + w[2] * a[2].exp()) / (w[0] + w[2])).ln();
        let v = c.forward(&x).unwrap();
        assert!((v - expected).abs() < 1e-14 * expected.abs().max(1.0));
        let xs = vec![x.clone(), Matrix::from_row_slice(1, 2, &[0.9, 0.1])];
        let many = c.forward_many(&xs).unwrap();
        assert_eq!(many[0], v);
        assert_eq!(many[1], c.forward(&xs[1]).unwrap());
        assert!(c.is_log() && !c.is_nonnegative());
        assert_eq!(c.expected_q(), Some(1));
    }

    #[test]
    fn gradients_match_finite_differences() {
        let c = WeightedLogMeanExp::new(terms(), vec![0.5, 0.3, 0.2]).unwrap();
        let x = Matrix::from_row_slice(1, 2, &[0.42, 0.27]);
        let (v, g) = c.forward_with_grad(&x).unwrap();
        assert_eq!(v, c.forward(&x).unwrap());
        let h = 1e-6;
        for k in 0..2 {
            let mut xp = x.clone();
            xp[(0, k)] += h;
            let mut xm = x.clone();
            xm[(0, k)] -= h;
            let fd = (c.forward(&xp).unwrap() - c.forward(&xm).unwrap()) / (2.0 * h);
            assert!(
                (fd - g[(0, k)]).abs() < 1e-6 * fd.abs().max(1.0),
                "{fd} {}",
                g[(0, k)]
            );
        }
    }

    #[test]
    fn single_term_is_the_term() {
        let t = terms();
        let c = WeightedLogMeanExp::new(vec![t[0].clone()], vec![2.5]).unwrap();
        let x = Matrix::from_row_slice(1, 2, &[0.1, 0.8]);
        let (v, g) = c.forward_with_grad(&x).unwrap();
        let (v0, g0) = t[0].forward_with_grad(&x).unwrap();
        assert!((v - v0).abs() < 1e-14 * v0.abs().max(1.0));
        assert!((&g - &g0).amax() < 1e-14 * g0.amax().max(1.0));
    }

    #[test]
    fn invalid_arguments() {
        let t = terms();
        assert!(WeightedLogMeanExp::new(Vec::new(), Vec::new()).is_err());
        assert!(WeightedLogMeanExp::new(t.clone(), vec![1.0]).is_err());
        assert!(WeightedLogMeanExp::new(t.clone(), vec![1.0, -1.0, 0.0]).is_err());
        assert!(WeightedLogMeanExp::new(t.clone(), vec![1.0, f64::NAN, 0.0]).is_err());
        assert!(WeightedLogMeanExp::new(t, vec![0.0, 0.0, 0.0]).is_err());
    }
}
