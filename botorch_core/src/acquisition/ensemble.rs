//! Acquisition functions of ensemble models (BoTorch's
//! `average_over_ensemble_models` and the ensemble dimension of
//! `SampleReducingMCAcquisitionFunction`).
//!
//! BoTorch evaluates an acquisition function of an ensemble model (e.g.
//! `EnsembleMapSaasSingleTaskGP`, whose posterior is a mixture of the GP
//! posteriors of its members) on the batch of member posteriors and averages
//! the values over the ensemble: the mean, or for log-scale acquisition
//! functions (BoTorch's `_log`) the log of the mean of the exponentiated
//! values (`logmeanexp`). Monte-Carlo acquisition functions reduce the
//! ensemble dimension together with the sample dimensions, with base samples
//! shared by all members, which gives the same value. [`EnsembleAverage`]
//! averages the acquisition functions of the members, each built on its
//! member's model with the same arguments (and sampler).
//!
//! With weights, the average is `Σ_k w_k α_k / Σ_k w_k` (log-scale:
//! `log(Σ_k w_k exp(α_k) / Σ_k w_k)`): the Monte-Carlo acquisition values of
//! BoTorch's `EnsembleModel`s, whose `IndexSampler` draws the ensemble member
//! of every sample, are the averages of the members' (deterministic) values
//! weighted by the frequencies of the members in the index samples.

use std::sync::Arc;

use rayon::prelude::*;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::Matrix;

/// The (weighted) average of the acquisition functions of the members of
/// an ensemble (see the [module documentation](self)): `mean_k α_k(X)`, or
/// `logmeanexp_k α_k(X)` if the members are log-scale. Members with weight
/// zero are not evaluated.
#[derive(Debug)]
pub struct EnsembleAverage {
    members: Vec<Arc<dyn AcquisitionFunction>>,
    log: bool,
    /// `(index, normalized weight)` of the members with a positive weight.
    active: Vec<(usize, f64)>,
}

impl EnsembleAverage {
    /// The average of `members` (log-scale if the first member is).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without members or with members of
    /// different scales; [`BotorchError::DimensionMismatch`] for members of
    /// different input dimensions.
    pub fn new(members: Vec<Arc<dyn AcquisitionFunction>>) -> Result<Self> {
        let k = members.len();
        Self::with_weights(members, vec![1.0; k])
    }

    /// The weighted average of `members` (log-scale if the first member
    /// is).
    ///
    /// # Errors
    /// See [`EnsembleAverage::new`]; [`BotorchError::InvalidArgument`] for a
    /// number of weights that differs from the number of members, negative
    /// or non-finite weights, or weights that are all zero.
    pub fn with_weights(
        members: Vec<Arc<dyn AcquisitionFunction>>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        let first = members.first().ok_or_else(|| {
            BotorchError::InvalidArgument("an ensemble needs at least one member".into())
        })?;
        let (d, log) = (first.dim(), first.is_log());
        for m in &members {
            if m.dim() != d {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the acquisition functions of the ensemble members must have the same \
                     input dimension, got {d} and {}",
                    m.dim()
                )));
            }
            if m.is_log() != log {
                return Err(BotorchError::InvalidArgument(
                    "the acquisition functions of the ensemble members must all be log-scale \
                     or all not"
                        .into(),
                ));
            }
        }
        if weights.len() != members.len() {
            return Err(BotorchError::InvalidArgument(format!(
                "got {} weights for {} ensemble members",
                weights.len(),
                members.len()
            )));
        }
        if let Some(w) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(BotorchError::InvalidArgument(format!(
                "the weights of the ensemble members must be finite and non-negative, got {w}"
            )));
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(BotorchError::InvalidArgument(
                "the weights of the ensemble members are all zero".into(),
            ));
        }
        let active = weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.0)
            .map(|(i, &w)| (i, w / total))
            .collect();
        Ok(Self {
            members,
            log,
            active,
        })
    }

    /// The acquisition functions of the members.
    pub fn members(&self) -> &[Arc<dyn AcquisitionFunction>] {
        &self.members
    }

    /// The average of the values `v` of the active members and the weights
    /// of the members in its derivative (the normalized weights, or the
    /// softmax weights of the weighted `logmeanexp`, as torch's `logsumexp`
    /// differentiates). With equal weights this is BoTorch's `mean(dim=-1)`
    /// / `logmeanexp(dim=-1)`.
    fn reduce(&self, v: &[f64]) -> (f64, Vec<f64>) {
        if !self.log {
            let value = v
                .iter()
                .zip(&self.active)
                .map(|(x, (_, w))| w * x)
                .sum::<f64>();
            return (value, self.active.iter().map(|&(_, w)| w).collect());
        }
        // logsumexp_k(log w_k + v_k) with normalized weights.
        let z: Vec<f64> = v
            .iter()
            .zip(&self.active)
            .map(|(x, (_, w))| w.ln() + x)
            .collect();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            let lse = if z.iter().any(|x| x.is_nan()) {
                f64::NAN
            } else {
                max
            };
            let p = z
                .iter()
                .map(|&x| if x == f64::INFINITY { 1.0 } else { 0.0 })
                .collect();
            return (lse, p);
        }
        let sum: f64 = z.iter().map(|&x| (x - max).exp()).sum();
        let lse = max + sum.ln();
        let p = z.iter().map(|&x| (x - lse).exp()).collect();
        (lse, p)
    }
}

impl AcquisitionFunction for EnsembleAverage {
    fn model(&self) -> &Arc<dyn Model> {
        self.members[0].model()
    }

    fn dim(&self) -> usize {
        self.members[0].dim()
    }

    fn expected_q(&self) -> Option<usize> {
        self.members[0].expected_q()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        let v = self
            .active
            .iter()
            .map(|&(i, _)| self.members[i].forward(x))
            .collect::<Result<Vec<f64>>>()?;
        Ok(self.reduce(&v).0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let mut v = Vec::with_capacity(self.active.len());
        let mut grads = Vec::with_capacity(self.active.len());
        for &(i, _) in &self.active {
            let (value, g) = self.members[i].forward_with_grad(x)?;
            v.push(value);
            grads.push(g);
        }
        let (value, p) = self.reduce(&v);
        let mut grad = Matrix::zeros(x.nrows(), x.ncols());
        for (pi, g) in p.iter().zip(&grads) {
            if *pi != 0.0 {
                grad += g * *pi;
            }
        }
        Ok((value, grad))
    }

    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        let per_member = self
            .active
            .iter()
            .map(|&(i, _)| self.members[i].forward_many(xs))
            .collect::<Result<Vec<_>>>()?;
        Ok((0..xs.len())
            .into_par_iter()
            .map(|b| {
                let v: Vec<f64> = per_member.iter().map(|values| values[b]).collect();
                self.reduce(&v).0
            })
            .collect())
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.members[0].x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        for m in &self.members {
            m.set_x_pending(x_pending)?;
        }
        Ok(())
    }

    fn is_nonnegative(&self) -> bool {
        !self.log && self.members.iter().all(|m| m.is_nonnegative())
    }

    fn is_log(&self) -> bool {
        self.log
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition::{LogExpectedImprovement, PosteriorMean};
    use crate::models::{SingleTaskGP, SingleTaskGPOptions};

    fn model(shift: f64) -> Arc<dyn Model> {
        let x = Matrix::from_fn(7, 2, |i, j| ((i * 3 + j * 5) % 7) as f64 / 7.0);
        let y = Matrix::from_fn(7, 1, |i, _| (5.0 * x[(i, 0)]).sin() - x[(i, 1)] + shift);
        Arc::new(SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap())
    }

    fn log_eis() -> Vec<Arc<dyn AcquisitionFunction>> {
        (0..3)
            .map(|k| {
                Arc::new(LogExpectedImprovement::new(model(0.2 * k as f64), 0.5, true).unwrap())
                    as Arc<dyn AcquisitionFunction>
            })
            .collect()
    }

    #[test]
    fn mean_and_logmeanexp_of_the_members() {
        let x = Matrix::from_row_slice(1, 2, &[0.35, 0.6]);
        let pm: Vec<Arc<dyn AcquisitionFunction>> = (0..3)
            .map(|k| {
                Arc::new(PosteriorMean::new(model(0.1 * k as f64), true).unwrap())
                    as Arc<dyn AcquisitionFunction>
            })
            .collect();
        let values: Vec<f64> = pm.iter().map(|a| a.forward(&x).unwrap()).collect();
        let avg = EnsembleAverage::new(pm).unwrap();
        let v = avg.forward(&x).unwrap();
        assert!((v - values.iter().sum::<f64>() / 3.0).abs() < 1e-15);
        assert!(!avg.is_log());

        let lei = log_eis();
        let values: Vec<f64> = lei.iter().map(|a| a.forward(&x).unwrap()).collect();
        let avg = EnsembleAverage::new(lei).unwrap();
        let expected = (values.iter().map(|v| v.exp()).sum::<f64>() / 3.0).ln();
        let (v, g) = avg.forward_with_grad(&x).unwrap();
        assert!((v - expected).abs() < 1e-13 * expected.abs().max(1.0));
        assert!(avg.is_log());
        let h = 1e-6;
        for k in 0..2 {
            let mut xp = x.clone();
            xp[(0, k)] += h;
            let mut xm = x.clone();
            xm[(0, k)] -= h;
            let fd = (avg.forward(&xp).unwrap() - avg.forward(&xm).unwrap()) / (2.0 * h);
            assert!((fd - g[(0, k)]).abs() < 1e-6 * fd.abs().max(1.0));
        }
        let many = avg.forward_many(std::slice::from_ref(&x)).unwrap();
        assert_eq!(many[0], v);
    }

    #[test]
    fn weighted_average() {
        let x = Matrix::from_row_slice(1, 2, &[0.35, 0.6]);
        let lei = log_eis();
        let values: Vec<f64> = lei.iter().map(|a| a.forward(&x).unwrap()).collect();
        let avg = EnsembleAverage::with_weights(lei, vec![3.0, 0.0, 1.0]).unwrap();
        let expected = ((3.0 * values[0].exp() + values[2].exp()) / 4.0).ln();
        let v = avg.forward(&x).unwrap();
        assert!((v - expected).abs() < 1e-13 * expected.abs().max(1.0));
    }

    #[test]
    fn invalid_members() {
        assert!(EnsembleAverage::new(Vec::new()).is_err());
        let a: Arc<dyn AcquisitionFunction> =
            Arc::new(PosteriorMean::new(model(0.0), true).unwrap());
        let b: Arc<dyn AcquisitionFunction> =
            Arc::new(LogExpectedImprovement::new(model(0.0), 0.5, true).unwrap());
        assert!(EnsembleAverage::new(vec![a.clone(), b]).is_err());
        assert!(EnsembleAverage::with_weights(vec![a.clone()], vec![0.0]).is_err());
        assert!(EnsembleAverage::with_weights(vec![a.clone()], vec![-1.0]).is_err());
        assert!(EnsembleAverage::with_weights(vec![a], vec![1.0, 2.0]).is_err());
    }
}
