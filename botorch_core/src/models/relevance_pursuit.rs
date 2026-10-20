//! Sparse outlier variances and joint relevance-pursuit optimization.
//!
//! The support is structural state; active variances are optimized jointly
//! with the usual GP hyperparameters. No dense covariance work leaves Rust.

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult};
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::SingleTaskGP;
use crate::priors::Prior;
use crate::{Matrix, Vector};

/// Sparse outlier-noise state in the transformed outcome space.
#[derive(Clone, Debug)]
pub struct SparseOutlierState {
    /// Training observation indices, in parameter order.
    pub support: Vec<usize>,
    /// One raw parameter for each active observation.
    pub raw: Vec<f64>,
    /// If true, `raw / (1 - raw)` multiplies the original covariance diagonal.
    pub convex: bool,
    /// Non-transformed optimizer bounds.
    pub bounds: (f64, f64),
    /// Prior on the dense variance factors (including inactive zeros).
    pub prior: Option<Prior>,
}

impl SparseOutlierState {
    /// Validates support indices, bounds and parameter values atomically.
    pub fn validate(&self, n: usize) -> Result<()> {
        let (lower, upper) = self.bounds;
        if lower < 0.0
            || !lower.is_finite()
            || upper <= lower
            || upper.is_nan()
            || (self.convex && upper >= 1.0)
        {
            return Err(BotorchError::InvalidArgument(
                "invalid sparse outlier bounds".into(),
            ));
        }
        if self.raw.len() != self.support.len() {
            return Err(BotorchError::DimensionMismatch(
                "one rho per support index is required".into(),
            ));
        }
        let mut seen = vec![false; n];
        for (&i, &r) in self.support.iter().zip(&self.raw) {
            if i >= n || seen[i] {
                return Err(BotorchError::InvalidArgument(
                    "outlier indices must be distinct and in range".into(),
                ));
            }
            seen[i] = true;
            if !r.is_finite() || r < lower || r > upper {
                return Err(BotorchError::InvalidArgument(
                    "rho is outside its non-transformed constraint".into(),
                ));
            }
        }
        Ok(())
    }

    /// Dense additive-variance factors.
    pub fn factors(&self, n: usize) -> Vec<f64> {
        let mut values = vec![0.0; n];
        for (&i, &r) in self.support.iter().zip(&self.raw) {
            values[i] = if self.convex { r / (1.0 - r) } else { r };
        }
        values
    }

    /// Adds outlier variances, retaining the original covariance diagonal.
    pub fn inflate(&self, covariance: &mut Matrix) -> Vec<f64> {
        let diagonal: Vec<f64> = covariance.diagonal().iter().copied().collect();
        for (&i, &r) in self.support.iter().zip(&self.raw) {
            let factor = if self.convex { r / (1.0 - r) } else { r };
            covariance[(i, i)] += if self.convex {
                diagonal[i] * factor
            } else {
                factor
            };
        }
        diagonal
    }

    /// Dense-factor prior value.
    pub fn log_prior(&self, n: usize) -> f64 {
        self.prior.map_or(0.0, |p| {
            self.factors(n).iter().map(|&v| p.log_prob(v)).sum()
        })
    }

    /// Rho gradients and the chain rule for base covariance parameters.
    pub fn gradients(&self, w: &mut Matrix, diagonal: &[f64], n: f64) -> Vector {
        let mut grad = Vector::zeros(self.raw.len());
        for (j, (&i, &r)) in self.support.iter().zip(&self.raw).enumerate() {
            let (factor, deriv) = if self.convex {
                (r / (1.0 - r), 1.0 / (1.0 - r).powi(2))
            } else {
                (r, 1.0)
            };
            let scale = if self.convex { diagonal[i] } else { 1.0 };
            grad[j] = (0.5 * w[(i, i)] * scale
                + self.prior.map_or(0.0, |p| p.grad_log_prob(factor)))
                * deriv
                / n;
            if self.convex {
                w[(i, i)] *= 1.0 + factor;
            }
        }
        grad
    }
}

impl SingleTaskGP {
    /// Active rho parameters followed by ordinary raw GP parameters.
    pub fn relevance_parameters(&self) -> Vector {
        let mut raw = self
            .sparse_outlier_noise()
            .map_or_else(Vec::new, |s| s.raw.clone());
        raw.extend_from_slice(self.raw_parameters().as_slice());
        Vector::from_vec(raw)
    }

    /// Sets active rho and ordinary GP parameters atomically.
    pub fn set_relevance_parameters(&mut self, values: &[f64]) -> Result<()> {
        let k = self.sparse_outlier_noise().map_or(0, |s| s.raw.len());
        if values.len() != k + self.parameter_layout().len() {
            return Err(BotorchError::DimensionMismatch(
                "wrong number of joint rho/GP parameters".into(),
            ));
        }
        let mut next = self.clone();
        if let Some(state) = self.sparse_outlier_noise() {
            let mut state = state.clone();
            state.raw = values[..k].to_vec();
            // Bound endpoints can be missed by a few ulps by line-search
            // arithmetic. Project only these numerical excursions, so model
            // state remains exactly feasible and priors never see negative rho.
            for value in &mut state.raw {
                let tolerance = 8.0 * f64::EPSILON * (1.0 + value.abs());
                if *value < state.bounds.0 && *value >= state.bounds.0 - tolerance {
                    *value = state.bounds.0;
                } else if *value > state.bounds.1 && *value <= state.bounds.1 + tolerance {
                    *value = state.bounds.1;
                }
            }
            next.set_sparse_outlier_noise(Some(state))?;
        }
        next.set_raw_parameters(&values[k..])?;
        *self = next;
        Ok(())
    }

    /// Optimizer bounds in joint parameter order.
    pub fn relevance_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = self
            .sparse_outlier_noise()
            .map_or_else(Vec::new, |s| vec![s.bounds; s.raw.len()]);
        bounds.extend(self.raw_parameter_bounds());
        bounds
    }

    /// Exact analytic gradients with respect to rho and all GP parameters.
    pub fn relevance_value_and_grad(&self, values: &[f64]) -> Result<(f64, Vector)> {
        let mut trial = self.clone();
        trial.set_relevance_parameters(values)?;
        let (value, gradient, rho) = ExactMarginalLogLikelihood::new(&trial)
            .value_and_rho_grad(trial.raw_parameters().as_slice())?;
        let mut joint = rho.as_slice().to_vec();
        joint.extend_from_slice(gradient.as_slice());
        Ok((value, Vector::from_vec(joint)))
    }

    /// LOO optimal variance increments, centered at the model's mean.
    pub fn relevance_expansion_objective(&self) -> Result<Vec<f64>> {
        let chol = self.train_cholesky()?;
        let inverse = chol.inverse();
        let residual = self
            .transformed_train_targets()
            .add_scalar(-self.constant());
        let alpha = chol.solve_vec(&residual);
        let factors = self.sparse_outlier_noise().map_or_else(
            || vec![0.0; self.num_train()],
            |s| s.factors(self.num_train()),
        );
        Ok((0..self.num_train())
            .map(|i| {
                let variance = 1.0 / inverse[(i, i)];
                ((alpha[i] * variance).powi(2) - variance - factors[i]).max(0.0)
            })
            .collect())
    }
}

/// Fits all active rho and dense parameters with the native bounded solver.
pub fn fit_sparse_outlier_gp(model: &mut SingleTaskGP, options: FitOptions) -> Result<FitResult> {
    let initial = model.relevance_parameters();
    let bounds = model.relevance_bounds();
    let (result, values) = minimize_with_retries(
        initial.as_slice(),
        &bounds,
        options,
        |p| {
            model
                .relevance_value_and_grad(p)
                .map(|(v, g)| (v, g.as_slice().to_vec()))
        },
        |rng| {
            let mut values = model
                .sparse_outlier_noise()
                .expect("sparse fit")
                .raw
                .clone();
            values.extend(crate::fit::sample_all_priors(
                model,
                model.hyperparameters(),
                rng,
            ));
            values
        },
    )?;
    model.set_relevance_parameters(&values)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SingleTaskGPOptions;
    use crate::OutcomeTransformChoice;

    fn model(convex: bool) -> SingleTaskGP {
        let mut model = SingleTaskGP::new(
            Matrix::from_column_slice(5, 1, &[0., 0.2, 0.45, 0.7, 1.]),
            Matrix::from_column_slice(5, 1, &[0., 0.5, 4., 0.7, -0.1]),
            SingleTaskGPOptions {
                outcome_transform: OutcomeTransformChoice::None,
                ..Default::default()
            },
        )
        .unwrap();
        model
            .set_sparse_outlier_noise(Some(SparseOutlierState {
                support: vec![2, 0],
                raw: vec![0.25, 0.15],
                convex,
                bounds: (0., if convex { 0.999 } else { f64::INFINITY }),
                prior: None,
            }))
            .unwrap();
        model
    }

    #[test]
    fn joint_analytic_gradient_matches_finite_difference() {
        for convex in [false, true] {
            let model = model(convex);
            let params = model.relevance_parameters();
            let (_, gradient) = model.relevance_value_and_grad(params.as_slice()).unwrap();
            for j in 0..params.len() {
                let step = 1e-6;
                let mut plus = params.as_slice().to_vec();
                plus[j] += step;
                let mut minus = params.as_slice().to_vec();
                minus[j] -= step;
                let a = model.relevance_value_and_grad(&plus).unwrap().0;
                let b = model.relevance_value_and_grad(&minus).unwrap().0;
                let numeric = (a - b) / (2. * step);
                assert!(
                    (numeric - gradient[j]).abs() < 2e-5 * (1. + numeric.abs()),
                    "{convex} parameter {j}: {numeric} != {}",
                    gradient[j]
                );
            }
        }
    }

    #[test]
    fn native_fit_improves_joint_objective_and_invalidates_prediction_cache() {
        let mut model = model(true);
        let x = Matrix::from_column_slice(2, 1, &[0.3, 0.8]);
        let before = model.posterior(&x, false).unwrap();
        let initial = ExactMarginalLogLikelihood::new(&model).value().unwrap();
        fit_sparse_outlier_gp(
            &mut model,
            FitOptions {
                maxiter: 100,
                max_attempts: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let final_value = ExactMarginalLogLikelihood::new(&model).value().unwrap();
        assert!(final_value > initial + 0.1);
        let after = model.posterior(&x, false).unwrap();
        assert!((before.mean - after.mean).norm() > 1e-4);
        assert!(model
            .sparse_outlier_noise()
            .unwrap()
            .raw
            .iter()
            .all(|&v| (0.0..=0.999).contains(&v)));
    }

    #[test]
    fn support_validation_is_atomic() {
        let mut model = model(true);
        let before = model.relevance_parameters();
        let mut state = model.sparse_outlier_noise().unwrap().clone();
        state.support = vec![2, 2];
        assert!(model.set_sparse_outlier_noise(Some(state)).is_err());
        assert_eq!(model.relevance_parameters(), before);
        let mut params = before.as_slice().to_vec();
        params[0] = 1.;
        assert!(model.set_relevance_parameters(&params).is_err());
        assert_eq!(model.relevance_parameters(), before);
    }
}
