//! Exact marginal log likelihood (`gpytorch.mlls.ExactMarginalLogLikelihood`)
//! with analytic gradients.
//!
//! For transformed training data `(X̃, ỹ)`, constant mean `c`, kernel matrix
//! `K` and noise `Σ` the value is
//!
//! ```text
//! mll = [ log N(ỹ | c·1, K + Σ) + Σ_priors log p(θ) ] / n
//! ```
//!
//! where every prior is evaluated on the actual (constrained) hyperparameter
//! value, exactly like GPyTorch (which also divides the prior terms by `n`).
//! The Cholesky factorization uses the same jitter fallback as
//! `psd_safe_cholesky`; the gradient is then that of the jittered objective,
//! which is what autograd produces in GPyTorch.

use crate::error::{BotorchError, Result};
use crate::linalg::psd_safe_cholesky;
use crate::models::{Hyperparameters, Likelihood, SingleTaskGP};
use crate::Vector;

/// `log(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// The exact marginal log likelihood of a [`SingleTaskGP`].
///
/// The MLL borrows the model (its data, priors and constraints) and can be
/// evaluated at arbitrary raw hyperparameters without modifying the model.
#[derive(Clone, Copy, Debug)]
pub struct ExactMarginalLogLikelihood<'a> {
    model: &'a SingleTaskGP,
}

impl<'a> ExactMarginalLogLikelihood<'a> {
    /// Creates the MLL of `model`.
    pub fn new(model: &'a SingleTaskGP) -> Self {
        Self { model }
    }

    /// The underlying model.
    pub fn model(&self) -> &'a SingleTaskGP {
        self.model
    }

    /// The MLL at the model's current hyperparameters.
    ///
    /// # Errors
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized.
    pub fn value(&self) -> Result<f64> {
        self.evaluate(self.model.raw_parameters().as_slice(), false)
            .map(|(v, _)| v)
    }

    /// The MLL at the given flat raw hyperparameters (see
    /// [`crate::ParameterLayout`]).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of parameters,
    /// [`BotorchError::InvalidArgument`] for invalid values and
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized.
    pub fn value_at(&self, raw_params: &[f64]) -> Result<f64> {
        self.evaluate(raw_params, false).map(|(v, _)| v)
    }

    /// The MLL and its gradient with respect to all raw hyperparameters (in
    /// the flat order of [`crate::ParameterLayout`]).
    ///
    /// # Errors
    /// See [`ExactMarginalLogLikelihood::value_at`].
    pub fn value_and_grad(&self, raw_params: &[f64]) -> Result<(f64, Vector)> {
        self.evaluate(raw_params, true)
            .map(|(v, g)| (v, g.expect("gradient requested")))
    }

    fn evaluate(&self, raw_params: &[f64], with_grad: bool) -> Result<(f64, Option<Vector>)> {
        let model = self.model;
        let layout = model.parameter_layout();
        let h = Hyperparameters::from_slice(layout, raw_params)?;
        if raw_params.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "non-finite raw hyperparameters: {raw_params:?}"
            )));
        }
        let spec = model.covar_module();
        let kernel = spec.build(h.raw_lengthscale.as_slice(), h.raw_outputscale)?;
        let xs = kernel.scale_points(model.transformed_train_inputs());
        let mut k = kernel.symmetric_scaled(&xs);
        let n = model.num_train();
        let nf = n as f64;

        // Likelihood noise.
        let mut noise_value = None;
        match model.likelihood() {
            Likelihood::Gaussian(lik) => {
                let raw = h
                    .raw_noise
                    .expect("homoskedastic likelihood has a raw noise");
                let noise = lik.noise_constraint.transform(raw);
                for i in 0..n {
                    k[(i, i)] += noise;
                }
                noise_value = Some(noise);
            }
            Likelihood::FixedNoise { noise } => {
                for i in 0..n {
                    k[(i, i)] += noise[i];
                }
            }
        }

        let chol = psd_safe_cholesky(&k)?;
        let resid = model
            .transformed_train_targets()
            .add_scalar(-h.raw_constant);
        let v = chol.solve_lower_vec(&resid);
        let inv_quad = v.dot(&v);
        let logdet = chol.logdet();
        let mut res = -0.5 * (inv_quad + logdet + nf * LN_2PI);

        // Prior terms, in GPyTorch's `named_priors()` order.
        let lengthscale = kernel.lengthscale();
        let outputscale = kernel.outputscale();
        if let (Likelihood::Gaussian(lik), Some(noise)) = (model.likelihood(), noise_value) {
            if let Some(prior) = lik.noise_prior {
                res += prior.log_prob(noise);
            }
        }
        if let (Some(scale), Some(s)) = (spec.scale, outputscale) {
            if let Some(prior) = scale.outputscale_prior {
                res += prior.log_prob(s);
            }
        }
        if let Some(prior) = spec.lengthscale_prior {
            res += lengthscale.iter().map(|&l| prior.log_prob(l)).sum::<f64>();
        }
        let value = res / nf;
        if !with_grad {
            return Ok((value, None));
        }

        // Gradient: dmll/dθ = [½ tr((ααᵀ - K⁻¹) ∂K/∂θ) + ∂ log p/∂θ] / n.
        let alpha = chol.solve_lower_transpose_vec(&v);
        let mut w = chol.inverse();
        w.ger(1.0, &alpha, &alpha, -1.0); // W = ααᵀ - K⁻¹
        let (gl, gs) = kernel.contract_hyper_grad(&xs, &w);

        let mut grad = Vector::zeros(layout.len());
        grad[layout.constant_index()] = alpha.sum() / nf;
        if let (Some(idx), Likelihood::Gaussian(lik), Some(noise)) =
            (layout.noise_index(), model.likelihood(), noise_value)
        {
            let mut g = 0.5 * w.trace();
            if let Some(prior) = lik.noise_prior {
                g += prior.grad_log_prob(noise);
            }
            let raw = h.raw_noise.expect("raw noise present");
            grad[idx] = g * lik.noise_constraint.transform_derivative(raw) / nf;
        }
        if let (Some(idx), Some(scale), Some(s), Some(gs)) =
            (layout.outputscale_index(), spec.scale, outputscale, gs)
        {
            let mut g = 0.5 * gs;
            if let Some(prior) = scale.outputscale_prior {
                g += prior.grad_log_prob(s);
            }
            let raw = h.raw_outputscale.expect("raw outputscale present");
            grad[idx] = g * scale.outputscale_constraint.transform_derivative(raw) / nf;
        }
        for (j, idx) in layout.lengthscale_range().enumerate() {
            let mut g = 0.5 * gl[j];
            if let Some(prior) = spec.lengthscale_prior {
                g += prior.grad_log_prob(lengthscale[j]);
            }
            grad[idx] = g * spec
                .lengthscale_constraint
                .transform_derivative(h.raw_lengthscale[j])
                / nf;
        }
        Ok((value, Some(grad)))
    }
}
