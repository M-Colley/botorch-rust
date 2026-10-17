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
//!
//! At raw hyperparameters where GPyTorch's covariance is NaN (overflowing
//! squared distances or lengthscales that underflowed to zero, see
//! [`crate::kernels::nonfinite`]), the value raises [`BotorchError::Nan`]
//! like linear_operator's `NanError`, and gradients that GPyTorch computes
//! as NaN are NaN.
//!
//! With learned input-transform parameters `φ` (e.g. `Warp`'s
//! concentrations), the transformed training inputs `X̃(φ)` are recomputed
//! in train mode at every evaluation (BoTorch transforms the training inputs
//! in the model's `forward`), the transform's priors are added, and the
//! gradient with respect to `φ` is `[Σ_a ∂mll/∂x̃_a · ∂x̃_a/∂φ]/n` with
//! `∂mll/∂x̃_a = Σ_b W_ab ∂₁k(x̃_a, x̃_b)` for `W = ααᵀ - K⁻¹`.

use crate::error::{BotorchError, Result};
use crate::kernels::{BuiltCovar, CovarModule, Prepared};
use crate::linalg::psd_safe_cholesky;
use crate::models::{Hyperparameters, Likelihood, SingleTaskGP};
use crate::priors::Prior;
use crate::transforms::input::TransformMode;
use crate::{Matrix, Vector};

/// `log(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// The MLL's derivative with respect to a raw hyperparameter whose value is
/// `v`: `(half_data + p'(v)) · deriv / n` with the data term `half_data`,
/// the prior `p` and the derivative `deriv` of the constraint transform.
/// If the prior's derivative overflows (e.g. `(α - 1) / v` for a softplus
/// value `v` that is subnormal), it is recomputed in torch's order, which
/// applies the `1/n` first ([`Prior::scaled_grad_log_prob`]) and overflows
/// only where GPyTorch's gradient does.
pub(crate) fn raw_gradient(
    half_data: f64,
    prior: Option<Prior>,
    v: f64,
    deriv: f64,
    nf: f64,
) -> f64 {
    let mut g = half_data;
    if let Some(p) = prior {
        g += p.grad_log_prob(v);
    }
    match prior {
        Some(p) if !g.is_finite() && half_data.is_finite() => {
            (half_data / nf + p.scaled_grad_log_prob(v, 1.0 / nf)) * deriv
        }
        _ => g * deriv / nf,
    }
}

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
    /// [`BotorchError::InvalidArgument`] for non-finite raw values and
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized (`Nan` also where GPyTorch's covariance is NaN,
    /// e.g. for lengthscales that underflowed to zero).
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
        // Hyperparameters from raw values may have underflowed to zero; the
        // covariance then has GPyTorch's NaN entries (`training_nan`), and
        // the Cholesky factorization raises `Nan` like linear_operator.
        let covar = model.build_covar_unchecked(&h)?;
        // Learned input-transform parameters: the training inputs at `φ`.
        let input = match model.input_chain() {
            Some(chain) if layout.input > 0 => {
                let chain = chain.with_params(h.raw_input.as_slice())?;
                // Non-finite transformed inputs propagate into the
                // covariance as in GPyTorch (`training_nan` below), where
                // the Cholesky factorization raises `Nan` (BoTorch's
                // closure: NaN), rather than ending the fit.
                let (x_tf, tape) =
                    chain.transform_with_tape_mode(model.train_inputs(), TransformMode::Train)?;
                Some((chain, x_tf, tape))
            }
            _ => None,
        };
        let train_x_tf = match &input {
            Some((_, x_tf, _)) => x_tf,
            None => model.transformed_train_inputs(),
        };
        let xs = covar.prepare(train_x_tf)?;
        let mut k = covar.symmetric(&xs);
        // Inputs from learned transforms require gradients in GPyTorch.
        let nan = covar.training_nan(train_x_tf, input.is_some());
        nan.inject(&mut k);
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
        if let (Likelihood::Gaussian(lik), Some(noise)) = (model.likelihood(), noise_value) {
            if let Some(prior) = lik.noise_prior {
                res += prior.log_prob(noise);
            }
        }
        let mean_prior = model.mean_module().prior;
        if let Some(prior) = mean_prior {
            if !prior.in_validated_support(h.raw_constant) {
                return Err(BotorchError::InvalidArgument(format!(
                    "Expected value argument ({}) to be within the support of the \
                     distribution {}Prior, but found invalid values",
                    h.raw_constant,
                    prior.name()
                )));
            }
            res += prior.log_prob(h.raw_constant);
        }
        let mut tree_prior_grad = None;
        match (model.covar_module(), &covar) {
            (CovarModule::Standard(spec), BuiltCovar::Standard(kernel)) => {
                let lengthscale = kernel.lengthscale();
                if let (Some(scale), Some(s)) = (spec.scale, kernel.outputscale()) {
                    if let Some(prior) = scale.outputscale_prior {
                        res += prior.log_prob(s);
                    }
                }
                if let Some(prior) = spec.lengthscale_prior {
                    res += lengthscale.iter().map(|&l| prior.log_prob(l)).sum::<f64>();
                }
            }
            (CovarModule::Tree(_), BuiltCovar::Tree(tree)) => {
                if with_grad {
                    let (lp, g) = tree.log_prior_and_grad()?;
                    res += lp;
                    tree_prior_grad = Some(g);
                } else {
                    res += tree.log_prior()?;
                }
            }
            _ => unreachable!("covariance module and kernel of different kinds"),
        }
        let input_prior = input
            .as_ref()
            .map(|(chain, _, _)| chain.log_prior_and_grad());
        if let Some((lp, _)) = &input_prior {
            res += lp;
        }
        let value = res / nf;
        if !with_grad {
            return Ok((value, None));
        }

        // Gradient: dmll/dθ = [½ tr((ααᵀ - K⁻¹) ∂K/∂θ) + ∂ log p/∂θ] / n.
        let alpha = chol.solve_lower_transpose_vec(&v);
        let mut w = chol.inverse();
        w.ger(1.0, &alpha, &alpha, -1.0); // W = ααᵀ - K⁻¹

        let mut grad = Vector::zeros(layout.len());
        let mut g_constant = alpha.sum();
        if let Some(prior) = mean_prior {
            g_constant += prior.grad_log_prob(h.raw_constant);
        }
        grad[layout.constant_index()] = g_constant / nf;
        if let (Some(idx), Likelihood::Gaussian(lik), Some(noise)) =
            (layout.noise_index(), model.likelihood(), noise_value)
        {
            let raw = h.raw_noise.expect("raw noise present");
            let deriv = lik.noise_constraint.transform_derivative(raw);
            grad[idx] = raw_gradient(0.5 * w.trace(), lik.noise_prior, noise, deriv, nf);
        }
        match (model.covar_module(), &covar, &xs) {
            (
                CovarModule::Standard(spec),
                BuiltCovar::Standard(kernel),
                Prepared::Standard(points),
            ) => {
                let (gl, gs) = kernel.contract_hyper_grad(points, &w);
                let lengthscale = kernel.lengthscale();
                let outputscale = kernel.outputscale();
                if let (Some(idx), Some(scale), Some(s), Some(gs)) =
                    (layout.outputscale_index(), spec.scale, outputscale, gs)
                {
                    let raw = h.raw_outputscale.expect("raw outputscale present");
                    let deriv = scale.outputscale_constraint.transform_derivative(raw);
                    grad[idx] = raw_gradient(0.5 * gs, scale.outputscale_prior, s, deriv, nf);
                }
                for (j, idx) in layout.lengthscale_range().enumerate() {
                    let deriv = spec
                        .lengthscale_constraint
                        .transform_derivative(h.raw_lengthscale[j]);
                    grad[idx] = raw_gradient(
                        0.5 * gl[j],
                        spec.lengthscale_prior,
                        lengthscale[j],
                        deriv,
                        nf,
                    );
                }
            }
            (CovarModule::Tree(spec), BuiltCovar::Tree(tree), Prepared::Tree(points)) => {
                let gk = tree.contract(points, &w);
                let gp = tree_prior_grad.expect("prior gradient computed");
                let derivs = spec.transform_derivatives(h.raw_kernel.as_slice());
                let mut scaled = None;
                for (j, idx) in layout.kernel_range().enumerate() {
                    let g = 0.5 * gk[j] + gp[j];
                    grad[idx] = if g.is_finite() || !gk[j].is_finite() {
                        g * derivs[j] / nf
                    } else {
                        // A prior derivative overflowed: torch's order (see
                        // `raw_gradient`).
                        if scaled.is_none() {
                            scaled = Some(tree.log_prior_scaled_grad(1.0 / nf)?);
                        }
                        let gs = scaled.as_ref().expect("computed")[j];
                        (0.5 * gk[j] / nf + gs) * derivs[j]
                    };
                }
            }
            _ => unreachable!("covariance module and kernel of different kinds"),
        }
        if let (Some((chain, x_tf, tape)), Some((_, prior_grad))) = (&input, &input_prior) {
            // ∂mll/∂x̃_a = Σ_b W_ab ∂₁k(x̃_a, x̃_b) (symmetric W and k).
            let dt = x_tf.ncols();
            let mut gx = Matrix::zeros(n, dt);
            for a in 0..n {
                let (_, dk) = covar.cross_with_grad(&xs, a, &xs);
                let g = dk.tr_mul(&w.row(a).transpose());
                for k in 0..dt {
                    gx[(a, k)] = g[k];
                }
            }
            let (_, pbar) = chain.backward_with_params(tape, 0..n, &gx)?;
            for (j, idx) in layout.input_range().enumerate() {
                grad[idx] = (pbar[j] + prior_grad[j]) / nf;
            }
        }
        // GPyTorch's NaN lengthscale derivatives (`RBFCovariance` /
        // `MaternCovariance` where `inf · 0` occurs).
        for &j in nan.nan_gradients() {
            let idx = match model.covar_module() {
                CovarModule::Standard(_) => layout.lengthscale_range().start + j,
                CovarModule::Tree(_) => layout.kernel_range().start + j,
            };
            grad[idx] = f64::NAN;
        }
        Ok((value, Some(grad)))
    }
}
