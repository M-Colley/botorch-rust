//! The noiseless fantasy models of BoTorch's fantasy-based analytic noisy
//! expected improvement (`_get_noiseless_fantasy_model` of
//! `botorch.acquisition.analytic`): copies of a fixed-noise [`SingleTaskGP`]
//! with the same hyperparameters and (eval-mode) transforms, trained on the
//! observed points with fantasized targets and a tiny fixed noise.

use std::sync::OnceLock;

use super::{Likelihood, PredictionCache, SingleTaskGP};
use crate::error::{BotorchError, Result};
use crate::linalg::psd_safe_cholesky;
use crate::{Matrix, Vector};

impl SingleTaskGP {
    /// The noiseless fantasy models of BoTorch's `_get_noiseless_fantasy_model(
    /// model, batch_X_observed, Y_fantasized)`: one model per row of `ys`
    /// (`n_f × n` fantasized targets in the original outcome space) on the
    /// raw inputs `x` (`n × d`), with the noise variance
    /// `noise_tf` (transformed space, not clamped: BoTorch assigns it to the
    /// likelihood after construction) at every point. The models share the
    /// factor of the training covariance.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] unless the model has a fixed-noise
    /// likelihood (BoTorch requires a `FixedNoiseGaussianLikelihood`);
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes; errors of
    /// the input transform and the factorization.
    pub(crate) fn noiseless_fantasies(
        &self,
        x: &Matrix,
        ys: &Matrix,
        noise_tf: f64,
    ) -> Result<Vec<SingleTaskGP>> {
        if !matches!(self.likelihood, Likelihood::FixedNoise { .. }) {
            return Err(BotorchError::Unsupported(
                "noiseless fantasy models require a fixed-noise likelihood".into(),
            ));
        }
        self.require_no_perturbation("the noiseless fantasy models")?;
        self.require_plain_outcome("the noiseless fantasy models")?;
        let n = x.nrows();
        if ys.ncols() != n || n == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected fantasized targets with {n} columns, got {}x{}",
                ys.nrows(),
                ys.ncols()
            )));
        }
        let x_tf = self.transform_inputs(x)?;
        let kernel = self.covar()?;
        let train_scaled = kernel.prepare(&x_tf)?;
        let mut k = kernel.symmetric(&train_scaled);
        for i in 0..n {
            k[(i, i)] += noise_tf;
        }
        let chol = psd_safe_cholesky(&k)?;
        let (mu, sd) = self.outcome_affine()?;
        let c = self.constant();
        let noise = Vector::from_element(n, noise_tf);
        let train_yvar = Some(Matrix::from_element(n, 1, noise_tf * sd * sd));
        (0..ys.nrows())
            .map(|f| {
                let y = Matrix::from_fn(n, 1, |i, _| ys[(f, i)]);
                let y_tf = Vector::from_fn(n, |i, _| (ys[(f, i)] - mu) / sd);
                let alpha = chol.solve_vec(&y_tf.add_scalar(-c));
                let cache = PredictionCache {
                    kernel: kernel.clone(),
                    train_scaled: train_scaled.clone(),
                    chol: chol.clone(),
                    alpha,
                };
                Ok(SingleTaskGP {
                    train_x: x.clone(),
                    train_y: y,
                    train_yvar: train_yvar.clone(),
                    train_x_tf: x_tf.clone(),
                    train_y_tf: y_tf,
                    input_transform: self.input_transform.clone(),
                    input_chain: self.input_chain.clone(),
                    construction_x_tf: None,
                    outcome_transform: self.outcome_transform.clone(),
                    outcome_chain: self.outcome_chain.clone(),
                    covar_module: self.covar_module.clone(),
                    likelihood: Likelihood::FixedNoise {
                        noise: noise.clone(),
                    },
                    mean_module: self.mean_module,
                    hyperparameters: self.hyperparameters.clone(),
                    point_noise: None,
                    cache: OnceLock::from(Ok(cache)),
                })
            })
            .collect()
    }
}
