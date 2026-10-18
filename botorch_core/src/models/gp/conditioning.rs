//! Conditioning a [`SingleTaskGP`] on new observations and fantasy models
//! (BoTorch's `condition_on_observations` and `fantasize`).
//!
//! A conditioned model has the training data of the original model followed
//! by the new points, the original hyperparameters and the original (fitted)
//! input and outcome transforms, applied in eval mode: `Normalize` keeps its
//! bounds and `Standardize` its statistics, exactly like BoTorch's fantasy
//! models. The new observations are given in the original outcome space;
//! their `noise` (observation-noise variances) in the *transformed* space,
//! as BoTorch's `SingleTaskGP.condition_on_observations` assumes.
//!
//! `fantasize` evaluates the input transform in BoTorch's fantasize mode
//! (steps with `transform_on_fantasize=False`, e.g. `InputPerturbation`, are
//! skipped) for both the posterior of the fantasized observations and the
//! new training inputs; with a nonlinear outcome transform the observations
//! are sampled from the `TransformedPosterior` (original space) and
//! transformed back by the outcome transform, like in BoTorch.
//!
//! The Cholesky factor of the enlarged training covariance is obtained by an
//! exact rank-`n'` update of the cached factor
//! ([`CholeskyFactor::bordered`], what GPyTorch's `get_fantasy_strategy`
//! does); the full factorization is used instead when the cached factor
//! needed jitter or the update is not positive definite. Fantasy models at
//! the same new points share the factor and differ only in their targets.
//!
//! [`FantasyGp`] holds the `n_f` fantasy models of one output at a common set
//! of new points (the fantasy batch of BoTorch's `fantasize`) with the
//! reverse-mode derivatives of their posteriors with respect to the test
//! points *and* the new points, through the fantasized observations and the
//! updated factor. One-shot acquisition functions (hypervolume knowledge
//! gradient) are built on it.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use super::{Likelihood, ObservationNoise, PredictionCache, SingleTaskGP, MIN_FIXED_NOISE};
use crate::error::{BotorchError, Result};
use crate::kernels::Prepared;
use crate::linalg::{psd_safe_cholesky, CholeskyFactor, CovarianceRoot};
use crate::posterior::Posterior;
use crate::sampling::SobolQMCNormalSampler;
use crate::transforms::input::TransformMode;
use crate::{Matrix, Vector};

/// Rows of `a` followed by the rows of `b` (same number of columns).
fn vstack(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.nrows() + b.nrows(), a.ncols());
    out.rows_mut(0, a.nrows()).copy_from(a);
    out.rows_mut(a.nrows(), b.nrows()).copy_from(b);
    out
}

/// The factor and scaled points of a model's training data enlarged by new
/// (transformed) points with the given noise variances.
struct Enlarged {
    full_scaled: Prepared,
    chol: CholeskyFactor,
}

impl SingleTaskGP {
    /// The noise variance (transformed space) on the diagonal of the
    /// training covariance at training point `i`, given the likelihood noise
    /// `noise` (one value, or one per point for fixed noise).
    pub(super) fn training_noise(&self, noise: &Vector, i: usize) -> f64 {
        if let Some(Some(v)) = self.point_noise.as_ref().map(|p| p[i]) {
            return v;
        }
        if noise.len() == 1 {
            noise[0]
        } else {
            noise[i]
        }
    }

    /// The noise variance (transformed space) that `fantasize` assigns to
    /// fantasy observations without an explicit `observation_noise`: `None`
    /// (the inferred noise level) for a homoskedastic likelihood, the mean of
    /// the training noise for fixed noise (BoTorch's `fantasize`).
    pub fn default_fantasy_noise(&self) -> Option<f64> {
        match &self.likelihood {
            Likelihood::Gaussian(_) => None,
            Likelihood::FixedNoise { noise } => Some(noise.sum() / noise.len() as f64),
        }
    }

    /// The factor of the training covariance enlarged by the transformed
    /// points `x_tf` with noise variances `noise_tf`, by a rank update of
    /// the cached factor where possible.
    fn enlarged(&self, x_tf: &Matrix, noise_tf: &[f64]) -> Result<Enlarged> {
        let cache = self.cache()?;
        let kernel = &cache.kernel;
        let n = self.num_train();
        let full_scaled = kernel.prepare(&vstack(&self.train_x_tf, x_tf))?;
        let new_scaled = kernel.prepare(x_tf)?;
        let cross = kernel.cross(&new_scaled, &cache.train_scaled);
        let mut corner = kernel.symmetric(&new_scaled);
        for (j, &v) in noise_tf.iter().enumerate() {
            corner[(j, j)] += v;
        }
        let chol = match cache.chol.bordered(&cross, &corner) {
            Some(c) => c,
            None => {
                let mut k = kernel.symmetric(&full_scaled);
                let noise = self.likelihood_noise();
                for i in 0..n {
                    k[(i, i)] += self.training_noise(&noise, i);
                }
                for (j, &v) in noise_tf.iter().enumerate() {
                    k[(n + j, n + j)] += v;
                }
                psd_safe_cholesky(&k)?
            }
        };
        Ok(Enlarged { full_scaled, chol })
    }

    /// The noise variances (transformed space) of `k` new observations:
    /// the given ones (clamped like GPyTorch's `FixedGaussianNoise` for
    /// fixed-noise models) or, for homoskedastic models, the inferred noise.
    ///
    /// # Errors
    /// [`BotorchError::Runtime`] for a fixed-noise model without `noise`
    /// (GPyTorch's message) and [`BotorchError::InputData`] for NaN or
    /// negative noise.
    pub(super) fn new_noise(&self, noise: Option<&[f64]>, k: usize) -> Result<Vec<f64>> {
        if let Some(v) = noise {
            if v.iter().any(|x| x.is_nan() || *x < 0.0) {
                return Err(BotorchError::InputData(
                    "the observation noise must be non-negative".into(),
                ));
            }
        }
        match (&self.likelihood, noise) {
            (Likelihood::FixedNoise { .. }, None) => Err(BotorchError::Runtime(
                "FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg".into(),
            )),
            (Likelihood::FixedNoise { .. }, Some(v)) => {
                Ok(v.iter().map(|&x| x.max(MIN_FIXED_NOISE)).collect())
            }
            (Likelihood::Gaussian(_), Some(v)) => Ok(v.to_vec()),
            (Likelihood::Gaussian(_), None) => {
                let s = self.likelihood_noise()[0];
                Ok(vec![s; k])
            }
        }
    }

    /// Validates new inputs (`k × d`) and returns them transformed (in
    /// eval or fantasize mode).
    pub(super) fn check_new_inputs(&self, x: &Matrix, mode: TransformMode) -> Result<Matrix> {
        super::check_finite(x, "X")?;
        self.transform_inputs_mode(x, mode)
    }

    /// Errors for a one-to-many input transform in `mode`: conditioning on
    /// observations needs one model input per observation (BoTorch fails on
    /// the shapes). In fantasize mode, BoTorch skips the steps with
    /// `transform_on_fantasize=False` (e.g. `InputPerturbation`).
    pub(crate) fn require_one_to_one(&self, mode: TransformMode, what: &str) -> Result<()> {
        if self.dim() != self.train_x.ncols() {
            return Err(BotorchError::Unsupported(format!(
                "{what} is not supported for models whose input transform maps points of {} \
                 dimensions (posteriors) and training inputs of {} to the kernel's dimensions",
                self.dim(),
                self.train_x.ncols()
            )));
        }
        let k = self
            .input_chain
            .as_ref()
            .map_or(1, |c| c.points_per_input_mode(mode));
        if k != 1 {
            return Err(BotorchError::Unsupported(format!(
                "{what} is not supported for models with a one-to-many input transform \
                 (InputPerturbation or AppendFeatures with {k} points per input): the transformed \
                 inputs would not match the observations"
            )));
        }
        Ok(())
    }

    /// The new observations `y` (`k × 1`, original space) of the points
    /// `x_tf` transformed by the outcome transform in eval mode.
    fn transform_new_targets(&self, y: &Matrix, x_tf: &Matrix) -> Result<Vector> {
        let k = y.nrows();
        if let Some(chain) = &self.outcome_chain {
            let (y_tf, _) = chain.transform(y, None, x_tf)?;
            return Ok(Vector::from_iterator(k, y_tf.iter().copied()));
        }
        let (mu, sd) = self.outcome_affine()?;
        Ok(Vector::from_iterator(k, y.iter().map(|&v| (v - mu) / sd)))
    }

    /// The observation-noise variances of new points in the original space
    /// (the bookkeeping of `train_yvar`): the transformed noise `noise_tf`
    /// un-transformed at the transformed observations `y_tf`.
    fn untransform_new_noise(
        &self,
        y_tf: &Vector,
        noise_tf: &[f64],
        x_tf: &Matrix,
    ) -> Result<Matrix> {
        let k = noise_tf.len();
        if let Some(chain) = &self.outcome_chain {
            let y = Matrix::from_column_slice(k, 1, y_tf.as_slice());
            let v = Matrix::from_column_slice(k, 1, noise_tf);
            let (_, v_u) = chain.untransform(&y, Some(&v), x_tf)?;
            return Ok(v_u.expect("noise given"));
        }
        let (_, sd) = self.outcome_affine()?;
        Ok(Matrix::from_fn(k, 1, |j, _| noise_tf[j] * sd * sd))
    }

    /// The model conditioned on new observations (BoTorch's
    /// `model.condition_on_observations(X, Y, noise=noise)`): `x` (`n' × d`)
    /// in the raw input space, `y` (`n' × 1`) in the original outcome space
    /// and optional observation-noise variances `noise` (`n' × 1`) in the
    /// *transformed* outcome space.
    ///
    /// The result has `n + n'` training points, the same hyperparameters and
    /// the same fitted transforms (applied in eval mode). The new points get
    /// the given noise; without `noise` a homoskedastic model uses its
    /// inferred noise level, and a fixed-noise model raises GPyTorch's error.
    /// For a homoskedastic model, explicit noise enters the posterior (mean
    /// and covariance, as with meta-pytorch/botorch#3396) but not the marginal
    /// log likelihood, like BoTorch's fantasy likelihood.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::InputData`] for NaN values or negative noise,
    /// [`BotorchError::Runtime`] for a fixed-noise model without `noise`, and
    /// errors of the factorization.
    pub fn condition_on_observations(
        &self,
        x: &Matrix,
        y: &Matrix,
        noise: Option<&Matrix>,
    ) -> Result<SingleTaskGP> {
        let mut out = self.condition_on_observations_many(x, std::slice::from_ref(y), noise)?;
        Ok(out.pop().expect("one conditioned model"))
    }

    /// [`SingleTaskGP::condition_on_observations`] for several sets of
    /// observations `ys` (`n' × 1` each) at the same inputs `x`, e.g. the
    /// fantasies of BoTorch's `fantasize`: the models share the update of
    /// the factor and differ only in their targets.
    ///
    /// # Errors
    /// See [`SingleTaskGP::condition_on_observations`].
    pub fn condition_on_observations_many(
        &self,
        x: &Matrix,
        ys: &[Matrix],
        noise: Option<&Matrix>,
    ) -> Result<Vec<SingleTaskGP>> {
        self.condition_mode(x, ys, noise, TransformMode::Eval)
    }

    /// [`SingleTaskGP::condition_on_observations_many`] with the input
    /// transform in `mode` (eval, or fantasize inside `fantasize`).
    pub(crate) fn condition_mode(
        &self,
        x: &Matrix,
        ys: &[Matrix],
        noise: Option<&Matrix>,
        mode: TransformMode,
    ) -> Result<Vec<SingleTaskGP>> {
        self.require_one_to_one(mode, "condition_on_observations")?;
        let k = x.nrows();
        if x.ncols() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {} columns (d), got {}",
                self.dim(),
                x.ncols()
            )));
        }
        for y in ys {
            if y.shape() != (k, 1) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected Y of shape {k} x 1 (the new points and one output), got {}x{}",
                    y.nrows(),
                    y.ncols()
                )));
            }
            super::check_finite(y, "Y")?;
        }
        if let Some(v) = noise {
            if v.shape() != (k, 1) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected noise of shape {k} x 1 (the shape of Y), got {}x{}",
                    v.nrows(),
                    v.ncols()
                )));
            }
        }
        if k == 0 {
            return Ok(vec![self.clone(); ys.len()]);
        }
        let x_tf = self.check_new_inputs(x, mode)?;
        let noise_tf = self.new_noise(noise.map(|v| v.as_slice()), k)?;
        let enlarged = self.enlarged(&x_tf, &noise_tf)?;
        let train_x = vstack(&self.train_x, x);
        let train_x_tf = vstack(&self.train_x_tf, &x_tf);
        let construction_x_tf = self.construction_x_tf.as_ref().map(|c| vstack(c, &x_tf));
        let fixed = matches!(self.likelihood, Likelihood::FixedNoise { .. });
        // Likelihood and noise bookkeeping of the conditioned model.
        let (likelihood, point_noise) = match &self.likelihood {
            Likelihood::FixedNoise { noise: old } => {
                let mut all = old.as_slice().to_vec();
                all.extend_from_slice(&noise_tf);
                (
                    Likelihood::FixedNoise {
                        noise: Vector::from_vec(all),
                    },
                    None,
                )
            }
            Likelihood::Gaussian(_) => {
                let point_noise = match (noise, &self.point_noise) {
                    (None, None) => None,
                    _ => {
                        let mut p = self
                            .point_noise
                            .clone()
                            .unwrap_or_else(|| vec![None; self.num_train()]);
                        p.extend(noise_tf.iter().map(|&v| noise.map(|_| v)));
                        Some(p)
                    }
                };
                (self.likelihood.clone(), point_noise)
            }
        };
        let cache = self.cache()?;
        let c = self.constant();
        ys.iter()
            .map(|y| {
                let y_tf = self.transform_new_targets(y, &x_tf)?;
                let train_yvar = if fixed {
                    let new_yvar = self.untransform_new_noise(&y_tf, &noise_tf, &x_tf)?;
                    self.train_yvar.as_ref().map(|v| vstack(v, &new_yvar))
                } else {
                    None
                };
                let mut train_y_tf = Vector::zeros(self.num_train() + k);
                train_y_tf
                    .rows_mut(0, self.num_train())
                    .copy_from(&self.train_y_tf);
                train_y_tf.rows_mut(self.num_train(), k).copy_from(&y_tf);
                let alpha = enlarged.chol.solve_vec(&train_y_tf.add_scalar(-c));
                let new_cache = PredictionCache {
                    kernel: cache.kernel.clone(),
                    train_scaled: enlarged.full_scaled.clone(),
                    chol: enlarged.chol.clone(),
                    alpha,
                };
                Ok(SingleTaskGP {
                    train_x: train_x.clone(),
                    train_y: vstack(&self.train_y, y),
                    train_yvar,
                    train_x_tf: train_x_tf.clone(),
                    train_y_tf,
                    input_transform: self.input_transform.clone(),
                    input_chain: self.input_chain.clone(),
                    construction_x_tf: construction_x_tf.clone(),
                    outcome_transform: self.outcome_transform.clone(),
                    outcome_chain: self.outcome_chain.clone(),
                    covar_module: self.covar_module.clone(),
                    likelihood: likelihood.clone(),
                    mean_module: self.mean_module,
                    hyperparameters: self.hyperparameters.clone(),
                    point_noise: point_noise.clone(),
                    cache: OnceLock::from(Ok(new_cache)),
                })
            })
            .collect()
    }

    /// The posterior at `x` (`q × d`) with the given observation-noise
    /// variances (`q`, transformed space) added to the diagonal (BoTorch's
    /// `posterior(X, observation_noise=noise)` with a tensor).
    ///
    /// # Errors
    /// See [`SingleTaskGP::posterior`]; [`BotorchError::DimensionMismatch`]
    /// if `noise` does not have one value per point.
    pub fn posterior_with_noise(&self, x: &Matrix, noise: &[f64]) -> Result<Posterior> {
        if noise.len() != x.nrows() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} observation-noise values, got {}",
                x.nrows(),
                noise.len()
            )));
        }
        self.posterior_impl(x, ObservationNoise::Explicit(noise), TransformMode::Eval)
    }

    /// Fantasy models (BoTorch's `model.fantasize(X, sampler,
    /// observation_noise)`): samples `n_f` sets of observations at `x`
    /// (`n' × d`) from the posterior including observation noise, with the
    /// sampler's base samples (`n_f × n'`), and conditions the model on each
    /// of them. `observation_noise` (`n'` values or one, transformed space)
    /// defaults to the inferred noise level, or the mean training noise of a
    /// fixed-noise model.
    ///
    /// # Errors
    /// See [`SingleTaskGP::condition_on_observations`]; errors of the
    /// posterior's covariance root.
    pub fn fantasize(
        &self,
        x: &Matrix,
        sampler: &SobolQMCNormalSampler,
        observation_noise: Option<&[f64]>,
    ) -> Result<Vec<SingleTaskGP>> {
        self.require_one_to_one(TransformMode::Fantasize, "fantasize")?;
        let k = x.nrows();
        let n_f = sampler.num_samples();
        if x.ncols() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {} columns (d), got {}",
                self.dim(),
                x.ncols()
            )));
        }
        if k == 0 {
            return Ok(vec![self.clone(); n_f]);
        }
        let noise: Option<Vec<f64>> = match observation_noise {
            Some(v) if v.len() == 1 => Some(vec![v[0]; k]),
            Some(v) if v.len() == k => Some(v.to_vec()),
            Some(v) => {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected one or {k} observation-noise values, got {}",
                    v.len()
                )))
            }
            None => self.default_fantasy_noise().map(|v| vec![v; k]),
        };
        let mode = TransformMode::Fantasize;
        let post = match &noise {
            Some(v) => self.posterior_impl(x, ObservationNoise::Explicit(v), mode)?,
            None => self.posterior_impl(x, ObservationNoise::Likelihood, mode)?,
        };
        let z = sampler.base_samples(k)?;
        let ys = post.rsample_from_base_samples(&z)?;
        let data = ys.data();
        let sets: Vec<Matrix> = (0..n_f)
            .map(|i| Matrix::from_fn(k, 1, |j, _| data[(j, i)]))
            .collect();
        let noise_m = noise.map(|v| Matrix::from_column_slice(k, 1, &v));
        self.condition_mode(x, &sets, noise_m.as_ref(), mode)
    }
}

/// The posterior of a fantasy model at test points with what its reverse
/// pass needs (see [`FantasyGp::posterior`]).
#[derive(Clone, Debug)]
pub(crate) struct FantasyPosterior {
    /// Index of the fantasy.
    fantasy: usize,
    /// The prepared transformed test points.
    xs: Prepared,
    /// `K(X_full, x)` (`N × p`).
    k_full: Matrix,
    /// `K_full⁻¹ K(X_full, x)` (`N × p`), with the covariance only.
    w: Option<Matrix>,
    /// Posterior mean (original outcome space, `p`).
    pub mean: Vector,
    /// Posterior covariance (original outcome space, `p × p`), if requested.
    pub covariance: Option<Matrix>,
}

/// The adjoints that the reverse passes of a [`FantasyGp`]'s posteriors
/// accumulate for its training data (linear in the adjoints of the
/// posteriors).
#[derive(Clone, Debug)]
pub(crate) struct FantasyAdjoint {
    /// Adjoint of the fantasized observations (original space, `n' × n_f`).
    ybar: Matrix,
    /// Coefficients of `∂k(x̃_new_j, x̃_b)/∂x̃_new_j` from the adjoint of the
    /// training covariance (`n' × N`).
    kbar: Matrix,
    /// Adjoint of the transformed new points from the test covariances
    /// (`n' × d`).
    xbar_tf: Matrix,
}

impl FantasyAdjoint {
    /// `self += scale · other`.
    pub fn add_scaled(&mut self, other: &FantasyAdjoint, scale: f64) {
        self.ybar += &other.ybar * scale;
        self.kbar += &other.kbar * scale;
        self.xbar_tf += &other.xbar_tf * scale;
    }
}

/// The `n_f` fantasy models of a single-output GP at the new points `x_new`
/// (one output of BoTorch's `fantasize`), sharing the updated factor, with
/// reverse-mode derivatives with respect to the test points and the new
/// points (through the fantasized observations and the factor).
#[derive(Debug)]
pub(crate) struct FantasyGp<'a> {
    gp: &'a SingleTaskGP,
    cache: &'a PredictionCache,
    x_new: Matrix,
    full_scaled: Cow<'a, Prepared>,
    chol: Cow<'a, CholeskyFactor>,
    /// Root of the posterior covariance at `x_new` (with observation noise)
    /// and the base samples (`n_f × n'`); `None` without new points.
    root: Option<(CovarianceRoot, Arc<Matrix>)>,
    /// `K_full⁻¹ (y_full - c)` of every fantasy (one shared vector without
    /// new points).
    alphas: Vec<Vector>,
    num_fantasies: usize,
}

impl<'a> FantasyGp<'a> {
    /// Fantasizes `num_fantasies` observations at `x_new` (`n' × d`, raw;
    /// `n' = 0` gives copies of the model) from the posterior with
    /// observation noise and the base samples `base` (`num_fantasies × n'`),
    /// with BoTorch's default fantasy noise.
    ///
    /// # Errors
    /// Shape mismatches and errors of the posterior and the factorization.
    pub fn new(
        gp: &'a SingleTaskGP,
        x_new: &Matrix,
        base: Arc<Matrix>,
        num_fantasies: usize,
    ) -> Result<Self> {
        gp.require_no_perturbation("fantasize")?;
        gp.require_plain_outcome("fantasy models with input derivatives")?;
        gp.require_affine_inputs("fantasy models with input derivatives")?;
        let cache = gp.cache()?;
        let k = x_new.nrows();
        let d = gp.dim();
        if x_new.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {d} columns (d), got {}",
                x_new.ncols()
            )));
        }
        if k == 0 {
            return Ok(Self {
                gp,
                cache,
                x_new: Matrix::zeros(0, d),
                full_scaled: Cow::Borrowed(&cache.train_scaled),
                chol: Cow::Borrowed(&cache.chol),
                root: None,
                alphas: Vec::new(),
                num_fantasies,
            });
        }
        if base.shape() != (num_fantasies, k) {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected base samples of shape {num_fantasies} x {k}, got {}x{}",
                base.nrows(),
                base.ncols()
            )));
        }
        let noise = gp.default_fantasy_noise();
        let post = match noise {
            Some(v) => gp.posterior_with_noise(x_new, &vec![v; k])?,
            None => gp.posterior(x_new, true)?,
        };
        let root = CovarianceRoot::new(&post.covariance)?;
        let x_new_tf = gp.check_new_inputs(x_new, TransformMode::Eval)?;
        let noise_tf = gp.new_noise(noise.map(|v| vec![v; k]).as_deref(), k)?;
        let enlarged = gp.enlarged(&x_new_tf, &noise_tf)?;
        let n = gp.num_train();
        let (mu, sd) = gp.outcome_affine()?;
        let c = gp.constant();
        let r = root.to_matrix();
        let mut r_full = gp.train_y_tf.add_scalar(-c);
        r_full = r_full.resize_vertically(n + k, 0.0);
        let alphas = (0..num_fantasies)
            .map(|i| {
                let z = base.row(i).transpose();
                let y = &post.mean + &r * z;
                let mut rhs = r_full.clone();
                for j in 0..k {
                    rhs[n + j] = (y[j] - mu) / sd - c;
                }
                enlarged.chol.solve_vec(&rhs)
            })
            .collect();
        Ok(Self {
            gp,
            cache,
            x_new: x_new.clone(),
            full_scaled: Cow::Owned(enlarged.full_scaled),
            chol: Cow::Owned(enlarged.chol),
            root: Some((root, base)),
            alphas,
            num_fantasies,
        })
    }

    /// Number of fantasies.
    pub fn num_fantasies(&self) -> usize {
        self.num_fantasies
    }

    /// `K_full⁻¹ (y_full - c)` of fantasy `i`.
    fn alpha(&self, i: usize) -> &Vector {
        if self.alphas.is_empty() {
            &self.cache.alpha
        } else {
            &self.alphas[i]
        }
    }

    /// Number of new points `n'`.
    pub fn num_new(&self) -> usize {
        self.x_new.nrows()
    }

    /// The posterior of fantasy `i` at the raw test points `x` (`p × d`),
    /// in the original outcome space, with its covariance if `with_cov`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns.
    pub fn posterior(&self, i: usize, x: &Matrix, with_cov: bool) -> Result<FantasyPosterior> {
        let xt = self.gp.transform_inputs(x)?;
        let kernel = &self.cache.kernel;
        let xs = kernel.prepare(&xt)?;
        let k_full = kernel.cross(&self.full_scaled, &xs);
        let (mu, sd) = self.gp.outcome_affine()?;
        let c = self.gp.constant();
        let mean = k_full.tr_mul(self.alpha(i)).map(|v| mu + sd * (c + v));
        let (w, covariance) = if with_cov {
            let mut v = k_full.clone();
            self.chol.solve_lower_mut(&mut v);
            let mut cov = kernel.symmetric(&xs);
            cov.gemm_tr(-1.0, &v, &v, 1.0);
            self.chol.solve_lower_transpose_mut(&mut v);
            (Some(v), Some(cov.map(|c| sd * c * sd)))
        } else {
            (None, None)
        };
        Ok(FantasyPosterior {
            fantasy: i,
            xs,
            k_full,
            w,
            mean,
            covariance,
        })
    }

    /// A zero accumulator for the reverse passes.
    pub fn adjoint(&self) -> FantasyAdjoint {
        let k = self.num_new();
        let n_full = self.chol.dim();
        FantasyAdjoint {
            ybar: Matrix::zeros(k, self.num_fantasies()),
            kbar: Matrix::zeros(k, n_full),
            xbar_tf: Matrix::zeros(k, self.gp.dim()),
        }
    }

    /// Reverse pass of [`FantasyGp::posterior`]: for the adjoints of the
    /// mean (`p`) and, if the covariance was computed, of the covariance
    /// (`p × p`, only its symmetric part matters), returns the gradient with
    /// respect to the raw test points (`p × d`) and accumulates the
    /// adjoints of the fantasy model's training data in `acc`.
    pub fn posterior_backward(
        &self,
        post: &FantasyPosterior,
        mean_bar: &Vector,
        cov_bar: Option<&Matrix>,
        acc: &mut FantasyAdjoint,
    ) -> Matrix {
        let kernel = &self.cache.kernel;
        let n = self.gp.num_train();
        let k = self.num_new();
        let n_full = n + k;
        let p = post.mean.len();
        let d = self.gp.dim();
        let sd = self
            .gp
            .outcome_affine()
            .expect("checked when the fantasy models were built")
            .1;
        let alpha = self.alpha(post.fantasy);
        let mbar = mean_bar * sd;
        // Adjoint of K(X_full, x) (N × p) and of K(x, x) (p × p).
        let mut kfull_bar = alpha * mbar.transpose();
        let abar = match (cov_bar, &post.w) {
            (Some(cb), Some(_)) => Some((cb + cb.transpose()) * (0.5 * sd * sd)),
            _ => None,
        };
        // β = K_full⁻¹ K(X_full, x) mbar: the adjoint of the targets.
        let beta = self.chol.solve_vec(&(&post.k_full * &mbar));
        for j in 0..k {
            acc.ybar[(j, post.fantasy)] += beta[n + j] / sd;
            for b in 0..n_full {
                acc.kbar[(j, b)] -= beta[n + j] * alpha[b] + beta[b] * alpha[n + j];
            }
        }
        if let (Some(a), Some(w)) = (&abar, &post.w) {
            kfull_bar.gemm(-2.0, w, a, 1.0);
            if k > 0 {
                // 2 (W A Wᵀ)[n + j, b] for the new rows.
                let wa = w.rows(n, k) * a;
                acc.kbar.gemm(2.0, &wa, &w.transpose(), 1.0);
            }
        }
        // Gradient with respect to the test points (transformed space).
        let jac = self
            .gp
            .input_jacobian()
            .expect("checked when the fantasy models were built");
        let mut out = Matrix::zeros(p, d);
        for t in 0..p {
            let (_, dk) = kernel.cross_with_grad(&post.xs, t, &self.full_scaled);
            let mut g = dk.tr_mul(&kfull_bar.column(t));
            if let Some(a) = &abar {
                let (_, dkk) = kernel.cross_with_grad(&post.xs, t, &post.xs);
                g += dkk.tr_mul(&a.column(t)) * 2.0;
            }
            for m in 0..d {
                out[(t, m)] = g[m] * jac[m];
            }
        }
        // The new points' side of K(X_full, x).
        for j in 0..k {
            let (_, dk) = kernel.cross_with_grad(&self.full_scaled, n + j, &post.xs);
            let g = dk.tr_mul(&kfull_bar.row(n + j).transpose());
            for m in 0..d {
                acc.xbar_tf[(j, m)] += g[m];
            }
        }
        out
    }

    /// Completes the reverse pass: the gradient with respect to the raw new
    /// points (`n' × d`) of everything accumulated in `acc`, through the
    /// training covariance of the fantasy models and the fantasized
    /// observations (the posterior at `x_new` and its covariance root).
    ///
    /// # Errors
    /// Errors of the covariance root's derivative (eigendecomposition
    /// fallback) and of the model's posterior derivative.
    pub fn backward(&self, acc: &FantasyAdjoint) -> Result<Matrix> {
        let k = self.num_new();
        let d = self.gp.dim();
        let Some((root, base)) = &self.root else {
            return Ok(Matrix::zeros(0, d));
        };
        let kernel = &self.cache.kernel;
        let n = self.gp.num_train();
        let mut xbar_tf = acc.xbar_tf.clone();
        for j in 0..k {
            let (_, dk) = kernel.cross_with_grad(&self.full_scaled, n + j, &self.full_scaled);
            let g = dk.tr_mul(&acc.kbar.row(j).transpose());
            for m in 0..d {
                xbar_tf[(j, m)] += g[m];
            }
        }
        let jac = self
            .gp
            .input_jacobian()
            .expect("checked when the fantasy models were built");
        let mut out = Matrix::from_fn(k, d, |j, m| xbar_tf[(j, m)] * jac[m]);
        // Through the fantasized observations y_i = μ + R z_i.
        let mean_bar = Vector::from_fn(k, |j, _| acc.ybar.row(j).sum());
        let root_bar = &acc.ybar * base.as_ref();
        let cov_bar = root.backward(&root_bar)?;
        out += self
            .gp
            .joint_posterior_vjp(&self.x_new, 0..k, &mean_bar, &cov_bar)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SingleTaskGPOptions;
    use crate::transforms::Normalize;

    fn data(n: usize, d: usize, shift: f64) -> (Matrix, Matrix) {
        let x = Matrix::from_fn(n, d, |i, j| {
            ((i * 7 + j * 3) % n) as f64 / n as f64 + 0.05 * j as f64 + shift
        });
        let y = Matrix::from_fn(n, 1, |i, _| {
            let s: f64 = x.row(i).iter().sum();
            (3.0 * s).sin() + 0.5 * s
        });
        (x, y)
    }

    fn model(fixed: bool) -> SingleTaskGP {
        let (x, y) = data(9, 2, 0.0);
        let yvar = fixed.then(|| Matrix::from_fn(9, 1, |i, _| 0.01 + 0.002 * i as f64));
        let mut gp = SingleTaskGP::new(
            x,
            y,
            SingleTaskGPOptions {
                train_yvar: yvar,
                input_transform: Some(Normalize::new(2)),
                ..Default::default()
            },
        )
        .unwrap();
        let mut h = gp.hyperparameters().clone();
        h.raw_lengthscale = Vector::from_vec(vec![0.4, 0.7]);
        h.raw_constant = 0.2;
        if h.raw_noise.is_some() {
            h.raw_noise = Some(0.02);
        }
        gp.set_hyperparameters(h).unwrap();
        gp
    }

    /// The conditioned model equals a model refitted on the enlarged data
    /// with the same hyperparameters and frozen transforms.
    #[test]
    fn conditioning_equals_refitting_with_fixed_hyperparameters() {
        for fixed in [false, true] {
            let gp = model(fixed);
            let xn = Matrix::from_row_slice(3, 2, &[0.15, 0.4, 0.8, 0.95, 0.33, 0.1]);
            let yn = Matrix::from_column_slice(3, 1, &[0.3, -0.2, 1.1]);
            let noise = fixed.then(|| Matrix::from_column_slice(3, 1, &[0.02, 0.05, 0.01]));
            let cond = gp
                .condition_on_observations(&xn, &yn, noise.as_ref())
                .unwrap();
            assert_eq!(cond.num_train(), 12);
            // Reference: a full factorization of the enlarged training data.
            let mut reference = cond.clone();
            reference.cache = OnceLock::new();
            let xt = Matrix::from_row_slice(4, 2, &[0.2, 0.2, 0.5, 0.6, 0.9, 0.1, 0.33, 0.12]);
            let a = cond.posterior(&xt, true).unwrap();
            let b = reference.posterior(&xt, true).unwrap();
            assert!((&a.mean - &b.mean).amax() < 1e-12);
            assert!((&a.covariance - &b.covariance).amax() < 1e-12);
            // The transforms were not refitted.
            assert_eq!(cond.input_transform(), gp.input_transform());
            assert_eq!(cond.outcome_transform(), gp.outcome_transform());
        }
        // Explicit noise of a homoskedastic model enters the posterior.
        let gp = model(false);
        let xn = Matrix::from_row_slice(1, 2, &[0.5, 0.5]);
        let yn = Matrix::from_element(1, 1, 2.0);
        let tight = gp
            .condition_on_observations(&xn, &yn, Some(&Matrix::from_element(1, 1, 1e-8)))
            .unwrap();
        let p = tight.posterior(&xn, false).unwrap();
        assert!((p.mean[0] - 2.0).abs() < 1e-4, "{}", p.mean[0]);
        assert!(model(true)
            .condition_on_observations(&xn, &yn, None)
            .is_err());
    }

    #[test]
    fn fantasy_gradients_match_finite_differences() {
        for fixed in [false, true] {
            let gp = model(fixed);
            let n_f = 3;
            let x_new = Matrix::from_row_slice(2, 2, &[0.3, 0.6, 0.7, 0.2]);
            let base = Arc::new(Matrix::from_fn(n_f, 2, |i, j| {
                0.7 * (i as f64 - 1.0) + 0.3 * j as f64
            }));
            let xt = Matrix::from_row_slice(3, 2, &[0.1, 0.9, 0.55, 0.45, 0.8, 0.7]);
            let mean_bar = Vector::from_vec(vec![0.7, -0.4, 1.3]);
            let cov_bar = Matrix::from_fn(3, 3, |a, b| ((a * 2 + b) % 3) as f64 * 0.3 - 0.2);
            // s(x_new, x_test) = Σ_i ⟨mean_bar, μ_i⟩ + ⟨cov_bar, Σ_i⟩.
            let s = |x_new: &Matrix, xt: &Matrix| -> f64 {
                let f = FantasyGp::new(&gp, x_new, Arc::clone(&base), n_f).unwrap();
                (0..n_f)
                    .map(|i| {
                        let p = f.posterior(i, xt, true).unwrap();
                        p.mean.dot(&mean_bar) + p.covariance.unwrap().component_mul(&cov_bar).sum()
                    })
                    .sum()
            };
            let f = FantasyGp::new(&gp, &x_new, Arc::clone(&base), n_f).unwrap();
            let mut acc = f.adjoint();
            let mut gt = Matrix::zeros(3, 2);
            for i in 0..n_f {
                let p = f.posterior(i, &xt, true).unwrap();
                gt += f.posterior_backward(&p, &mean_bar, Some(&cov_bar), &mut acc);
            }
            let gn = f.backward(&acc).unwrap();
            let h = 1e-6;
            for (pt, grad, is_new) in [(&xt, &gt, false), (&x_new, &gn, true)] {
                for r in 0..pt.nrows() {
                    for c in 0..2 {
                        let mut a = pt.clone();
                        let mut b = pt.clone();
                        a[(r, c)] += h;
                        b[(r, c)] -= h;
                        let fd = if is_new {
                            (s(&a, &xt) - s(&b, &xt)) / (2.0 * h)
                        } else {
                            (s(&x_new, &a) - s(&x_new, &b)) / (2.0 * h)
                        };
                        let an = grad[(r, c)];
                        assert!(
                            (fd - an).abs() < 1e-6 * fd.abs().max(1.0),
                            "fixed={fixed} new={is_new} ({r},{c}): {an} vs {fd}"
                        );
                    }
                }
            }
            // The fantasy posteriors equal those of the conditioned models.
            let sampler =
                SobolQMCNormalSampler::from_base_samples((*base).clone(), Some(0)).unwrap();
            let models = gp.fantasize(&x_new, &sampler, None).unwrap();
            for (i, m) in models.iter().enumerate() {
                let a = m.posterior(&xt, false).unwrap();
                let b = f.posterior(i, &xt, true).unwrap();
                assert!((&a.mean - &b.mean).amax() < 1e-12);
                assert!((&a.covariance - b.covariance.as_ref().unwrap()).amax() < 1e-12);
            }
        }
    }
}
