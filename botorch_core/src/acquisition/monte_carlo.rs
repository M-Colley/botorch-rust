//! Monte-Carlo batch acquisition functions mirroring
//! `botorch.acquisition.monte_carlo` (`SampleReducingMCAcquisitionFunction`
//! and its subclasses), for any `q` and with pending points.
//!
//! For a candidate set `X` (`q × d`) and pending points `X_p` (`m × d`), the
//! joint posterior `N(μ, Σ)` over `[X; X_p]` is sampled by the
//! reparameterization `y_s = μ + L z_s`, where `z_s` are the fixed base
//! samples of a [`SobolQMCNormalSampler`] for `q + m` points and `L` is the
//! covariance root gpytorch uses (`sqrt` for one point, jittered Cholesky
//! otherwise, see [`CovarianceRoot`]). A utility is computed per sample and
//! point, reduced over the `q + m` points (`max`) and averaged over the
//! samples, exactly as in BoTorch.
//!
//! Gradients are computed in reverse mode: the adjoint of the samples is
//! propagated to `μ` and `L`, through the Cholesky factorization
//! ([`crate::linalg::cholesky_backward`]) to `Σ`, and finally to `X` with
//! [`Model::joint_posterior_vjp`].

use std::ops::Range;
use std::sync::{Arc, RwLock};

use crate::acquisition::{check_candidates, check_pending, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::CovarianceRoot;
use crate::models::Model;
use crate::sampling::{SobolQMCNormalSampler, DEFAULT_MC_SAMPLES};
use crate::utils::safe_math::{sigmoid, sigmoid_grad};
use crate::{Matrix, Vector};

/// Options shared by the Monte-Carlo acquisition functions.
#[derive(Clone, Debug, Default)]
pub struct McAcquisitionOptions {
    /// The sampler of base samples (default: a [`SobolQMCNormalSampler`] with
    /// 512 samples, BoTorch's default, and a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points (`m × d`) concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
}

pub(crate) fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// The sampler of the options or a default one.
pub(crate) fn sampler_or_default(
    sampler: Option<Arc<SobolQMCNormalSampler>>,
) -> Result<Arc<SobolQMCNormalSampler>> {
    match sampler {
        Some(s) => Ok(s),
        None => Ok(Arc::new(SobolQMCNormalSampler::new(
            DEFAULT_MC_SAMPLES,
            None,
        )?)),
    }
}

/// Stacks matrices vertically (all with `d` columns).
pub(crate) fn vstack(parts: &[&Matrix], d: usize) -> Matrix {
    let n: usize = parts.iter().map(|m| m.nrows()).sum();
    let mut out = Matrix::zeros(n, d);
    let mut r = 0;
    for m in parts {
        out.rows_mut(r, m.nrows()).copy_from(m);
        r += m.nrows();
    }
    out
}

/// Joint posterior samples of a point set together with what is needed to
/// backpropagate through them.
pub(crate) struct JointSamples {
    /// The full input (`N × d`).
    pub x: Matrix,
    /// Posterior mean (`N`).
    pub mean: Vector,
    /// Root of the posterior covariance.
    pub root: CovarianceRoot,
    /// Base samples (`S × N`).
    pub base: Arc<Matrix>,
}

impl JointSamples {
    /// Computes the posterior over `x` and its root, and fetches the base
    /// samples for `x.nrows()` points.
    pub fn new(model: &dyn Model, x: Matrix, sampler: &SobolQMCNormalSampler) -> Result<Self> {
        let post = model.posterior(&x, false)?;
        let root = CovarianceRoot::new(&post.covariance)?;
        let base = sampler.base_samples(x.nrows())?;
        Ok(Self {
            x,
            mean: post.mean,
            root,
            base,
        })
    }

    /// Samples of the points `rows` (`rows.len() × S`; column `s` is MC
    /// sample `s`): `μ_i + Σ_k R_ik z_sk`.
    pub fn samples(&self, rows: Range<usize>) -> Matrix {
        let r = self.root.rows(rows.clone());
        let mut y = r * self.base.transpose();
        for (i, p) in rows.enumerate() {
            let m = self.mean[p];
            y.row_mut(i).iter_mut().for_each(|v| *v += m);
        }
        y
    }

    /// Propagates the adjoint `ybar` of the samples of the points `cols`
    /// (`cols.len() × S`) to the inputs `grad_rows` (`grad_rows.len() × d`).
    pub fn backward(
        &self,
        model: &dyn Model,
        cols: Range<usize>,
        ybar: &Matrix,
        grad_rows: Range<usize>,
    ) -> Result<Matrix> {
        let n = self.x.nrows();
        let mut mean_bar = Vector::zeros(n);
        let mut root_bar = Matrix::zeros(n, n);
        let rbar = ybar * &*self.base;
        for (i, c) in cols.enumerate() {
            mean_bar[c] = ybar.row(i).sum();
            root_bar.row_mut(c).copy_from(&rbar.row(i));
        }
        let cov_bar = self.root.backward(&root_bar)?;
        model.joint_posterior_vjp(&self.x, grad_rows, &mean_bar, &cov_bar)
    }
}

/// Model, sampler and pending points shared by the sample-reducing MC
/// acquisition functions.
#[derive(Debug)]
pub(crate) struct McCore {
    pub model: Arc<dyn Model>,
    pub sampler: Arc<SobolQMCNormalSampler>,
    pub x_pending: RwLock<Option<Matrix>>,
}

/// Result of a sample-level reduction: the value and, if requested, the
/// adjoint of the samples (`points × S`).
pub(crate) type Reduction = (f64, Option<Matrix>);

impl McCore {
    pub fn new(model: Arc<dyn Model>, options: McAcquisitionOptions) -> Result<Self> {
        if model.num_outputs() != 1 {
            return Err(BotorchError::Unsupported(
                "Must specify an objective or a posterior transform when using a multi-output model."
                    .into(),
            ));
        }
        if let Some(p) = &options.x_pending {
            check_pending(p, model.dim())?;
        }
        Ok(Self {
            sampler: sampler_or_default(options.sampler)?,
            x_pending: RwLock::new(options.x_pending),
            model,
        })
    }

    pub fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.x_pending).clone()
    }

    pub fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }

    /// Evaluates `reduce` on the samples of `[X; X_pending]` (BoTorch's
    /// `concatenate_pending_points` + `SampleReducingMCAcquisitionFunction`).
    pub fn evaluate<F>(
        &self,
        x: &Matrix,
        with_grad: bool,
        reduce: F,
    ) -> Result<(f64, Option<Matrix>)>
    where
        F: Fn(&Matrix, bool) -> Reduction,
    {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let x_full = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let n = x_full.nrows();
        let js = JointSamples::new(self.model.as_ref(), x_full, &self.sampler)?;
        let y = js.samples(0..n);
        let (value, ybar) = reduce(&y, with_grad);
        match ybar {
            Some(ybar) if with_grad => {
                let grad = js.backward(self.model.as_ref(), 0..n, &ybar, 0..q)?;
                Ok((value, Some(grad)))
            }
            _ => Ok((value, None)),
        }
    }
}

/// Per sample (column), the maximum over points and the weights with which
/// `amax` distributes its gradient (evenly between tied maxima).
pub(crate) fn amax_columns(u: &Matrix) -> (Vec<f64>, Matrix) {
    let (n, s) = u.shape();
    let mut maxima = vec![f64::NEG_INFINITY; s];
    let mut w = Matrix::zeros(n, s);
    for c in 0..s {
        let col = u.column(c);
        let m = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        maxima[c] = m;
        let ties = col.iter().filter(|&&v| v == m).count().max(1) as f64;
        for i in 0..n {
            if col[i] == m {
                w[(i, c)] = 1.0 / ties;
            }
        }
    }
    (maxima, w)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// `mean_s max_i u(y)_is` for an elementwise utility `u` with derivative `du`.
fn reduce_max_mean<U, D>(y: &Matrix, with_grad: bool, u: U, du: D) -> Reduction
where
    U: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let uv = y.map(&u);
    let (maxima, w) = amax_columns(&uv);
    let value = mean(&maxima);
    if !with_grad {
        return (value, None);
    }
    let s = y.ncols() as f64;
    let mut g = w;
    for (gi, &yi) in g.iter_mut().zip(y.iter()) {
        if *gi != 0.0 {
            *gi *= du(yi) / s;
        }
    }
    (value, Some(g))
}

macro_rules! mc_common {
    () => {
        fn model(&self) -> &Arc<dyn Model> {
            &self.core.model
        }

        fn x_pending(&self) -> Option<Matrix> {
            self.core.x_pending()
        }

        fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
            self.core.set_x_pending(x_pending)
        }

        fn forward(&self, x: &Matrix) -> Result<f64> {
            Ok(self.core.evaluate(x, false, |y, g| self.reduce(y, g))?.0)
        }

        fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
            let (v, g) = self.core.evaluate(x, true, |y, g| self.reduce(y, g))?;
            Ok((v, g.expect("gradient requested")))
        }
    };
}

/// MC-based batch Expected Improvement,
/// `qEI(X) = E[max(max_j y_j - best_f, 0)]` (BoTorch's
/// `qExpectedImprovement`). BoTorch recommends [`crate::acquisition::QLogExpectedImprovement`].
#[derive(Debug)]
pub struct QExpectedImprovement {
    core: McCore,
    best_f: f64,
}

impl QExpectedImprovement {
    /// Creates `qExpectedImprovement(model, best_f, sampler, X_pending=...)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models,
    /// [`BotorchError::InvalidArgument`] for a NaN `best_f` and
    /// [`BotorchError::DimensionMismatch`] for pending points of the wrong
    /// dimension.
    pub fn new(model: Arc<dyn Model>, best_f: f64, options: McAcquisitionOptions) -> Result<Self> {
        if best_f.is_nan() {
            return Err(BotorchError::InvalidArgument(
                "best_f must not be NaN".into(),
            ));
        }
        Ok(Self {
            core: McCore::new(model, options)?,
            best_f,
        })
    }

    /// The incumbent value `best_f`.
    pub fn best_f(&self) -> f64 {
        self.best_f
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.core.sampler
    }

    fn reduce(&self, y: &Matrix, with_grad: bool) -> Reduction {
        let b = self.best_f;
        reduce_max_mean(
            y,
            with_grad,
            |v| (v - b).max(0.0),
            |v| if v - b >= 0.0 { 1.0 } else { 0.0 },
        )
    }
}

impl AcquisitionFunction for QExpectedImprovement {
    mc_common!();

    fn is_nonnegative(&self) -> bool {
        true
    }
}

/// MC-based batch Probability of Improvement with a sigmoid-smoothed
/// indicator, `qPI(X) = E[max_j σ((y_j - best_f) / tau)]` (BoTorch's
/// `qProbabilityOfImprovement`).
#[derive(Debug)]
pub struct QProbabilityOfImprovement {
    core: McCore,
    best_f: f64,
    tau: f64,
}

impl QProbabilityOfImprovement {
    /// Creates `qProbabilityOfImprovement(model, best_f, sampler, X_pending,
    /// tau)`; BoTorch's default temperature is `tau = 1e-3`.
    ///
    /// # Errors
    /// As [`QExpectedImprovement::new`], and [`BotorchError::InvalidArgument`]
    /// unless `tau > 0`.
    pub fn new(
        model: Arc<dyn Model>,
        best_f: f64,
        tau: f64,
        options: McAcquisitionOptions,
    ) -> Result<Self> {
        if best_f.is_nan() {
            return Err(BotorchError::InvalidArgument(
                "best_f must not be NaN".into(),
            ));
        }
        if tau.is_nan() || tau <= 0.0 {
            return Err(BotorchError::InvalidArgument(format!(
                "tau must be positive, got {tau}"
            )));
        }
        Ok(Self {
            core: McCore::new(model, options)?,
            best_f,
            tau,
        })
    }

    fn reduce(&self, y: &Matrix, with_grad: bool) -> Reduction {
        let (b, t) = (self.best_f, self.tau);
        reduce_max_mean(
            y,
            with_grad,
            |v| sigmoid((v - b) / t),
            |v| sigmoid_grad((v - b) / t) / t,
        )
    }
}

impl AcquisitionFunction for QProbabilityOfImprovement {
    mc_common!();

    fn is_nonnegative(&self) -> bool {
        true
    }
}

/// MC-based batch Simple Regret, `qSR(X) = E[max_j y_j]` (BoTorch's
/// `qSimpleRegret`).
#[derive(Debug)]
pub struct QSimpleRegret {
    core: McCore,
}

impl QSimpleRegret {
    /// Creates `qSimpleRegret(model, sampler, X_pending=...)`.
    ///
    /// # Errors
    /// See [`QExpectedImprovement::new`].
    pub fn new(model: Arc<dyn Model>, options: McAcquisitionOptions) -> Result<Self> {
        Ok(Self {
            core: McCore::new(model, options)?,
        })
    }

    fn reduce(&self, y: &Matrix, with_grad: bool) -> Reduction {
        reduce_max_mean(y, with_grad, |v| v, |_| 1.0)
    }
}

impl AcquisitionFunction for QSimpleRegret {
    mc_common!();
}

/// MC-based batch Upper Confidence Bound (BoTorch's `qUpperConfidenceBound`,
/// Wilson et al. 2017): `qUCB(X) = E[max_j (m_j + β' |y_j - m_j|)]` with
/// `β' = sqrt(β π / 2)` and `m_j` the *sample* mean of `y_j` over the MC
/// samples.
#[derive(Debug)]
pub struct QUpperConfidenceBound {
    core: McCore,
    beta: f64,
    beta_prime: f64,
}

impl QUpperConfidenceBound {
    /// Creates `qUpperConfidenceBound(model, beta, sampler, X_pending=...)`.
    ///
    /// # Errors
    /// As [`QExpectedImprovement::new`], and [`BotorchError::InvalidArgument`]
    /// unless `beta` is finite and non-negative.
    pub fn new(model: Arc<dyn Model>, beta: f64, options: McAcquisitionOptions) -> Result<Self> {
        if !(beta >= 0.0 && beta.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "beta must be finite and non-negative, got {beta}"
            )));
        }
        Ok(Self {
            core: McCore::new(model, options)?,
            beta,
            beta_prime: (beta * std::f64::consts::PI / 2.0).sqrt(),
        })
    }

    /// The trade-off parameter `beta`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    fn reduce(&self, y: &Matrix, with_grad: bool) -> Reduction {
        let (n, s) = y.shape();
        let bp = self.beta_prime;
        let means: Vec<f64> = (0..n).map(|i| y.row(i).sum() / s as f64).collect();
        let u = Matrix::from_fn(n, s, |i, c| means[i] + bp * (y[(i, c)] - means[i]).abs());
        let (maxima, w) = amax_columns(&u);
        let value = mean(&maxima);
        if !with_grad {
            return (value, None);
        }
        let sf = s as f64;
        let mut g = Matrix::zeros(n, s);
        for i in 0..n {
            // Contribution through the sample mean m_i.
            let mut through_mean = 0.0;
            for c in 0..s {
                let a = w[(i, c)] / sf;
                let sgn = sign(y[(i, c)] - means[i]);
                g[(i, c)] = a * bp * sgn;
                through_mean += a * (1.0 - bp * sgn);
            }
            for c in 0..s {
                g[(i, c)] += through_mean / sf;
            }
        }
        (value, Some(g))
    }
}

/// `torch.sign`: 0 at 0.
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl AcquisitionFunction for QUpperConfidenceBound {
    mc_common!();
}
