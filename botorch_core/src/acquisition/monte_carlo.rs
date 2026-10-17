//! Monte-Carlo batch acquisition functions mirroring
//! `botorch.acquisition.monte_carlo` (`SampleReducingMCAcquisitionFunction`
//! and its subclasses), for any `q`, with pending points, MC objectives and
//! outcome constraints.
//!
//! For a candidate set `X` (`q × d`) and pending points `X_p`, the joint
//! posterior `N(μ, Σ)` over the `n` points `[X; X_p]` and the `m` model
//! outputs is sampled by the reparameterization `y_s = μ + L z_s` with the
//! fixed base samples `z_s` of a [`SobolQMCNormalSampler`] (`n·m` per MC
//! sample, BoTorch's `S × n × m` layout: entry `p·m + t` drives output `t` at
//! point `p`) and the covariance root gpytorch uses: one root per output
//! block for independent outputs (`sqrt` for one point, jittered Cholesky
//! otherwise, see [`CovarianceRoot`]). An MC objective maps the samples to
//! one value per point and sample (the samples themselves for single-output
//! models without an objective); a utility is computed per sample and point,
//! optionally weighted by a smoothed feasibility indicator of outcome
//! constraints, reduced over the points (`max`) and averaged over the
//! samples, exactly as in BoTorch.
//!
//! Gradients are computed in reverse mode: the adjoint of the utilities is
//! propagated through the objective and the constraints to the samples, to
//! `μ` and the roots, through the Cholesky factorizations
//! ([`crate::linalg::cholesky_backward`]) to `Σ`, and finally to `X` with
//! [`Model::joint_posterior_vjp`] (plus any direct dependence of the
//! objective on `X`).

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, RwLock};

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};

use crate::acquisition::mc_objective::{McObjective, McSamples, SamplePullback};
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::CovarianceRoot;
use crate::models::Model;
use crate::sampling::{SobolQMCNormalSampler, DEFAULT_MC_SAMPLES};
use crate::utils::objective::{compute_smoothed_feasibility_indicator, OutcomeConstraints};
use crate::utils::safe_math::{sigmoid, sigmoid_grad};
use crate::{Matrix, Vector};

/// BoTorch's error for multi-output models without an objective or a
/// scalarizing posterior transform.
pub(crate) const MULTI_OUTPUT_MESSAGE: &str =
    "Must specify an objective or a posterior transform when using a multi-output model.";

/// Options shared by the Monte-Carlo acquisition functions.
#[derive(Clone, Debug, Default)]
pub struct McAcquisitionOptions {
    /// The sampler of base samples (default: a [`SobolQMCNormalSampler`] with
    /// 512 samples, BoTorch's default, and a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points (`m × d`) concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
    /// The MC objective (BoTorch's `objective`); required for multi-output
    /// models without a posterior transform. `None` uses the samples of the
    /// single output (`IdentityMCObjective`).
    pub objective: Option<Arc<dyn McObjective>>,
    /// Outcome constraints weighting the utilities (BoTorch's `constraints`
    /// and `eta`; qExpectedImprovement and qProbabilityOfImprovement use the
    /// logistic sigmoid, BoTorch's `fat=False`). Not supported by
    /// qSimpleRegret and qUpperConfidenceBound (use a
    /// [`crate::acquisition::ConstrainedMcObjective`] there, as in BoTorch).
    pub constraints: Option<OutcomeConstraints>,
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

/// Checks that `model` can be used by an MC acquisition function with the
/// given objective: unbatched, and single-output unless there is an
/// objective (BoTorch's `MCAcquisitionFunction.__init__`).
pub(crate) fn check_mc_model(model: &dyn Model, objective: Option<&dyn McObjective>) -> Result<()> {
    check_unbatched(model)?;
    if objective.is_none() && model.num_outputs() != 1 {
        return Err(BotorchError::Unsupported(MULTI_OUTPUT_MESSAGE.into()));
    }
    Ok(())
}

/// BoTorch's error for outcome constraints with a one-to-many input transform
/// (`MultiObjectiveMCAcquisitionFunction`; the feasibility of the `q·n_w`
/// perturbed points cannot weight the `q` objective values).
pub(crate) fn check_constraints_one_to_one(model: &dyn Model) -> Result<()> {
    if model.points_per_input() != 1 {
        return Err(BotorchError::Unsupported(
            "Constraints are not supported with input perturbations, due to sample q-batch \
             shape being different than that of the inputs. Use a composite objective that \
             applies feasibility weighting to samples before calculating the risk measure."
                .into(),
        ));
    }
    Ok(())
}

/// Roots of a joint posterior covariance over `n` points and `m` outputs.
#[derive(Clone, Debug)]
pub(crate) enum SampleRoots {
    /// One root per output block (independent outputs: GPyTorch's root
    /// decomposition of a block-diagonal covariance, block by block).
    Blocks(Vec<CovarianceRoot>),
    /// One root of the full output-major covariance.
    Full(CovarianceRoot),
    /// The zero root of a deterministic model's posterior (see
    /// [`Model::is_deterministic`]): every sample is the mean, and only the
    /// mean receives adjoints.
    Zero,
}

impl SampleRoots {
    /// The roots of `cov` (`n·m × n·m`, output-major).
    pub fn new(cov: &Matrix, n: usize, m: usize) -> Result<Self> {
        if m == 1 {
            return Ok(Self::Blocks(vec![CovarianceRoot::new(cov)?]));
        }
        let independent = (0..m).all(|t| {
            (0..m)
                .filter(|&u| u != t)
                .all(|u| cov.view((t * n, u * n), (n, n)).iter().all(|&v| v == 0.0))
        });
        if independent {
            let roots = (0..m)
                .map(|t| CovarianceRoot::new(&cov.view((t * n, t * n), (n, n)).into_owned()))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self::Blocks(roots))
        } else {
            Ok(Self::Full(CovarianceRoot::new(cov)?))
        }
    }

    /// The jitter of every root (for cache keys).
    pub fn jitter_bits(&self) -> Option<Vec<u64>> {
        let bits = |r: &CovarianceRoot| match r {
            CovarianceRoot::Eigen(_) => None,
            r => Some(r.jitter().to_bits()),
        };
        match self {
            Self::Blocks(rs) => rs.iter().map(bits).collect(),
            Self::Full(r) => bits(r).map(|b| vec![b]),
            Self::Zero => Some(Vec::new()),
        }
    }
}

impl crate::posterior::Posterior {
    /// Posterior samples `μ + R z` from explicit base samples `z` (`S × q·m`,
    /// BoTorch's `S × q × m` layout: column `p·m + t` drives output `t` at
    /// point `p`), with GPyTorch's covariance roots (one per output block for
    /// independent outputs): BoTorch's `posterior.rsample_from_base_samples`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `base_samples` does not have
    /// `q·m` columns, and errors of the covariance root (NaN entries).
    pub fn rsample_from_base_samples(&self, base_samples: &Matrix) -> Result<McSamples> {
        let (q, m) = (self.num_points(), self.num_outputs());
        if base_samples.ncols() != q * m {
            return Err(BotorchError::DimensionMismatch(format!(
                "base samples must have {} columns ({q} points x {m} outputs), got {}",
                q * m,
                base_samples.ncols()
            )));
        }
        let js = JointSamples {
            x: Matrix::zeros(q, 0),
            n: q,
            m,
            mean: self.mean.clone(),
            roots: SampleRoots::new(&self.covariance, q, m)?,
            base: Arc::new(base_samples.clone()),
        };
        Ok(js.samples(0..q))
    }
}

/// Joint posterior samples of a point set together with what is needed to
/// backpropagate through them.
///
/// With a one-to-many input transform (see [`Model::points_per_input`]) the
/// posterior of the `x.nrows()` inputs is over `n = x.nrows()·n_w` points:
/// the samples, base samples and ranges of points refer to those `n` points,
/// while the rows of gradients refer to the inputs.
pub(crate) struct JointSamples {
    /// The full input (`n_in × d`).
    pub x: Matrix,
    /// Number of posterior points `n` (`n_in·n_w`).
    pub n: usize,
    /// Number of outputs `m`.
    pub m: usize,
    /// Posterior mean (`n·m`, output-major).
    pub mean: Vector,
    /// Roots of the posterior covariance.
    pub roots: SampleRoots,
    /// Base samples (`S × n·m`, column `p·m + t` for output `t` at point `p`).
    pub base: Arc<Matrix>,
}

impl JointSamples {
    /// Computes the posterior over `x` and its roots, and fetches the base
    /// samples for its `x.nrows()·n_w` points and all outputs.
    pub fn new(model: &dyn Model, x: Matrix, sampler: &SobolQMCNormalSampler) -> Result<Self> {
        let m = model.num_outputs();
        let base = sampler.base_samples(x.nrows() * model.points_per_input() * m)?;
        Self::with_base_samples(model, x, base)
    }

    /// Like [`JointSamples::new`] with explicit base samples (`S × n·m`).
    pub fn with_base_samples(model: &dyn Model, x: Matrix, base: Arc<Matrix>) -> Result<Self> {
        let post = model.posterior(&x, false)?;
        let n = post.num_points();
        let m = post.num_outputs();
        if base.ncols() != n * m {
            return Err(BotorchError::DimensionMismatch(format!(
                "base samples must have {} columns ({n} points x {m} outputs), got {}",
                n * m,
                base.ncols()
            )));
        }
        let roots = if model.is_deterministic() {
            SampleRoots::Zero
        } else {
            SampleRoots::new(&post.covariance, n, m)?
        };
        Ok(Self {
            x,
            n,
            m,
            mean: post.mean,
            roots,
            base,
        })
    }

    /// Number of posterior points `n`.
    pub fn num_points(&self) -> usize {
        self.n
    }

    /// Number of posterior points per input (`n_w`).
    pub fn points_per_input(&self) -> usize {
        self.n / self.x.nrows().max(1)
    }

    /// Number of MC samples `S`.
    pub fn num_samples(&self) -> usize {
        self.base.nrows()
    }

    /// The base samples of output `t` (`S × n`).
    fn base_of_output(&self, t: usize) -> Cow<'_, Matrix> {
        if self.m == 1 {
            return Cow::Borrowed(self.base.as_ref());
        }
        let (n, m) = (self.num_points(), self.m);
        Cow::Owned(Matrix::from_fn(self.base.nrows(), n, |s, j| {
            self.base[(s, j * m + t)]
        }))
    }

    /// The base samples in output-major order (`S × n·m`, column `t·n + p`).
    fn base_output_major(&self) -> Matrix {
        let (n, m) = (self.num_points(), self.m);
        Matrix::from_fn(self.base.nrows(), n * m, |s, k| {
            self.base[(s, (k % n) * m + k / n)]
        })
    }

    /// Samples of the points `rows` and all outputs: `μ + R z` (see
    /// [`McSamples`] for the layout).
    pub fn samples(&self, rows: Range<usize>) -> McSamples {
        let (n, m) = (self.num_points(), self.m);
        let k = rows.len();
        let s = self.num_samples();
        let mut data = Matrix::zeros(m * k, s);
        match &self.roots {
            SampleRoots::Blocks(roots) => {
                for (t, root) in roots.iter().enumerate() {
                    let r = root.rows(rows.clone());
                    let y = r * self.base_of_output(t).transpose();
                    for i in 0..k {
                        let mu = self.mean[t * n + rows.start + i];
                        let mut dst = data.row_mut(t * k + i);
                        for c in 0..s {
                            dst[c] = y[(i, c)] + mu;
                        }
                    }
                }
            }
            SampleRoots::Full(root) => {
                let z = self.base_output_major();
                let full = root.to_matrix();
                for t in 0..m {
                    for i in 0..k {
                        let row = t * n + rows.start + i;
                        let y = full.row(row) * z.transpose();
                        let mu = self.mean[row];
                        let mut dst = data.row_mut(t * k + i);
                        for c in 0..s {
                            dst[c] = y[c] + mu;
                        }
                    }
                }
            }
            SampleRoots::Zero => {
                for t in 0..m {
                    for i in 0..k {
                        let mu = self.mean[t * n + rows.start + i];
                        data.row_mut(t * k + i).fill(mu);
                    }
                }
            }
        }
        McSamples::new(data, k, m).expect("consistent sample shapes")
    }

    /// Joint samples of a given posterior (over the `n` points of `x`) with
    /// explicit base samples (`S × n·m`); `deterministic` uses the zero root
    /// (see [`Model::is_deterministic`]).
    pub fn from_posterior(
        x: Matrix,
        posterior: crate::posterior::Posterior,
        base: Arc<Matrix>,
        deterministic: bool,
    ) -> Result<Self> {
        let n = posterior.num_points();
        let m = posterior.num_outputs();
        if base.ncols() != n * m {
            return Err(BotorchError::DimensionMismatch(format!(
                "base samples must have {} columns ({n} points x {m} outputs), got {}",
                n * m,
                base.ncols()
            )));
        }
        let roots = if deterministic {
            SampleRoots::Zero
        } else {
            SampleRoots::new(&posterior.covariance, n, m)?
        };
        Ok(Self {
            x,
            n,
            m,
            mean: posterior.mean,
            roots,
            base,
        })
    }

    /// Propagates the adjoint `ybar` of the samples of the points `cols` (in
    /// the layout of [`JointSamples::samples`]) to the inputs `grad_rows`
    /// (rows of `x`; `grad_rows.len() × d`).
    pub fn backward(
        &self,
        model: &dyn Model,
        cols: Range<usize>,
        ybar: &Matrix,
        grad_rows: Range<usize>,
    ) -> Result<Matrix> {
        let (mean_bar, cov_bar) = self.posterior_adjoints(cols, ybar)?;
        model.joint_posterior_vjp(&self.x, grad_rows, &mean_bar, &cov_bar)
    }

    /// Propagates the adjoint `ybar` of the samples of the points `cols` to
    /// the adjoints of the posterior mean (`n·m`) and covariance (`n·m ×
    /// n·m`, output-major).
    pub fn posterior_adjoints(
        &self,
        cols: Range<usize>,
        ybar: &Matrix,
    ) -> Result<(Vector, Matrix)> {
        let (n, m) = (self.num_points(), self.m);
        let k = cols.len();
        let nm = n * m;
        let mut mean_bar = Vector::zeros(nm);
        for t in 0..m {
            for i in 0..k {
                mean_bar[t * n + cols.start + i] = ybar.row(t * k + i).sum();
            }
        }
        let mut cov_bar = Matrix::zeros(nm, nm);
        match &self.roots {
            SampleRoots::Blocks(roots) => {
                for (t, root) in roots.iter().enumerate() {
                    let yb = ybar.rows(t * k, k);
                    let rbar = yb * self.base_of_output(t).as_ref();
                    let mut root_bar = Matrix::zeros(n, n);
                    for i in 0..k {
                        root_bar.row_mut(cols.start + i).copy_from(&rbar.row(i));
                    }
                    let cb = root.backward(&root_bar)?;
                    cov_bar.view_mut((t * n, t * n), (n, n)).copy_from(&cb);
                }
            }
            SampleRoots::Full(root) => {
                let z = self.base_output_major();
                let mut root_bar = Matrix::zeros(nm, nm);
                for t in 0..m {
                    for i in 0..k {
                        let r = ybar.row(t * k + i) * &z;
                        root_bar.row_mut(t * n + cols.start + i).copy_from(&r);
                    }
                }
                cov_bar = root.backward(&root_bar)?;
            }
            SampleRoots::Zero => {}
        }
        Ok((mean_bar, cov_bar))
    }
}

/// Posterior samples of a point set and their MC objective values, with what
/// is needed to backpropagate (steps shared by all MC acquisition functions).
pub(crate) struct SampledObjective {
    pub js: JointSamples,
    /// Samples of all points.
    pub samples: McSamples,
    /// Objective values of all points (`n × S`).
    pub obj: Matrix,
    pullback: Option<SamplePullback>,
}

impl SampledObjective {
    /// Samples the posterior over `x` and evaluates the objective (`x` is
    /// passed to the objective as BoTorch's `X`).
    pub fn new(
        model: &dyn Model,
        x: Matrix,
        sampler: &SobolQMCNormalSampler,
        objective: Option<&dyn McObjective>,
        with_grad: bool,
    ) -> Result<Self> {
        let js = JointSamples::new(model, x, sampler)?;
        Self::from_joint(js, objective, with_grad)
    }

    /// Evaluates the objective on joint samples. The objective maps the
    /// samples of the `n` posterior points to one value per input (BoTorch's
    /// q-batch check of `MCAcquisitionObjective.__call__`; with a one-to-many
    /// input transform this needs an objective such as a risk measure).
    pub fn from_joint(
        js: JointSamples,
        objective: Option<&dyn McObjective>,
        with_grad: bool,
    ) -> Result<Self> {
        let n = js.num_points();
        let n_in = js.x.nrows();
        let samples = js.samples(0..n);
        let (obj, pullback) = match objective {
            Some(o) => {
                let v = o.evaluate(&samples, Some(&js.x), with_grad)?;
                if v.values.shape() != (n_in, samples.num_samples()) {
                    return Err(crate::acquisition::mc_objective::q_batch_mismatch(
                        v.values.nrows(),
                        n_in,
                    ));
                }
                (v.values, v.pullback)
            }
            None => {
                if samples.num_outputs() != 1 {
                    return Err(BotorchError::Unsupported(MULTI_OUTPUT_MESSAGE.into()));
                }
                if n != n_in {
                    return Err(crate::acquisition::mc_objective::q_batch_mismatch(n, n_in));
                }
                (samples.data().clone(), None)
            }
        };
        Ok(Self {
            js,
            samples,
            obj,
            pullback,
        })
    }

    /// Propagates the adjoint of the objective values (`n_in × S`) and an
    /// additional adjoint of the samples to the adjoints of the posterior
    /// mean and covariance and the direct adjoint of the inputs (`n_in ×
    /// d`, if the objective depends on them).
    pub fn posterior_adjoints(
        self,
        obj_bar: &Matrix,
        samples_bar: Option<Matrix>,
    ) -> Result<(Vector, Matrix, Option<Matrix>)> {
        let n = self.js.num_points();
        let (mut sbar, xbar) = match self.pullback {
            Some(pb) => pb(obj_bar)?,
            None => (obj_bar.clone(), None),
        };
        if let Some(extra) = samples_bar {
            sbar += extra;
        }
        let (mean_bar, cov_bar) = self.js.posterior_adjoints(0..n, &sbar)?;
        Ok((mean_bar, cov_bar, xbar))
    }

    /// Propagates the adjoint of the objective values (`n_in × S`) and an
    /// additional adjoint of the samples (e.g. from constraints) to the
    /// inputs `grad_rows`.
    pub fn backward(
        self,
        model: &dyn Model,
        obj_bar: &Matrix,
        samples_bar: Option<Matrix>,
        grad_rows: Range<usize>,
    ) -> Result<Matrix> {
        let n = self.js.num_points();
        let n_in = self.js.x.nrows();
        let (mut sbar, xbar) = match self.pullback {
            Some(pb) => pb(obj_bar)?,
            None => (obj_bar.clone(), None),
        };
        if let Some(extra) = samples_bar {
            sbar += extra;
        }
        let mut grad = self.js.backward(model, 0..n, &sbar, grad_rows.clone())?;
        if let Some(xb) = xbar {
            if xb.shape() != (n_in, self.js.x.ncols()) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the objective's input adjoint has shape {:?}, expected {n_in} x {}",
                    xb.shape(),
                    self.js.x.ncols()
                )));
            }
            grad += xb.rows(grad_rows.start, grad_rows.len());
        }
        Ok(grad)
    }
}

/// Result of a sample-level reduction: the value and, if requested, the
/// adjoints of the objective values and of the feasibility indicator
/// (`points × S` each).
pub(crate) struct Reduced {
    pub value: f64,
    pub obj_bar: Option<Matrix>,
    pub ind_bar: Option<Matrix>,
}

/// Model, sampler, pending points, objective and constraints shared by the
/// sample-reducing MC acquisition functions.
#[derive(Debug)]
pub(crate) struct McCore {
    pub model: Arc<dyn Model>,
    pub sampler: Arc<SobolQMCNormalSampler>,
    pub x_pending: RwLock<Option<Matrix>>,
    pub objective: Option<Arc<dyn McObjective>>,
    pub constraints: Option<OutcomeConstraints>,
    /// Smoothing of the feasibility indicator per constraint (BoTorch's
    /// `fat`).
    pub fat: Vec<Option<bool>>,
}

impl McCore {
    /// The core of an acquisition function whose constraints use the
    /// logistic sigmoid (BoTorch's default `fat=False`).
    pub fn new(model: Arc<dyn Model>, options: McAcquisitionOptions) -> Result<Self> {
        let k = options.constraints.as_ref().map_or(0, |c| c.len());
        Self::with_fat(model, options, vec![Some(false); k])
    }

    /// The core with one smoothing kind per constraint.
    pub fn with_fat(
        model: Arc<dyn Model>,
        options: McAcquisitionOptions,
        fat: Vec<Option<bool>>,
    ) -> Result<Self> {
        check_mc_model(model.as_ref(), options.objective.as_deref())?;
        if let Some(c) = &options.constraints {
            check_constraints_one_to_one(model.as_ref())?;
            crate::utils::objective::check_fat(c, &fat)?;
        }
        if let (Some(_), Some(o)) = (&options.constraints, &options.objective) {
            if o.is_constrained() {
                return Err(BotorchError::InvalidArgument(
                    "ConstrainedMCObjective as well as constraints passed to constructor.Choose \
                     one or the other, preferably the latter."
                        .into(),
                ));
            }
        }
        if let Some(p) = &options.x_pending {
            check_pending(p, model.dim())?;
        }
        Ok(Self {
            sampler: sampler_or_default(options.sampler)?,
            x_pending: RwLock::new(options.x_pending),
            model,
            objective: options.objective,
            constraints: options.constraints,
            fat,
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

    /// Rejects outcome constraints (for acquisition functions that do not take
    /// them in BoTorch).
    pub fn reject_constraints(&self, name: &str) -> Result<()> {
        if self.constraints.is_some() {
            return Err(BotorchError::Unsupported(format!(
                "{name} does not support outcome constraints; use a ConstrainedMcObjective."
            )));
        }
        Ok(())
    }

    /// Evaluates `reduce` on the objective values of the samples of
    /// `[X; X_pending]` (BoTorch's `concatenate_pending_points` +
    /// `SampleReducingMCAcquisitionFunction.forward`). `reduce` receives the
    /// objective values and, with constraints, the smoothed feasibility
    /// indicator (its logarithm if `log_indicator`).
    pub fn evaluate<F>(
        &self,
        x: &Matrix,
        with_grad: bool,
        log_indicator: bool,
        reduce: F,
    ) -> Result<(f64, Option<Matrix>)>
    where
        F: Fn(&Matrix, Option<&Matrix>, bool) -> Reduced,
    {
        let (v, g) = self.evaluate_weighted(x, None, with_grad, log_indicator, reduce)?;
        Ok((v, g.map(|(g, _)| g)))
    }

    /// [`McCore::evaluate`] with the utilities of every point (row of
    /// `[X; X_pending]`) multiplied by a weight (added to its log utilities
    /// if `log_indicator`), as BoTorch's `PriorGuidedAcquisitionFunction`
    /// weights the non-reduced utilities; returns the gradients with respect
    /// to `X` and to the weights.
    pub fn evaluate_weighted<F>(
        &self,
        x: &Matrix,
        weights: Option<&[f64]>,
        with_grad: bool,
        log_indicator: bool,
        reduce: F,
    ) -> Result<crate::acquisition::WeightedEvaluation>
    where
        F: Fn(&Matrix, Option<&Matrix>, bool) -> Reduced,
    {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let x_full = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let n_full = x_full.nrows();
        if let Some(w) = weights {
            if w.len() != n_full {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {n_full} point weights, got {}",
                    w.len()
                )));
            }
        }
        let so = SampledObjective::new(
            self.model.as_ref(),
            x_full,
            &self.sampler,
            self.objective.as_deref(),
            with_grad,
        )?;
        let ind = match &self.constraints {
            Some(c) => Some(compute_smoothed_feasibility_indicator(
                c,
                &self.fat,
                &so.samples,
                log_indicator,
                with_grad,
            )?),
            None => None,
        };
        let s = so.obj.ncols();
        let combined = weighted_indicator(
            ind.as_ref().map(|i| &i.values),
            weights,
            log_indicator,
            n_full,
            s,
        );
        let r = reduce(&so.obj, combined.as_ref(), with_grad);
        if !with_grad {
            return Ok((r.value, None));
        }
        let obj_bar = r.obj_bar.expect("gradient requested");
        let (ind_bar, w_bar) = split_indicator_adjoint(
            ind.as_ref().map(|i| &i.values),
            weights,
            log_indicator,
            r.ind_bar,
        );
        let samples_bar = match (ind, ind_bar) {
            (Some(ind), Some(ib)) => Some(ind.backward(&ib)?),
            _ => None,
        };
        let grad = so.backward(self.model.as_ref(), &obj_bar, samples_bar, 0..q)?;
        Ok((r.value, Some((grad, w_bar))))
    }
}

/// The indicator of the points combined with point weights (multiplied, or
/// added for log indicators), `n × S`; `None` without both.
pub(crate) fn weighted_indicator(
    ind: Option<&Matrix>,
    weights: Option<&[f64]>,
    log: bool,
    n: usize,
    s: usize,
) -> Option<Matrix> {
    match (weights, ind) {
        (None, Some(i)) => Some(i.clone()),
        (None, None) => None,
        (Some(w), Some(i)) => Some(Matrix::from_fn(n, s, |r, c| {
            if log {
                i[(r, c)] + w[r]
            } else {
                i[(r, c)] * w[r]
            }
        })),
        (Some(w), None) => Some(Matrix::from_fn(n, s, |r, _| w[r])),
    }
}

/// Splits the adjoint of a [`weighted_indicator`] into the adjoint of the
/// indicator and of the weights.
pub(crate) fn split_indicator_adjoint(
    ind: Option<&Matrix>,
    weights: Option<&[f64]>,
    log: bool,
    ind_bar: Option<Matrix>,
) -> (Option<Matrix>, Vec<f64>) {
    let Some(w) = weights else {
        return (ind_bar, Vec::new());
    };
    let Some(ib) = ind_bar else {
        return (None, vec![0.0; w.len()]);
    };
    let (n, s) = ib.shape();
    let mut w_bar = vec![0.0; n];
    for (row, wb) in w_bar.iter_mut().enumerate() {
        for c in 0..s {
            *wb += if log {
                ib[(row, c)]
            } else {
                ib[(row, c)] * ind.map_or(1.0, |i| i[(row, c)])
            };
        }
    }
    let ind_adj = ind.map(|_| {
        if log {
            ib.clone()
        } else {
            Matrix::from_fn(n, s, |row, c| ib[(row, c)] * w[row])
        }
    });
    (ind_adj, w_bar)
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

/// `mean_s max_i u(obj)_is · ind_is` for an elementwise utility `u` with
/// derivative `du` and an optional (non-log) feasibility indicator.
fn reduce_max_mean<U, D>(
    obj: &Matrix,
    ind: Option<&Matrix>,
    with_grad: bool,
    u: U,
    du: D,
) -> Reduced
where
    U: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let uv = obj.map(&u);
    let weighted = ind.map(|w| uv.component_mul(w));
    let (maxima, w) = amax_columns(weighted.as_ref().unwrap_or(&uv));
    let value = mean(&maxima);
    if !with_grad {
        return Reduced {
            value,
            obj_bar: None,
            ind_bar: None,
        };
    }
    let s = obj.ncols() as f64;
    match ind {
        None => {
            let mut g = w;
            for (gi, &yi) in g.iter_mut().zip(obj.iter()) {
                if *gi != 0.0 {
                    *gi *= du(yi) / s;
                }
            }
            Reduced {
                value,
                obj_bar: Some(g),
                ind_bar: None,
            }
        }
        Some(ind) => {
            let a_bar = w / s;
            let ind_bar = a_bar.component_mul(&uv);
            let mut g = a_bar.component_mul(ind);
            for (gi, &yi) in g.iter_mut().zip(obj.iter()) {
                if *gi != 0.0 {
                    *gi *= du(yi);
                }
            }
            Reduced {
                value,
                obj_bar: Some(g),
                ind_bar: Some(ind_bar),
            }
        }
    }
}

/// `mean_s max_i (α m_i + β |y_is - m_i|) · ind_is` with the sample means
/// `m_i = mean_s y_is` (qUpperConfidenceBound: `α = 1`, `β = β'`;
/// qLowerConfidenceBound: `β = -β'`; qPosteriorStandardDeviation: `α = 0`,
/// `β = sqrt(π/2)`), optionally weighted by a feasibility indicator.
pub(crate) fn reduce_centered(
    y: &Matrix,
    ind: Option<&Matrix>,
    alpha: f64,
    beta: f64,
    with_grad: bool,
) -> Reduced {
    let (n, s) = y.shape();
    let means: Vec<f64> = (0..n).map(|i| y.row(i).sum() / s as f64).collect();
    let u = Matrix::from_fn(n, s, |i, c| {
        alpha * means[i] + beta * (y[(i, c)] - means[i]).abs()
    });
    let weighted = ind.map(|w| u.component_mul(w));
    let (maxima, w) = amax_columns(weighted.as_ref().unwrap_or(&u));
    let value = mean(&maxima);
    if !with_grad {
        return Reduced {
            value,
            obj_bar: None,
            ind_bar: None,
        };
    }
    let sf = s as f64;
    // Adjoint of the utilities.
    let a_bar = w / sf;
    let (u_bar, ind_bar) = match ind {
        Some(ind) => (a_bar.component_mul(ind), Some(a_bar.component_mul(&u))),
        None => (a_bar, None),
    };
    let mut g = Matrix::zeros(n, s);
    for i in 0..n {
        // Contribution through the sample mean m_i.
        let mut through_mean = 0.0;
        for c in 0..s {
            let a = u_bar[(i, c)];
            let sgn = sign(y[(i, c)] - means[i]);
            g[(i, c)] = a * beta * sgn;
            through_mean += a * (alpha - beta * sgn);
        }
        for c in 0..s {
            g[(i, c)] += through_mean / sf;
        }
    }
    Reduced {
        value,
        obj_bar: Some(g),
        ind_bar,
    }
}

/// The per-sample utility of a sample-reducing Monte-Carlo acquisition
/// function (BoTorch's `_sample_forward`), reduced by the maximum over the
/// points and the mean over the samples, optionally weighted by a smoothed
/// feasibility indicator (BoTorch's `SampleReducingMCAcquisitionFunction`).
/// Used by the value functions of the knowledge gradient
/// ([`crate::acquisition::knowledge_gradient`]).
#[derive(Clone, Debug, PartialEq)]
pub enum SampleUtility {
    /// `max(obj - best_f, 0)` (qExpectedImprovement).
    ExpectedImprovement {
        /// The incumbent.
        best_f: f64,
    },
    /// `sigmoid((obj - best_f) / tau)` (qProbabilityOfImprovement).
    ProbabilityOfImprovement {
        /// The incumbent.
        best_f: f64,
        /// The temperature.
        tau: f64,
    },
    /// `obj` (qSimpleRegret).
    SimpleRegret,
    /// `m + beta_prime |obj - m|` with the sample mean `m`
    /// (qUpperConfidenceBound with `beta_prime = sqrt(beta π / 2)`,
    /// qLowerConfidenceBound with `-sqrt(beta π / 2)`).
    ConfidenceBound {
        /// `±sqrt(beta π / 2)`.
        beta_prime: f64,
    },
    /// `sqrt(π / 2) |obj - m|` (qPosteriorStandardDeviation).
    PosteriorStandardDeviation,
}

impl SampleUtility {
    /// The reduction of the objective values `y` (`n × S`) with an optional
    /// (non-log) feasibility indicator.
    pub(crate) fn reduce(&self, y: &Matrix, ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        match *self {
            Self::ExpectedImprovement { best_f: b } => reduce_max_mean(
                y,
                ind,
                with_grad,
                |v| (v - b).max(0.0),
                |v| if v - b >= 0.0 { 1.0 } else { 0.0 },
            ),
            Self::ProbabilityOfImprovement { best_f: b, tau: t } => reduce_max_mean(
                y,
                ind,
                with_grad,
                |v| sigmoid((v - b) / t),
                |v| sigmoid_grad((v - b) / t) / t,
            ),
            Self::SimpleRegret => reduce_max_mean(y, ind, with_grad, |v| v, |_| 1.0),
            Self::ConfidenceBound { beta_prime } => {
                reduce_centered(y, ind, 1.0, beta_prime, with_grad)
            }
            Self::PosteriorStandardDeviation => {
                reduce_centered(y, ind, 0.0, (std::f64::consts::PI / 2.0).sqrt(), with_grad)
            }
        }
    }

    /// Whether the utilities are known to be non-negative.
    pub fn is_nonnegative(&self) -> bool {
        matches!(
            self,
            Self::ExpectedImprovement { .. }
                | Self::ProbabilityOfImprovement { .. }
                | Self::PosteriorStandardDeviation
        )
    }
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
            Ok(self
                .core
                .evaluate(x, false, false, |y, w, g| self.reduce(y, w, g))?
                .0)
        }

        fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
            let (v, g) = self
                .core
                .evaluate(x, true, false, |y, w, g| self.reduce(y, w, g))?;
            Ok((v, g.expect("gradient requested")))
        }

        fn as_sample_reducing(&self) -> Option<&dyn crate::acquisition::SampleReducing> {
            Some(self)
        }
    };
}

/// [`crate::acquisition::SampleReducing`] of the sample-reducing MC
/// acquisition functions with non-log utilities.
macro_rules! sample_reducing {
    ($t:ty) => {
        impl crate::acquisition::SampleReducing for $t {
            fn evaluate_weighted(
                &self,
                x: &Matrix,
                weights: &[f64],
                with_grad: bool,
            ) -> Result<crate::acquisition::WeightedEvaluation> {
                self.core
                    .evaluate_weighted(x, Some(weights), with_grad, false, |y, w, g| {
                        self.reduce(y, w, g)
                    })
            }

            fn log_utilities(&self) -> bool {
                false
            }
        }
    };
}

/// MC-based batch Expected Improvement,
/// `qEI(X) = E[max_j (max(obj_j - best_f, 0) · ind_j)]` (BoTorch's
/// `qExpectedImprovement`, with the smoothed feasibility indicator `ind` of
/// outcome constraints). BoTorch recommends [`crate::acquisition::QLogExpectedImprovement`].
#[derive(Debug)]
pub struct QExpectedImprovement {
    core: McCore,
    best_f: f64,
}

impl QExpectedImprovement {
    /// Creates `qExpectedImprovement(model, best_f, sampler, objective,
    /// X_pending=..., constraints, eta)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models without an
    /// objective, [`BotorchError::InvalidArgument`] for a NaN `best_f` or
    /// constraints together with a [`crate::acquisition::ConstrainedMcObjective`],
    /// and [`BotorchError::DimensionMismatch`] for pending points of the wrong
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

    fn reduce(&self, y: &Matrix, ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        let b = self.best_f;
        reduce_max_mean(
            y,
            ind,
            with_grad,
            |v| (v - b).max(0.0),
            |v| if v - b >= 0.0 { 1.0 } else { 0.0 },
        )
    }
}

sample_reducing!(QExpectedImprovement);

impl AcquisitionFunction for QExpectedImprovement {
    mc_common!();

    fn is_nonnegative(&self) -> bool {
        true
    }
}

/// MC-based batch Probability of Improvement with a sigmoid-smoothed
/// indicator, `qPI(X) = E[max_j σ((obj_j - best_f) / tau) · ind_j]` (BoTorch's
/// `qProbabilityOfImprovement`).
#[derive(Debug)]
pub struct QProbabilityOfImprovement {
    core: McCore,
    best_f: f64,
    tau: f64,
}

impl QProbabilityOfImprovement {
    /// Creates `qProbabilityOfImprovement(model, best_f, sampler, objective,
    /// X_pending, tau, constraints, eta)`; BoTorch's default temperature is
    /// `tau = 1e-3`.
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

    fn reduce(&self, y: &Matrix, ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        let (b, t) = (self.best_f, self.tau);
        reduce_max_mean(
            y,
            ind,
            with_grad,
            |v| sigmoid((v - b) / t),
            |v| sigmoid_grad((v - b) / t) / t,
        )
    }
}

sample_reducing!(QProbabilityOfImprovement);

impl AcquisitionFunction for QProbabilityOfImprovement {
    mc_common!();

    fn is_nonnegative(&self) -> bool {
        true
    }
}

/// MC-based batch Simple Regret, `qSR(X) = E[max_j obj_j]` (BoTorch's
/// `qSimpleRegret`).
#[derive(Debug)]
pub struct QSimpleRegret {
    core: McCore,
}

impl QSimpleRegret {
    /// Creates `qSimpleRegret(model, sampler, objective, X_pending=...)`.
    ///
    /// # Errors
    /// See [`QExpectedImprovement::new`]; [`BotorchError::Unsupported`] for
    /// outcome constraints (as in BoTorch, use a
    /// [`crate::acquisition::ConstrainedMcObjective`]).
    pub fn new(model: Arc<dyn Model>, options: McAcquisitionOptions) -> Result<Self> {
        let core = McCore::new(model, options)?;
        core.reject_constraints("qSimpleRegret")?;
        Ok(Self { core })
    }

    fn reduce(&self, y: &Matrix, ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        reduce_max_mean(y, ind, with_grad, |v| v, |_| 1.0)
    }
}

sample_reducing!(QSimpleRegret);

impl AcquisitionFunction for QSimpleRegret {
    mc_common!();
}

/// MC-based batch Upper Confidence Bound (BoTorch's `qUpperConfidenceBound`,
/// Wilson et al. 2017): `qUCB(X) = E[max_j (m_j + β' |obj_j - m_j|)]` with
/// `β' = sqrt(β π / 2)` and `m_j` the *sample* mean of `obj_j` over the MC
/// samples.
#[derive(Debug)]
pub struct QUpperConfidenceBound {
    core: McCore,
    beta: f64,
    beta_prime: f64,
}

impl QUpperConfidenceBound {
    /// Creates `qUpperConfidenceBound(model, beta, sampler, objective,
    /// X_pending=...)`.
    ///
    /// # Errors
    /// As [`QSimpleRegret::new`], and [`BotorchError::InvalidArgument`]
    /// unless `beta` is finite and non-negative.
    pub fn new(model: Arc<dyn Model>, beta: f64, options: McAcquisitionOptions) -> Result<Self> {
        if !(beta >= 0.0 && beta.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "beta must be finite and non-negative, got {beta}"
            )));
        }
        let core = McCore::new(model, options)?;
        core.reject_constraints("qUpperConfidenceBound")?;
        Ok(Self {
            core,
            beta,
            beta_prime: (beta * std::f64::consts::PI / 2.0).sqrt(),
        })
    }

    /// The trade-off parameter `beta`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    fn reduce(&self, y: &Matrix, ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        reduce_centered(y, ind, 1.0, self.beta_prime, with_grad)
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

sample_reducing!(QUpperConfidenceBound);

impl AcquisitionFunction for QUpperConfidenceBound {
    mc_common!();
}

/// MC-based batched Lower Confidence Bound (BoTorch's
/// `qLowerConfidenceBound`): [`QUpperConfidenceBound`] with `-β'`,
/// `qLCB(X) = E[max_j (m_j - β' |obj_j - m_j|)]`, a pessimistic (risk-averse)
/// criterion to be maximized.
#[derive(Debug)]
pub struct QLowerConfidenceBound {
    core: McCore,
    beta: f64,
    beta_prime: f64,
}

impl QLowerConfidenceBound {
    /// Creates `qLowerConfidenceBound(model, beta, sampler, objective,
    /// X_pending=...)`.
    ///
    /// # Errors
    /// See [`QUpperConfidenceBound::new`].
    pub fn new(model: Arc<dyn Model>, beta: f64, options: McAcquisitionOptions) -> Result<Self> {
        if !(beta >= 0.0 && beta.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "beta must be finite and non-negative, got {beta}"
            )));
        }
        let core = McCore::new(model, options)?;
        core.reject_constraints("qLowerConfidenceBound")?;
        Ok(Self {
            core,
            beta,
            beta_prime: -(beta * std::f64::consts::PI / 2.0).sqrt(),
        })
    }

    /// The trade-off parameter `beta`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    fn reduce(&self, y: &Matrix, ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        reduce_centered(y, ind, 1.0, self.beta_prime, with_grad)
    }
}

sample_reducing!(QLowerConfidenceBound);

impl AcquisitionFunction for QLowerConfidenceBound {
    mc_common!();
}

/// MC-based batch posterior standard deviation (BoTorch's
/// `qPosteriorStandardDeviation`), an acquisition function for pure
/// exploration: `E[max_j sqrt(π/2) |obj_j - m_j| · ind_j]` with the sample
/// mean `m_j` of `obj_j` (for a Gaussian, `E|Y - μ| = σ sqrt(2/π)`) and the
/// smoothed feasibility indicator `ind` of optional outcome constraints.
#[derive(Debug)]
pub struct QPosteriorStandardDeviation {
    core: McCore,
}

impl QPosteriorStandardDeviation {
    /// Creates `qPosteriorStandardDeviation(model, sampler, objective,
    /// X_pending, constraints, eta)`.
    ///
    /// # Errors
    /// See [`QExpectedImprovement::new`].
    pub fn new(model: Arc<dyn Model>, options: McAcquisitionOptions) -> Result<Self> {
        Ok(Self {
            core: McCore::new(model, options)?,
        })
    }

    fn reduce(&self, y: &Matrix, ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        SampleUtility::PosteriorStandardDeviation.reduce(y, ind, with_grad)
    }
}

sample_reducing!(QPosteriorStandardDeviation);

/// Not in BoTorch's `is_nonnegative` list (so `optimize_acqf` uses the
/// Boltzmann initialization, as in BoTorch), although the values are
/// non-negative.
impl AcquisitionFunction for QPosteriorStandardDeviation {
    mc_common!();
}

/// Number of jitter attempts of the Cholesky factorization of the low-rank
/// update of the cached baseline root (`sample_cached_cholesky(max_tries=6)`).
const LOW_RANK_MAX_TRIES: usize = 6;

/// Number of uniformly random points at which BoTorch estimates a lower bound
/// of the objective when no baseline point is feasible.
const NEI_LOWER_BOUND_POINTS: usize = 32;
/// Salt of the seed of those points (derived from the sampler's seed).
const NEI_LOWER_BOUND_SALT: u64 = 0x6e65_695f_6c62_6e64;

/// Options of [`QNoisyExpectedImprovement`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QNoisyExpectedImprovementOptions {
    /// The sampler of base samples (default: 512 Sobol samples, random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points, appended to every candidate set.
    pub x_pending: Option<Matrix>,
    /// The MC objective (required for multi-output models without a
    /// posterior transform).
    pub objective: Option<Arc<dyn McObjective>>,
    /// Outcome constraints (BoTorch's `constraints` and `eta`, logistic
    /// sigmoid); the incumbent of every sample is the best *feasible*
    /// baseline value.
    pub constraints: Option<OutcomeConstraints>,
    /// Remove baseline points that are very unlikely to be the best
    /// (`prune_inferior_points`, default `true`) under the objective and the
    /// constraints.
    pub prune_baseline: bool,
    /// Settings of the pruning (2048 samples by default).
    pub prune_options: crate::acquisition::utils::PruneOptions,
    /// Sample the baseline once and the candidates by a low-rank update of
    /// the cached root of the baseline covariance (BoTorch's `cache_root`,
    /// default `true`; see [`QNoisyExpectedImprovement`]). Requires a model
    /// whose outputs are independent (otherwise the joint posterior is
    /// sampled with the same base samples).
    pub cache_root: bool,
    /// The points (`k × d`) at which the lower bound of the objective is
    /// estimated when a sample has no feasible baseline point. `None` (the
    /// default) draws 32 uniformly random points in the expanded bounding box
    /// of the baseline with a generator seeded by the sampler's seed (BoTorch
    /// draws them from torch's global generator).
    pub lower_bound_points: Option<Matrix>,
}

impl Default for QNoisyExpectedImprovementOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            x_pending: None,
            objective: None,
            constraints: None,
            prune_baseline: true,
            prune_options: crate::acquisition::utils::PruneOptions::default(),
            cache_root: true,
            lower_bound_points: None,
        }
    }
}

/// The cached quantities of [`QNoisyExpectedImprovement`] with `cache_root`:
/// the roots of the baseline covariance of every output and the incumbents
/// of every sample (BoTorch's `_baseline_L` and `_baseline_best_f`).
#[derive(Debug)]
struct NeiBaseline {
    /// Roots of the baseline covariance, one per output (`None` if the
    /// outputs are not independent or a root is not triangular: the joint
    /// posterior is sampled instead).
    roots: Option<Vec<CovarianceRoot>>,
    /// Base samples of the baseline (`S × N·m`).
    base: Arc<Matrix>,
    /// Best feasible baseline objective of every sample.
    best_f: Vec<f64>,
}

/// One output's part of the low-rank update: `L_XB`, the factor of the
/// Schur complement, and the baseline and new base samples (`nb × S`,
/// `nq × S`).
type LowRankBlock = (Matrix, crate::linalg::CholeskyFactor, Matrix, Matrix);

/// The samples of the new points and what their reverse pass needs.
enum NeiSamples {
    /// Joint samples of `[baseline; X; X_pending]`.
    Joint(SampledObjective),
    /// Samples of the new points from the low-rank update of the cached
    /// baseline roots.
    LowRank {
        x_full: Matrix,
        samples: McSamples,
        obj: Matrix,
        pullback: Option<SamplePullback>,
        /// Posterior points of the baseline and in total.
        nb: usize,
        n: usize,
        blocks: Vec<LowRankBlock>,
    },
}

/// MC-based batch Noisy Expected Improvement (BoTorch's
/// `qNoisyExpectedImprovement`):
/// `qNEI(X) = E[max_j (max(obj_j - max_i obj_i^baseline, 0) · ind_j)]` over
/// joint posterior samples of the candidates (with pending points appended)
/// and the previously evaluated (baseline) points, where the incumbent of
/// every sample is the best feasible objective value at the (pruned)
/// baseline and `ind` the smoothed feasibility indicator of outcome
/// constraints. BoTorch recommends
/// [`crate::acquisition::QLogNoisyExpectedImprovement`].
///
/// With `cache_root` (BoTorch's default for GP models), the baseline is
/// sampled once with the base samples of the sampler for its `N` points,
/// which give the incumbents, and the candidates' samples are
/// `μ_X + L_XB z_B + L_XX z_X` with the cached root `L_B` of the baseline
/// covariance of every output, `L_XB = Σ_XB L_B⁻ᵀ` and `L_XX = chol(Σ_XX -
/// L_XB L_XBᵀ)` (BoTorch's `sample_cached_cholesky`; up to six jitter
/// attempts, falling back to the joint posterior), where `z_B` are the
/// baseline's base samples and `z_X` rows `N..` of the base samples for all
/// `N + q` points (BoTorch's `_set_sampler`). Without `cache_root`, the joint
/// posterior of `[X_baseline; X; X_pending]` is sampled with the base samples
/// for all points and the incumbents come from the same samples. Incumbents
/// carry no gradient, as in BoTorch.
#[derive(Debug)]
pub struct QNoisyExpectedImprovement {
    core: McCore,
    x_baseline: Matrix,
    cache_root: bool,
    baseline: Option<NeiBaseline>,
    /// The base samples of `N + q'` points with the baseline's rows (cache
    /// root), per number of posterior points.
    joint_base: Mutex<HashMap<usize, Arc<Matrix>>>,
    lower_bound: Mutex<Option<f64>>,
    lower_bound_points: Option<Matrix>,
}

impl QNoisyExpectedImprovement {
    /// Creates `qNoisyExpectedImprovement(model, X_baseline, sampler,
    /// objective, X_pending, prune_baseline, cache_root, constraints, eta)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models without an
    /// objective, [`BotorchError::InvalidArgument`] for an empty baseline or
    /// constraints together with a [`crate::acquisition::ConstrainedMcObjective`],
    /// [`BotorchError::DimensionMismatch`] for inputs of the wrong dimension,
    /// and model errors from the pruning and the baseline posterior.
    pub fn new(
        model: Arc<dyn Model>,
        x_baseline: &Matrix,
        options: QNoisyExpectedImprovementOptions,
    ) -> Result<Self> {
        let d = model.dim();
        if x_baseline.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "X_baseline must have {d} columns, got {}",
                x_baseline.ncols()
            )));
        }
        if x_baseline.nrows() == 0 {
            return Err(BotorchError::InvalidArgument(
                "X_baseline must have at least one point.".into(),
            ));
        }
        if let Some(p) = &options.lower_bound_points {
            if p.ncols() != d || p.nrows() == 0 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "lower_bound_points must be k x {d} with k > 0, got {}x{}",
                    p.nrows(),
                    p.ncols()
                )));
            }
        }
        let core = McCore::new(
            model,
            McAcquisitionOptions {
                sampler: options.sampler,
                x_pending: options.x_pending,
                objective: options.objective,
                constraints: options.constraints,
            },
        )?;
        let x_baseline = if options.prune_baseline {
            let mut prune = options.prune_options;
            prune.objective = core.objective.clone();
            prune.constraints = core.constraints.as_ref().map(|c| c.constraints().to_vec());
            crate::acquisition::utils::prune_inferior_points(
                core.model.as_ref(),
                x_baseline,
                &prune,
            )?
        } else {
            x_baseline.clone()
        };
        let mut acqf = Self {
            core,
            x_baseline,
            cache_root: options.cache_root,
            baseline: None,
            joint_base: Mutex::new(HashMap::new()),
            lower_bound: Mutex::new(None),
            lower_bound_points: options.lower_bound_points,
        };
        if acqf.cache_root {
            acqf.baseline = Some(acqf.sample_baseline()?);
        }
        Ok(acqf)
    }

    /// The (pruned) baseline points (BoTorch's `X_baseline`).
    pub fn x_baseline(&self) -> &Matrix {
        &self.x_baseline
    }

    /// Whether the baseline root is cached (BoTorch's `_cache_root`).
    pub fn cache_root(&self) -> bool {
        self.cache_root
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.core.sampler
    }

    /// The incumbent of every sample computed from the baseline alone
    /// (BoTorch's `_baseline_best_f`; only with `cache_root`).
    pub fn baseline_best_f(&self) -> Option<&[f64]> {
        self.baseline.as_ref().map(|b| b.best_f.as_slice())
    }

    /// The points at which the lower bound of the objective is estimated for
    /// infeasible baselines (see
    /// [`QNoisyExpectedImprovementOptions::lower_bound_points`]).
    pub fn lower_bound_points(&self) -> Matrix {
        if let Some(p) = &self.lower_bound_points {
            return p.clone();
        }
        let xb = &self.x_baseline;
        let d = xb.ncols();
        let mut rng =
            Xoshiro256PlusPlus::seed_from_u64(self.core.sampler.seed() ^ NEI_LOWER_BOUND_SALT);
        let u = Matrix::from_fn(NEI_LOWER_BOUND_POINTS, d, |_, _| rng.random::<f64>());
        Matrix::from_fn(NEI_LOWER_BOUND_POINTS, d, |i, k| {
            let col = xb.column(k);
            let (lo, hi) = (col.min(), col.max());
            let padding = 0.1 * (hi - lo);
            lo - padding + u[(i, k)] * (hi - lo + 2.0 * padding)
        })
    }

    /// The lower bound of the objective used for infeasible baseline points,
    /// if an evaluation needed it.
    pub fn infeasible_lower_bound(&self) -> Option<f64> {
        *self.lower_bound.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn infeasible_value(&self) -> Result<f64> {
        let mut lb = self.lower_bound.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = *lb {
            return Ok(v);
        }
        let v = crate::acquisition::utils::estimate_objective_lower_bound(
            self.core.model.as_ref(),
            self.core.objective.as_deref(),
            &self.lower_bound_points(),
        )?;
        *lb = Some(v);
        Ok(v)
    }

    /// The incumbents of samples of the baseline (objective values `obj` of
    /// the `N` baseline inputs).
    fn best_feasible(&self, samples: &McSamples, obj: &Matrix) -> Result<Vec<f64>> {
        crate::acquisition::utils::compute_best_feasible_objective(
            samples,
            obj,
            self.core.constraints.as_ref().map(|c| c.constraints()),
            || self.infeasible_value(),
        )
    }

    /// Samples the baseline with its own base samples (BoTorch's `__init__`
    /// with `cache_root`).
    fn sample_baseline(&self) -> Result<NeiBaseline> {
        let model = self.core.model.as_ref();
        let js = JointSamples::new(model, self.x_baseline.clone(), &self.core.sampler)?;
        let nb = js.num_points();
        let samples = js.samples(0..nb);
        let obj = match &self.core.objective {
            Some(o) => o.evaluate(&samples, Some(&self.x_baseline), false)?.values,
            None => samples.data().clone(),
        };
        let best_f = self.best_feasible(&samples, &obj)?;
        let roots = match &js.roots {
            SampleRoots::Blocks(rs)
                if rs
                    .iter()
                    .all(|r| !matches!(r, CovarianceRoot::Eigen(_) | CovarianceRoot::Zero(_))) =>
            {
                Some(rs.clone())
            }
            _ => None,
        };
        Ok(NeiBaseline {
            roots,
            base: Arc::clone(&js.base),
            best_f,
        })
    }

    /// The base samples of all `n` posterior points with the baseline's
    /// points' replaced by the baseline's own (BoTorch's
    /// `_update_base_samples`).
    fn effective_base(&self, baseline: &NeiBaseline, n: usize) -> Result<Arc<Matrix>> {
        let mut cache = self.joint_base.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(z) = cache.get(&n) {
            return Ok(Arc::clone(z));
        }
        let m = self.core.model.num_outputs();
        let all = self.core.sampler.base_samples(n * m)?;
        let mut z = (*all).clone();
        let k = baseline.base.ncols();
        z.columns_mut(0, k).copy_from(baseline.base.as_ref());
        let z = Arc::new(z);
        cache.insert(n, Arc::clone(&z));
        Ok(z)
    }

    /// Samples of the new points from the low-rank update (`None` if it
    /// fails, as BoTorch's `NanError` / `NotPSDError` fallback).
    fn low_rank(
        &self,
        roots: &[CovarianceRoot],
        x_full: Matrix,
        x_new: &Matrix,
        z: &Matrix,
        with_grad: bool,
    ) -> Result<Option<NeiSamples>> {
        let model = self.core.model.as_ref();
        let post = model.posterior(&x_full, false)?;
        let (n, m) = (post.num_points(), post.num_outputs());
        if roots.len() != m {
            return Ok(None);
        }
        let nb = roots[0].dim();
        let nq = n - nb;
        let s = z.nrows();
        let mut data = Matrix::zeros(m * nq, s);
        let mut blocks = Vec::with_capacity(m);
        for (t, root) in roots.iter().enumerate() {
            let cov = post.covariance.view((t * n, t * n), (n, n));
            let bl = cov.view((nb, 0), (nq, nb)).into_owned();
            let bl_chol = match root {
                CovarianceRoot::Scalar { root, .. } => bl.map(|v| v / root),
                CovarianceRoot::Cholesky(f) => {
                    let mut ct = bl.transpose();
                    f.solve_lower_mut(&mut ct);
                    ct.transpose()
                }
                CovarianceRoot::Eigen(_) | CovarianceRoot::Zero(_) => return Ok(None),
            };
            let mut e = cov.view((nb, nb), (nq, nq)).into_owned();
            e.gemm(-1.0, &bl_chol, &bl_chol.transpose(), 1.0);
            let br = match crate::linalg::psd_safe_cholesky_with(
                &e,
                crate::linalg::CHOLESKY_JITTER,
                LOW_RANK_MAX_TRIES,
            ) {
                Ok(f) => f,
                Err(BotorchError::NotPsd { .. } | BotorchError::Nan { .. }) => return Ok(None),
                Err(err) => return Err(err),
            };
            let zb = Matrix::from_fn(nb, s, |p, c| z[(c, p * m + t)]);
            let zx = Matrix::from_fn(nq, s, |p, c| z[(c, (nb + p) * m + t)]);
            let mut y = &bl_chol * &zb;
            y.gemm(1.0, br.l(), &zx, 1.0);
            for p in 0..nq {
                let mu = post.mean[t * n + nb + p];
                let mut dst = data.row_mut(t * nq + p);
                for c in 0..s {
                    dst[c] = y[(p, c)] + mu;
                }
            }
            blocks.push((bl_chol, br, zb, zx));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        let samples = McSamples::new(data, nq, m)?;
        let (obj, pullback) = match &self.core.objective {
            Some(o) => {
                let v = o.evaluate(&samples, Some(x_new), with_grad)?;
                if v.values.shape() != (x_new.nrows(), s) {
                    return Err(crate::acquisition::mc_objective::q_batch_mismatch(
                        v.values.nrows(),
                        x_new.nrows(),
                    ));
                }
                (v.values, v.pullback)
            }
            None => {
                if nq != x_new.nrows() {
                    return Err(crate::acquisition::mc_objective::q_batch_mismatch(
                        nq,
                        x_new.nrows(),
                    ));
                }
                (samples.data().clone(), None)
            }
        };
        Ok(Some(NeiSamples::LowRank {
            x_full,
            samples,
            obj,
            pullback,
            nb,
            n,
            blocks,
        }))
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let (v, g) = self.evaluate_weighted(x, None, with_grad)?;
        Ok((v, g.map(|(g, _)| g)))
    }

    fn evaluate_weighted(
        &self,
        x: &Matrix,
        weights: Option<&[f64]>,
        with_grad: bool,
    ) -> Result<crate::acquisition::WeightedEvaluation> {
        let model = self.core.model.as_ref();
        let d = model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let x_new = match read_lock(&self.core.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let n_in = self.x_baseline.nrows();
        let x_full = vstack(&[&self.x_baseline, &x_new], d);
        // Samples and objective values of the new points; incumbents.
        let (path, best_f): (NeiSamples, Vec<f64>) = match &self.baseline {
            Some(bl) => {
                let n_points = x_full.nrows() * model.points_per_input();
                let z = self.effective_base(bl, n_points)?;
                let low = match &bl.roots {
                    Some(roots) => self.low_rank(roots, x_full.clone(), &x_new, &z, with_grad)?,
                    None => None,
                };
                let path = match low {
                    Some(p) => p,
                    None => {
                        let js = JointSamples::with_base_samples(model, x_full, z)?;
                        NeiSamples::Joint(SampledObjective::from_joint(
                            js,
                            self.core.objective.as_deref(),
                            with_grad,
                        )?)
                    }
                };
                (path, bl.best_f.clone())
            }
            None => {
                let so = SampledObjective::new(
                    model,
                    x_full,
                    &self.core.sampler,
                    self.core.objective.as_deref(),
                    with_grad,
                )?;
                let k = so.js.points_per_input();
                let best = self.best_feasible(
                    &so.samples.points(0..n_in * k),
                    &so.obj.rows(0, n_in).into_owned(),
                )?;
                (NeiSamples::Joint(so), best)
            }
        };
        // The objective values and samples of the new points.
        let (obj, new_samples) = match &path {
            NeiSamples::Joint(so) => {
                let k = so.js.points_per_input();
                let total = so.js.num_points();
                (
                    so.obj.rows(n_in, x_new.nrows()).into_owned(),
                    so.samples.points(n_in * k..total),
                )
            }
            NeiSamples::LowRank { samples, obj, .. } => (obj.clone(), samples.clone()),
        };
        let ind = match &self.core.constraints {
            Some(c) => Some(compute_smoothed_feasibility_indicator(
                c,
                &self.core.fat,
                &new_samples,
                false,
                with_grad,
            )?),
            None => None,
        };
        if let Some(w) = weights {
            if w.len() != obj.nrows() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {} point weights, got {}",
                    obj.nrows(),
                    w.len()
                )));
            }
        }
        let combined = weighted_indicator(
            ind.as_ref().map(|i| &i.values),
            weights,
            false,
            obj.nrows(),
            obj.ncols(),
        );
        let r = noisy_improvement_reduce(&obj, &best_f, combined.as_ref(), with_grad);
        if !with_grad {
            return Ok((r.value, None));
        }
        let obj_bar_new = r.obj_bar.expect("gradient requested");
        let (ind_bar, w_bar) =
            split_indicator_adjoint(ind.as_ref().map(|i| &i.values), weights, false, r.ind_bar);
        let samples_bar_new = match (ind, ind_bar) {
            (Some(ind), Some(ib)) => Some(ind.backward(&ib)?),
            _ => None,
        };
        let grad = match path {
            NeiSamples::Joint(so) => {
                let k = so.js.points_per_input();
                let total = so.js.num_points();
                let n_all = n_in + x_new.nrows();
                let mut obj_bar = Matrix::zeros(n_all, obj_bar_new.ncols());
                obj_bar
                    .rows_mut(n_in, x_new.nrows())
                    .copy_from(&obj_bar_new);
                let samples_bar =
                    samples_bar_new.map(|sb| so.samples.embed_points_adjoint(n_in * k..total, &sb));
                so.backward(model, &obj_bar, samples_bar, n_in..n_in + q)?
            }
            NeiSamples::LowRank {
                x_full,
                pullback,
                nb,
                n,
                blocks,
                ..
            } => {
                let nq = n - nb;
                let (mut sbar, xbar) = match pullback {
                    Some(pb) => pb(&obj_bar_new)?,
                    None => (obj_bar_new.clone(), None),
                };
                if let Some(extra) = samples_bar_new {
                    sbar += extra;
                }
                let m = blocks.len();
                let mut mean_bar = Vector::zeros(n * m);
                let mut cov_bar = Matrix::zeros(n * m, n * m);
                let roots = self
                    .baseline
                    .as_ref()
                    .and_then(|b| b.roots.as_ref())
                    .ok_or_else(|| {
                        BotorchError::Runtime("low-rank samples without cached roots".into())
                    })?;
                for (t, ((bl, br, zb, zx), root)) in blocks.iter().zip(roots).enumerate() {
                    let yb = sbar.rows(t * nq, nq);
                    for p in 0..nq {
                        mean_bar[t * n + nb + p] = yb.row(p).sum();
                    }
                    let mut blbar = yb * zb.transpose();
                    let brbar = yb * zx.transpose();
                    let ebar = crate::linalg::cholesky_backward(br.l(), &brbar);
                    blbar.gemm(-2.0, &ebar, bl, 1.0);
                    let cbar = match root {
                        CovarianceRoot::Scalar { root, .. } => blbar.map(|v| v / root),
                        CovarianceRoot::Cholesky(f) => {
                            let mut tb = blbar.transpose();
                            f.solve_lower_transpose_mut(&mut tb);
                            tb.transpose()
                        }
                        CovarianceRoot::Eigen(_) | CovarianceRoot::Zero(_) => {
                            return Err(BotorchError::Runtime(
                                "no low-rank path for eigendecomposition or zero roots".into(),
                            ))
                        }
                    };
                    cov_bar
                        .view_mut((t * n + nb, t * n), (nq, nb))
                        .copy_from(&cbar);
                    cov_bar
                        .view_mut((t * n + nb, t * n + nb), (nq, nq))
                        .copy_from(&ebar);
                }
                let mut grad =
                    model.joint_posterior_vjp(&x_full, n_in..n_in + q, &mean_bar, &cov_bar)?;
                if let Some(xb) = xbar {
                    grad += xb.rows(0, q);
                }
                grad
            }
        };
        Ok((r.value, Some((grad, w_bar))))
    }
}

impl crate::acquisition::SampleReducing for QNoisyExpectedImprovement {
    fn evaluate_weighted(
        &self,
        x: &Matrix,
        weights: &[f64],
        with_grad: bool,
    ) -> Result<crate::acquisition::WeightedEvaluation> {
        QNoisyExpectedImprovement::evaluate_weighted(self, x, Some(weights), with_grad)
    }

    fn log_utilities(&self) -> bool {
        false
    }
}

/// `mean_s max_j max(obj_js - best_f_s, 0) · ind_js` and the adjoints of
/// `obj` and `ind` (qNoisyExpectedImprovement's reduction).
fn noisy_improvement_reduce(
    obj: &Matrix,
    best_f: &[f64],
    ind: Option<&Matrix>,
    with_grad: bool,
) -> Reduced {
    let (n, s) = obj.shape();
    let u = Matrix::from_fn(n, s, |i, c| (obj[(i, c)] - best_f[c]).max(0.0));
    let weighted = ind.map(|w| u.component_mul(w));
    let (maxima, w) = amax_columns(weighted.as_ref().unwrap_or(&u));
    let value = mean(&maxima);
    if !with_grad {
        return Reduced {
            value,
            obj_bar: None,
            ind_bar: None,
        };
    }
    let a_bar = w / s as f64;
    let ind_bar = ind.map(|_| a_bar.component_mul(&u));
    let mut g = match ind {
        Some(ind) => a_bar.component_mul(ind),
        None => a_bar,
    };
    for i in 0..n {
        for c in 0..s {
            if g[(i, c)] != 0.0 && obj[(i, c)] - best_f[c] < 0.0 {
                g[(i, c)] = 0.0;
            }
        }
    }
    Reduced {
        value,
        obj_bar: Some(g),
        ind_bar,
    }
}

impl AcquisitionFunction for QNoisyExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.model
    }

    fn as_sample_reducing(&self) -> Option<&dyn crate::acquisition::SampleReducing> {
        Some(self)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.core.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.core.set_x_pending(x_pending)
    }

    fn is_nonnegative(&self) -> bool {
        true
    }
}
