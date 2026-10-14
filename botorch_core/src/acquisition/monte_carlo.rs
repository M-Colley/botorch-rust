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
use std::ops::Range;
use std::sync::{Arc, RwLock};

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
        model.joint_posterior_vjp(&self.x, grad_rows, &mean_bar, &cov_bar)
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
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let x_full = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
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
        let r = reduce(&so.obj, ind.as_ref().map(|i| &i.values), with_grad);
        if !with_grad {
            return Ok((r.value, None));
        }
        let obj_bar = r.obj_bar.expect("gradient requested");
        let samples_bar = match (ind, r.ind_bar) {
            (Some(ind), Some(ib)) => Some(ind.backward(&ib)?),
            _ => None,
        };
        let grad = so.backward(self.model.as_ref(), &obj_bar, samples_bar, 0..q)?;
        Ok((r.value, Some(grad)))
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

    fn reduce(&self, y: &Matrix, _ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        let (n, s) = y.shape();
        let bp = self.beta_prime;
        let means: Vec<f64> = (0..n).map(|i| y.row(i).sum() / s as f64).collect();
        let u = Matrix::from_fn(n, s, |i, c| means[i] + bp * (y[(i, c)] - means[i]).abs());
        let (maxima, w) = amax_columns(&u);
        let value = mean(&maxima);
        if !with_grad {
            return Reduced {
                value,
                obj_bar: None,
                ind_bar: None,
            };
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
        Reduced {
            value,
            obj_bar: Some(g),
            ind_bar: None,
        }
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
