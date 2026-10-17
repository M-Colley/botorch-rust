//! The machinery shared by qNEHVI and qLogNEHVI: BoTorch's
//! `NoisyExpectedHypervolumeMixin` (box decompositions of the Pareto fronts
//! of the baseline under every MC sample, cached pending points) and
//! `CachedCholeskyMCSamplerMixin` (`cache_root`: samples of new points from
//! low-rank updates of the cached root of the baseline covariance).
//!
//! # Base samples
//!
//! The baseline (including cached pending points) is sampled once with the
//! base samples `Z_N` of the sampler for its `N` points (`S × N·m`). For new
//! points `X` (the candidates and any uncached pending points, `q'` of them),
//! BoTorch's `_set_sampler` uses the base samples of the sampler for `N + q'`
//! points with the first `N` rows replaced by the baseline's, so that the
//! samples of `X` are consistent with the samples of the decompositions.
//! When pending points are cached, the base samples of the previously
//! decomposed points are kept and only the new rows are taken from the draw
//! for all points (the fix of meta-pytorch/botorch#3392; BoTorch main redraws
//! all rows, so that the cached hypervolumes and `_baseline_hvs` of
//! non-incremental NEHVI use different samples of the observed baseline).
//!
//! # Sampling new points
//!
//! With `cache_root` (BoTorch's default), the samples of `X` are
//! `μ_X + [L_XB, L_XX] z` with `L_XB = Σ_XB L_B⁻ᵀ` and `L_XX =
//! chol(Σ_XX - L_XB L_XBᵀ)` (up to five jitter retries) for the cached root
//! `L_B` of every output's baseline covariance; the cross-covariances come
//! from cached `K⁻¹ K(X_train, B)` products of the exact GPs of the model.
//! If the low-rank update fails (not positive definite, non-finite samples),
//! the joint posterior is sampled instead, as in BoTorch. Without
//! `cache_root`, the joint posterior of `[B; X]` is sampled with one root per
//! output (BoTorch's `cache_root=False`). Both agree to rounding unless the
//! joint covariance needs jitter.
//!
//! # One-to-many input transforms
//!
//! With `n_w` posterior points per input (BoTorch's `InputPerturbation`, see
//! [`crate::models::Model::points_per_input`]), every count of points above
//! is a count of posterior points (`N·n_w` for the baseline, `q'·n_w` for the
//! new points, BoTorch's `q_in = q * n_w`), and the objective (a risk
//! measure) maps them back to one (or, for set-valued objectives such as
//! `MVaR`, several) objective points per input.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use rayon::prelude::*;

use crate::acquisition::mc_objective::McSamples;
use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock, JointSamples, SampleRoots};
use crate::acquisition::multi_objective::hypervolume::HypercellBounds;
use crate::acquisition::multi_objective::objective::check_values;
use crate::acquisition::multi_objective::utils::{
    prune_inferior_points_multi_objective, PruneMultiObjectiveOptions,
};
use crate::acquisition::multi_objective::{MoMcCore, Reduction};
use crate::acquisition::{check_candidates, check_pending};
use crate::error::{BotorchError, Result};
use crate::linalg::{cholesky_backward, psd_safe_cholesky_with, CholeskyFactor, CovarianceRoot};
use crate::models::gp::BaselineCache;
use crate::utils::multi_objective::box_decompositions::utils::pad_batch_pareto_frontier;
use crate::utils::multi_objective::box_decompositions::{
    BoxDecomposition, BoxDecompositionList, DominatedPartitioning, FastNondominatedPartitioning,
    NondominatedPartitioning,
};
use crate::utils::multi_objective::PointBatch;
use crate::utils::objective::compute_feasibility_indicator;
use crate::utils::safe_math::logsumexp_with_grad;
use crate::Matrix;
use nalgebra::DMatrix;

/// Number of jitter attempts of the Cholesky factorization of the low-rank
/// update (`sample_cached_cholesky(max_tries=6)`).
const LOW_RANK_MAX_TRIES: usize = 6;

/// The settings of the NEHVI machinery (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub(crate) struct NehviConfig {
    pub ref_point: Vec<f64>,
    pub alpha: f64,
    pub cache_pending: bool,
    pub max_iep: usize,
    pub incremental: bool,
    pub cache_root: bool,
    pub prune_baseline: bool,
    pub prune_options: PruneMultiObjectiveOptions,
}

/// The cached roots of the baseline covariance of every output and what the
/// cross-covariances with new points need.
#[derive(Debug)]
struct CachedRoot {
    roots: Vec<CovarianceRoot>,
    caches: Vec<BaselineCache>,
    /// The baseline base samples of every output, transposed (`N × S`).
    zbt: Vec<Matrix>,
}

/// The state that depends on the baseline and the pending points.
#[derive(Debug)]
struct NehviState {
    /// The observed (pruned) baseline (`_X_baseline`).
    x_baseline: Matrix,
    /// The baseline and the cached pending points (`X_baseline`).
    x_cached: Matrix,
    /// All pending points.
    x_pending: Option<Matrix>,
    /// Base samples of the cached points (`base_sampler.base_samples`).
    base_z: Option<Arc<Matrix>>,
    /// Base samples of the observed baseline (`_baseline_base_sampler`).
    baseline_base_z: Option<Arc<Matrix>>,
    /// Whether a decomposition exists (`partitioning is not None`).
    decomposed: bool,
    cells: Arc<HypercellBounds>,
    /// Hypervolume of the cached points under every sample.
    hypervolumes: Vec<f64>,
    /// Hypervolume of the observed baseline (non-incremental NEHVI).
    baseline_hvs: Option<Vec<f64>>,
    prev_nehvi: f64,
    cached: Option<Arc<CachedRoot>>,
}

/// The NEHVI state machine shared by qNEHVI and qLogNEHVI.
#[derive(Debug)]
pub(crate) struct NehviCore {
    pub mc: MoMcCore,
    pub cfg: NehviConfig,
    state: RwLock<NehviState>,
    /// BoTorch's sampler state when it was last sized for the (unbatched)
    /// baseline without an evaluation since (it is reused if the baseline
    /// has to be sampled again without a base sampler).
    sampler_unbatched: Mutex<Option<(usize, Arc<Matrix>)>>,
    /// Base samples of `[baseline; new points]` per number of new points.
    joint_base: Mutex<HashMap<usize, Arc<Matrix>>>,
}

/// How the samples of the new points were computed (for the backward pass).
enum SamplePath {
    Joint {
        js: JointSamples,
        n: usize,
    },
    LowRank {
        cached: Arc<CachedRoot>,
        x_new: Matrix,
        /// Per output: `L_XB` (`q' × N`), `L_XX`, and the base samples of the
        /// new points (`q' × S`).
        outputs: Vec<(Matrix, CholeskyFactor, Matrix)>,
    },
}

impl NehviCore {
    pub fn new(
        mc: MoMcCore,
        cfg: NehviConfig,
        x_baseline: &Matrix,
        x_pending: Option<Matrix>,
    ) -> Result<Self> {
        if cfg.ref_point.len() < 2 {
            return Err(BotorchError::InvalidArgument(format!(
                "NoisyExpectedHypervolumeMixin supports m>=2 outcomes but ref_point has length \
                 {}, which is smaller than 2.",
                cfg.ref_point.len()
            )));
        }
        let d = mc.model.dim();
        if x_baseline.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "X_baseline must have {d} columns, got {}",
                x_baseline.ncols()
            )));
        }
        if let Some(p) = &x_pending {
            check_pending(p, d)?;
        }
        let x_baseline = if cfg.prune_baseline {
            let mut options = cfg.prune_options.clone();
            options.objective = Some(Arc::clone(&mc.objective));
            options.constraints = mc.constraints.as_ref().map(|c| c.constraints().to_vec());
            prune_inferior_points_multi_objective(
                mc.model.as_ref(),
                x_baseline,
                &cfg.ref_point,
                &options,
            )?
        } else {
            x_baseline.clone()
        };
        let mref = cfg.ref_point.len();
        let core = Self {
            state: RwLock::new(NehviState {
                x_cached: x_baseline.clone(),
                x_baseline,
                x_pending: None,
                base_z: None,
                baseline_base_z: None,
                decomposed: false,
                cells: Arc::new(HypercellBounds::new(1, 0, mref, Vec::new(), Vec::new())?),
                hypervolumes: Vec::new(),
                baseline_hvs: None,
                prev_nehvi: 0.0,
                cached: None,
            }),
            mc,
            cfg,
            sampler_unbatched: Mutex::new(None),
            joint_base: Mutex::new(HashMap::new()),
        };
        {
            let mut st = write_lock(&core.state);
            // For incremental NEHVI with enough pending points to cache, the
            // first decomposition is over the baseline and the pending points.
            let can_initialize_with_pending = x_pending.as_ref().is_some_and(|p| {
                core.cfg.cache_pending && core.cfg.incremental && p.nrows() > core.cfg.max_iep
            });
            if !can_initialize_with_pending {
                core.set_cell_bounds(&mut st, None)?;
            }
            if x_pending.is_some() {
                core.set_x_pending_locked(&mut st, x_pending)?;
            }
        }
        Ok(core)
    }

    pub fn x_baseline(&self) -> Matrix {
        read_lock(&self.state).x_baseline.clone()
    }

    pub fn x_cached(&self) -> Matrix {
        read_lock(&self.state).x_cached.clone()
    }

    pub fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.state).x_pending.clone()
    }

    pub fn cells(&self) -> Arc<HypercellBounds> {
        Arc::clone(&read_lock(&self.state).cells)
    }

    pub fn prev_nehvi(&self) -> f64 {
        read_lock(&self.state).prev_nehvi
    }

    pub fn hypervolumes(&self) -> Vec<f64> {
        read_lock(&self.state).hypervolumes.clone()
    }

    pub fn baseline_hvs(&self) -> Option<Vec<f64>> {
        read_lock(&self.state).baseline_hvs.clone()
    }

    pub fn base_samples(&self) -> Option<Matrix> {
        read_lock(&self.state).base_z.as_deref().cloned()
    }

    /// The number of outputs of the model.
    fn m(&self) -> usize {
        self.mc.model.num_outputs()
    }

    /// The number of posterior points per input (`n_w`).
    fn nw(&self) -> usize {
        self.mc.model.points_per_input()
    }

    /// BoTorch's `_set_cell_bounds(num_appended_points)`: samples the cached
    /// points, records the base samples and the roots, and decomposes the
    /// non-dominated space under every sample.
    fn set_cell_bounds(&self, st: &mut NehviState, _num_appended: Option<usize>) -> Result<()> {
        let m = self.m();
        let n = st.x_cached.nrows();
        // Posterior points of the cached points.
        let np = n * self.nw();
        let n_obs = st.x_baseline.nrows();
        let mref = self.cfg.ref_point.len();
        let num_samples = self.mc.sampler.num_samples();
        self.joint_base
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let (obj, feas) = if n > 0 {
            let z = match st.base_z.take() {
                None => {
                    // `get_posterior_samples`: the sampler's base samples for
                    // the baseline (kept if it is still sized for them).
                    let mut su = self
                        .sampler_unbatched
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let z = match su.take() {
                        Some((k, z)) if k == n => z,
                        _ => self.mc.sampler.base_samples(np * m)?,
                    };
                    if n == n_obs {
                        st.baseline_base_z = Some(Arc::clone(&z));
                    }
                    z
                }
                // Extend the base samples to the appended points, keeping
                // those of the previously decomposed points (#3392).
                Some(old) if old.ncols() == np * m => old,
                Some(old) => {
                    let mut z = (*self.mc.sampler.base_samples(np * m)?).clone();
                    let k = old.ncols().min(np * m);
                    z.columns_mut(0, k).copy_from(&old.columns(0, k));
                    Arc::new(z)
                }
            };
            st.base_z = Some(Arc::clone(&z));
            *self
                .sampler_unbatched
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some((n, Arc::clone(&z)));
            let js = JointSamples::with_base_samples(
                self.mc.model.as_ref(),
                st.x_cached.clone(),
                Arc::clone(&z),
            )?;
            let samples = js.samples(0..np);
            st.cached = if self.cfg.cache_root {
                self.cached_root(&js, &st.x_cached, &z)?
            } else {
                None
            };
            let obj = self
                .mc
                .objective
                .evaluate(&samples, Some(&st.x_cached), false)?
                .values;
            check_values(
                &obj,
                &samples,
                Some(&st.x_cached),
                self.mc.objective.verify_output_shape(),
            )?;
            let feas = match &self.mc.constraints {
                Some(c) => Some(compute_feasibility_indicator(c.constraints(), &samples)?),
                None => None,
            };
            (obj, feas)
        } else {
            st.cached = None;
            (
                McSamples::new(Matrix::zeros(0, num_samples), 0, mref)?,
                None,
            )
        };
        if obj.num_outputs() != mref {
            return Err(BotorchError::DimensionMismatch(format!(
                "The objective has {} outputs but the reference point has {mref} elements.",
                obj.num_outputs()
            )));
        }
        // Non-incremental NEHVI measures the improvement over HV(f(B)),
        // computed once before pending points are cached.
        if st.baseline_hvs.is_none() && !self.cfg.incremental {
            if n != n_obs {
                return Err(BotorchError::Runtime(
                    "_baseline_hvs must be computed from the observed baseline only.".into(),
                ));
            }
            st.baseline_hvs = Some(self.dominated_hypervolumes(&obj, feas.as_ref())?);
        }
        let (cells, hvs) = self.partition(&obj, feas.as_ref())?;
        st.cells = Arc::new(cells);
        st.hypervolumes = hvs;
        st.decomposed = true;
        Ok(())
    }

    /// The cached roots of the baseline covariance of every output (`None`
    /// if the model does not consist of independent exact GPs or a root is
    /// not a Cholesky factor).
    fn cached_root(
        &self,
        js: &JointSamples,
        x_cached: &Matrix,
        z: &Matrix,
    ) -> Result<Option<Arc<CachedRoot>>> {
        let Some(gps) = self.mc.model.output_gps() else {
            return Ok(None);
        };
        let SampleRoots::Blocks(roots) = &js.roots else {
            return Ok(None);
        };
        let m = self.m();
        if roots.len() != m || gps.len() != m {
            return Ok(None);
        }
        if roots
            .iter()
            .any(|r| matches!(r, CovarianceRoot::Eigen(_) | CovarianceRoot::Zero(_)))
        {
            return Ok(None);
        }
        // Posterior points of the cached points.
        let n = js.num_points();
        let caches = gps
            .iter()
            .map(|g| g.baseline_cache(x_cached))
            .collect::<Result<Vec<_>>>()?;
        let s = z.nrows();
        let zbt = (0..m)
            .map(|t| Matrix::from_fn(n, s, |b, c| z[(c, b * m + t)]))
            .collect();
        Ok(Some(Arc::new(CachedRoot {
            roots: roots.clone(),
            caches,
            zbt,
        })))
    }

    /// The feasible points of sample `c` (all points without constraints),
    /// row-major.
    fn feasible_points(obj: &McSamples, feas: Option<&DMatrix<bool>>, c: usize) -> Matrix {
        let (n, k) = (obj.num_points(), obj.num_outputs());
        let rows: Vec<usize> = (0..n).filter(|&p| feas.is_none_or(|f| f[(p, c)])).collect();
        Matrix::from_fn(rows.len(), k, |i, t| obj.value(c, rows[i], t))
    }

    /// The hypervolume of the feasible points of every sample
    /// (`_compute_baseline_hvs`).
    fn dominated_hypervolumes(
        &self,
        obj: &McSamples,
        feas: Option<&DMatrix<bool>>,
    ) -> Result<Vec<f64>> {
        let r = &self.cfg.ref_point;
        (0..obj.num_samples())
            .into_par_iter()
            .map(|c| {
                let y = Self::feasible_points(obj, feas, c);
                let dp = DominatedPartitioning::new(r, Some(&y))?;
                Ok(dp.compute_hypervolume()[0])
            })
            .collect()
    }

    /// The box decompositions of the non-dominated space under every sample
    /// (a padded batch for two objectives, a `BoxDecompositionList`
    /// otherwise) and their hypervolumes.
    fn partition(
        &self,
        obj: &McSamples,
        feas: Option<&DMatrix<bool>>,
    ) -> Result<(HypercellBounds, Vec<f64>)> {
        let r = &self.cfg.ref_point;
        let (n, k, s) = (obj.num_points(), obj.num_outputs(), obj.num_samples());
        let alpha = self.cfg.alpha;
        if k > 2 {
            let ys: Vec<Matrix> = (0..s)
                .map(|c| Self::feasible_points(obj, feas, c))
                .collect();
            let list = if alpha > 0.0 {
                BoxDecompositionList::par_build(&ys, |y| {
                    NondominatedPartitioning::new(r, Some(y), alpha)
                })?
            } else {
                BoxDecompositionList::par_build(&ys, |y| {
                    FastNondominatedPartitioning::new(r, Some(y))
                })?
            };
            let cells = HypercellBounds::from_cell_bounds(&list.get_hypercell_bounds()?)?;
            let hvs = list.compute_hypervolume().into_iter().flatten().collect();
            return Ok((cells, hvs));
        }
        let mut values = Vec::with_capacity(s * n * k);
        for c in 0..s {
            for p in 0..n {
                values.extend((0..k).map(|t| obj.value(c, p, t)));
            }
        }
        let y = PointBatch::new(vec![s], n, k, values)?;
        let mask: Option<Vec<bool>> = feas.map(|f| {
            (0..s)
                .flat_map(|c| (0..n).map(move |p| (c, p)))
                .map(|(c, p)| f[(p, c)])
                .collect()
        });
        let padded = pad_batch_pareto_frontier(&y, r, false, mask.as_deref())?;
        let decomposition: Box<dyn BoxDecomposition> = if alpha > 0.0 {
            Box::new(NondominatedPartitioning::from_points(
                r,
                Some(&padded),
                alpha,
            )?)
        } else {
            Box::new(FastNondominatedPartitioning::from_points(r, Some(&padded))?)
        };
        let bounds = decomposition.get_hypercell_bounds().ok_or_else(|| {
            BotorchError::Runtime("the box decomposition has no hypercell bounds".into())
        })?;
        let cells = HypercellBounds::from_cell_bounds(bounds)?;
        Ok((cells, decomposition.compute_hypervolume()))
    }

    /// BoTorch's `set_X_pending` (#3379 semantics: the complete pending set
    /// is replaced; appended points are cached incrementally).
    pub fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.mc.model.dim())?;
        }
        let mut st = write_lock(&self.state);
        self.set_x_pending_locked(&mut st, x_pending.cloned())
    }

    fn set_x_pending_locked(&self, st: &mut NehviState, new: Option<Matrix>) -> Result<()> {
        let old = std::mem::replace(&mut st.x_pending, new.clone());
        match (&old, &new) {
            (None, None) => return Ok(()),
            (Some(a), Some(b)) if a == b => return Ok(()),
            _ => {}
        }
        if !self.cfg.cache_pending {
            return Ok(());
        }
        let d = self.mc.model.dim();
        let empty = Matrix::zeros(0, d);
        let old_p = old.unwrap_or_else(|| empty.clone());
        let new_p = new.unwrap_or_else(|| empty.clone());
        let n_obs = st.x_baseline.nrows();
        let cached = st
            .x_cached
            .rows(n_obs, st.x_cached.nrows() - n_obs)
            .into_owned();
        let is_prefix = |prefix: &Matrix, x: &Matrix| {
            prefix.nrows() <= x.nrows() && *prefix == x.rows(0, prefix.nrows())
        };
        if is_prefix(&cached, &old_p) && is_prefix(&old_p, &new_p) {
            let num_uncached = new_p.nrows() - cached.nrows();
            if num_uncached <= self.cfg.max_iep {
                return Ok(());
            }
            st.x_cached = vstack(&[&st.x_baseline, &new_p], d);
            let num_appended = if st.decomposed {
                Some(num_uncached)
            } else {
                None
            };
            self.set_cell_bounds(st, num_appended)?;
        } else {
            // A replacement invalidates every cache derived from the previous
            // pending points; the base samples of the observed baseline are
            // kept (#3392), `_baseline_hvs` too.
            st.decomposed = false;
            st.cached = None;
            self.mc.subsets.clear();
            st.prev_nehvi = 0.0;
            st.base_z = st.baseline_base_z.clone();
            let to_cache = if new_p.nrows() > self.cfg.max_iep {
                new_p.clone()
            } else {
                empty
            };
            st.x_cached = vstack(&[&st.x_baseline, &to_cache], d);
            self.set_cell_bounds(st, None)?;
        }
        // _prev_nehvi holds only the contribution of the cached pending points.
        if !self.cfg.incremental && new_p.nrows() > self.cfg.max_iep {
            let base = st.baseline_hvs.as_ref().ok_or_else(|| {
                BotorchError::Runtime("_baseline_hvs has not been computed".into())
            })?;
            let diffs: Vec<f64> = st
                .hypervolumes
                .iter()
                .zip(base)
                .map(|(h, b)| crate::utils::multi_objective::clamp_min(h - b, 0.0))
                .collect();
            st.prev_nehvi = diffs.iter().sum::<f64>() / diffs.len() as f64;
        }
        Ok(())
    }

    /// The base samples of `[cached points; q_new new points]`: the cached
    /// points' followed by rows `N..` of the draw for all points.
    fn joint_base(&self, st: &NehviState, q_new: usize) -> Result<Arc<Matrix>> {
        let mut cache = self.joint_base.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(z) = cache.get(&q_new) {
            return Ok(Arc::clone(z));
        }
        let m = self.m();
        let n = st.x_cached.nrows();
        let all = self.mc.sampler.base_samples((n + q_new) * self.nw() * m)?;
        let z = match &st.base_z {
            Some(b) if n > 0 => {
                let mut z = (*all).clone();
                z.columns_mut(0, b.ncols()).copy_from(b.as_ref());
                Arc::new(z)
            }
            _ => all,
        };
        cache.insert(q_new, Arc::clone(&z));
        Ok(z)
    }

    /// Samples of the new points `x_new` from the low-rank update of the
    /// cached baseline roots (`None` if it fails); `n` is the number of
    /// posterior points of the cached points.
    fn low_rank_samples(
        &self,
        cached: &Arc<CachedRoot>,
        x_new: &Matrix,
        z: &Matrix,
        n: usize,
    ) -> Result<Option<(McSamples, SamplePath)>> {
        let Some(gps) = self.mc.model.output_gps() else {
            return Ok(None);
        };
        let m = self.m();
        // Posterior points of the new points.
        let q = x_new.nrows() * self.nw();
        let s = z.nrows();
        let mut data = Matrix::zeros(m * q, s);
        let mut outputs = Vec::with_capacity(m);
        for (t, g) in gps.iter().enumerate() {
            let cp = g.cross_posterior(&cached.caches[t], x_new)?;
            let bl = match &cached.roots[t] {
                CovarianceRoot::Scalar { root, .. } => cp.cross.map(|c| c / root),
                CovarianceRoot::Cholesky(f) => {
                    let mut ct = cp.cross.transpose();
                    f.solve_lower_mut(&mut ct);
                    ct.transpose()
                }
                CovarianceRoot::Eigen(_) | CovarianceRoot::Zero(_) => return Ok(None),
            };
            let mut e = cp.covariance.clone();
            e.gemm(-1.0, &bl, &bl.transpose(), 1.0);
            let br = match psd_safe_cholesky_with(
                &e,
                crate::linalg::CHOLESKY_JITTER,
                LOW_RANK_MAX_TRIES,
            ) {
                Ok(f) => f,
                Err(BotorchError::NotPsd { .. } | BotorchError::Nan { .. }) => return Ok(None),
                Err(err) => return Err(err),
            };
            let zn = Matrix::from_fn(q, s, |p, c| z[(c, (n + p) * m + t)]);
            let mut y = &bl * &cached.zbt[t];
            y.gemm(1.0, br.l(), &zn, 1.0);
            for p in 0..q {
                let mu = cp.mean[p];
                let mut dst = data.row_mut(t * q + p);
                for c in 0..s {
                    dst[c] = y[(p, c)] + mu;
                }
            }
            outputs.push((bl, br, zn));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        let samples = McSamples::new(data, q, m)?;
        Ok(Some((
            samples,
            SamplePath::LowRank {
                cached: Arc::clone(cached),
                x_new: x_new.clone(),
                outputs,
            },
        )))
    }

    /// Backpropagates the adjoint of the samples of the new points to the
    /// first `q` new points (the candidates).
    fn backward(&self, path: SamplePath, sbar: &Matrix, q: usize) -> Result<Matrix> {
        match path {
            SamplePath::Joint { js, n } => {
                // Posterior points of the cached points and of all points.
                let (np, total) = (n * js.points_per_input(), js.num_points());
                js.backward(self.mc.model.as_ref(), np..total, sbar, n..n + q)
            }
            SamplePath::LowRank {
                cached,
                x_new,
                outputs,
            } => {
                let gps =
                    self.mc.model.output_gps().ok_or_else(|| {
                        BotorchError::Runtime("the model lost its output GPs".into())
                    })?;
                // Posterior points of the new points.
                let qn = x_new.nrows() * self.nw();
                let mut grad = Matrix::zeros(x_new.nrows(), x_new.ncols());
                for (t, (g, (bl, br, zn))) in gps.iter().zip(&outputs).enumerate() {
                    let yb = sbar.rows(t * qn, qn);
                    let mbar = crate::Vector::from_fn(qn, |p, _| yb.row(p).sum());
                    let mut blbar = yb * cached.zbt[t].transpose();
                    let brbar = yb * zn.transpose();
                    let ebar = cholesky_backward(br.l(), &brbar);
                    blbar.gemm(-2.0, &ebar, bl, 1.0);
                    let cbar = match &cached.roots[t] {
                        CovarianceRoot::Scalar { root, .. } => blbar.map(|v| v / root),
                        CovarianceRoot::Cholesky(f) => {
                            let mut tb = blbar.transpose();
                            f.solve_lower_transpose_mut(&mut tb);
                            tb.transpose()
                        }
                        CovarianceRoot::Eigen(_) | CovarianceRoot::Zero(_) => {
                            unreachable!("no low-rank path for Eigen or zero roots")
                        }
                    };
                    grad +=
                        g.cross_posterior_vjp(&cached.caches[t], &x_new, &mbar, &cbar, &ebar)?;
                }
                Ok(grad.rows(0, q).into_owned())
            }
        }
    }

    /// The (log) NEHVI of the candidates `x` and its gradient: BoTorch's
    /// `forward` with `_compute_posterior_samples_and_concat_pending`.
    pub fn evaluate(
        &self,
        x: &Matrix,
        reduction: Reduction,
        with_grad: bool,
    ) -> Result<(f64, Option<Matrix>)> {
        let d = self.mc.model.dim();
        check_candidates(x, d, None)?;
        let st = read_lock(&self.state);
        let q = x.nrows();
        let n = st.x_cached.nrows();
        let num_cached_pending = n - st.x_baseline.nrows();
        let x_new = match &st.x_pending {
            Some(p) if p.nrows() > num_cached_pending => {
                let rest = p.rows(num_cached_pending, p.nrows() - num_cached_pending);
                vstack(&[x, &rest.into_owned()], d)
            }
            _ => x.clone(),
        };
        let q_new = x_new.nrows();
        *self
            .sampler_unbatched
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        let z = self.joint_base(&st, q_new)?;
        let nw = self.nw();
        let low_rank = match (&st.cached, self.cfg.cache_root) {
            (Some(c), true) => self.low_rank_samples(c, &x_new, &z, n * nw)?,
            _ => None,
        };
        let (samples, path) = match low_rank {
            Some(r) => r,
            None => {
                let x_full = vstack(&[&st.x_cached, &x_new], d);
                let js = JointSamples::with_base_samples(self.mc.model.as_ref(), x_full, z)?;
                let samples = js.samples(n * nw..(n + q_new) * nw);
                (samples, SamplePath::Joint { js, n })
            }
        };
        let cells = Arc::clone(&st.cells);
        let prev = st.prev_nehvi;
        drop(st);
        let r = self
            .mc
            .reduce_samples(&samples, &x_new, &cells, reduction, with_grad)?;
        let (value, scale) = match reduction {
            Reduction::Linear => (r.value + prev, 1.0),
            Reduction::Log(_) if self.cfg.incremental => (r.value, 1.0),
            Reduction::Log(_) => {
                let (v, w) = logsumexp_with_grad(&[r.value, prev.ln()]);
                (v, w[0])
            }
        };
        if !with_grad {
            return Ok((value, None));
        }
        let sbar = r.samples_bar.expect("gradient requested") * scale;
        let mut grad = self.backward(path, &sbar, q)?;
        if let Some(xb) = r.x_bar {
            grad += xb.rows(0, q) * scale;
        }
        Ok((value, Some(grad)))
    }
}

/// The NEHVI acquisition-function trait methods shared by qNEHVI and
/// qLogNEHVI.
macro_rules! nehvi_common {
    ($reduction:expr) => {
        fn model(&self) -> &std::sync::Arc<dyn crate::models::Model> {
            &self.core.mc.model
        }

        fn forward(&self, x: &crate::Matrix) -> crate::Result<f64> {
            Ok(self.core.evaluate(x, $reduction(self), false)?.0)
        }

        fn forward_with_grad(&self, x: &crate::Matrix) -> crate::Result<(f64, crate::Matrix)> {
            let (v, g) = self.core.evaluate(x, $reduction(self), true)?;
            Ok((v, g.expect("gradient requested")))
        }

        fn x_pending(&self) -> Option<crate::Matrix> {
            self.core.x_pending()
        }

        fn set_x_pending(&self, x_pending: Option<&crate::Matrix>) -> crate::Result<()> {
            self.core.set_x_pending(x_pending)
        }
    };
}

pub(crate) use nehvi_common;

/// Accessors of the NEHVI state shared by qNEHVI and qLogNEHVI.
macro_rules! nehvi_accessors {
    () => {
        /// The observed (pruned) baseline (BoTorch's `_X_baseline`).
        pub fn x_baseline(&self) -> crate::Matrix {
            self.core.x_baseline()
        }

        /// The baseline and the cached pending points (BoTorch's
        /// `X_baseline`).
        pub fn x_baseline_and_pending(&self) -> crate::Matrix {
            self.core.x_cached()
        }

        /// The cell bounds of the box decompositions under every MC sample
        /// (BoTorch's `cell_lower_bounds` / `cell_upper_bounds`).
        pub fn cell_bounds(&self) -> std::sync::Arc<HypercellBounds> {
            self.core.cells()
        }

        /// The expected hypervolume improvement of the cached pending points
        /// (non-incremental NEHVI; BoTorch's `_prev_nehvi`).
        pub fn prev_nehvi(&self) -> f64 {
            self.core.prev_nehvi()
        }

        /// The hypervolume of the cached points under every MC sample
        /// (BoTorch's `_hypervolumes`).
        pub fn hypervolumes(&self) -> Vec<f64> {
            self.core.hypervolumes()
        }

        /// The hypervolume of the observed baseline under every MC sample
        /// (non-incremental NEHVI; BoTorch's `_baseline_hvs`).
        pub fn baseline_hvs(&self) -> Option<Vec<f64>> {
            self.core.baseline_hvs()
        }

        /// The base samples of the cached points (`S × N·m`, BoTorch's
        /// `base_sampler.base_samples`).
        pub fn base_samples(&self) -> Option<crate::Matrix> {
            self.core.base_samples()
        }

        /// The sampler.
        pub fn sampler(&self) -> &std::sync::Arc<crate::sampling::SobolQMCNormalSampler> {
            &self.core.mc.sampler
        }
    };
}

pub(crate) use nehvi_accessors;
