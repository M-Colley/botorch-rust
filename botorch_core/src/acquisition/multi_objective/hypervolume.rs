//! The Monte-Carlo hypervolume improvement of a candidate set, computed
//! sample by sample with the inclusion–exclusion principle over the subsets
//! of the `q` candidates (Daulton et al., 2020), in linear space (qEHVI) or in
//! log space with fat-tailed smooth approximations (qLogEHVI; Ament et al.,
//! 2023), together with exact reverse-mode derivatives.
//!
//! For objective values `y_p` (`p = 1..q`) of one MC sample and a box
//! decomposition of the non-dominated space into cells `[l_k, u_k]`, the
//! hypervolume improvement is
//!
//! ```text
//! HVI = Σ_k Σ_{∅≠J⊆{1..q}} (-1)^{|J|+1} Π_j max(min(min_{p∈J} y_pj, u_kj) - l_kj, 0) · Π_{p∈J} w_p
//! ```
//!
//! with the smoothed feasibility `w_p` of outcome constraints (BoTorch's
//! `_compute_qehvi`). The log version (`_compute_log_qehvi`) replaces the
//! clamp by `log_fatplus(y - l, τ_relu)` (or `log_softplus`), the minima by
//! `fatmin` (or `smooth_amin`) with temperature `τ_max`, the inner minimum
//! with the upper bound by a smooth minimum with `log(min(u, 1e10) - l)`,
//! the sums by `logsumexp`, the alternating signs by `logdiffexp` of the
//! even- and odd-sized terms, and the mean over samples by `logmeanexp`.
//!
//! The computations follow BoTorch's tensor expressions element by element
//! (including its conventions for infinities, which make the empty cells that
//! pad per-sample decompositions contribute `0`, or `-inf` in log space, with
//! zero gradient). Unlike BoTorch, the smooth log-improvement of a point is
//! computed once per cell instead of once per subset containing it, as in
//! BoTorch's fused C++ kernel; samples are processed in parallel when there is
//! enough work.

use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::acquisition::mc_objective::McSamples;
use crate::utils::safe_math::{
    fatmax_grad_into, log1mexp, log1mexp_grad, log_fatplus_with_grad, log_softplus_with_grad,
    logsumexp_with_grad, smooth_amax_with_grad, ALPHA,
};
use crate::Matrix;

/// The upper cell bounds are clamped at this value before taking the log of
/// the cell lengths (BoTorch's `clamp_max(1e10)` in double precision).
pub(crate) const MAX_CELL_UPPER: f64 = 1e10;

/// Below this amount of work (samples × cells × subset elements × outcomes)
/// the samples are processed sequentially.
const PAR_WORK: usize = 1 << 15;

/// All `2^q - 1` non-empty subsets of `{0, …, q-1}` grouped by size
/// (BoTorch's `compute_subset_indices(q)`: entry `i - 1` holds the subsets
/// of size `i` in the order of `itertools.combinations`).
pub fn compute_subset_indices(q: usize) -> Vec<Vec<Vec<usize>>> {
    fn combinations(q: usize, i: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        if i > q {
            return out;
        }
        let mut idx: Vec<usize> = (0..i).collect();
        loop {
            out.push(idx.clone());
            // The rightmost position that can still be incremented.
            let mut k = i;
            while k > 0 && idx[k - 1] == k - 1 + q - i {
                k -= 1;
            }
            if k == 0 {
                return out;
            }
            idx[k - 1] += 1;
            for j in k..i {
                idx[j] = idx[j - 1] + 1;
            }
        }
    }
    (1..=q).map(|i| combinations(q, i)).collect()
}

/// The subsets of the candidates of the last evaluated `q` (BoTorch's
/// `SubsetIndexCachingMixin`): recomputed only when `q` changes.
#[derive(Debug, Default)]
pub struct SubsetIndexCache {
    cache: Mutex<Option<(usize, Arc<Subsets>)>>,
}

impl SubsetIndexCache {
    /// The subsets of `{0, …, q_out - 1}` (cached).
    pub fn get(&self, q_out: usize) -> Arc<Subsets> {
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((q, s)) = c.as_ref() {
            if *q == q_out {
                return Arc::clone(s);
            }
        }
        let s = Arc::new(Subsets::new(q_out));
        *c = Some((q_out, Arc::clone(&s)));
        s
    }

    /// Forgets the cached subsets (BoTorch resets `q_out` when pending points
    /// change).
    pub fn clear(&self) {
        *self.cache.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// The non-empty subsets of `{0, …, q-1}` flattened in BoTorch's order: by
/// size, then lexicographically.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subsets {
    q: usize,
    /// Members of all subsets, concatenated.
    members: Vec<usize>,
    /// Start of every subset in `members` (and the total length at the end).
    starts: Vec<usize>,
    /// Index of the first subset of every size `i = 1..=q` (and the total
    /// number of subsets at the end).
    size_starts: Vec<usize>,
}

impl Subsets {
    /// The subsets of `{0, …, q-1}`.
    pub fn new(q: usize) -> Self {
        let mut members = Vec::new();
        let mut starts = vec![0];
        let mut size_starts = vec![0];
        for by_size in compute_subset_indices(q) {
            for s in by_size {
                members.extend(s);
                starts.push(members.len());
            }
            size_starts.push(starts.len() - 1);
        }
        Self {
            q,
            members,
            starts,
            size_starts,
        }
    }

    /// The number of candidates `q`.
    pub fn q(&self) -> usize {
        self.q
    }

    /// The number of subsets, `2^q - 1`.
    pub fn len(&self) -> usize {
        self.starts.len() - 1
    }

    /// Whether there is no subset (`q = 0`).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Subset `g`.
    pub fn subset(&self, g: usize) -> &[usize] {
        &self.members[self.starts[g]..self.starts[g + 1]]
    }

    /// The indices of the subsets of size `i` (`1 ≤ i ≤ q`).
    pub fn of_size(&self, i: usize) -> std::ops::Range<usize> {
        self.size_starts[i - 1]..self.size_starts[i]
    }

    /// The total number of subset members, `Σ_J |J| = q 2^(q-1)`.
    pub fn total_members(&self) -> usize {
        self.members.len()
    }
}

/// The hypercell bounds of the box decompositions used by the MC
/// hypervolume acquisition functions: BoTorch's `cell_lower_bounds` and
/// `cell_upper_bounds`, either one decomposition shared by all MC samples
/// (`num_cells × m`, qEHVI) or one per MC sample (`S × num_cells × m`,
/// qNEHVI), row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct HypercellBounds {
    batches: usize,
    num_cells: usize,
    m: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// `log(min(u, 1e10) - l)` (BoTorch's `_log_cell_lengths`).
    log_lengths: Vec<f64>,
}

impl HypercellBounds {
    /// Bounds of `batches` decompositions of `num_cells` cells each over `m`
    /// outcomes (`batches = 1`: shared by all samples).
    ///
    /// # Errors
    /// [`crate::BotorchError::DimensionMismatch`] if the lengths do not match.
    pub fn new(
        batches: usize,
        num_cells: usize,
        m: usize,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> crate::Result<Self> {
        let len = batches * num_cells * m;
        if lower.len() != len || upper.len() != len || batches == 0 || m == 0 {
            return Err(crate::BotorchError::DimensionMismatch(format!(
                "cell bounds of {batches} x {num_cells} x {m} need {len} values each, got {} and {}",
                lower.len(),
                upper.len()
            )));
        }
        let log_lengths = lower
            .iter()
            .zip(&upper)
            .map(|(&l, &u)| (u.min(MAX_CELL_UPPER) - l).ln())
            .collect();
        Ok(Self {
            batches,
            num_cells,
            m,
            lower,
            upper,
            log_lengths,
        })
    }

    /// From [`CellBounds`](crate::utils::multi_objective::box_decompositions::CellBounds)
    /// of an unbatched decomposition (shared by all samples) or of a batch of
    /// decompositions (one per sample).
    ///
    /// # Errors
    /// [`crate::BotorchError::DimensionMismatch`] for more than one batch
    /// dimension.
    pub fn from_cell_bounds(
        bounds: &crate::utils::multi_objective::box_decompositions::CellBounds,
    ) -> crate::Result<Self> {
        if bounds.batch_shape().len() > 1 {
            return Err(crate::BotorchError::DimensionMismatch(format!(
                "cell bounds with batch shape {:?} are not supported",
                bounds.batch_shape()
            )));
        }
        Self::new(
            bounds.batch_size(),
            bounds.num_cells(),
            bounds.num_outcomes(),
            bounds.lower().to_vec(),
            bounds.upper().to_vec(),
        )
    }

    /// Number of decompositions (1 if shared by all samples).
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// Number of cells per decomposition.
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// Number of outcomes `m`.
    pub fn num_outcomes(&self) -> usize {
        self.m
    }

    /// The lower vertices (row-major `batches × num_cells × m`).
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// The upper vertices (row-major `batches × num_cells × m`).
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    fn block(&self, b: usize) -> (&[f64], &[f64], &[f64]) {
        let len = self.num_cells * self.m;
        let r = b * len..(b + 1) * len;
        (
            &self.lower[r.clone()],
            &self.upper[r.clone()],
            &self.log_lengths[r],
        )
    }

    /// The cells used for MC sample `s` of `num_samples`.
    fn for_sample(&self, s: usize) -> (&[f64], &[f64], &[f64]) {
        self.block(if self.batches == 1 { 0 } else { s })
    }
}

/// Settings of the log-space computation (qLogEHVI).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogSettings {
    /// Fat-tailed approximations (`log_fatplus`, `fatmin`) instead of
    /// `log_softplus` / `smooth_amin`.
    pub fat: bool,
    /// Temperature of the smooth ReLU.
    pub tau_relu: f64,
    /// Temperature of the smooth minima.
    pub tau_max: f64,
}

/// The reduced hypervolume improvement of all samples and, if requested,
/// the adjoints of the objective values (`(m·q) × S`, the layout of
/// [`McSamples`]) and of the (log-)feasibility weights (`q × S`).
pub struct Reduced {
    /// The value (mean, or `logmeanexp` for the log version).
    pub value: f64,
    /// The adjoint of the objective values.
    pub obj_bar: Option<Matrix>,
    /// The adjoint of the feasibility weights.
    pub feas_bar: Option<Matrix>,
}

/// The result of one MC sample: the value and its gradients with respect to
/// the objective values (`m·q`, index `j·q + p`) and the weights (`q`).
struct SampleResult {
    value: f64,
    gy: Vec<f64>,
    gw: Vec<f64>,
}

fn run_samples<F>(num_samples: usize, work: usize, f: F) -> Vec<SampleResult>
where
    F: Fn(usize) -> SampleResult + Sync + Send,
{
    if work >= PAR_WORK && num_samples > 1 {
        (0..num_samples).into_par_iter().map(f).collect()
    } else {
        (0..num_samples).map(f).collect()
    }
}

/// `torch.minimum(a, b)` and its derivative with respect to `a` (ties split
/// the gradient evenly, as in PyTorch).
#[inline]
fn minimum_grad_a(a: f64, b: f64) -> (f64, f64) {
    if a < b {
        (a, 1.0)
    } else if a == b {
        (a, 0.5)
    } else if b < a {
        (b, 0.0)
    } else {
        // NaN propagates (torch.minimum).
        (f64::NAN, if a.is_nan() { 1.0 } else { 0.0 })
    }
}

/// Mean over the MC samples of the hypervolume improvement (BoTorch's
/// `_compute_qehvi`): `obj` holds the objective values of `q` points and `m`
/// outcomes per sample, `feas` the smoothed feasibility weights (`q × S`) of
/// outcome constraints, `cells` the box decomposition(s).
pub fn qehvi(
    obj: &McSamples,
    feas: Option<&Matrix>,
    cells: &HypercellBounds,
    subsets: &Subsets,
    with_grad: bool,
) -> Reduced {
    let (q, m, ns) = (obj.num_points(), obj.num_outputs(), obj.num_samples());
    debug_assert_eq!(subsets.q(), q);
    debug_assert_eq!(cells.num_outcomes(), m);
    let nc = cells.num_cells();
    let data = obj.data();
    let work = ns * nc * subsets.total_members() * m;
    let results = run_samples(ns, work, |s| {
        let y = data.column(s);
        let (lower, upper, _) = cells.for_sample(s);
        let w = feas.map(|f| f.column(s).iter().copied().collect::<Vec<f64>>());
        qehvi_sample(
            y.as_slice(),
            q,
            m,
            w.as_deref(),
            lower,
            upper,
            nc,
            subsets,
            with_grad,
        )
    });
    // `areas_per_segment.sum(dim=-1).mean(dim=0)`.
    let value = results.iter().map(|r| r.value).sum::<f64>() / ns as f64;
    if !with_grad {
        return Reduced {
            value,
            obj_bar: None,
            feas_bar: None,
        };
    }
    let scale = 1.0 / ns as f64;
    let mut obj_bar = Matrix::zeros(m * q, ns);
    let mut feas_bar = feas.map(|_| Matrix::zeros(q, ns));
    for (s, r) in results.iter().enumerate() {
        for (i, g) in r.gy.iter().enumerate() {
            obj_bar[(i, s)] = g * scale;
        }
        if let Some(fb) = feas_bar.as_mut() {
            for (p, g) in r.gw.iter().enumerate() {
                fb[(p, s)] = g * scale;
            }
        }
    }
    Reduced {
        value,
        obj_bar: Some(obj_bar),
        feas_bar,
    }
}

/// The hypervolume improvement of one sample (`y[j·q + p]`) and its
/// gradients.
#[allow(clippy::too_many_arguments)]
fn qehvi_sample(
    y: &[f64],
    q: usize,
    m: usize,
    w: Option<&[f64]>,
    lower: &[f64],
    upper: &[f64],
    nc: usize,
    subsets: &Subsets,
    with_grad: bool,
) -> SampleResult {
    let mut seg = vec![0.0; nc];
    let mut areas_i = vec![0.0; nc];
    let mut overlap = vec![0.0; m];
    let mut arg = vec![0usize; m];
    let mut len = vec![0.0; m];
    let mut dlen = vec![0.0; m];
    let mut lgrad = vec![0.0; m];
    let mut d_overlap = vec![0.0; m];
    let mut gy = vec![0.0; if with_grad { m * q } else { 0 }];
    let mut gw = vec![0.0; if with_grad && w.is_some() { q } else { 0 }];
    for i in 1..=q {
        let sign = if i % 2 == 1 { 1.0 } else { -1.0 };
        areas_i.iter_mut().for_each(|a| *a = 0.0);
        for g in subsets.of_size(i) {
            let set = subsets.subset(g);
            // Component-wise minimum over the subset (gradient to the first
            // minimizer, as torch.min(dim) does).
            for j in 0..m {
                let mut best = y[j * q + set[0]];
                let mut a = set[0];
                for &p in &set[1..] {
                    let v = y[j * q + p];
                    if v < best || (v.is_nan() && !best.is_nan()) {
                        best = v;
                        a = p;
                    }
                }
                overlap[j] = best;
                arg[j] = a;
            }
            let wprod = w.map_or(1.0, |w| set.iter().fold(1.0, |acc, &p| acc * w[p]));
            d_overlap.iter_mut().for_each(|d| *d = 0.0);
            let mut sum_area_nofeas = 0.0;
            for k in 0..nc {
                let l = &lower[k * m..(k + 1) * m];
                let u = &upper[k * m..(k + 1) * m];
                for j in 0..m {
                    let (o, go) = minimum_grad_a(overlap[j], u[j]);
                    let raw = o - l[j];
                    // clamp_min(0): the gradient passes where raw >= 0.
                    len[j] = if raw < 0.0 { 0.0 } else { raw };
                    dlen[j] = if raw >= 0.0 { go } else { 0.0 };
                }
                let area = len.iter().fold(1.0, |acc, v| acc * v);
                areas_i[k] += area * wprod;
                if with_grad {
                    sum_area_nofeas += area;
                    if dlen.iter().any(|&d| d != 0.0) {
                        // Products of the other lengths (prefix / suffix).
                        let mut pre = 1.0;
                        for j in 0..m {
                            lgrad[j] = pre;
                            pre *= len[j];
                        }
                        let mut suf = 1.0;
                        for j in (0..m).rev() {
                            lgrad[j] *= suf;
                            suf *= len[j];
                        }
                        for j in 0..m {
                            d_overlap[j] += sign * wprod * lgrad[j] * dlen[j];
                        }
                    }
                }
            }
            if with_grad {
                for j in 0..m {
                    gy[j * q + arg[j]] += d_overlap[j];
                }
                if let Some(w) = w {
                    for (e, &p) in set.iter().enumerate() {
                        let others = set
                            .iter()
                            .enumerate()
                            .filter(|&(f, _)| f != e)
                            .fold(1.0, |acc, (_, &r)| acc * w[r]);
                        gw[p] += sign * sum_area_nofeas * others;
                    }
                }
            }
        }
        for (s, a) in seg.iter_mut().zip(&areas_i) {
            *s += sign * a;
        }
    }
    SampleResult {
        value: seg.iter().sum(),
        gy,
        gw,
    }
}

/// `logmeanexp` over the samples of the log hypervolume improvement (BoTorch's
/// `_compute_log_qehvi`); `log_feas` are the log-feasibility weights
/// (`q × S`).
pub fn log_qehvi(
    obj: &McSamples,
    log_feas: Option<&Matrix>,
    cells: &HypercellBounds,
    subsets: &Subsets,
    settings: LogSettings,
    with_grad: bool,
) -> Reduced {
    let (q, m, ns) = (obj.num_points(), obj.num_outputs(), obj.num_samples());
    debug_assert_eq!(subsets.q(), q);
    debug_assert_eq!(cells.num_outcomes(), m);
    let nc = cells.num_cells();
    let data = obj.data();
    let work = ns * nc * (subsets.total_members() + subsets.len()) * m;
    let results = run_samples(ns, work, |s| {
        let y = data.column(s);
        let (lower, _, log_lengths) = cells.for_sample(s);
        let lw = log_feas.map(|f| f.column(s).iter().copied().collect::<Vec<f64>>());
        log_qehvi_sample(
            y.as_slice(),
            q,
            m,
            lw.as_deref(),
            lower,
            log_lengths,
            nc,
            subsets,
            settings,
            with_grad,
        )
    });
    let values: Vec<f64> = results.iter().map(|r| r.value).collect();
    let (lse, weights) = logsumexp_with_grad(&values);
    let value = lse - (ns as f64).ln();
    if !with_grad {
        return Reduced {
            value,
            obj_bar: None,
            feas_bar: None,
        };
    }
    let mut obj_bar = Matrix::zeros(m * q, ns);
    let mut feas_bar = log_feas.map(|_| Matrix::zeros(q, ns));
    for (s, r) in results.iter().enumerate() {
        let ws = weights[s];
        for (i, g) in r.gy.iter().enumerate() {
            obj_bar[(i, s)] = g * ws;
        }
        if let Some(fb) = feas_bar.as_mut() {
            for (p, g) in r.gw.iter().enumerate() {
                fb[(p, s)] = g * ws;
            }
        }
    }
    Reduced {
        value,
        obj_bar: Some(obj_bar),
        feas_bar,
    }
}

/// `logsumexp` of two values and its gradient (BoTorch's `logplusexp`).
#[inline]
fn logplusexp_grad(a: f64, b: f64) -> (f64, f64, f64) {
    let (v, g) = logsumexp_with_grad(&[a, b]);
    (v, g[0], g[1])
}

/// BoTorch's `logdiffexp(log_a, log_b) = log_b + log1mexp(log_a - shift)`
/// with `shift = log_b` (`0` if `log_b = -inf`) and its gradients.
#[inline]
fn logdiffexp_grad(log_a: f64, log_b: f64) -> (f64, f64, f64) {
    let is_inf = log_b == f64::NEG_INFINITY;
    let shift = if is_inf { 0.0 } else { log_b };
    let x = log_a - shift;
    let value = log_b + log1mexp(x);
    let g = log1mexp_grad(x);
    (value, g, 1.0 - if is_inf { 0.0 } else { g })
}

/// The smooth minimum of `x` (`fatmin` or `smooth_amin` with temperature
/// `tau`) and its gradient (written to `grad`).
#[inline]
fn smooth_min_grad(x: &[f64], fat: bool, tau: f64, scratch: &mut [f64], grad: &mut [f64]) -> f64 {
    for (s, &v) in scratch.iter_mut().zip(x) {
        *s = -v;
    }
    if fat {
        // fatmin(x) = -fatmax(-x); its gradient is fatmax's gradient at -x.
        -fatmax_grad_into(scratch, tau, ALPHA, grad)
    } else {
        // smooth_amin(x) = -smooth_amax(-x).
        let (v, g) = smooth_amax_with_grad(scratch, tau);
        grad.copy_from_slice(&g);
        -v
    }
}

/// The log hypervolume improvement of one sample and its gradients.
#[allow(clippy::too_many_arguments)]
fn log_qehvi_sample(
    y: &[f64],
    q: usize,
    m: usize,
    log_w: Option<&[f64]>,
    lower: &[f64],
    log_lengths: &[f64],
    nc: usize,
    subsets: &Subsets,
    st: LogSettings,
    with_grad: bool,
) -> SampleResult {
    let nsub = subsets.len();
    // 1) The smooth log-improvement of every point over every cell's lower
    //    bound, L[k, p, j], and its derivative.
    let mut lim = vec![0.0; nc * q * m];
    let mut dlim = vec![0.0; if with_grad { nc * q * m } else { 0 }];
    for k in 0..nc {
        for p in 0..q {
            for j in 0..m {
                let z = y[j * q + p] - lower[k * m + j];
                let (v, d) = if st.fat {
                    log_fatplus_with_grad(z, st.tau_relu)
                } else {
                    log_softplus_with_grad(z, st.tau_relu)
                };
                let idx = (k * q + p) * m + j;
                lim[idx] = v;
                if with_grad {
                    dlim[idx] = d;
                }
            }
        }
    }
    // 2) Log areas per cell and subset; local derivatives of the log area of
    //    (k, J) with respect to L[k, p, j] for p ∈ J (coefficients `coef`,
    //    one per cell, subset member and outcome).
    let total = subsets.total_members();
    let mut log_area = vec![0.0; nc * nsub];
    let mut coef = vec![0.0; if with_grad { nc * total * m } else { 0 }];
    let mut vals = vec![0.0; q];
    let mut scratch = vec![0.0; q.max(2)];
    let mut gmin = vec![0.0; q.max(2)];
    let mut pair = [0.0; 2];
    let mut gpair = [0.0; 2];
    for g in 0..nsub {
        let set = subsets.subset(g);
        let off = subsets.starts[g];
        let w_sum = log_w.map_or(0.0, |lw| set.iter().map(|&p| lw[p]).sum::<f64>());
        for k in 0..nc {
            let mut acc = 0.0;
            for j in 0..m {
                // min over the subset members.
                let a = if set.len() == 1 {
                    gmin[0] = 1.0;
                    lim[(k * q + set[0]) * m + j]
                } else {
                    for (v, &p) in vals.iter_mut().zip(set) {
                        *v = lim[(k * q + p) * m + j];
                    }
                    let e = set.len();
                    smooth_min_grad(
                        &vals[..e],
                        st.fat,
                        st.tau_max,
                        &mut scratch[..e],
                        &mut gmin[..e],
                    )
                };
                // smooth minimum with the log cell length.
                pair[0] = a;
                pair[1] = log_lengths[k * m + j];
                let b = smooth_min_grad(&pair, st.fat, st.tau_max, &mut scratch[..2], &mut gpair);
                acc += b;
                if with_grad {
                    for e in 0..set.len() {
                        coef[(k * total + off + e) * m + j] = gpair[0] * gmin[e];
                    }
                }
            }
            log_area[k * nsub + g] = acc + w_sum;
        }
    }
    // 3) - 6) Per cell: logsumexp over the subsets of every size, the chains
    //    of logplusexp for odd and even sizes, logdiffexp; then logsumexp
    //    over the cells.
    let mut d_cells = vec![0.0; nc];
    // Per cell and size: (weights of the subsets, chain weights).
    let mut lse_w = vec![0.0; if with_grad { nc * nsub } else { 0 }];
    let mut chain_w = vec![(0.0, 0.0); if with_grad { nc * q } else { 0 }];
    let mut diff_w = vec![(0.0, 0.0); if with_grad { nc } else { 0 }];
    for k in 0..nc {
        let la = &log_area[k * nsub..(k + 1) * nsub];
        let mut acc = [f64::NEG_INFINITY; 2];
        for i in 1..=q {
            let r = subsets.of_size(i);
            let (lse, w) = logsumexp_with_grad(&la[r.clone()]);
            if with_grad {
                lse_w[k * nsub + r.start..k * nsub + r.end].copy_from_slice(&w);
            }
            let par = i % 2;
            let (v, w0, w1) = logplusexp_grad(acc[par], lse);
            acc[par] = v;
            if with_grad {
                chain_w[k * q + (i - 1)] = (w0, w1);
            }
        }
        let (d, ga, gb) = logdiffexp_grad(acc[0], acc[1]);
        d_cells[k] = d;
        if with_grad {
            diff_w[k] = (ga, gb);
        }
    }
    let (value, wk) = logsumexp_with_grad(&d_cells);
    if !with_grad {
        return SampleResult {
            value,
            gy: Vec::new(),
            gw: Vec::new(),
        };
    }
    // Backward.
    let mut gy = vec![0.0; m * q];
    let mut gw = vec![0.0; if log_w.is_some() { q } else { 0 }];
    let mut dl = vec![0.0; q * m];
    for k in 0..nc {
        let dd = wk[k];
        if dd == 0.0 {
            continue;
        }
        let (ga, gb) = diff_w[k];
        let mut dacc = [dd * ga, dd * gb];
        dl.iter_mut().for_each(|v| *v = 0.0);
        for i in (1..=q).rev() {
            let par = i % 2;
            let (w0, w1) = chain_w[k * q + (i - 1)];
            let dlse = dacc[par] * w1;
            dacc[par] *= w0;
            if dlse == 0.0 {
                continue;
            }
            for g in subsets.of_size(i) {
                let dla = dlse * lse_w[k * nsub + g];
                if dla == 0.0 {
                    continue;
                }
                let set = subsets.subset(g);
                let off = subsets.starts[g];
                if log_w.is_some() {
                    for &p in set {
                        gw[p] += dla;
                    }
                }
                for (e, &p) in set.iter().enumerate() {
                    let c = &coef[(k * total + off + e) * m..(k * total + off + e + 1) * m];
                    for j in 0..m {
                        dl[p * m + j] += dla * c[j];
                    }
                }
            }
        }
        for p in 0..q {
            for j in 0..m {
                let d = dl[p * m + j];
                if d != 0.0 {
                    gy[j * q + p] += d * dlim[(k * q + p) * m + j];
                }
            }
        }
    }
    SampleResult { value, gy, gw }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::multi_objective::box_decompositions::{
        BoxDecomposition, NondominatedPartitioning,
    };

    #[test]
    fn subsets_follow_itertools_order() {
        let s = compute_subset_indices(3);
        assert_eq!(s[0], vec![vec![0], vec![1], vec![2]]);
        assert_eq!(s[1], vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
        assert_eq!(s[2], vec![vec![0, 1, 2]]);
        let s4 = compute_subset_indices(4);
        assert_eq!(s4[1].len(), 6);
        assert_eq!(s4[1][3], vec![1, 2]);
        assert_eq!(
            s4[2],
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]]
        );
        assert!(compute_subset_indices(0).is_empty());
        let sub = Subsets::new(3);
        assert_eq!(sub.len(), 7);
        assert_eq!(sub.total_members(), 12);
        assert_eq!(sub.subset(4), &[0, 2]);
        assert_eq!(sub.of_size(2), 3..6);
        let cache = SubsetIndexCache::default();
        let a = cache.get(2);
        assert!(Arc::ptr_eq(&a, &cache.get(2)));
        assert_eq!(cache.get(3).len(), 7);
    }

    fn cells() -> HypercellBounds {
        let y = Matrix::from_row_slice(3, 2, &[1.0, 0.2, 0.6, 0.6, 0.1, 1.0]);
        let p = NondominatedPartitioning::new(&[0.0, 0.0], Some(&y), 0.0).unwrap();
        HypercellBounds::from_cell_bounds(p.get_hypercell_bounds().unwrap()).unwrap()
    }

    fn sample_obj(q: usize, s: usize) -> McSamples {
        let rm: Vec<f64> = (0..s * q * 2)
            .map(|i| 0.4 + 0.9 * (((i * 37) % 23) as f64 / 23.0 - 0.3))
            .collect();
        McSamples::from_row_major(&rm, s, q, 2).unwrap()
    }

    /// Exact hypervolume improvement by inclusion–exclusion is the
    /// hypervolume gain of adding the points to the front.
    #[test]
    fn qehvi_is_the_hypervolume_improvement() {
        use crate::utils::multi_objective::hypervolume::Hypervolume;
        let c = cells();
        let front = Matrix::from_row_slice(3, 2, &[1.0, 0.2, 0.6, 0.6, 0.1, 1.0]);
        let hv = Hypervolume::new(&[0.0, 0.0]);
        let base = hv.compute(&front).unwrap();
        for q in 1..=3 {
            let obj = sample_obj(q, 5);
            let sub = Subsets::new(q);
            let r = qehvi(&obj, None, &c, &sub, false);
            let mut expected = 0.0;
            for s in 0..5 {
                let mut all = front.clone().insert_rows(3, q, 0.0);
                for p in 0..q {
                    all[(3 + p, 0)] = obj.value(s, p, 0);
                    all[(3 + p, 1)] = obj.value(s, p, 1);
                }
                expected += hv.compute(&all).unwrap() - base;
            }
            expected /= 5.0;
            assert!(
                (r.value - expected).abs() < 1e-12,
                "q={q}: {} vs {expected}",
                r.value
            );
        }
    }

    fn fd_check<F: Fn(&McSamples, Option<&Matrix>) -> f64>(
        obj: &McSamples,
        w: Option<&Matrix>,
        gy: &Matrix,
        gw: Option<&Matrix>,
        f: F,
    ) {
        let (q, m, s) = (obj.num_points(), obj.num_outputs(), obj.num_samples());
        let rm = obj.to_row_major();
        let h = 1e-7;
        for i in 0..rm.len() {
            let eval = |d: f64| {
                let mut v = rm.clone();
                v[i] += d;
                f(&McSamples::from_row_major(&v, s, q, m).unwrap(), w)
            };
            let fd = (eval(h) - eval(-h)) / (2.0 * h);
            let (c, rest) = (i / (q * m), i % (q * m));
            let (p, t) = (rest / m, rest % m);
            let an = gy[(t * q + p, c)];
            assert!(
                (fd - an).abs() < 1e-5 * fd.abs().max(1.0),
                "y[{i}]: {an} vs {fd}"
            );
        }
        if let (Some(w), Some(gw)) = (w, gw) {
            for p in 0..q {
                for c in 0..s {
                    let eval = |d: f64| {
                        let mut wp = w.clone();
                        wp[(p, c)] += d;
                        f(obj, Some(&wp))
                    };
                    let fd = (eval(h) - eval(-h)) / (2.0 * h);
                    let an = gw[(p, c)];
                    assert!(
                        (fd - an).abs() < 1e-5 * fd.abs().max(1.0),
                        "w[{p},{c}]: {an} vs {fd}"
                    );
                }
            }
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        let c = cells();
        for q in 1..=3 {
            let obj = sample_obj(q, 4);
            let sub = Subsets::new(q);
            let w = Matrix::from_fn(q, 4, |p, s| 0.3 + 0.15 * p as f64 + 0.1 * s as f64);
            let r = qehvi(&obj, Some(&w), &c, &sub, true);
            fd_check(
                &obj,
                Some(&w),
                r.obj_bar.as_ref().unwrap(),
                r.feas_bar.as_ref(),
                |o, w| qehvi(o, w, &c, &sub, false).value,
            );
            for fat in [true, false] {
                let st = LogSettings {
                    fat,
                    tau_relu: 1e-2,
                    tau_max: 1e-2,
                };
                let lw = w.map(f64::ln);
                let r = log_qehvi(&obj, Some(&lw), &c, &sub, st, true);
                fd_check(
                    &obj,
                    Some(&lw),
                    r.obj_bar.as_ref().unwrap(),
                    r.feas_bar.as_ref(),
                    |o, w| log_qehvi(o, w, &c, &sub, st, false).value,
                );
            }
        }
    }

    /// As the temperatures go to zero, qLogEHVI approaches log(qEHVI).
    #[test]
    fn log_version_approaches_log_qehvi() {
        let c = cells();
        for q in 1..=3 {
            let obj = sample_obj(q, 6);
            let sub = Subsets::new(q);
            let exact = qehvi(&obj, None, &c, &sub, false).value.ln();
            let mut prev = f64::INFINITY;
            for tau in [1e-2, 1e-3, 1e-4, 1e-6] {
                let st = LogSettings {
                    fat: true,
                    tau_relu: tau,
                    tau_max: tau,
                };
                let v = log_qehvi(&obj, None, &c, &sub, st, false).value;
                let err = (v - exact).abs();
                assert!(err <= prev + 1e-12, "q={q} tau={tau}: {err} > {prev}");
                prev = err;
            }
            assert!(prev < 1e-4, "q={q}: {prev}");
        }
    }

    /// Empty (zero-padded) cells contribute nothing, also in log space.
    #[test]
    fn padded_cells_contribute_nothing() {
        let c = cells();
        let k = c.num_cells();
        let mut lower = c.lower().to_vec();
        let mut upper = c.upper().to_vec();
        lower.extend([0.0; 4]);
        upper.extend([0.0; 4]);
        let padded = HypercellBounds::new(1, k + 2, 2, lower, upper).unwrap();
        let obj = sample_obj(2, 3);
        let sub = Subsets::new(2);
        let a = qehvi(&obj, None, &c, &sub, true);
        let b = qehvi(&obj, None, &padded, &sub, true);
        assert!((a.value - b.value).abs() < 1e-15);
        assert_eq!(a.obj_bar, b.obj_bar);
        let st = LogSettings {
            fat: true,
            tau_relu: 1e-6,
            tau_max: 1e-2,
        };
        let a = log_qehvi(&obj, None, &c, &sub, st, true);
        let b = log_qehvi(&obj, None, &padded, &sub, st, true);
        assert!((a.value - b.value).abs() < 1e-14);
        let (ga, gb) = (a.obj_bar.unwrap(), b.obj_bar.unwrap());
        assert!((ga - gb).amax() < 1e-14);
    }
}
