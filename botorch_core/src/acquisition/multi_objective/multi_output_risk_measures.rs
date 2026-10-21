//! Multi-output risk measures as Monte-Carlo objectives
//! (`botorch.acquisition.multi_objective.multi_output_risk_measures`):
//! [`MultiOutputExpectation`], [`IndependentCVaR`], [`IndependentVaR`] and
//! [`MultiOutputWorstCase`] act on every output independently; [`MVaR`] is the
//! multivariate Value-at-Risk of Prékopa (2012) and Cousin & Di Bernardino
//! (2013), computed from the joint distribution of the outputs; [`Mars`]
//! approximates MVaR by the VaR of random Chebyshev scalarizations (Daulton
//! et al., *Robust Multi-Objective Bayesian Optimization Under Input Noise*,
//! ICML 2022).
//!
//! As for the single-output measures ([`crate::acquisition::risk_measures`]),
//! the samples of `q·n_w` points consist of blocks of `n_w` points per
//! candidate, and a multi-output preprocessing objective (BoTorch's
//! `preprocessing_function`, called without `X`; default: all outputs)
//! selects and aligns the outputs first.
//!
//! # MVaR
//!
//! For the `n_w` samples `y_i` of a candidate, MVaR is the set of maximal
//! points `z` of the grid spanned by the samples with `#{i: y_i ≥ z} ≥ k`,
//! `k = ceil(alpha·n_w)`. Like BoTorch, the grid is spanned by the
//! `n_w - k + 1` smallest values of every output (larger values cannot be
//! MVaR points), in BoTorch's order: the grid of the sorted values with
//! duplicates (`use_counting=False`, the default) or of the distinct values
//! in decreasing order (`use_counting=True`). The dominance counts are
//! computed with a histogram and suffix sums (the counting implementation's
//! idea) rather than by comparing every grid point with every sample.
//!
//! With ties in every output, an MVaR point can be dominated by more than
//! `k` samples; BoTorch main keeps only the points dominated by exactly `k`
//! samples (or by the smallest larger count) and misses them. This module
//! implements the fix of meta-pytorch/botorch#3392 ("Include tied MVaR points
//! with a larger CDF in MVaR"): a grid point is an MVaR point if it is
//! dominated by at least `k` samples and the next grid value along every
//! output is dominated by fewer. Without ties both agree.
//!
//! The MVaR points are padded to the largest set size over all candidates
//! and samples by repeating the last point (to `n_w` with `pad_to_n_w`), so
//! that the values have `q·k'` points per sample; `expectation=True`
//! returns the mean of the set instead. MVaR is piecewise constant in the
//! samples; BoTorch's gradient maps every component of an MVaR point to the
//! samples that attain it (`make_differentiable`, averaged over ties), which
//! is what the pullback computes.

use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::mc_objective::{McObjective, McSamples, SamplePullback, SampleValues};
use crate::acquisition::multi_objective::objective::{McMultiOutputObjective, MultiOutputValues};
use crate::acquisition::risk_measures::{
    alpha_idx, check_alpha, check_n_w, reduce_block, torch_cmp, view_error, RiskKind,
};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::utils::multi_objective::pareto::is_non_dominated;
use crate::Matrix;

/// The prepared samples of a multi-output risk measure: the preprocessed
/// values (`n_pts` points, `m'` outputs) and the pullback of the
/// preprocessing.
fn prepare(
    preprocessing: Option<&Arc<dyn McMultiOutputObjective>>,
    samples: &McSamples,
    n_w: usize,
    with_grad: bool,
) -> Result<(McSamples, Option<SamplePullback>)> {
    let (values, pullback) = match preprocessing {
        None => (samples.clone(), None),
        Some(p) => {
            // BoTorch calls the preprocessing function without `X`.
            let out = p.evaluate(samples, None, with_grad)?;
            if out.values.num_samples() != samples.num_samples() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the preprocessing function returned values for {} samples, expected {}",
                    out.values.num_samples(),
                    samples.num_samples()
                )));
            }
            (out.values, out.pullback)
        }
    };
    if values.num_points() % n_w != 0 {
        return Err(view_error(values.num_points(), n_w));
    }
    Ok((values, pullback))
}

/// Chains the pullback of the preprocessing after the adjoint of the
/// prepared values.
fn finish(
    pre_bar: Matrix,
    pre_pullback: Option<SamplePullback>,
) -> Result<(Matrix, Option<Matrix>)> {
    match pre_pullback {
        Some(pb) => {
            let (sbar, _) = pb(&pre_bar)?;
            Ok((sbar, None))
        }
        None => Ok((pre_bar, None)),
    }
}

/// Independent reduction of the `n_w` values of every candidate and output.
#[derive(Clone, Copy, Debug, PartialEq)]
enum IndependentKind {
    Risk(RiskKind),
    /// `sort(descending=True)[..., alpha_idx:, :].mean()`.
    CVaR(usize),
    /// `sort(descending=True)[..., alpha_idx, :]`.
    VaR(usize),
}

/// The descending (stable) order of `v` (torch's `sort(descending=True)`:
/// NaN first).
fn argsort_desc(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| torch_cmp(v[b], v[a]));
    idx
}

fn reduce_independent(kind: &IndependentKind, v: &[f64], w: &mut [f64]) -> f64 {
    match kind {
        IndependentKind::Risk(k) => reduce_block(k, v, w),
        IndependentKind::CVaR(a) => {
            w.iter_mut().for_each(|x| *x = 0.0);
            let order = argsort_desc(v);
            let k = v.len() - a;
            let mut s = 0.0;
            for &i in &order[*a..] {
                s += v[i];
                w[i] = 1.0 / k as f64;
            }
            s / k as f64
        }
        IndependentKind::VaR(a) => {
            w.iter_mut().for_each(|x| *x = 0.0);
            let i = argsort_desc(v)[*a];
            w[i] = 1.0;
            v[i]
        }
    }
}

/// The shared part of the multi-output risk measures (BoTorch's
/// `MultiOutputRiskMeasureMCObjective`).
#[derive(Clone, Debug)]
struct MoRiskBase {
    n_w: usize,
    preprocessing: Option<Arc<dyn McMultiOutputObjective>>,
}

impl MoRiskBase {
    fn new(n_w: usize, preprocessing: Option<Arc<dyn McMultiOutputObjective>>) -> Result<Self> {
        check_n_w(n_w)?;
        Ok(Self { n_w, preprocessing })
    }

    fn evaluate_independent(
        &self,
        kind: &IndependentKind,
        samples: &McSamples,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        let n_w = self.n_w;
        let (pre, pre_pb) = prepare(self.preprocessing.as_ref(), samples, n_w, with_grad)?;
        let (n, m, s) = (pre.num_points(), pre.num_outputs(), pre.num_samples());
        let q = n / n_w;
        let data = pre.data();
        let mut values = Matrix::zeros(m * q, s);
        let mut weights = if with_grad {
            Matrix::zeros(m * n, s)
        } else {
            Matrix::zeros(0, 0)
        };
        let mut v = vec![0.0; n_w];
        let mut w = vec![0.0; n_w];
        for c in 0..s {
            for t in 0..m {
                for g in 0..q {
                    for (i, vi) in v.iter_mut().enumerate() {
                        *vi = data[(t * n + g * n_w + i, c)];
                    }
                    values[(t * q + g, c)] = reduce_independent(kind, &v, &mut w);
                    if with_grad {
                        for (i, wi) in w.iter().enumerate() {
                            weights[(t * n + g * n_w + i, c)] = *wi;
                        }
                    }
                }
            }
        }
        let pullback: Option<SamplePullback> = with_grad.then(|| {
            Box::new(move |bar: &Matrix| {
                let mut pre_bar = weights;
                for c in 0..s {
                    for t in 0..m {
                        for r in 0..n {
                            pre_bar[(t * n + r, c)] *= bar[(t * q + r / n_w, c)];
                        }
                    }
                }
                finish(pre_bar, pre_pb)
            }) as SamplePullback
        });
        Ok(MultiOutputValues {
            values: McSamples::new(values, q, m)?,
            pullback,
        })
    }
}

macro_rules! mo_risk_common {
    () => {
        /// The number of points per candidate `n_w`.
        pub fn n_w(&self) -> usize {
            self.base.n_w
        }

        /// The preprocessing function (`None`: all outputs).
        pub fn preprocessing_function(&self) -> Option<&Arc<dyn McMultiOutputObjective>> {
            self.base.preprocessing.as_ref()
        }
    };
}

/// The expectation of every output (BoTorch's `MultiOutputExpectation`).
#[derive(Clone, Debug)]
pub struct MultiOutputExpectation {
    base: MoRiskBase,
}

impl MultiOutputExpectation {
    /// `MultiOutputExpectation(n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `n_w = 0`.
    pub fn new(n_w: usize, preprocessing: Option<Arc<dyn McMultiOutputObjective>>) -> Result<Self> {
        Ok(Self {
            base: MoRiskBase::new(n_w, preprocessing)?,
        })
    }

    mo_risk_common!();
}

impl McMultiOutputObjective for MultiOutputExpectation {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        self.base.evaluate_independent(
            &IndependentKind::Risk(RiskKind::Expectation),
            samples,
            with_grad,
        )
    }
}

/// The worst case of every output (BoTorch's `MultiOutputWorstCase`).
#[derive(Clone, Debug)]
pub struct MultiOutputWorstCase {
    base: MoRiskBase,
}

impl MultiOutputWorstCase {
    /// `MultiOutputWorstCase(n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `n_w = 0`.
    pub fn new(n_w: usize, preprocessing: Option<Arc<dyn McMultiOutputObjective>>) -> Result<Self> {
        Ok(Self {
            base: MoRiskBase::new(n_w, preprocessing)?,
        })
    }

    mo_risk_common!();
}

impl McMultiOutputObjective for MultiOutputWorstCase {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        self.base.evaluate_independent(
            &IndependentKind::Risk(RiskKind::WorstCase),
            samples,
            with_grad,
        )
    }
}

/// The CVaR of every output on its own (BoTorch's `IndependentCVaR`); more
/// optimistic than a CVaR of the joint distribution.
#[derive(Clone, Debug)]
pub struct IndependentCVaR {
    base: MoRiskBase,
    alpha: f64,
    alpha_idx: usize,
}

impl IndependentCVaR {
    /// `IndependentCVaR(alpha, n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `alpha ∈ (0, 1]` and
    /// `n_w ≥ 1`.
    pub fn new(
        alpha: f64,
        n_w: usize,
        preprocessing: Option<Arc<dyn McMultiOutputObjective>>,
    ) -> Result<Self> {
        let base = MoRiskBase::new(n_w, preprocessing)?;
        check_alpha(alpha, "alpha must be in (0.0, 1.0]")?;
        Ok(Self {
            base,
            alpha,
            alpha_idx: alpha_idx(alpha, n_w),
        })
    }

    /// The risk level.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// BoTorch's `alpha_idx`.
    pub fn alpha_idx(&self) -> usize {
        self.alpha_idx
    }

    mo_risk_common!();
}

impl McMultiOutputObjective for IndependentCVaR {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        self.base
            .evaluate_independent(&IndependentKind::CVaR(self.alpha_idx), samples, with_grad)
    }
}

/// The VaR of every output on its own (BoTorch's `IndependentVaR`).
#[derive(Clone, Debug)]
pub struct IndependentVaR {
    base: MoRiskBase,
    alpha: f64,
    alpha_idx: usize,
}

impl IndependentVaR {
    /// `IndependentVaR(alpha, n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// See [`IndependentCVaR::new`].
    pub fn new(
        alpha: f64,
        n_w: usize,
        preprocessing: Option<Arc<dyn McMultiOutputObjective>>,
    ) -> Result<Self> {
        let base = MoRiskBase::new(n_w, preprocessing)?;
        check_alpha(alpha, "alpha must be in (0.0, 1.0]")?;
        Ok(Self {
            base,
            alpha,
            alpha_idx: alpha_idx(alpha, n_w),
        })
    }

    /// The risk level.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// BoTorch's `alpha_idx`.
    pub fn alpha_idx(&self) -> usize {
        self.alpha_idx
    }

    mo_risk_common!();
}

impl McMultiOutputObjective for IndependentVaR {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        self.base
            .evaluate_independent(&IndependentKind::VaR(self.alpha_idx), samples, with_grad)
    }
}

/// Options of [`MVaR`] (BoTorch's keyword arguments).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MVaROptions {
    /// Return the mean of the MVaR set (Cousin & Di Bernardino, 2013) instead
    /// of the set (default `false`).
    pub expectation: bool,
    /// Pad every MVaR set to `n_w` points instead of the largest set size
    /// (default `false`).
    pub pad_to_n_w: bool,
    /// Return only the non-dominated alpha-level points (default `true`).
    pub filter_dominated: bool,
    /// BoTorch's `use_counting` (default `false`): the grid of distinct
    /// values in decreasing order instead of the sorted values with
    /// duplicates, which changes the order of the MVaR points (and, with
    /// `filter_dominated=false`, the duplicates).
    pub use_counting: bool,
}

impl Default for MVaROptions {
    fn default() -> Self {
        Self {
            expectation: false,
            pad_to_n_w: false,
            filter_dominated: true,
            use_counting: false,
        }
    }
}

/// The MVaR points of one candidate: the points and, for every point and
/// output, the samples that attain the component.
struct MVaRSet {
    points: Vec<Vec<f64>>,
}

/// The multivariate Value-at-Risk (BoTorch's `MVaR`; see the [module
/// documentation](self)).
#[derive(Clone, Debug)]
pub struct MVaR {
    base: MoRiskBase,
    alpha: f64,
    options: MVaROptions,
}

impl MVaR {
    /// `MVaR(n_w, alpha, expectation, preprocessing_function, pad_to_n_w,
    /// filter_dominated, use_counting)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `alpha ∈ (0, 1]` (BoTorch's
    /// message) and `n_w ≥ 1`.
    pub fn new(
        n_w: usize,
        alpha: f64,
        preprocessing: Option<Arc<dyn McMultiOutputObjective>>,
        options: MVaROptions,
    ) -> Result<Self> {
        let base = MoRiskBase::new(n_w, preprocessing)?;
        check_alpha(alpha, "`alpha` must be in (0.0, 1.0]")?;
        Ok(Self {
            base,
            alpha,
            options,
        })
    }

    /// The risk level.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The options.
    pub fn options(&self) -> &MVaROptions {
        &self.options
    }

    mo_risk_common!();

    /// The MVaR set of the `n_w` samples `y` (`n_w × m`, rows are samples) in
    /// BoTorch's order (BoTorch's `get_mvar_set_vectorized` or, with
    /// `use_counting`, `get_mvar_set_via_counting`, with the fix of
    /// meta-pytorch/botorch#3392 for ties).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] unless `y` has `n_w` rows.
    pub fn mvar_set(&self, y: &Matrix) -> Result<Vec<Vec<f64>>> {
        if y.nrows() != self.base.n_w || y.ncols() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} samples of a candidate, got a {}x{} matrix",
                self.base.n_w,
                y.nrows(),
                y.ncols()
            )));
        }
        Ok(self.set_of(y).points)
    }

    fn set_of(&self, y: &Matrix) -> MVaRSet {
        let (n_w, m) = y.shape();
        let var_idx = alpha_idx(self.alpha, n_w);
        let n_points = n_w - var_idx;
        let k = var_idx + 1; // ceil(alpha * n_w)
                             // The n_points smallest values of every output, ascending (topk with
                             // largest=False), and for counting their distinct values.
        let mut axes: Vec<Vec<f64>> = (0..m)
            .map(|t| {
                let mut col: Vec<f64> = y.column(t).iter().copied().collect();
                col.sort_by(|a, b| torch_cmp(*a, *b));
                col.truncate(n_points);
                col
            })
            .collect();
        // Ties in every output (#3392): only then can an MVaR point be
        // dominated by more than k samples.
        let ties = axes.iter().all(|a| a.windows(2).any(|w| w[0] == w[1]));
        if self.options.use_counting {
            for a in &mut axes {
                a.dedup_by(|x, y| *x == *y);
            }
        }
        let dims: Vec<usize> = axes.iter().map(Vec::len).collect();
        let size: usize = dims.iter().product();
        // Row-major strides of the grid.
        let mut strides = vec![1usize; m];
        for t in (0..m.saturating_sub(1)).rev() {
            strides[t] = strides[t + 1] * dims[t + 1];
        }
        // cdf[i] = #{samples y: y ≥ grid[i]}: a sample dominates the grid
        // points with i_t < c_t(y) = #{grid values of output t ≤ y_t} for all
        // t, so the counts are suffix sums of a histogram of (c_t(y) - 1).
        let mut cdf = vec![0usize; size];
        'samples: for i in 0..n_w {
            let mut idx = 0;
            for t in 0..m {
                let v = y[(i, t)];
                let c = if v.is_nan() {
                    0
                } else {
                    axes[t].partition_point(|&a| a <= v)
                };
                if c == 0 {
                    continue 'samples;
                }
                idx += (c - 1) * strides[t];
            }
            cdf[idx] += 1;
        }
        for t in 0..m {
            let (dim, stride) = (dims[t], strides[t]);
            for flat in (0..size).rev() {
                let it = (flat / stride) % dim;
                if it + 1 < dim {
                    cdf[flat] += cdf[flat + stride];
                }
            }
        }
        // Alpha-level points: dominated by exactly k samples, or (with ties)
        // by at least k samples while the next grid value along every output
        // is dominated by fewer.
        let is_alpha = |flat: usize| -> bool {
            let c = cdf[flat];
            if c == k {
                return true;
            }
            if !ties || c < k {
                return false;
            }
            (0..m).all(|t| {
                let it = (flat / strides[t]) % dims[t];
                it + 1 == dims[t] || cdf[flat + strides[t]] < k
            })
        };
        let point = |flat: usize| -> Vec<f64> {
            (0..m)
                .map(|t| axes[t][(flat / strides[t]) % dims[t]])
                .collect()
        };
        let mut points: Vec<Vec<f64>> = if self.options.use_counting {
            // Row-major over the distinct values in decreasing order.
            (0..size)
                .map(|f| {
                    // The flat index of the reversed multi-index.
                    (0..m)
                        .map(|t| (dims[t] - 1 - (f / strides[t]) % dims[t]) * strides[t])
                        .sum::<usize>()
                })
                .filter(|&flat| is_alpha(flat))
                .map(point)
                .collect()
        } else {
            (0..size).filter(|&f| is_alpha(f)).map(point).collect()
        };
        if self.options.filter_dominated && !points.is_empty() {
            let mat = Matrix::from_fn(points.len(), m, |i, t| points[i][t]);
            let mask = is_non_dominated(&mat, true, true);
            points = points
                .into_iter()
                .zip(mask)
                .filter(|(_, keep)| *keep)
                .map(|(p, _)| p)
                .collect();
        }
        MVaRSet { points }
    }
}

/// Component-wise mapping of supplied MVaR points to the samples attaining
/// each component. With an adjoint, returns its exact sample VJP instead.
/// Unattained components produce NaN values and zero sample derivatives,
/// matching BoTorch's masked mean (including its zero-count convention).
pub fn mvar_make_differentiable(
    prepared: &Matrix,
    mvars: &Matrix,
    adjoint: Option<&Matrix>,
) -> Result<Matrix> {
    let (nw, m) = prepared.shape();
    let k = mvars.nrows();
    if mvars.ncols() != m || adjoint.is_some_and(|bar| bar.shape() != mvars.shape()) {
        return Err(BotorchError::DimensionMismatch(
            "prepared samples and MVaR points must have equal output dimensions; adjoint must match MVaR shape".into(),
        ));
    }
    let mut result = if adjoint.is_some() {
        Matrix::zeros(nw, m)
    } else {
        Matrix::zeros(k, m)
    };
    for j in 0..k {
        for t in 0..m {
            let value = mvars[(j, t)];
            let mut count = 0;
            let mut sum = 0.0;
            for i in 0..nw {
                if prepared[(i, t)] == value {
                    count += 1;
                    sum += prepared[(i, t)];
                }
            }
            if let Some(bar) = adjoint {
                if count != 0 {
                    let weight = bar[(j, t)] / count as f64;
                    for i in 0..nw {
                        if prepared[(i, t)] == value {
                            result[(i, t)] += weight;
                        }
                    }
                }
            } else {
                result[(j, t)] = sum / count as f64;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod component_mapping_tests {
    use super::*;

    #[test]
    fn tied_components_share_adjoint_and_unattained_components_have_zero_derivative() {
        let samples = Matrix::from_row_slice(3, 2, &[1., 2., 1., 4., 3., 2.]);
        let points = Matrix::from_row_slice(2, 2, &[1., 2., 7., 4.]);
        let mapped = mvar_make_differentiable(&samples, &points, None).unwrap();
        assert_eq!(mapped[(0, 0)], 1.);
        assert!(mapped[(1, 0)].is_nan());
        let bar = Matrix::from_row_slice(2, 2, &[2., 6., 9., 5.]);
        let gradient = mvar_make_differentiable(&samples, &points, Some(&bar)).unwrap();
        assert_eq!(
            gradient,
            Matrix::from_row_slice(3, 2, &[1., 3., 1., 5., 0., 3.])
        );
    }
}

impl McMultiOutputObjective for MVaR {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        let n_w = self.base.n_w;
        let (pre, pre_pb) = prepare(self.base.preprocessing.as_ref(), samples, n_w, with_grad)?;
        let (n, m, s) = (pre.num_points(), pre.num_outputs(), pre.num_samples());
        let q = n / n_w;
        let data = pre.data();
        // The MVaR set of every sample and candidate.
        let sets: Vec<Vec<MVaRSet>> = (0..s)
            .map(|c| {
                (0..q)
                    .map(|g| {
                        let y = Matrix::from_fn(n_w, m, |i, t| data[(t * n + g * n_w + i, c)]);
                        self.set_of(&y)
                    })
                    .collect()
            })
            .collect();
        let k_out = if self.options.expectation {
            1
        } else if self.options.pad_to_n_w {
            n_w
        } else {
            sets.iter()
                .flatten()
                .map(|set| set.points.len())
                .max()
                .unwrap_or(1)
        };
        let np = q * k_out;
        let mut values = Matrix::zeros(m * np, s);
        // (output row, prepared row, weight) of the derivative per sample.
        let mut links: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); s];
        for c in 0..s {
            for g in 0..q {
                let set = &sets[c][g].points;
                let kset = set.len();
                // The samples of output t that attain value v, and the mean
                // over them (BoTorch's make_differentiable).
                let attained = |t: usize, v: f64| -> (f64, Vec<usize>) {
                    let rows: Vec<usize> = (0..n_w)
                        .filter(|&i| data[(t * n + g * n_w + i, c)] == v)
                        .collect();
                    let sum: f64 = (0..n_w)
                        .map(|i| {
                            let x = data[(t * n + g * n_w + i, c)];
                            if x == v {
                                x
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    (sum / rows.len() as f64, rows)
                };
                for t in 0..m {
                    if self.options.expectation {
                        // The mean of the set (with the gradient of every
                        // point mapped to the samples that attain it).
                        let out_row = t * np + g;
                        let mut total = 0.0;
                        for p in set {
                            let (v, rows) = attained(t, p[t]);
                            total += if with_grad { v } else { p[t] };
                            let w = 1.0 / (kset as f64 * rows.len() as f64);
                            for i in rows {
                                links[c].push((out_row, t * n + g * n_w + i, w));
                            }
                        }
                        values[(out_row, c)] = total / kset as f64;
                    } else {
                        for j in 0..k_out {
                            let p = &set[j.min(kset - 1)];
                            let out_row = t * np + g * k_out + j;
                            let (v, rows) = attained(t, p[t]);
                            values[(out_row, c)] = if with_grad { v } else { p[t] };
                            let w = 1.0 / rows.len() as f64;
                            for i in rows {
                                links[c].push((out_row, t * n + g * n_w + i, w));
                            }
                        }
                    }
                }
            }
        }
        let pullback: Option<SamplePullback> = with_grad.then(|| {
            Box::new(move |bar: &Matrix| {
                let mut pre_bar = Matrix::zeros(m * n, s);
                for (c, l) in links.iter().enumerate() {
                    for &(o, r, w) in l {
                        pre_bar[(r, c)] += bar[(o, c)] * w;
                    }
                }
                finish(pre_bar, pre_pb)
            }) as SamplePullback
        });
        Ok(MultiOutputValues {
            values: McSamples::new(values, np, m)?,
            pullback,
        })
    }

    fn verify_output_shape(&self) -> bool {
        false
    }
}

/// MVaR approximation based on random scalarizations (BoTorch's `MARS`): the
/// VaR of the Chebyshev scalarization `min_t w_t (ŷ_t - r̂_t)` of the outputs
/// normalized with bounds from the MVaR of the baseline (`ŷ`, the normalized
/// reference point `r̂` if given). A single-output objective, to be used with
/// single-output MC acquisition functions such as qLogNEI.
#[derive(Clone, Debug)]
pub struct Mars {
    alpha: f64,
    n_w: usize,
    var: RiskKind,
    weights: Vec<f64>,
    ref_point: Option<Vec<f64>>,
    preprocessing: Option<Arc<dyn McMultiOutputObjective>>,
    baseline_y: Option<Matrix>,
    /// The normalization bounds (`2 × m`) and the normalized reference point.
    scalarization: Option<(Matrix, Option<Vec<f64>>)>,
}

impl Mars {
    /// `MARS(alpha, n_w, chebyshev_weights, baseline_Y, ref_point,
    /// preprocessing_function)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `alpha ∈ (0, 1]` and `n_w ≥
    /// 1`, [`BotorchError::Unsupported`] for negative weights (BoTorch's
    /// message), and the errors of [`Mars::set_baseline_y`].
    pub fn new(
        alpha: f64,
        n_w: usize,
        chebyshev_weights: Vec<f64>,
        baseline_y: Option<Matrix>,
        ref_point: Option<Vec<f64>>,
        preprocessing: Option<Arc<dyn McMultiOutputObjective>>,
    ) -> Result<Self> {
        check_n_w(n_w)?;
        check_alpha(alpha, "alpha must be in (0.0, 1.0]")?;
        let idx = alpha_idx(alpha, n_w);
        let mut mars = Self {
            alpha,
            n_w,
            var: RiskKind::VaR {
                q: 1.0 - idx as f64 / n_w as f64,
            },
            weights: Vec::new(),
            ref_point,
            preprocessing,
            baseline_y: None,
            scalarization: None,
        };
        mars.set_chebyshev_weights(chebyshev_weights)?;
        if let Some(y) = baseline_y {
            mars.set_baseline_y(y)?;
        }
        Ok(mars)
    }

    /// The risk level.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The number of points per candidate `n_w`.
    pub fn n_w(&self) -> usize {
        self.n_w
    }

    /// The Chebyshev weights.
    pub fn chebyshev_weights(&self) -> &[f64] {
        &self.weights
    }

    /// The reference point, if any.
    pub fn ref_point(&self) -> Option<&[f64]> {
        self.ref_point.as_deref()
    }

    /// The baseline outcomes (`n' × m`), if set.
    pub fn baseline_y(&self) -> Option<&Matrix> {
        self.baseline_y.as_ref()
    }

    /// The preprocessing function (`None`: all outputs).
    pub fn preprocessing_function(&self) -> Option<&Arc<dyn McMultiOutputObjective>> {
        self.preprocessing.as_ref()
    }

    /// The normalization bounds of the scalarization (`2 × m`), once the
    /// baseline is set.
    pub fn normalization_bounds(&self) -> Option<&Matrix> {
        self.scalarization.as_ref().map(|(b, _)| b)
    }

    /// Sets the Chebyshev weights (BoTorch's `chebyshev_weights` setter).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for negative weights (BoTorch's
    /// message) and [`BotorchError::InvalidArgument`] for no weights.
    pub fn set_chebyshev_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        if weights.iter().any(|&w| w < 0.0) {
            return Err(BotorchError::Unsupported(
                "Negative weights are not supported in MARS.".into(),
            ));
        }
        if weights.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "MARS needs one Chebyshev weight per objective".into(),
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// Sets the baseline outcomes (BoTorch's `baseline_Y` setter) and
    /// computes the normalization bounds (BoTorch's
    /// `_get_Y_normalization_bounds`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the reference point does not
    /// match the outcomes.
    pub fn set_baseline_y(&mut self, baseline_y: Matrix) -> Result<()> {
        let bounds = y_normalization_bounds(&baseline_y, self.ref_point.as_deref())?;
        let ref_n = self.ref_point.as_ref().map(|r| {
            r.iter()
                .enumerate()
                .map(|(t, v)| (v - bounds[(0, t)]) / (bounds[(1, t)] - bounds[(0, t)]))
                .collect()
        });
        self.baseline_y = Some(baseline_y);
        self.scalarization = Some((bounds, ref_n));
        Ok(())
    }

    /// BoTorch's `set_baseline_Y(None, None, Y_samples)`: the non-dominated
    /// MVaR points (`MVaR(n_w, alpha)`) of the preprocessed predictions
    /// `y_samples` (`(n·n_w) × m`, `n_w` rows per baseline point) become the
    /// baseline.
    ///
    /// # Errors
    /// Errors of the preprocessing function and of [`Mars::set_baseline_y`].
    pub fn set_baseline_y_from_samples(&mut self, y_samples: &Matrix) -> Result<()> {
        let (n, m) = y_samples.shape();
        let samples = McSamples::new(
            Matrix::from_fn(m * n, 1, |r, _| y_samples[(r % n, r / n)]),
            n,
            m,
        )?;
        let mvar = MVaR::new(
            self.n_w,
            self.alpha,
            self.preprocessing.clone(),
            MVaROptions::default(),
        )?;
        let v = mvar.evaluate(&samples, None, false)?.values;
        let (np, mo) = (v.num_points(), v.num_outputs());
        let y = Matrix::from_fn(np, mo, |p, t| v.value(0, p, t));
        let mask = is_non_dominated(&y, true, true);
        let rows: Vec<usize> = (0..np).filter(|&i| mask[i]).collect();
        let pareto = Matrix::from_fn(rows.len(), mo, |i, t| y[(rows[i], t)]);
        self.set_baseline_y(pareto)
    }

    /// BoTorch's `set_baseline_Y(model, X_baseline)`: the posterior means of
    /// the `n_w` perturbations of every baseline point (each point evaluated
    /// on its own) passed to [`Mars::set_baseline_y_from_samples`].
    ///
    /// # Errors
    /// Errors of the model and of [`Mars::set_baseline_y_from_samples`].
    pub fn set_baseline_y_from_model(
        &mut self,
        model: &dyn Model,
        x_baseline: &Matrix,
    ) -> Result<()> {
        let k = model.points_per_input();
        let m = model.num_outputs();
        let n = x_baseline.nrows();
        let mut y = Matrix::zeros(n * k, m);
        for i in 0..n {
            let p = model.posterior(&x_baseline.rows(i, 1).into_owned(), false)?;
            let mean = if p.is_transformed() {
                let q = p.num_points();
                let mu = p.moments()?.0;
                Matrix::from_fn(q, m, |w, t| mu[t * q + w])
            } else {
                p.mean_matrix()
            };
            for w in 0..k {
                for t in 0..m {
                    y[(i * k + w, t)] = mean[(w, t)];
                }
            }
        }
        self.set_baseline_y_from_samples(&y)
    }
}

/// BoTorch's `MARS._get_Y_normalization_bounds`: `2 × m` bounds (nadir and
/// ideal point of the Pareto front of `y`, relative to `ref_point` if some
/// Pareto point is better than it; `[y, y + 1]` for a single point), with
/// empty ranges widened by one.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] if `ref_point` does not have one
/// value per column of `y`.
pub fn y_normalization_bounds(y: &Matrix, ref_point: Option<&[f64]>) -> Result<Matrix> {
    let m = y.ncols();
    if let Some(r) = ref_point {
        if r.len() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "The size of tensor a ({m}) must match the size of tensor b ({}) at \
                 non-singleton dimension 1",
                r.len()
            )));
        }
    }
    if y.nrows() == 0 {
        return Ok(Matrix::from_fn(2, m, |i, _| i as f64));
    }
    let mask = is_non_dominated(y, true, true);
    let rows: Vec<usize> = (0..y.nrows()).filter(|&i| mask[i]).collect();
    let pareto = Matrix::from_fn(rows.len(), m, |i, t| y[(rows[i], t)]);
    let better =
        |i: usize| -> bool { ref_point.is_some_and(|r| (0..m).all(|t| pareto[(i, t)] > r[t])) };
    let mut bounds = if pareto.nrows() == 1 {
        match ref_point {
            Some(r) if better(0) => {
                Matrix::from_fn(2, m, |i, t| if i == 0 { r[t] } else { pareto[(0, t)] })
            }
            _ => Matrix::from_fn(2, m, |i, t| pareto[(0, t)] + i as f64),
        }
    } else {
        let any_better = (0..pareto.nrows()).any(better);
        let (nadir, sel): (Vec<f64>, Vec<usize>) = match ref_point {
            Some(r) if any_better => (
                r.to_vec(),
                (0..pareto.nrows()).filter(|&i| better(i)).collect(),
            ),
            _ => {
                let all: Vec<usize> = (0..pareto.nrows()).collect();
                let nadir = (0..m)
                    .map(|t| {
                        all.iter()
                            .map(|&i| pareto[(i, t)])
                            .fold(f64::INFINITY, nan_min)
                    })
                    .collect();
                (nadir, all)
            }
        };
        let ideal: Vec<f64> = (0..m)
            .map(|t| {
                sel.iter()
                    .map(|&i| pareto[(i, t)])
                    .fold(f64::NEG_INFINITY, nan_max)
            })
            .collect();
        Matrix::from_fn(2, m, |i, t| if i == 0 { nadir[t] } else { ideal[t] })
    };
    for t in 0..m {
        let range =
            nan_max(bounds[(0, t)], bounds[(1, t)]) - nan_min(bounds[(0, t)], bounds[(1, t)]);
        if range <= 0.0 {
            bounds[(1, t)] += 1.0;
        }
    }
    Ok(bounds)
}

fn nan_min(a: f64, b: f64) -> f64 {
    crate::utils::multi_objective::nan_min(a, b)
}

fn nan_max(a: f64, b: f64) -> f64 {
    crate::utils::multi_objective::nan_max(a, b)
}

impl McObjective for Mars {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let Some((bounds, ref_n)) = &self.scalarization else {
            return Err(BotorchError::Runtime(
                "baseline_Y must be set before constructing the Chebyshev objective.".into(),
            ));
        };
        let n_w = self.n_w;
        let (pre, pre_pb) = prepare(self.preprocessing.as_ref(), samples, n_w, with_grad)?;
        let (n, m, s) = (pre.num_points(), pre.num_outputs(), pre.num_samples());
        if m != self.weights.len() || m != bounds.ncols() {
            return Err(BotorchError::DimensionMismatch(format!(
                "MARS has {} Chebyshev weights and bounds for {} outputs, but the preprocessed \
                 samples have {m} outputs",
                self.weights.len(),
                bounds.ncols()
            )));
        }
        let data = pre.data();
        // The Chebyshev scalarization of every point and the output that
        // attains the minimum.
        let mut scal = Matrix::zeros(n, s);
        let mut arg = vec![0usize; n * s];
        for c in 0..s {
            for p in 0..n {
                let mut best = f64::NAN;
                let mut best_t = 0;
                for t in 0..m {
                    let mut z =
                        (data[(t * n + p, c)] - bounds[(0, t)]) / (bounds[(1, t)] - bounds[(0, t)]);
                    if let Some(r) = ref_n {
                        z -= r[t];
                    }
                    let v = z * self.weights[t];
                    // min(dim=-1): the first minimum, NaN propagates.
                    if t == 0 || (!best.is_nan() && (v.is_nan() || v < best)) {
                        best = v;
                        best_t = t;
                    }
                }
                scal[(p, c)] = best;
                arg[c * n + p] = best_t;
            }
        }
        // VaR over the blocks of n_w points.
        let q = n / n_w;
        let mut values = Matrix::zeros(q, s);
        let mut weights = if with_grad {
            Matrix::zeros(n, s)
        } else {
            Matrix::zeros(0, 0)
        };
        let mut v = vec![0.0; n_w];
        let mut w = vec![0.0; n_w];
        for c in 0..s {
            for g in 0..q {
                for (i, vi) in v.iter_mut().enumerate() {
                    *vi = scal[(g * n_w + i, c)];
                }
                values[(g, c)] = reduce_block(&self.var, &v, &mut w);
                if with_grad {
                    for (i, wi) in w.iter().enumerate() {
                        weights[(g * n_w + i, c)] = *wi;
                    }
                }
            }
        }
        let pullback: Option<SamplePullback> = if with_grad {
            let factors: Vec<f64> = (0..m)
                .map(|t| self.weights[t] / (bounds[(1, t)] - bounds[(0, t)]))
                .collect();
            Some(Box::new(move |bar: &Matrix| {
                let mut pre_bar = Matrix::zeros(m * n, s);
                for c in 0..s {
                    for p in 0..n {
                        let g = weights[(p, c)] * bar[(p / n_w, c)];
                        if g != 0.0 {
                            let t = arg[c * n + p];
                            pre_bar[(t * n + p, c)] = g * factors[t];
                        }
                    }
                }
                finish(pre_bar, pre_pb)
            }))
        } else {
            None
        };
        Ok(SampleValues { values, pullback })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition::multi_objective::objective::WeightedMcMultiOutputObjective;
    use crate::Vector;

    fn samples(s: usize, n: usize, m: usize, seed: usize) -> McSamples {
        let rm: Vec<f64> = (0..s * n * m)
            .map(|i| (((i + seed) * 37 + 11) % 31) as f64 / 7.0 - 2.0 + 1e-3 * i as f64)
            .collect();
        McSamples::from_row_major(&rm, s, n, m).unwrap()
    }

    /// Brute-force MVaR: the maximal points of the full grid of sample values
    /// that are dominated by at least ceil(alpha·n_w) samples.
    fn brute_force(y: &Matrix, alpha: f64) -> Vec<Vec<f64>> {
        let (n_w, m) = y.shape();
        let k = (alpha * n_w as f64).ceil() as usize;
        let cols: Vec<Vec<f64>> = (0..m)
            .map(|t| {
                let mut c: Vec<f64> = y.column(t).iter().copied().collect();
                c.sort_by(|a, b| a.partial_cmp(b).unwrap());
                c.dedup();
                c
            })
            .collect();
        let mut grid = vec![Vec::new()];
        for col in &cols {
            grid = grid
                .into_iter()
                .flat_map(|p: Vec<f64>| {
                    col.iter().map(move |&v| {
                        let mut q = p.clone();
                        q.push(v);
                        q
                    })
                })
                .collect();
        }
        let count = |z: &[f64]| {
            (0..n_w)
                .filter(|&i| (0..m).all(|t| y[(i, t)] >= z[t]))
                .count()
        };
        let feasible: Vec<Vec<f64>> = grid.into_iter().filter(|z| count(z) >= k).collect();
        let mat = Matrix::from_fn(feasible.len(), m, |i, t| feasible[i][t]);
        let mask = is_non_dominated(&mat, true, true);
        let mut out: Vec<Vec<f64>> = feasible
            .into_iter()
            .zip(mask)
            .filter(|(_, k)| *k)
            .map(|(p, _)| p)
            .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    }

    #[test]
    fn mvar_matches_brute_force_with_and_without_ties() {
        for (case, (n_w, m)) in [(8, 2), (9, 3), (6, 2), (12, 2), (7, 3)]
            .into_iter()
            .enumerate()
        {
            for alpha in [0.2, 0.5, 0.75, 1.0] {
                for ties in [false, true] {
                    let y = Matrix::from_fn(n_w, m, |i, t| {
                        let v = ((i * 7 + t * 3 + case) % 11) as f64;
                        if ties {
                            (v / 3.0).floor()
                        } else {
                            v + 0.01 * (i * m + t) as f64
                        }
                    });
                    let expected = brute_force(&y, alpha);
                    for use_counting in [false, true] {
                        let mvar = MVaR::new(
                            n_w,
                            alpha,
                            None,
                            MVaROptions {
                                use_counting,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        let mut got = mvar.mvar_set(&y).unwrap();
                        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        assert_eq!(got, expected, "n_w={n_w} m={m} alpha={alpha} ties={ties}");
                    }
                }
            }
        }
    }

    #[test]
    fn mvar_points_are_dominated_by_alpha_of_the_samples() {
        let s = samples(5, 16, 2, 3);
        let mvar = MVaR::new(8, 0.6, None, MVaROptions::default()).unwrap();
        let v = mvar.evaluate(&s, None, false).unwrap().values;
        let k_out = v.num_points() / 2;
        for c in 0..5 {
            for g in 0..2 {
                for j in 0..k_out {
                    let p = g * k_out + j;
                    let dominating = (0..8)
                        .filter(|&i| (0..2).all(|t| s.value(c, g * 8 + i, t) >= v.value(c, p, t)))
                        .count();
                    assert!(dominating >= 5);
                }
            }
        }
        assert!(!mvar.verify_output_shape());
    }

    /// Central differences of `⟨bar, objective(samples)⟩` (MVaR is piecewise
    /// constant in the order of the samples, but the values attained move
    /// with the samples).
    fn check_mo_pullback(obj: &dyn McMultiOutputObjective, s: &McSamples) {
        let out = obj.evaluate(s, None, true).unwrap();
        let rows = out.values.data().nrows();
        let bar = Matrix::from_fn(rows, s.num_samples(), |r, c| {
            0.2 + 0.1 * r as f64 - 0.05 * c as f64
        });
        let (adj, _) = (out.pullback.unwrap())(&bar).unwrap();
        let (ns, n, m) = (s.num_samples(), s.num_points(), s.num_outputs());
        let rm = s.to_row_major();
        let h = 1e-6;
        for i in 0..rm.len() {
            let f = |delta: f64| {
                let mut v = rm.clone();
                v[i] += delta;
                let sp = McSamples::from_row_major(&v, ns, n, m).unwrap();
                obj.evaluate(&sp, None, false)
                    .unwrap()
                    .values
                    .data()
                    .component_mul(&bar)
                    .sum()
            };
            let fd = (f(h) - f(-h)) / (2.0 * h);
            let (c, rest) = (i / (n * m), i % (n * m));
            let (p, t) = (rest / m, rest % m);
            assert!(
                (fd - adj[(t * n + p, c)]).abs() < 1e-6,
                "{i}: {} vs {fd}",
                adj[(t * n + p, c)]
            );
        }
    }

    #[test]
    fn gradients() {
        let s = samples(3, 12, 3, 1);
        let pre: Arc<dyn McMultiOutputObjective> = Arc::new(
            WeightedMcMultiOutputObjective::new(
                Vector::from_vec(vec![1.0, -0.5]),
                Some(&[2, 0]),
                None,
            )
            .unwrap(),
        );
        for p in [None, Some(Arc::clone(&pre))] {
            check_mo_pullback(&MultiOutputExpectation::new(4, p.clone()).unwrap(), &s);
            check_mo_pullback(&MultiOutputWorstCase::new(4, p.clone()).unwrap(), &s);
            check_mo_pullback(&IndependentCVaR::new(0.6, 4, p.clone()).unwrap(), &s);
            check_mo_pullback(&IndependentVaR::new(0.6, 4, p.clone()).unwrap(), &s);
            for expectation in [false, true] {
                let mvar = MVaR::new(
                    6,
                    0.5,
                    p.clone(),
                    MVaROptions {
                        expectation,
                        ..Default::default()
                    },
                )
                .unwrap();
                check_mo_pullback(&mvar, &s);
            }
        }
    }

    #[test]
    fn expectation_and_worst_case_bound_the_independent_measures() {
        let s = samples(4, 10, 2, 5);
        let ex = MultiOutputExpectation::new(5, None)
            .unwrap()
            .evaluate(&s, None, false)
            .unwrap()
            .values;
        let wc = MultiOutputWorstCase::new(5, None)
            .unwrap()
            .evaluate(&s, None, false)
            .unwrap()
            .values;
        let cv = IndependentCVaR::new(0.7, 5, None)
            .unwrap()
            .evaluate(&s, None, false)
            .unwrap()
            .values;
        let va = IndependentVaR::new(0.7, 5, None)
            .unwrap()
            .evaluate(&s, None, false)
            .unwrap()
            .values;
        for c in 0..4 {
            for p in 0..2 {
                for t in 0..2 {
                    let (e, w) = (ex.value(c, p, t), wc.value(c, p, t));
                    assert!(w <= cv.value(c, p, t) && cv.value(c, p, t) <= e + 1e-15);
                    assert!(w <= va.value(c, p, t));
                    assert!(cv.value(c, p, t) <= va.value(c, p, t));
                }
            }
        }
    }

    #[test]
    fn mars_scalarization_and_bounds() {
        let base = Matrix::from_row_slice(3, 2, &[1.0, 3.0, 2.0, 2.0, 0.5, 0.5]);
        let mut mars = Mars::new(0.5, 4, vec![0.3, 0.7], Some(base.clone()), None, None).unwrap();
        let b = mars.normalization_bounds().unwrap().clone();
        assert_eq!(b, Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 3.0]));
        let s = samples(3, 8, 2, 7);
        let v = mars.evaluate(&s, None, false).unwrap().values;
        // VaR (alpha = 0.5, n_w = 4: the second largest) of the scalarization.
        for c in 0..3 {
            for g in 0..2 {
                let mut vals: Vec<f64> = (0..4)
                    .map(|i| {
                        let p = g * 4 + i;
                        let z0 = 0.3 * (s.value(c, p, 0) - 1.0) / 1.0;
                        let z1 = 0.7 * (s.value(c, p, 1) - 2.0) / 1.0;
                        z0.min(z1)
                    })
                    .collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!((v[(g, c)] - vals[2]).abs() < 1e-15);
            }
        }
        let out = mars.evaluate(&s, None, true).unwrap();
        assert!(out.pullback.is_some());
        // A reference point better than no Pareto point: min/max of the front.
        mars = Mars::new(0.5, 4, vec![0.3, 0.7], None, Some(vec![5.0, 5.0]), None).unwrap();
        assert!(mars.evaluate(&s, None, false).is_err());
        mars.set_baseline_y(base).unwrap();
        assert_eq!(
            mars.normalization_bounds().unwrap(),
            &Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 3.0])
        );
        assert!(Mars::new(0.5, 4, vec![-0.3, 0.7], None, None, None).is_err());
        // A single Pareto point.
        let one = y_normalization_bounds(&Matrix::from_row_slice(1, 2, &[1.0, 2.0]), None).unwrap();
        assert_eq!(one, Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 3.0]));
    }
}
