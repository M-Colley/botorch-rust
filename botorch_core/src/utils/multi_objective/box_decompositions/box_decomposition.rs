//! The base API of the box decompositions (`box_decomposition.py`):
//! [`BoxDecomposition`], [`FastPartitioning`] and their bounds,
//! [`CellBounds`], plus the state and logic shared by the implementations.

use std::fmt;

use super::utils::{
    compute_dominated_hypercell_bounds_2d, compute_non_dominated_hypercell_bounds_2d,
    expand_ref_point, get_partition_bounds, pad_batch_pareto_frontier, LocalUpperBounds,
};
use crate::error::{BotorchError, Result};
use crate::utils::multi_objective::pareto::is_non_dominated_rows;
use crate::utils::multi_objective::{clamp_min, nan_max, nan_min, PointBatch};
use crate::Matrix;

/// The bounds of the hypercells of a box decomposition: BoTorch's
/// `2 x (batch_shape) x num_cells x m` tensor, stored as two contiguous
/// row-major arrays of shape `(batch_shape) x num_cells x m` (the lower and
/// the upper vertices). Batch element `b`, cell `k`, outcome `j` is at index
/// `(b·num_cells + k)·m + j`.
#[derive(Clone, Debug, PartialEq)]
pub struct CellBounds {
    batch_shape: Vec<usize>,
    num_cells: usize,
    m: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CellBounds {
    /// Bounds of shape `(batch_shape) x num_cells x m`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the lengths do not match the
    /// shape.
    pub fn new(
        batch_shape: Vec<usize>,
        num_cells: usize,
        m: usize,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result<Self> {
        let len = batch_shape.iter().product::<usize>() * num_cells * m;
        if lower.len() != len || upper.len() != len {
            return Err(BotorchError::DimensionMismatch(format!(
                "cell bounds with {} and {} values do not match the shape {:?}",
                lower.len(),
                upper.len(),
                [batch_shape.as_slice(), &[num_cells, m]].concat()
            )));
        }
        Ok(Self {
            batch_shape,
            num_cells,
            m,
            lower,
            upper,
        })
    }

    /// The batch shape (empty for an unbatched decomposition).
    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    /// The number of batch elements (1 if unbatched).
    pub fn batch_size(&self) -> usize {
        self.batch_shape.iter().product()
    }

    /// The number of cells (per batch element).
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// The number of outcomes.
    pub fn num_outcomes(&self) -> usize {
        self.m
    }

    /// BoTorch's shape `[2, *batch_shape, num_cells, m]`.
    pub fn shape(&self) -> Vec<usize> {
        let mut s = vec![2];
        s.extend(&self.batch_shape);
        s.extend([self.num_cells, self.m]);
        s
    }

    /// The lower vertices, row-major `(batch_shape) x num_cells x m`.
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// The upper vertices, row-major `(batch_shape) x num_cells x m`.
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// The lower vertices of batch element `b` (`num_cells x m`, row-major).
    pub fn lower_of(&self, b: usize) -> &[f64] {
        let len = self.num_cells * self.m;
        &self.lower[b * len..(b + 1) * len]
    }

    /// The upper vertices of batch element `b` (`num_cells x m`, row-major).
    pub fn upper_of(&self, b: usize) -> &[f64] {
        let len = self.num_cells * self.m;
        &self.upper[b * len..(b + 1) * len]
    }

    /// The lower and upper vertex of cell `k` of batch element `b`.
    pub fn cell(&self, b: usize, k: usize) -> (&[f64], &[f64]) {
        let start = (b * self.num_cells + k) * self.m;
        (
            &self.lower[start..start + self.m],
            &self.upper[start..start + self.m],
        )
    }

    /// The lower vertices of batch element `b` as a `num_cells x m` matrix.
    pub fn lower_matrix(&self, b: usize) -> Matrix {
        Matrix::from_row_slice(self.num_cells, self.m, self.lower_of(b))
    }

    /// The upper vertices of batch element `b` as a `num_cells x m` matrix.
    pub fn upper_matrix(&self, b: usize) -> Matrix {
        Matrix::from_row_slice(self.num_cells, self.m, self.upper_of(b))
    }

    /// BoTorch's tensor flattened row-major: the lower vertices followed by
    /// the upper vertices.
    pub fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(2 * self.lower.len());
        out.extend(&self.lower);
        out.extend(&self.upper);
        out
    }

    /// The sum of the cell volumes of each batch element, `Σ_k Π_j (u_kj -
    /// l_kj)` (BoTorch's `(bounds[1] - bounds[0]).prod(-1).sum(-1)`).
    pub fn volumes(&self) -> Vec<f64> {
        (0..self.batch_size())
            .map(|b| {
                let (lo, up) = (self.lower_of(b), self.upper_of(b));
                sum(lo
                    .chunks_exact(self.m.max(1))
                    .zip(up.chunks_exact(self.m.max(1)))
                    .map(|(l, u)| prod(l.iter().zip(u).map(|(l, u)| u - l))))
            })
            .collect()
    }
}

/// A box decomposition of the dominated or the non-dominated space of a
/// Pareto frontier (BoTorch's abstract `BoxDecomposition`; maximization).
///
/// Implemented by [`NondominatedPartitioning`](super::NondominatedPartitioning),
/// [`FastNondominatedPartitioning`](super::FastNondominatedPartitioning) and
/// [`DominatedPartitioning`](super::DominatedPartitioning).
pub trait BoxDecomposition: Send + Sync + fmt::Debug {
    /// BoTorch's class name.
    fn class_name(&self) -> &'static str;

    /// The reference point: `m` values, or `b·m` values (row-major) for a
    /// batched reference point.
    fn ref_point(&self) -> &[f64];

    /// The shape of the reference point, `[m]` or `[b, m]`.
    fn ref_point_shape(&self) -> &[usize];

    /// `num_outcomes`: the number of outcomes of the reference point, or of
    /// the outcomes once they are given.
    fn num_outcomes(&self) -> usize;

    /// `batch_shape` (`None` before outcomes are given, where BoTorch has no
    /// such attribute).
    fn batch_shape(&self) -> Option<&[usize]>;

    /// `Y`: all outcomes given so far (`None` before any are given).
    fn y(&self) -> Option<&PointBatch>;

    /// `pareto_Y`: the Pareto frontier of the outcomes that are better than
    /// the reference point (`None` before outcomes are given). Batched
    /// frontiers are padded to the size of the largest one; with two
    /// outcomes (and always for
    /// [`NondominatedPartitioning`](super::NondominatedPartitioning)) the
    /// points are sorted by decreasing first outcome.
    fn pareto_y(&self) -> Option<PointBatch>;

    /// `update(Y)`: appends new outcomes (same batch shape) and updates the
    /// decomposition.
    ///
    /// # Errors
    /// [`BotorchError::InputData`] for NaN outcomes,
    /// [`BotorchError::DimensionMismatch`] or [`BotorchError::Unsupported`]
    /// for shapes BoTorch does not support.
    fn update(&mut self, y: &PointBatch) -> Result<()>;

    /// `reset()`: recomputes the Pareto frontier from all outcomes and, if
    /// it changed, the decomposition.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if no outcomes have been given.
    fn reset(&mut self) -> Result<()>;

    /// `partition_space()`: recomputes the decomposition of the current
    /// Pareto frontier.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if no outcomes have been given.
    fn partition_space(&mut self) -> Result<()>;

    /// `get_hypercell_bounds()`: the cell bounds in the outcome space
    /// (maximization), or `None` before outcomes are given.
    fn get_hypercell_bounds(&self) -> Option<&CellBounds>;

    /// `compute_hypervolume()`: the hypervolume dominated by the Pareto
    /// frontier of each batch element (one value if unbatched; `[0.0]`
    /// before outcomes are given).
    fn compute_hypervolume(&self) -> Vec<f64>;
}

/// The decompositions based on Algorithm 1 of Lacour et al. (2017)
/// (BoTorch's abstract `FastPartitioning`), with incremental updates for
/// more than two outcomes.
pub trait FastPartitioning: BoxDecomposition {
    /// The local upper bounds and defining points of the minimization
    /// problem (BoTorch's `_U`, `_Z`); `None` unless the decomposition has
    /// more than two outcomes and a non-empty Pareto frontier.
    fn local_upper_bounds(&self) -> Option<&LocalUpperBounds>;
}

/// Sequential sum (the order of BoTorch's small reductions).
pub(super) fn sum(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, |acc, v| acc + v)
}

/// Sequential product.
pub(super) fn prod(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(1.0, |acc, v| acc * v)
}

/// The state shared by all box decompositions (BoTorch's `BoxDecomposition`
/// attributes): the reference point, the outcomes and the negated Pareto
/// frontier.
#[derive(Clone, Debug)]
pub(super) struct Core {
    pub(super) name: &'static str,
    pub(super) ref_values: Vec<f64>,
    pub(super) ref_shape: Vec<usize>,
    pub(super) num_outcomes: usize,
    pub(super) sort: bool,
    /// `Y` (maximization; BoTorch stores `_neg_Y = -Y`).
    pub(super) y: Option<PointBatch>,
    /// `_neg_pareto_Y`: the negated, padded (and possibly sorted) frontier.
    pub(super) neg_pareto: Option<PointBatch>,
    pub(super) batch_shape: Option<Vec<usize>>,
}

impl Core {
    pub(super) fn new(
        name: &'static str,
        ref_point: &[f64],
        ref_shape: &[usize],
        sort: bool,
    ) -> Result<Self> {
        let Some(&m) = ref_shape.last() else {
            return Err(BotorchError::DimensionMismatch(
                "the reference point must have at least one dimension".into(),
            ));
        };
        if ref_shape.iter().product::<usize>() != ref_point.len() || ref_shape.len() > 2 {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected ref_point to be a `batch_shape x m` or `m`-dim tensor, but got {} \
                 values of shape {ref_shape:?}.",
                ref_point.len()
            )));
        }
        Ok(Self {
            name,
            ref_values: ref_point.to_vec(),
            ref_shape: ref_shape.to_vec(),
            num_outcomes: m,
            sort,
            y: None,
            neg_pareto: None,
            batch_shape: None,
        })
    }

    /// The shape of a flat reference point given the outcomes it is used
    /// with: `[b, m]` if it has one value per outcome and batch element of a
    /// batch with one dimension, else `[len]`.
    pub(super) fn infer_ref_shape(ref_point: &[f64], y: Option<&PointBatch>) -> Vec<usize> {
        if let Some(y) = y {
            if ref_point.len() != y.m() && y.batch_shape().len() == 1 && y.m() > 0 {
                let b = y.batch_shape()[0];
                if ref_point.len() == b * y.m() {
                    return vec![b, y.m()];
                }
            }
        }
        vec![ref_point.len()]
    }

    /// The NaN check of BoTorch's `__init__` and `_update_neg_Y`.
    pub(super) fn check_nan(y: &PointBatch) -> Result<()> {
        let count = y.values().iter().filter(|v| v.is_nan()).count();
        if count > 0 {
            return Err(BotorchError::InputData(format!(
                "NaN inputs are not supported. Got Y with {count} NaN values."
            )));
        }
        Ok(())
    }

    /// `_update_neg_Y`: stores or appends the outcomes; `true` if there were
    /// none before.
    pub(super) fn update_y(&mut self, y: &PointBatch) -> Result<bool> {
        Self::check_nan(y)?;
        match &mut self.y {
            None => {
                self.y = Some(y.clone());
                Ok(true)
            }
            Some(old) => {
                if old.batch_shape() != y.batch_shape() || old.m() != y.m() {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "new outcomes of shape {:?} cannot be appended to outcomes of shape {:?}",
                        y.shape(),
                        old.shape()
                    )));
                }
                let (b, n0, n1, m) = (old.batch_size(), old.n(), y.n(), old.m());
                let mut values = Vec::with_capacity(b * (n0 + n1) * m);
                for k in 0..b {
                    values.extend_from_slice(old.element(k));
                    values.extend_from_slice(y.element(k));
                }
                *old = PointBatch::new(old.batch_shape().to_vec(), n0 + n1, m, values)?;
                Ok(false)
            }
        }
    }

    /// `_validate_inputs`.
    pub(super) fn validate_inputs(&mut self) -> Result<()> {
        let y = self.y_or_err()?;
        let batch_shape = y.batch_shape().to_vec();
        let m = y.m();
        self.batch_shape = Some(batch_shape.clone());
        self.num_outcomes = m;
        if batch_shape.len() > 1 {
            return Err(BotorchError::Unsupported(format!(
                "{} only supports a single batch dimension, but got {} batch dimensions.",
                self.name,
                batch_shape.len()
            )));
        }
        if !batch_shape.is_empty() && m > 2 {
            return Err(BotorchError::Unsupported(format!(
                "{} only supports a batched box decompositions in the 2-objective setting.",
                self.name
            )));
        }
        Ok(())
    }

    pub(super) fn y_or_err(&self) -> Result<&PointBatch> {
        self.y
            .as_ref()
            .ok_or_else(|| BotorchError::InvalidArgument("Y data has not been initialized".into()))
    }

    pub(super) fn neg_pareto_or_err(&self) -> Result<&PointBatch> {
        self.neg_pareto.as_ref().ok_or_else(|| {
            BotorchError::InvalidArgument("pareto_Y has not been initialized".into())
        })
    }

    /// The reference point expanded to the batch shape (`b·m` values).
    pub(super) fn refs(&self) -> Result<Vec<f64>> {
        let y = self.y_or_err()?;
        let refs = expand_ref_point(&self.ref_values, &self.ref_shape, y.batch_shape())?;
        let m = self.ref_shape.last().copied().unwrap_or(0);
        if m != y.m() {
            return Err(BotorchError::DimensionMismatch(format!(
                "The reference point has {m} outcomes, but Y has {}.",
                y.m()
            )));
        }
        Ok(refs)
    }

    /// `_compute_pareto_Y`: the negated Pareto frontier of the outcomes that
    /// are better than the reference point, padded, and sorted by the first
    /// (negated) outcome if `sort`.
    pub(super) fn compute_pareto(&self) -> Result<PointBatch> {
        let y = self.y_or_err()?;
        let (m, n) = (y.m(), y.n());
        if n == 0 {
            return PointBatch::new(y.batch_shape().to_vec(), 0, m, Vec::new());
        }
        let refs = self.refs()?;
        let padded = pad_batch_pareto_frontier(y, &refs, false, None)?;
        let (n_p, b) = (padded.n(), padded.batch_size());
        let mut neg: Vec<f64> = padded.values().iter().map(|v| -v).collect();
        if self.sort && n_p > 1 {
            let mut order: Vec<usize> = Vec::with_capacity(n_p);
            let mut buf = vec![0.0; n_p * m];
            for k in 0..b {
                let block = &mut neg[k * n_p * m..(k + 1) * n_p * m];
                order.clear();
                order.extend(0..n_p);
                order.sort_by(|&a, &c| {
                    block[a * m]
                        .partial_cmp(&block[c * m])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (dst, &src) in order.iter().enumerate() {
                    buf[dst * m..(dst + 1) * m].copy_from_slice(&block[src * m..(src + 1) * m]);
                }
                block.copy_from_slice(&buf);
            }
        }
        PointBatch::new(padded.batch_shape().to_vec(), n_p, m, neg)
    }

    /// `_reset_pareto_Y`: `true` if the frontier changed.
    pub(super) fn reset_pareto(&mut self) -> Result<bool> {
        let pareto = self.compute_pareto()?;
        if self.neg_pareto.as_ref() != Some(&pareto) {
            self.neg_pareto = Some(pareto);
            return Ok(true);
        }
        Ok(false)
    }

    /// The body of BoTorch's `__init__` for given outcomes (before the space
    /// is partitioned).
    pub(super) fn init_outcomes(&mut self, y: &PointBatch) -> Result<()> {
        Self::check_nan(y)?;
        self.y = Some(y.clone());
        self.validate_inputs()?;
        self.neg_pareto = Some(self.compute_pareto()?);
        Ok(())
    }

    /// `pareto_Y`.
    pub(super) fn pareto_y(&self) -> Option<PointBatch> {
        let neg = self.neg_pareto.as_ref()?;
        PointBatch::new(
            neg.batch_shape().to_vec(),
            neg.n(),
            neg.m(),
            neg.values().iter().map(|v| -v).collect(),
        )
        .ok()
    }

    /// The hypervolume when there is nothing to decompose: `[0.0]` before
    /// outcomes are given, zeros if the frontier is empty, `None` otherwise.
    pub(super) fn trivial_hypervolume(&self) -> Option<Vec<f64>> {
        match &self.neg_pareto {
            None => Some(vec![0.0]),
            Some(p) if p.n() == 0 => Some(vec![0.0; p.batch_size()]),
            Some(_) => None,
        }
    }
}

/// The kind of a [`FastCore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FastKind {
    /// `FastNondominatedPartitioning`.
    NonDominated,
    /// `DominatedPartitioning`.
    Dominated,
}

/// The state and algorithms of BoTorch's `FastPartitioning` and its two
/// subclasses.
#[derive(Clone, Debug)]
pub(super) struct FastCore {
    pub(super) core: Core,
    pub(super) kind: FastKind,
    /// `_U`, `_Z` (more than two outcomes).
    pub(super) lub: Option<LocalUpperBounds>,
    /// `hypercell_bounds`.
    pub(super) cells: Option<CellBounds>,
}

impl FastCore {
    pub(super) fn new(
        kind: FastKind,
        ref_point: &[f64],
        ref_shape: &[usize],
        y: Option<&PointBatch>,
    ) -> Result<Self> {
        let name = match kind {
            FastKind::NonDominated => "FastNondominatedPartitioning",
            FastKind::Dominated => "DominatedPartitioning",
        };
        let sort = ref_shape.last() == Some(&2);
        let mut this = Self {
            core: Core::new(name, ref_point, ref_shape, sort)?,
            kind,
            lub: None,
            cells: None,
        };
        if let Some(y) = y {
            this.core.init_outcomes(y)?;
            this.partition_space()?;
        }
        Ok(this)
    }

    /// `FastPartitioning.partition_space`.
    pub(super) fn partition_space(&mut self) -> Result<()> {
        let n_p = self.core.neg_pareto_or_err()?.n();
        if n_p == 0 {
            self.single_cell()
        } else if self.core.num_outcomes == 2 {
            self.partition_space_2d()
        } else {
            self.partition_space_general()
        }
    }

    /// `_get_single_cell`: `[ref, inf]` (non-dominated) or the empty cell
    /// `[ref, ref]` (dominated) per batch element.
    fn single_cell(&mut self) -> Result<()> {
        let refs = self.core.refs()?;
        let neg = self.core.neg_pareto_or_err()?;
        let lower = refs.clone();
        let upper = match self.kind {
            FastKind::NonDominated => vec![f64::INFINITY; refs.len()],
            FastKind::Dominated => refs,
        };
        self.cells = Some(CellBounds::new(
            neg.batch_shape().to_vec(),
            1,
            neg.m(),
            lower,
            upper,
        )?);
        Ok(())
    }

    /// `_partition_space_2d`: the cells of the (padded, batched) frontier
    /// sorted by increasing first outcome.
    fn partition_space_2d(&mut self) -> Result<()> {
        let refs = self.core.refs()?;
        let neg = self.core.neg_pareto_or_err()?;
        // `self.pareto_Y.flip(-2)`: the negated frontier is sorted by
        // increasing negated first outcome.
        let (b, n_p, m) = (neg.batch_size(), neg.n(), neg.m());
        let mut values = Vec::with_capacity(b * n_p * m);
        for k in 0..b {
            for i in (0..n_p).rev() {
                values.extend(neg.point(k, i).iter().map(|v| -v));
            }
        }
        let sorted = PointBatch::new(neg.batch_shape().to_vec(), n_p, m, values)?;
        self.cells = Some(match self.kind {
            FastKind::NonDominated => compute_non_dominated_hypercell_bounds_2d(&sorted, &refs)?,
            FastKind::Dominated => compute_dominated_hypercell_bounds_2d(&sorted, &refs)?,
        });
        Ok(())
    }

    /// `_partition_space` (more than two outcomes, unbatched): the local
    /// upper bounds of the negated frontier from scratch.
    fn partition_space_general(&mut self) -> Result<()> {
        let neg = self.core.neg_pareto_or_err()?;
        if !neg.batch_shape().is_empty() {
            return Err(BotorchError::Unsupported(
                "_partition_space does not support batch dimensions.".into(),
            ));
        }
        // Use the reference point for maximization as the ideal point for
        // minimization (the dummy points of Lacour et al., Sec. 2.1, with
        // -inf instead of 0).
        let neg_ref: Vec<f64> = self.core.ref_values.iter().map(|r| -r).collect();
        let mut lub = LocalUpperBounds::initial(&neg_ref);
        for i in 0..neg.n() {
            lub.insert(neg.point(0, i))?;
        }
        self.lub = Some(lub);
        self.get_partitioning()
    }

    /// `_get_partitioning`: the cells from the local upper bounds.
    fn get_partitioning(&mut self) -> Result<()> {
        let lub = self.lub.as_ref().ok_or_else(|| {
            BotorchError::InvalidArgument("the local upper bounds are not initialized".into())
        })?;
        let m = lub.m();
        let cells = match self.kind {
            FastKind::Dominated => {
                let neg_ref: Vec<f64> = self.core.ref_values.iter().map(|r| -r).collect();
                let min_cells = get_partition_bounds(lub, &neg_ref)?;
                // Swap lower and upper bounds and negate.
                let lower = min_cells.upper().iter().map(|v| -v).collect();
                let upper = min_cells.lower().iter().map(|v| -v).collect();
                CellBounds::new(Vec::new(), min_cells.num_cells(), m, lower, upper)?
            }
            FastKind::NonDominated => {
                // The local lower bounds of the maximization problem, -U, are
                // the Pareto frontier of a minimization problem with the
                // reference point +inf, whose dominated space is partitioned
                // (Yang et al., 2019).
                let r = &self.core.ref_values;
                let mut z2 = Vec::with_capacity(m * m);
                for j in 0..m {
                    for (k, &rk) in r.iter().enumerate() {
                        z2.push(if k == j { f64::INFINITY } else { rk });
                    }
                }
                let mut lub2 = LocalUpperBounds::new(m, vec![f64::INFINITY; m], z2)?;
                let mut point = vec![0.0; m];
                for u in 0..lub.len() {
                    for (p, v) in point.iter_mut().zip(lub.upper_bound(u)) {
                        *p = -v;
                    }
                    lub2.insert(&point)?;
                }
                get_partition_bounds(&lub2, &vec![f64::INFINITY; m])?
            }
        };
        self.cells = Some(cells);
        Ok(())
    }

    /// `reset()`.
    pub(super) fn reset(&mut self) -> Result<()> {
        self.core.validate_inputs()?;
        if self.core.reset_pareto()? {
            self.partition_space()?;
        }
        Ok(())
    }

    /// `FastPartitioning.update`: with more than two outcomes and a
    /// non-empty frontier, the new Pareto points are inserted into the local
    /// upper bounds incrementally.
    pub(super) fn update(&mut self, y: &PointBatch) -> Result<()> {
        if self.core.update_y(y)? || self.core.num_outcomes == 2 {
            return self.reset();
        }
        let neg = self.core.neg_pareto_or_err()?;
        if neg.n() == 0 {
            return self.reset();
        }
        if !y.batch_shape().is_empty() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected new outcomes of shape `n x {}`, got {:?}.",
                neg.m(),
                y.shape()
            )));
        }
        let m = neg.m();
        // Only include points that are better than the reference point.
        let r = &self.core.ref_values;
        let new: Vec<&[f64]> = (0..y.n())
            .map(|i| y.point(0, i))
            .filter(|p| p.iter().zip(r).all(|(v, r)| v > r))
            .collect();
        if new.is_empty() {
            // The frontier and the decomposition do not change (BoTorch main
            // re-inserts the whole frontier here, with the same result;
            // skipped since meta-pytorch/botorch#3392).
            return Ok(());
        }
        // `is_non_dominated(-Y_all)` on the negated frontier followed by the
        // new points.
        let n_all = neg.n() + new.len();
        let mut all_max = Vec::with_capacity(n_all * m);
        all_max.extend(neg.values().iter().map(|v| -v));
        for p in &new {
            all_max.extend_from_slice(p);
        }
        let mask = is_non_dominated_rows(&all_max, n_all, m, true, true);
        let num_new = mask[neg.n()..].iter().filter(|&&k| k).count();
        let kept: Vec<f64> = (0..n_all)
            .filter(|&i| mask[i])
            .flat_map(|i| all_max[i * m..(i + 1) * m].iter().map(|v| -v))
            .collect();
        let n_kept = kept.len() / m;
        let new_neg = PointBatch::from_rows(n_kept, m, kept)?;
        if num_new > 0 {
            let mut lub = match self.lub.take() {
                Some(lub) => lub,
                None => {
                    self.core.neg_pareto = Some(new_neg);
                    return self.partition_space();
                }
            };
            for i in n_kept - num_new..n_kept {
                lub.insert(new_neg.point(0, i))?;
            }
            self.lub = Some(lub);
            self.core.neg_pareto = Some(new_neg);
            self.get_partitioning()?;
        } else {
            self.core.neg_pareto = Some(new_neg);
        }
        Ok(())
    }

    /// `_compute_hypervolume_if_y_has_data` (or the trivial values).
    pub(super) fn hypervolume(&self) -> Vec<f64> {
        if let Some(hv) = self.core.trivial_hypervolume() {
            return hv;
        }
        let (Some(neg), Some(cells)) = (&self.core.neg_pareto, &self.cells) else {
            return vec![0.0];
        };
        let b = neg.batch_size();
        match self.kind {
            FastKind::Dominated => cells.volumes(),
            FastKind::NonDominated => {
                let Ok(refs) = self.core.refs() else {
                    return vec![f64::NAN; b];
                };
                let (n_p, m) = (neg.n(), neg.m());
                (0..b)
                    .map(|k| {
                        // The ideal point of the (maximization) frontier.
                        let ideal: Vec<f64> = (0..m)
                            .map(|j| {
                                (0..n_p)
                                    .map(|i| -neg.point(k, i)[j])
                                    .fold(f64::NEG_INFINITY, nan_max)
                            })
                            .collect();
                        let r = &refs[k * m..(k + 1) * m];
                        let total = prod(ideal.iter().zip(r).map(|(i, r)| clamp_min(i - r, 0.0)));
                        let (lo, up) = (cells.lower_of(k), cells.upper_of(k));
                        let non_dom =
                            sum(lo.chunks_exact(m).zip(up.chunks_exact(m)).map(|(l, u)| {
                                prod((0..m).map(|j| {
                                    clamp_min(
                                        nan_min(u[j], ideal[j]) - nan_min(l[j], ideal[j]),
                                        0.0,
                                    )
                                }))
                            }));
                        total - non_dom
                    })
                    .collect()
            }
        }
    }
}

/// The constructors and the [`BoxDecomposition`] / [`FastPartitioning`]
/// implementations of a newtype around [`FastCore`].
macro_rules! impl_fast_partitioning {
    ($ty:ident, $kind:expr, $name:literal) => {
        impl $ty {
            #[doc = concat!("`", $name, "(ref_point, Y)` for an unbatched `n × m` matrix of outcomes (or none).")]
            ///
            /// # Errors
            /// [`BotorchError::InputData`](crate::BotorchError::InputData)
            /// for NaN outcomes,
            /// [`BotorchError::DimensionMismatch`](crate::BotorchError::DimensionMismatch)
            /// if the reference point does not have `m` values.
            pub fn new(ref_point: &[f64], y: Option<&$crate::Matrix>) -> $crate::Result<Self> {
                let y = y.map($crate::utils::multi_objective::PointBatch::from_matrix);
                Self::from_points(ref_point, y.as_ref())
            }

            /// A batch of two-objective decompositions, one per matrix (all
            /// of the same shape); `ref_point` has 2 values, or 2 per batch
            /// element.
            ///
            /// # Errors
            /// As `new`;
            /// [`BotorchError::Unsupported`](crate::BotorchError::Unsupported)
            /// for more than two objectives (as in BoTorch).
            pub fn new_batched(ref_point: &[f64], ys: &[$crate::Matrix]) -> $crate::Result<Self> {
                let y = $crate::utils::multi_objective::PointBatch::from_matrices(ys)?;
                Self::from_points(ref_point, Some(&y))
            }

            #[doc = concat!("`", $name, "(ref_point, Y)` for outcomes of shape `(batch_shape) x n x m`; `ref_point` has `m` values, or `b·m` for a batch of `b` frontiers.")]
            ///
            /// # Errors
            /// As `new_batched`.
            pub fn from_points(
                ref_point: &[f64],
                y: Option<&$crate::utils::multi_objective::PointBatch>,
            ) -> $crate::Result<Self> {
                let shape = Core::infer_ref_shape(ref_point, y);
                Self::with_ref_point_shape(ref_point, &shape, y)
            }

            /// Like `from_points` with an explicit shape of the reference
            /// point (`[m]` or `[*batch_shape, m]`).
            ///
            /// # Errors
            /// As `new_batched`.
            pub fn with_ref_point_shape(
                ref_point: &[f64],
                ref_shape: &[usize],
                y: Option<&$crate::utils::multi_objective::PointBatch>,
            ) -> $crate::Result<Self> {
                Ok(Self(FastCore::new($kind, ref_point, ref_shape, y)?))
            }

            /// `update(Y)` for an unbatched `n × m` matrix.
            ///
            /// # Errors
            /// As [`BoxDecomposition::update`].
            pub fn update_matrix(&mut self, y: &$crate::Matrix) -> $crate::Result<()> {
                self.update(&$crate::utils::multi_objective::PointBatch::from_matrix(y))
            }
        }

        impl BoxDecomposition for $ty {
            fn class_name(&self) -> &'static str {
                $name
            }

            fn ref_point(&self) -> &[f64] {
                &self.0.core.ref_values
            }

            fn ref_point_shape(&self) -> &[usize] {
                &self.0.core.ref_shape
            }

            fn num_outcomes(&self) -> usize {
                self.0.core.num_outcomes
            }

            fn batch_shape(&self) -> Option<&[usize]> {
                self.0.core.batch_shape.as_deref()
            }

            fn y(&self) -> Option<&$crate::utils::multi_objective::PointBatch> {
                self.0.core.y.as_ref()
            }

            fn pareto_y(&self) -> Option<$crate::utils::multi_objective::PointBatch> {
                self.0.core.pareto_y()
            }

            fn update(
                &mut self,
                y: &$crate::utils::multi_objective::PointBatch,
            ) -> $crate::Result<()> {
                self.0.update(y)
            }

            fn reset(&mut self) -> $crate::Result<()> {
                self.0.reset()
            }

            fn partition_space(&mut self) -> $crate::Result<()> {
                self.0.partition_space()
            }

            fn get_hypercell_bounds(&self) -> Option<&CellBounds> {
                self.0.cells.as_ref()
            }

            fn compute_hypervolume(&self) -> Vec<f64> {
                self.0.hypervolume()
            }
        }

        impl $crate::utils::multi_objective::box_decompositions::FastPartitioning for $ty {
            fn local_upper_bounds(
                &self,
            ) -> Option<&$crate::utils::multi_objective::box_decompositions::LocalUpperBounds>
            {
                self.0.lub.as_ref()
            }
        }
    };
}

pub(super) use impl_fast_partitioning;
