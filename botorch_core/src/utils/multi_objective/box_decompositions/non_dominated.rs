//! Partitionings of the non-dominated space (`non_dominated.py`):
//! [`NondominatedPartitioning`] and [`FastNondominatedPartitioning`].

use super::box_decomposition::{
    impl_fast_partitioning, prod, sum, BoxDecomposition, CellBounds, Core, FastCore, FastKind,
};
use crate::error::Result;
use crate::utils::multi_objective::PointBatch;
use crate::Matrix;

/// The binary partitioning of the non-dominated space of Couckuyt et al.
/// (2012) (BoTorch's `NondominatedPartitioning`; maximization).
///
/// The space between the ideal and the anti-ideal point is split
/// recursively along the coordinates of the Pareto points: a cell whose
/// upper vertex is not dominated by any Pareto point is accepted, a cell
/// that overlaps the frontier is halved along its longest edge (in sorted
/// index units), and dominated cells are discarded. With `alpha > 0` cells
/// whose volume is at most `alpha` times the volume of the bounding box are
/// not split further and are dropped (an approximation that underestimates
/// the non-dominated space; Watanabe, 2025). The cells are visited in
/// BoTorch's depth-first order, so they come out in BoTorch's order. With two
/// objectives the `n + 1` cells are built directly (also for a batch of
/// frontiers).
///
/// The bounds of the outer cells are `inf` (non-dominated side) and the
/// reference point. Exact partitionings are only practical for `m < 5`
/// objectives; [`FastNondominatedPartitioning`] is much faster.
#[derive(Clone, Debug)]
pub struct NondominatedPartitioning {
    core: Core,
    alpha: f64,
    cells: Option<NdCells>,
}

/// The cells of a [`NondominatedPartitioning`] as positions in the sorted,
/// augmented coordinates of the frontier (BoTorch's `hypercells`, which
/// index the rows of the augmented frontier sorted per outcome; positions
/// are independent of how ties are ordered).
#[derive(Clone, Debug)]
struct NdCells {
    num_cells: usize,
    n_p: usize,
    m: usize,
    /// `(b, num_cells, m)` positions of the lower and upper vertices:
    /// position 0 is the ideal side, `1..=n_p` the sorted coordinates of the
    /// frontier and `n_p + 1` the reference point.
    lower: Vec<u32>,
    upper: Vec<u32>,
    /// `(b, m, n_p)`: the negated frontier sorted per outcome.
    sorted: Vec<f64>,
    bounds: CellBounds,
}

impl NdCells {
    fn sorted_col(&self, k: usize, j: usize) -> &[f64] {
        let start = (k * self.m + j) * self.n_p;
        &self.sorted[start..start + self.n_p]
    }
}

/// The columns of each batch element of a negated frontier, sorted.
fn sorted_columns(neg: &PointBatch) -> Vec<f64> {
    let (b, n_p, m) = (neg.batch_size(), neg.n(), neg.m());
    let mut out = Vec::with_capacity(b * m * n_p);
    for k in 0..b {
        for j in 0..m {
            let start = out.len();
            out.extend((0..n_p).map(|i| neg.point(k, i)[j]));
            out[start..].sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    out
}

impl NondominatedPartitioning {
    /// `NondominatedPartitioning(ref_point, Y, alpha)` for an unbatched
    /// `n × m` matrix of outcomes (or none).
    ///
    /// # Errors
    /// [`BotorchError::InputData`](crate::BotorchError::InputData) for NaN
    /// outcomes, [`BotorchError::DimensionMismatch`](crate::BotorchError::DimensionMismatch)
    /// if the reference point does not have `m` values.
    pub fn new(ref_point: &[f64], y: Option<&Matrix>, alpha: f64) -> Result<Self> {
        let y = y.map(PointBatch::from_matrix);
        Self::from_points(ref_point, y.as_ref(), alpha)
    }

    /// A batch of two-objective decompositions, one per matrix (all of the
    /// same shape); `ref_point` has 2 values, or 2 per batch element.
    ///
    /// # Errors
    /// As [`NondominatedPartitioning::new`];
    /// [`BotorchError::Unsupported`](crate::BotorchError::Unsupported) for
    /// more than two objectives (as in BoTorch).
    pub fn new_batched(ref_point: &[f64], ys: &[Matrix], alpha: f64) -> Result<Self> {
        let y = PointBatch::from_matrices(ys)?;
        Self::from_points(ref_point, Some(&y), alpha)
    }

    /// `NondominatedPartitioning(ref_point, Y, alpha)` for outcomes of shape
    /// `(batch_shape) x n x m`; `ref_point` has `m` values, or `b·m` for a
    /// batch of `b` frontiers.
    ///
    /// # Errors
    /// As [`NondominatedPartitioning::new_batched`].
    pub fn from_points(ref_point: &[f64], y: Option<&PointBatch>, alpha: f64) -> Result<Self> {
        let shape = Core::infer_ref_shape(ref_point, y);
        Self::with_ref_point_shape(ref_point, &shape, y, alpha)
    }

    /// Like [`NondominatedPartitioning::from_points`] with an explicit shape
    /// of the reference point (`[m]` or `[*batch_shape, m]`).
    ///
    /// # Errors
    /// As [`NondominatedPartitioning::new_batched`].
    pub fn with_ref_point_shape(
        ref_point: &[f64],
        ref_shape: &[usize],
        y: Option<&PointBatch>,
        alpha: f64,
    ) -> Result<Self> {
        let mut this = Self {
            core: Core::new("NondominatedPartitioning", ref_point, ref_shape, true)?,
            alpha,
            cells: None,
        };
        if let Some(y) = y {
            this.core.init_outcomes(y)?;
            this.partition_space()?;
        }
        Ok(this)
    }

    /// The approximation parameter `alpha`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// `update(Y)` for an unbatched `n × m` matrix.
    ///
    /// # Errors
    /// As [`BoxDecomposition::update`].
    pub fn update_matrix(&mut self, y: &Matrix) -> Result<()> {
        self.update(&PointBatch::from_matrix(y))
    }

    /// `_partition_space_2d`: cell `c` (`c = 0..=n_p`) spans the sorted
    /// positions `(c, 0)` to `(c + 1, n_p + 1 - c)`.
    fn partition_space_2d(&self, neg: &PointBatch) -> (Vec<u32>, Vec<u32>, usize) {
        let (b, n_p) = (neg.batch_size(), neg.n() as u32);
        let num_cells = neg.n() + 1;
        let mut lower = Vec::with_capacity(b * num_cells * 2);
        let mut upper = Vec::with_capacity(b * num_cells * 2);
        for _ in 0..b {
            for c in 0..=n_p {
                lower.extend([c, 0]);
                upper.extend([c + 1, n_p + 1 - c]);
            }
        }
        (lower, upper, num_cells)
    }

    /// `_partition_space`: Couckuyt's binary partitioning (unbatched).
    fn partition_space_general(
        &self,
        neg: &PointBatch,
        sorted: &[f64],
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let (n_p, m) = (neg.n(), neg.m());
        let top = (n_p + 1) as u32;
        if n_p == 0 {
            // A single cell between -inf and the reference point.
            return (vec![0; m], vec![1; m], 1);
        }
        let col = |j: usize| &sorted[j * n_p..(j + 1) * n_p];
        // Extend the frontier by the ideal and the anti-ideal point.
        let ideal: Vec<f64> = (0..m).map(|j| col(j)[0] - 1.0).collect();
        let anti_ideal: Vec<f64> = (0..m).map(|j| col(j)[n_p - 1] + 1.0).collect();
        let value = |j: usize, pos: u32| -> f64 {
            if pos == 0 {
                ideal[j]
            } else if pos == top {
                anti_ideal[j]
            } else {
                col(j)[pos as usize - 1]
            }
        };
        let total_volume = prod(anti_ideal.iter().zip(&ideal).map(|(a, i)| a - i));
        let points: Vec<&[f64]> = (0..n_p).map(|i| neg.point(0, i)).collect();
        // Cells as `[lower positions; upper positions]` (2m values).
        let mut stack: Vec<u32> = Vec::with_capacity(4 * m);
        stack.extend(std::iter::repeat_n(0, m));
        stack.extend(std::iter::repeat_n(top, m));
        let (mut lower, mut upper) = (Vec::new(), Vec::new());
        let mut lo_v = vec![0.0; m];
        let mut hi_v = vec![0.0; m];
        while stack.len() >= 2 * m {
            let cell: Vec<u32> = stack.split_off(stack.len() - 2 * m);
            let (lo, hi) = cell.split_at(m);
            for j in 0..m {
                lo_v[j] = value(j, lo[j]);
                hi_v[j] = value(j, hi[j]);
            }
            let weakly_beats_all = |v: &[f64]| {
                points
                    .iter()
                    .all(|p| v.iter().zip(p.iter()).any(|(v, p)| v <= p))
            };
            if weakly_beats_all(&hi_v) {
                // The cell is entirely non-dominated.
                lower.extend_from_slice(lo);
                upper.extend_from_slice(hi);
            } else if weakly_beats_all(&lo_v) {
                // The cell overlaps the frontier.
                let any_not_adjacent = lo.iter().zip(hi).any(|(l, h)| h - l > 1);
                let cell_volume = prod(lo_v.iter().zip(&hi_v).map(|(l, h)| h - l));
                // Only divide cells that are not composed of adjacent indices
                // and whose share of the total volume exceeds alpha.
                if any_not_adjacent && cell_volume / total_volume > self.alpha {
                    // The longest edge (in index units; the first of ties).
                    let (mut longest, mut length) = (0, hi[0] - lo[0]);
                    for j in 1..m {
                        if hi[j] - lo[j] > length {
                            longest = j;
                            length = hi[j] - lo[j];
                        }
                    }
                    // Python's round(): halves round to even.
                    let new_length1 = (f64::from(length) / 2.0).round_ties_even() as u32;
                    let new_length2 = length - new_length1;
                    // Cell 1: the lower part, cell 2: the upper part (popped
                    // first).
                    let mut cell1 = cell.clone();
                    cell1[m + longest] -= new_length1;
                    stack.extend_from_slice(&cell1);
                    let mut cell2 = cell;
                    cell2[longest] += new_length2;
                    stack.extend_from_slice(&cell2);
                }
            }
        }
        let num_cells = lower.len() / m;
        (lower, upper, num_cells)
    }

    /// The cell bounds (maximization) from positions:
    /// `get_hypercell_bounds`, with `-inf` and the reference point at the
    /// ends of the sorted coordinates.
    fn bounds_of(&self, cells: &NdCells, refs: &[f64]) -> Result<CellBounds> {
        let (m, n_p, nc) = (cells.m, cells.n_p, cells.num_cells);
        let b = refs.len() / m.max(1);
        let value = |k: usize, j: usize, pos: u32| -> f64 {
            if pos == 0 {
                f64::NEG_INFINITY
            } else if pos as usize == n_p + 1 {
                -refs[k * m + j]
            } else {
                cells.sorted_col(k, j)[pos as usize - 1]
            }
        };
        let mut lower = Vec::with_capacity(b * nc * m);
        let mut upper = Vec::with_capacity(b * nc * m);
        for k in 0..b {
            for c in 0..nc {
                for j in 0..m {
                    let idx = (k * nc + c) * m + j;
                    // Swap the bounds and negate (minimization -> maximization).
                    lower.push(-value(k, j, cells.upper[idx]));
                    upper.push(-value(k, j, cells.lower[idx]));
                }
            }
        }
        let batch_shape = self.core.batch_shape.clone().unwrap_or_default();
        CellBounds::new(batch_shape, nc, m, lower, upper)
    }
}

impl BoxDecomposition for NondominatedPartitioning {
    fn class_name(&self) -> &'static str {
        "NondominatedPartitioning"
    }

    fn ref_point(&self) -> &[f64] {
        &self.core.ref_values
    }

    fn ref_point_shape(&self) -> &[usize] {
        &self.core.ref_shape
    }

    fn num_outcomes(&self) -> usize {
        self.core.num_outcomes
    }

    fn batch_shape(&self) -> Option<&[usize]> {
        self.core.batch_shape.as_deref()
    }

    fn y(&self) -> Option<&PointBatch> {
        self.core.y.as_ref()
    }

    fn pareto_y(&self) -> Option<PointBatch> {
        self.core.pareto_y()
    }

    fn update(&mut self, y: &PointBatch) -> Result<()> {
        self.core.update_y(y)?;
        self.reset()
    }

    fn reset(&mut self) -> Result<()> {
        self.core.validate_inputs()?;
        if self.core.reset_pareto()? {
            self.partition_space()?;
        }
        Ok(())
    }

    fn partition_space(&mut self) -> Result<()> {
        let refs = self.core.refs()?;
        let neg = self.core.neg_pareto_or_err()?;
        let sorted = sorted_columns(neg);
        let (lower, upper, num_cells) = if self.core.num_outcomes == 2 {
            self.partition_space_2d(neg)
        } else {
            self.partition_space_general(neg, &sorted)
        };
        let mut cells = NdCells {
            num_cells,
            n_p: neg.n(),
            m: neg.m(),
            lower,
            upper,
            sorted,
            bounds: CellBounds::new(Vec::new(), 0, 0, Vec::new(), Vec::new())?,
        };
        cells.bounds = self.bounds_of(&cells, &refs)?;
        self.cells = Some(cells);
        Ok(())
    }

    fn get_hypercell_bounds(&self) -> Option<&CellBounds> {
        self.cells.as_ref().map(|c| &c.bounds)
    }

    /// The volume of the box between the ideal point and the reference
    /// point minus the volume of the non-dominated cells within it.
    fn compute_hypervolume(&self) -> Vec<f64> {
        if let Some(hv) = self.core.trivial_hypervolume() {
            return hv;
        }
        let (Some(cells), Ok(refs)) = (&self.cells, self.core.refs()) else {
            return vec![0.0];
        };
        let (m, n_p, nc) = (cells.m, cells.n_p, cells.num_cells);
        let b = refs.len() / m;
        (0..b)
            .map(|k| {
                // Augment with the ideal point and the reference point
                // (minimization).
                let value = |j: usize, pos: u32| -> f64 {
                    if pos == 0 {
                        cells.sorted_col(k, j)[0]
                    } else if pos as usize == n_p + 1 {
                        -refs[k * m + j]
                    } else {
                        cells.sorted_col(k, j)[pos as usize - 1]
                    }
                };
                let total_volume =
                    prod((0..m).map(|j| -refs[k * m + j] - cells.sorted_col(k, j)[0]));
                let non_dom_volume = sum((0..nc).map(|c| {
                    prod((0..m).map(|j| {
                        let idx = (k * nc + c) * m + j;
                        value(j, cells.upper[idx]) - value(j, cells.lower[idx])
                    }))
                }));
                total_volume - non_dom_volume
            })
            .collect()
    }
}

/// The non-dominated space via two applications of Algorithm 1 of Lacour et
/// al. (2017) (BoTorch's `FastNondominatedPartitioning`; maximization):
/// first the local upper bounds of the negated frontier are computed
/// incrementally; their negations, the local lower bounds of the
/// maximization problem, are then treated as the frontier of a minimization
/// problem with the reference point `+inf`, whose dominated space is
/// partitioned (Yang et al., 2019). With two objectives the `n + 1` cells are
/// built directly (also for a batch of frontiers). For more than two
/// objectives, [`BoxDecomposition::update`] inserts new Pareto points
/// incrementally. The cells extend to `+inf`.
#[derive(Clone, Debug)]
pub struct FastNondominatedPartitioning(FastCore);

impl_fast_partitioning!(
    FastNondominatedPartitioning,
    FastKind::NonDominated,
    "FastNondominatedPartitioning"
);
