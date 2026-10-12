//! Helpers of the box decompositions (`box_decompositions/utils.py`): local
//! upper bounds (Lacour et al., 2017, Algorithm 1), the cell bounds of
//! Equation 2, the direct two-objective decompositions and the padding of
//! batched Pareto frontiers.

use super::box_decomposition::CellBounds;
use crate::error::{BotorchError, Result};
use crate::utils::multi_objective::pareto::is_non_dominated_rows;
use crate::utils::multi_objective::{nan_max, torch_size, PointBatch};

/// BoTorch's `_expand_ref_point(ref_point, batch_shape)`: the reference point
/// (of shape `ref_shape`, `[m]` or `[*batch_shape, m]`) for every batch
/// element, `prod(batch_shape)·m` values.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] if a batched reference point does not
/// have the batch shape (BoTorch's message).
pub fn expand_ref_point(
    ref_point: &[f64],
    ref_shape: &[usize],
    batch_shape: &[usize],
) -> Result<Vec<f64>> {
    let Some((_, ref_batch)) = ref_shape.split_last() else {
        return Err(BotorchError::DimensionMismatch(
            "the reference point must have at least one dimension".into(),
        ));
    };
    if ref_batch == batch_shape {
        return Ok(ref_point.to_vec());
    }
    if ref_shape.len() > 1 {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected ref_point to be a `batch_shape x m` or `m`-dim tensor, but got {}.",
            torch_size(ref_shape)
        )));
    }
    let b: usize = batch_shape.iter().product();
    Ok(ref_point.repeat(b))
}

/// The reference point of each of `b` batch elements: `m` values shared by
/// all, or `b·m` values.
fn batch_refs(ref_point: &[f64], b: usize, m: usize) -> Result<Vec<f64>> {
    if ref_point.len() == m {
        Ok(ref_point.repeat(b))
    } else if ref_point.len() == b * m {
        Ok(ref_point.to_vec())
    } else {
        Err(BotorchError::DimensionMismatch(format!(
            "Expected a reference point with {m} or {} values, got {}.",
            b * m,
            ref_point.len()
        )))
    }
}

/// BoTorch's `_pad_batch_pareto_frontier(Y, ref_point, is_pareto,
/// feasibility_mask)` (maximization): the Pareto points of each batch
/// element that are better than its reference point (`m` or `b·m` values),
/// in their original order. Unbatched, that is the frontier itself; in a
/// batch, every frontier is padded to the largest size by repeating its last
/// point, and an empty frontier is filled with the reference point.
/// Infeasible points (`feasibility_mask[b·n + i] == false`) are first
/// replaced by the reference point; with `is_pareto` all points are taken to
/// be non-dominated.
///
/// # Errors
/// [`BotorchError::Unsupported`] for more than one batch dimension,
/// [`BotorchError::DimensionMismatch`] for inconsistent sizes.
pub fn pad_batch_pareto_frontier(
    y: &PointBatch,
    ref_point: &[f64],
    is_pareto: bool,
    feasibility_mask: Option<&[bool]>,
) -> Result<PointBatch> {
    let batch_shape = y.batch_shape();
    if batch_shape.len() > 1 {
        return Err(BotorchError::Unsupported(format!(
            "_pad_batch_pareto_frontier only supports a single batch dimension, but got {} \
             batch dimensions.",
            batch_shape.len()
        )));
    }
    let (b, n, m) = (y.batch_size(), y.n(), y.m());
    let refs = batch_refs(ref_point, b, m)?;
    let mut values = y.values().to_vec();
    if let Some(feasible) = feasibility_mask {
        if feasible.len() != b * n {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected a feasibility mask with {} values, got {}.",
                b * n,
                feasible.len()
            )));
        }
        // Infeasible points become the reference point of their batch element.
        for k in 0..b {
            for i in 0..n {
                if !feasible[k * n + i] {
                    let start = (k * n + i) * m;
                    values[start..start + m].copy_from_slice(&refs[k * m..(k + 1) * m]);
                }
            }
        }
    }
    let mut frontiers: Vec<Vec<usize>> = Vec::with_capacity(b);
    for k in 0..b {
        let block = &values[k * n * m..(k + 1) * n * m];
        let r = &refs[k * m..(k + 1) * m];
        let pareto = if is_pareto {
            vec![true; n]
        } else {
            is_non_dominated_rows(block, n, m, true, true)
        };
        frontiers.push(
            (0..n)
                .filter(|&i| {
                    pareto[i] && block[i * m..(i + 1) * m].iter().zip(r).all(|(v, r)| v > r)
                })
                .collect(),
        );
    }
    let point = |k: usize, i: usize| &values[(k * n + i) * m..(k * n + i + 1) * m];
    if batch_shape.is_empty() {
        let idx = &frontiers[0];
        let out: Vec<f64> = idx
            .iter()
            .flat_map(|&i| point(0, i).iter().copied())
            .collect();
        return PointBatch::from_rows(idx.len(), m, out);
    }
    let max_n = frontiers.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = Vec::with_capacity(b * max_n * m);
    for (k, idx) in frontiers.iter().enumerate() {
        match idx.last() {
            Some(&last) => {
                for &i in idx {
                    out.extend_from_slice(point(k, i));
                }
                for _ in idx.len()..max_n {
                    out.extend_from_slice(point(k, last));
                }
            }
            None => {
                for _ in 0..max_n {
                    out.extend_from_slice(&refs[k * m..(k + 1) * m]);
                }
            }
        }
    }
    PointBatch::new(batch_shape.to_vec(), max_n, m, out)
}

/// Local upper bounds and their defining points (BoTorch's `U` and `Z`) of a
/// minimization problem: `U` holds `len` points of `m` values, and `Z[u, j]`
/// is the defining point `z^j(u)` of local upper bound `u` for outcome `j`
/// (Lacour et al., 2017, Sec. 2.1). The order of the bounds is BoTorch's.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalUpperBounds {
    m: usize,
    u: Vec<f64>,
    z: Vec<f64>,
}

impl LocalUpperBounds {
    /// Bounds `u` (`len·m` values, row-major) with defining points `z`
    /// (`len·m·m` values: `z[(i·m + j)·m + k]` is coordinate `k` of
    /// `z^j(u_i)`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent lengths.
    pub fn new(m: usize, u: Vec<f64>, z: Vec<f64>) -> Result<Self> {
        if m == 0 || !u.len().is_multiple_of(m) || z.len() != u.len() * m {
            return Err(BotorchError::DimensionMismatch(format!(
                "local upper bounds with {} values and defining points with {} values do not \
                 fit {m} outcomes",
                u.len(),
                z.len()
            )));
        }
        Ok(Self { m, u, z })
    }

    /// The initial bounds of a minimization problem with reference point
    /// `upper` (BoTorch's `FastPartitioning._partition_space`): `U = {upper}`
    /// with the dummy defining points `z^j = (-inf, ..., upper_j, ..., -inf)`.
    pub fn initial(upper: &[f64]) -> Self {
        let m = upper.len();
        let mut z = vec![f64::NEG_INFINITY; m * m];
        for j in 0..m {
            z[j * m + j] = upper[j];
        }
        Self {
            m,
            u: upper.to_vec(),
            z,
        }
    }

    /// The number of outcomes.
    pub fn m(&self) -> usize {
        self.m
    }

    /// The number of local upper bounds.
    pub fn len(&self) -> usize {
        self.u.len() / self.m
    }

    /// Whether there are no local upper bounds.
    pub fn is_empty(&self) -> bool {
        self.u.is_empty()
    }

    /// All local upper bounds, row-major (`len x m`; BoTorch's `U`).
    pub fn upper_bounds(&self) -> &[f64] {
        &self.u
    }

    /// All defining points, row-major (`len x m x m`; BoTorch's `Z`).
    pub fn defining_points(&self) -> &[f64] {
        &self.z
    }

    /// Local upper bound `i`.
    pub fn upper_bound(&self, i: usize) -> &[f64] {
        &self.u[i * self.m..(i + 1) * self.m]
    }

    /// The defining point `z^j(u_i)`.
    pub fn defining_point(&self, i: usize, j: usize) -> &[f64] {
        let start = (i * self.m + j) * self.m;
        &self.z[start..start + self.m]
    }

    /// Inserts the point `z` (BoTorch's `compute_local_upper_bounds`; the
    /// incremental Algorithm 1 of Lacour et al., 2017): every bound `u > z`
    /// is replaced by the bounds `(u_1, ..., z_j, ..., u_m)` for which `z_j`
    /// is at least the `j`-th coordinate of the other defining points of `u`.
    /// The bounds that are not dominated by `z` keep their order and are
    /// followed by the new bounds, grouped by `j`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `z` does not have `m` values;
    /// [`BotorchError::InvalidArgument`] for `m = 1` when a bound changes
    /// (where BoTorch's reduction over the other outcomes fails).
    pub fn insert(&mut self, z: &[f64]) -> Result<()> {
        let m = self.m;
        if z.len() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected a point with {m} values, got {}.",
                z.len()
            )));
        }
        let n = self.len();
        let mm = m * m;
        // The search zones that contain z: the bounds u > z.
        let dominated: Vec<usize> = (0..n)
            .filter(|&i| self.upper_bound(i).iter().zip(z).all(|(u, z)| u > z))
            .collect();
        if dominated.is_empty() {
            return Ok(());
        }
        if m < 2 {
            return Err(BotorchError::InvalidArgument(
                "max(): Expected reduction dim 1 to have non-zero size.".into(),
            ));
        }
        let a_u: Vec<f64> = dominated
            .iter()
            .flat_map(|&i| self.upper_bound(i).iter().copied())
            .collect();
        let a_z: Vec<f64> = dominated
            .iter()
            .flat_map(|&i| self.z[i * mm..(i + 1) * mm].iter().copied())
            .collect();
        // Keep the other bounds in their order (compacted in place).
        let mut kept = 0;
        let mut next = dominated.iter().peekable();
        for i in 0..n {
            if next.peek() == Some(&&i) {
                next.next();
                continue;
            }
            if kept != i {
                self.u.copy_within(i * m..(i + 1) * m, kept * m);
                self.z.copy_within(i * mm..(i + 1) * mm, kept * mm);
            }
            kept += 1;
        }
        self.u.truncate(kept * m);
        self.z.truncate(kept * mm);
        // Followed by the new bounds, grouped by j.
        for j in 0..m {
            for (u, zu) in a_u.chunks_exact(m).zip(a_z.chunks_exact(mm)) {
                // max_{k != j} z^k(u)_j
                let z_uj_max = (0..m)
                    .filter(|&k| k != j)
                    .map(|k| zu[k * m + j])
                    .fold(f64::NEG_INFINITY, nan_max);
                if z[j] >= z_uj_max {
                    let start = self.u.len();
                    self.u.extend_from_slice(u);
                    self.u[start + j] = z[j];
                    for k in 0..m {
                        if k == j {
                            self.z.extend_from_slice(z);
                        } else {
                            self.z.extend_from_slice(&zu[k * m..(k + 1) * m]);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// BoTorch's `compute_local_upper_bounds(U, Z, z)`: the bounds after
/// inserting `z` (see [`LocalUpperBounds::insert`]).
///
/// # Errors
/// As [`LocalUpperBounds::insert`].
pub fn compute_local_upper_bounds(
    bounds: &LocalUpperBounds,
    z: &[f64],
) -> Result<LocalUpperBounds> {
    let mut out = bounds.clone();
    out.insert(z)?;
    Ok(out)
}

/// BoTorch's `update_local_upper_bounds_incremental(new_pareto_Y, U, Z)`:
/// inserts the new Pareto points (an unbatched `n x m` point set, in order).
///
/// # Errors
/// As [`LocalUpperBounds::insert`].
pub fn update_local_upper_bounds_incremental(
    new_pareto_y: &PointBatch,
    bounds: &LocalUpperBounds,
) -> Result<LocalUpperBounds> {
    let mut out = bounds.clone();
    for k in 0..new_pareto_y.batch_size() {
        for i in 0..new_pareto_y.n() {
            out.insert(new_pareto_y.point(k, i))?;
        }
    }
    Ok(out)
}

/// BoTorch's `get_partition_bounds(Z, U, ref_point)`: the cells of the
/// dominated space of a minimization problem (Lacour et al., 2017,
/// Equation 2), one per local upper bound `u` (in order): lower vertex
/// `(z^1(u)_1, max_{k<2} z^k(u)_2, ..., max_{k<m} z^k(u)_m)` and upper vertex
/// `(ref_1, u_2, ..., u_m)`, without the empty cells.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] if `ref_point` does not have `m`
/// values.
pub fn get_partition_bounds(bounds: &LocalUpperBounds, ref_point: &[f64]) -> Result<CellBounds> {
    let m = bounds.m();
    if ref_point.len() != m {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected a reference point with {m} values, got {}.",
            ref_point.len()
        )));
    }
    let mut lower = Vec::with_capacity(bounds.u.len());
    let mut upper = Vec::with_capacity(bounds.u.len());
    let mut lo = vec![0.0; m];
    let mut up = vec![0.0; m];
    for i in 0..bounds.len() {
        lo[0] = bounds.defining_point(i, 0)[0];
        up[0] = ref_point[0];
        for j in 1..m {
            lo[j] = (0..j)
                .map(|k| bounds.defining_point(i, k)[j])
                .fold(f64::NEG_INFINITY, nan_max);
            up[j] = bounds.upper_bound(i)[j];
        }
        // Remove empty cells (also those with lower = upper = -inf).
        if !lo.iter().zip(&up).any(|(l, u)| u <= l) {
            lower.extend_from_slice(&lo);
            upper.extend_from_slice(&up);
        }
    }
    let num_cells = lower.len() / m;
    CellBounds::new(Vec::new(), num_cells, m, lower, upper)
}

/// Checks the input of the two-objective decompositions and returns the
/// reference point of every batch element.
fn check_2d(pareto_y_sorted: &PointBatch, ref_point: &[f64]) -> Result<Vec<f64>> {
    if pareto_y_sorted.m() != 2 {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected a frontier with 2 outcomes, got {}.",
            pareto_y_sorted.m()
        )));
    }
    if pareto_y_sorted.n() == 0 {
        return Err(BotorchError::InvalidArgument(
            "The Pareto frontier must have at least one point.".into(),
        ));
    }
    batch_refs(ref_point, pareto_y_sorted.batch_size(), 2)
}

/// BoTorch's `compute_non_dominated_hypercell_bounds_2d(pareto_Y_sorted,
/// ref_point)`: the `n + 1` cells of the non-dominated space of a frontier
/// of two outcomes (maximization) sorted by increasing first outcome, all
/// better than the reference point (2 values, or 2 per batch element). Cell
/// `k` spans `[P_{k-1,0}, P_{k,0}] x [P_{k,1}, inf]` with `P_{-1,0} = r_0`,
/// `P_{n,0} = inf` and `P_{n,1} = r_1`.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] or [`BotorchError::InvalidArgument`]
/// for frontiers that are not two-dimensional or empty.
pub fn compute_non_dominated_hypercell_bounds_2d(
    pareto_y_sorted: &PointBatch,
    ref_point: &[f64],
) -> Result<CellBounds> {
    let refs = check_2d(pareto_y_sorted, ref_point)?;
    let (b, n) = (pareto_y_sorted.batch_size(), pareto_y_sorted.n());
    let mut lower = Vec::with_capacity(b * (n + 1) * 2);
    let mut upper = Vec::with_capacity(b * (n + 1) * 2);
    for k in 0..b {
        let p = |i: usize| pareto_y_sorted.point(k, i);
        let r = &refs[2 * k..2 * k + 2];
        // The front with the boundary points (r_0, P_{0,1}) and (P_{n-1,0}, r_1).
        let front = |t: usize| -> [f64; 2] {
            if t == 0 {
                [r[0], p(0)[1]]
            } else if t <= n {
                [p(t - 1)[0], p(t - 1)[1]]
            } else {
                [p(n - 1)[0], r[1]]
            }
        };
        for c in 0..=n {
            lower.extend([front(c)[0], front(c + 1)[1]]);
            let x = if c < n {
                front(c + 1)[0]
            } else {
                f64::INFINITY
            };
            upper.extend([x, f64::INFINITY]);
        }
    }
    CellBounds::new(
        pareto_y_sorted.batch_shape().to_vec(),
        n + 1,
        2,
        lower,
        upper,
    )
}

/// BoTorch's `compute_dominated_hypercell_bounds_2d(pareto_Y_sorted,
/// ref_point)`: the `n` cells of the dominated space of a frontier of two
/// outcomes (maximization) sorted by increasing first outcome; cell `k`
/// spans `[P_{k-1,0}, P_{k,0}] x [r_1, P_{k,1}]` with `P_{-1,0} = r_0`.
///
/// # Errors
/// As [`compute_non_dominated_hypercell_bounds_2d`].
pub fn compute_dominated_hypercell_bounds_2d(
    pareto_y_sorted: &PointBatch,
    ref_point: &[f64],
) -> Result<CellBounds> {
    let refs = check_2d(pareto_y_sorted, ref_point)?;
    let (b, n) = (pareto_y_sorted.batch_size(), pareto_y_sorted.n());
    let mut lower = Vec::with_capacity(b * n * 2);
    let mut upper = Vec::with_capacity(b * n * 2);
    for k in 0..b {
        let p = |i: usize| pareto_y_sorted.point(k, i);
        let r = &refs[2 * k..2 * k + 2];
        for c in 0..n {
            let x = if c == 0 { r[0] } else { p(c - 1)[0] };
            lower.extend([x, r[1]]);
            upper.extend_from_slice(p(c));
        }
    }
    CellBounds::new(pareto_y_sorted.batch_shape().to_vec(), n, 2, lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_upper_bounds_of_two_points() {
        // Minimization with reference point (10, 10, 10).
        let mut lub = LocalUpperBounds::initial(&[10.0, 10.0, 10.0]);
        lub.insert(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(lub.len(), 3);
        assert_eq!(
            lub.upper_bounds(),
            &[1.0, 10.0, 10.0, 10.0, 2.0, 10.0, 10.0, 10.0, 3.0]
        );
        assert_eq!(lub.defining_point(1, 1), &[1.0, 2.0, 3.0]);
        assert_eq!(lub.defining_point(1, 0)[1], f64::NEG_INFINITY);
        // A dominated point does not change the bounds.
        let before = lub.clone();
        lub.insert(&[2.0, 3.0, 4.0]).unwrap();
        assert_eq!(lub, before);
        // The dominated space of a single point (min-space reference 10).
        let cells = get_partition_bounds(&lub, &[10.0, 10.0, 10.0]).unwrap();
        let vol: f64 = cells.volumes()[0];
        assert_eq!(vol, 9.0 * 8.0 * 7.0);
        assert!(LocalUpperBounds::initial(&[1.0]).insert(&[0.0]).is_err());
    }

    #[test]
    fn padding_and_expansion() {
        let y = PointBatch::new(
            vec![2],
            3,
            2,
            vec![
                1.0, 2.0, 2.0, 1.0, 0.5, 0.5, // two Pareto points
                -1.0, 3.0, 0.2, 0.1, 0.1, 0.2, // (0.2, 0.1) and (0.1, 0.2)
            ],
        )
        .unwrap();
        let padded = pad_batch_pareto_frontier(&y, &[0.0, 0.0], false, None).unwrap();
        assert_eq!(padded.shape(), vec![2, 2, 2]);
        assert_eq!(padded.values(), &[1.0, 2.0, 2.0, 1.0, 0.2, 0.1, 0.1, 0.2]);
        let mask = [true, false, true, true, true, false];
        let padded =
            pad_batch_pareto_frontier(&y, &[0.0, 0.0, 0.5, 0.5], false, Some(&mask)).unwrap();
        // Batch 0: only (1, 2) (the infeasible (2, 1) became the reference
        // point); batch 1: no point better than (0.5, 0.5).
        assert_eq!(padded.shape(), vec![2, 1, 2]);
        assert_eq!(padded.values(), &[1.0, 2.0, 0.5, 0.5]);
        assert_eq!(
            expand_ref_point(&[1.0, 2.0], &[2], &[3]).unwrap(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
        let err = expand_ref_point(&[1.0, 2.0], &[1, 2], &[]).unwrap_err();
        assert_eq!(
            err,
            BotorchError::DimensionMismatch(
                "Expected ref_point to be a `batch_shape x m` or `m`-dim tensor, but got \
                 torch.Size([1, 2])."
                    .into()
            )
        );
    }

    #[test]
    fn two_objective_cells() {
        let p = PointBatch::from_rows(3, 2, vec![3.0, 7.0, 5.0, 5.0, 8.0, 0.0]).unwrap();
        let nd = compute_non_dominated_hypercell_bounds_2d(&p, &[0.0, 0.0]).unwrap();
        assert_eq!(nd.num_cells(), 4);
        assert_eq!(nd.cell(0, 0), (&[0.0, 7.0][..], &[3.0, f64::INFINITY][..]));
        assert_eq!(nd.cell(0, 3), (&[8.0, 0.0][..], &[f64::INFINITY; 2][..]));
        let d = compute_dominated_hypercell_bounds_2d(&p, &[0.0, 0.0]).unwrap();
        assert_eq!(d.volumes(), vec![3.0 * 7.0 + 2.0 * 5.0]);
        assert!(compute_dominated_hypercell_bounds_2d(
            &PointBatch::from_rows(0, 2, vec![]).unwrap(),
            &[0.0, 0.0]
        )
        .is_err());
    }
}
