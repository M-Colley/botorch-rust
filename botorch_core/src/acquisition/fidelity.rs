//! Multi-fidelity helpers of `botorch.acquisition.utils`:
//! [`project_to_target_fidelity`], [`expand_trace_observations`], and the
//! input projections of multi-fidelity acquisition functions
//! ([`InputProjection`]; BoTorch's `project` callables).

use std::collections::BTreeMap;
use std::fmt::Debug;

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// BoTorch's `project_to_target_fidelity(X, target_fidelities, d)`: sets the
/// fidelity columns of the points of `x` to their target values. `x` has
/// either all `d` columns or the `d - d_f` non-fidelity columns, in which
/// case the fidelity columns are inserted. `target_fidelities` maps column
/// indices (negative ones count from the end) to target values.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for indices out of bounds or a
/// number of columns other than `d` and `d - d_f` (BoTorch's messages).
pub fn project_to_target_fidelity(
    x: &Matrix,
    target_fidelities: &[(isize, f64)],
    d: usize,
) -> Result<Matrix> {
    let tfs = normalize_fidelities(target_fidelities, d)?;
    let q = x.nrows();
    if x.ncols() == d {
        Ok(Matrix::from_fn(q, d, |p, i| {
            tfs.get(&i).copied().unwrap_or(x[(p, i)])
        }))
    } else if x.ncols() + target_fidelities.len() == d {
        let mut out = Matrix::zeros(q, d);
        let mut src = 0;
        for i in 0..d {
            match tfs.get(&i) {
                Some(&v) => out.column_mut(i).fill(v),
                None => {
                    out.column_mut(i).copy_from(&x.column(src));
                    src += 1;
                }
            }
        }
        Ok(out)
    } else {
        Err(BotorchError::DimensionMismatch(format!(
            "X must have a last dimension with size `d` or `d-d_f`, but got {}.",
            x.ncols()
        )))
    }
}

/// The target fidelities with non-negative indices (BoTorch's normalization
/// and bounds check).
fn normalize_fidelities(
    target_fidelities: &[(isize, f64)],
    d: usize,
) -> Result<BTreeMap<usize, f64>> {
    let mut tfs = BTreeMap::new();
    let mut out_of_bounds = Vec::new();
    for &(k, v) in target_fidelities {
        let i = if k < 0 { d as isize + k } else { k };
        if i < 0 || i >= d as isize {
            out_of_bounds.push(i);
        } else {
            tfs.insert(i as usize, v);
        }
    }
    if !out_of_bounds.is_empty() {
        out_of_bounds.sort_unstable();
        return Err(BotorchError::DimensionMismatch(format!(
            "Target fidelity indices {out_of_bounds:?} are out of bounds for `d={d}`."
        )));
    }
    Ok(tfs)
}

/// `torch.linspace(start, end, steps)` (computed from both ends like torch).
fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f64;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                start + step * i as f64
            } else {
                end - step * (steps - i - 1) as f64
            }
        })
        .collect()
}

/// BoTorch's `expand_trace_observations(X, fidelity_dims, num_trace_obs)`:
/// appends `num_trace_obs` copies of the `q` points of `x` whose fidelity
/// columns (default: the last one) are scaled by `(K - j) / (K + 1)`,
/// `j = 0..K` (`K = num_trace_obs`), the trace observations obtained while
/// evaluating `x`. Returns `(q + K·q) × d`.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for fidelity indices out of bounds.
pub fn expand_trace_observations(
    x: &Matrix,
    fidelity_dims: Option<&[isize]>,
    num_trace_obs: usize,
) -> Result<Matrix> {
    if num_trace_obs == 0 {
        return Ok(x.clone());
    }
    let (q, d) = x.shape();
    let dims: Vec<usize> = fidelity_dims
        .unwrap_or(&[-1])
        .iter()
        .map(|&k| {
            let i = if k < 0 { d as isize + k } else { k };
            if i < 0 || i >= d as isize {
                Err(BotorchError::DimensionMismatch(format!(
                    "index {k} is out of bounds for dimension with size {d}"
                )))
            } else {
                Ok(i as usize)
            }
        })
        .collect::<Result<_>>()?;
    let s_pad = 1.0 / (num_trace_obs as f64 + 1.0);
    let sf = linspace(1.0 - s_pad, s_pad, num_trace_obs);
    let mut out = Matrix::zeros(q * (1 + num_trace_obs), d);
    for r in 0..out.nrows() {
        let p = r % q;
        for c in 0..d {
            let scale = if r >= q && dims.contains(&c) {
                sf[(r - q) / q]
            } else {
                1.0
            };
            out[(r, c)] = scale * x[(p, c)];
        }
    }
    Ok(out)
}

/// A map of the points of a candidate set (`q × d`) to the points at which
/// an acquisition function evaluates its value function (`q' × d`), with
/// its reverse pass (BoTorch's `project` argument of the multi-fidelity
/// acquisition functions).
pub trait InputProjection: Send + Sync + Debug {
    /// The projected points.
    ///
    /// # Errors
    /// Implementation-specific.
    fn project(&self, x: &Matrix) -> Result<Matrix>;

    /// The gradient with respect to `x` of `⟨out_bar, project(x)⟩`.
    ///
    /// # Errors
    /// Implementation-specific.
    fn backward(&self, x: &Matrix, out_bar: &Matrix) -> Result<Matrix>;
}

/// The identity (BoTorch's default `project=lambda X: X`).
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityProjection;

impl InputProjection for IdentityProjection {
    fn project(&self, x: &Matrix) -> Result<Matrix> {
        Ok(x.clone())
    }

    fn backward(&self, _x: &Matrix, out_bar: &Matrix) -> Result<Matrix> {
        Ok(out_bar.clone())
    }
}

/// [`project_to_target_fidelity`] as a projection of points with all `d`
/// columns.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetFidelityProjection {
    target_fidelities: Vec<(isize, f64)>,
    d: usize,
}

impl TargetFidelityProjection {
    /// The projection onto the target fidelities of `d`-dimensional points.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for indices out of bounds.
    pub fn new(target_fidelities: Vec<(isize, f64)>, d: usize) -> Result<Self> {
        normalize_fidelities(&target_fidelities, d)?;
        Ok(Self {
            target_fidelities,
            d,
        })
    }

    /// The target fidelities by (non-negative) column.
    pub fn target_fidelities(&self) -> BTreeMap<usize, f64> {
        normalize_fidelities(&self.target_fidelities, self.d).expect("validated")
    }
}

impl InputProjection for TargetFidelityProjection {
    fn project(&self, x: &Matrix) -> Result<Matrix> {
        project_to_target_fidelity(x, &self.target_fidelities, self.d)
    }

    fn backward(&self, x: &Matrix, out_bar: &Matrix) -> Result<Matrix> {
        let tfs = normalize_fidelities(&self.target_fidelities, self.d)?;
        if x.ncols() != self.d {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected points with {} columns, got {}",
                self.d,
                x.ncols()
            )));
        }
        Ok(Matrix::from_fn(x.nrows(), self.d, |p, i| {
            if tfs.contains_key(&i) {
                0.0
            } else {
                out_bar[(p, i)]
            }
        }))
    }
}

/// [`expand_trace_observations`] as a map of candidate sets (BoTorch's
/// `expand` argument of `qMultiFidelityKnowledgeGradient`, typically
/// `lambda X: expand_trace_observations(X, fidelity_dims, num_trace_obs)`):
/// `q` points become `q (1 + num_trace_obs)` points.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceObservationExpansion {
    fidelity_dims: Option<Vec<isize>>,
    num_trace_obs: usize,
}

impl TraceObservationExpansion {
    /// The expansion with `num_trace_obs` trace observations whose fidelity
    /// columns `fidelity_dims` (default: the last column) are scaled.
    pub fn new(fidelity_dims: Option<Vec<isize>>, num_trace_obs: usize) -> Self {
        Self {
            fidelity_dims,
            num_trace_obs,
        }
    }
}

impl InputProjection for TraceObservationExpansion {
    fn project(&self, x: &Matrix) -> Result<Matrix> {
        expand_trace_observations(x, self.fidelity_dims.as_deref(), self.num_trace_obs)
    }

    fn backward(&self, x: &Matrix, out_bar: &Matrix) -> Result<Matrix> {
        let (q, d) = x.shape();
        let k = self.num_trace_obs;
        if out_bar.shape() != (q * (1 + k), d) {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected an adjoint of shape {}x{d}, got {}x{}",
                q * (1 + k),
                out_bar.nrows(),
                out_bar.ncols()
            )));
        }
        // The expansion is linear: the adjoint of x is the expansion of a
        // matrix of ones applied to the adjoint, row by row.
        let ones = Matrix::from_element(q, d, 1.0);
        let scale = expand_trace_observations(&ones, self.fidelity_dims.as_deref(), k)?;
        let mut out = Matrix::zeros(q, d);
        for r in 0..q * (1 + k) {
            for c in 0..d {
                out[(r % q, c)] += scale[(r, c)] * out_bar[(r, c)];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_and_trace_observations() {
        let x = Matrix::from_row_slice(2, 3, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let p = project_to_target_fidelity(&x, &[(-1, 1.0)], 3).unwrap();
        assert_eq!(
            p.column(2).iter().copied().collect::<Vec<_>>(),
            vec![1.0, 1.0]
        );
        assert_eq!(p[(1, 1)], 0.5);
        let short = Matrix::from_row_slice(1, 2, &[0.1, 0.2]);
        let p = project_to_target_fidelity(&short, &[(1, 0.9)], 3).unwrap();
        assert_eq!(p.as_slice(), &[0.1, 0.9, 0.2]);
        assert!(project_to_target_fidelity(&x, &[(3, 1.0)], 3).is_err());
        assert!(project_to_target_fidelity(&short, &[(0, 1.0)], 4).is_err());
        let e = expand_trace_observations(&x, None, 2).unwrap();
        assert_eq!(e.shape(), (6, 3));
        assert!((e[(2, 2)] - 0.3 * 2.0 / 3.0).abs() < 1e-15);
        assert!((e[(5, 2)] - 0.6 / 3.0).abs() < 1e-15);
        assert_eq!(e[(5, 1)], 0.5);
        assert_eq!(linspace(0.75, 0.25, 3), vec![0.75, 0.5, 0.25]);
        let tf = TargetFidelityProjection::new(vec![(0, 0.5)], 3).unwrap();
        let g = tf.backward(&x, &Matrix::from_element(2, 3, 1.0)).unwrap();
        assert_eq!(g.column(0).sum(), 0.0);
        assert_eq!(g.column(1).sum(), 2.0);
    }
}
