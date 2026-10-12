//! Multi-objective utilities (`botorch.utils.multi_objective`): Pareto
//! masks, hypervolumes, box decompositions and the Chebyshev scalarization;
//! plus `sample_simplex` and `sample_hypersphere` (`botorch.utils.sampling`).
//!
//! The box decompositions are stateful: [`PyBoxDecomposition`] keeps the
//! Rust decomposition behind a mutex that is locked only while the GIL is
//! released.

use std::sync::Mutex;

use botorch_core::utils::multi_objective::box_decompositions::utils::{
    compute_local_upper_bounds, expand_ref_point, get_partition_bounds, pad_batch_pareto_frontier,
    update_local_upper_bounds_incremental,
};
use botorch_core::utils::multi_objective::box_decompositions::{
    compute_dominated_hypercell_bounds_2d, compute_non_dominated_hypercell_bounds_2d,
    BoxDecomposition, CellBounds, DominatedPartitioning, FastNondominatedPartitioning,
    FastPartitioning, LocalUpperBounds, NondominatedPartitioning,
};
use botorch_core::utils::multi_objective::hypervolume::{
    get_hypervolume_maximizing_subset, infer_reference_point, Hypervolume,
};
use botorch_core::utils::multi_objective::pareto::is_non_dominated_batch;
use botorch_core::utils::multi_objective::scalarization::ChebyshevScalarization;
use botorch_core::utils::multi_objective::PointBatch;
use botorch_core::utils::sampling::{
    hypersphere_from_normal, sample_hypersphere, sample_simplex, simplex_from_uniform,
};
use botorch_core::{BotorchError, Matrix};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{dimension_error, to_pyerr, value_error};
use crate::threads;

/// torch's error for a reduction over an empty dimension, which BoTorch's
/// local upper bounds raise for a single outcome.
const EMPTY_REDUCTION: &str = "max(): Expected reduction dim 1 to have non-zero size.";

/// Maps core errors like [`crate::errors::IntoPyResult`], except torch's
/// empty-reduction error, which is an `IndexError` as in BoTorch.
fn mo_result<T>(r: botorch_core::Result<T>) -> PyResult<T> {
    r.map_err(|e| match e {
        BotorchError::InvalidArgument(msg) if msg == EMPTY_REDUCTION => PyIndexError::new_err(msg),
        e => to_pyerr(e),
    })
}

/// An array of shape `(*batch, n, m)` (at least two dimensions) as a
/// [`PointBatch`].
fn point_batch(a: &Array, name: &str) -> PyResult<PointBatch> {
    let n: usize = a.1.iter().product();
    if n != a.0.len() {
        return Err(dimension_error(format!(
            "{name}: {} values do not match the shape {:?}",
            a.0.len(),
            a.1
        )));
    }
    let Some((&m, rest)) = a.1.split_last() else {
        return Err(dimension_error(format!(
            "Expected `{name}` to have at least two dimensions (`... x n x m`)."
        )));
    };
    let Some((&n, batch)) = rest.split_last() else {
        return Err(dimension_error(format!(
            "Expected `{name}` to have at least two dimensions (`... x n x m`)."
        )));
    };
    mo_result(PointBatch::new(batch.to_vec(), n, m, a.0.clone()))
}

fn batch_out(p: &PointBatch) -> ArrayOut {
    (p.values().to_vec(), p.shape())
}

fn cells_out(c: &CellBounds) -> ArrayOut {
    (c.to_flat(), c.shape())
}

fn lub_out(l: &LocalUpperBounds) -> (ArrayOut, ArrayOut) {
    let (n, m) = (l.len(), l.m());
    (
        (l.upper_bounds().to_vec(), vec![n, m]),
        (l.defining_points().to_vec(), vec![n, m, m]),
    )
}

/// [`lub_out`] as buffers.
fn lub_buf(l: &LocalUpperBounds) -> (ArrayBuf, ArrayBuf) {
    let (u, z) = lub_out(l);
    (u.into(), z.into())
}

/// `U` (`n x m`) and `Z` (`n x m x m`) as [`LocalUpperBounds`].
fn lub_in(u: &Array, z: &Array) -> PyResult<LocalUpperBounds> {
    let (n, m) = match u.1.as_slice() {
        &[n, m] => (n, m),
        s => {
            return Err(dimension_error(format!(
                "Expected `U` to be a `n x m` tensor, got shape {s:?}."
            )))
        }
    };
    if z.1 != [n, m, m] {
        return Err(dimension_error(format!(
            "Expected `Z` to be a `{n} x {m} x {m}` tensor, got shape {:?}.",
            z.1
        )));
    }
    mo_result(LocalUpperBounds::new(m, u.0.clone(), z.0.clone()))
}

/// `is_non_dominated(Y, maximize, deduplicate)`: the mask (row-major) of an
/// array of shape `(*batch, n, m)`, of shape `(*batch, n)`.
#[pyfunction]
#[pyo3(signature = (y, maximize=true, deduplicate=true))]
pub fn is_non_dominated(
    py: Python<'_>,
    y: Array,
    maximize: bool,
    deduplicate: bool,
) -> PyResult<(Vec<bool>, Vec<usize>)> {
    let points = point_batch(&y, "Y")?;
    let mask = threads::run(py, || {
        is_non_dominated_batch(&points, maximize, deduplicate)
    });
    let shape = y.1[..y.1.len() - 1].to_vec();
    Ok((mask, shape))
}

/// `Hypervolume(ref_point).compute(pareto_Y)` for an `n x m` array.
#[pyfunction]
pub fn hypervolume(py: Python<'_>, ref_point: Vec<f64>, pareto_y: Array) -> PyResult<f64> {
    let y = pareto_y.matrix("pareto_Y")?;
    threads::run(py, || Hypervolume::new(&ref_point).compute(&y)).map_err(to_pyerr)
}

/// `infer_reference_point(pareto_Y, max_ref_point, scale, scale_max_ref_point)`.
#[pyfunction]
#[pyo3(
    name = "infer_reference_point",
    signature = (pareto_y, max_ref_point=None, scale=0.1, scale_max_ref_point=false)
)]
pub fn infer_reference_point_py(
    pareto_y: Array,
    max_ref_point: Option<Vec<f64>>,
    scale: f64,
    scale_max_ref_point: bool,
) -> PyResult<Vec<f64>> {
    let y = pareto_y.matrix("pareto_Y")?;
    infer_reference_point(&y, max_ref_point.as_deref(), scale, scale_max_ref_point)
        .map_err(to_pyerr)
}

/// `get_hypervolume_maximizing_subset(n, Y, ref_point)`; `choose(indices)`
/// selects a random remaining index (Python's `random.choice`). Returns the
/// subset (`n x m`) and the selected indices.
#[pyfunction]
#[pyo3(name = "get_hypervolume_maximizing_subset")]
pub fn get_hypervolume_maximizing_subset_py(
    py: Python<'_>,
    n: usize,
    y: Array,
    ref_point: Vec<f64>,
    choose: Py<PyAny>,
) -> PyResult<(ArrayOut, Vec<usize>)> {
    let y = y.matrix("Y")?;
    let error: Mutex<Option<PyErr>> = Mutex::new(None);
    let result = threads::run(py, || {
        get_hypervolume_maximizing_subset(n, &y, &ref_point, |remaining| {
            Python::attach(|py| {
                choose
                    .call1(py, (remaining.to_vec(),))
                    .and_then(|i| i.extract::<usize>(py))
            })
            .unwrap_or_else(|e| {
                let mut slot = error.lock().unwrap_or_else(|e| e.into_inner());
                slot.get_or_insert(e);
                remaining[0]
            })
        })
    });
    if let Some(e) = error.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(e);
    }
    let (subset, indices) = result.map_err(to_pyerr)?;
    Ok((matrix_out(&subset), indices))
}

/// The box decompositions behind [`PyBoxDecomposition`].
#[derive(Debug)]
enum AnyDecomposition {
    Nondominated(NondominatedPartitioning),
    FastNondominated(FastNondominatedPartitioning),
    Dominated(DominatedPartitioning),
}

impl AnyDecomposition {
    fn get(&self) -> &dyn BoxDecomposition {
        match self {
            Self::Nondominated(d) => d,
            Self::FastNondominated(d) => d,
            Self::Dominated(d) => d,
        }
    }

    fn get_mut(&mut self) -> &mut dyn BoxDecomposition {
        match self {
            Self::Nondominated(d) => d,
            Self::FastNondominated(d) => d,
            Self::Dominated(d) => d,
        }
    }

    fn local_upper_bounds(&self) -> Option<&LocalUpperBounds> {
        match self {
            Self::Nondominated(_) => None,
            Self::FastNondominated(d) => d.local_upper_bounds(),
            Self::Dominated(d) => d.local_upper_bounds(),
        }
    }
}

/// A box decomposition (`NondominatedPartitioning`,
/// `FastNondominatedPartitioning` or `DominatedPartitioning`).
#[pyclass(frozen, name = "BoxDecomposition", module = "botorch_rs._botorch_rs")]
pub struct PyBoxDecomposition {
    inner: Mutex<AnyDecomposition>,
}

impl PyBoxDecomposition {
    fn with<T, F>(&self, py: Python<'_>, f: F) -> T
    where
        F: FnOnce(&mut AnyDecomposition) -> T + Send,
        T: Send,
    {
        let inner = &self.inner;
        threads::run(py, || {
            f(&mut inner.lock().unwrap_or_else(|e| e.into_inner()))
        })
    }
}

#[pymethods]
impl PyBoxDecomposition {
    /// A decomposition of kind `"nondominated"` (with `alpha`),
    /// `"fast_nondominated"` or `"dominated"`, with reference point
    /// `ref_point` (`m` or `batch x m`) and outcomes `y` (`(batch) x n x m`).
    #[new]
    #[pyo3(signature = (kind, ref_point, y=None, alpha=0.0))]
    fn new(
        py: Python<'_>,
        kind: &str,
        ref_point: Array,
        y: Option<Array>,
        alpha: f64,
    ) -> PyResult<Self> {
        let y = y.map(|y| point_batch(&y, "Y")).transpose()?;
        let (r, shape) = (ref_point.0, ref_point.1);
        let inner = match kind {
            "nondominated" => threads::run(py, || {
                NondominatedPartitioning::with_ref_point_shape(&r, &shape, y.as_ref(), alpha)
                    .map(AnyDecomposition::Nondominated)
            }),
            "fast_nondominated" => threads::run(py, || {
                FastNondominatedPartitioning::with_ref_point_shape(&r, &shape, y.as_ref())
                    .map(AnyDecomposition::FastNondominated)
            }),
            "dominated" => threads::run(py, || {
                DominatedPartitioning::with_ref_point_shape(&r, &shape, y.as_ref())
                    .map(AnyDecomposition::Dominated)
            }),
            other => return Err(value_error(format!("unknown box decomposition {other:?}"))),
        };
        Ok(Self {
            inner: Mutex::new(mo_result(inner)?),
        })
    }

    /// `update(Y)`.
    fn update(&self, py: Python<'_>, y: Array) -> PyResult<()> {
        let y = point_batch(&y, "Y")?;
        mo_result(self.with(py, |d| d.get_mut().update(&y)))
    }

    /// `reset()`.
    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        mo_result(self.with(py, |d| d.get_mut().reset()))
    }

    /// `partition_space()`.
    fn partition_space(&self, py: Python<'_>) -> PyResult<()> {
        mo_result(self.with(py, |d| d.get_mut().partition_space()))
    }

    /// `compute_hypervolume()`: shape `batch_shape`; `None` before outcomes
    /// are given (BoTorch's scalar zero).
    fn compute_hypervolume(&self, py: Python<'_>) -> Option<ArrayBuf> {
        self.with(py, |d| {
            let d = d.get();
            d.pareto_y()?;
            let shape = d.batch_shape().map(<[usize]>::to_vec).unwrap_or_default();
            Some(ArrayBuf(d.compute_hypervolume(), shape))
        })
    }

    /// Whether outcomes have been given.
    #[getter]
    fn has_outcomes(&self, py: Python<'_>) -> bool {
        self.with(py, |d| d.get().y().is_some())
    }

    /// `get_hypercell_bounds()` (`2 x (batch) x num_cells x m`), `None`
    /// before outcomes are given.
    fn get_hypercell_bounds(&self, py: Python<'_>) -> Option<ArrayBuf> {
        self.with(py, |d| {
            d.get().get_hypercell_bounds().map(|c| cells_out(c).into())
        })
    }

    /// `pareto_Y`, `None` before outcomes are given.
    fn pareto_y(&self, py: Python<'_>) -> Option<ArrayBuf> {
        self.with(py, |d| {
            d.get().pareto_y().as_ref().map(|p| batch_out(p).into())
        })
    }

    /// `Y`, `None` before outcomes are given.
    fn y(&self, py: Python<'_>) -> Option<ArrayBuf> {
        self.with(py, |d| d.get().y().map(|p| batch_out(p).into()))
    }

    /// The reference point.
    #[getter]
    fn ref_point(&self, py: Python<'_>) -> ArrayOut {
        self.with(py, |d| {
            let d = d.get();
            (d.ref_point().to_vec(), d.ref_point_shape().to_vec())
        })
    }

    /// `num_outcomes`.
    #[getter]
    fn num_outcomes(&self, py: Python<'_>) -> usize {
        self.with(py, |d| d.get().num_outcomes())
    }

    /// `batch_shape`, `None` before outcomes are given.
    #[getter]
    fn batch_shape(&self, py: Python<'_>) -> Option<Vec<usize>> {
        self.with(py, |d| d.get().batch_shape().map(<[usize]>::to_vec))
    }

    /// BoTorch's class name.
    #[getter]
    fn class_name(&self, py: Python<'_>) -> &'static str {
        self.with(py, |d| d.get().class_name())
    }

    /// The local upper bounds `(U, Z)` of a fast partitioning (more than two
    /// outcomes, non-empty frontier), else `None`.
    fn local_upper_bounds(&self, py: Python<'_>) -> Option<(ArrayBuf, ArrayBuf)> {
        self.with(py, |d| d.local_upper_bounds().map(lub_buf))
    }
}

/// `_expand_ref_point(ref_point, batch_shape)`.
#[pyfunction]
#[pyo3(name = "expand_ref_point")]
pub fn expand_ref_point_py(ref_point: Array, batch_shape: Vec<usize>) -> PyResult<ArrayOut> {
    let values = mo_result(expand_ref_point(&ref_point.0, &ref_point.1, &batch_shape))?;
    let mut shape = batch_shape;
    shape.push(*ref_point.1.last().unwrap_or(&0));
    Ok((values, shape))
}

/// `_pad_batch_pareto_frontier(Y, ref_point, is_pareto, feasibility_mask)`.
#[pyfunction]
#[pyo3(
    name = "pad_batch_pareto_frontier",
    signature = (y, ref_point, is_pareto=false, feasibility_mask=None)
)]
pub fn pad_batch_pareto_frontier_py(
    py: Python<'_>,
    y: Array,
    ref_point: Array,
    is_pareto: bool,
    feasibility_mask: Option<Vec<bool>>,
) -> PyResult<ArrayBuf> {
    let points = point_batch(&y, "Y")?;
    let out = threads::run(py, || {
        pad_batch_pareto_frontier(
            &points,
            &ref_point.0,
            is_pareto,
            feasibility_mask.as_deref(),
        )
    });
    Ok(batch_out(&mo_result(out)?).into())
}

/// `compute_local_upper_bounds(U, Z, z)`: the new `(U, Z)`.
#[pyfunction]
#[pyo3(name = "compute_local_upper_bounds")]
pub fn compute_local_upper_bounds_py(
    u: Array,
    z: Array,
    point: Vec<f64>,
) -> PyResult<(ArrayBuf, ArrayBuf)> {
    let lub = lub_in(&u, &z)?;
    Ok(lub_buf(&mo_result(compute_local_upper_bounds(
        &lub, &point,
    ))?))
}

/// `update_local_upper_bounds_incremental(new_pareto_Y, U, Z)`.
#[pyfunction]
#[pyo3(name = "update_local_upper_bounds_incremental")]
pub fn update_local_upper_bounds_incremental_py(
    py: Python<'_>,
    new_pareto_y: Array,
    u: Array,
    z: Array,
) -> PyResult<(ArrayBuf, ArrayBuf)> {
    let points = point_batch(&new_pareto_y, "new_pareto_Y")?;
    let lub = lub_in(&u, &z)?;
    let out = threads::run(py, || update_local_upper_bounds_incremental(&points, &lub));
    Ok(lub_buf(&mo_result(out)?))
}

/// `get_partition_bounds(Z, U, ref_point)`: `2 x num_cells x m`.
#[pyfunction]
#[pyo3(name = "get_partition_bounds")]
pub fn get_partition_bounds_py(z: Array, u: Array, ref_point: Vec<f64>) -> PyResult<ArrayBuf> {
    let lub = lub_in(&u, &z)?;
    Ok(cells_out(&mo_result(get_partition_bounds(&lub, &ref_point))?).into())
}

/// `compute_non_dominated_hypercell_bounds_2d(pareto_Y_sorted, ref_point)`.
#[pyfunction]
#[pyo3(name = "compute_non_dominated_hypercell_bounds_2d")]
pub fn compute_non_dominated_hypercell_bounds_2d_py(
    pareto_y_sorted: Array,
    ref_point: Array,
) -> PyResult<ArrayBuf> {
    let points = point_batch(&pareto_y_sorted, "pareto_Y_sorted")?;
    let cells = compute_non_dominated_hypercell_bounds_2d(&points, &ref_point.0);
    Ok(cells_out(&mo_result(cells)?).into())
}

/// `compute_dominated_hypercell_bounds_2d(pareto_Y_sorted, ref_point)`.
#[pyfunction]
#[pyo3(name = "compute_dominated_hypercell_bounds_2d")]
pub fn compute_dominated_hypercell_bounds_2d_py(
    pareto_y_sorted: Array,
    ref_point: Array,
) -> PyResult<ArrayBuf> {
    let points = point_batch(&pareto_y_sorted, "pareto_Y_sorted")?;
    let cells = compute_dominated_hypercell_bounds_2d(&points, &ref_point.0);
    Ok(cells_out(&mo_result(cells)?).into())
}

/// The augmented Chebyshev scalarization (`get_chebyshev_scalarization`).
#[pyclass(
    frozen,
    name = "ChebyshevScalarization",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyChebyshevScalarization {
    /// The scalarization (for the acquisition functions that use it).
    pub inner: ChebyshevScalarization,
}

impl PyChebyshevScalarization {
    fn rows(&self, y: &Array) -> PyResult<(Vec<usize>, usize)> {
        let m = self.inner.num_outcomes();
        match y.1.split_last() {
            Some((&k, batch)) if k == m && y.0.len() == y.1.iter().product::<usize>() => {
                Ok((batch.to_vec(), m))
            }
            _ => Err(dimension_error(format!(
                "Expected Y to be a `... x {m}` tensor, got shape {:?}.",
                y.1
            ))),
        }
    }
}

#[pymethods]
impl PyChebyshevScalarization {
    /// The scalarization with `weights` (`m`) normalized by the outcomes `y`
    /// (`n x m`) and the weight `alpha` of the sum.
    #[new]
    fn new(weights: Vec<f64>, y: Array, alpha: f64) -> PyResult<Self> {
        let y = y.matrix("Y")?;
        Ok(Self {
            inner: ChebyshevScalarization::new(&weights, &y, alpha).map_err(to_pyerr)?,
        })
    }

    /// The scalarized values of `y` (`... x m`), shape `...`.
    fn evaluate(&self, py: Python<'_>, y: Array) -> PyResult<ArrayBuf> {
        let (batch, _) = self.rows(&y)?;
        let values = threads::run(py, || self.inner.evaluate_rows(&y.0));
        Ok(ArrayBuf(values, batch))
    }

    /// The values and their gradients with respect to `y`.
    fn evaluate_with_grad(&self, py: Python<'_>, y: Array) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let (batch, m) = self.rows(&y)?;
        let (values, grads) = threads::run(py, || {
            let mut grads = vec![0.0; y.0.len()];
            let values =
                y.0.chunks_exact(m)
                    .zip(grads.chunks_exact_mut(m))
                    .map(|(row, g)| self.inner.evaluate_with_grad(row, g))
                    .collect::<Vec<f64>>();
            (values, grads)
        });
        Ok((ArrayBuf(values, batch), ArrayBuf(grads, y.1)))
    }

    /// The weights.
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    /// `alpha`.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// The bounds of the negated outcomes (`2 x m`), `None` without
    /// observations.
    #[getter]
    fn bounds(&self) -> Option<ArrayOut> {
        self.inner.bounds().map(|(l, u)| {
            let mut data = l.to_vec();
            data.extend_from_slice(u);
            (data, vec![2, l.len()])
        })
    }
}

/// `sample_simplex(d, n, qmc, seed)`: `n x d`.
#[pyfunction]
#[pyo3(name = "sample_simplex", signature = (d, n=1, qmc=false, seed=None))]
pub fn sample_simplex_py(
    py: Python<'_>,
    d: usize,
    n: usize,
    qmc: bool,
    seed: Option<u64>,
) -> PyResult<ArrayBuf> {
    let x = threads::run(py, || sample_simplex(d, n, qmc, seed)).map_err(to_pyerr)?;
    Ok(matrix_out(&x).into())
}

/// `sample_hypersphere(d, n, qmc, seed)`: `n x d`.
#[pyfunction]
#[pyo3(name = "sample_hypersphere", signature = (d, n=1, qmc=false, seed=None))]
pub fn sample_hypersphere_py(
    py: Python<'_>,
    d: usize,
    n: usize,
    qmc: bool,
    seed: Option<u64>,
) -> PyResult<ArrayBuf> {
    let x = threads::run(py, || sample_hypersphere(d, n, qmc, seed)).map_err(to_pyerr)?;
    Ok(matrix_out(&x).into())
}

/// The simplex points of `sample_simplex` from uniform samples
/// (`n x (d - 1)`).
#[pyfunction]
#[pyo3(name = "simplex_from_uniform")]
pub fn simplex_from_uniform_py(u: Array) -> PyResult<ArrayBuf> {
    let u: Matrix = u.matrix("u")?;
    Ok(matrix_out(&simplex_from_uniform(&u)).into())
}

/// The sphere points of `sample_hypersphere` from normal samples (`n x d`).
#[pyfunction]
#[pyo3(name = "hypersphere_from_normal")]
pub fn hypersphere_from_normal_py(z: Array) -> PyResult<ArrayBuf> {
    let z: Matrix = z.matrix("z")?;
    Ok(matrix_out(&hypersphere_from_normal(&z)).into())
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoxDecomposition>()?;
    m.add_class::<PyChebyshevScalarization>()?;
    m.add_function(wrap_pyfunction!(is_non_dominated, m)?)?;
    m.add_function(wrap_pyfunction!(hypervolume, m)?)?;
    m.add_function(wrap_pyfunction!(infer_reference_point_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_hypervolume_maximizing_subset_py, m)?)?;
    m.add_function(wrap_pyfunction!(expand_ref_point_py, m)?)?;
    m.add_function(wrap_pyfunction!(pad_batch_pareto_frontier_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_local_upper_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        update_local_upper_bounds_incremental_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(get_partition_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_non_dominated_hypercell_bounds_2d_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compute_dominated_hypercell_bounds_2d_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sample_simplex_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_hypersphere_py, m)?)?;
    m.add_function(wrap_pyfunction!(simplex_from_uniform_py, m)?)?;
    m.add_function(wrap_pyfunction!(hypersphere_from_normal_py, m)?)?;
    Ok(())
}
