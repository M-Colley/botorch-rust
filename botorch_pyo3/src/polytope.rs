//! Polytope sampling (`botorch.utils.sampling`): `get_polytope_samples`,
//! `HitAndRunPolytopeSampler`, `find_interior_point`, `sample_polytope` and
//! `sparse_to_dense_constraints`.

use std::sync::Mutex;

use botorch_core::utils::polytope::{
    find_interior_point, get_polytope_samples, sample_polytope, sparse_to_dense_constraints,
    HitAndRunPolytopeSampler,
};
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{dimension_error, IntoPyResult};
use crate::optim::{linear_constraints, LinearArg};
use crate::threads;

/// A vector from a 1-d array or a `k x 1` column.
fn vector(a: &Array, name: &str) -> PyResult<Vector> {
    let n: usize = a.1.iter().product();
    if a.0.len() != n || a.1.len() > 2 || (a.1.len() == 2 && a.1[1] != 1) {
        return Err(dimension_error(format!(
            "`{name}` must be a vector or a column, got shape {:?}",
            a.1
        )));
    }
    Ok(Vector::from_vec(a.0.clone()))
}

/// Dense constraints `(A, b)` (`A x <= b` or `A x = b`).
fn dense(c: Option<(Array, Array)>, name: &str) -> PyResult<Option<(Matrix, Vector)>> {
    c.map(|(a, b)| Ok((a.matrix(name)?, vector(&b, name)?)))
        .transpose()
}

/// `get_polytope_samples(n, bounds, inequality_constraints,
/// equality_constraints, seed, n_burnin, n_thinning)`: `n x d` samples.
#[pyfunction]
#[pyo3(name = "get_polytope_samples", signature = (
    n,
    bounds,
    inequality_constraints=None,
    equality_constraints=None,
    seed=None,
    n_burnin=10_000,
    n_thinning=32,
))]
#[allow(clippy::too_many_arguments)]
pub fn get_polytope_samples_py(
    py: Python<'_>,
    n: usize,
    bounds: Array,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
    equality_constraints: Option<Vec<LinearArg<'_>>>,
    seed: Option<u64>,
    n_burnin: usize,
    n_thinning: usize,
) -> PyResult<ArrayBuf> {
    let bounds = bounds.matrix("bounds")?;
    let ineq = linear_constraints(inequality_constraints)?;
    let eq = linear_constraints(equality_constraints)?;
    let x = threads::run(py, || {
        get_polytope_samples(n, &bounds, &ineq, &eq, seed, n_burnin, n_thinning)
    })
    .into_py_result()?;
    Ok(matrix_out(&x).into())
}

/// `sparse_to_dense_constraints(d, constraints)`: `(A, b)` with
/// `A[i, indices] = coefficients`, `b[i] = rhs`.
#[pyfunction]
#[pyo3(name = "sparse_to_dense_constraints")]
pub fn sparse_to_dense_constraints_py(
    d: usize,
    constraints: Vec<LinearArg<'_>>,
) -> PyResult<(ArrayOut, ArrayOut)> {
    let cs = linear_constraints(Some(constraints))?;
    let (a, b) = sparse_to_dense_constraints(d, &cs).into_py_result()?;
    let k = b.len();
    Ok((matrix_out(&a), (b.as_slice().to_vec(), vec![k, 1])))
}

/// `find_interior_point(A, b, A_eq, b_eq)`: a point of `{A x <= b, A_eq x =
/// b_eq}` with maximal slack of the inequalities.
#[pyfunction]
#[pyo3(name = "find_interior_point", signature = (a, b, a_eq=None, b_eq=None))]
pub fn find_interior_point_py(
    a: Array,
    b: Array,
    a_eq: Option<Array>,
    b_eq: Option<Array>,
) -> PyResult<Vec<f64>> {
    let a = a.matrix("A")?;
    let b = vector(&b, "b")?;
    let eq = match (a_eq, b_eq) {
        (Some(ae), Some(be)) => {
            let ae = ae.matrix("A_eq")?;
            // BoTorch's A_eq has an extra (zero) column for the slack variable.
            let ae = if ae.ncols() == a.ncols() + 1 {
                ae.columns(0, a.ncols()).into_owned()
            } else {
                ae
            };
            Some((ae, vector(&be, "b_eq")?))
        }
        _ => None,
    };
    let x = find_interior_point(&a, &b, eq.as_ref().map(|(m, v)| (m, v))).into_py_result()?;
    Ok(x.as_slice().to_vec())
}

/// `sample_polytope(A, b, x0, n, n0, n_thinning, seed)`: hit-and-run samples
/// of `{A x <= b}` (`n x d`).
#[pyfunction]
#[pyo3(name = "sample_polytope", signature = (a, b, x0, n=10_000, n0=100, n_thinning=1, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn sample_polytope_py(
    py: Python<'_>,
    a: Array,
    b: Array,
    x0: Array,
    n: usize,
    n0: usize,
    n_thinning: usize,
    seed: Option<u64>,
) -> PyResult<ArrayBuf> {
    let a = a.matrix("A")?;
    let b = vector(&b, "b")?;
    let x0 = vector(&x0, "x0")?;
    let x = threads::run(py, || sample_polytope(&a, &b, &x0, n, n0, n_thinning, seed))
        .into_py_result()?;
    Ok(matrix_out(&x).into())
}

/// BoTorch's `HitAndRunPolytopeSampler`.
#[pyclass(
    frozen,
    name = "HitAndRunPolytopeSampler",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyHitAndRunPolytopeSampler {
    inner: Mutex<HitAndRunPolytopeSampler>,
}

#[pymethods]
impl PyHitAndRunPolytopeSampler {
    /// A sampler for `A x <= b` (`inequality_constraints`), `C x = e`
    /// (`equality_constraints`) and `bounds` (`2 x d`).
    #[new]
    #[pyo3(signature = (
        inequality_constraints=None,
        equality_constraints=None,
        bounds=None,
        interior_point=None,
        n_burnin=200,
        n_thinning=20,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        inequality_constraints: Option<(Array, Array)>,
        equality_constraints: Option<(Array, Array)>,
        bounds: Option<Array>,
        interior_point: Option<Array>,
        n_burnin: usize,
        n_thinning: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let ineq = dense(inequality_constraints, "inequality_constraints")?;
        let eq = dense(equality_constraints, "equality_constraints")?;
        let bounds = bounds.map(|b| b.matrix("bounds")).transpose()?;
        let interior = interior_point
            .map(|p| vector(&p, "interior_point"))
            .transpose()?;
        let sampler = threads::run(py, || {
            HitAndRunPolytopeSampler::new(
                ineq,
                eq,
                bounds.as_ref(),
                interior,
                n_burnin,
                n_thinning,
                seed,
            )
        })
        .into_py_result()?;
        Ok(Self {
            inner: Mutex::new(sampler),
        })
    }

    /// `n` samples (`n x d`); the chain continues on the next call.
    #[pyo3(signature = (n=1))]
    fn draw(&self, py: Python<'_>, n: usize) -> PyResult<ArrayBuf> {
        let inner = &self.inner;
        let x = threads::run(py, || {
            inner.lock().unwrap_or_else(|e| e.into_inner()).draw(n)
        })
        .into_py_result()?;
        Ok(matrix_out(&x).into())
    }
}
