//! Probability and safe-math utilities (phase 11a): the element-wise
//! functions and reductions of `botorch.utils.safe_math` and
//! `botorch.utils.probability` with derivatives, the stateful `MVNXPB` and
//! `PivotedCholesky` steps, `augment_cholesky`, the linear elliptical slice
//! sampler, the transforms of `botorch.sampling.qmc`, PyTorch's seeded CPU
//! generator and the null space of equality constraints.

use botorch_core::test_functions::single_objective::{
    default_max_step_size, round_nearest, TrajectoryConfig,
};
use botorch_core::test_functions::{SingleObjectiveKind, SingleObjectiveProblem};
use botorch_core::utils::elementwise::{
    elementwise, log_prob_feas_from_bounds, percentile_of_score, reduce, OutcomeBounds,
};
use botorch_core::utils::lin_ess::{CovarianceRoot, LinearEllipticalSliceSampler};
use botorch_core::utils::mvnxpb::mvnxpb_log_prob_with_grad;
use botorch_core::utils::mvnxpb_state::{
    augment_cholesky, pivoted_cholesky_pivot, pivoted_cholesky_update, MvnxpbState,
};
use botorch_core::utils::polytope::null_space;
use botorch_core::utils::qmc::{mvn_qmc_project, mvn_qmc_root, normal_qmc_transform};
use botorch_core::utils::sampling::SobolEngine;
use botorch_core::utils::torch_rng::TorchGenerator;
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

use crate::arrays::{float_data, float_values, matrix_out, Array, ArrayBuf};
use crate::errors::{dimension_error, unsupported, value_error, IntoPyResult};
use crate::threads;

/// The values of an array (a float buffer or a sequence of numbers).
#[derive(FromPyObject)]
pub struct Floats(#[pyo3(from_py_with = float_data)] pub Vec<f64>);

type Values<'py> = Vec<Bound<'py, PyAny>>;

/// `elementwise(name, inputs, params, single_precision, with_grad)`: the
/// element-wise function `name` of the (broadcast, flat) inputs; returns the
/// outputs and, with `with_grad`, `grads[output][input]`.
#[pyfunction]
#[pyo3(name = "elementwise", signature = (name, inputs, params=vec![], single_precision=false, with_grad=false))]
#[allow(clippy::type_complexity)]
fn elementwise_py<'py>(
    py: Python<'py>,
    name: &str,
    inputs: Vec<Floats>,
    params: Vec<f64>,
    single_precision: bool,
    with_grad: bool,
) -> PyResult<(Values<'py>, Option<Vec<Values<'py>>>)> {
    let slices: Vec<&[f64]> = inputs.iter().map(|f| f.0.as_slice()).collect();
    let out = threads::run(py, || {
        elementwise(name, &slices, &params, single_precision, with_grad)
    })
    .into_py_result()?;
    let values = out
        .values
        .iter()
        .map(|v| float_values(py, v))
        .collect::<PyResult<Vec<_>>>()?;
    let grads = match out.grads {
        Some(g) => Some(
            g.iter()
                .map(|per_input| {
                    per_input
                        .iter()
                        .map(|v| float_values(py, v))
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<Vec<_>>>()?,
        ),
        None => None,
    };
    Ok((values, grads))
}

/// `reduce(name, data, n, params, with_grad)`: the reduction `name` over the
/// rows of `n` values; returns the values and, with `with_grad`, the
/// derivatives with respect to every element.
#[pyfunction]
#[pyo3(name = "reduce", signature = (name, data, n, params=vec![], with_grad=false))]
fn reduce_py<'py>(
    py: Python<'py>,
    name: &str,
    data: Floats,
    n: usize,
    params: Vec<f64>,
    with_grad: bool,
) -> PyResult<(Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)> {
    let (values, grads) =
        threads::run(py, || reduce(name, &data.0, n, &params, with_grad)).into_py_result()?;
    Ok((
        float_values(py, &values)?,
        grads.map(|g| float_values(py, &g)).transpose()?,
    ))
}

/// `log_prob_feas_from_bounds(lower, upper, both, means, sigmas, m,
/// with_grad)` (BoTorch's `compute_log_prob_feas_from_bounds`).
#[pyfunction]
#[pyo3(name = "log_prob_feas_from_bounds", signature = (lower, upper, both, means, sigmas, m, with_grad=false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn log_prob_feas_py<'py>(
    py: Python<'py>,
    lower: Vec<(usize, f64)>,
    upper: Vec<(usize, f64)>,
    both: Vec<(usize, f64, f64)>,
    means: Floats,
    sigmas: Floats,
    m: usize,
    with_grad: bool,
) -> PyResult<(
    Bound<'py, PyAny>,
    Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
)> {
    let bounds = OutcomeBounds { lower, upper, both };
    let (v, g) =
        log_prob_feas_from_bounds(&bounds, &means.0, &sigmas.0, m, with_grad).into_py_result()?;
    let g = match g {
        Some((gm, gs)) => Some((float_values(py, &gm)?, float_values(py, &gs)?)),
        None => None,
    };
    Ok((float_values(py, &v)?, g))
}

/// `percentile_of_score(data, scores, n, single_precision)`.
#[pyfunction]
#[pyo3(name = "percentile_of_score")]
fn percentile_py<'py>(
    py: Python<'py>,
    data: Floats,
    scores: Floats,
    n: usize,
    single_precision: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if n > 0 && data.0.len() != scores.0.len() * n {
        return Err(dimension_error("data and scores do not match"));
    }
    float_values(
        py,
        &percentile_of_score(&data.0, &scores.0, n, single_precision),
    )
}

/// The state of a batch of `MVNXPB` solvers (`mvnxpbState`), exchanged with
/// the Python class.
#[pyclass(name = "MvnxpbState", module = "botorch_rs._botorch_rs")]
pub struct PyMvnxpbState {
    inner: MvnxpbState,
}

fn opt_values<'py>(py: Python<'py>, v: &Option<Vec<f64>>) -> PyResult<Option<Bound<'py, PyAny>>> {
    v.as_ref().map(|v| float_values(py, v)).transpose()
}

#[pymethods]
impl PyMvnxpbState {
    /// A state from its arrays (flat, row-major).
    #[new]
    #[pyo3(signature = (n, batch, step, pc_step, perm, bounds, tril, pc_perm, diag, plug_ins, log_prob, log_prob_extra))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        batch: usize,
        step: usize,
        pc_step: usize,
        perm: Vec<i64>,
        bounds: Floats,
        tril: Floats,
        pc_perm: Vec<i64>,
        diag: Option<Floats>,
        plug_ins: Floats,
        log_prob: Floats,
        log_prob_extra: Option<Floats>,
    ) -> PyResult<Self> {
        let inner = MvnxpbState {
            n,
            batch,
            step,
            pc_step,
            perm,
            bounds: bounds.0,
            tril: tril.0,
            pc_perm,
            diag: diag.map(|d| d.0),
            plug_ins: plug_ins.0,
            log_prob: log_prob.0,
            log_prob_extra: log_prob_extra.map(|e| e.0),
        };
        inner.validate().into_py_result()?;
        Ok(Self { inner })
    }

    /// BoTorch's `MVNXPB(covariance_matrix, bounds)` for `batch` problems.
    #[staticmethod]
    fn standardized(cov: Floats, bounds: Floats, batch: usize, n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: MvnxpbState::new(&cov.0, &bounds.0, batch, n).into_py_result()?,
        })
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
    #[getter]
    fn batch(&self) -> usize {
        self.inner.batch
    }
    #[getter]
    fn step(&self) -> usize {
        self.inner.step
    }
    #[getter]
    fn pc_step(&self) -> usize {
        self.inner.pc_step
    }
    #[getter]
    fn perm(&self) -> Vec<i64> {
        self.inner.perm.clone()
    }
    #[getter]
    fn pc_perm(&self) -> Vec<i64> {
        self.inner.pc_perm.clone()
    }
    #[getter]
    fn bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        float_values(py, &self.inner.bounds)
    }
    #[getter]
    fn tril<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        float_values(py, &self.inner.tril)
    }
    #[getter]
    fn diag<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        opt_values(py, &self.inner.diag)
    }
    #[getter]
    fn plug_ins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        float_values(py, &self.inner.plug_ins)
    }
    #[getter]
    fn log_prob<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        float_values(py, &self.inner.log_prob)
    }
    #[getter]
    fn log_prob_extra<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        opt_values(py, &self.inner.log_prob_extra)
    }

    /// `solve(num_steps, eps)`: runs the solver in place; returns the log
    /// probabilities.
    #[pyo3(signature = (num_steps=None, eps=1e-10))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        num_steps: Option<usize>,
        eps: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = &mut self.inner;
        let lp = threads::run(py, || state.solve(num_steps, eps)).into_py_result()?;
        float_values(py, &lp)
    }

    /// `select_pivot()`: the GGE pivot of every member.
    fn select_pivot(&self) -> Vec<usize> {
        self.inner.select_pivot()
    }

    /// `pivot_(pivot)`: swaps the variables at `step` and `pivot` in place.
    fn pivot(&mut self, pivot: Vec<usize>) -> PyResult<()> {
        self.inner.pivot(&pivot).into_py_result()
    }

    /// `augment(cov, bounds, cross, m, ...)`: the augmented state and
    /// whether the joint covariance had to be refactorized.
    #[pyo3(signature = (cov, bounds, cross, m, disable_pivoting=false, jitter=None, max_tries=None))]
    #[allow(clippy::too_many_arguments)]
    fn augment(
        &self,
        py: Python<'_>,
        cov: Floats,
        bounds: Floats,
        cross: Floats,
        m: usize,
        disable_pivoting: bool,
        jitter: Option<f64>,
        max_tries: Option<usize>,
    ) -> PyResult<(Self, bool)> {
        let state = &self.inner;
        let (inner, recovered) = threads::run(py, || {
            state.augment(
                &cov.0,
                &bounds.0,
                &cross.0,
                m,
                disable_pivoting,
                jitter,
                max_tries,
            )
        })
        .into_py_result()?;
        Ok((Self { inner }, recovered))
    }
}

/// `mvnxpb_log_prob_with_grad(cov, bounds, batch, n)`: the one-shot solve of
/// fresh problems with the derivatives with respect to the covariances
/// (lower triangles) and the bounds.
#[pyfunction]
#[pyo3(name = "mvnxpb_log_prob_with_grad")]
#[allow(clippy::type_complexity)]
fn mvnxpb_grad_py<'py>(
    py: Python<'py>,
    cov: Floats,
    bounds: Floats,
    batch: usize,
    n: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if cov.0.len() != batch * n * n || bounds.0.len() != batch * n * 2 {
        return Err(dimension_error("inconsistent covariance and bounds"));
    }
    let res = threads::run(py, || {
        let mut v = Vec::with_capacity(batch);
        let mut gc = Vec::with_capacity(batch * n * n);
        let mut gb = Vec::with_capacity(batch * n * 2);
        for b in 0..batch {
            let c = Matrix::from_row_slice(n, n, &cov.0[b * n * n..(b + 1) * n * n]);
            let bd = Matrix::from_row_slice(n, 2, &bounds.0[b * n * 2..(b + 1) * n * 2]);
            let (lp, g_cov, g_b) = mvnxpb_log_prob_with_grad(&c, &bd)?;
            v.push(lp);
            gc.extend(matrix_out(&g_cov).0);
            gb.extend(matrix_out(&g_b).0);
        }
        Ok::<_, botorch_core::BotorchError>((v, gc, gb))
    })
    .into_py_result()?;
    Ok((
        float_values(py, &res.0)?,
        float_values(py, &res.1)?,
        float_values(py, &res.2)?,
    ))
}

/// `pivoted_cholesky_update(tril, batch, n, step, eps)`: one decomposition
/// step of every member (`PivotedCholesky.update_`).
#[pyfunction]
#[pyo3(name = "pivoted_cholesky_update")]
fn pc_update_py(tril: Floats, batch: usize, n: usize, step: usize, eps: f64) -> PyResult<ArrayBuf> {
    let mut t = tril.0;
    if t.len() != batch * n * n || step >= n {
        return Err(dimension_error("inconsistent factor shape or step"));
    }
    for b in 0..batch {
        pivoted_cholesky_update(&mut t[b * n * n..(b + 1) * n * n], n, step, eps);
    }
    Ok(ArrayBuf(t, vec![batch, n, n]))
}

/// `pivoted_cholesky_pivot(tril, perm, diag, batch, n, step, pivots)`
/// (`PivotedCholesky.pivot_`).
#[pyfunction]
#[pyo3(name = "pivoted_cholesky_pivot", signature = (tril, perm, diag, batch, n, step, pivots))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pc_pivot_py<'py>(
    py: Python<'py>,
    tril: Floats,
    perm: Vec<i64>,
    diag: Option<Floats>,
    batch: usize,
    n: usize,
    step: usize,
    pivots: Vec<usize>,
) -> PyResult<(Bound<'py, PyAny>, Vec<i64>, Option<Bound<'py, PyAny>>)> {
    let mut t = tril.0;
    let mut p = perm;
    let mut d = diag.map(|d| d.0);
    if t.len() != batch * n * n
        || p.len() != batch * n
        || d.as_ref().is_some_and(|d| d.len() != batch * n)
        || pivots.len() != batch
        || pivots.iter().any(|&v| v < step || v >= n)
    {
        return Err(value_error(
            "Argument `pivot` does to match with batch shape`.",
        ));
    }
    for (b, &piv) in pivots.iter().enumerate() {
        pivoted_cholesky_pivot(
            &mut t[b * n * n..(b + 1) * n * n],
            &mut p[b * n..(b + 1) * n],
            d.as_mut().map(|d| &mut d[b * n..(b + 1) * n]),
            n,
            step,
            piv,
        );
    }
    Ok((float_values(py, &t)?, p, opt_values(py, &d)?))
}

/// `augment_cholesky(laa, kbb, kba, lba, jitter, batch, na, nb)`: the
/// factors of the augmented matrices (`batch x (na + nb) x (na + nb)`).
#[pyfunction]
#[pyo3(name = "augment_cholesky", signature = (laa, kbb, kba, lba, jitter, batch, na, nb))]
#[allow(clippy::too_many_arguments)]
fn augment_cholesky_py(
    py: Python<'_>,
    laa: Floats,
    kbb: Floats,
    kba: Option<Floats>,
    lba: Option<Floats>,
    jitter: Option<f64>,
    batch: usize,
    na: usize,
    nb: usize,
) -> PyResult<ArrayBuf> {
    let ok = laa.0.len() == batch * na * na
        && kbb.0.len() == batch * nb * nb
        && kba.as_ref().is_none_or(|k| k.0.len() == batch * nb * na)
        && lba.as_ref().is_none_or(|k| k.0.len() == batch * nb * na);
    if !ok {
        return Err(dimension_error(
            "inconsistent shapes of Laa, Kbb and Kba / Lba",
        ));
    }
    let n = na + nb;
    let out = threads::run(py, || {
        let mut out = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let l = Matrix::from_row_slice(na, na, &laa.0[b * na * na..(b + 1) * na * na]);
            let k = Matrix::from_row_slice(nb, nb, &kbb.0[b * nb * nb..(b + 1) * nb * nb]);
            let ba = kba
                .as_ref()
                .map(|v| Matrix::from_row_slice(nb, na, &v.0[b * nb * na..(b + 1) * nb * na]));
            let lb = lba
                .as_ref()
                .map(|v| Matrix::from_row_slice(nb, na, &v.0[b * nb * na..(b + 1) * nb * na]));
            let f = augment_cholesky(&l, &k, ba.as_ref(), lb.as_ref(), jitter)?;
            out.extend(matrix_out(&f).0);
        }
        Ok::<_, botorch_core::BotorchError>(out)
    })
    .into_py_result()?;
    Ok(ArrayBuf(out, vec![batch, n, n]))
}

/// The linear elliptical slice sampler (`LinearEllipticalSliceSampler`).
#[pyclass(
    name = "LinearEllipticalSliceSampler",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyLinearEllipticalSliceSampler {
    inner: LinearEllipticalSliceSampler,
}

#[pymethods]
impl PyLinearEllipticalSliceSampler {
    /// A sampler of `N(mean, Σ)` restricted to `A x <= b` started at `x0`;
    /// `root` is a lower-triangular root, `root_diag` a diagonal one.
    #[new]
    #[pyo3(signature = (a, b, x0, fixed_indices=None, mean=None, covariance_matrix=None, root=None, root_diag=None, num_chains=1, single_precision=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        a: Array,
        b: Floats,
        x0: Floats,
        fixed_indices: Option<Vec<usize>>,
        mean: Option<Floats>,
        covariance_matrix: Option<Array>,
        root: Option<Array>,
        root_diag: Option<Floats>,
        num_chains: usize,
        single_precision: bool,
    ) -> PyResult<Self> {
        let a = a.matrix("A")?;
        let root = match (root, root_diag) {
            (Some(r), None) => Some(CovarianceRoot::Lower(r.matrix("covariance_root")?)),
            (None, Some(d)) => Some(CovarianceRoot::Diagonal(Vector::from_vec(d.0))),
            (None, None) => None,
            _ => return Err(value_error("pass one of root and root_diag")),
        };
        let cov = covariance_matrix
            .map(|c| c.matrix("covariance_matrix"))
            .transpose()?;
        let inner = LinearEllipticalSliceSampler::new(
            a,
            Vector::from_vec(b.0),
            Vector::from_vec(x0.0),
            fixed_indices,
            mean.map(|m| Vector::from_vec(m.0)),
            cov,
            root,
            num_chains,
            single_precision,
        )
        .into_py_result()?;
        Ok(Self { inner })
    }

    /// `(k, num_chains)`: the shape of the directions of a step.
    #[getter]
    fn draw_shape(&self) -> (usize, usize) {
        (self.inner.free_dim(), self.inner.num_chains())
    }

    #[getter]
    fn lifetime_samples(&self) -> usize {
        self.inner.lifetime_samples()
    }

    /// The current points (`d x num_chains`).
    #[getter]
    fn x(&self) -> ArrayBuf {
        matrix_out(self.inner.current()).into()
    }

    /// The current standardized points (`k x num_chains`).
    #[getter]
    fn z(&self) -> ArrayBuf {
        matrix_out(self.inner.current_standardized()).into()
    }

    /// The standardized constraints `(A_z, b_z)`.
    #[getter]
    fn standardized(&self) -> (ArrayBuf, ArrayBuf) {
        let (a, b) = self.inner.standardized_constraints();
        (
            matrix_out(a).into(),
            ArrayBuf(b.as_slice().to_vec(), vec![b.len(), 1]),
        )
    }

    /// `steps` steps with the directions `nu` (`steps x k x num_chains`) and
    /// uniform numbers `u` (`steps x num_chains`), keeping every `keep`-th
    /// point (`keep = thinning + 1`); returns the kept points
    /// (`num_kept x d x num_chains`).
    #[pyo3(signature = (nu, u, steps, keep, check_feasibility=false))]
    fn run(
        &mut self,
        py: Python<'_>,
        nu: Floats,
        u: Floats,
        steps: usize,
        keep: usize,
        check_feasibility: bool,
    ) -> PyResult<ArrayBuf> {
        let (k, c) = (self.inner.free_dim(), self.inner.num_chains());
        if nu.0.len() != steps * k * c || u.0.len() != steps * c || keep == 0 {
            return Err(dimension_error("draws of the wrong shape"));
        }
        let sampler = &mut self.inner;
        let out = threads::run(py, || {
            let mut out = Vec::new();
            let mut kept = 0;
            for s in 0..steps {
                let dir = Matrix::from_row_slice(k, c, &nu.0[s * k * c..(s + 1) * k * c]);
                let x = sampler.step(&dir, &u.0[s * c..(s + 1) * c], check_feasibility)?;
                if (s + 1) % keep == 0 {
                    out.extend(matrix_out(x).0);
                    kept += 1;
                }
            }
            Ok::<_, botorch_core::BotorchError>((out, kept))
        })
        .into_py_result()?;
        let d = self.inner.current().nrows();
        Ok(ArrayBuf(out.0, vec![out.1, d, c]))
    }

    /// The intersection angles for the direction `nu` (flat `k x
    /// num_chains`): `m x 2 num_chains` (alpha, then beta; BoTorch's
    /// `_find_intersection_angles` concatenated along the last dimension).
    fn arc_ends(&self, nu: Floats) -> PyResult<ArrayBuf> {
        let (k, c) = (self.inner.free_dim(), self.inner.num_chains());
        if nu.0.len() != k * c {
            return Err(dimension_error("a direction of the wrong shape"));
        }
        let dir = Matrix::from_row_slice(k, c, &nu.0);
        let (alpha, beta) = self.inner.intersection_angles(&dir).into_py_result()?;
        let m = alpha.nrows();
        let mut out = Vec::with_capacity(2 * m * c);
        for r in 0..m {
            out.extend((0..c).map(|j| alpha[(r, j)]));
            out.extend((0..c).map(|j| beta[(r, j)]));
        }
        Ok(ArrayBuf(out, vec![m, 2 * c]))
    }

    /// Whether the columns of `points` (`d x M`) are feasible.
    #[pyo3(signature = (points, transformed=false))]
    fn is_feasible(&self, points: Array, transformed: bool) -> PyResult<Vec<bool>> {
        Ok(self
            .inner
            .is_feasible(&points.matrix("points")?, transformed))
    }
}

/// `normal_qmc_transform(uniform, n, sobol_dim, d, inv_transform,
/// single_precision)`.
#[pyfunction]
#[pyo3(name = "normal_qmc_transform")]
fn normal_qmc_py(
    py: Python<'_>,
    uniform: Floats,
    n: usize,
    sobol_dim: usize,
    d: usize,
    inv_transform: bool,
    single_precision: bool,
) -> PyResult<ArrayBuf> {
    let out = threads::run(py, || {
        normal_qmc_transform(&uniform.0, n, sobol_dim, d, inv_transform, single_precision)
    })
    .into_py_result()?;
    Ok(ArrayBuf(out, vec![n, d]))
}

/// `mvn_qmc_root(cov, single_precision)`: `MultivariateNormalQMCEngine`'s
/// `_corr_matrix`.
#[pyfunction]
#[pyo3(name = "mvn_qmc_root")]
fn mvn_qmc_root_py(py: Python<'_>, cov: Array, single_precision: bool) -> PyResult<ArrayBuf> {
    let cov = cov.matrix("cov")?;
    let root = threads::run(py, || mvn_qmc_root(&cov, single_precision)).into_py_result()?;
    Ok(matrix_out(&root).into())
}

/// Native matrix product and mean shift for multivariate QMC samples.
#[pyfunction]
#[pyo3(name = "mvn_qmc_project")]
fn mvn_qmc_project_py(
    py: Python<'_>,
    samples: Array,
    root: Array,
    mean: Floats,
    single_precision: bool,
) -> PyResult<ArrayBuf> {
    let samples = samples.matrix("samples")?;
    let root = root.matrix("root")?;
    let mean = Vector::from_vec(mean.0);
    let out = threads::run(py, || {
        mvn_qmc_project(&samples, &root, &mean, single_precision)
    })
    .into_py_result()?;
    Ok(matrix_out(&out).into())
}

/// Persistent Sobol state for the QMC engines when torch is unavailable.
#[pyclass(name = "SobolEngine", module = "botorch_rs._botorch_rs")]
struct PySobolEngine {
    inner: SobolEngine,
    #[pyo3(get)]
    num_generated: usize,
}

#[pymethods]
impl PySobolEngine {
    #[new]
    #[pyo3(signature = (dimension, seed=None))]
    fn new(dimension: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: SobolEngine::new(dimension, seed).into_py_result()?,
            num_generated: 0,
        })
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    fn draw(&mut self, py: Python<'_>, n: usize) -> PyResult<ArrayBuf> {
        let points =
            threads::run(py, || self.inner.draw(n, self.num_generated)).into_py_result()?;
        self.num_generated += n;
        Ok(matrix_out(&points).into())
    }
}

/// `torch_randn(seed, n)`: `torch.randn(n, dtype=torch.float64)` after
/// `torch.manual_seed(seed)`, without torch.
#[pyfunction]
#[pyo3(name = "torch_randn")]
fn torch_randn_py(seed: u64, n: usize) -> ArrayBuf {
    ArrayBuf(TorchGenerator::new(seed).randn(n), vec![n])
}

/// `torch_rand(seed, n, single_precision)`: `torch.rand(n)` in double (or
/// single) precision after `torch.manual_seed(seed)`, without torch.
#[pyfunction]
#[pyo3(name = "torch_rand", signature = (seed, n, single_precision=false))]
fn torch_rand_py(seed: u64, n: usize, single_precision: bool) -> ArrayBuf {
    let mut g = TorchGenerator::new(seed);
    let v = if single_precision {
        g.rand_f32(n).into_iter().map(f64::from).collect()
    } else {
        g.rand(n)
    };
    ArrayBuf(v, vec![n])
}

/// `null_space(C)`: an orthonormal basis (columns) of the null space of `C`.
#[pyfunction]
#[pyo3(name = "null_space")]
fn null_space_py(c: Array) -> PyResult<ArrayBuf> {
    Ok(matrix_out(&null_space(&c.matrix("C")?)).into())
}

/// A single-objective test problem of phase 11a (multi-fidelity,
/// sensitivity-analysis, mixed and constrained synthetic functions).
#[pyclass(
    frozen,
    name = "SingleObjectiveProblem",
    module = "botorch_rs._botorch_rs"
)]
pub struct PySingleObjectiveProblem {
    inner: SingleObjectiveProblem,
}

#[pymethods]
impl PySingleObjectiveProblem {
    /// The problem `name` (a BoTorch class name) with dimension `dim`,
    /// bounds (`2 x d`), discrete inputs, parameters and the trajectory
    /// configuration (`use_smooth_interp`, `max_step_size`).
    #[new]
    #[pyo3(signature = (name, dim, bounds, discrete=vec![], params=vec![], use_smooth_interp=None, max_step_size=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        dim: usize,
        bounds: Array,
        discrete: Vec<usize>,
        params: Vec<f64>,
        use_smooth_interp: Option<bool>,
        max_step_size: Option<f64>,
    ) -> PyResult<Self> {
        let kind = SingleObjectiveKind::from_name(name)
            .ok_or_else(|| unsupported(format!("Unknown test problem {name:?}.")))?;
        let trajectory = use_smooth_interp.map(|smooth| TrajectoryConfig {
            use_smooth_interp: smooth,
            max_step_size: max_step_size.unwrap_or_else(|| default_max_step_size(dim)),
        });
        let inner = SingleObjectiveProblem::new(
            kind,
            dim,
            bounds.matrix("bounds")?,
            discrete,
            params,
            trajectory,
        )
        .into_py_result()?;
        Ok(Self { inner })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bounds(&self) -> ArrayBuf {
        matrix_out(self.inner.bounds()).into()
    }

    #[getter]
    fn num_constraints(&self) -> usize {
        self.inner.num_constraints()
    }

    /// Noiseless values at the points `X` (`... x d`).
    fn evaluate_true(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        let (batch, points) = x.points("X")?;
        let v = threads::run(py, || self.inner.evaluate_true(&points)).into_py_result()?;
        Ok(ArrayBuf(v, batch))
    }

    /// Noiseless constraint slacks at the points `X` (`... x n_c`).
    fn evaluate_slack_true(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        let (mut batch, points) = x.points("X")?;
        let s = threads::run(py, || self.inner.evaluate_slack_true(&points)).into_py_result()?;
        batch.push(s.ncols());
        Ok(ArrayBuf(matrix_out(&s).0, batch))
    }

    /// The evaluation costs of the fidelities at the points `X`.
    fn cost(&self, x: Array) -> PyResult<ArrayBuf> {
        let (batch, points) = x.points("X")?;
        Ok(ArrayBuf(self.inner.cost(&points).into_py_result()?, batch))
    }
}

/// `round_nearest(X, increment, bounds)` element-wise.
#[pyfunction]
#[pyo3(name = "round_nearest", signature = (x, increment, bounds=None))]
fn round_nearest_py(x: Floats, increment: f64, bounds: Option<(f64, f64)>) -> Vec<f64> {
    x.0.iter()
        .map(|&v| round_nearest(v, increment, bounds))
        .collect()
}

/// `data_log_likelihoods(model)`: the log marginal likelihood of the
/// training targets of every element of a `SingleTaskGP` (not divided by
/// `n`, without the log priors), shape `_aug_batch_shape`.
#[pyfunction]
#[pyo3(name = "data_log_likelihoods")]
fn data_log_likelihoods_py(
    py: Python<'_>,
    model: PyRef<'_, crate::models::PySingleTaskGP>,
) -> PyResult<ArrayBuf> {
    let m = model.snapshot();
    let v = threads::run(py, || m.data_log_likelihoods()).into_py_result()?;
    Ok(ArrayBuf(v, m.aug_batch_shape()))
}

/// `sample_cached_cholesky(rows, baseline_l, mean_new, base, batch, n, q,
/// s, max_tries)`: for every batch element (a batch of the joint posterior
/// and an output), the samples `L_q z + μ_q` of the `q` new points from the
/// `q` bottom rows of the joint covariance (`q x (n + q)`), the cached
/// Cholesky factor of the baseline block (`n x n`), the new means (`q`) and
/// the base samples (`(n + q) x s`); returns `batch x q x s` values.
#[pyfunction]
#[pyo3(name = "sample_cached_cholesky")]
#[allow(clippy::too_many_arguments)]
fn sample_cached_cholesky_py(
    py: Python<'_>,
    rows: Floats,
    baseline_l: Floats,
    mean_new: Floats,
    base: Floats,
    batch: usize,
    n: usize,
    q: usize,
    s: usize,
    max_tries: usize,
) -> PyResult<ArrayBuf> {
    let nq = n + q;
    if rows.0.len() != batch * q * nq
        || baseline_l.0.len() != batch * n * n
        || mean_new.0.len() != batch * q
        || base.0.len() != batch * nq * s
    {
        return Err(dimension_error(
            "inconsistent shapes in sample_cached_cholesky",
        ));
    }
    let out = threads::run(py, || {
        let mut out = Vec::with_capacity(batch * q * s);
        for b in 0..batch {
            let r = Matrix::from_row_slice(q, nq, &rows.0[b * q * nq..(b + 1) * q * nq]);
            let l = Matrix::from_row_slice(n, n, &baseline_l.0[b * n * n..(b + 1) * n * n]);
            let bl = r.columns(0, n).into_owned();
            let br = r.columns(n, q).into_owned();
            // bl_chol = (L^{-1} bl^T)^T.
            let bl_chol = if n > 0 {
                l.solve_lower_triangular(&bl.transpose())
                    .ok_or(botorch_core::BotorchError::NotPsd { jitter: 0.0 })?
                    .transpose()
            } else {
                Matrix::zeros(q, 0)
            };
            let br_to_chol = &br - &bl_chol * bl_chol.transpose();
            let br_chol =
                botorch_core::linalg::psd_safe_cholesky_with(&br_to_chol, 1e-8, max_tries)?;
            let mut new_lq = Matrix::zeros(q, nq);
            new_lq.columns_mut(0, n).copy_from(&bl_chol);
            new_lq.columns_mut(n, q).copy_from(br_chol.l());
            let z = Matrix::from_row_slice(nq, s, &base.0[b * nq * s..(b + 1) * nq * s]);
            let res = &new_lq * z;
            for j in 0..q {
                for k in 0..s {
                    out.push(res[(j, k)] + mean_new.0[b * q + j]);
                }
            }
        }
        Ok::<_, botorch_core::BotorchError>(out)
    })
    .into_py_result()?;
    Ok(ArrayBuf(out, vec![batch, q, s]))
}

/// `psd_safe_cholesky(a, batch, n, jitter, max_tries)`: the lower Cholesky
/// factors of a batch of matrices with linear_operator's jitter retries.
#[pyfunction]
#[pyo3(name = "psd_safe_cholesky", signature = (a, batch, n, jitter=None, max_tries=None))]
fn psd_safe_cholesky_py(
    py: Python<'_>,
    a: Floats,
    batch: usize,
    n: usize,
    jitter: Option<f64>,
    max_tries: Option<usize>,
) -> PyResult<ArrayBuf> {
    if a.0.len() != batch * n * n {
        return Err(dimension_error("inconsistent shape of the matrices"));
    }
    let out = threads::run(py, || {
        let mut out = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let m = Matrix::from_row_slice(n, n, &a.0[b * n * n..(b + 1) * n * n]);
            let f = botorch_core::linalg::psd_safe_cholesky_with(
                &m,
                jitter.unwrap_or(1e-8),
                max_tries.unwrap_or(3),
            )?;
            out.extend(matrix_out(f.l()).0);
        }
        Ok::<_, botorch_core::BotorchError>(out)
    })
    .into_py_result()?;
    Ok(ArrayBuf(out, vec![batch, n, n]))
}

/// Registers the functions and classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(elementwise_py, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_prob_feas_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_py, m)?)?;
    m.add_class::<PyMvnxpbState>()?;
    m.add_function(wrap_pyfunction!(mvnxpb_grad_py, m)?)?;
    m.add_function(wrap_pyfunction!(pc_update_py, m)?)?;
    m.add_function(wrap_pyfunction!(pc_pivot_py, m)?)?;
    m.add_function(wrap_pyfunction!(augment_cholesky_py, m)?)?;
    m.add_class::<PyLinearEllipticalSliceSampler>()?;
    m.add_function(wrap_pyfunction!(normal_qmc_py, m)?)?;
    m.add_function(wrap_pyfunction!(mvn_qmc_root_py, m)?)?;
    m.add_function(wrap_pyfunction!(mvn_qmc_project_py, m)?)?;
    m.add_class::<PySobolEngine>()?;
    m.add_function(wrap_pyfunction!(torch_randn_py, m)?)?;
    m.add_function(wrap_pyfunction!(torch_rand_py, m)?)?;
    m.add_function(wrap_pyfunction!(null_space_py, m)?)?;
    m.add_class::<PySingleObjectiveProblem>()?;
    m.add_function(wrap_pyfunction!(round_nearest_py, m)?)?;
    m.add_function(wrap_pyfunction!(data_log_likelihoods_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_cached_cholesky_py, m)?)?;
    m.add_function(wrap_pyfunction!(psd_safe_cholesky_py, m)?)?;
    Ok(())
}
