//! Native preference-plane selection and continuous FPBI bindings.

use crate::arrays::{matrix_out, Array, BoolBuf};
use crate::errors::{dimension_error, to_pyerr};
use crate::threads;
use botorch_core::utils::multi_objective::decision_making::{
    knee_region_batch, select_preferred_batch,
};
use botorch_core::utils::multi_objective::fpbi::{
    fpbi_geometry, generate_fpbi_front, FpbiGeometry, FpbiOptions, VectorFunction,
};
use botorch_core::utils::multi_objective::PointBatch;
use botorch_core::Result;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn points(y: &Array) -> PyResult<PointBatch> {
    if y.1.len() < 2 {
        return Err(dimension_error("Y must have shape [..., n, m]"));
    }
    let m = y.1[y.1.len() - 1];
    let n = y.1[y.1.len() - 2];
    PointBatch::new(y.1[..y.1.len() - 2].to_vec(), n, m, y.0.clone()).map_err(to_pyerr)
}

#[pyfunction]
#[pyo3(signature=(y,alpha=0.5,maximize=true))]
fn decision_knee_region(py: Python<'_>, y: Array, alpha: f64, maximize: bool) -> PyResult<BoolBuf> {
    let y = points(&y)?;
    let mut shape = y.batch_shape().to_vec();
    shape.push(y.n());
    let mask = threads::run(py, || knee_region_batch(&y, alpha, maximize)).map_err(to_pyerr)?;
    Ok(BoolBuf(mask, shape))
}

#[pyfunction]
#[pyo3(signature=(y,preferences,alpha=0.5,maximize=true))]
fn decision_select_preferred(
    py: Python<'_>,
    y: Array,
    preferences: Vec<f64>,
    alpha: f64,
    maximize: bool,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let y = points(&y)?;
    let shape = y.batch_shape().to_vec();
    let indices = threads::run(py, || {
        select_preferred_batch(&y, &preferences, alpha, maximize)
    })
    .map_err(to_pyerr)?;
    Ok((indices, shape))
}

fn geometry_dict(py: Python<'_>, g: &FpbiGeometry) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("utopia", g.utopia.clone())?;
    out.set_item("nadir", g.nadir.clone())?;
    out.set_item("ranges", g.ranges.clone())?;
    out.set_item("active_objectives", g.active_objectives.clone())?;
    out.set_item("normalized_extremes", matrix_out(&g.normalized_extremes))?;
    out.set_item("farthest_pair", g.farthest_pair)?;
    out.set_item("center", g.center.clone())?;
    out.set_item("focus", g.focus.clone())?;
    out.set_item("search_direction", g.search_direction.clone())?;
    out.set_item("direction", g.search_direction.clone())?;
    // Public layout is objective dimensions x basis directions.
    out.set_item("basis", matrix_out(&g.basis.transpose()))?;
    out.set_item("grid", matrix_out(&g.anchors))?;
    out.set_item("anchors", matrix_out(&g.anchors))?;
    out.set_item("degenerate", g.degenerate)?;
    out.set_item("used_degenerate_fallback", g.used_degenerate_fallback)?;
    out.set_item("used_fallback_basis", g.used_degenerate_fallback)?;
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature=(extreme_y,n_steps=10,focus=None,maximize=true,max_grid_points=10000))]
fn decision_fpbi_geometry(
    py: Python<'_>,
    extreme_y: Array,
    n_steps: usize,
    focus: Option<Vec<f64>>,
    maximize: bool,
    max_grid_points: usize,
) -> PyResult<Py<PyDict>> {
    let extreme_y = extreme_y.matrix("extreme_Y")?;
    let options = FpbiOptions {
        maximize,
        grid_resolution: n_steps,
        max_grid_points,
        ..Default::default()
    };
    let geometry = threads::run(py, || fpbi_geometry(&extreme_y, focus.as_deref(), &options))
        .map_err(to_pyerr)?;
    geometry_dict(py, &geometry)
}

fn call_vector(callback: &Py<PyAny>, x: &[f64]) -> Result<Vec<f64>> {
    Python::attach(|py| {
        callback
            .bind(py)
            .call1((x.to_vec(),))?
            .extract::<Vec<f64>>()
    })
    .map_err(crate::callbacks::callback_error)
}

/// Low-level callbacks accept a single list of decisions and return a list.
/// Inequalities follow the BoTorch/SciPy `>= 0` sign convention.
#[allow(clippy::too_many_arguments)] // Mirrors the public Python keyword options.
#[pyfunction]
#[pyo3(signature=(objective,bounds,initial_conditions,n_steps=10,focus=None,maximize=true,maxiter=200,tol=1e-8,inequality_constraints=None,equality_constraints=None,feasibility_tolerance=1e-6,finite_difference_step=1e-6,max_grid_points=10000))]
fn decision_fpbi_front(
    py: Python<'_>,
    objective: Py<PyAny>,
    bounds: Array,
    initial_conditions: Array,
    n_steps: usize,
    focus: Option<Vec<f64>>,
    maximize: bool,
    maxiter: usize,
    tol: f64,
    inequality_constraints: Option<Py<PyAny>>,
    equality_constraints: Option<Py<PyAny>>,
    feasibility_tolerance: f64,
    finite_difference_step: f64,
    max_grid_points: usize,
) -> PyResult<Py<PyDict>> {
    let bounds = bounds.matrix("bounds")?;
    let starts = initial_conditions.matrix("initial_conditions")?;
    let options = FpbiOptions {
        maximize,
        grid_resolution: n_steps,
        max_grid_points,
        maxiter,
        ftol: tol,
        feasibility_tolerance,
        finite_difference_step,
    };
    let result = threads::run(py, || {
        let f = |x: &[f64]| call_vector(&objective, x);
        let ineq = |x: &[f64]| match &inequality_constraints {
            Some(c) => call_vector(c, x),
            None => Ok(vec![]),
        };
        let eq = |x: &[f64]| match &equality_constraints {
            Some(c) => call_vector(c, x),
            None => Ok(vec![]),
        };
        generate_fpbi_front(
            &f,
            &bounds,
            &starts,
            focus.as_deref(),
            inequality_constraints
                .as_ref()
                .map(|_| &ineq as &VectorFunction<'_>),
            equality_constraints
                .as_ref()
                .map(|_| &eq as &VectorFunction<'_>),
            &options,
        )
    })
    .map_err(to_pyerr)?;
    let out = PyDict::new(py);
    out.set_item("X", matrix_out(&result.x))?;
    out.set_item("Y", matrix_out(&result.y))?;
    out.set_item("anchor_indices", result.anchor_indices)?;
    out.set_item("extreme_X", matrix_out(&result.extreme_x))?;
    out.set_item("extreme_Y", matrix_out(&result.extreme_y))?;
    out.set_item("geometry", geometry_dict(py, &result.geometry)?)?;
    let mut statuses = Vec::new();
    for status in result.statuses {
        let item = PyDict::new(py);
        item.set_item("index", status.index)?;
        item.set_item("extreme", status.extreme)?;
        item.set_item("mode", status.mode)?;
        item.set_item("message", status.message)?;
        item.set_item("max_violation", status.max_violation)?;
        item.set_item("feasible", status.feasible)?;
        item.set_item("success", status.success)?;
        item.set_item("iterations", status.iterations)?;
        item.set_item("restarts", status.restarts)?;
        item.set_item("kappa", status.kappa)?;
        statuses.push(item.unbind());
    }
    out.set_item("statuses", statuses)?;
    Ok(out.unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decision_knee_region, m)?)?;
    m.add_function(wrap_pyfunction!(decision_select_preferred, m)?)?;
    m.add_function(wrap_pyfunction!(decision_fpbi_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(decision_fpbi_front, m)?)?;
    Ok(())
}
