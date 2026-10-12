//! Multi-output acquisition functions (`botorch.acquisition.multioutput_acquisition`)
//! and their optimization with NSGA-II
//! (`botorch.utils.multi_objective.optimize.optimize_with_nsgaii`), with the
//! deterministic NSGA-II building blocks (non-dominated sorting and crowding
//! distances as pymoo computes them).
//!
//! `optimize_with_nsgaii` evaluates the population either natively (a
//! [`PyMultiOutputAcquisitionFunction`] without objective or outcome
//! constraints: the whole run stays in Rust with the GIL released) or
//! through a Python callable `evaluate((data, shape))` returning the
//! objectives and the outcome-constraint values of the points (BoTorch's
//! `objective` and `constraints` callables applied to the acquisition
//! values).

use std::sync::Arc;

use botorch_core::acquisition::multioutput_acquisition::{
    MultiOutputAcquisitionFunction, MultiOutputAcquisitionFunctionWrapper, MultiOutputPosteriorMean,
};
use botorch_core::acquisition::AcquisitionFunction;
use botorch_core::utils::multi_objective::optimize::{
    crowding_distance, non_dominated_fronts, optimize_with_nsgaii, Evaluation,
    InequalityConstraint, OptimizeWithNsgaiiOptions,
};
use botorch_core::{BotorchError, Matrix};
use pyo3::prelude::*;

use crate::acquisition::PyAcquisitionFunction;
use crate::arrays::{float_values, matrix_out, Array, ArrayBuf};
use crate::callbacks::callback_error;
use crate::errors::{dimension_error, IntoPyResult};
use crate::input_perturbation::row_major;
use crate::models::PyModel;
use crate::threads;

/// A multi-output acquisition function evaluated in Rust.
#[pyclass(
    frozen,
    name = "MultiOutputAcquisitionFunction",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyMultiOutputAcquisitionFunction {
    inner: Arc<dyn MultiOutputAcquisitionFunction>,
}

#[pymethods]
impl PyMultiOutputAcquisitionFunction {
    /// `MultiOutputPosteriorMean(model, weights)`.
    #[staticmethod]
    #[pyo3(signature = (model, weights=None))]
    fn posterior_mean(model: PyRef<'_, PyModel>, weights: Option<Vec<f64>>) -> PyResult<Self> {
        let inner =
            MultiOutputPosteriorMean::new(Arc::clone(&model.inner), weights).into_py_result()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// `MultiOutputAcquisitionFunctionWrapper(acqfs)`.
    #[staticmethod]
    fn wrapper(acqfs: Vec<PyRef<'_, PyAcquisitionFunction>>) -> PyResult<Self> {
        let acqfs: Vec<Arc<dyn AcquisitionFunction>> =
            acqfs.iter().map(|a| Arc::clone(&a.inner)).collect();
        let inner = MultiOutputAcquisitionFunctionWrapper::new(acqfs).into_py_result()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// The number of outputs `m`.
    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    /// The input dimension `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The values (`... x m`) of the candidate sets `X` (`... x q x d`).
    fn forward(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        let (batch, sets) = x.batches("X")?;
        let acqf = &self.inner;
        let values = threads::run(py, || acqf.forward_many(&sets)).into_py_result()?;
        let mut shape = batch;
        shape.push(values.ncols());
        Ok(ArrayBuf(row_major(&values), shape))
    }
}

/// An `n × c` matrix from a `(data, shape)` pair returned by Python.
fn matrix_from(pair: &Array, rows: usize, what: &str) -> botorch_core::Result<Matrix> {
    let c = match pair.1.as_slice() {
        [r, c] if *r == rows => *c,
        [r] if *r == rows => 1,
        s => {
            return Err(BotorchError::DimensionMismatch(format!(
                "the {what} of {rows} points have shape {s:?}, expected {rows} x k"
            )))
        }
    };
    if pair.0.len() != rows * c {
        return Err(BotorchError::DimensionMismatch(format!(
            "the {what} do not match their shape"
        )));
    }
    Ok(Matrix::from_row_slice(rows, c, &pair.0))
}

/// BoTorch's `optimize_with_nsgaii` (see the module documentation); returns
/// the Pareto set, the Pareto front, BoTorch's warnings and the number of
/// NSGA-II runs.
#[pyfunction]
#[pyo3(
    name = "optimize_with_nsgaii",
    signature = (
        bounds,
        num_objectives,
        acqf=None,
        evaluate=None,
        q=None,
        ref_point=None,
        inequality_constraints=Vec::new(),
        population_size=250,
        max_gen=None,
        seed=None,
        fixed_features=Vec::new(),
        max_attempts=2,
        discrete_choices=Vec::new(),
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn optimize_with_nsgaii_py(
    py: Python<'_>,
    bounds: Array,
    num_objectives: usize,
    acqf: Option<PyRef<'_, PyMultiOutputAcquisitionFunction>>,
    evaluate: Option<Py<PyAny>>,
    q: Option<usize>,
    ref_point: Option<Vec<f64>>,
    inequality_constraints: Vec<InequalityConstraint>,
    population_size: usize,
    max_gen: Option<usize>,
    seed: Option<u64>,
    fixed_features: Vec<(usize, f64)>,
    max_attempts: usize,
    discrete_choices: Vec<(usize, Vec<f64>)>,
) -> PyResult<(ArrayBuf, ArrayBuf, Vec<String>, usize)> {
    let bounds = bounds.matrix("bounds")?;
    let options = OptimizeWithNsgaiiOptions {
        q,
        ref_point,
        inequality_constraints,
        population_size,
        max_gen,
        seed,
        fixed_features,
        max_attempts,
        discrete_choices,
    };
    let native = acqf.map(|a| Arc::clone(&a.inner));
    let result = match (native, evaluate) {
        (Some(acqf), None) => threads::run(py, || {
            optimize_with_nsgaii(
                |x: &Matrix| -> botorch_core::Result<Evaluation> {
                    let sets: Vec<Matrix> =
                        (0..x.nrows()).map(|i| x.rows(i, 1).into_owned()).collect();
                    Ok((acqf.forward_many(&sets)?, None))
                },
                &bounds,
                num_objectives,
                &options,
            )
        }),
        (None, Some(f)) => threads::run(py, || {
            optimize_with_nsgaii(
                |x: &Matrix| -> botorch_core::Result<Evaluation> {
                    let n = x.nrows();
                    let data = row_major(x);
                    let (obj, cons) = Python::attach(|py| -> PyResult<(Array, Option<Array>)> {
                        f.bind(py)
                            .call1(((float_values(py, &data)?, vec![n, x.ncols()]),))?
                            .extract()
                    })
                    .map_err(callback_error)?;
                    let obj = matrix_from(&obj, n, "objectives")?;
                    let cons = cons
                        .map(|c| matrix_from(&c, n, "constraint values"))
                        .transpose()?;
                    Ok((obj, cons))
                },
                &bounds,
                num_objectives,
                &options,
            )
        }),
        _ => {
            return Err(crate::errors::value_error(
                "optimize_with_nsgaii needs exactly one of `acqf` and `evaluate`",
            ))
        }
    }
    .into_py_result()?;
    Ok((
        matrix_out(&result.x).into(),
        matrix_out(&result.y).into(),
        result.warnings,
        result.attempts,
    ))
}

/// The Pareto fronts of `F` (`n x m`, minimized) as pymoo's
/// `NonDominatedSorting` computes them (each front in increasing index
/// order).
#[pyfunction]
#[pyo3(signature = (f, n_stop_if_ranked=None))]
pub fn nsga2_non_dominated_fronts(
    f: Array,
    n_stop_if_ranked: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    Ok(non_dominated_fronts(&f.matrix("F")?, n_stop_if_ranked))
}

/// The crowding distances of the points of one front `F` (`n x m`) as
/// pymoo's `calc_crowding_distance` computes them.
#[pyfunction]
pub fn nsga2_crowding_distance(f: Array) -> PyResult<Vec<f64>> {
    let f = f.matrix("F")?;
    if f.ncols() == 0 {
        return Err(dimension_error("F must have at least one objective"));
    }
    Ok(crowding_distance(&f))
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiOutputAcquisitionFunction>()?;
    m.add_function(wrap_pyfunction!(optimize_with_nsgaii_py, m)?)?;
    m.add_function(wrap_pyfunction!(nsga2_non_dominated_fronts, m)?)?;
    m.add_function(wrap_pyfunction!(nsga2_crowding_distance, m)?)?;
    Ok(())
}
