//! Reverse communication exposes independent native L-BFGS-B states to the
//! Python batching scheduler without storing Python objective callbacks.
use botorch_core::lbfgsb::{LbfgsbOptions, LbfgsbState, LbfgsbStatus, LbfgsbTask};
use pyo3::prelude::*;

use crate::errors::value_error;
use crate::threads;

#[pyclass(
    name = "_LbfgsbState",
    module = "botorch_rs._botorch_rs",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLbfgsbState {
    inner: LbfgsbState,
}

fn status_code(status: LbfgsbStatus) -> (i32, i32) {
    match status {
        LbfgsbStatus::ConvergedProjectedGradient => (4, 401),
        LbfgsbStatus::ConvergedRelativeReduction => (4, 402),
        LbfgsbStatus::MaxIterations => (5, 504),
        LbfgsbStatus::MaxEvaluations => (5, 502),
        LbfgsbStatus::CallbackStop => (5, 505),
        LbfgsbStatus::AbnormalLineSearch => (8, 0),
    }
}

#[pymethods]
impl PyLbfgsbState {
    #[new]
    #[pyo3(signature = (x0, lower, upper, maxcor=10, ftol=2.220446049250313e-9, gtol=1e-5, maxiter=15000, maxfun=usize::MAX, maxls=20))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x0: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        maxcor: usize,
        ftol: f64,
        gtol: f64,
        maxiter: usize,
        maxfun: usize,
        maxls: usize,
    ) -> PyResult<Self> {
        let options = LbfgsbOptions {
            maxcor,
            ftol,
            gtol,
            maxiter,
            maxfun,
            maxls,
        };
        Ok(Self {
            inner: LbfgsbState::new(&x0, &lower, &upper, options).map_err(value_error)?,
        })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }
    #[getter]
    fn f(&self) -> f64 {
        self.inner.value()
    }
    #[getter]
    fn g(&self) -> Vec<f64> {
        self.inner.gradient().to_vec()
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations()
    }
    #[getter]
    fn evaluations(&self) -> usize {
        self.inner.evaluations()
    }
    #[getter]
    fn updates(&self) -> usize {
        self.inner.updates()
    }
    #[getter]
    fn task(&self) -> (i32, i32) {
        match self.inner.task() {
            LbfgsbTask::Evaluate => (3, 0),
            LbfgsbTask::NewIterate => (1, 0),
            LbfgsbTask::Finished(status) => status_code(status),
        }
    }

    fn provide_evaluation(&mut self, py: Python<'_>, f: f64, gradient: Vec<f64>) -> PyResult<()> {
        threads::run(py, || self.inner.provide_evaluation(f, &gradient))
            .map(|_| ())
            .map_err(value_error)
    }

    fn resume(&mut self, py: Python<'_>, stop: bool) -> PyResult<()> {
        threads::run(py, || self.inner.resume(stop))
            .map(|_| ())
            .map_err(value_error)
    }

    fn corrections(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        self.inner.corrections()
    }

    fn inverse_hessian_product(&self, py: Python<'_>, vector: Vec<f64>) -> PyResult<Vec<f64>> {
        threads::run(py, || self.inner.inverse_hessian_product(&vector)).map_err(value_error)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLbfgsbState>()
}
