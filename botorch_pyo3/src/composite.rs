//! Composite and callable acquisition functions (botorch-rust extensions):
//!
//! * `WeightedLogMeanExp(terms, weights)` of `botorch_core`, the weighted
//!   log-mean-exp blend of acquisition functions, evaluated in Rust;
//! * acquisition functions given as Python callables on torch tensors (any
//!   `torch.nn.Module` whose `forward(X)` maps `batch x q x d` candidate sets
//!   to `batch` values, as BoTorch's `optimize_acqf` accepts them): values
//!   come from Python and gradients from `torch.autograd`, through
//!   `botorch_rs._torch_bridge`. Calls acquire the GIL, so they run one at a
//!   time while the optimizer's restarts run in parallel; a batch of
//!   candidate sets (e.g. the raw samples) is evaluated in one call.

use std::sync::Arc;

use botorch_core::acquisition::{AcquisitionFunction, WeightedLogMeanExp};
use botorch_core::{BotorchError, JointPosteriorGrad, Matrix, Model, Posterior, Result};
use pyo3::prelude::*;

use crate::acquisition::PyAcquisitionFunction;
use crate::arrays::{float_data, float_values};
use crate::callbacks::{callback_error, rows};
use crate::errors::{value_error, IntoPyResult};

const BRIDGE: &str = "botorch_rs._torch_bridge";

/// `WeightedLogMeanExp(acq_functions, weights)`: `log(Σ w_i exp(α_i(X)) /
/// Σ w_i)` over botorch_rs acquisition functions.
#[pyfunction]
pub fn weighted_log_mean_exp(
    terms: Vec<PyRef<'_, PyAcquisitionFunction>>,
    weights: Vec<f64>,
) -> PyResult<PyAcquisitionFunction> {
    let terms = terms.iter().map(|t| t.inner.clone()).collect();
    let a = WeightedLogMeanExp::new(terms, weights).into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "WeightedLogMeanExp",
    ))
}

/// An acquisition function given by a Python callable on torch tensors
/// (`callable(X)` with `X` of shape `batch x q x d`, returning `batch`
/// values), for candidate sets of dimension `dim`.
#[pyfunction]
pub fn callable_acquisition_function(
    callable: Py<PyAny>,
    dim: usize,
) -> PyResult<PyAcquisitionFunction> {
    if dim == 0 {
        return Err(value_error(
            "a callable acquisition function needs a positive input dimension",
        ));
    }
    let a = PyCallableAcquisition {
        callable,
        model: Arc::new(CallableInputs { dim }),
    };
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "CallableAcquisitionFunction",
    ))
}

/// The stand-in model of a callable acquisition function: only the input
/// dimension is known (the callable may use any models, or none).
#[derive(Debug)]
struct CallableInputs {
    dim: usize,
}

fn no_model<T>() -> Result<T> {
    Err(BotorchError::Unsupported(
        "a callable acquisition function has no botorch_rs model".into(),
    ))
}

impl Model for CallableInputs {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn posterior(&self, _x: &Matrix, _observation_noise: bool) -> Result<Posterior> {
        no_model()
    }

    fn joint_posterior_with_grad(&self, _x: &Matrix) -> Result<JointPosteriorGrad> {
        no_model()
    }
}

/// An acquisition function evaluated by a Python callable.
#[derive(Debug)]
struct PyCallableAcquisition {
    callable: Py<PyAny>,
    model: Arc<dyn Model>,
}

impl PyCallableAcquisition {
    /// Values (and with `with_grad`, the gradients, row-major like `xs`) at
    /// the candidate sets `xs` (all `q x d`), in one call of the callable.
    fn call(&self, xs: &[Matrix], with_grad: bool) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
        let d = self.model.dim();
        let q = xs.first().map_or(0, |x| x.nrows());
        let mut data = Vec::with_capacity(xs.len() * q * d);
        for x in xs {
            if x.ncols() != d || x.nrows() != q || q == 0 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected candidate sets of shape {q} x {d}, got {}x{}",
                    x.nrows(),
                    x.ncols()
                )));
            }
            data.extend(rows(x));
        }
        let shape = vec![xs.len(), q, d];
        let (values, grads) = Python::attach(|py| -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
            let bridge = py.import(BRIDGE)?;
            let res = bridge.call_method1(
                "acquisition_forward",
                (
                    self.callable.bind(py),
                    float_values(py, &data)?,
                    shape,
                    with_grad,
                ),
            )?;
            let (values, grads): (Bound<'_, PyAny>, Option<Bound<'_, PyAny>>) = res.extract()?;
            Ok((
                float_data(&values)?,
                grads.map(|g| float_data(&g)).transpose()?,
            ))
        })
        .map_err(callback_error)?;
        if values.len() != xs.len() || grads.as_ref().is_some_and(|g| g.len() != data.len()) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the acquisition function returned {} values for {} candidate sets",
                values.len(),
                xs.len()
            )));
        }
        Ok((values, grads))
    }
}

impl AcquisitionFunction for PyCallableAcquisition {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.call(std::slice::from_ref(x), false)?.0[0])
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (values, grads) = self.call(std::slice::from_ref(x), true)?;
        let grads = grads.expect("gradient requested");
        Ok((
            values[0],
            Matrix::from_row_slice(x.nrows(), x.ncols(), &grads),
        ))
    }

    /// One call of the callable for all candidate sets of the same size.
    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        if xs.iter().all(|x| x.nrows() == xs[0].nrows()) {
            return Ok(self.call(xs, false)?.0);
        }
        xs.iter().map(|x| self.forward(x)).collect()
    }

    /// The callable's `X_pending` attribute (if it has one).
    fn x_pending(&self) -> Option<Matrix> {
        let pending = Python::attach(|py| -> PyResult<Option<(Vec<f64>, Vec<usize>)>> {
            let bridge = py.import(BRIDGE)?;
            let res = bridge.call_method1("acquisition_x_pending", (self.callable.bind(py),))?;
            let pending: Option<(Bound<'_, PyAny>, Vec<usize>)> = res.extract()?;
            pending
                .map(|(data, shape)| Ok((float_data(&data)?, shape)))
                .transpose()
        })
        .ok()
        .flatten()?;
        match pending.1.as_slice() {
            &[r, c] if r * c == pending.0.len() => Some(Matrix::from_row_slice(r, c, &pending.0)),
            _ => None,
        }
    }

    /// The callable's `set_X_pending` (BoTorch's sequential optimization
    /// calls it; [`BotorchError::Unsupported`] if the callable has none).
    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        let arg = x_pending.map(|p| (rows(p), vec![p.nrows(), p.ncols()]));
        Python::attach(|py| -> PyResult<()> {
            let bridge = py.import(BRIDGE)?;
            let arg = match &arg {
                Some((data, shape)) => Some((float_values(py, data)?, shape.clone())),
                None => None,
            };
            bridge.call_method1("acquisition_set_x_pending", (self.callable.bind(py), arg))?;
            Ok(())
        })
        .map_err(callback_error)
    }
}

/// Registers the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(weighted_log_mean_exp, m)?)?;
    m.add_function(wrap_pyfunction!(callable_acquisition_function, m)?)?;
    Ok(())
}
