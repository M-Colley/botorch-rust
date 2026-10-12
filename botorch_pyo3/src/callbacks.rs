//! Python callables used by the Rust core: generic MC objectives and outcome
//! constraints of acquisition functions (BoTorch's `GenericMCObjective`,
//! `ConstrainedMCObjective`, multi-output objectives and `constraints`
//! callables), nonlinear parameter constraints of `optimize_acqf` and
//! initial-condition generators.
//!
//! The callables operate on torch tensors, exactly as in BoTorch. Every call
//! goes through the pure-Python module `botorch_rs._torch_bridge`: the Rust
//! side passes posterior samples (and inputs) as flat float64 buffers
//! (`array.array('d')`; lists for a few values, see [`crate::arrays`]), the
//! bridge builds tensors that require gradients, evaluates the callable and
//! keeps the autograd graph in an opaque handle; the reverse pass hands the
//! adjoint of the outputs back to the bridge, which computes the
//! vector-Jacobian product with `torch.autograd.grad`. Values and gradients
//! come back the same way. Calls acquire the GIL (they come from worker
//! threads while the GIL is released). A Python exception raised by a callable
//! is kept and re-raised unchanged when the Rust computation returns (see
//! [`take_error`]).

use std::sync::{Arc, Mutex};

use botorch_core::acquisition::multi_objective::{McMultiOutputObjective, MultiOutputValues};
use botorch_core::acquisition::{McObjective, McSamples, SamplePullback, SampleValues};
use botorch_core::optim::{IcGenerator, IcRequest, NonlinearInequalityConstraint};
use botorch_core::utils::objective::McConstraint;
use botorch_core::{BotorchError, Matrix, Result};
use pyo3::prelude::*;

use crate::arrays::{float_data, float_values, Array};

/// The first Python exception raised by a callable during the current
/// computation.
static PENDING: Mutex<Option<PyErr>> = Mutex::new(None);

/// Forgets a pending exception (called before every computation).
pub fn clear_error() {
    *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// The pending Python exception, if a callable raised one.
pub fn take_error() -> Option<PyErr> {
    PENDING.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Records a Python exception and returns the core error that aborts the
/// computation.
fn callback_error(e: PyErr) -> BotorchError {
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let msg = Python::attach(|py| e.value(py).to_string());
    if pending.is_none() {
        *pending = Some(e);
    }
    BotorchError::Runtime(format!("a Python callable raised an exception: {msg}"))
}

const BRIDGE: &str = "botorch_rs._torch_bridge";

/// Row-major data of a matrix.
fn rows(m: &Matrix) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len());
    for i in 0..m.nrows() {
        out.extend(m.row(i).iter());
    }
    out
}

/// Values of shape `(S, 1, n)` (row-major) as an `n × S` matrix.
fn values_matrix(values: &[f64], s: usize, n: usize) -> Result<Matrix> {
    if values.len() != s * n {
        return Err(BotorchError::Runtime(format!(
            "The q-batch shape of the objective values does not agree with the q-batch shape \
             of X: expected {s} x {n} values, got {}.",
            values.len()
        )));
    }
    Ok(Matrix::from_fn(n, s, |p, c| values[c * n + p]))
}

/// An `n × S` adjoint as a row-major `(S, 1, n)` list.
fn adjoint_list(bar: &Matrix) -> Vec<f64> {
    let (n, s) = bar.shape();
    let mut out = Vec::with_capacity(n * s);
    for c in 0..s {
        for p in 0..n {
            out.push(bar[(p, c)]);
        }
    }
    out
}

/// A callable `f(samples, X)` evaluated through the bridge: an MC objective
/// (`pass_x`: BoTorch's `objective(samples, X=X)`) or an outcome constraint
/// (`constraint(samples)`).
#[derive(Debug)]
pub struct PyCallback {
    callable: Py<PyAny>,
    kind: &'static str,
    pass_x: bool,
}

impl PyCallback {
    /// An MC objective (`kind = "objective"`) called as `f(samples, X=X)` if
    /// `pass_x`, else `f(samples, X=None)`.
    pub fn objective(callable: Py<PyAny>, pass_x: bool) -> Self {
        Self {
            callable,
            kind: "objective",
            pass_x,
        }
    }

    /// An outcome constraint called as `f(samples)`.
    pub fn constraint(callable: Py<PyAny>) -> Self {
        Self {
            callable,
            kind: "constraint",
            pass_x: false,
        }
    }

    /// A multi-output MC objective (BoTorch's `MCMultiOutputObjective`)
    /// called as `f(samples, X=X)`, returning `S x 1 x n x m'` values.
    pub fn mo_objective(callable: Py<PyAny>) -> Self {
        Self {
            callable,
            kind: "mo_objective",
            pass_x: true,
        }
    }

    fn call_mo(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        let (s, n, m) = (
            samples.num_samples(),
            samples.num_points(),
            samples.num_outputs(),
        );
        let x_arg: Option<(Vec<f64>, usize)> = x.map(|x| (rows(x), x.ncols()));
        let d = x_arg.as_ref().map(|(_, d)| *d);
        let (values, mo, handle) = Python::attach(|py| -> PyResult<(Vec<f64>, usize, Py<PyAny>)> {
            let bridge = py.import(BRIDGE)?;
            let x_arg = match &x_arg {
                Some((data, d)) => Some((float_values(py, data)?, *d)),
                None => None,
            };
            let res = bridge.call_method1(
                "forward_mo",
                (
                    self.callable.bind(py),
                    float_values(py, &samples.to_row_major())?,
                    (s, n, m),
                    x_arg,
                    with_grad,
                ),
            )?;
            let (values, mo, handle): (Bound<'_, PyAny>, usize, Py<PyAny>) = res.extract()?;
            Ok((float_data(&values)?, mo, handle))
        })
        .map_err(callback_error)?;
        let values = McSamples::from_row_major(&values, s, n, mo)?;
        let pullback: Option<SamplePullback> = if with_grad {
            Some(Box::new(move |bar: &Matrix| {
                let adjoint = McSamples::new(bar.clone(), n, mo)?.to_row_major();
                let (sbar, xbar) = Python::attach(|py| -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
                    let bridge = py.import(BRIDGE)?;
                    let adjoint = float_values(py, &adjoint)?;
                    let res = bridge.call_method1("backward", (handle, adjoint))?;
                    let (sbar, xbar): (Bound<'_, PyAny>, Option<Bound<'_, PyAny>>) =
                        res.extract()?;
                    Ok((
                        float_data(&sbar)?,
                        xbar.map(|x| float_data(&x)).transpose()?,
                    ))
                })
                .map_err(callback_error)?;
                let sbar = McSamples::from_row_major(&sbar, s, n, m)?.into_data();
                let xbar = match (xbar, d) {
                    (Some(v), Some(d)) if v.len() == n * d => {
                        Some(Matrix::from_row_slice(n, d, &v))
                    }
                    (Some(v), Some(d)) => {
                        return Err(BotorchError::DimensionMismatch(format!(
                            "the input gradient has {} values, expected {}",
                            v.len(),
                            n * d
                        )))
                    }
                    _ => None,
                };
                Ok((sbar, xbar))
            }))
        } else {
            None
        };
        Ok(MultiOutputValues { values, pullback })
    }

    fn call(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let (s, n, m) = (
            samples.num_samples(),
            samples.num_points(),
            samples.num_outputs(),
        );
        let x_arg: Option<(Vec<f64>, usize)> = if self.pass_x {
            x.map(|x| (rows(x), x.ncols()))
        } else {
            None
        };
        let d = x_arg.as_ref().map(|(_, d)| *d);
        let (values, handle) = Python::attach(|py| -> PyResult<(Vec<f64>, Py<PyAny>)> {
            let bridge = py.import(BRIDGE)?;
            let x_arg = match &x_arg {
                Some((data, d)) => Some((float_values(py, data)?, *d)),
                None => None,
            };
            let res = bridge.call_method1(
                "forward",
                (
                    self.callable.bind(py),
                    self.kind,
                    float_values(py, &samples.to_row_major())?,
                    (s, n, m),
                    x_arg,
                    with_grad,
                ),
            )?;
            let (values, handle): (Bound<'_, PyAny>, Py<PyAny>) = res.extract()?;
            Ok((float_data(&values)?, handle))
        })
        .map_err(callback_error)?;
        let values = values_matrix(&values, s, n)?;
        let pullback: Option<SamplePullback> = if with_grad {
            Some(Box::new(move |bar: &Matrix| {
                let (sbar, xbar) = Python::attach(|py| -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
                    let bridge = py.import(BRIDGE)?;
                    let adjoint = float_values(py, &adjoint_list(bar))?;
                    let res = bridge.call_method1("backward", (handle, adjoint))?;
                    let (sbar, xbar): (Bound<'_, PyAny>, Option<Bound<'_, PyAny>>) =
                        res.extract()?;
                    Ok((
                        float_data(&sbar)?,
                        xbar.map(|x| float_data(&x)).transpose()?,
                    ))
                })
                .map_err(callback_error)?;
                let sbar = McSamples::from_row_major(&sbar, s, n, m)?.into_data();
                let xbar = match (xbar, d) {
                    (Some(v), Some(d)) => {
                        if v.len() != n * d {
                            return Err(BotorchError::DimensionMismatch(format!(
                                "the input gradient has {} values, expected {}",
                                v.len(),
                                n * d
                            )));
                        }
                        Some(Matrix::from_row_slice(n, d, &v))
                    }
                    _ => None,
                };
                Ok((sbar, xbar))
            }))
        } else {
            None
        };
        Ok(SampleValues { values, pullback })
    }
}

impl McObjective for PyCallback {
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        self.call(samples, x, with_grad)
    }
}

impl McMultiOutputObjective for PyCallback {
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        self.call_mo(samples, x, with_grad)
    }
}

impl McConstraint for PyCallback {
    fn evaluate(&self, samples: &McSamples, with_grad: bool) -> Result<SampleValues> {
        self.call(samples, None, with_grad)
    }
}

/// A nonlinear parameter constraint `f(x) >= 0` given by a Python callable on
/// torch tensors (`x` of shape `d` for intra-point constraints, `q x d` for
/// inter-point constraints), with its gradient from torch.autograd.
#[derive(Debug)]
pub struct PyNonlinearConstraint {
    callable: Py<PyAny>,
    intrapoint: bool,
    d: usize,
}

impl PyNonlinearConstraint {
    /// The constraint `callable` for points of dimension `d`.
    pub fn new(callable: Py<PyAny>, intrapoint: bool, d: usize) -> Self {
        Self {
            callable,
            intrapoint,
            d,
        }
    }
}

impl NonlinearInequalityConstraint for PyNonlinearConstraint {
    fn is_intrapoint(&self) -> bool {
        self.intrapoint
    }

    fn evaluate(&self, x: &[f64]) -> Result<(f64, Vec<f64>)> {
        let shape: Vec<usize> = if self.intrapoint {
            vec![x.len()]
        } else {
            vec![x.len() / self.d.max(1), self.d]
        };
        let (v, g): (f64, Vec<f64>) = Python::attach(|py| -> PyResult<(f64, Vec<f64>)> {
            let bridge = py.import(BRIDGE)?;
            bridge
                .call_method1(
                    "nonlinear_constraint",
                    (self.callable.bind(py), x.to_vec(), shape),
                )?
                .extract()
        })
        .map_err(callback_error)?;
        if g.len() != x.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "a nonlinear constraint returned a gradient of {} values for {} inputs",
                g.len(),
                x.len()
            )));
        }
        Ok((v, g))
    }
}

/// An initial-condition generator calling `generate(q, num_restarts,
/// raw_samples)`, which returns `(data, shape)` of a `num_restarts x q x d`
/// array (the Python layer calls the user's `ic_generator` with BoTorch's
/// arguments).
pub fn ic_generator(generate: Py<PyAny>) -> IcGenerator {
    let generate = Arc::new(generate);
    IcGenerator::new(move |req: &IcRequest<'_>| {
        let d = req.bounds.ncols();
        let Array(data, shape) = Python::attach(|py| -> PyResult<Array> {
            generate
                .bind(py)
                .call1((req.q, req.num_restarts, req.raw_samples))?
                .extract()
        })
        .map_err(callback_error)?;
        let (b, q) = match shape.as_slice() {
            [b, q, dd] if *dd == d => (*b, *q),
            [q, dd] if *dd == d => (1, *q),
            s => {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the ic_generator returned an array of shape {s:?}, expected \
                     num_restarts x q x {d}"
                )))
            }
        };
        if data.len() != b * q * d {
            return Err(BotorchError::DimensionMismatch(
                "the ic_generator returned an inconsistent array".into(),
            ));
        }
        Ok((0..b)
            .map(|i| Matrix::from_row_slice(q, d, &data[i * q * d..(i + 1) * q * d]))
            .collect())
    })
}
