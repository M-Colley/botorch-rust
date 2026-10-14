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
///
/// Lock order: the GIL is never acquired while this lock is held. Callables
/// run on worker threads while other workers hold the GIL (a callable that
/// starts a nested botorch_rs computation clears or reads this slot with the
/// GIL held), so waiting for the GIL under this lock deadlocks.
static PENDING: Mutex<Option<PyErr>> = Mutex::new(None);

/// Forgets a pending exception (called before every outermost computation,
/// see [`crate::threads::run`]).
pub fn clear_error() {
    let stale = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
    drop(stale);
}

/// The pending Python exception, if a callable raised one.
pub fn take_error() -> Option<PyErr> {
    PENDING.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Records a Python exception and returns the core error that aborts the
/// computation.
pub(crate) fn callback_error(e: PyErr) -> BotorchError {
    let msg = Python::attach(|py| e.value(py).to_string());
    let unused = {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if pending.is_none() {
            *pending = Some(e);
            None
        } else {
            Some(e)
        }
    };
    drop(unused);
    BotorchError::Runtime(format!("a Python callable raised an exception: {msg}"))
}

const BRIDGE: &str = "botorch_rs._torch_bridge";

/// Row-major data of a matrix.
pub(crate) fn rows(m: &Matrix) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len());
    for i in 0..m.nrows() {
        out.extend(m.row(i).iter());
    }
    out
}

/// Values of shape `(S, 1, n')` (row-major) as an `n' × S` matrix; `n'` is
/// the number of sample points `n`, or fewer for an objective that reduces
/// the `n_w` points per input of a one-to-many input transform (a risk
/// measure; the acquisition function checks `n'` against `X`).
fn values_matrix(values: &[f64], s: usize, n: usize) -> Result<Matrix> {
    if s == 0 || !values.len().is_multiple_of(s) || values.len() / s > n {
        return Err(BotorchError::Runtime(format!(
            "The q-batch shape of the objective values does not agree with the q-batch shape \
             of X: expected {s} x {n} values, got {}.",
            values.len()
        )));
    }
    let n = values.len() / s;
    Ok(Matrix::from_fn(n, s, |p, c| values[c * n + p]))
}

/// The input adjoint returned by the bridge as an `x_rows × d` matrix (`X`
/// has fewer rows than the samples have points with a one-to-many input
/// transform: one per input point).
fn input_adjoint(
    xbar: Option<Vec<f64>>,
    x_rows: usize,
    d: Option<usize>,
) -> Result<Option<Matrix>> {
    match (xbar, d) {
        (Some(v), Some(d)) => {
            if v.len() != x_rows * d {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the input gradient has {} values, expected {}",
                    v.len(),
                    x_rows * d
                )));
            }
            Ok(Some(Matrix::from_row_slice(x_rows, d, &v)))
        }
        _ => Ok(None),
    }
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

    /// A preprocessing function of a multi-output risk measure (BoTorch's
    /// `preprocessing_function`), called as `f(samples)`.
    pub fn mo_preprocessing(callable: Py<PyAny>) -> Self {
        Self {
            callable,
            kind: "mo_objective",
            pass_x: false,
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
        let x_arg: Option<(Vec<f64>, usize)> = if self.pass_x {
            x.map(|x| (rows(x), x.ncols()))
        } else {
            None
        };
        let d = x_arg.as_ref().map(|(_, d)| *d);
        let x_rows = x.map_or(0, |x| x.nrows());
        type Forward = (Vec<f64>, usize, usize, Py<PyAny>);
        let (values, nv, mo, handle) = Python::attach(|py| -> PyResult<Forward> {
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
                    self.pass_x,
                ),
            )?;
            let (values, nv, mo, handle): (Bound<'_, PyAny>, usize, usize, Py<PyAny>) =
                res.extract()?;
            Ok((float_data(&values)?, nv, mo, handle))
        })
        .map_err(callback_error)?;
        let values = McSamples::from_row_major(&values, s, nv, mo)?;
        let pullback: Option<SamplePullback> = if with_grad {
            Some(Box::new(move |bar: &Matrix| {
                let adjoint = McSamples::new(bar.clone(), nv, mo)?.to_row_major();
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
                Ok((sbar, input_adjoint(xbar, x_rows, d)?))
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
        let x_rows = x.map_or(0, |x| x.nrows());
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
                Ok((sbar, input_adjoint(xbar, x_rows, d)?))
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

/// A Python callable of torch tensors evaluated through the bridge
/// (`call_tensor_function` / `tensor_function_vjp`), e.g. the `project` of a
/// multi-fidelity acquisition function, the cost of a
/// `GenericCostAwareUtility` or a cost model given as a callable.
#[derive(Debug)]
pub struct PyTensorFunction {
    callable: Py<PyAny>,
}

/// The outputs of a [`PyTensorFunction`]: the values (row-major) with their
/// shape and, with gradients, the handle of the autograd graph.
pub(crate) type TensorOutput = (Vec<f64>, Vec<usize>, Option<Py<PyAny>>);

impl PyTensorFunction {
    /// The callable `f`.
    pub fn new(callable: Py<PyAny>) -> Self {
        Self { callable }
    }

    /// The callable, bound to `py`.
    pub(crate) fn callable<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.callable.bind(py)
    }

    /// `f(*args)` for arguments given as `(row-major data, shape)`.
    pub(crate) fn call(
        &self,
        args: &[(Vec<f64>, Vec<usize>)],
        with_grad: bool,
    ) -> Result<TensorOutput> {
        Python::attach(|py| -> PyResult<TensorOutput> {
            let bridge = py.import(BRIDGE)?;
            let py_args = args
                .iter()
                .map(|(data, shape)| Ok((float_values(py, data)?, shape.clone())))
                .collect::<PyResult<Vec<_>>>()?;
            let res = bridge.call_method1(
                "call_tensor_function",
                (self.callable.bind(py), py_args, with_grad),
            )?;
            let (values, shape, handle): (Bound<'_, PyAny>, Vec<usize>, Option<Py<PyAny>>) =
                res.extract()?;
            Ok((float_data(&values)?, shape, handle))
        })
        .map_err(callback_error)
    }

    /// The adjoints of the arguments for the adjoint of the outputs.
    fn vjp(handle: &Py<PyAny>, adjoint: &[f64]) -> Result<Vec<Vec<f64>>> {
        Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
            let bridge = py.import(BRIDGE)?;
            let res = bridge.call_method1(
                "tensor_function_vjp",
                (handle.bind(py), float_values(py, adjoint)?),
            )?;
            let grads: Vec<Bound<'_, PyAny>> = res.extract()?;
            grads.iter().map(float_data).collect()
        })
        .map_err(callback_error)
    }
}

/// A projection `project(X)` of the points of a candidate set (BoTorch's
/// `project` argument), called with `1 x q x d` tensors.
#[derive(Debug)]
pub struct PyProjection(pub PyTensorFunction);

impl botorch_core::acquisition::fidelity::InputProjection for PyProjection {
    fn project(&self, x: &Matrix) -> Result<Matrix> {
        let d = x.ncols();
        let (values, shape, _) = self.0.call(&[(rows(x), vec![1, x.nrows(), d])], false)?;
        let cols = *shape.last().unwrap_or(&0);
        if cols != d || values.len() % d.max(1) != 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "the projection returned a tensor of shape {shape:?}, expected `1 x q' x {d}`"
            )));
        }
        Ok(Matrix::from_row_slice(values.len() / d, d, &values))
    }

    fn backward(&self, x: &Matrix, out_bar: &Matrix) -> Result<Matrix> {
        let d = x.ncols();
        let (_, _, handle) = self.0.call(&[(rows(x), vec![1, x.nrows(), d])], true)?;
        let handle = handle.expect("gradient requested");
        let grads = PyTensorFunction::vjp(&handle, &rows(out_bar))?;
        Ok(Matrix::from_row_slice(x.nrows(), d, &grads[0]))
    }
}

/// BoTorch's `GenericCostAwareUtility(cost)`: `cost(X, deltas)` with `X` of
/// shape `1 x q x d` and `deltas` of shape `n_f x 1`, returning `n_f x 1`
/// utilities.
pub fn generic_cost_utility(
    callable: Py<PyAny>,
) -> botorch_core::acquisition::cost_aware::GenericCostAwareUtility {
    use botorch_core::acquisition::cost_aware::{GenericCostAwareUtility, UtilityValues};
    let f = PyTensorFunction::new(callable);
    GenericCostAwareUtility::new(move |x: &Matrix, deltas: &[f64], with_grad: bool| {
        let (q, d) = x.shape();
        let n_f = deltas.len();
        let (values, _, handle) = f.call(
            &[(rows(x), vec![1, q, d]), (deltas.to_vec(), vec![n_f, 1])],
            with_grad,
        )?;
        if values.len() != n_f {
            return Err(BotorchError::DimensionMismatch(format!(
                "the cost-aware utility returned {} values for {n_f} improvements",
                values.len()
            )));
        }
        let pullback = handle.map(|h| {
            Box::new(move |vbar: &[f64]| {
                let grads = PyTensorFunction::vjp(&h, vbar)?;
                Ok((grads[1].clone(), Matrix::from_row_slice(q, d, &grads[0])))
            }) as botorch_core::acquisition::cost_aware::UtilityPullback
        });
        Ok(UtilityValues { values, pullback })
    })
}

/// A cost model given as a callable of `1 x q x d` tensors returning
/// `1 x q x m` costs (e.g. BoTorch's `GenericDeterministicModel`).
#[derive(Debug)]
pub struct PyCostModel {
    f: PyTensorFunction,
    num_outputs: usize,
}

impl PyCostModel {
    /// The cost model `callable` with `num_outputs` outputs.
    pub fn new(callable: Py<PyAny>, num_outputs: usize) -> Self {
        Self {
            f: PyTensorFunction::new(callable),
            num_outputs,
        }
    }
}

impl botorch_core::models::cost::CostModel for PyCostModel {
    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn forward(&self, x: &Matrix) -> Result<Matrix> {
        let (q, d) = x.shape();
        let (values, _, _) = self.f.call(&[(rows(x), vec![1, q, d])], false)?;
        if values.len() != q * self.num_outputs {
            return Err(BotorchError::DimensionMismatch(format!(
                "the cost model returned {} values, expected {q} x {}",
                values.len(),
                self.num_outputs
            )));
        }
        Ok(Matrix::from_row_slice(q, self.num_outputs, &values))
    }

    fn backward(&self, x: &Matrix, cost_bar: &Matrix) -> Result<Matrix> {
        let (q, d) = x.shape();
        let (_, _, handle) = self.f.call(&[(rows(x), vec![1, q, d])], true)?;
        let grads = PyTensorFunction::vjp(&handle.expect("gradient requested"), &rows(cost_bar))?;
        Ok(Matrix::from_row_slice(q, d, &grads[0]))
    }
}
