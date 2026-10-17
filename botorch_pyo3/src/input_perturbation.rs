//! `InputPerturbation` and the steps of a `ChainedInputTransform`
//! (`botorch.models.transforms.input`), the one-to-many input transforms of
//! robust and risk-averse Bayesian optimization.
//!
//! An [`PyInputPerturbation`] holds `botorch_core`'s
//! [`InputPerturbation`]: a fixed `n_w x k` perturbation set, or a Python
//! callable on torch tensors (BoTorch's callable `perturbation_set` for
//! heteroscedastic input noise), evaluated through
//! `botorch_rs._robust_bridge` with its vector-Jacobian products from
//! `torch.autograd`. A model receives its transform as a list of steps
//! (`Normalize` and `InputPerturbation` objects, [`chain_of`]) and reports
//! the fitted steps back ([`chain_state`]).

use std::sync::Arc;

use botorch_core::transforms::input::{
    ChainedInputTransform, InputPerturbation, InputTransformStep, PerturbationFunction,
    PerturbationSet,
};
use botorch_core::{BatchedInputTransform, BotorchError, Matrix};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::arrays::{float_data, float_values, matrix_out, Array, ArrayBuf, ArrayOut};
use crate::callbacks::callback_error;
use crate::errors::{unsupported, value_error, IntoPyResult};
use crate::threads;
use crate::transforms::PyNormalize;

/// The Python module that evaluates perturbation functions and cost
/// callables on torch tensors.
pub(crate) const ROBUST_BRIDGE: &str = "botorch_rs._robust_bridge";

/// Row-major data of a matrix.
pub(crate) fn row_major(m: &Matrix) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len());
    for i in 0..m.nrows() {
        out.extend(m.row(i).iter());
    }
    out
}

/// A perturbation function given by a Python callable `f(X)` on torch
/// tensors (`n x k` points to `n x n_w x k` perturbations).
#[derive(Debug)]
struct PyPerturbationFunction {
    callable: Py<PyAny>,
}

impl PerturbationFunction for PyPerturbationFunction {
    fn perturbations(&self, x: &Matrix) -> botorch_core::Result<Matrix> {
        let (n, k) = x.shape();
        let data = row_major(x);
        let values = Python::attach(|py| -> PyResult<Vec<f64>> {
            let bridge = py.import(ROBUST_BRIDGE)?;
            let res = bridge.call_method1(
                "perturbations",
                (self.callable.bind(py), float_values(py, &data)?, n, k),
            )?;
            float_data(&res)
        })
        .map_err(callback_error)?;
        if k == 0 || values.len() % k != 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "the perturbation function returned {} values for points with {k} perturbed \
                 inputs",
                values.len()
            )));
        }
        Ok(Matrix::from_row_slice(values.len() / k, k, &values))
    }

    fn vjp(&self, x: &Matrix, adjoint: &Matrix) -> botorch_core::Result<Matrix> {
        let (n, k) = x.shape();
        let data = row_major(x);
        let bar = row_major(adjoint);
        let values = Python::attach(|py| -> PyResult<Vec<f64>> {
            let bridge = py.import(ROBUST_BRIDGE)?;
            let res = bridge.call_method1(
                "perturbation_vjp",
                (
                    self.callable.bind(py),
                    float_values(py, &data)?,
                    n,
                    k,
                    float_values(py, &bar)?,
                ),
            )?;
            float_data(&res)
        })
        .map_err(callback_error)?;
        if values.len() != n * k {
            return Err(BotorchError::DimensionMismatch(format!(
                "the gradient of the perturbation function has {} values, expected {}",
                values.len(),
                n * k
            )));
        }
        Ok(Matrix::from_row_slice(n, k, &values))
    }
}

/// BoTorch's `InputPerturbation` (immutable; see the module documentation).
#[pyclass(frozen, name = "InputPerturbation", module = "botorch_rs._botorch_rs")]
pub struct PyInputPerturbation {
    /// The transform.
    pub inner: InputPerturbation,
}

#[pymethods]
impl PyInputPerturbation {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// `InputPerturbation(perturbation_set, bounds, indices, multiplicative,
    /// transform_on_train, transform_on_eval, transform_on_fantasize)` with a
    /// fixed `n_w x k` `perturbation_set` or a `perturbation_function`
    /// (exactly one of the two). `indices` must be non-negative (the Python
    /// layer resolves negative indices once the input dimension is known).
    #[new]
    #[pyo3(signature = (
        perturbation_set=None,
        perturbation_function=None,
        bounds=None,
        indices=None,
        multiplicative=false,
        transform_on_train=false,
        transform_on_eval=true,
        transform_on_fantasize=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        perturbation_set: Option<Array>,
        perturbation_function: Option<Py<PyAny>>,
        bounds: Option<Array>,
        indices: Option<Vec<usize>>,
        multiplicative: bool,
        transform_on_train: bool,
        transform_on_eval: bool,
        transform_on_fantasize: bool,
    ) -> PyResult<Self> {
        let set =
            match (perturbation_set, perturbation_function) {
                (Some(p), None) => {
                    if p.1.len() != 2 {
                        return Err(value_error(
                            "`perturbation_set` must be an `n_p x d`-dim tensor!",
                        ));
                    }
                    PerturbationSet::Fixed(p.matrix("perturbation_set")?)
                }
                (None, Some(f)) => {
                    PerturbationSet::Function(Arc::new(PyPerturbationFunction { callable: f }))
                }
                _ => return Err(value_error(
                    "exactly one of `perturbation_set` and `perturbation_function` must be given",
                )),
            };
        let mut inner = InputPerturbation::new(set).into_py_result()?;
        if let Some(b) = bounds {
            inner = inner.with_bounds(b.matrix("bounds")?).into_py_result()?;
        }
        if let Some(idx) = indices {
            inner = inner.with_indices(idx).into_py_result()?;
        }
        inner = inner
            .with_multiplicative(multiplicative)
            .with_transform_flags(
                transform_on_train,
                transform_on_eval,
                transform_on_fantasize,
            );
        Ok(Self { inner })
    }

    /// The number of perturbations `n_w` (`None` for a function that has not
    /// been evaluated by a model yet).
    #[getter]
    fn n_w(&self) -> Option<usize> {
        self.inner.n_w()
    }

    /// The fixed perturbation set (`n_w x k`), `None` for a function.
    #[getter]
    fn perturbation_set(&self) -> Option<ArrayOut> {
        match self.inner.perturbation_set() {
            PerturbationSet::Fixed(p) => Some(matrix_out(p)),
            PerturbationSet::Function(_) => None,
        }
    }

    /// The clamping bounds (`2 x k`), if any.
    #[getter]
    fn bounds(&self) -> Option<ArrayOut> {
        self.inner.bounds().map(matrix_out)
    }

    /// The perturbed inputs (`None`: all).
    #[getter]
    fn indices(&self) -> Option<Vec<usize>> {
        self.inner.indices().map(<[usize]>::to_vec)
    }

    /// Whether the perturbations multiply the inputs.
    #[getter]
    fn multiplicative(&self) -> bool {
        self.inner.multiplicative()
    }

    /// BoTorch's `transform_on_train`.
    #[getter]
    fn transform_on_train(&self) -> bool {
        self.inner.transform_on_train()
    }

    /// BoTorch's `transform_on_eval`.
    #[getter]
    fn transform_on_eval(&self) -> bool {
        self.inner.transform_on_eval()
    }

    /// BoTorch's `transform_on_fantasize`.
    #[getter]
    fn transform_on_fantasize(&self) -> bool {
        self.inner.transform_on_fantasize()
    }

    /// BoTorch's `InputPerturbation.transform`: the perturbed points of `X`
    /// (`batch_shape x q x d`), `batch_shape x (q * n_w) x d` with the
    /// perturbations of `X[..., i, :]` in rows `i * n_w .. (i + 1) * n_w`
    /// (applied regardless of the train/eval flags, as in BoTorch).
    fn transform(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        let (batch, mats) = x.batches("X")?;
        let d = x.1[x.1.len() - 1];
        let step = self.inner.clone().with_transform_flags(
            self.inner.transform_on_train(),
            true,
            self.inner.transform_on_fantasize(),
        );
        let chain = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(step)])
            .into_py_result()?;
        let outs = threads::run(py, || {
            mats.iter()
                .map(|m| chain.transform(m))
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let rows = outs.first().map_or(0, Matrix::nrows);
        let mut data = Vec::with_capacity(outs.len() * rows * d);
        for o in &outs {
            data.extend(row_major(o));
        }
        let mut shape = batch;
        shape.extend([rows, d]);
        Ok(ArrayBuf(data, shape))
    }
}

/// The `ChainedInputTransform` of the steps received from Python
/// (`Normalize`, `InputPerturbation` and `InputStep` objects, in order),
/// with the modes of the steps when the model is built (`training`) and the
/// parameter-name prefixes of the steps (`prefixes`), if given.
///
/// # Errors
/// `UnsupportedError` for other objects and for a `Normalize` with a batch
/// shape; errors of the chain.
pub fn chain_of(
    steps: &[Bound<'_, PyAny>],
    training: Option<Vec<bool>>,
    prefixes: Option<Vec<String>>,
) -> PyResult<ChainedInputTransform> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        if let Ok(p) = step.cast::<PyInputPerturbation>() {
            out.push(InputTransformStep::Perturbation(p.get().inner.clone()));
        } else if let Ok(s) = step.cast::<crate::transform_steps::PyInputStep>() {
            out.push(InputTransformStep::Step(s.get().inner.clone()));
        } else if let Ok(n) = step.cast::<PyNormalize>() {
            let n = n.get();
            if !n.batch_shape.is_empty() {
                return Err(unsupported(
                    "A Normalize with a batch shape is not supported in a chained or one-to-many                      input transform; use Normalize(d) (bounds shared by all batch elements).",
                ));
            }
            match n.for_model(&[])? {
                BatchedInputTransform::Shared(t) => out.push(InputTransformStep::Normalize(t)),
                BatchedInputTransform::PerBatch(mut ts) => {
                    out.push(InputTransformStep::Normalize(ts.remove(0)))
                }
            }
        } else {
            return Err(unsupported(format!(
                "Unsupported input transform step {}.",
                step.get_type().name()?
            )));
        }
    }
    let mut chain = ChainedInputTransform::new(out).into_py_result()?;
    if let Some(t) = training {
        chain = chain.with_training(t).into_py_result()?;
    }
    if let Some(p) = prefixes {
        chain = chain.with_prefixes(p).into_py_result()?;
    }
    Ok(chain)
}

/// The fitted steps of a model's chain for the Python layer: a `Normalize`
/// (with its fitted bounds) per normalization step, `n_w` per perturbation
/// and the fitted `InputStep` of every other step.
///
/// # Errors
/// Conversion errors.
pub fn chain_state(py: Python<'_>, chain: &ChainedInputTransform) -> PyResult<Vec<Py<PyAny>>> {
    chain
        .steps()
        .iter()
        .map(|s| match s {
            InputTransformStep::Normalize(n) => PyNormalize::fitted(
                &BatchedInputTransform::Shared(n.clone()),
                n.learn_bounds(),
                &[],
            )
            .into_py_any(py),
            InputTransformStep::Perturbation(p) => p.n_w().into_py_any(py),
            InputTransformStep::Step(s) => {
                crate::transform_steps::PyInputStep { inner: s.clone() }.into_py_any(py)
            }
        })
        .collect()
}

/// Registers the classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInputPerturbation>()?;
    Ok(())
}
