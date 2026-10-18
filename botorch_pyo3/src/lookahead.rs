//! The multi-step look-ahead acquisition function
//! (`botorch.acquisition.multi_step_lookahead.qMultiStepLookahead`), and the
//! vector-Jacobian products of posterior samples with respect to their
//! inputs that Python subclasses of `SampleReducingMCAcquisitionFunction`
//! differentiate through.
//!
//! Stage value functions arrive as `None` or `(name, kwargs, best_f_from_model,
//! sampler)`: the BoTorch class name of a native value function with its
//! arguments (see `so_acquisition::value_function_of`), whether `best_f` is
//! the best value of every fantasy model (BoTorch's `make_best_f`), and the
//! inner sampler of an MC value function.

use std::collections::HashMap;
use std::sync::Arc;

use botorch_core::acquisition::multi_step_lookahead::{
    QMultiStepLookahead, QMultiStepLookaheadOptions, StageValueFunction,
};
use botorch_core::acquisition::AcquisitionFunction;
use botorch_core::Vector;
use pyo3::prelude::*;

use crate::acquisition::{model_of, objective_of, sampler_of, PyAcquisitionFunction};
use crate::arrays::{Array, ArrayBuf};
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::models::PyModel;
use crate::sampling::PySobolQMCNormalSampler;
use crate::so_acquisition::value_function_of;
use crate::threads;

/// A stage value function: `(name, kwargs, best_f_from_model, sampler)`.
type StageArg<'py> = Option<(
    String,
    HashMap<String, Py<PyAny>>,
    bool,
    Option<PyRef<'py, PySobolQMCNormalSampler>>,
)>;

/// The state of a `qMultiStepLookahead` (its tree layout).
#[pyclass(
    frozen,
    name = "MultiStepLookaheadState",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyMultiStepLookahead {
    inner: Arc<QMultiStepLookahead>,
}

#[pymethods]
impl PyMultiStepLookahead {
    /// The number of auxiliary points of a candidate set (BoTorch's
    /// `_num_auxiliary`).
    fn num_auxiliary(&self) -> usize {
        self.inner.num_auxiliary()
    }

    /// The numbers of fantasies of the steps.
    fn num_fantasies(&self) -> Vec<usize> {
        self.inner.num_fantasies().to_vec()
    }

    /// The seeds of the samplers of the MC stage value functions (`None` for
    /// the others).
    fn inner_sampler_seeds(&self) -> Vec<Option<u64>> {
        self.inner
            .inner_samplers()
            .iter()
            .map(|s| s.as_ref().map(|s| s.seed()))
            .collect()
    }
}

/// `qMultiStepLookahead`; returns the acquisition function and its state.
#[pyfunction]
#[pyo3(signature = (
    model,
    batch_sizes,
    num_fantasies=None,
    samplers=None,
    sample_weights=None,
    value_functions=None,
    objective=None,
    transform=None,
    x_pending=None,
    collapse_fantasy_base_samples=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_multi_step_lookahead(
    model: PyRef<'_, PyModel>,
    batch_sizes: Vec<usize>,
    num_fantasies: Option<Vec<usize>>,
    samplers: Option<Vec<PyRef<'_, PySobolQMCNormalSampler>>>,
    sample_weights: Option<Vec<Option<Vec<f64>>>>,
    value_functions: Option<Vec<StageArg<'_>>>,
    objective: Option<Bound<'_, PyAny>>,
    transform: Option<(Vec<f64>, f64)>,
    x_pending: Option<Array>,
    collapse_fantasy_base_samples: bool,
) -> PyResult<(PyAcquisitionFunction, PyMultiStepLookahead)> {
    let value_functions = match value_functions {
        None => None,
        Some(v) => Some(
            v.into_iter()
                .map(|s| -> PyResult<Option<StageValueFunction>> {
                    let Some((name, kwargs, from_model, sampler)) = s else {
                        return Ok(None);
                    };
                    let kind = value_function_of(Some((name, kwargs)))?
                        .ok_or_else(|| value_error("missing value function"))?;
                    Ok(Some(StageValueFunction {
                        kind,
                        best_f_from_model: from_model,
                        sampler: sampler_of(sampler),
                    }))
                })
                .collect::<PyResult<Vec<_>>>()?,
        ),
    };
    let transform = transform
        .map(|(w, offset)| {
            botorch_core::acquisition::ScalarizedPosteriorTransform::new(
                Vector::from_vec(w),
                offset,
            )
        })
        .transpose()
        .into_py_result()?;
    let options = QMultiStepLookaheadOptions {
        batch_sizes,
        num_fantasies,
        samplers: samplers.map(|s| s.into_iter().map(|s| Arc::clone(&s.inner)).collect()),
        sample_weights: sample_weights.unwrap_or_default(),
        value_functions,
        objective: objective_of(objective.as_ref())?,
        posterior_transform: transform,
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        collapse_fantasy_base_samples,
    };
    let a = Arc::new(QMultiStepLookahead::new(model_of(&model), options).into_py_result()?);
    Ok((
        PyAcquisitionFunction::new_plain(
            a.clone() as Arc<dyn AcquisitionFunction>,
            "qMultiStepLookahead",
        ),
        PyMultiStepLookahead { inner: a },
    ))
}

/// The gradient with respect to the inputs `x` (`b x n x d`) of
/// `Σ samples_bar ⊙ samples` for the posterior samples of every t-batch of
/// `x` from the base samples `base` (`S x n·m`, column `p·m + t`, shared by
/// the t-batches; BoTorch's `rsample_from_base_samples`), with
/// `samples_bar` of shape `S x b x n x m` (the layout of
/// `Model.posterior_samples`).
#[pyfunction]
pub fn posterior_samples_vjp(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    x: Array,
    base: Array,
    samples_bar: Array,
) -> PyResult<ArrayBuf> {
    let m = model_of(&model);
    let (nb, n_in, d) = match x.1.as_slice() {
        &[b, n, d] => (b, n, d),
        s => {
            return Err(dimension_error(format!(
                "Expected X of shape b x n x d, got {s:?}."
            )))
        }
    };
    let (s, k) = match base.1.as_slice() {
        &[s, k] => (s, k),
        sh => {
            return Err(dimension_error(format!(
                "Expected base samples of shape S x n·m, got {sh:?}."
            )))
        }
    };
    if samples_bar.1.iter().product::<usize>() != s * nb * k || samples_bar.1.first() != Some(&s) {
        return Err(dimension_error(format!(
            "Expected samples_bar of shape {s} x {nb} x n x m, got {:?}.",
            samples_bar.1
        )));
    }
    let base_m = botorch_core::Matrix::from_row_slice(s, k, &base.0);
    let grads = threads::run(py, || {
        use rayon::prelude::*;
        (0..nb)
            .into_par_iter()
            .map(|b| {
                let xb = botorch_core::Matrix::from_row_slice(
                    n_in,
                    d,
                    &x.0[b * n_in * d..(b + 1) * n_in * d],
                );
                let w =
                    botorch_core::Matrix::from_fn(s, k, |i, c| samples_bar.0[(i * nb + b) * k + c]);
                botorch_core::acquisition::monte_carlo::posterior_samples_with_vjp(
                    m.as_ref(),
                    &xb,
                    &base_m,
                    &w,
                )
                .map(|(_, g)| g)
            })
            .collect::<botorch_core::Result<Vec<_>>>()
    })
    .into_py_result()?;
    let mut data = Vec::with_capacity(nb * n_in * d);
    for g in &grads {
        for i in 0..n_in {
            for j in 0..d {
                data.push(g[(i, j)]);
            }
        }
    }
    Ok(ArrayBuf(data, vec![nb, n_in, d]))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiStepLookahead>()?;
    m.add_function(wrap_pyfunction!(q_multi_step_lookahead, m)?)?;
    m.add_function(wrap_pyfunction!(posterior_samples_vjp, m)?)?;
    Ok(())
}
