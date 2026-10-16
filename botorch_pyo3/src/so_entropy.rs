//! Single-objective information-theoretic and sample-path acquisition
//! functions of phase 8: the max-value entropy search family
//! (`botorch.acquisition.max_value_entropy_search`), `qJointEntropySearch`,
//! `get_optimal_samples`, `PathwiseThompsonSampling` and
//! `qNegIntegratedPosteriorVariance`.
//!
//! The acquisition functions are returned as [`PyAcquisitionFunction`]s
//! (values and exact gradients, `optimize_acqf`), with a state handle where
//! the Python layer needs one (the max-value samples, the redraws of the
//! Thompson paths).

use std::sync::Arc;

use botorch_core::acquisition::max_value_entropy_search::{
    sample_max_value_gumbel, sample_max_value_thompson, MaxValueEntropyOptions,
    MultiFidelityMaxValueEntropyOptions, QLowerBoundMaxValueEntropy, QMaxValueEntropy,
    QMultiFidelityLowerBoundMaxValueEntropy, QMultiFidelityMaxValueEntropy,
};
use botorch_core::acquisition::{
    get_optimal_samples, AcquisitionFunction, JesEstimationType, OptimalSamplesOptions,
    PathwiseThompsonSampling, PathwiseThompsonSamplingOptions, QJointEntropySearch,
    QJointEntropySearchOptions, QNegIntegratedPosteriorVariance, SampleTransform, ScalarizedModel,
    ScalarizedPosteriorTransform,
};
use botorch_core::sampling::pathwise::DEFAULT_NUM_FEATURES;
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

use crate::acquisition::{model_of, objective_of, sampler_of, PyAcquisitionFunction};
use crate::arrays::{matrix_out, Array, ArrayOut};
use crate::errors::{unsupported, value_error, IntoPyResult};
use crate::hvkg::{projection_of, PyCostAwareUtility};
use crate::models::PyModel;
use crate::pathwise::{draws_of, DrawsArg};
use crate::sampling::PySobolQMCNormalSampler;
use crate::so_acquisition::expansion_of;
use crate::threads;

/// A posterior transform from `(weights, offset)`.
fn transform_of(t: Option<(Vec<f64>, f64)>) -> PyResult<Option<ScalarizedPosteriorTransform>> {
    t.map(|(w, offset)| ScalarizedPosteriorTransform::new(Vector::from_vec(w), offset))
        .transpose()
        .into_py_result()
}

/// The max-value entropy search behind a Python handle.
enum MaxValue {
    Mes(Arc<QMaxValueEntropy>),
    MfMes(Arc<QMultiFidelityMaxValueEntropy>),
    Gibbon(Arc<QLowerBoundMaxValueEntropy>),
    MfGibbon(Arc<QMultiFidelityLowerBoundMaxValueEntropy>),
}

/// The state of a max-value entropy search: its max-value samples and
/// candidate set.
#[pyclass(
    frozen,
    name = "MaxValueEntropyState",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyMaxValueState {
    inner: MaxValue,
}

#[pymethods]
impl PyMaxValueState {
    /// The max-value samples (BoTorch's `posterior_max_values`).
    fn posterior_max_values(&self) -> Vec<f64> {
        match &self.inner {
            MaxValue::Mes(a) => a.posterior_max_values(),
            MaxValue::MfMes(a) => a.as_mes().posterior_max_values(),
            MaxValue::Gibbon(a) => a.posterior_max_values(),
            MaxValue::MfGibbon(a) => a.as_gibbon().posterior_max_values(),
        }
    }

    /// Replaces the max-value samples.
    fn set_posterior_max_values(&self, values: Vec<f64>) -> PyResult<()> {
        match &self.inner {
            MaxValue::Mes(a) => a.set_posterior_max_values(values),
            MaxValue::MfMes(a) => a.as_mes().set_posterior_max_values(values),
            MaxValue::Gibbon(a) => a.set_posterior_max_values(values),
            MaxValue::MfGibbon(a) => a.as_gibbon().set_posterior_max_values(values),
        }
        .into_py_result()
    }

    /// The candidate set (with the training inputs; projected for the
    /// multi-fidelity variants).
    fn candidate_set(&self) -> ArrayOut {
        matrix_out(match &self.inner {
            MaxValue::Mes(a) => a.candidate_set(),
            MaxValue::MfMes(a) => a.as_mes().candidate_set(),
            MaxValue::Gibbon(a) => a.candidate_set(),
            MaxValue::MfGibbon(a) => a.as_gibbon().candidate_set(),
        })
    }
}

/// The max-value entropy search `kind` (`"qMaxValueEntropy"`,
/// `"qLowerBoundMaxValueEntropy"`, `"qMultiFidelityMaxValueEntropy"`,
/// `"qMultiFidelityLowerBoundMaxValueEntropy"`) with BoTorch's arguments;
/// `seed` seeds the random numbers of the max-value samples, `train_inputs`
/// defaults to the model's raw training inputs.
#[pyfunction]
#[pyo3(signature = (
    model,
    candidate_set,
    kind,
    num_fantasies=16,
    num_mv_samples=10,
    num_y_samples=128,
    transform=None,
    use_gumbel=true,
    maximize=true,
    x_pending=None,
    train_inputs=None,
    sampler=None,
    fantasies_sampler=None,
    seed=None,
    cost_aware_utility=None,
    project=None,
    expand=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_max_value_entropy(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    candidate_set: Array,
    kind: &str,
    num_fantasies: usize,
    num_mv_samples: usize,
    num_y_samples: usize,
    transform: Option<(Vec<f64>, f64)>,
    use_gumbel: bool,
    maximize: bool,
    x_pending: Option<Array>,
    train_inputs: Option<Array>,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    fantasies_sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    seed: Option<u64>,
    cost_aware_utility: Option<PyRef<'_, PyCostAwareUtility>>,
    project: Option<Bound<'_, PyAny>>,
    expand: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyAcquisitionFunction, PyMaxValueState)> {
    let m = model_of(&model);
    let d = m.dim();
    let cs = candidate_set.matrix("candidate_set")?;
    let mes = MaxValueEntropyOptions {
        num_mv_samples,
        posterior_transform: transform_of(transform)?,
        use_gumbel,
        maximize,
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        train_inputs: train_inputs.map(|t| t.matrix("train_inputs")).transpose()?,
        num_fantasies,
        num_y_samples,
        sampler: sampler_of(sampler),
        fantasies_sampler: sampler_of(fantasies_sampler),
        seed,
    };
    let mf = |mes: MaxValueEntropyOptions| -> PyResult<MultiFidelityMaxValueEntropyOptions> {
        Ok(MultiFidelityMaxValueEntropyOptions {
            mes,
            cost_aware_utility: cost_aware_utility.as_ref().map(|u| Arc::clone(&u.inner)),
            project: projection_of(project.as_ref(), d)?,
            expand: expansion_of(expand.as_ref())?,
        })
    };
    let (acqf, state): (Arc<dyn AcquisitionFunction>, MaxValue) = match kind {
        "qMaxValueEntropy" => {
            let a =
                Arc::new(threads::run(py, || QMaxValueEntropy::new(m, &cs, mes)).into_py_result()?);
            (a.clone(), MaxValue::Mes(a))
        }
        "qLowerBoundMaxValueEntropy" => {
            let a = Arc::new(
                threads::run(py, || QLowerBoundMaxValueEntropy::new(m, &cs, mes))
                    .into_py_result()?,
            );
            (a.clone(), MaxValue::Gibbon(a))
        }
        "qMultiFidelityMaxValueEntropy" => {
            let o = mf(mes)?;
            let a = Arc::new(
                threads::run(py, || QMultiFidelityMaxValueEntropy::new(m, &cs, o))
                    .into_py_result()?,
            );
            (a.clone(), MaxValue::MfMes(a))
        }
        "qMultiFidelityLowerBoundMaxValueEntropy" => {
            let o = mf(mes)?;
            let a = Arc::new(
                threads::run(py, || {
                    QMultiFidelityLowerBoundMaxValueEntropy::new(m, &cs, o)
                })
                .into_py_result()?,
            );
            (a.clone(), MaxValue::MfGibbon(a))
        }
        other => {
            return Err(unsupported(format!(
                "Unknown max-value entropy search {other:?}."
            )))
        }
    };
    let name: &'static str = match &state {
        MaxValue::Mes(_) => "qMaxValueEntropy",
        MaxValue::MfMes(_) => "qMultiFidelityMaxValueEntropy",
        MaxValue::Gibbon(_) => "qLowerBoundMaxValueEntropy",
        MaxValue::MfGibbon(_) => "qMultiFidelityLowerBoundMaxValueEntropy",
    };
    Ok((
        PyAcquisitionFunction::new_plain(acqf, name),
        PyMaxValueState { inner: state },
    ))
}

/// BoTorch's `_sample_max_value_Gumbel` for the uniform numbers `uniforms`
/// (BoTorch's `torch.rand(num_samples, 1)`).
#[pyfunction]
#[pyo3(name = "sample_max_value_gumbel", signature = (model, candidate_set, uniforms, maximize=true, transform=None))]
pub fn sample_max_value_gumbel_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    candidate_set: Array,
    uniforms: Vec<f64>,
    maximize: bool,
    transform: Option<(Vec<f64>, f64)>,
) -> PyResult<Vec<f64>> {
    let m = ScalarizedModel::wrap(model_of(&model), transform_of(transform)?).into_py_result()?;
    let cs = candidate_set.matrix("candidate_set")?;
    threads::run(py, || {
        sample_max_value_gumbel(m.as_ref(), &cs, maximize, &uniforms)
    })
    .into_py_result()
}

/// BoTorch's `_sample_max_value_Thompson` for the standard normal
/// `base_samples` (`num_samples x n`).
#[pyfunction]
#[pyo3(name = "sample_max_value_thompson", signature = (model, candidate_set, base_samples, maximize=true, transform=None))]
pub fn sample_max_value_thompson_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    candidate_set: Array,
    base_samples: Array,
    maximize: bool,
    transform: Option<(Vec<f64>, f64)>,
) -> PyResult<Vec<f64>> {
    let m = ScalarizedModel::wrap(model_of(&model), transform_of(transform)?).into_py_result()?;
    let cs = candidate_set.matrix("candidate_set")?;
    let z = base_samples.matrix("base_samples")?;
    threads::run(py, || {
        sample_max_value_thompson(m.as_ref(), &cs, maximize, &z)
    })
    .into_py_result()
}

/// `qJointEntropySearch(model, optimal_inputs, optimal_outputs,
/// condition_noiseless, posterior_transform, X_pending, estimation_type)`
/// with the MC sampler `sampler` and the per-output noise variances of the
/// conditioning `condition_noise` (outcome-transformed space).
#[pyfunction]
#[pyo3(signature = (
    model,
    optimal_inputs,
    optimal_outputs,
    condition_noiseless=true,
    condition_noise=None,
    transform=None,
    x_pending=None,
    estimation_type="LB",
    sampler=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_joint_entropy_search(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    optimal_inputs: Array,
    optimal_outputs: Array,
    condition_noiseless: bool,
    condition_noise: Option<Vec<f64>>,
    transform: Option<(Vec<f64>, f64)>,
    x_pending: Option<Array>,
    estimation_type: &str,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let xs = optimal_inputs.matrix("optimal_inputs")?;
    let ys = optimal_outputs.matrix("optimal_outputs")?;
    let options = QJointEntropySearchOptions {
        condition_noiseless,
        condition_noise,
        posterior_transform: transform_of(transform)?,
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        estimation_type: JesEstimationType::from_name(estimation_type).into_py_result()?,
        sampler: sampler_of(sampler),
    };
    let a = threads::run(py, || QJointEntropySearch::new(m, &xs, &ys, options)).into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "qJointEntropySearch",
    ))
}

/// `get_optimal_samples(model, bounds, num_optima, raw_samples,
/// num_restarts, posterior_transform, objective, return_transformed)`: the
/// optimal inputs (`num_optima x d`) and outputs (`num_optima x m`, or `x 1`
/// with `return_transformed`) of Matheron paths with `num_features`
/// features, from `draws` (one `(frequencies, weights, noise)` per output)
/// and `candidates` (`raw_samples x d`) or from `seed`.
#[pyfunction]
#[pyo3(name = "get_optimal_samples", signature = (
    model,
    bounds,
    num_optima,
    raw_samples=1024,
    num_restarts=20,
    transform=None,
    objective=None,
    return_transformed=false,
    num_features=DEFAULT_NUM_FEATURES,
    seed=None,
    draws=None,
    candidates=None,
    maxiter=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn get_optimal_samples_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    bounds: Array,
    num_optima: usize,
    raw_samples: usize,
    num_restarts: usize,
    transform: Option<(Vec<f64>, f64)>,
    objective: Option<Bound<'_, PyAny>>,
    return_transformed: bool,
    num_features: usize,
    seed: Option<u64>,
    draws: Option<Vec<DrawsArg>>,
    candidates: Option<Array>,
    maxiter: Option<usize>,
) -> PyResult<(ArrayOut, ArrayOut)> {
    let m = model_of(&model);
    let bounds = bounds.matrix("bounds")?;
    let transform = match (transform_of(transform)?, objective_of(objective.as_ref())?) {
        (Some(_), Some(_)) => {
            return Err(value_error(
                "Only one of `posterior_transform` and `objective` can be specified.",
            ))
        }
        (Some(t), None) => SampleTransform::Scalarize(t),
        (None, Some(o)) => SampleTransform::Objective(o),
        (None, None) => SampleTransform::Identity,
    };
    let mut options = OptimalSamplesOptions {
        raw_samples,
        num_restarts,
        transform,
        return_transformed,
        num_features,
        seed,
        draws: draws.map(draws_of).transpose()?,
        candidates: candidates.map(|c| c.matrix("candidates")).transpose()?,
        ..Default::default()
    };
    if let Some(it) = maxiter {
        options.lbfgsb.maxiter = it;
    }
    let (x, f) = threads::run(py, || {
        get_optimal_samples(m.as_ref(), &bounds, num_optima, &options)
    })
    .into_py_result()?;
    Ok((matrix_out(&x), matrix_out(&f)))
}

/// The state of a `PathwiseThompsonSampling` (its paths).
#[pyclass(
    frozen,
    name = "ThompsonSamplingState",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyThompsonState {
    inner: Arc<PathwiseThompsonSampling>,
}

#[pymethods]
impl PyThompsonState {
    /// BoTorch's `redraw(batch_size)`, from explicit `draws` (one
    /// `(frequencies, weights, noise)` per output with `batch_size` samples)
    /// or the next stream of the seed.
    #[pyo3(signature = (batch_size, draws=None))]
    fn redraw(
        &self,
        py: Python<'_>,
        batch_size: usize,
        draws: Option<Vec<DrawsArg>>,
    ) -> PyResult<()> {
        let draws = draws.map(draws_of).transpose()?;
        let a = Arc::clone(&self.inner);
        threads::run(py, || a.redraw(batch_size, draws)).into_py_result()
    }

    /// The batch size of the drawn paths (`None` before the first draw).
    #[getter]
    fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size()
    }
}

/// `PathwiseThompsonSampling(model, objective, posterior_transform)` with
/// `num_features` Fourier features and the seed of the paths.
#[pyfunction]
#[pyo3(signature = (model, objective=None, transform=None, num_features=DEFAULT_NUM_FEATURES, seed=None))]
pub fn pathwise_thompson_sampling(
    model: PyRef<'_, PyModel>,
    objective: Option<Bound<'_, PyAny>>,
    transform: Option<(Vec<f64>, f64)>,
    num_features: usize,
    seed: Option<u64>,
) -> PyResult<(PyAcquisitionFunction, PyThompsonState)> {
    let options = PathwiseThompsonSamplingOptions {
        objective: objective_of(objective.as_ref())?,
        posterior_transform: transform_of(transform)?,
        num_features,
        seed,
    };
    let a = Arc::new(PathwiseThompsonSampling::new(model_of(&model), options).into_py_result()?);
    Ok((
        PyAcquisitionFunction::new_plain(
            a.clone() as Arc<dyn AcquisitionFunction>,
            "PathwiseThompsonSampling",
        ),
        PyThompsonState { inner: a },
    ))
}

/// `qNegIntegratedPosteriorVariance(model, mc_points, posterior_transform,
/// X_pending)`.
#[pyfunction]
#[pyo3(signature = (model, mc_points, transform=None, x_pending=None))]
pub fn q_neg_integrated_posterior_variance(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    mc_points: Array,
    transform: Option<(Vec<f64>, f64)>,
    x_pending: Option<Array>,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let pts: Matrix = mc_points.matrix("mc_points")?;
    let t = transform_of(transform)?;
    let p = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
    let a = threads::run(py, || QNegIntegratedPosteriorVariance::new(m, &pts, t, p))
        .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "qNegIntegratedPosteriorVariance",
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMaxValueState>()?;
    m.add_class::<PyThompsonState>()?;
    m.add_function(wrap_pyfunction!(q_max_value_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_max_value_gumbel_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_max_value_thompson_py, m)?)?;
    m.add_function(wrap_pyfunction!(q_joint_entropy_search, m)?)?;
    m.add_function(wrap_pyfunction!(get_optimal_samples_py, m)?)?;
    m.add_function(wrap_pyfunction!(pathwise_thompson_sampling, m)?)?;
    m.add_function(wrap_pyfunction!(q_neg_integrated_posterior_variance, m)?)?;
    Ok(())
}
