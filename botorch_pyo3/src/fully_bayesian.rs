//! Fully Bayesian GPs (phase 10a): the NumPyro models of BoTorch's fully
//! Bayesian GPs as potential energies (`PyroModel`) and NumPyro's NUTS on
//! them, with seeded or recorded random draws.

use botorch_core::fully_bayesian::{
    find_valid_initial_params, sample_nuts, MassMatrix, NutsOptions, NutsRandom, PyroKernel,
    PyroModel, PyroModelSpec, RecordedDraw, RecordedDraws, SeededDraws, Support, TaskStructure,
};
use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, PathwiseThompsonSampling, PathwiseThompsonSamplingOptions,
    QBayesianActiveLearningByDisagreement, ScalarizedPosteriorTransform,
};
use botorch_core::sampling::pathwise::DEFAULT_NUM_FEATURES;
use botorch_core::Vector;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::acquisition::{model_of, objective_of, PyAcquisitionFunction};
use crate::arrays::Array;
use crate::errors::{value_error, IntoPyResult};
use crate::models::PyModel;
use crate::pathwise::{draws_of, DrawsArg};
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// A NumPyro model of a fully Bayesian GP (`SaasPyroModel`,
/// `MaternPyroModel`, `LinearPyroModel`, multi-task SAAS) as a potential
/// energy of its unconstrained parameters.
#[pyclass(name = "PyroModel", module = "botorch_rs._botorch_rs", frozen)]
pub struct PyPyroModel {
    inner: PyroModel,
}

fn parse_kernel(kernel: &str) -> PyResult<PyroKernel> {
    match kernel {
        "saas" => Ok(PyroKernel::Saas),
        "matern" => Ok(PyroKernel::Matern),
        "linear" => Ok(PyroKernel::Linear),
        other => Err(value_error(format!("unknown kernel {other:?}"))),
    }
}

fn parse_draws(draws: &Bound<'_, PyList>) -> PyResult<Vec<RecordedDraw>> {
    let mut out = Vec::with_capacity(draws.len());
    for item in draws.iter() {
        let kind: String = item.get_item(0)?.extract()?;
        match kind.as_str() {
            "n" => out.push(RecordedDraw::Normal(item.get_item(1)?.extract()?)),
            "u" => out.push(RecordedDraw::Uniform {
                u: item.get_item(1)?.extract()?,
                p: item.get_item(2)?.extract()?,
            }),
            other => return Err(value_error(format!("unknown draw kind {other:?}"))),
        }
    }
    Ok(out)
}

fn mass_matrix_out(m: &MassMatrix) -> (Vec<f64>, Vec<usize>) {
    match m {
        MassMatrix::Dense(m) => {
            let (r, c) = m.shape();
            let mut data = Vec::with_capacity(r * c);
            for i in 0..r {
                for j in 0..c {
                    data.push(m[(i, j)]);
                }
            }
            (data, vec![r, c])
        }
        MassMatrix::Diagonal(d) => (d.clone(), vec![d.len()]),
    }
}

#[pymethods]
impl PyPyroModel {
    /// `PyroModel(kernel, train_x, train_y, train_yvar, outputscale_prior,
    /// warp_indices, warp_eps, task)`: `kernel` is `"saas"`, `"matern"` or
    /// `"linear"`; `task` is `(task_feature, num_tasks, rank)` for a
    /// multi-task model.
    #[new]
    #[pyo3(signature = (kernel, train_x, train_y, train_yvar=None, outputscale_prior=None,
        warp_indices=None, warp_eps=1e-7, task=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kernel: &str,
        train_x: Array,
        train_y: Vec<f64>,
        train_yvar: Option<Vec<f64>>,
        outputscale_prior: Option<(f64, f64)>,
        warp_indices: Option<Vec<usize>>,
        warp_eps: f64,
        task: Option<(usize, usize, usize)>,
    ) -> PyResult<Self> {
        let spec = PyroModelSpec {
            kernel: parse_kernel(kernel)?,
            outputscale_prior,
            train_x: train_x.matrix("train_x")?,
            train_y,
            train_yvar,
            warp_indices,
            warp_eps,
            task: task.map(|(task_feature, num_tasks, rank)| TaskStructure {
                task_feature,
                num_tasks,
                rank,
            }),
        };
        Ok(Self {
            inner: PyroModel::new(spec).into_py_result()?,
        })
    }

    /// The number of unconstrained parameters.
    #[getter]
    fn dim(&self) -> usize {
        botorch_core::fully_bayesian::Potential::dim(&self.inner)
    }

    /// The latent sites in the order of the unconstrained parameters:
    /// `(name, shape, positive)`.
    fn sites(&self) -> Vec<(String, Vec<usize>, bool)> {
        self.inner
            .sites()
            .iter()
            .map(|s| {
                (
                    s.name.to_string(),
                    s.shape.clone(),
                    s.support == Support::Positive,
                )
            })
            .collect()
    }

    /// The potential energy at `z` and its gradient.
    fn potential(&self, z: Vec<f64>) -> PyResult<(f64, Vec<f64>)> {
        if z.len() != botorch_core::fully_bayesian::Potential::dim(&self.inner) {
            return Err(value_error(format!(
                "expected {} unconstrained values, got {}",
                botorch_core::fully_bayesian::Potential::dim(&self.inner),
                z.len()
            )));
        }
        Ok(self.inner.potential(&z))
    }

    /// BoTorch's post-processed samples of the unconstrained states
    /// `samples`: `[(name, shape of one sample, values)]`.
    fn postprocess(&self, samples: Vec<Vec<f64>>) -> Vec<(String, Vec<usize>, Vec<f64>)> {
        self.inner.postprocess(&samples)
    }

    /// Runs NumPyro's NUTS. With `draws` (NumPyro's recorded draws: `("n",
    /// values)` and `("u", u, p)`) and `z0`, the chain replays NumPyro's;
    /// otherwise the draws (and the initial values `init_to_uniform`) come
    /// from a generator seeded with `seed`. Returns a dict with `samples`
    /// (`num_samples` unconstrained states), `trace` (per iteration: state,
    /// potential energy, step size, acceptance probability, number of
    /// steps, divergence), `step_size`, `inverse_mass_matrix`, `stopped_at`
    /// (with `stop_when_draws_fail`: the iteration whose draws could not be
    /// taken, e.g. where a replay left the recorded chain) and, for
    /// recorded draws, `consumed`, `num_draws`, `max_probability_error`,
    /// `first_flip` and `min_margin`.
    #[pyo3(signature = (num_warmup, num_samples, max_tree_depth=(6, 6), seed=0, z0=None,
        draws=None, target_accept_prob=0.8, dense_mass=true, step_size=1.0,
        stop_when_draws_fail=false))]
    #[allow(clippy::too_many_arguments)]
    fn sample_nuts<'py>(
        &self,
        py: Python<'py>,
        num_warmup: usize,
        num_samples: usize,
        max_tree_depth: (usize, usize),
        seed: u64,
        z0: Option<Vec<f64>>,
        draws: Option<Bound<'py, PyList>>,
        target_accept_prob: f64,
        dense_mass: bool,
        step_size: f64,
        stop_when_draws_fail: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let options = NutsOptions {
            num_warmup,
            num_samples,
            max_tree_depth,
            target_accept_prob,
            dense_mass,
            step_size,
            stop_when_draws_fail,
            ..Default::default()
        };
        let recorded = draws.as_ref().map(parse_draws).transpose()?;
        let model = &self.inner;
        let (run, diagnostics) = crate::threads::run(py, move || {
            let result: botorch_core::Result<_> = (|| match recorded {
                Some(d) => {
                    let z0 = z0.ok_or_else(|| {
                        botorch_core::BotorchError::InvalidArgument(
                            "recorded draws need the initial values z0".into(),
                        )
                    })?;
                    let mut rng = RecordedDraws::new(d);
                    let run = sample_nuts(model, &z0, &options, &mut rng)?;
                    let diag = Some((
                        rng.consumed(),
                        rng.len(),
                        rng.max_probability_error,
                        rng.first_flip,
                        rng.min_margin,
                    ));
                    Ok((run, diag))
                }
                None => {
                    let mut rng = SeededDraws::new(seed);
                    let z0 = match z0 {
                        Some(z) => z,
                        None => find_valid_initial_params(model, &mut rng)?,
                    };
                    let run = sample_nuts(model, &z0, &options, &mut rng as &mut dyn NutsRandom)?;
                    Ok((run, None))
                }
            })();
            result
        })
        .into_py_result()?;
        let out = PyDict::new(py);
        out.set_item("samples", run.samples.clone())?;
        let trace = PyList::empty(py);
        for it in &run.trace {
            trace.append((
                it.z.clone(),
                it.potential_energy,
                it.step_size,
                it.accept_prob,
                it.num_steps,
                it.diverging,
            ))?;
        }
        out.set_item("trace", trace)?;
        out.set_item("step_size", run.step_size)?;
        out.set_item(
            "inverse_mass_matrix",
            mass_matrix_out(&run.inverse_mass_matrix),
        )?;
        out.set_item("num_divergences", run.num_divergences(num_warmup))?;
        out.set_item("stopped_at", run.stopped_at)?;
        if let Some((consumed, total, max_err, flip, margin)) = diagnostics {
            out.set_item("consumed", consumed)?;
            out.set_item("num_draws", total)?;
            out.set_item("max_probability_error", max_err)?;
            out.set_item("first_flip", flip)?;
            out.set_item("min_margin", margin)?;
        }
        Ok(out)
    }
}

/// `qBayesianActiveLearningByDisagreement(model, sampler, X_pending)` on
/// the components (MCMC samples) `members` of a fully Bayesian model.
#[pyfunction]
#[pyo3(signature = (members, sampler, x_pending=None))]
pub fn q_bayesian_active_learning_by_disagreement(
    py: Python<'_>,
    members: Vec<PyRef<'_, PyModel>>,
    sampler: PyRef<'_, PySobolQMCNormalSampler>,
    x_pending: Option<Array>,
) -> PyResult<PyAcquisitionFunction> {
    let members: Vec<_> = members.iter().map(|m| model_of(m)).collect();
    let sampler = Arc::clone(&sampler.inner);
    let p = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
    let a = threads::run(py, || {
        QBayesianActiveLearningByDisagreement::new(members, sampler, p)
    })
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "qBayesianActiveLearningByDisagreement",
    ))
}

/// The state of a `PathwiseThompsonSampling` of an ensemble (its paths and
/// ensemble indices).
#[pyclass(
    frozen,
    name = "EnsembleThompsonSamplingState",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyEnsembleThompsonState {
    inner: Arc<PathwiseThompsonSampling>,
}

#[pymethods]
impl PyEnsembleThompsonState {
    /// BoTorch's `redraw(batch_size)`, from explicit `draws` (one
    /// `(frequencies, weights, noise)` per member and output, member-major)
    /// and `ensemble_indices` (`q x m`, row-major) or the next stream of the
    /// seed.
    #[pyo3(signature = (batch_size, draws=None, ensemble_indices=None))]
    fn redraw(
        &self,
        py: Python<'_>,
        batch_size: usize,
        draws: Option<Vec<DrawsArg>>,
        ensemble_indices: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let draws = draws.map(draws_of).transpose()?;
        let a = Arc::clone(&self.inner);
        threads::run(py, || {
            a.redraw_ensemble(batch_size, draws, ensemble_indices)
        })
        .into_py_result()
    }

    /// The batch size of the drawn paths (`None` before the first draw).
    #[getter]
    fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size()
    }

    /// The ensemble indices (`q x m`, row-major; `None` before the first
    /// draw).
    #[getter]
    fn ensemble_indices(&self) -> Option<Vec<usize>> {
        self.inner.ensemble_indices()
    }
}

/// `PathwiseThompsonSampling(model, objective, posterior_transform)` of an
/// ensemble model with members `members`, `num_features` Fourier features
/// and the seed of the paths and ensemble indices.
#[pyfunction]
#[pyo3(signature = (members, objective=None, transform=None, num_features=DEFAULT_NUM_FEATURES, seed=None))]
pub fn ensemble_pathwise_thompson_sampling(
    members: Vec<PyRef<'_, PyModel>>,
    objective: Option<Bound<'_, PyAny>>,
    transform: Option<(Vec<f64>, f64)>,
    num_features: usize,
    seed: Option<u64>,
) -> PyResult<(PyAcquisitionFunction, PyEnsembleThompsonState)> {
    let posterior_transform = transform
        .map(|(w, offset)| ScalarizedPosteriorTransform::new(Vector::from_vec(w), offset))
        .transpose()
        .into_py_result()?;
    let options = PathwiseThompsonSamplingOptions {
        objective: objective_of(objective.as_ref())?,
        posterior_transform,
        num_features,
        seed,
    };
    let members: Vec<_> = members.iter().map(|m| model_of(m)).collect();
    let a = Arc::new(PathwiseThompsonSampling::new_ensemble(members, options).into_py_result()?);
    Ok((
        PyAcquisitionFunction::new_plain(
            a.clone() as Arc<dyn AcquisitionFunction>,
            "PathwiseThompsonSampling",
        ),
        PyEnsembleThompsonState { inner: a },
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPyroModel>()?;
    m.add_class::<PyEnsembleThompsonState>()?;
    m.add_function(wrap_pyfunction!(ensemble_pathwise_thompson_sampling, m)?)?;
    m.add_function(wrap_pyfunction!(
        q_bayesian_active_learning_by_disagreement,
        m
    )?)?;
    Ok(())
}
