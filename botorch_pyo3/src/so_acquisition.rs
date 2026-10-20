//! Single-objective acquisition functions of phase 8: the knowledge gradient
//! (`botorch.acquisition.knowledge_gradient`) with its one-shot initial
//! conditions, `ProjectedAcquisitionFunction`, `qNoisyExpectedImprovement`
//! and the analytic `(Log)NoisyExpectedImprovement`,
//! `LogProbabilityOfFeasibility`, `ScalarizedPosteriorMean` and
//! `qAnalyticProbabilityOfImprovement`.
//!
//! Value functions of the knowledge gradient arrive as `(name, kwargs)`:
//! the BoTorch class name of a native value function and its arguments
//! (`best_f`, `beta`, `tau`, `maximize`, `weights`). Expansions (`expand`)
//! arrive as `None`, `("trace", fidelity_dims, num_trace_obs)` or
//! `("callable", expand)`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use botorch_core::acquisition::fidelity::{InputProjection, TraceObservationExpansion};
use botorch_core::acquisition::knowledge_gradient::{
    KgValueFunction, ProjectedAcquisitionFunction, QKnowledgeGradient, QKnowledgeGradientOptions,
    QMultiFidelityKnowledgeGradient, QMultiFidelityKnowledgeGradientOptions,
};
use botorch_core::acquisition::one_shot::OneShotAcquisitionFunction;
use botorch_core::acquisition::utils::PruneOptions;
use botorch_core::acquisition::{
    AcquisitionFunction, LogNoisyExpectedImprovement, LogProbabilityOfFeasibility,
    NoisyExpectedImprovement, OutputConstraint, QAnalyticProbabilityOfImprovement,
    QNoisyExpectedImprovement, QNoisyExpectedImprovementOptions, SampleUtility,
    ScalarizedPosteriorMean, ScalarizedPosteriorTransform,
};
use botorch_core::lbfgsb::LbfgsbOptions;
use botorch_core::optim::{OptimizeAcqfOptions, TopnOptions, ValueFunctionInitialConditionOptions};
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::acquisition::{
    constraints_of, model_of, objective_of, sampler_of, PyAcquisitionFunction,
};
use crate::arrays::{matrix_out, stack_out, Array, ArrayOut};
use crate::callbacks::{PyProjection, PyTensorFunction};
use crate::errors::{unsupported, value_error, IntoPyResult};
use crate::hvkg::{projection_of, PyCostAwareUtility};
use crate::models::PyModel;
use crate::optim::{linear_constraints, LinearArg};
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// A posterior transform from `(weights, offset)`.
fn transform_of(t: Option<(Vec<f64>, f64)>) -> PyResult<Option<ScalarizedPosteriorTransform>> {
    t.map(|(w, offset)| ScalarizedPosteriorTransform::new(Vector::from_vec(w), offset))
        .transpose()
        .into_py_result()
}

/// An expansion from its specification (see the module documentation).
pub(crate) fn expansion_of(
    spec: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn InputProjection>>> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(None);
    };
    let kind: String = spec.get_item(0)?.extract()?;
    match kind.as_str() {
        "trace" => {
            let dims: Option<Vec<isize>> = spec.get_item(1)?.extract()?;
            let k: usize = spec.get_item(2)?.extract()?;
            Ok(Some(Arc::new(TraceObservationExpansion::new(dims, k))))
        }
        "callable" => Ok(Some(Arc::new(PyProjection(PyTensorFunction::new(
            spec.get_item(1)?.unbind(),
        ))))),
        other => Err(unsupported(format!("Unknown expansion {other:?}."))),
    }
}

/// A value function from `(name, kwargs)`.
pub(crate) fn value_function_of(
    spec: Option<(String, HashMap<String, Py<PyAny>>)>,
) -> PyResult<Option<KgValueFunction>> {
    let Some((name, kwargs)) = spec else {
        return Ok(None);
    };
    Python::attach(|py| {
        let f = |key: &str| -> PyResult<Option<f64>> {
            kwargs.get(key).map(|v| v.extract::<f64>(py)).transpose()
        };
        let req = |key: &str| -> PyResult<f64> {
            f(key)?.ok_or_else(|| value_error(format!("{name} requires `{key}`.")))
        };
        let maximize = kwargs
            .get("maximize")
            .map(|v| v.extract::<bool>(py))
            .transpose()?
            .unwrap_or(true);
        Ok(Some(match name.as_str() {
            "PosteriorMean" => KgValueFunction::PosteriorMean { maximize },
            "PosteriorStandardDeviation" => {
                KgValueFunction::PosteriorStandardDeviation { maximize }
            }
            "UpperConfidenceBound" => KgValueFunction::UpperConfidenceBound {
                beta: req("beta")?,
                maximize,
            },
            "ExpectedImprovement" => KgValueFunction::ExpectedImprovement {
                best_f: req("best_f")?,
                maximize,
            },
            "LogExpectedImprovement" => KgValueFunction::LogExpectedImprovement {
                best_f: req("best_f")?,
                maximize,
            },
            "ProbabilityOfImprovement" => KgValueFunction::ProbabilityOfImprovement {
                best_f: req("best_f")?,
                maximize,
            },
            "LogProbabilityOfImprovement" => KgValueFunction::LogProbabilityOfImprovement {
                best_f: req("best_f")?,
                maximize,
            },
            "ScalarizedPosteriorMean" => {
                let w: Vec<f64> = kwargs
                    .get("weights")
                    .ok_or_else(|| value_error("ScalarizedPosteriorMean requires `weights`."))?
                    .extract(py)?;
                KgValueFunction::ScalarizedPosteriorMean {
                    weights: Vector::from_vec(w),
                }
            }
            "qSimpleRegret" => KgValueFunction::MonteCarlo(SampleUtility::SimpleRegret),
            "qExpectedImprovement" => {
                KgValueFunction::MonteCarlo(SampleUtility::ExpectedImprovement {
                    best_f: req("best_f")?,
                })
            }
            "qProbabilityOfImprovement" => {
                KgValueFunction::MonteCarlo(SampleUtility::ProbabilityOfImprovement {
                    best_f: req("best_f")?,
                    tau: f("tau")?.unwrap_or(1e-3),
                })
            }
            "qUpperConfidenceBound" | "qLowerConfidenceBound" => {
                let beta = req("beta")?;
                let bp = (beta * std::f64::consts::PI / 2.0).sqrt();
                KgValueFunction::MonteCarlo(SampleUtility::ConfidenceBound {
                    beta_prime: if name == "qLowerConfidenceBound" {
                        -bp
                    } else {
                        bp
                    },
                })
            }
            "qPosteriorStandardDeviation" => {
                KgValueFunction::MonteCarlo(SampleUtility::PosteriorStandardDeviation)
            }
            other => {
                return Err(unsupported(format!(
                    "`valfunc_cls` {other} is not supported by botorch_rs (supported: \
                     PosteriorMean, PosteriorStandardDeviation, UpperConfidenceBound, \
                     (Log)ExpectedImprovement, (Log)ProbabilityOfImprovement, \
                     ScalarizedPosteriorMean, qSimpleRegret, qExpectedImprovement, \
                     qProbabilityOfImprovement, qUpperConfidenceBound, qLowerConfidenceBound, \
                     qPosteriorStandardDeviation)."
                )))
            }
        }))
    })
}

/// The knowledge gradient behind a Python handle.
enum Kg {
    Base(Arc<QKnowledgeGradient>),
    MultiFidelity(Arc<QMultiFidelityKnowledgeGradient>),
}

/// The state of a qKG / qMFKG (one-shot initial conditions, the inner
/// optimization of `evaluate` and the current value function).
#[pyclass(
    frozen,
    name = "KnowledgeGradientState",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyKnowledgeGradient {
    inner: Kg,
}

impl PyKnowledgeGradient {
    fn kg(&self) -> &QKnowledgeGradient {
        match &self.inner {
            Kg::Base(a) => a,
            Kg::MultiFidelity(a) => a.as_kg(),
        }
    }
}

#[pymethods]
impl PyKnowledgeGradient {
    /// `gen_one_shot_kg_initial_conditions`: `num_restarts x (q + n_f) x d`
    /// initial conditions and BoTorch's warnings.
    #[pyo3(signature = (
        bounds,
        q,
        num_restarts,
        raw_samples,
        *,
        fixed_features=None,
        inequality_constraints=None,
        equality_constraints=None,
        seed=None,
        eta=None,
        alpha=None,
        batch_limit=None,
        init_batch_limit=None,
        frac_random=None,
        num_inner_restarts=None,
        raw_inner_samples=None,
        one_shot_eta=None,
        nonnegative=false,
        n_burnin=None,
        n_thinning=None,
        topn=false,
        largest=true,
        sorted=true,
        sample_around_best=None,
        sample_around_best_sigma=None,
        sample_around_best_subset_sigma=None,
        sample_around_best_prob_perturb=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn gen_one_shot_initial_conditions(
        &self,
        py: Python<'_>,
        bounds: Array,
        q: usize,
        num_restarts: usize,
        raw_samples: usize,
        fixed_features: Option<BTreeMap<usize, f64>>,
        inequality_constraints: Option<Vec<LinearArg<'_>>>,
        equality_constraints: Option<Vec<LinearArg<'_>>>,
        seed: Option<u64>,
        eta: Option<f64>,
        alpha: Option<f64>,
        batch_limit: Option<usize>,
        init_batch_limit: Option<usize>,
        frac_random: Option<f64>,
        num_inner_restarts: Option<usize>,
        raw_inner_samples: Option<usize>,
        one_shot_eta: Option<f64>,
        nonnegative: bool,
        n_burnin: Option<usize>,
        n_thinning: Option<usize>,
        topn: bool,
        largest: bool,
        sorted: bool,
        sample_around_best: Option<Array>,
        sample_around_best_sigma: Option<f64>,
        sample_around_best_subset_sigma: Option<f64>,
        sample_around_best_prob_perturb: Option<f64>,
    ) -> PyResult<(ArrayOut, Vec<String>)> {
        let bounds = bounds.matrix("bounds")?;
        let d = bounds.ncols();
        let mut options = OptimizeAcqfOptions {
            seed,
            batch_limit,
            init_batch_limit,
            fixed_features: fixed_features.unwrap_or_default(),
            ..Default::default()
        };
        if let Some(e) = eta {
            options.eta = e;
        }
        if let Some(a) = alpha {
            options.alpha = a;
        }
        options.constraints.inequality = linear_constraints(inequality_constraints)?;
        options.constraints.equality = linear_constraints(equality_constraints)?;
        if let Some(f) = frac_random {
            options.one_shot.frac_random = f;
        }
        if let Some(n) = num_inner_restarts {
            options.one_shot.num_inner_restarts = n;
        }
        if let Some(n) = raw_inner_samples {
            options.one_shot.raw_inner_samples = n;
        }
        options.one_shot.eta = one_shot_eta;
        // The selection options of `gen_batch_initial_conditions`, which
        // BoTorch passes on with `options`.
        options.nonnegative = nonnegative;
        options.n_burnin = n_burnin.unwrap_or(options.n_burnin);
        options.n_thinning = n_thinning.unwrap_or(options.n_thinning);
        options.topn = topn.then_some(TopnOptions { largest, sorted });
        options.sample_around_best = crate::optim::sample_around_best_arg(
            sample_around_best,
            sample_around_best_sigma,
            sample_around_best_subset_sigma,
            sample_around_best_prob_perturb,
        )?;
        let acqf = self.kg();
        let q_aug = acqf.get_augmented_q_batch_size(q);
        let (ics, warnings) = threads::run(py, || {
            let seed = options
                .seed
                .unwrap_or_else(botorch_core::utils::sampling::random_seed);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x6f6e_655f_7368_6f74);
            acqf.gen_one_shot_initial_conditions(
                &bounds,
                q,
                num_restarts,
                raw_samples,
                &options,
                &mut rng,
            )
        })
        .into_py_result()?;
        Ok((stack_out(&ics, &[ics.len()], q_aug, d), warnings))
    }

    /// BoTorch's `evaluate(X, bounds, num_restarts, raw_samples, options,
    /// scipy_options)` for every candidate set of `X` (`... x q x d`).
    #[pyo3(signature = (
        x,
        bounds,
        num_restarts=20,
        raw_samples=1024,
        *,
        frac_random=None,
        num_inner_restarts=None,
        raw_inner_samples=None,
        eta=None,
        seed=None,
        maxiter=None,
        maxfun=None,
        ftol=None,
        gtol=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        py: Python<'_>,
        x: Array,
        bounds: Array,
        num_restarts: usize,
        raw_samples: usize,
        frac_random: Option<f64>,
        num_inner_restarts: Option<usize>,
        raw_inner_samples: Option<usize>,
        eta: Option<f64>,
        seed: Option<u64>,
        maxiter: Option<usize>,
        maxfun: Option<usize>,
        ftol: Option<f64>,
        gtol: Option<f64>,
    ) -> PyResult<(Vec<f64>, Vec<usize>)> {
        let (batch, sets) = x.batches("X")?;
        let bounds = bounds.matrix("bounds")?;
        let mut options = ValueFunctionInitialConditionOptions {
            seed,
            ..Default::default()
        };
        if let Some(f) = frac_random {
            options.frac_random = f;
        }
        if let Some(n) = num_inner_restarts {
            options.num_inner_restarts = n;
        }
        if let Some(n) = raw_inner_samples {
            options.raw_inner_samples = n;
        }
        if let Some(e) = eta {
            options.eta = e;
        }
        let mut lbfgsb = LbfgsbOptions {
            maxiter: 2000,
            ..Default::default()
        };
        if let Some(v) = maxiter {
            lbfgsb.maxiter = v;
        }
        if let Some(v) = maxfun {
            lbfgsb.maxfun = v;
        }
        if let Some(v) = ftol {
            lbfgsb.ftol = v;
        }
        if let Some(v) = gtol {
            lbfgsb.gtol = v;
        }
        let acqf = self.kg();
        let values = threads::run(py, || {
            let seed = seed.unwrap_or_else(botorch_core::utils::sampling::random_seed);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x6b67_5f65_7661_6c75);
            sets.iter()
                .map(|s| {
                    acqf.evaluate_with_inner_optimization(
                        s,
                        &bounds,
                        num_restarts,
                        raw_samples,
                        &options,
                        &lbfgsb,
                        &mut rng,
                    )
                })
                .collect::<botorch_core::Result<Vec<f64>>>()
        })
        .into_py_result()?;
        Ok((values, batch))
    }

    /// The value function of the current model (BoTorch's
    /// `_get_value_function` as used by the one-shot initial conditions).
    fn current_value_function(&self) -> PyResult<PyAcquisitionFunction> {
        let kg = self.kg();
        let vf = kg.current_value_function().into_py_result()?;
        let name = if kg.objective().is_some() {
            "qSimpleRegret"
        } else {
            "PosteriorMean"
        };
        Ok(PyAcquisitionFunction::new_plain(vf, name))
    }

    /// The number of fantasies.
    #[getter]
    fn num_fantasies(&self) -> usize {
        self.kg().num_fantasies()
    }

    /// The BoTorch class name of the value function.
    #[getter]
    fn value_function_name(&self) -> &'static str {
        self.kg().value_function_kind().name()
    }

    /// The seed of the inner sampler, if any.
    #[getter]
    fn inner_sampler_seed(&self) -> Option<u64> {
        self.kg().inner_sampler().map(|s| s.seed())
    }
}

/// `qKnowledgeGradient` (or, with `multi_fidelity`,
/// `qMultiFidelityKnowledgeGradient`); returns the acquisition function and
/// its state.
#[pyfunction]
#[pyo3(signature = (
    model,
    num_fantasies=Some(64),
    sampler=None,
    objective=None,
    transform=None,
    inner_sampler=None,
    x_pending=None,
    current_value=None,
    multi_fidelity=false,
    cost_aware_utility=None,
    project=None,
    expand=None,
    value_function=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_knowledge_gradient(
    model: PyRef<'_, PyModel>,
    num_fantasies: Option<usize>,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    objective: Option<Bound<'_, PyAny>>,
    transform: Option<(Vec<f64>, f64)>,
    inner_sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    x_pending: Option<Array>,
    current_value: Option<f64>,
    multi_fidelity: bool,
    cost_aware_utility: Option<PyRef<'_, PyCostAwareUtility>>,
    project: Option<Bound<'_, PyAny>>,
    expand: Option<Bound<'_, PyAny>>,
    value_function: Option<(String, HashMap<String, Py<PyAny>>)>,
) -> PyResult<(PyAcquisitionFunction, PyKnowledgeGradient)> {
    let m = model_of(&model);
    let d = m.dim();
    let kg = QKnowledgeGradientOptions {
        num_fantasies,
        sampler: sampler_of(sampler),
        objective: objective_of(objective.as_ref())?,
        posterior_transform: transform_of(transform)?,
        inner_sampler: sampler_of(inner_sampler),
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        current_value,
    };
    if !multi_fidelity {
        let a = Arc::new(QKnowledgeGradient::new(m, kg).into_py_result()?);
        return Ok((
            PyAcquisitionFunction::new_plain(
                a.clone() as Arc<dyn AcquisitionFunction>,
                "qKnowledgeGradient",
            ),
            PyKnowledgeGradient { inner: Kg::Base(a) },
        ));
    }
    let options = QMultiFidelityKnowledgeGradientOptions {
        kg,
        cost_aware_utility: cost_aware_utility.map(|u| Arc::clone(&u.inner)),
        project: projection_of(project.as_ref(), d)?,
        expand: expansion_of(expand.as_ref())?,
        value_function: value_function_of(value_function)?,
    };
    let a = Arc::new(QMultiFidelityKnowledgeGradient::new(m, options).into_py_result()?);
    Ok((
        PyAcquisitionFunction::new_plain(
            a.clone() as Arc<dyn AcquisitionFunction>,
            "qMultiFidelityKnowledgeGradient",
        ),
        PyKnowledgeGradient {
            inner: Kg::MultiFidelity(a),
        },
    ))
}

/// BoTorch's `ProjectedAcquisitionFunction(base_value_function, project)`
/// for a botorch_rs acquisition function and a projection specification.
#[pyfunction]
pub fn projected_acquisition_function(
    base: PyRef<'_, PyAcquisitionFunction>,
    project: Bound<'_, PyAny>,
) -> PyResult<PyAcquisitionFunction> {
    let d = base.inner.dim();
    let p = projection_of(Some(&project), d)?
        .ok_or_else(|| value_error("`project` must not be None"))?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(ProjectedAcquisitionFunction::new(
            Arc::clone(&base.inner),
            p,
        )),
        "ProjectedAcquisitionFunction",
    ))
}

/// The state of a qNEI (its pruned baseline and cached incumbents).
#[pyclass(
    frozen,
    name = "NoisyExpectedImprovementState",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyQNei {
    inner: Arc<QNoisyExpectedImprovement>,
}

#[pymethods]
impl PyQNei {
    /// The (pruned) baseline (`n x d`).
    fn x_baseline(&self) -> ArrayOut {
        matrix_out(self.inner.x_baseline())
    }

    /// The incumbents of the baseline samples (with `cache_root`).
    fn baseline_best_f(&self) -> Option<Vec<f64>> {
        self.inner.baseline_best_f().map(|v| v.to_vec())
    }

    /// The points of the lower-bound estimate for infeasible baselines.
    fn lower_bound_points(&self) -> ArrayOut {
        matrix_out(&self.inner.lower_bound_points())
    }

    /// The lower bound used for infeasible baseline points, if any.
    fn infeasible_lower_bound(&self) -> Option<f64> {
        self.inner.infeasible_lower_bound()
    }

    /// Whether the baseline root is cached.
    #[getter]
    fn cache_root(&self) -> bool {
        self.inner.cache_root()
    }
}

/// `qNoisyExpectedImprovement(model, X_baseline, ...)`; the baseline is
/// pruned with `prune_inferior_points` (`prune_num_samples` samples,
/// `prune_max_frac`, seeded by `prune_seed`) if `prune_baseline`.
#[pyfunction]
#[pyo3(signature = (
    model,
    x_baseline,
    sampler=None,
    x_pending=None,
    objective=None,
    constraints=None,
    eta=None,
    prune_baseline=true,
    prune_num_samples=2048,
    prune_max_frac=1.0,
    prune_seed=None,
    cache_root=true,
    lower_bound_points=None,
    objective_batch_shape=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_noisy_expected_improvement(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    x_baseline: Array,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    x_pending: Option<Array>,
    objective: Option<Bound<'_, PyAny>>,
    constraints: Option<Vec<Py<PyAny>>>,
    eta: Option<Vec<f64>>,
    prune_baseline: bool,
    prune_num_samples: usize,
    prune_max_frac: f64,
    prune_seed: Option<u64>,
    cache_root: bool,
    lower_bound_points: Option<Array>,
    objective_batch_shape: Option<Vec<usize>>,
) -> PyResult<(PyAcquisitionFunction, PyQNei)> {
    let m = model_of(&model);
    let x_baseline = x_baseline.matrix("X_baseline")?;
    let options = QNoisyExpectedImprovementOptions {
        sampler: sampler_of(sampler),
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        objective: objective_of(objective.as_ref())?,
        constraints: constraints_of(constraints, eta)?,
        prune_baseline,
        prune_options: PruneOptions {
            num_samples: prune_num_samples,
            max_frac: prune_max_frac,
            seed: prune_seed,
            ..Default::default()
        },
        cache_root,
        lower_bound_points: lower_bound_points
            .map(|p| p.matrix("lower_bound_points"))
            .transpose()?,
    };
    let acqf = threads::run(py, || {
        crate::callbacks::context::with_context(&[], None, || {
            QNoisyExpectedImprovement::new(m, &x_baseline, options)
        })
    })
    .into_py_result()?;
    let acqf = Arc::new(acqf);
    Ok((
        PyAcquisitionFunction::with_linear_nei(acqf.clone())
            .with_objective_batch(objective_batch_shape),
        PyQNei { inner: acqf },
    ))
}

/// `LogProbabilityOfFeasibility(model, constraints)` with constraints
/// `(output index, lower, upper)` (`None`: unbounded).
#[pyfunction]
pub fn log_probability_of_feasibility(
    model: PyRef<'_, PyModel>,
    constraints: Vec<OutputConstraint>,
) -> PyResult<PyAcquisitionFunction> {
    let a = LogProbabilityOfFeasibility::new(model_of(&model), constraints).into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "LogProbabilityOfFeasibility",
    ))
}

/// `ScalarizedPosteriorMean(model, weights)` (the model seen through its
/// posterior transform).
#[pyfunction]
pub fn scalarized_posterior_mean(
    model: PyRef<'_, PyModel>,
    weights: Vec<f64>,
) -> PyResult<PyAcquisitionFunction> {
    let a = ScalarizedPosteriorMean::new(model_of(&model), Vector::from_vec(weights))
        .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "ScalarizedPosteriorMean",
    ))
}

/// `qAnalyticProbabilityOfImprovement(model, best_f, maximize)` (the model
/// seen through its posterior transform).
#[pyfunction]
#[pyo3(signature = (model, best_f, maximize=true))]
pub fn q_analytic_probability_of_improvement(
    model: PyRef<'_, PyModel>,
    best_f: f64,
    maximize: bool,
) -> PyResult<PyAcquisitionFunction> {
    let a = QAnalyticProbabilityOfImprovement::new(model_of(&model), best_f, maximize)
        .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "qAnalyticProbabilityOfImprovement",
    ))
}

/// `NoisyExpectedImprovement(model, X_observed, num_fantasies, maximize)`
/// or, with `log`, `LogNoisyExpectedImprovement(..., posterior_transform)`;
/// returns the acquisition function and the incumbents of the fantasies.
#[pyfunction]
#[pyo3(signature = (model, x_observed, num_fantasies=20, maximize=true, sampler=None, log=false, transform=None))]
#[allow(clippy::too_many_arguments)]
pub fn noisy_expected_improvement(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    x_observed: Array,
    num_fantasies: usize,
    maximize: bool,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    log: bool,
    transform: Option<(Vec<f64>, f64)>,
) -> PyResult<(PyAcquisitionFunction, Vec<f64>)> {
    let m = model_of(&model);
    let x = x_observed.matrix("X_observed")?;
    let sampler = sampler_of(sampler);
    let transform = transform_of(transform)?;
    if log {
        let a = threads::run(py, || {
            LogNoisyExpectedImprovement::new(m, &x, num_fantasies, maximize, sampler, transform)
        })
        .into_py_result()?;
        let best = a.best_f().to_vec();
        return Ok((
            PyAcquisitionFunction::new_plain(Arc::new(a), "LogNoisyExpectedImprovement"),
            best,
        ));
    }
    let a = threads::run(py, || {
        NoisyExpectedImprovement::new(m, &x, num_fantasies, maximize, sampler)
    })
    .into_py_result()?;
    let best = a.best_f().to_vec();
    Ok((
        PyAcquisitionFunction::new_plain(Arc::new(a), "NoisyExpectedImprovement"),
        best,
    ))
}

/// The value of a matrix as a list of rows (for debugging helpers).
#[allow(dead_code)]
fn rows_of(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect()
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKnowledgeGradient>()?;
    m.add_class::<PyQNei>()?;
    m.add_function(wrap_pyfunction!(q_knowledge_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(projected_acquisition_function, m)?)?;
    m.add_function(wrap_pyfunction!(q_noisy_expected_improvement, m)?)?;
    m.add_function(wrap_pyfunction!(log_probability_of_feasibility, m)?)?;
    m.add_function(wrap_pyfunction!(scalarized_posterior_mean, m)?)?;
    m.add_function(wrap_pyfunction!(q_analytic_probability_of_improvement, m)?)?;
    m.add_function(wrap_pyfunction!(noisy_expected_improvement, m)?)?;
    Ok(())
}
