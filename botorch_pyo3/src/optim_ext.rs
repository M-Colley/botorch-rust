//! Phase 9 of the port (`botorch.optim` and `botorch.generation`): the mixed,
//! discrete, cyclic and homotopy optimizers, the torch-style candidate
//! generator, the sampling initializers, the sampling strategies' index
//! computations and the torch-style model fitting.
//!
//! Python callables (post-processing functions, candidate generators, the
//! replacement hooks of the random starting points) are called with flat
//! `(data, shape)` arrays; the Python layer converts them to the caller's
//! array type. Calls acquire the GIL (they may come from worker threads).

use std::collections::BTreeMap;
use std::sync::Arc;

use botorch_core::generation::{
    boltzmann_weights, constrained_scores, gen_candidates_torch, maximize_samples,
};
use botorch_core::optim::adam::AdamOptions;
use botorch_core::optim::cyclic::optimize_acqf_cyclic;
use botorch_core::optim::discrete::{gen_lattice_points, LocalSearchOptions};
use botorch_core::optim::fit::{fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch};
use botorch_core::optim::mixed::{
    continuous_step, discrete_step, generate_starting_points, get_categorical_neighbors,
    get_nearest_neighbors, get_spray_points, mixed_problem, relaxation_post_processing,
    round_discrete_dims, sample_feasible_points, DiscreteDims, InitializationStrategy,
    MixedAlternatingOptions, MixedProblem,
};
use botorch_core::optim::stopping::{expma_weights, ExpMaStoppingCriterion};
use botorch_core::optim::{
    gen_candidates_lbfgsb, gen_candidates_lbfgsb_joint, gen_candidates_slsqp,
    optimize_acqf_discrete, optimize_acqf_discrete_local_search, optimize_acqf_mixed,
    prune_candidates, sample_perturbed_subset_dims, sample_points_around,
    sample_truncated_normal_perturbations, truncated_normal_perturbations_from_uniform,
    CustomGenerator, GenRequest, OptimizationResult, OptimizerStatus, PostProcessing,
    SampleAroundBest, SlsqpOptions, TorchMinimizeOptions,
};
use botorch_core::{AcquisitionFunction, BotorchError, Matrix};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::acquisition::PyAcquisitionFunction;
use crate::arrays::{float_values, matrix_out, stack_out, Array, ArrayOut};
use crate::callbacks::callback_error;
use crate::errors::{value_error, IntoPyResult};
use crate::models::PySingleTaskGP;
use crate::optim::{acqf_options, kw, linear_constraints, LinearArg};
use crate::threads;

/// The rows of `m` in row-major order.
fn flat(m: &Matrix) -> Vec<f64> {
    matrix_out(m).0
}

/// A random number generator seeded from `seed` (or a random seed).
fn rng(seed: Option<u64>) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(
        seed.unwrap_or_else(botorch_core::utils::sampling::random_seed),
    )
}

/// The candidate sets returned by a Python callable as `(data, shape)` of
/// shape `b x q x d` (or `q x d` for one set).
fn sets_from(
    Array(data, shape): Array,
    b: usize,
    q: usize,
    d: usize,
) -> botorch_core::Result<Vec<Matrix>> {
    let ok = match shape.as_slice() {
        [bb, qq, dd] => *bb == b && *qq == q && *dd == d,
        [qq, dd] => b == 1 && *qq == q && *dd == d,
        _ => false,
    };
    if !ok || data.len() != b * q * d {
        return Err(BotorchError::DimensionMismatch(format!(
            "expected candidates of shape {:?}, got {shape:?}",
            [b, q, d]
        )));
    }
    Ok((0..b)
        .map(|i| Matrix::from_row_slice(q, d, &data[i * q * d..(i + 1) * q * d]))
        .collect())
}

/// A post-processing function calling `f(data, shape)` (shape `b x q x d`)
/// and expecting `(data, shape)` back.
pub fn py_post_processing(f: Py<PyAny>) -> PostProcessing {
    let f = Arc::new(f);
    PostProcessing::new(move |sets: &[Matrix]| {
        let b = sets.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        let (q, d) = sets[0].shape();
        let data: Vec<f64> = sets.iter().flat_map(flat).collect();
        let out: Array = Python::attach(|py| -> PyResult<Array> {
            f.bind(py)
                .call1((float_values(py, &data)?, vec![b, q, d]))?
                .extract()
        })
        .map_err(callback_error)?;
        sets_from(out, b, q, d)
    })
}

/// A candidate generator calling `generate(data, shape, timeout_sec,
/// fixed_features)` with the initial conditions (`b x q x d`), expecting
/// `((data, shape), values)`.
pub fn custom_generator(f: Py<PyAny>) -> CustomGenerator {
    let f = Arc::new(f);
    CustomGenerator::new(move |req: &GenRequest<'_>| {
        let b = req.initial_conditions.len();
        let (q, d) = req.initial_conditions[0].shape();
        let data: Vec<f64> = req.initial_conditions.iter().flat_map(flat).collect();
        let fixed: BTreeMap<usize, f64> = req.fixed_features.clone();
        let (out, values): (Array, Vec<f64>) =
            Python::attach(|py| -> PyResult<(Array, Vec<f64>)> {
                f.bind(py)
                    .call1((
                        float_values(py, &data)?,
                        vec![b, q, d],
                        req.timeout_sec,
                        fixed,
                    ))?
                    .extract()
            })
            .map_err(callback_error)?;
        Ok((sets_from(out, b, q, d)?, values))
    })
}

/// `(candidates, values, warnings)` of an optimization.
type CandidatesOut = (ArrayOut, Vec<f64>, Vec<String>);

/// `(candidates, value, restart candidates, restart values, warnings)` of
/// `optimize_acqf_mixed`.
type MixedOut = (ArrayOut, f64, ArrayOut, Vec<f64>, Vec<String>);

/// BoTorch's `optimize_acqf_mixed`: returns the candidates (`q x d`), the
/// value, all restarts of the best fixed features (`b x 1 x d`) with their
/// values (for `return_best_only=False`), and the warnings.
#[pyfunction]
#[pyo3(signature = (
    acq_function, bounds, q, num_restarts, fixed_features_list, raw_samples=None,
    return_best_only=true, **kwargs
))]
#[allow(clippy::too_many_arguments)]
fn optimize_acqf_mixed_py(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    bounds: Array,
    q: usize,
    num_restarts: usize,
    fixed_features_list: Vec<BTreeMap<usize, f64>>,
    raw_samples: Option<usize>,
    return_best_only: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<MixedOut> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let options = acqf_options(d, kwargs)?;
    let acqf = &acq_function.inner;
    let res = threads::run(py, || {
        optimize_acqf_mixed(
            acqf.as_ref(),
            &bounds,
            q,
            num_restarts,
            &fixed_features_list,
            raw_samples.unwrap_or(0),
            &options,
            return_best_only,
        )
    })
    .into_py_result()?;
    let mats: Vec<Matrix> = res.restarts.iter().map(|r| r.0.clone()).collect();
    let rq = mats.first().map_or(1, |m| m.nrows());
    let values = res.restarts.iter().map(|r| r.1).collect();
    Ok((
        matrix_out(&res.candidates),
        res.acq_value,
        stack_out(&mats, &[mats.len()], rq, d),
        values,
        res.warnings,
    ))
}

/// BoTorch's `optimize_acqf_discrete`.
#[pyfunction]
#[pyo3(signature = (acq_function, q, choices, unique=true, x_avoid=None, inequality_constraints=None))]
fn optimize_acqf_discrete_py(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    q: usize,
    choices: Array,
    unique: bool,
    x_avoid: Option<Array>,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
) -> PyResult<CandidatesOut> {
    let choices = choices.matrix("choices")?;
    let x_avoid = x_avoid.map(|a| a.matrix("X_avoid")).transpose()?;
    let ineq = match inequality_constraints {
        Some(cs) => Some(linear_constraints(Some(cs))?),
        None => None,
    };
    let acqf = &acq_function.inner;
    let res = threads::run(py, || {
        optimize_acqf_discrete(
            acqf.as_ref(),
            q,
            &choices,
            unique,
            x_avoid.as_ref(),
            ineq.as_deref(),
        )
    })
    .into_py_result()?;
    Ok((matrix_out(&res.candidates), res.acq_values, res.warnings))
}

/// BoTorch's `optimize_acqf_discrete_local_search`. `lattice_points`
/// optionally replaces the random starting points: it is called with the
/// points to avoid `(data, shape)` and returns `(data, shape)` of `n x d`.
#[pyfunction]
#[pyo3(signature = (
    acq_function, discrete_choices, q, num_restarts=20, raw_samples=4096,
    inequality_constraints=None, x_avoid=None, batch_initial_conditions=None, max_tries=100,
    unique=true, return_acq_values=true, lattice_points=None, seed=None
))]
#[allow(clippy::too_many_arguments)]
fn optimize_acqf_discrete_local_search_py(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    discrete_choices: Vec<Vec<f64>>,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
    x_avoid: Option<Array>,
    batch_initial_conditions: Option<Array>,
    max_tries: usize,
    unique: bool,
    return_acq_values: bool,
    lattice_points: Option<Py<PyAny>>,
    seed: Option<u64>,
) -> PyResult<(ArrayOut, Vec<f64>)> {
    let ineq = linear_constraints(inequality_constraints)?;
    let x_avoid = x_avoid.map(|a| a.matrix("X_avoid")).transpose()?;
    let bic = batch_initial_conditions
        .map(|a| a.points("batch_initial_conditions").map(|p| p.1))
        .transpose()?;
    let options = LocalSearchOptions {
        num_restarts,
        raw_samples,
        max_tries,
        unique,
    };
    let dim = discrete_choices.len();
    let hook = lattice_points.map(Arc::new);
    let acqf = &acq_function.inner;
    let res = threads::run(py, || {
        let generator = hook.as_ref().map(|f| {
            let f = f.clone();
            move |avoid: &Matrix| -> botorch_core::Result<Matrix> {
                let Array(data, shape) = Python::attach(|py| -> PyResult<Array> {
                    f.bind(py)
                        .call1((float_values(py, &flat(avoid))?, vec![avoid.nrows(), dim]))?
                        .extract()
                })
                .map_err(callback_error)?;
                match shape.as_slice() {
                    [n, dd] if *dd == dim && data.len() == n * dim => {
                        Ok(Matrix::from_row_slice(*n, dim, &data))
                    }
                    s => Err(BotorchError::DimensionMismatch(format!(
                        "the lattice points must be n x {dim}, got shape {s:?}"
                    ))),
                }
            }
        });
        let mut r = rng(seed);
        optimize_acqf_discrete_local_search(
            acqf.as_ref(),
            &discrete_choices,
            q,
            &options,
            &ineq,
            x_avoid.as_ref(),
            bic.as_ref(),
            generator
                .as_ref()
                .map(|g| g as &(dyn Fn(&Matrix) -> botorch_core::Result<Matrix> + Sync)),
            return_acq_values,
            &mut r,
        )
    })
    .into_py_result()?;
    Ok((matrix_out(&res.candidates), res.acq_values))
}

/// BoTorch's `_gen_batch_initial_conditions_local_search` (random lattice
/// points).
#[pyfunction]
#[pyo3(signature = (discrete_choices, raw_samples, x_avoid, inequality_constraints, min_points, max_tries=100, seed=None))]
#[allow(clippy::too_many_arguments)]
fn gen_lattice_points_py(
    py: Python<'_>,
    discrete_choices: Vec<Vec<f64>>,
    raw_samples: usize,
    x_avoid: Array,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
    min_points: usize,
    max_tries: usize,
    seed: Option<u64>,
) -> PyResult<ArrayOut> {
    let ineq = linear_constraints(inequality_constraints)?;
    let dim = discrete_choices.len();
    let avoid = if x_avoid.0.is_empty() {
        Matrix::zeros(0, dim)
    } else {
        x_avoid.matrix("X_avoid")?
    };
    let out = threads::run(py, || {
        let mut r = rng(seed);
        gen_lattice_points(
            &discrete_choices,
            raw_samples,
            &avoid,
            &ineq,
            min_points,
            max_tries,
            &mut r,
        )
    })
    .into_py_result()?;
    Ok(matrix_out(&out))
}

/// BoTorch's `optimize_acqf_cyclic`; `cyclic` holds the stopping criterion's
/// `(maxiter, minimize, n_window, eta, rel_tol)`.
#[pyfunction]
#[pyo3(signature = (acq_function, bounds, q, num_restarts, raw_samples, cyclic, **kwargs))]
#[allow(clippy::too_many_arguments)]
fn optimize_acqf_cyclic_py(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    bounds: Array,
    q: usize,
    num_restarts: usize,
    raw_samples: Option<usize>,
    cyclic: (usize, bool, usize, f64, f64),
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<CandidatesOut> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let options = acqf_options(d, kwargs)?;
    let stopping = ExpMaStoppingCriterion::new(cyclic.0, cyclic.1, cyclic.2, cyclic.3, cyclic.4);
    let acqf = &acq_function.inner;
    let res = threads::run(py, || {
        optimize_acqf_cyclic(
            acqf.as_ref(),
            &bounds,
            q,
            num_restarts,
            raw_samples.unwrap_or(0),
            &options,
            &stopping,
        )
    })
    .into_py_result()?;
    Ok((matrix_out(&res.candidates), res.acq_values, res.warnings))
}

/// The search space and settings of BoTorch's mixed alternating optimizer
/// (`opt_inputs` with the dimension maps), on which the building blocks of
/// `botorch.optim.optimize_mixed` run.
#[pyclass(frozen, name = "MixedProblem", module = "botorch_rs._botorch_rs")]
pub struct PyMixedProblem {
    acqf: Arc<dyn AcquisitionFunction>,
    bounds: Matrix,
    discrete_dims: DiscreteDims,
    cat_dims: DiscreteDims,
    cont_dims: Vec<usize>,
    relaxed: DiscreteDims,
    fixed_features: BTreeMap<usize, f64>,
    constraints: botorch_core::optim::ParameterConstraints,
    options: MixedAlternatingOptions,
    num_restarts: usize,
    raw_samples: usize,
    post_processing: Option<PostProcessing>,
    retry: bool,
    x_baseline: Option<Matrix>,
}

impl PyMixedProblem {
    fn problem(&self) -> MixedProblem<'_> {
        MixedProblem {
            acq_function: self.acqf.as_ref(),
            bounds: &self.bounds,
            discrete_dims: self.discrete_dims.clone(),
            cat_dims: self.cat_dims.clone(),
            cont_dims: self.cont_dims.clone(),
            fixed_features: self.fixed_features.clone(),
            constraints: self.constraints.clone(),
            options: self.options,
            num_restarts: self.num_restarts,
            raw_samples: self.raw_samples,
            post_processing: self.post_processing.clone(),
            retry_on_optimization_warning: self.retry,
            x_baseline: self.x_baseline.clone(),
        }
    }
}

/// The options of the alternating optimizer from keyword arguments.
fn alternating_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<MixedAlternatingOptions> {
    let d = MixedAlternatingOptions::default();
    let strategy: Option<String> = kw(kwargs, "initialization_strategy")?;
    let initialization_strategy = match strategy.as_deref() {
        None => None,
        Some("continuous_relaxation") => Some(InitializationStrategy::ContinuousRelaxation),
        Some("equally_spaced") => Some(InitializationStrategy::EquallySpaced),
        Some("random") => Some(InitializationStrategy::Random),
        Some(other) => {
            return Err(crate::errors::unsupported(format!(
                "Unsupported initialization strategy: {other}.Supported strategies are: \
                 {{'continuous_relaxation', 'equally_spaced', 'random'}}."
            )))
        }
    };
    Ok(MixedAlternatingOptions {
        initialization_strategy,
        tol: kw(kwargs, "tol")?.unwrap_or(d.tol),
        maxiter_alternating: kw(kwargs, "maxiter_alternating")?.unwrap_or(d.maxiter_alternating),
        maxiter_discrete: kw(kwargs, "maxiter_discrete")?.unwrap_or(d.maxiter_discrete),
        maxiter_continuous: kw(kwargs, "maxiter_continuous")?.unwrap_or(d.maxiter_continuous),
        maxiter_init: kw(kwargs, "maxiter_init")?.unwrap_or(d.maxiter_init),
        max_discrete_values: kw(kwargs, "max_discrete_values")?.unwrap_or(d.max_discrete_values),
        num_spray_points: kw(kwargs, "num_spray_points")?,
        std_cont_perturbation: kw(kwargs, "std_cont_perturbation")?
            .unwrap_or(d.std_cont_perturbation),
        batch_limit: kw(kwargs, "batch_limit")?.unwrap_or(d.batch_limit),
        init_batch_limit: kw(kwargs, "init_batch_limit")?,
    })
}

#[pymethods]
impl PyMixedProblem {
    /// Validates the arguments like BoTorch's `optimize_acqf_mixed_alternating`
    /// and builds the problem (`options` as keyword arguments).
    #[new]
    #[pyo3(signature = (
        acq_function, bounds, discrete_dims, cat_dims, num_restarts, raw_samples,
        fixed_features=None, inequality_constraints=None, equality_constraints=None,
        post_processing=None, retry_on_optimization_warning=true, x_baseline=None, **kwargs
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        acq_function: PyRef<'_, PyAcquisitionFunction>,
        bounds: Array,
        discrete_dims: BTreeMap<usize, Vec<f64>>,
        cat_dims: BTreeMap<usize, Vec<f64>>,
        num_restarts: usize,
        raw_samples: usize,
        fixed_features: Option<BTreeMap<usize, f64>>,
        inequality_constraints: Option<Vec<LinearArg<'_>>>,
        equality_constraints: Option<Vec<LinearArg<'_>>>,
        post_processing: Option<Py<PyAny>>,
        retry_on_optimization_warning: bool,
        x_baseline: Option<Array>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let bounds = bounds.matrix("bounds")?;
        let options = alternating_options(kwargs)?;
        let constraints = botorch_core::optim::ParameterConstraints {
            inequality: linear_constraints(inequality_constraints)?,
            equality: linear_constraints(equality_constraints)?,
            nonlinear: Vec::new(),
        };
        let fixed = fixed_features.unwrap_or_default();
        let x_baseline = x_baseline.map(|a| a.matrix("X_baseline")).transpose()?;
        let acqf = acq_function.inner.clone();
        let (problem, relaxed) = mixed_problem(
            acqf.as_ref(),
            &bounds,
            &discrete_dims,
            &cat_dims,
            &options,
            num_restarts,
            raw_samples,
            &fixed,
            &constraints,
            retry_on_optimization_warning,
            x_baseline.clone(),
        )
        .into_py_result()?;
        let user = post_processing.map(py_post_processing);
        let post_processing = relaxation_post_processing(relaxed.clone(), user);
        Ok(Self {
            discrete_dims: problem.discrete_dims.clone(),
            cat_dims: problem.cat_dims.clone(),
            cont_dims: problem.cont_dims.clone(),
            acqf: acqf.clone(),
            bounds: bounds.clone(),
            relaxed,
            fixed_features: fixed,
            constraints,
            options,
            num_restarts,
            raw_samples,
            post_processing,
            retry: retry_on_optimization_warning,
            x_baseline,
        })
    }

    /// The discrete dimensions of the discrete steps.
    #[getter]
    fn discrete_dims(&self) -> BTreeMap<usize, Vec<f64>> {
        self.discrete_dims.clone()
    }

    /// The categorical dimensions.
    #[getter]
    fn cat_dims(&self) -> BTreeMap<usize, Vec<f64>> {
        self.cat_dims.clone()
    }

    /// The continuous dimensions.
    #[getter]
    fn cont_dims(&self) -> Vec<usize> {
        self.cont_dims.clone()
    }

    /// The discrete dimensions optimized by continuous relaxation.
    #[getter]
    fn relaxed_dims(&self) -> BTreeMap<usize, Vec<f64>> {
        self.relaxed.clone()
    }

    /// `generate_starting_points`: `(points, values, warnings)`.
    #[pyo3(signature = (seed=None))]
    fn generate_starting_points(
        &self,
        py: Python<'_>,
        seed: Option<u64>,
    ) -> PyResult<(ArrayOut, Vec<f64>, Vec<String>)> {
        let (x, v, w) = threads::run(py, || {
            let mut r = rng(seed);
            generate_starting_points(&self.problem(), &mut r)
        })
        .into_py_result()?;
        Ok((matrix_out(&x), v, w))
    }

    /// `discrete_step` from the points `x` (`b x d`): `(points, values)`.
    #[pyo3(signature = (x, seed=None))]
    fn discrete_step(
        &self,
        py: Python<'_>,
        x: Array,
        seed: Option<u64>,
    ) -> PyResult<(ArrayOut, Vec<f64>)> {
        let mut x = x.matrix("current_x")?;
        let v = threads::run(py, || {
            let mut r = rng(seed);
            discrete_step(&self.problem(), &mut x, &mut r)
        })
        .into_py_result()?;
        Ok((matrix_out(&x), v))
    }

    /// `continuous_step` from the points `x` (`b x d`): `(points, values,
    /// warnings)`.
    fn continuous_step(
        &self,
        py: Python<'_>,
        x: Array,
    ) -> PyResult<(ArrayOut, Vec<f64>, Vec<String>)> {
        let x = x.matrix("current_x")?;
        let (out, v, w) =
            threads::run(py, || continuous_step(&self.problem(), &x)).into_py_result()?;
        Ok((matrix_out(&out), v, w))
    }

    /// `sample_feasible_points`.
    #[pyo3(signature = (num_points, seed=None))]
    fn sample_feasible_points(
        &self,
        py: Python<'_>,
        num_points: usize,
        seed: Option<u64>,
    ) -> PyResult<ArrayOut> {
        let out = threads::run(py, || {
            let mut r = rng(seed);
            sample_feasible_points(&self.problem(), num_points, &mut r)
        })
        .into_py_result()?;
        Ok(matrix_out(&out))
    }

    /// The post-processing of the candidates (`b x q x d`, flat): rounding of
    /// the relaxed dimensions and the user's function.
    fn post_process(&self, py: Python<'_>, x: Array) -> PyResult<ArrayOut> {
        let (batch, sets) = x.batches("X")?;
        let Some(pp) = &self.post_processing else {
            return Ok((x.0, x.1));
        };
        let out = threads::run(py, || pp.apply(&sets)).into_py_result()?;
        let (q, d) = out.first().map_or((0, 0), |m| m.shape());
        Ok(stack_out(&out, &batch, q, d))
    }

    /// `optimize_acqf_mixed_alternating` for `q` candidates: `(candidates,
    /// joint value, warnings)`. `starting_points` optionally replaces
    /// `generate_starting_points`: it is called without arguments and returns
    /// `((data, shape), values)`.
    #[pyo3(signature = (q, seed=None, starting_points=None, return_acq_values=true))]
    fn optimize(
        &self,
        py: Python<'_>,
        q: usize,
        seed: Option<u64>,
        starting_points: Option<Py<PyAny>>,
        return_acq_values: bool,
    ) -> PyResult<(ArrayOut, f64, Vec<String>)> {
        let d = self.bounds.ncols();
        let hook = starting_points.map(Arc::new);
        let res = threads::run(py, || {
            let f = hook.as_ref().map(|f| {
                let f = f.clone();
                move |_: &MixedProblem<'_>| -> botorch_core::Result<(Matrix, Vec<f64>)> {
                    let (Array(data, shape), values): (Array, Vec<f64>) =
                        Python::attach(|py| -> PyResult<(Array, Vec<f64>)> {
                            f.bind(py).call0()?.extract()
                        })
                        .map_err(callback_error)?;
                    match shape.as_slice() {
                        [n, dd] if *dd == d && data.len() == n * d => {
                            Ok((Matrix::from_row_slice(*n, d, &data), values))
                        }
                        s => Err(BotorchError::DimensionMismatch(format!(
                            "the starting points must be n x {d}, got shape {s:?}"
                        ))),
                    }
                }
            });
            botorch_core::optim::optimize_acqf_mixed_alternating(
                &self.problem(),
                q,
                seed,
                f.as_ref().map(|g| {
                    g as &(dyn Fn(&MixedProblem<'_>) -> botorch_core::Result<(Matrix, Vec<f64>)>
                          + Sync)
                }),
                return_acq_values,
            )
        })
        .into_py_result()?;
        Ok((matrix_out(&res.candidates), res.acq_value, res.warnings))
    }
}

/// BoTorch's `round_discrete_dims` on `n x d` points.
#[pyfunction]
fn round_discrete_dims_py(
    x: Array,
    discrete_dims: BTreeMap<usize, Vec<f64>>,
) -> PyResult<ArrayOut> {
    let (batch, mut m) = x.points("X")?;
    round_discrete_dims(&mut m, &discrete_dims);
    let d = m.ncols();
    let mut shape = batch;
    shape.push(d);
    Ok((flat(&m), shape))
}

/// BoTorch's `get_nearest_neighbors`.
#[pyfunction]
fn get_nearest_neighbors_py(
    current_x: Vec<f64>,
    bounds: Array,
    discrete_dims: BTreeMap<usize, Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    let bounds = bounds.matrix("bounds")?;
    if current_x.len() != bounds.ncols() {
        return Err(crate::errors::dimension_error(
            "current_x must have d entries",
        ));
    }
    if discrete_dims.keys().any(|&k| k >= current_x.len()) {
        return Err(value_error("discrete dimension out of range"));
    }
    Ok(get_nearest_neighbors(&current_x, &bounds, &discrete_dims))
}

/// BoTorch's `get_categorical_neighbors`.
#[pyfunction]
#[pyo3(signature = (current_x, cat_dims, max_num_cat_values=20, seed=None))]
fn get_categorical_neighbors_py(
    current_x: Vec<f64>,
    cat_dims: BTreeMap<usize, Vec<f64>>,
    max_num_cat_values: usize,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<f64>>> {
    if cat_dims.keys().any(|&k| k >= current_x.len()) {
        return Err(value_error("categorical dimension out of range"));
    }
    let mut r = rng(seed);
    Ok(get_categorical_neighbors(
        &current_x,
        &cat_dims,
        max_num_cat_values,
        &mut r,
    ))
}

/// BoTorch's `get_spray_points`.
#[pyfunction]
#[pyo3(signature = (x_baseline, cont_dims, discrete_dims, cat_dims, bounds, num_spray_points, std_cont_perturbation=0.1, seed=None))]
#[allow(clippy::too_many_arguments)]
fn get_spray_points_py(
    x_baseline: Array,
    cont_dims: Vec<usize>,
    discrete_dims: BTreeMap<usize, Vec<f64>>,
    cat_dims: BTreeMap<usize, Vec<f64>>,
    bounds: Array,
    num_spray_points: usize,
    std_cont_perturbation: f64,
    seed: Option<u64>,
) -> PyResult<ArrayOut> {
    let xb = x_baseline.matrix("X_baseline")?;
    let bounds = bounds.matrix("bounds")?;
    let mut r = rng(seed);
    let out = get_spray_points(
        &xb,
        &cont_dims,
        &discrete_dims,
        &cat_dims,
        &bounds,
        num_spray_points,
        std_cont_perturbation,
        &mut r,
    )
    .into_py_result()?;
    Ok(matrix_out(&out))
}

/// BoTorch's `prune_candidates`.
#[pyfunction]
fn prune_candidates_py(
    candidates: Array,
    acq_values: Vec<f64>,
    prune_tolerance: f64,
) -> PyResult<ArrayOut> {
    let c = candidates.matrix("candidates")?;
    let out = prune_candidates(&c, &acq_values, prune_tolerance).into_py_result()?;
    Ok(matrix_out(&out))
}

/// BoTorch's `gen_candidates_torch` from initial conditions (`b x q x d`):
/// `(candidates, values, steps)`. `callback(i, loss, grad_data, grad_shape)`
/// is called after every evaluation.
#[pyfunction]
#[pyo3(signature = (acq_function, initial_conditions, bounds, fixed_features=None, callback=None, **kwargs))]
fn gen_candidates_torch_py(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    initial_conditions: Array,
    bounds: Array,
    fixed_features: Option<BTreeMap<usize, f64>>,
    callback: Option<Py<PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(ArrayOut, Vec<f64>, usize)> {
    let (batch, ics) = initial_conditions.batches("initial_conditions")?;
    let bounds = bounds.matrix("bounds")?;
    let options = crate::optim::torch_gen_options(kwargs)?;
    let fixed = fixed_features.unwrap_or_default();
    let acqf = &acq_function.inner;
    let res = threads::run(py, || {
        let mut cb = callback.as_ref().map(|f| {
            move |i: usize, loss: f64, grads: &[Matrix]| -> botorch_core::Result<()> {
                let (q, nf) = grads.first().map_or((0, 0), |g| g.shape());
                let data: Vec<f64> = grads.iter().flat_map(flat).collect();
                Python::attach(|py| -> PyResult<()> {
                    f.bind(py).call1((
                        i,
                        loss,
                        float_values(py, &data)?,
                        vec![grads.len(), q, nf],
                    ))?;
                    Ok(())
                })
                .map_err(callback_error)
            }
        });
        gen_candidates_torch(
            &ics,
            acqf.as_ref(),
            &bounds,
            &fixed,
            &options,
            cb.as_mut()
                .map(|c| c as &mut dyn FnMut(usize, f64, &[Matrix]) -> botorch_core::Result<()>),
        )
    })
    .into_py_result()?;
    let (q, d) = res.candidates[0].shape();
    Ok((
        stack_out(&res.candidates, &batch, q, d),
        res.acq_values,
        res.steps,
    ))
}

/// BoTorch's `gen_candidates_scipy` from initial conditions (`b x q x d`):
/// independent L-BFGS-B runs (`aggregation=None`), groups of `aggregation`
/// restarts as one L-BFGS-B problem, or SLSQP with constraints. Returns
/// `(candidates, values, failed)` where `failed` reports BoTorch's
/// `OptimizationWarning`s.
#[pyfunction]
#[pyo3(signature = (acq_function, initial_conditions, bounds, fixed_features=None, aggregation=None, use_slsqp=false, timeout_sec=None, **kwargs))]
#[allow(clippy::too_many_arguments)]
fn gen_candidates_scipy_py(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    initial_conditions: Array,
    bounds: Array,
    fixed_features: Option<BTreeMap<usize, f64>>,
    aggregation: Option<usize>,
    use_slsqp: bool,
    timeout_sec: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(ArrayOut, Vec<f64>, Vec<String>)> {
    let (batch, ics) = initial_conditions.batches("initial_conditions")?;
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let options = acqf_options(d, kwargs)?;
    let fixed = fixed_features.unwrap_or_default();
    let acqf = &acq_function.inner;
    let (cands, values, warnings) = threads::run(py, || -> botorch_core::Result<_> {
        if use_slsqp || !options.constraints.is_empty() {
            let r = gen_candidates_slsqp(
                &ics,
                acqf.as_ref(),
                &bounds,
                &fixed,
                &options.constraints,
                &SlsqpOptions {
                    maxiter: options.slsqp.maxiter,
                    ftol: options.slsqp.ftol,
                },
            )?;
            return Ok((r.candidates, r.acq_values, r.warnings));
        }
        let rs = match aggregation {
            None => gen_candidates_lbfgsb(&ics, acqf.as_ref(), &bounds, &fixed, &options.lbfgsb)?,
            Some(k) => gen_candidates_lbfgsb_joint(
                &ics,
                acqf.as_ref(),
                &bounds,
                &fixed,
                &options.lbfgsb,
                k,
                timeout_sec,
            )?,
        };
        let failed = rs.iter().any(|r| {
            r.status
                == OptimizerStatus::Lbfgsb(botorch_core::lbfgsb::LbfgsbStatus::AbnormalLineSearch)
        });
        let warnings = if failed {
            vec!["ABNORMAL: ABNORMAL_TERMINATION_IN_LNSRCH".to_string()]
        } else {
            Vec::new()
        };
        Ok((
            rs.iter().map(|r| r.candidates.clone()).collect::<Vec<_>>(),
            rs.iter().map(|r| r.acq_value).collect(),
            warnings,
        ))
    })
    .into_py_result()?;
    let (q, dd) = cands.first().map_or((0, d), |m| m.shape());
    Ok((stack_out(&cands, &batch, q, dd), values, warnings))
}

/// BoTorch's `sample_truncated_normal_perturbations`.
#[pyfunction]
#[pyo3(signature = (x, n_discrete_points, sigma, bounds, qmc=true, seed=None))]
fn sample_truncated_normal_perturbations_py(
    x: Array,
    n_discrete_points: usize,
    sigma: f64,
    bounds: Array,
    qmc: bool,
    seed: Option<u64>,
) -> PyResult<ArrayOut> {
    let x = x.matrix("X")?;
    let bounds = bounds.matrix("bounds")?;
    let mut r = rng(seed);
    let out =
        sample_truncated_normal_perturbations(&x, n_discrete_points, sigma, &bounds, qmc, &mut r)
            .into_py_result()?;
    Ok(matrix_out(&out))
}

/// The deterministic transform of `sample_truncated_normal_perturbations`
/// for given centers and uniform draws.
#[pyfunction]
fn truncated_normal_perturbations_py(
    centers: Array,
    u: Array,
    sigma: f64,
    bounds: Array,
) -> PyResult<ArrayOut> {
    let out = truncated_normal_perturbations_from_uniform(
        &centers.matrix("centers")?,
        &u.matrix("u")?,
        sigma,
        &bounds.matrix("bounds")?,
    )
    .into_py_result()?;
    Ok(matrix_out(&out))
}

/// BoTorch's `sample_perturbed_subset_dims`.
#[pyfunction]
#[pyo3(signature = (x, bounds, n_discrete_points, sigma=0.1, qmc=true, prob_perturb=None, seed=None))]
fn sample_perturbed_subset_dims_py(
    x: Array,
    bounds: Array,
    n_discrete_points: usize,
    sigma: f64,
    qmc: bool,
    prob_perturb: Option<f64>,
    seed: Option<u64>,
) -> PyResult<ArrayOut> {
    let x = x.matrix("X")?;
    let bounds = bounds.matrix("bounds")?;
    let mut r = rng(seed);
    let out = sample_perturbed_subset_dims(
        &x,
        &bounds,
        n_discrete_points,
        sigma,
        qmc,
        prob_perturb,
        &mut r,
    )
    .into_py_result()?;
    Ok(matrix_out(&out))
}

/// The perturbations of `sample_points_around_best` around the given best
/// points.
#[pyfunction]
#[pyo3(signature = (best_x, n_discrete_points, sigma, bounds, subset_sigma=0.1, prob_perturb=None, seed=None))]
#[allow(clippy::too_many_arguments)]
fn sample_points_around_py(
    best_x: Array,
    n_discrete_points: usize,
    sigma: f64,
    bounds: Array,
    subset_sigma: f64,
    prob_perturb: Option<f64>,
    seed: Option<u64>,
) -> PyResult<ArrayOut> {
    let best = SampleAroundBest {
        best_x: best_x.matrix("best_X")?,
        sigma,
        subset_sigma,
        prob_perturb,
    };
    let bounds = bounds.matrix("bounds")?;
    let mut r = rng(seed);
    let out = sample_points_around(&best, n_discrete_points, &bounds, &mut r).into_py_result()?;
    Ok(matrix_out(&out))
}

/// The indices of `MaxPosteriorSampling.maximize_samples` (`batch x
/// num_samples`, flat) for objective values of shape `num_samples x batch x
/// n`.
#[pyfunction]
fn maximize_samples_py(
    obj: Vec<f64>,
    num_samples: usize,
    batch: usize,
    n: usize,
    replacement: bool,
) -> PyResult<Vec<usize>> {
    maximize_samples(&obj, num_samples, batch, n, replacement).into_py_result()
}

/// The scores of `ConstrainedMaxPosteriorSampling`.
#[pyfunction]
fn constrained_scores_py(
    y: Vec<f64>,
    c: Vec<f64>,
    rows: usize,
    n: usize,
    k: usize,
) -> PyResult<Vec<f64>> {
    constrained_scores(&y, &c, rows, n, k).into_py_result()
}

/// The Boltzmann weights of `boltzmann_sample` and the final temperature.
#[pyfunction]
#[pyo3(signature = (values, eta, temp_decrease=0.5))]
fn boltzmann_weights_py(values: Vec<f64>, eta: f64, temp_decrease: f64) -> (Vec<f64>, f64) {
    boltzmann_weights(&values, eta, temp_decrease)
}

/// The weights of BoTorch's `ExpMAStoppingCriterion` (torch's float32
/// values).
#[pyfunction]
fn expma_weights_py(eta: f64, n_window: usize) -> Vec<f64> {
    expma_weights(eta, n_window)
}

/// `(step, fval, status, runtime, message)` of an optimization result.
type ResultOut = (usize, f64, u8, Option<f64>, Option<String>);

fn result_out(r: &OptimizationResult) -> ResultOut {
    (
        r.step,
        r.fval,
        r.status.value(),
        r.runtime,
        r.message.clone(),
    )
}

/// BoTorch's `fit_gpytorch_mll_torch` with the default closure and Adam on a
/// `SingleTaskGP`: `mask` selects the optimized raw parameters (flat order),
/// `bounds` overrides their bounds; `stopping` is `(maxiter, minimize,
/// n_window, eta, rel_tol)` or `None`.
#[pyfunction]
#[pyo3(signature = (model, mask=None, bounds=None, step_limit=None, stopping=None, timeout_sec=None, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=false, maximize=false))]
#[allow(clippy::too_many_arguments)]
fn fit_gpytorch_mll_torch_py(
    py: Python<'_>,
    model: PyRef<'_, PySingleTaskGP>,
    mask: Option<Vec<bool>>,
    bounds: Option<Vec<(f64, f64)>>,
    step_limit: Option<usize>,
    stopping: Option<(usize, bool, usize, f64, f64)>,
    timeout_sec: Option<f64>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    amsgrad: bool,
    maximize: bool,
) -> PyResult<ResultOut> {
    let options = TorchMinimizeOptions {
        adam: AdamOptions {
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            maximize,
        },
        step_limit,
        timeout_sec,
        stopping_criterion: stopping.map(|s| ExpMaStoppingCriterion::new(s.0, s.1, s.2, s.3, s.4)),
    };
    let mut m = (*model.snapshot()).clone();
    let res = threads::run(py, || {
        fit_gpytorch_mll_torch(&mut m, mask.as_deref(), bounds.as_deref(), &options, None)
    })
    .into_py_result()?;
    model.replace_model(m);
    Ok(result_out(&res))
}

/// BoTorch's `fit_gpytorch_mll_scipy` with the default closure on a
/// `SingleTaskGP` (one attempt; batched models element by element).
#[pyfunction]
#[pyo3(signature = (model, mask=None, bounds=None, timeout_sec=None, maxiter=None, maxfun=None, ftol=None, gtol=None, maxcor=None, maxls=None))]
#[allow(clippy::too_many_arguments)]
fn fit_gpytorch_mll_scipy_py(
    py: Python<'_>,
    model: PyRef<'_, PySingleTaskGP>,
    mask: Option<Vec<bool>>,
    bounds: Option<Vec<(f64, f64)>>,
    timeout_sec: Option<f64>,
    maxiter: Option<usize>,
    maxfun: Option<usize>,
    ftol: Option<f64>,
    gtol: Option<f64>,
    maxcor: Option<usize>,
    maxls: Option<usize>,
) -> PyResult<ResultOut> {
    let mut o = botorch_core::lbfgsb::LbfgsbOptions::default();
    o.maxiter = maxiter.unwrap_or(o.maxiter);
    o.maxfun = maxfun.unwrap_or(o.maxfun);
    o.ftol = ftol.unwrap_or(o.ftol);
    o.gtol = gtol.unwrap_or(o.gtol);
    o.maxcor = maxcor.unwrap_or(o.maxcor);
    o.maxls = maxls.unwrap_or(o.maxls);
    let mut m = (*model.snapshot()).clone();
    let res = threads::run(py, || {
        fit_gpytorch_mll_scipy(&mut m, mask.as_deref(), bounds.as_deref(), &o, timeout_sec)
    })
    .into_py_result()?;
    model.replace_model(m);
    Ok(result_out(&res))
}

/// BoTorch's `sample_all_priors` on a `SingleTaskGP` (in place).
#[pyfunction]
#[pyo3(signature = (model, seed=None))]
fn sample_all_priors(model: PyRef<'_, PySingleTaskGP>, seed: Option<u64>) -> PyResult<()> {
    let mut m = (*model.snapshot()).clone();
    botorch_core::fit::sample_all_priors_batched(
        &mut m,
        seed.unwrap_or_else(botorch_core::utils::sampling::random_seed),
    )
    .into_py_result()?;
    model.replace_model(m);
    Ok(())
}

/// Registers the functions and classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMixedProblem>()?;
    m.add_function(wrap_pyfunction!(optimize_acqf_mixed_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_acqf_discrete_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_acqf_discrete_local_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(gen_lattice_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_acqf_cyclic_py, m)?)?;
    m.add_function(wrap_pyfunction!(round_discrete_dims_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_nearest_neighbors_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_categorical_neighbors_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_spray_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(prune_candidates_py, m)?)?;
    m.add_function(wrap_pyfunction!(gen_candidates_torch_py, m)?)?;
    m.add_function(wrap_pyfunction!(gen_candidates_scipy_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        sample_truncated_normal_perturbations_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(truncated_normal_perturbations_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_perturbed_subset_dims_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_points_around_py, m)?)?;
    m.add_function(wrap_pyfunction!(maximize_samples_py, m)?)?;
    m.add_function(wrap_pyfunction!(constrained_scores_py, m)?)?;
    m.add_function(wrap_pyfunction!(boltzmann_weights_py, m)?)?;
    m.add_function(wrap_pyfunction!(expma_weights_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gpytorch_mll_torch_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gpytorch_mll_scipy_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_all_priors, m)?)?;
    Ok(())
}
