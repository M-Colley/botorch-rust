//! Information-theoretic multi-objective acquisition functions
//! (`botorch.acquisition.multi_objective.{max_value_entropy_search,
//! joint_entropy_search, predictive_entropy_search}`) and their utilities
//! (`compute_sample_box_decomposition`, `random_search_optimizer`,
//! `sample_optimal_points` of `botorch.acquisition.multi_objective.utils`).
//!
//! The acquisition functions are returned as [`PyAcquisitionFunction`]s
//! (values and exact gradients, `optimize_acqf`); PES also returns a
//! [`PyPesState`] handle on the result of EP. Hypercell bounds arrive as
//! BoTorch's `num_pareto_samples x 2 x J x M` arrays, Pareto sets and fronts
//! as `num_pareto_samples x num_pareto_points x (d | M)` arrays.

use std::sync::Arc;

use botorch_core::acquisition::multi_objective::{
    compute_entropy_monte_carlo, compute_entropy_noiseless, compute_entropy_upper_bound,
    compute_sample_box_decomposition, log_cdf_robust, random_search_optimizer,
    sample_optimal_point_paths, sample_optimal_points, EntropyStatistics, EstimationType,
    LowerBoundMultiObjectiveEntropySearchOptions, QLowerBoundMultiObjectiveJointEntropySearch,
    QLowerBoundMultiObjectiveMaxValueEntropySearch, QMultiObjectivePredictiveEntropySearch,
    QMultiObjectivePredictiveEntropySearchOptions, RandomSearchOptions, SampleOptimalPointsOptions,
    SamplePartitioning,
};
use botorch_core::sampling::pathwise::{PathList, SamplePath};
use botorch_core::utils::multi_objective::box_decompositions::CellBounds;
use botorch_core::{BotorchError, Matrix, Result};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::acquisition::{model_of, sampler_of, PyAcquisitionFunction};
use crate::arrays::{Array, ArrayBuf};
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::models::PyModel;
use crate::pathwise::{draws_of, DrawsArg, PySamplePaths};
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// Hypercell bounds from BoTorch's `num_pareto_samples x 2 x J x M` array.
fn cells_of(cells: &Array) -> PyResult<CellBounds> {
    let (p, j, m) = match cells.1.as_slice() {
        [p, 2, j, m] => (*p, *j, *m),
        s => {
            return Err(dimension_error(format!(
                "The hypercell_bounds should have a shape of `num_pareto_samples x 2 x \
                 num_boxes x num_objectives`, got {s:?}."
            )))
        }
    };
    let block = j * m;
    if cells.0.len() != p * 2 * block {
        return Err(dimension_error("hypercell_bounds: inconsistent values"));
    }
    let mut lower = Vec::with_capacity(p * block);
    let mut upper = Vec::with_capacity(p * block);
    for b in 0..p {
        let base = b * 2 * block;
        lower.extend_from_slice(&cells.0[base..base + block]);
        upper.extend_from_slice(&cells.0[base + block..base + 2 * block]);
    }
    CellBounds::new(vec![p], j, m, lower, upper).into_py_result()
}

/// Hypercell bounds as BoTorch's `num_pareto_samples x 2 x J x M` array.
fn cells_out(cells: &CellBounds) -> ArrayBuf {
    let (p, j, m) = (cells.batch_size(), cells.num_cells(), cells.num_outcomes());
    let mut data = Vec::with_capacity(p * 2 * j * m);
    for b in 0..p {
        data.extend_from_slice(cells.lower_of(b));
        data.extend_from_slice(cells.upper_of(b));
    }
    ArrayBuf(data, vec![p, 2, j, m])
}

/// A `num_pareto_samples x n x k` array as `num_pareto_samples` matrices.
fn sample_matrices(a: &Array, name: &str) -> PyResult<Vec<Matrix>> {
    if a.1.len() != 3 {
        return Err(dimension_error(format!(
            "`{name}` must have shape `num_pareto_samples x num_pareto_points x k`, got {:?}.",
            a.1
        )));
    }
    Ok(a.batches(name)?.1)
}

/// Matrices of identical shape stacked into a `len x r x c` array.
fn stacked(mats: &[Matrix], r: usize, c: usize) -> ArrayBuf {
    let mut data = Vec::with_capacity(mats.len() * r * c);
    for m in mats {
        for i in 0..r {
            data.extend(m.row(i).iter());
        }
    }
    ArrayBuf(data, vec![mats.len(), r, c])
}

fn estimation_type_of(name: &str) -> PyResult<EstimationType> {
    EstimationType::from_name(name).into_py_result()
}

/// `compute_sample_box_decomposition(pareto_fronts, partitioning, maximize,
/// num_constraints)`: the hypercell bounds (`num_pareto_samples x 2 x J x (M
/// + K)`) of the sampled fronts (`num_pareto_samples x num_pareto_points x
/// M`); `partitioning` is `"dominated"` (BoTorch's default
/// `DominatedPartitioning`), `"fast_nondominated"` or `"nondominated"` (with
/// `alpha`).
#[pyfunction]
#[pyo3(name = "compute_sample_box_decomposition", signature = (pareto_fronts, partitioning="dominated", alpha=0.0, maximize=true, num_constraints=0))]
pub fn compute_sample_box_decomposition_py(
    py: Python<'_>,
    pareto_fronts: Array,
    partitioning: &str,
    alpha: f64,
    maximize: bool,
    num_constraints: usize,
) -> PyResult<ArrayBuf> {
    let fronts = sample_matrices(&pareto_fronts, "pareto_fronts")?;
    let partitioning = match partitioning {
        "dominated" => SamplePartitioning::Dominated,
        "fast_nondominated" => SamplePartitioning::FastNondominated,
        "nondominated" => SamplePartitioning::Nondominated(alpha),
        other => return Err(value_error(format!("unknown partitioning {other:?}"))),
    };
    let cells = threads::run(py, || {
        compute_sample_box_decomposition(&fronts, partitioning, maximize, num_constraints)
    })
    .into_py_result()?;
    Ok(cells_out(&cells))
}

/// Row-major values of a matrix.
fn rows(m: &Matrix) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len());
    for i in 0..m.nrows() {
        out.extend(m.row(i).iter());
    }
    out
}

/// The deterministic function optimized by `random_search_optimizer`: the
/// path values of sample `sample` of a [`PySamplePaths`] (`n x m`), or a
/// Python callable taking and returning `(data, shape)` (`n x d` to `n x
/// M`).
enum Objective {
    Paths(Arc<PathList>, usize),
    Callable(Py<PyAny>),
}

impl Objective {
    fn eval(&self, x: &Matrix) -> Result<Matrix> {
        match self {
            Objective::Paths(paths, sample) => {
                let values = paths.forward(x)?;
                Ok(Matrix::from_fn(x.nrows(), values.len(), |i, t| {
                    values[t][(*sample, i)]
                }))
            }
            Objective::Callable(f) => {
                let (n, d) = x.shape();
                let Array(data, shape) = Python::attach(|py| -> PyResult<Array> {
                    f.bind(py).call1(((rows(x), vec![n, d]),))?.extract()
                })
                .map_err(crate::callbacks::callback_error)?;
                match shape.as_slice() {
                    [r, m] if *r == n && data.len() == n * m => {
                        Ok(Matrix::from_row_slice(n, *m, &data))
                    }
                    s => Err(BotorchError::DimensionMismatch(format!(
                        "the model returned values of shape {s:?} for {n} points (expected `{n} \
                         x M`)"
                    ))),
                }
            }
        }
    }
}

/// `random_search_optimizer(model, bounds, num_points, maximize, pop_size,
/// max_tries)` of a path model (sample `sample` of `paths`) or of a Python
/// callable; the candidate sets are `candidates` (one `pop_size x d` array
/// per try) or Sobol points drawn from `seed`. Returns the optimal inputs
/// (`num_points x d`) and outputs (`num_points x M`).
#[pyfunction]
#[pyo3(name = "random_search_optimizer", signature = (model, bounds, num_points, maximize, pop_size=1024, max_tries=10, seed=None, candidates=None, sample=0))]
#[allow(clippy::too_many_arguments)]
pub fn random_search_optimizer_py(
    py: Python<'_>,
    model: Bound<'_, PyAny>,
    bounds: Array,
    num_points: usize,
    maximize: bool,
    pop_size: usize,
    max_tries: usize,
    seed: Option<u64>,
    candidates: Option<Vec<Array>>,
    sample: usize,
) -> PyResult<(ArrayBuf, ArrayBuf)> {
    let objective = match model.cast::<PySamplePaths>() {
        Ok(paths) => {
            let paths = paths.get();
            if sample >= paths.paths.num_samples() {
                return Err(value_error(format!(
                    "sample {sample} out of range for {} sample paths",
                    paths.paths.num_samples()
                )));
            }
            Objective::Paths(Arc::clone(&paths.paths), sample)
        }
        Err(_) => Objective::Callable(model.unbind()),
    };
    let bounds = bounds.matrix("bounds")?;
    let options = RandomSearchOptions {
        pop_size,
        max_tries,
        seed,
        candidates: candidates
            .map(|cs| {
                cs.iter()
                    .map(|c| c.matrix("candidates"))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?,
    };
    let (x, y) = threads::run(py, || {
        random_search_optimizer(
            &|x: &Matrix| objective.eval(x),
            &bounds,
            num_points,
            maximize,
            &options,
        )
    })
    .into_py_result()?;
    let (n, d, m) = (x.nrows(), x.ncols(), y.ncols());
    Ok((
        ArrayBuf(rows(&x), vec![n, d]),
        ArrayBuf(rows(&y), vec![n, m]),
    ))
}

type SampleDrawsArg = Option<Vec<Vec<DrawsArg>>>;

fn sample_options(
    pop_size: usize,
    max_tries: usize,
    num_features: usize,
    seed: Option<u64>,
    draws: SampleDrawsArg,
    candidates: Option<Vec<Vec<Array>>>,
) -> PyResult<SampleOptimalPointsOptions> {
    Ok(SampleOptimalPointsOptions {
        random_search: RandomSearchOptions {
            pop_size,
            max_tries,
            ..Default::default()
        },
        num_features,
        seed,
        draws: draws
            .map(|ds| ds.into_iter().map(draws_of).collect::<PyResult<Vec<_>>>())
            .transpose()?,
        candidates: candidates
            .map(|cs| {
                cs.iter()
                    .map(|c| {
                        c.iter()
                            .map(|a| a.matrix("candidates"))
                            .collect::<PyResult<Vec<_>>>()
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?,
    })
}

/// `sample_optimal_points(model, bounds, num_samples, num_points,
/// random_search_optimizer, maximize, {pop_size, max_tries})`: Pareto sets
/// (`num_samples x num_points x d`) and fronts (`num_samples x num_points x
/// M`) of Matheron paths with `num_features` features, from `draws` (one
/// list of per-output draws per sample) and `candidates` (one list of
/// candidate sets per sample) or from `seed`.
#[pyfunction]
#[pyo3(name = "sample_optimal_points", signature = (model, bounds, num_samples, num_points, maximize=true, pop_size=1024, max_tries=10, num_features=1024, seed=None, draws=None, candidates=None))]
#[allow(clippy::too_many_arguments)]
pub fn sample_optimal_points_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    bounds: Array,
    num_samples: usize,
    num_points: usize,
    maximize: bool,
    pop_size: usize,
    max_tries: usize,
    num_features: usize,
    seed: Option<u64>,
    draws: SampleDrawsArg,
    candidates: Option<Vec<Vec<Array>>>,
) -> PyResult<(ArrayBuf, ArrayBuf)> {
    let m = model_of(&model);
    let bounds = bounds.matrix("bounds")?;
    let options = sample_options(pop_size, max_tries, num_features, seed, draws, candidates)?;
    let (ps, pf) = threads::run(py, || {
        sample_optimal_points(
            m.as_ref(),
            &bounds,
            num_samples,
            num_points,
            maximize,
            &options,
        )
    })
    .into_py_result()?;
    let (d, mo) = (bounds.ncols(), m.num_outputs());
    Ok((stacked(&ps, num_points, d), stacked(&pf, num_points, mo)))
}

/// The Matheron path models of `sample_optimal_points` with the same
/// arguments (one handle of one sample path per Pareto sample), for
/// optimizers written in Python.
#[pyfunction]
#[pyo3(name = "sample_optimal_point_paths", signature = (model, num_samples, num_features=1024, seed=None, draws=None))]
pub fn sample_optimal_point_paths_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    num_samples: usize,
    num_features: usize,
    seed: Option<u64>,
    draws: SampleDrawsArg,
) -> PyResult<Vec<PySamplePaths>> {
    let m = model_of(&model);
    let options = sample_options(1024, 10, num_features, seed, draws, None)?;
    let models = threads::run(py, || {
        sample_optimal_point_paths(m.as_ref(), num_samples, &options)
    })
    .into_py_result()?;
    Ok(models
        .into_iter()
        .map(PySamplePaths::from_path_model)
        .collect())
}

fn entropy_options(
    x_pending: Option<Array>,
    estimation_type: &str,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
) -> PyResult<LowerBoundMultiObjectiveEntropySearchOptions> {
    Ok(LowerBoundMultiObjectiveEntropySearchOptions {
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        estimation_type: estimation_type_of(estimation_type)?,
        sampler: sampler_of(sampler),
    })
}

/// `qLowerBoundMultiObjectiveMaxValueEntropySearch(model, hypercell_bounds,
/// X_pending, estimation_type)` with the MC sampler `sampler` (`"MC"`).
#[pyfunction]
#[pyo3(signature = (model, hypercell_bounds, x_pending=None, estimation_type="LB", sampler=None))]
pub fn q_lower_bound_mo_max_value_entropy_search(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    hypercell_bounds: Array,
    x_pending: Option<Array>,
    estimation_type: &str,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let cells = cells_of(&hypercell_bounds)?;
    let options = entropy_options(x_pending, estimation_type, sampler)?;
    let a = threads::run(py, || {
        QLowerBoundMultiObjectiveMaxValueEntropySearch::new(m, cells, options)
    })
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "qLowerBoundMultiObjectiveMaxValueEntropySearch",
    ))
}

/// `qLowerBoundMultiObjectiveJointEntropySearch(model, pareto_sets,
/// pareto_fronts, hypercell_bounds, X_pending, estimation_type)` with the MC
/// sampler `sampler` (`"MC"`).
#[pyfunction]
#[pyo3(signature = (model, pareto_sets, pareto_fronts, hypercell_bounds, x_pending=None, estimation_type="LB", sampler=None))]
#[allow(clippy::too_many_arguments)]
pub fn q_lower_bound_mo_joint_entropy_search(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    pareto_sets: Array,
    pareto_fronts: Array,
    hypercell_bounds: Array,
    x_pending: Option<Array>,
    estimation_type: &str,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let sets = sample_matrices(&pareto_sets, "pareto_sets")?;
    let fronts = sample_matrices(&pareto_fronts, "pareto_fronts")?;
    let cells = cells_of(&hypercell_bounds)?;
    let options = entropy_options(x_pending, estimation_type, sampler)?;
    let a = threads::run(py, || {
        QLowerBoundMultiObjectiveJointEntropySearch::new(m, sets, fronts, cells, options)
    })
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "qLowerBoundMultiObjectiveJointEntropySearch",
    ))
}

/// The result of EP of a qMultiObjectivePredictiveEntropySearch.
#[pyclass(frozen, name = "PesState", module = "botorch_rs._botorch_rs")]
pub struct PyPesState {
    inner: Arc<QMultiObjectivePredictiveEntropySearch>,
    num_outputs: usize,
}

#[pymethods]
impl PyPesState {
    /// The number of EP iterations.
    #[getter]
    fn ep_iterations(&self) -> usize {
        self.inner.ep_summary().iterations
    }

    /// The final damping factors (`num_pareto_samples x M`; zero once
    /// converged).
    fn ep_damping(&self) -> ArrayBuf {
        let d = &self.inner.ep_summary().damping;
        ArrayBuf(
            d.clone(),
            vec![d.len() / self.num_outputs, self.num_outputs],
        )
    }

    /// The cached EP factors of the Pareto rows: natural means
    /// (`num_pareto_samples x M x P x P x 2`) and covariances (`... x 2 x 2`),
    /// BoTorch's `_omega_f_nat_mean[..., N:, :, :]` and
    /// `_omega_f_nat_cov[..., N:, :, :, :]`.
    fn cached_factors(&self) -> (ArrayBuf, ArrayBuf) {
        let factors = self.inner.cached_factors();
        let (r, p) = factors
            .first()
            .map_or((0, 0), |f| (f.rows(), f.num_pareto_points()));
        let s = factors.len() / self.num_outputs;
        let m = self.num_outputs;
        let means = factors
            .iter()
            .flat_map(|f| f.means().iter().copied())
            .collect();
        let covs = factors
            .iter()
            .flat_map(|f| f.covs().iter().copied())
            .collect();
        (
            ArrayBuf(means, vec![s, m, r, p, 2]),
            ArrayBuf(covs, vec![s, m, r, p, 2, 2]),
        )
    }
}

/// `qMultiObjectivePredictiveEntropySearch(model, pareto_sets, maximize,
/// X_pending, max_ep_iterations, ep_jitter, test_jitter, threshold)`: runs
/// EP and returns the acquisition function and the EP state.
#[pyfunction]
#[pyo3(signature = (model, pareto_sets, maximize=true, x_pending=None, max_ep_iterations=250, ep_jitter=1e-4, test_jitter=1e-4, threshold=1e-2))]
#[allow(clippy::too_many_arguments)]
pub fn q_mo_predictive_entropy_search(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    pareto_sets: Array,
    maximize: bool,
    x_pending: Option<Array>,
    max_ep_iterations: usize,
    ep_jitter: f64,
    test_jitter: f64,
    threshold: f64,
) -> PyResult<(PyAcquisitionFunction, PyPesState)> {
    let m = model_of(&model);
    let num_outputs = m.num_outputs();
    let sets = sample_matrices(&pareto_sets, "pareto_sets")?;
    let options = QMultiObjectivePredictiveEntropySearchOptions {
        maximize,
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        max_ep_iterations,
        ep_jitter,
        test_jitter,
        threshold,
    };
    let a = threads::run(py, || {
        QMultiObjectivePredictiveEntropySearch::new(m, sets, options)
    })
    .into_py_result()?;
    let a = Arc::new(a);
    Ok((
        PyAcquisitionFunction::new_plain(
            Arc::clone(&a) as Arc<dyn botorch_core::AcquisitionFunction>,
            "qMultiObjectivePredictiveEntropySearch",
        ),
        PyPesState {
            inner: a,
            num_outputs,
        },
    ))
}

/// The statistics `batch_shape x num_pareto_samples x q x 1 x M` of the
/// entropy estimators as one `P x M` matrix per `(batch, q)` element.
fn statistics(
    a: &Array,
    name: &str,
    p: usize,
    m: usize,
) -> PyResult<(Vec<usize>, usize, Vec<Matrix>)> {
    let shape = &a.1;
    let k = shape.len();
    if k < 4 || shape[k - 1] != m || shape[k - 2] != 1 || shape[k - 4] != p {
        return Err(dimension_error(format!(
            "`{name}` must have shape `batch_shape x {p} x q x 1 x {m}`, got {shape:?}."
        )));
    }
    let batch = shape[..k - 4].to_vec();
    let q = shape[k - 3];
    let nb: usize = batch.iter().product();
    if a.0.len() != nb * p * q * m {
        return Err(dimension_error(format!("`{name}`: inconsistent values")));
    }
    let mats = (0..nb * q)
        .map(|e| {
            let (b, i) = (e / q, e % q);
            Matrix::from_fn(p, m, |kk, j| a.0[((b * p + kk) * q + i) * m + j])
        })
        .collect();
    Ok((batch, q, mats))
}

/// BoTorch's `_compute_entropy_noiseless` (`"0"`),
/// `_compute_entropy_upper_bound` (`"LB"`, `"LB2"`: `only_diagonal`) or
/// `_compute_entropy_monte_carlo` (`"MC"`, with `samples`, `num_mc_samples x
/// batch_shape x P x q x 1 x M`, and `samples_log_prob`, `num_mc_samples x
/// batch_shape x P x q`): the entropy estimates (`batch_shape x q`).
#[pyfunction]
#[pyo3(signature = (estimation_type, hypercell_bounds, mean, variance, observation_noise, samples=None, samples_log_prob=None))]
#[allow(clippy::too_many_arguments)]
pub fn compute_entropy(
    py: Python<'_>,
    estimation_type: &str,
    hypercell_bounds: Array,
    mean: Array,
    variance: Array,
    observation_noise: Array,
    samples: Option<Array>,
    samples_log_prob: Option<Array>,
) -> PyResult<ArrayBuf> {
    let et = estimation_type_of(estimation_type)?;
    let cells = cells_of(&hypercell_bounds)?;
    let (p, m) = (cells.batch_size(), cells.num_outcomes());
    let (batch, q, means) = statistics(&mean, "mean", p, m)?;
    let (_, _, variances) = statistics(&variance, "variance", p, m)?;
    let (_, _, noises) = statistics(&observation_noise, "observation_noise", p, m)?;
    if variance.1 != mean.1 || observation_noise.1 != mean.1 {
        return Err(dimension_error(
            "mean, variance and observation_noise must have the same shape",
        ));
    }
    let ne = means.len();
    // The MC samples and log-densities of every element: `P` matrices of
    // `S x M` and an `S x P` matrix.
    let mc = if et == EstimationType::MonteCarlo {
        let (Some(s), Some(lp)) = (&samples, &samples_log_prob) else {
            return Err(value_error(
                "the \"MC\" estimate requires samples and samples_log_prob",
            ));
        };
        let ns = s.1.first().copied().unwrap_or(0);
        let mut expect = vec![ns];
        expect.extend_from_slice(&mean.1);
        let mut expect_lp = vec![ns];
        expect_lp.extend_from_slice(&mean.1[..mean.1.len() - 2]);
        if s.1 != expect || lp.1 != expect_lp || s.0.len() != ns * ne * p * m {
            return Err(dimension_error(format!(
                "samples must have shape {expect:?} and samples_log_prob {expect_lp:?}"
            )));
        }
        // Element e = b·q + i; sample index (s, b, k, i, j).
        Some(
            (0..ne)
                .map(|e| {
                    let (b, i) = (e / q, e % q);
                    let nb = ne / q;
                    let sm: Vec<Matrix> = (0..p)
                        .map(|k| {
                            Matrix::from_fn(ns, m, |r, j| {
                                s.0[(((r * nb + b) * p + k) * q + i) * m + j]
                            })
                        })
                        .collect();
                    let lpm = Matrix::from_fn(ns, p, |r, k| lp.0[((r * nb + b) * p + k) * q + i]);
                    (sm, lpm)
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let values = threads::run(py, || {
        (0..ne)
            .into_par_iter()
            .map(|e| {
                let stats = EntropyStatistics {
                    mean: means[e].clone(),
                    variance: variances[e].clone(),
                    observation_noise: noises[e].clone(),
                };
                match et {
                    EstimationType::Noiseless => compute_entropy_noiseless(&cells, &stats),
                    EstimationType::LowerBound => {
                        compute_entropy_upper_bound(&cells, &stats, false)
                    }
                    EstimationType::LowerBound2 => {
                        compute_entropy_upper_bound(&cells, &stats, true)
                    }
                    EstimationType::MonteCarlo => {
                        let (s, lp) = &mc.as_ref().expect("checked above")[e];
                        compute_entropy_monte_carlo(&cells, &stats, s, lp)
                    }
                }
            })
            .collect::<Result<Vec<_>>>()
    })
    .into_py_result()?;
    let mut shape = batch;
    shape.push(q);
    Ok(ArrayBuf(values, shape))
}

/// BoTorch's `log_cdf_robust` elementwise (same shape).
#[pyfunction]
#[pyo3(name = "log_cdf_robust")]
pub fn log_cdf_robust_py(x: Array) -> ArrayBuf {
    let Array(data, shape) = x;
    ArrayBuf(data.into_iter().map(log_cdf_robust).collect(), shape)
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPesState>()?;
    m.add_function(wrap_pyfunction!(compute_sample_box_decomposition_py, m)?)?;
    m.add_function(wrap_pyfunction!(random_search_optimizer_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_optimal_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_optimal_point_paths_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        q_lower_bound_mo_max_value_entropy_search,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(q_lower_bound_mo_joint_entropy_search, m)?)?;
    m.add_function(wrap_pyfunction!(q_mo_predictive_entropy_search, m)?)?;
    m.add_function(wrap_pyfunction!(compute_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(log_cdf_robust_py, m)?)?;
    Ok(())
}
