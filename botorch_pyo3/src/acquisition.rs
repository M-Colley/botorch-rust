//! Acquisition functions (`botorch.acquisition`) and `prune_inferior_points`.
//!
//! One Python class wraps every acquisition function of `botorch_core`; the
//! `botorch_rs.acquisition` package provides the BoTorch-named classes on top
//! of it. An acquisition function shares the model snapshot that is current
//! at construction (see [`crate::models`]).

use std::sync::Arc;

use botorch_core::acquisition::{
    prune_inferior_points, AcquisitionFunction, ExpectedImprovement, LogExpectedImprovement,
    LogProbabilityOfImprovement, McAcquisitionOptions, PosteriorMean, PosteriorStandardDeviation,
    ProbabilityOfImprovement, PruneOptions, QExpectedImprovement, QLogExpectedImprovement,
    QLogExpectedImprovementOptions, QLogNoisyExpectedImprovement,
    QLogNoisyExpectedImprovementOptions, QProbabilityOfImprovement, QSimpleRegret,
    QUpperConfidenceBound, UpperConfidenceBound,
};
use botorch_core::{Matrix, Model};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayOut};
use crate::errors::{unsupported, value_error, IntoPyResult};
use crate::models::PySingleTaskGP;
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// An acquisition function over candidate sets `X` (`batch x q x d`).
#[pyclass(
    frozen,
    name = "AcquisitionFunction",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyAcquisitionFunction {
    /// The acquisition function.
    pub inner: Arc<dyn AcquisitionFunction>,
    /// The same object when it is a qLogNEI (for its baseline accessors).
    nei: Option<Arc<QLogNoisyExpectedImprovement>>,
    name: &'static str,
}

impl PyAcquisitionFunction {
    fn new_plain(inner: Arc<dyn AcquisitionFunction>, name: &'static str) -> Self {
        Self {
            inner,
            nei: None,
            name,
        }
    }

    fn nei(&self) -> PyResult<&Arc<QLogNoisyExpectedImprovement>> {
        self.nei.as_ref().ok_or_else(|| {
            unsupported(format!(
                "{} has no baseline (only qLogNoisyExpectedImprovement does).",
                self.name
            ))
        })
    }
}

fn model_of(model: &PySingleTaskGP) -> Arc<dyn Model> {
    model.snapshot()
}

fn require(value: Option<f64>, name: &str, acqf: &str) -> PyResult<f64> {
    value.ok_or_else(|| value_error(format!("{acqf} requires `{name}`.")))
}

fn sampler_of(
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
) -> Option<Arc<botorch_core::SobolQMCNormalSampler>> {
    sampler.map(|s| Arc::clone(&s.inner))
}

/// Splits `X` (`... x q x d`) into candidate sets.
fn candidate_sets(x: &Array) -> PyResult<(Vec<usize>, Vec<Matrix>)> {
    x.batches("X")
}

/// The analytic acquisition functions (BoTorch class names).
const ANALYTIC: [&str; 7] = [
    "ExpectedImprovement",
    "LogExpectedImprovement",
    "ProbabilityOfImprovement",
    "LogProbabilityOfImprovement",
    "UpperConfidenceBound",
    "PosteriorMean",
    "PosteriorStandardDeviation",
];

/// The sample-reducing Monte-Carlo acquisition functions.
const MONTE_CARLO: [&str; 5] = [
    "qExpectedImprovement",
    "qProbabilityOfImprovement",
    "qLogExpectedImprovement",
    "qUpperConfidenceBound",
    "qSimpleRegret",
];

/// The `'static` name in `names` equal to `name`.
fn static_name(name: &str, names: &[&'static str]) -> PyResult<&'static str> {
    names
        .iter()
        .find(|n| **n == name)
        .copied()
        .ok_or_else(|| unsupported(format!("Unknown acquisition function {name:?}.")))
}

#[pymethods]
impl PyAcquisitionFunction {
    /// An analytic acquisition function (`q = 1`): `"ExpectedImprovement"`,
    /// `"LogExpectedImprovement"`, `"ProbabilityOfImprovement"`,
    /// `"LogProbabilityOfImprovement"` (with `best_f`),
    /// `"UpperConfidenceBound"` (with `beta`), `"PosteriorMean"` or
    /// `"PosteriorStandardDeviation"`.
    #[staticmethod]
    #[pyo3(signature = (name, model, best_f=None, beta=None, maximize=true))]
    fn analytic(
        name: &str,
        model: PyRef<'_, PySingleTaskGP>,
        best_f: Option<f64>,
        beta: Option<f64>,
        maximize: bool,
    ) -> PyResult<Self> {
        let name = static_name(name, &ANALYTIC)?;
        let m = model_of(&model);
        let best_f = || require(best_f, "best_f", name);
        let inner: Arc<dyn AcquisitionFunction> = match name {
            "ExpectedImprovement" => {
                Arc::new(ExpectedImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "LogExpectedImprovement" => {
                Arc::new(LogExpectedImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "ProbabilityOfImprovement" => {
                Arc::new(ProbabilityOfImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "LogProbabilityOfImprovement" => {
                Arc::new(LogProbabilityOfImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "UpperConfidenceBound" => Arc::new(
                UpperConfidenceBound::new(m, require(beta, "beta", name)?, maximize)
                    .into_py_result()?,
            ),
            "PosteriorMean" => Arc::new(PosteriorMean::new(m, maximize).into_py_result()?),
            _ => Arc::new(PosteriorStandardDeviation::new(m, maximize).into_py_result()?),
        };
        Ok(Self::new_plain(inner, name))
    }

    /// A Monte-Carlo acquisition function: `"qExpectedImprovement"`,
    /// `"qProbabilityOfImprovement"` (with `best_f`, `tau`),
    /// `"qLogExpectedImprovement"` (with `best_f`, `fat`, `tau_max`,
    /// `tau_relu`), `"qUpperConfidenceBound"` (with `beta`) or
    /// `"qSimpleRegret"`.
    #[staticmethod]
    #[pyo3(signature = (
        name,
        model,
        sampler=None,
        x_pending=None,
        best_f=None,
        beta=None,
        tau=1e-3,
        fat=true,
        tau_max=1e-2,
        tau_relu=1e-6,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn monte_carlo(
        name: &str,
        model: PyRef<'_, PySingleTaskGP>,
        sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
        x_pending: Option<Array>,
        best_f: Option<f64>,
        beta: Option<f64>,
        tau: f64,
        fat: bool,
        tau_max: f64,
        tau_relu: f64,
    ) -> PyResult<Self> {
        let name = static_name(name, &MONTE_CARLO)?;
        let m = model_of(&model);
        let sampler = sampler_of(sampler);
        let x_pending = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
        let options = McAcquisitionOptions {
            sampler: sampler.clone(),
            x_pending: x_pending.clone(),
        };
        let best_f = || require(best_f, "best_f", name);
        let inner: Arc<dyn AcquisitionFunction> = match name {
            "qExpectedImprovement" => {
                Arc::new(QExpectedImprovement::new(m, best_f()?, options).into_py_result()?)
            }
            "qProbabilityOfImprovement" => Arc::new(
                QProbabilityOfImprovement::new(m, best_f()?, tau, options).into_py_result()?,
            ),
            "qLogExpectedImprovement" => Arc::new(
                QLogExpectedImprovement::new(
                    m,
                    best_f()?,
                    QLogExpectedImprovementOptions {
                        sampler,
                        x_pending,
                        fat,
                        tau_max,
                        tau_relu,
                    },
                )
                .into_py_result()?,
            ),
            "qUpperConfidenceBound" => Arc::new(
                QUpperConfidenceBound::new(m, require(beta, "beta", name)?, options)
                    .into_py_result()?,
            ),
            _ => Arc::new(QSimpleRegret::new(m, options).into_py_result()?),
        };
        Ok(Self::new_plain(inner, name))
    }

    /// `qLogNoisyExpectedImprovement(model, X_baseline, ...)`; the baseline is
    /// pruned with `prune_inferior_points` (`prune_num_samples` samples,
    /// `prune_max_frac`, seeded by `prune_seed`) if `prune_baseline`.
    #[staticmethod]
    #[pyo3(signature = (
        model,
        x_baseline,
        sampler=None,
        x_pending=None,
        fat=true,
        prune_baseline=true,
        prune_num_samples=2048,
        prune_max_frac=1.0,
        prune_seed=None,
        tau_max=1e-2,
        tau_relu=1e-6,
        incremental=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn q_log_noisy_expected_improvement(
        py: Python<'_>,
        model: PyRef<'_, PySingleTaskGP>,
        x_baseline: Array,
        sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
        x_pending: Option<Array>,
        fat: bool,
        prune_baseline: bool,
        prune_num_samples: usize,
        prune_max_frac: f64,
        prune_seed: Option<u64>,
        tau_max: f64,
        tau_relu: f64,
        incremental: bool,
    ) -> PyResult<Self> {
        let m = model_of(&model);
        let x_baseline = x_baseline.matrix("X_baseline")?;
        let options = QLogNoisyExpectedImprovementOptions {
            sampler: sampler_of(sampler),
            x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
            fat,
            prune_baseline,
            prune_options: PruneOptions {
                num_samples: prune_num_samples,
                max_frac: prune_max_frac,
                seed: prune_seed,
                base_samples: None,
            },
            tau_max,
            tau_relu,
            incremental,
        };
        let acqf = threads::run(py, || {
            QLogNoisyExpectedImprovement::new(m, &x_baseline, options)
        })
        .into_py_result()?;
        let acqf = Arc::new(acqf);
        Ok(Self {
            inner: acqf.clone(),
            nei: Some(acqf),
            name: "qLogNoisyExpectedImprovement",
        })
    }

    /// The BoTorch class name.
    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

    /// Input dimension `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The required `q` (1 for analytic acquisition functions), or `None`.
    #[getter]
    fn expected_q(&self) -> Option<usize> {
        self.inner.expected_q()
    }

    /// Whether the values are non-negative (`initialize_q_batch_nonneg`).
    #[getter]
    fn is_nonnegative(&self) -> bool {
        self.inner.is_nonnegative()
    }

    /// Whether the values are on a log scale.
    #[getter]
    fn is_log(&self) -> bool {
        self.inner.is_log()
    }

    /// Values at the candidate sets `X` (`... x q x d`); shape `...`.
    fn forward(&self, py: Python<'_>, x: Array) -> PyResult<ArrayOut> {
        let (batch, sets) = candidate_sets(&x)?;
        let acqf = &self.inner;
        let values = threads::run(py, || acqf.forward_many(&sets)).into_py_result()?;
        Ok((values, batch))
    }

    /// Values (shape `...`) and gradients (shape of `X`) at the candidate
    /// sets `X` (`... x q x d`).
    fn forward_with_grad(&self, py: Python<'_>, x: Array) -> PyResult<(ArrayOut, ArrayOut)> {
        let (batch, sets) = candidate_sets(&x)?;
        let acqf = &self.inner;
        let results = threads::run(py, || {
            sets.par_iter()
                .map(|s| acqf.forward_with_grad(s))
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let mut values = Vec::with_capacity(results.len());
        let mut grads = Vec::with_capacity(x.0.len());
        for (v, g) in &results {
            values.push(*v);
            let (data, _) = matrix_out(g);
            grads.extend(data);
        }
        Ok(((values, batch), (grads, x.1.clone())))
    }

    /// The pending points (`m x d`), if any.
    fn x_pending(&self) -> Option<ArrayOut> {
        self.inner.x_pending().as_ref().map(matrix_out)
    }

    /// Sets (or with `None` clears) the pending points (`m x d`).
    #[pyo3(signature = (x_pending=None))]
    fn set_x_pending(&self, py: Python<'_>, x_pending: Option<Array>) -> PyResult<()> {
        let x_pending = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
        let acqf = &self.inner;
        threads::run(py, || acqf.set_x_pending(x_pending.as_ref())).into_py_result()
    }

    /// The (pruned) baseline of a qLogNEI (`n x d`).
    fn x_baseline(&self) -> PyResult<ArrayOut> {
        Ok(matrix_out(&self.nei()?.x_baseline()))
    }

    /// The incumbent points of a qLogNEI: the baseline and, in incremental
    /// mode, the pending points (BoTorch's `X_baseline` property).
    fn full_x_baseline(&self) -> PyResult<ArrayOut> {
        Ok(matrix_out(&self.nei()?.full_x_baseline()))
    }
}

/// `prune_inferior_points(model, X, num_samples, max_frac)`: the points of
/// `X` (`n x d`) with a non-zero estimated probability of being the best.
#[pyfunction]
#[pyo3(
    name = "prune_inferior_points",
    signature = (model, x, num_samples=2048, max_frac=1.0, seed=None, base_samples=None)
)]
pub fn prune_inferior_points_py(
    py: Python<'_>,
    model: PyRef<'_, PySingleTaskGP>,
    x: Array,
    num_samples: usize,
    max_frac: f64,
    seed: Option<u64>,
    base_samples: Option<Array>,
) -> PyResult<ArrayOut> {
    let m = model.snapshot();
    let x = x.matrix("X")?;
    let options = PruneOptions {
        num_samples,
        max_frac,
        seed,
        base_samples: base_samples.map(|z| z.matrix("base_samples")).transpose()?,
    };
    let pruned =
        threads::run(py, || prune_inferior_points(m.as_ref(), &x, &options)).into_py_result()?;
    Ok(matrix_out(&pruned))
}
