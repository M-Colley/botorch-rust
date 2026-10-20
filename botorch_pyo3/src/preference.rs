//! Preference learning (phase 10b): `PairwiseGP` with its Laplace
//! approximation and marginal log likelihood, and the pairwise
//! likelihoods.
//!
//! Comparisons arrive as float arrays of shape `batch x m x 2` (indices);
//! datapoints as `batch x n x d`. A model with batched data is a
//! [`BatchedPairwiseGP`]; posteriors and acquisition functions use the
//! generic model handle ([`PyModel`]).

use std::sync::{Arc, RwLock};

use botorch_core::acquisition::preference;
use botorch_core::fit::FitOptions;
use botorch_core::kernels::CovarModule;
use botorch_core::models::likelihoods::{Comparisons, PairwiseLikelihood};
use botorch_core::models::pairwise_gp::{
    initial_raw_parameters, pairwise_default_covar_module, LaplaceState,
};
use botorch_core::models::{BatchedPairwiseGP, Model, PairwiseGP, PairwiseGPOptions};
use botorch_core::transforms::input::ChainedInputTransform;
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::acquisition::{objective_of, PyAcquisitionFunction};
use crate::arrays::{broadcast_indices, broadcast_shapes, matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::PyModel;
use crate::sampling::PySobolQMCNormalSampler;
use crate::specs::PyKernelSpec;
use crate::threads;

/// `(loss, iterations, evaluations, status, message, attempts, elements)`.
type FitSummary = (
    f64,
    usize,
    usize,
    &'static str,
    String,
    usize,
    Vec<(f64, usize, usize, &'static str, String, usize)>,
);

fn likelihood_of(name: &str) -> PyResult<PairwiseLikelihood> {
    match name {
        "probit" => Ok(PairwiseLikelihood::Probit),
        "logit" => Ok(PairwiseLikelihood::Logit),
        other => Err(value_error(format!(
            "unknown pairwise likelihood {other:?}"
        ))),
    }
}

/// Comparisons from a float array of indices (`... x m x 2`), per batch
/// element.
fn comparisons_of(c: &Array, n: usize, batch: usize) -> PyResult<Vec<Vec<(usize, usize)>>> {
    if c.1.last() != Some(&2) || c.1.len() < 2 {
        return Err(dimension_error(format!(
            "comparisons must have shape `batch_shape x m x 2`, got {:?}",
            c.1
        )));
    }
    let m = c.1[c.1.len() - 2];
    let total: usize = c.1[..c.1.len() - 2].iter().product();
    if total != batch {
        return Err(dimension_error(format!(
            "comparisons have {total} batch elements, the datapoints {batch}"
        )));
    }
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut v = Vec::with_capacity(m);
        for k in 0..m {
            let mut pair = [0usize; 2];
            for (j, p) in pair.iter_mut().enumerate() {
                let x = c.0[(b * m + k) * 2 + j];
                // torch's `.long()` truncates; negative indices are out of
                // range for `scatter_`.
                let i = x.trunc();
                if !(i >= 0.0 && (i as usize) < n) {
                    return Err(crate::errors::to_pyerr(
                        botorch_core::BotorchError::Runtime(format!(
                            "index {} is out of bounds for dimension 1 with size {n}",
                            i as i64
                        )),
                    ));
                }
                *p = i as usize;
            }
            v.push((pair[0], pair[1]));
        }
        out.push(v);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum Inner {
    Single(PairwiseGP),
    Batched(BatchedPairwiseGP),
}

impl Inner {
    fn elements(&self) -> Vec<&PairwiseGP> {
        match self {
            Inner::Single(m) => vec![m],
            Inner::Batched(b) => b.elements().iter().collect(),
        }
    }

    fn first(&self) -> &PairwiseGP {
        match self {
            Inner::Single(m) => m,
            Inner::Batched(b) => &b.elements()[0],
        }
    }

    fn batch_shape(&self) -> Vec<usize> {
        match self {
            Inner::Single(_) => Vec::new(),
            Inner::Batched(b) => b.batch_shape().to_vec(),
        }
    }

    fn model(&self) -> Arc<dyn Model> {
        match self {
            Inner::Single(m) => Arc::new(m.clone()),
            Inner::Batched(b) => Arc::new(b.clone()),
        }
    }
}

/// BoTorch's `PairwiseGP` (see `botorch_core::models::pairwise_gp`).
#[pyclass(frozen, name = "PairwiseGP", module = "botorch_rs._botorch_rs")]
pub struct PyPairwiseGP {
    inner: RwLock<Inner>,
}

impl PyPairwiseGP {
    fn get(&self) -> Inner {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set(&self, inner: Inner) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = inner;
    }

    fn with<T>(&self, f: impl FnOnce(&Inner) -> T) -> T {
        f(&self.inner.read().unwrap_or_else(|e| e.into_inner()))
    }

    fn state_of(&self, what: &str) -> PyResult<Vec<LaplaceState>> {
        self.with(|i| {
            i.elements()
                .iter()
                .map(|e| {
                    e.state()
                        .cloned()
                        .ok_or_else(|| value_error(format!("the model has no data (no `{what}`)")))
                })
                .collect()
        })
    }

    fn stacked<F>(&self, what: &str, f: F) -> PyResult<ArrayOut>
    where
        F: Fn(&LaplaceState) -> Matrix,
    {
        let states = self.state_of(what)?;
        let batch = self.with(Inner::batch_shape);
        let mats: Vec<Matrix> = states.iter().map(f).collect();
        let (r, c) = mats[0].shape();
        Ok(crate::arrays::stack_out(&mats, &batch, r, c))
    }
}

fn vec_out(batch: &[usize], vecs: &[Vec<f64>]) -> ArrayOut {
    let mut shape = batch.to_vec();
    shape.push(vecs.first().map_or(0, Vec::len));
    (vecs.iter().flatten().copied().collect(), shape)
}

#[allow(clippy::too_many_arguments)]
fn fit_options(
    maxcor: Option<usize>,
    pgtol: Option<f64>,
    ftol: Option<f64>,
    maxiter: Option<usize>,
    maxfun: Option<usize>,
    maxls: Option<usize>,
    max_attempts: Option<usize>,
    seed: Option<u64>,
) -> PyResult<FitOptions> {
    let d = FitOptions::default();
    let options = FitOptions {
        maxcor: maxcor.unwrap_or(d.maxcor),
        pgtol: pgtol.unwrap_or(d.pgtol),
        ftol: ftol.unwrap_or(d.ftol),
        maxiter: maxiter.unwrap_or(d.maxiter),
        maxfun: maxfun.unwrap_or(d.maxfun),
        maxls: maxls.unwrap_or(d.maxls),
        max_attempts: max_attempts.unwrap_or(d.max_attempts),
        seed: seed.unwrap_or(d.seed),
    };
    if !(options.pgtol >= 0.0 && options.ftol >= 0.0) {
        return Err(value_error("gtol and ftol must be non-negative"));
    }
    Ok(options)
}

fn status_name(s: botorch_core::FitStatus) -> &'static str {
    match s {
        botorch_core::FitStatus::Success => "SUCCESS",
        botorch_core::FitStatus::Stopped => "STOPPED",
        botorch_core::FitStatus::Failure => "FAILURE",
    }
}

#[pymethods]
impl PyPairwiseGP {
    /// A model from `datapoints` (`batch_shape x n x d`) and `comparisons`
    /// (`batch_shape x m x 2`), or a prior-only model of dimension `dim`
    /// without them. `covar_spec` / `covar_tree` give the covariance module
    /// (BoTorch's default if neither); `raw` its raw parameters; for batched
    /// data `param_map` maps every element's raw parameters to `raw`.
    /// `draws`: the standard normal perturbations of the initial utility
    /// (`B·n` values after consolidation; see `needs_draws`).
    #[new]
    #[pyo3(signature = (
        datapoints,
        comparisons,
        dim,
        likelihood="probit",
        covar_spec=None,
        covar_tree=None,
        raw=None,
        param_map=None,
        input_chain=None,
        input_chain_training=None,
        input_chain_prefixes=None,
        jitter=1e-6,
        xtol=None,
        consolidate_rtol=0.0,
        consolidate_atol=1e-4,
        maxfev=None,
        mean_constant=0.0,
        float32=true,
        draws=None,
        seed=0,
        update=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        datapoints: Option<Array>,
        comparisons: Option<Array>,
        dim: usize,
        likelihood: &str,
        covar_spec: Option<PyRef<'_, PyKernelSpec>>,
        covar_tree: Option<PyRef<'_, PyKernelNode>>,
        raw: Option<Vec<f64>>,
        param_map: Option<Vec<Vec<usize>>>,
        input_chain: Option<Vec<Bound<'_, PyAny>>>,
        input_chain_training: Option<Vec<bool>>,
        input_chain_prefixes: Option<Vec<String>>,
        jitter: f64,
        xtol: Option<f64>,
        consolidate_rtol: f64,
        consolidate_atol: f64,
        maxfev: Option<usize>,
        mean_constant: f64,
        float32: bool,
        draws: Option<Vec<f64>>,
        seed: u64,
        update: bool,
    ) -> PyResult<Self> {
        let covar_module = match (covar_spec, covar_tree) {
            (Some(s), _) => Some(CovarModule::Standard(s.inner)),
            (None, Some(t)) => Some(CovarModule::Tree(t.inner.clone())),
            (None, None) => None,
        };
        let input_transform: Option<ChainedInputTransform> = match input_chain {
            Some(steps) => Some(crate::input_perturbation::chain_of(
                &steps,
                input_chain_training,
                input_chain_prefixes,
            )?),
            None => None,
        };
        let options = PairwiseGPOptions {
            likelihood: likelihood_of(likelihood)?,
            covar_module,
            raw_parameters: raw.clone(),
            input_transform,
            jitter,
            xtol,
            consolidate_rtol,
            consolidate_atol,
            maxfev,
            mean_constant,
            float32,
            seed,
        };
        let data = match (datapoints, comparisons) {
            (Some(x), Some(c)) => Some((x, c)),
            _ => None,
        };
        let Some((x, c)) = data else {
            let m = PairwiseGP::new(None, dim, options, None).into_py_result()?;
            return Ok(Self {
                inner: RwLock::new(Inner::Single(m)),
            });
        };
        let (batch, xs) = x.batches("datapoints")?;
        let n = xs[0].nrows();
        let comps = comparisons_of(&c, n, xs.len())?;
        let inner = threads::run(py, || -> botorch_core::Result<Inner> {
            if batch.is_empty() {
                let x0 = xs.into_iter().next().expect("one element");
                let c0 = comps.into_iter().next().expect("one element");
                if update {
                    Ok(Inner::Single(PairwiseGP::new(
                        Some((x0, c0)),
                        dim,
                        options,
                        draws.as_deref(),
                    )?))
                } else {
                    let mut m = PairwiseGP::new(None, dim, options, None)?;
                    m.set_train_data(x0, c0, false, None)?;
                    Ok(Inner::Single(m))
                }
            } else {
                let data: Vec<(Matrix, Vec<(usize, usize)>)> = xs.into_iter().zip(comps).collect();
                // `raw` holds the global parameters of the batch.
                let options = PairwiseGPOptions {
                    raw_parameters: None,
                    ..options
                };
                Ok(Inner::Batched(BatchedPairwiseGP::new(
                    batch.clone(),
                    data,
                    dim,
                    options,
                    raw,
                    param_map,
                    draws.as_deref(),
                )?))
            }
        })
        .into_py_result()?;
        Ok(Self {
            inner: RwLock::new(inner),
        })
    }

    /// A copy of the model.
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(self.get()),
        }
    }

    /// The batch shape.
    #[getter]
    fn batch_shape(&self) -> Vec<usize> {
        self.with(Inner::batch_shape)
    }

    /// The input dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.with(|i| i.first().input_dim())
    }

    /// Whether the model has data.
    #[getter]
    fn has_data(&self) -> bool {
        self.with(|i| i.first().data().is_some())
    }

    /// The number of (consolidated) datapoints `n`.
    #[getter]
    fn n(&self) -> usize {
        self.with(|i| i.first().num_datapoints())
    }

    /// The number of comparisons `m`.
    #[getter]
    fn m(&self) -> usize {
        self.with(|i| i.first().data().map_or(0, |d| d.comparisons.len()))
    }

    /// The likelihood's name (`"probit"` or `"logit"`).
    #[getter]
    fn likelihood(&self) -> &'static str {
        self.with(|i| match i.first().likelihood() {
            PairwiseLikelihood::Probit => "probit",
            PairwiseLikelihood::Logit => "logit",
        })
    }

    /// The number of standard normal draws a fresh start needs (0 if the
    /// next update is warm-started).
    #[getter]
    fn needs_draws(&self) -> usize {
        self.with(|i| {
            let els = i.elements();
            if els.iter().any(|e| e.needs_fresh_start()) {
                els.len() * els[0].num_datapoints()
            } else {
                0
            }
        })
    }

    /// The raw hyperparameters (global, flat).
    fn raw_parameters(&self) -> Vec<f64> {
        self.with(|i| match i {
            Inner::Single(m) => m.raw_parameters().to_vec(),
            Inner::Batched(b) => b.raw_parameters().to_vec(),
        })
    }

    /// Sets the raw hyperparameters (the Laplace approximation is not
    /// recomputed, as in BoTorch).
    fn set_raw_parameters(&self, raw: Vec<f64>) -> PyResult<()> {
        let mut inner = self.get();
        match &mut inner {
            Inner::Single(m) => m.set_raw_parameters(&raw),
            Inner::Batched(b) => b.set_raw_parameters(&raw),
        }
        .into_py_result()?;
        self.set(inner);
        Ok(())
    }

    /// The optimizer bounds of the raw hyperparameters.
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.with(|i| match i {
            Inner::Single(m) => m.raw_parameter_bounds(),
            Inner::Batched(b) => b.global_bounds(),
        })
    }

    /// The initial raw parameters of the default covariance module for
    /// `d` inputs (BoTorch's values).
    #[staticmethod]
    #[pyo3(signature = (d, float32=true))]
    fn default_kernel_spec(d: usize, float32: bool) -> PyResult<PyKernelSpec> {
        Ok(PyKernelSpec {
            inner: pairwise_default_covar_module(d, float32).into_py_result()?,
        })
    }

    /// The initial raw parameters of a covariance module.
    #[staticmethod]
    #[pyo3(signature = (covar_spec=None, covar_tree=None))]
    fn initial_raw(
        covar_spec: Option<PyRef<'_, PyKernelSpec>>,
        covar_tree: Option<PyRef<'_, PyKernelNode>>,
    ) -> PyResult<Vec<f64>> {
        match (covar_spec, covar_tree) {
            (Some(s), _) => Ok(initial_raw_parameters(&CovarModule::Standard(s.inner))),
            (None, Some(t)) => Ok(initial_raw_parameters(&CovarModule::Tree(t.inner.clone()))),
            _ => Err(value_error("a covariance module is required")),
        }
    }

    /// The `ConstantMean` constant.
    #[getter]
    fn mean_constant(&self) -> f64 {
        self.with(|i| i.first().mean_constant())
    }

    /// Sets the `ConstantMean` constant (no recomputation).
    #[setter]
    fn set_mean_constant(&self, c: f64) {
        let mut inner = self.get();
        match &mut inner {
            Inner::Single(m) => m.set_mean_constant(c),
            Inner::Batched(b) => {
                for e in b.elements_mut() {
                    e.set_mean_constant(c);
                }
            }
        }
        self.set(inner);
    }

    /// Train (`True`) or eval mode.
    #[getter]
    fn training(&self) -> bool {
        self.with(|i| i.first().training())
    }

    /// Sets train or eval mode.
    #[setter]
    fn set_training(&self, training: bool) {
        let mut inner = self.get();
        match &mut inner {
            Inner::Single(m) => m.set_training(training),
            Inner::Batched(b) => b.set_training(training),
        }
        self.set(inner);
    }

    /// BoTorch's `_update` (warm-started, or from `draws`).
    #[pyo3(signature = (draws=None))]
    fn update(&self, py: Python<'_>, draws: Option<Vec<f64>>) -> PyResult<()> {
        let mut inner = self.get();
        threads::run(py, || match &mut inner {
            Inner::Single(m) => m.update(draws.as_deref()),
            Inner::Batched(b) => b.update(draws.as_deref()),
        })
        .into_py_result()?;
        self.set(inner);
        Ok(())
    }

    /// BoTorch's `set_train_data(datapoints, comparisons, update_model)`
    /// (unbatched).
    #[pyo3(signature = (datapoints, comparisons, update_model=true, draws=None))]
    fn set_train_data(
        &self,
        py: Python<'_>,
        datapoints: Array,
        comparisons: Array,
        update_model: bool,
        draws: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let x = datapoints.matrix("datapoints")?;
        let c = comparisons_of(&comparisons, x.nrows(), 1)?
            .pop()
            .expect("one element");
        let mut inner = self.get();
        let Inner::Single(m) = &mut inner else {
            return Err(crate::errors::unsupported(
                "set_train_data on a batched PairwiseGP is not supported by botorch_rs",
            ));
        };
        threads::run(py, || {
            m.set_train_data(x, c, update_model, draws.as_deref())
        })
        .into_py_result()?;
        self.set(inner);
        Ok(())
    }

    /// The consolidated datapoints (`batch x n x d`).
    fn datapoints(&self) -> PyResult<Option<ArrayOut>> {
        self.with(|i| {
            let els = i.elements();
            if els[0].data().is_none() {
                return Ok(None);
            }
            let mats: Vec<Matrix> = els
                .iter()
                .map(|e| e.data().expect("data").datapoints.clone())
                .collect();
            let (r, c) = mats[0].shape();
            Ok(Some(crate::arrays::stack_out(
                &mats,
                &i.batch_shape(),
                r,
                c,
            )))
        })
    }

    /// The unconsolidated datapoints (`batch x n0 x d`).
    fn unconsolidated_datapoints(&self) -> Option<ArrayOut> {
        self.with(|i| {
            let els = i.elements();
            els[0].data()?;
            let mats: Vec<Matrix> = els
                .iter()
                .map(|e| e.data().expect("data").unconsolidated_datapoints.clone())
                .collect();
            let (r, c) = mats[0].shape();
            Some(crate::arrays::stack_out(&mats, &i.batch_shape(), r, c))
        })
    }

    /// The comparisons (`batch x m x 2`; consolidated unless
    /// `unconsolidated`).
    #[pyo3(signature = (unconsolidated=false))]
    fn comparisons(&self, unconsolidated: bool) -> Option<ArrayOut> {
        self.with(|i| {
            let els = i.elements();
            els[0].data()?;
            let mut data = Vec::new();
            let mut m = 0;
            for e in &els {
                let d = e.data().expect("data");
                let pairs: &[(usize, usize)] = if unconsolidated {
                    &d.unconsolidated_comparisons
                } else {
                    d.comparisons.pairs()
                };
                m = pairs.len();
                for &(a, b) in pairs {
                    data.push(a as f64);
                    data.push(b as f64);
                }
            }
            let mut shape = i.batch_shape();
            shape.extend([m, 2]);
            Some((data, shape))
        })
    }

    /// The new index of every original datapoint (`None` in batch mode or
    /// without data).
    fn consolidated_indices(&self) -> Option<Vec<usize>> {
        self.with(|i| match i {
            Inner::Single(m) => m.data().and_then(|d| d.consolidated_indices.clone()),
            Inner::Batched(_) => None,
        })
    }

    /// BoTorch's `D` (`batch x m x n`).
    #[allow(non_snake_case)]
    fn D(&self) -> Option<ArrayOut> {
        self.with(|i| {
            let els = i.elements();
            els[0].data()?;
            let mats: Vec<Matrix> = els
                .iter()
                .map(|e| e.data().expect("data").comparisons.d_matrix())
                .collect();
            let (r, c) = mats[0].shape();
            Some(crate::arrays::stack_out(&mats, &i.batch_shape(), r, c))
        })
    }

    /// The utility (`batch x n`).
    fn utility(&self) -> PyResult<ArrayOut> {
        let states = self.state_of("utility")?;
        let vecs: Vec<Vec<f64>> = states
            .iter()
            .map(|s| s.utility.as_slice().to_vec())
            .collect();
        Ok(vec_out(&self.with(Inner::batch_shape), &vecs))
    }

    /// The fsolve solution (BoTorch's `_x0`, `batch x n`), or `None`.
    fn x0(&self) -> Option<ArrayOut> {
        self.with(|i| {
            let vecs: Option<Vec<Vec<f64>>> = i
                .elements()
                .iter()
                .map(|e| e.warm_start().map(<[f64]>::to_vec))
                .collect();
            vecs.map(|v| vec_out(&i.batch_shape(), &v))
        })
    }

    /// Sets the warm start (BoTorch's `_x0`; `None` forces a fresh start).
    fn set_x0(&self, x0: Option<Vec<f64>>) {
        let mut inner = self.get();
        match &mut inner {
            Inner::Single(m) => m.set_warm_start(x0),
            Inner::Batched(b) => {
                let n = b.elements()[0].num_datapoints();
                for (k, e) in b.elements_mut().iter_mut().enumerate() {
                    e.set_warm_start(x0.as_ref().map(|v| v[k * n..(k + 1) * n].to_vec()));
                }
            }
        }
        self.set(inner);
    }

    /// `likelihood_hess` (`batch x n x n`).
    fn likelihood_hess(&self) -> PyResult<ArrayOut> {
        self.stacked("likelihood_hess", |s| s.likelihood_hess.clone())
    }

    /// `covar` (`batch x n x n`).
    fn covar(&self) -> PyResult<ArrayOut> {
        self.stacked("covar", |s| s.covar.clone())
    }

    /// `covar_chol` (`batch x n x n`).
    fn covar_chol(&self) -> PyResult<ArrayOut> {
        self.stacked("covar_chol", |s| s.covar_chol.clone())
    }

    /// `covar_inv` (`batch x n x n`).
    fn covar_inv(&self) -> PyResult<ArrayOut> {
        self.stacked("covar_inv", |s| s.covar_inv.clone())
    }

    /// `hlcov_eye` (`batch x n x n`).
    fn hlcov_eye(&self) -> PyResult<ArrayOut> {
        self.stacked("hlcov_eye", LaplaceState::hlcov_eye)
    }

    /// The MLL at the current state (no recomputation).
    fn mll(&self, py: Python<'_>) -> PyResult<f64> {
        let inner = self.get();
        threads::run(py, || match &inner {
            Inner::Single(m) => m.mll(),
            Inner::Batched(b) => b
                .mll_value_and_grad(b.raw_parameters(), false)
                .map(|(v, _, _)| v),
        })
        .into_py_result()
    }

    /// A training forward pass at the current (or the given) raw
    /// parameters: recomputes the Laplace approximation warm-started from
    /// the previous solution, installs it, and returns the MLL and its
    /// gradient with respect to the raw parameters.
    #[pyo3(signature = (raw=None))]
    fn forward_mll(&self, py: Python<'_>, raw: Option<Vec<f64>>) -> PyResult<(f64, Vec<f64>)> {
        let mut inner = self.get();
        let out = threads::run(py, || -> botorch_core::Result<(f64, Vec<f64>)> {
            match &mut inner {
                Inner::Single(m) => {
                    let raw = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
                    let (v, g, st) = m.mll_value_and_grad(&raw, true)?;
                    m.set_raw_parameters(&raw)?;
                    m.set_warm_start(Some(st.x0.clone()));
                    m.update(None)?;
                    Ok((v, g.expect("gradient")))
                }
                Inner::Batched(b) => {
                    let raw = raw.unwrap_or_else(|| b.raw_parameters().to_vec());
                    let (v, g, _) = b.mll_value_and_grad(&raw, true)?;
                    b.set_raw_parameters(&raw)?;
                    b.update(None)?;
                    Ok((v, g.expect("gradient")))
                }
            }
        })
        .into_py_result()?;
        self.set(inner);
        Ok(out)
    }

    /// `fit_gpytorch_mll` with the Laplace MLL (see `PairwiseGP::fit`).
    #[pyo3(signature = (
        maxcor=None,
        pgtol=None,
        ftol=None,
        maxiter=None,
        maxfun=None,
        maxls=None,
        max_attempts=None,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        maxcor: Option<usize>,
        pgtol: Option<f64>,
        ftol: Option<f64>,
        maxiter: Option<usize>,
        maxfun: Option<usize>,
        maxls: Option<usize>,
        max_attempts: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<FitSummary> {
        let options = fit_options(
            maxcor,
            pgtol,
            ftol,
            maxiter,
            maxfun,
            maxls,
            max_attempts,
            seed,
        )?;
        let mut inner = self.get();
        let r = threads::run(py, || match &mut inner {
            Inner::Single(m) => m.fit(options),
            Inner::Batched(b) => b.fit(options),
        })
        .into_py_result()?;
        self.set(inner);
        let status = status_name(r.status);
        Ok((
            r.loss,
            r.iterations,
            r.evaluations,
            status,
            r.message.clone(),
            r.attempts,
            vec![(
                r.loss,
                r.iterations,
                r.evaluations,
                status,
                r.message.clone(),
                r.attempts,
            )],
        ))
    }

    /// BoTorch's `condition_on_observations(X, Y)`: `x` (`n' x d` or
    /// `batch x n' x d`), comparisons `y` (`batch x m' x 2`, indices into
    /// `x`). Unbatched `y` on an unbatched model gives an unbatched,
    /// consolidated model; otherwise a batch (broadcast over `y`'s batch
    /// shape) without consolidation. With `update`, the Laplace
    /// approximation is recomputed (`draws`: the `B·n` initial
    /// perturbations, see `needs_draws`).
    #[pyo3(signature = (x, y, update=true, draws=None))]
    fn condition_on_observations(
        &self,
        py: Python<'_>,
        x: Array,
        y: Array,
        update: bool,
        draws: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let inner = self.get();
        let ybatch: Vec<usize> = y.1[..y.1.len().saturating_sub(2)].to_vec();
        let b: usize = ybatch.iter().product();
        let (xbatch, xs) = x.batches("X")?;
        let nx = xs[0].nrows();
        let comps = comparisons_of(&y, nx, b)?;
        let out = threads::run(py, || -> botorch_core::Result<Inner> {
            let mut out = match (&inner, ybatch.is_empty() && xbatch.is_empty()) {
                (Inner::Single(m), true) => {
                    Inner::Single(m.with_observations(&xs[0], &comps[0], false)?)
                }
                _ => {
                    // Batch mode: every element of Y's batch, with the
                    // model's (and X's) batch elements broadcast.
                    let model_els = inner.elements();
                    let nb = model_els.len();
                    let xb = xs.len();
                    let shape = if ybatch.is_empty() {
                        xbatch.clone()
                    } else {
                        ybatch.clone()
                    };
                    let total: usize = shape.iter().product();
                    let mut els = Vec::with_capacity(total);
                    for k in 0..total {
                        let base = model_els[if nb == 1 { 0 } else { k % nb }];
                        let xk = &xs[if xb == 1 { 0 } else { k % xb }];
                        let ck = &comps[if comps.len() == 1 { 0 } else { k }];
                        els.push(base.with_observations(xk, ck, true)?);
                    }
                    Inner::Batched(BatchedPairwiseGP::from_elements(shape, els)?)
                }
            };
            if update {
                match &mut out {
                    Inner::Single(m) => m.update(draws.as_deref())?,
                    Inner::Batched(bm) => bm.update(draws.as_deref())?,
                }
            }
            Ok(out)
        })
        .into_py_result()?;
        Ok(Self {
            inner: RwLock::new(out),
        })
    }

    /// The `ConstantMean` constant (one value; `ConstantMean` binding).
    #[getter]
    fn constant(&self) -> Vec<f64> {
        vec![self.with(|i| i.first().mean_constant())]
    }

    /// Sets the `ConstantMean` constant (no recomputation).
    fn set_constant(&self, value: Array) -> PyResult<()> {
        if value.0.len() != 1 {
            return Err(dimension_error(format!(
                "the constant of a PairwiseGP is a scalar, got shape {:?}",
                value.1
            )));
        }
        self.set_mean_constant(value.0[0]);
        Ok(())
    }

    /// The batch shape of the mean module (scalar).
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    /// The model handle for posteriors and acquisition functions.
    fn model_handle(&self) -> PyModel {
        PyModel {
            inner: self.with(Inner::model),
        }
    }

    /// The fitted input transform's steps for the Python layer (`None`
    /// without a transform).
    fn input_chain_states(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let inner = self.get();
        match inner.first().input_transform() {
            Some(chain) => Ok(Some(crate::input_perturbation::chain_state(py, chain)?)),
            None => Ok(None),
        }
    }

    /// `jitter`, `xtol`, `consolidate_rtol`, `consolidate_atol`, `maxfev`.
    fn settings(&self) -> (f64, Option<f64>, f64, f64, Option<usize>) {
        self.with(|i| i.first().settings())
    }
}

// ---------------------------------------------------------------------------
// Likelihood functions
// ---------------------------------------------------------------------------

/// A likelihood function of `utility` (`batch x n`) and `D` (`batch x m x
/// n`, BoTorch's comparison matrices): `"p"`, `"log_p"`,
/// `"negative_log_gradient_sum"` or `"negative_log_hessian_sum"`.
#[pyfunction]
pub fn pairwise_likelihood_eval(
    likelihood: &str,
    what: &str,
    utility: Array,
    d: Array,
) -> PyResult<ArrayBuf> {
    let lik = likelihood_of(likelihood)?;
    let (ubatch, us) = (
        utility.1[..utility.1.len().saturating_sub(1)].to_vec(),
        utility.points("utility")?.1,
    );
    let (dbatch, ds) = d.batches("D")?;
    if ubatch != dbatch || us.ncols() != ds[0].ncols() {
        return Err(dimension_error(format!(
            "utility of shape {:?} does not match D of shape {:?}",
            utility.1, d.1
        )));
    }
    let n = us.ncols();
    let mut out = Vec::new();
    let mut shape = ubatch.clone();
    for (b, dm) in ds.iter().enumerate() {
        let u = Vector::from_iterator(n, us.row(b).iter().copied());
        // D's rows as comparisons (+1 and -1 entries).
        let mut pairs = Vec::with_capacity(dm.nrows());
        for k in 0..dm.nrows() {
            let w = (0..n).find(|&j| dm[(k, j)] == 1.0);
            let l = (0..n).find(|&j| dm[(k, j)] == -1.0);
            match (w, l) {
                (Some(w), Some(l)) => pairs.push((w, l)),
                (None, None) => pairs.push((0, 0)),
                _ => {
                    return Err(value_error(
                        "every row of D must contain one 1 and one -1 (or be zero)",
                    ))
                }
            }
        }
        let comps = Comparisons::new(pairs, n).into_py_result()?;
        match what {
            "p" => out.extend(lik.p(&u, &comps).iter()),
            "log_p" => out.extend(lik.log_p(&u, &comps).iter()),
            "negative_log_gradient_sum" => out.extend(
                lik.negative_log_gradient_sum(&u, &comps)
                    .into_py_result()?
                    .iter(),
            ),
            "negative_log_hessian_sum" => {
                let h = lik.negative_log_hessian_sum(&u, &comps);
                out.extend(matrix_out(&h).0);
            }
            other => return Err(value_error(format!("unknown quantity {other:?}"))),
        }
        if b == 0 {
            match what {
                "p" | "log_p" => shape.push(dm.nrows()),
                "negative_log_gradient_sum" => shape.push(n),
                _ => shape.extend([n, n]),
            }
        }
    }
    Ok(ArrayBuf(out, shape))
}

// ---------------------------------------------------------------------------
// Preference acquisition functions
// ---------------------------------------------------------------------------

/// The preference model seen through a deterministic outcome model (the
/// model handle of qEUBO; `outcome=None` returns `pref`).
#[pyfunction]
#[pyo3(signature = (pref, outcome=None))]
pub fn compose_models(
    pref: PyRef<'_, PyModel>,
    outcome: Option<PyRef<'_, PyModel>>,
) -> PyResult<PyModel> {
    Ok(PyModel {
        inner: preference::compose(
            Arc::clone(&pref.inner),
            outcome.map(|o| Arc::clone(&o.inner)),
        )
        .into_py_result()?,
    })
}

/// `AnalyticExpectedUtilityOfBestOption(pref_model, outcome_model,
/// previous_winner)`.
#[pyfunction]
#[pyo3(signature = (pref, outcome=None, previous_winner=None))]
pub fn analytic_eubo(
    pref: PyRef<'_, PyModel>,
    outcome: Option<PyRef<'_, PyModel>>,
    previous_winner: Option<Vec<f64>>,
) -> PyResult<PyAcquisitionFunction> {
    let a = preference::AnalyticExpectedUtilityOfBestOption::new(
        Arc::clone(&pref.inner),
        outcome.map(|o| Arc::clone(&o.inner)),
        previous_winner.map(Vector::from_vec),
    )
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "AnalyticExpectedUtilityOfBestOption",
    ))
}

/// `PairwiseMCPosteriorVariance(model, objective, sampler)`.
#[pyfunction]
pub fn pairwise_mc_posterior_variance(
    model: PyRef<'_, PyModel>,
    objective: Bound<'_, PyAny>,
    sampler: PyRef<'_, PySobolQMCNormalSampler>,
) -> PyResult<PyAcquisitionFunction> {
    let objective = objective_of(Some(&objective))?
        .ok_or_else(|| value_error("PairwiseMCPosteriorVariance requires an objective"))?;
    let a = preference::PairwiseMcPosteriorVariance::new(
        Arc::clone(&model.inner),
        objective,
        Arc::clone(&sampler.inner),
    )
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "PairwiseMCPosteriorVariance",
    ))
}

/// `PairwiseBayesianActiveLearningByDisagreement`: evaluations with explicit
/// standard normal draws (`num_samples x batch`, as BoTorch draws them per
/// evaluation), and the acquisition function for `optimize_acqf` (which
/// draws from its own generator).
#[pyclass(frozen, name = "PairwiseBALD", module = "botorch_rs._botorch_rs")]
pub struct PyPairwiseBald {
    inner: Arc<preference::PairwiseBayesianActiveLearningByDisagreement>,
}

impl PyPairwiseBald {
    fn run(
        &self,
        py: Python<'_>,
        x: &Array,
        draws: &Array,
        grad: bool,
    ) -> PyResult<(ArrayBuf, Option<ArrayBuf>)> {
        let (batch, sets) = x.batches("X")?;
        let b = sets.len();
        let n = self.inner.num_samples();
        if draws.0.len() != n * b {
            return Err(dimension_error(format!(
                "expected {n} x {b} draws, got shape {:?}",
                draws.1
            )));
        }
        let acqf = &self.inner;
        let results = threads::run(py, || {
            sets.par_iter()
                .enumerate()
                .map(|(k, s)| {
                    let eps: Vec<f64> = (0..n).map(|i| draws.0[i * b + k]).collect();
                    acqf.evaluate_with_draws(s, &eps, grad)
                })
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let values: Vec<f64> = results.iter().map(|(v, _)| *v).collect();
        let grads = if grad {
            let mut g = Vec::with_capacity(x.0.len());
            for (_, m) in &results {
                g.extend(matrix_out(m.as_ref().expect("gradient")).0);
            }
            Some(ArrayBuf(g, x.1.clone()))
        } else {
            None
        };
        Ok((ArrayBuf(values, batch), grads))
    }
}

#[pymethods]
impl PyPairwiseBald {
    #[new]
    #[pyo3(signature = (pref, outcome=None, num_samples=1024, std_noise=0.0, seed=0))]
    fn new(
        pref: PyRef<'_, PyModel>,
        outcome: Option<PyRef<'_, PyModel>>,
        num_samples: usize,
        std_noise: f64,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                preference::PairwiseBayesianActiveLearningByDisagreement::new(
                    Arc::clone(&pref.inner),
                    outcome.map(|o| Arc::clone(&o.inner)),
                    num_samples,
                    std_noise,
                    seed,
                )
                .into_py_result()?,
            ),
        })
    }

    /// The acquisition function (fresh draws from its own generator at
    /// every evaluation).
    fn acquisition_function(&self) -> PyAcquisitionFunction {
        PyAcquisitionFunction::new_plain(
            self.inner.clone(),
            "PairwiseBayesianActiveLearningByDisagreement",
        )
    }

    /// Values at the candidate sets `x` (`batch x 2 x d`) with the draws
    /// `draws` (`num_samples x batch`).
    fn forward(&self, py: Python<'_>, x: Array, draws: Array) -> PyResult<ArrayBuf> {
        Ok(self.run(py, &x, &draws, false)?.0)
    }

    /// Values and gradients with the draws `draws`.
    fn forward_with_grad(
        &self,
        py: Python<'_>,
        x: Array,
        draws: Array,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let (v, g) = self.run(py, &x, &draws, true)?;
        Ok((v, g.expect("gradient")))
    }

    /// The number of draws per candidate set.
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }
}

/// BoTorch's `LearnedObjective` on outcome samples `y` (`B x q x d`, every
/// row-major batch element a set of `q` outcome vectors) with base samples
/// `z` (`N x B x q`): the samples of the preference model's posterior
/// (`N x B x q`) and, with the adjoint `adjoint` of them, the adjoint of
/// `y`.
#[pyfunction]
#[pyo3(signature = (model, y, z, adjoint=None, model_index=None))]
pub fn learned_objective_samples(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    y: Array,
    z: Array,
    adjoint: Option<Array>,
    model_index: Option<usize>,
) -> PyResult<(ArrayBuf, Option<ArrayBuf>)> {
    let (input_batch, sets) = y.batches("samples")?;
    let full_model_batch = model.inner.batch_shape();
    if model_index.is_some_and(|i| i >= full_model_batch.iter().product()) {
        return Err(value_error("preference model batch index is out of range"));
    }
    let model_batch = if model_index.is_some() {
        &[]
    } else {
        full_model_batch
    };
    let batch = broadcast_shapes(&input_batch, model_batch).ok_or_else(|| {
        dimension_error(format!(
            "outcome sample batch {input_batch:?} does not broadcast with preference batch {model_batch:?}"
        ))
    })?;
    let inputs = broadcast_indices(&input_batch, &batch);
    let models = model_index.map_or_else(
        || broadcast_indices(model_batch, &batch),
        |i| vec![i; inputs.len()],
    );
    let b = inputs.len();
    let q = sets.first().map_or(0, Matrix::nrows);
    let n = z.1.first().copied().unwrap_or(0);
    if n == 0 || z.0.len() != n * b * q {
        return Err(dimension_error(format!(
            "base samples of shape {:?} do not match {b} sets of {q} points",
            z.1
        )));
    }
    if adjoint.as_ref().is_some_and(|a| a.0.len() != n * b * q) {
        return Err(dimension_error(
            "preference sample adjoint has the wrong number of values",
        ));
    }
    let m = Arc::clone(&model.inner);
    let results = threads::run(py, || {
        inputs
            .par_iter()
            .enumerate()
            .map(|(e, &input)| {
                let member = if m.batch_shape().is_empty() {
                    m.as_ref()
                } else {
                    m.batch_element(models[e]).ok_or_else(|| {
                        botorch_core::BotorchError::Unsupported(
                            "the preference model does not expose native batch elements".into(),
                        )
                    })?
                };
                let base = Matrix::from_fn(n, q, |i, p| z.0[(i * b + e) * q + p]);
                let ybar = adjoint
                    .as_ref()
                    .map(|a| Matrix::from_fn(n, q, |i, p| a.0[(i * b + e) * q + p]));
                preference::posterior_samples_vjp(member, &sets[input], &base, ybar.as_ref())
            })
            .collect::<botorch_core::Result<Vec<_>>>()
    })
    .into_py_result()?;
    let mut values = vec![0.0; n * b * q];
    for (e, (s, _)) in results.iter().enumerate() {
        for i in 0..n {
            for p in 0..q {
                values[(i * b + e) * q + p] = s[(i, p)];
            }
        }
    }
    let grads = if adjoint.is_some() {
        let mut g = vec![0.0; y.0.len()];
        let size = q * y.1[y.1.len() - 1];
        for (&input, (_, gm)) in inputs.iter().zip(&results) {
            for (j, value) in matrix_out(gm.as_ref().expect("gradient"))
                .0
                .iter()
                .enumerate()
            {
                g[input * size + j] += value;
            }
        }
        Some(ArrayBuf(g, y.1.clone()))
    } else {
        None
    };
    let mut shape = vec![n];
    shape.extend(batch);
    shape.push(q);
    Ok((ArrayBuf(values, shape), grads))
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPairwiseGP>()?;
    m.add_function(wrap_pyfunction!(pairwise_likelihood_eval, m)?)?;
    m.add_class::<PyPairwiseBald>()?;
    m.add_function(wrap_pyfunction!(compose_models, m)?)?;
    m.add_function(wrap_pyfunction!(analytic_eubo, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_mc_posterior_variance, m)?)?;
    m.add_function(wrap_pyfunction!(learned_objective_samples, m)?)?;
    Ok(())
}
