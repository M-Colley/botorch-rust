//! Immutable native EM-prior model handles and analytic historical MLL fitting.

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::em_gp::{matrix, vector};
use crate::errors::{value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::PyModel;
use crate::specs::{PyConstraint, PyLikelihoodSpec, PyPrior};
use crate::threads;
use botorch_core::fit::{FitOptions, FitStatus};
use botorch_core::models::em_gp::{EMDataset, EMOptions};
use botorch_core::models::em_model::{EMEmpiricalGP, EMGPPrior, EMModelOptions, EMPrior};
use botorch_core::models::{ConstantMeanSpec, Model};
use botorch_core::optim::minimize::TorchMinimizeOptions;
use botorch_core::optim::{AdamOptions, ExpMaStoppingCriterion};
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};

type TorchFitSummary = (usize, f64, u8, Option<f64>, Option<String>);
type FitSummary = (f64, usize, usize, &'static str, String, usize);
type InducingState = (ArrayOut, ArrayBuf, ArrayBuf, ArrayOut);

#[pyclass(frozen, name = "EMGaussianProcess", module = "botorch_rs._botorch_rs")]
pub struct PyEMGaussianProcess {
    inner: RwLock<Arc<EMEmpiricalGP>>,
}
impl PyEMGaussianProcess {
    fn get(&self) -> Arc<EMEmpiricalGP> {
        Arc::clone(&self.inner.read().unwrap_or_else(|e| e.into_inner()))
    }
    fn set(&self, model: EMEmpiricalGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }
}

#[pymethods]
impl PyEMGaussianProcess {
    #[new]
    #[pyo3(signature=(x,y,history_x,history_y,inducing,mean,covariance,kernel,kernel_raw,likelihood,noise_raw,constant=0.0,learn_mean=false,mean_prior=None,mean_constraint=None,base=None,base_raw=Vec::new(),interpolate=true,inducing_observations=false,freeze_pretrained=false,em_iterations=None,em_mean_prior=false,em_covar_prior=false,em_iw_nu=None,em_tolerance=Some(1e-6),em_shrinkage=0.0,warm_start_em=false,learnable_inducing=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        x: Array,
        y: Array,
        history_x: Vec<Array>,
        history_y: Vec<Array>,
        inducing: Array,
        mean: Array,
        covariance: Array,
        kernel: PyRef<'_, PyKernelNode>,
        kernel_raw: Vec<f64>,
        likelihood: PyRef<'_, PyLikelihoodSpec>,
        noise_raw: f64,
        constant: f64,
        learn_mean: bool,
        mean_prior: Option<PyRef<'_, PyPrior>>,
        mean_constraint: Option<PyRef<'_, PyConstraint>>,
        base: Option<PyRef<'_, PyKernelNode>>,
        base_raw: Vec<f64>,
        interpolate: bool,
        inducing_observations: bool,
        freeze_pretrained: bool,
        em_iterations: Option<usize>,
        em_mean_prior: bool,
        em_covar_prior: bool,
        em_iw_nu: Option<f64>,
        em_tolerance: Option<f64>,
        em_shrinkage: f64,
        warm_start_em: bool,
        learnable_inducing: bool,
    ) -> PyResult<Self> {
        if !learn_mean && (mean_prior.is_some() || mean_constraint.is_some() || constant != 0.0) {
            return Err(value_error(
                "constant mean configuration requires learn_mean=true",
            ));
        }
        if em_iterations.is_none()
            && (em_mean_prior
                || em_covar_prior
                || em_iw_nu.is_some()
                || em_shrinkage != 0.0
                || em_tolerance != Some(1e-6))
        {
            return Err(value_error("EM training options require em_iterations"));
        }
        if history_x.len() != history_y.len() {
            return Err(value_error("historical EM dataset counts differ"));
        }
        if base.is_none() && !base_raw.is_empty() {
            return Err(value_error(
                "base raw parameters require an additive kernel",
            ));
        }
        let data = history_x
            .into_iter()
            .zip(history_y)
            .map(|(x, y)| {
                Ok(EMDataset {
                    x: matrix(x, "historical X")?,
                    y: vector(y, "historical Y")?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let x = matrix(x, "train_X")?;
        let y = vector(y, "train_Y")?;
        let prior = EMPrior {
            x: matrix(inducing, "inducing X")?,
            mean: vector(mean, "inducing mean")?,
            covariance: matrix(covariance, "inducing covariance")?,
        };
        let options = EMModelOptions {
            kernel: kernel.inner.clone(),
            kernel_raw,
            mean: learn_mean.then(|| ConstantMeanSpec {
                prior: mean_prior.map(|p| p.inner),
                constraint: mean_constraint.map(|c| c.inner),
            }),
            constant,
            likelihood: likelihood.inner,
            noise_raw,
            base: base.map(|b| (b.inner.clone(), base_raw)),
            interpolate,
            inducing_observations,
            freeze_pretrained,
            em: em_iterations.map(|iterations| EMOptions {
                iterations,
                mean_prior: em_mean_prior,
                covariance_prior: em_covar_prior,
                iw_nu: em_iw_nu,
                tolerance: em_tolerance,
                shrinkage: em_shrinkage,
                ..Default::default()
            }),
            warm_start_em,
            learnable_inducing,
        };
        let model =
            threads::run(py, || EMEmpiricalGP::new(x, y, data, prior, options)).into_py_result()?;
        Ok(Self {
            inner: RwLock::new(Arc::new(model)),
        })
    }
    fn raw_parameters(&self) -> Vec<f64> {
        self.get().raw_parameters().to_vec()
    }
    fn set_raw_parameters(&self, py: Python<'_>, raw: Vec<f64>) -> PyResult<()> {
        let mut model = (*self.get()).clone();
        threads::run(py, || model.set_raw_parameters(&raw)).into_py_result()?;
        self.set(model);
        Ok(())
    }
    fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.get().parameter_shapes()
    }
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.get()
            .parameter_shapes()
            .into_iter()
            .map(|(name, shape)| (name, shape.iter().product()))
            .collect()
    }
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.get().raw_parameter_bounds()
    }
    #[getter]
    fn noise(&self) -> Vec<f64> {
        vec![self.get().noise()]
    }
    fn set_noise(&self, py: Python<'_>, noise: Array) -> PyResult<()> {
        if noise.0.len() != 1
            || noise.1.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(1)
        {
            return Err(value_error("noise requires one scalar value"));
        }
        let raw = self
            .get()
            .raw_parameter_vector_for_noise(noise.0[0])
            .into_py_result()?;
        self.set_raw_parameters(py, raw)
    }
    #[getter]
    fn constant(&self) -> Vec<f64> {
        vec![self.get().constant()]
    }
    fn set_constant(&self, py: Python<'_>, constant: Array) -> PyResult<()> {
        if constant.0.len() != 1
            || constant.1.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(1)
        {
            return Err(value_error("constant requires one scalar value"));
        }
        let raw = self
            .get()
            .raw_parameter_vector_for_constant(constant.0[0])
            .into_py_result()?;
        self.set_raw_parameters(py, raw)
    }
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        vec![]
    }
    #[getter]
    fn dim(&self) -> usize {
        self.get().dim()
    }
    fn model_handle(&self) -> PyModel {
        PyModel { inner: self.get() }
    }
    fn prior_model_handle(&self) -> PyModel {
        PyModel {
            inner: Arc::new(EMGPPrior(self.get())),
        }
    }
    fn em_state(&self) -> InducingState {
        let (mean, covariance, factor, delta) = self.get().inducing_state();
        let n = mean.len();
        (
            (mean.as_slice().to_vec(), vec![n]),
            matrix_out(&covariance).into(),
            matrix_out(&factor).into(),
            (delta.as_slice().to_vec(), vec![n]),
        )
    }
    fn inducing_points(&self) -> ArrayBuf {
        matrix_out(self.get().inducing_points()).into()
    }
    fn refresh_em(&self, py: Python<'_>) -> PyResult<()> {
        let mut model = (*self.get()).clone();
        threads::run(py, || model.refresh_em()).into_py_result()?;
        self.set(model);
        Ok(())
    }
    #[pyo3(signature=(raw=None))]
    fn advance_mll(&self, py: Python<'_>, raw: Option<Vec<f64>>) -> PyResult<(ArrayOut, Vec<f64>)> {
        let mut model = (*self.get()).clone();
        let raw = raw.unwrap_or_else(|| model.raw_parameters().to_vec());
        let (value, gradient) = threads::run(py, || model.advance_mll(&raw)).into_py_result()?;
        self.set(model);
        Ok(((vec![value], vec![]), gradient))
    }
    fn freeze_pretrained(&self, freeze: bool) {
        let mut model = (*self.get()).clone();
        model.freeze_pretrained(freeze);
        self.set(model);
    }
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(self.get()),
        }
    }
    #[pyo3(signature=(raw=None))]
    fn mll(&self, py: Python<'_>, raw: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let m = self.get();
        let raw = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.mll_value_and_grad(&raw))
            .map(|(v, _)| (vec![v], vec![]))
            .into_py_result()
    }
    #[pyo3(signature=(raw=None))]
    fn mll_value_and_grad(
        &self,
        py: Python<'_>,
        raw: Option<Vec<f64>>,
    ) -> PyResult<(ArrayOut, Vec<f64>)> {
        let m = self.get();
        let raw = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.mll_value_and_grad(&raw))
            .map(|(v, g)| ((vec![v], vec![]), g))
            .into_py_result()
    }
    #[pyo3(signature=(seed=None))]
    fn sample_all_priors(&self, py: Python<'_>, seed: Option<u64>) -> PyResult<()> {
        use rand::SeedableRng;
        let m = self.get();
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            seed.unwrap_or_else(botorch_core::utils::sampling::random_seed),
        );
        self.set_raw_parameters(py, m.sample_all_priors(&mut rng))
    }
    #[pyo3(signature=(maxcor=None,pgtol=None,ftol=None,maxiter=None,maxfun=None,maxls=None,max_attempts=None,seed=None,mask=None,bounds=None))]
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
        mask: Option<Vec<bool>>,
        bounds: Option<Vec<(f64, f64)>>,
    ) -> PyResult<FitSummary> {
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
        let mut m = (*self.get()).clone();
        let r = threads::run(py, || {
            m.fit_mll_with_mask(options, mask.as_deref(), bounds.as_deref())
        })
        .into_py_result()?;
        self.set(m);
        let status = match r.status {
            FitStatus::Success => "SUCCESS",
            FitStatus::Stopped => "STOPPED",
            FitStatus::Failure => "FAILURE",
        };
        Ok((
            r.loss,
            r.iterations,
            r.evaluations,
            status,
            r.message,
            r.attempts,
        ))
    }
    #[pyo3(signature=(mask=None,bounds=None,step_limit=None,stopping=None,timeout_sec=None,lr=0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0,amsgrad=false,maximize=false))]
    #[allow(clippy::too_many_arguments)]
    fn fit_torch(
        &self,
        py: Python<'_>,
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
    ) -> PyResult<TorchFitSummary> {
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
            stopping_criterion: stopping
                .map(|s| ExpMaStoppingCriterion::new(s.0, s.1, s.2, s.3, s.4)),
        };
        let mut m = (*self.get()).clone();
        let r = threads::run(py, || {
            m.fit_mll_torch(mask.as_deref(), bounds.as_deref(), &options)
        })
        .into_py_result()?;
        self.set(m);
        Ok((r.step, r.fval, r.status.value(), r.runtime, r.message))
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEMGaussianProcess>()
}
