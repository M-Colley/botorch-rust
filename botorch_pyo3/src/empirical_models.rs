//! Native exact empirical Gaussian-process bindings.
use crate::arrays::{Array, ArrayOut};
use crate::errors::{value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::PyModel;
use crate::specs::PyLikelihoodSpec;
use crate::threads;
use botorch_core::fit::{FitOptions, FitStatus, Fittable};
use botorch_core::kernels::tree::KernelType;
use botorch_core::models::empirical::{EmpiricalFit, EmpiricalGP, EmpiricalTaskView};
use botorch_core::optim::fit::fit_gpytorch_mll_torch;
use botorch_core::optim::minimize::TorchMinimizeOptions;
use botorch_core::optim::{AdamOptions, ExpMaStoppingCriterion};
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};
type TorchFitSummary = (usize, f64, u8, Option<f64>, Option<String>);
#[pyclass(frozen, name = "EmpiricalGP", module = "botorch_rs._botorch_rs")]
pub struct PyEmpiricalGP {
    inner: RwLock<Arc<EmpiricalGP>>,
}
impl PyEmpiricalGP {
    fn get(&self) -> Arc<EmpiricalGP> {
        Arc::clone(&self.inner.read().unwrap_or_else(|e| e.into_inner()))
    }
    fn set(&self, m: EmpiricalGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(m);
    }
    fn wrap(m: EmpiricalGP) -> Self {
        Self {
            inner: RwLock::new(Arc::new(m)),
        }
    }
}
fn matrices(data: Vec<Array>) -> PyResult<Vec<botorch_core::Matrix>> {
    data.into_iter()
        .map(|a| {
            if a.1.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(a.0.len()) {
                return Err(value_error(
                    "empirical array length differs from declared shape",
                ));
            }
            a.matrix("empirical training array")
        })
        .collect()
}
#[pymethods]
impl PyEmpiricalGP {
    #[new]
    #[pyo3(signature=(x,y,yvar,mean,covar,noise,kernel_raw,noise_raw,correlated=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Vec<Array>,
        y: Vec<Array>,
        yvar: Option<Vec<Array>>,
        mean: PyRef<'_, PyKernelNode>,
        covar: PyRef<'_, PyKernelNode>,
        noise: PyRef<'_, PyLikelihoodSpec>,
        kernel_raw: Vec<f64>,
        noise_raw: Vec<f64>,
        correlated: bool,
    ) -> PyResult<Self> {
        let spec = match mean.inner.kind() {
            KernelType::EmpiricalBasis(spec) => spec.as_ref().clone(),
            _ => {
                return Err(value_error(
                    "empirical mean must contain an interpolation table",
                ))
            }
        };
        EmpiricalGP::new(
            matrices(x)?,
            matrices(y)?,
            yvar.map(matrices).transpose()?,
            spec,
            covar.inner.clone(),
            noise.inner,
            kernel_raw,
            noise_raw,
            correlated,
        )
        .map(Self::wrap)
        .into_py_result()
    }
    fn raw_parameters(&self) -> Vec<f64> {
        self.get().raw_parameters().to_vec()
    }
    fn set_raw_parameters(&self, raw: Vec<f64>) -> PyResult<()> {
        let mut m = (*self.get()).clone();
        m.set_raw_parameters(&raw).into_py_result()?;
        self.set(m);
        Ok(())
    }
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.get().parameter_names()
    }
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.get().raw_parameter_bounds()
    }
    #[getter]
    fn noise(&self) -> Vec<f64> {
        self.get().likelihood_noise()
    }
    fn set_noise(&self, noise: Array) -> PyResult<()> {
        if noise.1.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(noise.0.len()) {
            return Err(value_error(
                "noise array length differs from declared shape",
            ));
        }
        let mut m = (*self.get()).clone();
        m.set_noise(&noise.0).into_py_result()?;
        self.set(m);
        Ok(())
    }
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        self.get().aug_batch_shape()
    }
    #[getter]
    fn dim(&self) -> usize {
        1
    }
    fn model_handle(&self) -> PyModel {
        PyModel { inner: self.get() }
    }
    #[pyo3(signature=(x,y,noise=None))]
    fn condition_on_observations(
        &self,
        py: Python<'_>,
        x: Array,
        y: Array,
        noise: Option<Array>,
    ) -> PyResult<Self> {
        let x = x.matrix("X")?;
        let y = y.matrix("Y")?;
        let noise = noise.map(|v| v.matrix("noise")).transpose()?;
        let model = self.get();
        threads::run(py, || {
            model.condition_on_observations(&x, &y, noise.as_ref())
        })
        .map(Self::wrap)
        .into_py_result()
    }
    #[pyo3(signature=(outputs,task_feature=None))]
    fn task_model_handle(
        &self,
        outputs: Vec<usize>,
        task_feature: Option<usize>,
    ) -> PyResult<PyModel> {
        EmpiricalTaskView::new(self.get(), outputs, task_feature)
            .map(|view| PyModel {
                inner: Arc::new(view),
            })
            .into_py_result()
    }
    #[pyo3(signature=(raw=None))]
    fn mll(&self, py: Python<'_>, raw: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let m = self.get();
        let raw = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.mll_value_and_grad(&raw, false))
            .map(|(v, _)| {
                let s = if v.len() == 1 { vec![] } else { vec![v.len()] };
                (v, s)
            })
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
        threads::run(py, || m.mll_value_and_grad(&raw, true))
            .map(|(v, g)| {
                let s = if v.len() == 1 { vec![] } else { vec![v.len()] };
                ((v, s), g)
            })
            .into_py_result()
    }
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(self.get()),
        }
    }
    #[pyo3(signature=(seed=None))]
    fn sample_all_priors(&self, seed: Option<u64>) -> PyResult<()> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            seed.unwrap_or_else(botorch_core::utils::sampling::random_seed),
        );
        let mut m = (*self.get()).clone();
        let raw = m.sample_all_priors(&mut rng);
        m.set_raw_parameters(&raw).into_py_result()?;
        self.set(m);
        Ok(())
    }
    #[pyo3(signature=(maxcor=None,pgtol=None,ftol=None,maxiter=None,maxfun=None,maxls=None,max_attempts=None,seed=None))]
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
    ) -> PyResult<(f64, usize, usize, &'static str, String, usize)> {
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
        let mut fit = EmpiricalFit {
            model: (*self.get()).clone(),
        };
        let r = threads::run(py, || fit.fit_mll(options)).into_py_result()?;
        self.set(fit.model);
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
        let mut fit = EmpiricalFit {
            model: (*self.get()).clone(),
        };
        let r = threads::run(py, || {
            fit_gpytorch_mll_torch(&mut fit, mask.as_deref(), bounds.as_deref(), &options, None)
        })
        .into_py_result()?;
        self.set(fit.model);
        Ok((r.step, r.fval, r.status.value(), r.runtime, r.message))
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmpiricalGP>()
}
