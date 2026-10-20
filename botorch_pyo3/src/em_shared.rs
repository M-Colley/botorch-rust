//! Joint native optimization of exact GPs sharing their mean and covariance.
use crate::arrays::ArrayOut;
use crate::errors::{value_error, IntoPyResult};
use crate::models::PySingleTaskGP;
use crate::threads;
use botorch_core::fit::{FitOptions, FitStatus, Fittable};
use botorch_core::models::em_shared::SharedGPFit;
use botorch_core::models::Model;
use botorch_core::optim::fit::fit_gpytorch_mll_torch;
use botorch_core::optim::minimize::TorchMinimizeOptions;
use botorch_core::optim::{AdamOptions, ExpMaStoppingCriterion};
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};
type TorchFitSummary = (usize, f64, u8, Option<f64>, Option<String>);
type FitSummary = (f64, usize, usize, &'static str, String, usize);
#[pyclass(frozen, name = "SharedGPFit", module = "botorch_rs._botorch_rs")]
pub struct PySharedGPFit {
    inner: RwLock<Arc<SharedGPFit>>,
}
impl PySharedGPFit {
    fn get(&self) -> Arc<SharedGPFit> {
        Arc::clone(&self.inner.read().unwrap_or_else(|e| e.into_inner()))
    }
    fn set(&self, model: SharedGPFit) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }
}
#[pymethods]
impl PySharedGPFit {
    #[new]
    fn new(
        models: Vec<PyRef<'_, PySingleTaskGP>>,
        learn_mean: bool,
        freeze_noise: bool,
    ) -> PyResult<Self> {
        let mut single = Vec::new();
        for m in models {
            let snapshot = m.snapshot();
            if snapshot.num_outputs() != 1 || !snapshot.batch_shape().is_empty() {
                return Err(value_error("shared fitting requires unbatched scalar GPs"));
            }
            let gps = snapshot
                .output_gps()
                .ok_or_else(|| value_error("shared fitting requires exact scalar GPs"))?;
            single.push(gps[0].clone());
        }
        let model = SharedGPFit::new(single, learn_mean, freeze_noise).into_py_result()?;
        Ok(Self {
            inner: RwLock::new(Arc::new(model)),
        })
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
    fn local_parameters(&self) -> Vec<Vec<f64>> {
        self.get().local_parameters()
    }
    fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.get().parameter_shapes()
    }
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.get()
            .parameter_shapes()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product()))
            .collect()
    }
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.get().raw_parameter_bounds()
    }
    #[pyo3(signature=(raw=None))]
    fn mll(&self, py: Python<'_>, raw: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let m = self.get();
        let p = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.value_and_grad(&p))
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
        let p = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.value_and_grad(&p))
            .map(|(v, g)| ((vec![v], vec![]), g))
            .into_py_result()
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
        let r = threads::run(py, || m.fit_mll(options)).into_py_result()?;
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
            fit_gpytorch_mll_torch(&mut m, mask.as_deref(), bounds.as_deref(), &options, None)
        })
        .into_py_result()?;
        self.set(m);
        Ok((r.step, r.fval, r.status.value(), r.runtime, r.message))
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySharedGPFit>()
}
