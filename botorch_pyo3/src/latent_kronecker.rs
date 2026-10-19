//! Native latent Kronecker GP bindings.

use std::sync::{Arc, RwLock};

use botorch_core::fit::{FitOptions, FitStatus, Fittable};
use botorch_core::models::approximate_gp::VariationalMean;
use botorch_core::models::gp::ConstantMeanSpec;
use botorch_core::models::latent_kronecker_gp::{
    LatentKroneckerFit, LatentKroneckerGP, LatentKroneckerGPOptions,
};
use botorch_core::optim::fit::fit_gpytorch_mll_torch;
use botorch_core::optim::minimize::TorchMinimizeOptions;
use botorch_core::optim::{AdamOptions, ExpMaStoppingCriterion};
use botorch_core::transforms::input::ChainedInputTransform;
use botorch_core::Model;
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::PyModel;
use crate::specs::{PyConstraint, PyLikelihoodSpec, PyPrior};
use crate::threads;

/// Native latent Kronecker model, shared by immutable prediction handles.
#[pyclass(frozen, name = "LatentKroneckerGP", module = "botorch_rs._botorch_rs")]
pub struct PyLatentKroneckerGP {
    inner: RwLock<Arc<LatentKroneckerGP>>,
}
impl PyLatentKroneckerGP {
    fn get(&self) -> Arc<LatentKroneckerGP> {
        Arc::clone(&self.inner.read().unwrap_or_else(|e| e.into_inner()))
    }
    fn set(&self, value: LatentKroneckerGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(value);
    }
    fn wrap(value: LatentKroneckerGP) -> Self {
        Self {
            inner: RwLock::new(Arc::new(value)),
        }
    }
}

#[pymethods]
impl PyLatentKroneckerGP {
    #[new]
    #[pyo3(signature=(train_x, train_t, train_y, dim, covar_x, covar_t, noise, raw, mean_x="zero", mean_t="zero", mean_x_prior=None, mean_t_prior=None, mean_x_constraint=None, mean_t_constraint=None, input_chain=None, input_chain_training=None, input_chain_prefixes=None, raw_train_x=None, outcome=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        train_x: Array,
        train_t: Array,
        train_y: Array,
        dim: usize,
        covar_x: PyRef<'_, PyKernelNode>,
        covar_t: PyRef<'_, PyKernelNode>,
        noise: PyRef<'_, PyLikelihoodSpec>,
        raw: Vec<f64>,
        mean_x: &str,
        mean_t: &str,
        mean_x_prior: Option<PyRef<'_, PyPrior>>,
        mean_t_prior: Option<PyRef<'_, PyPrior>>,
        mean_x_constraint: Option<PyRef<'_, PyConstraint>>,
        mean_t_constraint: Option<PyRef<'_, PyConstraint>>,
        input_chain: Option<Vec<Bound<'_, PyAny>>>,
        input_chain_training: Option<Vec<bool>>,
        input_chain_prefixes: Option<Vec<String>>,
        raw_train_x: Option<Array>,
        outcome: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let mean = |name: &str,
                    prior: Option<PyRef<'_, PyPrior>>,
                    constraint: Option<PyRef<'_, PyConstraint>>|
         -> PyResult<VariationalMean> {
            match name {
                "zero" => Ok(VariationalMean::Zero),
                "constant" => Ok(VariationalMean::Constant(ConstantMeanSpec {
                    prior: prior.map(|p| p.inner),
                    constraint: constraint.map(|c| c.inner),
                })),
                _ => Err(value_error(
                    "only zero and constant mean factors are supported",
                )),
            }
        };
        let input_transform: Option<ChainedInputTransform> = match input_chain {
            Some(steps) => {
                let mut chain = crate::input_perturbation::chain_of(
                    &steps,
                    input_chain_training,
                    input_chain_prefixes,
                )?;
                chain
                    .fit(
                        &raw_train_x
                            .ok_or_else(|| value_error("an input chain requires raw_train_x"))?
                            .matrix("train_X")?,
                    )
                    .into_py_result()?;
                Some(chain)
            }
            None => None,
        };
        let options = LatentKroneckerGPOptions {
            covar_x: covar_x.inner.clone(),
            covar_t: covar_t.inner.clone(),
            mean_x: mean(mean_x, mean_x_prior, mean_x_constraint)?,
            mean_t: mean(mean_t, mean_t_prior, mean_t_constraint)?,
            noise: noise.inner,
            input_transform,
            outcome,
        };
        LatentKroneckerGP::new(
            train_x.matrix("train_X")?,
            train_t.matrix("train_T")?,
            train_y.matrix("train_Y")?,
            dim,
            Some(raw),
            options,
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
        vec![self.get().noise()]
    }
    fn set_noise(&self, value: Array) -> PyResult<()> {
        if value.0.len() != 1 {
            return Err(value_error("noise must be scalar"));
        }
        let mut m = (*self.get()).clone();
        let mut raw = m.raw_parameters().to_vec();
        raw[0] = m
            .options()
            .noise
            .noise_constraint
            .inverse_transform(value.0[0]);
        m.set_raw_parameters(&raw).into_py_result()?;
        self.set(m);
        Ok(())
    }
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        vec![]
    }
    #[getter]
    fn dim(&self) -> usize {
        Model::dim(&*self.get())
    }
    #[getter]
    fn num_train(&self) -> usize {
        self.get().train_inputs().nrows()
    }
    #[getter]
    fn num_observed(&self) -> usize {
        self.get().observed_indices().len()
    }
    #[getter]
    fn complete_grid(&self) -> bool {
        self.get().is_complete()
    }
    fn observed_indices(&self) -> Vec<usize> {
        self.get().observed_indices().to_vec()
    }
    fn train_inputs(&self) -> ArrayOut {
        matrix_out(self.get().train_inputs())
    }
    fn train_t(&self) -> ArrayOut {
        matrix_out(self.get().train_t())
    }
    fn train_targets(&self) -> ArrayOut {
        let m = self.get();
        let y = m.train_targets();
        (y.as_slice().to_vec(), vec![y.len()])
    }
    fn model_handle(&self) -> PyModel {
        PyModel { inner: self.get() }
    }
    #[pyo3(signature=(raw=None))]
    fn mll(&self, py: Python<'_>, raw: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let m = self.get();
        let raw = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.mll_value_and_grad(&raw, false))
            .map(|r| (vec![r.0], vec![]))
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
            .map(|(v, g)| ((vec![v], vec![]), g))
            .into_py_result()
    }
    fn posterior_grid(&self, py: Python<'_>, x: Array, t: Array) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let x = x.matrix("X")?;
        let t = t.matrix("T")?;
        let m = self.get();
        let posterior = threads::run(py, || m.posterior_grid(&x, &t)).into_py_result()?;
        Ok((
            ArrayBuf(
                posterior.mean.as_slice().to_vec(),
                vec![x.nrows(), t.nrows()],
            ),
            matrix_out(&posterior.covariance).into(),
        ))
    }
    fn samples(&self, py: Python<'_>, x: Array, t: Array, base: Array) -> PyResult<ArrayBuf> {
        let x = x.matrix("X")?;
        let t = t.matrix("T")?;
        let base = base.matrix("base_samples")?;
        let m = self.get();
        threads::run(py, || m.samples(&x, &t, &base))
            .map(|s| matrix_out(&s).into())
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
        let mut fit = LatentKroneckerFit {
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
    ) -> PyResult<(usize, f64, u8, Option<f64>, Option<String>)> {
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
        let mut fit = LatentKroneckerFit {
            model: (*self.get()).clone(),
        };
        let r = threads::run(py, || {
            fit_gpytorch_mll_torch(&mut fit, mask.as_deref(), bounds.as_deref(), &options, None)
        })
        .into_py_result()?;
        self.set(fit.model);
        Ok((r.step, r.fval, r.status.value(), r.runtime, r.message))
    }
    #[pyo3(signature=(seed=None))]
    fn sample_all_priors(&self, seed: Option<u64>) {
        use rand::SeedableRng;
        let mut m = (*self.get()).clone();
        let mut rng =
            StdRng::seed_from_u64(seed.unwrap_or_else(botorch_core::utils::sampling::random_seed));
        let raw = m.sample_all_priors(m.raw_parameters(), &mut rng);
        if m.set_raw_parameters(&raw).is_ok() {
            self.set(m);
        }
    }
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(self.get()),
        }
    }
}

use rand::rngs::StdRng;
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLatentKroneckerGP>()
}
