//! Native Gaussian KL and fitting objectives for empirical Bayes.

use botorch_core::fit::{FitOptions, FitStatus};
use botorch_core::models::em_kl::{kl_divergence_mvn, GaussianKlTarget, KernelPriorKl};
use botorch_core::Vector;
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf};
use crate::errors::{dimension_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::specs::PyConstraint;

type KlGradients = (ArrayBuf, ArrayBuf, ArrayBuf, ArrayBuf);
type KlFitSummary = (Vec<f64>, f64, usize, usize, &'static str, String, usize);

fn vector(array: Array, name: &str) -> PyResult<Vector> {
    if array.1.as_slice() != [array.0.len()] {
        return Err(dimension_error(format!("{name} must be a vector")));
    }
    Ok(Vector::from_vec(array.0))
}

#[pyfunction]
#[pyo3(signature = (mu_p, sigma_p, mu_q, sigma_q, gradients=false))]
pub fn gaussian_kl(
    py: Python<'_>,
    mu_p: Array,
    sigma_p: Array,
    mu_q: Array,
    sigma_q: Array,
    gradients: bool,
) -> PyResult<(f64, Option<KlGradients>)> {
    let a = vector(mu_p, "mu_p")?;
    let b = sigma_p.matrix("Sigma_p")?;
    let c = vector(mu_q, "mu_q")?;
    let d = sigma_q.matrix("Sigma_q")?;
    crate::threads::run(py, || {
        if gradients {
            let g = kl_divergence_mvn(&a, &b, &c, &d)?;
            let n = a.len();
            Ok((
                g.value,
                Some((
                    ArrayBuf(g.mean_p.as_slice().to_vec(), vec![n]),
                    matrix_out(&g.covariance_p).into(),
                    ArrayBuf(g.mean_q.as_slice().to_vec(), vec![n]),
                    matrix_out(&g.covariance_q).into(),
                )),
            ))
        } else {
            let target = GaussianKlTarget::new(a, b)?;
            target
                .value_q_gradient(&c, &d)
                .map(|(value, _, _)| (value, None))
        }
    })
    .into_py_result()
}

#[pyclass(frozen, name = "KernelPriorKl", module = "botorch_rs._botorch_rs")]
pub struct PyKernelPriorKl {
    inner: KernelPriorKl,
}

#[pymethods]
impl PyKernelPriorKl {
    #[new]
    #[pyo3(signature = (node, points, mean, covariance, learn_mean=true, mean_constraint=None))]
    fn new(
        py: Python<'_>,
        node: PyRef<'_, PyKernelNode>,
        points: Array,
        mean: Array,
        covariance: Array,
        learn_mean: bool,
        mean_constraint: Option<PyRef<'_, PyConstraint>>,
    ) -> PyResult<Self> {
        let node = node.inner.clone();
        let points = points.matrix("X_ref")?;
        let mean = vector(mean, "mu_em")?;
        let covariance = covariance.matrix("Sigma_em")?;
        let constraint = mean_constraint.map(|c| c.inner);
        crate::threads::run(py, || {
            KernelPriorKl::new(node, points, mean, covariance, learn_mean, constraint)
        })
        .map(|inner| Self { inner })
        .into_py_result()
    }

    fn value_and_grad(&self, py: Python<'_>, raw: Vec<f64>) -> PyResult<(f64, Vec<f64>)> {
        crate::threads::run(py, || self.inner.value_and_grad(&raw)).into_py_result()
    }

    #[pyo3(signature=(raw,maxcor=None,pgtol=None,ftol=None,maxiter=None,maxfun=None,maxls=None,max_attempts=None,seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        raw: Vec<f64>,
        maxcor: Option<usize>,
        pgtol: Option<f64>,
        ftol: Option<f64>,
        maxiter: Option<usize>,
        maxfun: Option<usize>,
        maxls: Option<usize>,
        max_attempts: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<KlFitSummary> {
        let default = FitOptions::default();
        let options = FitOptions {
            maxcor: maxcor.unwrap_or(default.maxcor),
            pgtol: pgtol.unwrap_or(default.pgtol),
            ftol: ftol.unwrap_or(default.ftol),
            maxiter: maxiter.unwrap_or(default.maxiter),
            maxfun: maxfun.unwrap_or(default.maxfun),
            maxls: maxls.unwrap_or(default.maxls),
            max_attempts: max_attempts.unwrap_or(default.max_attempts),
            seed: seed.unwrap_or(default.seed),
        };
        crate::threads::run(py, || self.inner.fit(&raw, options))
            .map(|(r, raw)| {
                let status = match r.status {
                    FitStatus::Success => "SUCCESS",
                    FitStatus::Stopped => "STOPPED",
                    FitStatus::Failure => "FAILURE",
                };
                (
                    raw,
                    r.loss,
                    r.iterations,
                    r.evaluations,
                    status,
                    r.message,
                    r.attempts,
                )
            })
            .into_py_result()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussian_kl, m)?)?;
    m.add_class::<PyKernelPriorKl>()?;
    Ok(())
}
