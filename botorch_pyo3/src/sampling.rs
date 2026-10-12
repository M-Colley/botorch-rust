//! Quasi-random sampling: `SobolQMCNormalSampler` (`botorch.sampling`),
//! `draw_sobol_samples` and `draw_sobol_normal_samples`
//! (`botorch.utils.sampling`).

use std::sync::Arc;

use botorch_core::utils::sampling::{draw_sobol_normal_samples, draw_sobol_samples};
use botorch_core::SobolQMCNormalSampler;
use pyo3::prelude::*;

use crate::arrays::{matrix_out, stack_out, Array, ArrayBuf};
use crate::errors::{value_error, IntoPyResult};
use crate::threads;

/// Quasi-MC sampler of standard-normal base samples from a scrambled Sobol
/// sequence (BoTorch's `SobolQMCNormalSampler`).
#[pyclass(
    frozen,
    name = "SobolQMCNormalSampler",
    module = "botorch_rs._botorch_rs"
)]
pub struct PySobolQMCNormalSampler {
    /// The sampler (shared with the acquisition functions that use it).
    pub inner: Arc<SobolQMCNormalSampler>,
}

#[pymethods]
impl PySobolQMCNormalSampler {
    /// A sampler drawing `num_samples` samples; `seed=None` draws a random
    /// seed.
    #[new]
    #[pyo3(signature = (num_samples, seed=None))]
    fn new(num_samples: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(SobolQMCNormalSampler::new(num_samples, seed).into_py_result()?),
        })
    }

    /// Number of MC samples.
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    /// The seed of the Sobol sequence.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// The base samples (`num_samples x dim`) for `dim` jointly sampled points.
    fn base_samples(&self, dim: usize) -> PyResult<ArrayBuf> {
        if dim == 0 {
            return Err(value_error("dim must be positive"));
        }
        let z = self.inner.base_samples(dim).into_py_result()?;
        Ok(matrix_out(&z).into())
    }

    /// Installs explicit base samples (`num_samples x dim`) for `dim` jointly
    /// sampled points.
    fn set_base_samples(&self, base_samples: Array) -> PyResult<()> {
        let z = base_samples.matrix("base_samples")?;
        self.inner.set_base_samples(z).into_py_result()
    }

    /// An independent copy with the same seed and base samples (e.g.
    /// `qHypervolumeKnowledgeGradient.cost_sampler`).
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        Self {
            inner: Arc::new(self.inner.as_ref().clone()),
        }
    }

    fn __copy__(&self) -> Self {
        Self {
            inner: Arc::new(self.inner.as_ref().clone()),
        }
    }
}

/// `draw_sobol_samples(bounds, n, q, seed)`: `n` scrambled Sobol `q`-batches
/// in the box `bounds` (`2 x d`); returns an array of shape `n x q x d`.
#[pyfunction]
#[pyo3(name = "draw_sobol_samples", signature = (bounds, n, q, seed=None))]
pub fn draw_sobol_samples_py(
    py: Python<'_>,
    bounds: Array,
    n: usize,
    q: usize,
    seed: Option<u64>,
) -> PyResult<ArrayBuf> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let batches = threads::run(py, || draw_sobol_samples(&bounds, n, q, seed)).into_py_result()?;
    Ok(stack_out(&batches, &[n], q, d).into())
}

/// `draw_sobol_normal_samples(d, n, seed)`: `n x d` quasi-random standard
/// normal samples.
#[pyfunction]
#[pyo3(name = "draw_sobol_normal_samples", signature = (d, n, seed=None))]
pub fn draw_sobol_normal_samples_py(
    py: Python<'_>,
    d: usize,
    n: usize,
    seed: Option<u64>,
) -> PyResult<ArrayBuf> {
    let z = threads::run(py, || draw_sobol_normal_samples(d, n, seed)).into_py_result()?;
    Ok(matrix_out(&z).into())
}
