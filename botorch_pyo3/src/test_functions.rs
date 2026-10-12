//! Synthetic test functions (`botorch.test_functions.synthetic`).

use botorch_core::test_functions::{SyntheticTestFunction, TestFunctionKind};
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{unsupported, IntoPyResult};

/// A synthetic test function with BoTorch's `forward` / `evaluate_true`
/// semantics.
#[pyclass(
    frozen,
    name = "SyntheticTestFunction",
    module = "botorch_rs._botorch_rs"
)]
pub struct PySyntheticTestFunction {
    inner: SyntheticTestFunction,
}

/// The names of the available test functions.
#[pyfunction]
pub fn test_function_names() -> Vec<&'static str> {
    TestFunctionKind::ALL.iter().map(|k| k.name()).collect()
}

impl PySyntheticTestFunction {
    fn evaluate(&self, x: &Array, noise: Option<bool>) -> PyResult<ArrayBuf> {
        let (batch, points) = x.points("X")?;
        let values = match noise {
            None => self.inner.evaluate_true(&points),
            Some(noise) => self.inner.forward(&points, noise),
        }
        .into_py_result()?;
        Ok(ArrayBuf(values.as_slice().to_vec(), batch))
    }
}

#[pymethods]
impl PySyntheticTestFunction {
    /// The test function `name` (a BoTorch class name) with dimension `dim`
    /// (`None`: the function's default), Shekel's `m`, observation noise
    /// `noise_std`, `negate`, custom `bounds` (`2 x d`) and the seed of the
    /// noise generator.
    #[new]
    #[pyo3(signature = (name, dim=None, m=None, noise_std=None, negate=false, bounds=None, seed=None))]
    fn new(
        name: &str,
        dim: Option<usize>,
        m: Option<usize>,
        noise_std: Option<f64>,
        negate: bool,
        bounds: Option<Array>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let kind = TestFunctionKind::ALL
            .into_iter()
            .find(|k| k.name() == name)
            .ok_or_else(|| unsupported(format!("Unknown test function {name:?}.")))?;
        let mut f = if kind == TestFunctionKind::Shekel {
            SyntheticTestFunction::shekel(m.unwrap_or(10))
        } else {
            SyntheticTestFunction::new(kind, dim)
        }
        .into_py_result()?
        .with_negate(negate);
        if let Some(s) = noise_std {
            f = f.with_noise_std(s).into_py_result()?;
        }
        if let Some(b) = bounds {
            f = f.with_bounds(b.matrix("bounds")?).into_py_result()?;
        }
        if let Some(seed) = seed {
            f = f.with_seed(seed);
        }
        Ok(Self { inner: f })
    }

    /// The BoTorch class name.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// The input dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Shekel's number of terms (`None` for other functions).
    #[getter]
    fn m(&self) -> Option<usize> {
        self.inner.shekel_m()
    }

    /// The `2 x d` bounds.
    #[getter]
    fn bounds(&self) -> ArrayOut {
        matrix_out(self.inner.bounds())
    }

    /// Whether the function is negated by `forward`.
    #[getter]
    fn negate(&self) -> bool {
        self.inner.negate()
    }

    /// The observation-noise standard deviation.
    #[getter]
    fn noise_std(&self) -> Option<f64> {
        self.inner.noise_std()
    }

    /// The optimal value (negated if `negate`), if known.
    #[getter]
    fn optimal_value(&self) -> Option<f64> {
        self.inner.optimal_value()
    }

    /// The known optimizers (one per row), if any.
    #[getter]
    fn optimizers(&self) -> Option<ArrayOut> {
        self.inner.optimizers().as_ref().map(matrix_out)
    }

    /// Whether the problem is a minimization problem (after `negate`).
    #[getter]
    fn is_minimization_problem(&self) -> bool {
        self.inner.is_minimization_problem()
    }

    /// Noiseless, un-negated values at the points `X` (`... x d`).
    fn evaluate_true(&self, x: Array) -> PyResult<ArrayBuf> {
        self.evaluate(&x, None)
    }

    /// Values with observation noise (if `noise`) and negation at `X`.
    #[pyo3(signature = (x, noise=true))]
    fn forward(&self, x: Array, noise: bool) -> PyResult<ArrayBuf> {
        self.evaluate(&x, Some(noise))
    }
}
