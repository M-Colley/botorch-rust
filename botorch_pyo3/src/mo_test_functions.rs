//! Multi-objective test problems (`botorch.test_functions.multi_objective` and
//! `multi_objective_multi_fidelity`).

use botorch_core::test_functions::{MultiObjectiveKind, MultiObjectiveTestProblem, NoiseStd};
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayOut};
use crate::errors::{dimension_error, unsupported, IntoPyResult};
use crate::threads;

/// A noise standard deviation from Python: a number (one for all outputs) or
/// a list (one per output).
#[derive(FromPyObject)]
pub enum NoiseArg {
    Scalar(f64),
    PerOutput(Vec<f64>),
}

impl From<NoiseArg> for NoiseStd {
    fn from(arg: NoiseArg) -> Self {
        match arg {
            NoiseArg::Scalar(s) => NoiseStd::Scalar(s),
            NoiseArg::PerOutput(v) => NoiseStd::PerOutput(v),
        }
    }
}

/// A multi-objective test problem with BoTorch's `forward`, `evaluate_true`,
/// `evaluate_slack`, `is_feasible` and `gen_pareto_front` semantics.
#[pyclass(
    frozen,
    name = "MultiObjectiveTestProblem",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyMultiObjectiveTestProblem {
    inner: MultiObjectiveTestProblem,
}

/// The names of the available multi-objective test problems.
#[pyfunction]
pub fn multi_objective_problem_names() -> Vec<&'static str> {
    MultiObjectiveKind::ALL.iter().map(|k| k.name()).collect()
}

/// `(leading shape, row-major data, number of columns)` of an array of points
/// (`... x d`).
fn points<'a>(x: &'a Array, name: &str) -> PyResult<(Vec<usize>, &'a [f64], usize)> {
    let Some((&d, batch)) = x.1.split_last() else {
        return Err(dimension_error(format!(
            "Expected `{name}` to have shape `(batch_shape) x n x d`, got a scalar."
        )));
    };
    if x.1.iter().product::<usize>() != x.0.len() {
        return Err(dimension_error(format!(
            "{name}: {} values do not match the shape {:?}",
            x.0.len(),
            x.1
        )));
    }
    Ok((batch.to_vec(), &x.0, d))
}

/// `values` (row-major, `width` per point) with the shape `(*batch, width)`.
fn with_shape(values: Vec<f64>, batch: &[usize], width: usize) -> ArrayOut {
    let mut shape = batch.to_vec();
    shape.push(width);
    (values, shape)
}

#[pymethods]
impl PyMultiObjectiveTestProblem {
    /// The problem `name` (a BoTorch class name) with the input dimension
    /// `dim` and `num_objectives` (`None`: fixed or BoTorch's default),
    /// observation noise of the objectives and the constraint slacks (a number
    /// or one value per output), `negate`, the seed of the noise generator
    /// (used for nested-list inputs) and the bounds against which inputs are
    /// validated (`2 x d`; `None`: BoTorch's bounds in float64).
    #[new]
    #[pyo3(signature = (name, dim=None, num_objectives=None, noise_std=None, constraint_noise_std=None, negate=false, seed=None, bounds=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        dim: Option<i64>,
        num_objectives: Option<i64>,
        noise_std: Option<NoiseArg>,
        constraint_noise_std: Option<NoiseArg>,
        negate: bool,
        seed: Option<u64>,
        bounds: Option<Array>,
    ) -> PyResult<Self> {
        let kind = MultiObjectiveKind::from_name(name)
            .ok_or_else(|| unsupported(format!("Unknown test problem {name:?}.")))?;
        let (d, m) = kind.resolve_sizes(dim, num_objectives).into_py_result()?;
        let mut p = MultiObjectiveTestProblem::new(kind, Some(d), Some(m))
            .into_py_result()?
            .with_negate(negate);
        if let Some(s) = noise_std {
            p = p.with_noise_std(s.into()).into_py_result()?;
        }
        if let Some(s) = constraint_noise_std {
            p = p.with_constraint_noise_std(s.into()).into_py_result()?;
        }
        if let Some(seed) = seed {
            p = p.with_seed(seed);
        }
        if let Some(b) = bounds {
            p = p.with_bounds(b.matrix("bounds")?).into_py_result()?;
        }
        Ok(Self { inner: p })
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

    /// The number of objectives.
    #[getter]
    fn num_objectives(&self) -> usize {
        self.inner.num_objectives()
    }

    /// The number of outcome constraints (0 for unconstrained problems).
    #[getter]
    fn num_constraints(&self) -> usize {
        self.inner.num_constraints()
    }

    /// The `2 x d` bounds.
    #[getter]
    fn bounds(&self) -> ArrayOut {
        matrix_out(self.inner.bounds())
    }

    /// Whether the objectives are negated.
    #[getter]
    fn negate(&self) -> bool {
        self.inner.negate()
    }

    /// The reference point (negated if `negate`).
    #[getter]
    fn ref_point(&self) -> Vec<f64> {
        self.inner.ref_point()
    }

    /// BoTorch's `_ref_point` (not negated).
    #[getter]
    fn raw_ref_point(&self) -> Vec<f64> {
        self.inner.raw_ref_point()
    }

    /// The maximum hypervolume, if known.
    #[getter]
    fn max_hv(&self) -> Option<f64> {
        self.inner.max_hv()
    }

    /// Whether `gen_pareto_front` is available.
    #[getter]
    fn has_pareto_front(&self) -> bool {
        self.inner.kind().has_pareto_front()
    }

    /// How botorch-rust deviates from BoTorch main for this problem, if it does.
    #[getter]
    fn deviation(&self) -> Option<&'static str> {
        self.inner.kind().deviation()
    }

    /// Noiseless, un-negated objectives at `X` (`... x d` -> `... x M`).
    fn evaluate_true(&self, py: Python<'_>, x: Array) -> PyResult<ArrayOut> {
        let (batch, data, d) = points(&x, "X")?;
        let f = threads::run(py, || self.inner.evaluate_true_rows(data, d)).into_py_result()?;
        Ok(with_shape(f, &batch, self.inner.num_objectives()))
    }

    /// Objectives with observation noise (if `noise`; drawn from the problem's
    /// seeded generator) and negation at `X`.
    #[pyo3(signature = (x, noise=true))]
    fn forward(&self, py: Python<'_>, x: Array, noise: bool) -> PyResult<ArrayOut> {
        let (batch, data, d) = points(&x, "X")?;
        let f = threads::run(py, || self.inner.forward_rows(data, d, noise)).into_py_result()?;
        Ok(with_shape(f, &batch, self.inner.num_objectives()))
    }

    /// Noiseless constraint slacks at `X` (`... x d` -> `... x n_c`).
    fn evaluate_slack_true(&self, py: Python<'_>, x: Array) -> PyResult<ArrayOut> {
        let (batch, data, d) = points(&x, "X")?;
        let c =
            threads::run(py, || self.inner.evaluate_slack_true_rows(data, d)).into_py_result()?;
        Ok(with_shape(c, &batch, self.inner.num_constraints()))
    }

    /// Constraint slacks with observation noise (if `noise`).
    #[pyo3(signature = (x, noise=true))]
    fn evaluate_slack(&self, py: Python<'_>, x: Array, noise: bool) -> PyResult<ArrayOut> {
        let (batch, data, d) = points(&x, "X")?;
        let c =
            threads::run(py, || self.inner.evaluate_slack_rows(data, d, noise)).into_py_result()?;
        Ok(with_shape(c, &batch, self.inner.num_constraints()))
    }

    /// Whether all (noisy, if `noise`) slacks at each point of `X` are
    /// non-negative: `(flags, leading shape)`.
    #[pyo3(signature = (x, noise=true))]
    fn is_feasible(
        &self,
        py: Python<'_>,
        x: Array,
        noise: bool,
    ) -> PyResult<(Vec<bool>, Vec<usize>)> {
        let (batch, data, d) = points(&x, "X")?;
        let flags =
            threads::run(py, || self.inner.is_feasible_rows(data, d, noise)).into_py_result()?;
        Ok((flags, batch))
    }

    /// `n` points of the Pareto front (`n x M`, negated if `negate`).
    #[pyo3(signature = (n, seed=None))]
    fn gen_pareto_front(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<ArrayOut> {
        let f = threads::run(py, || self.inner.gen_pareto_front(n, seed)).into_py_result()?;
        Ok(matrix_out(&f))
    }
}
