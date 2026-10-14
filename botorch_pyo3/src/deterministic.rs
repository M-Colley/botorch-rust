//! Deterministic models (`botorch.models.deterministic`) as model handles of
//! acquisition functions: `PosteriorMeanModel` (the posterior mean of a
//! botorch_rs model, evaluated in Rust) and `GenericDeterministicModel` (a
//! Python callable `f(X)` on torch tensors, evaluated through the bridge with
//! gradients from `torch.autograd`).
//!
//! Both are deterministic `botorch_core` models (`Model::is_deterministic`):
//! zero covariances, Monte-Carlo samples equal to the mean.

use std::ops::Range;
use std::sync::Arc;

use botorch_core::models::PosteriorMeanModel;
use botorch_core::{
    BotorchError, JointPosteriorGrad, Matrix, Model, PointPosteriors, Posterior, Result, Vector,
};
use pyo3::prelude::*;

use crate::arrays::{float_data, float_values};
use crate::callbacks::{callback_error, rows, PyTensorFunction};
use crate::errors::value_error;
use crate::models::PyModel;

const BRIDGE: &str = "botorch_rs._torch_bridge";

/// `PosteriorMeanModel(model)`: the posterior mean of a model handle (a
/// `SingleTaskGP` or `ModelListGP` snapshot).
#[pyfunction]
pub fn posterior_mean_model(model: PyRef<'_, PyModel>) -> PyModel {
    PyModel {
        inner: Arc::new(PosteriorMeanModel::new(model.inner.clone())),
    }
}

/// `GenericDeterministicModel(f, num_outputs)` on `dim`-dimensional inputs:
/// `f` maps `batch x q x d` tensors to `batch x q x m` tensors.
#[pyfunction]
pub fn generic_deterministic_model(
    callable: Py<PyAny>,
    dim: usize,
    num_outputs: usize,
) -> PyResult<PyModel> {
    if dim == 0 || num_outputs == 0 {
        return Err(value_error(
            "GenericDeterministicModel needs a positive input dimension and number of outputs.",
        ));
    }
    Ok(PyModel {
        inner: Arc::new(PyDeterministicModel {
            f: PyTensorFunction::new(callable),
            dim,
            num_outputs,
        }),
    })
}

/// A deterministic model given by a Python callable on torch tensors.
#[derive(Debug)]
struct PyDeterministicModel {
    f: PyTensorFunction,
    dim: usize,
    num_outputs: usize,
}

impl PyDeterministicModel {
    /// Checks that `f` returned `n·m` values for `n` points.
    fn check_values(&self, values: &[f64], shape: &[usize], n: usize) -> Result<()> {
        let m = self.num_outputs;
        if values.len() != n * m || shape.last() != Some(&m) {
            return Err(BotorchError::DimensionMismatch(format!(
                "GenericDeterministicModel: f returned a tensor of shape {shape:?} for {n} points, \
                 expected `batch_shape x q x {m}` (the outcome dimension must be explicit)"
            )));
        }
        Ok(())
    }

    /// `f(X)` at `X` (`1 x q x d`) with the Jacobian of the values (through
    /// the bridge's `deterministic_model_jacobian`).
    fn jacobian(&self, x: &Matrix) -> Result<(Vec<f64>, Vec<usize>, Vec<f64>)> {
        let shape = vec![1, x.nrows(), self.dim];
        Python::attach(|py| -> PyResult<(Vec<f64>, Vec<usize>, Vec<f64>)> {
            let bridge = py.import(BRIDGE)?;
            let res = bridge.call_method1(
                "deterministic_model_jacobian",
                (self.f.callable(py), float_values(py, &rows(x))?, shape),
            )?;
            let (values, shape, jac): (Bound<'_, PyAny>, Vec<usize>, Bound<'_, PyAny>) =
                res.extract()?;
            Ok((float_data(&values)?, shape, float_data(&jac)?))
        })
        .map_err(callback_error)
    }

    /// The vector-Jacobian product of `f` at `X` (`1 x q x d`) with the
    /// adjoint of its values (`1 x q x m`, row-major).
    fn vjp(&self, x: &Matrix, adjoint: &[f64]) -> Result<Vec<f64>> {
        let shape = vec![1, x.nrows(), self.dim];
        Python::attach(|py| -> PyResult<Vec<f64>> {
            let bridge = py.import(BRIDGE)?;
            let res = bridge.call_method1(
                "deterministic_model_vjp",
                (
                    self.f.callable(py),
                    float_values(py, &rows(x))?,
                    shape,
                    float_values(py, adjoint)?,
                ),
            )?;
            float_data(&res)
        })
        .map_err(callback_error)
    }

    fn check_input(&self, x: &Matrix) -> Result<()> {
        if x.ncols() != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {} columns (d), got {}",
                self.dim,
                x.ncols()
            )));
        }
        Ok(())
    }

    /// The values at the `q` points of `x` (`1 x q x d`) in output-major
    /// order (index `t·q + p`).
    fn mean(&self, x: &Matrix) -> Result<Vector> {
        self.check_input(x)?;
        let (q, m) = (x.nrows(), self.num_outputs);
        let (values, shape, _) = self.f.call(&[(rows(x), vec![1, q, self.dim])], false)?;
        self.check_values(&values, &shape, q)?;
        Ok(Vector::from_fn(q * m, |i, _| values[(i % q) * m + i / q]))
    }
}

impl Model for PyDeterministicModel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn posterior(&self, x: &Matrix, _observation_noise: bool) -> Result<Posterior> {
        let mean = self.mean(x)?;
        let n = mean.len();
        Ok(Posterior::multi_output(
            mean,
            Matrix::zeros(n, n),
            self.num_outputs,
        ))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.check_input(x)?;
        let (q, d, m) = (x.nrows(), self.dim, self.num_outputs);
        let (values, shape, jac) = self.jacobian(x)?;
        self.check_values(&values, &shape, q)?;
        // Value `p·m + t` (row of the Jacobian) is output `t` at point `p`;
        // input `p'·d + k` is coordinate `k` of point `p'`.
        let qd = q * d;
        let mean = Vector::from_fn(q * m, |i, _| values[(i % q) * m + i / q]);
        let dmean = (0..qd)
            .map(|j| Vector::from_fn(q * m, |i, _| jac[((i % q) * m + i / q) * qd + j]))
            .collect();
        Ok(JointPosteriorGrad {
            mean,
            covariance: Matrix::zeros(q * m, q * m),
            dmean,
            dcov: vec![Matrix::zeros(q * m, q * m); qd],
            num_outputs: m,
        })
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows_out: Range<usize>,
        mean_adjoint: &Vector,
        _covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        self.check_input(x)?;
        let (q, d, m) = (x.nrows(), self.dim, self.num_outputs);
        if mean_adjoint.len() != q * m || rows_out.end > q || rows_out.start > rows_out.end {
            return Err(BotorchError::DimensionMismatch(format!(
                "adjoint of length {} or rows {rows_out:?} do not fit {q} points and {m} outputs",
                mean_adjoint.len()
            )));
        }
        // The adjoint in the layout of f's values (`1 x q x m`).
        let adjoint: Vec<f64> = (0..q * m)
            .map(|k| mean_adjoint[(k % m) * q + k / m])
            .collect();
        let grad = self.vjp(x, &adjoint)?;
        if grad.len() != q * d {
            return Err(BotorchError::DimensionMismatch(format!(
                "GenericDeterministicModel: the gradient has {} values, expected {}",
                grad.len(),
                q * d
            )));
        }
        let g = Matrix::from_row_slice(q, d, &grad);
        Ok(g.rows(rows_out.start, rows_out.len()).into_owned())
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.check_input(x)?;
        let (b, m) = (x.nrows(), self.num_outputs);
        let (values, shape, _) = self.f.call(&[(rows(x), vec![b, 1, self.dim])], false)?;
        self.check_values(&values, &shape, b)?;
        Ok(PointPosteriors {
            mean: Matrix::from_row_slice(b, m, &values),
            covariance: Matrix::zeros(b, m * m),
        })
    }
}

/// Registers the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(posterior_mean_model, m)?)?;
    m.add_function(wrap_pyfunction!(generic_deterministic_model, m)?)?;
    Ok(())
}
