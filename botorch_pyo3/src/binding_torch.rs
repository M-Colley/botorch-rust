//! Graph-preserving Torch tensor algebra, dispatched without a libtorch ABI.
//!
//! Tensor values stay on their original device. PyO3 dispatches ordinary Torch
//! operations, so Torch owns both the numerical kernels and the autograd graph,
//! including higher derivatives. Torch is imported only when a function is used.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{dimension_error, unsupported, value_error};

type Tensor<'py> = Bound<'py, PyAny>;
type TensorPair<'py> = (Tensor<'py>, Tensor<'py>);

fn torch_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("torch")
}

fn require_tensor(torch: &Bound<'_, PyModule>, tensor: &Tensor<'_>, name: &str) -> PyResult<()> {
    if !tensor.is_instance(&torch.getattr("Tensor")?)? {
        return Err(value_error(format!("{name} must be a Torch tensor")));
    }
    if !tensor
        .call_method0("is_floating_point")?
        .extract::<bool>()?
    {
        return Err(value_error(format!(
            "{name} must be a real floating-point tensor"
        )));
    }
    Ok(())
}

fn shape(tensor: &Tensor<'_>) -> PyResult<Vec<usize>> {
    tensor.getattr("shape")?.extract()
}

fn transpose<'py>(tensor: &Tensor<'py>) -> PyResult<Tensor<'py>> {
    tensor.call_method1("transpose", (-1, -2))
}

fn diagonal<'py>(tensor: &Tensor<'py>) -> PyResult<Tensor<'py>> {
    tensor.call_method1("diagonal", (0, -2, -1))
}

fn add_diagonal<'py>(
    torch: &Bound<'py, PyModule>,
    matrix: &Tensor<'py>,
    values: &Tensor<'py>,
    jitter: f64,
) -> PyResult<Tensor<'py>> {
    let zeros = torch.call_method1("zeros_like", (diagonal(matrix)?,))?;
    let diag = values.add(zeros)?.add(jitter)?.call_method0("diag_embed")?;
    matrix.add(diag)
}

fn validate_jitter(jitter: f64) -> PyResult<()> {
    if !jitter.is_finite() || jitter < 0.0 {
        return Err(value_error("jitter must be finite and nonnegative"));
    }
    Ok(())
}

fn check_training_shapes(targets: &Tensor<'_>, covariance: &Tensor<'_>) -> PyResult<usize> {
    let ys = shape(targets)?;
    let ks = shape(covariance)?;
    let Some(&n) = ys.last() else {
        return Err(dimension_error("train_y must have shape batch_shape x n"));
    };
    if ks.len() < 2 || ks[ks.len() - 2..] != [n, n] {
        return Err(dimension_error(format!(
            "k_train must end in {n} x {n}; got {ks:?}"
        )));
    }
    Ok(n)
}

/// Dense RBF or Matern covariance using only differentiable Torch operations.
#[pyfunction]
#[pyo3(signature = (kind, x1, x2, lengthscale=None, nu=2.5, diag=false, last_dim_is_batch=false))]
#[allow(clippy::too_many_arguments)]
fn tensor_kernel<'py>(
    py: Python<'py>,
    kind: &str,
    x1: &Tensor<'py>,
    x2: &Tensor<'py>,
    lengthscale: Option<&Tensor<'py>>,
    nu: f64,
    diag: bool,
    last_dim_is_batch: bool,
) -> PyResult<Tensor<'py>> {
    let torch = torch_module(py)?;
    require_tensor(&torch, x1, "x1")?;
    require_tensor(&torch, x2, "x2")?;
    let (s1, s2) = (shape(x1)?, shape(x2)?);
    if s1.len() < 2 || s2.len() < 2 || s1.last() != s2.last() {
        return Err(dimension_error("Kernel inputs must end in n x d and m x d"));
    }
    if diag && s1[s1.len() - 2] != s2[s2.len() - 2] {
        return Err(dimension_error(
            "diag=True requires equal numbers of kernel input rows",
        ));
    }
    let (mut a, mut b) = if let Some(ls) = lengthscale {
        require_tensor(&torch, ls, "lengthscale")?;
        (x1.div(ls)?, x2.div(ls)?)
    } else {
        (x1.clone(), x2.clone())
    };
    if last_dim_is_batch {
        a = transpose(&a)?.call_method1("unsqueeze", (-1,))?;
        b = transpose(&b)?.call_method1("unsqueeze", (-1,))?;
    }
    let difference = if diag {
        a.sub(b)?
    } else {
        a.call_method1("unsqueeze", (-2,))?
            .sub(b.call_method1("unsqueeze", (-3,))?)?
    };
    let squared = difference
        .call_method0("square")?
        .call_method1("sum", (-1,))?;
    match kind.to_ascii_lowercase().as_str() {
        "rbf" => squared.mul(-0.5)?.call_method0("exp"),
        "matern" => {
            // Avoid sqrt(0)'s undefined adjoints in the unselected branch. At
            // coincidence use the exact Taylor terms required for the existing
            // derivatives: Matern-3/2 is C2, Matern-5/2 is C4 in the inputs.
            let radius = squared
                .call_method1("clamp_min", (1e-30,))?
                .call_method0("sqrt")?;
            let (full, at_zero) = if nu == 0.5 {
                (
                    radius.neg()?.call_method0("exp")?,
                    squared.mul(0.0)?.add(1.0)?,
                )
            } else if nu == 1.5 {
                let scaled = radius.mul(3.0_f64.sqrt())?;
                (
                    scaled.add(1.0)?.mul(scaled.neg()?.call_method0("exp")?)?,
                    squared.mul(-1.5)?.add(1.0)?,
                )
            } else if nu == 2.5 {
                let scaled = radius.mul(5.0_f64.sqrt())?;
                (
                    scaled
                        .add(1.0)?
                        .add(squared.mul(5.0 / 3.0)?)?
                        .mul(scaled.neg()?.call_method0("exp")?)?,
                    squared
                        .mul(-5.0 / 6.0)?
                        .add(1.0)?
                        .add(squared.call_method0("square")?.mul(25.0 / 24.0)?)?,
                )
            } else {
                return Err(unsupported(
                    "Tensor Matern kernels require nu in {0.5, 1.5, 2.5}",
                ));
            };
            torch.call_method1(
                "where",
                (squared.call_method1("eq", (0.0,))?, at_zero, full),
            )
        }
        _ => Err(unsupported(format!(
            "Tensor kernel {kind:?} is not implemented"
        ))),
    }
}

/// Condition a Gaussian using covariance blocks; independent outputs are batches.
#[pyfunction]
#[pyo3(signature = (train_y, k_train, k_train_query, k_query, train_mean, query_mean, train_noise, jitter=0.0, observation_noise=None))]
#[allow(clippy::too_many_arguments)]
fn tensor_exact_gp_posterior<'py>(
    py: Python<'py>,
    train_y: &Tensor<'py>,
    k_train: &Tensor<'py>,
    k_train_query: &Tensor<'py>,
    k_query: &Tensor<'py>,
    train_mean: &Tensor<'py>,
    query_mean: &Tensor<'py>,
    train_noise: &Tensor<'py>,
    jitter: f64,
    observation_noise: Option<&Tensor<'py>>,
) -> PyResult<TensorPair<'py>> {
    validate_jitter(jitter)?;
    let torch = torch_module(py)?;
    for (name, tensor) in [
        ("train_y", train_y),
        ("k_train", k_train),
        ("k_train_query", k_train_query),
        ("k_query", k_query),
        ("train_mean", train_mean),
        ("query_mean", query_mean),
        ("train_noise", train_noise),
    ] {
        require_tensor(&torch, tensor, name)?;
    }
    let n = check_training_shapes(train_y, k_train)?;
    let (cross, query) = (shape(k_train_query)?, shape(k_query)?);
    if cross.len() < 2
        || query.len() < 2
        || cross[cross.len() - 2] != n
        || cross[cross.len() - 1] != query[query.len() - 1]
        || query[query.len() - 2] != query[query.len() - 1]
    {
        return Err(dimension_error(
            "Covariance blocks must end in n x n, n x q, q x q",
        ));
    }
    let covariance = add_diagonal(&torch, k_train, train_noise, jitter)?;
    // No hidden retry/jitter policy: a failed Cholesky raises Torch's own error.
    let linalg = torch.getattr("linalg")?;
    let lower = linalg.call_method1("cholesky", (&covariance,))?;
    let residual = train_y.sub(train_mean)?.call_method1("unsqueeze", (-1,))?;
    let alpha = torch.call_method1("cholesky_solve", (residual, &lower))?;
    let mean = transpose(k_train_query)?
        .matmul(alpha)?
        .call_method1("squeeze", (-1,))?
        .add(query_mean)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("upper", false)?;
    let solved = linalg.call_method("solve_triangular", (&lower, k_train_query), Some(&kwargs))?;
    let mut covariance = k_query.sub(transpose(&solved)?.matmul(&solved)?)?;
    if let Some(noise) = observation_noise {
        require_tensor(&torch, noise, "observation_noise")?;
        covariance = add_diagonal(&torch, &covariance, noise, 0.0)?;
    }
    Ok((mean, covariance))
}

/// Exact Gaussian log likelihood, optionally divided by the training-set size.
#[pyfunction]
#[pyo3(signature = (train_y, k_train, train_mean, train_noise, jitter=0.0, normalize=false))]
fn tensor_exact_mll<'py>(
    py: Python<'py>,
    train_y: &Tensor<'py>,
    k_train: &Tensor<'py>,
    train_mean: &Tensor<'py>,
    train_noise: &Tensor<'py>,
    jitter: f64,
    normalize: bool,
) -> PyResult<Tensor<'py>> {
    validate_jitter(jitter)?;
    let torch = torch_module(py)?;
    for (name, tensor) in [
        ("train_y", train_y),
        ("k_train", k_train),
        ("train_mean", train_mean),
        ("train_noise", train_noise),
    ] {
        require_tensor(&torch, tensor, name)?;
    }
    let n = check_training_shapes(train_y, k_train)?;
    if normalize && n == 0 {
        return Err(value_error(
            "Cannot normalize a log likelihood with no training points",
        ));
    }
    let covariance = add_diagonal(&torch, k_train, train_noise, jitter)?;
    let lower = torch
        .getattr("linalg")?
        .call_method1("cholesky", (covariance,))?;
    let residual = train_y.sub(train_mean)?;
    let alpha = torch
        .call_method1(
            "cholesky_solve",
            (residual.call_method1("unsqueeze", (-1,))?, &lower),
        )?
        .call_method1("squeeze", (-1,))?;
    let quadratic = residual.mul(alpha)?.call_method1("sum", (-1,))?;
    let logdet_half = diagonal(&lower)?
        .call_method0("log")?
        .call_method1("sum", (-1,))?;
    let value = quadratic
        .mul(-0.5)?
        .sub(logdet_half)?
        .sub(0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln())?;
    if normalize {
        value.div(n as f64)
    } else {
        Ok(value)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tensor_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_exact_gp_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_exact_mll, m)?)?;
    Ok(())
}
