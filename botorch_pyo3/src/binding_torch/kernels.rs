//! Additional graph-preserving kernel primitives.

use super::{require_tensor, transpose, Tensor, TensorPair};
use crate::errors::{unsupported, value_error};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyTuple};

fn tensor_parameter<'py>(
    torch: &Bound<'py, PyModule>,
    parameters: Option<&Bound<'py, PyDict>>,
    name: &str,
) -> PyResult<Tensor<'py>> {
    let value = parameters
        .and_then(|p| p.get_item(name).transpose())
        .transpose()?
        .ok_or_else(|| value_error(format!("Tensor kernel requires parameter {name:?}")))?;
    require_tensor(torch, &value, name)?;
    Ok(value)
}

fn feature_batches<'py>(
    x1: &Tensor<'py>,
    x2: &Tensor<'py>,
    last: bool,
) -> PyResult<TensorPair<'py>> {
    if last {
        Ok((
            transpose(x1)?.call_method1("unsqueeze", (-1,))?,
            transpose(x2)?.call_method1("unsqueeze", (-1,))?,
        ))
    } else {
        Ok((x1.clone(), x2.clone()))
    }
}

fn squared_distance<'py>(a: &Tensor<'py>, b: &Tensor<'py>, diag: bool) -> PyResult<Tensor<'py>> {
    let delta = if diag {
        a.sub(b)?
    } else {
        a.call_method1("unsqueeze", (-2,))?
            .sub(b.call_method1("unsqueeze", (-3,))?)?
    };
    delta.call_method0("square")?.call_method1("sum", (-1,))
}

fn align_scalar<'py>(parameter: Tensor<'py>, diag: bool, last: bool) -> PyResult<Tensor<'py>> {
    let value = if diag {
        parameter
    } else {
        parameter.call_method1("unsqueeze", (-1,))?
    };
    if last {
        value.call_method1("unsqueeze", (-1,))
    } else {
        Ok(value)
    }
}

fn rational_quadratic<'py>(
    torch: &Bound<'py, PyModule>,
    x1: &Tensor<'py>,
    x2: &Tensor<'py>,
    lengthscale: Option<&Tensor<'py>>,
    diag: bool,
    last: bool,
    parameters: Option<&Bound<'py, PyDict>>,
) -> PyResult<Tensor<'py>> {
    let (a, b) = if let Some(ls) = lengthscale {
        require_tensor(torch, ls, "lengthscale")?;
        (x1.div(ls)?, x2.div(ls)?)
    } else {
        (x1.clone(), x2.clone())
    };
    let (a, b) = feature_batches(&a, &b, last)?;
    let squared = squared_distance(&a, &b, diag)?;
    let alpha = align_scalar(tensor_parameter(torch, parameters, "alpha")?, diag, last)?;
    squared
        .div(alpha.mul(2.0)?)?
        .add(1.0)?
        .call_method1("pow", (alpha.neg()?,))
}

fn periodic<'py>(
    torch: &Bound<'py, PyModule>,
    x1: &Tensor<'py>,
    x2: &Tensor<'py>,
    lengthscale: Option<&Tensor<'py>>,
    diag: bool,
    last: bool,
    parameters: Option<&Bound<'py, PyDict>>,
) -> PyResult<Tensor<'py>> {
    let period = tensor_parameter(torch, parameters, "period_length")?;
    let delta = if diag {
        x1.sub(x2)?.div(period)?
    } else {
        x1.call_method1("unsqueeze", (-2,))?
            .sub(x2.call_method1("unsqueeze", (-3,))?)?
            .div(period.call_method1("unsqueeze", (-2,))?)?
    };
    let mut exponent = delta
        .mul(std::f64::consts::PI)?
        .call_method0("sin")?
        .call_method0("square")?
        .mul(-2.0)?;
    if let Some(ls) = lengthscale {
        require_tensor(torch, ls, "lengthscale")?;
        exponent = exponent.div(if diag {
            ls.clone()
        } else {
            ls.call_method1("unsqueeze", (-2,))?
        })?;
    }
    if last {
        exponent = if diag {
            transpose(&exponent)?
        } else {
            exponent.call_method1("movedim", (-1, -3))?
        };
    } else {
        exponent = exponent.call_method1("sum", (-1,))?;
    }
    exponent.call_method0("exp")
}

fn cosine<'py>(
    torch: &Bound<'py, PyModule>,
    x1: &Tensor<'py>,
    x2: &Tensor<'py>,
    diag: bool,
    last: bool,
    parameters: Option<&Bound<'py, PyDict>>,
) -> PyResult<Tensor<'py>> {
    let period = tensor_parameter(torch, parameters, "period_length")?;
    let (a, b) = feature_batches(&x1.div(&period)?, &x2.div(&period)?, last)?;
    let squared = squared_distance(&a, &b, diag)?;
    // exp([[0,-pi²s],[1,0]])[0,0] = cos(pi sqrt(s)). Unlike sqrt's
    // usual clamp, this entire matrix function retains every derivative at s=0.
    let zeros = torch.call_method1("zeros_like", (&squared,))?;
    let ones = torch.call_method1("ones_like", (&squared,))?;
    let row1 = torch.call_method1(
        "stack",
        (
            PyTuple::new(
                torch.py(),
                [zeros.clone(), squared.mul(-std::f64::consts::PI.powi(2))?],
            )?,
            -1,
        ),
    )?;
    let row2 = torch.call_method1("stack", (PyTuple::new(torch.py(), [ones, zeros])?, -1))?;
    let matrix = torch.call_method1("stack", (PyTuple::new(torch.py(), [row1, row2])?, -2))?;
    torch
        .getattr("linalg")?
        .call_method1("matrix_exp", (matrix,))?
        .call_method1("select", (-1, 0))?
        .call_method1("select", (-1, 0))
}

fn dot_kernel<'py>(
    torch: &Bound<'py, PyModule>,
    kind: &str,
    x1: &Tensor<'py>,
    x2: &Tensor<'py>,
    diag: bool,
    last: bool,
    parameters: Option<&Bound<'py, PyDict>>,
) -> PyResult<Tensor<'py>> {
    let a = if kind == "linear" {
        x1.mul(tensor_parameter(torch, parameters, "variance")?)?
    } else {
        x1.clone()
    };
    let (a, b) = feature_batches(&a, x2, last)?;
    let dot = if diag {
        a.mul(b)?.call_method1("sum", (-1,))?
    } else {
        a.matmul(transpose(&b)?)?
    };
    if kind == "linear" {
        return Ok(dot);
    }
    let offset = align_scalar(tensor_parameter(torch, parameters, "offset")?, diag, last)?;
    let power = parameters
        .and_then(|p| p.get_item("power").transpose())
        .transpose()?
        .ok_or_else(|| value_error("Polynomial kernel requires power"))?;
    if !power.is_instance_of::<PyFloat>() && !power.is_instance_of::<PyInt>() {
        return Err(value_error(
            "Polynomial power must be a fixed Python number",
        ));
    }
    let power = power.extract::<f64>()?;
    if !power.is_finite() {
        return Err(value_error("Polynomial power must be finite"));
    }
    dot.add(offset)?.call_method1("pow", (power,))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate<'py>(
    torch: &Bound<'py, PyModule>,
    kind: &str,
    x1: &Tensor<'py>,
    x2: &Tensor<'py>,
    lengthscale: Option<&Tensor<'py>>,
    diag: bool,
    last: bool,
    parameters: Option<&Bound<'py, PyDict>>,
) -> PyResult<Tensor<'py>> {
    let kind = kind.to_ascii_lowercase();
    let allowed: &[&str] = match kind.as_str() {
        "rq" => &["alpha"],
        "periodic" | "cosine" => &["period_length"],
        "linear" => &["variance"],
        "polynomial" => &["offset", "power"],
        _ => {
            return Err(unsupported(format!(
                "Tensor kernel {kind:?} is not implemented"
            )))
        }
    };
    if let Some(parameters) = parameters {
        for (key, _) in parameters.iter() {
            let name = key.extract::<String>()?;
            if !allowed.contains(&name.as_str()) {
                return Err(value_error(format!(
                    "Unknown {kind} kernel parameter {name:?}"
                )));
            }
        }
    }
    if lengthscale.is_some() && matches!(kind.as_str(), "cosine" | "linear" | "polynomial") {
        return Err(value_error(format!(
            "{kind} kernels do not use lengthscale"
        )));
    }
    match kind.as_str() {
        "rq" => rational_quadratic(torch, x1, x2, lengthscale, diag, last, parameters),
        "periodic" => periodic(torch, x1, x2, lengthscale, diag, last, parameters),
        "cosine" => cosine(torch, x1, x2, diag, last, parameters),
        _ => dot_kernel(torch, &kind, x1, x2, diag, last, parameters),
    }
}
