//! Broadcast-aware supplied-factor HigherOrderGP variance evaluation.

use botorch_core::models::higher_order_gp::{
    posterior_variances_factors_vjp, posterior_variances_from_factors,
};
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

use crate::arrays::{broadcast_indices, broadcast_shapes, matrix_out, Array, ArrayBuf};
use crate::errors::{dimension_error, IntoPyResult};
use crate::threads;

struct Layout {
    factors: Vec<Vec<Matrix>>,
    indices: Vec<Vec<usize>>,
    noise_indices: Vec<usize>,
    batch: Vec<usize>,
    outputs: usize,
}

fn layout(factors: &[Array], train: usize, noise: &Array) -> PyResult<Layout> {
    if factors.is_empty() || train == 0 {
        return Err(dimension_error(
            "Need covariance factors and a positive training size",
        ));
    }
    if noise.1.iter().product::<usize>() != noise.0.len() || noise.0.is_empty() {
        return Err(dimension_error(
            "Noise values must match a nonempty batch shape",
        ));
    }
    let mut batch = noise.1.clone();
    let mut matrices = Vec::with_capacity(factors.len());
    let mut shapes = Vec::with_capacity(factors.len());
    let mut outputs = 1usize;
    for (i, factor) in factors.iter().enumerate() {
        let (shape, mats) = factor.batches("factor")?;
        let dims = &factor.1[factor.1.len() - 2..];
        if dims[0] != dims[1] || dims[0] == 0 || (i == 0 && dims[0] < train) {
            return Err(dimension_error(
                "Factors must be square and include all training inputs",
            ));
        }
        outputs *= if i == 0 { dims[0] - train } else { dims[0] };
        batch = broadcast_shapes(&batch, &shape).ok_or_else(|| {
            dimension_error("Covariance factor and noise batches do not broadcast")
        })?;
        shapes.push(shape);
        matrices.push(mats);
    }
    Ok(Layout {
        factors: matrices,
        indices: shapes
            .iter()
            .map(|s| broadcast_indices(s, &batch))
            .collect(),
        noise_indices: broadcast_indices(&noise.1, &batch),
        batch,
        outputs,
    })
}

#[pyfunction]
fn higher_order_variances(
    py: Python<'_>,
    factors: Vec<Array>,
    num_train: usize,
    noise: Array,
) -> PyResult<ArrayBuf> {
    let layout = layout(&factors, num_train, &noise)?;
    let values = threads::run(py, || {
        let mut out = Vec::new();
        for b in 0..layout.noise_indices.len() {
            let fs: Vec<_> = layout
                .factors
                .iter()
                .zip(&layout.indices)
                .map(|(f, i)| f[i[b]].clone())
                .collect();
            out.extend(
                posterior_variances_from_factors(&fs, num_train, noise.0[layout.noise_indices[b]])?
                    .iter()
                    .copied(),
            );
        }
        Ok::<_, botorch_core::BotorchError>(out)
    })
    .into_py_result()?;
    let mut shape = layout.batch;
    shape.push(layout.outputs);
    Ok(ArrayBuf(values, shape))
}

#[pyfunction]
fn higher_order_variances_vjp(
    py: Python<'_>,
    factors: Vec<Array>,
    num_train: usize,
    noise: Array,
    adjoint: Array,
) -> PyResult<(Vec<ArrayBuf>, ArrayBuf)> {
    let layout = layout(&factors, num_train, &noise)?;
    let mut expected = layout.batch.clone();
    expected.push(layout.outputs);
    if adjoint.1 != expected || adjoint.0.len() != expected.iter().product::<usize>() {
        return Err(dimension_error(
            "Variance adjoint must match the broadcast output shape",
        ));
    }
    let (gradients, noise_gradient) = threads::run(py, || {
        let mut gradients: Vec<Vec<Matrix>> = layout
            .factors
            .iter()
            .map(|fs| {
                fs.iter()
                    .map(|f| Matrix::zeros(f.nrows(), f.ncols()))
                    .collect()
            })
            .collect();
        let mut noise_gradient = vec![0.0; noise.0.len()];
        for b in 0..layout.noise_indices.len() {
            let fs: Vec<_> = layout
                .factors
                .iter()
                .zip(&layout.indices)
                .map(|(f, i)| f[i[b]].clone())
                .collect();
            let start = b * layout.outputs;
            let weights = Vector::from_column_slice(&adjoint.0[start..start + layout.outputs]);
            let (g, ng) = posterior_variances_factors_vjp(
                &fs,
                num_train,
                noise.0[layout.noise_indices[b]],
                &weights,
            )?;
            for (k, gradient) in g.iter().enumerate() {
                gradients[k][layout.indices[k][b]] += gradient;
            }
            noise_gradient[layout.noise_indices[b]] += ng;
        }
        Ok::<_, botorch_core::BotorchError>((gradients, noise_gradient))
    })
    .into_py_result()?;
    let out = gradients
        .iter()
        .zip(&factors)
        .map(|(gs, f)| {
            ArrayBuf(
                gs.iter().flat_map(|g| matrix_out(g).0).collect(),
                f.1.clone(),
            )
        })
        .collect();
    Ok((out, ArrayBuf(noise_gradient, noise.1)))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(higher_order_variances, m)?)?;
    m.add_function(wrap_pyfunction!(higher_order_variances_vjp, m)?)?;
    Ok(())
}
