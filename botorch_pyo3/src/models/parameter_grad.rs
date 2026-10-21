//! Raw parameter pullbacks of batched independent-output Gaussian posteriors.

use super::*;

pub(super) fn parameter_vjp(
    model: &BatchedMultiOutputGP,
    x: &Array,
    mean: &Array,
    covariance: &Array,
    noisy: bool,
) -> PyResult<ArrayBuf> {
    check_dim(x, model.dim())?;
    let (x_batch, points) = x.batches("X")?;
    let bc = broadcast(&x_batch, model.batch_shape())?;
    let q = x.1[x.1.len() - 2] * model.points_per_input();
    let outputs = model.num_outputs();
    let size = q * outputs;
    let mean_shape = [bc.shape.as_slice(), &[q, outputs]].concat();
    let covariance_shape = [bc.shape.as_slice(), &[size, size]].concat();
    if mean.1 != mean_shape
        || covariance.1 != covariance_shape
        || mean.0.len() != numel(&mean_shape)
        || covariance.0.len() != numel(&covariance_shape)
    {
        return Err(dimension_error(format!(
            "posterior mean/covariance adjoints must have shapes {mean_shape:?} and {covariance_shape:?}"
        )));
    }
    let layout = model.element_layout();
    let mut per_element = vec![Vector::zeros(layout.len()); model.num_elements()];
    for b in 0..bc.x_index.len() {
        for output in 0..outputs {
            let element = model.element(bc.model_index[b], output);
            let ma = Vector::from_fn(q, |i, _| mean.0[b * size + i * outputs + output]);
            let ca = Matrix::from_fn(q, q, |i, j| {
                covariance.0[b * size * size + (output * q + i) * size + output * q + j]
            });
            per_element[element] += model.models()[element]
                .posterior_parameter_vjp(&points[bc.x_index[b]], &ma, &ca, noisy)
                .into_py_result()?;
        }
    }
    let mut values = Vec::with_capacity(model.raw_parameters().len());
    let mut offset = 0;
    for (_, count) in model.models()[0].parameter_names() {
        for element in &per_element {
            values.extend_from_slice(&element.as_slice()[offset..offset + count]);
        }
        offset += count;
    }
    let size = values.len();
    Ok(ArrayBuf(values, vec![size]))
}
