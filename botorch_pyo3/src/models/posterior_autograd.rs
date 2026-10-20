//! First-order query adjoints for native posterior moments and samples.

use std::ops::Range;

use botorch_core::acquisition::monte_carlo::{
    posterior_samples_moment_vjp, posterior_samples_with_vjp,
};
use botorch_core::{BotorchError, JointPosteriorGrad, Result};

use super::*;

fn source_of(source: &Bound<'_, PyAny>) -> PyResult<ModelRef> {
    if let Ok(model) = source.extract::<PyRef<'_, PySingleTaskGP>>() {
        return Ok(ModelRef::Gp(model.snapshot()));
    }
    if let Ok(model) = source.extract::<PyRef<'_, PyModelListGP>>() {
        return Ok(ModelRef::List(model.inner.clone()));
    }
    if let Ok(model) = source.extract::<PyRef<'_, PyModel>>() {
        return Ok(ModelRef::Dyn(model.inner.clone()));
    }
    Err(unsupported(
        "Query gradients require a native Gaussian model source",
    ))
}

/// A single model-batch element. The VJP path borrows all cached factorizations.
#[derive(Debug)]
struct At<'a> {
    source: &'a ModelRef,
    batch: usize,
}

impl Model for At<'_> {
    fn dim(&self) -> usize {
        self.source.dim()
    }
    fn num_outputs(&self) -> usize {
        self.source.num_outputs()
    }
    fn points_per_input(&self) -> usize {
        self.source.points_per_input()
    }
    fn is_deterministic(&self) -> bool {
        matches!(self.source, ModelRef::Dyn(m) if m.is_deterministic())
    }
    fn posterior(&self, x: &Matrix, noise: bool) -> Result<Posterior> {
        self.source.posterior_at(self.batch, x, noise)
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        match self.source {
            ModelRef::Gp(m) => m.joint_posterior_with_grad_at(self.batch, x),
            ModelRef::List(m) => {
                let parts = m
                    .models()
                    .iter()
                    .map(|g| g.batch_element(self.batch))
                    .collect::<Result<Vec<_>>>()?;
                ModelListGP::new(parts)?.joint_posterior_with_grad(x)
            }
            ModelRef::Dyn(m) => m
                .batch_element(self.batch)
                .unwrap_or(m.as_ref())
                .joint_posterior_with_grad(x),
        }
    }
    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean: &Vector,
        covariance: &Matrix,
    ) -> Result<Matrix> {
        match self.source {
            ModelRef::Gp(m) => m.joint_posterior_vjp_at(self.batch, x, rows, mean, covariance),
            ModelRef::List(m) => {
                let size = x.nrows() * self.num_outputs() * self.points_per_input();
                if rows.start > rows.end
                    || rows.end > x.nrows()
                    || mean.len() != size
                    || covariance.shape() != (size, size)
                {
                    return Err(BotorchError::DimensionMismatch(
                        "Posterior adjoints have inconsistent shapes".into(),
                    ));
                }
                let mut gradient = Matrix::zeros(rows.len(), self.dim());
                let q = x.nrows() * self.points_per_input();
                let mut offset = 0;
                for child in m.models() {
                    let size = q * child.num_outputs();
                    gradient += child.joint_posterior_vjp_at(
                        self.batch,
                        x,
                        rows.clone(),
                        &mean.rows(offset, size).into_owned(),
                        &covariance.view((offset, offset), (size, size)).into_owned(),
                    )?;
                    offset += size;
                }
                Ok(gradient)
            }
            ModelRef::Dyn(m) => m
                .batch_element(self.batch)
                .unwrap_or(m.as_ref())
                .joint_posterior_vjp(x, rows, mean, covariance),
        }
    }
}

/// Pull a selected or scalarized Gaussian posterior adjoint back to its source.
/// Moment scalarization is handled separately because it is nonlinear.
fn linear_pullback(
    view: &View,
    post: &Posterior,
    mean: &Vector,
    covariance: &Matrix,
) -> (Vector, Matrix) {
    let q = post.num_points();
    let outputs: Vec<usize> = view
        .outputs
        .clone()
        .unwrap_or_else(|| (0..post.num_outputs()).collect());
    let mut ma = Vector::zeros(post.mean.len());
    let mut ca = Matrix::zeros(post.mean.len(), post.mean.len());
    for (t, &original) in outputs.iter().enumerate() {
        let weight = view.transform.as_ref().map_or(1.0, |v| v.weights()[t]);
        for p in 0..q {
            let selected = if view.transform.is_some() {
                p
            } else {
                t * q + p
            };
            ma[original * q + p] += weight * mean[selected];
            for (u, &other) in outputs.iter().enumerate() {
                let other_weight = view.transform.as_ref().map_or(1.0, |v| v.weights()[u]);
                for r in 0..q {
                    let selected_other = if view.transform.is_some() {
                        r
                    } else {
                        u * q + r
                    };
                    ca[(original * q + p, other * q + r)] +=
                        weight * other_weight * covariance[(selected, selected_other)];
                }
            }
        }
    }
    (ma, ca)
}

/// Source Gaussian adjoints from marginal-moment adjoints, including Log
/// transforms, variance clamps, and single-point PosteriorList scalarization.
fn marginal_pullback(
    view: &View,
    post: &Posterior,
    mean: &Vector,
    variance: &Vector,
    clamp_output: bool,
) -> Result<(Vector, Matrix)> {
    let q = post.num_points();
    let outputs: Vec<usize> = view
        .outputs
        .clone()
        .unwrap_or_else(|| (0..post.num_outputs()).collect());
    let selected = view.posterior(post.clone())?;
    let mut ma = Vector::zeros(post.mean.len());
    let mut ca = Matrix::zeros(post.mean.len(), post.mean.len());
    if view.transform.is_some() && !view.moments {
        let mut diagonal = Matrix::zeros(q, q);
        for p in 0..q {
            if !clamp_output || selected.covariance[(p, p)] > MIN_VARIANCE {
                diagonal[(p, p)] = variance[p];
            }
        }
        return Ok(linear_pullback(view, post, mean, &diagonal));
    }
    for (t, &original) in outputs.iter().enumerate() {
        let weight = view.transform.as_ref().map_or(1.0, |v| v.weights()[t]);
        for p in 0..q {
            let i = original * q + p;
            let selected_index = if view.moments { p } else { t * q + p };
            let mean_adjoint = mean[selected_index] * weight;
            let mut variance_adjoint = variance[selected_index] * weight * weight;
            if clamp_output && view.moments && selected.covariance[(p, p)] <= MIN_VARIANCE {
                variance_adjoint = 0.0;
            }
            if let Some(transform) = post.sample_transform.as_ref().filter(|_| !view.base) {
                let (_, _, a, b, c, d) =
                    transform.moments(original, p, post.mean[i], post.covariance[(i, i)])?;
                ma[i] += mean_adjoint * a + variance_adjoint * c;
                ca[(i, i)] += mean_adjoint * b + variance_adjoint * d;
            } else {
                ma[i] += mean_adjoint;
                if post.covariance[(i, i)] > MIN_VARIANCE {
                    ca[(i, i)] += variance_adjoint;
                }
            }
        }
    }
    Ok((ma, ca))
}

/// Posterior view used by the existing native reparameterized-sample VJP.
#[derive(Debug)]
struct SampleView<'a> {
    model: At<'a>,
    view: &'a View,
    noise: bool,
}

impl Model for SampleView<'_> {
    fn dim(&self) -> usize {
        self.model.dim()
    }
    fn num_outputs(&self) -> usize {
        self.view.num_outputs
    }
    fn points_per_input(&self) -> usize {
        self.model.points_per_input()
    }
    fn is_deterministic(&self) -> bool {
        self.model.is_deterministic()
    }
    fn posterior(&self, x: &Matrix, _: bool) -> Result<Posterior> {
        self.view.posterior(self.model.posterior(x, self.noise)?)
    }
    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean: &Vector,
        covariance: &Matrix,
    ) -> Result<Matrix> {
        let post = self.model.posterior(x, self.noise)?;
        let (ma, ca) = if self.view.moments {
            marginal_pullback(self.view, &post, mean, &covariance.diagonal(), false)?
        } else {
            linear_pullback(self.view, &post, mean, covariance)
        };
        self.model.joint_posterior_vjp(x, rows, &ma, &ca)
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let post = self.posterior(x, false)?;
        let size = post.mean.len();
        let mut dmean = vec![Vector::zeros(size); x.len()];
        let mut dcov = vec![Matrix::zeros(size, size); x.len()];
        // This adapter is normally consumed in reverse mode. Its forward-mode
        // implementation also uses exact native VJPs, avoiding a second backend.
        for i in 0..size {
            let mut ma = Vector::zeros(size);
            ma[i] = 1.0;
            let g = self.joint_posterior_vjp(x, 0..x.nrows(), &ma, &Matrix::zeros(size, size))?;
            for p in 0..x.nrows() {
                for k in 0..x.ncols() {
                    dmean[p * x.ncols() + k][i] = g[(p, k)];
                }
            }
            for j in 0..size {
                let mut ca = Matrix::zeros(size, size);
                ca[(i, j)] = 1.0;
                let g = self.joint_posterior_vjp(x, 0..x.nrows(), &Vector::zeros(size), &ca)?;
                for p in 0..x.nrows() {
                    for k in 0..x.ncols() {
                        dcov[p * x.ncols() + k][(i, j)] = g[(p, k)];
                    }
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: post.mean,
            covariance: post.covariance,
            dmean,
            dcov,
            num_outputs: post.num_outputs,
        })
    }
}

fn make_view(
    model: &ModelRef,
    outputs: Option<Vec<usize>>,
    transform: Option<TransformArg>,
    base: bool,
) -> PyResult<View> {
    let view = View::for_model(model, outputs, transform)?;
    if base {
        view.base()
    } else {
        Ok(view)
    }
}

/// Lift exposed moment adjoints to the full source Gaussian, then use its
/// analytic training-solve and kernel-parameter pullback. Selection,
/// scalarization, nonlinear outcome moments, and clamps share the query path.
#[allow(clippy::too_many_arguments)]
pub(super) fn moment_parameter_vjp(
    gp: Arc<BatchedMultiOutputGP>,
    x: &Array,
    adjoint: &Array,
    mode: &str,
    observation_noise: bool,
    output_indices: Option<Vec<usize>>,
    transform: Option<TransformArg>,
    base: bool,
) -> PyResult<ArrayBuf> {
    let model = ModelRef::Gp(gp.clone());
    let view = make_view(&model, output_indices, transform, base)?;
    check_dim(x, model.dim())?;
    let (xbatch, mats) = x.batches("X")?;
    let bc = broadcast(&xbatch, &model.batch_shape())?;
    let q = x.1[x.1.len() - 2] * model.points_per_input();
    let mo = view.num_outputs;
    let m = model.num_outputs();
    let covariance = mode == "covariance";
    if !matches!(mode, "mean" | "variance" | "covariance") {
        return Err(unsupported(
            "Parameter gradients currently support posterior moments",
        ));
    }
    let tail = if covariance {
        vec![q * mo, q * mo]
    } else {
        vec![q, mo]
    };
    if adjoint.1 != [bc.shape.as_slice(), &tail].concat() {
        return Err(dimension_error(
            "Posterior adjoint shape does not match its output",
        ));
    }
    let width = numel(&tail);
    let mut means = Vec::with_capacity(bc.x_index.len() * q * m);
    let mut covariances = Vec::with_capacity(bc.x_index.len() * q * m * q * m);
    for b in 0..bc.x_index.len() {
        let post = model
            .posterior_at(bc.model_index[b], &mats[bc.x_index[b]], observation_noise)
            .into_py_result()?;
        let values = &adjoint.0[b * width..(b + 1) * width];
        let (ma, ca) = if covariance {
            let matrix = Matrix::from_row_slice(q * mo, q * mo, values);
            if view.moments {
                marginal_pullback(
                    &view,
                    &post,
                    &Vector::zeros(q * mo),
                    &matrix.diagonal(),
                    false,
                )
                .into_py_result()?
            } else {
                linear_pullback(&view, &post, &Vector::zeros(q * mo), &matrix)
            }
        } else {
            let ordered = Vector::from_fn(q * mo, |i, _| values[(i % q) * mo + i / q]);
            let zero = Vector::zeros(q * mo);
            if mode == "mean" {
                marginal_pullback(&view, &post, &ordered, &zero, true).into_py_result()?
            } else {
                marginal_pullback(&view, &post, &zero, &ordered, true).into_py_result()?
            }
        };
        for p in 0..q {
            for t in 0..m {
                means.push(ma[t * q + p]);
            }
        }
        for i in 0..q * m {
            for j in 0..q * m {
                covariances.push(ca[(i, j)]);
            }
        }
    }
    let mean = Array(means, [bc.shape.as_slice(), &[q, m]].concat());
    let covariance = Array(covariances, [bc.shape.as_slice(), &[q * m, q * m]].concat());
    parameter_grad::parameter_vjp(&gp, x, &mean, &covariance, observation_noise)
}

/// Reparameterized sample parameter adjoints, reduced into original model
/// elements after lifting the selected/scalarized sample distribution.
#[allow(clippy::too_many_arguments)]
pub(super) fn sample_parameter_vjp(
    gp: Arc<BatchedMultiOutputGP>,
    x: &Array,
    normals: &Array,
    adjoint: &Array,
    observation_noise: bool,
    outputs: Option<Vec<usize>>,
    transform: Option<TransformArg>,
    base: bool,
) -> PyResult<ArrayBuf> {
    let model = ModelRef::Gp(gp.clone());
    let view = make_view(&model, outputs, transform, base)?;
    check_dim(x, model.dim())?;
    let (xbatch, mats) = x.batches("X")?;
    let bc = broadcast(&xbatch, &model.batch_shape())?;
    let q = x.1[x.1.len() - 2] * model.points_per_input();
    let size = q * view.num_outputs;
    let m = model.num_outputs();
    let n = *normals
        .1
        .first()
        .ok_or_else(|| dimension_error("Base samples require a sample dimension"))?;
    let nb = bc.x_index.len();
    let per_batch = match normals.1.as_slice() {
        [_, k] if *k == size => false,
        [_, b, k] if *b == nb && *k == size => true,
        _ => {
            return Err(dimension_error(
                "Base samples do not match the posterior event and batch shapes",
            ))
        }
    };
    if adjoint.0.len() != n * nb * size || normals.0.len() != numel(&normals.1) {
        return Err(dimension_error(
            "Sample adjoints or base samples have inconsistent shapes",
        ));
    }
    let mut means = Vec::with_capacity(nb * q * m);
    let mut covariances = Vec::with_capacity(nb * q * m * q * m);
    for b in 0..nb {
        let points = &mats[bc.x_index[b]];
        let at = SampleView {
            model: At {
                source: &model,
                batch: bc.model_index[b],
            },
            view: &view,
            noise: observation_noise,
        };
        let z = Matrix::from_fn(n, size, |s, k| {
            normals.0[if per_batch {
                (s * nb + b) * size + k
            } else {
                s * size + k
            }]
        });
        let weights = Matrix::from_fn(n, size, |s, k| adjoint.0[(s * nb + b) * size + k]);
        let (ma, ca) = posterior_samples_moment_vjp(&at, points, &z, &weights).into_py_result()?;
        let post = model
            .posterior_at(bc.model_index[b], points, observation_noise)
            .into_py_result()?;
        let (ma, ca) = if view.moments {
            marginal_pullback(&view, &post, &ma, &ca.diagonal(), false).into_py_result()?
        } else {
            linear_pullback(&view, &post, &ma, &ca)
        };
        for p in 0..q {
            for t in 0..m {
                means.push(ma[t * q + p]);
            }
        }
        for i in 0..q * m {
            for j in 0..q * m {
                covariances.push(ca[(i, j)]);
            }
        }
    }
    parameter_grad::parameter_vjp(
        &gp,
        x,
        &Array(means, [bc.shape.as_slice(), &[q, m]].concat()),
        &Array(covariances, [bc.shape.as_slice(), &[q * m, q * m]].concat()),
        observation_noise,
    )
}

/// Native VJP of posterior mean, variance, or covariance w.r.t. query inputs.
#[pyfunction]
#[pyo3(signature = (source, x, adjoint, mode, observation_noise=false, output_indices=None, transform=None, base=false))]
#[allow(clippy::too_many_arguments)]
fn posterior_query_vjp(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    x: Array,
    adjoint: Array,
    mode: &str,
    observation_noise: bool,
    output_indices: Option<Vec<usize>>,
    transform: Option<TransformArg>,
    base: bool,
) -> PyResult<ArrayBuf> {
    let model = source_of(source)?;
    let view = make_view(&model, output_indices, transform, base)?;
    check_dim(&x, model.dim())?;
    let (xbatch, mats) = x.batches("X")?;
    let bc = broadcast(&xbatch, &model.batch_shape())?;
    let q = x.1[x.1.len() - 2] * model.points_per_input();
    let mo = view.num_outputs;
    let covariance = mode == "covariance";
    if !matches!(mode, "mean" | "variance" | "covariance") {
        return Err(value_error("Unknown posterior gradient mode"));
    }
    let width = if covariance { q * mo * q * mo } else { q * mo };
    if adjoint.0.len() != bc.x_index.len() * width {
        return Err(dimension_error(
            "Posterior adjoint shape does not match its output",
        ));
    }
    let gradients = threads::run(py, || {
        (0..bc.x_index.len())
            .into_par_iter()
            .map(|b| {
                let points = &mats[bc.x_index[b]];
                let at = At {
                    source: &model,
                    batch: bc.model_index[b],
                };
                let post = at.posterior(points, observation_noise)?;
                let values = &adjoint.0[b * width..(b + 1) * width];
                let (ma, ca) = if covariance {
                    if view.moments {
                        marginal_pullback(
                            &view,
                            &post,
                            &Vector::zeros(q * mo),
                            &Matrix::from_row_slice(q * mo, q * mo, values).diagonal(),
                            false,
                        )?
                    } else {
                        linear_pullback(
                            &view,
                            &post,
                            &Vector::zeros(q * mo),
                            &Matrix::from_row_slice(q * mo, q * mo, values),
                        )
                    }
                } else {
                    let ordered = Vector::from_fn(q * mo, |i, _| values[(i % q) * mo + i / q]);
                    let zero = Vector::zeros(q * mo);
                    if mode == "mean" {
                        marginal_pullback(&view, &post, &ordered, &zero, true)?
                    } else {
                        marginal_pullback(&view, &post, &zero, &ordered, true)?
                    }
                };
                at.joint_posterior_vjp(points, 0..points.nrows(), &ma, &ca)
            })
            .collect::<Result<Vec<_>>>()
    })
    .into_py_result()?;
    sum_broadcast_gradients(&x, &mats, &bc, gradients)
}

fn sum_broadcast_gradients(
    x: &Array,
    mats: &[Matrix],
    bc: &Broadcast,
    gradients: Vec<Matrix>,
) -> PyResult<ArrayBuf> {
    let mut sum: Vec<Matrix> = mats
        .iter()
        .map(|m| Matrix::zeros(m.nrows(), m.ncols()))
        .collect();
    for (batch, gradient) in gradients.into_iter().enumerate() {
        sum[bc.x_index[batch]] += gradient;
    }
    let data = sum
        .iter()
        .flat_map(|m| (0..m.nrows()).flat_map(move |i| (0..m.ncols()).map(move |j| m[(i, j)])))
        .collect();
    Ok(ArrayBuf(data, x.1.clone()))
}

/// Native VJP of reparameterized samples, with selected outputs and batches.
#[pyfunction]
#[pyo3(signature = (source, x, base_samples, adjoint, observation_noise=false, output_indices=None, transform=None, base=false))]
#[allow(clippy::too_many_arguments)]
fn posterior_query_samples_vjp(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    x: Array,
    base_samples: Array,
    adjoint: Array,
    observation_noise: bool,
    output_indices: Option<Vec<usize>>,
    transform: Option<TransformArg>,
    base: bool,
) -> PyResult<ArrayBuf> {
    let model = source_of(source)?;
    let view = make_view(&model, output_indices, transform, base)?;
    check_dim(&x, model.dim())?;
    let (xbatch, mats) = x.batches("X")?;
    let bc = broadcast(&xbatch, &model.batch_shape())?;
    let q = x.1[x.1.len() - 2] * model.points_per_input();
    let size = q * view.num_outputs;
    let n = *base_samples
        .1
        .first()
        .ok_or_else(|| dimension_error("Base samples require a sample dimension"))?;
    let nb = bc.x_index.len();
    if base_samples.0.len() != base_samples.1.iter().product::<usize>() {
        return Err(dimension_error(
            "Base samples do not match their declared shape",
        ));
    }
    let per_batch = match *base_samples.1.as_slice() {
        [_, k] if k == size => false,
        [_, b, k] if b == nb && k == size => true,
        _ => {
            return Err(dimension_error(
                "Base samples do not match the posterior event and batch shapes",
            ))
        }
    };
    if adjoint.0.len() != n * nb * size {
        return Err(dimension_error(
            "Sample adjoints do not match posterior samples",
        ));
    }
    let gradients = threads::run(py, || {
        (0..nb)
            .into_par_iter()
            .map(|b| {
                let points = &mats[bc.x_index[b]];
                let at = SampleView {
                    model: At {
                        source: &model,
                        batch: bc.model_index[b],
                    },
                    view: &view,
                    noise: observation_noise,
                };
                let z = Matrix::from_fn(n, size, |s, k| {
                    base_samples.0[if per_batch {
                        (s * nb + b) * size + k
                    } else {
                        s * size + k
                    }]
                });
                let weights = Matrix::from_fn(n, size, |s, k| adjoint.0[(s * nb + b) * size + k]);
                posterior_samples_with_vjp(&at, points, &z, &weights).map(|(_, gradient)| gradient)
            })
            .collect::<Result<Vec<_>>>()
    })
    .into_py_result()?;
    sum_broadcast_gradients(&x, &mats, &bc, gradients)
}

/// Register the native query-adjoint bridge.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(posterior_query_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(posterior_query_samples_vjp, m)?)?;
    Ok(())
}
