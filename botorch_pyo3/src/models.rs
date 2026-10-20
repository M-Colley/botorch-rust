//! Models (`botorch.models`): `SingleTaskGP` with `m ≥ 1` outputs and an
//! optional batch shape, `ModelListGP`, posteriors (with output selection,
//! posterior transforms and batch broadcasting), marginal log likelihoods,
//! fitting and hyperparameter access, and the `Model` handle that
//! acquisition functions are built on.
//!
//! A `SingleTaskGP` is stored as an `Arc` snapshot behind a lock:
//! acquisition functions and model lists share the snapshot that is current
//! when they are created, while fitting or setting hyperparameters installs a
//! new snapshot (copy on write), so the GIL can be released during heavy work
//! without data races.

use std::sync::{Arc, RwLock};

use botorch_core::acquisition::{ScalarizedModel, ScalarizedPosteriorTransform};
use botorch_core::models::{ConstantMeanSpec, LcemTaskInputs};
use botorch_core::posterior::MIN_VARIANCE;
use botorch_core::{
    fit_gpytorch_mll, BatchedFitResult, BatchedMultiOutputGP, BatchedMultiOutputGPOptions,
    FitOptions, FitStatus, Hyperparameters, Likelihood, Matrix, Model, ModelList, ModelListGP,
    OutcomeTransformChoice, PointPosteriors, Posterior, Vector,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::arrays::{broadcast_indices, broadcast_shapes, broadcast_to, Array, ArrayBuf, ArrayOut};
use crate::contextual::PyLCEMGP;
use crate::errors::{dimension_error, unsupported, value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::specs::{PyConstraint, PyKernelSpec, PyLikelihoodSpec, PyPrior};
use crate::threads;
use crate::transforms::{numel, PyNormalize, PyStandardize};
use pyo3::IntoPyObjectExt;

pub mod posterior_autograd;

/// Rows per parallel work item of the marginal posterior.
const MARGINAL_CHUNK: usize = 256;

/// The fit summary returned to Python: `(loss, iterations, evaluations,
/// status, message, attempts, elements)`, with one `(loss, iterations,
/// evaluations, status, message, attempts)` tuple per batch element.
type ElementFit = (f64, usize, usize, &'static str, String, usize);
type FitSummary = (
    f64,
    usize,
    usize,
    &'static str,
    String,
    usize,
    Vec<ElementFit>,
);

/// A posterior transform received from Python: `(weights, offset)`.
type TransformArg = (Vec<f64>, f64);

fn status_name(status: FitStatus) -> &'static str {
    match status {
        FitStatus::Success => "SUCCESS",
        FitStatus::Stopped => "STOPPED",
        FitStatus::Failure => "FAILURE",
    }
}

/// A model evaluated per model batch element: a `SingleTaskGP`, a
/// `ModelListGP`, or any unbatched model (e.g. an `LCEMGP` or a list of them).
#[derive(Clone, Debug)]
enum ModelRef {
    Gp(Arc<BatchedMultiOutputGP>),
    List(ModelListGP),
    Dyn(Arc<dyn Model>),
}

impl ModelRef {
    fn dim(&self) -> usize {
        match self {
            ModelRef::Gp(m) => m.dim(),
            ModelRef::List(l) => l.dim(),
            ModelRef::Dyn(m) => m.dim(),
        }
    }

    fn num_outputs(&self) -> usize {
        match self {
            ModelRef::Gp(m) => m.num_outputs(),
            ModelRef::List(l) => l.num_outputs(),
            ModelRef::Dyn(m) => m.num_outputs(),
        }
    }

    fn batch_shape(&self) -> Vec<usize> {
        match self {
            ModelRef::Gp(m) => m.batch_shape().to_vec(),
            ModelRef::List(l) => l.batch_shape().to_vec(),
            ModelRef::Dyn(m) => m.batch_shape().to_vec(),
        }
    }

    /// Posterior points per input point (`n_w` with an `InputPerturbation`).
    fn points_per_input(&self) -> usize {
        match self {
            ModelRef::Gp(m) => m.points_per_input(),
            ModelRef::List(l) => l.points_per_input(),
            ModelRef::Dyn(m) => m.points_per_input(),
        }
    }

    /// Whether the posterior of each output is a `TransformedPosterior`.
    fn transformed_outputs(&self) -> Vec<bool> {
        match self {
            ModelRef::Gp(m) => m.transformed_outputs(),
            ModelRef::List(l) => Model::transformed_outputs(l),
            ModelRef::Dyn(m) => m.transformed_outputs(),
        }
    }

    /// Whether the model is a list of single-output models (whose posterior
    /// with transformed outputs is a `PosteriorList` that BoTorch scalarizes
    /// by moments).
    fn is_list_of_single_outputs(&self) -> bool {
        let sizes = match self {
            ModelRef::Gp(_) => None,
            ModelRef::List(l) => Model::list_output_sizes(l),
            ModelRef::Dyn(m) => m.list_output_sizes(),
        };
        sizes.is_some_and(|s| s.iter().all(|&k| k == 1))
    }

    fn posterior_at(&self, b: usize, x: &Matrix, noise: bool) -> botorch_core::Result<Posterior> {
        match self {
            ModelRef::Gp(m) => m.posterior_at(b, x, noise),
            ModelRef::List(l) => l.posterior_at(b, x, noise),
            ModelRef::Dyn(m) => m.batch_element(b).unwrap_or(m.as_ref()).posterior(x, noise),
        }
    }

    fn point_posteriors_at(
        &self,
        b: usize,
        x: &Matrix,
        noise: bool,
    ) -> botorch_core::Result<PointPosteriors> {
        match self {
            ModelRef::Gp(m) => m.point_posteriors_at(b, x, noise),
            ModelRef::List(l) => l.point_posteriors_at(b, x, noise),
            ModelRef::Dyn(m) if !noise => {
                m.batch_element(b).unwrap_or(m.as_ref()).point_posteriors(x)
            }
            ModelRef::Dyn(m) => m
                .batch_element(b)
                .unwrap_or(m.as_ref())
                .point_posteriors_with_noise(x),
        }
    }
}

/// BoTorch's `output_indices` and `posterior_transform` applied to the
/// posteriors of a model.
#[derive(Clone, Debug)]
struct View {
    outputs: Option<Vec<usize>>,
    transform: Option<ScalarizedPosteriorTransform>,
    num_outputs: usize,
    /// Whether each selected output is a `TransformedPosterior` (before a
    /// posterior transform).
    transformed: Vec<bool>,
    /// Whether the posterior transform scalarizes the marginal moments of a
    /// `PosteriorList` (BoTorch's `scalarize_posterior` of a model list with
    /// transformed outputs, `q = 1`).
    moments: bool,
    /// Report the Gaussian base posterior of transformed outputs (the
    /// `_posterior` of a `TransformedPosterior`).
    base: bool,
}

impl View {
    fn for_model(
        model: &ModelRef,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
    ) -> PyResult<Self> {
        let mut view = Self::new(model.num_outputs(), output_indices, transform)?;
        let all = model.transformed_outputs();
        view.transformed = match &view.outputs {
            Some(o) => o.iter().map(|&t| all[t]).collect(),
            None => all,
        };
        if view.transform.is_some() && view.transformed.iter().any(|&t| t) {
            if !model.is_list_of_single_outputs() {
                return Err(unsupported(
                    botorch_core::acquisition::SCALARIZE_TRANSFORMED_MESSAGE,
                ));
            }
            view.moments = true;
        }
        Ok(view)
    }

    /// The view of the Gaussian base posteriors (no sample transforms).
    fn base(mut self) -> PyResult<Self> {
        if self.transform.is_some() && self.transformed.iter().any(|&t| t) {
            return Err(unsupported(
                "the base posterior of a scalarized TransformedPosterior is not defined",
            ));
        }
        self.base = true;
        self.transformed = vec![false; self.transformed.len()];
        Ok(self)
    }

    fn new(
        m: usize,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
    ) -> PyResult<Self> {
        if let Some(o) = &output_indices {
            if o.is_empty() || o.iter().any(|&t| t >= m) {
                return Err(value_error(format!(
                    "output_indices {o:?} are invalid for a model with {m} outputs"
                )));
            }
        }
        let selected = output_indices.as_ref().map_or(m, |o| o.len());
        let transform = transform
            .map(|(w, offset)| {
                if w.len() != selected {
                    return Err(dimension_error(format!(
                        "Output shape not equal to that of weights. Output shape is {selected} \
                         and weights are [{}]",
                        w.len()
                    )));
                }
                ScalarizedPosteriorTransform::new(Vector::from_vec(w), offset).into_py_result()
            })
            .transpose()?;
        let num_outputs = if transform.is_some() { 1 } else { selected };
        Ok(Self {
            outputs: output_indices,
            transform,
            num_outputs,
            transformed: vec![false; selected],
            moments: false,
            base: false,
        })
    }

    fn posterior(&self, p: Posterior) -> botorch_core::Result<Posterior> {
        let mut p = match &self.outputs {
            Some(o) => p.select_outputs(o)?,
            None => p,
        };
        if self.base {
            p.sample_transform = None;
        }
        match &self.transform {
            Some(t) if self.moments => {
                // BoTorch's `scalarize_posterior` of a `PosteriorList`: a
                // Gaussian of the scalarized marginal moments of one point.
                if p.num_points() != 1 {
                    return Err(botorch_core::BotorchError::Unsupported(
                        "scalarize_posterior only works with a PosteriorList if each \
                         sub-posterior has q=1."
                            .into(),
                    ));
                }
                let (mean, var) = p.moments()?;
                let pp = PointPosteriors::independent(
                    Matrix::from_row_slice(1, mean.len(), mean.as_slice()),
                    &Matrix::from_row_slice(1, var.len(), var.as_slice()),
                );
                let s = t.transform_point_posteriors(&pp)?;
                Ok(Posterior::single_output(
                    Vector::from_element(1, s.mean[(0, 0)]),
                    Matrix::from_element(1, 1, s.variance(0, 0)),
                ))
            }
            Some(t) => t.transform_posterior(&p),
            None => Ok(p),
        }
    }

    fn points(&self, pp: PointPosteriors) -> botorch_core::Result<PointPosteriors> {
        let mut pp = match &self.outputs {
            Some(o) => pp.select_outputs(o)?,
            None => pp,
        };
        match &self.transform {
            Some(t) => {
                if self.moments {
                    // The variances of the Gaussian outputs of a
                    // `PosteriorList` are clamped (`posterior.variance`).
                    let m = pp.num_outputs();
                    for i in 0..pp.len() {
                        for (k, &tr) in self.transformed.iter().enumerate() {
                            let c = &mut pp.covariance[(i, k * m + k)];
                            if !tr && *c < MIN_VARIANCE {
                                *c = MIN_VARIANCE;
                            }
                        }
                    }
                }
                t.transform_point_posteriors(&pp)
            }
            None => Ok(pp),
        }
    }

    /// Whether the variance of selected output `t` is a moment of a
    /// `TransformedPosterior` (reported unclamped).
    fn unclamped(&self, t: usize) -> bool {
        self.transform.is_none() && self.transformed.get(t).copied().unwrap_or(false)
    }
}

/// The broadcast of the input batch shape of `X` with the model batch shape:
/// the output batch shape and, per output batch element, the indices of the
/// input point set and of the model batch element.
struct Broadcast {
    shape: Vec<usize>,
    x_index: Vec<usize>,
    model_index: Vec<usize>,
}

fn broadcast(xbatch: &[usize], model_batch: &[usize]) -> PyResult<Broadcast> {
    let shape = broadcast_shapes(xbatch, model_batch).ok_or_else(|| {
        dimension_error(format!(
            "The trailing batch dimensions of X must match the trailing batch dimensions of \
             the training inputs. Got X with batch shape {xbatch:?} and a model with batch \
             shape {model_batch:?}."
        ))
    })?;
    Ok(Broadcast {
        x_index: broadcast_indices(xbatch, &shape),
        model_index: broadcast_indices(model_batch, &shape),
        shape,
    })
}

fn check_dim(x: &Array, d: usize) -> PyResult<()> {
    match x.1.last() {
        Some(&k) if k == d && x.1.len() >= 2 => Ok(()),
        _ => Err(dimension_error(format!(
            "Expected X to be `batch_shape x q x d` with d = {d}, got shape {:?}.",
            x.1
        ))),
    }
}

/// Marginal posterior means and (clamped) variances at every point of `x`
/// (`... x q x d`): two arrays of shape `out_batch x q·k x m'` for `k`
/// posterior points per input (BoTorch's `q * n_w` points with an
/// `InputPerturbation`, point-major).
fn marginals(
    model: &ModelRef,
    x: &Array,
    noise: bool,
    view: &View,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    check_dim(x, model.dim())?;
    let (xbatch, mats) = x.batches("X")?;
    let bc = broadcast(&xbatch, &model.batch_shape())?;
    let q = x.1[x.1.len() - 2];
    let k = model.points_per_input();
    let qk = q * k;
    let mo = view.num_outputs;
    let nout = bc.x_index.len();
    let mut mean = vec![0.0; nout * qk * mo];
    let mut var = vec![0.0; nout * qk * mo];
    // Group the output elements by model batch element and evaluate all of
    // their points at once, in parallel chunks.
    let nmb = numel(&model.batch_shape());
    for b in 0..nmb {
        let outs: Vec<usize> = (0..nout).filter(|&o| bc.model_index[o] == b).collect();
        if outs.is_empty() || q == 0 {
            continue;
        }
        let rows = outs.len() * q;
        let d = model.dim();
        let stacked = Matrix::from_fn(rows, d, |r, j| mats[bc.x_index[outs[r / q]]][(r % q, j)]);
        let starts: Vec<usize> = (0..rows).step_by(MARGINAL_CHUNK).collect();
        let parts = starts
            .par_iter()
            .map(|&s| {
                let len = MARGINAL_CHUNK.min(rows - s);
                let pp = model.point_posteriors_at(b, &stacked.rows(s, len).into_owned(), noise)?;
                view.points(pp)
            })
            .collect::<botorch_core::Result<Vec<_>>>()
            .into_py_result()?;
        for (part, &s) in parts.iter().zip(&starts) {
            // `k` posterior points per stacked input row.
            for i in 0..part.len() {
                let r = s + i / k;
                let (o, p) = (outs[r / q], (r % q) * k + i % k);
                for t in 0..mo {
                    let idx = (o * qk + p) * mo + t;
                    mean[idx] = part.mean[(i, t)];
                    let v = part.variance(i, t);
                    var[idx] = if v < MIN_VARIANCE && !view.unclamped(t) {
                        MIN_VARIANCE
                    } else {
                        v
                    };
                }
            }
        }
    }
    let mut shape = bc.shape;
    shape.extend([qk, mo]);
    Ok((mean, var, shape))
}

/// Joint posteriors of every `q x d` point set of `x`.
fn joints(
    model: &ModelRef,
    x: &Array,
    noise: bool,
    view: &View,
) -> PyResult<(Vec<usize>, Vec<Posterior>)> {
    check_dim(x, model.dim())?;
    let (xbatch, mats) = x.batches("X")?;
    let bc = broadcast(&xbatch, &model.batch_shape())?;
    let posts = (0..bc.x_index.len())
        .into_par_iter()
        .map(|o| {
            let p = model.posterior_at(bc.model_index[o], &mats[bc.x_index[o]], noise)?;
            view.posterior(p)
        })
        .collect::<botorch_core::Result<Vec<_>>>()
        .into_py_result()?;
    Ok((bc.shape, posts))
}

/// Joint means (`out_batch x q x m'`) and covariances (`out_batch x q·m' x q·m'`).
fn joint_out(shape: Vec<usize>, posts: &[Posterior], q: usize, mo: usize) -> (ArrayBuf, ArrayBuf) {
    let mut mean = Vec::with_capacity(posts.len() * q * mo);
    let mut cov = Vec::with_capacity(posts.len() * q * q * mo * mo);
    for p in posts {
        for i in 0..q {
            for t in 0..mo {
                mean.push(p.mean[t * q + i]);
            }
        }
        let n = q * mo;
        for a in 0..n {
            for b in 0..n {
                cov.push(p.covariance[(a, b)]);
            }
        }
    }
    let mut mshape = shape.clone();
    mshape.extend([q, mo]);
    let mut cshape = shape;
    cshape.extend([q * mo, q * mo]);
    (ArrayBuf(mean, mshape), ArrayBuf(cov, cshape))
}

/// The marginal means and (clamped) variances of the Gaussian base
/// posteriors (the `_posterior` of a `TransformedPosterior`), from the joint
/// posteriors of the point sets of `x`: two arrays of shape `out_batch x q·k
/// x m'`.
fn base_marginals(
    py: Python<'_>,
    model: &ModelRef,
    x: &Array,
    noise: bool,
    view: View,
) -> PyResult<(ArrayBuf, ArrayBuf)> {
    let q = x.1.get(x.1.len().wrapping_sub(2)).copied().unwrap_or(0) * model.points_per_input();
    let mo = view.num_outputs;
    let (shape, posts) = threads::run(py, || joints(model, x, noise, &view))?;
    let mut mean = Vec::with_capacity(posts.len() * q * mo);
    let mut var = Vec::with_capacity(posts.len() * q * mo);
    for p in &posts {
        let v = p.variance();
        for i in 0..q {
            for t in 0..mo {
                mean.push(p.mean[t * q + i]);
                var.push(v[t * q + i]);
            }
        }
    }
    let mut out = shape;
    out.extend([q, mo]);
    Ok((ArrayBuf(mean, out.clone()), ArrayBuf(var, out)))
}

/// Posterior samples from base samples `z` of shape `S x q·m'` (shared by all
/// output batch elements) or `S x B x q·m'` (one set per output batch
/// element), with the base sample of point `p` and output `t` at index
/// `p·m' + t` (BoTorch's `S x q x m` base samples flattened); every output
/// uses the root of its own `q x q` covariance block, as GPyTorch does for
/// BoTorch's block-diagonal multi-output posteriors. Returns `S x out_batch x
/// q x m'`, where `q` counts the posterior points (`k` per input point with a
/// one-to-many input transform).
fn samples(model: &ModelRef, x: &Array, z: &Array, noise: bool, view: &View) -> PyResult<ArrayBuf> {
    let (shape, posts) = joints(model, x, noise, view)?;
    let q = x.1[x.1.len() - 2] * model.points_per_input();
    let mo = view.num_outputs;
    let nb = posts.len();
    let qm = q * mo;
    let per_batch = match z.1.as_slice() {
        &[_, k] if k == qm => false,
        &[_, b, k] if b == nb && k == qm => true,
        s => {
            return Err(dimension_error(format!(
                "base samples of shape {s:?} do not match {nb} batches of q = {q} points and \
                 {mo} outputs"
            )))
        }
    };
    if z.0.len() != z.1.iter().product::<usize>() {
        return Err(dimension_error("base samples do not match their shape"));
    }
    let s = z.1[0];
    let ys = posts
        .par_iter()
        .enumerate()
        .map(|(b, post)| {
            // The base samples of this batch (S x q·m', column p·m' + t) and the
            // samples in BoTorch's (p, t) order.
            let zb = Matrix::from_fn(s, qm, |i, k| {
                if per_batch {
                    z.0[(i * nb + b) * qm + k]
                } else {
                    z.0[i * qm + k]
                }
            });
            let samples = post.rsample_from_base_samples(&zb)?;
            Ok(Matrix::from_fn(s, qm, |i, k| {
                samples.value(i, k / mo, k % mo)
            }))
        })
        .collect::<botorch_core::Result<Vec<Matrix>>>()
        .into_py_result()?;
    let mut data = Vec::with_capacity(s * nb * qm);
    for i in 0..s {
        for y in &ys {
            data.extend(y.row(i).iter());
        }
    }
    let mut out_shape = vec![s];
    out_shape.extend(shape);
    out_shape.extend([q, mo]);
    Ok(ArrayBuf(data, out_shape))
}

#[allow(clippy::too_many_arguments)]
fn options_from(
    maxcor: Option<usize>,
    pgtol: Option<f64>,
    ftol: Option<f64>,
    maxiter: Option<usize>,
    maxfun: Option<usize>,
    maxls: Option<usize>,
    max_attempts: Option<usize>,
    seed: Option<u64>,
) -> PyResult<FitOptions> {
    let d = FitOptions::default();
    let options = FitOptions {
        maxcor: maxcor.unwrap_or(d.maxcor),
        pgtol: pgtol.unwrap_or(d.pgtol),
        ftol: ftol.unwrap_or(d.ftol),
        maxiter: maxiter.unwrap_or(d.maxiter),
        maxfun: maxfun.unwrap_or(d.maxfun),
        maxls: maxls.unwrap_or(d.maxls),
        max_attempts: max_attempts.unwrap_or(d.max_attempts),
        seed: seed.unwrap_or(d.seed),
    };
    if !(options.pgtol >= 0.0 && options.ftol >= 0.0) {
        return Err(value_error("gtol and ftol must be non-negative"));
    }
    Ok(options)
}

fn summary(r: &BatchedFitResult) -> FitSummary {
    let elements: Vec<ElementFit> = r
        .results
        .iter()
        .map(|e| {
            (
                e.loss,
                e.iterations,
                e.evaluations,
                status_name(e.status),
                e.message.clone(),
                e.attempts,
            )
        })
        .collect();
    (
        r.loss,
        r.iterations,
        r.results.iter().map(|e| e.evaluations).sum(),
        status_name(r.status),
        r.message.clone(),
        r.results.iter().map(|e| e.attempts).max().unwrap_or(1),
        elements,
    )
}

/// BoTorch's `SingleTaskGP` with `m ≥ 1` independent outputs and an optional
/// batch shape.
#[pyclass(frozen, name = "SingleTaskGP", module = "botorch_rs._botorch_rs")]
pub struct PySingleTaskGP {
    inner: RwLock<Arc<BatchedMultiOutputGP>>,
}

impl PySingleTaskGP {
    /// A handle on `model` (e.g. a conditioned or fantasy model).
    pub(crate) fn from_model(model: BatchedMultiOutputGP) -> Self {
        Self {
            inner: RwLock::new(Arc::new(model)),
        }
    }

    /// The current model.
    pub fn snapshot(&self) -> Arc<BatchedMultiOutputGP> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn replace(&self, model: BatchedMultiOutputGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }

    /// Replaces the model (after a fit).
    pub(crate) fn replace_model(&self, model: BatchedMultiOutputGP) {
        self.replace(model);
    }

    /// Applies `f` to a copy of the model and installs the copy on success.
    fn update<F>(&self, f: F) -> PyResult<()>
    where
        F: FnOnce(&mut BatchedMultiOutputGP) -> botorch_core::Result<()>,
    {
        let mut model = (*self.snapshot()).clone();
        f(&mut model).into_py_result()?;
        self.replace(model);
        Ok(())
    }

    fn model_ref(&self) -> ModelRef {
        ModelRef::Gp(self.snapshot())
    }

    /// Sets one hyperparameter of every element from `value` broadcast to
    /// `shape_per_element` (preceded by the `_aug_batch_shape`), or reshaped
    /// to it if it has as many values (GPyTorch's `Module.initialize`:
    /// `expand_as`, else `view_as`; e.g. a value of shape `(1,)` for a 0-dim
    /// outputscale).
    fn set_elementwise<F>(&self, value: &Array, per_element: &[usize], f: F) -> PyResult<()>
    where
        F: Fn(&mut Hyperparameters, &[f64]) -> PyResult<()>,
    {
        let model = self.snapshot();
        let mut target = model.aug_batch_shape();
        target.extend_from_slice(per_element);
        let values = broadcast_to(&value.0, &value.1, &target)
            .or_else(|| (value.0.len() == numel(&target)).then(|| value.0.clone()))
            .ok_or_else(|| {
                dimension_error(format!(
                    "a value of shape {:?} cannot be broadcast to the parameter shape {target:?}",
                    value.1
                ))
            })?;
        let k = numel(per_element);
        let mut hs: Vec<Hyperparameters> = (0..model.num_elements())
            .map(|e| model.element_hyperparameters(e).clone())
            .collect();
        for (e, h) in hs.iter_mut().enumerate() {
            f(h, &values[e * k..(e + 1) * k])?;
        }
        self.update(|m| {
            for (e, h) in hs.into_iter().enumerate() {
                m.set_element_hyperparameters(e, h)?;
            }
            Ok(())
        })
    }
}

#[pymethods]
impl PySingleTaskGP {
    /// Builds a model from `train_X` (`batch_shape x n x d`), `train_Y`
    /// (`batch_shape x n x m`) and optional observed noise variances
    /// `train_Yvar` (shape of `train_Y`).
    ///
    /// `outcome_transform=None` with `standardize_by_default=True` is BoTorch's
    /// `DEFAULT` (a fresh `Standardize(m, batch_shape)`); with `False` no
    /// outcome transform is used.
    #[new]
    #[pyo3(signature = (
        train_x,
        train_y,
        train_yvar=None,
        input_transform=None,
        outcome_transform=None,
        standardize_by_default=true,
        covar_module=None,
        likelihood=None,
        input_chain=None,
        covar_tree=None,
        input_chain_training=None,
        input_chain_prefixes=None,
        input_chains=None,
        outcome_chains=None,
        input_chain_params=None,
        mean_prior=None,
        mean_constraint=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        train_x: Array,
        train_y: Array,
        train_yvar: Option<Array>,
        input_transform: Option<PyRef<'_, PyNormalize>>,
        outcome_transform: Option<PyRef<'_, PyStandardize>>,
        standardize_by_default: bool,
        covar_module: Option<PyRef<'_, PyKernelSpec>>,
        likelihood: Option<PyRef<'_, PyLikelihoodSpec>>,
        input_chain: Option<Vec<Bound<'_, PyAny>>>,
        covar_tree: Option<PyRef<'_, PyKernelNode>>,
        input_chain_training: Option<Vec<bool>>,
        input_chain_prefixes: Option<Vec<String>>,
        input_chains: Option<Vec<Vec<Bound<'_, PyAny>>>>,
        outcome_chains: Option<Vec<Vec<PyRef<'_, crate::transform_steps::PyOutcomeStep>>>>,
        input_chain_params: Option<Vec<Vec<f64>>>,
        mean_prior: Option<PyRef<'_, PyPrior>>,
        mean_constraint: Option<PyRef<'_, PyConstraint>>,
    ) -> PyResult<Self> {
        let (batch, xs) = train_x.batches("train_X")?;
        let (ybatch, ys) = train_y.batches("train_Y")?;
        if ybatch != batch {
            return Err(dimension_error(format!(
                "Expected train_X and train_Y to have the same batch shape, got {batch:?} and \
                 {ybatch:?}."
            )));
        }
        let yvars = match train_yvar {
            Some(v) => {
                let (vb, vs) = v.batches("train_Yvar")?;
                if vb != batch {
                    return Err(dimension_error(
                        "Expected train_Yvar to have the shape of train_Y.",
                    ));
                }
                Some(vs)
            }
            None => None,
        };
        let outcome = match (outcome_transform, standardize_by_default) {
            _ if outcome_chains.is_some() => OutcomeTransformChoice::Chains(
                outcome_chains
                    .as_ref()
                    .expect("checked")
                    .iter()
                    .map(|steps| {
                        botorch_core::transforms::outcome::OutcomeChain::new(
                            steps.iter().map(|s| s.inner.clone()).collect(),
                        )
                    })
                    .collect(),
            ),
            (Some(st), _) => {
                if !st.batch_shape.is_empty() && st.batch_shape != batch {
                    return Err(dimension_error(format!(
                        "Expected Y.shape[:-2] to be {:?}, matching the `batch_shape` argument \
                         to `Standardize`, but got Y.shape[:-2]={batch:?}.",
                        st.batch_shape
                    )));
                }
                OutcomeTransformChoice::Standardize(
                    botorch_core::Standardize::new().with_min_stdv(st.inner[0].min_stdv()),
                )
            }
            (None, true) => OutcomeTransformChoice::Default,
            (None, false) => OutcomeTransformChoice::None,
        };
        let options = BatchedMultiOutputGPOptions {
            train_yvar: yvars,
            input_transform: input_transform.map(|t| t.for_model(&batch)).transpose()?,
            outcome_transform: outcome,
            covar_module: covar_module.map(|c| c.inner),
            covar_tree: covar_tree.map(|c| c.inner.clone()),
            likelihood: likelihood.map(|l| l.inner),
            input_chain: input_chain
                .as_deref()
                .map(|steps| {
                    crate::input_perturbation::chain_of(
                        steps,
                        input_chain_training.clone(),
                        input_chain_prefixes.clone(),
                    )
                })
                .transpose()?,
            input_chains: input_chains
                .as_deref()
                .map(|chains| {
                    chains
                        .iter()
                        .map(|steps| {
                            crate::input_perturbation::chain_of(
                                steps,
                                input_chain_training.clone(),
                                input_chain_prefixes.clone(),
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
                .transpose()?,
            input_chain_params,
            mean_module: (mean_prior.is_some() || mean_constraint.is_some()).then(|| {
                ConstantMeanSpec {
                    prior: mean_prior.map(|p| p.inner),
                    constraint: mean_constraint.map(|c| c.inner),
                }
            }),
            ..Default::default()
        };
        // (A perturbation function is called while the chain is fitted.)
        let model = threads::run(py, || BatchedMultiOutputGP::new(xs, ys, batch, options))
            .into_py_result()?;
        Ok(Self {
            inner: RwLock::new(Arc::new(model)),
        })
    }

    /// A deep copy of the model.
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(Arc::new((*self.snapshot()).clone())),
        }
    }

    /// The model of the outputs `idcs` (BoTorch's `subset_output`).
    fn subset_output(&self, idcs: Vec<usize>) -> PyResult<Self> {
        let model = self.snapshot().subset_output(&idcs).into_py_result()?;
        Ok(Self::from_model(model))
    }

    /// Sets the fixed `tau`s of the SAAS priors of every element's kernel
    /// tree (`taus[e]`: one value per SAAS prior with a fixed `tau`).
    fn set_saas_taus(&self, taus: Vec<Vec<f64>>) -> PyResult<()> {
        self.update(|m| m.set_saas_taus(&taus))
    }

    /// The unbatched model of batch element `batch` (row-major).
    fn batch_element(&self, batch: usize) -> PyResult<Self> {
        let model = self.snapshot().batch_element(batch).into_py_result()?;
        Ok(Self::from_model(model))
    }

    // ------------------------------------------------------------------
    // Data and configuration
    // ------------------------------------------------------------------

    /// Input dimension `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.snapshot().dim()
    }

    /// Number of training points `n` (per batch element).
    #[getter]
    fn num_train(&self) -> usize {
        self.snapshot().num_train()
    }

    /// Number of outputs `m`.
    #[getter]
    fn num_outputs(&self) -> usize {
        self.snapshot().num_outputs()
    }

    /// The batch shape of the training data (BoTorch's `model.batch_shape`).
    #[getter]
    fn batch_shape(&self) -> Vec<usize> {
        self.snapshot().batch_shape().to_vec()
    }

    /// BoTorch's `_aug_batch_shape` (the batch shape followed by `m` if
    /// `m > 1`).
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        self.snapshot().aug_batch_shape()
    }

    /// Whether the observation noise is fixed (`train_Yvar` given).
    #[getter]
    fn is_fixed_noise(&self) -> bool {
        self.snapshot().is_fixed_noise()
    }

    /// Training inputs in the original space (`batch_shape x n x d`).
    fn train_inputs(&self) -> ArrayBuf {
        let m = self.snapshot();
        let mats: Vec<Matrix> = (0..m.batch_size())
            .map(|b| m.train_inputs(b).clone())
            .collect();
        crate::arrays::stack_out(&mats, m.batch_shape(), m.num_train(), m.dim()).into()
    }

    /// Training targets in the original space (`batch_shape x n x m`).
    fn train_targets(&self) -> ArrayBuf {
        let m = self.snapshot();
        let mats: Vec<Matrix> = (0..m.batch_size())
            .map(|b| m.train_targets(b).clone())
            .collect();
        crate::arrays::stack_out(&mats, m.batch_shape(), m.num_train(), m.num_outputs()).into()
    }

    /// Observed noise variances in the original space, if given.
    fn train_yvar(&self) -> Option<ArrayBuf> {
        let m = self.snapshot();
        m.train_yvar(0)?;
        let mats: Vec<Matrix> = (0..m.batch_size())
            .map(|b| m.train_yvar(b).expect("fixed noise").clone())
            .collect();
        Some(
            crate::arrays::stack_out(&mats, m.batch_shape(), m.num_train(), m.num_outputs()).into(),
        )
    }

    /// Training inputs after the input transform (`batch_shape x n x d`).
    fn transformed_train_inputs(&self) -> ArrayBuf {
        let m = self.snapshot();
        let mats: Vec<Matrix> = (0..m.batch_size())
            .map(|b| m.model(b, 0).transformed_train_inputs().clone())
            .collect();
        let d_tf = mats[0].ncols();
        crate::arrays::stack_out(&mats, m.batch_shape(), m.num_train(), d_tf).into()
    }

    /// Training targets after the outcome transform, BoTorch's
    /// `model.train_targets` (`_aug_batch_shape x n`).
    fn transformed_train_targets(&self) -> ArrayBuf {
        let m = self.snapshot();
        let mut data = Vec::with_capacity(m.num_elements() * m.num_train());
        for gp in m.models() {
            data.extend(gp.transformed_train_targets().iter());
        }
        let mut shape = m.aug_batch_shape();
        shape.push(m.num_train());
        ArrayBuf(data, shape)
    }

    /// Efficient leave-one-out moments and observations in transformed space,
    /// all laid out as `batch x n x m`.
    #[pyo3(signature = (observation_noise=true))]
    fn loo_predictions(
        &self,
        py: Python<'_>,
        observation_noise: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf, ArrayBuf, Option<ArrayBuf>)> {
        let model = self.snapshot();
        let m = model.num_outputs();
        let n = model.num_train();
        let mut shape = model.batch_shape().to_vec();
        shape.extend([n, m]);
        let fixed = model.train_yvar(0).is_some();
        let (mean, variance, targets, noise) = threads::run(py, || -> botorch_core::Result<_> {
            let results = model
                .models()
                .par_iter()
                .map(|gp| gp.loo_predictions(observation_noise))
                .collect::<botorch_core::Result<Vec<_>>>()?;
            let mut mean = Vec::with_capacity(model.batch_size() * n * m);
            let mut variance = Vec::with_capacity(mean.capacity());
            let mut targets = Vec::with_capacity(mean.capacity());
            let mut noise = Vec::with_capacity(mean.capacity());
            let noises: Vec<_> = model
                .models()
                .iter()
                .map(|gp| gp.likelihood_noise())
                .collect();
            for b in 0..model.batch_size() {
                for i in 0..n {
                    for t in 0..m {
                        let gp = model.model(b, t);
                        let (mu, var) = &results[b * m + t];
                        mean.push(mu[i]);
                        variance.push(var[i]);
                        targets.push(gp.transformed_train_targets()[i]);
                        if fixed {
                            noise.push(noises[b * m + t][i]);
                        }
                    }
                }
            }
            Ok((mean, variance, targets, noise))
        })
        .into_py_result()?;
        Ok((
            ArrayBuf(mean, shape.clone()),
            ArrayBuf(variance, shape.clone()),
            ArrayBuf(targets, shape.clone()),
            fixed.then_some(ArrayBuf(noise, shape)),
        ))
    }

    /// Samples of independent Gaussian marginals, with collapsed or full
    /// batch base samples. Used by the leave-one-out posterior.
    #[staticmethod]
    fn diagonal_samples(
        py: Python<'_>,
        mean: Vec<f64>,
        variance: Vec<f64>,
        base: Array,
        qm: usize,
    ) -> PyResult<Vec<f64>> {
        if qm == 0 || !mean.len().is_multiple_of(qm) || variance.len() != mean.len() {
            return Err(dimension_error("inconsistent diagonal posterior shape"));
        }
        let shape = &base.1;
        let samples = shape.first().copied().unwrap_or(0);
        let collapsed = shape.as_slice() == [samples, qm];
        if !collapsed && shape.as_slice() != [samples, mean.len() / qm, qm] {
            return Err(dimension_error(
                "invalid diagonal posterior base sample shape",
            ));
        }
        Ok(threads::run(py, || {
            let mut out = Vec::with_capacity(samples * mean.len());
            for s in 0..samples {
                for i in 0..mean.len() {
                    let z = if collapsed {
                        base.0[s * qm + i % qm]
                    } else {
                        base.0[s * mean.len() + i]
                    };
                    out.push(mean[i] + variance[i].max(0.0).sqrt() * z);
                }
            }
            out
        }))
    }

    /// The fitted input transform, if any.
    fn input_transform(&self) -> Option<PyNormalize> {
        let m = self.snapshot();
        let tf = m.input_transform()?;
        let learn = m
            .model(0, 0)
            .input_transform()
            .map(|_| match tf {
                botorch_core::BatchedInputTransform::Shared(t) => t.learn_bounds(),
                botorch_core::BatchedInputTransform::PerBatch(ts) => ts[0].learn_bounds(),
            })
            .unwrap_or(false);
        Some(PyNormalize::fitted(tf, learn, m.batch_shape()))
    }

    /// The fitted steps of a chained (or one-to-many) input transform, if
    /// any: a `Normalize` per normalization step and the number of
    /// perturbations `n_w` per `InputPerturbation`.
    fn input_chain_state(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyAny>>>> {
        self.snapshot()
            .input_chain()
            .map(|c| crate::input_perturbation::chain_state(py, c))
            .transpose()
    }

    /// Posterior points per input point (`n_w` with an `InputPerturbation`).
    #[getter]
    fn points_per_input(&self) -> usize {
        self.snapshot().points_per_input()
    }

    /// The fitted steps of the input transform of every batch element (see
    /// `input_chain_state`), if the model has a general input transform.
    fn input_chain_states(&self, py: Python<'_>) -> PyResult<Option<Vec<Vec<Py<PyAny>>>>> {
        let m = self.snapshot();
        if m.input_chain_of(0).is_none() {
            return Ok(None);
        }
        (0..m.batch_size())
            .map(|b| {
                crate::input_perturbation::chain_state(
                    py,
                    m.input_chain_of(b).expect("every element has the chain"),
                )
            })
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    /// The training inputs as transformed when the model was built
    /// (`batch_shape x n x d'`; BoTorch's `transformed_X` in the
    /// constructor).
    fn construction_train_inputs(&self) -> ArrayBuf {
        let m = self.snapshot();
        let mats: Vec<Matrix> = (0..m.batch_size())
            .map(|b| m.model(b, 0).construction_train_inputs().clone())
            .collect();
        let d = mats[0].ncols();
        crate::arrays::stack_out(&mats, m.batch_shape(), m.num_train(), d).into()
    }

    /// The dimension of the transformed inputs.
    #[getter]
    fn transformed_dim(&self) -> usize {
        self.snapshot().models()[0].transformed_dim()
    }

    /// Whether the posterior of each output is a `TransformedPosterior`.
    #[getter]
    fn transformed_outputs(&self) -> Vec<bool> {
        self.snapshot().transformed_outputs()
    }

    /// The fitted outcome steps of every element (`_aug_batch_shape`
    /// elements, row-major), if the model has a general outcome transform.
    fn outcome_chain_states(&self, py: Python<'_>) -> PyResult<Option<Vec<Vec<Py<PyAny>>>>> {
        let m = self.snapshot();
        if m.models()[0].outcome_chain().is_none() {
            return Ok(None);
        }
        m.models()
            .iter()
            .map(|g| {
                g.outcome_chain()
                    .expect("every element has a chain")
                    .steps()
                    .iter()
                    .map(|s| {
                        crate::transform_steps::PyOutcomeStep { inner: s.clone() }.into_py_any(py)
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    /// The fitted outcome transform, if any (statistics of shape
    /// `batch_shape x 1 x m`).
    fn outcome_transform(&self) -> PyResult<Option<PyStandardize>> {
        let m = self.snapshot();
        if m.models()[0].outcome_transform().is_none() || m.models()[0].outcome_chain().is_some() {
            return Ok(None);
        }
        let mut means = Vec::with_capacity(m.batch_size());
        let mut stdvs = Vec::with_capacity(m.batch_size());
        let mut min_stdv = 0.0;
        for b in 0..m.batch_size() {
            let (mut mb, mut sb) = (Vec::new(), Vec::new());
            for t in 0..m.num_outputs() {
                let st = m
                    .outcome_transform(b, t)
                    .expect("all elements are standardized");
                mb.push(st.means()[0]);
                sb.push(st.stdvs()[0]);
                min_stdv = st.min_stdv();
            }
            means.push(mb);
            stdvs.push(sb);
        }
        PyStandardize::from_stats(means, stdvs, min_stdv, m.batch_shape().to_vec()).map(Some)
    }

    /// The covariance-module configuration (shared by all elements): the
    /// `KernelSpec`, or `None` for a kernel tree.
    fn covar_module(&self) -> Option<PyKernelSpec> {
        self.snapshot().models()[0]
            .covar_module()
            .as_standard()
            .map(|s| PyKernelSpec { inner: *s })
    }

    /// The kernel tree (shared by all elements), if the covariance module is
    /// one.
    fn covar_tree(&self) -> Option<PyKernelNode> {
        self.snapshot().models()[0]
            .covar_module()
            .as_tree()
            .map(|t| PyKernelNode { inner: t.clone() })
    }

    /// The homoskedastic likelihood configuration (`None` for fixed noise).
    fn likelihood_spec(&self) -> Option<PyLikelihoodSpec> {
        match self.snapshot().models()[0].likelihood() {
            Likelihood::Gaussian(spec) => Some(PyLikelihoodSpec { inner: *spec }),
            Likelihood::FixedNoise { .. } => None,
        }
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    /// BoTorch parameter names and total sizes in flat order.
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.snapshot().parameter_names()
    }

    /// BoTorch parameter names and shapes.
    fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.snapshot().parameter_shapes()
    }

    /// The raw hyperparameters in flat order (BoTorch's `named_parameters()`).
    fn raw_parameters(&self) -> Vec<f64> {
        self.snapshot().raw_parameters().as_slice().to_vec()
    }

    /// Sets the raw hyperparameters from a flat vector.
    fn set_raw_parameters(&self, values: Vec<f64>) -> PyResult<()> {
        self.update(|m| m.set_raw_parameters(&values))
    }

    /// Optimizer bounds of the raw hyperparameters in flat order.
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.snapshot().raw_parameter_bounds()
    }

    /// The lengthscales of all elements (`_aug_batch_shape x d`, flat).
    #[getter]
    fn lengthscale(&self) -> Vec<f64> {
        self.snapshot()
            .models()
            .iter()
            .flat_map(|g| g.lengthscale().iter().copied().collect::<Vec<_>>())
            .collect()
    }

    /// The outputscales of all elements (`None` without `ScaleKernel`).
    #[getter]
    fn outputscale(&self) -> Option<Vec<f64>> {
        self.snapshot()
            .models()
            .iter()
            .map(|g| g.outputscale())
            .collect()
    }

    /// The likelihood noise variances (transformed outcome space) of all
    /// elements, flat: one value per element, or one per training point and
    /// element for fixed noise.
    #[getter]
    fn noise(&self) -> Vec<f64> {
        self.snapshot()
            .models()
            .iter()
            .flat_map(|g| g.likelihood_noise().iter().copied().collect::<Vec<_>>())
            .collect()
    }

    /// The constant means (transformed outcome space) of all elements.
    #[getter]
    fn constant(&self) -> Vec<f64> {
        self.snapshot()
            .models()
            .iter()
            .map(|g| g.constant())
            .collect()
    }

    /// Sets the lengthscales from a value broadcastable to
    /// `_aug_batch_shape x 1 x d`.
    fn set_lengthscale(&self, value: Array) -> PyResult<()> {
        let model = self.snapshot();
        let d = model.dim();
        let Some(spec) = model.models()[0].covar_module().as_standard() else {
            return Err(unsupported(
                "set the lengthscales of a kernel tree through its kernels",
            ));
        };
        let constraint = spec.lengthscale_constraint;
        self.set_elementwise(&value, &[1, d], |h, v| {
            for (r, x) in h.raw_lengthscale.iter_mut().zip(v) {
                *r = constraint.inverse_transform(*x);
            }
            Ok(())
        })
    }

    /// Sets the outputscales from a value broadcastable to `_aug_batch_shape`.
    fn set_outputscale(&self, value: Array) -> PyResult<()> {
        let Some(scale) = self.snapshot().models()[0]
            .covar_module()
            .as_standard()
            .and_then(|s| s.scale)
        else {
            return Err(unsupported("The covariance module has no outputscale."));
        };
        self.set_elementwise(&value, &[], |h, v| {
            h.raw_outputscale = Some(scale.outputscale_constraint.inverse_transform(v[0]));
            Ok(())
        })
    }

    /// Sets the (homoskedastic) noise variances (transformed space) from a
    /// value broadcastable to `_aug_batch_shape x 1`.
    fn set_noise(&self, value: Array) -> PyResult<()> {
        let Likelihood::Gaussian(spec) = *self.snapshot().models()[0].likelihood() else {
            return Err(unsupported(
                "The noise of a fixed-noise likelihood (train_Yvar) cannot be set.",
            ));
        };
        self.set_elementwise(&value, &[1], |h, v| {
            h.raw_noise = Some(spec.noise_constraint.inverse_transform(v[0]));
            Ok(())
        })
    }

    /// Sets the constant means from a value broadcastable to `_aug_batch_shape`.
    fn set_constant(&self, value: Array) -> PyResult<()> {
        self.set_elementwise(&value, &[], |h, v| {
            h.raw_constant = v[0];
            Ok(())
        })
    }

    /// The observation-noise variances (original space) that
    /// `posterior(X, observation_noise=True)` adds, one per element.
    fn observation_noise(&self) -> Vec<f64> {
        self.snapshot()
            .models()
            .iter()
            .map(|g| g.observation_noise())
            .collect()
    }

    // ------------------------------------------------------------------
    // Posterior
    // ------------------------------------------------------------------

    /// Marginal posterior means and variances at the points of `X`
    /// (`... x q x d`): two arrays of shape `out_batch x q x m'` (variances
    /// clamped at 1e-10 like `posterior.variance`), where `out_batch` is the
    /// broadcast of the batch shapes of `X` and of the model and `m'` the
    /// number of selected outputs (1 with a posterior transform).
    #[pyo3(signature = (x, observation_noise=false, output_indices=None, transform=None, base=false))]
    fn posterior_marginals(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let model = self.model_ref();
        let view = View::for_model(&model, output_indices, transform)?;
        if base {
            return base_marginals(py, &model, &x, observation_noise, view.base()?);
        }
        let (mean, var, shape) =
            threads::run(py, || marginals(&model, &x, observation_noise, &view))?;
        Ok((ArrayBuf(mean, shape.clone()), ArrayBuf(var, shape)))
    }

    /// Joint posteriors of the point sets of `X` (`... x q x d`): means
    /// (`out_batch x q x m'`) and covariances (`out_batch x q·m' x q·m'`,
    /// BoTorch's output-major layout).
    #[pyo3(signature = (x, observation_noise=false, output_indices=None, transform=None, base=false))]
    fn posterior_joint(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let model = self.model_ref();
        let mut view = View::for_model(&model, output_indices, transform)?;
        if base {
            view = view.base()?;
        }
        let q = x.1.get(x.1.len().wrapping_sub(2)).copied().unwrap_or(0) * model.points_per_input();
        let mo = view.num_outputs;
        let (shape, posts) = threads::run(py, || joints(&model, &x, observation_noise, &view))?;
        Ok(joint_out(shape, &posts, q, mo))
    }

    /// Posterior samples at the point sets of `X` from base samples (see the
    /// module documentation); returns `S x out_batch x q x m'`.
    #[pyo3(signature = (x, base_samples, observation_noise=false, output_indices=None, transform=None, base=false))]
    #[allow(clippy::too_many_arguments)]
    fn posterior_samples(
        &self,
        py: Python<'_>,
        x: Array,
        base_samples: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<ArrayBuf> {
        let model = self.model_ref();
        let mut view = View::for_model(&model, output_indices, transform)?;
        if base {
            view = view.base()?;
        }
        threads::run(py, || {
            samples(&model, &x, &base_samples, observation_noise, &view)
        })
    }

    // ------------------------------------------------------------------
    // Marginal log likelihood and fitting
    // ------------------------------------------------------------------

    /// Replaces transformed training targets and optional fixed noise.
    #[pyo3(signature = (targets, fixed_noise=None, strict=true))]
    fn replace_transformed_train_data(
        &self,
        targets: Array,
        fixed_noise: Option<Array>,
        strict: bool,
    ) -> PyResult<()> {
        let model = self.snapshot();
        let mut shape = model.aug_batch_shape();
        shape.push(model.num_train());
        let size = model.num_elements() * model.num_train();
        if targets.0.len() != size
            || (strict && targets.1 != shape)
            || fixed_noise
                .as_ref()
                .is_some_and(|v| v.0.len() != size || (strict && v.1 != shape))
        {
            return Err(dimension_error(format!("replacement targets/noise must have shape {shape:?}; changing n requires new training inputs")));
        }
        let n = model.num_train();
        let values = targets.0.chunks(n).map(Vector::from_column_slice).collect();
        let noise = fixed_noise.map(|v| v.0.chunks(n).map(Vector::from_column_slice).collect());
        self.update(|m| m.replace_transformed_train_data(values, noise))
    }

    /// Installs sparse outlier variances, invalidating cached training factors.
    #[pyo3(signature = (support, raw, convex=true, lower=0.0, upper=0.999, prior=None))]
    fn set_sparse_outlier_noise(
        &self,
        support: Vec<usize>,
        raw: Vec<f64>,
        convex: bool,
        lower: f64,
        upper: f64,
        prior: Option<PyRef<'_, PyPrior>>,
    ) -> PyResult<()> {
        let state = botorch_core::models::relevance_pursuit::SparseOutlierState {
            support,
            raw,
            convex,
            bounds: (lower, upper),
            prior: prior.map(|p| p.inner),
        };
        self.update(|m| m.set_sparse_outlier_noise(state))
    }

    /// Current outlier support and active raw variances.
    fn sparse_outlier_state(&self) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let model = self.snapshot();
        let state = model.models()[0]
            .sparse_outlier_noise()
            .ok_or_else(|| value_error("model has no sparse outlier noise"))?;
        Ok((state.support.clone(), state.raw.clone()))
    }

    /// LOO expansion criterion (all observation indices).
    fn sparse_outlier_expansion(&self, py: Python<'_>, loo: bool) -> PyResult<Vec<f64>> {
        let model = self.snapshot();
        threads::run(py, || {
            let model = &model.models()[0];
            if loo {
                return model.relevance_expansion_objective();
            }
            let mut trial = model.clone();
            let mut state = model
                .sparse_outlier_noise()
                .ok_or_else(|| {
                    botorch_core::BotorchError::InvalidArgument(
                        "model has no sparse outlier noise".into(),
                    )
                })?
                .clone();
            let mut dense = vec![0.0; model.num_train()];
            for (&i, &r) in state.support.iter().zip(&state.raw) {
                dense[i] = r;
            }
            state.support = (0..model.num_train()).collect();
            state.raw = dense;
            trial.set_sparse_outlier_noise(Some(state))?;
            let (_, _, gradient) = botorch_core::mll::ExactMarginalLogLikelihood::new(&trial)
                .value_and_rho_grad(trial.raw_parameters().as_slice())?;
            Ok(gradient.as_slice().to_vec())
        })
        .into_py_result()
    }

    /// The exact MLL of every element (divided by `n`, priors included, as in
    /// GPyTorch) at the current or the given flat raw hyperparameters, with
    /// shape `_aug_batch_shape`.
    #[pyo3(signature = (raw_parameters=None))]
    fn mll(&self, py: Python<'_>, raw_parameters: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let model = self.snapshot();
        let values = threads::run(py, || match &raw_parameters {
            Some(p) => model.mll_values_and_grad(p).map(|(v, _)| v),
            None => model.mll_values(),
        })
        .into_py_result()?;
        Ok((values, model.aug_batch_shape()))
    }

    /// The exact MLL of every element and the gradient of their sum with
    /// respect to the flat raw hyperparameters.
    #[pyo3(signature = (raw_parameters=None))]
    fn mll_value_and_grad(
        &self,
        py: Python<'_>,
        raw_parameters: Option<Vec<f64>>,
    ) -> PyResult<(ArrayOut, Vec<f64>)> {
        let model = self.snapshot();
        let params = raw_parameters.unwrap_or_else(|| model.raw_parameters().as_slice().to_vec());
        let (v, g) = threads::run(py, || model.mll_values_and_grad(&params)).into_py_result()?;
        Ok(((v, model.aug_batch_shape()), g.as_slice().to_vec()))
    }

    /// Fits the hyperparameters by maximizing the MLL with L-BFGS-B
    /// (`fit_gpytorch_mll`; one independent problem per element for batched
    /// and multi-output models). On failure the model is left unchanged.
    #[pyo3(signature = (
        maxcor=None,
        pgtol=None,
        ftol=None,
        maxiter=None,
        maxfun=None,
        maxls=None,
        max_attempts=None,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        maxcor: Option<usize>,
        pgtol: Option<f64>,
        ftol: Option<f64>,
        maxiter: Option<usize>,
        maxfun: Option<usize>,
        maxls: Option<usize>,
        max_attempts: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<FitSummary> {
        let options = options_from(
            maxcor,
            pgtol,
            ftol,
            maxiter,
            maxfun,
            maxls,
            max_attempts,
            seed,
        )?;
        let mut model = (*self.snapshot()).clone();
        let result = threads::run(py, || fit_gpytorch_mll(&mut model, options)).into_py_result()?;
        self.replace(model);
        Ok(summary(&result))
    }
}

/// A list of independent models (BoTorch's `ModelListGP`), sharing the
/// current snapshots of its sub-models.
#[pyclass(frozen, name = "ModelListGP", module = "botorch_rs._botorch_rs")]
pub struct PyModelListGP {
    inner: ModelListGP,
}

#[pymethods]
impl PyModelListGP {
    /// A list of the current snapshots of `models`.
    #[new]
    fn new(models: Vec<PyRef<'_, PySingleTaskGP>>) -> PyResult<Self> {
        let inner = ModelListGP::from_shared(models.iter().map(|m| m.snapshot()).collect())
            .into_py_result()?;
        Ok(Self { inner })
    }

    /// Input dimension `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Total number of outputs.
    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    /// The common batch shape of the sub-models.
    #[getter]
    fn batch_shape(&self) -> Vec<usize> {
        self.inner.batch_shape().to_vec()
    }

    /// Posterior points per input point (`n_w` with an `InputPerturbation`).
    #[getter]
    fn points_per_input(&self) -> usize {
        self.inner.points_per_input()
    }

    /// See `SingleTaskGP.posterior_marginals`.
    #[pyo3(signature = (x, observation_noise=false, output_indices=None, transform=None, base=false))]
    fn posterior_marginals(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let model = ModelRef::List(self.inner.clone());
        let view = View::for_model(&model, output_indices, transform)?;
        if base {
            return base_marginals(py, &model, &x, observation_noise, view.base()?);
        }
        let (mean, var, shape) =
            threads::run(py, || marginals(&model, &x, observation_noise, &view))?;
        Ok((ArrayBuf(mean, shape.clone()), ArrayBuf(var, shape)))
    }

    /// See `SingleTaskGP.posterior_joint`.
    #[pyo3(signature = (x, observation_noise=false, output_indices=None, transform=None, base=false))]
    fn posterior_joint(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let model = ModelRef::List(self.inner.clone());
        let mut view = View::for_model(&model, output_indices, transform)?;
        if base {
            view = view.base()?;
        }
        let q = x.1.get(x.1.len().wrapping_sub(2)).copied().unwrap_or(0) * model.points_per_input();
        let mo = view.num_outputs;
        let (shape, posts) = threads::run(py, || joints(&model, &x, observation_noise, &view))?;
        Ok(joint_out(shape, &posts, q, mo))
    }

    /// See `SingleTaskGP.posterior_samples`.
    #[pyo3(signature = (x, base_samples, observation_noise=false, output_indices=None, transform=None, base=false))]
    #[allow(clippy::too_many_arguments)]
    fn posterior_samples(
        &self,
        py: Python<'_>,
        x: Array,
        base_samples: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<ArrayBuf> {
        let model = ModelRef::List(self.inner.clone());
        let mut view = View::for_model(&model, output_indices, transform)?;
        if base {
            view = view.base()?;
        }
        threads::run(py, || {
            samples(&model, &x, &base_samples, observation_noise, &view)
        })
    }
}

/// A model as seen by acquisition functions (BoTorch's `Model`), optionally
/// through a `ScalarizedPosteriorTransform`.
#[pyclass(frozen, name = "Model", module = "botorch_rs._botorch_rs")]
pub struct PyModel {
    /// The model.
    pub inner: Arc<dyn Model>,
}

fn with_transform(
    model: Arc<dyn Model>,
    transform: Option<TransformArg>,
) -> PyResult<Arc<dyn Model>> {
    let t = transform
        .map(|(w, offset)| ScalarizedPosteriorTransform::new(Vector::from_vec(w), offset))
        .transpose()
        .into_py_result()?;
    ScalarizedModel::wrap(model, t).into_py_result()
}

#[pymethods]
impl PyModel {
    /// The current snapshot of a `SingleTaskGP`, seen through the posterior
    /// transform `(weights, offset)` if given.
    #[staticmethod]
    #[pyo3(signature = (model, transform=None))]
    fn from_single_task_gp(
        model: PyRef<'_, PySingleTaskGP>,
        transform: Option<TransformArg>,
    ) -> PyResult<Self> {
        let m: Arc<dyn Model> = model.snapshot();
        Ok(Self {
            inner: with_transform(m, transform)?,
        })
    }

    /// The current snapshot of an `LCEMGP` (over its output tasks, or over
    /// the tasks `output_tasks` if given), seen through the posterior
    /// transform if given.
    #[staticmethod]
    #[pyo3(signature = (model, output_tasks=None, transform=None))]
    fn from_lcemgp(
        model: PyRef<'_, PyLCEMGP>,
        output_tasks: Option<Vec<i64>>,
        transform: Option<TransformArg>,
    ) -> PyResult<Self> {
        let snapshot = model.snapshot();
        let m: Arc<dyn Model> = match output_tasks {
            Some(tasks) => Arc::new(snapshot.with_output_tasks(&tasks).into_py_result()?),
            None => snapshot,
        };
        Ok(Self {
            inner: with_transform(m, transform)?,
        })
    }

    /// The current snapshot of an `LCEMGP` with inputs that include the
    /// task column (`d + 1` columns, one output).
    #[staticmethod]
    fn from_lcemgp_task_inputs(model: PyRef<'_, PyLCEMGP>) -> Self {
        Self {
            inner: Arc::new(LcemTaskInputs::new(model.snapshot())),
        }
    }

    /// A list of independent models (BoTorch's `ModelList`, e.g. a
    /// `ModelListGP` of `LCEMGP`s), seen through the posterior transform if
    /// given.
    #[staticmethod]
    #[pyo3(signature = (models, transform=None))]
    fn from_models(
        models: Vec<PyRef<'_, PyModel>>,
        transform: Option<TransformArg>,
    ) -> PyResult<Self> {
        let list = ModelList::new(models.iter().map(|m| Arc::clone(&m.inner)).collect())
            .into_py_result()?;
        Ok(Self {
            inner: with_transform(Arc::new(list), transform)?,
        })
    }

    /// See `SingleTaskGP.posterior_marginals` (unbatched models).
    #[pyo3(signature = (x, observation_noise=false, output_indices=None, transform=None, base=false))]
    fn posterior_marginals(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let model = ModelRef::Dyn(Arc::clone(&self.inner));
        let view = View::for_model(&model, output_indices, transform)?;
        if base {
            return base_marginals(py, &model, &x, observation_noise, view.base()?);
        }
        let (mean, var, shape) =
            threads::run(py, || marginals(&model, &x, observation_noise, &view))?;
        Ok((ArrayBuf(mean, shape.clone()), ArrayBuf(var, shape)))
    }

    /// See `SingleTaskGP.posterior_joint` (unbatched models).
    #[pyo3(signature = (x, observation_noise=false, output_indices=None, transform=None, base=false))]
    fn posterior_joint(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let model = ModelRef::Dyn(Arc::clone(&self.inner));
        let mut view = View::for_model(&model, output_indices, transform)?;
        if base {
            view = view.base()?;
        }
        let q = x.1.get(x.1.len().wrapping_sub(2)).copied().unwrap_or(0) * model.points_per_input();
        let mo = view.num_outputs;
        let (shape, posts) = threads::run(py, || joints(&model, &x, observation_noise, &view))?;
        Ok(joint_out(shape, &posts, q, mo))
    }

    /// See `SingleTaskGP.posterior_samples` (unbatched models).
    #[pyo3(signature = (x, base_samples, observation_noise=false, output_indices=None, transform=None, base=false))]
    #[allow(clippy::too_many_arguments)]
    fn posterior_samples(
        &self,
        py: Python<'_>,
        x: Array,
        base_samples: Array,
        observation_noise: bool,
        output_indices: Option<Vec<usize>>,
        transform: Option<TransformArg>,
        base: bool,
    ) -> PyResult<ArrayBuf> {
        let model = ModelRef::Dyn(Arc::clone(&self.inner));
        let mut view = View::for_model(&model, output_indices, transform)?;
        if base {
            view = view.base()?;
        }
        threads::run(py, || {
            samples(&model, &x, &base_samples, observation_noise, &view)
        })
    }

    /// A `ModelListGP`, seen through the posterior transform if given.
    #[staticmethod]
    #[pyo3(signature = (model, transform=None))]
    fn from_model_list(
        model: PyRef<'_, PyModelListGP>,
        transform: Option<TransformArg>,
    ) -> PyResult<Self> {
        let m: Arc<dyn Model> = Arc::new(model.inner.clone());
        Ok(Self {
            inner: with_transform(m, transform)?,
        })
    }

    /// Input dimension `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Number of outputs (1 with a posterior transform).
    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    /// The batch shape of the model.
    #[getter]
    fn batch_shape(&self) -> Vec<usize> {
        self.inner.batch_shape().to_vec()
    }

    /// Posterior points per input point (`n_w` with an `InputPerturbation`).
    #[getter]
    fn points_per_input(&self) -> usize {
        self.inner.points_per_input()
    }

    /// Whether the model is deterministic (e.g. a `PosteriorMeanModel`).
    #[getter]
    fn is_deterministic(&self) -> bool {
        self.inner.is_deterministic()
    }
}
