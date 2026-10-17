//! [`BatchedMultiOutputGP`]: BoTorch's `SingleTaskGP` with `m ≥ 1` outputs
//! and an optional batch shape (the semantics of BoTorch's
//! `BatchedMultiOutputGPyTorchModel`).
//!
//! BoTorch models the outputs of a multi-output `SingleTaskGP` independently
//! by moving them into a batch dimension: training data of shape
//! `batch_shape × n × d` / `batch_shape × n × m` become a batch of
//! `batch_shape × m` independent single-output GPs (`_aug_batch_shape`), each
//! with its own hyperparameters, standardization (`Standardize(m,
//! batch_shape)`) and noise. This module represents exactly that batch as one
//! [`SingleTaskGP`] per batch element and output, so every element reproduces
//! the single-output computations (validated against BoTorch to ~1e-14).
//!
//! * The joint posterior at `q` points of batch element `b` is the
//!   block-diagonal Gaussian over the `m` outputs in BoTorch's output-major
//!   layout (see [`crate::posterior`]).
//! * The flat hyperparameter vector follows BoTorch's `named_parameters()`:
//!   parameter by parameter (`raw_noise`, `raw_constant`, `raw_outputscale`,
//!   `raw_lengthscale`), each over all elements in row-major order of
//!   `_aug_batch_shape` (shapes `aug × 1`, `aug`, `aug`, `aug × 1 × d`).
//! * Fitting runs one independent L-BFGS-B problem per element (see
//!   [`crate::fit`]).
//!
//! A `Normalize` input transform can be shared by all batch elements (BoTorch's
//! `Normalize(d)`, with bounds learned from the inputs of *all* batch
//! elements) or given per batch element (`Normalize(d, batch_shape=...)` or
//! batched bounds); see [`BatchedInputTransform`]. The covariance module and
//! likelihood options are applied to every element with independent
//! hyperparameters (BoTorch's modules with `batch_shape = _aug_batch_shape`).

use std::ops::Range;

use crate::error::{BotorchError, Result};
use crate::kernels::{KernelKind, KernelNode, KernelSpec};
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::gp::{GaussianLikelihoodSpec, Hyperparameters, Likelihood, ParameterLayout};
use crate::models::model::check_vjp_shapes_expanded;
use crate::models::{Model, SingleTaskGP, SingleTaskGPOptions};
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::transforms::input::ChainedInputTransform;
use crate::transforms::outcome::SampleTransform;
use crate::transforms::{Normalize, OutcomeTransformChoice, Standardize};
use crate::{Matrix, Vector};

/// How a `Normalize` input transform applies to the batch elements of a
/// [`BatchedMultiOutputGP`].
#[derive(Clone, Debug, PartialEq)]
pub enum BatchedInputTransform {
    /// One transform for all batch elements (BoTorch's `Normalize(d)` without
    /// `batch_shape`): learned bounds are computed from the inputs of all
    /// batch elements together.
    Shared(Normalize),
    /// One transform per batch element in row-major order of the batch shape
    /// (BoTorch's `Normalize(d, batch_shape=batch_shape)` or batched bounds).
    PerBatch(Vec<Normalize>),
}

/// Options of [`BatchedMultiOutputGP::new`] (BoTorch's keyword arguments of
/// `SingleTaskGP`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BatchedMultiOutputGPOptions {
    /// Observed noise variances, one `n × m` matrix per batch element; if
    /// given, fixed-noise likelihoods are used.
    pub train_yvar: Option<Vec<Matrix>>,
    /// Optional input transform (BoTorch default: none).
    pub input_transform: Option<BatchedInputTransform>,
    /// A general input transform (`ChainedInputTransform` /
    /// `InputPerturbation`) instead of `input_transform`, shared by all batch
    /// elements and outputs; learnable `Normalize` bounds are fitted to the
    /// inputs of all batch elements together (BoTorch's `Normalize(d)`).
    pub input_chain: Option<ChainedInputTransform>,
    /// One general input transform per batch element (BoTorch's
    /// `BatchBroadcastedInputTransform` over the batch dimension, or a
    /// batched `Warp`), each fitted to the inputs of its batch element,
    /// instead of `input_chain`.
    pub input_chains: Option<Vec<ChainedInputTransform>>,
    /// The learned parameters of `input_chain` for every batch element (a
    /// `Warp` with `batch_shape`: one set of concentrations per batch
    /// element of a single-output model, while its normalization is learned
    /// from the inputs of all batch elements).
    pub input_chain_params: Option<Vec<Vec<f64>>>,
    /// Outcome transform (BoTorch default: `Standardize(m, batch_shape)`,
    /// i.e. one standardization per batch element and output). With
    /// [`OutcomeTransformChoice::Chains`], one chain per output (or one for
    /// all), fitted per batch element and output.
    pub outcome_transform: OutcomeTransformChoice,
    /// Kernel family of the default covariance module.
    pub kernel: KernelKind,
    /// Wrap the default covariance module in a `ScaleKernel` (see
    /// [`SingleTaskGPOptions::use_scale_kernel`]).
    pub use_scale_kernel: bool,
    /// Covariance-module override, applied to every element with independent
    /// hyperparameters.
    pub covar_module: Option<KernelSpec>,
    /// A general covariance module (kernel tree) instead of `covar_module`,
    /// applied to every element with independent hyperparameters.
    pub covar_tree: Option<KernelNode>,
    /// Homoskedastic-likelihood override, applied to every element with
    /// independent noise levels.
    pub likelihood: Option<GaussianLikelihoodSpec>,
}

/// BoTorch's `SingleTaskGP` with `m ≥ 1` independent outputs and an optional
/// batch shape: one [`SingleTaskGP`] per batch element and output (see the
/// [module documentation](self)).
///
/// Unbatched models implement [`Model`] (for acquisition functions); batched
/// models are evaluated per batch element with [`BatchedMultiOutputGP::posterior_at`].
#[derive(Clone, Debug)]
pub struct BatchedMultiOutputGP {
    batch_shape: Vec<usize>,
    num_outputs: usize,
    dim: usize,
    train_y: Vec<Matrix>,
    train_yvar: Option<Vec<Matrix>>,
    input_transform: Option<BatchedInputTransform>,
    input_chain: Option<ChainedInputTransform>,
    /// Element `b·m + t` models output `t` of batch element `b`.
    models: Vec<SingleTaskGP>,
}

fn column(m: &Matrix, t: usize) -> Matrix {
    m.columns(t, 1).into_owned()
}

impl BatchedMultiOutputGP {
    /// Builds a model from training inputs (one `n × d` matrix per batch
    /// element, in row-major order of `batch_shape`) and targets (one `n × m`
    /// matrix per batch element). An empty `batch_shape` means an unbatched
    /// model (one matrix each).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes and the
    /// errors of [`SingleTaskGP::new`].
    pub fn new(
        train_x: Vec<Matrix>,
        train_y: Vec<Matrix>,
        batch_shape: Vec<usize>,
        options: BatchedMultiOutputGPOptions,
    ) -> Result<Self> {
        let n_batch: usize = batch_shape.iter().product();
        if n_batch == 0 {
            return Err(BotorchError::InputData(format!(
                "the batch shape {batch_shape:?} has no elements"
            )));
        }
        if train_x.len() != n_batch || train_y.len() != n_batch {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {n_batch} batches of training data for batch shape {batch_shape:?}, \
                 got {} inputs and {} targets",
                train_x.len(),
                train_y.len()
            )));
        }
        let (n, d) = train_x[0].shape();
        let m = train_y[0].ncols();
        if m == 0 {
            return Err(BotorchError::DimensionMismatch(
                "train_y must have at least one column (output)".into(),
            ));
        }
        for (x, y) in train_x.iter().zip(&train_y) {
            if x.shape() != (n, d) || y.shape() != (n, m) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "all batches must have the same shapes: expected {n}x{d} inputs and \
                     {n}x{m} targets, got {}x{} and {}x{}",
                    x.nrows(),
                    x.ncols(),
                    y.nrows(),
                    y.ncols()
                )));
            }
        }
        if let Some(yvar) = &options.train_yvar {
            if yvar.len() != n_batch || yvar.iter().any(|v| v.shape() != (n, m)) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "train_yvar must contain {n_batch} matrices of shape {n}x{m} (the shape of \
                     train_y)"
                )));
            }
        }
        // Input transforms of the elements of every batch (fitted where shared).
        let (per_batch_tf, reported_tf) = match options.input_transform {
            None => (vec![None; n_batch], None),
            Some(BatchedInputTransform::Shared(mut tf)) => {
                if tf.dim() != d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "input transform has dimension {} but train_x has {d} columns",
                        tf.dim()
                    )));
                }
                if tf.learn_bounds() {
                    let all = Matrix::from_fn(n * n_batch, d, |i, j| train_x[i / n][(i % n, j)]);
                    tf.fit(&all)?;
                }
                (
                    vec![Some(tf.frozen()); n_batch],
                    Some(BatchedInputTransform::Shared(tf)),
                )
            }
            Some(BatchedInputTransform::PerBatch(tfs)) => {
                if tfs.len() != n_batch {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "expected {n_batch} input transforms (one per batch element), got {}",
                        tfs.len()
                    )));
                }
                (tfs.into_iter().map(Some).collect(), None)
            }
        };
        // A general input transform, fitted to the inputs of all batches.
        let has_tf = reported_tf.is_some() || per_batch_tf.iter().any(Option::is_some);
        if (options.input_chain.is_some() || options.input_chains.is_some()) && has_tf
            || options.input_chain.is_some() && options.input_chains.is_some()
        {
            return Err(BotorchError::InvalidArgument(
                "input_transform and input_chain cannot both be given".into(),
            ));
        }
        let shared_params = "learned input-transform parameters (e.g. Warp's concentrations) \
                             shared by several outputs or batch elements are not supported: \
                             BoTorch's batched fitting optimizes every output and batch element on \
                             its own (use a ModelListGP of single-output models, or a Warp with \
                             batch_shape for batched single-output models)";
        let per_batch_params = options.input_chain_params;
        if let Some(p) = &per_batch_params {
            if p.len() != n_batch {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected the input-transform parameters of {n_batch} batch elements, got {}",
                    p.len()
                )));
            }
        }
        let input_chain = match options.input_chain {
            Some(mut chain) => {
                let batched_ok = per_batch_params.is_some() && m == 1;
                if chain.num_params() > 0 && n_batch * m > 1 && !batched_ok {
                    return Err(BotorchError::Unsupported(shared_params.into()));
                }
                let all = Matrix::from_fn(n * n_batch, d, |i, j| train_x[i / n][(i % n, j)]);
                chain.fit(&all)?;
                Some(chain)
            }
            None => None,
        };
        let element_chains: Vec<Option<ChainedInputTransform>> = match options.input_chains {
            Some(chains) => {
                if chains.len() != n_batch {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "expected {n_batch} input transforms (one per batch element), got {}",
                        chains.len()
                    )));
                }
                if m > 1 && chains.iter().any(|c| c.num_params() > 0) {
                    return Err(BotorchError::Unsupported(shared_params.into()));
                }
                chains.into_iter().map(Some).collect()
            }
            None => match (&input_chain, &per_batch_params) {
                (Some(chain), Some(params)) => params
                    .iter()
                    .map(|p| chain.frozen().with_params(p).map(Some))
                    .collect::<Result<Vec<_>>>()?,
                _ => vec![input_chain.as_ref().map(ChainedInputTransform::frozen); n_batch],
            },
        };
        let outcome_for = |t: usize| -> Result<OutcomeTransformChoice> {
            Ok(match &options.outcome_transform {
                OutcomeTransformChoice::Chains(chains) => {
                    let chain = match chains.len() {
                        1 => chains[0].clone(),
                        k if k == m => chains[t].clone(),
                        k => {
                            return Err(BotorchError::DimensionMismatch(format!(
                                "expected one outcome chain or one per output ({m}), got {k}"
                            )))
                        }
                    };
                    OutcomeTransformChoice::Chains(vec![chain])
                }
                other => other.clone(),
            })
        };
        let mut models = Vec::with_capacity(n_batch * m);
        for (b, tf) in per_batch_tf.into_iter().enumerate() {
            for t in 0..m {
                let opts = SingleTaskGPOptions {
                    train_yvar: options.train_yvar.as_ref().map(|v| column(&v[b], t)),
                    input_transform: tf.clone(),
                    input_chain: element_chains[b].clone(),
                    outcome_transform: outcome_for(t)?,
                    kernel: options.kernel,
                    use_scale_kernel: options.use_scale_kernel,
                    covar_module: options.covar_module,
                    covar_tree: options.covar_tree.clone(),
                    likelihood: options.likelihood,
                };
                models.push(SingleTaskGP::new(
                    train_x[b].clone(),
                    column(&train_y[b], t),
                    opts,
                )?);
            }
        }
        let input_transform = match reported_tf {
            Some(shared) => Some(shared),
            None if models[0].input_transform().is_some() => Some(BatchedInputTransform::PerBatch(
                (0..n_batch)
                    .map(|b| {
                        models[b * m]
                            .input_transform()
                            .expect("all elements have an input transform")
                            .clone()
                    })
                    .collect(),
            )),
            None => None,
        };
        Ok(Self {
            batch_shape,
            num_outputs: m,
            // The dimension of the points of a posterior (see
            // `SingleTaskGP::dim`).
            dim: models[0].dim(),
            train_y,
            train_yvar: options.train_yvar,
            input_transform,
            input_chain,
            models,
        })
    }

    /// A model from its elements (`models[b·m + t]` for output `t` of batch
    /// element `b`), e.g. conditioned copies of another model's elements;
    /// the training targets (and observed noise) are collected from the
    /// elements.
    pub(crate) fn from_elements(
        batch_shape: Vec<usize>,
        num_outputs: usize,
        input_transform: Option<BatchedInputTransform>,
        models: Vec<SingleTaskGP>,
    ) -> Self {
        let m = num_outputs;
        let nb = models.len() / m;
        let columns = |b: usize, f: &dyn Fn(&SingleTaskGP) -> Matrix| -> Matrix {
            let cols: Vec<Matrix> = (0..m).map(|t| f(&models[b * m + t])).collect();
            Matrix::from_fn(cols[0].nrows(), m, |i, t| cols[t][(i, 0)])
        };
        let train_y = (0..nb)
            .map(|b| columns(b, &|g| g.train_targets().clone()))
            .collect();
        let train_yvar = models[0].train_yvar().is_some().then(|| {
            (0..nb)
                .map(|b| columns(b, &|g| g.train_yvar().expect("fixed noise").clone()))
                .collect()
        });
        Self {
            batch_shape,
            num_outputs: m,
            dim: models[0].dim(),
            train_y,
            train_yvar,
            input_transform,
            input_chain: models[0].input_chain().cloned(),
            models,
        }
    }

    /// An unbatched model with `m = train_y.ncols()` outputs sharing the
    /// training inputs `train_x` (`n × d`); `options.train_yvar` must be
    /// `n × m` and a `Normalize` input transform is shared by the outputs.
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::new`].
    pub fn multi_output(
        train_x: Matrix,
        train_y: Matrix,
        options: SingleTaskGPOptions,
    ) -> Result<Self> {
        Self::new(
            vec![train_x],
            vec![train_y],
            Vec::new(),
            BatchedMultiOutputGPOptions {
                train_yvar: options.train_yvar.map(|v| vec![v]),
                input_transform: options.input_transform.map(BatchedInputTransform::Shared),
                input_chain: options.input_chain,
                input_chains: None,
                input_chain_params: None,
                outcome_transform: options.outcome_transform,
                kernel: options.kernel,
                use_scale_kernel: options.use_scale_kernel,
                covar_module: options.covar_module,
                covar_tree: options.covar_tree,
                likelihood: options.likelihood,
            },
        )
    }

    // ------------------------------------------------------------------
    // Structure
    // ------------------------------------------------------------------

    /// The batch shape of the training data (BoTorch's `model.batch_shape`).
    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    /// Number of batch elements (the product of the batch shape, 1 if unbatched).
    pub fn batch_size(&self) -> usize {
        self.models.len() / self.num_outputs
    }

    /// Number of outputs `m`.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Input dimension `d`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of training points `n` (per batch element).
    pub fn num_train(&self) -> usize {
        self.models[0].num_train()
    }

    /// BoTorch's `_aug_batch_shape`: the batch shape, followed by `m` if the
    /// model has more than one output.
    pub fn aug_batch_shape(&self) -> Vec<usize> {
        let mut s = self.batch_shape.clone();
        if self.num_outputs > 1 {
            s.push(self.num_outputs);
        }
        s
    }

    /// Number of independent GPs (`batch_size · m`).
    pub fn num_elements(&self) -> usize {
        self.models.len()
    }

    /// The index of the GP of output `output` of batch element `batch`.
    pub fn element(&self, batch: usize, output: usize) -> usize {
        batch * self.num_outputs + output
    }

    /// The independent GPs, element `b·m + t` for output `t` of batch element `b`.
    pub fn models(&self) -> &[SingleTaskGP] {
        &self.models
    }

    pub(crate) fn models_mut(&mut self) -> &mut [SingleTaskGP] {
        &mut self.models
    }

    /// The GP of output `output` of batch element `batch`.
    ///
    /// # Panics
    /// If an index is out of range.
    pub fn model(&self, batch: usize, output: usize) -> &SingleTaskGP {
        assert!(output < self.num_outputs, "output index out of range");
        &self.models[self.element(batch, output)]
    }

    /// Training inputs of batch element `batch` in the original space (`n × d`).
    pub fn train_inputs(&self, batch: usize) -> &Matrix {
        self.models[self.element(batch, 0)].train_inputs()
    }

    /// Training targets of batch element `batch` in the original space (`n × m`).
    pub fn train_targets(&self, batch: usize) -> &Matrix {
        &self.train_y[batch]
    }

    /// Observed noise variances of batch element `batch` (`n × m`), if given.
    pub fn train_yvar(&self, batch: usize) -> Option<&Matrix> {
        self.train_yvar.as_ref().map(|v| &v[batch])
    }

    /// Whether the noise is fixed (`train_yvar` given).
    pub fn is_fixed_noise(&self) -> bool {
        matches!(self.models[0].likelihood(), Likelihood::FixedNoise { .. })
    }

    /// The fitted input transform, if any.
    pub fn input_transform(&self) -> Option<&BatchedInputTransform> {
        self.input_transform.as_ref()
    }

    /// The fitted general input transform (`ChainedInputTransform` /
    /// `InputPerturbation`), if any (that of the first batch element for
    /// per-batch transforms).
    pub fn input_chain(&self) -> Option<&ChainedInputTransform> {
        self.input_chain
            .as_ref()
            .or_else(|| self.models[0].input_chain())
    }

    /// The general input transform of batch element `batch` (with its
    /// learned parameters), if any.
    pub fn input_chain_of(&self, batch: usize) -> Option<&ChainedInputTransform> {
        self.models[self.element(batch, 0)].input_chain()
    }

    /// Whether the posterior of each output is a `TransformedPosterior` (see
    /// [`Model::transformed_outputs`]).
    pub fn transformed_outputs(&self) -> Vec<bool> {
        self.models[..self.num_outputs]
            .iter()
            .map(SingleTaskGP::has_transformed_posterior)
            .collect()
    }

    /// The number of posterior points per input point (see
    /// [`Model::points_per_input`]).
    pub fn points_per_input(&self) -> usize {
        self.models[0].points_per_input()
    }

    /// The fitted outcome transform of an element (`None` without outcome
    /// transform).
    pub fn outcome_transform(&self, batch: usize, output: usize) -> Option<&Standardize> {
        self.model(batch, output).outcome_transform()
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    /// The raw-parameter layout of every element (identical for all).
    pub fn element_layout(&self) -> ParameterLayout {
        self.models[0].parameter_layout()
    }

    /// Sizes per element of the parameter groups in BoTorch's order.
    fn groups(&self) -> Vec<usize> {
        self.models[0]
            .parameter_names()
            .into_iter()
            .map(|(_, size)| size)
            .collect()
    }

    /// BoTorch parameter names and total sizes (all elements) in flat order.
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        let e = self.num_elements();
        self.models[0]
            .parameter_names()
            .into_iter()
            .map(|(name, size)| (name, size * e))
            .collect()
    }

    /// BoTorch parameter names and shapes, e.g.
    /// `("covar_module.raw_lengthscale", [m, 1, d])` for `m > 1` outputs.
    pub fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let aug = self.aug_batch_shape();
        self.models[0]
            .parameter_element_shapes()
            .into_iter()
            .map(|(name, element)| {
                let mut shape = aug.clone();
                shape.extend(element);
                (name, shape)
            })
            .collect()
    }

    /// Splits a flat (BoTorch-ordered) parameter vector into one raw-parameter
    /// vector per element.
    fn split(&self, flat: &[f64]) -> Result<Vec<Vec<f64>>> {
        let e = self.num_elements();
        let groups = self.groups();
        let per: usize = groups.iter().sum();
        if flat.len() != per * e {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                per * e,
                flat.len()
            )));
        }
        let mut out = vec![Vec::with_capacity(per); e];
        let mut offset = 0;
        for &k in &groups {
            for (i, v) in out.iter_mut().enumerate() {
                v.extend_from_slice(&flat[offset + i * k..offset + (i + 1) * k]);
            }
            offset += k * e;
        }
        Ok(out)
    }

    /// Flattens per-element vectors (in the element layout) into BoTorch's order.
    fn join(&self, per_element: &[Vector]) -> Vector {
        let e = self.num_elements();
        let groups = self.groups();
        let per: usize = groups.iter().sum();
        let mut flat = Vec::with_capacity(per * e);
        let mut start = 0;
        for &k in &groups {
            for v in per_element {
                flat.extend_from_slice(&v.as_slice()[start..start + k]);
            }
            start += k;
        }
        Vector::from_vec(flat)
    }

    /// Splits a vector in BoTorch's flat parameter order (e.g. a mask or
    /// bounds) into the per-element vectors.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong length.
    pub fn split_flat(&self, values: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.split(values)
    }

    /// The raw hyperparameters of all elements in BoTorch's flat order.
    pub fn raw_parameters(&self) -> Vector {
        let per: Vec<Vector> = self.models.iter().map(|m| m.raw_parameters()).collect();
        self.join(&per)
    }

    /// Sets all raw hyperparameters from a flat vector in BoTorch's order.
    /// On error the model is left unchanged.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong length and the errors
    /// of [`SingleTaskGP::set_raw_parameters`].
    pub fn set_raw_parameters(&mut self, values: &[f64]) -> Result<()> {
        let per = self.split(values)?;
        let mut models = self.models.clone();
        for (m, v) in models.iter_mut().zip(&per) {
            m.set_raw_parameters(v)?;
        }
        self.models = models;
        Ok(())
    }

    /// Optimizer bounds of the raw hyperparameters in flat order.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let e = self.num_elements();
        let bounds = self.models[0].raw_parameter_bounds();
        let mut out = Vec::with_capacity(bounds.len() * e);
        let mut start = 0;
        for k in self.groups() {
            for _ in 0..e {
                out.extend_from_slice(&bounds[start..start + k]);
            }
            start += k;
        }
        out
    }

    /// The raw hyperparameters of element `element`.
    pub fn element_hyperparameters(&self, element: usize) -> &Hyperparameters {
        self.models[element].hyperparameters()
    }

    /// Sets the raw hyperparameters of element `element`.
    ///
    /// # Errors
    /// See [`SingleTaskGP::set_hyperparameters`].
    pub fn set_element_hyperparameters(
        &mut self,
        element: usize,
        hyperparameters: Hyperparameters,
    ) -> Result<()> {
        self.models[element].set_hyperparameters(hyperparameters)
    }

    /// The exact marginal log likelihood of every element (BoTorch's
    /// `mll(model(*train_inputs), train_targets)`, a tensor of shape
    /// `_aug_batch_shape`), in row-major order.
    ///
    /// # Errors
    /// See [`ExactMarginalLogLikelihood::value`].
    pub fn mll_values(&self) -> Result<Vec<f64>> {
        self.models
            .iter()
            .map(|m| ExactMarginalLogLikelihood::new(m).value())
            .collect()
    }

    /// The marginal log likelihoods of all elements at the flat raw
    /// hyperparameters `raw` and the gradient of their *sum* with respect to
    /// `raw` (what autograd computes for `mll(...).sum()` in BoTorch).
    ///
    /// # Errors
    /// See [`ExactMarginalLogLikelihood::value_and_grad`].
    pub fn mll_values_and_grad(&self, raw: &[f64]) -> Result<(Vec<f64>, Vector)> {
        let per = self.split(raw)?;
        let mut values = Vec::with_capacity(per.len());
        let mut grads = Vec::with_capacity(per.len());
        for (m, p) in self.models.iter().zip(&per) {
            let (v, g) = ExactMarginalLogLikelihood::new(m).value_and_grad(p)?;
            values.push(v);
            grads.push(g);
        }
        Ok((values, self.join(&grads)))
    }

    // ------------------------------------------------------------------
    // Posteriors per batch element
    // ------------------------------------------------------------------

    fn check_batch(&self, batch: usize) -> Result<()> {
        if batch >= self.batch_size() {
            return Err(BotorchError::DimensionMismatch(format!(
                "batch index {batch} out of range for batch shape {:?}",
                self.batch_shape
            )));
        }
        Ok(())
    }

    fn outputs_of(&self, batch: usize) -> &[SingleTaskGP] {
        let m = self.num_outputs;
        &self.models[batch * m..(batch + 1) * m]
    }

    /// The joint posterior of batch element `batch` at the `q × d` inputs `x`
    /// over all `m` outputs (block diagonal, output-major), with the
    /// likelihood noise of every output if `observation_noise`.
    ///
    /// # Errors
    /// See [`SingleTaskGP::posterior`]; [`BotorchError::DimensionMismatch`]
    /// for a batch index out of range.
    pub fn posterior_at(
        &self,
        batch: usize,
        x: &Matrix,
        observation_noise: bool,
    ) -> Result<Posterior> {
        self.check_batch(batch)?;
        let outputs = self.outputs_of(batch);
        if outputs.len() == 1 {
            return outputs[0].posterior(x, observation_noise);
        }
        let parts = outputs
            .iter()
            .map(|g| g.posterior(x, observation_noise))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal(&parts))
    }

    /// Posteriors of every row of `x` on its own (`q = 1`) for batch element
    /// `batch` over all outputs: means and *unclamped* (diagonal) output
    /// covariances, including the observation noise of every output if
    /// `observation_noise`. With a one-to-many input transform, the
    /// marginals of the `n_w` transformed points of every row (`b·n_w` rows,
    /// point-major).
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::posterior_at`].
    pub fn point_posteriors_at(
        &self,
        batch: usize,
        x: &Matrix,
        observation_noise: bool,
    ) -> Result<PointPosteriors> {
        self.check_batch(batch)?;
        let outputs = self.outputs_of(batch);
        let b = x.nrows() * self.points_per_input();
        let m = outputs.len();
        let mut mean = Matrix::zeros(b, m);
        let mut var = Matrix::zeros(b, m);
        for (t, g) in outputs.iter().enumerate() {
            let (mu, v) = g.marginals_with_noise(x, observation_noise)?;
            for i in 0..b {
                mean[(i, t)] = mu[i];
                var[(i, t)] = v[i];
            }
        }
        Ok(PointPosteriors::independent(mean, &var))
    }

    /// The joint posterior of batch element `batch` (without observation
    /// noise) with forward-mode derivatives (see
    /// [`Model::joint_posterior_with_grad`]).
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::posterior_at`].
    pub fn joint_posterior_with_grad_at(
        &self,
        batch: usize,
        x: &Matrix,
    ) -> Result<JointPosteriorGrad> {
        self.check_batch(batch)?;
        let outputs = self.outputs_of(batch);
        if outputs.len() == 1 {
            return outputs[0].joint_posterior_with_grad(x);
        }
        let parts = outputs
            .iter()
            .map(|g| g.joint_posterior_with_grad(x))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal_grad(&parts))
    }

    /// The posterior of batch element `batch` at a single point over all
    /// outputs (unclamped) with derivatives (see
    /// [`Model::point_posterior_with_grad`]).
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::posterior_at`].
    pub fn point_posterior_with_grad_at(
        &self,
        batch: usize,
        x: &[f64],
    ) -> Result<JointPosteriorGrad> {
        self.check_batch(batch)?;
        let parts = self
            .outputs_of(batch)
            .iter()
            .map(|g| Model::point_posterior_with_grad(g, x))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal_grad(&parts))
    }

    /// Reverse-mode derivative of the joint posterior of batch element
    /// `batch` (see [`Model::joint_posterior_vjp`]).
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::posterior_at`].
    pub fn joint_posterior_vjp_at(
        &self,
        batch: usize,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        self.check_batch(batch)?;
        let k = self.points_per_input();
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_outputs,
            k,
        )?;
        let q = x.nrows() * k;
        let mut out = Matrix::zeros(rows.len(), x.ncols());
        for (t, g) in self.outputs_of(batch).iter().enumerate() {
            let r = t * q..(t + 1) * q;
            let ma = mean_adjoint.rows(r.start, q).into_owned();
            let ca = covariance_adjoint
                .view((r.start, r.start), (q, q))
                .into_owned();
            out += g.joint_posterior_vjp(x, rows.clone(), &ma, &ca)?;
        }
        Ok(out)
    }

    /// Errors unless every input maps to a single posterior point.
    fn require_one_point_per_input(&self) -> Result<()> {
        let k = self.points_per_input();
        if k != 1 {
            return Err(BotorchError::Unsupported(format!(
                "this evaluates every input point on its own, which requires one posterior \
                 point per input; the model's one-to-many input transform (InputPerturbation) \
                 gives {k}"
            )));
        }
        Ok(())
    }

    fn require_unbatched(&self) -> Result<()> {
        if !self.batch_shape.is_empty() {
            return Err(BotorchError::Unsupported(format!(
                "batched models (batch_shape = {:?}) cannot be used through the Model trait \
                 (e.g. by acquisition functions); evaluate them per batch element with \
                 posterior_at",
                self.batch_shape
            )));
        }
        Ok(())
    }
}

impl From<SingleTaskGP> for BatchedMultiOutputGP {
    /// An unbatched single-output model wrapping `gp`.
    fn from(gp: SingleTaskGP) -> Self {
        let input_transform = gp
            .input_transform()
            .map(|tf| BatchedInputTransform::PerBatch(vec![tf.clone()]));
        Self {
            batch_shape: Vec::new(),
            num_outputs: 1,
            dim: gp.dim(),
            train_y: vec![gp.train_targets().clone()],
            train_yvar: gp.train_yvar().map(|v| vec![v.clone()]),
            input_transform,
            input_chain: gp.input_chain().cloned(),
            models: vec![gp],
        }
    }
}

/// The block-diagonal joint posterior of independent outputs (each part over
/// the same points, possibly itself multi-output), output-major, with the
/// sample transforms of the parts.
pub(crate) fn block_diagonal(parts: &[Posterior]) -> Posterior {
    let n: usize = parts.iter().map(|p| p.len()).sum();
    let m: usize = parts.iter().map(|p| p.num_outputs()).sum();
    let mut mean = Vector::zeros(n);
    let mut cov = Matrix::zeros(n, n);
    let mut o = 0;
    for p in parts {
        let k = p.len();
        mean.rows_mut(o, k).copy_from(&p.mean);
        cov.view_mut((o, o), (k, k)).copy_from(&p.covariance);
        o += k;
    }
    let q = parts.first().map_or(0, Posterior::num_points);
    let transform = if parts.iter().any(Posterior::is_transformed) {
        let pieces: Vec<_> = parts
            .iter()
            .map(|p| (p.sample_transform.as_ref(), p.num_outputs()))
            .collect();
        SampleTransform::concat(&pieces, q)
    } else {
        None
    };
    Posterior::multi_output(mean, cov, m).with_sample_transform(transform)
}

/// The block-diagonal [`JointPosteriorGrad`] of independent outputs.
pub(crate) fn block_diagonal_grad(parts: &[JointPosteriorGrad]) -> JointPosteriorGrad {
    if parts.len() == 1 {
        return parts[0].clone();
    }
    let n: usize = parts.iter().map(|p| p.mean.len()).sum();
    let m: usize = parts.iter().map(|p| p.num_outputs).sum();
    let nj = parts[0].dmean.len();
    let mut mean = Vector::zeros(n);
    let mut cov = Matrix::zeros(n, n);
    let mut dmean = vec![Vector::zeros(n); nj];
    let mut dcov = vec![Matrix::zeros(n, n); nj];
    let mut o = 0;
    for p in parts {
        let k = p.mean.len();
        mean.rows_mut(o, k).copy_from(&p.mean);
        cov.view_mut((o, o), (k, k)).copy_from(&p.covariance);
        for j in 0..nj {
            dmean[j].rows_mut(o, k).copy_from(&p.dmean[j]);
            dcov[j].view_mut((o, o), (k, k)).copy_from(&p.dcov[j]);
        }
        o += k;
    }
    JointPosteriorGrad {
        mean,
        covariance: cov,
        dmean,
        dcov,
        num_outputs: m,
    }
}

impl Model for BatchedMultiOutputGP {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    fn output_gps(&self) -> Option<Vec<&SingleTaskGP>> {
        (self.batch_shape.is_empty() && self.models.iter().all(SingleTaskGP::has_plain_outcome))
            .then(|| self.models.iter().collect())
    }

    fn transformed_outputs(&self) -> Vec<bool> {
        BatchedMultiOutputGP::transformed_outputs(self)
    }

    fn points_per_input(&self) -> usize {
        BatchedMultiOutputGP::points_per_input(self)
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        self.require_unbatched()?;
        self.posterior_at(0, x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.require_unbatched()?;
        self.joint_posterior_with_grad_at(0, x)
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        self.require_unbatched()?;
        self.joint_posterior_vjp_at(0, x, rows, mean_adjoint, covariance_adjoint)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.require_unbatched()?;
        self.require_one_point_per_input()?;
        self.point_posteriors_at(0, x, false)
    }

    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        self.require_unbatched()?;
        self.point_posterior_with_grad_at(0, x)
    }

    fn posterior_marginals(&self, x: &Matrix) -> Result<(Vector, Vector)> {
        self.require_unbatched()?;
        self.require_one_point_per_input()?;
        if self.num_outputs == 1 {
            return SingleTaskGP::posterior_marginals(&self.models[0], x);
        }
        Err(BotorchError::Unsupported(format!(
            "this method requires a single-output model, got {} outputs",
            self.num_outputs
        )))
    }

    fn posterior_q1_with_grad(&self, x: &[f64]) -> Result<(f64, f64, Vector, Vector)> {
        self.require_unbatched()?;
        if self.num_outputs == 1 {
            return self.models[0].posterior_q1_with_grad(x);
        }
        Err(BotorchError::Unsupported(format!(
            "this method requires a single-output model, got {} outputs",
            self.num_outputs
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, d: usize, m: usize, shift: f64) -> (Matrix, Matrix) {
        let x = Matrix::from_fn(n, d, |i, j| {
            ((i * 7 + j * 3) % n) as f64 / n as f64 + 0.01 * j as f64
        });
        let y = Matrix::from_fn(n, m, |i, t| {
            let s: f64 = x.row(i).iter().sum();
            (s * (1.0 + t as f64)).sin() + shift * t as f64
        });
        (x, y)
    }

    #[test]
    fn elements_match_single_output_models() {
        let (x, y) = data(9, 2, 3, 0.5);
        let gp = BatchedMultiOutputGP::multi_output(
            x.clone(),
            y.clone(),
            SingleTaskGPOptions {
                input_transform: Some(Normalize::new(2)),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((gp.num_outputs(), gp.num_elements()), (3, 3));
        assert_eq!(gp.aug_batch_shape(), vec![3]);
        let xt = Matrix::from_row_slice(2, 2, &[0.2, 0.3, 0.7, 0.1]);
        let post = gp.posterior(&xt, true).unwrap();
        for t in 0..3 {
            let single = SingleTaskGP::new(
                x.clone(),
                y.columns(t, 1).into_owned(),
                SingleTaskGPOptions {
                    input_transform: Some(Normalize::new(2)),
                    ..Default::default()
                },
            )
            .unwrap();
            let p = single.posterior(&xt, true).unwrap();
            let pt = post.select_outputs(&[t]).unwrap();
            assert_eq!(pt.mean, p.mean);
            assert_eq!(pt.covariance, p.covariance);
        }
        // Off-diagonal output blocks are zero.
        assert_eq!(post.covariance[(0, 2)], 0.0);
    }

    #[test]
    fn flat_parameters_follow_botorch_order() {
        let (x, y) = data(6, 2, 2, 0.0);
        let mut gp =
            BatchedMultiOutputGP::multi_output(x, y, SingleTaskGPOptions::default()).unwrap();
        let shapes = gp.parameter_shapes();
        assert_eq!(
            shapes[0],
            ("likelihood.noise_covar.raw_noise".into(), vec![2, 1])
        );
        assert_eq!(shapes[1], ("mean_module.raw_constant".into(), vec![2]));
        assert_eq!(
            shapes[2],
            ("covar_module.raw_lengthscale".into(), vec![2, 1, 2])
        );
        let values = [0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        gp.set_raw_parameters(&values).unwrap();
        assert_eq!(gp.raw_parameters().as_slice(), &values);
        assert_eq!(
            gp.element_hyperparameters(1).to_vector().as_slice(),
            &[0.02, 0.2, 0.5, 0.6]
        );
        let (v, g) = gp.mll_values_and_grad(&values).unwrap();
        assert_eq!(v, gp.mll_values().unwrap());
        assert_eq!(g.len(), 8);
        assert!(gp.set_raw_parameters(&values[..7]).is_err());
        assert_eq!(gp.raw_parameter_bounds().len(), 8);
    }

    #[test]
    fn batched_models_and_vjp() {
        let (x0, y0) = data(7, 2, 2, 0.1);
        let (x1, y1) = data(7, 2, 2, -0.3);
        let x1 = x1.map(|v| v * 0.5 + 0.2);
        let gp = BatchedMultiOutputGP::new(
            vec![x0, x1],
            vec![y0, y1],
            vec![2],
            BatchedMultiOutputGPOptions {
                input_transform: Some(BatchedInputTransform::Shared(Normalize::new(2))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(gp.aug_batch_shape(), vec![2, 2]);
        let xt = Matrix::from_row_slice(2, 2, &[0.25, 0.5, 0.6, 0.9]);
        assert!(gp.posterior(&xt, false).is_err());
        let p1 = gp.posterior_at(1, &xt, false).unwrap();
        assert_eq!(p1.num_outputs(), 2);
        assert!(gp.posterior_at(2, &xt, false).is_err());
        // Shared Normalize: bounds learned from both batches.
        match gp.input_transform() {
            Some(BatchedInputTransform::Shared(tf)) => assert!(tf.learn_bounds()),
            other => panic!("unexpected transform {other:?}"),
        }
        // VJP equals the contraction of the forward-mode derivatives.
        let g = gp.joint_posterior_with_grad_at(1, &xt).unwrap();
        let n = 4;
        let ma = Vector::from_fn(n, |i, _| 0.3 + i as f64);
        let ca = Matrix::from_fn(n, n, |i, j| ((i * 3 + j) % 5) as f64 / 5.0 - 0.2);
        let vjp = gp.joint_posterior_vjp_at(1, &xt, 0..2, &ma, &ca).unwrap();
        for p in 0..2 {
            for k in 0..2 {
                let j = p * 2 + k;
                let expected = ma.dot(&g.dmean[j]) + ca.component_mul(&g.dcov[j]).sum();
                assert!((vjp[(p, k)] - expected).abs() < 1e-12 * expected.abs().max(1.0));
            }
        }
    }
}
