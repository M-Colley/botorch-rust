//! One-to-many and chained input transforms mirroring
//! `botorch.models.transforms.input.InputPerturbation` and
//! `ChainedInputTransform`.
//!
//! [`InputPerturbation`] replaces every input point by `n_w` perturbed copies
//! (additive or multiplicative perturbations of a subset of the inputs,
//! optionally clamped to bounds), which is how BoTorch models input noise for
//! risk-averse and robust multi-objective Bayesian optimization (Cakmak et
//! al., 2020; Daulton et al., 2022). A [`ChainedInputTransform`] applies
//! [`InputTransformStep`]s (`Normalize` and `InputPerturbation`) in order.
//!
//! # Layout
//!
//! For `q` points the perturbed set has `q·n_w` rows, point-major: row
//! `i·n_w + w` holds perturbation `w` of point `i`, exactly BoTorch's
//! `batch_shape x (q * n_w) x d` output (`output[..., i * n_w: (i + 1) * n_w,
//! :]` belongs to `X[..., i, :]`). Chained perturbations nest: after steps
//! with `n_1` and `n_2` perturbations, row `(i·n_1 + w_1)·n_2 + w_2`.
//!
//! # Train and eval mode
//!
//! A model applies the steps whose `transform_on_train` is set to its
//! training inputs (BoTorch's `preprocess_transform`; learnable `Normalize`
//! bounds are fitted on the way, as BoTorch does in train mode) and the steps
//! whose `transform_on_eval` is set to the points of a posterior. `Normalize`
//! always applies in both modes (the only configuration this crate
//! supports); `InputPerturbation` defaults to eval only. A perturbation that
//! also applies in train mode would give the exact GP `n·n_w` training inputs
//! for `n` targets, which BoTorch cannot evaluate either, so models reject
//! it. `transform_on_fantasize` is recorded only: BoTorch fantasizes in a
//! "fantasize mode" that skips perturbations with `transform_on_fantasize =
//! false`, which is not ported, so models with a one-to-many chain reject
//! `condition_on_observations` and `fantasize`.
//!
//! # Derivatives
//!
//! [`ChainedInputTransform::transform_with_tape`] records what the
//! reverse-mode derivative needs ([`InputTape::backward`]): the coefficients
//! of `Normalize`, and for perturbations the derivative of the clamp, which
//! follows torch's `minimum`/`maximum` (half the gradient to each argument
//! when they are equal) so that gradients at the bounds match BoTorch's
//! autograd. Heteroscedastic perturbations ([`PerturbationFunction`]) also
//! contribute the vector-Jacobian product of the perturbation function.

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::transforms::input_steps::{NewStep, StepRecord};
pub use crate::transforms::input_steps::{TransformFlags, TransformMode};
use crate::transforms::Normalize;
use crate::{Matrix, Vector};

/// A perturbation set that depends on the input (BoTorch's callable
/// `perturbation_set`, e.g. heteroscedastic input noise).
///
/// The function must act pointwise (the perturbations of a point depend only
/// on that point), as BoTorch's documentation of the callable states.
pub trait PerturbationFunction: Send + Sync + Debug {
    /// The perturbations of the points `x` (`n × k`, the perturbed columns):
    /// an `(n·n_w) × k` matrix whose row `i·n_w + w` is perturbation `w` of
    /// point `i` (BoTorch's callable returns `n x n_w x k`).
    ///
    /// # Errors
    /// Implementation-specific.
    fn perturbations(&self, x: &Matrix) -> Result<Matrix>;

    /// The vector-Jacobian product: for the adjoint of the perturbations of
    /// `x` (`(n·n_w) × k`, the layout of [`PerturbationFunction::perturbations`]),
    /// the adjoint of `x` (`n × k`).
    ///
    /// # Errors
    /// Implementation-specific.
    fn vjp(&self, x: &Matrix, adjoint: &Matrix) -> Result<Matrix>;
}

/// The perturbations of an [`InputPerturbation`].
#[derive(Clone, Debug)]
pub enum PerturbationSet {
    /// The same `n_w × k` perturbations for every point (BoTorch's tensor
    /// `perturbation_set`).
    Fixed(Matrix),
    /// Point-dependent perturbations (BoTorch's callable `perturbation_set`).
    Function(Arc<dyn PerturbationFunction>),
}

impl PartialEq for PerturbationSet {
    /// Fixed sets compare by value, functions by identity.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Fixed(a), Self::Fixed(b)) => a == b,
            (Self::Function(a), Self::Function(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// BoTorch's `InputPerturbation`: replaces every point by `n_w` perturbed
/// copies (see the [module documentation](self)).
#[derive(Clone, Debug, PartialEq)]
pub struct InputPerturbation {
    set: PerturbationSet,
    bounds: Option<Matrix>,
    indices: Option<Vec<usize>>,
    multiplicative: bool,
    transform_on_train: bool,
    transform_on_eval: bool,
    transform_on_fantasize: bool,
    /// The number of perturbations (known for a fixed set, determined by the
    /// first evaluation of a function).
    n_w: Option<usize>,
}

impl InputPerturbation {
    /// `InputPerturbation(perturbation_set)` with BoTorch's defaults: no
    /// bounds, all inputs, additive, applied in eval mode only.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty or non-finite fixed
    /// perturbation set.
    pub fn new(set: PerturbationSet) -> Result<Self> {
        let n_w = match &set {
            PerturbationSet::Fixed(p) => {
                if p.nrows() == 0 || p.ncols() == 0 {
                    return Err(BotorchError::InvalidArgument(
                        "`perturbation_set` must be an `n_p x d`-dim tensor with n_p >= 1!".into(),
                    ));
                }
                if p.iter().any(|v| !v.is_finite()) {
                    return Err(BotorchError::InvalidArgument(
                        "`perturbation_set` must be finite".into(),
                    ));
                }
                Some(p.nrows())
            }
            PerturbationSet::Function(_) => None,
        };
        Ok(Self {
            set,
            bounds: None,
            indices: None,
            multiplicative: false,
            transform_on_train: false,
            transform_on_eval: true,
            transform_on_fantasize: false,
            n_w,
        })
    }

    /// Clamps the perturbed inputs to `bounds` (`2 × k`: lower bounds in row
    /// 0, one column per perturbed input).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a wrong shape (BoTorch's message
    /// for a fixed set) or NaN bounds.
    pub fn with_bounds(mut self, bounds: Matrix) -> Result<Self> {
        if bounds.nrows() != 2 {
            return Err(BotorchError::InvalidArgument(format!(
                "`bounds` must be a `2 x d` tensor, got {}x{}",
                bounds.nrows(),
                bounds.ncols()
            )));
        }
        if let PerturbationSet::Fixed(p) = &self.set {
            if bounds.ncols() != p.ncols() {
                return Err(BotorchError::InvalidArgument(format!(
                    "`bounds` must have the same number of columns (last dimension) as the \
                     `perturbation_set`! Got {} and {}.",
                    bounds.ncols(),
                    p.ncols()
                )));
            }
        }
        if bounds.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::InvalidArgument(
                "`bounds` must not contain NaN".into(),
            ));
        }
        self.bounds = Some(bounds);
        Ok(self)
    }

    /// Perturbs only the inputs `indices` (BoTorch's `indices`); the other
    /// inputs are copied.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for empty or repeated indices.
    pub fn with_indices(mut self, indices: Vec<usize>) -> Result<Self> {
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if indices.is_empty() || sorted.len() != indices.len() {
            return Err(BotorchError::InvalidArgument(
                "`indices` must be a non-empty list of distinct indices".into(),
            ));
        }
        self.indices = Some(indices);
        Ok(self)
    }

    /// Multiplies the inputs by the perturbations instead of adding them
    /// (BoTorch's `multiplicative=True`).
    pub fn with_multiplicative(mut self, multiplicative: bool) -> Self {
        self.multiplicative = multiplicative;
        self
    }

    /// Sets `transform_on_train`, `transform_on_eval` and
    /// `transform_on_fantasize` (BoTorch's defaults: false, true, false).
    pub fn with_transform_flags(mut self, train: bool, eval: bool, fantasize: bool) -> Self {
        self.transform_on_train = train;
        self.transform_on_eval = eval;
        self.transform_on_fantasize = fantasize;
        self
    }

    /// The perturbation set.
    pub fn perturbation_set(&self) -> &PerturbationSet {
        &self.set
    }

    /// The clamping bounds (`2 × k`), if any.
    pub fn bounds(&self) -> Option<&Matrix> {
        self.bounds.as_ref()
    }

    /// The perturbed inputs (`None`: all).
    pub fn indices(&self) -> Option<&[usize]> {
        self.indices.as_deref()
    }

    /// Whether the perturbations are multiplicative.
    pub fn multiplicative(&self) -> bool {
        self.multiplicative
    }

    /// BoTorch's `transform_on_train`.
    pub fn transform_on_train(&self) -> bool {
        self.transform_on_train
    }

    /// BoTorch's `transform_on_eval`.
    pub fn transform_on_eval(&self) -> bool {
        self.transform_on_eval
    }

    /// BoTorch's `transform_on_fantasize`.
    pub fn transform_on_fantasize(&self) -> bool {
        self.transform_on_fantasize
    }

    /// The number of perturbations `n_w` (`None` for a function that has not
    /// been evaluated yet).
    pub fn n_w(&self) -> Option<usize> {
        self.n_w
    }

    /// The perturbed columns of a `d`-dimensional input.
    fn columns(&self, d: usize) -> Result<Vec<usize>> {
        let cols: Vec<usize> = match &self.indices {
            Some(idx) => idx.clone(),
            None => (0..d).collect(),
        };
        if let Some(&bad) = cols.iter().find(|&&c| c >= d) {
            return Err(BotorchError::DimensionMismatch(format!(
                "index {bad} is out of bounds for dimension with size {d}"
            )));
        }
        let k = cols.len();
        if let PerturbationSet::Fixed(p) = &self.set {
            if p.ncols() != k {
                return Err(BotorchError::DimensionMismatch(format!(
                    "The size of tensor a ({k}) must match the size of tensor b ({}) at \
                     non-singleton dimension 3",
                    p.ncols()
                )));
            }
        }
        if let Some(b) = &self.bounds {
            if b.ncols() != k {
                return Err(BotorchError::DimensionMismatch(format!(
                    "The size of tensor a ({k}) must match the size of tensor b ({}) at \
                     non-singleton dimension 3",
                    b.ncols()
                )));
            }
        }
        Ok(cols)
    }

    /// The perturbations of the points `xi` (`n × k`): `(n·n_w) × k`.
    fn perturbations_of(&self, xi: &Matrix) -> Result<(Matrix, usize)> {
        let n = xi.nrows();
        match &self.set {
            PerturbationSet::Fixed(p) => {
                let nw = p.nrows();
                Ok((
                    Matrix::from_fn(n * nw, p.ncols(), |r, c| p[(r % nw, c)]),
                    nw,
                ))
            }
            PerturbationSet::Function(f) => {
                let p = f.perturbations(xi)?;
                if n == 0 {
                    return Ok((p, self.n_w.unwrap_or(1)));
                }
                if p.ncols() != xi.ncols() || p.nrows() % n != 0 || p.nrows() == 0 {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "the perturbation function returned a {}x{} matrix for {n} points with \
                         {} perturbed inputs (expected n·n_w rows)",
                        p.nrows(),
                        p.ncols(),
                        xi.ncols()
                    )));
                }
                let nw = p.nrows() / n;
                if let Some(expected) = self.n_w {
                    if nw != expected {
                        return Err(BotorchError::DimensionMismatch(format!(
                            "the perturbation function returned {nw} perturbations per point, \
                             but {expected} before"
                        )));
                    }
                }
                Ok((p, nw))
            }
        }
    }

    /// Determines `n_w` of a perturbation function by evaluating it at `x`
    /// (one or more `d`-dimensional points); a no-op for fixed sets.
    ///
    /// # Errors
    /// Errors of the function and dimension mismatches.
    pub fn probe(&mut self, x: &Matrix) -> Result<()> {
        let cols = self.columns(x.ncols())?;
        if self.n_w.is_some() || x.nrows() == 0 {
            return Ok(());
        }
        let xi = Matrix::from_fn(x.nrows(), cols.len(), |i, c| x[(i, cols[c])]);
        let (_, nw) = self.perturbations_of(&xi)?;
        self.n_w = Some(nw);
        Ok(())
    }

    /// The perturbed points of `x` (`n × d`, eval mode): `(n·n_w) × d`, and
    /// what the derivative needs.
    fn apply(&self, x: &Matrix) -> Result<(Matrix, PerturbationRecord)> {
        let (n, d) = x.shape();
        let cols = self.columns(d)?;
        let k = cols.len();
        let xi = Matrix::from_fn(n, k, |i, c| x[(i, cols[c])]);
        let (p, nw) = self.perturbations_of(&xi)?;
        let fixed = matches!(self.set, PerturbationSet::Fixed(_));
        let rows = n * nw;
        let mut out = Matrix::from_fn(rows, d, |r, j| x[(r / nw, j)]);
        // Derivative of the clamp (torch's minimum/maximum) per entry.
        let mut dclamp = Matrix::from_element(rows, k, 1.0);
        for r in 0..rows {
            let i = r / nw;
            for c in 0..k {
                let xv = xi[(i, c)];
                let pv = p[(r, c)];
                let y = if self.multiplicative {
                    xv * pv
                } else {
                    xv + pv
                };
                let v = match &self.bounds {
                    Some(b) => {
                        let (lo, hi) = (b[(0, c)], b[(1, c)]);
                        // torch.maximum(torch.minimum(Y, hi), lo), NaN-propagating.
                        let (z, f1) = if y.is_nan() {
                            (f64::NAN, 1.0)
                        } else if y < hi {
                            (y, 1.0)
                        } else if y == hi {
                            (y, 0.5)
                        } else {
                            (hi, 0.0)
                        };
                        let (v, f2) = if z.is_nan() {
                            (f64::NAN, 1.0)
                        } else if z > lo {
                            (z, 1.0)
                        } else if z == lo {
                            (z, 0.5)
                        } else {
                            (lo, 0.0)
                        };
                        dclamp[(r, c)] = f1 * f2;
                        v
                    }
                    None => y,
                };
                out[(r, cols[c])] = v;
            }
        }
        Ok((
            out,
            PerturbationRecord {
                nw,
                cols,
                xi,
                p,
                dclamp,
                multiplicative: self.multiplicative,
                function: if fixed {
                    None
                } else {
                    match &self.set {
                        PerturbationSet::Function(f) => Some(Arc::clone(f)),
                        PerturbationSet::Fixed(_) => None,
                    }
                },
            },
        ))
    }
}

/// What the derivative of a perturbation step needs.
#[derive(Clone, Debug)]
struct PerturbationRecord {
    nw: usize,
    cols: Vec<usize>,
    /// The perturbed columns of the step's input (`n × k`).
    xi: Matrix,
    /// The perturbations (`(n·n_w) × k`).
    p: Matrix,
    /// The derivative of the clamp (`(n·n_w) × k`).
    dclamp: Matrix,
    multiplicative: bool,
    function: Option<Arc<dyn PerturbationFunction>>,
}

/// A step of a [`ChainedInputTransform`].
#[derive(Clone, Debug, PartialEq)]
pub enum InputTransformStep {
    /// `Normalize` of all inputs with BoTorch's defaults (applied in train
    /// and eval mode; learnable bounds are fitted to the training inputs as
    /// transformed by the preceding steps).
    Normalize(Normalize),
    /// `InputPerturbation`.
    Perturbation(InputPerturbation),
    /// Any other transform of `botorch.models.transforms.input` (see
    /// [`crate::transforms::input_steps`]).
    Step(NewStep),
}

impl InputTransformStep {
    /// Whether the size of the step's output is determined at the first
    /// evaluation (feature and perturbation functions, interaction features
    /// of all dimensions).
    fn sized_by_inputs(&self) -> bool {
        match self {
            InputTransformStep::Normalize(_) => false,
            InputTransformStep::Perturbation(p) => matches!(p.set, PerturbationSet::Function(_)),
            InputTransformStep::Step(NewStep::Append(a)) => a.sized_by_inputs(),
            InputTransformStep::Step(_) => false,
        }
    }

    /// The transform flags of the step.
    pub fn flags(&self) -> TransformFlags {
        match self {
            InputTransformStep::Normalize(_) => TransformFlags::ALL,
            InputTransformStep::Perturbation(p) => TransformFlags {
                train: p.transform_on_train,
                eval: p.transform_on_eval,
                fantasize: p.transform_on_fantasize,
            },
            InputTransformStep::Step(s) => s.flags(),
        }
    }

    /// BoTorch's class name of the step.
    pub fn name(&self) -> &'static str {
        match self {
            InputTransformStep::Normalize(_) => "Normalize",
            InputTransformStep::Perturbation(_) => "InputPerturbation",
            InputTransformStep::Step(s) => s.name(),
        }
    }

    /// The number of learned parameters of the step (fitted with the GP).
    pub fn num_params(&self) -> usize {
        match self {
            InputTransformStep::Step(s) => s.num_params(),
            _ => 0,
        }
    }

    /// Whether the step learns coefficients from the training inputs in
    /// train mode.
    fn learns_coefficients(&self) -> bool {
        match self {
            InputTransformStep::Normalize(n) => n.learn_bounds(),
            InputTransformStep::Perturbation(_) => false,
            InputTransformStep::Step(s) => s.learns_coefficients(),
        }
    }

    /// The number of points per input point if the step applies in `mode`
    /// (`None` for a callable whose size is not known yet).
    fn expand(&self, mode: TransformMode) -> Option<usize> {
        if !self.flags().applies(mode) {
            return Some(1);
        }
        match self {
            InputTransformStep::Normalize(_) => Some(1),
            InputTransformStep::Perturbation(p) => p.n_w,
            InputTransformStep::Step(s) => s.expand(),
        }
    }

    /// The output dimension for `d` input dimensions if the step applies.
    fn out_dim(&self, d: usize) -> Result<usize> {
        match self {
            InputTransformStep::Normalize(n) => {
                if n.dim() != d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "Wrong input dimension. Received {d}, expected {}.",
                        n.dim()
                    )));
                }
                Ok(d)
            }
            InputTransformStep::Perturbation(p) => {
                p.columns(d)?;
                Ok(d)
            }
            InputTransformStep::Step(s) => s.out_dim(d),
        }
    }
}

/// BoTorch's `ChainedInputTransform` (a single step represents the plain
/// transform): the steps are applied in order, each according to its flags
/// and the mode (see [`TransformMode`]).
#[derive(Clone, Debug, PartialEq)]
pub struct ChainedInputTransform {
    steps: Vec<InputTransformStep>,
    /// Whether each step was in train mode when the model was built
    /// (BoTorch transforms the training inputs at construction in the
    /// transforms' current modes; `true` by default).
    training: Vec<bool>,
    /// The parameter-name prefix of each step (e.g. `input_transform.` or
    /// `input_transform.warp.`).
    prefixes: Vec<String>,
    /// The dimension of the points of a posterior (determined by
    /// [`ChainedInputTransform::fit`]; `None`: that of the training inputs).
    eval_dim: Option<usize>,
}

/// The record of an evaluation of a [`ChainedInputTransform`] for its
/// derivatives (see [`ChainedInputTransform::transform_with_tape`]).
#[derive(Clone, Debug)]
pub struct InputTape {
    n: usize,
    d: usize,
    /// The output dimension.
    d_out: usize,
    steps: Vec<TapeStep>,
}

#[derive(Clone, Debug)]
enum TapeStep {
    Normalize(Vector),
    Perturbation(PerturbationRecord),
    /// A new step: its index in the chain, its parameter offset and record.
    Step(usize, usize, StepRecord),
}

impl TapeStep {
    fn expand(&self) -> usize {
        match self {
            TapeStep::Perturbation(r) => r.nw,
            TapeStep::Normalize(_) => 1,
            TapeStep::Step(_, _, r) => r.expand(),
        }
    }
}

impl ChainedInputTransform {
    /// A chain of `steps`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty chain.
    pub fn new(steps: Vec<InputTransformStep>) -> Result<Self> {
        if steps.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "a ChainedInputTransform needs at least one transform".into(),
            ));
        }
        let k = steps.len();
        let prefixes = if k == 1 {
            vec!["input_transform.".to_string()]
        } else {
            (0..k).map(|i| format!("input_transform.{i}.")).collect()
        };
        Ok(Self {
            steps,
            training: vec![true; k],
            prefixes,
            eval_dim: None,
        })
    }

    /// Sets whether each step is in train mode when the model is built
    /// (one flag per step).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of flags.
    pub fn with_training(mut self, training: Vec<bool>) -> Result<Self> {
        if training.len() != self.steps.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} training flags, got {}",
                self.steps.len(),
                training.len()
            )));
        }
        self.training = training;
        Ok(self)
    }

    /// Sets the parameter-name prefixes of the steps (one per step).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of prefixes.
    pub fn with_prefixes(mut self, prefixes: Vec<String>) -> Result<Self> {
        if prefixes.len() != self.steps.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} prefixes, got {}",
                self.steps.len(),
                prefixes.len()
            )));
        }
        self.prefixes = prefixes;
        Ok(self)
    }

    /// The steps.
    pub fn steps(&self) -> &[InputTransformStep] {
        &self.steps
    }

    /// The modes of the steps at construction (see
    /// [`ChainedInputTransform::with_training`]).
    pub fn training(&self) -> &[bool] {
        &self.training
    }

    /// Whether every step is a plain `Normalize` or an `InputPerturbation`
    /// (the transforms that existed before the other transforms were
    /// ported; all model features support them).
    pub fn is_basic(&self) -> bool {
        self.steps
            .iter()
            .all(|s| !matches!(s, InputTransformStep::Step(_)))
    }

    /// Whether some step maps a point to several points in eval mode
    /// (BoTorch's `is_one_to_many`, restricted to the steps that apply).
    pub fn is_one_to_many(&self) -> bool {
        self.points_per_input_mode(TransformMode::Eval) > 1
    }

    /// The number of points per input point in eval mode: the product of
    /// `n_w` (and `n_f`) over the one-to-many steps that apply (1 if none
    /// does).
    ///
    /// # Panics
    /// If a perturbation or feature function has not been probed (see
    /// [`ChainedInputTransform::fit`]).
    pub fn points_per_input(&self) -> usize {
        self.points_per_input_mode(TransformMode::Eval)
    }

    /// [`ChainedInputTransform::points_per_input`] in `mode`.
    ///
    /// # Panics
    /// See [`ChainedInputTransform::points_per_input`].
    pub fn points_per_input_mode(&self, mode: TransformMode) -> usize {
        self.steps
            .iter()
            .map(|s| {
                s.expand(mode).expect(
                    "perturbation and feature functions are probed when the transform is fitted",
                )
            })
            .product()
    }

    /// A copy whose learned coefficients are never refitted (to share
    /// bounds learned from all batch elements of a batched model).
    pub fn frozen(&self) -> Self {
        let mut out = self.clone();
        for s in &mut out.steps {
            match s {
                InputTransformStep::Normalize(n) => *n = n.frozen(),
                InputTransformStep::Step(NewStep::Affine(a)) => {
                    *a = a.clone().frozen();
                }
                InputTransformStep::Step(NewStep::Warp(w)) => {
                    *w = w.clone().frozen();
                }
                _ => {}
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // Learned parameters
    // ------------------------------------------------------------------

    /// The number of learned parameters (e.g. `Warp`'s concentrations).
    pub fn num_params(&self) -> usize {
        self.steps.iter().map(InputTransformStep::num_params).sum()
    }

    /// The learned parameters in BoTorch's `named_parameters()` order.
    pub fn params(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.num_params());
        for s in &self.steps {
            if let InputTransformStep::Step(n) = s {
                out.extend(n.params());
            }
        }
        out
    }

    /// Sets the learned parameters.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of values.
    pub fn set_params(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.num_params() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} input transform parameters, got {}",
                self.num_params(),
                values.len()
            )));
        }
        let mut o = 0;
        for s in &mut self.steps {
            if let InputTransformStep::Step(n) = s {
                let k = n.num_params();
                n.set_params(&values[o..o + k])?;
                o += k;
            }
        }
        Ok(())
    }

    /// A copy with the learned parameters `values`.
    ///
    /// # Errors
    /// See [`ChainedInputTransform::set_params`].
    pub fn with_params(&self, values: &[f64]) -> Result<Self> {
        let mut out = self.clone();
        out.set_params(values)?;
        Ok(out)
    }

    /// The names (with the steps' prefixes) and shapes of the learned
    /// parameters, in order.
    pub fn param_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        for (s, prefix) in self.steps.iter().zip(&self.prefixes) {
            if let InputTransformStep::Step(n) = s {
                for (name, shape) in n.param_shapes() {
                    out.push((format!("{prefix}{name}"), shape));
                }
            }
        }
        out
    }

    /// The optimizer bounds of the learned parameters.
    pub fn param_bounds(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for s in &self.steps {
            if let InputTransformStep::Step(n) = s {
                out.extend(n.param_bounds());
            }
        }
        out
    }

    /// The priors of the learned parameters (one per parameter).
    pub fn param_priors(&self) -> Vec<Option<crate::priors::Prior>> {
        let mut out = Vec::new();
        for s in &self.steps {
            if let InputTransformStep::Step(n) = s {
                out.extend(n.param_priors());
            }
        }
        out
    }

    /// The log prior density of the learned parameters and its gradient.
    pub fn log_prior_and_grad(&self) -> (f64, Vec<f64>) {
        let mut lp = 0.0;
        let mut g = Vec::with_capacity(self.num_params());
        for s in &self.steps {
            if let InputTransformStep::Step(n) = s {
                let (v, gs) = n.log_prior_and_grad();
                lp += v;
                g.extend(gs);
            }
        }
        (lp, g)
    }

    // ------------------------------------------------------------------
    // Fitting and application
    // ------------------------------------------------------------------

    /// Fits the chain to training inputs (what BoTorch does when the model
    /// is built): every step in train mode (see
    /// [`ChainedInputTransform::with_training`]) that applies to the
    /// training inputs learns its coefficients from the inputs as
    /// transformed by the preceding steps; steps in eval mode transform them
    /// without learning. The sizes of perturbation and feature functions are
    /// determined at the first training point (as transformed by the
    /// preceding eval-mode steps). Returns the training inputs as
    /// transformed at construction (BoTorch's `transformed_X`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a one-to-many step that applies in
    /// train mode (an exact GP cannot have more inputs than targets), for
    /// steps whose output dimension differs between the modes and for
    /// learned coefficients that would depend on learned parameters;
    /// [`BotorchError::DimensionMismatch`] for inconsistent dimensions and
    /// errors of perturbation and feature functions.
    pub fn fit(&mut self, train_x: &Matrix) -> Result<Matrix> {
        let mut x = train_x.clone();
        // A point as the eval-mode steps see it (`None` once a step cannot
        // transform the training point: the points of a posterior then have
        // other dimensions than the training inputs, e.g. with
        // `AppendFeatures` not applied in training).
        let mut probe = Some(train_x.rows(0, train_x.nrows().min(1)).into_owned());
        let mut seen_params = false;
        for (step, &training) in self.steps.iter_mut().zip(&self.training) {
            let flags = step.flags();
            if let InputTransformStep::Perturbation(p) = step {
                if p.transform_on_train {
                    return Err(BotorchError::Unsupported(
                        "InputPerturbation with transform_on_train=True is not supported: the \
                         perturbed training inputs (n * n_w points) would not match the n \
                         training targets of the model."
                            .into(),
                    ));
                }
            }
            if let InputTransformStep::Step(NewStep::Warp(w)) = step {
                if w.reverse() {
                    return Err(BotorchError::Unsupported(
                        "Warp with reverse=True is not supported in models.".into(),
                    ));
                }
            }
            if let InputTransformStep::Step(s) = step {
                if let Some(pr) = &probe {
                    if s.probe(pr).is_err() {
                        probe = None;
                    }
                }
                if flags.train && s.expand() != Some(1) {
                    return Err(BotorchError::Unsupported(format!(
                        "{} with transform_on_train=True and {} features per point is not \
                         supported: the training inputs would not match the training targets of \
                         the model.",
                        s.name(),
                        s.expand().unwrap_or(0)
                    )));
                }
            }
            if step.learns_coefficients() && flags.train && seen_params {
                return Err(BotorchError::Unsupported(format!(
                    "a {} that learns its coefficients after a transform with learned parameters \
                     (e.g. Warp) is not supported: its coefficients would depend on the learned \
                     parameters.",
                    step.name()
                )));
            }
            seen_params |= step.num_params() > 0 && flags.train;
            // The construction-time transform of the training inputs.
            let applies = if training { flags.train } else { flags.eval };
            if applies {
                match step {
                    InputTransformStep::Normalize(n) => {
                        if n.dim() != x.ncols() {
                            return Err(BotorchError::DimensionMismatch(format!(
                                "Wrong input dimension. Received {}, expected {}.",
                                x.ncols(),
                                n.dim()
                            )));
                        }
                        if training {
                            n.fit(&x)?;
                        }
                        x = n.transform(&x)?;
                    }
                    InputTransformStep::Perturbation(p) => {
                        p.probe(probe.as_ref().unwrap_or(&x))?;
                        x = p.apply(&x)?.0;
                    }
                    InputTransformStep::Step(s) => {
                        if training && flags.train {
                            s.fit(&x)?;
                        }
                        x = s.apply(&x)?.0;
                    }
                }
            } else if let (InputTransformStep::Perturbation(p), Some(pr)) = (&mut *step, &probe) {
                if p.probe(pr).is_err() {
                    probe = None;
                }
            }
            // The probe point follows the eval-mode steps.
            if flags.eval {
                probe = probe.and_then(|pr| Self::probe_step(step, &pr).ok());
            }
        }
        let d = train_x.ncols();
        let eval_dim = self.find_eval_dim(d)?;
        if eval_dim != d && self.steps.iter().any(InputTransformStep::sized_by_inputs) {
            return Err(BotorchError::Unsupported(
                "feature or perturbation functions (and InteractionFeatures of all \
                 dimensions) in an input transform whose posterior points have other \
                 dimensions than the training inputs are not supported"
                    .into(),
            ));
        }
        self.eval_dim = (eval_dim != d).then_some(eval_dim);
        Ok(x)
    }

    /// The first transformed point of `probe` under `step` in eval mode.
    fn probe_step(step: &mut InputTransformStep, probe: &Matrix) -> Result<Matrix> {
        let out = match step {
            InputTransformStep::Normalize(n) => n.transform(probe)?,
            InputTransformStep::Perturbation(p) => {
                p.probe(probe)?;
                p.apply(probe)?.0
            }
            InputTransformStep::Step(s) => s.apply(probe)?.0,
        };
        Ok(out.rows(0, out.nrows().min(1)).into_owned())
    }

    /// The dimension of the points of a posterior: that of the training
    /// inputs (`d`) if the chain maps them in eval mode to the dimension of
    /// the transformed training inputs (which the kernel sees), otherwise the
    /// smallest dimension it maps there (BoTorch's `AppendFeatures` with
    /// training inputs that include the features).
    fn find_eval_dim(&self, d: usize) -> Result<usize> {
        let dt = self.out_dim_mode(d, TransformMode::Train)?;
        if self.out_dim_mode(d, TransformMode::Eval).ok() == Some(dt) {
            return Ok(d);
        }
        (1..=d + dt)
            .find(|&de| self.out_dim_mode(de, TransformMode::Eval).ok() == Some(dt))
            .ok_or_else(|| {
                BotorchError::Unsupported(format!(
                    "the input transform maps the training inputs to {dt} dimensions but no \
                     points of a posterior (eval mode) to {dt}; a model needs the same number \
                     of dimensions"
                ))
            })
    }

    /// The dimension of the points of a posterior for training inputs of
    /// `d` dimensions (see [`ChainedInputTransform::fit`]).
    pub fn input_dim(&self, d: usize) -> usize {
        self.eval_dim.unwrap_or(d)
    }

    /// Checks that the chain maps `d`-dimensional points in `mode` to the
    /// `d_tf` dimensions of the transformed training inputs.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] otherwise.
    pub fn check_mode_dim(&self, d: usize, d_tf: usize, mode: TransformMode) -> Result<()> {
        let dm = self.out_dim_mode(d, mode)?;
        if dm != d_tf {
            return Err(BotorchError::DimensionMismatch(format!(
                "the input transform maps the points ({mode:?} mode) to {dm} dimensions, but \
                 the transformed training inputs have {d_tf}"
            )));
        }
        Ok(())
    }

    /// The output dimension of the chain in `mode` for `d` input dimensions.
    ///
    /// # Errors
    /// Dimension errors of the steps.
    pub fn out_dim_mode(&self, d: usize, mode: TransformMode) -> Result<usize> {
        let mut dim = d;
        for s in &self.steps {
            if s.flags().applies(mode) {
                dim = s.out_dim(dim)?;
            }
        }
        Ok(dim)
    }

    /// The Jacobian of the chain in `mode` for `d`-dimensional inputs if it
    /// is constant and diagonal (only `Normalize` and other affine steps
    /// apply): the factor of every dimension; `None` otherwise.
    pub fn constant_diag_jacobian(&self, mode: TransformMode, d: usize) -> Option<Vec<f64>> {
        let mut jac = vec![1.0; d];
        for s in &self.steps {
            if !s.flags().applies(mode) {
                continue;
            }
            let f: Vec<f64> = match s {
                InputTransformStep::Normalize(n) => {
                    n.coefficient().iter().map(|c| 1.0 / c).collect()
                }
                InputTransformStep::Step(NewStep::Affine(a)) => a.diag_jacobian(),
                _ => return None,
            };
            if f.len() != d {
                return None;
            }
            for (j, v) in jac.iter_mut().zip(&f) {
                *j *= v;
            }
        }
        Some(jac)
    }

    /// The output dimension of the chain in eval mode.
    ///
    /// # Errors
    /// Dimension errors of the steps.
    pub fn out_dim(&self, d: usize) -> Result<usize> {
        self.out_dim_mode(d, TransformMode::Eval)
    }

    /// The training inputs as the model sees them (BoTorch's
    /// `preprocess_transform` and the train-mode transform: the steps that
    /// apply in train mode, with the current coefficients and parameters).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent dimensions.
    pub fn preprocess(&self, train_x: &Matrix) -> Result<Matrix> {
        Ok(self
            .transform_with_tape_mode(train_x, TransformMode::Train)?
            .0)
    }

    /// The transformed points of `x` (`n × d`) in eval mode: `(n·N) × d'`
    /// for `N` = [`ChainedInputTransform::points_per_input`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent dimensions and
    /// errors of perturbation and feature functions.
    pub fn transform(&self, x: &Matrix) -> Result<Matrix> {
        Ok(self.transform_with_tape(x)?.0)
    }

    /// The transformed points of `x` in `mode`.
    ///
    /// # Errors
    /// See [`ChainedInputTransform::transform`].
    pub fn transform_mode(&self, x: &Matrix, mode: TransformMode) -> Result<Matrix> {
        Ok(self.transform_with_tape_mode(x, mode)?.0)
    }

    /// [`ChainedInputTransform::transform`] with the record of its
    /// derivatives.
    ///
    /// # Errors
    /// See [`ChainedInputTransform::transform`].
    pub fn transform_with_tape(&self, x: &Matrix) -> Result<(Matrix, InputTape)> {
        self.transform_with_tape_mode(x, TransformMode::Eval)
    }

    /// [`ChainedInputTransform::transform_mode`] with the record of its
    /// derivatives.
    ///
    /// # Errors
    /// See [`ChainedInputTransform::transform`].
    pub fn transform_with_tape_mode(
        &self,
        x: &Matrix,
        mode: TransformMode,
    ) -> Result<(Matrix, InputTape)> {
        let mut cur = x.clone();
        let mut steps = Vec::with_capacity(self.steps.len());
        let mut poff = 0;
        for (si, step) in self.steps.iter().enumerate() {
            let np = step.num_params();
            if step.flags().applies(mode) {
                match step {
                    InputTransformStep::Normalize(n) => {
                        cur = n.transform(&cur)?;
                        steps.push(TapeStep::Normalize(n.coefficient().clone()));
                    }
                    InputTransformStep::Perturbation(p) => {
                        let (out, rec) = p.apply(&cur)?;
                        cur = out;
                        steps.push(TapeStep::Perturbation(rec));
                    }
                    InputTransformStep::Step(s) => {
                        let (out, rec) = s.apply(&cur)?;
                        cur = out;
                        steps.push(TapeStep::Step(si, poff, rec));
                    }
                }
            }
            poff += np;
        }
        let d_out = cur.ncols();
        Ok((
            cur,
            InputTape {
                n: x.nrows(),
                d: x.ncols(),
                d_out,
                steps,
            },
        ))
    }

    /// Inverts the steps in reverse order (BoTorch's `untransform`; every
    /// step is inverted, as in BoTorch).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] (`NotImplementedError`) for steps
    /// without an inverse.
    pub fn untransform(&self, x: &Matrix) -> Result<Matrix> {
        let mut cur = x.clone();
        for step in self.steps.iter().rev() {
            cur = match step {
                InputTransformStep::Normalize(n) => n.untransform(&cur)?,
                InputTransformStep::Perturbation(_) => {
                    return Err(BotorchError::Unsupported(
                        "InputPerturbation does not implement the `untransform` method.".into(),
                    ))
                }
                InputTransformStep::Step(s) => s.untransform(&cur)?,
            };
        }
        Ok(cur)
    }

    /// The reverse pass of a tape, with the adjoint of the learned
    /// parameters (see [`InputTape::backward_with_params`]).
    ///
    /// # Errors
    /// See [`InputTape::backward`].
    pub fn backward_with_params(
        &self,
        tape: &InputTape,
        rows: Range<usize>,
        adjoint: &Matrix,
    ) -> Result<(Matrix, Vec<f64>)> {
        tape.backward_impl(self, rows, adjoint, true)
    }
}

impl InputTape {
    /// The number of raw points.
    pub fn num_inputs(&self) -> usize {
        self.n
    }

    /// The raw input dimension.
    pub fn input_dim(&self) -> usize {
        self.d
    }

    /// The transformed dimension.
    pub fn output_dim(&self) -> usize {
        self.d_out
    }

    /// The number of transformed points per raw point.
    pub fn points_per_input(&self) -> usize {
        self.steps.iter().map(TapeStep::expand).product()
    }

    /// Reverse-mode derivative: for the adjoint of the transformed points of
    /// the raw points `rows` (`rows.len()·N × d'`, the rows `rows.start·N ..
    /// rows.end·N` of the transformed points), the adjoint of those raw
    /// points (`rows.len() × d`).
    ///
    /// The derivatives of the new steps need the chain that recorded the
    /// tape; use [`InputTape::backward_through`] for chains with them.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes and errors
    /// of perturbation functions; [`BotorchError::Unsupported`] for tapes of
    /// new steps.
    pub fn backward(&self, rows: Range<usize>, adjoint: &Matrix) -> Result<Matrix> {
        if self.steps.iter().any(|s| matches!(s, TapeStep::Step(..))) {
            return Err(BotorchError::Unsupported(
                "the tape records transforms whose reverse pass needs the chain; use \
                 backward_through"
                    .into(),
            ));
        }
        let empty = ChainedInputTransform {
            steps: Vec::new(),
            training: Vec::new(),
            prefixes: Vec::new(),
            eval_dim: None,
        };
        Ok(self.backward_impl(&empty, rows, adjoint, false)?.0)
    }

    /// [`InputTape::backward`] through the steps of `chain` (the chain that
    /// recorded the tape).
    ///
    /// # Errors
    /// See [`InputTape::backward`].
    pub fn backward_through(
        &self,
        chain: &ChainedInputTransform,
        rows: Range<usize>,
        adjoint: &Matrix,
    ) -> Result<Matrix> {
        Ok(self.backward_impl(chain, rows, adjoint, false)?.0)
    }

    /// [`InputTape::backward_through`] with the adjoint of the chain's
    /// learned parameters.
    ///
    /// # Errors
    /// See [`InputTape::backward`].
    pub fn backward_with_params(
        &self,
        chain: &ChainedInputTransform,
        rows: Range<usize>,
        adjoint: &Matrix,
    ) -> Result<(Matrix, Vec<f64>)> {
        self.backward_impl(chain, rows, adjoint, true)
    }

    fn backward_impl(
        &self,
        chain: &ChainedInputTransform,
        rows: Range<usize>,
        adjoint: &Matrix,
        with_params: bool,
    ) -> Result<(Matrix, Vec<f64>)> {
        let total = self.points_per_input();
        if rows.end > self.n || adjoint.shape() != (rows.len() * total, self.d_out) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the adjoint of the transformed points of rows {rows:?} must be {}x{}, got {}x{}",
                rows.len() * total,
                self.d_out,
                adjoint.nrows(),
                adjoint.ncols()
            )));
        }
        // The number of points per raw point after each step.
        let mut per = Vec::with_capacity(self.steps.len() + 1);
        per.push(1usize);
        for s in &self.steps {
            let last = *per.last().expect("non-empty");
            per.push(last * s.expand());
        }
        let mut pbar = vec![0.0; if with_params { chain.num_params() } else { 0 }];
        let mut bar = adjoint.clone();
        for (s, step) in self.steps.iter().enumerate().rev() {
            match step {
                TapeStep::Normalize(coef) => {
                    for j in 0..bar.ncols() {
                        let c = coef[j];
                        for r in 0..bar.nrows() {
                            bar[(r, j)] /= c;
                        }
                    }
                }
                TapeStep::Perturbation(rec) => {
                    let start = rows.start * per[s];
                    let n_in = rows.len() * per[s];
                    bar = rec.backward(start, n_in, &bar)?;
                }
                TapeStep::Step(si, poff, rec) => {
                    let InputTransformStep::Step(ns) = &chain.steps[*si] else {
                        return Err(BotorchError::InvalidArgument(
                            "the tape was not recorded by this chain".into(),
                        ));
                    };
                    let start = rows.start * per[s];
                    let n_in = rows.len() * per[s];
                    let np = ns.num_params();
                    let slot = if with_params && np > 0 {
                        Some(&mut pbar[*poff..*poff + np])
                    } else {
                        None
                    };
                    bar = ns.backward(rec, start, n_in, &bar, slot)?;
                }
            }
        }
        Ok((bar, pbar))
    }

    /// The Jacobian of every transformed point with respect to its raw point
    /// (`d' × d`, entry `[l, j] = ∂x̃_l / ∂x_j`), in the order of the
    /// transformed points.
    ///
    /// # Errors
    /// Errors of perturbation and feature functions.
    pub fn jacobians(&self) -> Result<Vec<Matrix>> {
        self.jacobians_through(&ChainedInputTransform {
            steps: Vec::new(),
            training: Vec::new(),
            prefixes: Vec::new(),
            eval_dim: None,
        })
    }

    /// [`InputTape::jacobians`] through the steps of `chain`.
    ///
    /// # Errors
    /// See [`InputTape::jacobians`].
    pub fn jacobians_through(&self, chain: &ChainedInputTransform) -> Result<Vec<Matrix>> {
        let total = self.points_per_input();
        let (n, d, dout) = (self.n, self.d, self.d_out);
        let mut out = vec![Matrix::zeros(dout, d); n * total];
        for slot in 0..total {
            for l in 0..dout {
                let mut e = Matrix::zeros(n * total, dout);
                for i in 0..n {
                    e[(i * total + slot, l)] = 1.0;
                }
                let g = self.backward_impl(chain, 0..n, &e, false)?.0;
                for i in 0..n {
                    let jac = &mut out[i * total + slot];
                    for j in 0..d {
                        jac[(l, j)] = g[(i, j)];
                    }
                }
            }
        }
        Ok(out)
    }
}

impl PerturbationRecord {
    /// The adjoint of the step's input rows `start .. start + n_in` from the
    /// adjoint of their perturbed rows (`n_in·n_w × d`).
    fn backward(&self, start: usize, n_in: usize, bar: &Matrix) -> Result<Matrix> {
        let nw = self.nw;
        let d = bar.ncols();
        let k = self.cols.len();
        let mut out = Matrix::zeros(n_in, d);
        // Adjoint of the perturbations (function sets only).
        let mut p_bar = self.function.as_ref().map(|_| Matrix::zeros(n_in * nw, k));
        for il in 0..n_in {
            let i = start + il;
            for w in 0..nw {
                let rl = il * nw + w;
                let r = i * nw + w;
                for j in 0..d {
                    out[(il, j)] += bar[(rl, j)];
                }
                for (c, &j) in self.cols.iter().enumerate() {
                    // Undo the plain copy of the perturbed column.
                    out[(il, j)] -= bar[(rl, j)];
                    let g = bar[(rl, j)] * self.dclamp[(r, c)];
                    let dy_dx = if self.multiplicative {
                        self.p[(r, c)]
                    } else {
                        1.0
                    };
                    out[(il, j)] += g * dy_dx;
                    if let Some(pb) = p_bar.as_mut() {
                        pb[(rl, c)] = if self.multiplicative {
                            g * self.xi[(i, c)]
                        } else {
                            g
                        };
                    }
                }
            }
        }
        if let (Some(f), Some(pb)) = (&self.function, p_bar) {
            let xi = self.xi.rows(start, n_in).into_owned();
            let xbar = f.vjp(&xi, &pb)?;
            if xbar.shape() != (n_in, k) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the vector-Jacobian product of the perturbation function has shape {}x{}, \
                     expected {n_in}x{k}",
                    xbar.nrows(),
                    xbar.ncols()
                )));
            }
            for il in 0..n_in {
                for (c, &j) in self.cols.iter().enumerate() {
                    out[(il, j)] += xbar[(il, c)];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn botorch_example() -> (Matrix, Matrix) {
        // BoTorch's test_input_perturbation: X (2 batches) and the set.
        let p = Matrix::from_row_slice(3, 2, &[0.5, -0.3, 0.2, 0.4, -0.7, 0.1]);
        let x = Matrix::from_row_slice(2, 2, &[0.5, 0.5, 0.9, 0.7]);
        (p, x)
    }

    #[test]
    fn perturbs_point_major_like_botorch() {
        let (p, x) = botorch_example();
        let tf = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
            InputPerturbation::new(PerturbationSet::Fixed(p.clone())).unwrap(),
        )])
        .unwrap();
        let out = tf.transform(&x).unwrap();
        let expected = Matrix::from_row_slice(
            6,
            2,
            &[1.0, 0.2, 0.7, 0.9, -0.2, 0.6, 1.4, 0.4, 1.1, 1.1, 0.2, 0.8],
        );
        assert!((out - expected).amax() < 1e-15);
        // With bounds.
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.2, 1.2, 0.9]);
        let tf = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
            InputPerturbation::new(PerturbationSet::Fixed(p))
                .unwrap()
                .with_bounds(bounds)
                .unwrap(),
        )])
        .unwrap();
        let out = tf.transform(&x).unwrap();
        let expected = Matrix::from_row_slice(
            6,
            2,
            &[1.0, 0.2, 0.7, 0.9, 0.0, 0.6, 1.2, 0.4, 1.1, 0.9, 0.2, 0.8],
        );
        assert!((out - expected).amax() < 1e-15);
        assert_eq!(tf.points_per_input(), 3);
    }

    #[test]
    fn multiplicative_and_indices() {
        let p = Matrix::from_row_slice(2, 1, &[0.5, 2.0]);
        let x = Matrix::from_row_slice(2, 3, &[0.3, 0.2, 0.1, 0.6, 0.4, 0.8]);
        let tf = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
            InputPerturbation::new(PerturbationSet::Fixed(p))
                .unwrap()
                .with_indices(vec![1])
                .unwrap()
                .with_multiplicative(true),
        )])
        .unwrap();
        let out = tf.transform(&x).unwrap();
        assert_eq!(out.shape(), (4, 3));
        assert_eq!(
            out.row(1).iter().copied().collect::<Vec<_>>(),
            vec![0.3, 0.4, 0.1]
        );
        assert_eq!(
            out.row(2).iter().copied().collect::<Vec<_>>(),
            vec![0.6, 0.2, 0.8]
        );
    }

    /// Central differences of `⟨bar, T(x)⟩`.
    fn check_backward(tf: &ChainedInputTransform, x: &Matrix) {
        let (out, tape) = tf.transform_with_tape(x).unwrap();
        let bar = Matrix::from_fn(out.nrows(), out.ncols(), |r, c| {
            0.3 + 0.1 * r as f64 - 0.2 * c as f64
        });
        let g = tape.backward(0..x.nrows(), &bar).unwrap();
        let h = 1e-6;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let f = |delta: f64| {
                    let mut xp = x.clone();
                    xp[(i, j)] += delta;
                    tf.transform(&xp).unwrap().component_mul(&bar).sum()
                };
                let fd = (f(h) - f(-h)) / (2.0 * h);
                assert!(
                    (fd - g[(i, j)]).abs() < 1e-7,
                    "{i},{j}: {} vs {fd}",
                    g[(i, j)]
                );
            }
        }
        // Jacobians agree with the reverse mode.
        let jac = tape.jacobians().unwrap();
        let total = tape.points_per_input();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut s = 0.0;
                for w in 0..total {
                    for l in 0..x.ncols() {
                        s += bar[(i * total + w, l)] * jac[i * total + w][(l, j)];
                    }
                }
                assert!((s - g[(i, j)]).abs() < 1e-12);
            }
        }
        // A sub-range of rows.
        let g1 = tape
            .backward(1..2, &bar.rows(total, total).into_owned())
            .unwrap();
        assert!((g1.row(0) - g.row(1)).amax() < 1e-15);
    }

    #[test]
    fn chained_gradients() {
        let x = Matrix::from_row_slice(2, 2, &[0.5, 0.3, 0.9, 0.7]);
        let p = Matrix::from_row_slice(3, 2, &[0.05, -0.1, 0.2, 0.02, -0.07, 0.1]);
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let mut tf = ChainedInputTransform::new(vec![
            InputTransformStep::Normalize(
                Normalize::with_bounds(&Matrix::from_row_slice(2, 2, &[0.0, -1.0, 2.0, 1.0]))
                    .unwrap(),
            ),
            InputTransformStep::Perturbation(
                InputPerturbation::new(PerturbationSet::Fixed(p.clone()))
                    .unwrap()
                    .with_bounds(bounds.clone())
                    .unwrap(),
            ),
            InputTransformStep::Perturbation(
                InputPerturbation::new(PerturbationSet::Fixed(p.rows(0, 2).map(|v| 1.0 + v)))
                    .unwrap()
                    .with_multiplicative(true),
            ),
        ])
        .unwrap();
        tf.fit(&x).unwrap();
        assert_eq!(tf.points_per_input(), 6);
        check_backward(&tf, &x);
    }

    #[test]
    fn clamp_gradient_is_halved_at_the_bound() {
        // A zero perturbation of a point on the bound: torch splits the
        // gradient of minimum/maximum between equal arguments.
        let p = Matrix::from_row_slice(1, 1, &[0.0]);
        let tf = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
            InputPerturbation::new(PerturbationSet::Fixed(p))
                .unwrap()
                .with_bounds(Matrix::from_row_slice(2, 1, &[0.0, 1.0]))
                .unwrap(),
        )])
        .unwrap();
        let (_, tape) = tf
            .transform_with_tape(&Matrix::from_row_slice(2, 1, &[1.0, 0.5]))
            .unwrap();
        let g = tape
            .backward(0..2, &Matrix::from_row_slice(2, 1, &[1.0, 1.0]))
            .unwrap();
        assert_eq!(g[(0, 0)], 0.5);
        assert_eq!(g[(1, 0)], 1.0);
    }

    #[derive(Debug)]
    struct Scaled;

    impl PerturbationFunction for Scaled {
        fn perturbations(&self, x: &Matrix) -> Result<Matrix> {
            // BoTorch's test: stack([X * 0.1, X * 0.2]) (plus a nonlinearity).
            Ok(Matrix::from_fn(2 * x.nrows(), x.ncols(), |r, c| {
                let v = x[(r / 2, c)];
                if r % 2 == 0 {
                    0.1 * v * v
                } else {
                    0.2 * v
                }
            }))
        }

        fn vjp(&self, x: &Matrix, adjoint: &Matrix) -> Result<Matrix> {
            Ok(Matrix::from_fn(x.nrows(), x.ncols(), |i, c| {
                0.2 * x[(i, c)] * adjoint[(2 * i, c)] + 0.2 * adjoint[(2 * i + 1, c)]
            }))
        }
    }

    #[test]
    fn heteroscedastic_perturbations() {
        let x = Matrix::from_row_slice(3, 3, &[0.5, 0.3, 0.2, 0.9, 0.7, 0.1, 0.2, 0.4, 0.6]);
        for multiplicative in [false, true] {
            let mut tf = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
                InputPerturbation::new(PerturbationSet::Function(Arc::new(Scaled)))
                    .unwrap()
                    .with_indices(vec![2, 0])
                    .unwrap()
                    .with_multiplicative(multiplicative)
                    .with_bounds(Matrix::from_row_slice(2, 2, &[0.0, 0.0, 0.62, 0.9]))
                    .unwrap(),
            )])
            .unwrap();
            tf.fit(&x).unwrap();
            assert_eq!(tf.points_per_input(), 2);
            check_backward(&tf, &x);
        }
    }

    #[test]
    fn errors() {
        assert!(InputPerturbation::new(PerturbationSet::Fixed(Matrix::zeros(0, 2))).is_err());
        let p = InputPerturbation::new(PerturbationSet::Fixed(Matrix::zeros(2, 2))).unwrap();
        let e = p.clone().with_bounds(Matrix::zeros(2, 3)).unwrap_err();
        assert!(e.to_string().contains("same number of columns"));
        let mut tf = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
            p.with_transform_flags(true, true, false),
        )])
        .unwrap();
        assert!(matches!(
            tf.fit(&Matrix::zeros(3, 2)),
            Err(BotorchError::Unsupported(_))
        ));
        assert!(ChainedInputTransform::new(Vec::new()).is_err());
    }
}
