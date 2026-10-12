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
    /// `Normalize` (applied in train and eval mode; learnable bounds are
    /// fitted to the training inputs as transformed by the preceding steps).
    Normalize(Normalize),
    /// `InputPerturbation`.
    Perturbation(InputPerturbation),
}

/// BoTorch's `ChainedInputTransform` (a single step represents the plain
/// transform): the steps are applied in order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainedInputTransform {
    steps: Vec<InputTransformStep>,
}

/// The record of an evaluation of a [`ChainedInputTransform`] for its
/// derivatives (see [`ChainedInputTransform::transform_with_tape`]).
#[derive(Clone, Debug)]
pub struct InputTape {
    n: usize,
    d: usize,
    steps: Vec<TapeStep>,
}

#[derive(Clone, Debug)]
enum TapeStep {
    Normalize(Vector),
    Perturbation(PerturbationRecord),
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
        Ok(Self { steps })
    }

    /// The steps.
    pub fn steps(&self) -> &[InputTransformStep] {
        &self.steps
    }

    /// Whether some step maps a point to several points in eval mode
    /// (BoTorch's `is_one_to_many`, restricted to the steps that apply).
    pub fn is_one_to_many(&self) -> bool {
        self.steps
            .iter()
            .any(|s| matches!(s, InputTransformStep::Perturbation(p) if p.transform_on_eval))
    }

    /// The number of points per input point in eval mode: the product of
    /// `n_w` over the perturbations that apply (1 if none does).
    ///
    /// # Panics
    /// If a perturbation function has not been probed (see
    /// [`ChainedInputTransform::fit`]).
    pub fn points_per_input(&self) -> usize {
        self.steps
            .iter()
            .map(|s| match s {
                InputTransformStep::Perturbation(p) if p.transform_on_eval => p
                    .n_w
                    .expect("perturbation functions are probed when the transform is fitted"),
                _ => 1,
            })
            .product()
    }

    /// A copy whose `Normalize` bounds are never refitted (to share bounds
    /// learned from all batch elements of a batched model).
    pub fn frozen(&self) -> Self {
        Self {
            steps: self
                .steps
                .iter()
                .map(|s| match s {
                    InputTransformStep::Normalize(n) => InputTransformStep::Normalize(n.frozen()),
                    other => other.clone(),
                })
                .collect(),
        }
    }

    /// Fits the chain to training inputs (what BoTorch does in train mode):
    /// learnable `Normalize` bounds are fitted to the inputs as transformed by
    /// the preceding steps, and the number of perturbations of perturbation
    /// functions is determined at the first training point (as transformed
    /// by the preceding eval-mode steps).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a perturbation that applies in train
    /// mode (an exact GP cannot have more inputs than targets),
    /// [`BotorchError::DimensionMismatch`] for inconsistent dimensions and
    /// errors of perturbation functions.
    pub fn fit(&mut self, train_x: &Matrix) -> Result<()> {
        let mut x = train_x.clone();
        let mut probe = train_x.rows(0, train_x.nrows().min(1)).into_owned();
        for step in &mut self.steps {
            match step {
                InputTransformStep::Normalize(n) => {
                    if n.dim() != x.ncols() {
                        return Err(BotorchError::DimensionMismatch(format!(
                            "Wrong input dimension. Received {}, expected {}.",
                            x.ncols(),
                            n.dim()
                        )));
                    }
                    n.fit(&x)?;
                    x = n.transform(&x)?;
                    probe = n.transform(&probe)?;
                }
                InputTransformStep::Perturbation(p) => {
                    if p.transform_on_train {
                        return Err(BotorchError::Unsupported(
                            "InputPerturbation with transform_on_train=True is not supported: \
                             the perturbed training inputs (n * n_w points) would not match the \
                             n training targets of the model."
                                .into(),
                        ));
                    }
                    p.probe(&probe)?;
                    if p.transform_on_eval {
                        probe = p
                            .apply(&probe)?
                            .0
                            .rows(0, probe.nrows().min(1))
                            .into_owned();
                    }
                }
            }
        }
        Ok(())
    }

    /// The training inputs as the model sees them (BoTorch's
    /// `preprocess_transform`: the steps that apply in train mode, with the
    /// current bounds).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent dimensions.
    pub fn preprocess(&self, train_x: &Matrix) -> Result<Matrix> {
        let mut x = train_x.clone();
        for step in &self.steps {
            match step {
                InputTransformStep::Normalize(n) => x = n.transform(&x)?,
                InputTransformStep::Perturbation(p) => {
                    if p.transform_on_train {
                        x = p.apply(&x)?.0;
                    }
                }
            }
        }
        Ok(x)
    }

    /// The transformed points of `x` (`n × d`) in eval mode: `(n·N) × d` for
    /// `N` = [`ChainedInputTransform::points_per_input`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent dimensions and
    /// errors of perturbation functions.
    pub fn transform(&self, x: &Matrix) -> Result<Matrix> {
        Ok(self.transform_with_tape(x)?.0)
    }

    /// [`ChainedInputTransform::transform`] with the record of its
    /// derivatives.
    ///
    /// # Errors
    /// See [`ChainedInputTransform::transform`].
    pub fn transform_with_tape(&self, x: &Matrix) -> Result<(Matrix, InputTape)> {
        let mut cur = x.clone();
        let mut steps = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            match step {
                InputTransformStep::Normalize(n) => {
                    cur = n.transform(&cur)?;
                    steps.push(TapeStep::Normalize(n.coefficient().clone()));
                }
                InputTransformStep::Perturbation(p) => {
                    if p.transform_on_eval {
                        let (out, rec) = p.apply(&cur)?;
                        cur = out;
                        steps.push(TapeStep::Perturbation(rec));
                    }
                }
            }
        }
        Ok((
            cur,
            InputTape {
                n: x.nrows(),
                d: x.ncols(),
                steps,
            },
        ))
    }
}

impl InputTape {
    /// The number of raw points.
    pub fn num_inputs(&self) -> usize {
        self.n
    }

    /// The number of transformed points per raw point.
    pub fn points_per_input(&self) -> usize {
        self.steps
            .iter()
            .map(|s| match s {
                TapeStep::Perturbation(r) => r.nw,
                TapeStep::Normalize(_) => 1,
            })
            .product()
    }

    /// Reverse-mode derivative: for the adjoint of the transformed points of
    /// the raw points `rows` (`rows.len()·N × d`, the rows `rows.start·N ..
    /// rows.end·N` of the transformed points), the adjoint of those raw
    /// points (`rows.len() × d`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes and errors
    /// of perturbation functions.
    pub fn backward(&self, rows: Range<usize>, adjoint: &Matrix) -> Result<Matrix> {
        let total = self.points_per_input();
        if rows.end > self.n || adjoint.shape() != (rows.len() * total, self.d) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the adjoint of the transformed points of rows {rows:?} must be {}x{}, got {}x{}",
                rows.len() * total,
                self.d,
                adjoint.nrows(),
                adjoint.ncols()
            )));
        }
        // The number of points per raw point after each step.
        let mut per = Vec::with_capacity(self.steps.len() + 1);
        per.push(1usize);
        for s in &self.steps {
            let last = *per.last().expect("non-empty");
            per.push(
                last * match s {
                    TapeStep::Perturbation(r) => r.nw,
                    TapeStep::Normalize(_) => 1,
                },
            );
        }
        let mut bar = adjoint.clone();
        for (s, step) in self.steps.iter().enumerate().rev() {
            match step {
                TapeStep::Normalize(coef) => {
                    for j in 0..self.d {
                        let c = coef[j];
                        for r in 0..bar.nrows() {
                            bar[(r, j)] /= c;
                        }
                    }
                }
                TapeStep::Perturbation(rec) => {
                    // Input rows of this step for the raw rows `rows`.
                    let start = rows.start * per[s];
                    let n_in = rows.len() * per[s];
                    bar = rec.backward(start, n_in, &bar)?;
                }
            }
        }
        Ok(bar)
    }

    /// The Jacobian of every transformed point with respect to its raw point
    /// (`d × d`, entry `[l, j] = ∂x̃_l / ∂x_j`), in the order of the
    /// transformed points.
    ///
    /// # Errors
    /// Errors of perturbation functions.
    pub fn jacobians(&self) -> Result<Vec<Matrix>> {
        let total = self.points_per_input();
        let (n, d) = (self.n, self.d);
        let mut out = vec![Matrix::zeros(d, d); n * total];
        for slot in 0..total {
            for l in 0..d {
                let mut e = Matrix::zeros(n * total, d);
                for i in 0..n {
                    e[(i * total + slot, l)] = 1.0;
                }
                let g = self.backward(0..n, &e)?;
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
