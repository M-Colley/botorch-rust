//! The input transforms of `botorch.models.transforms.input` other than
//! `Normalize` (without options) and `InputPerturbation`, as steps of a
//! [`crate::transforms::input::ChainedInputTransform`]:
//!
//! * [`AffineStep`]: `AffineInputTransform`, `Normalize` (with `indices`,
//!   `center`, `reverse`, transform flags, learned or fixed bounds) and
//!   `InputStandardize`;
//! * [`WarpStep`]: `Warp`, the Kumaraswamy-CDF input warping with learned
//!   concentrations (fitted jointly with the GP; priors and the
//!   `GreaterThan(1e-4)` bound of BoTorch);
//! * [`RoundStep`]: `Round` (exact rounding with straight-through gradients,
//!   or the differentiable `approximate_round`);
//! * [`Log10Step`] (`Log10`);
//! * [`AppendStep`]: `AppendFeatures` (a fixed feature set, a callable, or
//!   the pairwise products of `InteractionFeatures`);
//! * [`FilterStep`] (`FilterFeatures`);
//! * [`OneHotToNumericStep`] and [`NumericToCategoricalStep`]: the
//!   categorical encodings;
//! * [`ImputationStep`]: `LearnedFeatureImputation` (learned imputation
//!   values of the features a task does not observe).
//!
//! Every step maps `n × d_in` inputs to `n·k × d_out` outputs (`k > 1` only
//! for `AppendFeatures` with several features, point-major like BoTorch) and
//! records what its reverse-mode derivative needs: the vector-Jacobian
//! product with respect to the inputs (what acquisition functions need) and,
//! for `Warp` and `LearnedFeatureImputation`, with respect to the learned
//! parameters (what the marginal log likelihood's gradient needs). The
//! derivatives follow torch's autograd, including its conventions at
//! non-differentiable points (`clamp` with scalar bounds passes the gradient
//! on the open interval only, as in torch ≥ 2.14; `round`, `argmax` and
//! `.long()` follow BoTorch's straight-through estimators or have zero
//! gradient).

use std::fmt::Debug;
use std::sync::Arc;

use crate::constraints::GreaterThan;
use crate::error::{BotorchError, Result};
use crate::priors::Prior;
use crate::transforms::utils::{interaction_features, kumaraswamy_warp_with_grad};
use crate::Matrix;

/// When a transform applies (BoTorch's `transform_on_train`,
/// `transform_on_eval`, `transform_on_fantasize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformFlags {
    /// Apply in train mode (to the training inputs).
    pub train: bool,
    /// Apply in eval mode (to the points of a posterior).
    pub eval: bool,
    /// Apply in eval mode inside `fantasize`.
    pub fantasize: bool,
}

impl TransformFlags {
    /// All three flags set (BoTorch's default for most transforms).
    pub const ALL: TransformFlags = TransformFlags {
        train: true,
        eval: true,
        fantasize: true,
    };

    /// Whether a transform with these flags applies in `mode`.
    pub fn applies(&self, mode: TransformMode) -> bool {
        match mode {
            TransformMode::Train => self.train,
            TransformMode::Eval => self.eval,
            TransformMode::Fantasize => self.eval && self.fantasize,
        }
    }
}

impl Default for TransformFlags {
    fn default() -> Self {
        Self::ALL
    }
}

/// The mode in which an input transform is applied: to the training inputs
/// (BoTorch's train mode and `preprocess_transform`), to the points of a
/// posterior (eval mode), or to the points of a posterior inside
/// `fantasize` (eval mode with BoTorch's fantasize flag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformMode {
    /// Train mode / training inputs.
    Train,
    /// Eval mode.
    Eval,
    /// Eval mode inside `fantasize`.
    Fantasize,
}

fn check_indices(indices: &[usize], d: usize, what: &str) -> Result<()> {
    if let Some(&bad) = indices.iter().find(|&&i| i >= d) {
        return Err(BotorchError::DimensionMismatch(format!(
            "index {bad} of {what} is out of bounds for dimension with size {d}"
        )));
    }
    Ok(())
}

fn check_unique(indices: &[usize], what: &str) -> Result<()> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != indices.len() {
        return Err(BotorchError::InvalidArgument(format!(
            "Elements of `{what}` tensor must be unique!"
        )));
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Affine transforms
// ----------------------------------------------------------------------------

/// How an [`AffineStep`] learns its coefficients in train mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AffineKind {
    /// `Normalize`: offset = min, coefficient = max − min (ranges below
    /// `min_range` are not normalized), shifted by `(0.5 − center) · coefficient`.
    Normalize {
        /// BoTorch's `min_range`.
        min_range: f64,
        /// BoTorch's `center`.
        center: f64,
    },
    /// `InputStandardize`: offset = mean, coefficient = unbiased standard
    /// deviation (dimensions whose standard deviation is below `min_std`, or
    /// NaN for a single point, are not standardized).
    Standardize {
        /// BoTorch's `min_std`.
        min_std: f64,
    },
    /// `AffineInputTransform` with fixed coefficients.
    Fixed,
}

/// BoTorch's `AffineInputTransform` and its subclasses `Normalize` and
/// `InputStandardize`: `x_j ↦ (x_j − offset_j) / coefficient_j` for the
/// dimensions `indices` (all by default), or the inverse with `reverse`.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineStep {
    d: usize,
    indices: Option<Vec<usize>>,
    offset: Vec<f64>,
    coefficient: Vec<f64>,
    kind: AffineKind,
    learn: bool,
    reverse: bool,
    flags: TransformFlags,
}

impl AffineStep {
    /// An affine transform of the `indices` (all `d` dimensions if `None`)
    /// with the given offsets and coefficients (one per transformed
    /// dimension).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for empty, repeated or out-of-range
    /// indices (BoTorch's messages) and [`BotorchError::DimensionMismatch`]
    /// for coefficients of the wrong length.
    pub fn new(
        d: usize,
        indices: Option<Vec<usize>>,
        offset: Vec<f64>,
        coefficient: Vec<f64>,
        kind: AffineKind,
        learn: bool,
    ) -> Result<Self> {
        if let Some(idx) = &indices {
            if idx.is_empty() {
                return Err(BotorchError::InvalidArgument(
                    "`indices` list is empty!".into(),
                ));
            }
            if idx.len() > d {
                return Err(BotorchError::InvalidArgument(
                    "Can provide at most `d` indices!".into(),
                ));
            }
            if idx.iter().any(|&i| i >= d) {
                return Err(BotorchError::InvalidArgument(
                    "Elements of `indices` have to be smaller than `d`!".into(),
                ));
            }
            check_unique(idx, "indices")?;
        }
        let k = indices.as_ref().map_or(d, Vec::len);
        if offset.len() != k || coefficient.len() != k {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {k} offsets and coefficients, got {} and {}",
                offset.len(),
                coefficient.len()
            )));
        }
        Ok(Self {
            d,
            indices,
            offset,
            coefficient,
            kind,
            learn,
            reverse: false,
            flags: TransformFlags::ALL,
        })
    }

    /// Sets `reverse` (the forward pass un-transforms).
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The transform with its current coefficients, never refitted.
    pub fn frozen(mut self) -> Self {
        self.learn = false;
        self
    }

    /// The input dimension `d`.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// The transformed dimensions (`None`: all).
    pub fn indices(&self) -> Option<&[usize]> {
        self.indices.as_deref()
    }

    /// The offsets (one per transformed dimension).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// The coefficients.
    pub fn coefficient(&self) -> &[f64] {
        &self.coefficient
    }

    /// How the coefficients are learned.
    pub fn kind(&self) -> AffineKind {
        self.kind
    }

    /// Whether the coefficients are learned in train mode (BoTorch's
    /// `learn_coefficients`).
    pub fn learn(&self) -> bool {
        self.learn
    }

    /// BoTorch's `reverse`.
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    fn columns(&self) -> Vec<usize> {
        match &self.indices {
            Some(i) => i.clone(),
            None => (0..self.d).collect(),
        }
    }

    fn check(&self, x: &Matrix) -> Result<()> {
        if x.ncols() != self.d {
            return Err(BotorchError::DimensionMismatch(format!(
                "Wrong input dimension. Received {}, expected {}.",
                x.ncols(),
                self.d
            )));
        }
        Ok(())
    }

    /// Learns the coefficients from `x` (BoTorch's `_update_coefficients`,
    /// over all points); a no-op unless they are learned.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns.
    pub fn fit(&mut self, x: &Matrix) -> Result<()> {
        self.check(x)?;
        if !self.learn || x.nrows() == 0 {
            return Ok(());
        }
        let cols = self.columns();
        let n = x.nrows() as f64;
        for (c, &j) in cols.iter().enumerate() {
            let col = x.column(j);
            match self.kind {
                AffineKind::Normalize { min_range, center } => {
                    let min = col.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;
                    let almost_zero = range < min_range;
                    let coef = if almost_zero { 1.0 } else { range };
                    let off = if almost_zero { 0.0 } else { min };
                    self.coefficient[c] = coef;
                    self.offset[c] = off + (0.5 - center) * coef;
                }
                AffineKind::Standardize { min_std } => {
                    let mean = col.iter().sum::<f64>() / n;
                    let ss = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
                    let std = (ss / (n - 1.0)).sqrt();
                    // A NaN standard deviation (one point) is treated as zero
                    // spread (meta-pytorch/botorch#3389).
                    let almost_zero = std.partial_cmp(&min_std).is_none_or(|o| o.is_lt());
                    self.coefficient[c] = if almost_zero { 1.0 } else { std };
                    self.offset[c] = if almost_zero { 0.0 } else { mean };
                }
                AffineKind::Fixed => {}
            }
        }
        Ok(())
    }

    /// Applies the step (the forward transform, or its inverse with
    /// `reverse`).
    fn apply(&self, x: &Matrix) -> Result<Matrix> {
        self.check(x)?;
        let mut out = x.clone();
        for (c, &j) in self.columns().iter().enumerate() {
            let (o, s) = (self.offset[c], self.coefficient[c]);
            for i in 0..x.nrows() {
                out[(i, j)] = if self.reverse {
                    s * x[(i, j)] + o
                } else {
                    (x[(i, j)] - o) / s
                };
            }
        }
        Ok(out)
    }

    /// The inverse of [`AffineStep::apply`].
    fn invert(&self, x: &Matrix) -> Result<Matrix> {
        self.check(x)?;
        let mut out = x.clone();
        for (c, &j) in self.columns().iter().enumerate() {
            let (o, s) = (self.offset[c], self.coefficient[c]);
            for i in 0..x.nrows() {
                out[(i, j)] = if self.reverse {
                    (x[(i, j)] - o) / s
                } else {
                    s * x[(i, j)] + o
                };
            }
        }
        Ok(out)
    }

    /// The (constant, diagonal) derivative of every output column.
    pub(crate) fn diag_jacobian(&self) -> Vec<f64> {
        let mut jac = vec![1.0; self.d];
        for (c, &j) in self.columns().iter().enumerate() {
            jac[j] = if self.reverse {
                self.coefficient[c]
            } else {
                1.0 / self.coefficient[c]
            };
        }
        jac
    }
}

// ----------------------------------------------------------------------------
// Warp
// ----------------------------------------------------------------------------

/// BoTorch's `Warp._min_concentration_level`.
pub const WARP_MIN_CONCENTRATION: f64 = 1e-4;

/// BoTorch's `Warp`: the dimensions `indices` are normalized to the unit cube
/// (by bounds given or learned in train mode) and warped through the
/// Kumaraswamy CDF `1 − (1 − x^c1)^c0` with learned concentrations
/// (`concentration0`, `concentration1`, one per warped dimension, initially
/// 1), constrained to `≥ 1e-4` and optionally with priors.
#[derive(Clone, Debug, PartialEq)]
pub struct WarpStep {
    indices: Vec<usize>,
    eps: f64,
    normalize: AffineStep,
    c0: Vec<f64>,
    c1: Vec<f64>,
    c0_prior: Option<Prior>,
    c1_prior: Option<Prior>,
    reverse: bool,
    flags: TransformFlags,
    batched: bool,
}

impl WarpStep {
    /// `Warp(d, indices, eps, bounds)`: `bounds` (lower and upper bounds of
    /// the warped dimensions, or of all `d`) normalize the inputs, or the
    /// bounds are learned from the training inputs if `None`.
    ///
    /// # Errors
    /// Errors of the normalization's indices and bounds.
    pub fn new(
        d: usize,
        indices: Vec<usize>,
        eps: f64,
        bounds: Option<(Vec<f64>, Vec<f64>)>,
    ) -> Result<Self> {
        if indices.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "`indices` list is empty!".into(),
            ));
        }
        let k = indices.len();
        let normalize = match bounds {
            Some((lo, hi)) => {
                let (lo, hi) = if lo.len() == d && k != d {
                    (
                        indices.iter().map(|&i| lo[i]).collect::<Vec<_>>(),
                        indices.iter().map(|&i| hi[i]).collect::<Vec<_>>(),
                    )
                } else {
                    (lo, hi)
                };
                if lo.len() != k || hi.len() != k {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "Dimensions of provided `bounds` are incompatible with transform_dimension = {k}!"
                    )));
                }
                let coef: Vec<f64> = (0..k).map(|c| hi[c] - lo[c]).collect();
                AffineStep::new(
                    d,
                    Some(indices.clone()),
                    lo,
                    coef,
                    AffineKind::Normalize {
                        min_range: 1e-8,
                        center: 0.5,
                    },
                    false,
                )?
            }
            None => AffineStep::new(
                d,
                Some(indices.clone()),
                vec![0.0; k],
                vec![1.0; k],
                AffineKind::Normalize {
                    min_range: 1e-8,
                    center: 0.5,
                },
                true,
            )?,
        };
        Ok(Self {
            indices,
            eps,
            normalize,
            c0: vec![1.0; k],
            c1: vec![1.0; k],
            c0_prior: None,
            c1_prior: None,
            reverse: false,
            flags: TransformFlags::ALL,
            batched: false,
        })
    }

    /// Sets the priors on `concentration0` and `concentration1`.
    pub fn with_priors(mut self, c0: Option<Prior>, c1: Option<Prior>) -> Self {
        self.c0_prior = c0;
        self.c1_prior = c1;
        self
    }

    /// Sets `reverse`.
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The transform with its current bounds, never refitted.
    pub fn frozen(mut self) -> Self {
        self.normalize.learn = false;
        self
    }

    /// Marks the concentrations as an element of a batched `Warp` (BoTorch's
    /// `batch_shape`: parameter shape `batch_shape x 1 x k` instead of `k`).
    pub fn with_batched(mut self, batched: bool) -> Self {
        self.batched = batched;
        self
    }

    /// Sets the concentrations.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for wrong lengths.
    pub fn with_concentrations(mut self, c0: Vec<f64>, c1: Vec<f64>) -> Result<Self> {
        if c0.len() != self.indices.len() || c1.len() != self.indices.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} concentrations each, got {} and {}",
                self.indices.len(),
                c0.len(),
                c1.len()
            )));
        }
        self.c0 = c0;
        self.c1 = c1;
        Ok(self)
    }

    /// The warped dimensions.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// `eps`.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The normalization of the warped dimensions.
    pub fn normalize(&self) -> &AffineStep {
        &self.normalize
    }

    /// `concentration0`.
    pub fn concentration0(&self) -> &[f64] {
        &self.c0
    }

    /// `concentration1`.
    pub fn concentration1(&self) -> &[f64] {
        &self.c1
    }

    /// The priors on `concentration0` and `concentration1`.
    pub fn priors(&self) -> (Option<Prior>, Option<Prior>) {
        (self.c0_prior, self.c1_prior)
    }

    /// BoTorch's `reverse`.
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    /// Whether the concentrations belong to a batched `Warp`.
    pub fn batched(&self) -> bool {
        self.batched
    }

    /// The constraint of the concentrations: `GreaterThan(1e-4,
    /// transform=None, initial_value=1.0)` (float32 constants, like GPyTorch).
    pub fn constraint() -> GreaterThan {
        GreaterThan::new(WARP_MIN_CONCENTRATION, None, Some(1.0)).rounded_to_f32()
    }

    /// The log prior density of the concentrations and its gradient (`c0`
    /// then `c1`), in GPyTorch's order (the `concentration0` prior first).
    pub fn log_prior_and_grad(&self) -> (f64, Vec<f64>) {
        let k = self.indices.len();
        let mut lp = 0.0;
        let mut g = vec![0.0; 2 * k];
        if let Some(p) = self.c0_prior {
            lp += self.c0.iter().map(|&c| p.log_prob(c)).sum::<f64>();
            for (j, &c) in self.c0.iter().enumerate() {
                g[j] = p.grad_log_prob(c);
            }
        }
        if let Some(p) = self.c1_prior {
            lp += self.c1.iter().map(|&c| p.log_prob(c)).sum::<f64>();
            for (j, &c) in self.c1.iter().enumerate() {
                g[k + j] = p.grad_log_prob(c);
            }
        }
        (lp, g)
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        self.normalize.fit(x)
    }

    /// The forward warp (no `reverse`) with the derivatives of every warped
    /// value with respect to its input and the concentrations.
    fn warp(&self, x: &Matrix) -> Result<(Matrix, WarpRecord)> {
        let xn = self.normalize.apply(x)?;
        let n = x.nrows();
        let k = self.indices.len();
        let mut out = xn.clone();
        let mut dx = Matrix::zeros(n, k);
        let mut dc0 = Matrix::zeros(n, k);
        let mut dc1 = Matrix::zeros(n, k);
        for (c, &j) in self.indices.iter().enumerate() {
            let scale = self.normalize.diag_jacobian()[j];
            for i in 0..n {
                let (v, gx, g0, g1) =
                    kumaraswamy_warp_with_grad(xn[(i, j)], self.c0[c], self.c1[c], self.eps);
                out[(i, j)] = v;
                dx[(i, c)] = gx * scale;
                dc0[(i, c)] = g0;
                dc1[(i, c)] = g1;
            }
        }
        Ok((out, WarpRecord { dx, dc0, dc1 }))
    }

    /// BoTorch's `_untransform`: the inverse Kumaraswamy CDF, then the
    /// inverse normalization.
    fn unwarp(&self, x: &Matrix) -> Result<Matrix> {
        let mut out = x.clone();
        for (c, &j) in self.indices.iter().enumerate() {
            for i in 0..x.nrows() {
                out[(i, j)] = crate::transforms::utils::inv_kumaraswamy_warp(
                    x[(i, j)],
                    self.c0[c],
                    self.c1[c],
                    self.eps,
                );
            }
        }
        self.normalize.invert(&out)
    }
}

#[derive(Clone, Debug)]
struct WarpRecord {
    /// ∂ output / ∂ input of the warped columns (`n × k`).
    dx: Matrix,
    /// ∂ output / ∂ c0 (`n × k`).
    dc0: Matrix,
    /// ∂ output / ∂ c1 (`n × k`).
    dc1: Matrix,
}

// ----------------------------------------------------------------------------
// Round, Log10
// ----------------------------------------------------------------------------

/// BoTorch's `Round`: rounds the `integer_indices` (half to even, like
/// `torch.round`, with straight-through gradients, or the differentiable
/// `approximate_round` with temperature `tau`) and discretizes one-hot
/// encoded categoricals (`start`, `cardinality`) to the largest entry.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundStep {
    integer_indices: Vec<usize>,
    categorical: Vec<(usize, usize)>,
    approximate: bool,
    tau: f64,
    flags: TransformFlags,
}

impl RoundStep {
    /// A rounding transform.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for approximate rounding of categoricals
    /// (BoTorch's `NotImplementedError`).
    pub fn new(
        integer_indices: Vec<usize>,
        categorical: Vec<(usize, usize)>,
        approximate: bool,
        tau: f64,
    ) -> Result<Self> {
        if approximate && !categorical.is_empty() {
            return Err(BotorchError::Unsupported(
                "Approximate rounding of categorical features is not implemented.".into(),
            ));
        }
        Ok(Self {
            integer_indices,
            categorical,
            approximate,
            tau,
            flags: TransformFlags::ALL,
        })
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The integer dimensions.
    pub fn integer_indices(&self) -> &[usize] {
        &self.integer_indices
    }

    /// The categorical features `(start, cardinality)`.
    pub fn categorical_features(&self) -> &[(usize, usize)] {
        &self.categorical
    }

    /// Whether the rounding is approximate.
    pub fn approximate(&self) -> bool {
        self.approximate
    }

    /// The temperature of approximate rounding.
    pub fn tau(&self) -> f64 {
        self.tau
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    fn check(&self, d: usize) -> Result<()> {
        check_indices(&self.integer_indices, d, "integer_indices")?;
        for &(s, c) in &self.categorical {
            if s + c > d {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the categorical feature at {s} with cardinality {c} exceeds the input \
                     dimension {d}"
                )));
            }
        }
        Ok(())
    }

    /// The rounded inputs and the derivative of every integer entry (1 for
    /// exact rounding).
    fn apply(&self, x: &Matrix) -> Result<(Matrix, Matrix)> {
        let (n, d) = x.shape();
        self.check(d)?;
        let mut out = x.clone();
        let mut dint = Matrix::from_element(n, self.integer_indices.len(), 1.0);
        for (c, &j) in self.integer_indices.iter().enumerate() {
            for i in 0..n {
                let v = x[(i, j)];
                out[(i, j)] = if self.approximate {
                    let offset = v.floor();
                    let r = (v - offset - 0.5) / self.tau;
                    let t = r.tanh();
                    dint[(i, c)] = (1.0 - t * t) / (2.0 * self.tau);
                    offset + (t + 1.0) / 2.0
                } else {
                    v.round_ties_even()
                };
            }
        }
        for &(start, card) in &self.categorical {
            for i in 0..n {
                // argmax: the first maximum (NaN counts as the maximum, as in torch).
                let mut best = start;
                for j in start..start + card {
                    let (v, b) = (x[(i, j)], x[(i, best)]);
                    if v.is_nan() && !b.is_nan() || v > b {
                        best = j;
                    }
                }
                for j in start..start + card {
                    out[(i, j)] = if j == best { 1.0 } else { 0.0 };
                }
            }
        }
        Ok((out, dint))
    }
}

/// BoTorch's `Log10` of the dimensions `indices` (`10^x` with `reverse`).
#[derive(Clone, Debug, PartialEq)]
pub struct Log10Step {
    indices: Vec<usize>,
    reverse: bool,
    flags: TransformFlags,
}

impl Log10Step {
    /// A `Log10` of `indices`.
    pub fn new(indices: Vec<usize>) -> Self {
        Self {
            indices,
            reverse: false,
            flags: TransformFlags::ALL,
        }
    }

    /// Sets `reverse`.
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The transformed dimensions.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// BoTorch's `reverse`.
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    fn map(&self, x: &Matrix, inverse: bool) -> Result<(Matrix, Matrix)> {
        check_indices(&self.indices, x.ncols(), "indices")?;
        let mut out = x.clone();
        let mut der = Matrix::zeros(x.nrows(), self.indices.len());
        for (c, &j) in self.indices.iter().enumerate() {
            for i in 0..x.nrows() {
                let v = x[(i, j)];
                if inverse {
                    let y = 10f64.powf(v);
                    out[(i, j)] = y;
                    der[(i, c)] = y * std::f64::consts::LN_10;
                } else {
                    out[(i, j)] = v.log10();
                    der[(i, c)] = 1.0 / (v * std::f64::consts::LN_10);
                }
            }
        }
        Ok((out, der))
    }
}

// ----------------------------------------------------------------------------
// AppendFeatures, InteractionFeatures, FilterFeatures
// ----------------------------------------------------------------------------

/// A callable that generates features to append (BoTorch's `AppendFeatures`
/// with `f`): for `n` points (`n × k`, the columns `indices`), the `n_f`
/// features of every point, `(n·n_f) × d_f`, row `i·n_f + j` for feature
/// `j` of point `i`.
pub trait FeatureFunction: Send + Sync + Debug {
    /// The features of the points `x`.
    ///
    /// # Errors
    /// Implementation-specific.
    fn features(&self, x: &Matrix) -> Result<Matrix>;

    /// The vector-Jacobian product: the adjoint of `x` (`n × k`) for the
    /// adjoint of the features (`(n·n_f) × d_f`).
    ///
    /// # Errors
    /// Implementation-specific.
    fn vjp(&self, x: &Matrix, adjoint: &Matrix) -> Result<Matrix>;
}

/// The features of an [`AppendStep`].
#[derive(Clone, Debug)]
pub enum FeatureSource {
    /// A fixed `n_f × d_f` feature set, appended to every point.
    Fixed(Matrix),
    /// A callable on the columns `indices` (all if `None`).
    Function(Arc<dyn FeatureFunction>, Option<Vec<usize>>),
    /// The pairwise products of the columns `indices` (all if `None`):
    /// BoTorch's `InteractionFeatures`.
    Interaction(Option<Vec<usize>>),
}

impl PartialEq for FeatureSource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Fixed(a), Self::Fixed(b)) => a == b,
            (Self::Function(a, i), Self::Function(b, j)) => Arc::ptr_eq(a, b) && i == j,
            (Self::Interaction(a), Self::Interaction(b)) => a == b,
            _ => false,
        }
    }
}

/// BoTorch's `AppendFeatures` (and `InteractionFeatures`): every point is
/// replaced by `n_f` points with the features appended (point-major), or,
/// with `skip_expand`, the `q·n_f` points of the input (e.g. after an
/// `InputPerturbation` with `n_f` perturbations) get feature `r mod n_f`.
#[derive(Clone, Debug, PartialEq)]
pub struct AppendStep {
    source: FeatureSource,
    skip_expand: bool,
    flags: TransformFlags,
    /// The number of features `n_f` (known for a fixed set and interaction
    /// features; determined by the first evaluation of a function).
    n_f: Option<usize>,
    /// The feature dimension `d_f` (as `n_f`).
    d_f: Option<usize>,
}

impl AppendStep {
    /// Whether the size of the appended features is determined by the
    /// first evaluation (a feature function or interaction features of all
    /// dimensions).
    pub(crate) fn sized_by_inputs(&self) -> bool {
        matches!(
            self.source,
            FeatureSource::Function(..) | FeatureSource::Interaction(None)
        )
    }

    /// An `AppendFeatures` with BoTorch's default flags (eval only).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty feature set or
    /// `indices`, repeated indices, and `skip_expand` with a callable
    /// (BoTorch's messages).
    pub fn new(source: FeatureSource, skip_expand: bool) -> Result<Self> {
        let (n_f, d_f) = match &source {
            FeatureSource::Fixed(f) => {
                if f.nrows() == 0 {
                    return Err(BotorchError::InvalidArgument(
                        "`feature_set` must be an `n_f x d_f`-dim tensor!".into(),
                    ));
                }
                (Some(f.nrows()), Some(f.ncols()))
            }
            FeatureSource::Function(_, idx) => {
                if skip_expand {
                    return Err(BotorchError::InvalidArgument(
                        "`skip_expand` option is not supported in case of using a callable".into(),
                    ));
                }
                if let Some(i) = idx {
                    if i.is_empty() {
                        return Err(BotorchError::InvalidArgument(
                            "`indices` list is empty!".into(),
                        ));
                    }
                    check_unique(i, "indices")?;
                }
                (None, None)
            }
            FeatureSource::Interaction(idx) => {
                if let Some(i) = idx {
                    if i.is_empty() {
                        return Err(BotorchError::InvalidArgument(
                            "`indices` list is empty!".into(),
                        ));
                    }
                    check_unique(i, "indices")?;
                    (Some(1), Some(i.len() * (i.len() - 1) / 2))
                } else {
                    (Some(1), None)
                }
            }
        };
        Ok(Self {
            source,
            skip_expand,
            flags: TransformFlags {
                train: false,
                eval: true,
                fantasize: false,
            },
            n_f,
            d_f,
        })
    }

    /// BoTorch's `InteractionFeatures(indices)` (applied in all modes).
    pub fn interaction(indices: Option<Vec<usize>>) -> Result<Self> {
        Ok(Self::new(FeatureSource::Interaction(indices), false)?.with_flags(TransformFlags::ALL))
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The feature source.
    pub fn source(&self) -> &FeatureSource {
        &self.source
    }

    /// BoTorch's `skip_expand`.
    pub fn skip_expand(&self) -> bool {
        self.skip_expand
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    /// The number of features per point, if known.
    pub fn n_f(&self) -> Option<usize> {
        self.n_f
    }

    /// The number of points per input point (`n_f`, or 1 with
    /// `skip_expand`).
    fn expand(&self) -> Option<usize> {
        if self.skip_expand {
            Some(1)
        } else {
            self.n_f
        }
    }

    fn feature_columns(&self, d: usize) -> Result<Vec<usize>> {
        let idx = match &self.source {
            FeatureSource::Fixed(_) => return Ok(Vec::new()),
            FeatureSource::Function(_, i) | FeatureSource::Interaction(i) => i.clone(),
        };
        let cols = idx.unwrap_or_else(|| (0..d).collect());
        check_indices(&cols, d, "indices")?;
        Ok(cols)
    }

    /// The features of the points `x` (`(n·n_f) × d_f`), with `n_f`.
    fn features(&self, x: &Matrix) -> Result<(Matrix, usize)> {
        let n = x.nrows();
        match &self.source {
            FeatureSource::Fixed(f) => {
                let nf = f.nrows();
                Ok((
                    Matrix::from_fn(n * nf, f.ncols(), |r, c| f[(r % nf, c)]),
                    nf,
                ))
            }
            FeatureSource::Interaction(_) => {
                let cols = self.feature_columns(x.ncols())?;
                let m = cols.len() * cols.len().saturating_sub(1) / 2;
                let mut out = Matrix::zeros(n, m);
                for i in 0..n {
                    let p: Vec<f64> = cols.iter().map(|&j| x[(i, j)]).collect();
                    for (c, v) in interaction_features(&p).into_iter().enumerate() {
                        out[(i, c)] = v;
                    }
                }
                Ok((out, 1))
            }
            FeatureSource::Function(f, _) => {
                let cols = self.feature_columns(x.ncols())?;
                let xi = Matrix::from_fn(n, cols.len(), |i, c| x[(i, cols[c])]);
                let feats = f.features(&xi)?;
                if n == 0 {
                    return Ok((feats, self.n_f.unwrap_or(1)));
                }
                if feats.nrows() % n != 0 || feats.nrows() == 0 {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "the feature function returned {} rows for {n} points (expected n·n_f)",
                        feats.nrows()
                    )));
                }
                let nf = feats.nrows() / n;
                if let Some(e) = self.n_f {
                    if e != nf || self.d_f != Some(feats.ncols()) {
                        return Err(BotorchError::DimensionMismatch(format!(
                            "the feature function returned {nf} features of dimension {} per \
                             point, but {e} of dimension {:?} before",
                            feats.ncols(),
                            self.d_f
                        )));
                    }
                }
                Ok((feats, nf))
            }
        }
    }

    /// Determines `n_f` and `d_f` of a callable from one point (a no-op
    /// otherwise).
    fn probe(&mut self, x: &Matrix) -> Result<()> {
        if self.n_f.is_some() && self.d_f.is_some() {
            return Ok(());
        }
        if let FeatureSource::Interaction(None) = &self.source {
            let d = x.ncols();
            self.d_f = Some(d * d.saturating_sub(1) / 2);
            return Ok(());
        }
        if x.nrows() == 0 {
            return Ok(());
        }
        let (f, nf) = self.features(&x.rows(0, 1).into_owned())?;
        self.n_f = Some(nf);
        self.d_f = Some(f.ncols());
        Ok(())
    }

    fn out_dim(&self, d: usize) -> Result<usize> {
        match (&self.source, self.d_f) {
            (FeatureSource::Interaction(None), _) => Ok(d + d * d.saturating_sub(1) / 2),
            (_, Some(df)) => Ok(d + df),
            _ => Err(BotorchError::Runtime(
                "the dimension of the appended features is unknown before the transform is \
                 evaluated"
                    .into(),
            )),
        }
    }

    fn apply(&self, x: &Matrix) -> Result<(Matrix, AppendRecord)> {
        let (n, d) = x.shape();
        if self.skip_expand {
            let nf = self.n_f.expect("fixed feature sets know n_f");
            if n % nf != 0 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "with skip_expand, the number of points ({n}) must be a multiple of the \
                     number of features ({nf})"
                )));
            }
            let FeatureSource::Fixed(f) = &self.source else {
                unreachable!("skip_expand requires a fixed feature set")
            };
            let df = f.ncols();
            let out = Matrix::from_fn(n, d + df, |r, c| {
                if c < d {
                    x[(r, c)]
                } else {
                    f[(r % nf, c - d)]
                }
            });
            return Ok((out, AppendRecord { nf: 1, d, xi: None }));
        }
        let (feats, nf) = self.features(x)?;
        let df = feats.ncols();
        let out = Matrix::from_fn(n * nf, d + df, |r, c| {
            if c < d {
                x[(r / nf, c)]
            } else {
                feats[(r, c - d)]
            }
        });
        let xi = match &self.source {
            FeatureSource::Fixed(_) => None,
            _ => {
                let cols = self.feature_columns(d)?;
                Some((
                    cols.clone(),
                    Matrix::from_fn(n, cols.len(), |i, c| x[(i, cols[c])]),
                ))
            }
        };
        Ok((out, AppendRecord { nf, d, xi }))
    }

    fn backward(
        &self,
        rec: &AppendRecord,
        start: usize,
        n_in: usize,
        bar: &Matrix,
    ) -> Result<Matrix> {
        let (nf, d) = (rec.nf, rec.d);
        let mut out = Matrix::zeros(n_in, d);
        for il in 0..n_in {
            for w in 0..nf {
                let r = il * nf + w;
                for j in 0..d {
                    out[(il, j)] += bar[(r, j)];
                }
            }
        }
        let Some((cols, xi)) = &rec.xi else {
            return Ok(out);
        };
        let df = bar.ncols() - d;
        let fbar = Matrix::from_fn(n_in * nf, df, |r, c| bar[(r, d + c)]);
        let xs = xi.rows(start, n_in).into_owned();
        let gx = match &self.source {
            FeatureSource::Function(f, _) => f.vjp(&xs, &fbar)?,
            FeatureSource::Interaction(_) => {
                let k = cols.len();
                let mut g = Matrix::zeros(n_in, k);
                for i in 0..n_in {
                    let mut c = 0;
                    for a in 0..k {
                        for b in a + 1..k {
                            g[(i, a)] += fbar[(i, c)] * xs[(i, b)];
                            g[(i, b)] += fbar[(i, c)] * xs[(i, a)];
                            c += 1;
                        }
                    }
                }
                g
            }
            FeatureSource::Fixed(_) => unreachable!("fixed features record no inputs"),
        };
        if gx.shape() != (n_in, cols.len()) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the vector-Jacobian product of the feature function has shape {}x{}, expected \
                 {n_in}x{}",
                gx.nrows(),
                gx.ncols(),
                cols.len()
            )));
        }
        for i in 0..n_in {
            for (c, &j) in cols.iter().enumerate() {
                out[(i, j)] += gx[(i, c)];
            }
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct AppendRecord {
    nf: usize,
    d: usize,
    /// The feature columns and their values (`n × k`) for callables and
    /// interaction features.
    xi: Option<(Vec<usize>, Matrix)>,
}

/// BoTorch's `FilterFeatures`: keeps the columns `feature_indices`.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterStep {
    feature_indices: Vec<usize>,
    flags: TransformFlags,
}

impl FilterStep {
    /// A `FilterFeatures` keeping `feature_indices` (in that order).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for repeated indices.
    pub fn new(feature_indices: Vec<usize>) -> Result<Self> {
        check_unique(&feature_indices, "feature_indices")?;
        Ok(Self {
            feature_indices,
            flags: TransformFlags::ALL,
        })
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The kept columns.
    pub fn feature_indices(&self) -> &[usize] {
        &self.feature_indices
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }
}

// ----------------------------------------------------------------------------
// Categorical encodings
// ----------------------------------------------------------------------------

/// BoTorch's `OneHotToNumeric`: one-hot encoded categoricals (`start`,
/// `cardinality`) become their (float) category index, numerical features
/// are kept.
#[derive(Clone, Debug, PartialEq)]
pub struct OneHotToNumericStep {
    dim: usize,
    categorical: Vec<(usize, usize)>,
    numerical_idx: Vec<usize>,
    ordinal_idx: Vec<usize>,
    new_numerical_idx: Vec<usize>,
    flags: TransformFlags,
}

impl OneHotToNumericStep {
    /// `OneHotToNumeric(dim, categorical_features)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for overlapping features or features
    /// beyond `dim` (BoTorch's messages).
    pub fn new(dim: usize, mut categorical: Vec<(usize, usize)>) -> Result<Self> {
        categorical.sort_by_key(|&(s, _)| s);
        let mut onehot: Vec<usize> = Vec::new();
        for &(s, c) in &categorical {
            onehot.extend(s..s + c);
        }
        let mut uniq = onehot.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() != onehot.len() {
            return Err(BotorchError::InvalidArgument(
                "Categorical features overlap.".into(),
            ));
        }
        if onehot.iter().any(|&i| i >= dim) {
            return Err(BotorchError::InvalidArgument(
                "Categorical features exceed the provided dimension.".into(),
            ));
        }
        let numerical_idx: Vec<usize> = (0..dim).filter(|i| !onehot.contains(i)).collect();
        let mut ordinal_idx = Vec::new();
        let mut offset = 0;
        for &(s, c) in &categorical {
            ordinal_idx.push(s - offset);
            offset += c - 1;
        }
        let reduced = ordinal_idx.len() + numerical_idx.len();
        let new_numerical_idx: Vec<usize> =
            (0..reduced).filter(|i| !ordinal_idx.contains(i)).collect();
        Ok(Self {
            dim,
            categorical,
            numerical_idx,
            ordinal_idx,
            new_numerical_idx,
            flags: TransformFlags::ALL,
        })
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The one-hot input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The categorical features `(start, cardinality)`, sorted.
    pub fn categorical_features(&self) -> &[(usize, usize)] {
        &self.categorical
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    fn out_dim(&self, d: usize) -> Result<usize> {
        if self.categorical.is_empty() {
            return Ok(d);
        }
        if d < self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "OneHotToNumeric expects {} input dimensions, got {d}",
                self.dim
            )));
        }
        Ok(self.new_numerical_idx.len() + self.categorical.len())
    }

    fn apply(&self, x: &Matrix) -> Result<Matrix> {
        if self.categorical.is_empty() {
            return Ok(x.clone());
        }
        let k = self.out_dim(x.ncols())?;
        let mut out = Matrix::from_fn(x.nrows(), k, |i, j| x[(i, j)]);
        for i in 0..x.nrows() {
            for (&new, &old) in self.new_numerical_idx.iter().zip(&self.numerical_idx) {
                out[(i, new)] = x[(i, old)];
            }
            for (c, &(s, card)) in self.categorical.iter().enumerate() {
                let mut best = 0;
                for j in 0..card {
                    let (v, b) = (x[(i, s + j)], x[(i, s + best)]);
                    if v.is_nan() && !b.is_nan() || v > b {
                        best = j;
                    }
                }
                out[(i, self.ordinal_idx[c])] = best as f64;
            }
        }
        Ok(out)
    }

    fn invert(&self, x: &Matrix) -> Result<Matrix> {
        if self.categorical.is_empty() {
            return Ok(x.clone());
        }
        let width = self.numerical_idx.len() + self.categorical.iter().map(|c| c.1).sum::<usize>();
        let mut out = Matrix::zeros(x.nrows(), width);
        for i in 0..x.nrows() {
            for (&new, &old) in self.new_numerical_idx.iter().zip(&self.numerical_idx) {
                out[(i, old)] = x[(i, new)];
            }
            for (c, &(s, card)) in self.categorical.iter().enumerate() {
                let v = x[(i, self.ordinal_idx[c])] as i64;
                if v < 0 || v as usize >= card {
                    return Err(BotorchError::InvalidArgument(format!(
                        "Class values must be smaller than num_classes ({card}), got {v}"
                    )));
                }
                out[(i, s + v as usize)] = 1.0;
            }
        }
        Ok(out)
    }
}

/// An encoder of a categorical feature (BoTorch's `encoders` of
/// `NumericToCategoricalEncoding`): the `cardinality`-dimensional encodings
/// of integer category values.
pub trait CategoricalEncoder: Send + Sync + Debug {
    /// The encodings of `values` (`n × width`).
    ///
    /// # Errors
    /// Implementation-specific.
    fn encode(&self, values: &[i64]) -> Result<Matrix>;
}

/// The one-hot encoder (`partial(one_hot, num_classes=cardinality)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotEncoder {
    /// The number of classes.
    pub num_classes: usize,
}

impl CategoricalEncoder for OneHotEncoder {
    fn encode(&self, values: &[i64]) -> Result<Matrix> {
        let mut out = Matrix::zeros(values.len(), self.num_classes);
        for (i, &v) in values.iter().enumerate() {
            if v < 0 || v as usize >= self.num_classes {
                return Err(BotorchError::InvalidArgument(format!(
                    "Class values must be smaller than num_classes ({}), got {v}",
                    self.num_classes
                )));
            }
            out[(i, v as usize)] = 1.0;
        }
        Ok(out)
    }
}

/// BoTorch's `NumericToCategoricalEncoding`: integer-encoded categoricals
/// (`index`, `cardinality`) are replaced in place by their encodings.
#[derive(Clone, Debug)]
pub struct NumericToCategoricalStep {
    dim: usize,
    categorical: Vec<(usize, usize)>,
    encoders: Vec<Arc<dyn CategoricalEncoder>>,
    numerical_idx: Vec<usize>,
    new_numerical_idx: Vec<usize>,
    encoded_start: Vec<usize>,
    flags: TransformFlags,
}

impl PartialEq for NumericToCategoricalStep {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.categorical == other.categorical
            && self.flags == other.flags
            && self.encoders.len() == other.encoders.len()
            && self
                .encoders
                .iter()
                .zip(&other.encoders)
                .all(|(a, b)| Arc::ptr_eq(a, b))
    }
}

impl NumericToCategoricalStep {
    /// `NumericToCategoricalEncoding(dim, categorical_features, encoders)`
    /// with one encoder per categorical feature (in the order of
    /// `categorical`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for BoTorch's argument errors.
    pub fn new(
        dim: usize,
        categorical: Vec<(usize, usize)>,
        encoders: Vec<Arc<dyn CategoricalEncoder>>,
    ) -> Result<Self> {
        if categorical.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "At least one categorical feature for encoding must be provided.".into(),
            ));
        }
        if categorical.len() > dim {
            return Err(BotorchError::InvalidArgument(format!(
                "The number of categorical features ({}) exceeds the provided dimension ({dim}).",
                categorical.len()
            )));
        }
        for &(idx, card) in &categorical {
            if card <= 1 {
                return Err(BotorchError::InvalidArgument(format!(
                    "Categorical feature at index {idx} has cardinality {card}. All categorical \
                     features must be an integer and have cardinality greater than 1."
                )));
            }
        }
        if encoders.len() != categorical.len() {
            return Err(BotorchError::InvalidArgument(
                "The keys of `encoders` must match the keys of `categorical_features`.".into(),
            ));
        }
        let mut pairs: Vec<((usize, usize), Arc<dyn CategoricalEncoder>)> =
            categorical.into_iter().zip(encoders).collect();
        pairs.sort_by_key(|p| p.0 .0);
        let categorical: Vec<(usize, usize)> = pairs.iter().map(|p| p.0).collect();
        let encoders = pairs.into_iter().map(|p| p.1).collect();
        let ordinal: Vec<usize> = categorical.iter().map(|c| c.0).collect();
        let numerical_idx: Vec<usize> = (0..dim).filter(|i| !ordinal.contains(i)).collect();
        let mut new_numerical_idx = Vec::new();
        let mut encoded_start = Vec::new();
        let mut offset = 0;
        for idx in 0..dim {
            if let Some(c) = categorical.iter().find(|c| c.0 == idx) {
                encoded_start.push(idx + offset);
                offset += c.1 - 1;
            } else {
                new_numerical_idx.push(idx + offset);
            }
        }
        Ok(Self {
            dim,
            categorical,
            encoders,
            numerical_idx,
            new_numerical_idx,
            encoded_start,
            flags: TransformFlags::ALL,
        })
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The numerical input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The categorical features `(index, cardinality)`, sorted.
    pub fn categorical_features(&self) -> &[(usize, usize)] {
        &self.categorical
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    fn out_dim(&self, d: usize) -> Result<usize> {
        if d != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "NumericToCategoricalEncoding expects {} input dimensions, got {d}",
                self.dim
            )));
        }
        Ok(self.numerical_idx.len() + self.categorical.iter().map(|c| c.1).sum::<usize>())
    }

    fn apply(&self, x: &Matrix) -> Result<Matrix> {
        let k = self.out_dim(x.ncols())?;
        let n = x.nrows();
        let mut out = Matrix::zeros(n, k);
        for i in 0..n {
            for (&new, &old) in self.new_numerical_idx.iter().zip(&self.numerical_idx) {
                out[(i, new)] = x[(i, old)];
            }
        }
        for (c, &(idx, card)) in self.categorical.iter().enumerate() {
            // torch's `.long()` truncates towards zero.
            let values: Vec<i64> = (0..n).map(|i| x[(i, idx)].trunc() as i64).collect();
            let enc = self.encoders[c].encode(&values)?;
            if enc.shape() != (n, card) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the encoder of the categorical feature {idx} returned {}x{} values, \
                     expected {n}x{card}",
                    enc.nrows(),
                    enc.ncols()
                )));
            }
            for i in 0..n {
                for j in 0..card {
                    out[(i, self.encoded_start[c] + j)] = enc[(i, j)];
                }
            }
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// LearnedFeatureImputation
// ----------------------------------------------------------------------------

/// BoTorch's `LearnedFeatureImputation`: inputs have `d` features and a task
/// column (last); the features a task does not observe are replaced by
/// learned values (`raw_imputation_values`, one per task and column, mapped
/// into `bounds` by a sigmoid if given). Inputs without the task column get
/// the column of `target_task`.
#[derive(Clone, Debug, PartialEq)]
pub struct ImputationStep {
    d: usize,
    task_values: Vec<i64>,
    /// `missing[t][j]`: column `j` (of `d + 1`) is imputed for task `t`.
    missing: Vec<Vec<bool>>,
    target_task: Option<i64>,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    raw: Vec<f64>,
    flags: TransformFlags,
}

impl ImputationStep {
    /// `LearnedFeatureImputation(feature_indices, d, target_task, bounds)`:
    /// `feature_indices` maps every task value to its observed columns.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for BoTorch's argument errors.
    pub fn new(
        feature_indices: Vec<(i64, Vec<usize>)>,
        d: usize,
        target_task: Option<i64>,
        bounds: Option<(Vec<f64>, Vec<f64>)>,
    ) -> Result<Self> {
        let mut fi = feature_indices;
        fi.sort_by_key(|p| p.0);
        if let Some(t) = target_task {
            if !fi.iter().any(|p| p.0 == t) {
                let keys: Vec<i64> = fi.iter().map(|p| p.0).collect();
                return Err(BotorchError::InvalidArgument(format!(
                    "target_task={t} is not a key in feature_indices. Available tasks: {keys:?}."
                )));
            }
        }
        if let Some((lo, hi)) = &bounds {
            if lo.len() != d || hi.len() != d {
                return Err(BotorchError::InvalidArgument(format!(
                    "bounds must have shape (2, {d}), got (2, {})",
                    lo.len()
                )));
            }
        }
        let mut missing = Vec::with_capacity(fi.len());
        for (t, cols) in &fi {
            if cols.contains(&d) {
                return Err(BotorchError::InvalidArgument(format!(
                    "feature_indices[{t}] contains the task column index {d}. Feature indices \
                     must not include the task column."
                )));
            }
            check_indices(cols, d + 1, "feature_indices")?;
            let mut m = vec![true; d + 1];
            m[d] = false;
            for &c in cols {
                m[c] = false;
            }
            missing.push(m);
        }
        let num = fi.len() * (d + 1);
        Ok(Self {
            d,
            task_values: fi.iter().map(|p| p.0).collect(),
            missing,
            target_task,
            bounds,
            raw: vec![0.0; num],
            flags: TransformFlags::ALL,
        })
    }

    /// Sets the transform flags.
    pub fn with_flags(mut self, flags: TransformFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Sets the raw imputation values (`num_tasks · (d + 1)`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong length.
    pub fn with_raw(mut self, raw: Vec<f64>) -> Result<Self> {
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw imputation values, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        self.raw = raw;
        Ok(self)
    }

    /// The number of features `d` (without the task column).
    pub fn dim(&self) -> usize {
        self.d
    }

    /// The sorted task values.
    pub fn task_values(&self) -> &[i64] {
        &self.task_values
    }

    /// The target task.
    pub fn target_task(&self) -> Option<i64> {
        self.target_task
    }

    /// The bounds of the imputation values.
    pub fn bounds(&self) -> Option<&(Vec<f64>, Vec<f64>)> {
        self.bounds.as_ref()
    }

    /// The raw imputation values.
    pub fn raw(&self) -> &[f64] {
        &self.raw
    }

    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        self.flags
    }

    /// The bounds `(lower, upper)` of entry `r` of the raw values (padded
    /// with `[0, 1]` for the task column, like BoTorch).
    fn entry_bounds(&self, r: usize) -> Option<(f64, f64)> {
        let (lo, hi) = self.bounds.as_ref()?;
        let j = r % (self.d + 1);
        Some(if j < self.d {
            (lo[j], hi[j])
        } else {
            (0.0, 1.0)
        })
    }

    /// The imputation values (`num_tasks · (d + 1)`) and their derivatives
    /// with respect to the raw values (GPyTorch's `Interval` maps the raw
    /// values through `lower + (upper − lower) · sigmoid(raw)`).
    pub fn imputation_values(&self) -> (Vec<f64>, Vec<f64>) {
        let mut v = Vec::with_capacity(self.raw.len());
        let mut g = Vec::with_capacity(self.raw.len());
        for (r, &raw) in self.raw.iter().enumerate() {
            match self.entry_bounds(r) {
                Some((lo, hi)) => {
                    let s = crate::constraints::sigmoid(raw);
                    v.push(s * (hi - lo) + lo);
                    g.push((hi - lo) * s * (1.0 - s));
                }
                None => {
                    v.push(raw);
                    g.push(1.0);
                }
            }
        }
        (v, g)
    }

    fn out_dim(&self, d: usize) -> Result<usize> {
        if d == self.d {
            if self.target_task.is_none() {
                return Err(BotorchError::InvalidArgument(format!(
                    "Received d-dim input (X.shape[-1]={}) but no target_task was configured. \
                     When X lacks a task column, target_task must be specified at init so the \
                     transform knows which task's imputation pattern to apply.",
                    self.d
                )));
            }
            Ok(d + 1)
        } else if d == self.d + 1 {
            Ok(d)
        } else {
            Err(BotorchError::InvalidArgument(format!(
                "Expected X.shape[-1] to be {} (no task column) or {} (with task column), got \
                 {d}.",
                self.d,
                self.d + 1
            )))
        }
    }

    fn apply(&self, x: &Matrix) -> Result<(Matrix, Vec<Option<usize>>)> {
        let k = self.out_dim(x.ncols())?;
        let n = x.nrows();
        let mut out = Matrix::from_fn(n, k, |i, j| {
            if j < x.ncols() {
                x[(i, j)]
            } else {
                self.target_task.expect("checked") as f64
            }
        });
        let (vals, _) = self.imputation_values();
        let mut tasks = Vec::with_capacity(n);
        for i in 0..n {
            let t = out[(i, self.d)];
            let pos = if t.is_nan() {
                None
            } else {
                self.task_values.binary_search(&(t.trunc() as i64)).ok()
            };
            if let Some(p) = pos {
                for j in 0..=self.d {
                    if self.missing[p][j] {
                        out[(i, j)] = vals[p * (self.d + 1) + j];
                    }
                }
            }
            tasks.push(pos);
        }
        Ok((out, tasks))
    }
}

// ----------------------------------------------------------------------------
// Step records (for the tape of a chain)
// ----------------------------------------------------------------------------

/// What the reverse pass of a new step needs (see
/// [`crate::transforms::input::InputTape`]).
#[derive(Clone, Debug)]
pub(crate) enum StepRecord {
    /// A diagonal Jacobian per output column (affine, `Log10`, rounding),
    /// constant or per point (`n × d`).
    Diagonal(DiagJac),
    /// Warp: the per-point derivatives of the warped columns.
    Warp(WarpRecordPub),
    /// AppendFeatures.
    Append(AppendRecordPub),
    /// A selection of input columns (`out[:, c] = in[:, cols[c]]`, `None`:
    /// a zero column).
    Select {
        cols: Vec<Option<usize>>,
        d_in: usize,
    },
    /// Imputation: the task position of every point.
    Impute {
        tasks: Vec<Option<usize>>,
        d_in: usize,
    },
}

/// A diagonal Jacobian.
#[derive(Clone, Debug)]
pub(crate) enum DiagJac {
    /// The same factor per column for every point.
    Constant(Vec<f64>),
    /// Factors `n × d` (the identity outside `cols`), columns `cols`.
    PerPoint(Vec<usize>, Matrix),
}

#[derive(Clone, Debug)]
pub(crate) struct WarpRecordPub(WarpRecord, Vec<usize>, Vec<f64>);

#[derive(Clone, Debug)]
pub(crate) struct AppendRecordPub(AppendRecord);

impl StepRecord {
    /// The number of output points per input point.
    pub(crate) fn expand(&self) -> usize {
        match self {
            StepRecord::Append(a) => a.0.nf,
            _ => 1,
        }
    }
}

/// The interface of the new steps used by the chain (see
/// [`crate::transforms::input`]).
#[derive(Clone, Debug, PartialEq)]
pub enum NewStep {
    /// `AffineInputTransform`, `Normalize`, `InputStandardize`.
    Affine(AffineStep),
    /// `Warp`.
    Warp(WarpStep),
    /// `Round`.
    Round(RoundStep),
    /// `Log10`.
    Log10(Log10Step),
    /// `AppendFeatures`, `InteractionFeatures`.
    Append(AppendStep),
    /// `FilterFeatures`.
    Filter(FilterStep),
    /// `OneHotToNumeric`.
    OneHotToNumeric(OneHotToNumericStep),
    /// `NumericToCategoricalEncoding`.
    NumericToCategorical(NumericToCategoricalStep),
    /// `LearnedFeatureImputation`.
    Imputation(ImputationStep),
}

impl NewStep {
    /// The transform flags.
    pub fn flags(&self) -> TransformFlags {
        match self {
            NewStep::Affine(s) => s.flags,
            NewStep::Warp(s) => s.flags,
            NewStep::Round(s) => s.flags,
            NewStep::Log10(s) => s.flags,
            NewStep::Append(s) => s.flags,
            NewStep::Filter(s) => s.flags,
            NewStep::OneHotToNumeric(s) => s.flags,
            NewStep::NumericToCategorical(s) => s.flags,
            NewStep::Imputation(s) => s.flags,
        }
    }

    /// BoTorch's class name.
    pub fn name(&self) -> &'static str {
        match self {
            NewStep::Affine(s) => match s.kind {
                AffineKind::Normalize { .. } => "Normalize",
                AffineKind::Standardize { .. } => "InputStandardize",
                AffineKind::Fixed => "AffineInputTransform",
            },
            NewStep::Warp(_) => "Warp",
            NewStep::Round(_) => "Round",
            NewStep::Log10(_) => "Log10",
            NewStep::Append(s) => match s.source {
                FeatureSource::Interaction(_) => "InteractionFeatures",
                _ => "AppendFeatures",
            },
            NewStep::Filter(_) => "FilterFeatures",
            NewStep::OneHotToNumeric(_) => "OneHotToNumeric",
            NewStep::NumericToCategorical(_) => "NumericToCategoricalEncoding",
            NewStep::Imputation(_) => "LearnedFeatureImputation",
        }
    }

    /// Whether the step learns coefficients from the training inputs in
    /// train mode (`Normalize` / `InputStandardize` with learned
    /// coefficients, `Warp` with learned bounds).
    pub fn learns_coefficients(&self) -> bool {
        match self {
            // BoTorch learns in `_transform`, which a `reverse` transform's
            // forward pass does not call.
            NewStep::Affine(a) => a.learn && a.kind != AffineKind::Fixed && !a.reverse,
            NewStep::Warp(w) => w.normalize.learn,
            _ => false,
        }
    }

    /// The number of learned parameters (fitted with the GP).
    pub fn num_params(&self) -> usize {
        match self {
            NewStep::Warp(w) => 2 * w.indices.len(),
            NewStep::Imputation(s) => s.raw.len(),
            _ => 0,
        }
    }

    /// The learned parameters (Warp: `concentration0` then
    /// `concentration1`; imputation: the raw values).
    pub fn params(&self) -> Vec<f64> {
        match self {
            NewStep::Warp(w) => w.c0.iter().chain(&w.c1).copied().collect(),
            NewStep::Imputation(s) => s.raw.clone(),
            _ => Vec::new(),
        }
    }

    /// Sets the learned parameters.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of values.
    pub fn set_params(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.num_params() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} parameters of {}, got {}",
                self.num_params(),
                self.name(),
                values.len()
            )));
        }
        match self {
            NewStep::Warp(w) => {
                let k = w.indices.len();
                w.c0 = values[..k].to_vec();
                w.c1 = values[k..].to_vec();
            }
            NewStep::Imputation(s) => s.raw = values.to_vec(),
            _ => {}
        }
        Ok(())
    }

    /// The names (relative to the step) and shapes of the learned
    /// parameters, in order.
    pub fn param_shapes(&self) -> Vec<(String, Vec<usize>)> {
        match self {
            NewStep::Warp(w) => {
                let k = w.indices.len();
                let shape = if w.batched { vec![1, k] } else { vec![k] };
                vec![
                    ("concentration0".into(), shape.clone()),
                    ("concentration1".into(), shape),
                ]
            }
            NewStep::Imputation(s) => vec![("raw_imputation_values".into(), vec![s.raw.len()])],
            _ => Vec::new(),
        }
    }

    /// The optimizer bounds of the learned parameters.
    pub fn param_bounds(&self) -> Vec<(f64, f64)> {
        match self {
            NewStep::Warp(w) => vec![WarpStep::constraint().raw_bounds(); 2 * w.indices.len()],
            NewStep::Imputation(s) => vec![(f64::NEG_INFINITY, f64::INFINITY); s.raw.len()],
            _ => Vec::new(),
        }
    }

    /// The log prior density of the learned parameters and its gradient.
    pub fn log_prior_and_grad(&self) -> (f64, Vec<f64>) {
        match self {
            NewStep::Warp(w) => w.log_prior_and_grad(),
            _ => (0.0, vec![0.0; self.num_params()]),
        }
    }

    /// The priors of the learned parameters, one per parameter (for
    /// BoTorch's `sample_all_priors`).
    pub fn param_priors(&self) -> Vec<Option<Prior>> {
        match self {
            NewStep::Warp(w) => {
                let k = w.indices.len();
                let mut out = vec![w.c0_prior; k];
                out.extend(std::iter::repeat_n(w.c1_prior, k));
                out
            }
            _ => vec![None; self.num_params()],
        }
    }

    /// The step with its current coefficients, never refitted.
    pub fn frozen(&self) -> NewStep {
        match self {
            NewStep::Affine(a) => NewStep::Affine(a.clone().frozen()),
            NewStep::Warp(w) => NewStep::Warp(w.clone().frozen()),
            other => other.clone(),
        }
    }

    /// Learns the coefficients from the inputs (train mode).
    ///
    /// # Errors
    /// Dimension errors.
    pub fn fit(&mut self, x: &Matrix) -> Result<()> {
        match self {
            NewStep::Affine(a) if a.reverse => Ok(()),
            NewStep::Affine(a) => a.fit(x),
            NewStep::Warp(w) => w.fit(x),
            _ => Ok(()),
        }
    }

    /// Determines the sizes of callable features from one point.
    ///
    /// # Errors
    /// Errors of the callable.
    pub fn probe(&mut self, x: &Matrix) -> Result<()> {
        match self {
            NewStep::Append(a) => a.probe(x),
            _ => Ok(()),
        }
    }

    /// The output dimension for `d` input dimensions.
    ///
    /// # Errors
    /// Dimension errors.
    pub fn out_dim(&self, d: usize) -> Result<usize> {
        match self {
            NewStep::Affine(a) => {
                if d != a.d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "Wrong input dimension. Received {d}, expected {}.",
                        a.d
                    )));
                }
                Ok(d)
            }
            NewStep::Warp(w) => {
                check_indices(&w.indices, d, "indices")?;
                if d != w.normalize.d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "Wrong input dimension. Received {d}, expected {}.",
                        w.normalize.d
                    )));
                }
                Ok(d)
            }
            NewStep::Round(r) => {
                r.check(d)?;
                Ok(d)
            }
            NewStep::Log10(l) => {
                check_indices(&l.indices, d, "indices")?;
                Ok(d)
            }
            NewStep::Append(a) => a.out_dim(d),
            NewStep::Filter(f) => {
                check_indices(&f.feature_indices, d, "feature_indices")?;
                Ok(f.feature_indices.len())
            }
            NewStep::OneHotToNumeric(o) => o.out_dim(d),
            NewStep::NumericToCategorical(c) => c.out_dim(d),
            NewStep::Imputation(s) => s.out_dim(d),
        }
    }

    /// The number of points per input point (`n_f` of `AppendFeatures`).
    pub fn expand(&self) -> Option<usize> {
        match self {
            NewStep::Append(a) => a.expand(),
            _ => Some(1),
        }
    }

    /// Applies the step (the forward transform; `reverse` steps
    /// un-transform) with the record of its derivatives.
    pub(crate) fn apply(&self, x: &Matrix) -> Result<(Matrix, StepRecord)> {
        let d_in = x.ncols();
        match self {
            NewStep::Affine(a) => Ok((
                a.apply(x)?,
                StepRecord::Diagonal(DiagJac::Constant(a.diag_jacobian())),
            )),
            NewStep::Warp(w) => {
                if w.reverse {
                    // The inverse warp is not differentiated (BoTorch's
                    // `reverse=True` is for un-normalized search spaces).
                    let out = w.unwarp(x)?;
                    return Ok((
                        out,
                        StepRecord::Diagonal(DiagJac::Constant(vec![f64::NAN; d_in])),
                    ));
                }
                let (out, rec) = w.warp(x)?;
                Ok((
                    out,
                    StepRecord::Warp(WarpRecordPub(
                        rec,
                        w.indices.clone(),
                        w.normalize.diag_jacobian(),
                    )),
                ))
            }
            NewStep::Round(r) => {
                let (out, dint) = r.apply(x)?;
                Ok((
                    out,
                    StepRecord::Diagonal(DiagJac::PerPoint(r.integer_indices.clone(), dint)),
                ))
            }
            NewStep::Log10(l) => {
                let (out, der) = l.map(x, l.reverse)?;
                Ok((
                    out,
                    StepRecord::Diagonal(DiagJac::PerPoint(l.indices.clone(), der)),
                ))
            }
            NewStep::Append(a) => {
                let (out, rec) = a.apply(x)?;
                Ok((out, StepRecord::Append(AppendRecordPub(rec))))
            }
            NewStep::Filter(f) => {
                self.out_dim(d_in)?;
                let cols = &f.feature_indices;
                Ok((
                    Matrix::from_fn(x.nrows(), cols.len(), |i, c| x[(i, cols[c])]),
                    StepRecord::Select {
                        cols: cols.iter().map(|&c| Some(c)).collect(),
                        d_in,
                    },
                ))
            }
            NewStep::OneHotToNumeric(o) => {
                let out = o.apply(x)?;
                let k = out.ncols();
                let mut cols: Vec<Option<usize>> = (0..k)
                    .map(|j| o.categorical.is_empty().then_some(j))
                    .collect();
                if !o.categorical.is_empty() {
                    for (&new, &old) in o.new_numerical_idx.iter().zip(&o.numerical_idx) {
                        cols[new] = Some(old);
                    }
                }
                Ok((out, StepRecord::Select { cols, d_in }))
            }
            NewStep::NumericToCategorical(c) => {
                let out = c.apply(x)?;
                let mut cols = vec![None; out.ncols()];
                for (&new, &old) in c.new_numerical_idx.iter().zip(&c.numerical_idx) {
                    cols[new] = Some(old);
                }
                Ok((out, StepRecord::Select { cols, d_in }))
            }
            NewStep::Imputation(s) => {
                let (out, tasks) = s.apply(x)?;
                Ok((out, StepRecord::Impute { tasks, d_in }))
            }
        }
    }

    /// Applies the step (BoTorch's `transform`: the forward transform, or
    /// the inverse of a `reverse` step).
    ///
    /// # Errors
    /// Dimension errors and errors of callables.
    pub fn transform(&self, x: &Matrix) -> Result<Matrix> {
        Ok(self.apply(x)?.0)
    }

    /// The inverse (BoTorch's `untransform`), where defined.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] (`NotImplementedError`) for transforms
    /// without an `untransform`.
    pub fn untransform(&self, x: &Matrix) -> Result<Matrix> {
        match self {
            NewStep::Affine(a) => a.invert(x),
            NewStep::Warp(w) => {
                if w.reverse {
                    Ok(w.warp(x)?.0)
                } else {
                    w.unwarp(x)
                }
            }
            NewStep::Log10(l) => Ok(l.map(x, !l.reverse)?.0),
            NewStep::OneHotToNumeric(o) => o.invert(x),
            _ => Err(BotorchError::Unsupported(format!(
                "{} does not implement the `untransform` method.",
                self.name()
            ))),
        }
    }

    /// The reverse pass: the adjoint of the input rows `start .. start +
    /// n_in` of the step from the adjoint `bar` of their output rows
    /// (`n_in·expand × d_out`), accumulating the adjoint of the learned
    /// parameters in `pbar` if given.
    pub(crate) fn backward(
        &self,
        rec: &StepRecord,
        start: usize,
        n_in: usize,
        bar: &Matrix,
        pbar: Option<&mut [f64]>,
    ) -> Result<Matrix> {
        match rec {
            StepRecord::Diagonal(DiagJac::Constant(f)) => {
                Ok(Matrix::from_fn(n_in, bar.ncols(), |i, j| {
                    bar[(i, j)] * f[j]
                }))
            }
            StepRecord::Diagonal(DiagJac::PerPoint(cols, m)) => {
                let mut out = bar.clone();
                for (c, &j) in cols.iter().enumerate() {
                    for i in 0..n_in {
                        out[(i, j)] = bar[(i, j)] * m[(start + i, c)];
                    }
                }
                Ok(out)
            }
            StepRecord::Warp(WarpRecordPub(r, cols, norm)) => {
                let mut out = Matrix::from_fn(n_in, bar.ncols(), |i, j| bar[(i, j)] * norm[j]);
                for (c, &j) in cols.iter().enumerate() {
                    for i in 0..n_in {
                        out[(i, j)] = bar[(i, j)] * r.dx[(start + i, c)];
                    }
                }
                if let Some(pb) = pbar {
                    let k = cols.len();
                    for (c, &j) in cols.iter().enumerate() {
                        for i in 0..n_in {
                            pb[c] += bar[(i, j)] * r.dc0[(start + i, c)];
                            pb[k + c] += bar[(i, j)] * r.dc1[(start + i, c)];
                        }
                    }
                }
                Ok(out)
            }
            StepRecord::Append(AppendRecordPub(r)) => {
                let NewStep::Append(a) = self else {
                    unreachable!("append records belong to append steps")
                };
                a.backward(r, start, n_in, bar)
            }
            StepRecord::Select { cols, d_in } => {
                let mut out = Matrix::zeros(n_in, *d_in);
                for (c, src) in cols.iter().enumerate() {
                    if let Some(j) = src {
                        for i in 0..n_in {
                            out[(i, *j)] += bar[(i, c)];
                        }
                    }
                }
                Ok(out)
            }
            StepRecord::Impute { tasks, d_in } => {
                let NewStep::Imputation(s) = self else {
                    unreachable!("imputation records belong to imputation steps")
                };
                let mut out = Matrix::zeros(n_in, *d_in);
                let (_, dv) = s.imputation_values();
                let mut pbar = pbar;
                for i in 0..n_in {
                    let task = tasks[start + i];
                    for j in 0..bar.ncols() {
                        let imputed = task.is_some_and(|p| s.missing[p][j]);
                        if imputed {
                            if let (Some(pb), Some(p)) = (pbar.as_deref_mut(), task) {
                                let r = p * (s.d + 1) + j;
                                pb[r] += bar[(i, j)] * dv[r];
                            }
                        } else if j < *d_in {
                            out[(i, j)] += bar[(i, j)];
                        }
                    }
                }
                Ok(out)
            }
        }
    }
}
