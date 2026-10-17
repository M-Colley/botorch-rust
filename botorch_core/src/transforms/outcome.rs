//! Outcome transforms beyond a plain `Standardize`, mirroring
//! `botorch.models.transforms.outcome`: `Log`, `Power`, `Bilog`,
//! `StratifiedStandardize` and `ChainedOutcomeTransform`, together with the
//! un-transformation of posteriors (BoTorch's `untransform_posterior` and
//! `TransformedPosterior`).
//!
//! # Per-output chains
//!
//! Models keep one single-output GP per output and batch element (see
//! [`crate::BatchedMultiOutputGP`]), so an outcome transform is stored as an
//! [`OutcomeChain`] per element: the steps of BoTorch's
//! `ChainedOutcomeTransform` (or the single transform) that apply to that
//! output (BoTorch's `outputs` argument selects the transformed outputs; the
//! other outputs pass through unchanged). In train mode a chain fits the
//! statistics of its standardizations to the training targets as transformed
//! by the preceding steps ([`OutcomeChain::fit_transform`]); in eval mode it
//! transforms new observations with the fitted statistics
//! ([`OutcomeChain::transform`], used by conditioning).
//!
//! # Posteriors
//!
//! The posterior of the GP in the transformed space is Gaussian. BoTorch
//! un-transforms it step by step in reverse order: as long as the steps are
//! affine (`Standardize`, `StratifiedStandardize`) the result stays a
//! `GPyTorchPosterior` with a rescaled mean and covariance; the first
//! nonlinear step (`Log`, `Power`, `Bilog`) turns it into a
//! `TransformedPosterior`, and every later step (also affine ones) wraps it
//! further. [`OutcomeChain::posterior_parts`] returns exactly that split: the
//! folded affine map of the leading affine steps ([`PointAffine`], per point
//! for `StratifiedStandardize`) and the elementwise [`SampleStep`]s of the
//! rest, which [`SampleTransform`] applies to posterior samples
//! (`rsample_from_base_samples` of the `TransformedPosterior`) with their
//! derivatives, and to the marginal moments where BoTorch defines them
//! (`Log`: [`crate::transforms::utils::norm_to_lognorm_mean`] /
//! `_variance` of the clamped variance; affine steps: the rescaled moments;
//! `Power` and `Bilog` have none, BoTorch's `NotImplementedError`).

use crate::error::{BotorchError, Result};
use crate::posterior::MIN_VARIANCE;
use crate::transforms::utils::{nanmean, nanstd, norm_to_lognorm_mean, norm_to_lognorm_variance};
use crate::transforms::{Standardize, STANDARDIZE_MIN_STDV};
use crate::Matrix;

/// The lower clamp of the targets in `Log`'s delta-method noise transform
/// (`Var[log Y] ≈ Var[Y] / max(Y, 1e-8)²`).
pub const LOG_NOISE_CLAMP: f64 = 1e-8;

/// torch's `sign` (0 at 0, NaN for NaN).
#[inline]
pub(crate) fn torch_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// BoTorch's `StratifiedStandardize`: a standardization of a single output
/// per stratum, where the stratum of a point is the (truncated) integer
/// value of its input `stratification_idx` (of the transformed inputs, as
/// BoTorch passes them).
///
/// Strata in `task_values` without training data, and values outside
/// `task_values`, are not standardized (mean 0, standard deviation 1;
/// BoTorch warns for the latter).
#[derive(Clone, Debug, PartialEq)]
pub struct StratifiedStandardize {
    stratification_idx: usize,
    task_values: Vec<i64>,
    means: Vec<f64>,
    stdvs: Vec<f64>,
    min_stdv: f64,
    trained: bool,
}

impl StratifiedStandardize {
    /// An untrained transform for the strata `all_task_values` (sorted and
    /// de-duplicated, like BoTorch's `all_task_values.unique(sorted=True)`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without task values.
    pub fn new(stratification_idx: usize, all_task_values: &[i64], min_stdv: f64) -> Result<Self> {
        let mut values = all_task_values.to_vec();
        values.sort_unstable();
        values.dedup();
        if values.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "all_task_values must contain at least one task value".into(),
            ));
        }
        let t = values.len();
        Ok(Self {
            stratification_idx,
            task_values: values,
            means: vec![0.0; t],
            stdvs: vec![1.0; t],
            min_stdv,
            trained: false,
        })
    }

    /// The input column that holds the stratum.
    pub fn stratification_idx(&self) -> usize {
        self.stratification_idx
    }

    /// The sorted task values.
    pub fn task_values(&self) -> &[i64] {
        &self.task_values
    }

    /// The means per stratum (in the order of [`Self::task_values`]).
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// The standard deviations per stratum.
    pub fn stdvs(&self) -> &[f64] {
        &self.stdvs
    }

    /// The `min_stdv` setting.
    pub fn min_stdv(&self) -> f64 {
        self.min_stdv
    }

    /// Whether the statistics have been fitted.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// A fitted transform with the given statistics (e.g. restored from a
    /// copy).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the lengths differ from the
    /// number of task values.
    pub fn with_stats(mut self, means: Vec<f64>, stdvs: Vec<f64>) -> Result<Self> {
        if means.len() != self.task_values.len() || stdvs.len() != self.task_values.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} means and standard deviations, got {} and {}",
                self.task_values.len(),
                means.len(),
                stdvs.len()
            )));
        }
        self.means = means;
        self.stdvs = stdvs;
        self.trained = true;
        Ok(self)
    }

    /// The stratum (position in [`Self::task_values`]) of a point, `None`
    /// for an unknown task value.
    fn stratum(&self, x: &Matrix, i: usize) -> Result<Option<usize>> {
        if self.stratification_idx >= x.ncols() {
            return Err(BotorchError::DimensionMismatch(format!(
                "index {} is out of bounds for dimension with size {}",
                self.stratification_idx,
                x.ncols()
            )));
        }
        let v = x[(i, self.stratification_idx)];
        if v.is_nan() {
            return Ok(None);
        }
        // torch's `.long()` truncates towards zero.
        let s = v.trunc() as i64;
        Ok(self.task_values.binary_search(&s).ok())
    }

    /// The mean and standard deviation of every point of `x`.
    fn per_point(&self, x: &Matrix) -> Result<Vec<(f64, f64)>> {
        (0..x.nrows())
            .map(|i| {
                Ok(match self.stratum(x, i)? {
                    Some(s) => (self.means[s], self.stdvs[s]),
                    None => (0.0, 1.0),
                })
            })
            .collect()
    }

    fn fit(&mut self, y: &Matrix, x: &Matrix) -> Result<()> {
        let n = y.nrows();
        if n < 1 {
            return Err(BotorchError::InputData(format!(
                "Can't standardize with no observations. Y has shape {n}x1."
            )));
        }
        if x.nrows() != n {
            return Err(BotorchError::DimensionMismatch(format!(
                "X has {} rows but Y has {n}",
                x.nrows()
            )));
        }
        let strata = (0..n)
            .map(|i| self.stratum(x, i))
            .collect::<Result<Vec<_>>>()?;
        for s in 0..self.task_values.len() {
            if !strata.contains(&Some(s)) {
                continue;
            }
            let col: Vec<f64> = (0..n)
                .map(|i| {
                    if strata[i] == Some(s) {
                        y[(i, 0)]
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let std = if n == 1 { 1.0 } else { nanstd(&col) };
            self.stdvs[s] = if std >= self.min_stdv { std } else { 1.0 };
            self.means[s] = nanmean(&col);
        }
        self.trained = true;
        Ok(())
    }
}

/// A step of an outcome transform chain (for one output).
#[derive(Clone, Debug, PartialEq)]
pub enum OutcomeStep {
    /// BoTorch's `Standardize`.
    Standardize(Standardize),
    /// BoTorch's `StratifiedStandardize`.
    StratifiedStandardize(StratifiedStandardize),
    /// BoTorch's `Log`.
    Log,
    /// BoTorch's `Power(power)`.
    Power(f64),
    /// BoTorch's `Bilog`.
    Bilog,
}

fn noise_unsupported(name: &str) -> BotorchError {
    BotorchError::Unsupported(format!(
        "{name} does not yet support transforming observation noise"
    ))
}

fn check_y(y: &Matrix, yvar: Option<&Matrix>) -> Result<()> {
    if y.ncols() != 1 {
        return Err(BotorchError::DimensionMismatch(format!(
            "an outcome chain transforms one output, got {} columns",
            y.ncols()
        )));
    }
    if let Some(v) = yvar {
        if v.shape() != y.shape() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Yvar has shape {:?} but Y has shape {:?}",
                v.shape(),
                y.shape()
            )));
        }
    }
    Ok(())
}

impl OutcomeStep {
    /// Whether the step is affine (BoTorch's `_is_linear`).
    pub fn is_linear(&self) -> bool {
        matches!(
            self,
            OutcomeStep::Standardize(_) | OutcomeStep::StratifiedStandardize(_)
        )
    }

    /// BoTorch's class name of the step.
    pub fn name(&self) -> &'static str {
        match self {
            OutcomeStep::Standardize(_) => "Standardize",
            OutcomeStep::StratifiedStandardize(_) => "StratifiedStandardize",
            OutcomeStep::Log => "Log",
            OutcomeStep::Power(_) => "Power",
            OutcomeStep::Bilog => "Bilog",
        }
    }

    /// The step in train mode (`forward` of a module in train mode): fits
    /// the statistics of a standardization to `y`, then transforms `y` and
    /// `yvar` (`x`: the transformed inputs of the points).
    ///
    /// # Errors
    /// See [`OutcomeStep::transform`].
    pub fn fit_transform(
        &mut self,
        y: &Matrix,
        yvar: Option<&Matrix>,
        x: &Matrix,
    ) -> Result<(Matrix, Option<Matrix>)> {
        check_y(y, yvar)?;
        match self {
            OutcomeStep::Standardize(st) => st.fit_transform(y, yvar),
            OutcomeStep::StratifiedStandardize(st) => {
                st.fit(y, x)?;
                self.transform(y, yvar, x)
            }
            _ => self.transform(y, yvar, x),
        }
    }

    /// The step in eval mode: transforms `y` (`n × 1`) and `yvar` with the
    /// fitted statistics.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for observation noise with `Power` and
    /// `Bilog` (BoTorch's `NotImplementedError`), errors of an untrained
    /// standardization and shape mismatches.
    pub fn transform(
        &self,
        y: &Matrix,
        yvar: Option<&Matrix>,
        x: &Matrix,
    ) -> Result<(Matrix, Option<Matrix>)> {
        check_y(y, yvar)?;
        match self {
            OutcomeStep::Standardize(st) => st.transform(y, yvar),
            OutcomeStep::StratifiedStandardize(st) => {
                let pp = st.per_point(x)?;
                if pp.len() != y.nrows() {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "X has {} rows but Y has {}",
                        pp.len(),
                        y.nrows()
                    )));
                }
                let y_tf = Matrix::from_fn(y.nrows(), 1, |i, _| (y[(i, 0)] - pp[i].0) / pp[i].1);
                let v_tf = yvar
                    .map(|v| Matrix::from_fn(v.nrows(), 1, |i, _| v[(i, 0)] / (pp[i].1 * pp[i].1)));
                Ok((y_tf, v_tf))
            }
            OutcomeStep::Log => {
                let y_tf = y.map(f64::ln);
                let v_tf = yvar.map(|v| {
                    Matrix::from_fn(v.nrows(), 1, |i, _| {
                        let c = y[(i, 0)];
                        let c = if c.is_nan() {
                            c
                        } else {
                            c.max(LOG_NOISE_CLAMP)
                        };
                        v[(i, 0)] / (c * c)
                    })
                });
                Ok((y_tf, v_tf))
            }
            OutcomeStep::Power(p) => {
                if yvar.is_some() {
                    return Err(noise_unsupported("Power"));
                }
                Ok((y.map(|v| v.powf(*p)), None))
            }
            OutcomeStep::Bilog => {
                if yvar.is_some() {
                    return Err(noise_unsupported("Bilog"));
                }
                Ok((y.map(|v| torch_sign(v) * (v.abs() + 1.0).ln()), None))
            }
        }
    }

    /// Inverts the step (BoTorch's `untransform`).
    ///
    /// # Errors
    /// See [`OutcomeStep::transform`]; [`BotorchError::Runtime`] for an
    /// untrained standardization.
    pub fn untransform(
        &self,
        y: &Matrix,
        yvar: Option<&Matrix>,
        x: &Matrix,
    ) -> Result<(Matrix, Option<Matrix>)> {
        check_y(y, yvar)?;
        match self {
            OutcomeStep::Standardize(st) => {
                if !st.is_trained() {
                    return Err(BotorchError::Runtime(
                        "`Standardize` transforms must be called on outcome data (e.g. \
                         `transform(Y)`) before calling `untransform`, since means and standard \
                         deviations need to be computed."
                            .into(),
                    ));
                }
                st.untransform(y, yvar)
            }
            OutcomeStep::StratifiedStandardize(st) => {
                let pp = st.per_point(x)?;
                if pp.len() != y.nrows() {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "X has {} rows but Y has {}",
                        pp.len(),
                        y.nrows()
                    )));
                }
                let y_u = Matrix::from_fn(y.nrows(), 1, |i, _| pp[i].0 + pp[i].1 * y[(i, 0)]);
                let v_u = yvar
                    .map(|v| Matrix::from_fn(v.nrows(), 1, |i, _| (pp[i].1 * pp[i].1) * v[(i, 0)]));
                Ok((y_u, v_u))
            }
            OutcomeStep::Log => {
                let y_u = y.map(f64::exp);
                let v_u = yvar.map(|v| {
                    Matrix::from_fn(v.nrows(), 1, |i, _| v[(i, 0)] * (2.0 * y[(i, 0)]).exp())
                });
                Ok((y_u, v_u))
            }
            OutcomeStep::Power(p) => {
                if yvar.is_some() {
                    return Err(noise_unsupported("Power"));
                }
                let e = 1.0 / p;
                Ok((y.map(|v| v.powf(e)), None))
            }
            OutcomeStep::Bilog => {
                if yvar.is_some() {
                    return Err(noise_unsupported("Bilog"));
                }
                Ok((y.map(|v| torch_sign(v) * v.abs().exp_m1()), None))
            }
        }
    }
}

/// The affine map `y = offset + scale · f` of the posterior of every point
/// (the folded affine un-transforms; constant unless a
/// `StratifiedStandardize` is involved).
#[derive(Clone, Debug, PartialEq)]
pub enum PointAffine {
    /// The same map for every point.
    Constant(f64, f64),
    /// One `(offset, scale)` per point.
    PerPoint(Vec<(f64, f64)>),
}

impl PointAffine {
    /// The identity.
    pub const IDENTITY: PointAffine = PointAffine::Constant(0.0, 1.0);

    /// `(offset, scale)` of point `i`.
    #[inline]
    pub fn at(&self, i: usize) -> (f64, f64) {
        match self {
            PointAffine::Constant(o, s) => (*o, *s),
            PointAffine::PerPoint(v) => v[i],
        }
    }

    /// The scale of point `i`.
    #[inline]
    pub fn scale(&self, i: usize) -> f64 {
        self.at(i).1
    }

    /// The offset of point `i`.
    #[inline]
    pub fn offset(&self, i: usize) -> f64 {
        self.at(i).0
    }

    /// Whether the map is the same for every point.
    pub fn is_constant(&self) -> bool {
        matches!(self, PointAffine::Constant(..))
    }

    /// `outer ∘ self` (first `self`, then `outer`).
    fn then(&self, outer: &PointAffine, n: usize) -> PointAffine {
        match (self, outer) {
            (PointAffine::Constant(o1, s1), PointAffine::Constant(o2, s2)) => {
                PointAffine::Constant(o2 + s2 * o1, s2 * s1)
            }
            _ => PointAffine::PerPoint(
                (0..n)
                    .map(|i| {
                        let (o1, s1) = self.at(i);
                        let (o2, s2) = outer.at(i);
                        (o2 + s2 * o1, s2 * s1)
                    })
                    .collect(),
            ),
        }
    }
}

/// An elementwise map of posterior samples (one un-transform of a
/// `TransformedPosterior`).
#[derive(Clone, Debug, PartialEq)]
pub enum SampleStep {
    /// `offset + scale · s` (a standardization after a nonlinear step), per
    /// point.
    Affine(PointAffine),
    /// `exp(s)` (`Log`).
    Exp,
    /// `s^exponent` (`Power(p)`: `exponent = 1 / p`).
    Power(f64),
    /// `sign(s) · expm1(|s|)` (`Bilog`).
    Bilog,
}

impl SampleStep {
    /// The value at a sample `s` of point `i`.
    #[inline]
    pub fn apply(&self, s: f64, i: usize) -> f64 {
        match self {
            SampleStep::Affine(a) => {
                let (o, c) = a.at(i);
                o + c * s
            }
            SampleStep::Exp => s.exp(),
            SampleStep::Power(e) => s.powf(*e),
            SampleStep::Bilog => torch_sign(s) * s.abs().exp_m1(),
        }
    }

    /// The derivative at a sample `s` of point `i` (torch's autograd: `sign`
    /// and `abs` have zero / `sign` derivatives at 0).
    #[inline]
    pub fn derivative(&self, s: f64, i: usize) -> f64 {
        match self {
            SampleStep::Affine(a) => a.scale(i),
            SampleStep::Exp => s.exp(),
            SampleStep::Power(e) => e * s.powf(e - 1.0),
            SampleStep::Bilog => {
                let g = torch_sign(s);
                g * g * s.abs().exp()
            }
        }
    }

    /// BoTorch's class name of the un-transformed outcome transform (for
    /// error messages).
    fn has_moments(&self) -> bool {
        matches!(self, SampleStep::Affine(_) | SampleStep::Exp)
    }
}

/// The marginal moments of a `TransformedPosterior` and their derivatives
/// with respect to the Gaussian base mean and *unclamped* variance:
/// `(mean, variance, ∂mean/∂μ, ∂mean/∂σ², ∂var/∂μ, ∂var/∂σ²)`.
pub type MomentsWithGrad = (f64, f64, f64, f64, f64, f64);

/// The elementwise sample maps of a `TransformedPosterior` over `q` points
/// and `m` outputs (output-major, like [`crate::Posterior`]): the steps of
/// every output, applied in order (an empty list leaves an output Gaussian).
#[derive(Clone, Debug, PartialEq)]
pub struct SampleTransform {
    steps: Vec<Vec<SampleStep>>,
    num_points: usize,
}

/// BoTorch's error for the moments of a `TransformedPosterior` without a
/// mean (or variance) transform.
pub fn no_moment_transform(what: &str) -> BotorchError {
    BotorchError::Unsupported(format!("No {what} transform provided."))
}

impl SampleTransform {
    /// A transform with the given steps per output over `num_points` points
    /// (`None` if no output has a step).
    pub fn new(steps: Vec<Vec<SampleStep>>, num_points: usize) -> Option<Self> {
        steps
            .iter()
            .any(|s| !s.is_empty())
            .then_some(Self { steps, num_points })
    }

    /// Number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.steps.len()
    }

    /// Number of points.
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// The steps of output `t`.
    pub fn steps(&self, t: usize) -> &[SampleStep] {
        &self.steps[t]
    }

    /// Whether output `t` is transformed.
    pub fn is_transformed(&self, t: usize) -> bool {
        !self.steps[t].is_empty()
    }

    /// The transformed value of a sample `s` of output `t` at point `p`.
    #[inline]
    pub fn apply(&self, t: usize, p: usize, s: f64) -> f64 {
        self.steps[t].iter().fold(s, |v, step| step.apply(v, p))
    }

    /// The transformed value of a sample and its derivative.
    #[inline]
    pub fn apply_with_derivative(&self, t: usize, p: usize, s: f64) -> (f64, f64) {
        let mut v = s;
        let mut d = 1.0;
        for step in &self.steps[t] {
            d *= step.derivative(v, p);
            v = step.apply(v, p);
        }
        (v, d)
    }

    /// The transform of the outputs `outputs` (in that order).
    pub fn select_outputs(&self, outputs: &[usize]) -> Option<SampleTransform> {
        Self::new(
            outputs.iter().map(|&t| self.steps[t].clone()).collect(),
            self.num_points,
        )
    }

    /// The transform of independent posteriors stacked output-wise (each
    /// `None` part contributes untransformed outputs).
    pub fn concat(parts: &[(Option<&SampleTransform>, usize)], num_points: usize) -> Option<Self> {
        let mut steps = Vec::new();
        for (part, m) in parts {
            match part {
                Some(p) => steps.extend(p.steps.iter().cloned()),
                None => steps.extend(std::iter::repeat_n(Vec::new(), *m)),
            }
        }
        Self::new(steps, num_points)
    }

    /// Whether the moments of output `t` are defined (all its steps are
    /// affine or `exp`).
    pub fn has_moments(&self, t: usize) -> bool {
        self.steps[t].iter().all(SampleStep::has_moments)
    }

    /// The marginal mean and variance of output `t` at point `p` of the
    /// `TransformedPosterior` (BoTorch's `posterior.mean` / `.variance`)
    /// from the Gaussian base mean `mu` and *unclamped* variance `var`
    /// (clamped at [`MIN_VARIANCE`] like the base `posterior.variance`),
    /// with the derivatives of [`MomentsWithGrad`].
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] (`NotImplementedError`) if a step has
    /// no moment transform (`Power`, `Bilog`).
    pub fn moments(&self, t: usize, p: usize, mu: f64, var: f64) -> Result<MomentsWithGrad> {
        if !self.has_moments(t) {
            return Err(no_moment_transform("mean"));
        }
        // The base posterior's variance is clamped (zero derivative there).
        let (mut v, mut dv_dvar) = if var < MIN_VARIANCE {
            (MIN_VARIANCE, 0.0)
        } else {
            (var, 1.0)
        };
        let mut m = mu;
        // Derivatives of (m, v) with respect to (μ, σ²).
        let (mut dm_dmu, mut dm_dvar, mut dv_dmu) = (1.0, 0.0, 0.0);
        for step in &self.steps[t] {
            match step {
                SampleStep::Affine(a) => {
                    let (o, s) = a.at(p);
                    m = o + s * m;
                    v *= s * s;
                    dm_dmu *= s;
                    dm_dvar *= s;
                    dv_dmu *= s * s;
                    dv_dvar *= s * s;
                }
                SampleStep::Exp => {
                    let nm = norm_to_lognorm_mean(m, v);
                    let nv = norm_to_lognorm_variance(m, v);
                    // ∂nm/∂m = nm, ∂nm/∂v = nm / 2;
                    // ∂nv/∂m = 2 nv, ∂nv/∂v = exp(v) exp(2m + v) + nv.
                    let e2b = (2.0 * (m + 0.5 * v)).exp();
                    let a_m = nm;
                    let a_v = 0.5 * nm;
                    let b_m = 2.0 * nv;
                    let b_v = v.exp() * e2b + nv;
                    let (pm_mu, pm_var, pv_mu, pv_var) = (dm_dmu, dm_dvar, dv_dmu, dv_dvar);
                    dm_dmu = a_m * pm_mu + a_v * pv_mu;
                    dm_dvar = a_m * pm_var + a_v * pv_var;
                    dv_dmu = b_m * pm_mu + b_v * pv_mu;
                    dv_dvar = b_m * pm_var + b_v * pv_var;
                    m = nm;
                    v = nv;
                }
                SampleStep::Power(_) | SampleStep::Bilog => unreachable!("checked above"),
            }
        }
        Ok((m, v, dm_dmu, dm_dvar, dv_dmu, dv_dvar))
    }
}

/// The outcome transform of one output: BoTorch's `ChainedOutcomeTransform`
/// restricted to the steps that apply to the output (see the [module
/// documentation](self)).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct OutcomeChain {
    steps: Vec<OutcomeStep>,
}

impl OutcomeChain {
    /// A chain of `steps` (in forward order).
    pub fn new(steps: Vec<OutcomeStep>) -> Self {
        Self { steps }
    }

    /// The steps in forward order.
    pub fn steps(&self) -> &[OutcomeStep] {
        &self.steps
    }

    /// Whether the chain has no steps (the identity).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Whether every step is affine (BoTorch's `_is_linear`; the posterior
    /// stays Gaussian).
    pub fn is_linear(&self) -> bool {
        self.steps.iter().all(OutcomeStep::is_linear)
    }

    /// The single plain `Standardize` of the chain, if that is all it is.
    pub fn as_standardize(&self) -> Option<&Standardize> {
        match self.steps.as_slice() {
            [OutcomeStep::Standardize(st)] => Some(st),
            _ => None,
        }
    }

    /// Whether some step depends on the inputs (`StratifiedStandardize`).
    pub fn depends_on_inputs(&self) -> bool {
        self.steps
            .iter()
            .any(|s| matches!(s, OutcomeStep::StratifiedStandardize(_)))
    }

    /// The chain in train mode (BoTorch's model constructor): fits every
    /// step to the targets as transformed by the preceding steps and
    /// returns the transformed targets and noise.
    ///
    /// # Errors
    /// See [`OutcomeStep::transform`].
    pub fn fit_transform(
        &mut self,
        y: &Matrix,
        yvar: Option<&Matrix>,
        x: &Matrix,
    ) -> Result<(Matrix, Option<Matrix>)> {
        let mut y = y.clone();
        let mut v = yvar.cloned();
        for step in &mut self.steps {
            let (ny, nv) = step.fit_transform(&y, v.as_ref(), x)?;
            y = ny;
            v = nv;
        }
        Ok((y, v))
    }

    /// The chain in eval mode (e.g. new observations of
    /// `condition_on_observations`).
    ///
    /// # Errors
    /// See [`OutcomeStep::transform`].
    pub fn transform(
        &self,
        y: &Matrix,
        yvar: Option<&Matrix>,
        x: &Matrix,
    ) -> Result<(Matrix, Option<Matrix>)> {
        let mut y = y.clone();
        let mut v = yvar.cloned();
        for step in &self.steps {
            let (ny, nv) = step.transform(&y, v.as_ref(), x)?;
            y = ny;
            v = nv;
        }
        Ok((y, v))
    }

    /// The inverse of the chain (steps in reverse order).
    ///
    /// # Errors
    /// See [`OutcomeStep::untransform`].
    pub fn untransform(
        &self,
        y: &Matrix,
        yvar: Option<&Matrix>,
        x: &Matrix,
    ) -> Result<(Matrix, Option<Matrix>)> {
        let mut y = y.clone();
        let mut v = yvar.cloned();
        for step in self.steps.iter().rev() {
            let (ny, nv) = step.untransform(&y, v.as_ref(), x)?;
            y = ny;
            v = nv;
        }
        Ok((y, v))
    }

    /// The un-transformation of the Gaussian posterior at the transformed
    /// points `x` (one row per posterior point): the folded affine map of
    /// the trailing affine steps (applied first, in reverse order) and the
    /// sample steps of the rest (see the [module documentation](self)).
    ///
    /// # Errors
    /// Shape errors of a `StratifiedStandardize`.
    pub fn posterior_parts(&self, x: &Matrix) -> Result<(PointAffine, Vec<SampleStep>)> {
        let n = x.nrows();
        let mut affine = PointAffine::IDENTITY;
        let mut samples = Vec::new();
        let mut gaussian = true;
        for step in self.steps.iter().rev() {
            let a = match step {
                OutcomeStep::Standardize(st) => {
                    Some(PointAffine::Constant(st.means()[0], st.stdvs()[0]))
                }
                OutcomeStep::StratifiedStandardize(st) => {
                    Some(PointAffine::PerPoint(st.per_point(x)?))
                }
                _ => None,
            };
            match (gaussian, a) {
                (true, Some(a)) => {
                    affine = if affine == PointAffine::IDENTITY {
                        a
                    } else {
                        affine.then(&a, n)
                    };
                }
                (_, a) => {
                    gaussian = false;
                    samples.push(match (step, a) {
                        (_, Some(a)) => SampleStep::Affine(a),
                        (OutcomeStep::Log, _) => SampleStep::Exp,
                        (OutcomeStep::Power(p), _) => SampleStep::Power(1.0 / p),
                        (OutcomeStep::Bilog, _) => SampleStep::Bilog,
                        _ => unreachable!("affine steps map to Some"),
                    });
                }
            }
        }
        Ok((affine, samples))
    }
}

/// The default `min_stdv` of `StratifiedStandardize`.
pub const STRATIFIED_MIN_STDV: f64 = STANDARDIZE_MIN_STDV;

#[cfg(test)]
mod tests {
    use super::*;

    fn col(v: &[f64]) -> Matrix {
        Matrix::from_column_slice(v.len(), 1, v)
    }

    #[test]
    fn log_power_bilog_round_trips() {
        let y = col(&[0.5, 2.0, 3.5]);
        let x = Matrix::zeros(3, 1);
        for step in [
            OutcomeStep::Log,
            OutcomeStep::Power(0.5),
            OutcomeStep::Bilog,
        ] {
            let (t, _) = step.transform(&y, None, &x).unwrap();
            let (u, _) = step.untransform(&t, None, &x).unwrap();
            assert!((u - &y).amax() < 1e-14, "{step:?}");
        }
        let yv = col(&[0.1, 0.2, 0.3]);
        let (_, v) = OutcomeStep::Log.transform(&y, Some(&yv), &x).unwrap();
        assert!((v.unwrap()[(1, 0)] - 0.2 / 4.0).abs() < 1e-16);
        assert!(OutcomeStep::Power(2.0)
            .transform(&y, Some(&yv), &x)
            .is_err());
        assert!(OutcomeStep::Bilog.untransform(&y, Some(&yv), &x).is_err());
        assert_eq!(torch_sign(0.0), 0.0);
    }

    #[test]
    fn posterior_parts_fold_trailing_standardizations() {
        let mut chain = OutcomeChain::new(vec![
            OutcomeStep::Log,
            OutcomeStep::Standardize(Standardize::new()),
        ]);
        let y = col(&[0.5, 2.0, 3.5, 1.0]);
        let x = Matrix::zeros(4, 1);
        chain.fit_transform(&y, None, &x).unwrap();
        assert!(!chain.is_linear());
        let (aff, samples) = chain.posterior_parts(&x).unwrap();
        let st = match &chain.steps()[1] {
            OutcomeStep::Standardize(st) => st.clone(),
            _ => unreachable!(),
        };
        assert_eq!(aff, PointAffine::Constant(st.means()[0], st.stdvs()[0]));
        assert_eq!(samples, vec![SampleStep::Exp]);
        // Standardize before Log: Log is applied first, then the affine sample step.
        let mut chain = OutcomeChain::new(vec![
            OutcomeStep::Standardize(Standardize::new()),
            OutcomeStep::Log,
        ]);
        let y = col(&[5.0, 7.0, 9.0]);
        let x = Matrix::zeros(3, 1);
        assert!(chain.fit_transform(&y, None, &x).is_ok());
        let (aff, samples) = chain.posterior_parts(&x).unwrap();
        assert_eq!(aff, PointAffine::IDENTITY);
        assert_eq!(samples.len(), 2);
        assert!(matches!(samples[1], SampleStep::Affine(_)));
    }

    #[test]
    fn moments_and_derivatives() {
        let st = SampleTransform::new(
            vec![vec![
                SampleStep::Exp,
                SampleStep::Affine(PointAffine::Constant(0.3, 2.0)),
            ]],
            1,
        )
        .unwrap();
        let (mu, var) = (0.2, 0.4);
        let (m, v, a, b, c, d) = st.moments(0, 0, mu, var).unwrap();
        assert!((m - (0.3 + 2.0 * norm_to_lognorm_mean(mu, var))).abs() < 1e-14);
        assert!((v - 4.0 * norm_to_lognorm_variance(mu, var)).abs() < 1e-14);
        let h = 1e-6;
        let f = |dm: f64, dv: f64| st.moments(0, 0, mu + dm, var + dv).unwrap();
        assert!(((f(h, 0.0).0 - f(-h, 0.0).0) / (2.0 * h) - a).abs() < 1e-7);
        assert!(((f(0.0, h).0 - f(0.0, -h).0) / (2.0 * h) - b).abs() < 1e-7);
        assert!(((f(h, 0.0).1 - f(-h, 0.0).1) / (2.0 * h) - c).abs() < 1e-7);
        assert!(((f(0.0, h).1 - f(0.0, -h).1) / (2.0 * h) - d).abs() < 1e-7);
        let p = SampleTransform::new(vec![vec![SampleStep::Power(0.5)]], 1).unwrap();
        assert!(p.moments(0, 0, 1.0, 1.0).is_err());
        let (val, der) = p.apply_with_derivative(0, 0, 4.0);
        assert_eq!((val, der), (2.0, 0.25));
    }

    #[test]
    fn stratified_standardize() {
        let x = Matrix::from_row_slice(5, 2, &[0.1, 0.0, 0.2, 1.0, 0.3, 0.0, 0.4, 1.0, 0.5, 1.0]);
        let y = col(&[1.0, 2.0, 3.0, 6.0, 7.0]);
        let mut step = OutcomeStep::StratifiedStandardize(
            StratifiedStandardize::new(1, &[0, 1, 2], STRATIFIED_MIN_STDV).unwrap(),
        );
        let (t, _) = step.fit_transform(&y, None, &x).unwrap();
        let OutcomeStep::StratifiedStandardize(st) = &step else {
            unreachable!()
        };
        assert_eq!(st.means(), &[2.0, 5.0, 0.0]);
        assert!((st.stdvs()[0] - 2f64.sqrt()).abs() < 1e-15);
        assert_eq!(st.stdvs()[2], 1.0);
        assert!((t[(0, 0)] + 1.0 / 2f64.sqrt()).abs() < 1e-15);
        let (u, _) = step.untransform(&t, None, &x).unwrap();
        assert!((u - y).amax() < 1e-14);
    }
}
