//! Risk measures as Monte-Carlo objectives (`botorch.acquisition.risk_measures`;
//! Cakmak et al., *Bayesian Optimization of Risk Measures*, NeurIPS 2020):
//! [`CVaR`], [`VaR`], [`WorstCase`] and [`Expectation`].
//!
//! A risk measure maps posterior samples of `q·n_w` points, where every
//! block of `n_w` consecutive points belongs to one candidate (the layout of
//! a model with an `InputPerturbation`, see [`crate::transforms::input`]), to
//! one value per candidate and sample. The samples are first mapped to one
//! value per point by a preprocessing objective (BoTorch's
//! `preprocessing_function`, called without `X`; by default the samples of a
//! single-output model). As in BoTorch, the risk measures work on the lower
//! tail (larger values are better).
//!
//! Gradients are BoTorch's autograd gradients: the order statistics pass
//! the adjoint to the selected samples (`topk`, `quantile`, `min`), the mean
//! to all of them.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::mc_objective::{McObjective, McSamples, SamplePullback, SampleValues};
use crate::error::{BotorchError, Result};
use crate::Matrix;

/// BoTorch's error for multi-output samples without a scalarizing
/// preprocessing function.
pub const MULTI_OUTPUT_RISK_MESSAGE: &str =
    "Multi-output samples should be scalarized using a `preprocessing_function`.";

/// torch's order of floating-point values in `sort`/`topk` (NaN after all
/// numbers).
pub(crate) fn torch_cmp(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).expect("not NaN"),
    }
}

/// The indices `0..n` sorted by `values` in ascending order (stable).
pub(crate) fn argsort(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| torch_cmp(values[a], values[b]));
    idx
}

/// BoTorch's `ceil(n_w * alpha) - 1` (`alpha_idx`), evaluated in double
/// precision like Python.
pub(crate) fn alpha_idx(alpha: f64, n_w: usize) -> usize {
    ((n_w as f64 * alpha).ceil() as usize).saturating_sub(1)
}

/// Checks a risk level `alpha ∈ (0, 1]` (BoTorch's message).
pub(crate) fn check_alpha(alpha: f64, message: &str) -> Result<()> {
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(BotorchError::InvalidArgument(message.into()));
    }
    Ok(())
}

/// Checks `n_w ≥ 1`.
pub(crate) fn check_n_w(n_w: usize) -> Result<()> {
    if n_w == 0 {
        return Err(BotorchError::InvalidArgument(
            "n_w must be a positive integer".into(),
        ));
    }
    Ok(())
}

/// torch's error for a `view(..., -1, n_w)` of a number of points that is
/// not a multiple of `n_w`.
pub(crate) fn view_error(points: usize, n_w: usize) -> BotorchError {
    BotorchError::Runtime(format!(
        "shape '[-1, {n_w}]' is invalid for input of size {points}: the q-batch of the \
         samples ({points} points) must consist of blocks of n_w = {n_w} points per candidate"
    ))
}

/// The reduction of the `n_w` values of a candidate in one sample: the value
/// and the weights of the values in its derivative.
pub(crate) fn reduce_block(kind: &RiskKind, v: &[f64], w: &mut [f64]) -> f64 {
    let n_w = v.len();
    w.iter_mut().for_each(|x| *x = 0.0);
    match *kind {
        RiskKind::CVaR { alpha_idx } => {
            // topk(n_w - alpha_idx, largest=False).values.mean()
            let k = n_w - alpha_idx;
            let order = argsort(v);
            let mut s = 0.0;
            for &i in &order[..k] {
                s += v[i];
                w[i] = 1.0 / k as f64;
            }
            s / k as f64
        }
        RiskKind::VaR { q } => {
            // torch.quantile(q, interpolation="lower"): the sorted value at
            // floor(q · (n_w - 1)); NaN if any value is NaN.
            let order = argsort(v);
            let rank = if v.iter().any(|x| x.is_nan()) {
                n_w - 1
            } else {
                (q * (n_w - 1) as f64).floor() as usize
            };
            let i = order[rank.min(n_w - 1)];
            w[i] = 1.0;
            v[i]
        }
        RiskKind::WorstCase => {
            // min(dim=-1): the first minimum (NaN propagates).
            let mut best = 0;
            for i in 1..n_w {
                if v[best].is_nan() {
                    break;
                }
                if v[i].is_nan() || v[i] < v[best] {
                    best = i;
                }
            }
            w[best] = 1.0;
            v[best]
        }
        RiskKind::Expectation => {
            let s: f64 = v.iter().sum();
            w.iter_mut().for_each(|x| *x = 1.0 / n_w as f64);
            s / n_w as f64
        }
    }
}

/// The reduction of a risk measure (see [`reduce_block`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RiskKind {
    CVaR { alpha_idx: usize },
    VaR { q: f64 },
    WorstCase,
    Expectation,
}

/// The shared part of the single-output risk measures (BoTorch's
/// `RiskMeasureMCObjective`): `n_w` and the preprocessing function.
#[derive(Clone, Debug)]
struct RiskBase {
    n_w: usize,
    preprocessing: Option<Arc<dyn McObjective>>,
}

impl RiskBase {
    fn new(n_w: usize, preprocessing: Option<Arc<dyn McObjective>>) -> Result<Self> {
        check_n_w(n_w)?;
        Ok(Self { n_w, preprocessing })
    }

    /// BoTorch's `_prepare_samples` followed by the reduction of every block
    /// of `n_w` values: `q × S` values and their pullback.
    fn evaluate(
        &self,
        kind: &RiskKind,
        samples: &McSamples,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let (pre, pre_pullback) = match &self.preprocessing {
            None => {
                if samples.num_outputs() != 1 {
                    return Err(BotorchError::Runtime(MULTI_OUTPUT_RISK_MESSAGE.into()));
                }
                (samples.data().clone(), None)
            }
            Some(p) => {
                // BoTorch calls the preprocessing function without `X`.
                let v = p.evaluate(samples, None, with_grad)?;
                if v.values.ncols() != samples.num_samples() {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "the preprocessing function returned values for {} samples, expected {}",
                        v.values.ncols(),
                        samples.num_samples()
                    )));
                }
                (v.values, v.pullback)
            }
        };
        let (n, s) = pre.shape();
        let n_w = self.n_w;
        if n % n_w != 0 {
            return Err(view_error(n, n_w));
        }
        let q = n / n_w;
        let mut values = Matrix::zeros(q, s);
        let mut weights = if with_grad {
            Matrix::zeros(n, s)
        } else {
            Matrix::zeros(0, 0)
        };
        let mut v = vec![0.0; n_w];
        let mut w = vec![0.0; n_w];
        for c in 0..s {
            for g in 0..q {
                for (i, vi) in v.iter_mut().enumerate() {
                    *vi = pre[(g * n_w + i, c)];
                }
                values[(g, c)] = reduce_block(kind, &v, &mut w);
                if with_grad {
                    for (i, wi) in w.iter().enumerate() {
                        weights[(g * n_w + i, c)] = *wi;
                    }
                }
            }
        }
        let pullback: Option<SamplePullback> = if with_grad {
            Some(Box::new(move |bar: &Matrix| {
                let mut pre_bar = weights;
                for c in 0..s {
                    for r in 0..n {
                        pre_bar[(r, c)] *= bar[(r / n_w, c)];
                    }
                }
                match pre_pullback {
                    Some(pb) => {
                        let (sbar, _) = pb(&pre_bar)?;
                        Ok((sbar, None))
                    }
                    None => Ok((pre_bar, None)),
                }
            }))
        } else {
            None
        };
        Ok(SampleValues { values, pullback })
    }
}

macro_rules! risk_common {
    () => {
        /// The number of points per candidate `n_w`.
        pub fn n_w(&self) -> usize {
            self.base.n_w
        }

        /// The preprocessing function (`None`: the samples of a single-output
        /// model).
        pub fn preprocessing_function(&self) -> Option<&Arc<dyn McObjective>> {
            self.base.preprocessing.as_ref()
        }
    };
}

/// The Conditional Value-at-Risk (BoTorch's `CVaR`): the mean of the
/// `n_w - alpha_idx` smallest values of every candidate, `alpha_idx =
/// ceil(n_w·alpha) - 1`.
#[derive(Clone, Debug)]
pub struct CVaR {
    base: RiskBase,
    alpha: f64,
    alpha_idx: usize,
}

impl CVaR {
    /// `CVaR(alpha, n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `alpha ∈ (0, 1]` (BoTorch's
    /// message) and `n_w ≥ 1`.
    pub fn new(
        alpha: f64,
        n_w: usize,
        preprocessing: Option<Arc<dyn McObjective>>,
    ) -> Result<Self> {
        let base = RiskBase::new(n_w, preprocessing)?;
        check_alpha(alpha, "alpha must be in (0.0, 1.0]")?;
        Ok(Self {
            base,
            alpha,
            alpha_idx: alpha_idx(alpha, n_w),
        })
    }

    /// The risk level.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// BoTorch's `alpha_idx`.
    pub fn alpha_idx(&self) -> usize {
        self.alpha_idx
    }

    risk_common!();
}

impl McObjective for CVaR {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        self.base.evaluate(
            &RiskKind::CVaR {
                alpha_idx: self.alpha_idx,
            },
            samples,
            with_grad,
        )
    }
}

/// The Value-at-Risk (BoTorch's `VaR`): the `1 - alpha_idx / n_w` quantile
/// (`interpolation="lower"`) of the values of every candidate, i.e. the
/// `alpha_idx`-th largest value.
#[derive(Clone, Debug)]
pub struct VaR {
    base: RiskBase,
    alpha: f64,
    alpha_idx: usize,
    q: f64,
}

impl VaR {
    /// `VaR(alpha, n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// See [`CVaR::new`].
    pub fn new(
        alpha: f64,
        n_w: usize,
        preprocessing: Option<Arc<dyn McObjective>>,
    ) -> Result<Self> {
        let base = RiskBase::new(n_w, preprocessing)?;
        check_alpha(alpha, "alpha must be in (0.0, 1.0]")?;
        let idx = alpha_idx(alpha, n_w);
        Ok(Self {
            base,
            alpha,
            alpha_idx: idx,
            q: 1.0 - idx as f64 / n_w as f64,
        })
    }

    /// The risk level.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// BoTorch's `alpha_idx`.
    pub fn alpha_idx(&self) -> usize {
        self.alpha_idx
    }

    risk_common!();
}

impl McObjective for VaR {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        self.base
            .evaluate(&RiskKind::VaR { q: self.q }, samples, with_grad)
    }
}

/// The worst case (BoTorch's `WorstCase`): the minimum of the values of
/// every candidate.
#[derive(Clone, Debug)]
pub struct WorstCase {
    base: RiskBase,
}

impl WorstCase {
    /// `WorstCase(n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `n_w = 0`.
    pub fn new(n_w: usize, preprocessing: Option<Arc<dyn McObjective>>) -> Result<Self> {
        Ok(Self {
            base: RiskBase::new(n_w, preprocessing)?,
        })
    }

    risk_common!();
}

impl McObjective for WorstCase {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        self.base.evaluate(&RiskKind::WorstCase, samples, with_grad)
    }
}

/// The expectation (BoTorch's `Expectation`): the mean of the values of
/// every candidate.
#[derive(Clone, Debug)]
pub struct Expectation {
    base: RiskBase,
}

impl Expectation {
    /// `Expectation(n_w, preprocessing_function)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `n_w = 0`.
    pub fn new(n_w: usize, preprocessing: Option<Arc<dyn McObjective>>) -> Result<Self> {
        Ok(Self {
            base: RiskBase::new(n_w, preprocessing)?,
        })
    }

    risk_common!();
}

impl McObjective for Expectation {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        self.base
            .evaluate(&RiskKind::Expectation, samples, with_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition::mc_objective::LinearMcObjective;
    use crate::Vector;

    fn samples(s: usize, n: usize, m: usize) -> McSamples {
        let rm: Vec<f64> = (0..s * n * m)
            .map(|i| ((i * 37 + 11) % 29) as f64 / 7.0 - 2.0 + 1e-3 * i as f64)
            .collect();
        McSamples::from_row_major(&rm, s, n, m).unwrap()
    }

    /// Central differences of `⟨bar, objective(samples)⟩`.
    fn check_pullback(obj: &dyn McObjective, s: &McSamples) {
        let out = obj.evaluate(s, None, true).unwrap();
        let bar = Matrix::from_fn(out.values.nrows(), out.values.ncols(), |r, c| {
            0.4 + 0.3 * r as f64 - 0.1 * c as f64
        });
        let (adj, _) = (out.pullback.unwrap())(&bar).unwrap();
        let (ns, n, m) = (s.num_samples(), s.num_points(), s.num_outputs());
        let rm = s.to_row_major();
        let h = 1e-7;
        for i in 0..rm.len() {
            let f = |delta: f64| {
                let mut v = rm.clone();
                v[i] += delta;
                let sp = McSamples::from_row_major(&v, ns, n, m).unwrap();
                obj.evaluate(&sp, None, false)
                    .unwrap()
                    .values
                    .component_mul(&bar)
                    .sum()
            };
            let fd = (f(h) - f(-h)) / (2.0 * h);
            let (c, rest) = (i / (n * m), i % (n * m));
            let (p, t) = (rest / m, rest % m);
            assert!((fd - adj[(t * n + p, c)]).abs() < 1e-6, "{i}");
        }
    }

    #[test]
    fn order_statistics_of_blocks() {
        let s = samples(3, 8, 1);
        let n_w = 4;
        let cvar = CVaR::new(0.5, n_w, None).unwrap();
        assert_eq!(cvar.alpha_idx(), 1);
        let var = VaR::new(0.5, n_w, None).unwrap();
        let wc = WorstCase::new(n_w, None).unwrap();
        let ex = Expectation::new(n_w, None).unwrap();
        let block = |c: usize, g: usize| -> Vec<f64> {
            let mut v: Vec<f64> = (0..n_w).map(|i| s.value(c, g * n_w + i, 0)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        let vc = cvar.evaluate(&s, None, false).unwrap().values;
        let vv = var.evaluate(&s, None, false).unwrap().values;
        let vw = wc.evaluate(&s, None, false).unwrap().values;
        let ve = ex.evaluate(&s, None, false).unwrap().values;
        for c in 0..3 {
            for g in 0..2 {
                let b = block(c, g);
                assert_eq!(vc[(g, c)], (b[0] + b[1] + b[2]) / 3.0);
                // VaR: the alpha_idx-th largest value.
                assert_eq!(vv[(g, c)], b[n_w - 1 - 1]);
                assert_eq!(vw[(g, c)], b[0]);
                assert!((ve[(g, c)] - b.iter().sum::<f64>() / 4.0).abs() < 1e-15);
                // The expectation and the worst case bound the others.
                for v in [vc[(g, c)], vv[(g, c)]] {
                    assert!(vw[(g, c)] <= v && v <= b[3]);
                }
                assert!(vc[(g, c)] <= ve[(g, c)]);
            }
        }
        for obj in [
            &cvar as &dyn McObjective,
            &var as &dyn McObjective,
            &wc as &dyn McObjective,
            &ex as &dyn McObjective,
        ] {
            check_pullback(obj, &s);
        }
    }

    #[test]
    fn preprocessing_and_errors() {
        let s = samples(2, 6, 2);
        let lin: Arc<dyn McObjective> =
            Arc::new(LinearMcObjective::new(Vector::from_vec(vec![0.7, -1.2])).unwrap());
        let cvar = CVaR::new(0.8, 3, Some(Arc::clone(&lin))).unwrap();
        check_pullback(&cvar, &s);
        let e = CVaR::new(0.8, 3, None)
            .unwrap()
            .evaluate(&s, None, false)
            .unwrap_err();
        assert!(e.to_string().contains("preprocessing_function"));
        assert!(VaR::new(0.8, 4, Some(lin))
            .unwrap()
            .evaluate(&s, None, false)
            .is_err());
        assert!(CVaR::new(0.0, 3, None).is_err());
        assert!(CVaR::new(1.5, 3, None).is_err());
        // ceil(25 * 0.28) - 1 = 7 in floating point (25 * 0.28 = 7.000000000000001).
        assert_eq!(CVaR::new(0.28, 25, None).unwrap().alpha_idx(), 7);
    }
}
