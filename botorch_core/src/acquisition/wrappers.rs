//! Acquisition functions that wrap another acquisition function:
//!
//! * [`ProximalAcquisitionFunction`] (`botorch.acquisition.proximal`): the
//!   base acquisition function weighted by a squared exponential centered at
//!   the last training point;
//! * [`FixedFeatureAcquisitionFunction`] (`botorch.acquisition.fixed_feature`):
//!   the base acquisition function with some input columns fixed;
//! * [`PriorGuidedAcquisitionFunction`] (`botorch.acquisition.prior_guided`):
//!   the base acquisition function weighted by a prior over the inputs
//!   (πBO, Hvarfner et al. 2022), per point and MC sample for sample-reducing
//!   Monte-Carlo acquisition functions.
//!
//! Gradients are exact (chain rule through the wrapped acquisition function,
//! the weights and the user functions).

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock};
use crate::acquisition::penalized::PointPullback;
use crate::acquisition::{check_candidates, check_pending, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::transforms::Normalize;
use crate::{Matrix, Vector};

/// `torch.nn.functional.softplus(x, beta)` (linear above `beta·x > 20`) and
/// its derivative.
fn softplus(x: f64, beta: f64) -> (f64, f64) {
    let bx = beta * x;
    if bx > 20.0 {
        (x, 1.0)
    } else {
        let e = bx.exp();
        ((e.ln_1p()) / beta, e / (1.0 + e))
    }
}

/// BoTorch's `ProximalAcquisitionFunction(acq_function, proximal_weights,
/// transformed_weighting, beta)`: `acqf(x) · exp(-0.5 ||(x - x_last) /
/// w||²)` for `q = 1`, where `x_last` is the last training point (in the
/// model's input-transformed space with `transformed_weighting`, in which
/// case `x` is transformed too); with `beta`, `softplus(acqf(x), beta)`
/// replaces the base value (otherwise negative base values are an error).
#[derive(Debug)]
pub struct ProximalAcquisitionFunction {
    base: Arc<dyn AcquisitionFunction>,
    proximal_weights: Vector,
    last_x: Vector,
    input_transform: Option<Normalize>,
    transformed_weighting: bool,
    beta: Option<f64>,
}

impl ProximalAcquisitionFunction {
    /// The proximal weighting of `base` around the raw last training point
    /// `last_x` (`d`) with the model's input transform (if any).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for weights of the wrong length
    /// (BoTorch's message) and [`BotorchError::Unsupported`] for a base
    /// acquisition function with pending points (BoTorch's message).
    pub fn new(
        base: Arc<dyn AcquisitionFunction>,
        proximal_weights: Vector,
        last_x: Vector,
        input_transform: Option<Normalize>,
        transformed_weighting: bool,
        beta: Option<f64>,
    ) -> Result<Self> {
        if base.x_pending().is_some() {
            return Err(BotorchError::Unsupported(
                "Proximal acquisition function requires `X_pending` to be None.".into(),
            ));
        }
        if proximal_weights.len() != last_x.len() {
            return Err(BotorchError::InvalidArgument(
                "`proximal_weights` must be a one dimensional tensor with same feature \
                 dimension as model."
                    .into(),
            ));
        }
        Ok(Self {
            base,
            proximal_weights,
            last_x,
            input_transform,
            transformed_weighting,
            beta,
        })
    }

    /// The proximal weight and its gradient with respect to `x` (`1 × d`).
    fn weight(&self, x: &Matrix) -> Result<(f64, Vector)> {
        let d = x.ncols();
        let (diff, jac): (Vector, Vector) =
            match (&self.input_transform, self.transformed_weighting) {
                (Some(tf), true) => {
                    let xt = tf.transform(x)?;
                    let lt = tf.transform(&Matrix::from_row_slice(1, d, self.last_x.as_slice()))?;
                    (
                        Vector::from_fn(d, |k, _| xt[(0, k)] - lt[(0, k)]),
                        tf.coefficient().map(|c| 1.0 / c),
                    )
                }
                _ => (
                    Vector::from_fn(d, |k, _| x[(0, k)] - self.last_x[k]),
                    Vector::from_element(d, 1.0),
                ),
            };
        let scaled = diff.component_div(&self.proximal_weights);
        let n = scaled.norm();
        let m = n * n;
        let w = (-0.5 * m).exp();
        // d w / d x_k = -w (diff_k / pw_k²) ∂diff_k/∂x_k.
        let g = Vector::from_fn(d, |k, _| -w * scaled[k] / self.proximal_weights[k] * jac[k]);
        Ok((w, g))
    }

    fn base_value(&self, v: f64) -> Result<(f64, f64)> {
        match self.beta {
            Some(beta) => Ok(softplus(v, beta)),
            None => {
                if v < 0.0 {
                    return Err(BotorchError::Runtime(
                        "Cannot use proximal biasing for negative acquisition function values, \
                         set a value for beta to fix this with a softplus transform"
                            .into(),
                    ));
                }
                Ok((v, 1.0))
            }
        }
    }
}

impl AcquisitionFunction for ProximalAcquisitionFunction {
    fn model(&self) -> &Arc<dyn Model> {
        self.base.model()
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        check_candidates(x, self.base.dim(), Some(1))?;
        let (b, _) = self.base_value(self.base.forward(x)?)?;
        Ok(b * self.weight(x)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.base.dim(), Some(1))?;
        let (v, g) = self.base.forward_with_grad(x)?;
        let (b, db) = self.base_value(v)?;
        let (w, gw) = self.weight(x)?;
        let grad = g * (db * w) + Matrix::from_row_slice(1, gw.len(), gw.as_slice()) * b;
        Ok((b * w, grad))
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.base.set_x_pending(x_pending)
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.base.x_pending()
    }
}

/// BoTorch's `FixedFeatureAcquisitionFunction(acq_function, d, columns,
/// values)`: the base acquisition function of `d` inputs evaluated at
/// candidate sets of `d - d_f` columns with the columns `columns` fixed to
/// `values` (one value per fixed column, or one row of values per point of
/// the candidate sets).
#[derive(Debug)]
pub struct FixedFeatureAcquisitionFunction {
    base: Arc<dyn AcquisitionFunction>,
    d: usize,
    columns: Vec<usize>,
    /// `1 × d_f` (broadcast) or `q × d_f`.
    values: Matrix,
}

impl FixedFeatureAcquisitionFunction {
    /// Fixes the columns `columns` (each `< d`) of the inputs of `base` to
    /// `values` (`1 × d_f` or `q × d_f`, in the order of `columns` sorted).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for columns out of range or
    /// duplicates and values of the wrong shape.
    pub fn new(
        base: Arc<dyn AcquisitionFunction>,
        d: usize,
        columns: Vec<usize>,
        values: Matrix,
    ) -> Result<Self> {
        let mut sorted = columns.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != columns.len() || sorted.iter().any(|&c| c >= d) {
            return Err(BotorchError::InvalidArgument(format!(
                "columns must be distinct indices below d = {d}, got {columns:?}"
            )));
        }
        if values.ncols() != columns.len() || values.nrows() == 0 {
            return Err(BotorchError::InvalidArgument(format!(
                "values must have {} columns (one per fixed column), got {}x{}",
                columns.len(),
                values.nrows(),
                values.ncols()
            )));
        }
        Ok(Self {
            base,
            d,
            columns: sorted,
            values,
        })
    }

    /// BoTorch's `_construct_X_full`: the full inputs (`q × d`) of `x`
    /// (`q × (d - d_f)`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a wrong number of columns
    /// (BoTorch's message) or of rows of the values.
    pub fn construct_x_full(&self, x: &Matrix) -> Result<Matrix> {
        let d_f = self.columns.len();
        if x.ncols() + d_f != self.d {
            return Err(BotorchError::InvalidArgument(format!(
                "Feature dimension d' ({}) of input must be d - d_f ({}).",
                x.ncols(),
                self.d - d_f
            )));
        }
        let q = x.nrows();
        if self.values.nrows() != 1 && self.values.nrows() != q {
            return Err(BotorchError::InvalidArgument(format!(
                "the values of the fixed features have {} rows but X has {q} points",
                self.values.nrows()
            )));
        }
        let mut out = Matrix::zeros(q, self.d);
        for i in 0..q {
            let vr = if self.values.nrows() == 1 { 0 } else { i };
            let (mut jx, mut jf) = (0, 0);
            for c in 0..self.d {
                if jf < d_f && self.columns[jf] == c {
                    out[(i, c)] = self.values[(vr, jf)];
                    jf += 1;
                } else {
                    out[(i, c)] = x[(i, jx)];
                    jx += 1;
                }
            }
        }
        Ok(out)
    }

    fn free_columns(&self) -> Vec<usize> {
        (0..self.d).filter(|c| !self.columns.contains(c)).collect()
    }
}

impl AcquisitionFunction for FixedFeatureAcquisitionFunction {
    fn model(&self) -> &Arc<dyn Model> {
        self.base.model()
    }

    fn dim(&self) -> usize {
        self.d - self.columns.len()
    }

    fn expected_q(&self) -> Option<usize> {
        self.base.expected_q()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        self.base.forward(&self.construct_x_full(x)?)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.base.forward_with_grad(&self.construct_x_full(x)?)?;
        let free = self.free_columns();
        Ok((
            v,
            Matrix::from_fn(x.nrows(), free.len(), |p, j| g[(p, free[j])]),
        ))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.base.x_pending()
    }

    /// Sets the full pending points of the base acquisition function
    /// (BoTorch's `X_pending` setter).
    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        match x_pending {
            Some(p) => self.base.set_x_pending(Some(&self.construct_x_full(p)?)),
            None => self.base.set_x_pending(None),
        }
    }

    fn is_log(&self) -> bool {
        self.base.is_log()
    }

    fn is_nonnegative(&self) -> bool {
        self.base.is_nonnegative()
    }
}

/// The function of a prior over the inputs: the (log) prior density of every
/// point of a candidate set (`q`) and, if requested, the reverse pass.
pub type PriorFn = dyn Fn(&Matrix, bool) -> Result<(Vector, Option<PointPullback>)> + Send + Sync;

/// A prior over the inputs (BoTorch's `prior_module`).
pub struct PriorModule {
    f: Arc<PriorFn>,
}

impl Debug for PriorModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PriorModule")
    }
}

impl PriorModule {
    /// The prior of the function `f(x, with_grad)`.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Matrix, bool) -> Result<(Vector, Option<PointPullback>)> + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(Vector, Option<PointPullback>)> {
        let (v, g) = (self.f)(x, with_grad)?;
        if v.len() != x.nrows() {
            return Err(BotorchError::DimensionMismatch(format!(
                "the prior returned {} values for {} points",
                v.len(),
                x.nrows()
            )));
        }
        Ok((v, g))
    }
}

/// BoTorch's `PriorGuidedAcquisitionFunction(acq_function, prior_module,
/// log, prior_exponent, X_pending)`: the base acquisition function weighted
/// by `prior(x)^e` (added `e · log prior(x)` with `log`). For sample-reducing
/// Monte-Carlo acquisition functions every point's utilities are weighted
/// before the reductions over the points and the samples (the pending
/// points are appended to the candidates first); otherwise `q = 1`.
#[derive(Debug)]
pub struct PriorGuidedAcquisitionFunction {
    base: Arc<dyn AcquisitionFunction>,
    prior: PriorModule,
    log: bool,
    prior_exponent: f64,
    x_pending: RwLock<Option<Matrix>>,
}

impl PriorGuidedAcquisitionFunction {
    /// The prior-guided acquisition function.
    ///
    /// # Errors
    /// [`BotorchError::Runtime`] if the base acquisition function has pending
    /// points (BoTorch's message), [`BotorchError::Unsupported`] for a
    /// sample-reducing acquisition function whose utilities are on a log
    /// scale while `log` is false or vice versa.
    pub fn new(
        base: Arc<dyn AcquisitionFunction>,
        prior: PriorModule,
        log: bool,
        prior_exponent: f64,
        x_pending: Option<Matrix>,
    ) -> Result<Self> {
        if base.x_pending().is_some() {
            return Err(BotorchError::Runtime(
                "X_pending is set on acq_function, but should be set on \
                 `PriorGuidedAcquisitionFunction`."
                    .into(),
            ));
        }
        if let Some(sr) = base.as_sample_reducing() {
            if sr.log_utilities() != log {
                return Err(BotorchError::Unsupported(
                    "botorch_rs weights the utilities of a sample-reducing acquisition function \
                     by the prior only with `log` matching the acquisition function's log scale"
                        .into(),
                ));
            }
        }
        if let Some(p) = &x_pending {
            check_pending(p, base.dim())?;
        }
        Ok(Self {
            base,
            prior,
            log,
            prior_exponent,
            x_pending: RwLock::new(x_pending),
        })
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.base.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let xf = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let n = xf.nrows();
        let (prior, gprior) = self.prior.evaluate(&xf, with_grad)?;
        let e = self.prior_exponent;
        // The weight of every point and its derivative w.r.t. the prior.
        let (w, dw): (Vec<f64>, Vec<f64>) = prior
            .iter()
            .map(|&p| {
                if self.log {
                    (p * e, e)
                } else {
                    let pe = p.powf(e);
                    (pe, if p == 0.0 { 0.0 } else { e * pe / p })
                }
            })
            .unzip();
        if let Some(sr) = self.base.as_sample_reducing() {
            let (v, g) = sr.evaluate_weighted(&xf, &w, with_grad)?;
            let Some((gx, wbar)) = g else {
                return Ok((v, None));
            };
            let gp = gprior
                .ok_or_else(|| BotorchError::Runtime("the prior returned no gradient".into()))?;
            let adj = Vector::from_fn(n, |i, _| wbar[i] * dw[i]);
            let gpx = gp(&adj)?;
            // Gradients w.r.t. the candidates only (not the pending points).
            let grad = gx.rows(0, q) + gpx.rows(0, q);
            return Ok((v, Some(grad)));
        }
        if n > 1 {
            return Err(BotorchError::Unsupported(
                "q-batches with q>1 are only supported using SampleReducingMCAcquisitionFunction."
                    .into(),
            ));
        }
        if !with_grad {
            let a = self.base.forward(&xf)?;
            return Ok((if self.log { a + w[0] } else { a * w[0] }, None));
        }
        let (a, ga) = self.base.forward_with_grad(&xf)?;
        let gp =
            gprior.ok_or_else(|| BotorchError::Runtime("the prior returned no gradient".into()))?;
        if self.log {
            let gpx = gp(&Vector::from_element(1, dw[0]))?;
            Ok((a + w[0], Some(ga + gpx)))
        } else {
            let gpx = gp(&Vector::from_element(1, a * dw[0]))?;
            Ok((a * w[0], Some(ga * w[0] + gpx)))
        }
    }
}

impl AcquisitionFunction for PriorGuidedAcquisitionFunction {
    fn model(&self) -> &Arc<dyn Model> {
        self.base.model()
    }

    fn dim(&self) -> usize {
        self.base.dim()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.x_pending).clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.base.dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }

    fn is_log(&self) -> bool {
        self.log
    }
}
