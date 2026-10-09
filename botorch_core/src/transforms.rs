//! Input and outcome transforms mirroring
//! `botorch.models.transforms.input.Normalize` and
//! `botorch.models.transforms.outcome.Standardize`.

use crate::error::{BotorchError, Result};
use crate::{Matrix, Vector};

/// Default `min_range` of BoTorch's `Normalize`.
pub const NORMALIZE_MIN_RANGE: f64 = 1e-8;

/// Default `min_stdv` of BoTorch's `Standardize`.
pub const STANDARDIZE_MIN_STDV: f64 = 1e-8;

/// Affine input normalization to the unit cube: `x ↦ (x - offset) / coefficient`.
///
/// With learned bounds (the default, `Normalize(d=d)` in BoTorch) the offset
/// and coefficient are the per-column minimum and range of the training
/// inputs; columns whose range is below `min_range` are left untransformed
/// (offset 0, coefficient 1). With explicit bounds, `offset = bounds[0]` and
/// `coefficient = bounds[1] - bounds[0]`. The transform always applies to all
/// input dimensions and uses BoTorch's default `center = 0.5`.
#[derive(Clone, Debug, PartialEq)]
pub struct Normalize {
    d: usize,
    offset: Vector,
    coefficient: Vector,
    learn_bounds: bool,
    min_range: f64,
}

impl Normalize {
    /// Normalization with bounds learned from the training inputs
    /// (`Normalize(d=d)`). Until it is fitted the transform is the identity.
    pub fn new(d: usize) -> Self {
        Self {
            d,
            offset: Vector::zeros(d),
            coefficient: Vector::from_element(d, 1.0),
            learn_bounds: true,
            min_range: NORMALIZE_MIN_RANGE,
        }
    }

    /// Normalization with explicit bounds (`Normalize(d=d, bounds=bounds)`),
    /// where `bounds` is a `2 × d` matrix of lower (row 0) and upper (row 1)
    /// bounds.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `bounds` is not `2 × d` and
    /// [`BotorchError::InvalidArgument`] if an upper bound does not exceed the
    /// lower bound or a bound is not finite.
    pub fn with_bounds(bounds: &Matrix) -> Result<Self> {
        if bounds.nrows() != 2 || bounds.ncols() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "bounds must be a 2 x d matrix, got {}x{}",
                bounds.nrows(),
                bounds.ncols()
            )));
        }
        let d = bounds.ncols();
        let offset = Vector::from_iterator(d, bounds.row(0).iter().copied());
        let coefficient = Vector::from_iterator(d, (0..d).map(|j| bounds[(1, j)] - bounds[(0, j)]));
        if bounds.iter().any(|v| !v.is_finite()) || coefficient.iter().any(|&c| c <= 0.0) {
            return Err(BotorchError::InvalidArgument(
                "Normalize bounds must be finite with upper > lower in every dimension".into(),
            ));
        }
        Ok(Self {
            d,
            offset,
            coefficient,
            learn_bounds: false,
            min_range: NORMALIZE_MIN_RANGE,
        })
    }

    /// Sets the minimum range below which a learned dimension is not
    /// normalized (BoTorch default `1e-8`).
    pub fn with_min_range(mut self, min_range: f64) -> Self {
        self.min_range = min_range;
        self
    }

    /// Input dimension.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Whether the bounds are learned from the training data.
    pub fn learn_bounds(&self) -> bool {
        self.learn_bounds
    }

    /// The `min_range` setting.
    pub fn min_range(&self) -> f64 {
        self.min_range
    }

    /// Per-dimension offsets (the lower bounds).
    pub fn offset(&self) -> &Vector {
        &self.offset
    }

    /// Per-dimension coefficients (the ranges).
    pub fn coefficient(&self) -> &Vector {
        &self.coefficient
    }

    /// The `2 × d` bounds `[offset; offset + coefficient]`.
    pub fn bounds(&self) -> Matrix {
        Matrix::from_fn(2, self.d, |i, j| {
            if i == 0 {
                self.offset[j]
            } else {
                self.offset[j] + self.coefficient[j]
            }
        })
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

    /// Learns the bounds from training inputs if `learn_bounds` is set (what
    /// BoTorch does when the transform is called in train mode); a no-op for
    /// explicit bounds.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] on a column mismatch and
    /// [`BotorchError::InputData`] if `x` has no rows.
    pub fn fit(&mut self, x: &Matrix) -> Result<()> {
        self.check(x)?;
        if !self.learn_bounds {
            return Ok(());
        }
        if x.nrows() == 0 {
            return Err(BotorchError::InputData(
                "cannot learn bounds from zero points".into(),
            ));
        }
        for j in 0..self.d {
            let col = x.column(j);
            let min = col.iter().copied().fold(f64::INFINITY, f64::min);
            let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = max - min;
            if range < self.min_range {
                self.offset[j] = 0.0;
                self.coefficient[j] = 1.0;
            } else {
                self.offset[j] = min;
                self.coefficient[j] = range;
            }
        }
        Ok(())
    }

    /// Applies the normalization (eval mode, i.e. with the current bounds).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] on a column mismatch.
    pub fn transform(&self, x: &Matrix) -> Result<Matrix> {
        self.check(x)?;
        Ok(Matrix::from_fn(x.nrows(), self.d, |i, j| {
            (x[(i, j)] - self.offset[j]) / self.coefficient[j]
        }))
    }

    /// Normalizes a single point.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] on a length mismatch.
    pub fn transform_point(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.d {
            return Err(BotorchError::DimensionMismatch(format!(
                "Wrong input dimension. Received {}, expected {}.",
                x.len(),
                self.d
            )));
        }
        Ok(x.iter()
            .enumerate()
            .map(|(j, v)| (v - self.offset[j]) / self.coefficient[j])
            .collect())
    }

    /// Inverts the normalization.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] on a column mismatch.
    pub fn untransform(&self, x: &Matrix) -> Result<Matrix> {
        self.check(x)?;
        Ok(Matrix::from_fn(x.nrows(), self.d, |i, j| {
            self.coefficient[j] * x[(i, j)] + self.offset[j]
        }))
    }
}

/// Outcome standardization (zero mean, unit variance per output column).
///
/// When fitted on training data (BoTorch's train-mode `forward`) it computes
/// the column means and *unbiased* standard deviations; a standard deviation
/// below `min_stdv` is replaced by 1 and with a single observation the
/// standard deviation is 1. Observation-noise variances are divided by the
/// squared standard deviation.
#[derive(Clone, Debug, PartialEq)]
pub struct Standardize {
    means: Vec<f64>,
    stdvs: Vec<f64>,
    min_stdv: f64,
    trained: bool,
}

impl Default for Standardize {
    fn default() -> Self {
        Self::new()
    }
}

impl Standardize {
    /// An untrained standardization with BoTorch's default `min_stdv = 1e-8`.
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stdvs: Vec::new(),
            min_stdv: STANDARDIZE_MIN_STDV,
            trained: false,
        }
    }

    /// Sets `min_stdv`.
    pub fn with_min_stdv(mut self, min_stdv: f64) -> Self {
        self.min_stdv = min_stdv;
        self
    }

    /// Whether the transform has been fitted.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Per-output means (empty until fitted).
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Per-output standard deviations (empty until fitted).
    pub fn stdvs(&self) -> &[f64] {
        &self.stdvs
    }

    /// The `min_stdv` setting.
    pub fn min_stdv(&self) -> f64 {
        self.min_stdv
    }

    /// Learns the means and standard deviations from `y` (`n × m`) and returns
    /// the standardized outcomes and noise variances (BoTorch's train-mode
    /// `forward`).
    ///
    /// # Errors
    /// [`BotorchError::InputData`] if `y` has no rows and
    /// [`BotorchError::DimensionMismatch`] if `yvar` does not match `y`.
    pub fn fit_transform(
        &mut self,
        y: &Matrix,
        yvar: Option<&Matrix>,
    ) -> Result<(Matrix, Option<Matrix>)> {
        let (n, m) = y.shape();
        if n < 1 {
            return Err(BotorchError::InputData(format!(
                "Can't standardize with no observations. Y has shape {n}x{m}."
            )));
        }
        let nf = n as f64;
        let mut means = Vec::with_capacity(m);
        let mut stdvs = Vec::with_capacity(m);
        for j in 0..m {
            let col = y.column(j);
            let mean = col.iter().sum::<f64>() / nf;
            let std = if n == 1 {
                1.0
            } else {
                // nanstd: mean of squared deviations times n / (n - 1), then sqrt.
                let msd = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / nf;
                (msd * nf / (nf - 1.0)).sqrt()
            };
            means.push(mean);
            stdvs.push(if std >= self.min_stdv { std } else { 1.0 });
        }
        self.means = means;
        self.stdvs = stdvs;
        self.trained = true;
        self.transform(y, yvar)
    }

    fn check_trained(&self, m: usize) -> Result<()> {
        if !self.trained {
            return Err(BotorchError::InvalidArgument(
                "Standardize must be fitted on outcome data before use".into(),
            ));
        }
        if m != self.means.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Wrong output dimension. Received {m}; expected {}.",
                self.means.len()
            )));
        }
        Ok(())
    }

    /// Standardizes outcomes with the learned statistics (eval mode).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if not fitted and
    /// [`BotorchError::DimensionMismatch`] on a shape mismatch.
    pub fn transform(&self, y: &Matrix, yvar: Option<&Matrix>) -> Result<(Matrix, Option<Matrix>)> {
        self.check_trained(y.ncols())?;
        if let Some(v) = yvar {
            if v.shape() != y.shape() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Yvar has shape {:?} but Y has shape {:?}",
                    v.shape(),
                    y.shape()
                )));
            }
        }
        let y_tf = Matrix::from_fn(y.nrows(), y.ncols(), |i, j| {
            (y[(i, j)] - self.means[j]) / self.stdvs[j]
        });
        let yvar_tf = yvar.map(|v| {
            Matrix::from_fn(v.nrows(), v.ncols(), |i, j| {
                v[(i, j)] / (self.stdvs[j] * self.stdvs[j])
            })
        });
        Ok((y_tf, yvar_tf))
    }

    /// Un-standardizes outcomes (`mean + std · y`) and variances (`std² · var`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if not fitted and
    /// [`BotorchError::DimensionMismatch`] on a shape mismatch.
    pub fn untransform(
        &self,
        y: &Matrix,
        yvar: Option<&Matrix>,
    ) -> Result<(Matrix, Option<Matrix>)> {
        self.check_trained(y.ncols())?;
        let y_utf = Matrix::from_fn(y.nrows(), y.ncols(), |i, j| {
            self.means[j] + self.stdvs[j] * y[(i, j)]
        });
        let yvar_utf = yvar.map(|v| {
            Matrix::from_fn(v.nrows(), v.ncols(), |i, j| {
                (self.stdvs[j] * self.stdvs[j]) * v[(i, j)]
            })
        });
        Ok((y_utf, yvar_utf))
    }
}

/// Choice of outcome transform for [`crate::SingleTaskGP`], mirroring the
/// `outcome_transform` argument of BoTorch's `SingleTaskGP`.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum OutcomeTransformChoice {
    /// BoTorch's `DEFAULT`: a fresh `Standardize` fitted on the training data.
    #[default]
    Default,
    /// No outcome transform (`outcome_transform=None`).
    None,
    /// A user-configured `Standardize` (refitted on the training data, like
    /// BoTorch, which calls `.train()` on the transform during construction).
    Standardize(Standardize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_learns_bounds_with_min_range() {
        let x = Matrix::from_row_slice(
            3,
            3,
            &[1.0, 5.0, 2.0, 3.0, 5.0, -1.0, 2.0, 5.0 + 1e-10, 0.0],
        );
        let mut nz = Normalize::new(3);
        nz.fit(&x).unwrap();
        assert_eq!(nz.offset().as_slice(), &[1.0, 0.0, -1.0]);
        assert_eq!(nz.coefficient().as_slice(), &[2.0, 1.0, 3.0]);
        let xt = nz.transform(&x).unwrap();
        assert_eq!(xt[(1, 0)], 1.0);
        assert_eq!(xt[(0, 2)], 1.0);
        let back = nz.untransform(&xt).unwrap();
        assert!((back - x).amax() < 1e-15);
    }

    #[test]
    fn normalize_with_explicit_bounds_ignores_fit() {
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, -1.0, 2.0, 1.0]);
        let mut nz = Normalize::with_bounds(&bounds).unwrap();
        nz.fit(&Matrix::from_row_slice(1, 2, &[5.0, 5.0])).unwrap();
        assert_eq!(nz.bounds(), bounds);
        assert!(Normalize::with_bounds(&Matrix::from_row_slice(2, 1, &[1.0, 1.0])).is_err());
    }

    #[test]
    fn standardize_uses_unbiased_std() {
        let y = Matrix::from_column_slice(4, 1, &[1.0, 2.0, 3.0, 6.0]);
        let yvar = Matrix::from_element(4, 1, 0.5);
        let mut st = Standardize::new();
        let (yt, vt) = st.fit_transform(&y, Some(&yvar)).unwrap();
        let mean = 3.0;
        let std = ((4.0 + 1.0 + 0.0 + 9.0) / 3.0f64).sqrt();
        assert!((st.means()[0] - mean).abs() < 1e-15);
        assert!((st.stdvs()[0] - std).abs() < 1e-15);
        assert!((yt[(3, 0)] - (6.0 - mean) / std).abs() < 1e-15);
        assert!((vt.unwrap()[(0, 0)] - 0.5 / (std * std)).abs() < 1e-15);
        let (back, _) = st.untransform(&yt, None).unwrap();
        assert!((back - y).amax() < 1e-14);
    }

    #[test]
    fn standardize_special_cases() {
        let mut st = Standardize::new();
        let (yt, _) = st
            .fit_transform(&Matrix::from_element(1, 1, 3.0), None)
            .unwrap();
        assert_eq!(st.stdvs(), &[1.0]);
        assert_eq!(yt[(0, 0)], 0.0);
        let mut st = Standardize::new();
        st.fit_transform(&Matrix::from_element(5, 1, 2.0), None)
            .unwrap();
        assert_eq!(st.stdvs(), &[1.0]);
        assert_eq!(st.means(), &[2.0]);
        assert!(Standardize::new()
            .transform(&Matrix::zeros(1, 1), None)
            .is_err());
    }
}
