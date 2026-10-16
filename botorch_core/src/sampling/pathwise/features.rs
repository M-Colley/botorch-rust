//! Feature maps of the pathwise samplers (`botorch.sampling.pathwise.features`):
//! random Fourier features of a stationary kernel ([`KernelFeatureMap`], the
//! basis of the prior paths) and kernel evaluations at fixed points
//! ([`KernelEvaluationMap`], the basis of the update paths).
//!
//! Feature maps act on the inputs of the kernel, i.e. on the model's
//! *transformed* inputs; paths apply the model's input transform first.

use std::fmt::Debug;

use crate::error::{BotorchError, Result};
use crate::kernels::{BuiltCovar, Kernel, Prepared};
use crate::sampling::pathwise::utils::{check_num_features, draw_fourier_frequencies};
use crate::{Matrix, Vector};

/// The default number of Fourier features of the prior paths (BoTorch's
/// `num_features=1024` of `_draw_kernel_feature_paths_fallback`).
pub const DEFAULT_NUM_FEATURES: usize = 1024;

/// A feature map `φ: R^d → R^L` (BoTorch's `FeatureMap`), evaluated on the
/// inputs of the kernel.
pub trait FeatureMap: Send + Sync + Debug {
    /// The input dimension `d`.
    fn dim(&self) -> usize;

    /// The number of features `L` (BoTorch's `num_outputs`).
    fn num_features(&self) -> usize;

    /// The features of the rows of `z` (`n × d`): an `n × L` matrix.
    fn features(&self, z: &Matrix) -> Matrix;

    /// The features of one point (length `L`) and their derivatives with
    /// respect to its coordinates (`L × d`).
    fn features_with_grad(&self, z: &[f64]) -> (Vector, Matrix);

    /// Checks that the rows of `z` are valid inputs (e.g. task indices in
    /// range for index kernels, the unit cube for an
    /// `OrthogonalAdditiveKernel`); [`FeatureMap::features`] may assume it.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for invalid inputs.
    fn validate(&self, _z: &Matrix) -> Result<()> {
        Ok(())
    }
}

/// Random Fourier features of a stationary kernel (BoTorch's
/// `KernelFeatureMap` from `gen_kernel_features`): with frequencies `ω_l`
/// (the rows of `frequencies`, `L/2 × d`),
///
/// ```text
/// φ(z) = √s · √(2/L) · [sin(u_1), ..., sin(u_{L/2}), cos(u_1), ..., cos(u_{L/2})],
/// u_l = ⟨z ⊙ ℓ⁻¹, ω_l⟩,
/// ```
///
/// so that `φ(z)ᵀ φ(z')` approximates `k(z, z')` for frequencies drawn from
/// the kernel's spectral density. `ℓ` are the lengthscales and `s` the
/// outputscale of a `ScaleKernel` (1 without). The operations follow
/// BoTorch's `InverseLengthscaleTransform`, `SineCosineTransform` and
/// `OutputscaleTransform`; with `active_dims` (BoTorch's `FeatureSelector`),
/// `z` are the selected inputs.
#[derive(Clone, Debug)]
pub struct KernelFeatureMap {
    kernel: Kernel,
    frequencies: Matrix,
    inv_lengthscale: Vec<f64>,
    scale: f64,
    active_dims: Option<Vec<usize>>,
    input_dim: usize,
}

impl KernelFeatureMap {
    /// The feature map of `kernel` with the given frequencies (BoTorch's
    /// `feature_map.weight`, `L/2 × d`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the frequencies do not have
    /// `kernel.dim()` columns or no rows, [`BotorchError::InvalidArgument`]
    /// for non-finite frequencies.
    pub fn new(kernel: Kernel, frequencies: Matrix) -> Result<Self> {
        if frequencies.ncols() != kernel.dim() || frequencies.nrows() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "Fourier frequencies of a kernel with {} lengthscales must have {} columns and at \
                 least one row, got {}x{}",
                kernel.dim(),
                kernel.dim(),
                frequencies.nrows(),
                frequencies.ncols()
            )));
        }
        if frequencies.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "Fourier frequencies must be finite".into(),
            ));
        }
        let num_features = 2 * frequencies.nrows();
        let inv_lengthscale = kernel.lengthscale().iter().map(|l| 1.0 / l).collect();
        let input_dim = kernel.dim();
        Ok(Self {
            kernel,
            frequencies,
            inv_lengthscale,
            scale: (2.0 / num_features as f64).powf(0.5),
            active_dims: None,
            input_dim,
        })
    }

    /// The map acting on the columns `active_dims` of `input_dim`-dimensional
    /// inputs (BoTorch's `FeatureSelector(active_dims)` before the
    /// lengthscales); the kernel has one lengthscale per active dimension.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the kernel does not have one
    /// lengthscale per active dimension or an index is out of range.
    pub fn with_active_dims(mut self, active_dims: Vec<usize>, input_dim: usize) -> Result<Self> {
        if active_dims.len() != self.kernel.dim() || active_dims.iter().any(|&j| j >= input_dim) {
            return Err(BotorchError::DimensionMismatch(format!(
                "active_dims {active_dims:?} of {input_dim} inputs do not match a kernel with {} \
                 lengthscales",
                self.kernel.dim()
            )));
        }
        self.active_dims = Some(active_dims);
        self.input_dim = input_dim;
        Ok(self)
    }

    /// The `active_dims`, if any.
    pub fn active_dims(&self) -> Option<&[usize]> {
        self.active_dims.as_deref()
    }

    /// The column of the inputs of feature dimension `c`.
    #[inline]
    fn column(&self, c: usize) -> usize {
        self.active_dims.as_ref().map_or(c, |dims| dims[c])
    }

    /// The kernel.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// The frequencies `ω_l` (`L/2 × d`, BoTorch's `weight`).
    pub fn frequencies(&self) -> &Matrix {
        &self.frequencies
    }

    /// `u = (z ⊙ ℓ⁻¹) ωᵀ` for one point.
    fn projections(&self, z: &[f64]) -> Vec<f64> {
        let zs: Vec<f64> = self
            .inv_lengthscale
            .iter()
            .enumerate()
            .map(|(c, r)| r * z[self.column(c)])
            .collect();
        (0..self.frequencies.nrows())
            .map(|l| {
                zs.iter()
                    .enumerate()
                    .map(|(k, v)| v * self.frequencies[(l, k)])
                    .sum()
            })
            .collect()
    }

    /// `√s · (√(2/L) · v)`, BoTorch's order of the two scalings.
    #[inline]
    fn scaled(&self, v: f64) -> f64 {
        let v = self.scale * v;
        match self.kernel.outputscale() {
            Some(s) => s.sqrt() * v,
            None => v,
        }
    }
}

impl FeatureMap for KernelFeatureMap {
    fn dim(&self) -> usize {
        self.input_dim
    }

    fn num_features(&self) -> usize {
        2 * self.frequencies.nrows()
    }

    fn features(&self, z: &Matrix) -> Matrix {
        let (n, d) = z.shape();
        debug_assert_eq!(d, self.dim());
        let half = self.frequencies.nrows();
        // (z ⊙ ℓ⁻¹) ωᵀ as one matrix product.
        let zs = Matrix::from_fn(n, self.kernel.dim(), |i, k| {
            self.inv_lengthscale[k] * z[(i, self.column(k))]
        });
        let u = zs * self.frequencies.transpose();
        let mut out = Matrix::zeros(n, 2 * half);
        for l in 0..half {
            for i in 0..n {
                let (s, c) = u[(i, l)].sin_cos();
                out[(i, l)] = self.scaled(s);
                out[(i, half + l)] = self.scaled(c);
            }
        }
        out
    }

    fn features_with_grad(&self, z: &[f64]) -> (Vector, Matrix) {
        let d = self.dim();
        let half = self.frequencies.nrows();
        let u = self.projections(z);
        let factor = self.scaled(1.0);
        let mut f = Vector::zeros(2 * half);
        let mut df = Matrix::zeros(2 * half, d);
        for (l, &ul) in u.iter().enumerate() {
            let (s, c) = ul.sin_cos();
            f[l] = self.scaled(s);
            f[half + l] = self.scaled(c);
            for k in 0..self.kernel.dim() {
                let du = self.frequencies[(l, k)] * self.inv_lengthscale[k];
                let col = self.column(k);
                df[(l, col)] += factor * c * du;
                df[(half + l, col)] += -factor * s * du;
            }
        }
        (f, df)
    }
}

/// Random Fourier features of `kernel` on `num_inputs` inputs with
/// `num_features` features (BoTorch's `gen_kernel_features`), with
/// frequencies drawn by
/// [`draw_fourier_frequencies`] from
/// `seed` (RBF: standard normal; Matérn-ν: multivariate Student-t with
/// `2ν` degrees of freedom; a
/// `ScaleKernel` scales the features by the square root of its
/// outputscale).
///
/// # Errors
/// [`BotorchError::Unsupported`] for an odd number of features (BoTorch's
/// message) and [`BotorchError::DimensionMismatch`] if `num_inputs` differs
/// from the kernel's dimension.
pub fn gen_kernel_features(
    kernel: &Kernel,
    num_inputs: usize,
    num_features: usize,
    seed: u64,
) -> Result<KernelFeatureMap> {
    check_num_features(num_features)?;
    if num_inputs != kernel.dim() {
        return Err(BotorchError::DimensionMismatch(format!(
            "the kernel has {} lengthscales but num_inputs is {num_inputs}",
            kernel.dim()
        )));
    }
    let w = draw_fourier_frequencies(kernel.kind(), num_inputs, num_features, seed)?;
    KernelFeatureMap::new(kernel.clone(), w)
}

/// The kernel evaluated at fixed points, `φ(z) = [k(z, p_1), ..., k(z, p_n)]`
/// (BoTorch's `KernelEvaluationMap`), the basis of the update paths of
/// Matheron's rule; any covariance module (also a kernel tree).
#[derive(Clone, Debug)]
pub struct KernelEvaluationMap {
    kernel: BuiltCovar,
    points: Matrix,
    scaled: Prepared,
}

impl KernelEvaluationMap {
    /// The map of `kernel` centered at the rows of `points` (`n × d`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `points` does not have
    /// `kernel.dim()` columns.
    pub fn new(kernel: Kernel, points: Matrix) -> Result<Self> {
        if points.ncols() != kernel.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Shape mismatch: the points have {} columns, but the kernel has {} lengthscales.",
                points.ncols(),
                kernel.dim()
            )));
        }
        Self::from_covar(BuiltCovar::Standard(kernel), points)
    }

    /// The map of a covariance module centered at the rows of `points`.
    ///
    /// # Errors
    /// Errors of [`BuiltCovar::prepare`] (e.g. a column mismatch).
    pub fn from_covar(kernel: BuiltCovar, points: Matrix) -> Result<Self> {
        let scaled = kernel.prepare(&points)?;
        Ok(Self {
            kernel,
            points,
            scaled,
        })
    }

    /// The covariance module.
    pub fn kernel(&self) -> &BuiltCovar {
        &self.kernel
    }

    /// The points (`n × d`).
    pub fn points(&self) -> &Matrix {
        &self.points
    }
}

impl FeatureMap for KernelEvaluationMap {
    fn dim(&self) -> usize {
        self.points.ncols()
    }

    fn num_features(&self) -> usize {
        self.points.nrows()
    }

    fn features(&self, z: &Matrix) -> Matrix {
        let p = self
            .kernel
            .prepare(z)
            .expect("path inputs are validated (FeatureMap::validate)");
        self.kernel.cross(&p, &self.scaled)
    }

    fn features_with_grad(&self, z: &[f64]) -> (Vector, Matrix) {
        let p = self
            .kernel
            .prepare(&Matrix::from_row_slice(1, z.len(), z))
            .expect("path inputs are validated (FeatureMap::validate)");
        self.kernel.cross_with_grad(&p, 0, &self.scaled)
    }

    fn validate(&self, z: &Matrix) -> Result<()> {
        self.kernel.prepare(z).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::KernelKind;

    fn points() -> Matrix {
        Matrix::from_row_slice(3, 2, &[0.1, 0.2, 0.5, 0.9, 0.35, 0.3])
    }

    #[test]
    fn fourier_features_approximate_the_kernel() {
        for kind in [KernelKind::Rbf, KernelKind::Matern52] {
            let kernel = Kernel::new(kind, Vector::from_vec(vec![0.4, 0.7]), Some(1.6)).unwrap();
            let map = gen_kernel_features(&kernel, 2, 2 * 16384, 7).unwrap();
            assert_eq!(map.num_features(), 32768);
            let x = points();
            let f = map.features(&x);
            let approx = &f * f.transpose();
            let exact = kernel.covariance_symmetric(&x).unwrap();
            let err = (approx - &exact).amax();
            assert!(err < 0.05, "{kind:?}: {err}");
        }
        let kernel = Kernel::new(KernelKind::Rbf, Vector::from_vec(vec![0.4]), None).unwrap();
        assert!(gen_kernel_features(&kernel, 1, 5, 0).is_err());
        assert!(gen_kernel_features(&kernel, 2, 4, 0).is_err());
    }

    #[test]
    fn feature_gradients_match_finite_differences() {
        let kernel = Kernel::new(
            KernelKind::Matern52,
            Vector::from_vec(vec![0.3, 0.8]),
            Some(0.9),
        )
        .unwrap();
        let fourier = gen_kernel_features(&kernel, 2, 64, 3).unwrap();
        let eval = KernelEvaluationMap::new(kernel.clone(), points()).unwrap();
        let maps: [&dyn FeatureMap; 2] = [&fourier, &eval];
        let z = [0.41, 0.66];
        for map in maps {
            let (f, df) = map.features_with_grad(&z);
            let fm = map.features(&Matrix::from_row_slice(1, 2, &z));
            assert!((fm.row(0).transpose() - &f).amax() < 1e-15);
            for k in 0..2 {
                let h = 1e-6;
                let mut zp = z;
                zp[k] += h;
                let mut zm = z;
                zm[k] -= h;
                let fd =
                    (map.features_with_grad(&zp).0 - map.features_with_grad(&zm).0) / (2.0 * h);
                assert!((fd - df.column(k)).amax() < 1e-7);
            }
        }
    }
}
