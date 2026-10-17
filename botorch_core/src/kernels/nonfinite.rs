//! Where GPyTorch's training covariance is NaN.
//!
//! During a fit, L-BFGS-B's line search can try extreme raw hyperparameters,
//! e.g. a softplus lengthscale of `1e-293` or exactly zero. botorch_rs's
//! kernels then compute the mathematical limit (for a vanishing lengthscale
//! `K = s·I`), but GPyTorch's arithmetic overflows: its `sq_dist` centers the
//! scaled inputs `c = x/ℓ - mean(x/ℓ)` and expands the squared distance as
//! `‖cᵢ‖² + ‖cⱼ‖² - 2 cᵢ·cⱼ`, so squares above `1.8e308` give `inf - inf =
//! NaN`; a zero lengthscale gives `x / 0` and NaN after the centering. The
//! NaN covariance makes linear_operator raise `NanError`, BoTorch's closure
//! returns a NaN loss and NaN gradients, and scipy's L-BFGS-B line search
//! continues with them (typically it fails and the iteration restarts from
//! the last iterate). A finite value at such a point sends the line search
//! elsewhere and the fit to another optimum.
//!
//! This module reproduces GPyTorch's arithmetic where it decides
//! finiteness, for the kernels that go through `sq_dist` (RBF, Matérn,
//! rational quadratic, cosine and periodic kernels), and reports the NaN
//! entries of the training covariance `K(X, X)` and the hyperparameters
//! whose gradient is NaN. The marginal log likelihoods inject the NaN
//! entries into their own (finite) covariance, so that the Cholesky
//! factorization raises [`crate::BotorchError::Nan`] like linear_operator.
//! Finite values are never changed.
//!
//! GPyTorch evaluates kernels in two ways, which differ here:
//!
//! * the autograd path (several lengthscales, or inputs that require
//!   gradients): `sq_dist` with a diagonal computed like every other entry;
//! * the `RBFCovariance` / `MaternCovariance` autograd functions (a single
//!   lengthscale and inputs without gradients): `sq_dist` with the diagonal
//!   set to zero, and hand-written lengthscale derivatives such as
//!   `sq_dist · K / ℓ` for RBF, which are NaN where `inf · 0` occurs although
//!   the covariance is finite.
//!
//! Nothing is computed beyond a scan of the centered inputs unless one of
//! them is non-finite or larger than [`SAFE_MAGNITUDE`] in magnitude: below
//! that, no squared distance, kernel value or derivative factor can
//! overflow, so GPyTorch's values are finite. They still differ from
//! botorch_rs's exact values by GPyTorch's cancellation error, about
//! `1e-16 ‖c‖²` in the squared distances (its diagonal
//! `‖cᵢ‖² + ‖cᵢ‖² - 2 cᵢ·cᵢ` is not zero), which depends on the BLAS
//! summation order and cannot be reproduced; it matters only for scaled
//! inputs above about `1e4`, where BoTorch's own fits vary between twins.
//! Gradients of the autograd path that are NaN while the covariance is
//! finite (narrow bands around the overflow thresholds for RBF and Matérn
//! kernels, wider ones for the `α` of rational quadratic kernels and the
//! periods of periodic kernels) are not reproduced either.

use crate::kernels::KernelKind;

/// Magnitude of GPyTorch's centered scaled inputs up to which none of its
/// distance computations can overflow.
pub const SAFE_MAGNITUDE: f64 = 1e100;

const SQRT3: f64 = 1.732_050_807_568_877_2;
const SQRT5: f64 = 2.236_067_977_499_79;

/// GPyTorch's NaN entries of a training covariance `K(X, X)` and the
/// hyperparameters whose gradient is NaN.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingNan {
    /// Row-major `n × n` mask of the NaN entries (empty if there are none).
    mask: Vec<bool>,
    n: usize,
    /// Indices of the hyperparameters with a NaN gradient (lengthscale
    /// indices for a standard kernel, flat parameter offsets for a tree).
    grad: Vec<usize>,
}

impl TrainingNan {
    /// Whether some entry of the covariance is NaN.
    pub fn any(&self) -> bool {
        !self.mask.is_empty()
    }

    /// The hyperparameters whose gradient is NaN.
    pub fn nan_gradients(&self) -> &[usize] {
        &self.grad
    }

    /// Sets the NaN entries of `k` (`n × n`) to NaN.
    pub fn inject(&self, k: &mut crate::Matrix) {
        if self.mask.is_empty() {
            return;
        }
        for i in 0..self.n {
            for j in 0..self.n {
                if self.mask[i * self.n + j] {
                    k[(i, j)] = f64::NAN;
                }
            }
        }
    }

    /// Adds the NaN entries of an `n × n` kernel matrix (row-major mask).
    pub(crate) fn add_mask(&mut self, n: usize, mask: &[bool]) {
        if !mask.iter().any(|&m| m) {
            return;
        }
        if self.mask.is_empty() {
            self.mask = vec![false; n * n];
            self.n = n;
        }
        for (a, &b) in self.mask.iter_mut().zip(mask) {
            *a |= b;
        }
    }

    /// Marks the gradient of hyperparameter `index` as NaN.
    pub(crate) fn add_grad(&mut self, index: usize) {
        if !self.grad.contains(&index) {
            self.grad.push(index);
        }
    }

    /// Adds the NaN entries and gradients of `other`.
    pub(crate) fn merge(&mut self, other: TrainingNan) {
        if other.any() {
            let n = other.n;
            self.add_mask(n, &other.mask);
        }
        for g in other.grad {
            self.add_grad(g);
        }
    }
}

/// `res.clamp_min(1e-30).sqrt()` as in `gpytorch.kernels.kernel.dist`
/// (torch's clamp keeps NaN, unlike `f64::max`).
#[inline]
fn clamped_sqrt(r: f64) -> f64 {
    if r < 1e-30 {
        1e-30f64.sqrt()
    } else {
        r.sqrt()
    }
}

/// `x1.mean(dim=-2)` of a row-major `n × k` buffer.
fn column_means(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut m = vec![0.0; k];
    for i in 0..n {
        for (a, v) in m.iter_mut().zip(&x[i * k..(i + 1) * k]) {
            *a += v;
        }
    }
    let nf = n as f64;
    m.iter().map(|s| s / nf).collect()
}

/// GPyTorch's squared distances `sq_dist(x1_, x1_, x1_eq_x2=True)` of the
/// rows of `x` (GPyTorch's scaled inputs, row-major `n × k`), after its
/// `clamp_min_(0)`, with the diagonal set to zero if `zero_diag` (inputs
/// without gradients). `None` if the centered inputs are finite and at most
/// [`SAFE_MAGNITUDE`] in magnitude (then all distances are finite).
pub(crate) fn sq_dist(x: &[f64], n: usize, k: usize, zero_diag: bool) -> Option<Vec<f64>> {
    if n == 0 {
        return None;
    }
    let adj = column_means(x, n, k);
    let c: Vec<f64> = x.iter().enumerate().map(|(i, v)| v - adj[i % k]).collect();
    if c.iter().all(|v| v.abs() <= SAFE_MAGNITUDE) {
        return None;
    }
    // x1_ = [-2 c, ‖c‖², 1] and x2_ = [c, 1, ‖c‖²]; res = x1_ x2_ᵀ.
    let norms: Vec<f64> = (0..n)
        .map(|i| c[i * k..(i + 1) * k].iter().map(|v| v * v).sum())
        .collect();
    let mut res = vec![0.0; n * n];
    for i in 0..n {
        let ci = &c[i * k..(i + 1) * k];
        for j in 0..n {
            let r = if zero_diag && i == j {
                0.0
            } else {
                let cj = &c[j * k..(j + 1) * k];
                let mut acc = 0.0;
                for (a, b) in ci.iter().zip(cj) {
                    acc += (-2.0 * a) * b;
                }
                acc + norms[i] + norms[j]
            };
            // `clamp_min_(0)` keeps NaN.
            res[i * n + j] = if r < 0.0 { 0.0 } else { r };
        }
    }
    Some(res)
}

/// GPyTorch's training-mode NaN entries of a stationary RBF / Matérn kernel
/// on the rows of `x` (`n × k`, row-major, after `active_dims`) with
/// lengthscales `ls` (one per column, or one shared). `function_path`:
/// GPyTorch's `RBFCovariance` / `MaternCovariance` (a single lengthscale and
/// inputs without gradients). Gradient flags refer to lengthscale index 0.
pub(crate) fn stationary(
    kind: KernelKind,
    x: &[f64],
    n: usize,
    k: usize,
    ls: &[f64],
    function_path: bool,
) -> TrainingNan {
    let mut out = TrainingNan::default();
    let l = |j: usize| if ls.len() == 1 { ls[0] } else { ls[j] };
    // RBF divides first; Matérn subtracts the mean of the inputs first.
    let scaled: Vec<f64> = match kind {
        KernelKind::Rbf => x.iter().enumerate().map(|(i, v)| v / l(i % k)).collect(),
        _ => {
            let mean = column_means(x, n, k);
            x.iter()
                .enumerate()
                .map(|(i, v)| (v - mean[i % k]) / l(i % k))
                .collect()
        }
    };
    let Some(res) = sq_dist(&scaled, n, k, function_path) else {
        return out;
    };
    let ell = ls[0];
    let mut mask = vec![false; n * n];
    let mut grad_nan = false;
    for (m, &r) in mask.iter_mut().zip(&res) {
        let (value, d_out) = match kind {
            KernelKind::Rbf => {
                let kv = (r / -2.0).exp();
                (kv, r * kv / ell)
            }
            _ => {
                let dist = clamped_sqrt(r);
                if function_path {
                    matern_function(kind, dist, ell)
                } else {
                    (matern_autograd(kind, dist), 0.0)
                }
            }
        };
        *m = value.is_nan();
        grad_nan |= function_path && d_out.is_nan();
    }
    out.add_mask(n, &mask);
    if grad_nan && !out.any() {
        out.add_grad(0);
    }
    out
}

/// `MaternKernel.forward` (autograd path) from the clamped distance.
fn matern_autograd(kind: KernelKind, dist: f64) -> f64 {
    match kind {
        KernelKind::Matern12 => (-dist).exp(),
        KernelKind::Matern32 => (SQRT3 * dist + 1.0) * (-SQRT3 * dist).exp(),
        KernelKind::Matern52 => {
            (SQRT5 * dist + 1.0 + 5.0 / 3.0 * (dist * dist)) * (-SQRT5 * dist).exp()
        }
        KernelKind::Rbf => unreachable!("not a Matérn kernel"),
    }
}

/// `MaternCovariance.forward`: the value and `d_output_d_input`.
fn matern_function(kind: KernelKind, dist: f64, ell: f64) -> (f64, f64) {
    match kind {
        KernelKind::Matern12 => {
            let s = dist;
            let e = (-s).exp();
            (e, s / ell * e)
        }
        KernelKind::Matern32 => {
            let s = dist * SQRT3;
            let e = (-s).exp();
            ((s + 1.0) * e, s * s / ell * e)
        }
        KernelKind::Matern52 => {
            let s = dist * SQRT5;
            let lin = s + 1.0;
            let quad = s * s / 3.0;
            let e = (-s).exp();
            ((lin + quad) * e, lin * quad * e / ell)
        }
        KernelKind::Rbf => unreachable!("not a Matérn kernel"),
    }
}

/// `RQKernel.forward` on the rows of `x`: `(1 + sq_dist / (2α))^(-α)` of
/// the inputs divided by the lengthscales.
pub(crate) fn rational_quadratic(
    x: &[f64],
    n: usize,
    k: usize,
    ls: &[f64],
    alpha: f64,
) -> TrainingNan {
    let mut out = TrainingNan::default();
    let l = |j: usize| if ls.len() == 1 { ls[0] } else { ls[j] };
    let scaled: Vec<f64> = x.iter().enumerate().map(|(i, v)| v / l(i % k)).collect();
    if let Some(res) = sq_dist(&scaled, n, k, false) {
        let mask: Vec<bool> = res
            .iter()
            .map(|&r| (1.0 + r / (2.0 * alpha)).powf(-alpha).is_nan())
            .collect();
        out.add_mask(n, &mask);
    }
    out
}

/// `CosineKernel.forward` on the rows of `x`: `cos(π · dist(x / p))`.
pub(crate) fn cosine(x: &[f64], n: usize, k: usize, period: f64) -> TrainingNan {
    let mut out = TrainingNan::default();
    let scaled: Vec<f64> = x.iter().map(|v| v / period).collect();
    if let Some(res) = sq_dist(&scaled, n, k, false) {
        let mask: Vec<bool> = res
            .iter()
            .map(|&r| (clamped_sqrt(r) * std::f64::consts::PI).cos().is_nan())
            .collect();
        out.add_mask(n, &mask);
    }
    out
}

/// `PeriodicKernel.forward` on the rows of `x`: per-dimension distances of
/// `x / (p / π)` (`last_dim_is_batch`), `exp(-2 Σ sin²(dist) / ℓ)`.
pub(crate) fn periodic(x: &[f64], n: usize, k: usize, ls: &[f64], period: &[f64]) -> TrainingNan {
    let mut out = TrainingNan::default();
    let pick = |v: &[f64], j: usize| if v.len() == 1 { v[0] } else { v[j] };
    let mut sums: Option<Vec<f64>> = None;
    for j in 0..k {
        let div = pick(period, j) / std::f64::consts::PI;
        let col: Vec<f64> = (0..n).map(|i| x[i * k + j] / div).collect();
        if let Some(res) = sq_dist(&col, n, 1, false) {
            let s = sums.get_or_insert_with(|| vec![0.0; n * n]);
            let l = pick(ls, j);
            for (a, &r) in s.iter_mut().zip(&res) {
                let d = clamped_sqrt(r);
                let sn = d.sin();
                *a += sn * sn / l * -2.0;
            }
        }
    }
    if let Some(s) = sums {
        let mask: Vec<bool> = s.iter().map(|v| v.exp().is_nan()).collect();
        out.add_mask(n, &mask);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, usize, usize) {
        // 4 points in 2 dimensions.
        (vec![0.1, 0.2, 0.5, 0.9, 0.3, 0.3, 0.95, 0.05], 4, 2)
    }

    #[test]
    fn moderate_lengthscales_are_finite_without_work() {
        let (x, n, k) = data();
        for kind in [
            KernelKind::Rbf,
            KernelKind::Matern12,
            KernelKind::Matern32,
            KernelKind::Matern52,
        ] {
            let r = stationary(kind, &x, n, k, &[1e-6, 3.0], false);
            assert!(!r.any() && r.nan_gradients().is_empty());
        }
        let scaled: Vec<f64> = x.iter().map(|v| v / 1e-50).collect();
        assert!(sq_dist(&scaled, n, k, false).is_none());
    }

    #[test]
    fn overflowing_squares_give_nan_like_gpytorch() {
        // GPyTorch: ScaleKernel(RBFKernel) with lengthscales 1e-293 .. 1e-109
        // gives a NaN covariance (torch reference: all 16 entries NaN).
        let (x, n, k) = data();
        let r = stationary(KernelKind::Rbf, &x, n, k, &[1e-293, 0.5], false);
        assert!(r.any());
        let mut m = crate::Matrix::zeros(n, n);
        r.inject(&mut m);
        assert!(m[(0, 0)].is_nan() && m[(1, 1)].is_nan());
        // Squares of about 1e216 do not overflow: finite.
        let r = stationary(KernelKind::Rbf, &x, n, k, &[1e-108, 0.5], false);
        assert!(!r.any());
    }

    #[test]
    fn zero_lengthscales_give_nan() {
        let (x, n, k) = data();
        for kind in [KernelKind::Rbf, KernelKind::Matern52] {
            for function_path in [false, true] {
                let ls = if function_path {
                    vec![0.0]
                } else {
                    vec![0.0, 0.4]
                };
                let r = stationary(kind, &x, n, k, &ls, function_path);
                assert!(r.any(), "{kind:?} {function_path}");
            }
        }
    }

    #[test]
    fn function_path_gradients_are_nan_where_inf_times_zero_occurs() {
        // Two points on either side of their mean: the off-diagonal squared
        // distance overflows to +inf without NaN (the diagonal is zeroed), so
        // K is finite but `sq_dist · K / ℓ = inf · 0` is NaN.
        let x = vec![-0.5, 0.5];
        let r = stationary(KernelKind::Rbf, &x, 2, 1, &[1e-160], true);
        assert!(!r.any());
        assert_eq!(r.nan_gradients(), &[0]);
        // Matérn-5/2: `(1 + s) · s²/3 · exp(-s)` overflows far earlier.
        let r = stationary(KernelKind::Matern52, &x, 2, 1, &[1e-110], true);
        assert!(!r.any());
        assert_eq!(r.nan_gradients(), &[0]);
        // Matérn-3/2 with an infinite distance: (1 + inf) · 0 is NaN.
        let r = stationary(KernelKind::Matern32, &x, 2, 1, &[1e-160], true);
        assert!(r.any());
    }

    #[test]
    fn other_kernels() {
        let (x, n, k) = data();
        assert!(rational_quadratic(&x, n, k, &[1e-200], 0.7).any());
        assert!(!rational_quadratic(&x, n, k, &[1e-90], 0.7).any());
        assert!(cosine(&x, n, k, 1e-200).any());
        assert!(periodic(&x, n, k, &[0.5], &[1e-200]).any());
        assert!(!periodic(&x, n, k, &[0.5], &[0.3]).any());
    }
}
