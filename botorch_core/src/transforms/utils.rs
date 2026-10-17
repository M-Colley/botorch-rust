//! Helpers of the transforms, mirroring `botorch.models.transforms.utils`:
//! the log-normal moment maps of `Log`'s `TransformedPosterior`
//! ([`norm_to_lognorm_mean`], [`norm_to_lognorm_variance`],
//! [`norm_to_lognorm`], [`lognorm_to_norm`]), the Kumaraswamy CDF of `Warp`
//! ([`kumaraswamy_warp`], [`inv_kumaraswamy_warp`]) with its derivatives,
//! [`interaction_features`] and [`nanstd`].

use crate::{Matrix, Vector};

/// The mean of `exp(X)` for `X ~ N(mu, var)`: `exp(mu + var / 2)`
/// (BoTorch's `norm_to_lognorm_mean`).
#[inline]
pub fn norm_to_lognorm_mean(mu: f64, var: f64) -> f64 {
    (mu + 0.5 * var).exp()
}

/// The variance of `exp(X)` for `X ~ N(mu, var)`:
/// `expm1(var) · exp(2 (mu + var / 2))` (BoTorch's
/// `norm_to_lognorm_variance`).
#[inline]
pub fn norm_to_lognorm_variance(mu: f64, var: f64) -> f64 {
    let b = mu + 0.5 * var;
    var.exp_m1() * (2.0 * b).exp()
}

/// The mean vector and covariance matrix of `exp(X)` for `X ~ N(mu, cov)`
/// (BoTorch's `norm_to_lognorm`):
/// `mu_ln_i = exp(b_i)`, `cov_ln_ij = expm1(cov_ij) exp(b_i + b_j)` with
/// `b = mu + diag(cov) / 2`.
///
/// # Panics
/// If `cov` is not `n × n` for `n = mu.len()`.
pub fn norm_to_lognorm(mu: &Vector, cov: &Matrix) -> (Vector, Matrix) {
    let n = mu.len();
    assert_eq!(cov.shape(), (n, n), "covariance must be n x n");
    let b = Vector::from_fn(n, |i, _| mu[i] + 0.5 * cov[(i, i)]);
    let mu_ln = b.map(f64::exp);
    let cov_ln = Matrix::from_fn(n, n, |i, j| cov[(i, j)].exp_m1() * (b[i] + b[j]).exp());
    (mu_ln, cov_ln)
}

/// The mean vector and covariance matrix of `log(Y)` for a log-normal `Y`
/// with mean `mu` and covariance `cov` (BoTorch's `lognorm_to_norm`):
/// `cov_n_ij = log1p(cov_ij / (mu_i mu_j))`,
/// `mu_n_i = log(mu_i) - cov_n_ii / 2`.
///
/// # Panics
/// If `cov` is not `n × n` for `n = mu.len()`.
pub fn lognorm_to_norm(mu: &Vector, cov: &Matrix) -> (Vector, Matrix) {
    let n = mu.len();
    assert_eq!(cov.shape(), (n, n), "covariance must be n x n");
    let cov_n = Matrix::from_fn(n, n, |i, j| (cov[(i, j)] / (mu[i] * mu[j])).ln_1p());
    let mu_n = Vector::from_fn(n, |i, _| mu[i].ln() - 0.5 * cov_n[(i, i)]);
    (mu_n, cov_n)
}

/// The Kumaraswamy CDF of BoTorch's `kumaraswamy_warp` at a single value:
/// the input is first mapped to `[eps, 1 - eps]`
/// (`clamp(x (1 - 2 eps) + eps, eps, 1 - eps)`), then
/// `1 - (1 - x^c1)^c0`.
#[inline]
pub fn kumaraswamy_warp(x: f64, c0: f64, c1: f64, eps: f64) -> f64 {
    let xr = clamp_unit(x, eps).0;
    1.0 - (1.0 - xr.powf(c1)).powf(c0)
}

/// `clamp(x (1 - 2 eps) + eps, eps, 1 - eps)` and the derivative of the
/// clamp with respect to `x` (torch's `clamp` passes the gradient on the
/// closed interval).
#[inline]
fn clamp_unit(x: f64, eps: f64) -> (f64, f64) {
    let range = 1.0 - 2.0 * eps;
    let y = x * range + eps;
    let (lo, hi) = (eps, 1.0 - eps);
    if y.is_nan() {
        (y, range)
    } else if y < lo {
        (lo, 0.0)
    } else if y > hi {
        (hi, 0.0)
    } else {
        (y, range)
    }
}

/// [`kumaraswamy_warp`] with its derivatives with respect to `x`, `c0` and
/// `c1` (what torch's autograd computes, including the clamp).
pub fn kumaraswamy_warp_with_grad(x: f64, c0: f64, c1: f64, eps: f64) -> (f64, f64, f64, f64) {
    let (xr, dxr) = clamp_unit(x, eps);
    let p = xr.powf(c1); // x^c1
    let u = 1.0 - p; // 1 - x^c1
    let up = u.powf(c0); // (1 - x^c1)^c0
    let value = 1.0 - up;
    // d/dc0: -(u^c0 ln u); d/dc1: c0 u^(c0-1) x^c1 ln x; d/dx: c0 u^(c0-1) c1 x^(c1-1).
    let u_c0m1 = u.powf(c0 - 1.0);
    let dc0 = -up * u.ln();
    let dc1 = c0 * u_c0m1 * p * xr.ln();
    let dx = c0 * u_c0m1 * c1 * xr.powf(c1 - 1.0) * dxr;
    (value, dx, dc0, dc1)
}

/// BoTorch's `inv_kumaraswamy_warp` at a single value:
/// `(1 - (1 - x)^(1/c0))^(1/c1)` mapped back from `[eps, 1 - eps]` and
/// clamped to `[0, 1]`.
#[inline]
pub fn inv_kumaraswamy_warp(x: f64, c0: f64, c1: f64, eps: f64) -> f64 {
    let range = 1.0 - 2.0 * eps;
    let untf = (1.0 - (1.0 - x).powf(1.0 / c0)).powf(1.0 / c1);
    let v = (untf - eps) / range;
    if v.is_nan() {
        v
    } else {
        v.clamp(0.0, 1.0)
    }
}

/// The first-order interaction features `x_i x_j` (`i < j`) of a point, in
/// the row-major order of the upper triangle (BoTorch's
/// `interaction_features`, `torch.triu_indices(d, d, offset=1)`).
pub fn interaction_features(x: &[f64]) -> Vec<f64> {
    let d = x.len();
    let mut out = Vec::with_capacity(d * d.saturating_sub(1) / 2);
    for i in 0..d {
        for j in i + 1..d {
            out.push(x[i] * x[j]);
        }
    }
    out
}

/// The standard deviation of `values` ignoring NaNs, with Bessel's
/// correction (BoTorch's `nanstd`): the mean of the squared deviations from
/// the NaN-mean, times `n / (n - 1)` for `n` non-NaN values, square-rooted.
/// NaN for fewer than two values.
pub fn nanstd(values: &[f64]) -> f64 {
    let (sum, count) = values
        .iter()
        .filter(|v| !v.is_nan())
        .fold((0.0, 0usize), |(s, c), &v| (s + v, c + 1));
    let n = count as f64;
    let mean = sum / n;
    let msd = values
        .iter()
        .filter(|v| !v.is_nan())
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / n;
    (msd * n / (n - 1.0)).sqrt()
}

/// The mean of `values` ignoring NaNs (torch's `nanmean`); NaN without
/// values.
pub fn nanmean(values: &[f64]) -> f64 {
    let (sum, count) = values
        .iter()
        .filter(|v| !v.is_nan())
        .fold((0.0, 0usize), |(s, c), &v| (s + v, c + 1));
    sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lognormal_round_trip() {
        let mu = Vector::from_vec(vec![0.3, -0.2]);
        let cov = Matrix::from_row_slice(2, 2, &[0.5, 0.1, 0.1, 0.2]);
        let (m, c) = norm_to_lognorm(&mu, &cov);
        assert!((m[0] - norm_to_lognorm_mean(0.3, 0.5)).abs() < 1e-15);
        assert!((c[(1, 1)] - norm_to_lognorm_variance(-0.2, 0.2)).abs() < 1e-15);
        let (m2, c2) = lognorm_to_norm(&m, &c);
        assert!((m2 - mu).amax() < 1e-14);
        assert!((c2 - cov).amax() < 1e-14);
    }

    #[test]
    fn kumaraswamy_identity_and_inverse() {
        // c0 = c1 = 1 is the identity up to the eps squeeze.
        let eps = 1e-7;
        let x = 0.37;
        assert!((kumaraswamy_warp(x, 1.0, 1.0, eps) - (x * (1.0 - 2.0 * eps) + eps)).abs() < 1e-15);
        let y = kumaraswamy_warp(x, 1.7, 0.6, eps);
        assert!((inv_kumaraswamy_warp(y, 1.7, 0.6, eps) - x).abs() < 1e-12);
        // Derivatives against central differences.
        let (v, dx, dc0, dc1) = kumaraswamy_warp_with_grad(x, 1.7, 0.6, eps);
        assert_eq!(v, y);
        let h = 1e-6;
        let fd = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
        assert!((fd(&|t| kumaraswamy_warp(x + t, 1.7, 0.6, eps)) - dx).abs() < 1e-8);
        assert!((fd(&|t| kumaraswamy_warp(x, 1.7 + t, 0.6, eps)) - dc0).abs() < 1e-8);
        assert!((fd(&|t| kumaraswamy_warp(x, 1.7, 0.6 + t, eps)) - dc1).abs() < 1e-8);
        // Outside the unit interval the clamp has a zero derivative.
        assert_eq!(kumaraswamy_warp_with_grad(1.5, 1.7, 0.6, eps).1, 0.0);
    }

    #[test]
    fn interaction_and_nanstd() {
        assert_eq!(interaction_features(&[1.0, 2.0, 3.0]), vec![2.0, 3.0, 6.0]);
        assert!(interaction_features(&[1.0]).is_empty());
        let s = nanstd(&[1.0, f64::NAN, 3.0, 5.0]);
        assert!((s - 2.0).abs() < 1e-15);
        assert!(nanstd(&[1.0]).is_nan());
        assert_eq!(nanmean(&[1.0, f64::NAN, 3.0]), 2.0);
    }
}
