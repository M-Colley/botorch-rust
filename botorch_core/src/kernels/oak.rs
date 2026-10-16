//! Helpers of BoTorch's `OrthogonalAdditiveKernel`
//! (`botorch.models.kernels.orthogonal_additive_kernel`): Gauss-Legendre
//! quadrature (`leggauss`) and the second-order index helpers.

/// The minimum of the normalizers of the orthogonalized base kernels
/// (`OrthogonalAdditiveKernel.normalizer(eps=1e-6)`).
pub const OAK_NORMALIZER_EPS: f64 = 1e-6;

/// Legendre polynomial `P_n(x)` and its derivative.
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// Gauss-Legendre nodes and weights of degree `deg` on `[-1, 1]`
/// (`numpy.polynomial.legendre.leggauss`): ascending nodes, symmetrized
/// like NumPy's, weights normalized to sum to 2.
pub fn leggauss(deg: usize) -> (Vec<f64>, Vec<f64>) {
    let n = deg;
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        // Newton's method from Tricomi's initial guess (descending roots).
        let mut r = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre(n, r);
            let step = p / dp;
            r -= step;
            if step.abs() <= 1e-17 * r.abs().max(1.0) {
                break;
            }
        }
        let (_, dp) = legendre(n, r);
        x[n - 1 - i] = r;
        w[n - 1 - i] = 2.0 / ((1.0 - r * r) * dp * dp);
    }
    // NumPy symmetrizes the nodes and weights and rescales the weights.
    let xs: Vec<f64> = (0..n).map(|i| (x[i] - x[n - 1 - i]) / 2.0).collect();
    let ws: Vec<f64> = (0..n).map(|i| (w[i] + w[n - 1 - i]) / 2.0).collect();
    let total: f64 = ws.iter().sum();
    let ws = ws.iter().map(|v| v * (2.0 / total)).collect();
    (xs, ws)
}

/// BoTorch's `leggauss(deg, a=0, b=1)`: the nodes `(x + 1) / 2` and weights
/// `w / 2` of [`leggauss`] on `[0, 1]`.
pub fn gauss_legendre_01(deg: usize) -> (Vec<f64>, Vec<f64>) {
    let (x, w) = leggauss(deg);
    (
        x.iter().map(|v| (v + 1.0) / 2.0).collect(),
        w.iter().map(|v| v * 0.5).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_polynomials_exactly() {
        for deg in [1, 2, 5, 8, 32] {
            let (x, w) = gauss_legendre_01(deg);
            for p in 0..(2 * deg) {
                let q: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(p as i32)).sum();
                let exact = 1.0 / (p as f64 + 1.0);
                assert!((q - exact).abs() < 1e-14, "deg {deg} p {p}: {q} vs {exact}");
            }
            assert!(x.windows(2).all(|p| p[0] < p[1]));
        }
        // numpy.polynomial.legendre.leggauss(3) and (8) (NumPy's weights are
        // accurate to a few ulps; these are closer to the exact values).
        let (x, w) = leggauss(3);
        assert!((x[0] + 0.7745966692414834).abs() < 2e-16);
        assert_eq!(x[1], 0.0);
        assert!((w[0] - 0.5555555555555557).abs() < 4e-16, "{}", w[0]);
        assert!((w[1] - 0.8888888888888888).abs() < 4e-16, "{}", w[1]);
        let (x, w) = leggauss(8);
        for (a, e) in x
            .iter()
            .zip([-0.9602898564975362, -0.7966664774136267, -0.525532409916329])
        {
            assert!((a - e).abs() < 1e-15, "{a} {e}");
        }
        for (a, e) in w
            .iter()
            .zip([0.10122853629037706, 0.22238103445337443, 0.3137066458778869])
        {
            assert!((a - e).abs() < 1e-14 * e, "{a} {e}");
        }
    }
}
