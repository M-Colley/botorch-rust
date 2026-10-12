//! The DTLZ problems of Deb et al. (2005): `DTLZ1`, `DTLZ2`, `DTLZ3`,
//! `DTLZ4`, `DTLZ5` and `DTLZ7` with `M` objectives and `d > M` inputs; the
//! last `k = d - M + 1` inputs are the distance variables `X_m`.

use std::f64::consts::{FRAC_PI_2, PI};

/// The exponent of the position variables of DTLZ4.
const DTLZ4_ALPHA: f64 = 100.0;

/// `f_i = scale * prod_{j < M-1-i} cos(theta_j pi / 2) * sin(theta_{M-1-i} pi / 2)`
/// (no sine for `i = 0`): the spherical objectives of DTLZ2-5.
fn sphere_objectives(scale: f64, theta: impl Fn(usize) -> f64, m: usize, out: &mut [f64]) {
    for (i, o) in out.iter_mut().enumerate().take(m) {
        let idx = m - 1 - i;
        let prod: f64 = (0..idx).map(|j| (theta(j) * FRAC_PI_2).cos()).product();
        let mut f = scale * prod;
        if i > 0 {
            f *= (theta(idx) * FRAC_PI_2).sin();
        }
        *o = f;
    }
}

/// `sum_{X_m} (x - 0.5)^2`.
fn g_sphere(x_m: &[f64]) -> f64 {
    x_m.iter().map(|&v| (v - 0.5) * (v - 0.5)).sum()
}

/// `100 (k + sum_{X_m} ((x - 0.5)^2 - cos(20 pi (x - 0.5))))`: Rastrigin-like
/// distance function of DTLZ1 and DTLZ3.
fn g_rastrigin(x_m: &[f64]) -> f64 {
    let sum: f64 = x_m
        .iter()
        .map(|&v| {
            let t = v - 0.5;
            t * t - (20.0 * PI * t).cos()
        })
        .sum();
    100.0 * (x_m.len() as f64 + sum)
}

/// DTLZ1: `f_i = 0.5 (1 + g) prod_{j < M-1-i} x_j (1 - x_{M-1-i})`.
pub(super) fn dtlz1(x: &[f64], m: usize, out: &mut [f64]) {
    let k = x.len() - m + 1;
    let g_term = 0.5 * (1.0 + g_rastrigin(&x[x.len() - k..]));
    for (i, o) in out.iter_mut().enumerate().take(m) {
        let idx = m - 1 - i;
        let prod: f64 = x[..idx].iter().product();
        let mut f = g_term * prod;
        if i > 0 {
            f *= 1.0 - x[idx];
        }
        *o = f;
    }
}

/// DTLZ2: spherical objectives with `g = sum (x - 0.5)^2`.
pub(super) fn dtlz2(x: &[f64], m: usize, out: &mut [f64]) {
    let k = x.len() - m + 1;
    let g = g_sphere(&x[x.len() - k..]);
    sphere_objectives(1.0 + g, |j| x[j], m, out);
}

/// DTLZ3: spherical objectives with DTLZ1's distance function.
pub(super) fn dtlz3(x: &[f64], m: usize, out: &mut [f64]) {
    let k = x.len() - m + 1;
    let g = g_rastrigin(&x[x.len() - k..]);
    sphere_objectives(1.0 + g, |j| x[j], m, out);
}

/// DTLZ4: DTLZ2 of the position variables mapped to `x_j^100`
/// (BoTorch PR #3402; BoTorch main's DTLZ4 equals DTLZ2).
pub(super) fn dtlz4(x: &[f64], m: usize, out: &mut [f64]) {
    let k = x.len() - m + 1;
    let g = g_sphere(&x[x.len() - k..]);
    sphere_objectives(1.0 + g, |j| x[j].powf(DTLZ4_ALPHA), m, out);
}

/// DTLZ5: spherical objectives of the angles
/// `theta_j = (1 + 2 g x_j) / (2 (1 + g))` (`theta_0 = x_0`).
pub(super) fn dtlz5(x: &[f64], m: usize, out: &mut [f64]) {
    let k = x.len() - m + 1;
    let g = g_sphere(&x[x.len() - k..]);
    if m == 1 {
        // BoTorch prepends x_0 to the (here empty) angles, so that its single
        // objective is (1 + g) cos(x_0 pi / 2).
        out[0] = (g + 1.0) * (x[0] * FRAC_PI_2).cos();
        return;
    }
    // torch evaluates `1 / t` as `t.reciprocal() * 1`.
    let c = 1.0 / (2.0 * (1.0 + g));
    let theta = |j: usize| {
        if j == 0 {
            x[0]
        } else {
            c * (1.0 + 2.0 * g * x[j])
        }
    };
    sphere_objectives(g + 1.0, theta, m, out);
}

/// DTLZ7: `f_i = x_i` for `i < M - 1` and `f_{M-1} = (1 + g) h` with
/// `g = 1 + 9 / k sum X_m` and `h = M - sum_i f_i / (1 + g) (1 + sin(3 pi f_i))`.
pub(super) fn dtlz7(x: &[f64], m: usize, out: &mut [f64]) {
    let k = x.len() - m + 1;
    let g = 1.0 + 9.0 / k as f64 * x[x.len() - k..].iter().sum::<f64>();
    let sum: f64 = x[..m - 1]
        .iter()
        .map(|&f| f / (1.0 + g) * (1.0 + (3.0 * PI * f).sin()))
        .sum();
    let h = m as f64 - sum;
    out[..m - 1].copy_from_slice(&x[..m - 1]);
    out[m - 1] = (1.0 + g) * h;
}

/// The radius of the feasible regions of C2-DTLZ2 for `M` objectives (Jain
/// & Deb, 2014; 0.2 for `M = 2`, as in pymoo).
pub(super) fn c2dtlz2_radius(m: usize) -> f64 {
    match m {
        2 => 0.2,
        3 => 0.4,
        _ => 0.5,
    }
}

/// `min(a, b)` that propagates NaN (`torch.minimum`).
fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// The constraint slack of C2-DTLZ2 at the objectives `f` (Jain & Deb, 2014):
/// `-(min(min_i [(f_i - 1)^2 + sum_{j != i} f_j^2], sum_i (f_i - 1 / sqrt(M))^2)
/// - r^2)`, i.e. feasible within distance `r` of a unit vector `e_i` or of
/// `(1, ..., 1) / sqrt(M)`. BoTorch main subtracts `r^2` inside the sums and
/// uses `r = 0.2` for all `M` (fixed in BoTorch PR #3402).
pub(super) fn c2dtlz2_slack_at_objectives(f: &[f64]) -> f64 {
    let m = f.len();
    let r = c2dtlz2_radius(m);
    let sq_sum: f64 = f.iter().map(|v| v * v).sum();
    let to_unit_vectors = f
        .iter()
        .map(|&v| (v - 1.0) * (v - 1.0) + sq_sum - v * v)
        .fold(f64::INFINITY, nan_min);
    let c = 1.0 / (m as f64).sqrt();
    let to_center: f64 = f.iter().map(|&v| (v - c) * (v - c)).sum();
    -(nan_min(to_unit_vectors, to_center) - r * r)
}

/// C2-DTLZ2: the slack at the DTLZ2 objectives of `x`.
pub(super) fn c2dtlz2_slack(x: &[f64], m: usize, out: &mut [f64]) {
    let mut buf = [0.0; 16];
    let mut heap = Vec::new();
    let f = if m <= buf.len() {
        &mut buf[..m]
    } else {
        heap.resize(m, 0.0);
        &mut heap[..]
    };
    dtlz2(x, m, f);
    out[0] = c2dtlz2_slack_at_objectives(f);
}

/// The maximum hypervolume of C2-DTLZ2 for `M = 2` (`None` otherwise): the
/// feasible Pareto front consists of the arcs of the unit circle with angles
/// in `[0, a]`, `[pi/4 - a, pi/4 + a]` and `[pi/2 - a, pi/2]` with
/// `a = 2 asin(r / 2)`; the hypervolume of the quarter circle is reduced by
/// the area between the circle and the two corners that bridge the gaps
/// (BoTorch PR #3402; BoTorch main uses an NSGA-II estimate for all `M`).
pub(super) fn c2dtlz2_max_hv(m: usize) -> Option<f64> {
    if m != 2 {
        return None;
    }
    let a = 2.0 * (c2dtlz2_radius(m) / 2.0).asin();
    // The integral of sqrt(1 - t^2) from lo to hi.
    let antiderivative = |t: f64| 0.5 * (t * (1.0 - t * t).sqrt() + t.asin());
    let area_under_circle = |lo: f64, hi: f64| antiderivative(hi) - antiderivative(lo);
    let (lo, hi) = (a, PI / 4.0 - a);
    let gap = (lo.cos() - hi.cos()) * (hi.sin() - lo.sin())
        - (area_under_circle(hi.cos(), lo.cos()) - lo.sin() * (lo.cos() - hi.cos()));
    Some(1.1f64.powf(2.0) - PI / 4.0 - 2.0 * gap)
}

/// `Gamma(n / 2)` for `n >= 1`, by the exact recursions from `Gamma(1) = 1`
/// and `Gamma(1/2) = sqrt(pi)`.
fn gamma_half(n: usize) -> f64 {
    if n.is_multiple_of(2) {
        (1..n / 2).map(|j| j as f64).product()
    } else {
        (0..n / 2).fold(PI.sqrt(), |acc, j| acc * (j as f64 + 0.5))
    }
}

/// `pi^(M/2) / Gamma(M/2 + 1) / 2^M`: the volume of the positive orthant of
/// the unit ball in `R^M`, i.e. the hypervolume dominated by the Pareto front
/// of DTLZ2-4 with respect to the origin.
pub(super) fn positive_ball_volume(m: usize) -> f64 {
    PI.powf(m as f64 / 2.0) / gamma_half(m + 2) / 2f64.powi(m as i32)
}

/// `0.5^M / M!`: the volume of the simplex `{f >= 0: sum f_i <= 0.5}` (the
/// Pareto front of DTLZ1; BoTorch PR #3402 adds the factor `1 / M!`).
pub(super) fn dtlz1_simplex_volume(m: usize) -> f64 {
    let factorial = if m <= 34 {
        // Exact in u128 and correctly rounded, like Python's int-to-float.
        (1..=m as u128).product::<u128>() as f64
    } else {
        (1..=m).map(|j| j as f64).product()
    };
    0.5f64.powf(m as f64) / factorial
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_and_volumes() {
        // Gamma(M/2 + 1) as computed by scipy.special.gamma.
        let expected = [
            0.886_226_925_452_757_9,
            1.0,
            1.329_340_388_179_137,
            2.0,
            3.323_350_970_447_843,
            6.0,
            11.631_728_396_567_45,
            24.0,
        ];
        for (m, e) in expected.iter().enumerate() {
            let g = gamma_half(m + 3);
            assert!(
                (g - e).abs() <= 2.0 * f64::EPSILON * e,
                "M={}: {g} vs {e}",
                m + 1
            );
        }
        // The quarter disk and the octant of the unit ball.
        assert!((positive_ball_volume(2) - PI / 4.0).abs() < 1e-15);
        assert!((positive_ball_volume(3) - PI / 6.0).abs() < 1e-15);
        assert_eq!(dtlz1_simplex_volume(2), 0.125);
        assert_eq!(dtlz1_simplex_volume(3), 0.125 / 6.0);
    }

    #[test]
    fn pareto_optimal_points_lie_on_the_fronts() {
        // x_m = 0.5 (DTLZ1-5) and x_m = 0 (DTLZ7) are Pareto optimal.
        let mut f = [0.0; 3];
        let x = [0.3, 0.8, 0.5, 0.5, 0.5];
        dtlz1(&x, 3, &mut f);
        assert!((f.iter().sum::<f64>() - 0.5).abs() < 1e-15);
        for eval in [dtlz2, dtlz3, dtlz4, dtlz5] {
            eval(&x, 3, &mut f);
            assert!((f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-15);
        }
        dtlz7(&[0.3, 0.8, 0.0, 0.0, 0.0], 3, &mut f);
        assert_eq!(&f[..2], &[0.3, 0.8]);
    }

    #[test]
    fn c2dtlz2_feasible_regions() {
        // On the front, feasible near e_1 and the center, not in between.
        let r = c2dtlz2_radius(2);
        assert!(c2dtlz2_slack_at_objectives(&[1.0, 0.0]) > 0.0);
        let c = std::f64::consts::FRAC_1_SQRT_2;
        assert!((c2dtlz2_slack_at_objectives(&[c, c]) - r * r).abs() < 1e-15);
        let t = 0.3f64; // angle beyond a = 2 asin(0.1) = 0.2003
        assert!(c2dtlz2_slack_at_objectives(&[t.cos(), t.sin()]) < 0.0);
        // Below the hypervolume of the unconstrained front (BoTorch PR #3402:
        // 0.3823336).
        let hv = c2dtlz2_max_hv(2).unwrap();
        assert!(hv < 1.21 - PI / 4.0 && (hv - 0.3823336).abs() < 1e-7);
        assert!(c2dtlz2_max_hv(3).is_none());
    }
}
