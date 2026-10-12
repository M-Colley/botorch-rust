//! The DH problems of Deb & Gupta (2005) for robust multi-objective
//! optimization (`DH1`, `DH2`, `DH3`, `DH4`).

use std::f64::consts::PI;

/// `torch.nan_to_num`: NaN to 0 and infinities to the largest finite values.
fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        f64::MIN
    } else {
        v
    }
}

/// DH1 (`beta = 1`) and DH2 (`beta = 10`) with `alpha = 1`:
/// `f_1 = h(x_0) + g(x) S(x_0)`.
pub(super) fn dh1(x: &[f64], beta: f64, out: &mut [f64]) {
    let alpha = 1.0;
    let x0 = x[0];
    let h = 1.0 - x0 * x0;
    let g: f64 = x[1..]
        .iter()
        .map(|&v| 10.0 + v * v - 10.0 * (4.0 * PI * v).cos())
        .sum();
    // torch evaluates `alpha / t` as `t.reciprocal() * alpha`.
    let s = (1.0 / (0.2 + x0)) * alpha + beta * (x0 * x0);
    out[0] = x0;
    out[1] = h + nan_to_num(g * s);
}

/// `-((x - 0.35) / 0.25)^2` and `-((x - 0.85) / 0.03)^2`.
fn exp_args(v: f64) -> (f64, f64) {
    let a = (v - 0.35) / 0.25;
    let b = (v - 0.85) / 0.03;
    (-(a * a), -(b * b))
}

/// DH3 (`h` depends on `x_1`) and DH4 (`h` depends on `x_0 + x_1` and
/// `x_0`): `f_1 = h (g(x) + S(x_0))`.
pub(super) fn dh3(x: &[f64], dh4: bool, out: &mut [f64]) {
    let x0 = x[0];
    let h = if dh4 {
        let (a1, a2) = exp_args(x0 + x[1]);
        2.0 - x0 - 0.8 * a1.exp() - a2.exp()
    } else {
        let (a1, a2) = exp_args(x[1]);
        2.0 - 0.8 * a1.exp() - a2.exp()
    };
    let g = 50.0 * x[2..].iter().map(|&v| v * v).sum::<f64>();
    let s = 1.0 - x0.sqrt();
    out[0] = x0;
    out[1] = h * (g + s);
}
