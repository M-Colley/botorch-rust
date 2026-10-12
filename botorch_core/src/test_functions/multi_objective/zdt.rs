//! The ZDT problems of Zitzler et al. (2000): `ZDT1`, `ZDT2` and `ZDT3` with
//! two objectives `f_0 = x_0` and `f_1 = g(x) h(f_0, g(x))`.

use std::f64::consts::PI;

/// `g(x) = 1 + 9 mean(x_1, ..., x_{d-1})`.
fn g(x: &[f64]) -> f64 {
    let rest = &x[1..];
    1.0 + 9.0 * (rest.iter().sum::<f64>() / rest.len() as f64)
}

/// ZDT1: `h = 1 - sqrt(f_0 / g)` (convex front).
pub(super) fn zdt1(x: &[f64], out: &mut [f64]) {
    let g = g(x);
    out[0] = x[0];
    out[1] = g * (1.0 - (x[0] / g).sqrt());
}

/// ZDT2: `h = 1 - (f_0 / g)^2` (concave front).
pub(super) fn zdt2(x: &[f64], out: &mut [f64]) {
    let g = g(x);
    let r = x[0] / g;
    out[0] = x[0];
    out[1] = g * (1.0 - r * r);
}

/// ZDT3: `h = 1 - sqrt(f_0 / g) - f_0 / g sin(10 pi f_0)` (disconnected
/// front). BoTorch main omits the factor `g` of `f_1` (fixed in BoTorch PR
/// #3402).
pub(super) fn zdt3(x: &[f64], out: &mut [f64]) {
    let g = g(x);
    let r = x[0] / g;
    out[0] = x[0];
    out[1] = g * (1.0 - r.sqrt() - r * (10.0 * PI * x[0]).sin());
}
