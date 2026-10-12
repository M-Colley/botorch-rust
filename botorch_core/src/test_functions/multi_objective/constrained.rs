//! Constrained multi-objective problems with closed-form objectives and
//! constraints: `BNH`, `CONSTR`, `SRN` (from Garrido-Merchán &
//! Hernández-Lobato, 2020), `OSY` (Osyczka & Kundu, 1995) and `MW7` (Ma &
//! Wang, 2019). The slacks are feasible if non-negative.

/// `BNH`: `(4 |x|^2, |x - 5|^2)`.
pub(super) fn bnh(x: &[f64], out: &mut [f64]) {
    out[0] = 4.0 * (x[0] * x[0] + x[1] * x[1]);
    out[1] = (x[0] - 5.0) * (x[0] - 5.0) + (x[1] - 5.0) * (x[1] - 5.0);
}

/// `BNH`: `25 - (x_0 - 5)^2 - x_1^2` and `(x_0 - 8)^2 + (x_1 + 3)^2 - 7.7`.
pub(super) fn bnh_slack(x: &[f64], out: &mut [f64]) {
    out[0] = 25.0 - (x[0] - 5.0) * (x[0] - 5.0) - x[1] * x[1];
    out[1] = (x[0] - 8.0) * (x[0] - 8.0) + (x[1] + 3.0) * (x[1] + 3.0) - 7.7;
}

/// `CONSTR`: `(x_0, (1 + x_1) / x_0)`.
pub(super) fn constr(x: &[f64], out: &mut [f64]) {
    out[0] = x[0];
    out[1] = (1.0 + x[1]) / x[0];
}

/// `CONSTR`: `9 x_0 + x_1 - 6` and `9 x_0 - x_1 - 1`.
pub(super) fn constr_slack(x: &[f64], out: &mut [f64]) {
    out[0] = 9.0 * x[0] + x[1] - 6.0;
    out[1] = 9.0 * x[0] - x[1] - 1.0;
}

/// `OSY`: `-(25 (x_0 - 2)^2 + (x_1 - 2)^2 + (x_2 - 1)^2 + (x_3 - 4)^2 +
/// (x_4 - 1)^2)` and `|x|^2`.
pub(super) fn osy(x: &[f64], out: &mut [f64]) {
    let sq = |v: f64| v * v;
    out[0] = -(25.0 * sq(x[0] - 2.0)
        + sq(x[1] - 2.0)
        + sq(x[2] - 1.0)
        + sq(x[3] - 4.0)
        + sq(x[4] - 1.0));
    out[1] = x.iter().map(|&v| v * v).sum();
}

/// `OSY`: six linear and quadratic constraints.
pub(super) fn osy_slack(x: &[f64], out: &mut [f64]) {
    out[0] = x[0] + x[1] - 2.0;
    out[1] = 6.0 - x[0] - x[1];
    out[2] = 2.0 - x[1] + x[0];
    out[3] = 2.0 - x[0] + 3.0 * x[1];
    out[4] = 4.0 - (x[2] - 3.0) * (x[2] - 3.0) - x[3];
    out[5] = (x[4] - 3.0) * (x[4] - 3.0) + x[5] - 4.0;
}

/// `SRN` (Srinivas & Deb, 1994): `(2 + (x_0 - 2)^2 + (x_1 - 1)^2,
/// 9 x_0 - (x_1 - 1)^2)`. BoTorch main uses `(x_1 - 2)^2` in the first
/// objective (fixed in BoTorch PR #3402).
pub(super) fn srn(x: &[f64], out: &mut [f64]) {
    out[0] = 2.0 + (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 1.0) * (x[1] - 1.0);
    out[1] = 9.0 * x[0] - (x[1] - 1.0) * (x[1] - 1.0);
}

/// `SRN`: `225 - x_0^2 - x_1^2` and `3 x_1 - x_0 - 10`. BoTorch main uses
/// `225 - x_0^4 - x_1^4` (fixed in BoTorch PR #3402).
pub(super) fn srn_slack(x: &[f64], out: &mut [f64]) {
    out[0] = 225.0 - (x[0] * x[0] + x[1] * x[1]);
    out[1] = -10.0 - x[0] + 3.0 * x[1];
}

/// `MW7` (adapted by BoTorch from pymoo): `f_0 = g x_0` and
/// `f_1 = g sqrt(1 - (f_0 / g)^2)` with
/// `g = 1 + sum_i 2 (x_{i+1} + (x_i - 0.5)^2 - 1)^2`.
pub(super) fn mw7(x: &[f64], out: &mut [f64]) {
    let g = 1.0
        + x.windows(2)
            .map(|w| {
                let a = w[0] - 0.5;
                let t = w[1] + a * a - 1.0;
                2.0 * (t * t)
            })
            .sum::<f64>();
    let f0 = g * x[0];
    let r = f0 / g;
    out[0] = f0;
    out[1] = g * (1.0 - r * r).sqrt();
}

/// pymoo's `LA2(A, B, C, D, theta) = A sin(B theta^C)^D`.
fn la2(a: f64, b: f64, c: f64, d: f64, theta: f64) -> f64 {
    a * (b * theta.powf(c)).sin().powf(d)
}

/// `MW7`: the negated pymoo constraints (two rings in objective space).
pub(super) fn mw7_slack(x: &[f64], out: &mut [f64]) {
    let mut f = [0.0; 2];
    mw7(x, &mut f);
    let (f0, f1) = (f[0], f[1]);
    let atan = (f1 / f0).atan();
    let outer = 1.2 + la2(0.4, 4.0, 1.0, 16.0, atan).abs();
    let g0 = f0 * f0 + f1 * f1 - outer * outer;
    let inner = 1.15 - la2(0.2, 4.0, 1.0, 8.0, atan);
    let g1 = inner * inner - f0 * f0 - f1 * f1;
    out[0] = -g0;
    out[1] = -g1;
}
