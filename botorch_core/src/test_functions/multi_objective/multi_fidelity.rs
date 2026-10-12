//! Multi-objective multi-fidelity problems of Irshad et al. (2021)
//! (`botorch.test_functions.multi_objective_multi_fidelity`): `MOMFBraninCurrin`
//! and `MOMFPark`. The last input is the fidelity `s`.

use std::f64::consts::PI;

/// `MOMFBraninCurrin`: the negated, rescaled Branin and Currin functions with
/// fidelity-dependent coefficients.
pub(super) fn branin_currin(x: &[f64], out: &mut [f64]) {
    let (x1, x2, s) = (x[0], x[1], x[2]);
    // Modified Branin.
    let x11 = 15.0 * x1 - 5.0;
    let x22 = 15.0 * x2;
    let b = 5.1 / (4.0 * (PI * PI)) - 0.01 * (1.0 - s);
    let c = 5.0 / PI - 0.1 * (1.0 - s);
    let r = 6.0;
    let t = 1.0 / (8.0 * PI) + 0.05 * (1.0 - s);
    let inner = x22 - b * (x11 * x11) + c * x11 - r;
    let y = inner * inner + 10.0 * (1.0 - t) * x11.cos() + 10.0;
    let branin = (21.0 - y) / 22.0;
    // Modified Currin.
    let (x1_2, x1_3) = (x1 * x1, x1 * x1 * x1);
    let a = 2300.0 * x1_3 + 1900.0 * x1_2 + 2092.0 * x1 + 60.0;
    let bb = 100.0 * x1_3 + 500.0 * x1_2 + 4.0 * x1 + 20.0;
    // torch evaluates `-1 / t` as `t.reciprocal() * -1`, i.e. `-(1 / t)`.
    let e = (-(1.0 / (2.0 * x2))).exp();
    let yc = (1.0 - 0.1 * (1.0 - s) * e) * a / bb;
    let currin = (-yc + 14.0) / 15.0;
    out[0] = -branin;
    out[1] = -currin;
}

/// `MOMFPark`: the negated modified Park functions of the transformed inputs.
pub(super) fn park(x: &[f64], out: &mut [f64]) {
    let s = x[4];
    let x1 = 1.0 - 2.0 * ((x[0] - 0.6) * (x[0] - 0.6));
    let x2 = x[1];
    let x3 = 1.0 - 3.0 * ((x[2] - 0.5) * (x[2] - 0.5));
    let x4 = 1.0 - (x[3] - 0.8) * (x[3] - 0.8);
    let a = 0.9 + 0.1 * s;
    let b = 0.1 * (1.0 - s);
    let t1 = (x1 + 1e-3 * (1.0 - s)) / 2.0 * (1.0 + (x2 + x3 * x3) * x4 / (x1 * x1 + 1e-4)).sqrt();
    let t2 = (x1 + 3.0 * x4) * (1.0 + x3.sin()).exp();
    let park1 = a * (t1 + t2 - b) / 22.0 - 0.8;
    let park2 = a * (5.0 - 2.0 / 3.0 * (x1 + x2).exp() + x4 * x3.sin() * a - x3 + b) / 4.0 - 0.7;
    out[0] = -park1;
    out[1] = -park2;
}
