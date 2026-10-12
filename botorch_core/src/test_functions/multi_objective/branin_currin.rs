//! `BraninCurrin` and the constraint of `ConstrainedBraninCurrin`.

use crate::test_functions::SyntheticTestFunction;

/// Currin's function on `[0, 1]^2` (BoTorch's `BraninCurrin._currin`).
pub(super) fn currin(x0: f64, x1: f64) -> f64 {
    // torch evaluates `-1 / t` as `t.reciprocal() * -1`, i.e. `-(1 / t)`.
    let factor1 = 1.0 - (-(1.0 / (2.0 * x1))).exp();
    let (x0_2, x0_3) = (x0 * x0, x0 * x0 * x0);
    let numer = 2300.0 * x0_3 + 1900.0 * x0_2 + 2092.0 * x0 + 60.0;
    let denom = 100.0 * x0_3 + 500.0 * x0_2 + 4.0 * x0 + 20.0;
    factor1 * numer / denom
}

/// `(Branin(15 x_0 - 5, 15 x_1), Currin(x_0, x_1))`.
pub(super) fn evaluate(branin: &SyntheticTestFunction, x: &[f64], out: &mut [f64]) {
    out[0] = branin.value_unchecked(&[15.0 * x[0] - 5.0, 15.0 * x[1]]);
    out[1] = currin(x[0], x[1]);
}

/// The disk constraint of Gelbart et al. (2014) on Branin's domain:
/// `50 - (x_0' - 2.5)^2 - (x_1' - 7.5)^2 >= 0` with `x' = unnormalize(x)`.
pub(super) fn constraint_slack(x: &[f64], out: &mut [f64]) {
    // `unnormalize(X, [(-5, 10), (0, 15)])` = `X * (upper - lower) + lower`.
    let u0 = x[0] * 15.0 + -5.0;
    let u1 = x[1] * 15.0 + 0.0;
    out[0] = 50.0 - (u0 - 2.5) * (u0 - 2.5) - (u1 - 7.5) * (u1 - 7.5);
}
