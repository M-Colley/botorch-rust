//! `ToyRobust` (Daulton et al., 2022): a one-dimensional problem whose
//! Pareto front is sensitive to input noise. The first objective mixes a
//! linear and a sinusoidal function, the second is a modified Levy function.

use crate::test_functions::SyntheticTestFunction;

/// The negated objectives `-(f_1(x), f_2(x))`; `levy` is BoTorch's
/// two-dimensional `Levy` function.
pub(super) fn evaluate(levy: &SyntheticTestFunction, x: &[f64], out: &mut [f64]) {
    let v = x[0];
    let p1 = 2.4 - 10.0 * v - 0.1 * (v * v);
    let p2 = 2.0 * v - 0.1 * (v * v);
    let smoother = (v - 0.5) * (v - 0.5) + (30.0 * v).sin() * 0.1;
    // torch.sigmoid(a) = 1 / (1 + exp(-a)).
    let mask = 1.0 / (1.0 + (-((0.2 - v) / 0.005)).exp());
    let f1 = -(p1 * mask + p2 * (1.0 - mask) + smoother) * 30.0 + 30.0;
    // Levy at unnormalize([x, 0] * 0.95 + 0.03, [-10, 10]^2) (the first part
    // of the function is cut out), minus 0.75 x_0'^2.
    let u0 = (v * 0.95 + 0.03) * 20.0 + -10.0;
    let u1 = (0.0 * 0.95 + 0.03) * 20.0 + -10.0;
    let f2 = levy.value_unchecked(&[u0, u1]) - u0 * u0 * 0.75;
    out[0] = -f1;
    out[1] = -f2;
}
