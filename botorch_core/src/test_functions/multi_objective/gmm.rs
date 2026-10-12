//! `GMM` (Frohlich et al., 2020; Daulton et al., 2022): each of the `M`
//! objectives (2 to 4) is a negated mixture of three isotropic Gaussian
//! densities on `[0, 1]^2`.
//!
//! The parameters are float64, as in BoTorch PR #3405: BoTorch main creates
//! them in float32, so that it evaluates float64 inputs in float32 and
//! returns float32 values.

use std::f64::consts::PI;

/// Means of the components (`[objective][component]`).
const POSITIONS: [[[f64; 2]; 3]; 4] = [
    [[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]],
    [[0.07, 0.2], [0.4, 0.8], [0.85, 0.1]],
    [[0.08, 0.21], [0.45, 0.75], [0.86, 0.11]],
    [[0.09, 0.19], [0.44, 0.72], [0.89, 0.13]],
];
/// Standard deviations of the components.
const STDS: [[f64; 3]; 4] = [
    [0.2, 0.1, 0.1],
    [0.2, 0.1, 0.05],
    [0.2, 0.1, 0.07],
    [0.15, 0.07, 0.09],
];
/// Weights of the normalized densities (`norm = 2 pi var weight`).
const WEIGHTS: [[f64; 3]; 4] = [
    [0.5, 0.7, 0.7],
    [0.5, 0.7, 0.7],
    [0.5, 0.7, 0.9],
    [0.5, 0.7, 0.9],
];

/// `-sum_k norm_k N(x; mu_k, var_k I)` for each objective, computed like
/// torch's `MultivariateNormal.log_prob` (Cholesky factor `sqrt(var) I`).
pub(super) fn evaluate(x: &[f64], m: usize, out: &mut [f64]) {
    let log_2pi_times_2 = 2.0 * (2.0 * PI).ln();
    for (i, o) in out.iter_mut().enumerate().take(m) {
        let mut sum = 0.0;
        for k in 0..3 {
            let var = STDS[i][k] * STDS[i][k];
            let norm = 2.0 * PI * var * WEIGHTS[i][k];
            let l = var.sqrt();
            let z0 = (x[0] - POSITIONS[i][k][0]) / l;
            let z1 = (x[1] - POSITIONS[i][k][1]) / l;
            let half_log_det = l.ln() + l.ln();
            let log_prob = -0.5 * (log_2pi_times_2 + (z0 * z0 + z1 * z1)) - half_log_det;
            sum += norm * log_prob.exp();
        }
        *o = -sum;
    }
}

/// BoTorch's reference point (the first `M` entries).
pub(super) const REF_POINT: [f64; 4] = [-0.2338, -0.2211, -0.5180, -0.1866];
