//! `Penicillin` (Liang & Lai, 2021): a simulator of penicillin production in
//! a fed-batch fermenter, adapted by BoTorch from
//! <https://github.com/HarryQL/TuRBO-Penicillin>.
//!
//! The seven inputs are the culture volume `V`, the biomass concentration
//! `X`, the temperature `T`, the glucose concentration `S`, the substrate
//! feed rate `F`, the substrate feed concentration `s_f` and the H+
//! concentration exponent. The ODE is integrated with explicit Euler steps of
//! one time unit for at most 2500 steps; a point stops at the first step after
//! which the volume exceeds `V_max`, the glucose concentration is negative or
//! the penicillin production rate `dP/dt` is below `1e-11`, and that step is
//! the third objective (the time to ferment; 2500 if it never stops). The
//! objectives (all minimized) are the negated penicillin concentration, the
//! CO2 concentration and the time.
//!
//! BoTorch integrates all points at once, masking the stopped ones; the
//! points are independent, so they are simulated one by one here (and in
//! parallel). The temperature does not change, so the temperature- and
//! pH-dependent factors are computed once per point; the floating-point
//! operations are otherwise BoTorch's, in BoTorch's order, so that the step at
//! which a point stops is the same and the values agree to rounding errors
//! accumulated over the steps (about 1e-14 relative).

/// BoTorch's constants (class attributes; e.g. `K_2 = 7 * 10 ** (-5)`).
const Y_XS: f64 = 0.45;
const Y_PS: f64 = 0.90;
const K_1: f64 = 1e-10;
const K_2: f64 = 7.000000000000001e-05;
const M_X: f64 = 0.014;
const ALPHA_1: f64 = 0.143;
const ALPHA_2: f64 = 4e-07;
const ALPHA_3: f64 = 0.0001;
const MU_X: f64 = 0.092;
const K_X: f64 = 0.15;
const MU_P: f64 = 0.005;
const K_P: f64 = 0.0002;
const K_I: f64 = 0.10;
const K: f64 = 0.04;
const K_G: f64 = 7000.0;
const E_G: f64 = 5100.0;
const K_D: f64 = 1e33;
const E_D: f64 = 50000.0;
const LAMBD: f64 = 0.00025;
/// Temperatures in Kelvin.
const T_V: f64 = 273.0;
const T_O: f64 = 373.0;
/// The gas constant in cal / (mol K).
const R: f64 = 1.9872;
const V_MAX: f64 = 180.0;
/// The maximum number of time steps.
const STEPS: usize = 2500;
/// A point stops once `dP/dt < 10e-12` (i.e. `1e-11`).
const MIN_DP_DT: f64 = 10e-12;

/// The objectives `(-P, CO2, t)` at one point.
pub(super) fn evaluate(x: &[f64], out: &mut [f64]) {
    let (mut v, mut xb, temp, mut s, f, s_f, h_exp) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
    let mut p = 0.0;
    let mut co2 = 0.0;
    let h = 10f64.powf(-h_exp);
    // Factors that only depend on the (constant) temperature and pH. torch
    // evaluates `c / t` (Python float / tensor) as `t.reciprocal() * c`.
    let loss_rate = (5.0 * ((temp - T_O) / (T_V - T_O))).exp_m1();
    let ph_factor = (1.0 / (1.0 + (1.0 / h) * K_1 + h / K_2)) * MU_X;
    let inv_rt = 1.0 / (R * temp);
    let growth = K_G * (inv_rt * -E_G).exp() - K_D * (inv_rt * -E_D).exp();
    let mut t_end = STEPS as f64;
    for step in 1..=STEPS {
        let f_loss = v * LAMBD * loss_rate;
        let dv = f - f_loss;
        let mu = ph_factor * (s / (K_X * xb + s)) * growth;
        let dx = mu * xb - (xb / v) * dv;
        let mu_pp = MU_P * (s / (K_P + s + s * s / K_I));
        let ds = -(mu / Y_XS) * xb - (mu_pp / Y_PS) * xb - M_X * xb + f * s_f / v - (s / v) * dv;
        let dp = (mu_pp * xb) - K * p - (p / v) * dv;
        let dco2 = ALPHA_1 * dx + ALPHA_2 * xb + ALPHA_3;
        p += dp;
        v += dv;
        xb += dx;
        s += ds;
        co2 += dco2;
        if v > V_MAX || s < 0.0 || dp < MIN_DP_DT {
            t_end = step as f64;
            break;
        }
    }
    out[0] = -p;
    out[1] = co2;
    out[2] = t_end;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulations_stop_within_the_horizon() {
        let mut out = [0.0; 3];
        // Lower and upper corners of the domain, and a point in between.
        for x in [
            [60.0, 0.05, 293.0, 0.05, 0.01, 500.0, 5.0],
            [120.0, 18.0, 303.0, 18.0, 0.5, 700.0, 6.5],
            [90.0, 9.0, 298.0, 9.0, 0.25, 600.0, 5.75],
        ] {
            evaluate(&x, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
            assert!(out[2] >= 1.0 && out[2] <= STEPS as f64 && out[2].fract() == 0.0);
        }
    }
}
