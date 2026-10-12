//! Real-world engineering problems: `VehicleSafety` (vehicle
//! crash-worthiness), `CarSideImpact` and `DiscBrake` from the suite of
//! Tanabe & Ishibuchi (2020), and `WeldedBeam` (adapted by BoTorch from
//! pymoo).
//!
//! The coefficients and the order of the operations are BoTorch's; e.g.
//! `0.1106 * X1.pow(2)` is `0.1106 * (x1 * x1)`.

/// `VehicleSafety`: mass, acceleration and toe-board intrusion of a vehicle
/// as functions of the thicknesses of five reinforced members.
pub(super) fn vehicle_safety(x: &[f64], out: &mut [f64]) {
    let (x1, x2, x3, x4, x5) = (x[0], x[1], x[2], x[3], x[4]);
    out[0] = 1640.2823
        + 2.3573285 * x1
        + 2.3220035 * x2
        + 4.5688768 * x3
        + 7.7213633 * x4
        + 4.4559504 * x5;
    out[1] = 6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.8364 * x4 - 0.3695 * x1 * x4
        + 0.0861 * x1 * x5
        + 0.3628 * x2 * x4
        - 0.1106 * (x1 * x1)
        - 0.3437 * (x3 * x3)
        + 0.1764 * (x4 * x4);
    out[2] = -0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 - 0.0073 * x1 * x2 + 0.024 * x2 * x3
        - 0.0118 * x2 * x4
        - 0.0204 * x3 * x4
        - 0.008 * x3 * x5
        - 0.0241 * (x2 * x2)
        + 0.0109 * (x4 * x4);
}

/// `CarSideImpact`: weight, pubic force, average velocity of the V-pillars
/// and the sum of the violations of ten constraints (the fourth objective).
pub(super) fn car_side_impact(x: &[f64], out: &mut [f64]) {
    let (x1, x2, x3, x4, x5, x6, x7) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
    let f1 =
        1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7;
    let f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3;
    let v_mbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2;
    let v_fd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6;
    let f3 = 0.5 * (v_mbp + v_fd);
    let g = [
        1.0 - 1.16 + 0.3717 * x2 * x4 + 0.0092928 * x3,
        0.32 - 0.261 + 0.0159 * x1 * x2 + 0.06486 * x1 + 0.019 * x2 * x7
            - 0.0144 * x3 * x5
            - 0.0154464 * x6,
        0.32 - 0.214 - 0.00817 * x5 + 0.045195 * x1 + 0.0135168 * x1 - 0.03099 * x2 * x6
            + 0.018 * x2 * x7
            - 0.007176 * x3
            - 0.023232 * x3
            + 0.00364 * x5 * x6
            + 0.018 * (x2 * x2),
        0.32 - 0.74 + 0.61 * x2 + 0.031296 * x3 + 0.031872 * x7 - 0.227 * (x2 * x2),
        32.0 - 28.98 - 3.818 * x3 + 4.2 * x1 * x2 - 1.27296 * x6 + 2.68065 * x7,
        32.0 - 33.86 - 2.95 * x3 + 5.057 * x1 * x2 + 3.795 * x2 + 3.4431 * x7 - 1.45728,
        32.0 - 46.36 + 9.9 * x2 + 4.4505 * x1,
        4.0 - f2,
        9.9 - v_mbp,
        15.7 - v_fd,
    ];
    // The total violation: sum of -g_i over the violated constraints.
    let f4: f64 = g.iter().map(|&v| if v < 0.0 { -v } else { 0.0 }).sum();
    out.copy_from_slice(&[f1, f2, f3, f4]);
}

/// `DiscBrake`: the mass of the brake and the stopping time, as functions of
/// the inner and outer radii, the engaging force and the number of friction
/// surfaces.
pub(super) fn disc_brake(x: &[f64], out: &mut [f64]) {
    let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
    let sq_diff = x2 * x2 - x1 * x1;
    out[0] = 4.9 * 1e-5 * sq_diff * (x4 - 1.0);
    out[1] = (9.82 * 1e6) * sq_diff / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1));
}

/// `DiscBrake`: four constraints (the minimum distance between the radii,
/// the maximum pressure, temperature and torque).
// `3.14` is BoTorch's (and Tanabe & Ishibuchi's) literal, not pi.
#[allow(clippy::approx_constant)]
pub(super) fn disc_brake_slack(x: &[f64], out: &mut [f64]) {
    let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
    let sq_diff = x2 * x2 - x1 * x1;
    let cub_diff = x2 * x2 * x2 - x1 * x1 * x1;
    out[0] = x2 - x1 - 20.0;
    out[1] = 0.4 - x3 / (3.14 * sq_diff);
    out[2] = 1.0 - (2.22 * 1e-3 * x3 * cub_diff) / (sq_diff * sq_diff);
    out[3] = (2.66 * 1e-2 * x3 * x4 * cub_diff) / sq_diff - 900.0;
}

/// `WeldedBeam`: the cost of the beam and its end deflection.
pub(super) fn welded_beam(x: &[f64], out: &mut [f64]) {
    let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
    out[0] = 1.10471 * (x1 * x1) * x2 + 0.04811 * x3 * x4 * (14.0 + x2);
    // torch evaluates `c / t` (Python float / tensor) as `t.reciprocal() * c`.
    out[1] = (1.0 / (x4 * (x3 * x3 * x3))) * 2.1952;
}

/// `WeldedBeam`: shear stress, normal stress, the order of the thicknesses
/// and the buckling load, normalized. These are pymoo's constraint values
/// (feasible if non-positive), negated as in BoTorch PR #3402: BoTorch main
/// returns them unchanged, so that its feasibility is inverted.
pub(super) fn welded_beam_slack(x: &[f64], out: &mut [f64]) {
    let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
    let p = 6000.0;
    let l = 14.0;
    let t_max = 13600.0;
    let s_max = 30000.0;
    let r = (0.25 * (x2 * x2 + (x1 + x3) * (x1 + x3))).sqrt();
    let m = p * (l + x2 / 2.0);
    let j = 2.0 * 0.5f64.sqrt() * x1 * x2 * (x2 * x2 / 12.0 + 0.25 * ((x1 + x3) * (x1 + x3)));
    let t1 = (1.0 / (2f64.sqrt() * x1 * x2)) * p;
    let t2 = m * r / j;
    let t = (t1 * t1 + t1 * t2 * x2 / r + t2 * t2).sqrt();
    let s = (1.0 / (x4 * (x3 * x3))) * (6.0 * p * l);
    let p_c = 64746.022 * (1.0 - 0.0282346 * x3) * x3 * (x4 * x4 * x4);
    let g = [
        (t - t_max) / t_max,
        (s - s_max) / s_max,
        1.0 / (5.0 - 0.125) * (x1 - x4),
        (p - p_c) / p,
    ];
    for (o, v) in out.iter_mut().zip(g) {
        *o = -v;
    }
}
