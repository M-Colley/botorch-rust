//! Finite-difference checks of every analytic gradient: the MLL with respect
//! to all raw hyperparameters, kernel matrices with respect to hyperparameters
//! and inputs, and posterior moments with respect to the (raw-space) test
//! inputs for q = 1 and q > 1, through Normalize and Standardize; plus
//! consistency of the reverse-mode posterior derivative (`joint_posterior_vjp`)
//! with the forward-mode one and of the batched marginals with the q = 1
//! posteriors.

use botorch_core::kernels::get_matern_kernel_with_gamma_prior;
use botorch_core::{
    get_gaussian_likelihood_with_gamma_prior, ExactMarginalLogLikelihood, Kernel, KernelKind,
    Matrix, Model, Normalize, OutcomeTransformChoice, SingleTaskGP, SingleTaskGPOptions, Vector,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Deterministic test data: `n` points in `[lo, hi]^d` and a smooth response
/// with an offset and scale (so that Standardize is non-trivial).
fn data(seed: u64, n: usize, d: usize, lo: f64, hi: f64) -> (Matrix, Matrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Matrix::from_fn(n, d, |_, _| lo + (hi - lo) * rng.random::<f64>());
    let y = Matrix::from_fn(n, 1, |i, _| {
        let u: f64 = (0..d).map(|j| (x[(i, j)] - lo) / (hi - lo)).sum();
        4.0 + 2.5 * ((3.0 * u).sin() + 0.3 * u * u) + 0.05 * rng.random::<f64>()
    });
    (x, y)
}

/// A collection of model configurations covering all parameterizations.
fn models() -> Vec<(&'static str, SingleTaskGP)> {
    let mut out = Vec::new();
    let (x, y) = data(1, 12, 3, 0.0, 1.0);
    out.push((
        "rbf",
        SingleTaskGP::new(x.clone(), y.clone(), SingleTaskGPOptions::default()).unwrap(),
    ));
    out.push((
        "matern52",
        SingleTaskGP::new(
            x.clone(),
            y.clone(),
            SingleTaskGPOptions {
                kernel: KernelKind::Matern52,
                ..Default::default()
            },
        )
        .unwrap(),
    ));
    out.push((
        "rbf + scale kernel",
        SingleTaskGP::new(
            x.clone(),
            y.clone(),
            SingleTaskGPOptions {
                use_scale_kernel: true,
                ..Default::default()
            },
        )
        .unwrap(),
    ));
    out.push((
        "legacy gamma priors (softplus constraints)",
        SingleTaskGP::new(
            x.clone(),
            y.clone(),
            SingleTaskGPOptions {
                covar_module: Some(get_matern_kernel_with_gamma_prior(3)),
                likelihood: Some(get_gaussian_likelihood_with_gamma_prior()),
                ..Default::default()
            },
        )
        .unwrap(),
    ));
    let yvar = Matrix::from_fn(12, 1, |i, _| 0.01 + 0.002 * i as f64);
    out.push((
        "fixed noise",
        SingleTaskGP::new(
            x.clone(),
            y.clone(),
            SingleTaskGPOptions {
                train_yvar: Some(yvar),
                ..Default::default()
            },
        )
        .unwrap(),
    ));
    let (xn, yn) = data(2, 10, 2, -3.0, 5.0);
    out.push((
        "normalize + matern",
        SingleTaskGP::new(
            xn.clone(),
            yn.clone(),
            SingleTaskGPOptions {
                input_transform: Some(Normalize::new(2)),
                kernel: KernelKind::Matern52,
                ..Default::default()
            },
        )
        .unwrap(),
    ));
    out.push((
        "normalize (bounds) + no outcome transform",
        SingleTaskGP::new(
            xn,
            yn,
            SingleTaskGPOptions {
                input_transform: Some(
                    Normalize::with_bounds(&Matrix::from_row_slice(2, 2, &[-4.0, -4.0, 6.0, 6.0]))
                        .unwrap(),
                ),
                outcome_transform: OutcomeTransformChoice::None,
                ..Default::default()
            },
        )
        .unwrap(),
    ));
    out
}

/// Non-default hyperparameters (so that prior and likelihood gradients are not
/// accidentally zero).
fn perturbed(model: &SingleTaskGP, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let layout = model.parameter_layout();
    let mut p = model.raw_parameters().as_slice().to_vec();
    if let Some(i) = layout.noise_index() {
        p[i] = if p[i] > 1.0 {
            0.03
        } else {
            0.03 + 0.02 * rng.random::<f64>()
        };
    }
    p[layout.constant_index()] = 0.2 * rng.random::<f64>() - 0.1;
    if let Some(i) = layout.outputscale_index() {
        p[i] += 0.3;
    }
    for i in layout.lengthscale_range() {
        p[i] = if model
            .covar_module()
            .as_standard()
            .unwrap()
            .lengthscale_constraint
            .enforced()
        {
            -0.5 + rng.random::<f64>()
        } else {
            0.2 + 0.6 * rng.random::<f64>()
        };
    }
    p
}

fn assert_close(what: &str, analytic: f64, fd: f64, rtol: f64, atol: f64) {
    assert!(
        (analytic - fd).abs() <= rtol * analytic.abs().max(fd.abs()) + atol,
        "{what}: analytic {analytic:e} vs finite difference {fd:e}"
    );
}

#[test]
fn mll_gradient_matches_finite_differences() {
    for (name, model) in models() {
        let mll = ExactMarginalLogLikelihood::new(&model);
        for seed in 0..3 {
            let p = perturbed(&model, seed);
            let (_, grad) = mll.value_and_grad(&p).unwrap();
            for i in 0..p.len() {
                let h = 1e-6 * p[i].abs().max(1.0);
                let mut pp = p.clone();
                pp[i] += h;
                let mut pm = p.clone();
                pm[i] -= h;
                let fd = (mll.value_at(&pp).unwrap() - mll.value_at(&pm).unwrap()) / (2.0 * h);
                assert_close(
                    &format!("{name}: d mll / d theta[{i}]"),
                    grad[i],
                    fd,
                    1e-6,
                    1e-8,
                );
            }
        }
    }
}

#[test]
fn kernel_gradients_match_finite_differences() {
    let (x, _) = data(3, 7, 3, 0.0, 1.0);
    for kind in [KernelKind::Rbf, KernelKind::Matern52] {
        let ls = vec![0.3, 0.6, 1.1];
        let s = 1.4;
        let k = Kernel::new(kind, Vector::from_vec(ls.clone()), Some(s)).unwrap();
        let grads = k.hyperparameter_gradients(&x).unwrap();
        for m in 0..3 {
            let h = 1e-6;
            let mut lp = ls.clone();
            lp[m] += h;
            let mut lm = ls.clone();
            lm[m] -= h;
            let kp = Kernel::new(kind, Vector::from_vec(lp), Some(s))
                .unwrap()
                .covariance_symmetric(&x)
                .unwrap();
            let km = Kernel::new(kind, Vector::from_vec(lm), Some(s))
                .unwrap()
                .covariance_symmetric(&x)
                .unwrap();
            let fd = (kp - km) / (2.0 * h);
            assert!(
                (fd - &grads.lengthscale[m]).amax() < 1e-8,
                "{kind:?} dK/dl[{m}]"
            );
        }
        let h = 1e-6;
        let kp = Kernel::new(kind, Vector::from_vec(ls.clone()), Some(s + h))
            .unwrap()
            .covariance_symmetric(&x)
            .unwrap();
        let km = Kernel::new(kind, Vector::from_vec(ls.clone()), Some(s - h))
            .unwrap()
            .covariance_symmetric(&x)
            .unwrap();
        let fd = (kp - km) / (2.0 * h);
        assert!(
            (fd - grads.outputscale.unwrap()).amax() < 1e-8,
            "{kind:?} dK/ds"
        );

        // Gradient with respect to the first argument.
        let pt = [0.37, 0.52, 0.18];
        let (_, dk) = k.covariance_with_grad_x(&pt, &x).unwrap();
        for m in 0..3 {
            let mut xp = pt;
            xp[m] += h;
            let mut xm = pt;
            xm[m] -= h;
            let fd = (k.covariance_with_grad_x(&xp, &x).unwrap().0
                - k.covariance_with_grad_x(&xm, &x).unwrap().0)
                / (2.0 * h);
            assert!((fd - dk.column(m)).amax() < 1e-8, "{kind:?} dk/dx[{m}]");
        }
    }
}

fn test_points(model: &SingleTaskGP, q: usize, seed: u64) -> Matrix {
    // Points spanning (and slightly exceeding) the range of the training inputs.
    let x = model.train_inputs();
    let mut rng = StdRng::seed_from_u64(seed);
    Matrix::from_fn(q, x.ncols(), |_, j| {
        let col = x.column(j);
        let (lo, hi) = (col.min(), col.max());
        lo - 0.1 * (hi - lo) + 1.2 * (hi - lo) * rng.random::<f64>()
    })
}

#[test]
fn q1_posterior_gradients_match_finite_differences() {
    for (name, mut model) in models() {
        let p = perturbed(&model, 7);
        model.set_raw_parameters(&p).unwrap();
        let xs = test_points(&model, 5, 11);
        for r in 0..xs.nrows() {
            let x: Vec<f64> = xs.row(r).iter().copied().collect();
            let (mean, var, dmean, dvar) = model.posterior_q1_with_grad(&x).unwrap();
            let post = model
                .posterior(&Matrix::from_row_slice(1, x.len(), &x), false)
                .unwrap();
            assert_close(&format!("{name}: mean"), mean, post.mean[0], 1e-12, 1e-14);
            assert_close(
                &format!("{name}: variance"),
                var,
                post.variance()[0],
                1e-10,
                1e-14,
            );
            for k in 0..x.len() {
                let h = 1e-6 * x[k].abs().max(1.0);
                let mut xp = x.clone();
                xp[k] += h;
                let mut xm = x.clone();
                xm[k] -= h;
                let (mp, vp, _, _) = model.posterior_q1_with_grad(&xp).unwrap();
                let (mm, vm, _, _) = model.posterior_q1_with_grad(&xm).unwrap();
                assert_close(
                    &format!("{name}: dmean/dx[{k}]"),
                    dmean[k],
                    (mp - mm) / (2.0 * h),
                    1e-6,
                    1e-8,
                );
                assert_close(
                    &format!("{name}: dvar/dx[{k}]"),
                    dvar[k],
                    (vp - vm) / (2.0 * h),
                    1e-6,
                    1e-8,
                );
            }
        }
    }
}

#[test]
fn joint_posterior_gradients_match_finite_differences() {
    for (name, mut model) in models() {
        let p = perturbed(&model, 5);
        model.set_raw_parameters(&p).unwrap();
        let q = 3;
        let xs = test_points(&model, q, 13);
        let d = xs.ncols();
        let g = model.joint_posterior_with_grad(&xs).unwrap();
        let post = model.posterior(&xs, false).unwrap();
        assert!((&g.mean - &post.mean).amax() < 1e-12);
        assert!((&g.covariance - &post.covariance).amax() < 1e-12);
        for pidx in 0..q {
            for k in 0..d {
                let j = pidx * d + k;
                let h = 1e-6 * xs[(pidx, k)].abs().max(1.0);
                let mut xp = xs.clone();
                xp[(pidx, k)] += h;
                let mut xm = xs.clone();
                xm[(pidx, k)] -= h;
                let pp = model.posterior(&xp, false).unwrap();
                let pm = model.posterior(&xm, false).unwrap();
                let fd_mean = (pp.mean - pm.mean) / (2.0 * h);
                let fd_cov = (pp.covariance - pm.covariance) / (2.0 * h);
                for a in 0..q {
                    assert_close(
                        &format!("{name}: dmean[{a}]/dx[{pidx},{k}]"),
                        g.dmean[j][a],
                        fd_mean[a],
                        1e-6,
                        1e-8,
                    );
                    for b in 0..q {
                        assert_close(
                            &format!("{name}: dcov[{a},{b}]/dx[{pidx},{k}]"),
                            g.dcov[j][(a, b)],
                            fd_cov[(a, b)],
                            1e-6,
                            1e-8,
                        );
                    }
                }
            }
        }
    }
}

/// Forward-mode reference for the VJP: contracts `joint_posterior_with_grad`.
fn vjp_reference(
    model: &SingleTaskGP,
    x: &Matrix,
    rows: std::ops::Range<usize>,
    mean_adj: &Vector,
    cov_adj: &Matrix,
) -> Matrix {
    let d = x.ncols();
    let g = model.joint_posterior_with_grad(x).unwrap();
    let mut out = Matrix::zeros(rows.len(), d);
    for (r, p) in rows.enumerate() {
        for k in 0..d {
            let j = p * d + k;
            out[(r, k)] = mean_adj.dot(&g.dmean[j]) + cov_adj.component_mul(&g.dcov[j]).sum();
        }
    }
    out
}

#[test]
fn joint_posterior_vjp_matches_forward_mode() {
    for (name, mut model) in models() {
        let p = perturbed(&model, 9);
        model.set_raw_parameters(&p).unwrap();
        let n = 5;
        let xs = test_points(&model, n, 17);
        let mut rng = StdRng::seed_from_u64(3);
        let mean_adj = Vector::from_fn(n, |_, _| rng.random::<f64>() - 0.5);
        // A non-symmetric adjoint: only its symmetric part may matter.
        let cov_adj = Matrix::from_fn(n, n, |_, _| rng.random::<f64>() - 0.5);
        for rows in [0..n, 2..5, 0..1, 4..5, 3..3] {
            let vjp = model
                .joint_posterior_vjp(&xs, rows.clone(), &mean_adj, &cov_adj)
                .unwrap();
            let reference = vjp_reference(&model, &xs, rows.clone(), &mean_adj, &cov_adj);
            assert_eq!(vjp.shape(), (rows.len(), xs.ncols()));
            let scale = reference.amax().max(1e-300);
            assert!(
                (&vjp - &reference).amax() <= 1e-12 * scale,
                "{name}: vjp rows {rows:?}: {vjp} vs {reference}"
            );
            // The trait's default implementation (forward mode) agrees as well.
            let via_trait = <SingleTaskGP as Model>::joint_posterior_vjp(
                &model,
                &xs,
                rows.clone(),
                &mean_adj,
                &cov_adj,
            )
            .unwrap();
            assert!((&via_trait - &vjp).amax() <= 1e-12 * scale);
        }
    }
}

#[test]
fn joint_posterior_vjp_validates_shapes() {
    let (_, model) = models().remove(0);
    let xs = test_points(&model, 3, 1);
    let ma = Vector::zeros(3);
    let ca = Matrix::zeros(3, 3);
    assert!(model.joint_posterior_vjp(&xs, 0..4, &ma, &ca).is_err());
    assert!(model
        .joint_posterior_vjp(&xs, 0..3, &Vector::zeros(2), &ca)
        .is_err());
    assert!(model
        .joint_posterior_vjp(&xs, 0..3, &ma, &Matrix::zeros(3, 2))
        .is_err());
}

#[test]
fn posterior_marginals_match_q1_posteriors() {
    for (name, mut model) in models() {
        let p = perturbed(&model, 4);
        model.set_raw_parameters(&p).unwrap();
        let mut xs = test_points(&model, 6, 23);
        // Include a training point (variance clamped for tiny noise).
        xs.set_row(5, &model.train_inputs().row(0));
        let (mean, var) = model.posterior_marginals(&xs).unwrap();
        let (dmean, dvar) = Model::posterior_marginals(&model, &xs).unwrap();
        assert_eq!(mean, dmean);
        assert_eq!(var, dvar);
        for i in 0..xs.nrows() {
            let x: Vec<f64> = xs.row(i).iter().copied().collect();
            let (m, v, _, _) = model.posterior_q1_with_grad(&x).unwrap();
            assert_eq!(m, mean[i], "{name}: mean {i}");
            assert_eq!(v, var[i], "{name}: variance {i}");
            let post = model.posterior(&xs.rows(i, 1).into_owned(), false).unwrap();
            assert_close(
                &format!("{name}: mean"),
                mean[i],
                post.mean[0],
                1e-13,
                1e-14,
            );
            assert_close(
                &format!("{name}: var"),
                var[i],
                post.variance()[0],
                1e-10,
                1e-14,
            );
        }
    }
}
