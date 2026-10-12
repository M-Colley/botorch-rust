//! Models with one-to-many and chained input transforms
//! (`InputPerturbation`, `ChainedInputTransform`), independent of BoTorch:
//! the posterior equals the posterior of the same GP at the transformed
//! points, reverse- and forward-mode input derivatives agree with central
//! finite differences, the cached-baseline blocks agree with the joint
//! posterior, and multi-output and list models are block diagonal over the
//! perturbed points. BoTorch parity is in `mo_robust_parity.rs`.

use std::sync::Arc;

use botorch_core::models::BatchedMultiOutputGPOptions;
use botorch_core::transforms::input::{
    ChainedInputTransform, InputPerturbation, InputTransformStep, PerturbationFunction,
    PerturbationSet,
};
use botorch_core::{
    BatchedMultiOutputGP, BotorchError, Matrix, Model, ModelListGP, Normalize, SingleTaskGP,
    SingleTaskGPOptions, Vector,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn data(seed: u64, n: usize, d: usize, m: usize) -> (Matrix, Matrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Matrix::from_fn(n, d, |_, _| 0.2 + 1.5 * rng.random::<f64>());
    let y = Matrix::from_fn(n, m, |i, t| {
        let u: f64 = (0..d).map(|j| x[(i, j)]).sum();
        ((2.0 + t as f64) * u).sin() + 0.3 * u + 0.05 * rng.random::<f64>()
    });
    (x, y)
}

fn perturbation(d: usize, nw: usize, seed: u64) -> InputPerturbation {
    let mut rng = StdRng::seed_from_u64(seed);
    let p = Matrix::from_fn(nw, d, |_, _| 0.1 * (rng.random::<f64>() - 0.5));
    InputPerturbation::new(PerturbationSet::Fixed(p)).unwrap()
}

/// `Normalize(learned) -> InputPerturbation(bounds [0, 1])` and the plain
/// chain.
fn chains(d: usize, nw: usize) -> Vec<ChainedInputTransform> {
    let bounds = Matrix::from_fn(2, d, |i, _| i as f64);
    vec![
        ChainedInputTransform::new(vec![InputTransformStep::Perturbation(perturbation(
            d, nw, 1,
        ))])
        .unwrap(),
        ChainedInputTransform::new(vec![
            InputTransformStep::Normalize(Normalize::new(d)),
            InputTransformStep::Perturbation(
                perturbation(d, nw, 2).with_bounds(bounds.clone()).unwrap(),
            ),
        ])
        .unwrap(),
        ChainedInputTransform::new(vec![
            InputTransformStep::Perturbation(
                perturbation(1, nw, 3)
                    .with_multiplicative(true)
                    .with_indices(vec![d - 1])
                    .unwrap(),
            ),
            InputTransformStep::Normalize(Normalize::new(d)),
        ])
        .unwrap(),
    ]
}

fn perturbed_gp(seed: u64, chain: ChainedInputTransform) -> SingleTaskGP {
    let (x, y) = data(seed, 9, 2, 1);
    let mut gp = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            input_chain: Some(chain),
            ..Default::default()
        },
    )
    .unwrap();
    gp.set_raw_parameters(&[0.03, 0.2, 0.4, 0.7]).unwrap();
    gp
}

fn points(seed: u64, q: usize, d: usize) -> Matrix {
    let mut rng = StdRng::seed_from_u64(seed);
    Matrix::from_fn(q, d, |_, _| 0.3 + 1.2 * rng.random::<f64>())
}

#[test]
fn posterior_is_the_posterior_at_the_transformed_points() {
    for chain in chains(2, 3) {
        let gp = perturbed_gp(1, chain.clone());
        assert_eq!(gp.points_per_input(), 3);
        // The same GP on the preprocessed training inputs, without transform.
        let fitted = gp.input_chain().unwrap();
        let plain_x = fitted.preprocess(gp.train_inputs()).unwrap();
        assert_eq!(&plain_x, gp.transformed_train_inputs());
        let mut plain = SingleTaskGP::new(
            plain_x,
            gp.train_targets().clone(),
            SingleTaskGPOptions::default(),
        )
        .unwrap();
        plain
            .set_raw_parameters(gp.raw_parameters().as_slice())
            .unwrap();
        let x = points(2, 2, 2);
        let xt = fitted.transform(&x).unwrap();
        assert_eq!(xt.nrows(), 6);
        for noise in [false, true] {
            let a = gp.posterior(&x, noise).unwrap();
            let b = plain.posterior(&xt, noise).unwrap();
            assert_eq!(a, b);
        }
        // Marginals of all transformed points.
        let (m, v) = gp.posterior_marginals_unclamped(&x).unwrap();
        let (mp, vp) = plain.posterior_marginals_unclamped(&xt).unwrap();
        assert_eq!((m, v), (mp, vp));
        // Methods that evaluate inputs on their own need one point per input.
        assert!(matches!(
            Model::point_posteriors(&gp, &x),
            Err(BotorchError::Unsupported(_))
        ));
        assert!(gp.posterior_q1_with_grad(&[0.5, 0.5]).is_err());
    }
}

/// Central differences of `⟨ma, μ(x)⟩ + ⟨ca, Σ(x)⟩` for rows `rows`.
fn check_vjp(model: &dyn Model, x: &Matrix, rows: std::ops::Range<usize>) {
    let n = x.nrows() * model.points_per_input() * model.num_outputs();
    let ma = Vector::from_fn(n, |i, _| 0.3 - 0.07 * i as f64);
    let ca = Matrix::from_fn(n, n, |i, j| ((i * 3 + j * 5) % 7) as f64 / 7.0 - 0.4);
    let g = model
        .joint_posterior_vjp(x, rows.clone(), &ma, &ca)
        .unwrap();
    let f = |xp: &Matrix| {
        let p = model.posterior(xp, false).unwrap();
        ma.dot(&p.mean) + ca.component_mul(&p.covariance).sum()
    };
    let h = 1e-6;
    for (r, i) in rows.clone().enumerate() {
        for j in 0..x.ncols() {
            let mut xp = x.clone();
            xp[(i, j)] += h;
            let mut xm = x.clone();
            xm[(i, j)] -= h;
            let fd = (f(&xp) - f(&xm)) / (2.0 * h);
            assert!(
                (fd - g[(r, j)]).abs() < 1e-6 * fd.abs().max(1.0),
                "row {i}, dim {j}: {} vs {fd}",
                g[(r, j)]
            );
        }
    }
    // Forward mode agrees with the reverse mode.
    let jg = model.joint_posterior_with_grad(x).unwrap();
    assert_eq!(jg.mean.len(), n);
    for (r, i) in rows.enumerate() {
        for j in 0..x.ncols() {
            let k = i * x.ncols() + j;
            let v = ma.dot(&jg.dmean[k]) + ca.component_mul(&jg.dcov[k]).sum();
            assert!((v - g[(r, j)]).abs() < 1e-11 * v.abs().max(1.0));
        }
    }
}

#[test]
fn input_derivatives() {
    for chain in chains(2, 3) {
        let gp = perturbed_gp(3, chain);
        let x = points(4, 3, 2);
        check_vjp(&gp, &x, 0..3);
        check_vjp(&gp, &x, 1..3);
    }
}

#[test]
fn baseline_blocks_and_their_derivatives() {
    let gp = perturbed_gp(5, chains(2, 2).remove(1));
    let xb = points(6, 4, 2);
    let x = points(7, 2, 2);
    let cache = gp.baseline_cache(&xb).unwrap();
    assert_eq!(cache.len(), 8);
    let cp = gp.cross_posterior(&cache, &x).unwrap();
    let mut full = Matrix::zeros(6, 2);
    full.rows_mut(0, 4).copy_from(&xb);
    full.rows_mut(4, 2).copy_from(&x);
    let joint = gp.posterior(&full, false).unwrap();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12 * b.abs().max(1.0);
    for i in 0..4 {
        assert!(close(cp.mean[i], joint.mean[8 + i]));
        for j in 0..8 {
            assert!(close(cp.cross[(i, j)], joint.covariance[(8 + i, j)]));
        }
        for j in 0..4 {
            assert!(close(
                cp.covariance[(i, j)],
                joint.covariance[(8 + i, 8 + j)]
            ));
        }
    }
    // Reverse mode of the blocks.
    let ma = Vector::from_fn(4, |i, _| 0.2 * i as f64 - 0.3);
    let cb = Matrix::from_fn(4, 8, |i, j| ((i + 2 * j) % 5) as f64 / 5.0 - 0.3);
    let ca = Matrix::from_fn(4, 4, |i, j| ((3 * i + j) % 4) as f64 / 4.0);
    let g = gp.cross_posterior_vjp(&cache, &x, &ma, &cb, &ca).unwrap();
    let f = |xp: &Matrix| {
        let c = gp.cross_posterior(&cache, xp).unwrap();
        ma.dot(&c.mean) + cb.component_mul(&c.cross).sum() + ca.component_mul(&c.covariance).sum()
    };
    let h = 1e-6;
    for i in 0..2 {
        for j in 0..2 {
            let mut xp = x.clone();
            xp[(i, j)] += h;
            let mut xm = x.clone();
            xm[(i, j)] -= h;
            let fd = (f(&xp) - f(&xm)) / (2.0 * h);
            assert!((fd - g[(i, j)]).abs() < 1e-6 * fd.abs().max(1.0));
        }
    }
}

#[test]
fn multi_output_and_list_models() {
    let (x, y) = data(8, 10, 2, 2);
    let mo = BatchedMultiOutputGP::new(
        vec![x.clone()],
        vec![y.clone()],
        Vec::new(),
        BatchedMultiOutputGPOptions {
            input_chain: Some(chains(2, 3).remove(1)),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(Model::points_per_input(&mo), 3);
    let xq = points(9, 2, 2);
    let post = mo.posterior(&xq, false).unwrap();
    assert_eq!(post.num_points(), 6);
    for t in 0..2 {
        let single = mo.model(0, t).posterior(&xq, false).unwrap();
        assert_eq!(post.select_outputs(&[t]).unwrap().mean, single.mean);
    }
    assert_eq!(post.covariance[(0, 6)], 0.0);
    check_vjp(&mo, &xq, 0..2);
    // Bounds learned from all batch elements: the elements share them.
    let fitted = mo.input_chain().unwrap();
    match &fitted.steps()[0] {
        InputTransformStep::Normalize(n) => {
            assert!(n.learn_bounds());
            assert_eq!(n.offset()[0], x.column(0).min());
        }
        _ => panic!("expected Normalize"),
    }
    // A list of a perturbed multi-output and a perturbed single-output model.
    let (x2, y2) = data(10, 7, 2, 1);
    let so = BatchedMultiOutputGP::new(
        vec![x2],
        vec![y2],
        Vec::new(),
        BatchedMultiOutputGPOptions {
            input_chain: Some(chains(2, 3).remove(0)),
            ..Default::default()
        },
    )
    .unwrap();
    let list = ModelListGP::new(vec![mo.clone(), so]).unwrap();
    assert_eq!(Model::points_per_input(&list), 3);
    let lp = Model::posterior(&list, &xq, false).unwrap();
    assert_eq!(lp.num_points(), 6);
    assert_eq!(lp.num_outputs(), 3);
    check_vjp(&list, &xq, 0..2);
    assert!(Model::point_posteriors(&list, &xq).is_err());
    // Sub-models with different numbers of perturbations are rejected.
    let (x3, y3) = data(11, 7, 2, 1);
    let other = BatchedMultiOutputGP::new(
        vec![x3],
        vec![y3],
        Vec::new(),
        BatchedMultiOutputGPOptions {
            input_chain: Some(chains(2, 2).remove(0)),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(ModelListGP::new(vec![mo, other]).is_err());
}

#[derive(Debug)]
struct Heteroscedastic;

impl PerturbationFunction for Heteroscedastic {
    fn perturbations(&self, x: &Matrix) -> botorch_core::Result<Matrix> {
        Ok(Matrix::from_fn(2 * x.nrows(), x.ncols(), |r, c| {
            let v = x[(r / 2, c)];
            if r % 2 == 0 {
                0.05 * v * v
            } else {
                -0.08 * v
            }
        }))
    }

    fn vjp(&self, x: &Matrix, adjoint: &Matrix) -> botorch_core::Result<Matrix> {
        Ok(Matrix::from_fn(x.nrows(), x.ncols(), |i, c| {
            0.1 * x[(i, c)] * adjoint[(2 * i, c)] - 0.08 * adjoint[(2 * i + 1, c)]
        }))
    }
}

#[test]
fn heteroscedastic_perturbations() {
    let chain = ChainedInputTransform::new(vec![
        InputTransformStep::Normalize(Normalize::new(2)),
        InputTransformStep::Perturbation(
            InputPerturbation::new(PerturbationSet::Function(Arc::new(Heteroscedastic))).unwrap(),
        ),
    ])
    .unwrap();
    let gp = perturbed_gp(12, chain);
    assert_eq!(gp.points_per_input(), 2);
    check_vjp(&gp, &points(13, 3, 2), 0..3);
}

#[test]
fn rejected_configurations() {
    let (x, y) = data(14, 6, 2, 1);
    let options = |chain| SingleTaskGPOptions {
        input_chain: Some(chain),
        ..Default::default()
    };
    // Perturbing the training inputs.
    let train = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
        perturbation(2, 3, 1).with_transform_flags(true, true, false),
    )])
    .unwrap();
    assert!(matches!(
        SingleTaskGP::new(x.clone(), y.clone(), options(train)),
        Err(BotorchError::Unsupported(_))
    ));
    // A perturbation set of the wrong width.
    let wide = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(perturbation(
        3, 3, 1,
    ))])
    .unwrap();
    assert!(SingleTaskGP::new(x.clone(), y.clone(), options(wide)).is_err());
    // Both kinds of input transforms.
    let both = SingleTaskGPOptions {
        input_transform: Some(Normalize::new(2)),
        input_chain: Some(chains(2, 2).remove(0)),
        ..Default::default()
    };
    assert!(SingleTaskGP::new(x.clone(), y.clone(), both).is_err());
    // A perturbation that does not apply in eval mode is the identity.
    let off = ChainedInputTransform::new(vec![InputTransformStep::Perturbation(
        perturbation(2, 3, 1).with_transform_flags(false, false, false),
    )])
    .unwrap();
    let gp = SingleTaskGP::new(x.clone(), y.clone(), options(off)).unwrap();
    assert_eq!(gp.points_per_input(), 1);
    let plain = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
    let xq = points(15, 2, 2);
    assert_eq!(
        gp.posterior(&xq, false).unwrap(),
        plain.posterior(&xq, false).unwrap()
    );
    assert!(Model::point_posteriors(&gp, &xq).is_ok());
}

#[test]
fn conditioning_and_fantasies_need_one_input_per_observation() {
    let chain = chains(2, 3).remove(0);
    let gp = perturbed_gp(16, chain);
    assert!(gp.points_per_input() > 1);
    let xn = points(17, 2, 2);
    let yn = Matrix::from_element(2, 1, 0.5);
    assert!(matches!(
        gp.condition_on_observations(&xn, &yn, None),
        Err(BotorchError::Unsupported(_))
    ));
    let sampler = botorch_core::sampling::SobolQMCNormalSampler::new(4, Some(0)).unwrap();
    assert!(matches!(
        gp.fantasize(&xn, &sampler, None),
        Err(BotorchError::Unsupported(_))
    ));
    let batched = BatchedMultiOutputGP::from(gp);
    let xb = botorch_core::models::BatchedMatrix::new(Vec::new(), vec![xn]).unwrap();
    assert!(matches!(
        batched.fantasize(&xb, &sampler, None),
        Err(BotorchError::Unsupported(_))
    ));
}
