//! Regression tests for bug fixes and cache/performance invariants across
//! `botorch_core`.

use std::collections::BTreeMap;
use std::sync::Arc;

use botorch_core::acquisition::cost_aware::InverseCostWeightedUtility;
use botorch_core::acquisition::multi_objective::{
    ExpectedHypervolumeImprovement, QHypervolumeKnowledgeGradient,
    QHypervolumeKnowledgeGradientOptions, QNoisyExpectedHypervolumeImprovement,
    QNoisyExpectedHypervolumeImprovementOptions,
};
use botorch_core::acquisition::wrappers::{
    FixedFeatureAcquisitionFunction, ProximalAcquisitionFunction,
};
use botorch_core::acquisition::{
    AcquisitionFunction, LogExpectedImprovement, QExpectedImprovement,
};
use botorch_core::models::cost::{AffineFidelityCostModel, CostModel};
use botorch_core::models::multitask::{KroneckerMultiTaskGP, MultiTaskGP, MultiTaskGPOptions};
use botorch_core::models::BoolMatrix;
use botorch_core::optim::mixed::{get_nearest_neighbors, optimize_acqf_mixed, DiscreteDims};
use botorch_core::optim::{optimize_acqf, OptimizeAcqfOptions};
use botorch_core::utils::elementwise::reduce;
use botorch_core::utils::multi_objective::box_decompositions::FastNondominatedPartitioning;
use botorch_core::utils::polytope::HitAndRunPolytopeSampler;
use botorch_core::{
    psd_safe_cholesky, BatchedMultiOutputGP, LCEMGPOptions, Matrix, Model, SingleTaskGP,
    SingleTaskGPOptions, SobolQMCNormalSampler, Vector, LCEMGP,
};

#[test]
fn cholesky_inverse_is_exact_and_rejects_infinite_diagonal() {
    let a = Matrix::from_row_slice(
        3,
        3,
        &[
            4.0, 1.2, -0.5, //
            1.2, 3.5, 0.8, //
            -0.5, 0.8, 2.7,
        ],
    );
    let chol = psd_safe_cholesky(&a).unwrap();
    let inv = chol.inverse();
    let prod = &a * &inv;
    let eye = Matrix::identity(3, 3);
    assert!((prod - eye).amax() < 1e-14);

    let mut inf_mat = Matrix::identity(2, 2);
    inf_mat[(1, 1)] = f64::INFINITY;
    assert!(psd_safe_cholesky(&inf_mat).is_err());
}

#[test]
fn fatmin_negative_tau_checks_min_extremum_correctly() {
    // For [-inf, 0, 0], -row = [+inf, 0, 0] has infinite max so y_no_inf == 0 -> valid!
    let (vals, _) = reduce(
        "fatmin",
        &[f64::NEG_INFINITY, 0.0, 0.0],
        3,
        &[-1.0, 2.0],
        true,
    )
    .unwrap();
    assert_eq!(vals, vec![f64::NEG_INFINITY]);

    // For [0, +inf], min = 0 (finite) and +inf > 0 -> _pareto(-inf) must be rejected!
    let err = reduce("fatmin", &[0.0, f64::INFINITY], 2, &[-1.0, 2.0], false).unwrap_err();
    assert!(err.to_string().contains("non-negative"), "{err}");
}

#[test]
fn hit_and_run_handles_infinite_and_equal_bounds() {
    // x0 >= 0, x0 + x1 <= 2, with x1 fixed to 0.5 via equal bounds and upper bound +inf on x0.
    let a = Matrix::from_row_slice(2, 2, &[-1.0, 0.0, 1.0, 1.0]);
    let b = Vector::from_vec(vec![0.0, 2.0]);
    let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.5, f64::INFINITY, 0.5]);
    let mut sampler =
        HitAndRunPolytopeSampler::new(Some((a, b)), None, Some(&bounds), None, 20, 4, Some(42))
            .unwrap();
    let draws = sampler.draw(8).unwrap();
    assert_eq!(draws.shape(), (8, 2));
    for i in 0..8 {
        assert!(draws[(i, 0)].is_finite());
        assert!(draws[(i, 0)] >= -1e-9 && draws[(i, 0)] <= 1.5 + 1e-9);
        assert!((draws[(i, 1)] - 0.5).abs() < 1e-12);
    }
}

#[test]
fn model_clones_share_caches_and_eval_unobserved_is_idempotent() {
    // MultiTaskGP with task 2 unobserved in training data.
    let train_x = Matrix::from_row_slice(
        4,
        2,
        &[
            0.1, 0.0, //
            0.4, 0.0, //
            0.2, 1.0, //
            0.8, 1.0,
        ],
    );
    let train_y = Matrix::from_column_slice(4, 1, &[1.0, 1.5, 0.5, 0.9]);
    let opts = MultiTaskGPOptions {
        all_tasks: Some(vec![0, 1, 2]),
        ..Default::default()
    };
    let mut mt = MultiTaskGP::new(train_x.clone(), train_y.clone(), 1, opts).unwrap();
    let mut raw: Vec<f64> = mt
        .raw_parameters()
        .iter()
        .enumerate()
        .map(|(i, _)| 0.1 * (i + 1) as f64)
        .collect();
    raw[0] = 0.05;
    mt.set_raw_parameters(&raw).unwrap();
    assert!(mt.eval_unobserved_task_factors());
    // Calling a second time must be a no-op and return false across all platforms.
    assert!(!mt.eval_unobserved_task_factors());
    // Updating raw parameters re-arms unobserved synchronization.
    raw[0] = 0.25;
    mt.set_raw_parameters(&raw).unwrap();
    assert!(mt.eval_unobserved_task_factors());
    assert!(!mt.eval_unobserved_task_factors());

    let snap = mt.clone();
    let x_test = Matrix::from_row_slice(2, 1, &[0.3, 0.7]);
    let p1 = snap.posterior(&x_test, false).unwrap();
    let p2 = mt.posterior(&x_test, false).unwrap();
    assert_eq!(p1.mean, p2.mean);
    assert_eq!(p1.covariance, p2.covariance);

    // KroneckerMultiTaskGP clone shares cache.
    let kx = Matrix::from_row_slice(3, 2, &[0.1, 0.2, 0.4, 0.5, 0.8, 0.9]);
    let ky = Matrix::from_row_slice(3, 2, &[1.0, 2.0, 1.5, 2.5, 0.7, 1.2]);
    let kron = KroneckerMultiTaskGP::new(kx, ky, Default::default()).unwrap();
    let kron_snap = kron.clone();
    let qx = Matrix::from_row_slice(2, 2, &[0.25, 0.35, 0.6, 0.7]);
    assert_eq!(
        kron_snap.posterior(&qx, false).unwrap().mean,
        kron.posterior(&qx, false).unwrap().mean
    );

    // LCEMGP idempotent eval_unobserved_means and renorm_embeddings.
    let mut lcem = LCEMGP::new(
        train_x,
        train_y,
        1,
        LCEMGPOptions {
            all_tasks: Some(vec![0, 1, 2]),
            ..Default::default()
        },
    )
    .unwrap();
    let _ = lcem.eval_unobserved_means();
    assert!(!lcem.eval_unobserved_means());
    let _ = lcem.renorm_embeddings();
    assert!(!lcem.renorm_embeddings());
}

#[test]
fn fixed_feature_x_pending_roundtrips_and_proximal_validates_inputs() {
    let x = Matrix::from_fn(6, 3, |i, j| ((i * 3 + j + 1) % 7) as f64 / 6.0);
    let y = Matrix::from_fn(6, 1, |i, _| x[(i, 0)] + 0.5 * x[(i, 1)] - x[(i, 2)]);
    let gp: Arc<dyn Model> =
        Arc::new(SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap());
    let qei: Arc<dyn AcquisitionFunction> =
        Arc::new(QExpectedImprovement::new(gp.clone(), 0.5, Default::default()).unwrap());

    // Fix column 1 to 0.25 -> reduced dimension d' = 2.
    let ff = FixedFeatureAcquisitionFunction::new(
        qei,
        3,
        vec![1],
        Matrix::from_row_slice(1, 1, &[0.25]),
    )
    .unwrap();
    let pend = Matrix::from_row_slice(1, 2, &[0.2, 0.8]);
    ff.set_x_pending(Some(&pend)).unwrap();
    assert_eq!(ff.x_pending().unwrap(), pend);

    let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    let opts = OptimizeAcqfOptions {
        seed: Some(7),
        sequential: true,
        ..Default::default()
    };
    let (cands, _) = optimize_acqf(&ff, &bounds, 2, 2, 16, &opts).unwrap();
    assert_eq!(cands.shape(), (2, 2));
    // Original pending point (1 x 2) is restored cleanly.
    assert_eq!(ff.x_pending().unwrap(), pend);

    // Proximal validation: non-positive weight and setting X_pending are rejected.
    let lei: Arc<dyn AcquisitionFunction> =
        Arc::new(LogExpectedImprovement::new(gp, 0.5, true).unwrap());
    assert!(ProximalAcquisitionFunction::new(
        lei.clone(),
        Vector::from_vec(vec![1.0, 0.0, 1.0]),
        Vector::from_vec(vec![0.5, 0.5, 0.5]),
        None,
        false,
        Some(1.0),
    )
    .is_err());
    let prox = ProximalAcquisitionFunction::new(
        lei,
        Vector::from_vec(vec![1.0, 1.0, 1.0]),
        Vector::from_vec(vec![0.5, 0.5, 0.5]),
        None,
        false,
        Some(1.0),
    )
    .unwrap();
    assert!(prox
        .set_x_pending(Some(&Matrix::from_row_slice(1, 3, &[0.1, 0.2, 0.3])))
        .is_err());
}

#[test]
fn mixed_optimizer_rejects_out_of_bounds_fixed_feature_and_empty_neighbors_safe() {
    let x = Matrix::from_row_slice(4, 2, &[0.1, 0.2, 0.4, 0.5, 0.7, 0.3, 0.9, 0.8]);
    let y = Matrix::from_column_slice(4, 1, &[0.2, 0.8, 0.6, 1.1]);
    let gp: Arc<dyn Model> =
        Arc::new(SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap());
    let acqf = LogExpectedImprovement::new(gp, 1.1, true).unwrap();
    let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    let bad_ff = vec![BTreeMap::from([(2usize, 0.5)])];
    let err = optimize_acqf_mixed(
        &acqf,
        &bounds,
        1,
        2,
        &bad_ff,
        16,
        &OptimizeAcqfOptions::default(),
        true,
    )
    .unwrap_err();
    assert!(err.to_string().contains("out of range"), "{err}");

    let empty_dims: DiscreteDims = BTreeMap::from([(0usize, Vec::new())]);
    assert!(get_nearest_neighbors(&[0.5, 0.5], &bounds, &empty_dims).is_empty());
}

#[test]
fn ehvi_forward_many_matches_forward_and_log_hvkg_handles_high_current_value() {
    let x = Matrix::from_fn(6, 2, |i, j| ((i * 5 + j * 2 + 1) % 7) as f64 / 6.0);
    let y = Matrix::from_fn(6, 2, |i, j| {
        if j == 0 {
            x[(i, 0)] + 0.2 * x[(i, 1)]
        } else {
            1.0 - x[(i, 0)] + 0.3 * x[(i, 1)]
        }
    });
    let model: Arc<dyn Model> =
        Arc::new(BatchedMultiOutputGP::multi_output(x, y.clone(), Default::default()).unwrap());
    let ref_point = [0.0, 0.0];
    let part = FastNondominatedPartitioning::new(&ref_point, Some(&y)).unwrap();
    let ehvi = ExpectedHypervolumeImprovement::new(model.clone(), &ref_point, &part).unwrap();
    let pts: Vec<Matrix> = (0..70)
        .map(|i| {
            let u = i as f64 / 69.0;
            Matrix::from_row_slice(1, 2, &[u, 1.0 - u])
        })
        .collect();
    let batch_vals = ehvi.forward_many(&pts).unwrap();
    for (p, &bv) in pts.iter().zip(&batch_vals) {
        let sv = ehvi.forward(p).unwrap();
        assert!((bv - sv).abs() < 1e-14, "{bv} vs {sv}");
    }

    // qHVKG in log mode with current_value above some/all fantasies must not NaN.
    let hvkg = QHypervolumeKnowledgeGradient::new(
        model.clone(),
        &ref_point,
        QHypervolumeKnowledgeGradientOptions {
            num_fantasies: 4,
            num_pareto: 2,
            current_value: Some(10.0), // log(HV) = 10 -> exp(10) >> fantasy HV, so v <= cv
            log: true,
            ..Default::default()
        },
    )
    .unwrap();
    // 1 candidate + 4 * 2 Pareto points = 9 rows.
    let x_aug = Matrix::from_fn(9, 2, |i, j| ((i * 3 + j + 1) % 10) as f64 / 10.0);
    let (_, grad) = hvkg.forward_with_grad(&x_aug).unwrap();
    assert!(grad.iter().all(|v| !v.is_nan()));

    // qNEHVI invalidates joint_base cache if sampler base_samples are replaced.
    let sampler = Arc::new(SobolQMCNormalSampler::new(8, Some(1)).unwrap());
    let x_base = Matrix::from_row_slice(2, 2, &[0.2, 0.3, 0.7, 0.8]);
    let nehvi = QNoisyExpectedHypervolumeImprovement::new(
        model,
        &ref_point,
        &x_base,
        QNoisyExpectedHypervolumeImprovementOptions {
            sampler: Some(sampler.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let cand = Matrix::from_row_slice(1, 2, &[0.5, 0.5]);
    let v1 = nehvi.forward(&cand).unwrap();
    sampler
        .set_base_samples(Matrix::from_element(8, 6, 3.0))
        .unwrap();
    let v2 = nehvi.forward(&cand).unwrap();
    assert_ne!(v1, v2);

    // Single-output cost model accepts multi-output evaluation masks selecting output 0.
    let cost_model: Arc<dyn CostModel> = Arc::new(AffineFidelityCostModel::new(None, 0.1).unwrap());
    let util = InverseCostWeightedUtility::new(cost_model, true, None, false).unwrap();
    let x_q = Matrix::from_row_slice(2, 2, &[0.2, 0.5, 0.4, 1.0]);
    let mask_ok = BoolMatrix::from_row_slice(2, 2, &[true, false, true, false]);
    let (cost, grad) = util.total_cost(&x_q, Some(&mask_ok), true).unwrap();
    assert!((cost - 1.7).abs() < 1e-12);
    assert!(grad.unwrap().iter().all(|v| v.is_finite()));
    let mask_oob = BoolMatrix::from_row_slice(2, 2, &[false, true, false, true]);
    assert!(util.total_cost(&x_q, Some(&mask_oob), false).is_err());
}
