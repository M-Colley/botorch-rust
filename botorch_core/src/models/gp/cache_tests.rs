//! Cache reuse is shared by value snapshots, while mutations remain isolated.

use super::*;

fn model(fixed: bool) -> SingleTaskGP {
    let x = Matrix::from_fn(11, 2, |i, j| ((3 * i + 2 * j) % 11) as f64 / 11.0);
    let y = Matrix::from_fn(11, 1, |i, _| (x[(i, 0)] * 4.0).sin() + x[(i, 1)]);
    SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            outcome_transform: OutcomeTransformChoice::None,
            train_yvar: fixed.then(|| Matrix::from_element(11, 1, 0.03)),
            ..Default::default()
        },
    )
    .unwrap()
}

fn query() -> Matrix {
    Matrix::from_row_slice(3, 2, &[0.23, 0.51, 0.67, 0.32, 0.81, 0.42])
}

fn assert_same(a: &(Vector, Vector), b: &(Vector, Vector)) {
    assert!((&a.0 - &b.0).amax() < 1e-12);
    assert!((&a.1 - &b.1).amax() < 1e-12);
}

#[test]
fn snapshot_initializes_the_originals_cache_and_later_clones_share_it() {
    let gp = model(false);
    let snapshot = gp.clone();
    assert!(gp.cache.get().is_none());
    let before = snapshot.posterior_marginals(&query()).unwrap();
    assert!(gp.cache.get().is_some());
    assert!(std::ptr::eq(
        gp.train_cholesky().unwrap(),
        snapshot.train_cholesky().unwrap()
    ));
    let later = gp.clone();
    assert!(std::ptr::eq(
        snapshot.train_cholesky().unwrap(),
        later.train_cholesky().unwrap()
    ));
    assert_same(&before, &later.posterior_marginals(&query()).unwrap());
}

#[test]
fn concurrent_snapshots_share_one_lazy_factor() {
    let gp = model(false);
    let snapshots: Vec<_> = (0..4).map(|_| gp.clone()).collect();
    let addresses = std::thread::scope(|scope| {
        snapshots
            .iter()
            .map(|copy| scope.spawn(move || copy.train_cholesky().unwrap().l().as_ptr() as usize))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    let address = gp.train_cholesky().unwrap().l().as_ptr() as usize;
    assert!(addresses.iter().all(|&value| value == address));
}

#[test]
fn raw_parameter_and_sparse_noise_mutations_detach_without_changing_snapshots() {
    let gp = model(false);
    let old = gp.posterior_marginals(&query()).unwrap();
    let snapshot = gp.clone();
    let mut changed = gp.clone();
    let mut raw = changed.raw_parameters();
    raw[0] = 0.04;
    raw[1] = 0.23;
    raw[2] = 0.37;
    changed.set_raw_parameters(raw.as_slice()).unwrap();
    assert!(!Arc::ptr_eq(&gp.cache, &changed.cache));
    let mut fresh = model(false);
    fresh.set_raw_parameters(raw.as_slice()).unwrap();
    assert_same(
        &changed.posterior_marginals(&query()).unwrap(),
        &fresh.posterior_marginals(&query()).unwrap(),
    );
    changed
        .set_sparse_outlier_noise(Some(super::super::relevance_pursuit::SparseOutlierState {
            support: vec![2],
            raw: vec![0.3],
            convex: false,
            bounds: (0.0, 10.0),
            prior: None,
        }))
        .unwrap();
    assert!(changed.cache.get().is_none());
    changed.posterior_marginals(&query()).unwrap();
    assert_same(&old, &gp.posterior_marginals(&query()).unwrap());
    assert_same(&old, &snapshot.posterior_marginals(&query()).unwrap());
}

#[test]
fn replacement_targets_and_fixed_noise_preserve_the_old_model_law() {
    let gp = model(true);
    let old = gp.posterior_marginals(&query()).unwrap();
    let mut changed = gp.clone();
    let targets = gp.train_y_tf.map(|value| value + 0.7);
    let noise = Vector::from_element(gp.num_train(), 0.12);
    changed
        .replace_transformed_train_data(targets.clone(), Some(noise))
        .unwrap();
    assert!(!Arc::ptr_eq(&gp.cache, &changed.cache));
    let fresh = SingleTaskGP::new(
        gp.train_inputs().clone(),
        Matrix::from_column_slice(targets.len(), 1, targets.as_slice()),
        SingleTaskGPOptions {
            outcome_transform: OutcomeTransformChoice::None,
            train_yvar: Some(Matrix::from_element(gp.num_train(), 1, 0.12)),
            ..Default::default()
        },
    )
    .unwrap();
    assert_same(
        &changed.posterior_marginals(&query()).unwrap(),
        &fresh.posterior_marginals(&query()).unwrap(),
    );
    assert_same(&old, &gp.posterior_marginals(&query()).unwrap());
}

#[test]
fn conditioned_models_share_only_their_own_new_factor() {
    let mut gp = model(false);
    let old = gp.posterior_marginals(&query()).unwrap();
    let conditioned = gp
        .condition_on_observations(
            &Matrix::from_row_slice(1, 2, &[0.41, 0.72]),
            &Matrix::from_element(1, 1, 0.8),
            None,
        )
        .unwrap();
    let snapshot = conditioned.clone();
    let expected = snapshot.posterior_marginals(&query()).unwrap();
    assert!(!Arc::ptr_eq(&gp.cache, &conditioned.cache));
    assert!(std::ptr::eq(
        conditioned.train_cholesky().unwrap(),
        snapshot.train_cholesky().unwrap()
    ));
    let mut raw = gp.raw_parameters();
    raw[1] += 0.4;
    gp.set_raw_parameters(raw.as_slice()).unwrap();
    assert_same(&expected, &snapshot.posterior_marginals(&query()).unwrap());
    assert!((&old.0 - gp.posterior_marginals(&query()).unwrap().0).amax() > 1e-8);
}
