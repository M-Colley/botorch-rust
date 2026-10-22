//! Source-derived and analytic checks for Schmitt et al. (2022).
use approx::assert_abs_diff_eq;
use botorch_core::utils::multi_objective::decision_making::{
    analyze_knee_region, knee_region, knee_region_batch, select_preferred, select_preferred_batch,
};
use botorch_core::utils::multi_objective::fpbi::{fpbi_geometry, generate_fpbi_front, FpbiOptions};
use botorch_core::utils::multi_objective::PointBatch;
use botorch_core::Matrix;
use std::cell::Cell;

#[test]
fn signed_plane_distance_and_alpha_endpoints() {
    let y = Matrix::from_row_slice(3, 3, &[1., 1., 0., 0., 0., 1., 0.8, 0.8, 0.8]);
    let report = analyze_knee_region(&y, 0., false).unwrap();
    assert_eq!(report.distance_plane_index, 0);
    assert_abs_diff_eq!(
        report.signed_distances[2],
        -0.4 / 3.0_f64.sqrt(),
        epsilon = 1e-14
    );
    assert_eq!(report.mask, vec![true, true, false]);
    assert_eq!(
        knee_region(&y, 1., false).unwrap(),
        vec![false, true, false]
    );
}

#[test]
fn dominated_rows_do_not_distort_normalization_and_duplicates_are_stable() {
    let y = Matrix::from_row_slice(
        5,
        2,
        &[0., 1., 0.25, 0.25, 1., 0., 1000., 1000., 0.25, 0.25],
    );
    assert_eq!(
        knee_region(&y, 0.5, false).unwrap(),
        vec![false, true, false, false, false]
    );
    assert_eq!(select_preferred(&y, &[1., 100.], 0., false).unwrap(), 2);
    assert_eq!(select_preferred(&y, &[1., 100.], 1., false).unwrap(), 1);
    let affine = Matrix::from_fn(5, 2, |i, j| {
        if j == 0 {
            1e3 * y[(i, j)] + 75.
        } else {
            0.01 * y[(i, j)] - 3.
        }
    });
    assert_eq!(
        knee_region(&affine, 0.5, false).unwrap(),
        knee_region(&y, 0.5, false).unwrap()
    );
    assert_eq!(select_preferred(&(-&y), &[1., 100.], 0., true).unwrap(), 2);
}

#[test]
fn batch_flat_axes_one_row_and_validation() {
    let a = Matrix::from_row_slice(3, 3, &[0., 1., 7., 0.25, 0.25, 7., 1., 0., 7.]);
    let b = -&a;
    let batch = PointBatch::from_matrices(&[a.clone(), a.clone()]).unwrap();
    assert_eq!(
        knee_region_batch(&batch, 0.5, false).unwrap(),
        vec![false, true, false, false, true, false]
    );
    assert_eq!(
        select_preferred_batch(&batch, &[1., 100., 0.], 0., false).unwrap(),
        vec![2, 2]
    );
    assert_eq!(select_preferred(&b, &[1., 100., 0.], 0., true).unwrap(), 2);
    let single = Matrix::from_row_slice(1, 2, &[3., 3.]);
    assert_eq!(knee_region(&single, 1., true).unwrap(), vec![true]);
    for bad in [f64::NAN, -0.1, 1.1, f64::INFINITY] {
        assert!(knee_region(&a, bad, true).is_err());
    }
    for p in [
        vec![0., 0., 0.],
        vec![1., -1., 0.],
        vec![f64::NAN, 1., 1.],
        vec![1.],
    ] {
        assert!(select_preferred(&a, &p, 0.5, true).is_err());
    }
    assert!(knee_region(&Matrix::zeros(0, 2), 0.5, true).is_err());
    assert!(knee_region(
        &Matrix::from_row_slice(1, 2, &[0., f64::INFINITY]),
        0.5,
        true
    )
    .is_err());
    let huge = Matrix::from_row_slice(2, 2, &[-f64::MAX, f64::MAX, f64::MAX, -f64::MAX]);
    assert_eq!(knee_region(&huge, 0., false).unwrap(), vec![true, true]);
}

fn options() -> FpbiOptions {
    FpbiOptions {
        maximize: false,
        grid_resolution: 2,
        ..Default::default()
    }
}

#[test]
fn paper_fpbi_grid_is_asymmetric_and_focus_direction_not_normalized() {
    let e = Matrix::from_row_slice(2, 2, &[0., 1., 1., 0.]);
    let g = fpbi_geometry(&e, None, &options()).unwrap();
    assert_eq!(g.farthest_pair, (0, 1));
    assert_eq!(g.center, vec![0.5, 0.5]);
    assert_eq!(g.search_direction, vec![-0.5, -0.5]);
    assert_abs_diff_eq!(g.anchors[(0, 0)], 0.5, epsilon = 1e-14);
    assert_abs_diff_eq!(g.anchors[(1, 0)], 1.0, epsilon = 1e-14);
    assert_abs_diff_eq!(g.anchors[(1, 1)], 0.0, epsilon = 1e-14);
    let focused = fpbi_geometry(&e, Some(&[-0.2, 0.1]), &options()).unwrap();
    assert_eq!(focused.search_direction, vec![-0.7, -0.4]);
    let mut excessive = options();
    excessive.max_grid_points = 1;
    assert!(fpbi_geometry(&e, None, &excessive).is_err());
}

#[test]
fn flat_axes_and_dominated_extremes_have_deterministic_geometry() {
    let e = Matrix::from_row_slice(3, 3, &[0., 1., 7., 1., 0., 7., 0.5, 0.5, 7.]);
    let g = fpbi_geometry(&e, None, &options()).unwrap();
    assert_eq!(g.active_objectives, vec![true, true, false]);
    assert_eq!(g.basis.shape(), (2, 3));
    assert!((0..g.anchors.nrows()).all(|i| g.anchors[(i, 2)] == 0.));
    assert!((0..3).all(|j| g.basis[(1, j)] == 0.));
    assert!(fpbi_geometry(&e, Some(&[0., 0., 1.]), &options()).is_err());
    let flat = fpbi_geometry(&Matrix::from_element(3, 3, 4.), None, &options()).unwrap();
    assert!(flat.degenerate);
    assert_eq!(flat.farthest_pair, (0, 0));
    assert_eq!(flat.anchors.shape(), (4, 3));
    assert!(flat.anchors.iter().all(|v| *v == 0.));
    let dominated = Matrix::from_row_slice(3, 3, &[0., 1., 0., 1., 0., 0., 100., 100., 100.]);
    let geometry = fpbi_geometry(&dominated, None, &options()).unwrap();
    assert_eq!(geometry.ranges, vec![1., 1., 0.]);
}

#[test]
fn analytic_continuous_fpbi_front_and_constraint_feasibility() {
    let objective = |x: &[f64]| Ok(vec![x[0] * x[0], (x[0] - 1.).powi(2)]);
    let bounds = Matrix::from_row_slice(2, 1, &[0., 1.]);
    let starts = Matrix::from_row_slice(3, 1, &[0.2, 0.5, 0.8]);
    let front =
        generate_fpbi_front(&objective, &bounds, &starts, None, None, None, &options()).unwrap();
    assert_eq!(front.x.nrows(), 3);
    for expected in [0., 0.5, 1.] {
        assert!((0..front.x.nrows()).any(|i| (front.x[(i, 0)] - expected).abs() < 1e-5));
    }
    let middle = front
        .statuses
        .iter()
        .find(|s| !s.extreme && s.index == 0)
        .unwrap();
    assert_abs_diff_eq!(middle.kappa.unwrap(), 0.5, epsilon = 1e-5);
    assert!(front
        .statuses
        .iter()
        .all(|s| s.success && s.max_violation <= 1e-6));
    let constraints = |x: &[f64]| Ok(vec![x[0] - 0.2, 0.8 - x[0]]);
    let constrained = generate_fpbi_front(
        &objective,
        &bounds,
        &starts,
        None,
        Some(&constraints),
        None,
        &options(),
    )
    .unwrap();
    assert!((0..constrained.x.nrows())
        .all(|i| constrained.x[(i, 0)] >= 0.2 - 1e-6 && constrained.x[(i, 0)] <= 0.8 + 1e-6));
    let middle = constrained
        .statuses
        .iter()
        .find(|s| !s.extreme && s.index == 0)
        .unwrap();
    assert_abs_diff_eq!(middle.kappa.unwrap(), 0.3, epsilon = 1e-5);
    let max_objective = |x: &[f64]| Ok(vec![-x[0] * x[0], -(x[0] - 1.).powi(2)]);
    let max_options = FpbiOptions {
        maximize: true,
        ..options()
    };
    let max_front = generate_fpbi_front(
        &max_objective,
        &bounds,
        &starts,
        None,
        None,
        None,
        &max_options,
    )
    .unwrap();
    assert_eq!(max_front.x.nrows(), 3);
}

#[test]
fn equalities_input_units_failures_and_degenerate_fronts() {
    let bounds = Matrix::from_row_slice(2, 2, &[10., 20., 110., 220.]);
    let objective = |x: &[f64]| {
        let a = (x[0] - 10.) / 100.;
        let b = (x[1] - 20.) / 200.;
        Ok(vec![a * a, (b - 1.).powi(2)])
    };
    let equality = |x: &[f64]| Ok(vec![(x[0] - 10.) / 100. - (x[1] - 20.) / 200.]);
    let starts = Matrix::from_row_slice(2, 2, &[30., 60., 80., 160.]);
    let front = generate_fpbi_front(
        &objective,
        &bounds,
        &starts,
        None,
        None,
        Some(&equality),
        &options(),
    )
    .unwrap();
    assert!((0..front.x.nrows()).any(|i| (front.x[(i, 0)] - 60.).abs() < 1e-3));
    assert!(front.statuses.iter().all(|s| s.feasible));
    let impossible = |_: &[f64]| Ok(vec![-1.]);
    assert!(generate_fpbi_front(
        &objective,
        &bounds,
        &starts,
        None,
        Some(&impossible),
        None,
        &options()
    )
    .is_err());
    let flat_objective = |_: &[f64]| Ok(vec![7., 7.]);
    let flat = generate_fpbi_front(
        &flat_objective,
        &bounds,
        &starts,
        None,
        None,
        None,
        &options(),
    )
    .unwrap();
    assert!(flat.geometry.degenerate);
    assert_eq!(flat.x.nrows(), 1);
    assert!(generate_fpbi_front(
        &objective,
        &bounds,
        &starts,
        Some(&[1., 1.]),
        None,
        None,
        &options()
    )
    .is_err());
}

#[test]
fn generalized_cross_rank_fallback_and_negative_kappa() {
    let e = Matrix::from_row_slice(3, 3, &[0., 1., 1., 1., 0., 1., 1., 1., 0.]);
    let geometry = fpbi_geometry(&e, None, &options()).unwrap();
    assert!(!geometry.used_degenerate_fallback);
    for j in 0..3 {
        let expected = if j == 2 { 1.0 } else { -1.0 } / 3.0_f64.sqrt();
        assert_abs_diff_eq!(geometry.basis[(1, j)], expected, epsilon = 1e-14);
    }
    let bounds = Matrix::from_fn(2, 3, |i, _| i as f64);
    let objective = |x: &[f64]| Ok(x.to_vec());
    let equality = |x: &[f64]| Ok(vec![x.iter().sum::<f64>() - 2.]);
    let front = generate_fpbi_front(
        &objective,
        &bounds,
        &e,
        None,
        None,
        Some(&equality),
        &options(),
    )
    .unwrap();
    assert!(front
        .statuses
        .iter()
        .any(|s| s.success && s.kappa.is_some_and(|k| k < -0.5)));
    let rank_deficient = Matrix::from_row_slice(
        4,
        4,
        &[
            0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0.,
        ],
    );
    let completed = fpbi_geometry(&rank_deficient, None, &options()).unwrap();
    assert!(completed.used_degenerate_fallback);
    assert!(completed.basis.iter().all(|v| v.is_finite()));
    for i in 0..3 {
        assert_abs_diff_eq!(completed.basis.row(i).norm(), 1., epsilon = 1e-12);
        for j in 0..i {
            assert_abs_diff_eq!(
                completed.basis.row(i).dot(&completed.basis.row(j)),
                0.,
                epsilon = 1e-12
            );
        }
    }
}

#[test]
fn fpbi_preserves_actual_high_endpoints_after_cancellation() {
    let costs = Matrix::from_row_slice(2, 2, &[-1e16, 1., 1., -1e16]);
    for maximize in [false, true] {
        let y = if maximize { -&costs } else { costs.clone() };
        let geometry = fpbi_geometry(
            &y,
            None,
            &FpbiOptions {
                maximize,
                ..options()
            },
        )
        .unwrap();
        assert_eq!(geometry.utopia, vec![-1e16, -1e16]);
        assert_eq!(geometry.nadir, vec![1., 1.]);
        // The rounded span cannot be used to reconstruct the original high.
        assert_eq!(geometry.utopia[0] + geometry.ranges[0], 0.);
        assert_eq!(
            geometry.normalized_extremes,
            Matrix::from_row_slice(2, 2, &[0., 1., 1., 0.])
        );
    }
}

#[test]
fn invalid_fpbi_grid_is_rejected_before_extremization() {
    let bounds = Matrix::from_row_slice(2, 1, &[0., 1.]);
    let starts = Matrix::from_row_slice(2, 1, &[0.25, 0.75]);
    for settings in [
        FpbiOptions {
            grid_resolution: 2,
            max_grid_points: 3,
            ..options()
        },
        FpbiOptions {
            grid_resolution: usize::MAX,
            max_grid_points: usize::MAX,
            ..options()
        },
    ] {
        let objective_calls = Cell::new(0);
        let constraint_calls = Cell::new(0);
        let objective = |x: &[f64]| {
            objective_calls.set(objective_calls.get() + 1);
            Ok(vec![x[0] * x[0], (1. - x[0]).powi(2), x[0]])
        };
        let constraints = |_: &[f64]| {
            constraint_calls.set(constraint_calls.get() + 1);
            Ok(vec![1.])
        };
        let error = generate_fpbi_front(
            &objective,
            &bounds,
            &starts,
            None,
            Some(&constraints),
            None,
            &settings,
        )
        .unwrap_err();
        assert!(error.to_string().contains("grid"), "{error}");
        assert_eq!(
            objective_calls.get(),
            1,
            "only the dimension probe should run"
        );
        assert_eq!(
            constraint_calls.get(),
            0,
            "constraints should not run on an invalid grid"
        );
    }
}
