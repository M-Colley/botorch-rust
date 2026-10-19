//! Integrated white noise kernel: exact formulas and analytic derivatives.

use botorch_core::kernels::{KernelNode, TreeKernel};
use botorch_core::Matrix;

#[test]
fn integrated_white_noise_matches_closed_forms_and_is_positive_semidefinite() {
    let x = Matrix::from_column_slice(6, 1, &[0.0, 0.1, 0.3, 0.7, 1.2, 2.0]);
    for order in 1..=8 {
        let node = KernelNode::integrated_white_noise(order).unwrap();
        assert_eq!(node.num_raw(), 0);
        let kernel = TreeKernel::build(&node, &[], 1).unwrap();
        let prep = kernel.prepare(&x).unwrap();
        let covariance = kernel.symmetric(&prep);
        assert!((&covariance - covariance.transpose()).amax() == 0.0);
        assert!(covariance.clone().symmetric_eigen().eigenvalues.min() > -1e-10);
        assert!(kernel.contract(&prep, &Matrix::identity(6, 6)).is_empty());
        for i in 0..6 {
            assert_eq!(kernel.diag_mode(&prep, i), covariance[(i, i)]);
            for j in 0..6 {
                let min = x[(i, 0)].min(x[(j, 0)]);
                let max = x[(i, 0)].max(x[(j, 0)]);
                let expected = match order {
                    1 => min,
                    2 => min.powi(2) * (3.0 * max - min) / 6.0,
                    3 => min.powi(3) * (min.powi(2) - 5.0 * min * max + 10.0 * max.powi(2)) / 120.0,
                    _ => continue,
                };
                assert!((covariance[(i, j)] - expected).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn integrated_white_noise_input_gradients_match_finite_differences() {
    let other = Matrix::from_column_slice(4, 2, &[7.0, 8.0, 9.0, 10.0, 0.1, 0.35, 0.7, 1.4]);
    for order in 1..=8 {
        let node = KernelNode::scale(
            KernelNode::integrated_white_noise(order)
                .unwrap()
                .with_active_dims(vec![1])
                .unwrap(),
        );
        let raw = node.initial_raw();
        let kernel = TreeKernel::build(&node, &raw, 2).unwrap();
        let other_prep = kernel.prepare(&other).unwrap();
        for time in [0.0, 0.35, 0.9] {
            let query = Matrix::from_row_slice(1, 2, &[5.0, time]);
            let prep = kernel.prepare(&query).unwrap();
            let (_, gradient) = kernel.cross_with_grad(&prep, 0, &other_prep);
            assert!(gradient.column(0).iter().all(|&v| v == 0.0));
            let h = 1e-6;
            let plus = Matrix::from_row_slice(1, 2, &[5.0, time + h]);
            let minus = Matrix::from_row_slice(1, 2, &[5.0, time - h]);
            let fd = (kernel.covariance(&plus, &other).unwrap()
                - kernel.covariance(&minus, &other).unwrap())
                / (2.0 * h);
            for j in 0..other.nrows() {
                assert!((fd[(0, j)] - gradient[(j, 1)]).abs() < 1e-7 * (1.0 + fd[(0, j)].abs()));
            }
        }
    }
}

#[test]
fn integrated_white_noise_validates_dimensions_and_handles_large_orders() {
    assert!(KernelNode::integrated_white_noise(0).is_err());
    let node = KernelNode::integrated_white_noise(1).unwrap();
    assert!(node.validate(2).is_err());
    for (order, time) in [(60, 20.0_f64), (150, 100.0)] {
        let node = KernelNode::integrated_white_noise(order).unwrap();
        let kernel = TreeKernel::build(&node, &[], 1).unwrap();
        let point = Matrix::from_element(1, 1, time);
        let value = kernel.covariance_symmetric(&point).unwrap()[(0, 0)];
        let log_expected = (2 * order - 1) as f64 * time.ln()
            - 2.0 * (1..order).map(|i| (i as f64).ln()).sum::<f64>()
            - ((2 * order - 1) as f64).ln();
        assert!(value.is_finite() && value > 0.0);
        assert!((value.ln() - log_expected).abs() < 1e-10);
    }
}
