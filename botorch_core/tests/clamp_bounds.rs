//! Derivatives exactly at the bounds of BoTorch's clamps follow PyTorch
//! ≥ 2.14 (pytorch#191142): a scalar clamp bound is a fixed parameter, so the
//! subgradient at equality is 0 (PyTorch ≤ 2.13 passed the gradient on the
//! closed interval). Away from the bounds the derivatives are the analytic
//! ones (checked against central differences). The comparisons with BoTorch
//! are `botorch_pyo3/tests/test_parity_clamp_bounds.py`.

use std::sync::Arc;

use botorch_core::acquisition::{AcquisitionFunction, McAcquisitionOptions, QExpectedImprovement};
use botorch_core::kernels::{KernelNode, TreeKernel};
use botorch_core::models::AffineDeterministicModel;
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::transforms::utils::kumaraswamy_warp_with_grad;
use botorch_core::{KernelKind, Matrix, Vector};

const TRAIN: [f64; 15] = [
    0.1, 0.7, 0.0, //
    0.4, 0.2, 1.0, //
    0.8, 0.5, 0.5, //
    0.3, 0.9, 0.0, //
    0.6, 0.1, 1.0,
];

/// The cross-covariance gradients of `x` (one row) against the training
/// rows, and their central (or one-sided, `sign`) differences in column `c`.
fn cross_grad(k: &TreeKernel, x: &[f64], c: usize, sign: f64) -> (Matrix, Vector) {
    let d = x.len();
    let train = Matrix::from_row_slice(5, d, &TRAIN[..5 * d]);
    let b = k.prepare(&train).unwrap();
    let a = k.prepare(&Matrix::from_row_slice(1, d, x)).unwrap();
    let (_, g) = k.cross_with_grad(&a, 0, &b);
    let h = 1e-6;
    let at = |t: f64| {
        let mut xs = x.to_vec();
        xs[c] += t;
        let a = k.prepare(&Matrix::from_row_slice(1, d, &xs)).unwrap();
        k.cross(&a, &b).row(0).transpose()
    };
    let fd = if sign == 0.0 {
        (at(h) - at(-h)) / (2.0 * h)
    } else {
        (at(sign * h) - at(0.0)) / (sign * h)
    };
    (g, fd)
}

#[test]
fn linear_truncated_fidelity_gradients_vanish_at_the_fidelity_bounds() {
    for fid in [vec![2], vec![1, 2]] {
        let non_fid = 3 - fid.len();
        let st = |k: KernelKind| KernelNode::stationary(k, Some(non_fid)).unwrap();
        let node = KernelNode::linear_truncated_fidelity(
            fid.clone(),
            st(KernelKind::Matern52),
            st(KernelKind::Rbf),
        )
        .unwrap();
        let raw: Vec<f64> = (0..node.num_raw())
            .map(|i| -0.4 + 0.15 * i as f64)
            .collect();
        let k = TreeKernel::build(&node, &raw, 3).unwrap();
        for f in [0.0, 1.0, 0.25, 0.5, -0.2, 1.3] {
            let x = [0.35, 0.6, f];
            let (g, fd) = cross_grad(&k, &x, 2, 0.0);
            if 0.0 < f && f < 1.0 {
                for j in 0..5 {
                    assert!(
                        (g[(j, 2)] - fd[j]).abs() < 1e-7,
                        "{fid:?} {f}: {} {}",
                        g[(j, 2)],
                        fd[j]
                    );
                }
                assert!(g.column(2).iter().any(|&v| v != 0.0));
            } else {
                assert!(g.column(2).iter().all(|&v| v == 0.0), "{fid:?} {f}: {g}");
            }
            if f == 0.0 || f == 1.0 {
                // The one-sided derivative from inside (PyTorch ≤ 2.13's
                // gradient at the bound) does not vanish.
                let (_, inside) = cross_grad(&k, &x, 2, if f == 0.0 { 1.0 } else { -1.0 });
                assert!(
                    inside.iter().any(|v| v.abs() > 1e-3),
                    "{fid:?} {f}: {inside}"
                );
            }
        }
    }
}

#[test]
fn warp_gradient_vanishes_at_the_eps_bounds() {
    // x = 0 maps to eps exactly; x = 1 maps to 1 - eps exactly for eps =
    // 2^-20, but below it for eps = 1e-7.
    for (eps, upper_tie) in [(1e-7, false), (2f64.powi(-20), true)] {
        assert_eq!(1.0 * (1.0 - 2.0 * eps) + eps == 1.0 - eps, upper_tie);
        let (_, d0, _, _) = kumaraswamy_warp_with_grad(0.0, 1.4, 0.7, eps);
        let (_, d1, _, _) = kumaraswamy_warp_with_grad(1.0, 1.4, 0.7, eps);
        assert_eq!(d0, 0.0);
        assert_eq!(d1 == 0.0, upper_tie);
        let (_, dm, _, _) = kumaraswamy_warp_with_grad(0.5, 1.4, 0.7, eps);
        assert!(dm > 0.0);
    }
}

#[test]
fn qei_gradient_vanishes_at_an_improvement_of_zero() {
    // y = x_0 exactly: samples tie with best_f at x_0 = 0.3.
    let model = Arc::new(
        AffineDeterministicModel::new(Matrix::from_row_slice(2, 1, &[1.0, 0.0]), Vector::zeros(1))
            .unwrap(),
    );
    let opts = || McAcquisitionOptions {
        sampler: Some(Arc::new(SobolQMCNormalSampler::new(1, Some(0)).unwrap())),
        ..Default::default()
    };
    let qei = QExpectedImprovement::new(model, 0.3, opts()).unwrap();
    let cases: [(&[f64], f64, &[f64]); 4] = [
        (&[0.3, 0.7], 0.0, &[0.0, 0.0]),
        (&[0.31, 0.7], 0.31 - 0.3, &[1.0, 0.0]),
        (&[0.2, 0.7], 0.0, &[0.0, 0.0]),
        // q = 2, both at the bound.
        (&[0.3, 0.7, 0.3, 0.1], 0.0, &[0.0, 0.0, 0.0, 0.0]),
    ];
    for (x, value, grad) in cases {
        let x = Matrix::from_row_slice(x.len() / 2, 2, x);
        let (v, g) = qei.forward_with_grad(&x).unwrap();
        assert_eq!(v, value);
        assert_eq!(g.transpose().as_slice(), grad, "{x}");
    }
}
