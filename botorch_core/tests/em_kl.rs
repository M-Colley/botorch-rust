use botorch_core::kernels::{KernelKind, KernelNode};
use botorch_core::models::em_kl::{kl_divergence_mvn, KernelPriorKl};
use botorch_core::{Matrix, Vector};

#[test]
fn gaussian_kl_gradients_match_symmetric_finite_differences() {
    let p = Vector::from_vec(vec![0.2, -0.5]);
    let q = Vector::from_vec(vec![0.8, 0.3]);
    let a = Matrix::from_row_slice(2, 2, &[1.3, 0.2, 0.2, 0.8]);
    let b = Matrix::from_row_slice(2, 2, &[0.9, -0.1, -0.1, 1.4]);
    let gradient = kl_divergence_mvn(&p, &a, &q, &b).unwrap();
    let epsilon = 1e-6;
    for which in 0..4 {
        let count = if which % 2 == 0 { 2 } else { 3 };
        for index in 0..count {
            let mut pp = p.clone();
            let mut pm = p.clone();
            let mut qp = q.clone();
            let mut qm = q.clone();
            let mut ap = a.clone();
            let mut am = a.clone();
            let mut bp = b.clone();
            let mut bm = b.clone();
            let expected = match which {
                0 => {
                    pp[index] += epsilon;
                    pm[index] -= epsilon;
                    gradient.mean_p[index]
                }
                2 => {
                    qp[index] += epsilon;
                    qm[index] -= epsilon;
                    gradient.mean_q[index]
                }
                _ => {
                    let (row, col) = if index == 0 {
                        (0, 0)
                    } else if index == 1 {
                        (0, 1)
                    } else {
                        (1, 1)
                    };
                    let (plus, minus, g) = if which == 1 {
                        (&mut ap, &mut am, &gradient.covariance_p)
                    } else {
                        (&mut bp, &mut bm, &gradient.covariance_q)
                    };
                    plus[(row, col)] += epsilon;
                    minus[(row, col)] -= epsilon;
                    if row != col {
                        plus[(col, row)] += epsilon;
                        minus[(col, row)] -= epsilon;
                    }
                    g[(row, col)] * if row == col { 1.0 } else { 2.0 }
                }
            };
            let plus = kl_divergence_mvn(&pp, &ap, &qp, &bp).unwrap().value;
            let minus = kl_divergence_mvn(&pm, &am, &qm, &bm).unwrap().value;
            assert!(((plus - minus) / (2.0 * epsilon) - expected).abs() < 2e-8);
        }
    }
}

#[test]
fn kernel_kl_raw_adjoint_matches_finite_differences() {
    let node = KernelNode::stationary(KernelKind::Rbf, Some(2)).unwrap();
    let points = Matrix::from_row_slice(3, 2, &[0.1, 0.2, 0.4, 0.7, 0.9, 0.5]);
    let mean = Vector::from_vec(vec![0.4, 0.5, 0.7]);
    let covariance = Matrix::from_row_slice(3, 3, &[1.1, 0.3, 0.1, 0.3, 1.2, 0.2, 0.1, 0.2, 0.9]);
    let objective = KernelPriorKl::new(node, points, mean, covariance, true, None).unwrap();
    let raw = vec![0.2, -0.4, -0.7];
    let (_, gradient) = objective.value_and_grad(&raw).unwrap();
    for index in 0..raw.len() {
        let mut plus = raw.clone();
        let mut minus = raw.clone();
        plus[index] += 1e-6;
        minus[index] -= 1e-6;
        let difference = (objective.value_and_grad(&plus).unwrap().0
            - objective.value_and_grad(&minus).unwrap().0)
            / 2e-6;
        assert!((difference - gradient[index]).abs() < 2e-7);
    }
}
