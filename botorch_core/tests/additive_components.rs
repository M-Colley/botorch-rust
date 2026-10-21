use botorch_core::constraints::Constraint;
use botorch_core::kernels::{KernelKind, KernelNode};
use botorch_core::models::gp::additive::AdditiveComponentModel;
use botorch_core::models::{BatchedMultiOutputGP, BatchedMultiOutputGPOptions, Model};
use botorch_core::{Matrix, Vector};

#[test]
fn component_means_reconstruct_and_jacobians_match_finite_differences() {
    for kind in [KernelKind::Rbf, KernelKind::Matern32, KernelKind::Matern52] {
        let kernel = KernelNode::orthogonal_additive(
            2,
            KernelNode::stationary(kind, None).unwrap(),
            true,
            12,
            true,
            true,
            Constraint::positive(),
        )
        .unwrap();
        let train =
            Matrix::from_row_slice(5, 2, &[0.1, 0.2, 0.3, 0.7, 0.5, 0.4, 0.7, 0.9, 0.9, 0.6]);
        let targets = Matrix::from_column_slice(5, 1, &[7.2, 7.9, 7.7, 8.3, 8.8]);
        let gp = BatchedMultiOutputGP::new(
            vec![train],
            vec![targets],
            vec![],
            BatchedMultiOutputGPOptions {
                covar_tree: Some(kernel),
                ..Default::default()
            },
        )
        .unwrap();
        let model = AdditiveComponentModel::new(&gp, 0).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.23, 0.45, 0.68, 0.76]);
        let mut sum = Vector::zeros(2);
        for component in 0..4 {
            let leaf = model.batch_element(component).unwrap();
            let analytic = leaf.joint_posterior_with_grad(&x).unwrap();
            sum += &analytic.mean;
            for point in 0..2 {
                for column in 0..2 {
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus[(point, column)] += 1e-6;
                    minus[(point, column)] -= 1e-6;
                    let a = leaf.posterior(&plus, false).unwrap();
                    let b = leaf.posterior(&minus, false).unwrap();
                    let dm = (a.mean - b.mean) / 2e-6;
                    let dc = (a.covariance - b.covariance) / 2e-6;
                    assert!(
                        (&analytic.dmean[point * 2 + column] - dm).amax() < 2e-7,
                        "{kind:?} component{component}"
                    );
                    assert!(
                        (&analytic.dcov[point * 2 + column] - dc).amax() < 2e-7,
                        "{kind:?} component{component}"
                    );
                }
            }
        }
        let full = gp.posterior_at(0, &x, false).unwrap();
        assert!((sum - full.mean).amax() < 1e-12);
    }
}
