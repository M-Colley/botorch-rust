//! Compare the efficient identity to explicit conditional Gaussians using
//! the kernel matrices, noise and targets recorded from BoTorch.

mod common;

use botorch_core::{psd_safe_cholesky, Matrix, Vector};
use common::{build_model, load_fixtures};

#[test]
fn loo_matches_conditionals_of_botorch_fixtures() {
    for fixture in load_fixtures() {
        let model = build_model(&fixture);
        let n = model.num_train();
        let noise = model.likelihood_noise();
        let mut covariance = Matrix::from_fn(n, n, |i, j| fixture.kernel.k_train[i][j]);
        for i in 0..n {
            covariance[(i, i)] += noise[if noise.len() == 1 { 0 } else { i }];
        }
        for noisy in [false, true] {
            let (mean, variance) = model.loo_predictions(noisy).unwrap();
            for i in 0..n {
                let keep: Vec<_> = (0..n).filter(|&j| j != i).collect();
                let mut expected_mean = model.constant();
                let mut expected_variance = covariance[(i, i)];
                if n > 1 {
                    let train =
                        Matrix::from_fn(n - 1, n - 1, |r, c| covariance[(keep[r], keep[c])]);
                    let cross = Vector::from_fn(n - 1, |r, _| covariance[(keep[r], i)]);
                    let residual = Vector::from_fn(n - 1, |r, _| {
                        fixture.transformed.train_y[keep[r]] - model.constant()
                    });
                    let chol = psd_safe_cholesky(&train).unwrap();
                    expected_mean += cross.dot(&chol.solve_vec(&residual));
                    expected_variance -= cross.dot(&chol.solve_vec(&cross));
                }
                if !noisy {
                    expected_variance -= noise[if noise.len() == 1 { 0 } else { i }];
                }
                let mean_error = (mean[i] - expected_mean).abs();
                let variance_error = (variance[i] - expected_variance.max(0.0)).abs();
                assert!(mean_error < 1e-8, "{}: mean {mean_error}", fixture.name);
                assert!(
                    variance_error < 1e-8,
                    "{}: variance {variance_error}",
                    fixture.name
                );
            }
        }
    }
}
