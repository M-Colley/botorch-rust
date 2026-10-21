//! Query adjoint of Kronecker Matheron draws without changing their base layout.

use super::*;

impl KroneckerMultiTaskGP {
    /// Query VJP of public point-major mean and clamped marginal variance.
    pub fn marginal_vjp(
        &self,
        x: &Matrix,
        mean: &Matrix,
        variance: &Matrix,
        noise: bool,
    ) -> Result<Matrix> {
        let (q, m) = (x.nrows(), self.num_tasks());
        if mean.shape() != (q, m) || variance.shape() != (q, m) {
            return Err(BotorchError::DimensionMismatch(
                "Invalid Kronecker marginal adjoint shape".into(),
            ));
        }
        let p = self.posterior(x, noise)?;
        let ma = Vector::from_fn(q * m, |i, _| mean[(i % q, i / q)]);
        let ca = Matrix::from_diagonal(&Vector::from_fn(q * m, |i, _| {
            if p.covariance[(i, i)] > 1e-10 {
                variance[(i % q, i / q)]
            } else {
                0.0
            }
        }));
        self.joint_posterior_vjp(x, 0..q, &ma, &ca)
    }

    /// VJP of point-major Matheron samples with respect to raw query features.
    pub fn matheron_samples_vjp(
        &self,
        x: &Matrix,
        base: &Matrix,
        adjoint: &Matrix,
        observation_noise: bool,
    ) -> Result<Matrix> {
        let c = self.cache()?;
        let (xt, jacobian) = self.transform_x(x)?;
        let (n, q, m) = (self.num_train(), xt.nrows(), self.num_tasks());
        let joint = (n + q) * m;
        let total = joint + n * m + if observation_noise { q * m } else { 0 };
        if base.ncols() != total || adjoint.shape() != (base.nrows(), q * m) {
            return Err(BotorchError::DimensionMismatch(
                "Invalid Kronecker Matheron adjoint shape".into(),
            ));
        }
        let all = Matrix::from_fn(n + q, xt.ncols(), |i, j| {
            if i < n {
                self.x_tf[(i, j)]
            } else {
                xt[(i - n, j)]
            }
        });
        let root = psd_safe_cholesky(&c.data.covariance_symmetric(&all)?)?;
        let task_root = psd_safe_cholesky(&c.b)?;
        let noise_root = psd_safe_cholesky(&c.noise)?;
        let residual = self.residuals(self.raw.as_slice());
        let (_, scale) = self.outcome_affine();
        let mut root_adjoint = Matrix::zeros(n + q, n + q);
        let mut cross_adjoint = Matrix::zeros(q, n);
        for sample in 0..base.nrows() {
            let normals = Matrix::from_fn(n + q, m, |i, t| base[(sample, i * m + t)]);
            let mixed = normals * task_root.l().transpose();
            let prior = root.l() * &mixed;
            let noise = Matrix::from_fn(n, m, |i, t| base[(sample, joint + i * m + t)])
                * noise_root.l().transpose();
            let diff = &residual
                - Vector::from_fn(n * m, |i, _| prior[(i / m, i % m)] + noise[(i / m, i % m)]);
            let solved = c.chol.solve_vec(&diff);
            let update = Matrix::from_fn(n, m, |i, t| {
                (0..m).map(|u| c.b[(t, u)] * solved[i * m + u]).sum()
            });
            let weights = Matrix::from_fn(q, m, |i, t| scale[t] * adjoint[(sample, i * m + t)]);
            let rows = &weights * mixed.transpose();
            for i in 0..q {
                for j in 0..n + q {
                    root_adjoint[(n + i, j)] += rows[(i, j)];
                }
            }
            cross_adjoint += weights * update.transpose();
        }
        let gradient = crate::models::matheron::data_query_vjp(
            &c.data,
            &all,
            n,
            root.l(),
            &root_adjoint,
            &cross_adjoint,
        )?;
        Ok(Matrix::from_fn(q, x.ncols(), |i, j| {
            gradient[(i, j)] * jacobian[j]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matheron_query_vjp_matches_finite_differences() {
        for rank in [0, 2] {
            for standardized in [false, true] {
                for noise in [false, true] {
                    let model = super::super::tests::model(rank, standardized);
                    let x = Matrix::from_row_slice(2, 2, &[0.23, 0.37, 0.62, 0.76]);
                    let n = model.num_train();
                    let m = model.num_tasks();
                    let total = (2 * n + 2) * m + if noise { 2 * m } else { 0 };
                    let base =
                        Matrix::from_fn(3, total, |i, j| ((i * 11 + j * 3) as f64 * 0.31).sin());
                    let adjoint =
                        Matrix::from_fn(3, 2 * m, |i, j| ((i * 3 + j * 7) as f64 * 0.17).cos());
                    let gradient = model
                        .matheron_samples_vjp(&x, &base, &adjoint, noise)
                        .unwrap();
                    for i in 0..2 {
                        for j in 0..2 {
                            let mut plus = x.clone();
                            plus[(i, j)] += 1e-6;
                            let mut minus = x.clone();
                            minus[(i, j)] -= 1e-6;
                            let finite = (model
                                .matheron_samples(&plus, &base, noise)
                                .unwrap()
                                .component_mul(&adjoint)
                                .sum()
                                - model
                                    .matheron_samples(&minus, &base, noise)
                                    .unwrap()
                                    .component_mul(&adjoint)
                                    .sum())
                                / 2e-6;
                            assert!(
                                (finite - gradient[(i, j)]).abs() < 2e-5 * (1.0 + finite.abs()),
                                "rank={rank} std={standardized} noise={noise}: FD={finite} VJP={}",
                                gradient[(i, j)]
                            );
                        }
                    }
                }
            }
        }
    }
}
