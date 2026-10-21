//! Exact query adjoints of the native Kronecker Matheron reparameterization.

use super::*;

impl HigherOrderGP {
    /// Query VJP of point-major marginal moments in the requested outcome units.
    pub fn mean_variance_vjp_in(
        &self,
        x: &Matrix,
        mean: &Matrix,
        variance: &Matrix,
        original_space: bool,
    ) -> Result<Matrix> {
        let (q, s) = (x.nrows(), self.num_targets());
        if mean.shape() != (q, s) || variance.shape() != (q, s) {
            return Err(BotorchError::DimensionMismatch(
                "Invalid HOGP marginal adjoint shape".into(),
            ));
        }
        let scale = if original_space { 1.0 } else { self.affine().1 };
        let ma = Vector::from_fn(q * s, |i, _| mean[(i % q, i / q)] / scale);
        let ca = Matrix::from_diagonal(&Vector::from_fn(q * s, |i, _| {
            variance[(i % q, i / q)] / (scale * scale)
        }));
        Model::joint_posterior_vjp(self, x, 0..q, &ma, &ca)
    }

    /// Query-input VJP of fixed-base Matheron samples, with held-fixed model
    /// parameters and normals. Adjoint columns use point-major target layout.
    pub fn rsample_vjp_in(
        &self,
        x: &Matrix,
        base: &Matrix,
        adjoint: &Matrix,
        original_space: bool,
    ) -> Result<Matrix> {
        let xt = self.transform_inputs(x)?;
        let c = self.cache()?;
        let f0 = &c.factors[0];
        let (n, q, s) = (self.train_x.nrows(), xt.nrows(), self.num_targets());
        let joint = (n + q) * s;
        if base.ncols() != joint + n * s || adjoint.shape() != (base.nrows(), q * s) {
            return Err(BotorchError::DimensionMismatch(
                "Invalid Matheron sample adjoint shape".into(),
            ));
        }
        let all = Matrix::from_fn(n + q, xt.ncols(), |i, j| {
            if i < n {
                self.train_x[(i, j)]
            } else {
                xt[(i - n, j)]
            }
        });
        let prep = f0.kernel.prepare(&all)?;
        let mut covariance = f0.kernel.symmetric(&prep);
        covariance.view_mut((0, 0), (n, n)).copy_from(&f0.k);
        let root = psd_safe_cholesky(&covariance)?;
        let latent_roots = c
            .factors
            .iter()
            .skip(1)
            .map(|f| psd_safe_cholesky(&f.k))
            .collect::<Result<Vec<_>>>()?;
        let mut dims = vec![n + q];
        dims.extend(&self.target_shape);
        let mut root_adjoint = Matrix::zeros(n + q, n + q);
        let mut cross_adjoint = Matrix::zeros(q, n);
        let scale = if original_space { self.affine().1 } else { 1.0 };
        for sample in 0..base.nrows() {
            let mut mixed = Tensor::new(
                dims.clone(),
                (0..joint).map(|k| base[(sample, k)]).collect(),
            );
            for (mode, latent) in latent_roots.iter().enumerate() {
                mixed = mixed.mode_mul(mode + 1, latent.l());
            }
            let mixed_matrix = Matrix::from_row_slice(n + q, s, &mixed.data);
            let prior = root.l() * &mixed_matrix;
            let mut rhs = self.y_tensor();
            for k in 0..n * s {
                rhs.data[k] -= prior[(k / s, k % s)] + c.noise.sqrt() * base[(sample, joint + k)];
            }
            let mut solved = Self::solve(&c.factors, &c.d, &rhs);
            for (mode, factor) in c.factors.iter().enumerate().skip(1) {
                solved = solved.mode_mul(mode, &factor.k);
            }
            let update = Matrix::from_row_slice(n, s, &solved.data);
            let weights = Matrix::from_fn(q, s, |i, t| scale * adjoint[(sample, i * s + t)]);
            let root_rows = &weights * mixed_matrix.transpose();
            for i in 0..q {
                for j in 0..n + q {
                    root_adjoint[(n + i, j)] += root_rows[(i, j)];
                }
            }
            cross_adjoint += weights * update.transpose();
        }
        let transformed = crate::models::matheron::data_query_vjp(
            &f0.kernel,
            &all,
            n,
            root.l(),
            &root_adjoint,
            &cross_adjoint,
        )?;
        let Some(chain) = &self.options.input_transform else {
            return Ok(transformed);
        };
        let (_, tape) = chain.transform_with_tape_mode(x, TransformMode::Eval)?;
        let jacobians = tape.jacobians_through(chain)?;
        Ok(Matrix::from_fn(q, x.ncols(), |i, j| {
            (0..xt.ncols())
                .map(|k| transformed[(i, k)] * jacobians[i][(k, j)])
                .sum()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matheron_query_vjp_matches_finite_differences() {
        for tensor in [false, true] {
            let model = super::super::tests::model(false, tensor);
            let x = Matrix::from_row_slice(2, 2, &[0.24, 0.36, 0.61, 0.73]);
            let n = model.train_inputs().nrows();
            let outputs = model.num_targets();
            let base = Matrix::from_fn(3, (2 * n + 2) * outputs, |i, j| {
                ((i * 11 + j * 3) as f64 * 0.37).sin()
            });
            let adjoint =
                Matrix::from_fn(3, 2 * outputs, |i, j| ((i * 3 + j * 7) as f64 * 0.13).cos());
            for original in [false, true] {
                let gradient = model.rsample_vjp_in(&x, &base, &adjoint, original).unwrap();
                for i in 0..2 {
                    for j in 0..2 {
                        let mut plus = x.clone();
                        plus[(i, j)] += 1e-6;
                        let mut minus = x.clone();
                        minus[(i, j)] -= 1e-6;
                        let finite = (model
                            .rsample_in(&plus, &base, original)
                            .unwrap()
                            .component_mul(&adjoint)
                            .sum()
                            - model
                                .rsample_in(&minus, &base, original)
                                .unwrap()
                                .component_mul(&adjoint)
                                .sum())
                            / 2e-6;
                        assert!(
                            (finite - gradient[(i, j)]).abs() < 2e-5 * (1.0 + finite.abs()),
                            "tensor={tensor} original={original}: FD={finite} VJP={}",
                            gradient[(i, j)]
                        );
                    }
                }
            }
        }
    }
}
