//! Exact feature/time query adjoints for native latent-grid Matheron draws.

use super::*;
use crate::linalg::cholesky_backward;
use crate::models::matheron::{cross_self_query_vjp, kronecker_vjp};

impl LatentKroneckerGP {
    fn feature_pullback(&self, x: &Matrix, transformed: &Matrix) -> Result<Matrix> {
        let Some(chain) = &self.options.input_transform else {
            return Ok(transformed.clone());
        };
        let (_, tape) = chain.transform_with_tape_mode(x, TransformMode::Eval)?;
        let jacobians = tape.jacobians_through(chain)?;
        Ok(Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            (0..transformed.ncols())
                .map(|k| transformed[(i, k)] * jacobians[i][(k, j)])
                .sum()
        }))
    }

    /// Input VJP of posterior-grid moments, for both feature and time queries.
    pub fn posterior_grid_vjp(
        &self,
        x: &Matrix,
        t: &Matrix,
        mean: &Vector,
        covariance: &Matrix,
    ) -> Result<(Matrix, Matrix)> {
        Self::validate_t(t)?;
        let xt = self.transformed(x)?;
        let cartesian = cartesian_inputs(&xt, t);
        let size = cartesian.nrows();
        if mean.len() != size || covariance.shape() != (size, size) {
            return Err(BotorchError::DimensionMismatch(
                "Grid posterior adjoints have inconsistent shapes".into(),
            ));
        }
        let dense;
        let model = match self.backend()? {
            Backend::Masked(m) => m.as_ref(),
            Backend::Full(_) => {
                dense = self.dense(&self.raw)?;
                &dense
            }
        };
        let scale = self.affine().1;
        let gradient = model.joint_posterior_vjp(
            &cartesian,
            0..size,
            &(mean * scale),
            &(covariance * (scale * scale)),
        )?;
        let features = Matrix::from_fn(x.nrows(), xt.ncols(), |i, j| {
            (0..t.nrows())
                .map(|a| gradient[(i * t.nrows() + a, j)])
                .sum()
        });
        let times = Matrix::from_fn(t.nrows(), 1, |a, _| {
            (0..x.nrows())
                .map(|i| gradient[(i * t.nrows() + a, xt.ncols())])
                .sum()
        });
        Ok((self.feature_pullback(x, &features)?, times))
    }

    /// Feature/time input VJP of the exact fixed-base native Matheron map.
    pub fn samples_vjp(
        &self,
        x: &Matrix,
        t: &Matrix,
        base: &Matrix,
        adjoint: &Matrix,
    ) -> Result<(Matrix, Matrix)> {
        Self::validate_t(t)?;
        let xt = self.transformed(x)?;
        let (full, observed, test) = (self.y.len(), self.observed.len(), x.nrows() * t.nrows());
        if base.ncols() != full + observed + test || adjoint.shape() != (base.nrows(), test) {
            return Err(BotorchError::DimensionMismatch(
                "Invalid latent Matheron adjoint shape".into(),
            ));
        }
        let (kx, kt) = self.kernels(&self.raw)?;
        let root_x = psd_safe_cholesky(&kx.covariance_symmetric(&self.x)?)?;
        let root_t = psd_safe_cholesky(&kt.covariance_symmetric(&self.t)?)?;
        let cross_x = kx.covariance(&self.x, &xt)?;
        let cross_t = kt.covariance(&self.t, t)?;
        let mut ax = cross_x.clone();
        root_x.solve_lower_mut(&mut ax);
        let mut at = cross_t.clone();
        root_t.solve_lower_mut(&mut at);
        let a = ax.kronecker(&at);
        let self_x = kx.covariance_symmetric(&xt)?;
        let self_t = kt.covariance_symmetric(t)?;
        let root = psd_safe_cholesky(&(self_x.kronecker(&self_t) - a.transpose() * &a))?;
        let targets = self.train_targets().add_scalar(-self.mean(&self.raw));
        let mut a_bar = Matrix::zeros(full, test);
        let mut cross_bar = Matrix::zeros(full, test);
        let mut root_bar = Matrix::zeros(test, test);
        let scale = self.affine().1;
        for sample in 0..base.nrows() {
            let z = Vector::from_fn(full, |i, _| base[(sample, i)]);
            let prior = Tensor::new(vec![self.x.nrows(), self.t.nrows()], z.as_slice().to_vec())
                .mode_mul(0, root_x.l())
                .mode_mul(1, root_t.l());
            let rhs = Vector::from_fn(observed, |i, _| {
                targets[i]
                    - prior.data[self.observed[i]]
                    - self.noise().sqrt() * base[(sample, full + i)]
            });
            let solved = self.solve_observed(&rhs)?;
            let w = Vector::from_fn(test, |i, _| base[(sample, full + observed + i)]);
            let weights = Vector::from_fn(test, |i, _| scale * adjoint[(sample, i)]);
            a_bar += &z * weights.transpose();
            root_bar += &weights * w.transpose();
            for i in 0..observed {
                for j in 0..test {
                    cross_bar[(self.observed[i], j)] += solved[i] * weights[j];
                }
            }
        }
        let self_bar = cholesky_backward(root.l(), &root_bar);
        a_bar -= &a * (&self_bar + self_bar.transpose());
        let (mut cx_bar, mut ct_bar) = kronecker_vjp(&ax, &at, &a_bar);
        root_x.solve_lower_transpose_mut(&mut cx_bar);
        root_t.solve_lower_transpose_mut(&mut ct_bar);
        let (additional_x, additional_t) = kronecker_vjp(&cross_x, &cross_t, &cross_bar);
        cx_bar += additional_x;
        ct_bar += additional_t;
        let (sx_bar, st_bar) = kronecker_vjp(&self_x, &self_t, &self_bar);
        let features = cross_self_query_vjp(&kx, &self.x, &xt, &cx_bar, &sx_bar)?;
        let times = cross_self_query_vjp(&kt, &self.t, t, &ct_bar, &st_bar)?;
        Ok((self.feature_pullback(x, &features)?, times))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matheron_query_vjp_matches_grid_and_sample_finite_differences() {
        for missing in [false, true] {
            let model = super::super::tests::model(missing);
            let x = Matrix::from_column_slice(2, 1, &[0.28, 0.68]);
            let t = Matrix::from_column_slice(2, 1, &[0.23, 0.77]);
            let base = Matrix::from_fn(3, model.y.len() + model.observed.len() + 4, |i, j| {
                ((i * 11 + j * 3) as f64 * 0.31).sin()
            });
            let adjoint = Matrix::from_fn(3, 4, |i, j| ((i * 3 + j * 7) as f64 * 0.17).cos());
            let ma = Vector::from_fn(4, |i, _| 0.2 + i as f64 * 0.3);
            let ca = Matrix::from_fn(4, 4, |i, j| ((i * 3 + j * 7) as f64 * 0.21).cos());
            for samples in [false, true] {
                let (gx, gt) = if samples {
                    model.samples_vjp(&x, &t, &base, &adjoint).unwrap()
                } else {
                    model.posterior_grid_vjp(&x, &t, &ma, &ca).unwrap()
                };
                let value = |x: &Matrix, t: &Matrix| {
                    if samples {
                        model
                            .samples(x, t, &base)
                            .unwrap()
                            .component_mul(&adjoint)
                            .sum()
                    } else {
                        let p = model.posterior_grid(x, t).unwrap();
                        p.mean.dot(&ma) + p.covariance.component_mul(&ca).sum()
                    }
                };
                for time in [false, true] {
                    for i in 0..2 {
                        let (mut xp, mut xm, mut tp, mut tm) =
                            (x.clone(), x.clone(), t.clone(), t.clone());
                        if time {
                            tp[(i, 0)] += 1e-6;
                            tm[(i, 0)] -= 1e-6;
                        } else {
                            xp[(i, 0)] += 1e-6;
                            xm[(i, 0)] -= 1e-6;
                        }
                        let finite = (value(&xp, &tp) - value(&xm, &tm)) / 2e-6;
                        let gradient = if time { gt[(i, 0)] } else { gx[(i, 0)] };
                        assert!((finite-gradient).abs()<2e-5*(1.0+finite.abs()),"missing={missing} samples={samples} time={time}: FD={finite} VJP={gradient}");
                    }
                }
            }
        }
    }
}
