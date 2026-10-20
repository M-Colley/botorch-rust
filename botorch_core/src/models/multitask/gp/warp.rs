//! Fixed per-MCMC-sample Kumaraswamy transformations of non-task features.

use super::*;
use crate::transforms::utils::{kumaraswamy_warp, kumaraswamy_warp_with_grad};

#[derive(Clone, Debug)]
pub(super) struct InputWarp {
    indices: Vec<usize>,
    c0: Vec<f64>,
    c1: Vec<f64>,
    eps: f64,
}

impl InputWarp {
    pub(super) fn apply(&self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        for (j, &c) in self.indices.iter().enumerate() {
            for i in 0..x.nrows() {
                out[(i, c)] = kumaraswamy_warp(x[(i, c)], self.c0[j], self.c1[j], self.eps);
            }
        }
        out
    }

    pub(super) fn with_factors(&self, x: &Matrix) -> (Matrix, Matrix) {
        let mut out = x.clone();
        let mut derivative = Matrix::from_element(x.nrows(), x.ncols(), 1.0);
        for (j, &c) in self.indices.iter().enumerate() {
            for i in 0..x.nrows() {
                let (value, dx, _, _) =
                    kumaraswamy_warp_with_grad(x[(i, c)], self.c0[j], self.c1[j], self.eps);
                out[(i, c)] = value;
                derivative[(i, c)] = dx;
            }
        }
        (out, derivative)
    }
}

impl MultiTaskGP {
    /// Set fixed Kumaraswamy concentrations after the fitted normalization.
    /// These are per-MCMC-sample transformations, not extra MAP parameters.
    /// Original task labels and task derivatives remain unchanged.
    ///
    /// # Errors
    /// Invalid/duplicate/task indices, nonpositive concentrations, invalid
    /// margin or a failed transformation of the existing training data.
    pub fn set_input_warp(
        &mut self,
        indices: Vec<usize>,
        c0: Vec<f64>,
        c1: Vec<f64>,
        eps: f64,
    ) -> Result<()> {
        let cols = self.train_x.ncols();
        let mut unique = indices.clone();
        unique.sort_unstable();
        unique.dedup();
        if indices.is_empty()
            || unique.len() != indices.len()
            || indices
                .iter()
                .any(|&i| i >= cols || i == self.layout.task_feature())
        {
            return Err(BotorchError::InvalidArgument(
                "Warp indices must be unique non-task feature columns".into(),
            ));
        }
        if c0.len() != indices.len()
            || c1.len() != indices.len()
            || c0.iter().chain(&c1).any(|c| !c.is_finite() || *c <= 0.0)
            || !eps.is_finite()
            || eps <= 0.0
            || eps >= 0.5
        {
            return Err(BotorchError::InvalidArgument(
                "Warp concentrations must be finite positive vectors and eps in (0,0.5)".into(),
            ));
        }
        let warp = InputWarp {
            indices,
            c0,
            c1,
            eps,
        };
        let normalized = apply_input_transform(
            self.input_transform.as_ref(),
            self.input_indices.as_deref(),
            &self.train_x,
        )?;
        let (kx, tasks) = map_task_column(&self.layout, &warp.apply(&normalized), self.validate)?;
        self.train_kx = kx;
        self.train_tasks = tasks;
        self.input_warp = Some(warp);
        self.cache = OnceLock::new();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warped_multitask_query_adjoints_and_conditioning_match_finite_differences() {
        let x = Matrix::from_row_slice(
            6,
            3,
            &[
                0.1, 3.0, 0.2, 0.3, 7.0, 0.8, 0.8, 3.0, 0.5, 0.4, 7.0, 0.6, 0.5, 3.0, 0.3, 0.9,
                7.0, 0.7,
            ],
        );
        let y = Matrix::from_fn(6, 1, |i, _| (x[(i, 0)] + x[(i, 2)]).sin());
        let mut model = MultiTaskGP::new(x, y, 1, MultiTaskGPOptions::default()).unwrap();
        model
            .set_input_warp(vec![0, 2], vec![0.7, 1.4], vec![1.3, 0.6], 0.03)
            .unwrap();
        let query = Matrix::from_row_slice(2, 2, &[0.2, 0.7, 0.6, 0.4]);
        let derivatives = model.joint_posterior_with_grad(&query).unwrap();
        let ma = Vector::from_element(4, 0.4);
        let ca = Matrix::from_fn(4, 4, |i, j| 0.03 * (i + 2 * j) as f64);
        let gradient = model.joint_posterior_vjp(&query, 0..2, &ma, &ca).unwrap();
        for p in 0..2 {
            for c in 0..2 {
                let mut changed = query.clone();
                changed[(p, c)] += 1e-6;
                let plus = model.posterior(&changed, false).unwrap();
                changed[(p, c)] -= 2e-6;
                let minus = model.posterior(&changed, false).unwrap();
                let dm = (plus.mean - minus.mean) * 5e5;
                let dc = (plus.covariance - minus.covariance) * 5e5;
                assert!((&dm - &derivatives.dmean[p * 2 + c]).amax() < 1e-6);
                assert!((&dc - &derivatives.dcov[p * 2 + c]).amax() < 1e-6);
                assert!(
                    (ma.dot(&dm) + ca.component_mul(&dc).sum() - gradient[(p, c)]).abs() < 1e-6
                );
            }
        }
        let full = Matrix::from_row_slice(1, 3, &[0.25, 7.0, 0.45]);
        let conditioned = model
            .condition_on_observations(&full, &Matrix::from_element(1, 1, 0.8), None)
            .unwrap();
        assert!(conditioned.input_warp.is_some());
        assert_eq!(conditioned.train_x.nrows(), 7);
        assert!(conditioned
            .posterior(&query, false)
            .unwrap()
            .mean
            .iter()
            .all(|v| v.is_finite()));
    }
}
