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
    pub(super) fn apply_feature_offsets(&self, x: &mut Matrix) {
        for &(column, offset) in &self.input_feature_offsets {
            for row in 0..x.nrows() {
                x[(row, column)] += offset;
            }
        }
    }

    /// Set fixed additive feature offsets after normalization and warping.
    /// This preserves the categorical task column and has unit derivative.
    /// An empty pair of vectors clears the offsets.
    ///
    /// # Errors
    /// Invalid, duplicate or task indices, non-finite offsets, unequal lengths,
    /// or invalid transformed training inputs. Failure leaves the model intact.
    pub fn set_input_feature_offsets(
        &mut self,
        indices: Vec<usize>,
        offsets: Vec<f64>,
    ) -> Result<()> {
        let mut unique = indices.clone();
        unique.sort_unstable();
        unique.dedup();
        if indices.len() != offsets.len()
            || unique.len() != indices.len()
            || indices
                .iter()
                .any(|&i| i >= self.train_x.ncols() || i == self.layout.task_feature())
            || offsets.iter().any(|x| !x.is_finite())
        {
            return Err(BotorchError::InvalidArgument(
                "Feature offsets must be finite and match unique non-task columns".into(),
            ));
        }
        let mut candidate = self.clone();
        candidate.input_feature_offsets = indices.into_iter().zip(offsets).collect();
        let transformed = candidate.transform_full(&candidate.train_x)?;
        let (kx, tasks) = map_task_column(&self.layout, &transformed, self.validate)?;
        self.input_feature_offsets = candidate.input_feature_offsets;
        self.train_kx = kx;
        self.train_tasks = tasks;
        self.cache = OnceLock::new();
        Ok(())
    }

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
        let mut candidate = self.clone();
        candidate.input_warp = Some(warp);
        let transformed = candidate.transform_full(&candidate.train_x)?;
        let (kx, tasks) = map_task_column(&self.layout, &transformed, self.validate)?;
        self.train_kx = kx;
        self.train_tasks = tasks;
        self.input_warp = candidate.input_warp;
        self.cache = OnceLock::new();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_warp_matches_explicit_linear_inputs_and_survives_conditioning() {
        let x = Matrix::from_row_slice(
            6,
            3,
            &[
                0.1, 3.0, 0.2, 0.3, 7.0, 0.8, 0.8, 3.0, 0.5, 0.4, 7.0, 0.6, 0.5, 3.0, 0.3, 0.9,
                7.0, 0.7,
            ],
        );
        let y = Matrix::from_fn(6, 1, |i, _| (x[(i, 0)] + x[(i, 2)]).sin());
        let options = MultiTaskGPOptions {
            covar_module: Some(KernelNode::linear(Some(2)).unwrap()),
            ..Default::default()
        };
        let mut model = MultiTaskGP::new(x.clone(), y.clone(), 1, options.clone()).unwrap();
        model
            .set_input_feature_offsets(vec![0, 2], vec![-0.5, -0.5])
            .unwrap();
        model
            .set_input_warp(vec![0, 2], vec![0.7, 1.4], vec![1.3, 0.6], 0.03)
            .unwrap();
        let transform = |points: &Matrix| {
            let mut out = points.clone();
            for row in 0..out.nrows() {
                out[(row, 0)] = kumaraswamy_warp(points[(row, 0)], 0.7, 1.3, 0.03) - 0.5;
                out[(row, 2)] = kumaraswamy_warp(points[(row, 2)], 1.4, 0.6, 0.03) - 0.5;
            }
            out
        };
        let explicit = MultiTaskGP::new(transform(&x), y, 1, options).unwrap();
        assert!((&model.train_kx - &explicit.train_kx).amax() < 1e-15);
        let full_query = Matrix::from_row_slice(2, 3, &[0.2, 7.0, 0.7, 0.6, 3.0, 0.4]);
        let check_law = |actual: &MultiTaskGP, reference: &MultiTaskGP| {
            let a = actual.posterior_with_tasks(&full_query, false).unwrap();
            let b = reference
                .posterior_with_tasks(&transform(&full_query), false)
                .unwrap();
            assert!((&a.mean - &b.mean).amax() < 1e-12);
            assert!((&a.covariance - &b.covariance).amax() < 1e-12);
        };
        check_law(&model, &explicit);
        let ma = Vector::from_element(2, 0.4);
        let ca = Matrix::from_row_slice(2, 2, &[0.2, -0.1, 0.3, 0.4]);
        let view = MultiTaskTaskInputs::new(Arc::new(model.clone()));
        let grad = view
            .joint_posterior_vjp(&full_query, 0..2, &ma, &ca)
            .unwrap();
        for row in 0..2 {
            assert_eq!(grad[(row, 1)], 0.0);
            for col in [0, 2] {
                let mut changed = full_query.clone();
                changed[(row, col)] += 1e-6;
                let plus = model.posterior_with_tasks(&changed, false).unwrap();
                changed[(row, col)] -= 2e-6;
                let minus = model.posterior_with_tasks(&changed, false).unwrap();
                let fd = (ma.dot(&(plus.mean - minus.mean))
                    + ca.component_mul(&(plus.covariance - minus.covariance))
                        .sum())
                    * 5e5;
                assert!((fd - grad[(row, col)]).abs() < 1e-6);
            }
        }
        let added = Matrix::from_row_slice(1, 3, &[0.25, 7.0, 0.45]);
        let target = Matrix::from_element(1, 1, 0.8);
        let conditioned = model
            .condition_on_observations(&added, &target, None)
            .unwrap();
        let explicit_conditioned = explicit
            .condition_on_observations(&transform(&added), &target, None)
            .unwrap();
        check_law(&conditioned, &explicit_conditioned);
        assert_eq!(
            conditioned.input_feature_offsets,
            vec![(0, -0.5), (2, -0.5)]
        );
        for (indices, offsets) in [
            (vec![1], vec![0.1]),
            (vec![0, 0], vec![0.1, 0.2]),
            (vec![0], vec![f64::NAN]),
            (vec![0], vec![]),
        ] {
            assert!(model.set_input_feature_offsets(indices, offsets).is_err());
            check_law(&model, &explicit);
        }
        model.set_input_feature_offsets(vec![], vec![]).unwrap();
        assert!(model.input_feature_offsets.is_empty());
    }

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
