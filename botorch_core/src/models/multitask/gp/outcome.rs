//! Fitted positive affine maps for GPs built from transformed MCMC targets.

use super::*;

impl MultiTaskGP {
    /// Attach `raw_y = offset + scale * transformed_y` without refitting it.
    /// The model must currently have no outcome transform. Its training solve
    /// and noise already use transformed units; raw data metadata is restored
    /// to the declared original units. Conditioning also uses this fitted map.
    ///
    /// # Errors
    /// An existing transform, invalid/nonpositive scale, non-finite offset,
    /// or an overflowing transformation of the stored data. Failure is atomic.
    pub fn set_fitted_outcome_affine(&mut self, offset: f64, scale: f64) -> Result<()> {
        if self.outcome_transform.is_some() {
            return Err(BotorchError::InvalidArgument(
                "A fitted affine map requires a model without an outcome transform".into(),
            ));
        }
        let transform = Standardize::fitted(vec![offset], vec![scale], 1e-8)?;
        let raw_y = self.train_y.map(|value| offset + scale * value);
        let raw_variance = self
            .train_yvar
            .as_ref()
            .map(|v| v.map(|x| x * scale * scale));
        if raw_y.iter().any(|x| !x.is_finite())
            || raw_variance
                .as_ref()
                .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
        {
            return Err(BotorchError::InvalidArgument(
                "The fitted affine map overflows the stored original-space data".into(),
            ));
        }
        self.train_y = raw_y;
        self.train_yvar = raw_variance;
        self.outcome_transform = Some(transform);
        self.cache = OnceLock::new();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitted_affine_preserves_native_law_metadata_conditioning_and_adjoints() {
        let x = Matrix::from_row_slice(4, 2, &[0.1, 3.0, 0.4, 7.0, 0.6, 3.0, 0.9, 7.0]);
        let y = Matrix::from_column_slice(4, 1, &[-0.7, 0.3, 0.5, -0.1]);
        let variance = Matrix::from_element(4, 1, 0.03);
        let base = MultiTaskGP::new(
            x,
            y.clone(),
            1,
            MultiTaskGPOptions {
                train_yvar: Some(variance.clone()),
                outcome_transform: OutcomeTransformChoice::None,
                ..Default::default()
            },
        )
        .unwrap();
        let (offset, scale) = (1.7, 2.3);
        let mut model = base.clone();
        model.set_fitted_outcome_affine(offset, scale).unwrap();
        assert!((&model.train_y - y.map(|v| offset + scale * v)).amax() < 1e-15);
        assert!((model.train_yvar.as_ref().unwrap() - variance * (scale * scale)).amax() < 1e-15);
        assert_eq!(model.y_tf, base.y_tf);
        let query = Matrix::from_row_slice(2, 1, &[0.2, 0.7]);
        let check = |actual: &MultiTaskGP, reference: &MultiTaskGP| {
            let a = actual.posterior(&query, true).unwrap();
            let b = reference.posterior(&query, true).unwrap();
            assert!((a.mean - b.mean.map(|v| offset + scale * v)).amax() < 1e-12);
            assert!((a.covariance - b.covariance * (scale * scale)).amax() < 1e-12);
        };
        check(&model, &base);
        let ma = Vector::from_element(4, 0.4);
        let ca = Matrix::from_fn(4, 4, |i, j| 0.03 * (i + 2 * j) as f64);
        let actual = model.joint_posterior_vjp(&query, 0..2, &ma, &ca).unwrap();
        let expected = base
            .joint_posterior_vjp(&query, 0..2, &(&ma * scale), &(&ca * (scale * scale)))
            .unwrap();
        assert!((actual - expected).amax() < 1e-12);
        let added = Matrix::from_row_slice(1, 2, &[0.25, 7.0]);
        let raw_target = Matrix::from_element(1, 1, 0.8);
        let transformed_target = raw_target.map(|v| (v - offset) / scale);
        let a = model
            .condition_on_observations(&added, &raw_target, Some(&[0.04]))
            .unwrap();
        let b = base
            .condition_on_observations(&added, &transformed_target, Some(&[0.04]))
            .unwrap();
        check(&a, &b);
        assert!(model.set_fitted_outcome_affine(0.0, 1.0).is_err());
        check(&model, &base);
        for (offset, scale) in [
            (f64::NAN, 1.0),
            (0.0, -1.0),
            (0.0, 0.0),
            (0.0, f64::INFINITY),
            (1e308, 1e308),
        ] {
            let mut candidate = base.clone();
            assert!(candidate.set_fitted_outcome_affine(offset, scale).is_err());
            assert!(candidate.outcome_transform.is_none());
            assert_eq!(candidate.train_y, base.train_y);
        }
    }
}
