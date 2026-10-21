//! Updating fitted inverse maps without refitting stored training data.

use super::*;
use crate::transforms::outcome::OutcomeStep;

impl SingleTaskGP {
    /// Replace the fitted outcome map while preserving training targets,
    /// observation noise, hyperparameters, and the training factorization.
    ///
    /// # Errors
    /// Unfitted or incorrectly sized maps, or both representations supplied.
    /// Failure leaves the model unchanged.
    pub fn set_fitted_outcome_state(
        &mut self,
        standardize: Option<Standardize>,
        chain: Option<OutcomeChain>,
    ) -> Result<bool> {
        if standardize.is_some() && chain.is_some() {
            return Err(BotorchError::InvalidArgument(
                "Supply one fitted outcome-map representation".into(),
            ));
        }
        let validate = |st: &Standardize| {
            if !st.is_trained() || st.means().len() != 1 {
                Err(BotorchError::InvalidArgument(
                    "A fitted single-output standardization is required".into(),
                ))
            } else {
                Ok(())
            }
        };
        if let Some(st) = &standardize {
            validate(st)?;
        }
        if let Some(chain) = &chain {
            for step in chain.steps() {
                match step {
                    OutcomeStep::Standardize(st) => validate(st)?,
                    OutcomeStep::StratifiedStandardize(st) if !st.is_trained() => {
                        return Err(BotorchError::InvalidArgument(
                            "A fitted stratified standardization is required".into(),
                        ));
                    }
                    _ => {}
                }
            }
        }
        let changed = self.outcome_transform != standardize || self.outcome_chain != chain;
        if changed {
            self.outcome_transform = standardize;
            self.outcome_chain = chain;
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_maps_preserve_the_fitted_training_law_and_original_snapshots() {
        let x = Matrix::from_column_slice(4, 1, &[0.1, 0.35, 0.7, 0.9]);
        let y = Matrix::from_column_slice(4, 1, &[1.2, 1.7, 2.4, 3.1]);
        let mut model = SingleTaskGP::new(
            x,
            y,
            SingleTaskGPOptions {
                train_yvar: Some(Matrix::from_element(4, 1, 0.04)),
                ..Default::default()
            },
        )
        .unwrap();
        let query = Matrix::from_column_slice(2, 1, &[0.2, 0.65]);
        let original = model.clone();
        let before = model.posterior(&query, true).unwrap();
        let targets = model.train_y_tf.clone();
        let noise = model.train_yvar.clone();
        let mll = crate::mll::ExactMarginalLogLikelihood::new(&model)
            .value()
            .unwrap();
        let old = model.outcome_transform().unwrap();
        let (offset, scale) = (old.means()[0], old.stdvs()[0]);
        let fitted = Standardize::fitted(vec![100.0], vec![1.7], 1e-8).unwrap();
        assert!(model
            .set_fitted_outcome_state(Some(fitted.clone()), None)
            .unwrap());
        assert!(!model.set_fitted_outcome_state(Some(fitted), None).unwrap());
        let after = model.posterior(&query, true).unwrap();
        let ratio = 1.7 / scale;
        assert!((after.mean - before.mean.map(|x| (x - offset) * ratio + 100.0)).amax() < 1e-12);
        assert!((after.covariance - &before.covariance * ratio.powi(2)).amax() < 1e-12);
        assert_eq!(model.train_y_tf, targets);
        assert_eq!(model.train_yvar, noise);
        assert_eq!(
            crate::mll::ExactMarginalLogLikelihood::new(&model)
                .value()
                .unwrap(),
            mll
        );
        assert_eq!(original.posterior(&query, true).unwrap(), before);

        let chain = OutcomeChain::new(vec![
            OutcomeStep::Standardize(Standardize::fitted(vec![1.0], vec![2.0], 1e-8).unwrap()),
            OutcomeStep::Standardize(Standardize::fitted(vec![3.0], vec![4.0], 1e-8).unwrap()),
        ]);
        model.set_fitted_outcome_state(None, Some(chain)).unwrap();
        let after = model.posterior(&query, true).unwrap();
        let ratio = 8.0 / scale;
        assert!((after.mean - before.mean.map(|x| (x - offset) * ratio + 7.0)).amax() < 1e-12);
        assert!((after.covariance - &before.covariance * ratio.powi(2)).amax() < 1e-12);
        assert_eq!(model.train_y_tf, targets);
        assert_eq!(
            crate::mll::ExactMarginalLogLikelihood::new(&model)
                .value()
                .unwrap(),
            mll
        );
        let saved = model.outcome_chain.clone();
        assert!(model
            .set_fitted_outcome_state(Some(Standardize::new()), None)
            .is_err());
        assert_eq!(model.outcome_chain, saved);
    }
}
