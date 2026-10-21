use super::*;
use crate::acquisition::mc_objective::SampleValues;
use crate::models::{SingleTaskGP, SingleTaskGPOptions};
use crate::utils::objective::LinearOutcomeConstraint;
use crate::OutcomeTransformChoice;

#[derive(Debug)]
struct AugmentedObjective;

impl McObjective for AugmentedObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let (n, s) = (samples.num_points(), samples.num_samples());
        let data = samples.data().clone();
        let values = Matrix::from_fn(n, 2 * s, |p, c| {
            let a = if c < s { 0.7 } else { 1.3 };
            a * data[(p, c % s)].powi(2) + 0.2 * a
        });
        let pullback: Option<SamplePullback> = with_grad.then(|| {
            Box::new(move |bar: &Matrix| {
                Ok((
                    Matrix::from_fn(n, s, |p, c| {
                        2.0 * data[(p, c)] * (0.7 * bar[(p, c)] + 1.3 * bar[(p, c + s)])
                    }),
                    None,
                ))
            }) as SamplePullback
        });
        Ok(SampleValues { values, pullback })
    }
}

#[test]
fn augmented_objectives_and_constraints_preserve_native_noisy_adjoints() {
    let train_x = Matrix::from_column_slice(5, 1, &[0.0, 0.2, 0.4, 0.7, 1.0]);
    let train_y = train_x.map(|x: f64| (5.0 * x).sin());
    let model: Arc<dyn Model> = Arc::new(
        SingleTaskGP::new(
            train_x,
            train_y,
            SingleTaskGPOptions {
                train_yvar: Some(Matrix::from_element(5, 1, 0.04)),
                outcome_transform: OutcomeTransformChoice::None,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let baseline = Matrix::from_column_slice(2, 1, &[0.15, 0.8]);
    let query = Matrix::from_column_slice(2, 1, &[0.3, 0.6]);
    for cached in [false, true] {
        let acquisition = QNoisyExpectedImprovement::new(
            Arc::clone(&model),
            &baseline,
            QNoisyExpectedImprovementOptions {
                sampler: Some(Arc::new(SobolQMCNormalSampler::new(16, Some(28)).unwrap())),
                objective: Some(Arc::new(AugmentedObjective)),
                constraints: Some(
                    OutcomeConstraints::uniform(
                        vec![Arc::new(
                            LinearOutcomeConstraint::new(Vector::from_element(1, 1.0), -0.6)
                                .unwrap(),
                        )],
                        0.1,
                    )
                    .unwrap(),
                ),
                prune_baseline: false,
                cache_root: cached,
                ..Default::default()
            },
        )
        .unwrap();
        let (_, gradient) = acquisition.forward_with_grad(&query).unwrap();
        for p in 0..2 {
            let mut plus = query.clone();
            let mut minus = query.clone();
            plus[(p, 0)] += 1e-6;
            minus[(p, 0)] -= 1e-6;
            let finite =
                (acquisition.forward(&plus).unwrap() - acquisition.forward(&minus).unwrap()) / 2e-6;
            assert!((finite - gradient[(p, 0)]).abs() < 2e-6);
        }
    }
}
