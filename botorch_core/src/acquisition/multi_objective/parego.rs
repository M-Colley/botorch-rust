//! qLogNParEGO (`botorch.acquisition.multi_objective.parego.qLogNParEGO`;
//! Daulton et al., 2020, with the log-improvement of Ament et al., 2023;
//! Knowles, 2006, for ParEGO): qLogNEI of an augmented Chebyshev
//! scalarization of the objectives with (random) weights from the simplex.

use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::logei::{
    QLogNoisyExpectedImprovement, QLogNoisyExpectedImprovementOptions, TAU_MAX, TAU_RELU,
};
use crate::acquisition::mc_objective::{McObjective, McSamples, SamplePullback, SampleValues};
use crate::acquisition::multi_objective::mo_sampler_or_default;
use crate::acquisition::multi_objective::objective::{
    check_values, IdentityMcMultiOutputObjective, McMultiOutputObjective,
};
use crate::acquisition::utils::PruneOptions;
use crate::acquisition::{check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::multi_objective::scalarization::ChebyshevScalarization;
use crate::utils::objective::OutcomeConstraints;
use crate::utils::sampling::sample_simplex;
use crate::Matrix;

/// The composite objective of qLogNParEGO: the augmented Chebyshev
/// scalarization of a multi-output objective (BoTorch's
/// `GenericMCObjective(lambda samples, X: chebyshev(org_objective(samples, X), X))`),
/// evaluated natively with exact gradients.
#[derive(Debug)]
pub struct ChebyshevMcObjective {
    objective: Arc<dyn McMultiOutputObjective>,
    scalarization: ChebyshevScalarization,
}

impl ChebyshevMcObjective {
    /// The scalarization of `objective`'s outputs.
    pub fn new(
        objective: Arc<dyn McMultiOutputObjective>,
        scalarization: ChebyshevScalarization,
    ) -> Self {
        Self {
            objective,
            scalarization,
        }
    }
}

impl McObjective for ChebyshevMcObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let out = self.objective.evaluate(samples, x, with_grad)?;
        check_values(
            &out.values,
            samples,
            x,
            self.objective.verify_output_shape(),
        )?;
        let (n, k, s) = (
            out.values.num_points(),
            out.values.num_outputs(),
            out.values.num_samples(),
        );
        let mw = self.scalarization.num_outcomes();
        if k != mw {
            return Err(BotorchError::DimensionMismatch(format!(
                "The size of tensor a ({k}) must match the size of tensor b ({mw}) at \
                 non-singleton dimension 2"
            )));
        }
        let data = out.values.data();
        let mut values = Matrix::zeros(n, s);
        let mut grads = if with_grad {
            Matrix::zeros(k * n, s)
        } else {
            Matrix::zeros(0, 0)
        };
        let mut y = vec![0.0; k];
        let mut g = vec![0.0; k];
        for c in 0..s {
            for p in 0..n {
                for (t, yt) in y.iter_mut().enumerate() {
                    *yt = data[(t * n + p, c)];
                }
                values[(p, c)] = if with_grad {
                    let v = self.scalarization.evaluate_with_grad(&y, &mut g);
                    for (t, gt) in g.iter().enumerate() {
                        grads[(t * n + p, c)] = *gt;
                    }
                    v
                } else {
                    self.scalarization.evaluate(&y)
                };
            }
        }
        let pullback: Option<SamplePullback> = match out.pullback {
            Some(inner) if with_grad => Some(Box::new(move |bar: &Matrix| {
                let mut obj_bar = grads;
                for t in 0..k {
                    for p in 0..n {
                        for c in 0..s {
                            obj_bar[(t * n + p, c)] *= bar[(p, c)];
                        }
                    }
                }
                inner(&obj_bar)
            })),
            _ => None,
        };
        Ok(SampleValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        self.objective.is_pointwise()
    }
}

/// Options of [`QLogNParEGO`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QLogNParEGOOptions {
    /// The weights of the Chebyshev scalarization (one per objective);
    /// `None` draws them uniformly from the simplex.
    pub scalarization_weights: Option<Vec<f64>>,
    /// The seed of the random weights (`None`: a random seed; BoTorch draws
    /// them from torch's global generator).
    pub weights_seed: Option<u64>,
    /// The sampler (default: 128 Sobol samples with a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The multi-output objective applied before the scalarization
    /// (default: the identity on all outputs).
    pub objective: Option<Arc<dyn McMultiOutputObjective>>,
    /// Outcome constraints with their temperatures (default `eta = 1e-3`).
    pub constraints: Option<OutcomeConstraints>,
    /// Pending points.
    pub x_pending: Option<Matrix>,
    /// Fat-tailed approximations (default `true`).
    pub fat: bool,
    /// Prune the baseline with `prune_inferior_points` under the composite
    /// objective (default `false`).
    pub prune_baseline: bool,
    /// Settings of the pruning.
    pub prune_options: PruneOptions,
    /// Temperature of the smooth ReLU (default `1e-6`).
    pub tau_relu: f64,
    /// Temperature of the smooth maximum (default `1e-2`).
    pub tau_max: f64,
    /// Incremental improvement over pending points (default `true`).
    pub incremental: bool,
}

impl Default for QLogNParEGOOptions {
    fn default() -> Self {
        Self {
            scalarization_weights: None,
            weights_seed: None,
            sampler: None,
            objective: None,
            constraints: None,
            x_pending: None,
            fat: true,
            prune_baseline: false,
            prune_options: PruneOptions::default(),
            tau_relu: TAU_RELU,
            tau_max: TAU_MAX,
            incremental: true,
        }
    }
}

/// q-LogNParEGO (BoTorch's `qLogNParEGO`): qLogNEI of the augmented
/// Chebyshev scalarization (`alpha = 0.05`) of the objectives, normalized by
/// the objective values of the posterior mean at the baseline.
#[derive(Debug)]
pub struct QLogNParEGO {
    inner: Arc<QLogNoisyExpectedImprovement>,
    weights: Vec<f64>,
    y_baseline: Matrix,
    scalarization: ChebyshevScalarization,
}

impl QLogNParEGO {
    /// `qLogNParEGO(model, X_baseline, scalarization_weights, ...)`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the weights do not have one
    /// value per objective, [`BotorchError::Unsupported`] for batched models
    /// or negative weights without baseline points, and errors of qLogNEI.
    pub fn new(
        model: Arc<dyn Model>,
        x_baseline: &Matrix,
        options: QLogNParEGOOptions,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        let objective: Arc<dyn McMultiOutputObjective> = options
            .objective
            .unwrap_or_else(|| Arc::new(IdentityMcMultiOutputObjective::default()));
        // Y_baseline = objective(model.posterior(X_baseline).mean).
        let post = model.posterior(x_baseline, false)?;
        let (n, m) = (x_baseline.nrows(), model.num_outputs());
        let post_mean = if post.is_transformed() {
            post.moments()?.0
        } else {
            post.mean.clone()
        };
        let mean = McSamples::new(
            Matrix::from_column_slice(m * n, 1, post_mean.as_slice()),
            n,
            m,
        )?;
        let yb = objective.evaluate(&mean, None, false)?.values;
        let k = yb.num_outputs();
        let y_baseline = Matrix::from_fn(yb.num_points(), k, |p, t| yb.value(0, p, t));
        let weights = match options.scalarization_weights {
            Some(w) => w,
            None => sample_simplex(k, 1, false, options.weights_seed)?
                .row(0)
                .iter()
                .copied()
                .collect(),
        };
        let scalarization = ChebyshevScalarization::new(&weights, &y_baseline, 0.05)?;
        let composite = Arc::new(ChebyshevMcObjective::new(objective, scalarization.clone()));
        let inner = QLogNoisyExpectedImprovement::new(
            model,
            x_baseline,
            QLogNoisyExpectedImprovementOptions {
                sampler: Some(mo_sampler_or_default(options.sampler)?),
                x_pending: options.x_pending,
                fat: options.fat,
                prune_baseline: options.prune_baseline,
                prune_options: options.prune_options,
                tau_max: options.tau_max,
                tau_relu: options.tau_relu,
                incremental: options.incremental,
                objective: Some(composite),
                constraints: options.constraints,
                constraint_fat: None,
                lower_bound_points: None,
            },
        )?;
        Ok(Self {
            inner: Arc::new(inner),
            weights,
            y_baseline,
            scalarization,
        })
    }

    /// The scalarization weights.
    pub fn scalarization_weights(&self) -> &[f64] {
        &self.weights
    }

    /// The objective values of the posterior mean at the baseline (BoTorch's
    /// `Y_baseline`).
    pub fn y_baseline(&self) -> &Matrix {
        &self.y_baseline
    }

    /// The Chebyshev scalarization.
    pub fn scalarization(&self) -> &ChebyshevScalarization {
        &self.scalarization
    }

    /// The underlying qLogNEI (of the scalarized objective).
    pub fn log_nei(&self) -> &Arc<QLogNoisyExpectedImprovement> {
        &self.inner
    }
}

impl AcquisitionFunction for QLogNParEGO {
    fn model(&self) -> &Arc<dyn Model> {
        self.inner.model()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        self.inner.forward(x)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        self.inner.forward_with_grad(x)
    }

    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        self.inner.forward_many(xs)
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.inner.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.inner.set_x_pending(x_pending)
    }

    fn is_log(&self) -> bool {
        true
    }
}
