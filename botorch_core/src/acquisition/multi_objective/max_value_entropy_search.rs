//! Lower-bound max-value entropy search for multi-objective Bayesian
//! optimization (`botorch.acquisition.multi_objective.max_value_entropy_search`,
//! MESMO; see Tu et al., NeurIPS 2022, for the estimators).

use std::sync::Arc;

use crate::acquisition::multi_objective::joint_entropy_search::{
    entropy_constant, initial_entropy, LowerBoundMultiObjectiveEntropySearch,
    LowerBoundMultiObjectiveEntropySearchOptions, RawPointPosterior,
};
use crate::acquisition::AcquisitionFunction;
use crate::error::Result;
use crate::models::Model;
use crate::utils::multi_objective::box_decompositions::CellBounds;
use crate::{Matrix, Vector};

/// Options of [`QLowerBoundMultiObjectiveMaxValueEntropySearch`].
pub type QLowerBoundMultiObjectiveMaxValueEntropySearchOptions =
    LowerBoundMultiObjectiveEntropySearchOptions;

/// The lower-bound multi-objective max-value entropy search acquisition
/// function (BoTorch's `qLowerBoundMultiObjectiveMaxValueEntropySearch`):
/// the mutual information between the observations at the candidates and
/// the Pareto optimal outputs,
///
/// ```text
/// α(X) = H[p(y_X | D)] - Σ_{x ∈ X} (1/P) Σ_p H[p(y_x | D, Y*_p)],
/// ```
///
/// where the conditional entropies are those of the model's posterior
/// truncated to the dominated region of each sampled Pareto front (the
/// hypercells `P × J × M`), estimated per point with the chosen
/// [`EstimationType`](super::joint_entropy_search::EstimationType). The value
/// can be negative.
#[derive(Debug)]
pub struct QLowerBoundMultiObjectiveMaxValueEntropySearch {
    base: LowerBoundMultiObjectiveEntropySearch,
}

impl QLowerBoundMultiObjectiveMaxValueEntropySearch {
    /// MES with the hypercell bounds of the sampled Pareto fronts (batch
    /// shape `[num_pareto_samples]`, e.g. from
    /// [`compute_sample_box_decomposition`](super::pareto_samples::compute_sample_box_decomposition)).
    ///
    /// # Errors
    /// See [`LowerBoundMultiObjectiveEntropySearch`].
    pub fn new(
        model: Arc<dyn Model>,
        hypercell_bounds: CellBounds,
        options: QLowerBoundMultiObjectiveMaxValueEntropySearchOptions,
    ) -> Result<Self> {
        Ok(Self {
            base: LowerBoundMultiObjectiveEntropySearch::new(model, hypercell_bounds, options)?,
        })
    }

    /// The shared state (cells, estimate, sampler).
    pub fn base(&self) -> &LowerBoundMultiObjectiveEntropySearch {
        &self.base
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let base = &self.base;
        let full = base.full_candidates(x)?;
        let (n, q, d) = (full.nrows(), x.nrows(), x.ncols());
        let gps = base.model().output_gps().expect("checked at construction");
        let m = gps.len();
        let p = base.num_pareto_samples();
        let mut grad = with_grad.then(|| Matrix::zeros(q, d));
        let h0 = entropy_constant(m) + initial_entropy(&gps, &full, q, grad.as_mut())?;
        // The marginal posteriors of every point (q = 1 each), shared by the
        // Pareto samples.
        let marginals = gps
            .iter()
            .map(|gp| gp.posterior_marginals_unclamped(&full))
            .collect::<Result<Vec<_>>>()?;
        let noise: Vec<f64> = gps.iter().map(|gp| gp.observation_noise()).collect();
        let z = base.mc_base_samples()?;
        let ce = base.conditional_entropy(false, z.as_deref());
        let mut cond_total = 0.0;
        for i in 0..n {
            let mean: Vec<f64> = marginals.iter().map(|(mu, _)| mu[i]).collect();
            let var: Vec<f64> = marginals.iter().map(|(_, v)| v[i]).collect();
            let post = RawPointPosterior {
                mean: &mean,
                variance: &var,
                noise: &noise,
            };
            let point_grad = with_grad && i < q;
            let mut hi = 0.0;
            let mut mean_bar = vec![0.0; m];
            let mut var_bar = vec![0.0; m];
            for k in 0..p {
                let (h, adj) = ce.sample_entropy(k, &post, point_grad);
                hi += h;
                if let Some(a) = adj {
                    for t in 0..m {
                        mean_bar[t] += a.mean[t];
                        var_bar[t] += a.variance[t];
                    }
                }
            }
            cond_total += hi / p as f64;
            if let (Some(g), true) = (grad.as_mut(), point_grad) {
                let xi: Vec<f64> = x.row(i).iter().copied().collect();
                let mut gi = Vector::zeros(d);
                for (t, gp) in gps.iter().enumerate() {
                    let (_, _, dmean, dvar) = gp.posterior_q1_with_grad_unclamped(&xi)?;
                    gi += dmean * mean_bar[t] + dvar * var_bar[t];
                }
                for k in 0..d {
                    g[(i, k)] -= gi[k] / p as f64;
                }
            }
        }
        Ok((h0 - cond_total, grad))
    }
}

impl AcquisitionFunction for QLowerBoundMultiObjectiveMaxValueEntropySearch {
    fn model(&self) -> &Arc<dyn Model> {
        self.base.model()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.base.pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.base.set_pending(x_pending)
    }
}
