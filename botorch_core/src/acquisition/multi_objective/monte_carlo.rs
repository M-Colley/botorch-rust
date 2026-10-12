//! Monte-Carlo hypervolume improvement (`botorch.acquisition.multi_objective.monte_carlo`):
//! [`QExpectedHypervolumeImprovement`] (qEHVI, Daulton et al., NeurIPS 2020)
//! (BoTorch recommends its log version in [`super::logei`]).

use std::sync::{Arc, RwLock};

use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock, JointSamples};
use crate::acquisition::multi_objective::hypervolume::HypercellBounds;
use crate::acquisition::multi_objective::objective::McMultiOutputObjective;
use crate::acquisition::multi_objective::{check_ref_point, MoMcCore, Reduction};
use crate::acquisition::{check_candidates, check_pending, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::multi_objective::box_decompositions::BoxDecomposition;
use crate::utils::objective::OutcomeConstraints;
use crate::Matrix;

/// Options of [`QExpectedHypervolumeImprovement`] (BoTorch's keyword
/// arguments).
#[derive(Clone, Debug, Default)]
pub struct QExpectedHypervolumeImprovementOptions {
    /// The sampler (default: 128 Sobol samples with a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The multi-output objective (default: the identity on all outputs).
    pub objective: Option<Arc<dyn McMultiOutputObjective>>,
    /// Outcome constraints (feasible iff `<= 0`) with their temperatures
    /// (BoTorch's `constraints` and `eta`, default `eta = 1e-3`).
    pub constraints: Option<OutcomeConstraints>,
    /// Pending points concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
    /// Smooth the feasibility indicator with the fat-tailed `fatmoid`
    /// instead of the logistic sigmoid (BoTorch's `fat`, default `false`).
    pub fat: bool,
}

/// The cell bounds of an unbatched box decomposition.
pub(crate) fn cells_of(partitioning: &dyn BoxDecomposition) -> Result<HypercellBounds> {
    let bounds = partitioning.get_hypercell_bounds().ok_or_else(|| {
        BotorchError::InvalidArgument(
            "The partitioning has no hypercell bounds (no outcomes were given).".into(),
        )
    })?;
    if !bounds.batch_shape().is_empty() {
        return Err(BotorchError::Unsupported(
            "batched box decompositions are not supported; use one decomposition of the \
             current Pareto front"
                .into(),
        ));
    }
    HypercellBounds::from_cell_bounds(bounds)
}

/// The shared implementation of qEHVI and qLogEHVI.
#[derive(Debug)]
pub(crate) struct EhviCore {
    pub mc: MoMcCore,
    pub ref_point: Vec<f64>,
    pub cells: Arc<HypercellBounds>,
    pub x_pending: RwLock<Option<Matrix>>,
}

impl EhviCore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        cells: HypercellBounds,
        sampler: Option<Arc<SobolQMCNormalSampler>>,
        objective: Option<Arc<dyn McMultiOutputObjective>>,
        constraints: Option<OutcomeConstraints>,
        x_pending: Option<Matrix>,
        fat: bool,
        log: bool,
    ) -> Result<Self> {
        check_ref_point(ref_point, cells.num_outcomes(), log)?;
        if cells.batches() != 1 {
            return Err(BotorchError::Unsupported(
                "batched box decompositions are not supported; use one decomposition of the \
                 current Pareto front"
                    .into(),
            ));
        }
        let mc = MoMcCore::new(model, sampler, objective, constraints, fat)?;
        if let Some(p) = &x_pending {
            check_pending(p, mc.model.dim())?;
        }
        Ok(Self {
            mc,
            ref_point: ref_point.to_vec(),
            cells: Arc::new(cells),
            x_pending: RwLock::new(x_pending),
        })
    }

    /// BoTorch's `forward` with `concatenate_pending_points`.
    pub fn evaluate(
        &self,
        x: &Matrix,
        reduction: Reduction,
        with_grad: bool,
    ) -> Result<(f64, Option<Matrix>)> {
        let d = self.mc.model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let x_full = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let n = x_full.nrows();
        let js = JointSamples::new(self.mc.model.as_ref(), x_full, &self.mc.sampler)?;
        let samples = js.samples(0..n);
        let r = self
            .mc
            .reduce_samples(&samples, &js.x, &self.cells, reduction, with_grad)?;
        if !with_grad {
            return Ok((r.value, None));
        }
        let sbar = r.samples_bar.expect("gradient requested");
        let mut grad = js.backward(self.mc.model.as_ref(), 0..n, &sbar, 0..q)?;
        if let Some(xb) = r.x_bar {
            grad += xb.rows(0, q);
        }
        Ok((r.value, Some(grad)))
    }

    pub fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.x_pending).clone()
    }

    pub fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.mc.model.dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }
}

/// The acquisition-function trait methods shared by qEHVI and qLogEHVI.
macro_rules! ehvi_common {
    ($reduction:expr) => {
        fn model(&self) -> &std::sync::Arc<dyn crate::models::Model> {
            &self.core.mc.model
        }

        fn forward(&self, x: &crate::Matrix) -> crate::Result<f64> {
            Ok(self.core.evaluate(x, $reduction(self), false)?.0)
        }

        fn forward_with_grad(&self, x: &crate::Matrix) -> crate::Result<(f64, crate::Matrix)> {
            let (v, g) = self.core.evaluate(x, $reduction(self), true)?;
            Ok((v, g.expect("gradient requested")))
        }

        fn x_pending(&self) -> Option<crate::Matrix> {
            self.core.x_pending()
        }

        fn set_x_pending(&self, x_pending: Option<&crate::Matrix>) -> crate::Result<()> {
            self.core.set_x_pending(x_pending)
        }
    };
}

pub(crate) use ehvi_common;

/// MC-based batch Expected Hypervolume Improvement (BoTorch's
/// `qExpectedHypervolumeImprovement`): the expected (feasibility-weighted)
/// hypervolume improvement of the candidates over a box decomposition of the
/// space not dominated by the current Pareto front, computed with the
/// inclusion–exclusion principle over the subsets of the candidates.
#[derive(Debug)]
pub struct QExpectedHypervolumeImprovement {
    core: EhviCore,
}

impl QExpectedHypervolumeImprovement {
    /// `qExpectedHypervolumeImprovement(model, ref_point, partitioning, ...)`
    /// with an unbatched box decomposition of the current Pareto front (of
    /// the objective values; feasible points only with constraints).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if the reference point does not
    /// have one value per outcome of the decomposition (BoTorch's message),
    /// [`BotorchError::Unsupported`] for batched models or decompositions,
    /// and [`BotorchError::DimensionMismatch`] for pending points of the
    /// wrong dimension.
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        partitioning: &dyn BoxDecomposition,
        options: QExpectedHypervolumeImprovementOptions,
    ) -> Result<Self> {
        Self::from_cell_bounds(model, ref_point, cells_of(partitioning)?, options)
    }

    /// Like [`QExpectedHypervolumeImprovement::new`] with the hypercell
    /// bounds of the decomposition (e.g. of a decomposition computed
    /// elsewhere).
    ///
    /// # Errors
    /// See [`QExpectedHypervolumeImprovement::new`].
    pub fn from_cell_bounds(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        cells: HypercellBounds,
        options: QExpectedHypervolumeImprovementOptions,
    ) -> Result<Self> {
        Ok(Self {
            core: EhviCore::new(
                model,
                ref_point,
                cells,
                options.sampler,
                options.objective,
                options.constraints,
                options.x_pending,
                options.fat,
                false,
            )?,
        })
    }

    /// The reference point.
    pub fn ref_point(&self) -> &[f64] {
        &self.core.ref_point
    }

    /// The hypercell bounds (BoTorch's `cell_lower_bounds` /
    /// `cell_upper_bounds`).
    pub fn cell_bounds(&self) -> &HypercellBounds {
        &self.core.cells
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.core.mc.sampler
    }
}

impl AcquisitionFunction for QExpectedHypervolumeImprovement {
    ehvi_common!(|_: &Self| Reduction::Linear);

    fn is_nonnegative(&self) -> bool {
        true
    }
}
