//! A list of box decompositions (`box_decomposition_list.py`):
//! [`BoxDecompositionList`].

use rayon::prelude::*;

use super::box_decomposition::{BoxDecomposition, CellBounds};
use crate::error::{BotorchError, Result};
use crate::utils::multi_objective::PointBatch;

/// A list of unbatched box decompositions (BoTorch's `BoxDecompositionList`),
/// e.g. one per Monte-Carlo sample when a batched decomposition is not
/// available (more than two objectives). Construction with
/// [`BoxDecompositionList::par_build`] and [`BoxDecompositionList::update`]
/// process the decompositions in parallel.
#[derive(Debug, Default)]
pub struct BoxDecompositionList {
    decompositions: Vec<Box<dyn BoxDecomposition>>,
}

impl BoxDecompositionList {
    /// `BoxDecompositionList(*box_decompositions)`.
    pub fn new(decompositions: Vec<Box<dyn BoxDecomposition>>) -> Self {
        Self { decompositions }
    }

    /// One decomposition per item, built in parallel with `build`, e.g.
    /// `BoxDecompositionList::par_build(&samples, |y| FastNondominatedPartitioning::new(&ref_point, Some(y)))`.
    ///
    /// # Errors
    /// The first error returned by `build`.
    pub fn par_build<T, D, F>(items: &[T], build: F) -> Result<Self>
    where
        T: Sync,
        D: BoxDecomposition + 'static,
        F: Fn(&T) -> Result<D> + Sync,
    {
        let decompositions = items
            .par_iter()
            .map(|item| build(item).map(|d| Box::new(d) as Box<dyn BoxDecomposition>))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { decompositions })
    }

    /// The number of decompositions.
    pub fn len(&self) -> usize {
        self.decompositions.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.decompositions.is_empty()
    }

    /// The decompositions.
    pub fn decompositions(&self) -> &[Box<dyn BoxDecomposition>] {
        &self.decompositions
    }

    /// The decompositions (mutable).
    pub fn decompositions_mut(&mut self) -> &mut [Box<dyn BoxDecomposition>] {
        &mut self.decompositions
    }

    /// `pareto_Y`: the frontier of each decomposition.
    pub fn pareto_y(&self) -> Vec<Option<PointBatch>> {
        self.decompositions.iter().map(|d| d.pareto_y()).collect()
    }

    /// `ref_point`: the reference points of all decompositions, stacked
    /// (`n_box_decompositions x m`, row-major).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the reference points have
    /// different shapes or the list is empty.
    pub fn ref_point(&self) -> Result<Vec<f64>> {
        let shape = self.common_ref_shape()?;
        let mut out = Vec::with_capacity(self.len() * shape.iter().product::<usize>());
        for d in &self.decompositions {
            out.extend_from_slice(d.ref_point());
        }
        Ok(out)
    }

    fn common_ref_shape(&self) -> Result<Vec<usize>> {
        let first = self.decompositions.first().ok_or_else(|| {
            BotorchError::DimensionMismatch("stack expects a non-empty list".into())
        })?;
        let shape = first.ref_point_shape().to_vec();
        if self
            .decompositions
            .iter()
            .any(|d| d.ref_point_shape() != shape)
        {
            return Err(BotorchError::DimensionMismatch(
                "stack expects each reference point to have the same shape".into(),
            ));
        }
        Ok(shape)
    }

    /// `get_hypercell_bounds()`: the cell bounds of all decompositions,
    /// padded with empty cells (zeros) to the largest number of cells, as a
    /// batch of shape `n_box_decompositions x num_cells x m`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if a decomposition has no
    /// outcomes, [`BotorchError::Unsupported`] for batched decompositions,
    /// [`BotorchError::DimensionMismatch`] for different numbers of outcomes
    /// or an empty list.
    pub fn get_hypercell_bounds(&self) -> Result<CellBounds> {
        if self.decompositions.is_empty() {
            return Err(BotorchError::DimensionMismatch(
                "stack expects a non-empty list".into(),
            ));
        }
        let mut all = Vec::with_capacity(self.len());
        for d in &self.decompositions {
            let bounds = d.get_hypercell_bounds().ok_or_else(|| {
                BotorchError::InvalidArgument(format!(
                    "{} has no hypercell bounds (no outcomes were given).",
                    d.class_name()
                ))
            })?;
            if !bounds.batch_shape().is_empty() {
                return Err(BotorchError::Unsupported(
                    "BoxDecompositionList.get_hypercell_bounds supports unbatched box \
                     decompositions only."
                        .into(),
                ));
            }
            all.push(bounds);
        }
        let m = all[0].num_outcomes();
        if all.iter().any(|b| b.num_outcomes() != m) {
            return Err(BotorchError::DimensionMismatch(
                "the box decompositions have different numbers of outcomes".into(),
            ));
        }
        let max_cells = all.iter().map(|b| b.num_cells()).max().unwrap_or(0);
        let mut lower = Vec::with_capacity(all.len() * max_cells * m);
        let mut upper = Vec::with_capacity(all.len() * max_cells * m);
        for b in &all {
            let pad = (max_cells - b.num_cells()) * m;
            lower.extend_from_slice(b.lower());
            lower.extend(std::iter::repeat_n(0.0, pad));
            upper.extend_from_slice(b.upper());
            upper.extend(std::iter::repeat_n(0.0, pad));
        }
        CellBounds::new(vec![all.len()], max_cells, m, lower, upper)
    }

    /// `update(Y)`: updates decomposition `i` with `ys[i]`, in parallel.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if there is not one batch of new
    /// outcomes per decomposition (BoTorch's message); otherwise the first
    /// error of an update (all updates are attempted).
    pub fn update(&mut self, ys: &[PointBatch]) -> Result<()> {
        if ys.len() != self.decompositions.len() {
            return Err(BotorchError::DimensionMismatch(
                "BoxDecompositionList.update requires either a batched tensor Y, with one batch \
                 per box decomposition or a list of tensors with one element per box \
                 decomposition."
                    .into(),
            ));
        }
        self.decompositions
            .par_iter_mut()
            .zip(ys.par_iter())
            .map(|(d, y)| d.update(y))
            .collect::<Vec<Result<()>>>()
            .into_iter()
            .collect()
    }

    /// `compute_hypervolume()`: the hypervolume of each decomposition (each
    /// entry holds one value per batch element of that decomposition).
    pub fn compute_hypervolume(&self) -> Vec<Vec<f64>> {
        self.decompositions
            .par_iter()
            .map(|d| d.compute_hypervolume())
            .collect()
    }
}
