//! The partitioning of the dominated space (`dominated.py`):
//! [`DominatedPartitioning`].

use super::box_decomposition::{
    impl_fast_partitioning, BoxDecomposition, CellBounds, Core, FastCore, FastKind,
};

/// The dominated space as disjoint boxes via Algorithm 1 of Lacour et al.
/// (2017) (BoTorch's `DominatedPartitioning`; maximization): the local
/// upper bounds of the negated frontier are computed incrementally and every
/// local upper bound defines one cell (Equation 2). With two objectives the
/// `n` cells are built directly (also for a batch of frontiers). For more
/// than two objectives, [`BoxDecomposition::update`] inserts new Pareto
/// points incrementally. The hypervolume is the sum of the cell volumes;
/// without Pareto points the decomposition is a single empty cell at the
/// reference point.
#[derive(Clone, Debug)]
pub struct DominatedPartitioning(FastCore);

impl_fast_partitioning!(
    DominatedPartitioning,
    FastKind::Dominated,
    "DominatedPartitioning"
);
