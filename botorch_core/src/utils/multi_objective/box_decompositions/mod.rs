//! Box decompositions of the dominated and non-dominated space
//! (`botorch.utils.multi_objective.box_decompositions`).
//!
//! * [`NondominatedPartitioning`]: the binary partitioning of the
//!   non-dominated space of Couckuyt et al. (2012), with the approximation
//!   parameter `alpha` (Watanabe, 2025), and a direct construction for two
//!   objectives.
//! * [`FastNondominatedPartitioning`]: the non-dominated space via two
//!   applications of Algorithm 1 of Lacour et al. (2017) (local upper bounds
//!   of the Pareto front, then the space dominated by them; Yang et al.,
//!   2019), with incremental updates.
//! * [`DominatedPartitioning`]: the dominated space via Algorithm 1 of Lacour
//!   et al. (2017) (Klamroth et al., 2015), with incremental updates.
//! * [`BoxDecompositionList`]: one decomposition per element of a list (e.g.
//!   per Monte-Carlo sample for more than two objectives), built and updated
//!   in parallel.
//!
//! All classes assume maximization and work internally on the negated
//! outcomes, as BoTorch does. With two objectives they support a batch of
//! Pareto fronts (a [`PointBatch`](super::PointBatch) with one batch
//! dimension); fronts of different sizes are padded by repeating a Pareto
//! point (or with the reference point if a front is empty), as in BoTorch,
//! so that all batch elements have the same number of cells.
//!
//! References:
//!
//! * I. Couckuyt, D. Deschrijver and T. Dhaene. *Towards efficient
//!   multiobjective optimization: multiobjective statistical criterions.*
//!   IEEE CEC 2012.
//! * R. Lacour, K. Klamroth and C. M. Fonseca. *A box decomposition algorithm
//!   to compute the hypervolume indicator.* Computers & Operations Research
//!   79, 2017.
//! * K. Klamroth, R. Lacour and D. Vanderpooten. *On the representation of
//!   the search region in multi-objective optimization.* European Journal of
//!   Operational Research 245(3), 2015.
//! * K. Yang, M. Emmerich, A. Deutz and T. Bäck. *Efficient computation of
//!   expected hypervolume improvement using box decomposition algorithms.*
//!   Journal of Global Optimization 75, 2019.
//! * S. Watanabe. *Approximation of box decomposition algorithm for fast
//!   hypervolume-based multi-objective optimization.* arXiv:2512.05825, 2025.

mod box_decomposition;
mod box_decomposition_list;
mod dominated;
mod non_dominated;
pub mod utils;

pub use box_decomposition::{BoxDecomposition, CellBounds, FastPartitioning};
pub use box_decomposition_list::BoxDecompositionList;
pub use dominated::DominatedPartitioning;
pub use non_dominated::{FastNondominatedPartitioning, NondominatedPartitioning};
pub use utils::{
    compute_dominated_hypercell_bounds_2d, compute_non_dominated_hypercell_bounds_2d,
    LocalUpperBounds,
};
