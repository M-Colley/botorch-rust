//! Multi-objective acquisition functions (`botorch.acquisition.multi_objective`).
//!
//! * [`objective`] — multi-output MC objectives (`IdentityMCMultiOutputObjective`,
//!   `WeightedMCMultiOutputObjective`, `GenericMCMultiOutputObjective`,
//!   `FeasibilityWeightedMCMultiOutputObjective`).
//! * [`utils`] — `get_default_partitioning_alpha`,
//!   `prune_inferior_points_multi_objective`.

pub mod objective;
pub mod utils;

pub use objective::{
    get_infeasible_cost_per_output, FeasibilityWeightedMcMultiOutputObjective,
    GenericMcMultiOutputObjective, IdentityMcMultiOutputObjective, McMultiOutputObjective,
    MultiOutputValues, WeightedMcMultiOutputObjective,
};
pub use utils::{
    get_default_partitioning_alpha, prune_inferior_points_multi_objective,
    prune_inferior_points_multi_objective_indices, PruneMultiObjectiveOptions,
};
