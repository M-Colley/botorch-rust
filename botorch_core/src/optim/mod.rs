//! Optimization of acquisition functions (`botorch.optim`):
//! [`optimize_acqf`] with multi-start L-BFGS-B, and the initialization
//! heuristics of `botorch.optim.initializers`.

pub mod initializers;
pub mod optimize;

pub use initializers::{
    gen_batch_initial_conditions, initialize_q_batch, initialize_q_batch_nonneg,
    InitialConditionOptions,
};
pub use optimize::{
    default_lbfgsb_options, gen_candidates_lbfgsb, optimize_acqf, optimize_acqf_detailed,
    OptimizeAcqfOptions, OptimizeAcqfResult, RestartResult,
};
