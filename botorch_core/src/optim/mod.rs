//! Optimization of acquisition functions (`botorch.optim`):
//! [`optimize_acqf`] with multi-start L-BFGS-B (box constraints) or SLSQP
//! (linear and nonlinear parameter constraints), the initialization
//! heuristics of `botorch.optim.initializers` and the constraint handling of
//! `botorch.optim.parameter_constraints`.

pub mod initializers;
pub mod optimize;
pub mod parameter_constraints;
pub mod slsqp;

pub use initializers::{
    gen_batch_initial_conditions, initialize_q_batch, initialize_q_batch_nonneg,
    sample_q_batches_from_polytope, transform_constraints, InitialConditionOptions,
};
pub use optimize::{
    default_lbfgsb_options, gen_candidates_lbfgsb, optimize_acqf, optimize_acqf_detailed,
    IcGenerator, IcRequest, OptimizeAcqfOptions, OptimizeAcqfResult, OptimizerStatus,
    RestartResult,
};
pub use parameter_constraints::{
    evaluate_feasibility, ConstraintIndices, FnNonlinearConstraint, LinearConstraint,
    NonlinearInequalityConstraint, ParameterConstraints, CONSTRAINT_TOLERANCE,
};
pub use slsqp::{gen_candidates_slsqp, project_to_feasible_space, SlsqpChunkResult, SlsqpOptions};
