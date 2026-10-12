//! Optimization of acquisition functions (`botorch.optim`):
//! [`optimize_acqf`] with multi-start L-BFGS-B (box constraints) or SLSQP
//! (linear and nonlinear parameter constraints), the initialization
//! heuristics of `botorch.optim.initializers` and the constraint handling of
//! `botorch.optim.parameter_constraints`; [`optimize_acqf_list`] for lists of
//! acquisition functions; the initial conditions of one-shot acquisition
//! functions ([`one_shot`]).

pub mod initializers;
pub mod one_shot;
pub mod optimize;
pub mod optimize_list;
pub mod parameter_constraints;
pub mod slsqp;

pub use initializers::{
    gen_batch_initial_conditions, gen_batch_initial_conditions_with_fantasies, initialize_q_batch,
    initialize_q_batch_nonneg, sample_q_batches_from_polytope, transform_constraints,
    InitialConditionOptions,
};
pub use one_shot::{
    boltzmann_sample, gen_one_shot_hvkg_initial_conditions, OneShotInitialConditionOptions,
};
pub use optimize::{
    default_lbfgsb_options, gen_candidates_lbfgsb, optimize_acqf, optimize_acqf_detailed,
    IcGenerator, IcRequest, OptimizationMethod, OptimizeAcqfOptions, OptimizeAcqfResult,
    OptimizerStatus, RestartResult,
};
pub use optimize_list::{optimize_acqf_list, OptimizeAcqfListResult};
pub use parameter_constraints::{
    evaluate_feasibility, ConstraintIndices, FnNonlinearConstraint, LinearConstraint,
    NonlinearInequalityConstraint, ParameterConstraints, CONSTRAINT_TOLERANCE,
};
pub use slsqp::{gen_candidates_slsqp, project_to_feasible_space, SlsqpChunkResult, SlsqpOptions};
