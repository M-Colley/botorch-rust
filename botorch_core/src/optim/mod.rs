//! Optimization of acquisition functions (`botorch.optim`):
//! [`optimize_acqf`] with multi-start L-BFGS-B (box constraints) or SLSQP
//! (linear and nonlinear parameter constraints), the initialization
//! heuristics of `botorch.optim.initializers` and the constraint handling of
//! `botorch.optim.parameter_constraints`; [`optimize_acqf_list`] for lists of
//! acquisition functions; the initial conditions of one-shot acquisition
//! functions ([`one_shot`]); the mixed, discrete, cyclic and homotopy
//! optimizers ([`mixed`], [`discrete`], [`cyclic`], [`homotopy`]); torch-style
//! optimization ([`adam`], [`stopping`], [`minimize`]) and model fitting with
//! torch optimizers ([`fit`]).

pub mod adam;
pub mod cyclic;
pub mod discrete;
pub mod fit;
pub mod homotopy;
pub mod initializers;
pub mod minimize;
pub mod mixed;
pub mod one_shot;
pub mod optimize;
pub mod optimize_list;
pub mod parameter_constraints;
pub mod slsqp;
pub mod stopping;

pub use adam::{Adam, AdamOptions};
pub use cyclic::{optimize_acqf_cyclic, CyclicResult};
pub use discrete::{
    optimize_acqf_discrete, optimize_acqf_discrete_local_search, DiscreteResult, LocalSearchOptions,
};
pub use homotopy::prune_candidates;
pub use initializers::{
    gen_batch_initial_conditions, gen_batch_initial_conditions_with_fantasies, initialize_q_batch,
    initialize_q_batch_nonneg, initialize_q_batch_topn, sample_perturbed_subset_dims,
    sample_points_around, sample_q_batches_from_polytope, sample_truncated_normal_perturbations,
    transform_constraints, truncated_normal_perturbations_from_uniform, InitialConditionOptions,
    SampleAroundBest, TopnOptions,
};
pub use minimize::{torch_minimize, OptimizationResult, OptimizationStatus, TorchMinimizeOptions};
pub use mixed::{
    optimize_acqf_mixed, optimize_acqf_mixed_alternating, should_use_mixed_alternating_optimizer,
    MixedAlternatingOptions, MixedResult,
};
pub use one_shot::{
    boltzmann_sample, gen_one_shot_hvkg_initial_conditions, optimize_hvkg_value_function,
    OneShotInitialConditionOptions, ValueFunctionSolutions,
};
pub use optimize::{
    default_lbfgsb_options, gen_candidates_lbfgsb, gen_candidates_lbfgsb_joint, optimize_acqf,
    optimize_acqf_detailed, CandidateGenerator, CustomGenerator, GenRequest, IcGenerator,
    IcRequest, OptimizationMethod, OptimizeAcqfOptions, OptimizeAcqfResult, OptimizerStatus,
    PostProcessing, RestartResult,
};
pub use optimize_list::{optimize_acqf_list, OptimizeAcqfListResult};
pub use parameter_constraints::{
    evaluate_feasibility, ConstraintIndices, FnNonlinearConstraint, LinearConstraint,
    NonlinearInequalityConstraint, ParameterConstraints, CONSTRAINT_TOLERANCE,
};
pub use slsqp::{gen_candidates_slsqp, project_to_feasible_space, SlsqpChunkResult, SlsqpOptions};
pub use stopping::{ExpMaStoppingCriterion, StoppingCriterion};
