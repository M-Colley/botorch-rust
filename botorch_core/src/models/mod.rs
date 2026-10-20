//! Gaussian process models and the [`Model`] trait used by acquisition
//! functions.
//!
//! * [`SingleTaskGP`]: a single-output exact GP (BoTorch's `SingleTaskGP`
//!   with one output);
//! * [`BatchedMultiOutputGP`]: BoTorch's `SingleTaskGP` with `m ≥ 1` outputs
//!   and an optional batch shape, i.e. independent `SingleTaskGP`s per batch
//!   element and output;
//! * [`ModelListGP`]: independent sub-models with possibly different training
//!   data;
//! * [`LCEMGP`]: BoTorch's latent context embedding multi-output GP, a
//!   multi-task GP with a kernel on context embeddings (with the task
//!   handling of `MultiTaskGP`, [`multitask`]);
//! * [`ModelList`]: a list of arbitrary models (BoTorch's `ModelList`), e.g.
//!   `ModelListGP(*lcemgps)`;
//! * [`PosteriorMeanModel`]: the posterior mean of a model as a deterministic
//!   model (see [`deterministic`]).
//!
//! The first three can be conditioned on new observations and fantasized
//! (`condition_on_observations`, `fantasize`; see [`fantasize`]).

pub mod approximate_gp;
pub mod batched;
pub mod contextual_multioutput;
pub mod cost;
pub mod deterministic;
pub mod em_gp;
pub mod em_kl;
pub mod em_model;
pub mod em_shared;
pub mod empirical;
pub mod ensemble;
pub mod fantasize;
pub mod gp;
pub mod heterogeneous;
pub mod higher_order_gp;
pub mod inducing_point_allocators;
pub mod latent_kronecker_gp;
pub mod likelihoods;
mod matheron;
pub mod model;
pub mod model_batch;
pub mod model_list;
pub mod model_list_dyn;
pub mod multitask;
pub mod pairwise_gp;
pub mod relevance_pursuit;

pub use approximate_gp::{
    SingleTaskVariationalGP, SingleTaskVariationalGPOptions, VariationalLikelihood, VariationalMean,
};
pub use batched::{BatchedInputTransform, BatchedMultiOutputGP, BatchedMultiOutputGPOptions};
pub use contextual_multioutput::{LCEMGPOptions, LcemParameters, LcemTaskInputs, LCEMGP};
pub use deterministic::{AffineDeterministicModel, FixedSingleSampleModel, PosteriorMeanModel};
pub use ensemble::EnsembleMomentsModel;
pub use fantasize::{BatchedMatrix, BoolMatrix, FantasySampler};
pub use gp::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    ConstantMeanSpec, GaussianLikelihoodSpec, Hyperparameters, Likelihood, ParameterLayout,
    SingleTaskGP, SingleTaskGPOptions, MIN_FIXED_NOISE, MIN_INFERRED_NOISE_LEVEL,
};
pub use higher_order_gp::{HigherOrderGP, HigherOrderGPOptions};
pub use model::Model;
pub use model_list::ModelListGP;
pub use model_list_dyn::ModelList;
pub use multitask::{MultiTaskGP, MultiTaskGPOptions, TaskLayout};
pub use pairwise_gp::{BatchedPairwiseGP, PairwiseGP, PairwiseGPOptions};
