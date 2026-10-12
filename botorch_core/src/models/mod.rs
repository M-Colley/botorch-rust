//! Gaussian process models and the [`Model`] trait used by acquisition
//! functions.
//!
//! * [`SingleTaskGP`]: a single-output exact GP (BoTorch's `SingleTaskGP`
//!   with one output);
//! * [`BatchedMultiOutputGP`]: BoTorch's `SingleTaskGP` with `m ≥ 1` outputs
//!   and an optional batch shape, i.e. independent `SingleTaskGP`s per batch
//!   element and output;
//! * [`ModelListGP`]: independent sub-models with possibly different training
//!   data.
//!
//! All three can be conditioned on new observations and fantasized
//! (`condition_on_observations`, `fantasize`; see [`fantasize`]).

pub mod batched;
pub mod cost;
pub mod fantasize;
pub mod gp;
pub mod model;
pub mod model_list;

pub use batched::{BatchedInputTransform, BatchedMultiOutputGP, BatchedMultiOutputGPOptions};
pub use fantasize::{BatchedMatrix, BoolMatrix, FantasySampler};
pub use gp::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    GaussianLikelihoodSpec, Hyperparameters, Likelihood, ParameterLayout, SingleTaskGP,
    SingleTaskGPOptions, MIN_FIXED_NOISE, MIN_INFERRED_NOISE_LEVEL,
};
pub use model::Model;
pub use model_list::ModelListGP;
