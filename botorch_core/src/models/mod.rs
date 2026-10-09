//! Gaussian process models and the [`Model`] trait used by acquisition
//! functions.

pub mod gp;
pub mod model;

pub use gp::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    GaussianLikelihoodSpec, Hyperparameters, Likelihood, ParameterLayout, SingleTaskGP,
    SingleTaskGPOptions, MIN_FIXED_NOISE, MIN_INFERRED_NOISE_LEVEL,
};
pub use model::Model;
