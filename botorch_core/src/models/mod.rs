//! Gaussian process models.

pub mod gp;

pub use gp::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    GaussianLikelihoodSpec, Hyperparameters, Likelihood, ParameterLayout, SingleTaskGP,
    SingleTaskGPOptions, MIN_FIXED_NOISE, MIN_INFERRED_NOISE_LEVEL,
};
