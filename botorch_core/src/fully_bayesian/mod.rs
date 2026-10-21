//! Fully Bayesian GPs (`botorch.models.fully_bayesian`,
//! `botorch.models.fully_bayesian_multitask`, `botorch.fit.
//! fit_fully_bayesian_model_nuts`): the hyperparameters are sampled from
//! their posterior with the No-U-Turn sampler, and the model is the batch of
//! GPs of the samples (an ensemble whose posterior is a Gaussian mixture).
//!
//! * [`nuts`]: NumPyro's NUTS (step-size and mass-matrix adaptation,
//!   iterative tree building), with seeded or recorded random draws;
//! * [`pyro_models`]: the potential energies of BoTorch's NumPyro models
//!   (`SaasPyroModel`, `MaternPyroModel`, `LinearPyroModel` and the
//!   multi-task SAAS model) with analytic gradients, and the
//!   post-processing of their samples.
//!
//! The batched GP built from the samples is a [`crate::BatchedMultiOutputGP`]
//! (the bindings assemble it from the samples as BoTorch's
//! `load_mcmc_samples` does).

pub mod nuts;
pub mod pyro_models;

pub use nuts::{
    find_valid_initial_params, sample_nuts, sample_nuts_with_callback, MassMatrix, NutsOptions,
    NutsRandom, NutsRun, Potential, RecordedDraw, RecordedDraws, SeededDraws, UniformKind,
};
pub use pyro_models::{
    PyroKernel, PyroModel, PyroModelSpec, Site, Support, TaskStructure, MIN_INFERRED_NOISE_LEVEL,
};
