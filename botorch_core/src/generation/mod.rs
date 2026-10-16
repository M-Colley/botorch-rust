//! Candidate generation (`botorch.generation`): the torch-style optimizer of
//! acquisition functions ([`gen::gen_candidates_torch`], Adam with BoTorch's
//! exponential-moving-average stopping criterion), [`gen::get_best_candidates`]
//! and the sampling strategies of `botorch.generation.sampling`
//! ([`sampling::maximize_samples`] for `MaxPosteriorSampling` and its
//! constrained variant, [`sampling::boltzmann_weights`] for
//! `BoltzmannSampling`).
//!
//! `gen_candidates_scipy` is [`crate::optim::gen_candidates_lbfgsb`] (independent
//! L-BFGS-B runs, BoTorch's default parallel mode),
//! [`crate::optim::gen_candidates_lbfgsb_joint`] (BoTorch's serial mode, one
//! problem over several restarts) and [`crate::optim::gen_candidates_slsqp`]
//! (constraints).

pub mod gen;
pub mod sampling;

pub use gen::{gen_candidates_torch, get_best_candidates, TorchGenOptions, TorchGenResult};
pub use sampling::{boltzmann_weights, constrained_scores, maximize_samples};
