//! Likelihoods of BoTorch's models (`botorch.models.likelihoods`): the
//! pairwise likelihoods of preference models.

pub mod pairwise;

pub use pairwise::{Comparisons, PairwiseLikelihood};
