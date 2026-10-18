//! One-shot acquisition functions (BoTorch's `OneShotAcquisitionFunction`).
//!
//! A one-shot acquisition function is optimized over an *augmented*
//! candidate set: the `q` candidates followed by auxiliary points (e.g. the
//! solutions of the inner optimization problems of the knowledge gradient
//! for every fantasy), so that a nested optimization becomes a single one.
//! [`crate::optim::optimize_acqf`] optimizes `get_augmented_q_batch_size(q)`
//! points, starts from the function's own initial conditions (BoTorch's
//! `get_ic_generator`, e.g. `gen_one_shot_hvkg_initial_conditions`) unless an
//! `ic_generator` or `batch_initial_conditions` is given, and returns
//! `extract_candidates` of the best augmented set.

use rand::Rng;

use crate::acquisition::AcquisitionFunction;
use crate::error::Result;
use crate::optim::OptimizeAcqfOptions;
use crate::Matrix;

/// The one-shot interface (see the [module documentation](self)).
pub trait OneShotAcquisitionFunction: AcquisitionFunction {
    /// The number of points of the augmented candidate set for `q`
    /// candidates (BoTorch's `get_augmented_q_batch_size`).
    fn get_augmented_q_batch_size(&self, q: usize) -> usize;

    /// The number of points of the candidate sets that
    /// [`crate::optim::optimize_acqf`] optimizes when called with `q`: the
    /// augmented size (the default), or `q` itself for acquisition functions
    /// that BoTorch's `optimize_acqf` is called with the augmented size for
    /// (`qMultiStepLookahead`).
    fn optimized_q_batch_size(&self, q: usize) -> usize {
        self.get_augmented_q_batch_size(q)
    }

    /// The `q` candidates of an augmented candidate set (`q_aug × d`),
    /// BoTorch's `extract_candidates`.
    ///
    /// # Errors
    /// [`crate::BotorchError::DimensionMismatch`] if `x_full` has too few
    /// points.
    fn extract_candidates(&self, x_full: &Matrix) -> Result<Matrix>;

    /// The default initial conditions of `optimize_acqf`: `num_restarts`
    /// augmented candidate sets for `q` candidates (BoTorch's
    /// `get_ic_generator()` of the acquisition function), and the messages
    /// of BoTorch's warnings.
    ///
    /// # Errors
    /// Implementation-specific.
    fn gen_one_shot_initial_conditions(
        &self,
        bounds: &Matrix,
        q: usize,
        num_restarts: usize,
        raw_samples: usize,
        options: &OptimizeAcqfOptions,
        rng: &mut dyn Rng,
    ) -> Result<(Vec<Matrix>, Vec<String>)>;
}
