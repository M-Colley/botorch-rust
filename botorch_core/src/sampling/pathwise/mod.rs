//! Pathwise posterior sampling (`botorch.sampling.pathwise`): functions drawn
//! from (an approximation of) a GP posterior that can be evaluated anywhere,
//! cheaply and with derivatives.
//!
//! A *sample path* is a deterministic function `x ↦ f_s(x)` for every sample
//! `s` of a batch of draws. Following Wilson et al. (2020, 2021), a posterior
//! path is built with *Matheron's rule*
//!
//! ```text
//!               "prior path"                 "update path"
//! (f | y)(·) =    f(·)     +  k(·, X) (K + Σ)⁻¹ (y - f(X) - ε),
//! ```
//!
//! where the prior path `f ~ GP(c, k)` is a Bayesian linear model on random
//! Fourier features of the kernel (Rahimi & Recht, 2007; Sutherland &
//! Schneider, 2015), `ε ~ N(0, Σ)` is a draw of the observation noise and
//! `K + Σ` is the training covariance of the model.
//!
//! | BoTorch | botorch_core |
//! |---|---|
//! | `gen_kernel_features(kernel, num_inputs, num_outputs)` | [`features::gen_kernel_features`]` -> `[`features::KernelFeatureMap`] (RBF, Matérn, `active_dims`, `ScaleKernel`) |
//! | `KernelEvaluationMap(kernel, points)` | [`features::KernelEvaluationMap`] |
//! | `GeneralizedLinearPath`, `PathList`, `PathDict`, `SamplePath` | [`paths::GeneralizedLinearPath`], [`paths::PathList`], [`paths::PathDict`], the [`paths::SamplePath`] trait |
//! | `draw_kernel_feature_paths(model, sample_shape)` | [`prior_samplers::draw_kernel_feature_paths`] |
//! | `gaussian_update(model, sample_values)` | [`update_strategies::gaussian_update`] |
//! | `draw_matheron_paths(model, sample_shape)`, `MatheronPath` | [`posterior_samplers::draw_matheron_paths`]` -> `[`paths::PathList`] of [`posterior_samplers::MatheronPath`]s |
//! | `get_matheron_path_model(model)`, `MatheronPathModel` | [`posterior_samplers::get_matheron_path_model`], [`posterior_samplers::MatheronPathModel`] |
//! | the transforms of `pathwise.utils` (input transforms, `OutcomeUntransformer`, `InverseLengthscaleTransform`, `SineCosineTransform`, `OutputscaleTransform`) | [`utils::OutcomeUntransform`]; the model's [`crate::Normalize`]; built into the feature maps |
//!
//! ## Conventions
//!
//! * Paths act on raw (original-space) inputs and return values in the
//!   original outcome space, like BoTorch's paths of a model with input and
//!   outcome transforms. Input transforms are applied with eval-mode
//!   semantics (meta-pytorch/botorch#3385; BoTorch main re-fits a learnable
//!   `Normalize` when a path of a model in train mode is evaluated).
//! * Evaluating a path at the `n` rows of `x` returns one `num_samples × n`
//!   matrix per output ([`paths::PathValues`]): BoTorch's `sample_shape x n`
//!   for a single-output model, `sample_shape x m x n` for a multi-output
//!   model, and the list of its sub-models' values for a `ModelListGP`.
//! * A path value at a point depends only on that point, so the derivatives
//!   of [`paths::SamplePath::forward_with_grad`] are one `n × d` matrix per
//!   output and sample ([`paths::PathGradients`]).
//!
//! ## Random numbers
//!
//! A Matheron path of one output GP needs three sets of random numbers,
//! collected in [`utils::MatheronPathDraws`]:
//!
//! * the Fourier frequencies (`num_features / 2 × d`): quasi-random standard
//!   normal vectors (BoTorch's `draw_sobol_normal_samples`), divided by the
//!   square root of a `Gamma(ν, ν)` variable for the Matérn kernel;
//! * the prior weights (`num_samples × num_features`), quasi-random standard
//!   normal vectors;
//! * the noise (`num_samples × n`), i.i.d. standard normal (BoTorch's
//!   `torch.randn_like`), scaled by the square root of the noise variance.
//!
//! BoTorch draws them from torch's global generator. botorch_core draws them
//! from its own generators ([`crate::utils::sampling::SobolEngine`] and a
//! seeded Xoshiro256++ generator), with the same distributions; for the same
//! seed the numbers differ from BoTorch's. Given BoTorch's draws (e.g. read
//! from its path objects), [`utils::MatheronPathDraws`] reproduces BoTorch's
//! paths to rounding.
//!
//! References:
//!
//! * J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky and M. Deisenroth.
//!   *Efficiently sampling functions from Gaussian process posteriors.* ICML
//!   2020.
//! * J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky and M. Deisenroth.
//!   *Pathwise Conditioning of Gaussian Processes.* JMLR 22, 2021.
//! * A. Rahimi and B. Recht. *Random features for large-scale kernel
//!   machines.* NeurIPS 2007.
//! * D. J. Sutherland and J. Schneider. *On the error of random Fourier
//!   features.* UAI 2015.

pub mod features;
pub mod paths;
pub mod posterior_samplers;
pub mod prior_samplers;
pub mod update_strategies;
pub mod utils;

pub use features::{
    gen_kernel_features, FeatureMap, KernelEvaluationMap, KernelFeatureMap, DEFAULT_NUM_FEATURES,
};
pub use paths::{
    GeneralizedLinearPath, PathDict, PathGradients, PathJoin, PathList, PathValues, SamplePath,
};
pub use posterior_samplers::{
    draw_matheron_paths, get_matheron_path_model, matheron_path_draws, MatheronPath,
    MatheronPathModel, MatheronPathOptions,
};
pub use prior_samplers::{draw_kernel_feature_paths, kernel_feature_path};
pub use update_strategies::{draw_update_noise, gaussian_update};
pub use utils::{MatheronPathDraws, OutcomeUntransform};
