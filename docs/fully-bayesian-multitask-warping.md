# Native fully Bayesian multitask input warping

Audited against official BoTorch revision `b0dd01be6`. The comparison tests
correct the reference's documented warp errors explicitly; they do not claim
that its unmodified loader has the behavior described here.

`SaasFullyBayesianMultiTaskGP` now supports
`MultitaskSaasPyroModel(use_input_warping=True)` through native NUTS fitting,
MCMC sample loading, GP prediction, conditioning, and query derivatives.
The optional NumPyro program and native Rust potential use the same warp.
Native fitting and inference do not call Torch, JAX, or BoTorch for their
numerical computations.

Each MCMC member applies its own fixed `c0` and `c1` concentrations after
the common fitted `Normalize` transform. The transformation is
`u = clamp(x * (1 - 2 eps) + eps, eps, 1 - eps)`, followed by
`1 - (1 - u**c1)**c0`. Custom `eps` is retained when samples are loaded.
`Standardize` retains the existing multitask outcome-transform behavior.
These fixed sampled concentrations are not additional MAP parameters.

`indices_to_warp` identifies columns of the original full input, including
the task column in that coordinate system. The default selects every
non-task column, even when the task column is first or interior. Explicit
negative indices resolve against the full width. Task columns, duplicates,
empty selections, and invalid indices are rejected. Concentrations must be
finite and positive; `eps` must be finite and strictly between zero and
one half. External task labels remain unchanged by normalization/warping
and are mapped to contiguous task indices separately.

Training inputs and cached GP solves are rebuilt when a member's sampled
warp changes. The public member `input_transform` reflects its actual
native transform, and mutations of its concentrations update the native
state transactionally. Previously returned posteriors and deep copies keep
their snapshots. Conditioning retains the sampled transformation. Query
adjoints apply the row-dependent warp Jacobian and normalization factors;
the task coordinate has zero derivative. First-order Torch query graphs
use these native adjoints.

The common outer `input_transform` remains the user-supplied normalization;
the sample-specific transformations are exposed on individual MCMC members,
like the port's existing per-member covariance and likelihood modules.
The native multitask model's existing restrictions still apply: the common
input transform is `Normalize`, the outcome transform is `Standardize`,
and arbitrary custom Python transforms are not evaluated inside Rust.

## Official issues and their corrections

The pinned official source has several related failures:

- `SaasFullyBayesianMultiTaskGP.load_mcmc_samples` discards the `Warp`
  returned by its Pyro model. With warping enabled, load dummy samples and
  inspect `getattr(model, "input_transform", None)`: without another
  transform it is `None`. Keep the returned transform and apply it to both
  training and prediction, after any common normalization.
- That returned transform is constructed with `d=ard_num_dims`, although
  multitask inputs contain one additional task column. An explicit physical
  last-feature index of `2` for a three-column input with task column `1`
  raises `IndexError`. Use the full input width and physical non-task
  indices. Default `range(ard_num_dims)` also incorrectly includes an
  interior task column and omits the last real feature.
- The official NumPyro `_warp_jax` swaps the canonical `c0`/`c1` exponents
  and omits the affine interior rescaling used by the loaded Torch `Warp`.
  Use the canonical formula above in the potential as well as prediction.
- The loader does not pass the Pyro model's custom `eps` to `Warp`.
  Preserve the sampled model's configured margin.

The corrected reference tests bypass only the discarded transform's broken
construction, load the same mean/kernel/noise samples, and explicitly attach
the official Torch `Warp` with full-width indices, sampled concentrations,
and matching `eps`. All GP computations in the reference remain official.
This makes the intended canonical model a reproducible comparison target.

## Validation

The focused Python tests cover list, NumPy, and Torch inputs; all three task
column positions; default and subset feature selection; external labels;
inferred and fixed observation noise; normalization and standardization;
corrected-reference posterior means, variances, and query gradients;
sample mutation, snapshots, copies, conditioning, configuration validation;
and actual short native NUTS fits for each array kind. Reference posterior
tolerances are `3e-7` relative / `3e-8` absolute, and query-adjoint tolerances
are `3e-6` / `3e-7`.

Native tests compare multitask posterior mean/covariance derivatives and
contracted VJPs with finite differences, and recheck warped multitask NUTS
potential gradients. The NumPyro trace tests now also compare warped
multitask unconstrained potential values and gradients to native Rust.
