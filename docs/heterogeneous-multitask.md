# Heterogeneous multitask Gaussian processes

The port provides `CombinatorialCovarModule`, `MultiTaskConditionalKernel`, and
`HeterogeneousMTGP` under the matching BoTorch module paths. Reference tests use
official BoTorch revision `b0dd01be6`, rather than the separate local corrected
BoTorch fork.

The conditional kernel partitions features into maximal groups with identical
task memberships. Each group has its own scaled Matérn-5/2 covariance, included
only when both tasks contain those features. An optional scaled Matérn-5/2
kernel operates on binary membership vectors. The Rust evaluator prepares those
vectors once per point set, evaluates all covariances and input derivatives,
and contracts exact parameter derivatives. Public parameter paths, constraints,
learned SAAS shrinkage parameters, and prior terms follow the reference.

BoTorch registers the combinatorial covariance even when
`use_combinatorial_kernel=False`. Its SAAS priors still contribute to the MLL;
its covariance parameters without priors have zero likelihood gradient. The
native implementation preserves this behavior. The combinatorial kernel uses
SAAS whenever its ARD dimension exceeds one, independently of the flag that
controls SAAS on the spatial subkernels.

The GP embeds each task's observations into full feature space, imputes missing
features using empirical means reduced in Rust, and reuses the native exact
multitask GP. Predictions use task 0. Public query inputs contain target features
in their declared order followed by a zero task column. The native acquisition
adapter applies that same mapping, with forward and reverse derivatives
projected back to target coordinates. Optimization bounds must fix the final
task coordinate to zero. Target conditioning and fantasies also embed their
inputs before reaching the native GP.

Supported configurations include unbatched training data with unequal task row
counts, batched query inputs, fixed or learned task noise, arbitrary declared
target feature order, full or reduced task rank, optional SAAS and combinatorial
covariance, and `Standardize` outcomes. Input transforms use the native
`MultiTaskGP` support for `Normalize`; specify non-task `indices` explicitly.
Transforming a categorical task indicator is rejected. Each task must contain
at least one feature, feature indices must be unique and in range, and task
values must be `0..len(feature_indices)-1`. Fractional task indicators are
rejected, rather than silently truncated. Other outcome transforms and
tensor-valued prediction noise retain the explicit native multitask limits.

## Verified upstream conditioning bug and solution

At official revision `b0dd01be6`, create a `HeterogeneousMTGP` with feature maps
`[[0, 1], [1, 2]]` and `full_feature_dim=3`. After predicting at
`X=[[0.3, 0.4, 0.0]]`, calling `condition_on_observations(X, Y)` raises
`RuntimeError: Sizes of tensors must match ... Expected size 4 but got size 3`.
The posterior maps these three-column target inputs to full inputs, but the
inherited conditioning method concatenates them directly with four-column
training inputs.

The fix maps target inputs before invoking the base conditioning method. This
port applies that fix to conditioning and fantasies. A regression reproduces
the reference failure and compares native conditioned posterior means and
covariances with the reference after explicitly mapping its conditioning inputs
to full space. The same mapping is required by native acquisition model handles
to preserve the public model's input contract.

Validation includes native finite differences for every covariance parameter,
continuous input derivatives, forward/reverse target-embedding derivatives,
and zero task-indicator derivatives; official kernel, likelihood, parameter
gradient, posterior, and expected-improvement comparisons; and native fitting,
conditioning, fantasy, and acquisition-optimization checks across list, NumPy,
and torch forms. Configuration availability is limited to those documented
above; these classes do not delegate inference to Python BoTorch or GPyTorch.
