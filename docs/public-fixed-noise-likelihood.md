# Public fixed Gaussian likelihood

`FixedNoiseGaussianLikelihood(noise, learn_additional_noise=False,
batch_shape=(), noise_prior=..., noise_constraint=...)` accepts Tensor
observations directly and returns an actual Torch module. It retains the fixed
observation-data graph and optional learned noise Parameters. Scalar exact,
fully Bayesian and supported scalar-event structured models adopt the supplied
module through their Gaussian likelihood protocol without importing GPyTorch.

The module exposes `noise_covar`, `second_noise_covar`, `noise`, `second_noise`,
Gaussian marginals, pointwise log probabilities, expected log probabilities and
the fixed-noise fantasy hook. It preserves source state keys, dtype/device
migration, hooks, copies and serialization. Fixed observations are ordinary
Tensor data, matching the source's checkpoint semantics. Native array models
continue to receive fixed observation variances through `train_Yvar`; an
unbound array descriptor exposes its supplied fixed values without importing
Torch.

Explicit observation noise replaces the fixed component while retaining the
additional learned component. This follows the repaired Gaussian contract in
[GPyTorch #2826](https://github.com/cornellius-gp/gpytorch/pull/2826), rather than
the pinned implementation's duplicated explicit override. Conditioning appends
fixed observations once and independently copies learned likelihood parameters.

Twenty-four focused CPU/CUDA cases compare source values, all active first and
mixed second derivatives, actual module protocols and independent corrected
override/conditioning laws. All 204 related integrated cases pass. The focused
module passes twelve cases on Torch 2.14 CPU; with reference packages blocked,
eight pass and sixteen source-only comparisons skip. Configured formatting,
Ruff and Mypy pass. These focused selections overlap and do not replace the
complete release verification.
