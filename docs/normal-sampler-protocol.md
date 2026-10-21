# Normal sampler public protocol

`IIDNormalSampler` and `SobolQMCNormalSampler` accept an empty `sample_shape=()`:
this draws one event without introducing a sample axis. Multidimensional
sample shapes retain all their sample axes, including explicitly assigned
base normals.

A caller-supplied Torch posterior needs only public `dtype`, `device`,
`base_sample_shape`, `batch_range`, and `rsample_from_base_samples` members.
No `_kind` or `_tensor_backend` marker is required. The samplers collapse the
declared batch axes while constructing normals, then expand those axes to
`sample_shape + base_sample_shape` before invoking the posterior's method.
This preserves common random numbers across batches and lets the posterior
control its own sampling algebra and autograd graph. Native model posteriors
continue to use their Rust sampler cache. Resetting Sobol `base_samples` to
`None` clears both Python and native explicit-normal caches while retaining
the sampler seed.

The Torch protocol generates IID normals on the posterior device with a
private generator. Sobol integer sequence generation follows Torch's CPU
Sobol engine; only the newly generated uniforms are transferred to the
requested device, and inverse-normal transformation runs there. No posterior
data or covariance is copied to CPU. This is sampling interoperability for
an explicitly supplied posterior, not delegation to BoTorch inference.

`test_normal_sampler_public_protocol.py` checks strict keyword-only custom
posteriors, expanded base layouts, unchanged global RNG state, cached reuse,
CPU/CUDA first and second derivatives, empty-shape real Gaussian sampling,
explicit normals, and reset semantics. CPU seeded values agree with pinned
official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`. The initial focused
run passed 71 cases including existing sampling, cache-prefix, and sampler
mixin regressions. This change does not extend the separate structured
Matheron `_update_base_samples` layout boundary.

The follow-up broadcast repair inserts omitted batch axes after every MC
sample axis, so explicitly supplied `sample_shape x q` normals retain their
identity and gradients when sampled across model batches. Full public
single-output layouts take precedence over the legacy `q x 1` shorthand:
when `q=1`, valid trailing singleton batch/event axes are not squeezed or
reinterpreted as an output axis. The same shape-only expansion is shared by
the sampler and the Tensor Gaussian posterior's direct base-sample method.

The expanded protocol suite passes 76 cases on Torch 2.13 CPU/CUDA, including
independent Gaussian sample values and all first/second derivatives through
means, covariance matrices and explicit normals. Torch 2.14 CPU passes the
same available cases plus the previous explicit-normal model regression
(45 passed, 32 CUDA skips). The explicit model regression and complete KG
and multistep lookahead modules pass 62 CPU/CUDA cases without changing
numerical tolerances or acquisition formulas.
