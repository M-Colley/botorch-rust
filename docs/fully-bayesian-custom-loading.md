# Fully Bayesian returned-module loading

Audited against official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`.

Scalar Tensor fully Bayesian models call explicit `PyroModel.load_mcmc_samples`
overrides and the public `_build_mean_module`, `_get_covar_module`, and
`_build_multitask_covariance` extension hooks. The latter hooks receive the
source's dtype/device keywords. Inherited default behavior uses the existing
fast family loader; explicit hooks can return real Torch mean, covariance,
Gaussian likelihood and input-transform modules. The model retains those
module and Parameter objects, including custom forward behavior. Port-native
descriptors for supported kernels and means are converted to Tensor modules.

The supported prediction law is scalar Gaussian conditioning, with one batch
per MCMC sample. A kernel can return a dense Tensor or a differentiable operator
with `to_dense()`. Observation noise must be a supported diagonal Gaussian law:
the port's Gaussian/fixed-noise likelihoods or explicitly supplied GPyTorch
Gaussian/fixed-noise likelihoods with their canonical Gaussian marginal.
Overridden arbitrary likelihood marginals are rejected, rather than evaluated
as a different Gaussian law. Returned modules must already use the training
dtype/device; loading does not silently move caller-owned parameters.

A returned likelihood owns its observation law, even when the constructor
received `train_Yvar`. Returned fixed noise is used per training observation;
noisy predictions use its mean, matching the fully
Bayesian source policy. The ordinary descriptor loader continues to recompute
transformed constructor noise from the original Tensor so training-data and
noise graphs stay live. An explicitly supplied fixed-noise Tensor likewise
retains its graph.

Loading stages the model and fitted user transforms, validates an isolated copy
of returned mutable buffers, and checks component shapes and finite Gaussian
predictions before committing. Success preserves the caller's module and
parameter identities. Failure leaves the fitted model, held modules, fitted
transform state and prediction law intact. Arbitrary external side effects
performed by user hook code are outside this transaction. Custom state loading
uses the program's `get_dummy_mcmc_samples` hook, following the source's
`mean_module.raw_constant` state-layout convention.

Custom sampling/postprocessing programs still use optional NumPyro inference.
As in the source's explicit CPU/NumPy conversion, JAX's installed backend and
precision select that inference device and precision. MCMC draws are detached
from Torch differentiation. This adapter does not force arbitrary NumPyro
programs through Torch NUTS or import upstream BoTorch inference. After loading,
prediction executes the port's Tensor Gaussian conditioning on the training
device, with training-input, target, query, module-parameter and mixed
higher-order graphs. Direct custom-module loading and prediction require no
host Tensor data conversion. Built-in programs with only module-loader hooks
retain the Tensor NUTS fast path.

Multitask custom returned-module composition remains a separate implementation
block. The existing built-in latent-feature multitask loading and custom-prior
programs remain supported; arbitrary task-aware returned-module hooks are not
silently accepted. General non-Gaussian likelihood inference and custom state
layouts without the source's constant-mean key are also outside this protocol.

Tests exercise actual custom nonlinear mean/kernel/input modules on CPU/CUDA,
their retained identities, independent dense Gaussian values and mixed HVPs,
explicit GPyTorch Gaussian/fixed-noise noise ownership and graphs, original
module-buffer preservation on failed validation, real custom NumPyro fitting,
source hook keywords and dummy-state loading. No reference model or acquisition
is used in production prediction.

Validation: 21 new CPU/CUDA cases pass on Torch 2.13, together with 145 existing
scalar/native/custom-program cases. The new checks include query counts equal
to and different from the training count for fixed-noise predictions. Ruff and
Mypy pass for the changed production modules.
