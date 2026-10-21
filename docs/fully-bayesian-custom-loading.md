# Fully Bayesian returned-module loading

Audited against official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`.

Scalar and latent-task Tensor fully Bayesian models call explicit `PyroModel.load_mcmc_samples`
overrides and the public `_build_mean_module`, `_get_covar_module`, and
`_build_multitask_covariance` extension hooks. The latter hooks receive the
source's dtype/device keywords. Inherited default behavior uses the existing
fast family loader; explicit hooks can return real Torch mean, covariance,
Gaussian likelihood and input-transform modules. The model retains those
module and Parameter objects, including custom forward behavior. Port-native
descriptors for supported kernels and means are converted to Tensor modules.

The supported prediction law is Gaussian conditioning, with one batch
per MCMC sample. Multitask models retain the joint cross-task covariance and
use a long-format scalar training table. A kernel can return a dense Tensor or a differentiable operator
with `to_dense()`. Observation noise uses the port's Gaussian/fixed-noise
likelihoods or explicitly supplied GPyTorch Gaussian likelihoods with their
canonical Gaussian marginal. These include correlated, input-dependent noise
operators supplied through `_GaussianLikelihoodBase.noise_covar`.
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

For an explicitly returned likelihood, conditioning evaluates that module's
actual Gaussian marginal on the training prior and uses the full observed
covariance in the port's native Tensor solve. Noisy query predictions apply
the actual marginal again, with the source's explicit-noise/fixed-noise
override rules. Training receives the source's one-element input sequence;
prediction receives the query Tensor. No noise vector is substituted for a
correlated covariance. Training and test noise are applied separately, matching
the source prediction strategy rather than inventing a cross-noise block.
Optional GPyTorch distributions/operators are used only to call an explicitly
supplied GPyTorch likelihood; inference and conditioning use no upstream model.
The adapter validates finite symmetric PSD noise per MCMC component before
adoption, without changing its higher-order differentiation graph.

Loading stages the model and fitted user transforms, validates an isolated copy
of returned mutable buffers, and checks component shapes and finite Gaussian
predictions before committing. Success preserves the caller's module and
parameter identities. Failure leaves the fitted model, held modules, fitted
transform state and prediction law intact. Arbitrary external side effects
performed by user hook code are outside this transaction. Custom state loading
uses the program's `get_dummy_mcmc_samples` hook, following the source's
`mean_module.raw_constant` or multitask
`mean_module.base_means.0.raw_constant` state-layout convention.

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

Task-aware returned-module hooks retain the actual mean, covariance,
Gaussian likelihood and input transformation. The public task-index hook uses
mapped task identifiers and physical non-task columns. The default task
composition selects those continuous columns from a caller's actual data
kernel, then multiplies by the sampled latent-feature covariance. Interior
task columns, external labels, selected-column Warp and epsilon are preserved.
The Linear composition uses the corrected mean/covariance dispatch and
continuous-only centering described in [the multitask notes](torch-fully-bayesian-multitask.md).
Fixed-noise conditioning appends the returned likelihood's actual noise and
retains its training-noise graph, even if it differs from constructor noise.
General non-Gaussian likelihood inference and custom state
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

The correlated-likelihood checkpoint adds CPU/CUDA independent full covariance
values, all training-input/target/query/module-parameter first derivatives and
mixed HVPs; direct GPyTorch ExactGP values and first derivatives; explicit noisy
query overrides; host-conversion guards; and failed-loading rollback, including
mixed-scale component noise PSD validation. It also checks that substituting
only the public diagonal noise vector produces a different answer, so a
diagonal shortcut cannot satisfy the regression.

The 35 module-loader/correlated-likelihood cases pass on Torch 2.13 CPU/CUDA;
Torch 2.14 CPU passes 18 cases with 17 CUDA skips. A custom model's virtual
Gaussian prior and its own scale Parameter are tested through the shared
`_prior_blocks` hook, independently of the returned covariance module.

The task-aware checkpoint adds 38 CPU/CUDA cases. Actual GPyTorch RBF/Scale
data kernels match the pinned source's joint laws and query gradients across
all three task-column positions, fixed/inferred noise and sampled warping.
Independent RBF-times-task-factor Gaussian formulas check complete data,
query, noise and module-parameter first derivatives and mixed HVPs, including
conditioning and full correlated observation covariance. Real custom NumPyro
fits adopt their task-aware returned modules, followed by device predictions
with full derivative graphs. The scalar/task/correlated-loader selection
passes 38 Torch 2.14 CPU cases, with 37 CUDA cases skipped in that CPU build.
The focused task/scalar-loader, custom-program, prior and multitask selection
passes 257 cases on Torch 2.13 CPU/CUDA. Ruff and Mypy pass for all seven
changed production modules.

Validation copies use a fresh Python object shell and cloned mutable buffers;
they do not invoke a supplied module's copy/pickle hooks. This is necessary
because GPyTorch main `e1302d24cbd334649ada70608cbe98e903a47a83` returns its live
`Kernel.__dict__` from `__getstate__`: `copy.copy(kernel).__dict__ is kernel.__dict__`,
so assigning an ordinary copied attribute also changes the original.
Copying/deepcopying/pickling also clears the original `distance_module` cache.
The separate source repair [GPyTorch #2825](https://github.com/cornellius-gp/gpytorch/pull/2825)
returns a separate state dictionary and clears only its serialized cache.
Normal shallow sharing of Parameters, constraints and child modules remains
intentional. Original-source regressions fail seven alias/cache checks;
48 serialization/RBF/Scale tests pass after repair on Torch 2.13 CPU/CUDA
and Torch 2.14 CPU. The port has independent scalar and multitask supplied-kernel
identity/registry/rollback controls, so it does not depend on this upstream
repair being installed.
