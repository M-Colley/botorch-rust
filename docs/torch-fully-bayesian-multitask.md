# Fully Bayesian multitask models on the Torch backend

`SaasFullyBayesianMultiTaskGP(..., backend="torch")` keeps continuous GP
calculations and NUTS states on the training Tensor's CPU or CUDA device.
CUDA tensors, training tensors requiring gradients, and Torch modules also
select this backend automatically. The ordinary native CPU backend is preserved.

The default model uses a SAAS data kernel multiplied by a Matérn-5/2 kernel of
sampled latent task embeddings. Per-task means, latent embeddings, task
lengthscales, inferred noise, shrinkage, and optional input-warp parameters
are included in the joint potential with their priors and unconstraining
Jacobians. `LatentFeatureMultiTaskPyroMixin` compositions with the port's
`MaternPyroModel` and `LinearPyroModel` are also supported. Task columns may
appear anywhere, and noncontiguous labels are mapped consistently. Sampled
warps affect selected physical non-task columns and retain their epsilon.
The Linear composition centers continuous features while preserving task labels.

Loaded samples construct a batch of correlated Gaussian components with
actual Torch Parameters for data-kernel, task-kernel, per-task mean, and
inferred likelihood parameters. The posterior is a
`TorchGaussianMixturePosterior`; it retains within-component cross-task
covariances, mixture moments, samples, output selection, and posterior
transforms. Queries may omit the task column to request all output tasks,
or include it to request particular tasks. Fixed and inferred noise,
Normalize/Standardize, reset/reload, and state-dict round trips are covered.
Training data, queries, and loaded parameters support first derivatives and
mixed Hessian-vector products. Categorical task identifiers have zero
continuous derivatives. Loaded MCMC draws are detached, as in standard MCMC;
there is no derivative through random/discrete NUTS transitions.

The existing device-resident multinomial NUTS implementation is reused,
including dense/diagonal mass adaptation, dual averaging, thinning,
private-generator random draws, and replay diagnostics. Python synchronizes
scalar tree decisions; continuous states and matrix algebra stay on device.
The eager sampler rejects `jit_compile=True`. Arbitrary custom probabilistic
or loader overrides are rejected rather than silently replaced by a built-in
potential. A Gaussian mixture requires a linear outcome transform. Tensor
observation-noise arguments retain the ordinary multitask model's explicit
unsupported boundary. Inputs are an unbatched long-format training table
with one observed scalar per row, as in the reference family.

## Independent validation

The focused suite contains 99 CPU/CUDA cases. It compares the potential and its complete first derivatives
against both the native implementation and independent JAX/NumPyro source;
potential Hessian-vector products are checked against native gradient
differences. A recorded NumPyro run supplies every momentum and uniform draw:
all warmup and sampling states and adapted step sizes are replayed on CPU and
CUDA, including dense mass adaptation. Separate public fitting cases execute
real NUTS, validate thinning and private RNG behavior, and differentiate
subsequent predictions. Loaded laws and all data/query/parameter gradients
are checked against official source
`b0dd01be61d517c387e234903a6e2cc38c057ea2`; mixed second derivatives are compared
with differences of independent reference first gradients.

The tests identify reference defects explicitly instead of treating incorrect
reference behavior as the desired law:

* The pinned multitask loader discards the returned sampled input transform;
  its default Warp construction also uses continuous width where full physical
  input width is needed. The parity oracle installs the selected-column,
  full-width sampled Warp with the sampled epsilon. This is the previously
  documented loader defect.
* `LinearPyroModel.load_mcmc_samples` hardcodes `ConstantMean(batch_shape=[S])`
  and assigns the mixin's `[S,T]` means to it. For `S=3,T=2`, the public
  `LatentFeatureMultiTaskPyroMixin, LinearPyroModel` composition fails with
  `RuntimeError: shape '[3]' is invalid for input of size 6`. It also bypasses
  the mixin's task-covariance construction, and its returned centering
  transform is discarded. Repair the Linear loader to use the same
  `_build_mean_module` / `_build_multitask_covariance` dispatch as the Matern
  loader, then retain a full-width transform that centers only continuous
  columns. Tests reproduce the exception before constructing that corrected
  reference oracle.

The reference lazy product operator can add jitter while rooting its singular
Linear covariance before multiplying it by the task covariance. For that
composition, the covariance oracle evaluates the two reference kernels
independently, multiplies their dense matrices, and applies the exact noisy
Gaussian conditioning equations. This selects the physical law without
depending on a particular singular-root decomposition; it is not a claim
that low-rank covariance roots are invalid.
