# Device-resident fully Bayesian models

`FullyBayesianSingleTaskGP`, `SaasFullyBayesianSingleTaskGP` and
`FullyBayesianLinearSingleTaskGP` support `backend="torch"`. CUDA training
data, or training tensors requiring gradients, select it with `backend="auto"`.
Explicit `backend="native"` retains the existing CPU/f64 implementation.
The Torch classes remain instances of their public model families.

`load_mcmc_samples` creates registered mean, covariance and observation-noise
Parameters on the training device. Matérn models use the sampled ARD
lengthscales and an optional sampled outputscale; linear models use sampled
ARD weight variances and the reference's input centering. Both support sampled
Kumaraswamy warps, a subset of warped features, user input transforms and linear
outcome transforms. Fixed noise is recomputed from the original training data
and outcome transform; its fully Bayesian noise floor is `1e-4` after
transforming, matching the reference. Inferred noise is clamped to the same
floor. A noise sample exactly on the floor is represented by raw `-inf`, with a
finite constrained noise and zero derivatives through that saturated value.

Training data stay unbatched until conditioning or a sampled warp broadcasts
them. The MCMC axis is the final model-batch axis. Predictions use the shared
Rust Torch dispatcher for kernels and exact-GP factorization/conditioning.
No original data, predictions or derivatives are flattened into the CPU/f64
model, and no BoTorch or GPyTorch model computes this backend's predictions.

The loaded sample values must have the training dtype/device and the declared
sample count and feature width. They are detached when installed as Parameters,
as in the official loader: gradients target the model's registered leaves,
rather than the caller's input sample dictionary. Original training data,
fixed observation variances, queries and loaded mean/kernel/noise/warp
Parameters retain functional first- and higher-order Torch graphs. Static
transformed-input fitting follows the functional semantics described in
[the module notes](torch-backend-modules.md). In particular, these graphs retain
training-input derivatives that the reference's evaluation-time
`_set_transformed_inputs` intentionally detaches.

The returned `TorchGaussianMixturePosterior` is also an instance of the public
`GaussianMixturePosterior`. Its mean, variance, covariance and reparameterized
samples describe every Gaussian component; `mixture_mean`,
`mixture_variance` and `mixture_covariance_matrix` reduce the MCMC axis.
These operations, scalarization and output selection preserve the device and
graph. Normal samplers collapse their base samples across the MCMC axis, so
all components use matching normal draws. Marginal mixture quantiles locate
the Gaussian-mixture CDF root and use differentiable Newton refinements for
implicit derivatives, rather than differentiating bisection decisions.

Calling `train()` resets loaded components; `train(reset=False)` retains them.
Repeated sample loading replaces sampled transforms instead of accumulating
warps. Standard Torch state-dict loading initializes the correct sample batch
before loading its Parameters. Saved posterior moments retain their earlier
law after public parameter edits; standard Torch in-place version checks still
apply if a parameter used by an old graph is mutated before backward.

The focused checkpoint has 60 CPU/CUDA cases. It compares all three families,
fixed/inferred noise, sampled warps, Normalize/Standardize, batched queries,
mixture moments and covariance, canonical shared-normal samples and median
hyperparameters against official source
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. All training/query/loaded-parameter
first derivatives are compared, and complete mixed Hessian-vector products
are checked against differences of independent reference first gradients.
Tests also cover reset/reload/state behavior, immutable saved laws, quantile
query derivatives, input validation and automatic backend selection. For
reference training-data gradients, `propagate_grads(True)` is enabled; the
linear model's identical transformed training values are attached to their
functional graph, because evaluation otherwise detaches that graph.

`fit_fully_bayesian_model_nuts` now fits these three scalar families directly
on their training device. The Tensor potential evaluates sorted unconstrained
latent sites, their prior densities and Jacobians, and the GP likelihood using
port-owned Torch-dispatched Rust primitives. Multinomial NUTS implements
leapfrog integration, generalized U-turn checks, progressive proposal selection,
dual-averaging step adaptation and Welford/Stan-window dense mass adaptation.
Continuous states, momenta, mass matrices and random draws stay on device;
Python synchronizes scalar values for tree decisions and diagnostics. This is
GPU execution with scalar control overhead, rather than a fully compiled GPU
sampler. Float32 and float64 are supported. Each fit uses a private device-local
Torch generator and preserves global RNG state. A fixed seed repeats a chain
on the same device; CPU/CUDA/JAX generators produce different chains.

Draw replay can use recorded NumPyro momenta, uniforms and initial sites. Tests
compare every warmup and sampling state and adapted step size against a pinned
NumPyro run on both CPU and CUDA, including dense mass adaptation. Potential
values and gradients are independently compared against JAX and the existing
native implementation; Hessian-vector products are compared against differences
of native gradients. Known correlated Gaussian targets test the sampler law.
Public fitting tests cover thinning, refitting, fixed/inferred noise, sampled
warps, and prediction/data/parameter first and second derivatives after fitting.

Markov transitions, random/discrete tree choices and fitted MCMC draws are
intentionally detached, as in standard MCMC. The potential itself and loaded
component predictions support full Torch graphs. Invalid fitting requests leave
previously loaded components intact. The legacy `jit_compile` flag is accepted
as a no-op, matching pinned BoTorch `b0dd01be`, whose fitter never reads it.
The Tensor sampler remains eager; the native sampler is compiled. Progress bars
are currently omitted by both backends. MAP MLL fitting is explicitly rejected
for fully Bayesian classes; ordinary MAP SAAS models are a separate family.
Training inputs remain `n x d` with single-output `n x 1` observations, matching
the official family. The subsequent [multitask checkpoint](torch-fully-bayesian-multitask.md)
adds latent-feature multitask loaded inference and device NUTS. Arbitrary
custom Pyro-model overrides remain unsupported. A Gaussian mixture requires
linear outcome transforms. Ensemble acquisitions use the Torch adapter.

## Canonical sampling and rectangular roots

A loaded reference LinearKernel can retain a `q x d` covariance root when
`d > q`. GPyTorch correctly advertises `d` base-normal coordinates for that
root, and accepts samples with its advertised shape. An earlier version of
these notes incorrectly supplied `q` coordinates and described the resulting
shape error as an upstream bug. That finding is retracted; no upstream repair
is warranted by this example.

The device-resident port uses a square Cholesky root. This is the same Gaussian
law, with a different valid base-normal representation. To compare samples
using common `q`-dimensional normals, the parity tests construct an independent
dense GPyTorch MVN from the official posterior's existing mean and covariance.
Original reference models supply all compared moments. This oracle selects the
same canonical square root; it is not a workaround for a reference defect.

The loaded scalar families also expose `forward(X)` as their prior Gaussian
law, separately from conditioned prediction through `posterior(X)`. As in
the pinned source, training mode (`train(reset=False)`) applies the input
transform, while evaluation mode expects inputs in kernel coordinates.
The returned law uses transformed outcome units. Native list/NumPy priors
evaluate their covariance and draw samples in Rust. Torch query inputs use
the port's Tensor modules and preserve tested query gradients and Hessian-vector
products. These Tensor prior modules are snapshots of native model parameters;
use the Torch model backend for persistent model-parameter graphs.

For a one-input linear model queried at two points, the latent covariance is
rank one. Even when both libraries select dense roots, roundoff in its zero
eigenvalue can change whether a Cholesky factorization succeeds or adds
`1e-8` diagonal jitter. Identical base normals then need not produce identical
realized samples. Four affected CI configurations have this rank-one geometry;
controlled Torch 2.13 and 2.14 probes show posterior means and covariance
entries agreeing to ordinary floating-point precision, while the strict
sampled acquisition comparison depends on this platform-specific root choice.
This does not establish an upstream sampling defect.

The randomized scalar-MC comparison therefore uses `q <= d` for linear
models, with its existing value and gradient tolerances unchanged. It still
checks two-point posterior moments. Separate fixed- and inferred-noise
rank-one tests compare the joint posterior and its query derivatives to
closed-form Bayesian linear regression. Joint two-point qEI, qSimpleRegret,
and smooth qLogEI values and gradients are also checked against independent
closed forms using explicitly supplied zero base normals, which remove the
root-dependent sample term. These are deterministic contraction tests, not
claims about a Monte Carlo estimator's convergence.
