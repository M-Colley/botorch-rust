# Device-resident loaded-MCMC fully Bayesian models

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

This checkpoint implements loaded-sample inference. NUTS fitting on this
backend raises `UnsupportedError` before resetting a fitted model; the existing
native CPU NUTS implementation remains available. MAP MLL fitting is likewise
explicitly rejected for these fully Bayesian models. Training inputs remain
`n x d` with single-output `n x 1` observations, matching the official family;
this does not add GPU support for fully Bayesian multitask models or arbitrary
custom Pyro model loaders. A Gaussian mixture requires linear outcome
transforms. Ensemble acquisition reductions are implemented by the separate
Torch acquisition adapter.

## Overcomplete linear covariance roots in GPyTorch

With GPyTorch 1.15.2, a loaded fully Bayesian linear model with input dimension
`d=3` and `q=2` queried points can retain a `2 x 3` covariance root. The public
posterior advertises two base-normal coordinates, but
`MultivariateNormal.rsample` reshapes them using the root's three columns and
raises. This also occurs without BoTorch:

```python
import torch
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import RootLinearOperator
from botorch.posteriors.gpytorch import GPyTorchPosterior

root = torch.tensor([[1., .2, .3], [.5, 1.1, .7]], dtype=torch.double)
posterior = GPyTorchPosterior(MultivariateNormal(
    torch.zeros(2, dtype=torch.double), RootLinearOperator(root)
))
assert posterior.base_sample_shape == torch.Size([2])
posterior.rsample_from_base_samples(torch.Size([4]), torch.zeros(4, 2))
# RuntimeError: shape ... is invalid for input of size ...
```

A conservative workaround constructs a dense GPyTorch MVN from the official
posterior's existing mean and covariance. This preserves exactly the same
Gaussian law and uses a canonical square Cholesky root for positive-definite
covariances. The parity tests use that independent dense reference law for
shared-normal sampling; original official models supply all compared moments.
An upstream repair can compress an overcomplete `q x r` root (`r > q`) before
reshaping the advertised base samples. A reduced QR of its transpose gives
`R.T`, a `q x q` root with the same covariance and no added jitter. This keeps
the root dimensions consistent with the public event/base-sample shape.
