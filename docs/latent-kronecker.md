# Latent Kronecker GP

`botorch_rs.models.LatentKroneckerGP(train_X, train_T, train_Y)` implements
exact inference on a Cartesian feature/time grid. `train_X` is `n x d`,
`train_T` is `t x 1`, and `train_Y` is `n x t`. Nonfinite observations are
missing. The API follows the official BoTorch source at revision
`b0dd01be6`, rather than the older constructor in release 0.18.1.

Likelihoods, parameter gradients, feature-input gradients, optimization,
posterior covariance and sampling run in Rust. Python handles array
conversion, module registration, shapes and random draws. Complete grids
reuse the HigherOrderGP Kronecker eigensolver. Missing grids use exact
masked dense covariance solves; novel query times and feature gradients
also use exact dense inference. These dense paths are not a claim of the
large masked-grid iterative performance described by upstream.

Supported modules include native covariance kernel trees, zero/constant
factor means, Gaussian likelihoods, fixed input transforms and scalar
Standardize. LBFGSB and Adam fitting optimize the likelihood and all
registered factor kernel/mean parameters. `construct_inputs` factorizes
unbatched scalar SupervisedDataset rows using the feature named `step`.
Query feature/time batches broadcast, and Sobol/IID samplers use the
structured posterior's base-sample protocol. The base-sample layout is
full training-grid prior, observed training noise, conditional test prior.

On the native backend, training batches, learned input transforms and iterative
solver controls raise explicit unsupported errors. Observation noise in the Python
posterior, posterior transforms and conditioning raise NotImplementedError,
as in the audited official model. Model-specific acquisition adapters and
iterative masked-grid scalability remain separate gaps.

## Tensor CPU/CUDA backend

`LatentKroneckerGP(..., backend="torch")` evaluates the same masked correlated
feature/time Gaussian law on device, with persistent Torch parameters and
first and higher derivatives through training features, training times,
observed targets, query features/times, likelihood, factor means/kernels
and learned input transforms. CUDA inputs, gradient-bearing data or actual
Torch modules select this backend automatically. Tensor operations call the
Rust exact-factor/prediction bindings without detaching or copying numerical
values to the host. Native CPU/list behavior remains available.

Training batches are supported when the finite-observation mask agrees across
all members, as required by the pinned official model. Batch/query/time axes
broadcast. Means are the product of the two factor means, covariance is
`K_X` Kronecker `K_T`, and the actual observed rows/columns select the training
system. Default Standardize fits only observed cells per model batch;
raw-space moments and samples share the same affine law and retain the target
graph. The MLL normalizes by the number of observed cells. Public Torch fitting
optimizes these actual leaves; custom Torch covariance and Gaussian modules
retain their identities and derivative contracts.

Matheron sampling draws the full train-grid/query-grid joint prior, plus
independent noise only for observed training cells. Its normal vector uses
`n_train_full + n_observed + n_test` entries, in the advertised train-prior,
observed-noise, test-prior order. The block root preserves the complete Schur
complement, including the feature/time cross terms missing from the pinned
reference sampler. Both moments and draws use `batch x q x t` original-space
shape; `distribution` exposes the full point-major joint covariance.

Configured-noise conditioning and fantasies append observations on the
existing time grid and preserve the original transform fitting data. Missing
new observations are allowed with a shared mask. Fantasy observations use
the supplied sampler's full noisy Gaussian law, with no additional ambient
random draws outside its base-sample protocol. Feature/time/parameter
higher derivatives remain connected when `propagate_grads=True`.

This implementation uses dense exact masked solves and joint roots; it does
not claim iterative masked-grid scalability. Iterative solver controls,
explicit observation-noise posterior overrides and arbitrary posterior or
non-Standardize outcome transforms raise explicit unsupported errors.
Changing the existing training missing-value pattern requires reconstruction.
Dataset factorization continues to use the existing native adapter.

`test_torch_latent_kronecker.py` compares CPU/CUDA float32/float64 values,
likelihoods and full-graph gradients against a Gaussian conditional built
from the pinned official factor modules. It covers model batches, missing
cells, novel times, mixed Hessian products, actual sample covariance,
Standardize/Warp, custom modules, public fitting and deterministic fantasies.

## Official-source bugs and proposed corrections

The reference checkout and the user's separate BoTorch audit fork are not
modified. The following observations concern official revision `b0dd01be6`.

### Conditional sampling loses feature variance

The official `_rsample_from_base_samples` constructs the conditional test
covariance as a product of factor residuals:

```
(Kx_test - Ax) ⊗ (Kt_test - At)
```

where `Ax = Kx_test_train Kx_train^-1 Kx_train_test`, and analogously for
time. The correct covariance conditional on the full training grid is:

```
Kx_test ⊗ Kt_test - Ax ⊗ At
```

These expressions differ. In particular, querying the training times
makes the time residual zero and removes almost all conditional feature
uncertainty even at remote feature inputs. Reproduction with the complete
official checkout on PYTHONPATH:

```python
import torch
from botorch.models.latent_kronecker_gp import LatentKroneckerGP

torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
model = LatentKroneckerGP(
    torch.tensor([[0.0]]),
    torch.tensor([[0.0], [1.0]]),
    torch.tensor([[1.0, 2.0]]),
    outcome_transform=None,
).eval()
posterior = model.posterior(torch.tensor([[10.0]]))
print(posterior.variance)                       # approximately [0.6931, 0.6931]
print(posterior.rsample(torch.Size([4096])).var(0))  # approximately 1e-8
```

A correctness-first upstream fix forms the full conditional covariance
above and computes its PSD-safe root. If `B = Lx^-1 Kx_train_test ⊗
Lt^-1 Kt_train_test`, this is `K_test - B.T @ B`; retain `B.T @ w_train`
as the conditional mean term. The Rust implementation applies this fix.
A structured square-root method could improve the dense test-root cost
without changing the covariance. Regression tests compare native sample
means and full covariance with exact posterior moments on complete and
missing grids, at training and novel time inputs.

### Posterior moments use a different shape and outcome space

The official posterior wraps a flattened MultivariateNormal while setting
`_is_mt=True`: `posterior.mean` and `.variance` are flat, while `rsample`
returns `q x t`. Its posterior method also leaves the distribution in
standardized units when Standardize is present, while sampling explicitly
untransforms outcomes. The proposed fix reshapes moments to `q x t` and
applies the outcome transform to both distribution moments and samples.
For scalar fitted mean `mu` and standard deviation `sd`, use
`mean_original = mu + sd * mean_standardized` and
`cov_original = sd**2 * cov_standardized`.

This port consistently returns original-space `batch x q x t` moments
and samples. Eight official-source parity cases compare likelihoods,
autograd parameter gradients and complete joint covariance, explicitly
reshaping/untransforming the official moments before comparison. Native
Rust tests independently compare dense covariance likelihoods, all raw
parameter finite differences and analytic feature-input gradients.

Direct Tensor evaluation calls accept transformed feature coordinates and
return a flattened conditional in model outcome units, with an optional
query time grid. Eval `forward` computes the prior and preserves the source's
training-prefix missing-value mask on a joint feature/time grid. Public
`posterior` continues to accept raw features and return original-space
structured moments and samples. New direct-call comparisons cover CPU/CUDA,
normalization and explicit query/model batch axes; the combined structured
model and native fantasy regression run passes 173 cases. The 24 initial
direct-call cases also pass on CPU/CUDA, with all 12 CPU cases passing on
Torch 2.14. These overlapping counts do not certify a new release.
