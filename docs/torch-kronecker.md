# Correlated Kronecker Tensor inference

`KroneckerMultiTaskGP(..., backend="torch")` keeps the actual correlated
input-by-task law on the Tensor device. CUDA data and training data requiring
gradients select it automatically. The covariance is `K_X ⊗ K_tasks`; training
noise is `I_n ⊗ N_tasks`. Task noise can be diagonal or a learned factor product
plus global noise. No independent-output approximation is substituted.

The initial implementation materializes this joint covariance and uses the
Rust Tensor exact-Gaussian dispatcher. This establishes the exact mathematical
law and higher derivatives; it does not claim structured-solver scalability.
Memory is quadratic in the number of training points times tasks and dense
factorization is cubic. Training data, model/query batches, task factors,
task variances, means, noise, queries and outcome statistics keep their graphs.
Custom Gaussian multitask likelihood modules retain their actual Parameters;
the adapter builds their task-noise covariance with Tensor operations to avoid
a lazy operator's first-order-only backward.

The posterior is a `MultitaskGPPosterior` and samples with Matheron's rule:
a correlated prior over training/query outputs, independent training noise,
and optional query noise, followed by the exact conditioning update. The
advertised normal vector contains `(n+q)m + nm` coordinates, plus `qm` for
observed predictions. Multiple sample axes and sampler batch collapsing retain
that layout. Roots use differentiable Cholesky factors with the shared adaptive
jitter policy, rather than the reference's eigenvector roots; identical normal
vectors therefore need not give identical samples, while the joint Gaussian
law agrees. Sampling and double backward perform no CPU/list/NumPy conversion.

Unlike the reference's marginal-only public distribution view, the Tensor
posterior exposes the actual full joint conditional covariance in output-major
order, consistent with the other Tensor Gaussian posteriors. Sampling uses the
same correlated law. Output selection and scalarization use that full matrix.
Standardize applies its captured affine transformation to moments and Matheron
samples. [General outcome transforms](structured-outcome-transforms.md) also
support nonlinear maps, chains, and custom differentiable affine modules while
retaining the structured sampling protocol.
Conditioning and fantasies append complete task vectors at new input points;
fantasy batches and `propagate_grads=True` retain the full derivative graph.
Explicit per-observation conditioning noise is retained in transformed outcome
units. Fantasies use the configured Gaussian task-noise model.

Task covariance has a learned factor and positive diagonal, with the default
LKJ correlation / marginal-scale prior. The prior keeps the existing native
port's documented correction: one correlation log density plus the sum of
scale log densities. GPyTorch 1.15.2 repeats the correlation term for each task
when the scale prior is univariate. Tests separate the exact data MLL from this
prior difference and compare the corrected reference density and gradients.
As in GPyTorch IndexKernel, the matrix prior has no inverse-setting closure
for arbitrary low-rank-plus-diagonal Parameters; prior resampling reports this
missing setter explicitly.

`test_torch_kronecker.py` covers CPU/CUDA, float32/float64, batched models,
rank-one correlated task noise, exact dense conditional laws using the actual
reference covariance modules, data MLLs, parameter/training/query derivatives,
mixed Hessian-vector products, full Matheron sample derivatives, 24,000-draw
joint covariance checks, Standardize, selection, conditioning and fantasies.

## GPyTorch batched task-factor repetition

GPyTorch 1.15.2 `MultitaskKernel.forward` repeats an already batched task
covariance by the input batch sizes. A task batch of size two and an input
batch of size two therefore produce task batch four against data batch two:

```python
import torch
from gpytorch.kernels import MultitaskKernel, RBFKernel

X = torch.rand(2, 4, 1, dtype=torch.double)
kernel = MultitaskKernel(
    RBFKernel(batch_shape=torch.Size([2])), num_tasks=2, rank=1,
    batch_shape=torch.Size([2]),
).to(X)
kernel(X, X).to_dense()
# RuntimeError: Batch shapes ... are incompatible for a Kronecker product.
```

The concrete repair removes the blind
`covar_i.repeat(*x1.shape[:-2], 1, 1)` and lets
`KroneckerProductLinearOperator` broadcast the data and task batch shapes.
The regression asserts the original CPU/CUDA failure and applies that narrow
forward repair in a test context, verifying both full and diagonal covariance
against separate per-batch Kronecker products. This is distinct from the
existing active-dimension dispatch issue: no feature-selection behavior needs
to change to repair batch repetition. Batched model comparisons form the exact
law from the unmodified reference's actual data and task component modules,
avoiding its broken composite wrapper.

Direct Tensor evaluation calls consume already transformed inputs and return
the latent conditional in standardized model units. `forward` evaluates the
prior in both train and eval modes, applying input transforms only in train
mode. Public `posterior` still accepts raw features and returns original
outcome units. The structured direct-call regressions compare values and query
gradients and check finite query Hessians on CPU/CUDA, including query/model batch axes.
The source batch-factor defect is filed as
[GPyTorch #2817](https://github.com/cornellius-gp/gpytorch/pull/2817).

Tensor factory selection also preserves caller-defined subclasses. Their
`forward` Gaussian defines the full prior used by prediction and exact MLLs;
both interleaved actual Gaussian and port output-major covariance layouts are
recognized. The [subclass checks](builtin-model-subclasses.md) cover complete
first/higher graphs, actual source controls, registered state and serialization.
