# Device Gaussian utility paths

`utils.low_rank.sample_cached_cholesky` uses Tensor triangular solves,
Schur complements and the native Torch-dispatch safe Cholesky operation
when its posterior exposes Torch values. Numerical arrays stay on their
device. Means, covariance blocks, supplied baseline factors and normal
draws retain their complete Torch graph, including higher derivatives.
The helper does not detach the baseline factor; callers that intend a
fixed cache must pass a detached factor themselves.

Normal draws retain the public point-major layout and are shared across
model/t-batches. Empty, one-dimensional and multidimensional sample shapes
produce `sample_shape x batch_shape x q x m` results. Independent outputs
are required: correlated multitask covariance blocks raise `BotorchError`.
Both output-major Tensor Gaussian laws and interleaved supplied Gaussian
distributions are handled. Native list and NumPy calls retain the Rust
sampling path. Adaptive jitter applies only to failing conditional blocks,
using the existing dtype-specific policy and `max_tries`.

CPU/CUDA tests compare values and weighted first/Hessian-vector derivatives
with the pinned reference on symmetric positive definite covariance
coordinates. General sample shapes additionally match full-joint Gaussian
draws with identical normals. This avoids the source's multidimensional
reshape defect, already covered by [BoTorch #3501](https://github.com/meta-pytorch/botorch/pull/3501).
Host-transfer guards, independent multi-output batches, supplied posterior
distributions, zero baseline size and singular conditional recovery are
tested separately.

`utils.evaluation.compute_in_sample_model_fit_metric` evaluates a Tensor
model's actual Gaussian likelihood on device. It returns a Python float
under `no_grad`, as the public source API specifies; it does not promise a
training graph. MLL is the unnormalized data log probability without priors,
averaged over output/model/MCMC batches. AIC/BIC count actual registered
parameters, with the source's MCMC adjustment. Loaded MCMC samples survive
the training-mode transition and the model finishes in evaluation mode.

For a default multitask model, the unchanged source metric omits the task
indices required by its Hadamard likelihood and raises
`ValueError("Task indices must be provided.")`. The port supplies mapped
training task inputs to that likelihood. Its regression preserves the
original source failure and compares the repaired result with the explicit
source Gaussian law, parameter count and information-criterion formulas.
The proposed upstream repair is [BoTorch #3515](https://github.com/meta-pytorch/botorch/pull/3515).
