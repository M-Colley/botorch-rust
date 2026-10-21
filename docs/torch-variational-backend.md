# Device-resident variational GP checkpoint

`SingleTaskVariationalGP(..., backend="torch")` uses a port-owned whitened
sparse variational model with actual Torch Parameters. CUDA inputs or training
Tensors requiring gradients select this path with `backend="auto"`.
`backend="native"` retains the existing CPU/f64 implementation.

The implemented family is `q(v) = Normal(m, L L.T)` under the whitened prior
`Normal(0, I)`. Inducing locations are Parameters when learned and buffers when
fixed. The model uses the port's Torch kernel, mean, likelihood and transform
modules. Kernels and positive-definite factorization dispatch through Rust to
the installed Torch runtime; triangular solves and variational contractions use
Torch tensor operations. No BoTorch or GPyTorch model computes predictions,
losses or derivatives for this backend.

Predictive marginals use `A = chol(K_ZZ + jitter I)^-1 K_ZX`, with mean
`mean(X) + A.T m` and variance
`diag(K_XX) + jitter + diag(A.T (L L.T - I) A)`. Computing an ELBO therefore
does not materialize an N-by-N training covariance. Full covariance is
materialized lazily when requested. Scalar training calls retain GPyTorch's
diagonal-only covariance update when N exceeds the number of inducing points;
evaluation and independent multitask prediction use the full update.

`VariationalELBO(model.likelihood, model.model, num_data=...)` dispatches to a
Torch module that is also an instance of the public `VariationalELBO`, with
ordinary `.parameters()`, `.backward()`, `.zero_grad()` and
Torch optimizer behavior. It computes the analytic Gaussian expected log
likelihood, whitened Gaussian KL and registered hyperparameter priors. Explicit
minibatch inputs and targets are supported. `combine_terms=False` returns the
same likelihood, scaled KL and prior terms as the reference. A saved q(f) output
retains its noise, KL, prior terms and lazy kernel parameter snapshot; passing a
prior output to this ELBO raises an explicit error.

Both `fit_gpytorch_mll` and the low-level SciPy/Torch fitters optimize the
actual Parameters. The default SciPy path filters frozen leaves and uses the
Tensor ELBO graph rather than the CPU native variational fitter. External
`DataLoader` closures evaluate each supplied minibatch and restart their
iterator at the end of an epoch; an empty loader raises `ValueError`. The
gradient closure works with both optimizer adapters, including actual Adam
updates from full or smaller minibatches on CPU and CUDA. As with the reference,
external inputs and targets should already be in the inner model's transformed
space. `combine_terms=False` is evaluated directly, rather than by the fitting
closure.

Single and independent multiple outputs are supported, including batched
training with automatic or explicit inducing points. Default multioutput observations use
rank-zero multitask Gaussian noise, with global and task variances. Positive-rank
`MultitaskGaussianLikelihood` also supports learned task factors, full cross-task
noise in predictive/marginal covariances, and all first/second noise-factor
derivatives. The expected ELBO log-likelihood and sample-conditional distribution
use the task-noise diagonal, matching GPyTorch's variational likelihood semantics.
Gaussian
noise, variational means/factors, inducing locations, kernel parameters, means,
training inputs/targets and posterior queries participate in the Torch graph.
Fitted transform graphs are recomputed from the original training data when
needed, retaining functional training-data derivatives.

Inducing allocation uses device-resident pivoted Cholesky, including model
batches, `GreedyImprovementReduction`, and custom `InducingPointAllocator`
quality functions or allocator implementations. Kernel/quality evaluation and
discrete pivot selection stay on the input device. Selected input values retain
their first- and higher-order data graphs; pivot choices are discrete. Independent
outputs share locations selected using the first output's covariance.
Reinitialization preserves the inducing leaf's identity and resets lazy
variational initialization. Cloning the model rewires strategy ownership to
the clone. A prior-only call does not initialize q(v) or consume random draws.

The model suite passes 35 CPU/CUDA cases. It compares q(f), full
covariance and all ELBO terms against the pinned official reference, compares
all first data/parameter derivatives, checks every data/parameter Hessian-vector
product against finite differences of independent reference gradients, exercises
float32/float64 batched graphs, and trains real minibatch Adam optimizers with
learned and fixed inducing locations.

An additional 31 fitting integration cases check public ELBO identity, parameter
freezing and clearing, default SciPy fitting, DataLoader validation and cycling,
and full/minibatch Adam values, gradients and actual parameter updates against
the independent pinned reference. The optional detached `value_and_grad()`
extension evaluates current Parameters and returns a flat gradient in their
registered order; explicit alternate raw vectors are unsupported on this
backend, so assign the Parameters before calling it.

An additional allocation suite checks CPU/CUDA, float32/64, quality-weighted
pivots, selected-input first/second derivatives, custom allocators, batched
construction/ELBO derivatives and reinitialization against independent reference
computations. For model batches with distinct hyperparameters, each batch uses
its own covariance. Pinned `b0dd01be6` instead recomputes all batched kernels for
each input row and takes `kernel_element[0]` repeatedly, so its public batched
allocator incorrectly uses the first model's hyperparameters for every batch.
The regression compares each batch against the reference's standalone pivot
routine with that batch's actual covariance.

Twelve additional CPU/CUDA cases compare rank-zero, rank-one and full-rank task
noise against independent GPyTorch/BoTorch likelihood, posterior and ELBO calls.
They check complete first parameter derivatives, noise Hessians, noisy posterior
query/parameter derivatives and the sample-conditional event shape.

This checkpoint does not implement other variational distributions or
strategies or non-Gaussian observations. Those
configurations raise `UnsupportedError`. Reallocation cannot resize the variational family.
It does not imply GPU support for arbitrary `ApproximateGPyTorchModel` wrappers
or every other approximate-model family.
