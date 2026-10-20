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
training with explicit inducing points. Default multioutput observations use
rank-zero multitask Gaussian noise, with global and task variances. Gaussian
noise, variational means/factors, inducing locations, kernel parameters, means,
training inputs/targets and posterior queries participate in the Torch graph.
Fitted transform graphs are recomputed from the original training data when
needed, retaining functional training-data derivatives.

Default unbatched inducing allocation is Torch pivoted Cholesky with the first
output's covariance for multiple outputs, matching GreedyVarianceReduction.
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

This checkpoint does not implement other variational distributions or
strategies, non-Gaussian observations, rank-positive multitask noise, or custom
inducing allocators. Those configurations raise `UnsupportedError`. Automatic
allocation currently requires unbatched training inputs; batched models require
explicit inducing points. Reallocation cannot resize the variational family.
It does not imply GPU support for arbitrary `ApproximateGPyTorchModel` wrappers
or every other approximate-model family.
