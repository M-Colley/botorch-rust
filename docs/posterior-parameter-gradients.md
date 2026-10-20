# Posterior hyperparameter derivatives

`SingleTaskGP.posterior_parameter_vjp(X, mean_adjoint, covariance_adjoint,
observation_noise=False)` computes an exact first-order native derivative
of a weighted posterior mean and joint covariance. The result follows the
flat order of `named_parameters()`, including noise, mean, kernel-tree and
learned input-transform parameters. It excludes prior-density terms.

The mean adjoint has the posterior mean's `... x q x m` layout. The
covariance adjoint has the joint normal's output-major
`... x (q*m) x (q*m)` layout. Both match the broadcast model/query batch
shape. Contributions from query batches are summed into the original
model parameters, and cross-output adjoints vanish for independent outputs.

The native pullback reuses the cached training factor and weights. It
differentiates the conditional mean and Schur complement with respect to
the training, cross and candidate kernel blocks, then contracts their
symmetric joint adjoint with analytic kernel derivatives. Learned input
transforms receive both training-point and query-point contributions.
Production gradients do not use finite differences.

Reference comparisons use BoTorch's `propagate_grads(True)` and exact
predictive covariance settings. BoTorch normally detaches training caches;
that default graph omits derivatives through the training solve. The native
explicit pullback computes the full conditional derivative. Native finite
differences independently check this behavior with inferred/fixed noise,
kernel trees, affine outcome transforms and learned input warping.

Ordinary `SingleTaskGP` constructed from CPU Torch tensors now exposes
persistent `torch.nn.Parameter` leaves through `parameters()`,
`named_parameters()`, and bound raw module attributes. Posterior mean,
variance and joint covariance connect these leaves to the native pullback,
including selected/scalarized outputs, independent output/batch parameters,
kernel trees and learned input warping. These graphs also retain existing
query-input derivatives. They compute the full conditional derivative,
equivalent to the reference settings described above, regardless of
BoTorch's default detached-cache setting. Only first derivatives are supported.

Torch optimizer edits are pushed before posterior evaluation, native MLL
evaluation, native acquisition construction and native fitting. Native
fitting/setters are pulled into the same leaves on the next public parameter
or model use; leaf identities and `requires_grad` flags are preserved.
Float32 native updates are rounded to their exposed parameter dtype.
Concurrent unsynchronized edits of both native parameters and Torch leaves
raise instead of overwriting either update. Private `_rs` calls bypass
automatic pre-use synchronization; synchronize via `named_parameters()`
before editing Torch leaves after private native writes.

Posterior graphs retain immutable native snapshots. Later fitting/setters
change future posteriors, while an earlier graph differentiates its original
law into the same parameter leaves. NumPy/list queries preserve their array
types, and `torch.no_grad()` suppresses parameter graph attachment.

This checkpoint deliberately scopes automatic parameter gradients to the
concrete ordinary `SingleTaskGP`; specialized GP subclasses and model lists
retain their previous parameter protocols. Sparse-outlier likelihood
parameters and conditioned input snapshots that do not match the stored
input-transform program remain explicit unsupported native pullbacks.
Posterior sample **parameter** backward raises `UnsupportedError`;
sample query-input differentiation still works. Exact MLL values and native
acquisition outputs retain their established explicit native-gradient/query
interfaces; this checkpoint does not attach their parameter backward graphs.
Native acquisitions capture model snapshots, so rebuild them after parameter
updates. External Torch optimizers can already train a model using a loss
built from the supported posterior moments.
