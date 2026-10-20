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

This explicit derivative interface is a foundation for model-parameter
autograd. It does not make `parameters()` return persistent Torch
parameters or attach ordinary `posterior()` outputs to those parameters.
Current automatic posterior graphs cover query derivatives; callers can
use this pullback directly for parameter sensitivities. Sparse-outlier
likelihood parameters and conditioned input snapshots that do not match
the stored input-transform program remain explicit unsupported cases.
