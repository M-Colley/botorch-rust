# Supplied variational distributions

The Tensor variational backend retains a caller-supplied GPyTorch
`_VariationalDistribution` module itself under
`model.model.variational_strategy._variational_distribution` (under the base
strategy for multiple outputs). Its registered parameters, buffers, forward
implementation, initialization method and trainability flags stay intact.
Subclasses may add parameters or change the returned Gaussian law. The forward
result must be a GPyTorch `MultivariateNormal` or `Delta`, matching the source
strategy contract. Initialization receives the actual inducing prior in the
distribution's coordinates.

The port computes the GP conditional, Gaussian/Delta KL and observation ELBO.
Each output's conditional and saved KL use the same evaluation of the supplied
distribution. A later model call cannot replace a saved output's variational
law. Explicit-normal Matheron updates sample the supplied distribution's own
root, preserving its sample coupling and device graph.

This includes `NaturalVariationalDistribution` and
`TrilNaturalVariationalDistribution`. Their custom backward rules return natural
gradients rather than Euclidean derivatives of their stored natural parameters.
The port preserves those rules; it does not substitute ordinary inverse-matrix
autograd. Use `gpytorch.optim.NGD` or another optimizer that follows precisely
that gradient direction for these parameters, as required by GPyTorch. The inner
model exposes `variational_parameters()` / `named_variational_parameters()` and
the complementary hyperparameter iterators for separate optimizers. Natural
parameter Hessians are not advertised as ordinary Euclidean Hessians.

CPU and CUDA tests cover whitened and unwhitened strategies, independent outputs,
model batches, float32/float64 execution, full data/parameter VJPs, repeated NGD
steps, first-call initialization, custom-module registration and current-q path
updates. Custom Gaussian-module mixed Hessians are compared with finite
differences of independent reference first gradients because the pinned
`linear_operator` triangular operations have a separately reported matrix-Hessian
defect ([linear_operator #152](https://github.com/cornellius-gp/linear_operator/pull/152)).

This checkpoint does not add `BatchDecoupledVariationalStrategy` or
`CiqVariationalStrategy`; their different inducing/whitening laws require their
own implementations. Strategies whose constructors require an additional base
strategy, grid or task arguments are not interchangeable with the
`SingleTaskVariationalGP` class argument in the upstream API. The source also
does not register default unwhitened or independent-multitask Matheron updates.

The separately verified scalar-probit log-CDF repair is published as
[GPyTorch #2820](https://github.com/cornellius-gp/gpytorch/pull/2820). Against
official base `e1302d24cbd334649ada70608cbe98e903a47a83`, the four new test
methods expose eight failing subtests and four gradcheck errors; the two existing
methods pass. The repair passes 256 relevant upstream function, Bernoulli and
variational-strategy tests with CPU/CUDA checks, plus the upstream formatting and
lint configuration. The port's corrected probit primitive and disclosed reference
comparison remain described in [torch-variational-probit.md](torch-variational-probit.md).
