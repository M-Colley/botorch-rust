# Finite ensembles on the Tensor backend

Caller-defined `EnsembleModel` subclasses use the public Torch module
adapter. Their `forward(X)` produces `batch x members x q x outputs`
tensors. `EnsemblePosterior` now keeps Tensor values and weights on their
device, with differentiable weighted means, reliability-weight sample
variances, mixture moments, and selected-member samples. List/NumPy
posteriors retain the existing array implementation.

`IndexSampler` draws categorical member indices on that device using a
local seeded generator. It preserves the global CPU and CUDA RNG states.
Each index selects the same member across all query points and outputs,
retaining its complete joint function draw and the member/query parameter
graph. Its random coordinates do not grow when query points are appended.
As in the reference, a cached index table remains fixed while its sample
and posterior batch shapes are unchanged. Selection is discrete: sample
gradients reach member values, while sampling probabilities have no
pathwise derivative. Posterior moments do differentiate the weights.

Scalar MC and hypervolume acquisitions select `IndexSampler` by default
for finite `EnsembleModel` instances. Fully Bayesian Gaussian model
ensembles continue using normal samplers. Noisy acquisitions, pruning,
and pending prefixes retain a complete selected-member baseline/candidate
draw. Gaussian root caching does not apply to finite ensembles. Analytic
EI/PI, constrained moment formulas, posterior means, standard deviations,
and UCB use the ensemble moments, following the source. The multivariate
Gaussian approximation required by `qAnalyticProbabilityOfImprovement`
does not accept a finite-ensemble posterior, as in the source.

Input and outcome transforms, posterior output selection, model modes,
registered parameters, and forward hooks retain the Torch module
behavior. Tensor-valued observation noise is rejected; the Boolean
observation-noise flag is ignored, following `EnsembleModel.posterior`.

For batch-specific member weights, variance uses a separate denominator
`1 - sum(weights**2, dim=members)` in every model batch. The pinned
reference sums squared weights across all batches, producing infinite
variance for two batches each weighted `[0.5, 0.5, 0]`. The port uses the
independent weighted-variance law.

Identical ensemble members can place analytic acquisitions far into the
normal tail. Tensor LogEI now evaluates that tail using Laplace's
continued fraction for the Mills-ratio gap. This avoids the reference's
`erfcx`/`log1mexp` gradient cancellation while retaining differentiable
Tensor operations. LogPI and log feasibility use the port's stable
`log_ndtr` primitive rather than Torch's inaccurate far-tail backward.
Independent 80-digit controls verify the LogEI value and its first through
fourth derivatives at standardized improvements from `-6` to `-1e6`.

Focused CPU/CUDA float32/float64 checks cover source moment and sampling
parity, weighted value/parameter/query Hessians, explicit and default
sampling, RNG state, transforms, empty and singleton events, noisy
pruning, pending transitions, and scalar/hypervolume acquisition graphs.
Known defective source weighted-variance and extreme LogEI derivatives
are checked against independent mathematical controls.
