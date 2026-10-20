# Native acquisition parameter gradients

Ordinary CPU Torch `SingleTaskGP` models and unbatched Gaussian `ModelListGP`
containers now connect their actual persistent parameter leaves to these
acquisition values:

- Analytic `ExpectedImprovement`, `LogExpectedImprovement`,
  `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`,
  `UpperConfidenceBound`, `PosteriorMean`, and `PosteriorStandardDeviation`.
- Monte Carlo `qExpectedImprovement`, `qLogExpectedImprovement`,
  `qProbabilityOfImprovement`, `qSimpleRegret`, `qUpperConfidenceBound`,
  `qLowerConfidenceBound`, and `qPosteriorStandardDeviation`.

The implementation supports independent multioutput children through Gaussian
scalarization, different child training sets, repeated child/parameter aliases,
fixed or inferred noise, supported kernel trees, affine outcome transforms,
learned input warping, and single-output nonlinear outcome transforms. The
public acquisition constructor still rejects model batch shapes: this
checkpoint does not map copied ensemble member parameters into parent leaves.
Torch candidate batches remain independent and may have arbitrary upstream
loss weights. Frozen leaves and `torch.no_grad()` retain their normal meaning.

Each native forward pass records the underlying Gaussian mean/covariance
adjoints after analytic moment transformations or the MC objective, constraint,
reparameterization, and sample-transformation reverse pass. Backward contracts
those retained bars with the model's analytic full-conditional parameter VJP.
It does not approximate parameter derivatives with finite differences or
rerun the acquisition/objective. MC tapes include pending points; their
parameter contributions are retained while only candidate coordinates receive
query gradients. Direct objective derivatives with respect to candidate inputs
remain on the separate query graph.

The tape retains one immutable source and adjoint pair per candidate set.
Backward receives the actual upstream weight of each acquisition value, then
sums the resulting raw parameter contributions into the original parameter
layout. Independent output/list covariance blocks and Gaussian scalarization
are lifted before each model pullback. Aliased Torch leaves accumulate all
uses. Parameter-only and query-only graph branches are separate, so
`torch.autograd.grad(value.sum(), X)` does not run a parameter pullback.
Only first derivatives are supported.

Native acquisitions capture a model snapshot at construction. Forward and
`value_and_grad` now detect changed native model generations, source identities,
or leaf values and raise with an instruction to reconstruct the acquisition.
Rebuild after fitting, setters, training-data changes, or Torch optimizer steps.
Already-created backward graphs retain the original law even after such
changes. Private native calls bypass the public validation/synchronization
protocol. Native candidate optimization uses its original fixed acquisition
snapshot; construct that acquisition after synchronizing the model.

Noisy EI, knowledge-gradient, entropy, multiobjective, and acquisition-wrapper
parameter pullbacks are outside this checkpoint. When an unsupported native
family is used with persistent trainable leaves, requesting its parameter
backward raises an explicit `UnsupportedError`; its query-only derivative
interface remains available. Nonlinear moment-matched scalarization also
rejects parameter backward. Specialized model families retain their existing
parameter protocols. Learned-objective preference-model parameters are not
connected by the outcome-model tape, even when the objective's native sample
pullback supports outcome sensitivities.

Tests compare values, arbitrarily weighted model derivatives, and query
coordinates against official BoTorch `b0dd01be6` with `propagate_grads(True)`
and exact predictive covariance. That setting includes the training solve;
BoTorch's default detached predictive caches can yield different parameter
graphs. The mixed-multioutput list test uses the separately validated exact
conditional described in `joint-model-list-autograd.md`, avoiding the pinned
reference's defective lazy task-block slicing. Explicit shared normal draws
remove sampler-seed differences. The tests also cover minimizing analytic
objectives, Warp training-input chains, Log outcome chains, pending points,
constraints and direct-input objectives, frozen leaves, immutable old graphs,
stale-source errors, query-branch pruning, and no objective reevaluation.
