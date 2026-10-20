# Device restart initialization and optimizer checks

The Torch acquisition backend generates raw Sobol candidates and selects
restarts using the pinned BoTorch scalar heuristics. Standard initialization
uses standardized Boltzmann weights and ensures the maximum is included.
Nonnegative initialization retains the positive quota, reduces the alpha
threshold until enough points qualify, and samples relative-value weights.
`topn=True` honors `largest` and `sorted`; `largest` defaults to the acquisition's
`maximize` flag. Constant or unusable scores trigger random selection and the
reference's four increasing raw-sample attempts.

`options['seed']` seeds both Sobol generation and a device-local selection
generator. This makes the complete initializer reproducible without changing
the caller's global selection RNG. The reference's standalone selection
functions use the global generator; exact RNG parity tests seed that generator
to the same state. Inference scores, selection weights and candidates remain
on device. Only integer counts and branch decisions are read as scalars.
`init_batch_limit` chunks raw scoring. Automatic initialization requires one
scalar acquisition value per raw candidate; constrained initialization still
requires feasible initial conditions or a custom generator.

The optimizer now rejects unsupported candidate `batch_limit`, forced parallel
SciPy mode, aggregation limits and extra initializer keywords without a custom
generator. Inter-point constraints are rejected before sequential q=1
optimization. Supported initialization options are removed before candidate
optimizer dispatch; unsupported specialized initialization options raise errors.

Postprocessing must retain candidate shape, dtype and device. Bounds, fixed
features and linear/nonlinear feasibility are checked again using the port's
dtype-specific constraint tolerance. An infeasible postprocessed candidate raises
`CandidateGenerationError`; this checkpoint does not implement feasibility
projection or repair. Optimization warnings trigger one regenerated-IC retry
when enabled, with the seed advanced and the remaining candidate-generation
time budget. Supplied initial conditions retain the warning without regeneration.

LBFGS receives a closure that evaluates the current projected candidates and
their actual derivative at every invocation, allowing a real line search.
GP parameter `.grad` fields remain untouched. This differs deliberately from
the pinned reference's `assign_grad` closure, which returns a loss and gradient
captured before the optimizer step; Adam and SGD retain their previous path.

Validation covers CPU and an NVIDIA RTX 4050 Laptop GPU: exact standalone
reference restart choices with synchronized RNG, positive quotas, alpha
relaxation, tie/nonfinite fallbacks, top-n ordering, raw-scoring chunks with
forbidden host data conversions, LBFGS quadratic/GP searches, postprocessing
feasibility, warning retries and unsupported-option errors. The focused
initializer/optimizer run passes 62 tests, including the existing 18 optimizer
regressions. Reference revision: `b0dd01be61d517c387e234903a6e2cc38c057ea2`.
