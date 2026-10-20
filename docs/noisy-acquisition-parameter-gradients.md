# Noisy acquisition parameter gradients and cached constants

`qNoisyExpectedImprovement`, `qLogNoisyExpectedImprovement`, and their
`qLogNParEGO` scalarization wrapper expose native first-order model-parameter tapes for the ordinary unbatched CPU Torch
GP sources supported by `acquisition-parameter-gradients.md`. This includes
Gaussian scalarization, independent-output model lists and aliases, custom
objectives, candidate constraints, learned input warping, single-output
nonlinear outcome transformations, pending points, and qLogNEI incremental
or non-incremental pending handling. Stochastic or batched learned objectives
retain the existing joint-sampling path; the tape differentiates the outcome
model, without adding a preference-model parameter graph.

The derivative contract follows the actual official BoTorch graph, which
intentionally holds some construction-time quantities constant:

- Baseline incumbents are detached in both cached and joint modes, including
  constrained incumbents and inferred infeasible lower bounds.
- A successful cached low-rank update holds its saved baseline factor `L_B`
  constant. Its live candidate mean, candidate-to-baseline covariance, and
  candidate covariance retain the full native training-solve derivative.
- An uncached evaluation or actual cached-root failure fallback differentiates
  the joint covariance factor used by that evaluation, with the incumbent
  still detached. Its baseline covariance adjoint can therefore be nonzero.
- Baseline pruning selects a fixed stored point set. The discrete selection
  itself is not differentiated.

These conventions mean equal cached and joint values can have different
parameter gradients. This feature does **not** claim full baseline-cache
hypergradients or the derivative obtained by rebuilding all baseline caches
and incumbents after every perturbed parameter value.

For one independent output, cached sampling uses
`T = C_XB L_B^{-T}`, `R = chol(C_XX - T T^T)`, and
`Y_X = mu_X + T z_B + R z_X`. The retained objective/constraint reverse pass
computes `mu_X_bar`, `C_XB_bar`, and `C_XX_bar`, treating `L_B` as constant.
Only those compact blocks are stored. The baseline-to-baseline block stays
zero and a full Gaussian adjoint is materialized only when parameter backward
is requested. The model's analytic VJP then includes the dependence of live
cross blocks on baseline input transformations and the training solve.

Joint tapes retain the existing sample-factor adjoints instead. The exact path
selected by the original forward is retained; backward does not resample,
recompute the objective, request another posterior, or try a different root.
Saved normal layouts remain point-major (`point * outputs + output`) at the
sampler boundary and output-major within each independent Gaussian block.
Cached baseline columns come from the baseline's original normal matrix;
candidate columns come from the baseline-adjusted joint normal matrix.

The native query-only cache path continues to compute only candidate blocks
from cached GP training solves. Recording a cached tape adds no baseline-sized
quadratic allocation. A counted-model regression verifies that both recorded
cached evaluation and parameter contraction request no additional model
posterior or full query VJP. Existing public source-generation and leaf-value
validation requires acquisition reconstruction after model changes. It now
also detects changes in parameter-boundary metadata, such as replacing public
modules with a shared binding. Old backward graphs retain their original law.
Sampler or pending edits after a forward similarly do not change its tape.

Validation uses official BoTorch `b0dd01be6`, shared explicit normal draws,
`propagate_grads(True)`, and exact predictive covariance. Official baseline
construction populates predictive caches under `no_grad`; tests clear those
model caches before the live forward, preserving the acquisition's saved
baseline factor and incumbent. This isolates the intended full live
training-solve derivative from incidental detached predictive caches.
Independent directional finite differences hold the saved baseline factor
and incumbent fixed while updating live GP cross/candidate solves. A forced
fallback test uses a live joint factor with the same saved incumbent, and
checks that its parameter graph differs from the successful cached path.

`qLogNParEGO` delegates directly to its saved qLogNEI backend. Its existing
Chebyshev objective pullback carries sample and direct-input bars through the
fixed constructor scalarization. Weights and posterior-mean baseline
normalization stay constant, matching official construction under `no_grad`.
Tests verify cached/joint parameter and query graphs with signed loss weights,
negative scalarization weights, transformed multioutput objectives, constraints,
and both incremental pending modes. No separate model or objective evaluation
is added by the delegate.

Other native multiobjective acquisitions and general acquisition wrappers
retain explicit unsupported parameter backward; their query derivatives
continue to work. Eigen-root derivatives and specialized model parameter
layouts retain their existing explicit unsupported boundaries.
