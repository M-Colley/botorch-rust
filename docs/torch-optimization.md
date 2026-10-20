# Device acquisition optimization

The Tensor optimization driver supports candidate `batch_limit` and
`max_optimization_problem_aggregation_size` as actual limits on groups of
restarts. Bounds and per-restart fixed values are sliced consistently;
`fixed_features={i: None}` retains the original frozen values. Torch
optimizers retain all candidate algebra on the device. SciPy transfers only
its explicit candidate/value/gradient interface, as before. Every group
receives a share of the remaining timeout budget.

Automatic box initialization and best-restart selection support model
batches. Boltzmann restarts are drawn independently per model batch, and
the final candidate is selected independently for each batch. Nonnegative
restart selection likewise applies its positive-quota/threshold law per
batch. Public `initialize_q_batch`, `initialize_q_batch_nonneg` and
`initialize_q_batch_topn` retain Tensor values/graphs on their device.
The public `gen_batch_initial_conditions` also accepts a Tensor raw-sample
generator and `fixed_X_fantasies`, without host inference conversions.

`optimize_acqf(..., acq_function_sequence=[...], sequential=True)` runs one
actual device acquisition per step. Each acquisition retains its own
original pending set, appends the previous selected candidates during its
step, and has its original state restored on success or failure. Joint
one-shot acquisitions and inter-point constraints remain excluded from
sequential mode, as in BoTorch.

The 88-case checkpoint combines new CPU/CUDA configuration tests, existing
Tensor optimizer/initializer tests and native optimization smoke tests.
It checks actual Torch/SciPy chunk sizes and convergence, fixed-value
mapping, model-batched initialization and best selection, sequence cleanup
on errors, public generators/fantasies, seeded reference laws and forbidden
host conversions. Targeted mypy and Ruff checks pass.

The pinned official reference is BoTorch
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. Batched Boltzmann parity uses the
category-axis standardization correction already proposed in
[PR 3390](https://github.com/meta-pytorch/botorch/pull/3390), matching the
existing native port. The unmodified reference standardizes across batches
and can sample nearly uniformly despite unequal values within a batch.
Scalar laws continue to match the unmodified reference and RNG.

Discrete enumeration, discrete local search, mixed continuous/discrete
optimization, acquisition lists and cyclic search now run device acquisitions
and Tensor candidate operations on CPU or CUDA. Enumeration chunks choice
scores, supports intra-point inequalities and avoidance, and selects choices
independently per model batch. Lattice local search evaluates one-coordinate
neighbors and supports supplied or generated lattice starts. Mixed search
optimizes each supplied fixed-feature combination, chooses the winning
combination, and uses greedy pending-point conditioning for q > 1. Cyclic
search reoptimizes each candidate against the others with a device exponential
moving-average stopping criterion. The explicit SciPy interface remains
available for the continuous subproblem; `gen_candidates_torch` keeps that
subproblem on the device.

Acquisition lists append prior selected candidates to each acquisition's own
original pending points and restore every original state on success or failure.
This deliberately follows the corrected law in
[PR 3395](https://github.com/meta-pytorch/botorch/pull/3395), rather than the
pinned reference's accidental reuse of the first acquisition's baseline.
The same correction applies to CPU native acquisition orchestration. Cyclic
search defaults to maximization, another PR 3395 correction; an explicit
`cyclic_options={"minimize": True}` still selects the requested stopping law.
Discrete local search restores pending state even when values are disabled.

The second checkpoint passes 230 tests across the new device search cases,
previous device optimizer/initializer configurations, native sequence and
optimization tests, and the phase-9 parity suite (two CPU-callback/CUDA cases
are skipped). Tests include real batched exact-GP choice selection, source
candidate/value comparisons, both CPU and CUDA, no host conversion guards,
actual mixed Torch/SciPy optimization, and error-state cleanup. Cyclic parity
uses the corrected maximization setting on both implementations; existing
criterion-view aliasing is isolated by cloning reference values as before.
Targeted mypy and Ruff checks pass.

A separate exception defect remains beyond PR 3395's success-only cleanup:
if the second list optimization raises, `optimize_acqf_list` leaves its
augmented pending points installed. A regression against unmodified pinned
BoTorch fails in all four ordinary/mixed and None/Tensor-pending cases.
A focused upstream `try/finally` repair and regression are prepared in
`D:/botorch-pr-list-pending`; publication is tracked centrally. PR 3469
repairs the separate `optimize_acqf` sequential dispatcher, so it does not
cover this list API failure path.

Automatic linear-constrained initialization now uses device affine-polytope
sampling. Intra-point constraints are repeated across q points, inter-point
indices are flattened jointly, and scalar fixed features are included as
exact equalities. An SVD nullspace removes equality constraints; Dykstra
projections find a feasible relative-interior point, with smaller margins
for narrow polytopes. Independent vectorized hit-and-run chains then draw
q-batches on the bounds' device. `n_burnin` and `n_thinning` control chain
transitions, including the reference's q multiplier for inter-point thinning.
The sampler shares the reference's uniform stationary law; its independent
chains and device RNG do not reproduce the serial CPU chain's draws.

Public `sample_q_batches_from_polytope` and constraint-transform helpers keep
Tensor constraints and points on their device. Bounds must be finite. The
polytope's affine hull must be represented by explicit equality constraints;
inconsistent equalities or failure to find a feasible relative interior raise
an explicit error. As with the reference's finite burn-in, uniformity is a
stationary-law guarantee, rather than an assertion of independent exact
uniform draws after arbitrary user-specified transition counts.

Sample-around-best initialization is also available on the device. It scores
baseline posterior means, handles scalar objectives/Pareto sets, output
constraint callables and analytic constraint dictionaries, and generates
bounds-truncated perturbations. The public perturbation helpers retain input
graphs and support repeatable seeded draws. Subset perturbations retain the
port's corrections: every perturbed coordinate uses the same selected center,
and `subset_sigma` actually controls subset noise. Around-best raw batches
are checked for parameter-constraint feasibility after fixed features are
applied, so infeasible extras cannot become initial conditions. Equality
constraints and custom raw generators exclude around-best mode, as in the
reference. Nonlinear constrained automatic initialization still requires a
user-supplied feasible initializer or initial conditions. Adam/SGD candidate
optimization remains box/fixed-feature constrained; general parameter
constraints use the explicit SLSQP interface.

The constrained/around-best checkpoint passes 185 targeted optimizer,
initializer, native optimization and phase-9 parity tests. Its focused
CPU/CUDA tests check independent simplex moments, equalities/inter-point
constraints, fixed and degenerate features, actual SLSQP convergence,
reference perturbation draws, gradient and gradient-gradient checks,
analytic constraint dictionaries, a single subset draw, repeatability and
forbidden host conversions. Narrow-polytope interior selection is tested
separately. Targeted mypy and Ruff checks pass.

Specialized KG starting-point heuristics, mixed alternating search and further
public initialization helpers are subsequent checkpoints. Local lattice
search retains the reference's unbatched, non-one-shot restriction. Sequential
mixed search retains the reference's inter-point-constraint restriction.
Forced parallel SciPy mode remains unsupported; this driver uses serial SciPy
calls for each restart group. No optimization speedup is claimed here.
