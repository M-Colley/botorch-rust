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
A focused upstream `try/finally` repair and regression now extend
[PR #3395](https://github.com/meta-pytorch/botorch/pull/3395). The complete list
test class passes seven tests and four new exception subtests. PR 3469
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

One-shot KG and HVKG now select their specialized default initializers.
`gen_one_shot_kg_initial_conditions` first scores augmented raw starts, then
replaces the prescribed fantasy tail with Boltzmann draws from optimized
current-model decisions. A zero-length optimized tail leaves the random
starts intact, following the existing PR 3395 repair for a single fantasy.
`gen_one_shot_hvkg_initial_conditions` optimizes current-model hypervolume
sets, fills the prescribed restart quota with sampled sets, and scores
candidate starts conditional on each fixed fantasy set. Multi-fidelity
HVKG optimizes full physical coordinates with target fidelity features fixed.
`gen_value_function_initial_conditions` mixes sampled current-model optima
with Sobol candidates and selects starts independently for each fantasy/model
batch. Scalar and model-batched KG, batched HVKG and conditional fantasy
layouts are tested. Fixed fantasy inputs may carry batch dimensions that
broadcast to the acquisition's batch shape.

The inner problem uses the port's actual value acquisition and device Torch
candidate optimizer by default. Specified SciPy solver options or linear
constraints select the explicit SciPy/SLSQP interface; inference remains on
the model's device. Inner restart/sample counts, optimizer settings, random
fraction, Boltzmann temperature and seeded draws are honored. The device
solver can converge to different optima from reference SciPy; no identical
optimization trajectory is claimed. Independent formula tests synchronize
raw and optimized decisions to compare the exact KG tail law to the pinned
reference, and separate tests run genuine inner optimization and high-level
default dispatch. Batched weighting follows PR 3390's category-axis correction.
The KG initializer checkpoint passes 140 CPU/CUDA KG/HVKG, optimizer and
initializer tests, including no-host-transfer guards; mypy and Ruff pass.

`optimize_acqf_mixed_alternating` and its public building blocks now support
device Tensor domains. Random, equally-spaced binary and continuously-relaxed
starts are implemented; ordinal/categorical nearest neighbors, spray points,
feasible-point sampling and Boltzmann restart selection retain candidate data
on the device. The search alternates feasible discrete/categorical moves and
actual continuous optimization, with independent optimizer state per restart.
Large unconstrained ordinal domains use continuous relaxation and rounding;
dimensions participating in constraints remain discrete. Fixed features,
intra-point inequalities, continuous equalities, q > 1 pending conditioning,
scoring limits and postprocessing are honored. General linear constraints
select the explicit SLSQP interface; box/fixed-feature continuous steps use
Torch L-BFGS. Postprocessing must retain a feasible in-bounds candidate in its
specified discrete domain. The reference algorithm's unbatched scalar,
non-one-shot restriction remains explicit.

Bounded Torch L-BFGS now projects its candidate state before each fresh closure
and uses the physical query gradient, masking outward motion at active bounds.
This corrects a port defect: for one restart x=0.2, objective -(x-0.73)^2 and
bounds [0,1], an unconstrained line-search trial can exceed 1. Differentiating
its clamp reports zero and falsely accepts x=1. The projected closure returns
x=0.73 and still recognizes genuine boundary optima. Mixed continuous steps
honor independent restart aggregation rather than coupling their L-BFGS
histories. Candidate optimization does not accumulate model parameter grads.

The mixed checkpoint passes 34 focused CPU/CUDA tests, including independent
reference building blocks, real GP candidate/value parity, all three start
strategies, constrained optimization, relaxation, fixed features, no-host
conversion guards, pending rollback, postprocessing rejection, and twelve
single/multiple-restart interior/boundary L-BFGS cases. The broader mixed,
discrete, polytope, KG, native and phase-9 parity checkpoint passes 226 tests
with two CPU-callback/CUDA exclusions. Targeted mypy and Ruff checks pass.

The pinned source also leaves augmented pending points installed if a later
`optimize_acqf_mixed_alternating` initialization fails. Two source regressions
(None and Tensor pending states) fail on unchanged b0dd01be6; a `try/finally`
repair makes them pass alongside existing value/retry tests. The patch extends
the existing [PR 3395](https://github.com/meta-pytorch/botorch/pull/3395). The native Rust
alternating optimizer already restores pending state before propagating an
error; the device implementation does likewise.

Device homotopy optimization now executes real scheduled solves, distance
pruning, greedy pending updates and the final solve without flattening tensor
candidates or values. Plain, enumerated mixed and alternating mixed domains
are tested. Each solve starts from independent decision leaves; a graph-attached
previous stage does not become a non-leaf optimizer parameter. Homotopy always
restores its original parameter values and pending points, including failed
stages and postprocessing. Cleanup failures do not replace an already-active
optimization exception. Postprocessing is checked against bounds, fixed
features, constraints and enumerated domains before a candidate is returned.
The reference's unbatched scalar, non-one-shot pruning restriction remains
explicit. General constraints use the explicit SciPy interface.

Public `columnwise_clamp`, `fix_features`, `get_best_candidates`,
`evaluate_feasibility` and `nonlinear_constraint_is_feasible` now retain tensor
device and selected graphs. Fixed values can carry restart/model batches;
`None` freezes just its column. Linear intra-point/inter-point constraints,
equalities, scalar nonlinear callbacks, dtype-specific tolerances and empty
batches are tested. Constraint transforms also retain the existing array API's
integer-valued float-index compatibility, rejecting fractional indices.
Native acquisition callers using an explicit Torch optimizer reuse the fresh
closure and detached-start loop, retaining their list/NumPy/Torch output kind;
the default native Adam path remains Rust.

This checkpoint passes 46 focused CPU/CUDA tests plus 160 existing optimizer,
constraint and phase-9 parity tests. Independent source formulas, first and
second derivatives, source stage/value parity, genuine optimizations and
forbidden-host-conversion guards are included. Targeted mypy and Ruff pass.

Two independently reproduced source defects are prepared for upstream review
against official b0dd01be6. `generation/gen.py` makes graph-attached initial
conditions optimizer parameters without detaching, so Adam raises
`ValueError: can't optimize a non-leaf Tensor`, including consecutive homotopy
stages. Detaching the new decision variables fixes all four CPU/CUDA and
fixed/free regression cases and repeated stages; the patch is
`D:/botorch-pr-generator-leaf`, commit `e71f02396`. Port stage-parity tests
explicitly use this corrected reference generator. Separately,
`optimize_acqf_homotopy` only restores parameters and pending points after
successful optimization. All four q=1/q=2 and None/Tensor pending regression
cases fail on the pinned source. The prepared finally-based repair in
`D:/botorch-pr-homotopy-rollback`, commit `08d1bb9ab`, passes the 13 source
homotopy tests and four regression subcases. These are source patches prepared
for publication; they are not claims that official main already contains them.

Further public optimization helpers are being audited. Local lattice search
retains the reference's unbatched, non-one-shot restriction. Sequential mixed
search retains the reference's inter-point-constraint restriction. Forced
parallel SciPy mode remains unsupported; this driver uses serial SciPy calls
for each restart group. No optimization speedup is claimed here.
