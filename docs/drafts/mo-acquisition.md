# Multi-objective acquisition functions: README additions (draft)

Draft text for `README.md`, to be integrated by the maintainer (branch
`mo/acquisition`). Measured on Windows 11 (Intel Core Ultra 7 155H, 22 logical
CPUs) with Python 3.13.12, PyTorch 2.13.0 (CPU), SciPy 1.18.0, GPyTorch
1.15.2, linear_operator 0.6.1 and BoTorch main at da00e04; "BoTorch with
#3392" is main plus the commits of meta-pytorch/botorch#3392 (012d077).

## Supported API: rows to add

| BoTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.acquisition.multi_objective.qLogNoisyExpectedHypervolumeImprovement`, `qNoisyExpectedHypervolumeImprovement` | `botorch_rs.acquisition.multi_objective.*` | `ref_point`, `X_baseline`, `sampler` (default: 128 Sobol samples), `objective`, outcome `constraints` and `eta` (a float or one per constraint), `X_pending` / `set_X_pending` (cached in the box decompositions as in BoTorch: `cache_pending`, `max_iep`, `incremental_nehvi`), `prune_baseline`, `alpha`, `cache_root` (default `True`: BoTorch's low-rank update of the cached root of the baseline covariance; `False`: the joint posterior), `fat`, `tau_relu`, `tau_max`; the state `X_baseline`, `cell_lower_bounds` / `cell_upper_bounds` (one decomposition per MC sample), `_hypervolumes`, `_baseline_hvs`, `_prev_nehvi` |
| `qLogExpectedHypervolumeImprovement`, `qExpectedHypervolumeImprovement` | `botorch_rs.acquisition.multi_objective.*` | `ref_point`, `partitioning` (any unbatched box decomposition of botorch_rs or BoTorch), `sampler`, `objective`, `constraints`, `eta`, `X_pending`, `fat`, `tau_relu`, `tau_max`; `cell_lower_bounds`, `cell_upper_bounds` |
| `ExpectedHypervolumeImprovement` | `botorch_rs.acquisition.multi_objective.ExpectedHypervolumeImprovement` | analytic, `q = 1`, independent outputs |
| `qLogNParEGO` | `botorch_rs.acquisition.multi_objective.qLogNParEGO` | `scalarization_weights` (or random weights, see the migration notes), `sampler`, `objective`, `constraints`, `eta`, `X_pending`, `fat`, `prune_baseline`, `tau_relu`, `tau_max`, `incremental`; `scalarization_weights`, `Y_baseline`, `chebyshev_scalarization` |
| `MCMultiOutputObjective`, `IdentityMCMultiOutputObjective`, `WeightedMCMultiOutputObjective`, `GenericMCMultiOutputObjective`, `FeasibilityWeightedMCMultiOutputObjective` | `botorch_rs.acquisition.multi_objective.objective.*` | identity (`outcomes`, `num_outcomes`), weighted and feasibility-weighted objectives are evaluated in Rust; generic objectives and subclasses are callables on torch tensors, evaluated from Rust with gradients from `torch.autograd` (requires torch); BoTorch's own objective objects are accepted too |
| `botorch.acquisition.multi_objective.utils.get_default_partitioning_alpha`, `prune_inferior_points_multi_objective` | same paths | `objective`, `constraints`, `num_samples`, `max_frac`, and a keyword-only `sampler` (a botorch_rs extension) |
| `botorch.utils.multi_objective.hypervolume.compute_subset_indices`, `SubsetIndexCachingMixin`, `NoisyExpectedHypervolumeMixin` | same path | |
| `botorch.optim.optimize.optimize_acqf_list` | `botorch_rs.optim.optimize.optimize_acqf_list` | sequential greedy optimization for `q = 1` per acquisition function with the previous candidates as `X_pending`; `fixed_features` / `fixed_features_list` (BoTorch's `optimize_acqf_mixed` per step), parameter constraints, `options`, `ic_generator`, `return_acq_values` |
| `acqf(X)`, `torch.autograd.grad(acqf(X), X)`, `acqf.value_and_grad(X)` | same | exact gradients of all multi-objective acquisition functions, computed in Rust (reverse mode through the inclusion–exclusion, the smoothing, the objectives, the constraints, the low-rank sample update and the posterior) |

In Rust: `botorch_core::acquisition::multi_objective` (the acquisition
functions, objectives and utilities) and `botorch_core::optim::optimize_acqf_list`.
`cargo run --release --example mobo_loop` runs the qLogNEHVI tutorial loop.

## What is not ported yet: replace the multi-objective bullet

- **Multi-objective optimization:** the acquisition functions qLogNEHVI,
  qLogEHVI, qNEHVI, qEHVI, the analytic EHVI and qLogNParEGO, their
  objectives and utilities, `optimize_acqf_list`, the Pareto, hypervolume,
  box-decomposition and scalarization utilities and the test problems are
  ported; not yet:
  - `qHypervolumeKnowledgeGradient`, `MOMF`, multi-objective entropy search
    (`qMultiObjectiveMaxValueEntropy`, `qLowerBoundMultiObjectiveMaxValueEntropySearch`,
    the multi-objective JES and PES) and multi-objective risk measures
    (`MARS`, `MVaR`, ...);
  - `compute_sample_box_decomposition`, `random_search_optimizer` and
    `sample_optimal_points` of `acquisition.multi_objective.utils` (used by
    the entropy-search acquisition functions);
  - `marginalize_dim` (fully Bayesian models), per-constraint `fat` lists,
    posterior transforms in the multi-objective acquisition functions, and
    batched models or batched box decompositions;
  - `optimize_with_nsgaii` (NSGA-II via pymoo).

With these, botorch-rust provides **165 of the 773** public classes and
functions of BoTorch (21%, was 146); in the coverage table,
`acquisition.multi_objective` becomes 15 of 33 (45%), `optim` 6 of 58 (10%)
and `utils.multi_objective` 19 of 22 (86%) (regenerate `docs/api_coverage.md`
with `scripts/api_coverage.py`). Methods still missing on the ported classes:
`psi` and `nu` of the analytic EHVI, `get_posterior_samples` and
`sample_shape` of the MC acquisition functions (as for the single-objective
ones), and qLogNParEGO's `compute_best_f`.

In the Development section's list of fixture phases, add `--phase 5b` (and
`--phase 5b-3392`, with a BoTorch that includes meta-pytorch/botorch#3392)
for the multi-objective acquisition functions.

## Equivalence with BoTorch: additions

In the list of fixture phases: `--phase 5b` multi-objective acquisition
functions (`--phase 5b-3392` writes the cases that need
meta-pytorch/botorch#3392 and runs only with a BoTorch that includes it). In
the list of Python test modules: `test_parity_mo_acquisition.py`
(multi-objective acquisition functions, objectives, pruning,
`optimize_acqf_list`) and `test_equivalence_multi_objective.py` (randomized,
`-m equivalence`); `test_mobo_loop.py` compares whole optimization loops.

### Multi-objective acquisition functions

The acquisition functions are compared with identical Monte-Carlo base
samples (every draw of BoTorch's samplers is recorded or replaced) on
multi-output `SingleTaskGP`s with m = 2, 3, 4 outputs and `ModelListGP`s of
single- and multi-output models, for q = 1, 2, 3, t-batches, pending points
(at construction, cached or not, appended, replaced and cleared), outcome
constraints with `eta`, `fat`, `tau_relu` / `tau_max`, all objectives,
exact and approximate (`alpha`) partitionings, `cache_root` true and false,
incremental and non-incremental NEHVI and pruned baselines.

- **Rust fixtures** (`botorch_core/tests/mo_acquisition_parity.rs`, 4 models,
  27 qEHVI / qLogEHVI, 55 qNEHVI / qLogNEHVI, 8 analytic EHVI, 16 qLogNParEGO
  and 16 pruning cases, plus 12 `set_X_pending` sequences of five steps
  against BoTorch with #3392).
- **Python parity tests** (`test_parity_mo_acquisition.py`, 135 tests).
- **Randomized suite** (`test_equivalence_multi_objective.py`, 143 tests, 13
  scenarios: n ∈ {6, 10, 14}, d ∈ {1, …, 4}, m ∈ {2, 3, 4}, inferred or fixed
  noise, with and without `Normalize` / `Standardize`, RBF or Matérn-5/2, three
  input ranges, three `ModelListGP`s; random reference points, fronts, q,
  pending points, objectives, constraints, `eta`, `fat`, `alpha`,
  `cache_root`, `incremental_nehvi`, `max_iep`).

| Quantity | Tolerance | Rust fixtures | Python parity | Randomized suite |
|---|---|---|---|---|
| Multi-output objectives (identity with `outcomes`, weighted, generic, feasibility-weighted), infeasible costs | 1e-12 (relative; 1e-14 in the parity tests) | 4.1e-14, 2.0e-14 | exact, exact | 4.6e-15 |
| qEHVI values / gradients | 1e-9 / 1e-8 | 9.7e-15 / 2.9e-14 | 1.5e-14 / 5.4e-14 | 1.4e-13 / 9.2e-12 |
| qLogEHVI values / gradients | 1e-9 (absolute below 1) / 1e-8 | 4.5e-14 / 8.6e-14 | 9.8e-15 / 5.0e-14 | 2.6e-14 / 6.0e-13 |
| qNEHVI values / gradients | 1e-9 / 1e-8, or 10× BoTorch's noise floor if larger (errors scaled to the nominal tolerance) | 1.3e-12 / 9.5e-12 | 6.8e-11 / 5.0e-9 | 9.6e-10 / 4.7e-9 |
| qLogNEHVI values / gradients | as above | 3.6e-13 / 3.6e-12 | 3.7e-12 / 4.9e-10 | 3.5e-10 / 5.9e-9 |
| qNEHVI / qLogNEHVI after `set_X_pending` calls that cache pending points (BoTorch with #3392): values / gradients | 1e-9 / 1e-8 | 1.3e-12 / 9.5e-12 (see above) | 2.7e-12 / 2.6e-12 | 3.7e-10 / 2.6e-9 |
| ... non-incremental qLogNEHVI there: `exp(value)` relative to the hypervolume / gradients of `exp(value)` | 1e-12 / 1e-8 | 7.6e-16 / 1.4e-14 | 4.1e-13 / 1.8e-12 | 9.6e-15 / 2.1e-12 |
| NEHVI state: cached baselines, numbers of cells / per-sample cell bounds and hypervolumes / improvement of cached pending points (relative to the hypervolume) | identical / 1e-9 / 1e-12, or 10× BoTorch's noise floor | identical / 2.4e-14 / 7.6e-16 | identical / 1.5e-13 / exact | identical (one decomposition differing at a near tie within the noise floor, with agreeing hypervolumes) / 6.4e-10 / 4.2e-13 |
| Analytic EHVI values / gradients | 1e-9 / 1e-8 | 6.1e-15 / 7.4e-15 | 3.1e-15 / 2.4e-15 | 6.6e-15 / 4.8e-15 |
| qLogNParEGO values / gradients (and `Y_baseline`) | 1e-9 / 1e-8, or 10× the noise floor (1e-12) | 3.2e-13 / 1.4e-11 (4.7e-15) | 4.4e-11 / 2.6e-10 (1.2e-15) | 2.8e-10 / 7.0e-9 (1.0e-15) |
| Box decompositions of qEHVI / qLogEHVI (numbers of cells, cell bounds) | identical / 1e-12 | identical / exact | identical / exact | identical / exact |
| `prune_inferior_points_multi_objective`, `compute_subset_indices`, `get_default_partitioning_alpha` | identical | identical | identical | identical |
| Unscaled errors of all evaluations of the noisy acquisition functions (values / gradients) | | | | 6.4e-3 / 2.0e-2 (1-D scenarios; BoTorch's own change there: up to 3.4e-2 / 7.8e-2) |
| NEHVI evaluations on numerically singular joint posteriors (jitter-dependent) | finite, 0.1 | | | 4 evaluations, 4.0e-3 |

Independent of BoTorch, qEHVI equals the average over its posterior samples
of the exact hypervolume improvement (`Hypervolume` of the front with the
sample minus that of the front; worst 1.8e-15 relative to the hypervolume),
the analytic EHVI agrees with qEHVI over 16,384 quasi-random samples (worst
5.5e-5 of the largest value), qLogEHVI converges to `log qEHVI` as `tau_relu =
tau_max → 0` (2.7e-11 relative at 1e-6), gradients agree with finite
differences, the `cache_root` and joint computations agree, caching pending
points equals a fresh
acquisition function with the enlarged baseline, results do not depend on
the number of threads (bitwise), and a `ModelListGP` of the outputs gives the
values of the equivalent multi-output model.

Noise floors. The joint posteriors of the noisy acquisition functions over
their baseline (the training points) are nearly singular: under a relative
perturbation of the training inputs by 1e-14, BoTorch's own NEHVI values and
gradients change by up to 3.9e-11 on the fixture models and 1.1e-11 / 2.3e-9
on the models of the parity tests. In the 1-D scenarios of the randomized
suite (6 to 14 points on a line) they change by up to 6.4e-3 / 2.0e-2 (NEHVI)
and 3.4e-2 / 7.8e-2 (qLogNParEGO, which samples the joint posterior of the
baseline and the candidates); there the tolerance is ten times that change.
Where a covariance block is numerically singular (smallest eigenvalue below
1e-12 of the largest), whether Cholesky needs jitter depends on rounding: 4
evaluations, all in 1-D scenarios after `set_X_pending`, agree to 4.0e-3.

From identical initial conditions, every restart of `optimize_acqf`
(qLogNEHVI, L-BFGS-B) reaches BoTorch's value, sequential optimization
agrees to 2.1e-14 in its first step (3.1e-10 over both steps with #3392,
where BoTorch main caches the first candidate with redrawn base samples), and
`optimize_acqf_list` agrees to 4.7e-14 (qLogNParEGO); a candidate that
lands on a pending point makes the joint covariance singular, and its value
then depends on jitter (3.7e-2). `test_mobo_loop.py` compares whole
qLogNEHVI loops (6 iterations) on BraninCurrin from the same initial
designs. Single runs differ because the optimizers draw different initial
conditions, by 5.7% of the maximum hypervolume (standard deviation, with
heavy tails), so the CI test only guards against gross failures (median
shortfall over four designs at most 5%; observed 0.65% against main, 1.9%
with #3392, 0 against 0.18.1). Over 24 designs the loops are statistically
indistinguishable: final hypervolumes 50.54 (botorch-rust) and 49.90
(BoTorch) on average, medians 50.85 and 50.69, shortfall of botorch-rust
−1.1% ± 1.2% (mean ± standard error).

## Migration notes: additions

- **Multi-objective acquisition functions.**
  - **Cached pending points** (meta-pytorch/botorch#3392). When qNEHVI or
    qLogNEHVI caches pending points in its box decompositions, botorch-rust
    keeps the base samples of the previously decomposed points and draws
    only the new ones. BoTorch main redraws all of them, so after
    `set_X_pending` the cached hypervolumes (and the baseline hypervolumes of
    non-incremental NEHVI) use different samples of the same observed
    points, and the improvement of the pending points mixes sampling noise
    into the result. For example, a pending point equal to a baseline point
    improves the hypervolume by 3.6e-3 on average in BoTorch main (by 1e-15
    with #3392; 16 samples, two outputs), and the log value of a candidate
    that adds nothing either rises from -15.68 to -5.63. In the parity tests'
    16-sample cases, botorch-rust's values differ from BoTorch main's by
    0.5–1.1 (relative, qNEHVI) and up to 42 (log values, qLogNEHVI). These
    cases agree with BoTorch including #3392 (Rust fixtures and Python
    tests).
  - **Feasibility weighting** (meta-pytorch/botorch#3392). With several
    `constraint_idcs`, `FeasibilityWeightedMCMultiOutputObjective` weights by
    every constraint; BoTorch main evaluates the last index for every
    constraint (a late-binding closure). With one constraint the results are
    identical.
  - **`cache_root`.** The default (`True`) is BoTorch's computation: new
    points are sampled from the low-rank update `[Σ_XB L_B⁻ᵀ, chol(Σ_XX −
    Σ_XB Σ_B⁻¹ Σ_BX)]` of the cached root `L_B` of every output's baseline
    covariance, with BoTorch's jitter retries and the same fallback to the
    joint posterior; its cost grows as O(N² q) instead of O((N + q)³).
    `cache_root=False` samples the joint posterior. Both agree with BoTorch's
    corresponding paths and with each other to rounding unless the joint
    covariance needs jitter.
  - **Random numbers.** `prune_baseline` prunes with 2048 samples seeded from
    the sampler's seed (BoTorch draws a random seed), and qLogNParEGO without
    `scalarization_weights` draws its weights with `sample_simplex` from
    botorch-rust's generator seeded from the sampler's seed (BoTorch uses
    torch's global generator); explicit weights reproduce BoTorch exactly.
    `prune_inferior_points_multi_objective(..., sampler=...)` accepts an
    explicit sampler.
  - **qLogNParEGO** is qLogNEI of the Chebyshev scalarization, computed like
    BoTorch's `cache_root=False` (as botorch-rust's qLogNEI); `cache_root` is
    accepted and ignored.
  - **Non-incremental qLogNEHVI** returns `log(NEHVI + prev_nehvi)`, where
    `prev_nehvi`, the improvement of cached pending points, is a mean of
    differences of hypervolumes and exact only to their rounding (about 1e-15
    of the hypervolume). If the pending points add nothing, both libraries
    get 0 or about 1e-15, and where the candidates add nothing either (log
    improvements below about -34 plus the log hypervolume), the logarithm
    turns that into differences of order one in the log value.
    This is a numerical property of BoTorch's formula, not of either
    implementation; the tests compare these values in linear space.
  - **BoTorch's fused kernel.** BoTorch main can compile a fused C++ kernel
    for the log hypervolume improvement (`fat=True` on the CPU, if a C++
    compiler is available). It evaluates the same formulas with reassociated
    arithmetic; botorch-rust follows the pure-Python path, which BoTorch uses
    when the kernel is not available (as on the machine of these
    measurements).
  - **Not supported** (each raises `UnsupportedError`): per-constraint `fat`
    lists (BoTorch's multi-objective acquisition functions use the truth
    value of `fat` for all smoothings), posterior transforms, `marginalize_dim`,
    batched models and batched box decompositions.
  - Default sample shapes (128), legacy warnings (`NumericsWarning` for qEHVI
    and qNEHVI), argument validation and error messages follow BoTorch.
- **`optimize_acqf_list`.** BoTorch 0.18.1 appends pending points in
  `set_X_pending` and optimizes all L-BFGS-B restarts jointly (SciPy ≥
  1.16), so its sequential candidates can differ from main's; botorch-rust
  follows main.

## Benchmarks: rows to add

| Task | botorch-rust, 1 thread | BoTorch, 1 thread | speedup | botorch-rust, 22 threads | BoTorch, 16 threads | speedup |
|---|---|---|---|---|---|---|
| MOBO loop, qLogNEHVI, 20 iterations | 10.36 s | 120.78 s | 12× | 2.54 s | 91.00 s | 36× |
| optimize_acqf, qLogNEHVI, m = 2, q = 2 | 295.8 ms | 2.83 s | 9.6× | 70.5 ms | 2.63 s | 37× |
| optimize_acqf, qLogEHVI, m = 3, q = 2 | 1.31 s | 11.39 s | 8.7× | 186.1 ms | 5.29 s | 28× |

Tasks:

- **MOBO loop:** the loop of `examples/mobo_loop.rs` (BoTorch's qLogNEHVI
  tutorial on BraninCurrin): 6 Sobol points, then 20 iterations, each
  fitting a two-output `SingleTaskGP` with `Normalize` and optimizing
  qLogNEHVI (128 Sobol samples, pruned baseline) for q = 2 candidates with
  `sequential=True`, 10 restarts and 512 raw samples. The result is the final
  hypervolume (maximum 59.41).
- **`optimize_acqf`, qLogNEHVI and qLogEHVI:** 10 restarts and 512 raw
  samples on 30-point DTLZ2 models (d = 6, fixed hyperparameters):
  qLogNEHVI with m = 2 and q = 2 (`sequential=True`, 128 Sobol samples, so
  the libraries' base samples differ), qLogEHVI with m = 3 and q = 2 (the same
  128 explicit base samples in both libraries; the partitioning of the
  posterior mean at the training points).

| Task | botorch-rust | BoTorch |
|---|---|---|
| MOBO loop, qLogNEHVI, 20 iterations | 57.822098 | 57.934464 |
| optimize_acqf, qLogNEHVI, m = 2, q = 2 | -4.633851 | -4.555190 |
| optimize_acqf, qLogEHVI, m = 3, q = 2 | -1.493424 | -1.492716 |

The loop reaches 97% of the maximum hypervolume in both libraries (a single
design; see above for the comparison over designs). qLogNEHVI differs
because the libraries' Sobol sequences differ (sum of the two sequential
acquisition values). qLogEHVI uses the same base samples; the optima
differ by 7e-4 because the random initial conditions lead to different
local optima. The multi-objective tasks were timed once, on the same
machine as the table above (`python scripts/benchmark.py -k HVI`), while
other jobs ran on it (for the default numbers of threads, a test suite at
low priority on two threads).
