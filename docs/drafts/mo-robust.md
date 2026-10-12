# Robust multi-objective optimization, MOMF and NSGA-II (draft)

Draft documentation of the robust and multi-fidelity multi-objective ports
(branch `mo/robust-momf-nsga2`), to be merged into the README and
`docs/api_coverage.md`. Reference: BoTorch main `da00e0401`; the live tests
also run against the released BoTorch 0.18.1, whose modules of these features
are identical to main.

## Supported API

| BoTorch | botorch_rs (Python) | botorch_core (Rust) | Notes |
|---|---|---|---|
| `models.transforms.input.InputPerturbation` | same path | `transforms::input::InputPerturbation` | `perturbation_set` as an `n_w x k` array or a callable on torch tensors (heteroscedastic noise; `n x k` points to `n x n_w x k`), `bounds`, `indices` (negative indices count from the end), `multiplicative`, `transform_on_eval`, `transform_on_fantasize` (recorded); `transform`, `forward` (train/eval flags); `n_w` (botorch_rs extension). `transform_on_train=True` is rejected by the models. |
| `models.transforms.input.ChainedInputTransform` | same path | `transforms::input::ChainedInputTransform` | chains of `Normalize` (without a batch shape; learned bounds are fitted to the inputs as transformed by the preceding train-mode steps) and `InputPerturbation`, in any order; mapping interface (`[]`, `keys`, `values`, `items`, iteration). |
| `SingleTaskGP(..., input_transform=...)`, `ModelListGP` | same paths | `SingleTaskGPOptions::input_chain`, `BatchedMultiOutputGPOptions::input_chain`, `Model::points_per_input` | posteriors over the `q * n_w` perturbed points (point-major, as BoTorch: rows `i * n_w .. (i + 1) * n_w` belong to input `i`), with and without observation noise; samples, means, covariances; exact gradients through the perturbations (torch's `minimum`/`maximum` tie rule at the bounds; callables through `torch.autograd`). Model lists require the same `n_w` in all sub-models. |
| `acquisition.risk_measures`: `RiskMeasureMCObjective`, `CVaR`, `VaR`, `WorstCase`, `Expectation` | same path | `acquisition::risk_measures` | `preprocessing_function`: botorch_rs or BoTorch MC objectives, or callables on torch tensors. Values and gradients when called directly (torch autograd). |
| `acquisition.multi_objective.multi_output_risk_measures`: `MultiOutputRiskMeasureMCObjective`, `MultiOutputExpectation`, `IndependentCVaR`, `IndependentVaR`, `MultiOutputWorstCase`, `MVaR`, `MARS` | same path | `acquisition::multi_objective::multi_output_risk_measures` | `MVaR(expectation, pad_to_n_w, filter_dominated, use_counting)`, `get_mvar_set_vectorized`, `get_mvar_set_via_counting`; `MARS(chebyshev_weights, baseline_Y, ref_point)`, `set_baseline_Y(model, X_baseline, Y_samples)`, `chebyshev_weights` / `baseline_Y` setters, `_get_Y_normalization_bounds`, `mvar`; `normalization_bounds` (botorch_rs extension). |
| risk measures as objectives | qEHVI, qLogEHVI, qNEHVI, qLogNEHVI (`cache_root` true or false), `prune_inferior_points_multi_objective`; qLogNEI, qLogEI, qEI, qPI, qUCB, qSimpleRegret, `prune_inferior_points` | the same | evaluated natively in Rust; BoTorch's own risk-measure objects are converted. Outcome constraints are rejected with one-to-many transforms (BoTorch's message), as in BoTorch's multi-objective acquisition functions. |
| `acquisition.multi_objective.multi_fidelity.MOMF` | same path (and `acquisition.multi_objective`) | `acquisition::multi_objective::multi_fidelity::Momf` | `cost_call`: any callable on torch tensors (`batch x q x d` to `batch x q x 1`, differentiated with autograd), default BoTorch's `AffineFidelityCostModel(fidelity_weights={-1: 1.0}, fixed_cost=1.0)` computed by the internal `affine_fidelity_cost` (BoTorch's cost models themselves are ported separately); `X_pending`, `constraints`, `eta`, `objective`. MOMF test problems (`MOMFBraninCurrin`, `MOMFPark`) are on main. |
| `acquisition.multioutput_acquisition`: `MultiOutputAcquisitionFunction`, `MultiOutputPosteriorMean`, `MultiOutputAcquisitionFunctionWrapper` | same path (`MultiOutputAcquisitionFunction` also from `acquisition`) | `acquisition::multioutput_acquisition` | the wrapper of botorch_rs acquisition functions and the posterior mean are evaluated in Rust; other wrapped functions are called. |
| `utils.multi_objective.optimize.optimize_with_nsgaii` | same path | `utils::multi_objective::optimize::{optimize_with_nsgaii, nsga2}` and the building blocks `non_dominated_fronts`, `crowding_distance`, `sbx`, `polynomial_mutation` | every argument of BoTorch's: `q`, `ref_point`, `objective`, `constraints`, `inequality_constraints`, `population_size`, `max_gen`, `seed`, `fixed_features`, `max_attempts`, `discrete_choices`, `post_processing_func`. pymoo is not needed. Without `objective`/`constraints`, native multi-output acquisition functions run entirely in Rust with the GIL released. |

NSGA-II implementation. BoTorch delegates to pymoo's `NSGA2`. The Rust crate
`moors` (the only maintained NSGA-II crate) was evaluated and not used: it
depends on `faer` 0.23, `ndarray` 0.16 and `rand` 0.9 (the workspace uses
`nalgebra` and `rand` 0.10), has no polynomial mutation and only fixed
iteration counts, so it could not reproduce pymoo's defaults that BoTorch
relies on. `botorch_core` implements NSGA-II (Deb, Pratap, Agarwal and
Meyarivan, *IEEE TEC* 6(2), 2002) following pymoo 0.6.2 as BoTorch configures
it: `FloatRandomSampling`; binary tournament selection on concatenated random
permutations (`compare` by constraint violation, dominance, crowding
distance, then at random); SBX (`eta = 15`, mating probability 0.9,
per-variable probability 0.5, children exchanged with probability 0.5) and
polynomial mutation (`eta = 20`, probability 0.9, per-variable probability
`min(0.5, 1/d)`), both clamped to the bounds; duplicate elimination (`eps =
1e-16`, at most 100 mating rounds per generation); rank-and-crowding survival
with infeasible points ranked by constraint violation and random tie breaks;
pymoo's `DefaultMultiObjectiveTermination` (`xtol = 5e-4`, `cvtol = 1e-8`,
`ftol = 5e-3`, `n_skip = 5`, `period = 50`, at most 1000 generations or
100,000 evaluations, robust over the window) or exactly `max_gen`
generations; the result is the first front of the final population.

## Equivalence with BoTorch

Rust fixture tests (`botorch_core/tests/mo_robust_parity.rs`, fixtures from
`scripts/fixtures_mo_robust.py`, BoTorch main `da00e0401`; worst error per
quantity; tolerance 1e-9 for values, 1e-8 for gradients, or 10x BoTorch's
own change under 1e-14 perturbations of its training inputs (its "noise
floor") where that is larger):

| Quantity | Worst error |
|---|---|
| `InputPerturbation` / chains: transformed points, vector-Jacobian products | 0 |
| model posteriors over `q * n_w` points: means / covariances / gradients | 1.5e-15 / 5.6e-15 / 3.8e-15 |
| risk measures (all, incl. MVaR options and MARS): values / gradients | 1.6e-16 / 0 |
| MVaR with ties and with an `m`-changing preprocessing function (vs the #3392 branch) | 0 |
| qEHVI / qNEHVI with risk measures: values | 6.9e-11 |
| qLogEHVI / qLogNEHVI with risk measures: log values / gradients | 1.1e-9 / 1.6e-6 (noise floors 2.5e-5 / 7.6e-4) |
| `MARS.set_baseline_Y(model)`: baseline and normalization bounds | 3.9e-16 |
| qLogNEI(MARS): log values / gradients | 9.6e-10 / 3.8e-7 (noise floor 3.6e-7) |
| MOMF (default and exponential costs, pending points, constraints): values / gradients | 4.3e-15 / 0 |
| NSGA-II building blocks (fronts, crowding distance, SBX, polynomial mutation given the random numbers) | 0 |

Live tests (`botorch_pyo3/tests/test_parity_mo_robust.py`, through both
public Python APIs with identical base samples; BoTorch main and 0.18.1;
`-m equivalence` adds a randomized sweep of models, risk measures and
acquisition functions):

| Quantity | Worst error | Tolerance |
|---|---|---|
| transformed points (`InputPerturbation`, chains, `transform`) | 0 | 1e-15 |
| posterior mean / variance / covariance (`q * n_w` points) | 3.0e-16 / 3.0e-15 / 3.8e-15 | 1e-9 |
| posterior samples from identical base samples | 2.4e-14 | 1e-8 |
| fitted `Normalize` bounds of a chain | 0 | 0 |
| CVaR, VaR, WorstCase, Expectation: values / gradients | 2.1e-16 / 0 | 1e-9 / 1e-8 |
| MultiOutputExpectation, IndependentCVaR/VaR, MultiOutputWorstCase: values / gradients | 1.0e-16 / 0 | 1e-9 / 1e-8 |
| MVaR (all options): values / gradients; MVaR sets | 0 / 9.4e-17; 0 | 1e-9 / 1e-8; 0 |
| MARS: values / gradients / baseline / bounds | 4.2e-16 / 0 / 1.0e-16 / 1.0e-16 | 1e-9 / 1e-8 |
| qEHVI, qLogEHVI with risk measures: values / gradients | 1.6e-14 / 2.8e-14 | 1e-9 / 1e-8 |
| qNEHVI, qLogNEHVI with risk measures: values / gradients | 1.9e-10 / 4.1e-9 where the nominal tolerance applies; up to 7.2e-5 / 1.8e-4 where BoTorch's own noise floor (up to 2.5e-4 / 7.0e-3) is larger | 1e-9 / 1e-8 or 10x floor |
| qLogNEI (CVaR, VaR, MARS), qLogEI, qSimpleRegret: values / gradients | 5.8e-10 / 6.8e-9 (6.2e-9 / 1.1e-8 above a noise floor) | 1e-9 / 1e-8 or 10x floor |
| MOMF: values / gradients | 3.4e-16 / 3.7e-16 | 1e-9 / 1e-8 |
| NSGA-II vs pymoo (fronts, crowding distance) | 0 (needs pymoo) | 1e-15 |
| `optimize_with_nsgaii` vs BoTorch + pymoo: mean hypervolume over 10 seeds | within 4 standard errors or 1% (needs pymoo) | statistical |

Property tests (`botorch_pyo3/tests/test_mo_robust.py`, every array type):
posteriors equal the plain posterior at the perturbed points; MVaR equals the
brute-force definition (with ties); NSGA-II reaches 99.5% of the optimal
hypervolume on ZDT1 (points within 0.02 of the front) and at least 90% of a
uniform 105-point set on DTLZ2 (3 objectives); the front of a GP's posterior
mean has at least 99.5% of the hypervolume of a 4096-point Sobol grid.

Near-zero improvements. The joint posteriors of the noisy acquisition
functions over their perturbed baseline are nearly singular, and when the
expected improvement is negligible the log values (around -42) are dominated
by the smoothing (`tau_relu = 1e-6`): BoTorch's own values then change by up
to 2.5e-4 (log) and its gradients by 7e-3 under 1e-14 perturbations of the
training inputs, which bounds the attainable agreement.

## Timings

Windows 11, 6 threads (torch and rayon), BoTorch main `da00e0401`, median of
5 runs after a warm-up. Model: two-output `SingleTaskGP`, `d = 3`, 20
training points (the baseline), fixed hyperparameters, `InputPerturbation`
with 16 Sobol-normal perturbations (scale 0.05) clamped to the unit cube;
128 MC samples.

| Task | BoTorch | botorch_rs | Speed-up |
|---|---|---|---|
| qNEHVI(MVaR, alpha 0.8): construction | 0.544 s | 0.021 s | 26x |
| qNEHVI(MVaR): values and gradients of 64 candidates (q = 1) | 2.00 s | 0.057 s | 35x |
| `optimize_acqf`(qNEHVI(MVaR)), 4 restarts, 64 raw samples | 29.8 s | 0.32 s | 92x |
| `MARS.set_baseline_Y` + qLogNEI(MARS) construction | 0.058 s | 0.0007 s | 81x |
| qLogNEI(MARS): values and gradients of 64 candidates (q = 1) | 0.37 s | 1.37 s | 0.27x |
| `optimize_with_nsgaii`(MultiOutputPosteriorMean), population 250, 100 generations | >= 1.35 s (acquisition evaluations alone; pymoo not installed) | 0.57 s | >= 2.4x |
| the same with pymoo's default termination | (needs pymoo) | 0.36 s | |

qLogNEI is slower: botorch-rust's qLogNEI (an earlier port) samples the
joint posterior of the baseline and the candidates for every candidate set
(BoTorch's `cache_root=False`), which costs a Cholesky factorization of
`(n + q) * n_w` points per output; BoTorch's default `cache_root=True`
updates a cached factor of the `n * n_w` baseline points (low rank). With
`n_w = 16` that is a 336-point factorization per candidate set. Porting the
low-rank path (as qNEHVI already has) to qLogNEI is the remaining
performance item.

## Deviations from BoTorch main

- MVaR keeps tied alpha-level points that BoTorch drops (meta-pytorch/botorch#3392,
  6ce9d2135).
- MVaR applies the preprocessing function before reading the number of
  outputs, so preprocessing functions may change it (BoTorch main fails;
  #3392, 4310b6538).
- Non-incremental qNEHVI with MVaR compares cached points, not samples
  (#3392, edeee815f).
- `MVaR(expectation=True)` has gradients (the mean of the attaining samples'
  gradients); BoTorch returns NaN of the wrong shape when the samples require
  gradients, and qEHVI then enumerates `2^(q * batch)` subsets.
- Inside acquisition functions every candidate set's MVaR sets are padded on
  their own; BoTorch pads to the largest set of the whole t-batch, so its
  qLogEHVI/qLogNEHVI values of a candidate set depend on the other candidate
  sets of the batch (equal when each set is evaluated on its own).
- Python floats (`alpha`, Chebyshev weights, MOMF's affine cost weights) are
  used as exact float64 values (BoTorch stores some in float32).
- A perturbation function's `n_w` is determined at model construction from
  the first training point and must be constant; the function must act
  pointwise (as BoTorch's documentation states).
- `InputPerturbation(transform_on_train=True)` is rejected by the models (an
  exact GP cannot have `n * n_w` training inputs for `n` targets).
- Outcome constraints are rejected with any one-to-many transform, including
  chained ones and the single-objective acquisition functions (BoTorch only
  checks a direct `InputPerturbation` in its multi-objective base class).
- Analytic acquisition functions and point posteriors need one posterior
  point per input and raise `UnsupportedError` with `n_w > 1`.
- qLogNEI with one-to-many transforms works with `cache_root=False` (BoTorch
  splits the samples by `[n_baseline, q]` instead of `n_w`-blocks and fails).
- NSGA-II's random streams differ from pymoo's (results agree
  statistically); the hypervolume-subset tie breaks and the random dominated
  points that fill up a front shorter than `q` use a seeded generator instead
  of Python's and NumPy's global ones; `post_processing_func` is applied to
  the deduplicated front.
- `InputPerturbation.transform` and `ChainedInputTransform.transform` called
  directly are not connected to torch's autograd graph (acquisition
  gradients through the transforms are exact).
- Acquisition functions keep the state of a `MARS` objective at their
  construction (BoTorch reads it at every evaluation).

## BoTorch bugs found

Minimal reproductions (BoTorch main `da00e0401` and 0.18.1):

1. `MVaR(expectation=True)` on samples that require gradients returns NaN of
   the wrong shape:
   `MVaR(n_w=6, alpha=0.5, expectation=True)(torch.rand(2, 6, 2, requires_grad=True))`
   is `2 x 2 x 2` NaN instead of `2 x 1 x 2`.
2. As a consequence, qEHVI with `MVaR(expectation=True)` and gradients runs
   its inclusion-exclusion over `2^(q * batch)` subsets (memory blow-up).
3. MVaR pads the MVaR sets to the largest set of the whole t-batch, so
   qLogEHVI/qLogNEHVI values of a candidate set depend on the other candidate
   sets evaluated with it.
4. qLogNEI with `cache_root=False` fails for models with an
   `InputPerturbation` (`split_with_sizes` with `[n_baseline, q]` on
   `(n_baseline + q) * n_w` points).
5. The constraint check of the multi-objective base class only recognizes a
   direct `InputPerturbation`: with a `ChainedInputTransform` containing one,
   qNEHVI with outcome constraints is constructed and then fails with
   `RuntimeError: The size of tensor a (18) must match the size of tensor b
   (12)` (its message also lacks two spaces: "due tosample", "inputs.Use").

## Migration notes

- Import paths are BoTorch's: `from botorch_rs.models.transforms.input import
  InputPerturbation`, `from botorch_rs.acquisition.risk_measures import CVaR`,
  `from botorch_rs.acquisition.multi_objective.multi_output_risk_measures
  import MVaR, MARS`, `from botorch_rs.acquisition.multi_objective import
  MOMF`, `from botorch_rs.utils.multi_objective.optimize import
  optimize_with_nsgaii`.
- BoTorch's risk-measure objects can be passed to botorch_rs acquisition
  functions; they are converted to the Rust implementations.
- `optimize_with_nsgaii` needs no pymoo; with `objective` or `constraints`
  (callables on torch tensors) the population is evaluated through Python.
- MOMF's `cost_call` receives `1 x n x d` tensors and must return one cost
  per point.

## Not ported

- `InputPerturbation` in train mode, `Normalize` with a batch shape in
  chains, other input transforms in chains (`AppendFeatures`, `Warp`,
  `Round`, ...), risk measures on batched models.
- BoTorch's "fantasize mode" (`transform_on_fantasize`, recorded only):
  `condition_on_observations` and `fantasize` raise `UnsupportedError` for
  models with a one-to-many transform (BoTorch fantasizes at the
  unperturbed points).
- Other pymoo algorithms and options of `optimize_with_nsgaii`'s internals
  (pymoo's `Result` object, callbacks).
- `MVaR.make_differentiable` as public API (its gradient is part of the
  objective).
