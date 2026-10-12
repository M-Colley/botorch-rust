# Draft: README additions for the multi-objective utilities

This draft collects the README changes for the port of BoTorch's
multi-objective utilities (`botorch.utils.multi_objective` without
`optimize`, plus `sample_simplex` and `sample_hypersphere`). It is meant to
be merged into `README.md` (and the API coverage regenerated with
`scripts/api_coverage.py`).

## Supported API (new rows)

| BoTorch / GPyTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.utils.multi_objective.pareto.is_non_dominated` | same path | `(batch_shape) x n x m` outcomes, `maximize`, `deduplicate` (the first of identical points is kept); BoTorch's mask for every input size (BoTorch's all-pairs and loop variants agree); points with a NaN are dominated (meta-pytorch/botorch#3392) |
| `botorch.utils.multi_objective.hypervolume.Hypervolume`, `infer_reference_point`, `get_hypervolume_maximizing_subset` | same paths | the dimension-sweep algorithm of Fonseca et al. (2006) ported line by line (float64 results identical to BoTorch's); `infer_reference_point` with NaN entries of `max_ref_point`, `scale`, `scale_max_ref_point`; the greedy subset with BoTorch's `random.choice` fallback (identical selections under `random.seed`) |
| `botorch.utils.multi_objective.box_decompositions.NondominatedPartitioning`, `FastNondominatedPartitioning`, `DominatedPartitioning`, `BoxDecompositionList` | same paths | stateful, backed by Rust: `pareto_Y`, `Y`, `ref_point`, `num_outcomes`, `batch_shape`, `update(Y)` (incremental for `m > 2` in the fast partitionings), `reset()`, `partition_space()`, `compute_hypervolume()`, `get_hypercell_bounds()`, `hypercell_bounds`, `_U`, `_Z`; `alpha` (approximate binary partitioning); batches of two-objective fronts of different sizes (padded as in BoTorch), batched reference points; cells and local upper bounds in BoTorch's order |
| `botorch.utils.multi_objective.box_decompositions.box_decomposition.BoxDecomposition`, `FastPartitioning` | same path | abstract base classes (`isinstance` checks) |
| `botorch.utils.multi_objective.box_decompositions.utils.*` | same path | `compute_local_upper_bounds`, `get_partition_bounds`, `update_local_upper_bounds_incremental`, `compute_non_dominated_hypercell_bounds_2d`, `compute_dominated_hypercell_bounds_2d`, `_expand_ref_point`, `_pad_batch_pareto_frontier` (`is_pareto`, `feasibility_mask`) |
| `botorch.utils.multi_objective.scalarization.get_chebyshev_scalarization` | same path | returns a `ChebyshevScalarization` object callable as `obj(Y, X=None)` with BoTorch's semantics (normalization by the observed outcomes, negative weights for minimization, `alpha`); evaluated in Rust, differentiable with `torch.autograd` (exact gradients from Rust), e.g. inside `GenericMCObjective` |
| `botorch.utils.sampling.sample_simplex`, `sample_hypersphere` | `botorch_rs.utils.sampling.*` | `d`, `n`, `qmc`, `seed`, `device`, `dtype` |

Rust (`botorch_core`), for the module overview table:

| BoTorch | botorch_core |
|---|---|
| `is_non_dominated(Y, maximize, deduplicate)` | [`utils::multi_objective::pareto::is_non_dominated`] (`n × m` matrix), `is_non_dominated_rows` (row-major slice), `is_non_dominated_batch` (a `PointBatch`, parallel over the batch) |
| `Hypervolume(ref_point).compute(pareto_Y)`, `infer_reference_point`, `get_hypervolume_maximizing_subset` | [`utils::multi_objective::hypervolume`] (`Hypervolume::new(&r).compute(&y)`, `infer_reference_point`, `get_hypervolume_maximizing_subset(n, &y, &r, choose)`) |
| `NondominatedPartitioning`, `FastNondominatedPartitioning`, `DominatedPartitioning`, `BoxDecompositionList`, `box_decompositions.utils` | [`utils::multi_objective::box_decompositions`] (the `BoxDecomposition` trait, contiguous `CellBounds`, `BoxDecompositionList::par_build`) |
| `get_chebyshev_scalarization(weights, Y, alpha)` | [`utils::multi_objective::scalarization::ChebyshevScalarization`] (values and gradients) |
| `sample_simplex`, `sample_hypersphere` | [`utils::sampling::sample_simplex`], [`utils::sampling::sample_hypersphere`] (and the transformations `simplex_from_uniform`, `hypersphere_from_normal`) |

"What is not ported yet", multi-objective bullet: the utilities are ported;
`botorch.utils.multi_objective.optimize` (`optimize_with_nsgaii`, NSGA-II via
pymoo) is not, nor are the NEHVI helpers of `hypervolume.py`
(`SubsetIndexCachingMixin`, `compute_subset_indices`,
`NoisyExpectedHypervolumeMixin`), which belong to the multi-objective
acquisition functions.

## Equivalence with BoTorch (new rows)

Fixture phase: `python scripts/generate_fixtures.py --phase 5a-utils`
(`botorch_core/tests/fixtures/phase5a_utils/`, Rust tests
`botorch_core/tests/mo_utils_parity.rs`); property tests independent of
BoTorch in `botorch_core/tests/mo_utils_properties.rs`; live comparisons in
`botorch_pyo3/tests/test_parity_mo_utils.py` (section "multi-objective
utilities" of the pytest summary; the randomized sweep is part of
`-m equivalence`).

Coverage: m = 1..5 objectives, n = 0..300 points (1100–1200 for BoTorch's
loop variant of `is_non_dominated`, also in float32 and with batches above
its 5 MB limit), uniform points, points near the unit sphere (fronts of up to
300 Pareto points) and grid points (ties, duplicates), infinities, NaNs,
`maximize=False`, `deduplicate=False`, points that do not dominate the
reference point, empty fronts, batches of fronts of different sizes (shared
and batched reference points), incremental updates with new Pareto points,
with dominated points only and with points worse than the reference point,
`reset()`, `alpha` ∈ {1e-3, 1e-2, 0.1} (and outside [0, 1]), float32 inputs,
error types and messages.

| Quantity | Tolerance | Rust fixtures | Python API (live, BoTorch main) |
|---|---|---|---|
| `is_non_dominated` masks: all flags, m = 1..5, n = 0..300, BoTorch's loop variant (n = 1100–1200, float32, batches above 5 MB), ties, duplicates, ±inf | identical | identical (380 masks) | identical |
| `is_non_dominated` with NaN outcomes (vs BoTorch with meta-pytorch/botorch#3392; live: also vs BoTorch's mask of the points without NaNs) | identical | identical (16 masks) | identical |
| `Hypervolume.compute` (m = 1..5, n ≤ 300, Pareto and raw sets, points worse than the reference point, ±inf) | 1e-12 (relative) | exact | exact |
| `Hypervolume.compute`, float32 inputs: vs BoTorch in float64 on the same values / vs BoTorch in float32 | 1e-12 / 1e-6 | | exact / 1.3e-7 |
| `infer_reference_point` (NaN entries of `max_ref_point`, single points, constant objectives, empty sets, both scalings) | 1e-12 | exact | exact |
| `get_hypervolume_maximizing_subset` (incl. BoTorch's `random.choice` fallback) | identical selections | identical | identical (under `random.seed`) |
| Box decompositions (3 classes): Pareto sets, cells, local upper bounds `_U`, `_Z` (in BoTorch's order), after construction, updates and `reset`; batches; `alpha` | identical | identical (bitwise) | identical (bitwise) |
| Box decompositions: hypervolumes | 1e-12 | 3.0e-15 | 7.4e-16 |
| Box decompositions, float32 inputs: cells / hypervolume (vs float64 BoTorch: the rounding of the result; vs float32 BoTorch) | identical / 2⁻²⁴ / 1e-6 | | identical / 8.9e-9 / 1.1e-7 |
| `BoxDecompositionList`: padded cells / hypervolumes | identical / 1e-12 | | identical / 1.8e-16 |
| `box_decompositions.utils` (local upper bounds step by step, `get_partition_bounds`, 2-d cells, padding, `_expand_ref_point`) | identical | identical | identical |
| `get_chebyshev_scalarization`: values / autograd gradients | 1e-12 | exact / 9.4e-17 | exact / 8.4e-17 |
| `sample_simplex` / `sample_hypersphere`: transformation of given random numbers | exact / 1e-15 | exact / exact | exact / exact |
| `sample_simplex` / `sample_hypersphere` (i.i.d. and QMC, 16384 samples): mean and covariance vs BoTorch's and vs the exact moments (statistical) | 0.03 / 0.015 (i.i.d.), 0.002 (QMC) | | ≤ 6.2e-3 (i.i.d.), ≤ 6.1e-5 (QMC) |
| Error types and messages (NaN outcomes, batch dimensions, batched m > 2, reference point shapes, uninitialized state, single outcome, hypervolume shapes, empty Pareto sets, subset sizes, scalarization weights), warnings, signatures | identical | | identical; no warnings in either library |

Independent of BoTorch (`mo_utils_properties.rs`, `test_multi_objective.py`):
the dimension sweep and all three decompositions agree with an exact grid
computation of the hypervolume (worst 3.4e-15) and with Monte-Carlo
estimates; random points lie in exactly one dominated or non-dominated cell
(the cells tile the box between the reference point and the ideal point);
batched decompositions equal their elements; incremental updates equal fresh
decompositions; `BoxDecompositionList` pads and updates correctly in parallel;
the samplers have the moments of the uniform distributions on the simplex and
the sphere.

Order of the outputs: the algorithms are ported step by step, so the Pareto
points, the hypercells and the local upper bounds come out in BoTorch's order
and are compared in order (no canonical sorting needed). Hypervolumes of the
decompositions are sums over the cells, which torch reduces in a different
order; the relative differences (≤ 3e-15) stay far below the tolerance.

### Benchmarks (new section rows or subsection)

`scripts/benchmark_mo_utils.py` (Python APIs, single thread: `torch.set_num_threads(1)`,
`botorch_rs.set_num_threads(1)`; median of 3 runs of at least 0.1 s; BoTorch
main at da00e0401, Intel Core Ultra 7 155H, Windows 11, Python 3.13, torch
2.13; measured while other jobs ran on the machine). Points near the unit
sphere, so that nearly all points are Pareto optimal; "decomposition" =
construction, `get_hypercell_bounds()` and `compute_hypervolume()`.

| Task | m | n = 10 | n = 50 | n = 100 | n = 200 |
|---|---|---|---|---|---|
| `Hypervolume.compute` | 2 | 48 µs vs 808 µs (17×) | 79 µs vs 2.39 ms (30×) | 65 µs vs 2.65 ms (41×) | 92 µs vs 7.97 ms (87×) |
| | 3 | 70 µs vs 4.17 ms (59×) | 72 µs vs 60.8 ms (840×) | 96 µs vs 183 ms (1900×) | 155 µs vs 538 ms (3500×) |
| | 4 | 65 µs vs 7.33 ms (112×) | 138 µs vs 339 ms (2500×) | 334 µs vs 1.43 s (4300×) | 1.52 ms vs 6.92 s (4600×) |
| `NondominatedPartitioning` | 2 | 209 µs vs 1.07 ms (5×) | 228 µs vs 1.18 ms (5×) | 188 µs vs 1.79 ms (10×) | 279 µs vs 5.17 ms (19×) |
| | 3 | 307 µs vs 117 ms (380×) | 7.87 ms vs 3.92 s (500×) | 44.9 ms vs 16.7 s (370×) | 195 ms vs – |
| | 4 | 1.68 ms vs 1.06 s (630×) | 387 ms vs 226 s (580×) | 3.48 s vs – | 65 s vs – |
| `FastNondominatedPartitioning` | 2 | 286 µs vs 937 µs (3×) | 226 µs vs 958 µs (4×) | 349 µs vs 1.47 ms (4×) | 323 µs vs 3.92 ms (12×) |
| | 3 | 308 µs vs 42.4 ms (138×) | 499 µs vs 140 ms (280×) | 1.23 ms vs 385 ms (310×) | 2.09 ms vs 624 ms (300×) |
| | 4 | 401 µs vs 117 ms (290×) | 2.46 ms vs 551 ms (220×) | 8.72 ms vs 1.24 s (143×) | 34.7 ms vs 3.73 s (107×) |
| `DominatedPartitioning` | 2 | 161 µs vs 527 µs (3×) | 208 µs vs 681 µs (3×) | 180 µs vs 1.34 ms (7×) | 355 µs vs 4.53 ms (13×) |
| | 3 | 244 µs vs 14.1 ms (58×) | 264 µs vs 67.8 ms (260×) | 888 µs vs 118 ms (132×) | 1.28 ms vs 196 ms (153×) |
| | 4 | 207 µs vs 27.3 ms (132×) | 645 µs vs 77.8 ms (121×) | 1.08 ms vs 178 ms (165×) | 3.27 ms vs 476 ms (145×) |
| `FastNondominatedPartitioning` + `update` (5 points) | 3 | 335 µs vs 83.6 ms (250×) | 658 µs vs 343 ms (520×) | 2.29 ms vs 599 ms (260×) | 3.35 ms vs 984 ms (290×) |
| | 4 | 599 µs vs 229 ms (380×) | 4.42 ms vs 1.27 s (290×) | 16.7 ms vs 2.35 s (141×) | 80.1 ms vs 7.48 s (93×) |

("–": BoTorch not run after a run exceeded 10 s.) qNEHVI-style
decompositions of 128 Monte-Carlo samples of 20 points: a batched
`FastNondominatedPartitioning` (m = 2) 783 µs vs 12.0 ms (15×); a
`BoxDecompositionList` of 128 `FastNondominatedPartitioning`s (m = 3) 32.5 ms
vs 9.83 s (300×); `is_non_dominated` of 128 × 200 points (m = 3) 39.4 ms vs
314 ms (8×), 14.2 ms with botorch_rs's 22 threads.

For small inputs the Python timings are dominated by the bindings (array
conversion and dispatch to the thread pool); the native timings
(`cargo run --release --example benchmark_mo_utils`) are what Rust callers
such as the acquisition functions see, e.g. for n = 100: `Hypervolume`
1.6 / 17 / 226 µs (m = 2 / 3 / 4), `FastNondominatedPartitioning` 11 µs /
426 µs / 7.0 ms, `DominatedPartitioning` 10 / 140 / 527 µs; 128 samples of 20
points: batched m = 2 decomposition 0.42 ms, `BoxDecompositionList::par_build`
(m = 3) 7.2 ms on one thread and 1.2 ms on 22 threads.

## Migration notes and known differences (new bullets)

- **Multi-objective utilities: NaN outcomes.** `is_non_dominated` treats a
  point with a NaN in any outcome as dominated, and such points never
  eliminate other points, as documented by BoTorch and implemented in
  meta-pytorch/botorch#3392. BoTorch main (and 0.18.1) only masks out points
  whose outcomes are *all* NaN: a partially NaN point is reported as
  non-dominated (with `deduplicate=True` only if it is the first point), and
  in BoTorch's loop variant (n > 1000 or large batches) it can eliminate
  genuine Pareto points. The box decompositions reject NaN outcomes as in
  BoTorch (`ValueError`), so they are not affected.
- **Multi-objective utilities: incremental updates.** For more than two
  objectives, `FastNondominatedPartitioning.update` and
  `DominatedPartitioning.update` return early when no new point is better
  than the reference point (meta-pytorch/botorch#3392); BoTorch main
  re-inserts its whole frontier into the local upper bounds there. The
  results are identical (checked bitwise on all fixture states), only faster.
- **Multi-objective utilities: batched reference points.** With a batched
  reference point and no Pareto point in any batch element,
  `FastNondominatedPartitioning` and `DominatedPartitioning` return one cell
  per batch element (BoTorch fails to broadcast the reference point,
  `RuntimeError`); `_pad_batch_pareto_frontier` accepts one reference point
  for all batch elements (BoTorch requires one per batch element when a
  front is empty, `IndexError`).
- **Multi-objective utilities: float32 and other dtypes.** Everything is
  computed in float64; results have the dtype of the outcomes (e.g. float32
  cell bounds are the float32 inputs exactly, float32 hypervolumes are the
  float64 result rounded once: BoTorch, computing in float32, differs by up to
  ~1e-7 relative). `Hypervolume.compute` returns a Python float as in BoTorch.
  `compute_hypervolume()` of a decomposition without outcomes returns BoTorch's
  `torch.tensor(0.0)` (torch's default dtype) for torch inputs.
- **Multi-objective utilities: hypervolume sums.** The decompositions'
  hypervolumes sum over cells sequentially; torch's reductions use a
  different order, so values can differ in the last bits (observed: 3e-15
  relative, after the cancellation in "box minus non-dominated cells").
  `Hypervolume` (dimension sweep), all cells, Pareto sets and local upper
  bounds are bitwise identical to BoTorch's in float64, in BoTorch's order.
- **`get_chebyshev_scalarization`** returns a `ChebyshevScalarization` object
  instead of a closure; it is called like BoTorch's (`obj(Y, X=None)`), is
  differentiable with `torch.autograd` (gradients computed in Rust) and
  exposes `weights`, `alpha` and BoTorch's normalization bounds
  (`Y_bounds`). Acquisition functions can evaluate it natively through its
  Rust object.
- **`get_hypervolume_maximizing_subset`** draws its random fallback choice
  with Python's `random.choice`, like BoTorch, so `random.seed` reproduces
  BoTorch's selection exactly; from Rust, the choice is a closure.
- **`sample_simplex`, `sample_hypersphere`** use botorch-rust's random streams
  (Xoshiro256++ and the Owen-scrambled Sobol sequence for `qmc=True`), so
  for the same seed the samples differ from BoTorch's (torch's generator and
  `SobolEngine`), with the same distribution; the transformations of given
  random numbers are BoTorch's exactly. Without `dtype`, results are float64
  tensors if torch has been imported (BoTorch: torch's default dtype, and an
  int64 tensor for `sample_hypersphere(d=1)`).
- **Unsupported shapes.** A `BoxDecompositionList` of batched
  decompositions, more than one batch dimension and batched decompositions
  with more than two objectives are not supported (as in BoTorch, which
  raises `NotImplementedError` for the latter two). `BoxDecompositionList.update`
  keeps BoTorch's argument check (a 3-dimensional tensor with more batches
  than decompositions is accepted and the extra batches ignored).
- **Not ported:** `botorch.utils.multi_objective.optimize`
  (`optimize_with_nsgaii`, NSGA-II via pymoo); the NEHVI helpers of
  `hypervolume.py` (`SubsetIndexCachingMixin`, `compute_subset_indices`,
  `NoisyExpectedHypervolumeMixin`) come with the acquisition functions.

## Notes for the maintainers (not for the README)

### Versions checked

The live comparisons pass against BoTorch main (da00e0401), the released
0.18.1 (whose multi-objective utilities are identical to main's) and
meta-pytorch/botorch#3392 (012d0777b, where the NaN cases are compared with
BoTorch directly). The fixtures were recorded with main, `pareto_nan.json`
with #3392; the decomposition fixtures are bitwise identical under both.

### Crates evaluated

- `wfg_rs` (MIT, ndarray): the WFG hypervolume algorithm, not BoTorch's
  dimension sweep (results equal only to rounding, a new ndarray dependency);
  not used.
- `hv-fonseca-et-al-2006-sys` (LGPL-2.1, release candidate) and `hv-wfg-sys`
  (GPL-3.0): C wrappers whose licenses do not fit an MIT crate; not used.
- `pareto_front`, `pare`: incremental Pareto fronts / skyline queries without
  BoTorch's mask semantics (first of identical points, flags, batches);
  not used.
- No crate implements BoTorch's box decompositions (Couckuyt et al.'s binary
  partitioning, Lacour et al.'s local upper bounds) or its Chebyshev
  scalarization, so the algorithms are ported from BoTorch with the papers
  cited in the doc comments.

### BoTorch issues found (not fixed in BoTorch; reported here)

1. `FastNondominatedPartitioning` / `DominatedPartitioning` with a batched
   reference point fail when no batch element has Pareto points
   (`_get_single_cell` does not broadcast `ref_point`):

   ```python
   FastNondominatedPartitioning(ref_point=torch.full((3, 2), 2.0), Y=torch.rand(3, 4, 2))
   # RuntimeError: The expanded size of the tensor (1) must match the existing size (3) ...
   DominatedPartitioning(ref_point=torch.full((3, 2), 2.0), Y=torch.rand(3, 4, 2))
   # RuntimeError: The expanded size of the tensor (1) must match the existing size (3) ...
   ```

   botorch_rs returns one cell per batch element (deviation).
2. `_pad_batch_pareto_frontier` with an `m`-dim reference point and a batch
   whose element `i >= 1` has no Pareto point:

   ```python
   _pad_batch_pareto_frontier(torch.tensor([[[1.0, 2.0]], [[-1.0, -1.0]]]), torch.zeros(2))
   # IndexError: index 1 is out of bounds for dimension 0 with size 1
   ```

   BoTorch's callers expand the reference point first; botorch_rs broadcasts.
3. `BoxDecompositionList.update` checks
   `Y.ndim != 3 and Y.shape[0] != len(...)` (probably meant `or`): a 3-d
   tensor with more batches than decompositions is accepted and the extra
   batches are ignored silently; a 2-d tensor passes the check and fails later
   with `RuntimeError: Tensors must have same number of dimensions`.
   botorch_rs keeps the check.
4. Already fixed by meta-pytorch/botorch#3392: partially NaN points in
   `is_non_dominated`, and the needless re-insertion of the whole frontier in
   `FastPartitioning.update` when no new point is better than the reference
   point.

A minor inconsistency kept for compatibility: `compute_hypervolume()` of a
decomposition without outcomes returns `torch.tensor(0.0)` in torch's default
dtype, whatever the dtype of the reference point.

### Checks

`ruff check` / `ruff format --check` on all of `scripts` fail on main already
(`generate_fixtures.py`: B023 in phase 4b and formatting;
`benchmark_optimize_acqf.py`: formatting); the new code is clean, and the
files CI lints pass. `scripts/benchmark_mo_utils.py` could be added to CI's
ruff command.
