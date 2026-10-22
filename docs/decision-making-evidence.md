# Decision methods: what worked and what did not

## 22 October correctness follow-up

Native FPBI now checks the grid limit immediately after its required output
dimension probe, before calling constraints or optimizing extrema. Geometry
stores the actual upper endpoints; it does not reconstruct them from a
rounded lower endpoint and span. Cancellation-sensitive bounds agree with
OpenBO and the community port in both objective orientations. The release
core suite passed 10 tests, and the rebuilt checkout extension passed all
24 Python decision tests. An installable wheel release remains pending.

The same audit fixed the external replay's portable dependency fixture and
the community benchmark tests' CI import boundary. Historical numerical
measurements below were preserved and were not rerun for these fixes.

These are optional research utilities, with evidence collected on 21 October
2026. The [algorithm and API guide](decision-making.md) explains their use.
The [upstream draft PR #3531](https://github.com/meta-pytorch/botorch/pull/3531)
contains the proposed community implementation, notebook and evidence bundle;
[issue #3530](https://github.com/meta-pytorch/botorch/issues/3530) records the
scope discussion. A draft proposal is not an accepted BoTorch API.

## Functional behavior that worked

The implementations reproduce Schmitt et al.'s signed knee-distance test,
preference hyperplane, asymmetric FPBI grid and unrestricted ray parameter.
Tests cover original-row indices, stable ties, duplicates, dominated rows,
flat columns, batched preferences, objective orientation, finite extreme
ranges, constrained continuous solves, negative ray parameters and explicit
solver failures. The model adapter was exercised with real BoTorch-compatible
deterministic posterior means. These checks establish behavior, not superior
decisions, robust human preferences or MPC stability.

Recorded local validation:

| Check | Observed result |
| --- | --- |
| OpenBO selection/FPBI tests | 35 passed; one CUDA skip in the CPU environment |
| Rust Python decision tests | 21 passed |
| Native Rust decision tests | Eight passed; release example ran |
| Initial upstream selector CPU/CUDA suite | 18 passed |
| Expanded upstream selector/FPBI/harness CPU check | 36 passed, ten CUDA skips, ten subtests passed |
| Dedicated upstream FPBI CUDA boundary check | One passed |
| Both library CUDA facades | 144 exact mask/index comparisons passed |
| Fixed-archive replay tests | Five passed |
| HITL repository complete suite | 1,250 passed, 11 runtime skips, one collection-skipped module; exit 0 |
| Static paper checks | Passed; LaTeX compilation was not performed |

Both newly ported CLI harnesses also completed functional smokes: the three
FPBI arms on the convex case, seed 19, and 12 CPU selector configurations
with exact parity. The expanded upstream FPBI port and harnesses are later
artifacts. These checks do not rerun the historical numerical measurements
below on the new FPBI module bytes.

## Knee filtering did not improve noisy final decisions here

The log-only replay fixes 100 archives from four analytic MOBO problems,
qLogNEHVI, seeds 12–16 and 50 trials per archive. Eighty archives contain
Gaussian objective errors of one landscape SD, with onset 0/20 and objective
error correlation 0/.85; 20 are clean optimization archives. Every selector
uses the same candidates within an archive. No designs, objective evaluations
or GP fits were added. The replay scores 98,100 preference/condition/rule rows.

Linear and weighted quadratic-distance utilities are declared separately,
each with its matching ordinary reference. The following pools the 80 noisy
archives with exact preference weights. Values are final selected-design
regret in fixed within-archive true-front-normalized utility units; lower is
better. Clean cost compares each method on the same archive with exact
outcomes and true preferences.

| Declared utility | Ordinary reference | Reference noisy regret | Knee noisy regret | Knee gain | Knee clean cost |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear | Preference-weighted sum | 0.237706 | 0.332108 | -0.094402 | +0.072301 |
| Quadratic distance to ideal | Weighted nearest ideal | 0.452719 | 0.547495 | -0.094777 | +0.032571 |

The third rule also remains visible: nearest-ideal selection under linear
truth has noisy regret 0.269813, gain -0.032106 and clean cost +0.033651.
Ordinary weighted selection under quadratic truth has regret 0.448362,
gain +0.004357 and clean cost +0.031272. That small descriptive gain does
not establish superiority and should be read with its clean loss.

The linear knee gain is negative on all four problems. This is not solely
caused by single-objective preferences: among 24 Dirichlet trade-offs per
seed, linear regret is 0.237492 for ordinary weighted selection and 0.324670
for the knee rule, with a clean cost of 0.066320. Equal supplied weights
produce the same weighted and knee-weighted choice, as the formulas predict.
Separate multiplicative log-weight noise also did not establish a general
knee robustness benefit. It preserves exactly zero weights by design.

Four correlated problem clusters support descriptive comparisons only.
The specified utility families do not reward compromise independently of
their stated costs. A person who explicitly wants extremes excluded has a
different decision target. This replay does not compare posterior denoising,
LCB rules, additional confirmation trials or real human benefit.

## FPBI worked as a solver, but lost this small comparison

Thirty-six runs compare FPBI with BoTorch's NSGA-II wrapper on four exact,
low-dimensional deterministic-model cases and three seeds. The median
hypervolume ratio and IGD favored NSGA-II in every case, both under a common
output cap and under approximately matched objective-point counts. The
approximately point-matched comparison is:

| Case | FPBI / NSGA-II HV ratio | FPBI / NSGA-II generation seconds |
| --- | --- | --- |
| Convex 2D | 0.9220 / 0.9519 | 0.3219 / 0.1220 |
| Concave 2D | 0.8502 / 0.8693 | 0.7665 / 0.2282 |
| Constrained 2D | 0.8940 / 0.9228 | 0.4854 / 0.1204 |
| Sphere 3D | 0.5575 / 0.7287 | 0.9405 / 0.2728 |

No complete runs failed. All 72 nominal FPBI rays succeeded, but five of 369
restart attempts were rejected. The maximum returned constraint violation
was approximately 2.8e-16. A common cap does not give identical retained cardinality:
the 2D medians are six FPBI points versus seven NSGA-II points; sphere medians
are 11 versus 12. Point counts and posterior-call counts differ because
NSGA-II batches evaluations. Generation timing excludes reference metrics,
common output capping and imports. Cheap vectorized objectives favor this
batching; these measurements are not a general optimizer ranking or a noisy
BO experiment. Local FPBI solves do not certify a global Pareto front.

## Vectorization had mixed runtime results

All 396 selected indices and 79,104 knee-mask entries matched exactly across
48 paired function configurations. Full selection improved in 11/12 CPU and
4/12 CUDA configurations. The largest observed gains were 4.07x CPU and
26.73x CUDA. The largest full-selection regression was about 24.1x slower:
CUDA float32, shape `(1,1024,3)`, 6.862 ms to 165.3 ms.

That case used BoTorch's memory-bounded Pareto loop. Incremental live CUDA
allocation fell from 6,314,496 to 53,760 bytes while latency increased. This
does not establish a universal memory improvement. The benchmark used one
machine, two warmups, five repeats and observed independent background
workloads; it is descriptive, not a cross-hardware speed claim. The baseline
is the earlier Torch prototype; the measured candidate is the vectorized
upstream selector. Its saved `after.py` hash matches that upstream selector
at packaging time. The later FPBI port is a separate module and was not
retimed in this selector comparison.

## When to use each part

- Use ordinary weighted selection when the declared target is linear utility;
  use nearest-ideal selection when that distance is the declared target.
- Use a knee restriction when the decision maker explicitly wants a compromise
  region and accepts exclusion of some preferred extremes. It is not a
  demonstrated noise-robust default. `alpha=0` can still exclude points.
- Keep FPBI experimental for inexpensive objectives or surrogate means,
  particularly with few objectives. Check failures, normalization and local
  extrema, and compare front generators on the actual application.
- Keep experiment acquisition and preference learning separate. These methods
  do not supply an exploration incentive, learn a person's utility, or inherit
  the thesis's MPC feasibility/stability guarantees.

## Reproduce and inspect the evidence

Run `python -m pytest botorch_pyo3/tests/test_decision_making.py -q` and
`cargo test -p botorch_core --test decision_making` from the Rust checkout.
The draft PR's notebook directory contains compact CSV results, protocols,
version/source hashes, historical source snapshots and reproduction notes.
Its new runnable benchmark harnesses are distinguished from those snapshots.

The selection replay needs the external
[HITL robustness repository](https://github.com/M-Colley/hitl-mobo-robustness-tests),
its helper modules and the 100 input CSVs listed by hash. Raw MOBO logs are
not included in the upstream bundle and may require access from the authors;
cloning the repository alone may not reproduce that measurement. The replay
source is not advertised as a standalone upstream experiment.
