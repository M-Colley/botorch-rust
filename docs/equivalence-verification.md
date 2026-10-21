# Interpreting equivalence verification

Passing test counts establish the collected cases at the recorded source,
native artifact and reference revisions. They do not establish every public
configuration, GPU coverage or a universal speedup. The API availability audit
and behavioral tests are separate measurements.

The unified suite uses several kinds of checks:

| Check | Meaning |
| --- | --- |
| Deterministic values / gradients | The mixed elementwise/normwise error is bounded by `1e-9` / `1e-8`; log values use the separately documented log error. |
| Common-normal MC values / gradients | The same bounds, or ten times the reference's measured sensitivity to a `1e-14` input perturbation; raw and scaled errors are reported. |
| Numerically singular MC laws | Finiteness and a loose `0.1` value-error bound when the smallest covariance eigenvalue is below `1e-12` of the largest. This is a stabilization check, not strict pointwise equivalence. |
| Identical-start candidate optimization | At least 80% of restarts reach the reference value within relative `1e-6`. |
| Three-seed candidate optimization | The port's best result must reach the reference's median within `0.05 * max(1, abs(median))`, with feasibility and returned-value checks. This is a regression guard, not strict optimizer trajectory equivalence. |
| GP fitting | A separate two-stage first-attempt outcome comparison, described in `fitting.md`, with explicit retries and statistical controls for multiple local optima. |

The optimizer table in the original suite docstring said `1e-3` while its
existing `OPT_TOL` was `0.05`. The docstring now states the actual bound; the
implementation tolerance has not been loosened.

An October 21 audit found that the shared numeric error helper treated two NaNs
as equal. It now rejects any NaN in a numerical equivalence check. Successful
first-attempt fit outcomes must also be finite before statistical comparison;
nonfinite outcomes cannot be hidden among other fit results. Explicit tests of
invalid inputs continue to compare their nonfinite classes separately.
Regressions cover shared and one-sided NaNs and nonfinite first-attempt results
from either implementation. The existing deliberately degraded optimizer test
still verifies that the fitting comparison rejects a systematic regression.

Oracle adjustments are scoped to the affected tests. The source checkout stays
pinned and clean. Model-list comparisons can materialize its actual covariance
blocks to work around documented operator-indexing defects; noisy constrained
improvement can supply common auxiliary lower-bound points. Where a source
derivative is demonstrably wrong, tests separately check unmodified source
values/first derivatives and independent Gaussian identities or convergent
finite differences. Such cases do not certify the defective source derivative.
Confirmed repairs are linked in `upstream-pull-requests.md`.

The post-release Linux Torch 2.14.1 CPU run exposed a preference test-oracle
failure while exploring the alternate outcome of an ambiguous Cholesky
decision. Its `2 * abs(lambda_min)` diagonal shift can be zero for a valid
rank-one covariance. Unmodified source handles that matrix with ordinary
jitter. The test helper now uses a representable positive margin, increasing
it only within the existing Demmel roundoff budget and a finite attempt
limit. New controls cover scaled rank-one matrices, unchanged positive-definite
values/gradients, exact-zero and unrepresentable budgets, and rejection of
genuinely indefinite/nonfinite matrices. The comparison tolerances, source,
runtime package and published 0.3.0 tag remain unchanged by this repair.

The stricter audit exposed one randomized `Power(2)` transform fixture with
negative Gaussian latent draws. Its inverse is a square root, so these samples
and the affected acquisition gradients are undefined in both implementations.
The transform suite now compares NaN/positive-infinity/negative-infinity classes
explicitly for this inverse-power domain, and applies numerical tolerances only
to finite entries. Undefined entries are reported as domain behavior, never
zero numerical error. Ordinary gradient comparisons now reject every nonfinite
entry. Separate regressions reject mismatched classes and finite-value errors;
a fully defined positive-latent case checks common-normal samples, source first
derivatives and independent dense Hessians.

The broader Linux run at `637ee4a` exposed 21 more legacy comparisons of
intentional nonfinite values. Percentile tests now derive the expected NaN
mask directly from their deliberately missing data/score inputs. MVNXPB tests
derive the exact uncomputed plug-in sentinel mask from the completed step
count; every computed entry still receives a numerical comparison, and a
regression rejects shared NaNs in a computed entry. The periodic-kernel stress
case separately checks its undefined aligned diagonal only after a lengthscale
has underflowed to exactly zero. Finite covariance/diagonal comparisons retain
their existing tolerances. These state/domain checks do not count matching NaNs
as numerical equivalence, and no runtime behavior changes in that correction.

The [0.3.0 installed-wheel run](verification-release-030-2026-10-21.json)
accounts for all 14,560 canonical CPU/CUDA cases exactly once: 14,509 pass and
51 skip, including 1,222 passes and 15 skips in the 1,237-case equivalence
selection. Python/native, wheel, verifier and pinned reference fingerprints
remain stable. The finite case set retains the scoped distinctions above.
Earlier successful checkpoints and focused runs remain separate evidence;
their overlapping pass counts are not added into the release total.
