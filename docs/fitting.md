# Fitting at extreme hyperparameters

How `fit_gpytorch_mll` in botorch-rust reproduces BoTorch's fits when
L-BFGS-B visits extreme hyperparameters, and how fits are compared with
BoTorch: the bug, GPyTorch's non-finite values reproduced, the deviations,
the equivalence results, the two-stage comparison of fits and the BoTorch
bug found. Reference: BoTorch main at da00e0401 and BoTorch 0.18.1, with
torch 2.13 and 2.14.1.

Rust: `botorch_core/src/kernels/nonfinite.rs` (new), `fit.rs`, `mll.rs`,
`kernels.rs`, `kernels/eval.rs`, `kernels/covar.rs`, `priors.rs` and the
marginal log likelihoods of `LCEMGP`, `MultiTaskGP` and
`KroneckerMultiTaskGP`. Tests: `botorch_core/tests/fit_nonfinite.rs`,
`botorch_pyo3/tests/test_parity_fit_nonfinite.py`,
`botorch_pyo3/tests/test_parity.py`; the comparison of fits in
`botorch_pyo3/tests/parity_helpers.py` and
`botorch_core/tests/common/mod.rs`, its tests in
`botorch_pyo3/tests/test_fit_comparison.py` and
`botorch_core/tests/kernels_6a_parity.rs`, and its Monte Carlo in
`scripts/fit_comparison_sim.py`.

## Supported API

No new BoTorch names. Behaviour changes of existing rows:

| BoTorch | botorch_rs | Notes |
| --- | --- | --- |
| `fit_gpytorch_mll(mll)` | `botorch_rs.fit.fit_gpytorch_mll`, `botorch_core::fit_gpytorch_mll` | Non-finite losses and gradients go to L-BFGS-B for all models, as in BoTorch; GPyTorch's NaN covariances at extreme hyperparameters are reproduced (`botorch_core::kernels::nonfinite`). |
| `ExactMarginalLogLikelihood(likelihood, model)(...)` | `ExactMarginalLogLikelihood` | Raises `NanError` where GPyTorch's covariance is NaN (e.g. lengthscales of `1e-293`, or zero from raw values). |

Rust additions: `Kernel::training_nan`, `TreeKernel::training_nan`,
`BuiltCovar::training_nan` and `kernels::TrainingNan` (GPyTorch's NaN
entries of a training covariance), `TreeKernel::log_prior_scaled_grad` and
`Prior::scaled_grad_log_prob` (prior derivatives in torch's order).

## The bug

`test_parity.py::test_fit_parity` failed on Linux CI. Its data were seeded
with `hash(config)`, which Python salts per process, so every run fitted
different data. A scan of the seeds 7..1006 for all eight `CONFIGS`
(`scratchpad/fitflake/scan2.py`) found two kinds of failures:

- `custom_scale_rbf`, seed 514: botorch_rs's loss 1.0853, BoTorch's 0.8678
  (BoTorch's 16 twins agree to 2e-11).
- `legacy_gamma`, seeds 104, 203, 417, 665, 667, 753: botorch_rs's
  `fit_gpytorch_mll` raised `InvalidArgumentError: lengthscales must be
  positive and finite, got [1.88e-249, 0.0]` or `outputscale must be positive
  and finite, got 0`. BoTorch fits these data without problems.

scipy's L-BFGS-B on both objectives (`replay2.py`) showed that the objectives
agree to about 1e-15 everywhere except at extreme trial points of the line
search, where the two libraries computed different things:

1. Raw lengthscales of about -674, -359 and -251 (lengthscales 1e-293 ..
   1e-109). GPyTorch's `sq_dist` centers the scaled inputs `c = x/ℓ -
   mean(x/ℓ)` and expands `‖cᵢ‖² + ‖cⱼ‖² - 2 cᵢ·cⱼ`; the squares overflow,
   `inf - inf` is NaN, linear_operator raises `NanError` and BoTorch's closure
   returns NaN. botorch_rs computed the mathematical limit `K = s·I + σ²I`
   (loss 224.66), so the line search took another step and the fit ended in
   another optimum.
2. A raw outputscale of -3074: `softplus` underflows to 0. GPyTorch's loss is
   `+inf` (`GammaPrior(2, 0.15).log_prob(0) = -inf`) with a NaN gradient
   (`inf · softplus'(-3074) = inf · 0`); botorch_rs's kernel validation raised
   `InvalidArgument`, which ends the fit (as a prior-support `ValueError`
   does in BoTorch).
3. A lengthscale of exactly 0 (softplus underflow): GPyTorch's `x / 0` gives
   NaN after the centering; botorch_rs raised.

(`custom_scale_rbf` passes GPyTorch-style kernel objects, which botorch_rs
evaluates as a kernel tree; `legacy_gamma` uses BoTorch's helper, which
botorch_rs evaluates with its specialized RBF / Matérn module. Both computed
the exact limits.)

In addition, only kernel trees handed non-finite values to L-BFGS-B; the
standard RBF / Matérn path ended the attempt and retried from prior samples
(drawn with another random number generator than BoTorch's).

## The fix: GPyTorch's non-finite values

botorch_rs now gives L-BFGS-B what BoTorch's closure gives scipy at every
point: the same finite values, the same `+inf`, and NaN (with NaN gradients)
wherever GPyTorch's covariance is NaN.

- **`kernels::nonfinite`** reproduces GPyTorch's arithmetic where it decides
  finiteness: the centering of `sq_dist` (for Matérn kernels after
  `MaternKernel.forward`'s own centering of the unscaled inputs), the
  expansion `x1_ = [-2c, ‖c‖², 1]`, `x2_ = [c, 1, ‖c‖²]`, `res = x1_ x2_ᵀ`,
  `clamp_min_(0)` (which keeps NaN) and `dist`'s `clamp_min(1e-30).sqrt()`,
  and the kernels' post-processing: RBF `exp(-res/2)`; Matérn
  `(√5 d + 1 + 5/3 d²) · exp(-√5 d)` etc., NaN where `inf · 0` occurs
  (Matérn-3/2 and 5/2 at infinite distances); rational quadratic
  `(1 + res/(2α))^(-α)`; cosine `cos(π d)` (NaN at infinite distances);
  periodic per-dimension distances (`last_dim_is_batch`) and
  `exp(-2 Σ sin²(d)/ℓ)`. Two evaluation paths differ, as in GPyTorch:
  - the autograd path (several lengthscales, or learned inputs such as
    LCEMGP's embeddings): the diagonal of `res` is computed like every
    other entry;
  - `RBFCovariance` / `MaternCovariance` (a single lengthscale, inputs
    without gradients): the diagonal of `res` is zero, and the hand-written
    lengthscale derivatives (`res · K / ℓ` for RBF,
    `(1 + s) · s²/3 · exp(-s) / ℓ` for Matérn-5/2, ...) are NaN where
    `inf · 0` occurs although the covariance is finite (the lengthscale
    gradient is then NaN, as in GPyTorch).

  The MLLs inject the NaN entries into botorch_rs's own (exact, finite)
  covariance, so the Cholesky factorization raises `BotorchError::Nan` with
  linear_operator's count; finite values are never changed. Nothing is
  computed beyond a scan of the centered inputs unless one exceeds `1e100`
  in magnitude or is not finite, so all existing results are bitwise
  unchanged.
- **Hyperparameters from raw values** are no longer validated in the MLL:
  `KernelSpec::build_unchecked` accepts lengthscales and outputscales that
  underflowed to zero (GPyTorch evaluates them). `initialize`, constructors
  and `set_raw_parameters` still raise for them with the specialized RBF /
  Matérn module, as before (kernel trees accepted any finite raw values
  already, like GPyTorch).
- **Prior derivatives in torch's order.** For a softplus value `v` that is
  subnormal, `(α - 1)/v` overflows in botorch_rs but not in torch, whose
  backward pass applies the MLL's `1/n` first (`xlogy`'s backward
  `grad · (α - 1) / v`). Where botorch_rs's prior derivative overflows, it is
  recomputed in torch's order (`Prior::scaled_grad_log_prob`, `GammaPrior`
  and `LogNormalPrior`; standard kernels, kernel trees, noise). Finite
  derivatives are unchanged.
- **All fits hand non-finite values to L-BFGS-B** (`fit.rs`: the
  `nan_tolerant` switch is gone): a NaN covariance gives NaN loss and
  gradients (BoTorch's `_handle_numerical_errors`), other non-finite values
  pass unchanged. The L-BFGS-B port reproduces scipy's handling of them
  (e.g. `fit_nonfinite.rs::lbfgsb_handles_nan_trial_points_like_scipy`: the
  same 27 evaluation points as scipy 1.18, the same result bitwise).
  BoTorch's retry semantics are kept: `NotPSDError` retries, a prior-support
  `ValueError` ends the fit.
- The other `Fittable` models: `LCEMGP` (data kernel on the inputs, task
  kernel on the learned embeddings), `MultiTaskGP` and
  `KroneckerMultiTaskGP` (kernel trees) inject GPyTorch's NaN entries too;
  the multi-fidelity, mixed and MAP-SAAS models are `SingleTaskGP`s with
  kernel trees and follow the `SingleTaskGP` path.
- **Learned input transforms** (phase 6b: `Warp`, `LearnedFeatureImputation`
  fitted jointly with the GP): GPyTorch's NaN entries are computed on the
  transformed training inputs at the current transform parameters, and
  non-finite transformed inputs propagate into the covariance (all entries
  NaN after `sq_dist`'s centering, so the Cholesky raises `Nan` and the fit
  sees NaN, as BoTorch's closure does) instead of ending the fit with an
  `InputData` error. Inputs computed by a learned transform require
  gradients in torch, so GPyTorch evaluates every kernel on its autograd
  path (no `RBFCovariance` / `MaternCovariance` with their zeroed diagonal),
  and so does the emulation. `Warp`'s concentrations are bounded below by `1e-4`
  (L-BFGS-B bounds) and its inputs clamped to `[eps, 1 - eps]`, so the
  Kumaraswamy CDF stays finite in `[0, 1]` for any concentrations; the
  imputation values are mapped into their bounds by a sigmoid if bounds are
  given, else used as they are (extreme values then overflow in the
  kernel's `sq_dist` like any other input). `Warp` models were probed like
  the kernels (see the equivalence tables).
- **Batched fits** (`BatchedMultiOutputGP`, `fit_gpytorch_mll_scipy`): every
  element sees only its own NaN covariances, with NaN gradients. This
  deviates from BoTorch, see bug 1 below.

## Deviations from BoTorch (migration notes)

1. **Batched fits are independent also at NaN points** (BoTorch bug 1):
   BoTorch's batched closure gives every element of the batch a NaN loss
   with zero gradients when any element's covariance is NaN. botorch_rs's
   elements match BoTorch's unbatched fits instead
   (`test_batched_elements_are_independent`).
2. **Finite values at extreme lengthscales.** GPyTorch's squared distances
   carry a cancellation error of about `1e-16 ‖c‖²` (`c` the centered scaled
   inputs); its diagonal `‖cᵢ‖² + ‖cᵢ‖² - 2 cᵢ·cᵢ` is not zero. Above
   `‖c‖ ≈ 1e4` this changes the kernel values visibly, and above ~1e8 the
   diagonal of `K` is 0 or `s` depending on the sign of BLAS rounding noise.
   botorch_rs keeps the exact values. The noise depends on the BLAS
   summation order (no sequential order, with or without FMA, reproduces
   MKL's here), so it cannot be reproduced; BoTorch's own fits vary between
   twins where it matters (seeds 505 and 856 of `legacy_gamma` below). The
   same holds for last-bit differences of inputs computed by a learned
   transform (`Warp`'s Kumaraswamy CDF is not bitwise torch's): divided by a
   lengthscale below ~1e-170 they alone decide whether squared distances
   overflow (e.g. a saturated warp gives a constant column in one library
   and not quite in the other), and with subnormal lengthscales
   botorch_rs's uncentered scaling `x/ℓ` overflows where GPyTorch's Matérn
   kernel, which centers first, does not (gradients NaN, values equal).
3. **Gradient NaNs of GPyTorch's autograd path while the covariance is
   finite** are not reproduced: narrow bands around the overflow thresholds
   for RBF / Matérn kernels with several lengthscales; for the `α` of
   rational quadratic kernels (`0 · inf` in `pow`'s backward when
   `res / (2α)` overflows) and the periods of periodic kernels (botorch_rs's
   closed-form period derivative is NaN where torch's chain gives 0) they
   are more frequent. The covariance's NaN entries are reproduced for these
   kernels.

## Existing fixtures

Instrumenting the fits of all Rust tests shows non-finite losses or
gradients only in `kernels_6a_parity.rs` (kernel trees, which passed them to
L-BFGS-B before as well) and in the new `fit_nonfinite.rs`; all other
fixture fits never reach the changed code paths, and their outcomes are
unchanged. In `kernels_6a_parity.rs`, one twin of `product_scale_d3` visits
points where GPyTorch's covariance is NaN but botorch_rs's exact one was
finite: with the NaN injection its first attempt succeeds like BoTorch's
(botorch_rs's first-attempt successes went from 63 to 64 of 65; BoTorch:
65 of 65). Every other printed outcome of the 6a test is identical with and
without the injection.

## Scan of seeds 7..1006 (all eight `CONFIGS`, 8000 fits)

`scratchpad/fitflake/scan3.py` (BoTorch main, single-threaded torch): every
seed where botorch_rs raises, BoTorch raises, or botorch_rs's loss exceeds
BoTorch's by more than 1e-6, with the twin criterion on 16 twins.

| config, seed | before | after | BoTorch's spread at its best (twins) | botorch_rs best vs BoTorch best (twins) | class |
| --- | --- | --- | --- | --- | --- |
| `custom_scale_rbf` 514 | +2.2e-1 | 3e-16 | – | – | fixed (overflow NaN) |
| `legacy_gamma` 104 | raises | -3e-14 | – | – | fixed (zero lengthscale) |
| `legacy_gamma` 203 | raises | 7e-13 | – | – | fixed |
| `legacy_gamma` 417 | raises | -3e-11 | – | – | fixed |
| `legacy_gamma` 665 | raises | 8e-14 | – | – | fixed |
| `legacy_gamma` 667 | raises | -7e-16 | – | – | fixed (zero outputscale, `+inf`) |
| `legacy_gamma` 753 | raises | 3e-14 | – | – | fixed |
| `custom_scale_rbf` 400 | +2.2e-6 | +2.2e-6 | 2.3e-6 | -4e-15 | within BoTorch's twin spread (flat ridge) |
| `legacy_gamma` 505 | +3.3e-6 | +3.3e-6 | 6.7e-5 | +3.3e-6 (tol 6.7e-4) | within BoTorch's twin spread (rounding-noise regime, deviation 2) |
| `legacy_gamma` 856 | +2.3e-5 | +2.3e-5 | 5.2e-5 | +2.3e-5 (tol 5.2e-4) | within BoTorch's twin spread (rounding-noise regime, deviation 2) |

No other seed of any configuration fails; no fit raises. The same scan
after merging main (phase 6b) gives the identical three lines.

**torch 2.14.1** (CI's parity job; gradients at clamp bounds changed in
pytorch#191142), same scan with BoTorch main: four seeds, all within
BoTorch's twin spread, no fit raises:

| config, seed | gap | BoTorch's spread at its best (twins) | botorch_rs best vs BoTorch best (twins) |
| --- | --- | --- | --- |
| `custom_scale_rbf` 400 | +2.2e-6 | 2.3e-6 | -5.8e-15 (tol 2.3e-5) |
| `custom_scale_rbf` 933 | +3.2e-6 | 3.3e-6 | -1.5e-10 (tol 3.3e-5) |
| `legacy_gamma` 505 | +3.3e-6 | 6.7e-5 | +3.3e-6 (tol 6.7e-4) |
| `legacy_gamma` 856 | +2.3e-5 | 5.2e-5 | +2.3e-5 (tol 5.2e-4) |

(`custom_scale_rbf` 933 is a flat ridge on which BoTorch 2.14's first
outcome lands at the better end.) BoTorch's losses at the regression seeds
are the same under both torch versions to 4e-13, so the constants of
`fit_nonfinite.rs` hold for both; `test_parity_fit_nonfinite.py` and
`test_parity.py` pass under torch 2.14.1 with the same worst errors.

## Equivalence tables

Rust (`botorch_core/tests/fit_nonfinite.rs`, values measured with BoTorch /
GPyTorch, no fixture files):

| quantity | tolerance | result |
| --- | --- | --- |
| MLL at raw lengthscales -800, -674 (zero, `1e-293`) | NaN (`Nan`, 144 of 144 entries as linear_operator counts) | NaN |
| MLL at a raw lengthscale of -251 (`1e-109`) | finite, like GPyTorch | finite |
| MLL at a raw outputscale of -3074 (zero) | `-inf`, gradient NaN only for the outputscale | as expected |
| user-specified zero lengthscale / outputscale | `InvalidArgument` | raises |
| L-BFGS-B on `x - ln x` with NaN for `x <= 0` | scipy's 27 evaluation points, `x`, `f`, `nit` bitwise | identical |
| fits of `legacy_gamma` 104 and 667 and `custom_scale_rbf` 514 | BoTorch's loss within 1e-6, first attempt | `< 1e-10` |
| `nonfinite` unit tests (overflow, zero lengthscales, `inf · 0` derivatives, RQ / cosine / periodic) | GPyTorch's classes | pass |

Live (`test_parity_fit_nonfinite.py`, BoTorch main):

| quantity | tolerance | worst |
| --- | --- | --- |
| loss class (finite / NaN / `+inf`) at random extreme raw hyperparameters, 60 points each: RBF and Matérn-1/2, 3/2, 5/2 with 1..3 inputs, kernel trees with a shared and with ARD lengthscales, RQ, periodic, cosine | equal | equal at all 1242 classified points (cosine: 18 points where BoTorch or botorch_rs raise `NotPSDError`, whose jitter decision depends on rounding) |
| NaN pattern of the gradient (RBF, Matérn, trees) | equal | equal |
| loss class at random extreme raw hyperparameters of `Warp` models (concentrations `1e-4 .. 1e12`, lengthscales down to `1e-169`; legacy Matérn-5/2 with 1 and 2 inputs, RBF tree), 60 points each | equal | equal (180 points) |
| finite values at moderate points (lengthscales `>= 1e-3`) | 1e-9 | 4.1e-15 |
| finite gradients at moderate points | 1e-8 | 1.2e-14 |
| regression seeds (`custom_scale_rbf` 514, `legacy_gamma` 104, 203, 417, 665, 667, 753): botorch_rs's loss minus BoTorch's | 1e-6 (twin criterion) | 3.0e-11 |
| sweep, 200 random data seeds over the eight `CONFIGS`: loss above BoTorch's | 1e-6, twin criterion where above | 1.1e-8 (1 seed decided by twins: within, 2e-3 of its tolerance) |
| batched fit (two outputs, one through a NaN covariance): element loss above BoTorch's unbatched fit | 1e-6 | 1.6e-15 (BoTorch's batched fit is 2.72 worse, bug 1) |

`test_parity.py::test_fit_parity` (deterministic seeds, 16 twins, all decided
by stage 1 of the comparison below): best loss above BoTorch's best 3.0e-8
of the tolerance, BoTorch's spread at its best optimum up to 9.1e-9, no
failed first attempts.

## Comparing fits

BoTorch's own fit depends on rounding: on twins whose training inputs are
perturbed at 1e-14 .. 1e-11, long L-BFGS-B runs end in other optima or stop
early on flat regions. Every fit test therefore compares botorch_rs's fits
with BoTorch's on the same twins, first attempts only (retries start from
prior samples drawn by different random number generators).

### The flake

The fit tests bounded one sample: botorch_rs's fit of the model itself had
to end no higher than BoTorch's worst of 5 (live tests) or 17 (fixtures)
outcomes. Wherever both libraries share a tail of outcomes, this fails by
chance. CI's Linux parity job failed
`test_parity_6b_transforms.py::test_fit_from_identical_initial_values[warp_priors_bounds]`
(81.9 × the tolerance). A diagnostic on Linux (torch 2.14.1, the model and
64 twins in both libraries) showed that both libraries draw their outcomes
from the same distribution. Most fits reach -0.97612859, and a tail stops
early on a flat region near -0.97592 (relative-reduction stop): BoTorch 6 of
65, botorch_rs 4 of 65. The two objectives agree at both optima to 1e-12.
botorch_rs's fit of the model itself happened to land in the tail. With a
shared tail of probability `p`, the bound fails with probability about
`p (1 - p)^5`, 5% here. The median and quartile comparison used before for
chaotic cases (on 17 outcomes) failed by chance in the same way: on CI for
7a `default_t3_d2` on Linux and macOS, and in 3 of 11 emulated platform
replicates.

### The two-stage procedure

`parity_helpers.compare_twin_fits` (built on `two_stage_fit_comparisons`) and
`tests/common/mod.rs::compare_fits` (built on `two_stage_fit_comparison`)
replace the former checks in every fit test:

- live tests, 4 twins in stage 1: 6a, 6b, 7a (`MultiTaskGP`,
  `KroneckerMultiTaskGP`), 7b (per model element; the all-categorical mixed
  model has no twins and only stage 1), and LCEMGP (the Unity models and
  the `Interval` `SingleTaskGP`s, with BoTorch's fit with L-BFGS-B's
  variables reversed as one more BoTorch outcome); `test_parity.py` and the
  `test_parity_fit_nonfinite.py` sweep with 16 twins, its regression seeds
  with 4; `test_equivalence.py::test_fit` with none (the model itself);
- Rust fixture tests, 16 twins in stage 1: 6a, 6b, 7a (both models), 7b and
  LCEMGP (the fixtures record no attempts there, so all fits count, as
  before).

1. **Stage 1** (deterministic, on the twins fitted anyway): botorch_rs's
   best loss must reach BoTorch's best, and botorch_rs's fit of the model
   itself (live tests; if its first attempt succeeded) or every one of its
   stage-1 fits (fixture tests) must not end above BoTorch's worst, both
   within `max(1e-6, 10 ×` BoTorch's spread at its best optimum`)` (the
   spread of BoTorch's outcomes within 1e-4 of its best). If both hold, the
   case passes.
2. **Stage 2** (only if stage 1 fails): both libraries fit twins up to 64
   (the Rust fixtures hold BoTorch's outcomes on 64 twins, and botorch_rs
   fits the same twins), and on all 65 outcomes per library:
   - a one-sided Mann–Whitney U test at level `1e-3` that botorch_rs's
     losses are not stochastically larger than BoTorch's: `U = #{(r, b): r >
     b + 1e-6} + ½ #{|r - b| <= 1e-6}`, `z = (U - n₁n₂/2 - ½) /
     sqrt(n₁n₂(n₁+n₂+1)/12)`, fail if `z > 3.09`;
   - a rank bound: botorch_rs's best loss must not exceed BoTorch's `k`-th
     smallest by more than 1e-6, with the smallest `k` such that `C(n_bt, k)
     / C(n_rs + n_bt, k) <= 1e-3` (`k = 10` for 65 against 65 and against
     66). If at least `k` of BoTorch's outcomes lie in its best cluster,
     botorch_rs's best lies in it as well.

A case decided by stage 2 prints its diagnostics: stage 1's excesses, `z`,
`k`, BoTorch's `k`-th best and the sorted outcomes of both libraries. pytest
shows them for a failing test and repeats them in the assertion message; the
Rust tests print them before the error table, which lists the
`fit (stage 2)` rows. The pytest summary and the Rust tables record stage
1's quantities for cases decided by stage 1 and stage 2's otherwise.

The second check of stage 1 differs because stage 2 costs differ. In the
fixture tests, stage 2 costs only botorch_rs's fits (fractions of a second), and
checking every stage-1 fit sends more regressions to stage 2 (power table
below). In the live tests, stage 2 needs up to 64 BoTorch fits (18 to 32 s
each for 7a `default_t3_d2`). Checking every stage-1 fit there sent
`default_t3_d2` to stage 2 on Windows (one of its 5 botorch_rs fits 1.48 ×
the tolerance above BoTorch's worst of 5; stage 2 then passed with `z =
-0.71`), and its test took 1074 s instead of about 80 s. The live tests
therefore keep the former check of the model's own fit.

### False-positive rate

Under the null hypothesis that botorch_rs's outcomes are another sample of
BoTorch's (exchangeable values), stage 2 alone fails with probability at
most `2e-3`: each check at most `1e-3`. Ties within 1e-6 and the positive
dependence within a twin pair (fitted on the same inputs) only reduce the
variance of `U` and the chance that the `k` smallest values all come from
BoTorch, so both checks are conservative. Stage 2 uses a fixed set of twins
(0..64), whatever stage 1 found, and the procedure fails only if both stages
fail:

`P(fail) = P(stage 1 fails and stage 2 fails) <= P(stage 2 fails) <= 2e-3`.

Stage 1 only adds a way to pass, so the procedure is never less reliable
than stage 2 alone, for any stage-1 check.

### Power

`P(fail) = P(stage 2 fails) - P(stage 1 passes and stage 2 fails)`. A
regression that moves botorch_rs's outcomes beyond stage 1's tolerance on
the stage-1 twins fails stage 1 with certainty and then stage 2 as often as
stage 2 alone does. This covers botorch_rs never reaching BoTorch's best
optimum, every fit ending higher, and every fit stopping early. A regression
that only makes an existing tail or optimum more frequent passes stage 1
whenever botorch_rs's fits on the stage-1 twins happen to stay within
BoTorch's range. It is then caught less often than by stage 2 alone. With
the live tests' check of the model's own fit, it is caught at most as often
as by the former one-sample bound (which failed on the same events, and on
chance ones as well). Chaotic cases, which the previous version sent
straight to stage 2's checks, now pass when stage 1 passes: a doubled spread
fails 40% of the time in the live tests instead of 59%, the price of not
fitting 64 twins for every chaotic case. A shift smaller than stage 1's
tolerance passes by design.

Monte Carlo (`scripts/fit_comparison_sim.py`: 200000 replications under the
null hypothesis, 20000 per regression; failure rates; "own" checks the
model's own fit in stage 1 as the live tests do, "every" every stage-1 fit
as the fixture tests do; `n₁` outcomes per library in stage 1, 65 in stage
2):

| outcomes of BoTorch and botorch_rs | stage 2 alone | two-stage, own, `n₁ = 5` | own, `n₁ = 17` | every, `n₁ = 17` |
| --- | --- | --- | --- | --- |
| *null hypothesis (same distribution)* | | | | |
| one optimum (spread 1e-9) | 0 | 0 | 0 | 0 |
| tail of early stops, 8% (Linux, 6b warp) | 0 | 0 (stage 1 fails 7.1%) | 0 (4.3%) | 0 (44%) |
| tail of early stops, 25% | 0.003% | 0.001% | 0.001% | 0.002% |
| chaotic (exponential, scale 1e-3) | 0.17% | 0.097% | 0.025% | 0.088% |
| four optima, 30/30/20/20% | 0.059% | 0.021% | 0.006% | 0.011% |
| four optima, best rare 5/45/30/20% | 0.045% | 0.025% | 0.025% | 0.026% |
| *regressions (botorch_rs worse)* | | | | |
| optimum never reached (tail 8% to 100%) | 100% | 100% | 100% | 100% |
| optimum 2e-6 higher | 100% | 100% | 100% | 100% |
| fits stopped early (`maxiter = 5`) | 100% | 100% | 100% | 100% |
| best optimum never reached (four optima, 30% to 0%) | 99.8% | 85% | 99.6% | 99.6% |
| tail 8% to 70% | 100% | 59% | 32% | 83% |
| tail 8% to 40% | 46% | 16% | 11% | 38% |
| tail 8% to 20% | 0.05% | 0.02% | 0.02% | 0.05% |
| four optima, best 30% to 10% | 29% | 18% | 7.0% | 7.6% |
| chaotic, scale 1e-3 to 2e-3 | 59% | 40% | 18% | 53% |
| chaotic, scale 1e-3 to 1.5e-3 | 12% | 8.3% | 3.0% | 10% |

Real fits: botorch_rs stopped after 5 L-BFGS-B iterations fails both stages
in `test_fit_comparison.py::test_degraded_fits_fail_both_stages`
(`test_parity.py`'s `default` configuration: stage 1 8.3e3 × the tolerance;
stage 2 `z = 9.83`, the largest possible value, every botorch_rs outcome
above every BoTorch outcome; best 8.3e-3 above BoTorch's 10-th best) and in
`kernels_6a_parity.rs::degraded_fits_fail_the_two_stage_fit_comparison`
(`rbf_shared_d3` and `matern32_scale_d2` against their 64-twin fixtures:
`z = 9.83`, best 7.5e-2 and 7.2e-2 above BoTorch's 10-th best).

### The Linux diagnostic

`test_fit_comparison.py::test_linux_warp_outcomes_fail_stage_one_and_pass_stage_two`
and `kernels_6a_parity.rs::linux_warp_outcomes_pass_the_two_stage_fit_comparison`
hold its 65 + 65 outcomes. Stage 1 with 4 twins fails: botorch_rs's fit of
the model itself ends 213 × the tolerance above BoTorch's worst of 5. Stage
2 passes: `z = -0.452`, `k = 10`, and botorch_rs's best -0.9761285871 equals
BoTorch's 10-th best (at the printed 10 decimals).

### Fixtures

`scripts/generate_fixtures.py` (BoTorch main da00e0401, torch 2.13,
single-threaded) now records 64 twins for every fit case:
`fixtures_6a_kernels.FIT_TWINS` (used by the 6a, 6b, 7a and 7b generators)
and `fixtures_lcemgp.FIT_TWINS` (the LCEMGP Monte-Carlo twins stay at 16).
The regenerated fit cases: 6a 21 (the phase took 1120 s), 6b 6 (173 s), 7a 2
`MultiTaskGP` and 1 `KroneckerMultiTaskGP` (1939 s, mostly BoTorch's slow fits
of `mtgp_default_t3_d2`, which had 64 twins before as well, and
`kron_default_m3_d2`), 7b 9 (394 s), LCEMGP 10 (1012 s). The cases that had
64 twins already (6a `index_d3`, `oak_d3`, `periodic_shared_d2`,
`positive_index_d3`, `product_scale_d3`, 7a `default_t3_d2`, 7b `mixed_last`)
are unchanged. A JSON comparison with the previous fixtures (all 102 files of
the five phases): the previous twins are a prefix of the new ones (bitwise),
and every other field is byte-identical (the new file, with its twin arrays
replaced by the old ones, serializes to the old file's bytes).

### Results

Windows, BoTorch main, torch 2.13:

- Rust fixture tests: every fit case passes. Two go to stage 2: 6a
  `positive_index_d3` (`z = -0.68`, botorch_rs's best -0.8031547 below
  BoTorch's 10-th best -0.8030219) and 7b `mixed_last` (`z = 0.79`, best
  0.4448777 below BoTorch's 10-th best 0.4449453). The cases that were
  compared statistically before (6a `index_d3`, `oak_d3`,
  `periodic_shared_d2`, `product_scale_d3`, 7a `default_t3_d2`) pass stage 1
  on their first 16 twins. The fit tests take as long as before (6a 36 s,
  7a 32 s, debug build).
- Live tests (BoTorch main and 0.18.1, the same outcomes): 6a `oak` (`z =
  -0.23`) and `product_scale` (`z = -0.25`; one botorch_rs twin reaches
  0.339, far below BoTorch's best 0.787) go to stage 2; every other fit case
  of `test_parity.py`, `test_parity_fit_nonfinite.py`, 6a, 6b, 7a, 7b, LCEMGP
  and `test_equivalence.py` passes stage 1. The slowest fit tests: 6a `oak`
  90 s and `product_scale` 87 s, 7a `default_t3_d2` 76 s, Kronecker
  `default_m3_d2` 74 s.
- torch 2.14.1 (BoTorch main, CI's parity job): the live fit tests pass, 6a
  `oak` (`z = -0.23`) and `product_scale` (`z = 0.28`) by stage 2, all
  others by stage 1.

Stage 2's checks were measured before as the criterion for chaotic cases
(`scratchpad/fitflake/stability_mt.py`): botorch_rs's fits of 7a
`default_t3_d2` on all 65 twins with the inputs perturbed again at 1e-15
(10 replicates plus the unperturbed run) pass in all 11 (`z` from -1.15 to
+0.96, best 1.0e-5 .. 1.6e-5 below BoTorch's 10-th best), and under torch
2.14.1 in all 11 against fixtures regenerated with torch 2.14.1 (`z` from
-0.17 to +1.89). The CI outcomes reported for Linux (best 0.0924141922) and
macOS (best 0.0924154162) lie below BoTorch's 10-th best of the fixture's 65
outcomes. BoTorch's own outcomes shift between the torch versions by `z =
0.80` (`default_t3_d2`; other cases `|z| <= 0.75`).

**`default_t3_d2` and the non-finite fix**: botorch_rs's path for this case
(1815 evaluations, raw `var` down to -1.1e4 and `covar_factor` up to 7.4e3)
visits no non-finite point, and BoTorch's objective along that path agrees
with botorch_rs's to 4.5e-15 (values) and 1.2e-13 (gradients): the case is
chaotic through round-off amplification alone, and this fix does not change
its outcomes.

## Not done

- Finite kernel values in GPyTorch's rounding-noise regime (deviation 2)
  and autograd-path gradient NaNs while the covariance is finite
  (deviation 3).
- Kernels whose arithmetic does not go through `sq_dist` (linear,
  polynomial, index, categorical, exponential decay, infinite-width BNN,
  orthogonal additive): no emulation; their overflow behaviour at extreme
  hyperparameters was not compared.
- `LCEMGP`, `MultiTaskGP` and `KroneckerMultiTaskGP` inject GPyTorch's NaN
  entries but were not probed live at extreme points (their default modules
  bound the lengthscales; `KroneckerMultiTaskGP`'s MLL in GPyTorch uses
  eigendecompositions of the Kronecker factors, whose NaN handling may
  differ from the Cholesky's `NanError`). The torch-order prior derivatives
  cover `SingleTaskGP` (standard kernels and trees, the noise prior) only.
- The fit comparisons use first attempts only. botorch_rs's first attempt
  fails more often than BoTorch's on 6a `periodic_shared_d2` (on 32 of its
  65 twins, BoTorch's on 2), which no test checks; not investigated.
- The LCEMGP fixtures record no attempts, so their comparison counts all
  fits (as before); the live LCEMGP tests compare first attempts.

## BoTorch bugs found

1. **Batched independent fitting couples the elements through NaN
   covariances** (`botorch/optim/closures/core.py`,
   `BatchedNDarrayOptimizationClosure.__call__`; `fit_gpytorch_mll` of every
   model with `_aug_batch_shape` of more than one element). The closure
   evaluates the trial points of all active elements in one batched forward
   pass; if any element's covariance is NaN, linear_operator raises
   `NanError` for the whole batch and the closure returns NaN values with
   **zero** gradients for every active element. An element that meets this
   at a sound trial point can stop with "CONVERGENCE: NORM OF PROJECTED
   GRADIENT <= PGTOL" and a NaN loss, at hyperparameters far from its
   optimum; the fit reports `SUCCESS` with `fval` NaN. Reproduction (BoTorch
   main da00e0401, GPyTorch 1.15.2, torch 2.13):

   ```python
   import torch
   from botorch.fit import fit_gpytorch_mll
   from botorch.models import SingleTaskGP
   from botorch.models.utils.gpytorch_modules import (
       get_gaussian_likelihood_with_gamma_prior, get_matern_kernel_with_gamma_prior)
   from gpytorch.mlls import ExactMarginalLogLikelihood

   torch.set_num_threads(1)
   DT = torch.float64
   g = torch.Generator().manual_seed(104)
   X = torch.rand(12, 2, generator=g, dtype=DT)
   Y1 = torch.sin(6 * X[:, :1]) + X[:, 1:] ** 2 + 0.05 * torch.randn(12, 1, generator=g, dtype=DT)
   g = torch.Generator().manual_seed(2)
   Y2 = torch.sin(3 * X[:, 1:]) * (1 + torch.rand(1, generator=g, dtype=DT))
   Y2 = Y2 + 0.1 * torch.randn(12, 1, generator=g, dtype=DT)

   def fit(Y):
       bs = torch.Size([Y.shape[-1]]) if Y.shape[-1] > 1 else torch.Size()
       m = SingleTaskGP(X, Y, covar_module=get_matern_kernel_with_gamma_prior(2, batch_shape=bs),
                        likelihood=get_gaussian_likelihood_with_gamma_prior(batch_shape=bs))
       mll = ExactMarginalLogLikelihood(m.likelihood, m)
       fit_gpytorch_mll(mll)
       m.train()
       return (-mll(m(*m.train_inputs), m.train_targets)).detach().reshape(-1).tolist()

   print(fit(torch.cat([Y1, Y2], -1)))  # [1.1416, 4.1846]: output 1 stopped at a NaN point
   print(fit(Y2))                       # [1.4640]: the same output fitted alone
   ```

   Output 0 visits a zero lengthscale (NaN covariance) in its 30th
   evaluation; output 1, evaluated in the same batch, gets NaN with zero
   gradients and stops after 9 iterations. Fix: on `NanError`, evaluate the
   active elements one at a time (or mask the failing elements) and give
   only the failing ones NaN, with NaN gradients as in the unbatched
   closure (meta-pytorch/botorch#3450).
