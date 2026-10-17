# Fits through extreme hyperparameters (`fit-nonfinite`)

Draft for README / `docs/api_coverage.md`. Rust:
`botorch_core/src/kernels/nonfinite.rs` (new), `fit.rs`, `mll.rs`,
`kernels.rs`, `kernels/eval.rs`, `kernels/covar.rs`, `priors.rs` and the
marginal log likelihoods of `LCEMGP`, `MultiTaskGP` and
`KroneckerMultiTaskGP`. Tests: `botorch_core/tests/fit_nonfinite.rs`,
`botorch_pyo3/tests/test_parity_fit_nonfinite.py`,
`botorch_pyo3/tests/test_parity.py`; the statistical comparison of chaotic
fits in `botorch_pyo3/tests/parity_helpers.py` and
`botorch_core/tests/common/mod.rs`.

## Supported API (rows for README)

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
   twins where it matters (seeds 505 and 856 of `legacy_gamma` below).
3. **Gradient NaNs of GPyTorch's autograd path while the covariance is
   finite** are not reproduced: narrow bands around the overflow thresholds
   for RBF / Matérn kernels with several lengthscales; for the `α` of
   rational quadratic kernels (`0 · inf` in `pow`'s backward when
   `res / (2α)` overflows) and the periods of periodic kernels (botorch_rs's
   closed-form period derivative is NaN where torch's chain gives 0) they
   are more frequent. The covariance's NaN entries are reproduced for these
   kernels.

## Existing fixtures

All fixture comparisons are unchanged (printed outcomes identical), except
one: in `kernels_6a_parity.rs`, `product_scale_d3`, a twin's fit visits
points where GPyTorch's covariance is NaN (scaled inputs that overflow in
the product kernel's RBF factor). With the NaN injection, its first attempt
succeeds like BoTorch's: botorch_rs's first-attempt successes went from 63
to 64 of 65 (BoTorch: 65 of 65); every other printed outcome of the 6a, 7a,
7b, LCEMGP and core fixture tests is identical with and without the
injection. Instrumenting the fits of all Rust tests shows non-finite values
only in `kernels_6a_parity.rs` (kernel trees, which passed them to L-BFGS-B
before as well) and in the new `fit_nonfinite.rs`.

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

No other seed of any configuration fails; no fit raises.

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
| finite values at moderate points (lengthscales `>= 1e-3`) | 1e-9 | 4.1e-15 |
| finite gradients at moderate points | 1e-8 | 1.2e-14 |
| regression seeds (`custom_scale_rbf` 514, `legacy_gamma` 104, 203, 417, 665, 667, 753): botorch_rs's loss minus BoTorch's | 1e-6 (twin criterion) | 3.0e-11 |
| sweep, 200 random data seeds over the eight `CONFIGS`: loss above BoTorch's | 1e-6, twin criterion where above | 1.1e-8 (1 seed decided by twins: within, 2e-3 of its tolerance) |
| batched fit (two outputs, one through a NaN covariance): element loss above BoTorch's unbatched fit | 1e-6 | 1.6e-15 (BoTorch's batched fit is 2.72 worse, bug 1) |

`test_parity.py::test_fit_parity` (deterministic seeds, 16 twins): best loss
above BoTorch's best 3.0e-8 of the tolerance, BoTorch's spread at its best
optimum up to 9.1e-9, no failed first attempts.

## Chaotic fits: a statistical criterion

Where BoTorch's own fit outcomes on twins spread over more than 1e-3 (or,
in the 7b tests, fewer than a quarter of them lie within 1e-4 of the best),
the fit tests compared botorch_rs's best with BoTorch's median and its
median with BoTorch's upper quartile, on 17 outcomes each. That is not a
sound test: with outcomes drawn from the same distribution it fails often.
Main CI failed it on Linux and macOS for `multitask_7a_parity.rs`, case
`default_t3_d2` (errors 6.6e-5 and 1.2e-3), and under emulated platform
differences (below) it fails 3 of 11 times on Windows.

The new criterion (`parity_helpers.compare_chaotic_fits`,
`tests/common/mod.rs::compare_chaotic_fits`), used consistently in the Rust
6a, 7a (`MultiTaskGP` and `KroneckerMultiTaskGP`) and 7b fixture tests, the
Python 6a, 7a and 7b live tests and `test_parity.py`:

- **64 twins** (plus the model itself) for chaotic cases: the fixture
  generators (`fixtures_6a_kernels.is_chaotic`, used by 6a, 7a and 7b) add
  twins 17..64 when the first 17 outcomes are chaotic or multimodal; the
  live tests run `parity_helpers.CHAOTIC_TWINS = 64` twins.
- **One-sided Mann–Whitney U test** at level `1e-3` that botorch_rs's
  first-attempt losses are not stochastically larger than BoTorch's:
  `U = #{(r, b): r > b + 1e-6} + ½ #{|r - b| <= 1e-6}`, `z = (U - n₁n₂/2 -
  ½) / sqrt(n₁n₂(n₁+n₂+1)/12)`, fail if `z > 3.09`.
- **Best-cluster bound**: botorch_rs's best loss must not exceed BoTorch's
  `k`-th smallest by more than 1e-6, with the smallest `k` such that
  `C(n_bt, k) / C(n_rs + n_bt, k) <= 1e-3` (`k = 10` for 65 against 65,
  `k = 9` for 17 against 17). If at least `k` BoTorch outcomes lie in its
  best cluster, botorch_rs's best lies in it as well.

False-positive rate: under the null hypothesis that botorch_rs's outcomes
are another sample of BoTorch's (exchangeable values), each check fails with
probability at most `1e-3`, a chaotic case at most `2e-3`. Ties within
1e-6 and the positive dependence within a twin pair (both libraries fit the
same perturbed inputs) only reduce the variance of `U` and the chance that
the `k` smallest values all come from BoTorch, so both checks are
conservative. Measured: Monte Carlo with continuous outcomes, 400000
replications, `z > 3.09` in 0.090% (65 against 65), 0.067% (17), 0.090%
(33); with a discrete four-mode distribution plus noise and ties at 1e-6,
20000 replications: 0.13% (Mann–Whitney) and 0.085% (rank bound).

Results on the regenerated fixtures (Windows): `default_t3_d2` z = 0.86,
botorch_rs's best 0.0924066 below BoTorch's best 0.0924093; 6a chaotic
cases (`index_d3`, `oak_d3`, `periodic_shared_d2`, `positive_index_d3`,
`product_scale_d3`) z <= 0.81; 7b `mixed_last` z = 0.79; every best within
BoTorch's 10-th best.

**Emulated platform differences** (`scratchpad/fitflake/stability_mt.py`):
botorch_rs's fits of `default_t3_d2` on all 65 twins with the inputs
perturbed again at 1e-15 (10 replicates, plus the unperturbed run): the new
criterion passes in all 11 (z from -1.15 to +0.96, best 1.0e-5 .. 1.6e-5
below BoTorch's 10-th best 0.0924190), the old one on 17 outcomes fails in
3 of 11. The CI outcomes reported for Linux (best 0.0924141922) and macOS
(best 0.0924154162) lie below BoTorch's 10-th best of the 65 fixture
outcomes, so they pass the best-cluster bound; the Mann–Whitney test needs
their full samples, which match BoTorch's distribution if the platforms'
outcomes are statistically consistent with it.

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
       return (-mll(m(X), m.train_targets)).detach().reshape(-1).tolist()

   print(fit(torch.cat([Y1, Y2], -1)))  # [1.1416, 4.1846]: output 1 stopped at a NaN point
   print(fit(Y2))                       # [1.4640]: the same output fitted alone
   ```

   Output 0 visits a zero lengthscale (NaN covariance) in its 30th
   evaluation; output 1, evaluated in the same batch, gets NaN with zero
   gradients and stops after 9 iterations. Fix: on `NanError`, evaluate the
   active elements one at a time (or mask the failing elements) and give
   only the failing ones NaN, with NaN gradients as in the unbatched
   closure.
