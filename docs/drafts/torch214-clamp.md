# Subgradients at clamp bounds (PyTorch 2.14)

PyTorch 2.14.0 changed the gradients of the `clamp` family exactly at its
bounds (pytorch#191142, release notes, section "Autograd"). botorch-rust
follows the PyTorch ≥ 2.14 convention under every torch version. This draft
records the change, the audit of every clamp-like operation on a
differentiable path of the BoTorch, GPyTorch and linear_operator code that
botorch_rs ports, the implementation, the tests and the results under torch
2.13 and 2.14.1. Reference: BoTorch main at da00e0401 and BoTorch 0.18.1,
GPyTorch 1.15.2, linear_operator 0.6.1.

## The change

From the release notes, checked with a probe of every operation under torch
2.13.0 and 2.14.1 (reverse and forward mode):

| Operation | torch ≤ 2.13 at the bound | torch ≥ 2.14 at the bound |
|---|---|---|
| `clamp` / `clamp_min` / `clamp_max` / `clip` (and in-place variants) with a **scalar** bound (Python number) | input gradient 1 (closed interval) | 0 (open interval; minimum-norm subgradient) |
| the same with a **tensor** bound, tie `x == bound` | all to the input `(1, 0)` | split `(½, ½)`, also when the bound does not require a gradient (the input gets ½) |
| tensor bounds with `lo == hi == x` | `(1, 0, 0)` | `(1, 0, 0)` (unchanged) |
| tensor bounds with `lo > hi` | all to `hi` | all to `hi` (unchanged) |
| `fmin` / `fmax` at a tie | all to one argument | split `(½, ½)` |
| NaN input, scalar bounds | 0 | 0 |
| NaN input or bound, tensor bounds | 0 | mostly 1 to the input (irregular) |
| forward-mode AD of all of the above | as reverse mode | as reverse mode |
| `torch.minimum` / `torch.maximum`, binary `torch.max` / `torch.min` | `(½, ½)` | unchanged |
| `max(dim)` / `min(dim)` reductions | to the first index | unchanged |
| `amax` / `amin` / full `max()` / `min()` | split evenly | unchanged |
| `relu`, `hardtanh`, `threshold`, `abs`, `where`, `softplus`, `sort`, `topk`, `cummax`, `logsumexp` | | unchanged |

Values are unchanged (the forward pass is the same in both versions; a
signed zero survives `clamp_min(-0.0, 0.0)` in both).

## Audit

Every `clamp` / `clamp_min` / `clamp_max` / `clip` / `fmin` / `fmax` and every
`minimum` / `maximum` / relu-like operation in BoTorch main, GPyTorch 1.15.2
and linear_operator 0.6.1 (268 matches in BoTorch alone), with botorch_rs's
counterpart. "At the bound?" says whether inputs can sit exactly at the
bound in practice; "measure zero" means only by an exact floating-point
coincidence. "Old / new" is the input's subgradient at the bound under torch
≤ 2.13 / ≥ 2.14. Every row marked "changed" now applies the ≥ 2.14 rule;
values are bitwise unchanged everywhere, and so are derivatives away from
the exact bounds.

### Kernels

| BoTorch / GPyTorch | botorch_rs | At the bound? | Old / new | botorch_rs |
|---|---|---|---|---|
| `LinearTruncatedFidelityKernel.forward`: `x.index_select(fidelity).clamp(0, 1)` of `x1` and `x2` (`models/kernels/linear_truncated_fidelity.py:208-221`, scalar bounds) | `kernels/eval.rs` `ltf_bias` (cross covariances, `diag_with_grad`) | **Yes**: fidelities 0 and 1 are the usual bounds (target fidelity 1, L-BFGS-B iterates on the bounds) | 1 / 0 | changed: the fidelity derivative passes on `0 < v < 1` only (was `0 ≤ v ≤ 1`) |
| `DownsamplingKernel`: `1 - x`, `pow(1 + power)` (no clamp) | `eval.rs` `Downsampling` | yes | no clamp | unchanged (control test) |
| `ExponentialDecayKernel` → GPyTorch `dist(x1, -x2)` = `cdist(...).clamp_min(1e-15)` (`gpytorch/kernels/kernel.py:58`) | `eval.rs` `ExpDecay` `cross_grad` / `contract` | measure zero (a zero distance at `x1 = x2 = 0` lies strictly below 1e-15) | 1 / 0 | changed: `dist > 1e-15` (was `≥`) |
| GPyTorch `sq_dist(...).clamp_min_(0)` (`kernel.py:49`; RBF, RQ, Matérn on the training covariance) | distances computed directly as `Σ (x − y)²` (never negative, no clamp) | only identical rows, where the expansion's gradient `2 (x1 − x2)` is 0, or rows equal up to rounding | rounding-level only | unchanged |
| GPyTorch `dist`: `cdist(...).clamp_min(1e-15)` and `sq_dist(...).clamp_min_(1e-30).sqrt_()` (`kernel.py:58,60`; Matérn ½, 3/2, 5/2, periodic per dimension, cosine) | `kernels.rs` `KernelKind::value_and_factor`; `eval.rs` periodic / cosine `cross_grad` and `contract` | measure zero (identical points lie strictly below) | 1 / 0 | changed: `r ≤ 1e-15` gives 0 (was `<`); periodic / cosine pass on `> 1e-15` |
| `InfiniteWidthBNNKernel`: `fraction.clamp(-1 + eps, 1 - eps)` (`infinite_width_bnn.py:141`) | `eval.rs` `bnn_pair` | measure zero (identical points give 1, strictly clamped) | 1 / 0 | changed: open interval |
| `OrthogonalAdditiveKernel.normalizers`: `.clamp(eps)` (`orthogonal_additive_kernel.py:602`; hyperparameter gradient) | `eval.rs` `OakState::compute_normalizers` | measure zero | 1 / 0 | changed: clamped at `total ≤ eps` (was `<`) |
| `PositiveIndexKernel` setter, `map_saas` initialization (`positive_index.py:155-166`, `map_saas.py:102,125`) | `kernels/tree.rs` sampling / initialization | not differentiated | | unchanged |
| GPyTorch `HammingIMQKernel`, inducing-point / grid / interpolation kernels | not ported | | | |

### Priors and constraints

| BoTorch / GPyTorch | botorch_rs | At the bound? | Old / new | botorch_rs |
|---|---|---|---|---|
| `SmoothedBoxPrior.log_prob`: `((x - c).abs() - r).clamp(min=0)` | `priors.rs` `SmoothedBox` | yes (a value on the box edge) | the density's derivative `-z / σ²` vanishes at `z = 0`, so the result is 0 either way | unchanged |
| GPyTorch constraints (softplus / sigmoid transforms), BoTorch `optim/core.py:235` (`.data` clamp) | `constraints.rs` | no clamp on a gradient path | | unchanged |

### Transforms

| BoTorch | botorch_rs | At the bound? | Old / new | botorch_rs |
|---|---|---|---|---|
| `kumaraswamy_warp`: `clamp(X * (1 - 2 eps) + eps, eps, 1 - eps)` (`models/transforms/utils.py:182`; `Warp`) | `transforms/utils.rs` `clamp_unit` (used by `Warp` in `input_steps.rs`) | **Yes**: `x = 0` maps to `eps` exactly. `x = 1` maps to `1 - eps` exactly only for some `eps` (e.g. powers of two); for the default `1e-7`, `1 · (1 − 2e−7) + 1e−7 < 1 − 1e−7`, so it stays inside | 1 / 0 (input gradient; the concentrations' gradients do not involve the clamp) | changed: open interval |
| `inv_kumaraswamy_warp`: `.clamp(0, 1)` (`utils.py:209`) | `inv_kumaraswamy_warp` | not differentiated (untransform) | | unchanged |
| `InputPerturbation` with bounds: `torch.maximum(torch.minimum(Y, hi), lo)` (`input.py:1621`) | `transforms/input.rs` `dclamp` (½ at ties) | yes | `(½, ½)` in both versions | unchanged |
| `Normalize`, `InputStandardize`, `Round` (`approximate_round`: tanh), `OneHotToNumeric`, `Log10`, `AppendFeatures`, `LearnedFeatureImputation`; outcome `Log`, `Power`, `Bilog` (`sign` / `abs`) | | no clamp | | unchanged |
| outcome `Log`'s delta-method noise `Y.clamp(min=1e-8)` (`outcome.py:810`, training data) | `transforms/outcome.rs` `LOG_NOISE_CLAMP` | not differentiated | | unchanged |

### Posterior variances and covariance roots

| BoTorch / GPyTorch / linear_operator | botorch_rs | At the bound? | Old / new | botorch_rs |
|---|---|---|---|---|
| GPyTorch `MultivariateNormal.variance`: `clamp_min(min_variance)` (1e-10 in float64; `distributions/multivariate_normal.py:381`) | `models/gp.rs` `posterior_q1_with_grad`, `models/model.rs` (default), `models/deterministic.rs` `FixedSingleSampleModel`, `acquisition/objective.rs` (scalarized), `active_learning.rs` (qNIPV), `knowledge_gradient.rs` (analytic value functions), `transforms/outcome.rs` `moments`, `acquisition/analytic.rs`, entropy search (`Dual::clamp_min`, MO `clamp_variance`) | measure zero (a smaller variance has gradient 0 in both versions) | 1 / 0 | changed: clamped at `v ≤ 1e-10` (was `<`) |
| `_mean_and_sigma`: `variance.clamp_min(1e-12)` (`acquisition/analytic.py:112`) | `analytic.rs` `mean_sigma_with_grad`, `output_moments`, posterior-transform path | measure zero | 1 / 0 | changed: `var > max(floor, 1e-12)` |
| analytic EHVI: `variance.clamp_min(1e-9)` (`multi_objective/analytic.py:175`); `cell_upper_bounds.clamp_max(1e10)` (constants) | `multi_objective/analytic.rs` | measure zero | 1 / 0 | changed |
| linear_operator 1 × 1 Cholesky `clamp_min(0).sqrt()` (`_linear_operator.py:524`; Monte-Carlo samples at `q = 1`) | `linalg.rs` `CovarianceRoot::Scalar` | measure zero (`v = 0` exactly) | inf or NaN / 0 | unchanged: botorch_rs already passes the gradient only for `v > 0` |
| linear_operator `symeig` / `root_decomposition` `evals.clamp_min(0)` (896, 2193, 2196), Kronecker log-determinants (`clamp(min=1e-7)`, only for n ≥ 800 or without noise), `root_inv_decomposition`, `minres` | `CovarianceRoot::Eigen` (not differentiated) or not on botorch_rs's paths | | | unchanged |
| GPyTorch `FixedGaussianNoise` `clamp_min(min_noise)` (training noise), `GaussianLikelihood` 1e-8 clamp (approximate GPs), variational strategies | `MIN_FIXED_NOISE` (values), not ported | not differentiated | | unchanged |

### Acquisition functions

| BoTorch | botorch_rs | At the bound? | Old / new | botorch_rs |
|---|---|---|---|---|
| `qExpectedImprovement._sample_forward`: `(obj - best_f).clamp_min(0)` (`monte_carlo.py:437`) | `acquisition/monte_carlo.rs` `SampleUtility::ExpectedImprovement`, `QExpectedImprovement::reduce` | **Yes**: a sample exactly at `best_f` (deterministic models, e.g. a candidate at the incumbent) | 1 / 0 | changed: passes on `obj − best_f > 0` (was `≥`) |
| `qNoisyExpectedImprovement._sample_forward`: `(obj - best_f).clamp_min(0)` (`monte_carlo.py:616`) | `monte_carlo.rs` `noisy_improvement_reduce` | **Yes**: a candidate tying with the best baseline sample (deterministic models) | 1 / 0 | changed |
| `apply_constraints_nonnegative_soft`: `obj.clamp_min(0)` (`utils/objective.py:98`; `ConstrainedMCObjective`, `FeasibilityWeightedMCMultiOutputObjective`) | `mc_objective.rs` `ConstrainedMcObjective`, `multi_objective/objective.rs`; the torch forwards of `ConstrainedMCObjective` and `FeasibilityWeightedMCMultiOutputObjective` (`_clamp_min_zero`) | **Yes**: `obj + infeasible_cost == 0` (the default cost 0 and an objective of 0) | 1 / 0 | changed; the torch forwards use `torch.where(x > 0, x, clamped.detach())`, so they follow ≥ 2.14 under every torch |
| qEHVI: `torch.min(overlap, cell_upper)` then `(… - cell_lower).clamp_min(0)` (`multi_objective/monte_carlo.py:200-206`; also qNEHVI, MOMF and the non-log value function of qHVKG) | `multi_objective/hypervolume.rs` (`minimum_grad_a`, `dlen`) | **Yes**: an objective value on a cell bound (deterministic models) | `min`: ½ / ½ (unchanged); `clamp_min`: 1 / 0 | changed (`clamp_min`); `torch.min` already split |
| qLogEI / qLogNEI / qLogEHVI / qLogNEHVI / qLogNParEGO: `log_fatplus`, `log_softplus`, `fatmax`, `smooth_amax`, `logmeanexp`, `log1mexp` | `logei.rs`, `utils/safe_math.rs` | no clamp (smooth; `log1mexp` uses `where`) | | unchanged |
| analytic EI / LogEI / PI / LogPI / UCB / constrained EI | `analytic.rs` | only `_mean_and_sigma` (above) | | |
| MES / GIBBON / multi-fidelity MES: `clamp_min(CLAMP_LB)` of variances and CDFs (`max_value_entropy_search.py:160,383,407,414,538,558,570,778`) | `max_value_entropy_search.rs` through `Dual::clamp_min` | measure zero (a saturated CDF of 0 lies strictly below 1e-8) | 1 / 0 | changed (`Dual::clamp_min`: zero tangent at equality, value kept) |
| GIBBON: `variance_m.clamp(inv_quad_term * (1 + eps))` (tensor bound, `:627`) | `Dual::clamp_min_var` | measure zero | `(1, 0)` / `(½, ½)` | changed |
| Gumbel max-value sampling `variance.clamp_min(1e-8)` (`:993`) | `max_value_entropy_search.rs` (values) | not differentiated | | unchanged |
| qJES: `clamp_min(CLAMP_LB)` (`joint_entropy_search.py:248,254,331,347,348`) | `joint_entropy_search.rs` (`Dual`) | measure zero | 1 / 0 | changed |
| MO JES / MO MES: variances, noise and box probabilities `clamp_min(CLAMP_LB)` (`multi_objective/joint_entropy_search.py:389-401,502,572,596,701,716,723`; `multi_objective/max_value_entropy_search.py:142,153`) | `multi_objective/joint_entropy_search.rs` `clamp_min_active`, `clamp_variance` | measure zero | 1 / 0 | changed: passes on `v > lo` |
| MO JES: `W = Σ W_j`, `W.clamp_max(1.0)` (`:507,578,706,728`) | `multi_objective/joint_entropy_search.rs` (`w_active`) | **possible**: `W` rounds to exactly 1 when every probability saturates (Φ = 1 in double above z ≈ 8.3), with derivatives of ~1e-15 | 1 / 0 | changed: passes on `W < 1` |
| PES: `log_cdf.clamp(finfo.min, -eps)`, `clamp_min(CLAMP_LB)`, `clamp_min(finfo.min)`, `exp(...).clamp(finfo.min, finfo.max)` (`predictive_entropy_search.py:517,749,750,762,764,768,780`) | `predictive_entropy_search.rs` (`log_cdf_robust_with_grad`, `clamp_min_active`, `lz_active`, `rho_active`) | measure zero (saturated `log Φ = -0.0` lies strictly above `-eps`) | 1 / 0 | changed: open intervals |
| qAnalyticProbabilityOfImprovement → MVNXPB / `bvn`: `.clip(0, 1)` of probabilities (`bvn.py:96,170,237`), `comp_r2.clip(min=0)`, `(Φ(s) − Φ(h)).clip(min=0)` (`:193,203`), `Lii.clip(min=0)` (`linalg.py:115`, `mvnxpb.py:180`), `bounds.clip(std * lim)` (tensor bounds, `mvnxpb.py:86,391`), `safe_math.exp` (`x.clip(max=tensor)`), `safe_math.log` (`x.clip(min=tiny)`) | `utils/mvnxpb.rs` (`clip01`, `Scalar::clamp_min`, `safe_exp`, `safe_log`, the bound clip) | probabilities of exactly 0 or 1 and differences of exactly 0 when both CDFs saturate (derivatives ~1e-15); the rest measure zero | scalar: 1 / 0; tensor: `(1, 0)` / `(½, ½)` | changed (values kept) |
| `bvn._bvnu_taylor`: `torch.maximum(h, k)` (`bvn.py:202`) | `utils/mvnxpb.rs` `bvnu_taylor` | **yes**: two identical candidate points (`r = 1`, `h = k`) | `(½, ½)` in both versions | fixed: botorch_rs gave all to `h` (a deviation independent of torch 2.14) |
| `get_infeasible_cost` (`acquisition/utils.py:258,273`), `initializers.py:1233`, `infer_reference_point` (`hypervolume.py:129,142`), `compute_hypervolume` (`non_dominated.py:452,457`), `NoisyExpectedHypervolumeMixin` (`hypervolume.py:876`), `ConstrainedMaxPosteriorSampling` (`generation/sampling.py:310`), polytope / truncated-normal sampling (`utils/sampling.py:291,1165`), `lin_ess.py`, LOO cross-validation (`cross_validation.py:561`), `optimize_mixed.py`, `fully_bayesian*.py`, `empirical_gps/utils.py` | value-only counterparts (`acquisition/utils.rs`, `utils/multi_objective/*`, `optim/*`, ...) or not ported | not differentiated | | unchanged |
| preference acquisition functions (`preference.py:122,262`), `PairwiseGP` likelihoods | not ported | | | |
| risk measures (`VaR`, `CVaR`, `WorstCase`, `Expectation`, MVaR): `sort`, `topk`, `kthvalue`, `amin`, `mean` | `risk_measures.rs` | no clamp | unchanged ops | unchanged |

### Optimization

| BoTorch | botorch_rs | At the bound? | Old / new | botorch_rs |
|---|---|---|---|---|
| `columnwise_clamp` (`optim/utils/acquisition_utils.py:59`, tensor bounds) in `gen_candidates_scipy` (applied to the initial conditions and the result, outside autograd) and `gen_candidates_torch` (under `torch.no_grad()`); `fix_features` (no clamp); `fmin_l_bfgs_b_batched`'s `np.clip` | `optim/slsqp.rs` `clamp_to_bounds`, `generation/gen.rs` `clamp_columns`, `lbfgsb.rs`; `gen_candidates_torch`'s Python loop (under `no_grad`) | never differentiated | | unchanged |
| L-BFGS-B's projected gradient at the bounds | `lbfgsb.rs` | **yes**: it uses the acquisition gradient at the bound, so the kernel / transform rows above decide whether an iterate on a bound leaves it | | covered by the kernel and transform changes (test below) |

Torch callables (`GenericDeterministicModel`, `CallableAcquisitionFunction`,
torch MC objectives) are differentiated by the installed torch in both
libraries, so they agree with BoTorch under every version.

## Implementation

The derivative masks were inclusive (`≥` / `≤`, `RangeInclusive::contains`)
or the "clamped" tests strict (`v < lo`); they are now exclusive at scalar
bounds. Values are untouched (an equality keeps its own value, e.g. a signed
zero), and NaN behaviour is unchanged (NaN comparisons are false either way).
For the generic forward-mode dual numbers of the entropy-search and MVNXPB
code (`acquisition/dual.rs`): `Scalar::clamp_min` gives a zero tangent at
equality; `Scalar::clamp_min_var` (tensor bound) and the new `Scalar::tie`
split the tangent evenly at a tie. The Python forwards of
`ConstrainedMCObjective` and `FeasibilityWeightedMCMultiOutputObjective`,
which torch differentiates, use `torch.where(x > 0, x, x.clamp_min(0).detach())`
so that they follow the ≥ 2.14 rule under torch ≤ 2.13 too.

No fixture changed: the Rust fixtures (generated under torch 2.13) have no
derivative at an exact clamp bound, and all of them pass unchanged.

## Tests

* `botorch_pyo3/tests/clamp_subgradients.py`: `torch214_clamp_subgradients()`
  makes BoTorch, GPyTorch and linear_operator compute the ≥ 2.14
  subgradients under torch ≤ 2.13. It replaces `torch.clamp` / `clip` /
  `clamp_min` / `clamp_max` and the `Tensor` methods (also in place) by an
  autograd function whose forward is the native clamp (bitwise) and whose
  backward is the ≥ 2.14 rule (scalar bounds: open interval; tensor bounds:
  the even split, the `lo == hi` and `lo > hi` cases as torch 2.14). Under
  torch ≥ 2.14 it does nothing, so the tests compare directly with BoTorch.
  NaN inputs and forward-mode AD are not emulated.
* `botorch_pyo3/tests/test_parity_clamp_bounds.py` (new):
  * the emulation against native torch ≥ 2.14 on 420 random cases with
    ties, infinite, inverted and degenerate bounds and broadcasting (exact
    equality of values and gradients; a one-off check with 2 100 cases also
    found no difference), and against the release-note examples and the
    patched functions / methods (also in place) under every version;
  * `SingleTaskMultiFidelityGP` at fidelities 0 and 1 (one data fidelity,
    two data fidelities, iteration + data fidelity, and the downsampling /
    exponential decay kernels as a control): the MLL and its gradient, and
    the values and input gradients of `PosteriorMean`,
    `PosteriorStandardDeviation`, `ExpectedImprovement` and
    `qExpectedImprovement` (`q = 2`, BoTorch's base samples). botorch_rs's
    derivative with respect to a fidelity at 0 or 1 is exactly 0 where the
    fidelities enter only through the clamp;
  * `Warp` at its `eps` bounds (`eps = 1e-7`, where only `x = 0` ties, and
    `eps = 2^-20`, where `x = 1` ties too), with training inputs on the
    bounds: MLL, `PosteriorMean` and `ExpectedImprovement`;
  * `optimize_acqf` on a multi-fidelity GP (`PosteriorMean`,
    `ExpectedImprovement`) from 8 starts whose fidelity is on a bound, on both
    L-BFGS-B paths, all restarts compared: no restart leaves its bound
    (the projected gradient vanishes there), while BoTorch under torch 2.13
    moves 4 of the 8 off the bound;
  * `qExpectedImprovement` (`q = 1, 2`), `qNoisyExpectedImprovement`, a
    `ConstrainedMCObjective` (in `qSimpleRegret` and its torch forward),
    `FeasibilityWeightedMCMultiOutputObjective` (torch forward) and
    `qExpectedHypervolumeImprovement` with samples exactly at the bound of
    their `clamp_min(0)` (affine deterministic models); botorch_rs's
    gradients there are exactly 0.

  Under torch ≤ 2.13 every case also records the change of BoTorch's native
  gradients and requires it to be real (> 1e-12; the emulation's autograd
  graph alone changes gradients by ≤ 1e-17, measured 9.5e-18 on the control),
  so the cases do sit at the bounds; the control without clamps requires no
  change.
* `test_parity_6b_transforms.py::test_transforms_on_phase7_models_match_botorch`
  (the two failures under torch 2.14): its query points have fidelities 0 and
  1, so BoTorch's acquisition values and gradients are computed under
  `torch214_clamp_subgradients()` (direct comparison under ≥ 2.14).
* `botorch_core/tests/clamp_bounds.rs` (new): the linear truncated fidelity
  kernel's cross-covariance gradients at fidelities 0, 1 (exactly 0), inside
  (central differences) and outside, with the one-sided derivative from
  inside nonzero at the bounds (one and two fidelity dimensions); the Warp
  derivative at the eps bounds; qEI at an improvement of exactly 0. Unit
  tests in `dual.rs` (scalar and tensor ties), `transforms/utils.rs` (Warp)
  and `utils/mvnxpb.rs` (`clip01`, `safe_log`, `safe_exp` and the symmetric
  subgradients of `torch.maximum(h, k)` at `h == k`).

## Results

Branch `fix/torch214-clamp` after merging main f3840b8 (with
`fix/fit-nonfinite`). Before the change, under torch 2.14.1 the full suite
had exactly two failures against both BoTorch main and 0.18.1 (the
multi-fidelity cases of `test_transforms_on_phase7_models_match_botorch`).

| Check | torch 2.13.0 (main env) | torch 2.14.1+cpu (scratch venv) |
|---|---|---|
| `pytest botorch_pyo3/tests` vs BoTorch main (`BOTORCH_RS_REQUIRE_BOTORCH=1`; includes the 974 `equivalence` tests) | 3527 passed, 36 skipped, 0 failed | 3527 passed, 36 skipped, 0 failed |
| the same vs BoTorch 0.18.1 | 3507 passed, 56 skipped, 0 failed | 3507 passed, 56 skipped, 0 failed |
| `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace` | 393 passed, 0 failed | |
| `scripts/develop.py`, ruff check / format (CI's file list), mypy | pass | |
| Python 3.11 without torch / NumPy | 462 passed, 74 skipped | |

No Rust fixture changed: none has a derivative at an exact clamp bound.

Worst errors of the new comparisons (identical under both torch versions;
under 2.13 against BoTorch with the ≥ 2.14 rule):

| Quantity | Tolerance | Worst error |
|---|---|---|
| multi-fidelity kernels (4 cases): MLL / MLL gradient | 1e-9 / 1e-8 | 5.0e-16 / 4.7e-15 |
| multi-fidelity: `PosteriorMean`, `PosteriorStandardDeviation`, EI, qEI (`q = 2`) values | 1e-9 | 5.9e-15 |
| multi-fidelity: the same, input gradients | 1e-8 | 2.6e-14 |
| `Warp` (`eps = 1e-7`, `2^-20`): MLL / MLL gradient | 1e-9 / 1e-8 | 3.6e-16 / 6.3e-15 |
| `Warp`: `PosteriorMean`, EI values / input gradients | 1e-9 / 1e-8 | 4.5e-15 / 5.5e-15 |
| `optimize_acqf` on a multi-fidelity GP (2 acquisitions × 2 L-BFGS-B paths, 8 restarts each): candidates / values | 1e-7 / 1e-9 | 1.8e-15 / 2.4e-15 |
| qEI (`q = 1, 2`), qNEI, constrained objectives, qEHVI at the bound: values / gradients | 1e-9 / 1e-8 | 0 / 0 |
| `test_transforms_on_phase7_models_match_botorch`: EI and qEI values / input gradients | 1e-9 / 1e-8 | 7.3e-15 / 8.1e-15 |

Change of BoTorch's own gradients from the ≤ 2.13 to the ≥ 2.14 rule in
these cases (relative to the largest component, measured under torch 2.13):
0.13 – 2.5 for the linear truncated fidelity kernels, 5.5e-7 – 30 for `Warp`
(small where the warped derivative at `eps` is small: concentration above
1), 0.5 – 1 for the Monte-Carlo improvements, 0 for the downsampling /
exponential decay control (9.5e-18 for its qEI: the emulation's graph). In
`optimize_acqf`, 4 of the 8 restarts leave the fidelity bound under BoTorch
with torch 2.13; none does with torch 2.14 or botorch_rs.

## Deviations from BoTorch (migration notes)

* botorch-rust follows PyTorch ≥ 2.14's subgradients at clamp bounds under
  every torch version. With torch ≤ 2.13, BoTorch's gradients differ exactly
  at such bounds, e.g. at fidelities of 0 or 1 in the
  `LinearTruncatedFidelityKernel`, at inputs of 0 under `Warp`, or for a
  Monte-Carlo improvement of exactly 0; `optimize_acqf` from starts on such
  bounds can then end elsewhere. Away from the bounds nothing changes.
* `bvn`'s `torch.maximum(h, k)` now splits the gradient at `h == k` as torch
  does (botorch_rs gave all of it to `h`).

## BoTorch observations

No BoTorch bug follows from the torch change itself: BoTorch inherits torch's
convention. One quirk surfaced while writing the tests:

* `ExpectedImprovement`, `LogExpectedImprovement`, `qExpectedImprovement`
  (and the other acquisition functions that call `torch.as_tensor(best_f)`)
  store a Python-float `best_f` in torch's default dtype, float32 unless
  changed, also for float64 models:
  `ExpectedImprovement(model, best_f=0.3).best_f` is
  `tensor(0.30000001192092896)`. The acquisition values of a float64 model
  are then those of a rounded incumbent (relative differences of ~1e-7 in
  EI), and a sample exactly at `0.3` no longer ties. botorch-rust keeps
  float64. The parity tests pass a float64 tensor (or a value exact in
  float32, such as 2.5). Possible upstream fix: `torch.as_tensor(best_f,
  dtype=torch.float64)` or the model's dtype.

## Remaining

* `fmin` / `fmax` and NaN-input clamp gradients changed too, but no ported
  code differentiates them.
* README's "Versions checked" could add torch 2.14.1 once main runs on it.
