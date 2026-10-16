# Phase 6a: kernels, priors and constraints

`SingleTaskGP` (one or several outputs, batched, in a `ModelListGP`) takes any
tree of GPyTorch's and BoTorch's kernels as `covar_module`: `ScaleKernel`
around any kernel, `AdditiveKernel` and `ProductKernel` (also `k1 + k2` and
`k1 * k2`), `active_dims`, `ard_num_dims`, `batch_shape`, constraints and
priors. Parameter names, shapes, initial values, `named_parameters()`,
`initialize()`, optimizer bounds and the order of the flat raw parameters are
GPyTorch's; the MLL (with the priors), its gradient with respect to every raw
parameter, posteriors, the exact input gradients of every acquisition
function, `fit_gpytorch_mll` (BoTorch's parameter order, bounds and retries),
conditioning and fantasies work with every tree. Kernel matrices and their
derivatives (hyperparameters by reverse mode, inputs analytically) are
computed in Rust (`botorch_core::kernels::{tree, eval, oak}`); there are no
Python calls per evaluation.

The default models are unchanged: BoTorch's preset modules
(`get_covar_module_with_dim_scaled_prior`, `get_matern_kernel_with_gamma_prior`,
...) keep the specialized RBF / Matérn-5/2 kernel, whose outputs are bitwise
identical to `main` (125 quantities checked: MLLs and gradients, fits,
posteriors, `LogExpectedImprovement`, `qLogNoisyExpectedImprovement`,
conditioning; single- and multi-output, fixed noise, `Normalize`).

## Performance

Single thread (`torch.set_num_threads(1)`, `RAYON_NUM_THREADS=1`), tree
`ScaleKernel(RBF(ARD, dims 0-1) * Periodic(dim 2) + Matern-3/2(ARD, dims 2-3))`
on 4 inputs, BoTorch main vs botorch_rs:

| n | MLL + gradient | `fit_gpytorch_mll` | `PosteriorMean` + input gradient, 256 points |
|---|---|---|---|
| 50 | 5.0 ms vs 0.23 ms (22x) | 1.61 s vs 0.10 s (16x) | 12.4 ms vs 2.7 ms (4.6x) |
| 200 | 9.7 ms vs 7.6 ms (1.3x) | 2.53 s vs 2.37 s (1.1x) | 32.9 ms vs 6.0 ms (5.5x) |

At n = 200 and above the MLL is dominated by the dense linear algebra
(Cholesky, inverse), where botorch_rs's kernel trees are as fast as its
specialized default kernel (2.3 ms for both with `ScaleKernel(RBF)`) and
MKL catches up (BoTorch 15 ms vs botorch_rs 18-20 ms at n = 400 for the
default model: not specific to this phase).

## Supported API rows

| BoTorch name | botorch_rs name | notes |
|---|---|---|
| `gpytorch.kernels.RBFKernel` | `botorch_rs.kernels.RBFKernel` | ARD or a shared lengthscale, `active_dims`, `batch_shape`, any constraint / prior |
| `gpytorch.kernels.MaternKernel` | `botorch_rs.kernels.MaternKernel` | `nu` = 0.5, 1.5, 2.5 |
| `gpytorch.kernels.RQKernel` | `botorch_rs.kernels.RQKernel` | `alpha_constraint` |
| `gpytorch.kernels.PeriodicKernel` | `botorch_rs.kernels.PeriodicKernel` | ARD lengthscales and period lengths |
| `gpytorch.kernels.LinearKernel` | `botorch_rs.kernels.LinearKernel` | `variance` per dimension with `ard_num_dims` |
| `gpytorch.kernels.PolynomialKernel` | `botorch_rs.kernels.PolynomialKernel` | any `power` |
| `gpytorch.kernels.CosineKernel` | `botorch_rs.kernels.CosineKernel` | |
| `gpytorch.kernels.IndexKernel` | `botorch_rs.kernels.IndexKernel` | `rank`, `prior` (any univariate prior or `LKJCovariancePrior`); random `covar_factor` from torch's generator |
| `gpytorch.kernels.ScaleKernel` | `botorch_rs.kernels.ScaleKernel` | around any kernel |
| `gpytorch.kernels.AdditiveKernel`, `ProductKernel`, `k1 + k2`, `k1 * k2` | same names in `botorch_rs.kernels` | flattened like GPyTorch's `__add__` / `__mul__` |
| `gpytorch.kernels.Kernel.__call__` | `Kernel.__call__` | `k(x1, x2)`, `k(x1)`, `diag=True`; dense arrays of `x1`'s type |
| `botorch.models.kernels.CategoricalKernel` | `botorch_rs.models.kernels.CategoricalKernel` | |
| `botorch.models.kernels.DownsamplingKernel` | `botorch_rs.models.kernels.DownsamplingKernel` | `diag=True` deviates for several inputs (bug below) |
| `botorch.models.kernels.ExponentialDecayKernel` | `botorch_rs.models.kernels.ExponentialDecayKernel` | |
| `botorch.models.kernels.InfiniteWidthBNNKernel` | `botorch_rs.models.kernels.InfiniteWidthBNNKernel` | |
| `botorch.models.kernels.LinearTruncatedFidelityKernel` | `botorch_rs.models.kernels.LinearTruncatedFidelityKernel` | one or two fidelity dimensions, `nu` |
| `botorch.models.kernels.OrthogonalAdditiveKernel`, `leggauss` | `botorch_rs.models.kernels.OrthogonalAdditiveKernel`, `...orthogonal_additive_kernel.leggauss` | first and second order, `check_hypercube`, shared or per-dimension lengthscales |
| `botorch.models.kernels.PositiveIndexKernel` | `botorch_rs.models.kernels.PositiveIndexKernel` | `task_prior` on the correlations |
| `gpytorch.priors.NormalPrior`, `UniformPrior`, `HalfCauchyPrior`, `HalfNormalPrior` | `botorch_rs.priors.*` | parameters in torch's default dtype, like GPyTorch |
| `gpytorch.priors.LKJPrior`, `LKJCholeskyFactorPrior`, `LKJCovariancePrior` | `botorch_rs.priors.*` | `LKJCovariancePrior` as `IndexKernel(prior=...)` |
| `botorch.models.utils.priors.BetaPrior` | `botorch_rs.models.utils.priors.BetaPrior` | |
| `gpytorch.constraints.LessThan` | `botorch_rs.constraints.LessThan` | softplus or `transform=None` (optimizer bound) |
| `draw_matheron_paths`, `draw_kernel_feature_paths` with kernel trees | `botorch_rs.sampling.pathwise.*` | RBF and Matérn (any ν) with ARD or a shared lengthscale and `active_dims`, with or without `ScaleKernel`: the kernels BoTorch's pathwise sampling supports; others raise `NotImplementedError` like BoTorch |

## Not ported (and why)

- `IntegratedWhiteNoiseKernel`, `LCEAKernel` (with `get_order`,
  `get_permutation`, `is_contiguous`), `SACKernel` and the heterogeneous
  multi-task kernels (`CombinatorialCovarModule`, `DeltaKernel`,
  `MultiTaskConditionalKernel`, `find_subsets`, `map_subsets`): they belong to
  models of later phases (contextual and heterogeneous multi-task GPs).
- Other GPyTorch kernels (spectral mixture, piecewise polynomial, grid /
  inducing-point kernels, ...), priors (`SmoothedBoxPrior`,
  `MultivariateNormalPrior`, `HorseshoePrior`) and prior transforms: not in
  the 6a scope; `SmoothedBoxPrior` comes with `MultiTaskGP` (phase 7).
- `last_dim_is_batch`, `LinearOperator` results of kernel calls.
- One kernel object used twice in a tree (GPyTorch shares its parameters):
  `UnsupportedError`.
- Kernel batch shapes other than the model's `_aug_batch_shape` (GPyTorch
  broadcasts a kernel shared by all outputs; BoTorch's batched fit cannot fit
  such modules either): `UnsupportedError` / `BotorchTensorDimensionError`.
- Kernel trees in models other than `SingleTaskGP` / `ModelListGP`
  (`MultiTaskGP`, multi-fidelity and mixed models are phase 7).

## Deviations from BoTorch (migration notes)

- Kernel calls return dense arrays of the type of `x1`, not `LinearOperator`s;
  unknown keyword arguments raise `UnsupportedError` (GPyTorch ignores them).
- Errors of inconsistent kernel and input dimensions (`ard_num_dims`,
  `active_dims` out of range, task indices out of range, an index kernel on
  several inputs) are `RuntimeError`s, as torch raises them, with messages
  that name the kernel and the dimension. GPyTorch raises them when the kernel
  is evaluated; botorch_rs checks dimensions when the model is built and task
  indices when the kernel is evaluated.
- `IndexKernel` and `PositiveIndexKernel` draw their random initial values
  from torch's global generator when torch is installed (identical values to
  GPyTorch's for the same seed), else from Python's `random` module.
- Fits of kernel-tree models hand non-finite MLL values and gradients to
  L-BFGS-B, like BoTorch (whose line search backtracks from them, e.g. when a
  periodic lengthscale underflows); models with the preset RBF / Matérn-5/2
  modules keep ending the attempt on them, as before this phase. A
  hyperparameter leaving the support of its prior (e.g. a `UniformPrior`)
  ends the fit with `ValueError` in both libraries (BoTorch retries only
  `NotPSDError`). linear_operator's structure-specific failures (e.g. the
  `LinAlgError` of `add_low_rank` on a NaN `LinearKernel`) are not
  reproduced: botorch_rs reports a NaN covariance and retries.
- In degenerate regimes, where BoTorch's own results are rounding noise,
  botorch_rs's values differ: periodic kernels with period lengths near zero
  evaluate `sin` of arguments ~1e20, which GPyTorch computes from a
  quadratic-expansion distance with cancellation and botorch_rs from exact
  differences. Fits whose line searches step there (the fixture
  `periodic_shared_d2`) have more failed first attempts in botorch_rs (9 of
  17 twins) than in BoTorch (1 of 17); the outcomes are compared statistically.
- Fixed BoTorch / GPyTorch bugs (below): the second-order prior of
  `OrthogonalAdditiveKernel` applies to the `dim * (dim - 1) / 2`
  coefficients; `LKJCovariancePrior.log_prob` is the density
  `log p_LKJ(R) + sum_i log p(sd_i)` (one value per matrix);
  `DownsamplingKernel(x, diag=True)` is the diagonal of `k(x, x)` for several
  inputs; `HalfCauchyPrior` and the LKJ normalizing constants are evaluated
  in double precision; distances of a point to itself are zero in training
  mode too (GPyTorch's `sq_dist` leaves rounding noise on the diagonal when
  inputs require gradients, which changes Matérn-1/2 MLLs at ~1e-9).
- Prior and constraint parameters given as Python floats are rounded to
  float32 when torch's default dtype is float32 (or torch is not installed),
  as GPyTorch stores them; a tensor keeps its dtype.

## Equivalence tests

Tolerances: values 1e-9, gradients 1e-8 (relative to the largest
component). Fits are compared on the model and 16 twins (training inputs
perturbed at 1e-14 .. 1e-11, fixture) or 4 twins (live, 16 if BoTorch's
outcomes spread over more than 1e-3): botorch_rs's best first-attempt loss
must reach BoTorch's best and its own loss must not end above BoTorch's worst,
within `max(1e-6, 10 x` BoTorch's spread at its best optimum`)` (scaled errors
below, tolerance 1); chaotic fits (BoTorch's outcomes spread over more than
1e-3) statistically: botorch_rs's best within 1e-6 of BoTorch's median, its
median within 1e-6 of BoTorch's upper quartile. Only first attempts are
compared (retries start from prior samples drawn by different generators).

### Rust fixtures (`botorch_core/tests/kernels_6a_parity.rs`, 28 models)

Generated against BoTorch main (da00e0401) by
`scripts/generate_fixtures.py --phase 6a-kernels`: every kernel class, trees
with `active_dims`, ARD and shared parameters, constraints (`Interval`,
`GreaterThan`, `LessThan`, non-enforced), all priors, fixed noise, no outcome
transform, two outputs.

| quantity | tolerance | worst error |
|---|---|---|
| bounds, default raw parameters | exact | 0 |
| default MLL / gradient | 1e-9 / 1e-8 | 8.0e-14 / 1.1e-13 |
| MLL / gradient at random parameters | 1e-9 / 1e-8 | 4.7e-15 / 6.3e-14 |
| kernel matrix | 1e-9 | 1.3e-15 |
| q=1 posterior mean / variance / noisy variance | 1e-9 | 3.0e-15 / 3.2e-14 / 9.6e-15 |
| q=1 d mean / dx, d variance / dx | 1e-8 | 1.2e-14 / 2.8e-14 |
| joint posterior mean / covariance / noisy covariance | 1e-9 | 4.0e-15 / 6.9e-14 / 1.5e-14 |
| joint d (weighted sum) / dx | 1e-8 | 3.9e-14 |
| fit: best loss above BoTorch's best (scaled) | 1 | 3.2e-4 |
| fit: loss above BoTorch's worst (scaled) | 1 | 5.8e-2 |
| fit (chaotic: periodic_shared, positive_index, product_scale, oak, index) | statistical | passed (0) |

### Live parity (`botorch_pyo3/tests/test_parity_6a_kernels.py`)

Against BoTorch main (da00e0401) and 0.18.1 (all pass on both). Every
kernel class stand-alone (unbatched and batched); 25 `SingleTaskGP` models
(every kernel class in trees, all priors, `LessThan`, fixed noise, no outcome
transform); multi-output and batched trees; `ModelListGP`; conditioning and
fantasies; Matheron and prior paths with BoTorch's random draws; 40 random
kernel trees (`equivalence` sweep: random leaves of 10 kernel classes with
`active_dims`, ARD, constraints, priors, sums, products, `ScaleKernel`s).
BoTorch's quirks fixed for the comparison (and measured) as listed under
the bugs.

| quantity | tolerance | worst error |
|---|---|---|
| kernel `k(x1, x2)`, `k(x1)`, `k(x1, diag=True)` (all kernels, batched) | 1e-9 | 3.9e-16 |
| kernel arithmetic trees, periodic with vanishing lengthscales, Downsampling with d = 3 | 1e-9 | 2.7e-16 |
| default raw parameters (kernels and models, incl. random index-kernel factors) | exact | 0 |
| optimizer bounds, parameter names and shapes | exact | equal |
| MLL at default / random parameters | 1e-9 | 4.2e-14 / 3.9e-15 |
| MLL gradient at default / random parameters | 1e-8 | 4.8e-14 / 3.0e-14 |
| posterior mean / variance / covariance (with and without observation noise) | 1e-9 | 8.6e-15 / 2.5e-14 / 6.1e-14 |
| `PosteriorMean`, `PosteriorStandardDeviation` values / input gradients | 1e-9 / 1e-8 | 1.4e-14 / 3.2e-14 |
| `qSimpleRegret` (identical base samples) values / input gradients | 1e-9 / 1e-8 | 2.4e-15 / 1.6e-14 |
| multi-output and batched trees: MLL / gradient / posterior | 1e-9 / 1e-8 / 1e-9 | 1.7e-15 / 1.6e-14 / 1.3e-14 |
| `ModelListGP` posterior, conditioned and fantasy posteriors | 1e-9 | 9.5e-15 |
| Matheron paths values / input gradients, prior paths | 1e-9 / 1e-8 / 1e-9 | 4.0e-15 / 7.0e-15 / 1.7e-16 |
| prior log densities (Normal, Uniform, HalfNormal, HalfCauchy, LogNormal, Gamma, Beta) | 1e-9 | 2.5e-15 |
| `LKJPrior`, `LKJCholeskyFactorPrior`, `LKJCovariancePrior` | 1e-9 | 4.6e-16 |
| `LessThan` transform / inverse / initial value, MLL / gradient | 1e-9 / 1e-8 | 0 / 2.5e-14 |
| sweep: MLL / gradient | 1e-9 / 1e-8 | 2.8e-15 / 9.1e-14 |
| sweep: posterior mean / covariance | 1e-9 | 5.5e-15 / 5.8e-14 |
| sweep: `PosteriorMean`, `PosteriorStandardDeviation` values / gradients | 1e-9 / 1e-8 | 3.4e-14 / 4.1e-13 |
| fit: best loss above BoTorch's best (scaled) | 1 | 5.6e-2 |
| fit: loss above BoTorch's worst (scaled) | 1 | 0 |
| fit (chaotic fits): best vs median, median vs upper quartile | 1e-6 | 0 / 0 |
| fit leaving a prior's support: `ValueError` in both, hyperparameters kept | exact | 0 |

Measured effects of BoTorch's quirks (fixed in the comparisons, see the
bugs): HalfCauchy log density 2.0e-8 (MLL 1.8e-8), LKJ normalizer in float32
2.0e-8 (MLL), OAK second-order prior on zeros 0.135 (MLL), `LKJCovariancePrior`
correlation term counted `n` times 0.24 (MLL; 4.0 in the log density),
`DownsamplingKernel` diagonal 0.37 (d = 3).

### Verification

`cargo fmt --check`, clippy (`-D warnings`), `cargo test --workspace` (317
tests), ruff, mypy; `pytest botorch_pyo3/tests` with
`BOTORCH_RS_REQUIRE_BOTORCH=1`: 2644 passed, 18 skipped against BoTorch main
(da00e0401), all passing against 0.18.1, `-m equivalence` 621 passed; the
package imports on Python 3.11 without torch and NumPy. The fixtures
regenerate byte for byte against BoTorch main.

## BoTorch / GPyTorch bugs found

New:

1. **`OrthogonalAdditiveKernel`: the second-order prior is evaluated on the
   full `dim x dim` coefficient matrix, including its zeros.** Priors without
   support at 0 make the MLL `-inf`:

   ```python
   k = OrthogonalAdditiveKernel(dim=2, base_kernel=RBFKernel(), second_order=True,
                                coeffs_2_prior=GammaPrior(2.0, 1.0), dtype=torch.float64)
   m = SingleTaskGP(X, Y, covar_module=k)  # X: 8 x 2 in [0, 1]
   ExactMarginalLogLikelihood(m.likelihood, m)(m(X), m.train_targets)  # -inf
   ```

   With a `HalfCauchyPrior(1.0)` and `dim = 2`, the MLL is off by
   `3 log(2 / pi) / n` (measured 0.135 for n = 10).
2. **`DownsamplingKernel(x, diag=True)` is not the diagonal of `k(x, x)` for
   more than one input**: `offset + (sum_d (1 - x_d)^2)^(1 + power)` instead of
   `offset + sum_d ((1 - x_d)^2)^(1 + power)` (relative error 0.37 for
   `d = 3`; `DownsamplingKernel()(x, diag=True)` vs
   `DownsamplingKernel()(x).to_dense().diagonal()`).
3. **`LinearTruncatedFidelityKernel` docstring**: the default lengthscale
   priors are documented as `Gamma(1.1, 1/20)` and `Gamma(5, 1/20)`; the code
   uses `GammaPrior(3, 6)` and `GammaPrior(6, 2)`.

GPyTorch / torch:

4. **`LKJCovariancePrior.log_prob` with a univariate `sd_prior` returns `n`
   values** `log p_LKJ(R) + log p(sd_i)` (the correlation term broadcast over
   the standard deviations), so a model's MLL, which adds their sum, counts
   the LKJ term `n` times:
   `LKJCovariancePrior(3, torch.tensor(1.5), GammaPrior(2.0, 1.5)).log_prob(S)`
   has shape `(3,)`. With `SmoothedBoxPrior` (event shape `(1,)`) the result
   is correct, which is why BoTorch's `MultiTaskGP` default is unaffected.
5. **LKJ normalizing constants in float32 in double-precision models**: the
   `concentration` of `LKJCholesky` is not a buffer, so `.to(torch.float64)`
   leaves it in float32 and `lgamma` runs in single precision (MLL change
   2.0e-8 when cast).
6. **`HalfCauchyPrior` evaluates in float32 in double-precision models**:
   torch's `HalfCauchy.log_prob` casts its argument to the base `Cauchy`'s
   dtype, which GPyTorch's `.to(torch.float64)` does not convert (log density
   error 2.0e-8, MLL change 1.8e-8).
7. **`sq_dist` leaves rounding noise on the diagonal of `K(X, X)` when the
   inputs require gradients** (it zeroes the diagonal only without
   gradients), so training-mode Matérn-1/2 (and periodic) kernels see
   distances ~1e-8 instead of 0 (Matérn-1/2 MLL change 1.5e-9, gradient
   3.6e-8 in the fixtures).
8. **A batched `IndexKernel` on non-batched inputs ignores `diag=True`**:
   `IndexKernel(num_tasks=3, rank=2, batch_shape=torch.Size([2]))(x, diag=True)`
   returns the full `2 x n x n` matrix (`Kernel.__call__` takes the diagonal
   only if the result has the inputs' batch dimensions).
9. MKL's single-precision `log` is not always correctly rounded (torch's
   float32 `HalfNormal` / `Uniform` log densities; botorch_rs uses torch's
   value when torch is installed).

Already fixed in the fork (earlier PRs): `ExponentialDecayKernel`'s
`power_constraint`, `InfiniteWidthBNNKernel`'s diagonal, the dtype of
`OrthogonalAdditiveKernel`'s quadrature, `posterior(output_indices=...)` of
multi-output models.
