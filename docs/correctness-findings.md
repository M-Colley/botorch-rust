# Correctness findings and proposed upstream fixes

These findings are reproducible against official BoTorch revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`, with released 0.18.1 comparisons
where that API exists. The separate local BoTorch fork at `c98dd9452`
contains earlier proposed fixes; it is not the official reference.
The linked chapters contain reproductions, corrected formulas and tests.
Verified repairs have been submitted upstream. The
[submission ledger](upstream-pull-requests.md) links existing and new PRs and
identifies findings whose upstream regression or submission is still pending.

| Area | Failure | Solution and evidence |
|---|---|---|
| Batched LOO | Standardize statistics broadcast against the fold axis; fully Bayesian target consistency can index a nonexistent dimension. | Untransform before restoring `q=1`; expand targets to the predictive ensemble layout. [LOO notes](cross-validation.md). |
| SAC priors | One shared lengthscale prior appears `M+1` times through registered aliases, changing fitting values and gradients. | Deduplicate visited modules during prior enumeration, or avoid registering shared aliases. The Rust objective counts it once. [SAC notes](contextual-sac.md). |
| SAC construction | A dataset without noise omits a required constructor argument. | Include `train_Yvar=training_data.Yvar`, even when it is None. [SAC reproduction](contextual-sac.md). |
| Kronecker sampling | Remote-input posterior variance can be about 0.693 while sample variance collapses to roughly `1e-8`. | Root the full conditional covariance `Kx* ⊗ Kt* - Ax ⊗ At`; the product of separately conditioned residuals is incorrect. [Kronecker reproduction and shape fixes](latent-kronecker.md). |
| Boltzmann weighting | Batched scores normalize across a batch dimension instead of categories; identical fantasy rows can become uniform despite unequal values. | Standardize `values.unsqueeze(-1)` then squeeze, or normalize the final category axis explicitly. [Sampling notes](drafts/11a-utils.md). |
| Integrated white noise | Public `diag=True` can ignore a different second input. | Evaluate paired cross-covariance diagonals using both inputs. [Kernel notes](kernels-and-priors.md). |
| Expectation transforms | All-zero weight columns normalize to NaN. | Require finite nonnegative weights and positive column sums. [Expectation notes](expectation-posterior-transform.md). |
| Context embeddings | Sorted feature permutations lose declared context pairings; noncontiguous categorical labels can index outside embedding tables. | Preserve each context's declared order and map categorical labels to contiguous indices. [LCEA reproductions](contextual-lcea.md). |
| Hierarchical kernels | Global category checks reject unrelated parent labels; unfitted input transforms and interior task columns can produce wrong branch values; closed cycles disappear. | Validate per parent, derive gates after fitting transforms, compose non-task indices, and validate all graph components. [Hierarchy notes](hierarchical-models.md). |
| Batched L-BFGS-B | Failed line searches can return a rejected trial's value with a restored point and gradient; the same failure occurs in SciPy's C-backed solver. | Preserve the accepted objective value and restore all three together. [Reproduction and tested upstream patch](batched-lbfgsb.md). |
| Heterogeneous conditioning | Target-task inputs accepted by `posterior` cannot be concatenated with full-feature training inputs during conditioning. | Embed the target inputs in the declared full feature layout before conditioning. [Reproduction and native fix](heterogeneous-multitask.md). |
| Empirical multitask noise | A fixed-noise prediction for a historical task absent from training averages an empty task mask and returns NaN. | Use observed task-specific noise where available and an observed global mean for unseen tasks. [Empirical models](empirical-gp-models.md). |
| Sparse outliers and robust fitting | Batched indexing, duplicate support, small-data support priors, fixed-noise prediction and restoration after fitting errors have reproducible defects. | Validate support, use the proper event axis and noise semantics, center LOO residuals, and restore state in `finally`. [Seven reproductions and proposed fixes](relevance-pursuit.md). |
| Single selected output | A multi-output GP selecting one output attempts to construct a multitask distribution from one MVN and raises. | Return the sole MVN when only one output is selected. [Reproduction](posterior-query-autograd.md). |
| Additive components | Bias covariance and component-axis handling fail for batched queries; `Standardize` adds the full outcome location to every component. | Broadcast both point sets and the bias, and assign the outcome location only to the bias. [Tested broadcast patch and location formula](additive-components.md). |
| Empty Beta expansion | Validated expansion to an empty batch can reject a legitimate empty Beta distribution. | Preserve concentration broadcasting and validate scalar parameter values before expanding. [Reproduction and corrected distribution comparison](risk-prior-members.md). |
| Fantasy noise | A multidimensional sample shape leaves fantasy dimensions in forwarded observation noise. | Remove all sample dimensions before expanding noise. [Protocol tests and proposed patch](fantasy-model-protocol.md). |
| EM coordinate ascent | Subsetting historical datasets leaves stale experiment indices, so equal-sized datasets can silently use another dataset's inputs. | Rebuild observation indices from the selected datasets. [Reproductions and solutions](empirical-gp-models.md). |
| Constant EM objective | Default fixed-noise, noninducing coordinate fitting can call backward on a constant detached likelihood and raise. | Detect a constant objective and complete a zero-gradient optimization step. [Reproduction](empirical-gp-models.md). |
| Fully Bayesian traces | Known-noise prior prediction broadcasts a length-n latent vector against n-by-1 scales, producing n-by-n observations; default multitask warping can include the task column. | Flatten per-observation scales and select actual non-task features. [Trace parity and corrected formulas](numpyro-trace-interoperability.md). |
| Higher-order fantasy caches | A generic fantasy training cache can be an `AddedDiagLinearOperator`, so the posterior's Kronecker-specific `.linear_op` access raises. Batch-repeat wrappers also require explicit unwrapping. | Inspect the actual added-diagonal and repeated-factor structures before extracting noiseless factors. Native fantasy predictions use their own cached factors. [Reproduction and validation](higher-order-public-methods.md). |
| Sequential acquisitions | A legal `None` primary acquisition fails during dataclass replacement; exceptions leak pending points; all-fixed optimization bypasses the sequence and returns a scalar. | Retain the sequence through initialization, restore pending state in `finally`, and evaluate each fixed candidate's actual acquisition. [Verified patch and parity tests](sequence-optimization.md). |
| All-fixed constraints | The all-features-fixed optimizer shortcut returns an infeasible candidate without checking parameter constraints. | Pass constraints into the shortcut and call `evaluate_feasibility` before returning; reject an impossible fixed candidate. [Isolated reproduction and proposed repair](#all-fixed-optimization-ignores-constraints). |
| Correlated empirical fantasies | Generic conditioning treats correlated outputs as independent batches, creating an extra fantasy dimension or dropping fixed observation noise. | Repeat inputs per output and flatten targets and noise together before `get_fantasy_model`. Native conditioning uses the joint event directly. [Tested workaround](empirical-conditioning.md). |
| Empty data loaders | A closure retries empty loader epochs indefinitely. | Raise after one exhausted restart. The port evaluates real variational minibatches and their native derivatives. [Regression and proposed patch](variational-data-loader-closures.md). |
| Learned preference caches | Sampling learned preferences separately for baseline and candidates breaks their joint law; a duplicate baseline candidate can report improvement about `0.45`. | Sample preferences jointly over baseline and candidates. Native stochastic learned objectives use full-joint noisy improvement; deterministic preference caches remain available. [Reproduction and solution](learned-objective-acquisitions.md). |
| Kernel feature batches | Categorical full covariances exchange point axes; composite lazy shapes can count unselected features; the generic diagonal wrapper can take a second diagonal when the point and feature counts coincide. | Move the feature axis without reversing point axes, derive composite feature counts from children, and include the feature axis in the diagonal rank check. [Verified reproductions, patch and native implementation](kernels-and-priors.md#verified-upstream-feature-batch-bugs). |
| Model-list covariance blocks | Lazy task-block indexing can lose off-diagonal covariance or move variances between tasks of a multioutput child. | Index the dense covariance as a conservative workaround, or repair the underlying lazy block extraction. [Pinned reproduction, tested patch and native block pullbacks](joint-model-list-autograd.md). |
| Variational zero-mean prior (GPyTorch 1.15.2) | An unbatched multioutput `prior=True` call with `ZeroMean(batch_shape=())` passes a rank-one mean into `from_batch_mvn` and raises. | Expand the mean to the MVN's batch and event shape before permuting it; an output-batched `ZeroMean` is a workaround. The port returns the valid multitask prior. [Reproduction, independently tested repair and parity workaround](approximate-models.md). |
| Neural-kernel diagonal conventions | Intentional `acos_eps` clipping affects full self covariances but is bypassed by `diag=True`, producing different values and derivatives; the native kernel matches both. | Preserve the model's lazy marginal variance in analytic acquisitions. For upstream consistency, choose a common differentiable self-variance recurrence; avoid evaluating singular self `acos` terms. [Reproduction, closed forms, and tested repair prototype](infinite-width-bnn-diagonals.md). |
| Constant-target Standardize | Taking a square root of zero variance before replacing its result can propagate NaN first and second target derivatives through the inactive branch. | Clamp variance before `sqrt`, then select the constant-one branch using the original variance. [CPU/CUDA reproduction and finite-difference checks](torch-backend-modules.md#verification-and-reference-limitations). |
| Periodic/cosine higher derivatives (GPyTorch 1.15.2 / Torch 2.13.0) | A second query derivative raises because `_cdist_backward` has no derivative in this runtime. | Use differentiable pairwise arithmetic for periodic distances and a nonsingular cosine-of-squared-distance primitive. These are port primitives and independent reference-formula checks, not a submitted upstream fix. [Evidence and formulas](tensor-native-algebra.md#reference-issues-and-solutions). |
| Polynomial feature batches (GPyTorch 1.15.2) | A model-batched offset lacks the extra feature axis under `last_dim_is_batch=True`, causing incorrect broadcasting or an error. | Insert the feature axis into the offset for both full and diagonal evaluations. [Concrete proposed repair and per-feature reference oracle](tensor-native-algebra.md#reference-issues-and-solutions). |
| Ensemble posterior standard deviation | The reference reshapes standard deviations to the query batch before reducing its MCMC axis, and raises when both axes are present. | Remove the singleton output axis with `squeeze(-1)` while preserving the MCMC axis for the existing ensemble reduction. [Pinned reproduction and direct posterior oracle](torch-backend-acquisitions.md#ensemble-acquisition-reductions). |
| Fully Bayesian explicit prediction noise | The reference inserts an MCMC axis into queries but leaves tensor-valued observation noise unchanged, causing a broadcast error for query batches. | Insert the same singleton MCMC axis into explicit observation noise. [Unmodified failure and independent component-law comparison](torch-backend-acquisitions.md#fully-bayesian-model-lists). |

## All-fixed optimization ignores constraints

In official revision `b0dd01be61d517c387e234903a6e2cc38c057ea2`,
`botorch/optim/optimize.py:831` dispatches directly to
`_optimize_acqf_all_features_fixed` (defined at line 195) when every input
feature is fixed. The helper receives no parameter constraints and returns the
fixed point without checking feasibility. This is separate from the sequential
acquisition issue above and occurs in an ordinary scalar optimization:

```python
import torch
from botorch.optim import optimize_acqf

class Quadratic(torch.nn.Module):
    def forward(self, X):
        return -X.square().sum(dim=(-2, -1))

candidate, value = optimize_acqf(
    Quadratic(), torch.tensor([[0.], [1.]], dtype=torch.double),
    q=1, num_restarts=1, raw_samples=4,
    fixed_features={0: 0.},
    inequality_constraints=[(torch.tensor([0]), torch.tensor([1.]), 0.5)],
)
# candidate == tensor([[0.]]) despite the required x >= 0.5.
```

The concrete upstream repair is to forward `inequality_constraints`,
`equality_constraints`, and `nonlinear_inequality_constraints` at the line-831
dispatch and add those keyword arguments to the helper. After constructing
`X`, call `evaluate_feasibility(X, ...)` with all three constraint collections;
raise `CandidateGenerationError` if any fixed q-batch is infeasible. Also check
the fixed values against `bounds`. There are no free coordinates to project or
optimize, so returning a constraint-violating point cannot resolve this case.
If postprocessing is supported in this shortcut, apply it before the final
feasibility check and recompute the returned acquisition value at that point.

`test_all_fixed_constraint_regression.py` preserves the unmodified pinned
reference failure and verifies that the Tensor port raises
`CandidateGenerationError` on CPU and CUDA. It supplies initial conditions to
exercise the port's constrained solver directly, without depending on a
constraint-aware automatic initializer. The upstream repair and regressions are
submitted in [BoTorch #3470](https://github.com/meta-pytorch/botorch/pull/3470).

The port also fixes its own integration issues: batched outcome untransform
layout, unseeded QMC stream continuation, native projection and infeasible
SLSQP result handling, list-sampler forwarding, explicit Gaussian posterior
sampling and scalarization composition, cached-root broadcasting, frozen posterior parameter snapshots, and lazy
optional-dependency adapters. Regression tests cover these independently
of upstream comparisons.

Native SingleTaskGP, higher-order and Kronecker posteriors retain parameter snapshots.
Previously, cached moments could describe an earlier model while sampling
re-evaluated a mutable model after its noise or other parameters changed.
The fix snapshots mutable native source handles when constructing a posterior;
regression tests cover raw and transformed outcomes.

## Measured release performance

`python scripts/benchmark_loo.py --repeats 21` checks matching means and
variances before timing cached LOO evaluation on identical float64 models.
Both libraries use one worker and hyperparameter autograd is disabled.
The 2026-10-20 Windows run on an Intel Core Ultra 7 155H produced:

| Training points | BoTorch 0.18.1 (ms) | Rust release (ms) | Ratio |
|---|---:|---:|---:|
| 16 | 1.1255 | 0.3306 | 3.40x |
| 64 | 1.0358 | 0.3891 | 2.66x |
| 128 | 1.2842 | 0.3961 | 3.24x |
| 256 | 3.2681 | 1.6761 | 1.95x |

These are medians of local measurements of this diagnostic, excluding
fitting and imports. Other checks were active on the machine. They do not
establish speedups for other models, workloads, machines or thread counts.

For noisy improvement, `scripts/benchmark_noisy_cache.py` checks values
using shared normal draws before timing four variants in shuffled order in
each measured round. A release run with 32 training points, 180 baseline
points, eight candidate batches of two points, 128 samples, one thread,
three warmups and 21 measured rounds produced:

| qLogNEI evaluation | BoTorch pinned reference (ms) | Rust release (ms) | BoTorch/Rust |
|---|---:|---:|---:|
| Full joint sampling | 15.7400 | 15.8125 | 0.995x |
| Cached sampling | 8.3588 | 1.3996 | 5.97x |

The cached native block computation improves this workload; full joint
evaluation is roughly equal in this run. Maximum value differences are
`8.55e-7` and `1.54e-7`, respectively, within the configured absolute and
relative tolerances of `1e-6`. The test includes ill-conditioned latent
baseline covariances. Timing excludes fitting, construction and imports;
it evaluates forward values with Torch gradients disabled. Heavy builds and
test runs were stopped for this measurement. Timings varied across rounds;
all raw measurements are retained.

[Raw timings and artifact provenance](benchmarks/noisy-cache-complete-release-2026-10-20.json)
record the source checkouts, settings and exact loaded extension path and
SHA256. The native build checkpoint was `4a8b8bd`; its extension hash is
`543749d9f2df2aa1b89faa2985b9f706edd2e7eca5014d7bf49235e074f6def9`.
The report's checkout revision includes subsequent Python-only changes.

An [earlier release measurement](benchmarks/noisy-cache-release-2026-10-20.json)
with other work active measured 5.80x for cached sampling and 0.96x for full
joint sampling. A [measurement before the final Python dispatch correction](benchmarks/noisy-cache-final-release-2026-10-20.json)
using the same native binary measured 7.02x and 0.97x. These local results
support an improvement for cached sampling; they do not establish a general
speedup across BoTorch workloads.
