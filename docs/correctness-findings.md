# Correctness findings and proposed upstream fixes

These findings are reproducible against official BoTorch revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`, with released 0.18.1 comparisons
where that API exists. The separate local BoTorch fork at `c98dd9452`
contains earlier proposed fixes; it is not the official reference.
The linked chapters contain reproductions, corrected formulas and tests.
No upstream issues or pull requests have been submitted by this work.

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
| Full joint sampling | 13.8765 | 14.4090 | 0.96x |
| Cached sampling | 7.6639 | 1.3206 | 5.80x |

The cached native block computation improves this workload; uncached
evaluation is slightly slower in this run. Maximum value differences are
`8.55e-7` and `1.54e-7`, respectively, within the configured absolute and
relative tolerances of `1e-6`. The test includes ill-conditioned latent
baseline covariances. Timing excludes fitting, construction and imports;
it evaluates forward values with Torch gradients disabled. Other work was
active on the machine.

[Raw timings and artifact provenance](benchmarks/noisy-cache-release-2026-10-20.json)
record the source checkouts, settings and exact loaded extension path and
SHA256. The native build checkpoint was `1ec588a`; its extension hash is
`55b7161267609ce3fccb381f70287ad337e3f29dae18276095bdf1e43e9339a0`.
The report's checkout revision includes subsequent Python-only changes.
