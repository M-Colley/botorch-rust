# Roadmap to a full port

botorch-rust 0.2.0's consolidated development branch includes the GP core
with kernel trees, priors and all input and outcome transforms; multi-task,
multi-fidelity, mixed, MAP SAAS, ensemble, deterministic and contextual
models; fully Bayesian GPs and NUTS; preference, variational and higher-order
GPs; the single-objective acquisition functions and all of
`botorch.acquisition.multi_objective`; acquisition optimization; efficient
leave-one-out cross-validation; probability utilities and samplers.
The generated [coverage report](api_coverage.md) and
[JSON gap inventory](api_coverage.json) record public names and missing API
members against a pinned official BoTorch revision. Symbol availability
does not prove that every configuration, gradient or algorithm is supported.
The goal is the whole of BoTorch, with every ported component
tested for equivalence like the existing ones (Rust fixtures generated with
BoTorch, live side-by-side tests, randomized sweeps). This file orders the
remaining work. Foundations come first, because most models and acquisition
functions build on them.

## Phase 6: modeling building blocks

- **6a: kernels, priors and constraints (done; [details](kernels-and-priors.md)).**
  Kernel trees in `SingleTaskGP`, the GPyTorch kernels (Matérn ν = 0.5, 1.5,
  2.5, RBF, RQ, periodic, linear, polynomial, cosine, `IndexKernel`), BoTorch's
  own kernels, the Normal / Uniform / half-Cauchy / half-normal / LKJ / Beta
  priors and `LessThan`. Remaining: the kernels of later model phases,
  `SmoothedBoxPrior` and other priors, `last_dim_is_batch`.
- **6b: transforms (done; [details](transforms.md)).** All 35 names of
  `botorch.models.transforms`: the input transforms (`Normalize` in full,
  `InputStandardize`, `Warp`, `Round`, `AppendFeatures`, `FilterFeatures`,
  `InteractionFeatures`, `Log10`, `OneHotToNumeric`,
  `NumericToCategoricalEncoding`, `AffineInputTransform`,
  `BatchBroadcastedInputTransform`, `LearnedFeatureImputation`), the outcome
  transforms (`Log`, `Power`, `Bilog`, `ChainedOutcomeTransform`,
  `StratifiedStandardize`), `TransformedPosterior` and the utilities.
  Remaining: transforms beyond `Normalize` / `Standardize` on the multi-task
  models and `LCEMGP`, nonlinear outcome transforms in the knowledge
  gradient and entropy searches, `Warp(reverse=True)` in models.

## Phase 7: models

- **7a: multi-task models (done; [details](multitask.md)).** `MultiTaskGP`
  and `KroneckerMultiTaskGP`, with `MultitaskGaussianLikelihood`,
  `HadamardGaussianLikelihood`, `MultitaskKernel` and `SmoothedBoxPrior`.
  `HeterogeneousMTGP` and its composite kernels are also integrated
  ([details](heterogeneous-multitask.md)). Specialized batching and transform
  combinations remain documented per model.
- **7b: other models (done; [details](models.md)).** Ported:
  - `SingleTaskMultiFidelityGP`, `MixedSingleTaskGP`;
  - the MAP SAAS models: `get_map_saas_model`, `AdditiveMapSaasSingleTaskGP`,
    `EnsembleMapSaasSingleTaskGP`;
  - `AffineDeterministicModel`, `FixedSingleSampleModel`, `EnsembleModel`,
    `ModelList`, `ModelDict`, `OrthogonalAdditiveGP`;
  - `subset_output`, `construct_inputs`, `condition_on_observations` and
    `fantasize` for these models.

  Remaining:
  - the knowledge gradient, entropy search and multi-objective acquisition
    functions on ensemble models.

## Phase 8: single-objective acquisition functions (done)

Ported, with equivalence tests ([details](single-objective-acquisition.md)):
- the knowledge gradient, also multi-fidelity;
- qNEI, qLCB, qPSD;
- analytic noisy EI and LogNEI, `LogProbabilityOfFeasibility`,
  `ScalarizedPosteriorMean`, qAnalyticPI;
- max-value entropy search and GIBBON, also multi-fidelity;
- joint and predictive entropy search;
- Thompson sampling, `qNegIntegratedPosteriorVariance`;
- the penalized, proximal, prior-guided and fixed-feature wrappers.

Also ported (phase 8b, [details](acquisition-construction.md)): the input
constructors with `get_acquisition_function`, `qMultiStepLookahead`,
`SampleReducingMCAcquisitionFunction` as a public base class, and t-batched
max-value samplers and fixed features.

Remaining:
- the input constructors of BALD and EUBO (with Phase 10);
- `qMultiStepLookahead` with other value functions and models;
- the active learning that needs pairwise or fully Bayesian models
  (Phase 10).

## Phase 9: optimization and candidate generation (done; [details](optimization.md))

Ported:
- `optimize_acqf_mixed` (also alternating), `optimize_acqf_discrete` (also
  local search), `optimize_acqf_cyclic`, `optimize_acqf_homotopy`;
- `post_processing_func`, `timeout_sec`, `gen_candidates`, `topn` and
  `sample_around_best` in `optimize_acqf`;
- `MaxPosteriorSampling`, `BoltzmannSampling`, `get_best_candidates`;
- fitting with torch optimizers, the closures and stopping criteria;
- `batch_cross_validation`.

Sequential acquisition sequences now optimize each acquisition through the
native candidate generator and restore pending points after failures
([scope and upstream fixes](sequence-optimization.md)). Remaining:
`data_loader` closures.

`loo_cv`, `efficient_loo_cv` and `ensemble_loo_cv` now use native cached
factorizations, with batched independent outputs, fixed noise, outcome
untransforms and ensemble sample transforms. They keep the existing model
hyperparameters, as in BoTorch; `batch_cross_validation` refits each fold.

## Phase 10: fully Bayesian, preference and approximate models

- Fully Bayesian single-task, linear and multi-task GPs, native NUTS and
  BALD are integrated ([implementation and parity notes](drafts/10a-fully-bayesian.md)).
- `PairwiseGP` with its Laplace MLL and likelihoods, EUBO and the preference
  BALD (done; [details](preference.md)). Remaining: `LearnedObjective` in the
  Monte-Carlo acquisition functions, learned input transforms in
  `PairwiseGP`.
- `SingleTaskVariationalGP` with the inducing-point allocators and
  `HigherOrderGP` are integrated ([details and limits](approximate-models.md)).
- `LatentKroneckerGP`, contextual `SACGP` / `LCEAGP` and hierarchical
  conditional GPs are integrated with explicit configuration limits
  ([Kronecker](latent-kronecker.md), [SAC](contextual-sac.md),
  [LCEA](contextual-lcea.md), [hierarchical](hierarchical-models.md)).
- Native empirical covariance, basis, interpolation, PSD and shrinkage helpers
  are integrated ([utilities](empirical-gp-utils.md)).
- Scalar, independent/correlated-output and heterogeneous multitask empirical
  GPs are integrated ([models](empirical-gp-models.md)).
- Relevance pursuit, robust GPs and sparse-outlier likelihoods are integrated
  ([scope and upstream fixes](relevance-pursuit.md)).
- Native additive-component inference is integrated
  ([details](additive-components.md)).
- EM empirical prior pretraining, historical likelihoods, exact target
  conditioning, shared-kernel fitting and coordinate ascent are integrated.
  Differentiable EM reruns and learned inducing locations remain configuration
  work ([details](empirical-gp-models.md)).

## Phase 11: the rest

- Probability utilities, including bivariate and truncated multivariate
  normals, MVNXPB and linear elliptical slice sampling, are integrated.
- IID, normal QMC, multivariate QMC, forked RNG and list samplers are
  integrated. Native QMC engines retain their stream state and perform
  multivariate projection in Rust.
- Multi-fidelity, constrained single-objective and sensitivity-analysis
  test functions are integrated ([scope and verification](drafts/11a-utils.md)).
- Resumable native batched L-BFGS-B and atomic training target/noise replacement
  are integrated ([solver](batched-lbfgsb.md), [model state](model-training-state.md)).
- Gaussian and structured Matheron posterior query adjoints,
  supplied-distribution/operator constructors, public path constructors, and
  fully Bayesian NumPyro effect-handler interfaces are integrated
  ([paths](pathwise-public-constructors.md),
  [optional trace interface](numpyro-trace-interoperability.md)).
  Specialized fantasy-batch and model-state configurations remain explicit.
- Lazy JAX conversion adapters, a validated atomic LCBench cache/loader with
  native log scaling, and native-aware optimization mocks are integrated.
- `botorch_community`.

For an explicit completeness check, run
`python scripts/api_coverage.py --require-full --json-out docs/api_coverage.json`.
This core API check now passes against the pinned official reference: 776
public names, no missing audited constructor parameters or public members.
It exits unsuccessfully if those gaps recur. This is an API milestone; the
numerical parity suites, documented configuration limits and release
benchmarks remain separate requirements. `botorch_community` is tracked
separately.

## Cross-cutting

- Batched models and batched box decompositions in the multi-objective
  acquisition functions.
- Autograd through models: posteriors as torch tensors with gradients with
  respect to the hyperparameters, not only the inputs.
- Faster dense linear algebra for larger n. At n = 300, fitting is only 1.4×
  faster than BoTorch, which uses multithreaded BLAS/LAPACK.

Correctness deviations and reproductions are indexed in
[correctness-findings.md](correctness-findings.md). Public symbol availability
is not completion: missing parameters/members and numerical, tensor-autograd,
configuration and performance coverage remain separate work.
