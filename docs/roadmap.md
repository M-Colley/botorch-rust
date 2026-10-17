# Roadmap to a full port

botorch-rust 0.1.0 covers 222 of BoTorch's 773 public classes and functions
([coverage report](api_coverage.md)): the GP core, analytic and Monte Carlo
acquisition, constrained `optimize_acqf`, all of
`botorch.acquisition.multi_objective`, pathwise sampling, deterministic and
contextual models. The goal is the whole of BoTorch, with every ported component
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
  Remaining: `HeterogeneousMTGP` and its kernels, batched training data,
  candidates with the task column.
- **7b: other models (done; [details](models.md)).** Ported:
  - `SingleTaskMultiFidelityGP`, `MixedSingleTaskGP`;
  - the MAP SAAS models: `get_map_saas_model`, `AdditiveMapSaasSingleTaskGP`,
    `EnsembleMapSaasSingleTaskGP`;
  - `AffineDeterministicModel`, `FixedSingleSampleModel`, `EnsembleModel`,
    `ModelList`, `ModelDict`, `OrthogonalAdditiveGP`;
  - `subset_output`, `construct_inputs`, `condition_on_observations` and
    `fantasize` for these models.

  Remaining:
  - the per-component posteriors of `OrthogonalAdditiveGP`;
  - the knowledge gradient, entropy search and multi-objective acquisition
    functions on ensemble models.

## Phase 8: single-objective acquisition functions (mostly done)

Ported, with equivalence tests ([details](single-objective-acquisition.md)):
- the knowledge gradient, also multi-fidelity;
- qNEI, qLCB, qPSD;
- analytic noisy EI and LogNEI, `LogProbabilityOfFeasibility`,
  `ScalarizedPosteriorMean`, qAnalyticPI;
- max-value entropy search and GIBBON, also multi-fidelity;
- joint and predictive entropy search;
- Thompson sampling, `qNegIntegratedPosteriorVariance`;
- the penalized, proximal, prior-guided and fixed-feature wrappers.

Remaining:
- `qMultiStepLookahead`;
- the input constructors with `get_acquisition_function`;
- `SampleReducingMCAcquisitionFunction` as a public base class;
- t-batched candidate sets in max-value entropy search;
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

Remaining:
- `acq_function_sequence`;
- `loo_cv`, `efficient_loo_cv` and `ensemble_loo_cv`;
- `data_loader` closures.

## Phase 10: fully Bayesian, preference and approximate models

- `SaasFullyBayesianSingleTaskGP` (NUTS, reusing an existing Rust NUTS
  implementation such as `nuts-rs`), the linear and multi-task fully Bayesian
  models, and BALD.
- `PairwiseGP` with its Laplace MLL and likelihoods, EUBO and the preference
  BALD.
- `SingleTaskVariationalGP` with the inducing-point allocators.
- `HigherOrderGP`, `LatentKroneckerGP`, the contextual `SACGP` / `LCEAGP`,
  `HeterogeneousMTGP`, relevance pursuit, and the empirical and hierarchical
  GP packages.

## Phase 11: the rest

- `botorch.utils.probability` (bivariate and truncated multivariate normal,
  linear-ellipsoid sampling) and the remaining utilities.
- Samplers other than Sobol (`IIDNormalSampler`, ...).
- The remaining test functions: multi-fidelity, constrained single-objective
  and sensitivity analysis.
- `botorch_community`.

## Cross-cutting

- Batched models and batched box decompositions in the multi-objective
  acquisition functions.
- Autograd through models: posteriors as torch tensors with gradients with
  respect to the hyperparameters, not only the inputs.
- Faster dense linear algebra for larger n. At n = 300, fitting is only 1.4×
  faster than BoTorch, which uses multithreaded BLAS/LAPACK.
