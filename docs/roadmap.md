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

- **6a: kernels, priors and constraints.**
  - GPyTorch kernels BoTorch users pass as `covar_module`:
    - Matérn with ν = 0.5, 1.5 and 2.5, RBF, rational quadratic, periodic,
      linear, polynomial, cosine;
    - `AdditiveKernel`, `ProductKernel` and `ScaleKernel` on any kernel, with
      `active_dims`;
    - `IndexKernel`.
  - BoTorch's own kernels: `CategoricalKernel`, `LinearTruncatedFidelityKernel`,
    `DownsamplingKernel`, `ExponentialDecayKernel`, `InfiniteWidthBNNKernel`,
    `OrthogonalAdditiveKernel`, `PositiveIndexKernel`.
  - Priors: Normal, Uniform, half-Cauchy, half-normal, LKJ and BoTorch's
    `BetaPrior`. Constraints: `LessThan`.
  - `SingleTaskGP` with arbitrary kernel trees: parameter names,
    `initialize`, fitting and posteriors.
- **6b: transforms.**
  - Input transforms: `InputStandardize`, `Warp` (Kumaraswamy), `Round`,
    `AppendFeatures`, `FilterFeatures`, `InteractionFeatures`, `Log10`,
    `OneHotToNumeric`, `NumericToCategoricalEncoding`,
    `AffineInputTransform`, `BatchBroadcastedInputTransform`,
    `LearnedFeatureImputation`.
  - Outcome transforms: `Log`, `Power`, `Bilog`, `ChainedOutcomeTransform`,
    `StratifiedStandardize`.
  - `botorch.models.transforms.utils`.

## Phase 7: models

- `MultiTaskGP` itself and `KroneckerMultiTaskGP`.
- `SingleTaskMultiFidelityGP`, `MixedSingleTaskGP`.
- MAP SAAS models: `get_map_saas_model`, `AdditiveMapSaasSingleTaskGP`,
  `EnsembleMapSaasSingleTaskGP`.
- `AffineDeterministicModel`, `FixedSingleSampleModel`, `EnsembleModel`,
  `ModelList`, `ModelDict`.
- For every model: `subset_output`, `construct_inputs`,
  `condition_on_observations` and `fantasize`.

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

## Phase 9: optimization and candidate generation

- `optimize_acqf_mixed` (also alternating), `optimize_acqf_discrete` (also
  with local search), `optimize_acqf_cyclic`, `optimize_acqf_homotopy`.
- In `optimize_acqf`: `gen_candidates_torch`, `sample_around_best`,
  `post_processing_func`, `timeout_sec`.
- `MaxPosteriorSampling`, `BoltzmannSampling`, `get_best_candidates`.
- Fitting with torch optimizers, BoTorch's closures and stopping criteria.
- `botorch.cross_validation`.

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
