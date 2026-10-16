# Single-objective acquisition functions

The port of BoTorch's single-objective acquisition functions (roadmap phase 8):
the supported API, the deviations from BoTorch, the equivalence results and the
BoTorch bugs found. Everything listed here evaluates natively in Rust with exact
gradients with respect to `X`, works with `optimize_acqf` (also from given
initial conditions), with `SingleTaskGP` (single- and multi-output, with
`Normalize` / `Standardize`, fixed noise) and `ModelListGP`, accepts
BoTorch's objectives, posterior transforms, samplers and callables where the
existing botorch_rs acquisition functions do, and raises BoTorch's exceptions.

## Supported API

| BoTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.acquisition.knowledge_gradient.qKnowledgeGradient` | same path (also `botorch_rs.acquisition`) | one-shot KG: fantasies of the candidates (and pending points) with the sampler's base samples (`SingleTaskGP`, single- and multi-output, `ModelListGP`), value function the posterior mean (through `posterior_transform`) or, with an MC `objective`, `qSimpleRegret` with `inner_sampler`; `current_value`, `X_pending`, `get_augmented_q_batch_size`, `extract_candidates`, `evaluate(X, bounds, num_restarts, raw_samples, options)` (inner maximization: `gen_value_function_initial_conditions` and one joint L-BFGS-B problem). Exact gradients also with MC objectives (BoTorch's are wrong, #3427) |
| `qMultiFidelityKnowledgeGradient`, `ProjectedAcquisitionFunction` | same path | `cost_aware_utility` (`InverseCostWeightedUtility`, `GenericCostAwareUtility`), `project` / `expand` (native for `functools.partial`s of `project_to_target_fidelity` / `expand_trace_observations`, else torch callables), `valfunc_cls` / `valfunc_argfac` for the native value functions `PosteriorMean`, `PosteriorStandardDeviation`, `UpperConfidenceBound`, `(Log)ExpectedImprovement`, `(Log)ProbabilityOfImprovement`, `ScalarizedPosteriorMean`, `qSimpleRegret`, `qExpectedImprovement`, `qProbabilityOfImprovement`, `qUpper/LowerConfidenceBound`, `qPosteriorStandardDeviation` |
| `botorch.optim.initializers.gen_one_shot_kg_initial_conditions` | `botorch_rs.optim.initializers` (also used by `optimize_acqf` for qKG / qMFKG) | `frac_random`, `num_inner_restarts`, `raw_inner_samples`, `eta`; fixed features and linear constraints |
| `botorch.acquisition.monte_carlo.qNoisyExpectedImprovement` | same path | `cache_root` (BoTorch's low-rank update of the cached baseline root with its base-sample layout), `prune_baseline`, `X_pending`, objectives, outcome `constraints` / `eta`, `X_baseline`; keyword-only `lower_bound_points` |
| `qLowerConfidenceBound`, `qPosteriorStandardDeviation` | same path | sample-reducing like `qUpperConfidenceBound` |
| `botorch.acquisition.analytic.NoisyExpectedImprovement`, `LogNoisyExpectedImprovement` | same path | noiseless fantasy models of a fixed-noise `SingleTaskGP` (BoTorch's `1e-7` noise); keyword-only `sampler` for explicit fantasy base samples; `best_f` of every fantasy |
| `LogProbabilityOfFeasibility`, `ScalarizedPosteriorMean` | same path | multi-output constraints `{i: (lower, upper)}`; `weights` of the `q` points |
| `qAnalyticProbabilityOfImprovement` | same path | MVNXPB (bivariate conditioning, GGE ordering, Genz's bivariate normal) ported step by step; exact gradients by forward-mode dual numbers through MVNXPB. Rust: `botorch_core::utils::mvnxpb::{bvn, bvnmom, mvnxpb_log_prob}` |
| `botorch.acquisition.max_value_entropy_search.qMaxValueEntropy`, `qLowerBoundMaxValueEntropy`, `qMultiFidelityMaxValueEntropy`, `qMultiFidelityLowerBoundMaxValueEntropy`, `MaxValueBase` | same path | Gumbel or Thompson max values over candidate set, training inputs and pending points; `posterior_max_values` (settable), fantasies of the pending points (qMES), GIBBON's repulsion; `num_fantasies`, `num_y_samples`, `use_gumbel`, `maximize`, `posterior_transform`, `train_inputs`, `cost_aware_utility`, `project`, `expand`. Keyword-only `seed`, `sampler`, `fantasies_sampler` |
| `_sample_max_value_Gumbel`, `_sample_max_value_Thompson` | same path | keyword-only `uniforms` / `base_samples` or `seed` |
| `botorch.acquisition.joint_entropy_search.qJointEntropySearch` | same path | `"LB"` and `"MC"`, `condition_noiseless`, `posterior_transform`, `X_pending`, `num_samples` (keyword-only `sampler`); conditioning per optimum and output |
| `botorch.acquisition.predictive_entropy_search.qPredictiveEntropySearch` | same path | the multi-objective PES with one optimal input per sample; `ep_iterations`, `ep_damping`, `cached_factors` |
| `botorch.acquisition.utils.get_optimal_samples` | same path | Matheron paths optimized like `optimize_posterior_samples` (raw Sobol points, top-`num_restarts`, one joint L-BFGS-B problem, `maxiter=2000`); `posterior_transform` or `objective`, `return_transformed`; keyword-only `num_features`, `seed`, `draws`, `candidates`, `maxiter` |
| `botorch.acquisition.thompson_sampling.PathwiseThompsonSampling` | same path | `objective`, `posterior_transform`, `redraw(batch_size)`, `batch_size`, BoTorch's batch-size error; keyword-only `num_features`, `seed`, `redraw(..., draws=...)` |
| `botorch.acquisition.active_learning.qNegIntegratedPosteriorVariance` | same path | closed form of the variance after conditioning (no fantasies needed for exact GPs; `sampler` accepted), `posterior_transform`, `X_pending` |
| `botorch.acquisition.penalized.*` | same path | `L2Penalty`, `L1Penalty`, `GaussianPenalty`, `GroupLassoPenalty`, `L0Approximation`, `L0PenaltyApprox` (native in `PenalizedAcquisitionFunction`; BoTorch's own modules and other torch callables through torch), `L1PenaltyObjective`, `L0PenaltyApproxObjective`, `PenalizedMCObjective` (native, also BoTorch's), `group_lasso_regularizer`, `narrow_gaussian`, `nnz_approx` |
| `botorch.acquisition.proximal.ProximalAcquisitionFunction` | same path | `transformed_weighting`, `beta` (softplus), `Normalize` input transforms, `ModelListGP` with equal training inputs |
| `botorch.acquisition.fixed_feature.FixedFeatureAcquisitionFunction` | same path | values per fixed column or per point (`q x d_f`), `X_pending` setter, `_construct_X_full` |
| `botorch.acquisition.prior_guided.PriorGuidedAcquisitionFunction` | same path | prior: a torch callable (gradients by `torch.autograd`); sample-reducing MC acquisition functions with `q > 1` and `X_pending`, analytic ones with `q = 1` |
| `botorch.acquisition.decoupled.DecoupledAcquisitionFunction` | same path | the abstract base class: `X_evaluation_mask`, `set_X_pending(X_pending, X_pending_evaluation_mask)`, `construct_evaluation_mask` |
| `botorch.acquisition.OneShotAcquisitionFunction` | `botorch_rs.acquisition` | base class (now exported) |

## Not ported in this area

* `qMultiStepLookahead` (and `make_best_f`, `warmstart_multistep`), the
  input constructors with `get_acquisition_function` /
  `get_acqf_input_constructor`.
* `PairwiseMCPosteriorVariance`, `qBayesianActiveLearningByDisagreement`,
  `PairwiseBayesianActiveLearningByDisagreement`, the expected utility of the
  best option (need pairwise / fully Bayesian models), `LearnedObjective`.
* `SampleReducingMCAcquisitionFunction` and
  `LogImprovementMCAcquisitionFunction` as public base classes (the Rust
  acquisition functions implement the sample reduction; Python subclasses
  that override `_sample_forward` are not supported).
* Value functions of the knowledge gradient other than the native list
  above (arbitrary Python classes), and argument factories whose arguments
  depend on the fantasy model.
* The Python classes `MVNXPB` (`augment`, `concat`, ...) and
  `PivotedCholesky`, and Python access to `bvn` / `bvnmom` (Rust only).
* t-batched candidate sets of `qMaxValueEntropy` (`candidate_set` with a batch
  dimension), t-batched fixed values of `FixedFeatureAcquisitionFunction`,
  batched models in the proximal and entropy-search acquisition functions,
  fully Bayesian models (BoTorch's ensemble paths).

## Deviations from BoTorch (migration notes)

* **Knowledge gradient.** `valfunc_argfac` is called once with the current
  model (BoTorch calls it with every fantasy model). `evaluate` draws its
  random restarts from botorch_rs's generator (`options["seed"]`) and applies
  a cost-aware utility to the candidates as `forward` does. The gradient with
  an MC objective or an MC value function is exact (BoTorch's misses the
  fantasy covariances, meta-pytorch/botorch#3427). Multi-output models work
  with a single t-batch (BoTorch raises a reshape error).
* **qNEI.** The pruning sampler (2048 Sobol samples) and the 32 random points
  of the objective's lower bound (when a sample has no feasible baseline
  point) are seeded from the sampler's seed (BoTorch: unseeded); explicit
  points with `lower_bound_points`.
* **Max-value entropy search.** The max values come from botorch_rs's
  generator (`seed`) and can be set through `posterior_max_values`. The
  training inputs appended to the candidate set are the raw ones (BoTorch
  appends `model.train_inputs`, which a model with an input transform holds
  transformed, and transforms them again). The Gumbel quantiles stay in
  float64 (BoTorch rounds them to float32). GIBBON's repulsion uses the
  posterior cross-covariances in the outcome space of the posterior (BoTorch
  mixes outcome-transformed and untransformed covariances and skips the input
  transform). `qMultiFidelityLowerBoundMaxValueEntropy` uses the `X_pending`
  of its constructor (BoTorch drops it).
* **qJointEntropySearch.** The model is conditioned on the raw optimal inputs
  (meta-pytorch/botorch#3399), the `"MC"` estimate compares with the
  transformed optimal values (#3399), the conditioning noise enters the
  conditioned posterior mean (#3396), and the `"MC"` estimate of `q > 1`
  points (also through pending points) is the information gain of the batch
  (meta-pytorch/botorch#3428; e.g. additive over independent points, which
  BoTorch's is not). For `q = 1` and with `"LB"` both agree.
* **qPredictiveEntropySearch.** As the multi-objective PES: raw training
  inputs (#3421) and the fallback / damping of every candidate set on its
  own. Kept from BoTorch: the factors of the training inputs are not used at
  the candidates (meta-pytorch/botorch#3429).
* **PathwiseThompsonSampling.** The paths come from botorch_rs's generator
  (`seed`, a new stream per redraw) or explicit `draws`. With a posterior
  transform and `q = 1` the result has the shape of the t-batch (BoTorch
  sums over the t-batch).
* **qNegIntegratedPosteriorVariance.** Exact gradients (BoTorch's are wrong,
  `fast_pred_var`); the variance after conditioning in closed form, so the
  `sampler` does not matter.
* **qAnalyticProbabilityOfImprovement.** Exact gradients for every t-batch
  (BoTorch's are wrong when the members of a t-batch need different pivots,
  and NaN where the value is 1); the quadrature order of the polar
  integration of `bvn` is chosen per pair (BoTorch: from the largest
  correlation of the whole t-batch; differences at the quadrature error,
  ~1e-16).
* **PriorGuidedAcquisitionFunction.** For a sample-reducing acquisition
  function, `log` must match the scale of its utilities (log acquisition
  functions: `log=True`), otherwise `UnsupportedError`; BoTorch also
  multiplies log-utilities by the prior.
* **ProximalAcquisitionFunction** supports `Normalize` input transforms
  (not `ChainedInputTransform` / `InputPerturbation`).

## Equivalence

Tolerances: values 1e-9, gradients 1e-8 (relative, mixed elementwise /
normwise; Rust: entries within 1e-14 count as equal). Where BoTorch is wrong
the reference is a fixed BoTorch (the `botorch-kg` branch of #3427, the
#3399 branch, BoTorch without `fast_pred_var`, BoTorch one candidate set at a
time) or central differences of botorch_rs's values (step 1e-6, tolerance
1e-7 in Python and 1e-6 in Rust: smooth cases agree to 2e-9; rounding of the
values limits the differences to 6.4e-8 for qAnalyticPI on nearly coincident
points, where step 1e-5 agrees to 3e-9). `get_optimal_samples` and
`optimize_acqf`: L-BFGS-B stops on rounding-dependent criteria, so the
optima agree to 1e-6 (values) and candidates to 1e-3 (the square root of the
value tolerance near a quadratic optimum; measured worst below).

### Rust fixtures (`botorch_core/tests/so_acquisition_parity.rs`, BoTorch main)

Six models (single-output RBF, Matern-5/2 with `Normalize`, fixed noise, no
outcome transform, two-output `SingleTaskGP`, `ModelListGP` with a
`Normalize` sub-model; `scripts/generate_fixtures.py --phase
8-so-acquisition`). Gradient entries within 1e-14 count as equal, so exact
agreement shows as 0. qKG with MC objectives: values against BoTorch,
gradients against central differences (BoTorch's are wrong).

| quantity | tolerance | worst error |
|---|---|---|
| FixedFeature gradients | 1e-8 | 0.0e+00 |
| FixedFeature values | 1e-9 | 2.0e-16 |
| LogNEI best_f | 1e-9 | 1.1e-15 |
| LogNEI gradients | 1e-8 | 4.9e-14 |
| LogNEI values | 1e-9 | 2.5e-14 |
| LogProbabilityOfFeasibility gradients | 1e-8 | 3.9e-15 |
| LogProbabilityOfFeasibility values | 1e-9 | 2.4e-15 |
| NEI best_f | 1e-9 | 8.8e-16 |
| NEI gradients | 1e-8 | 0.0e+00 |
| NEI values | 1e-9 | 6.1e-15 |
| PathwiseThompsonSampling gradients | 1e-8 | 3.7e-15 |
| PathwiseThompsonSampling values | 1e-9 | 9.0e-16 |
| Penalized Gaussian gradients | 1e-8 | 0.0e+00 |
| Penalized Gaussian values | 1e-9 | 0.0e+00 |
| Penalized GroupLasso gradients | 1e-8 | 0.0e+00 |
| Penalized GroupLasso values | 1e-9 | 0.0e+00 |
| Penalized L0 gradients | 1e-8 | 0.0e+00 |
| Penalized L0 values | 1e-9 | 0.0e+00 |
| Penalized L1 gradients | 1e-8 | 0.0e+00 |
| Penalized L1 values | 1e-9 | 0.0e+00 |
| Penalized L2 gradients | 1e-8 | 0.0e+00 |
| Penalized L2 values | 1e-9 | 2.2e-17 |
| PriorGuided ExpectedImprovement pending=false gradients | 1e-8 | 0.0e+00 |
| PriorGuided ExpectedImprovement pending=false values | 1e-9 | 4.7e-18 |
| PriorGuided LogExpectedImprovement pending=false gradients | 1e-8 | 2.8e-16 |
| PriorGuided LogExpectedImprovement pending=false values | 1e-9 | 1.8e-16 |
| PriorGuided qExpectedImprovement pending=false gradients | 1e-8 | 0.0e+00 |
| PriorGuided qExpectedImprovement pending=false values | 1e-9 | 9.9e-17 |
| PriorGuided qExpectedImprovement pending=true gradients | 1e-8 | 0.0e+00 |
| PriorGuided qExpectedImprovement pending=true values | 1e-9 | 2.6e-16 |
| PriorGuided qLogExpectedImprovement pending=false gradients | 1e-8 | 0.0e+00 |
| PriorGuided qLogExpectedImprovement pending=false values | 1e-9 | 5.5e-16 |
| PriorGuided qLogExpectedImprovement pending=true gradients | 1e-8 | 0.0e+00 |
| PriorGuided qLogExpectedImprovement pending=true values | 1e-9 | 8.3e-16 |
| Proximal gradients | 1e-8 | 0.0e+00 |
| Proximal values | 1e-9 | 2.0e-18 |
| ScalarizedPosteriorMean gradients | 1e-8 | 0.0e+00 |
| ScalarizedPosteriorMean values | 1e-9 | 8.1e-16 |
| get_optimal_samples inputs | 1e-6 | 6.3e-14 |
| get_optimal_samples outputs | 1e-6 | 9.6e-16 |
| optimize_acqf qKnowledgeGradient: candidates (agreeing restarts) | 1e-3 | 2.7e-15 |
| optimize_acqf qKnowledgeGradient: restarts below BoTorch's value | 1 restart | 0.0e+00 |
| optimize_acqf qLowerBoundMaxValueEntropy: candidates (agreeing restarts) | 1e-3 | 0.0e+00 |
| optimize_acqf qLowerBoundMaxValueEntropy: restarts below BoTorch's value | 1 restart | 0.0e+00 |
| qAnalyticProbabilityOfImprovement gradients | 1e-8 | 2.9e-14 |
| qAnalyticProbabilityOfImprovement values | 1e-9 | 3.3e-15 |
| qEI PenalizedMCObjective gradients | 1e-8 | 0.0e+00 |
| qEI PenalizedMCObjective values | 1e-9 | 2.8e-16 |
| qJES LB gradients | 1e-8 | 7.1e-13 |
| qJES LB values | 1e-9 | 3.1e-12 |
| qJES MC gradients | 1e-8 | 2.9e-14 |
| qJES MC values | 1e-9 | 1.2e-14 |
| qKG current_value gradients | 1e-8 | 0.0e+00 |
| qKG current_value values | 1e-9 | 3.9e-16 |
| qKG objective gradients vs finite differences | 1e-6 (finite differences) | 2.8e-10 |
| qKG objective values | 1e-9 | 5.3e-16 |
| qKG pending gradients | 1e-8 | 0.0e+00 |
| qKG pending values | 1e-9 | 5.7e-16 |
| qKG pm gradients | 1e-8 | 0.0e+00 |
| qKG pm values | 1e-9 | 2.6e-16 |
| qLowerBoundMaxValueEntropy pending gradients | 1e-8 | 0.0e+00 |
| qLowerBoundMaxValueEntropy pending values | 1e-9 | 1.6e-15 |
| qLowerBoundMaxValueEntropy plain gradients | 1e-8 | 0.0e+00 |
| qLowerBoundMaxValueEntropy plain values | 1e-9 | 2.6e-15 |
| qLowerBoundMaxValueEntropy transform gradients | 1e-8 | 0.0e+00 |
| qLowerBoundMaxValueEntropy transform values | 1e-9 | 3.5e-15 |
| qLowerConfidenceBound gradients | 1e-8 | 3.4e-15 |
| qLowerConfidenceBound values | 1e-9 | 7.1e-16 |
| qMFKG cost gradients | 1e-8 | 0.0e+00 |
| qMFKG cost values | 1e-9 | 8.8e-17 |
| qMFKG project gradients | 1e-8 | 0.0e+00 |
| qMFKG project values | 1e-9 | 0.0e+00 |
| qMFKG trace gradients | 1e-8 | 0.0e+00 |
| qMFKG trace values | 1e-9 | 2.1e-16 |
| qMaxValueEntropy minimize gradients | 1e-8 | 0.0e+00 |
| qMaxValueEntropy minimize values | 1e-9 | 1.3e-14 |
| qMaxValueEntropy pending gradients | 1e-8 | 3.1e-14 |
| qMaxValueEntropy pending values | 1e-9 | 3.9e-14 |
| qMaxValueEntropy plain gradients | 1e-8 | 8.7e-15 |
| qMaxValueEntropy plain values | 1e-9 | 5.0e-15 |
| qMaxValueEntropy transform gradients | 1e-8 | 0.0e+00 |
| qMaxValueEntropy transform values | 1e-9 | 2.7e-14 |
| qMultiFidelityLowerBoundMaxValueEntropy project gradients | 1e-8 | 0.0e+00 |
| qMultiFidelityLowerBoundMaxValueEntropy project values | 1e-9 | 5.5e-16 |
| qMultiFidelityMaxValueEntropy project gradients | 1e-8 | 0.0e+00 |
| qMultiFidelityMaxValueEntropy project values | 1e-9 | 4.1e-14 |
| qMultiFidelityMaxValueEntropy trace gradients | 1e-8 | 1.5e-13 |
| qMultiFidelityMaxValueEntropy trace values | 1e-9 | 1.2e-14 |
| qNEI cache_root=false gradients | 1e-8 | 9.4e-12 |
| qNEI cache_root=false values | 1e-9 | 3.0e-12 |
| qNEI cache_root=true gradients | 1e-8 | 2.6e-13 |
| qNEI cache_root=true values | 1e-9 | 2.0e-12 |
| qNIPV gradients | 1e-8 | 0.0e+00 |
| qNIPV values | 1e-9 | 1.6e-15 |
| qPES gradients | 1e-8 | 0.0e+00 |
| qPES values | 1e-9 | 2.9e-15 |
| qPosteriorStandardDeviation gradients | 1e-8 | 5.3e-15 |
| qPosteriorStandardDeviation values | 1e-9 | 2.1e-15 |

### Live parity (`test_parity_so_acquisition.py`)

Against BoTorch main (da00e0401) and 0.18.1 (the worse of the two runs;
`BOTORCH_RS_REQUIRE_BOTORCH=1`). Gradients of qKG with an MC objective were
also compared with the `botorch-kg` branch (#3427): 5.8e-16; qJES `"MC"`
with posterior transforms with the #3399 branch: passes at nominal
tolerances. `optimize_acqf` qKG: candidate tolerance 1e-4.

| quantity | tolerance | worst error |
|---|---|---|
| BoTorch module penalty | 1.0e-09 | 0.0e+00 |
| FixedFeature _construct_X_full | 0.0e+00 | 0.0e+00 |
| FixedFeatureAcquisitionFunction [0, 2]: gradients | 1.0e-08 | 3.8e-16 |
| FixedFeatureAcquisitionFunction [0, 2]: values | 1.0e-09 | 8.7e-16 |
| FixedFeatureAcquisitionFunction [1]: gradients | 1.0e-08 | 0.0e+00 |
| FixedFeatureAcquisitionFunction [1]: values | 1.0e-09 | 0.0e+00 |
| FixedFeatureAcquisitionFunction [2, 0]: gradients | 1.0e-08 | 3.3e-16 |
| FixedFeatureAcquisitionFunction [2, 0]: values | 1.0e-09 | 1.1e-15 |
| Gaussian penalty | 1.0e-09 | 0.0e+00 |
| L0 penalty | 1.0e-09 | 0.0e+00 |
| L1 penalty | 1.0e-09 | 0.0e+00 |
| L2 penalty | 1.0e-09 | 0.0e+00 |
| LogNoisyExpectedImprovement: best_f | 1.0e-09 | 1.7e-16 |
| LogNoisyExpectedImprovement: gradients | 1.0e-08 | 4.5e-14 |
| LogNoisyExpectedImprovement: values | 1.0e-09 | 1.5e-14 |
| LogProbabilityOfFeasibility list: gradients | 1.0e-08 | 3.4e-14 |
| LogProbabilityOfFeasibility list: values | 1.0e-09 | 1.4e-13 |
| LogProbabilityOfFeasibility m1_rbf: gradients | 1.0e-08 | 2.6e-15 |
| LogProbabilityOfFeasibility m1_rbf: values | 1.0e-09 | 2.1e-15 |
| LogProbabilityOfFeasibility m2: gradients | 1.0e-08 | 9.7e-15 |
| LogProbabilityOfFeasibility m2: values | 1.0e-09 | 1.1e-14 |
| NoisyExpectedImprovement: best_f | 1.0e-09 | 1.7e-16 |
| NoisyExpectedImprovement: gradients | 1.0e-08 | 6.8e-15 |
| NoisyExpectedImprovement: values | 1.0e-09 | 2.2e-15 |
| PathwiseThompsonSampling q=1: gradients | 1.0e-08 | 1.8e-15 |
| PathwiseThompsonSampling q=1: values | 1.0e-09 | 1.8e-16 |
| PathwiseThompsonSampling q=2: gradients | 1.0e-08 | 2.2e-15 |
| PathwiseThompsonSampling q=2: values | 1.0e-09 | 1.3e-15 |
| PenalizedAcquisitionFunction BoTorch module q=2: gradients | 1.0e-08 | 2.4e-16 |
| PenalizedAcquisitionFunction BoTorch module q=2: values | 1.0e-09 | 2.7e-16 |
| PenalizedAcquisitionFunction BoTorch module: gradients | 1.0e-08 | 1.6e-16 |
| PenalizedAcquisitionFunction BoTorch module: values | 1.0e-09 | 4.7e-17 |
| PenalizedAcquisitionFunction Gaussian q=2: gradients | 1.0e-08 | 4.1e-17 |
| PenalizedAcquisitionFunction Gaussian q=2: values | 1.0e-09 | 6.1e-17 |
| PenalizedAcquisitionFunction Gaussian: gradients | 1.0e-08 | 9.5e-17 |
| PenalizedAcquisitionFunction Gaussian: values | 1.0e-09 | 0.0e+00 |
| PenalizedAcquisitionFunction GroupLasso: gradients | 1.0e-08 | 1.7e-16 |
| PenalizedAcquisitionFunction GroupLasso: values | 1.0e-09 | 0.0e+00 |
| PenalizedAcquisitionFunction L0 q=2: gradients | 1.0e-08 | 7.9e-17 |
| PenalizedAcquisitionFunction L0 q=2: values | 1.0e-09 | 8.0e-17 |
| PenalizedAcquisitionFunction L0: gradients | 1.0e-08 | 1.5e-16 |
| PenalizedAcquisitionFunction L0: values | 1.0e-09 | 0.0e+00 |
| PenalizedAcquisitionFunction L1 q=2: gradients | 1.0e-08 | 2.7e-16 |
| PenalizedAcquisitionFunction L1 q=2: values | 1.0e-09 | 9.8e-17 |
| PenalizedAcquisitionFunction L1: gradients | 1.0e-08 | 1.9e-16 |
| PenalizedAcquisitionFunction L1: values | 1.0e-09 | 0.0e+00 |
| PenalizedAcquisitionFunction L2 q=2: gradients | 1.0e-08 | 2.4e-16 |
| PenalizedAcquisitionFunction L2 q=2: values | 1.0e-09 | 2.7e-16 |
| PenalizedAcquisitionFunction L2: gradients | 1.0e-08 | 1.6e-16 |
| PenalizedAcquisitionFunction L2: values | 1.0e-09 | 4.7e-17 |
| PriorGuided MC log=False: gradients | 1.0e-08 | 3.7e-16 |
| PriorGuided MC log=False: values | 1.0e-09 | 0.0e+00 |
| PriorGuided MC log=True: gradients | 1.0e-08 | 3.5e-15 |
| PriorGuided MC log=True: values | 1.0e-09 | 0.0e+00 |
| PriorGuided analytic log=False: gradients | 1.0e-08 | 1.7e-15 |
| PriorGuided analytic log=False: values | 1.0e-09 | 1.5e-15 |
| PriorGuided analytic log=True: gradients | 1.0e-08 | 9.8e-16 |
| PriorGuided analytic log=True: values | 1.0e-09 | 2.5e-16 |
| ProjectedAcquisitionFunction: gradients | 1.0e-08 | 6.2e-16 |
| ProjectedAcquisitionFunction: values | 1.0e-09 | 9.8e-17 |
| ProximalAcquisitionFunction: gradients | 1.0e-08 | 4.9e-15 |
| ProximalAcquisitionFunction: values | 1.0e-09 | 7.2e-15 |
| ScalarizedPosteriorMean: gradients | 1.0e-08 | 1.2e-15 |
| ScalarizedPosteriorMean: values | 1.0e-09 | 5.9e-17 |
| _sample_max_value_Gumbel | 1.0e-09 | 8.6e-17 |
| _sample_max_value_Gumbel vs BoTorch (float32 quantiles) | 1.0e-06 | 2.6e-08 |
| _sample_max_value_Thompson | 1.0e-09 | 2.5e-15 |
| get_optimal_samples: inputs | 1.0e-06 | 6.7e-15 |
| get_optimal_samples: outputs | 1.0e-06 | 4.3e-15 |
| optimize_acqf GIBBON candidate | 1.0e-03 | 2.5e-16 |
| optimize_acqf GIBBON value | 1.0e-06 | 1.7e-16 |
| optimize_acqf GIBBON: BoTorch's value at the candidate | 1.0e-06 | 0.0e+00 |
| optimize_acqf qKG candidate | 1.0e-04 | 1.8e-05 |
| optimize_acqf qKG value | 1.0e-06 | 1.1e-12 |
| optimize_acqf qNEI candidate | 1.0e-03 | 1.6e-07 |
| optimize_acqf qNEI value | 1.0e-06 | 1.1e-12 |
| optimize_acqf qNEI: BoTorch's value at the candidate | 1.0e-06 | 1.1e-12 |
| optimize_acqf qNIPV candidate | 1.0e-03 | 1.2e-04 |
| optimize_acqf qNIPV value | 1.0e-06 | 3.6e-09 |
| optimize_acqf qNIPV: BoTorch's value at the candidate | 1.0e-06 | 3.6e-09 |
| qAnalyticPI where BoTorch's gradient is NaN: vs finite differences | 1.0e-07 | 0.0e+00 |
| qAnalyticProbabilityOfImprovement: gradients | 1.0e-08 | 3.4e-14 |
| qAnalyticProbabilityOfImprovement: values | 1.0e-09 | 1.3e-14 |
| qEI with BoTorch's PenalizedMCObjective(L0): gradients | 1.0e-08 | 1.8e-15 |
| qEI with BoTorch's PenalizedMCObjective(L0): values | 1.0e-09 | 3.5e-15 |
| qEI with BoTorch's PenalizedMCObjective(L1): gradients | 1.0e-08 | 1.9e-15 |
| qEI with BoTorch's PenalizedMCObjective(L1): values | 1.0e-09 | 3.0e-15 |
| qEI with botorch_rs's PenalizedMCObjective(L0): gradients | 1.0e-08 | 1.8e-15 |
| qEI with botorch_rs's PenalizedMCObjective(L0): values | 1.0e-09 | 3.5e-15 |
| qEI with botorch_rs's PenalizedMCObjective(L1): gradients | 1.0e-08 | 1.9e-15 |
| qEI with botorch_rs's PenalizedMCObjective(L1): values | 1.0e-09 | 3.0e-15 |
| qJointEntropySearch LB: gradients | 1.0e-08 | 3.7e-14 |
| qJointEntropySearch LB: values | 1.0e-09 | 7.0e-14 |
| qJointEntropySearch MC: gradients | 1.0e-08 | 6.2e-14 |
| qJointEntropySearch MC: values | 1.0e-09 | 2.3e-15 |
| qKG current_value: gradients | 1.0e-08 | 8.4e-16 |
| qKG current_value: values | 1.0e-09 | 9.4e-17 |
| qKG mc_objective: gradients vs finite differences | 1.0e-07 | 6.3e-10 |
| qKG mc_objective: values | 1.0e-09 | 2.1e-16 |
| qKG objective list: gradients vs finite differences | 1.0e-07 | 6.4e-10 |
| qKG objective list: values | 1.0e-09 | 7.1e-17 |
| qKG objective m2: gradients vs finite differences | 1.0e-07 | 1.3e-09 |
| qKG objective m2: values | 1.0e-09 | 6.2e-16 |
| qKG pending: gradients | 1.0e-08 | 6.7e-16 |
| qKG pending: values | 1.0e-09 | 7.8e-17 |
| qKG pm: gradients | 1.0e-08 | 8.4e-16 |
| qKG pm: values | 1.0e-09 | 7.8e-17 |
| qKG transform list: gradients | 1.0e-08 | 2.1e-15 |
| qKG transform list: values | 1.0e-09 | 1.5e-16 |
| qKG transform m2: gradients | 1.0e-08 | 1.8e-15 |
| qKG transform m2: values | 1.0e-09 | 2.9e-16 |
| qLowerBoundMaxValueEntropy minimize: gradients | 1.0e-08 | 1.5e-15 |
| qLowerBoundMaxValueEntropy minimize: values | 1.0e-09 | 2.9e-15 |
| qLowerBoundMaxValueEntropy pending: gradients | 1.0e-08 | 6.0e-16 |
| qLowerBoundMaxValueEntropy pending: values | 1.0e-09 | 3.6e-15 |
| qLowerBoundMaxValueEntropy plain: gradients | 1.0e-08 | 4.8e-15 |
| qLowerBoundMaxValueEntropy plain: values | 1.0e-09 | 3.0e-15 |
| qLowerBoundMaxValueEntropy transform: gradients | 1.0e-08 | 1.3e-15 |
| qLowerBoundMaxValueEntropy transform: values | 1.0e-09 | 3.3e-15 |
| qLowerBoundMaxValueEntropy: posterior_max_values | 0.0e+00 | 0.0e+00 |
| qLowerConfidenceBound: gradients | 1.0e-08 | 8.3e-15 |
| qLowerConfidenceBound: values | 1.0e-09 | 7.3e-16 |
| qMFKG cost: gradients | 1.0e-08 | 1.1e-16 |
| qMFKG cost: values | 1.0e-09 | 6.0e-17 |
| qMFKG expand: gradients | 1.0e-08 | 1.2e-15 |
| qMFKG expand: values | 1.0e-09 | 5.8e-17 |
| qMFKG project: gradients | 1.0e-08 | 6.4e-16 |
| qMFKG project: values | 1.0e-09 | 9.9e-17 |
| qMFKG project_lambda: gradients | 1.0e-08 | 6.4e-16 |
| qMFKG project_lambda: values | 1.0e-09 | 9.9e-17 |
| qMFKG valfunc_ei: gradients vs finite differences | 1.0e-07 | 3.7e-10 |
| qMFKG valfunc_ei: values | 1.0e-09 | 4.2e-16 |
| qMFKG valfunc_qei: gradients vs finite differences | 1.0e-07 | 2.4e-10 |
| qMFKG valfunc_qei: values | 1.0e-09 | 5.0e-16 |
| qMFKG valfunc_ucb_min: gradients vs finite differences | 1.0e-07 | 6.4e-10 |
| qMFKG valfunc_ucb_min: values | 1.0e-09 | 0.0e+00 |
| qMaxValueEntropy minimize: gradients | 1.0e-08 | 2.6e-15 |
| qMaxValueEntropy minimize: values | 1.0e-09 | 4.8e-14 |
| qMaxValueEntropy pending: gradients | 1.0e-08 | 6.8e-15 |
| qMaxValueEntropy pending: values | 1.0e-09 | 3.6e-14 |
| qMaxValueEntropy plain: gradients | 1.0e-08 | 7.4e-15 |
| qMaxValueEntropy plain: values | 1.0e-09 | 4.3e-14 |
| qMaxValueEntropy transform: gradients | 1.0e-08 | 1.6e-15 |
| qMaxValueEntropy transform: values | 1.0e-09 | 3.6e-14 |
| qMaxValueEntropy: posterior_max_values | 0.0e+00 | 0.0e+00 |
| qMultiFidelityLowerBoundMaxValueEntropy gibbon: gradients | 1.0e-08 | 1.7e-15 |
| qMultiFidelityLowerBoundMaxValueEntropy gibbon: values | 1.0e-09 | 2.0e-15 |
| qMultiFidelityLowerBoundMaxValueEntropy gibbon_pending: gradients | 1.0e-08 | 2.5e-15 |
| qMultiFidelityLowerBoundMaxValueEntropy gibbon_pending: values | 1.0e-09 | 7.4e-16 |
| qMultiFidelityMaxValueEntropy pending: gradients | 1.0e-08 | 9.5e-15 |
| qMultiFidelityMaxValueEntropy pending: values | 1.0e-09 | 8.0e-15 |
| qMultiFidelityMaxValueEntropy project: gradients | 1.0e-08 | 2.3e-14 |
| qMultiFidelityMaxValueEntropy project: values | 1.0e-09 | 1.1e-14 |
| qMultiFidelityMaxValueEntropy trace: gradients | 1.0e-08 | 7.5e-15 |
| qMultiFidelityMaxValueEntropy trace: values | 1.0e-09 | 1.6e-14 |
| qNEI cache_root=False: gradients | 1.0e-08 | 6.1e-11 |
| qNEI cache_root=False: values | 1.0e-09 | 6.2e-11 |
| qNEI cache_root=True: gradients | 1.0e-08 | 8.3e-11 |
| qNEI cache_root=True: values | 1.0e-09 | 1.0e-10 |
| qNEI pruned baseline (symmetric difference) | 2.0e+00 | 0.0e+00 |
| qNEI with PenalizedMCObjective(L0): gradients | 1.0e-08 | 5.6e-15 |
| qNEI with PenalizedMCObjective(L0): values | 1.0e-09 | 6.6e-14 |
| qNEI with PenalizedMCObjective(L1): gradients | 1.0e-08 | 2.7e-15 |
| qNEI with PenalizedMCObjective(L1): values | 1.0e-09 | 6.1e-14 |
| qNegIntegratedPosteriorVariance: gradients | 1.0e-08 | 3.6e-15 |
| qNegIntegratedPosteriorVariance: gradients vs finite differences | 1.0e-07 | 3.9e-09 |
| qNegIntegratedPosteriorVariance: values | 1.0e-09 | 1.0e-15 |
| qPosteriorStandardDeviation: gradients | 1.0e-08 | 2.4e-15 |
| qPosteriorStandardDeviation: values | 1.0e-09 | 1.2e-15 |
| qPredictiveEntropySearch: gradients | 1.0e-08 | 2.1e-15 |
| qPredictiveEntropySearch: values | 1.0e-09 | 5.6e-15 |

### Randomized sweep (`test_equivalence_so_acquisition.py`, `-m equivalence`)

12 scenarios (10 `SingleTaskGP`s with n = 6, 10, 14, d = 1-3, one or two
outputs, random fixed noise / `Normalize` / `Standardize` / kernel / input
range, and two `ModelListGP`s), seeded options, against BoTorch main and
0.18.1. qNEI is compared with the regime of BoTorch's outcomes on twins
whose training inputs are perturbed at 1e-14 .. 1e-11 (its joint
posterior over the training points is nearly singular, and BoTorch's own
result jumps between Cholesky-jitter regimes, e.g. 2.5e-4 vs 1.3e-4 for
the same candidates in a one-dimensional fixed-noise model): within 10x
the regime's spread (the scaled excess must be at most 1).

| quantity | tolerance | worst error |
|---|---|---|
| PathwiseThompsonSampling: gradients | 1.0e-08 | 4.9e-15 |
| PathwiseThompsonSampling: values | 1.0e-09 | 2.3e-15 |
| get_optimal_samples: inputs | 1.0e-06 | 7.9e-15 |
| get_optimal_samples: outputs | 1.0e-06 | 2.1e-15 |
| qAnalyticPI where BoTorch's gradient is NaN: vs finite differences | 1.0e-07 | 6.4e-08 |
| qAnalyticProbabilityOfImprovement: gradients | 1.0e-08 | 4.9e-15 |
| qAnalyticProbabilityOfImprovement: values | 1.0e-09 | 2.4e-14 |
| qJointEntropySearch LB: gradients | 1.0e-08 | 2.0e-13 |
| qJointEntropySearch LB: values | 1.0e-09 | 1.9e-13 |
| qJointEntropySearch MC: gradients | 1.0e-08 | 1.9e-14 |
| qJointEntropySearch MC: values | 1.0e-09 | 3.5e-14 |
| qKG: gradients | 1.0e-08 | 6.7e-15 |
| qKG: gradients vs finite differences | 1.0e-07 | 1.0e-08 |
| qKG: values | 1.0e-09 | 1.3e-15 |
| qLowerBoundMaxValueEntropy: gradients | 1.0e-08 | 5.7e-14 |
| qLowerBoundMaxValueEntropy: values | 1.0e-09 | 8.9e-14 |
| qLowerConfidenceBound: gradients | 1.0e-08 | 1.6e-14 |
| qLowerConfidenceBound: values | 1.0e-09 | 1.6e-15 |
| qMaxValueEntropy: gradients | 1.0e-08 | 5.2e-14 |
| qMaxValueEntropy: values | 1.0e-09 | 6.8e-14 |
| qNEI values and gradients (scaled) | 1.0e+00 | 4.1e-02 |
| qNegIntegratedPosteriorVariance: gradients | 1.0e-08 | 2.5e-14 |
| qNegIntegratedPosteriorVariance: values | 1.0e-09 | 6.4e-15 |
| qPosteriorStandardDeviation: gradients | 1.0e-08 | 6.4e-11 |
| qPosteriorStandardDeviation: values | 1.0e-09 | 2.5e-13 |
| qNEI values (unscaled, nearest BoTorch outcome) | (recorded) | 7.1e-10 |
| qNEI gradients (unscaled, nearest BoTorch outcome) | (recorded; BoTorch's spread 6.8e-8) | 2.8e-8 |

### Performance

Value and gradient of a batch of 32 candidate sets on a `SingleTaskGP` with
n = 30, d = 4 (BoTorch main with one torch thread; botorch_rs with 5 Rayon
threads; Windows laptop; an ad-hoc timing script, not part of the repository).

| acquisition function (batch of 32 candidate sets) | BoTorch | botorch_rs | speed-up |
|---|---|---|---|
| qKnowledgeGradient (q = 2, 16 fantasies) | 47.3 ms | 1.74 ms | 27x |
| qNoisyExpectedImprovement (q = 2, 256 samples, baseline 30) | 16.2 ms | 3.83 ms | 4x |
| qMaxValueEntropy (q = 1) | 14.0 ms | 0.78 ms | 18x |
| qLowerBoundMaxValueEntropy (q = 1) | 8.6 ms | 0.70 ms | 12x |
| qJointEntropySearch LB (q = 2, 8 optima) | 23.8 ms | 1.52 ms | 16x |
| qAnalyticProbabilityOfImprovement (q = 4) | 66.0 ms | 1.46 ms | 45x |
| qNegIntegratedPosteriorVariance (q = 2, 256 points) | 92.9 ms | 1.21 ms | 77x |

## BoTorch bugs found

Reproductions run against BoTorch main (da00e0401) and 0.18.1, with
`X = torch.rand(8, 2, dtype=torch.float64)`, `Y = torch.sin(6 * X).sum(-1,
keepdim=True)`, `model = SingleTaskGP(X, Y).eval()` unless stated otherwise.

1. **qKnowledgeGradient / qMultiFidelityKnowledgeGradient: wrong gradients
   with MC objectives and MC value functions** (reported:
   meta-pytorch/botorch#3427, fixed on the `fix/kg-mc-objective-gradients`
   branch). The value function is evaluated under `fast_pred_var`, whose
   cached root does not track the fantasy covariances' dependence on `X`.
   botorch_rs's gradients agree with central differences (worst 2.8e-10) and
   with the fixed branch (5.8e-16).

2. **qKnowledgeGradient of a multi-output model fails for a single
   t-batch** (new).

   ```python
   m2 = SingleTaskGP(X, torch.cat([Y, -Y], -1)).eval()
   tf = ScalarizedPosteriorTransform(torch.tensor([0.5, 0.5], dtype=torch.float64))
   kg = qKnowledgeGradient(m2, num_fantasies=2, posterior_transform=tf)
   kg(torch.rand(1, 3, 2, dtype=torch.float64))
   # RuntimeError: shape '[2, 1, 2, 1]' is invalid for input of size 8
   kg(torch.rand(2, 3, 2, dtype=torch.float64))  # two t-batches work
   ```

3. **qNegIntegratedPosteriorVariance: wrong gradients** (new). The fantasy
   model's posterior is computed under BoTorch's `gpt_posterior_settings`
   (`fast_pred_var`), whose cached root misses the dependence of the fantasy
   covariances on `X`; without `fast_pred_var` the gradient is right.

   ```python
   acqf = qNegIntegratedPosteriorVariance(model, torch.rand(16, 2, dtype=torch.float64))
   x = torch.rand(1, 1, 2, dtype=torch.float64, requires_grad=True)
   acqf(x).sum().backward()
   # x.grad: [-0.038, 0.864]; central differences: [0.072, 0.072]
   ```

4. **qLowerBoundMaxValueEntropy (GIBBON) with pending points and an outcome
   or input transform: wrong repulsion term** (new).
   `_compute_information_gain` takes the cross-covariances `A` from
   `self.model(X_batches)`, which in eval mode skips the input transform and
   returns the posterior in the outcome-transformed space, and combines them
   with `B` and `variance_m` from `self.model.posterior(...)` in the original
   space. With the default `Standardize`, BoTorch's values differ from
   botorch_rs's by 50% (relative) in the parity tests, and its repulsion
   term is often the log of the clamped determinant (values near -12);
   without transforms botorch_rs and BoTorch agree to 2e-14.

   ```python
   cs = torch.rand(20, 2, dtype=torch.float64)
   gibbon = qLowerBoundMaxValueEntropy(model, cs, X_pending=torch.rand(2, 2, dtype=torch.float64))
   gibbon(torch.rand(3, 1, 2, dtype=torch.float64))
   ```

5. **qMultiFidelityLowerBoundMaxValueEntropy ignores the `X_pending` of its
   constructor** (new): `__init__` does not pass it to
   `qMultiFidelityMaxValueEntropy.__init__`, so neither the max values nor
   the repulsion use it.

   ```python
   acqf = qMultiFidelityLowerBoundMaxValueEntropy(model, cs, X_pending=torch.rand(2, 2, dtype=torch.float64))
   acqf.X_pending  # None
   ```

6. **MaxValueBase appends the transformed training inputs to the candidate
   set** (new; the same pattern as meta-pytorch/botorch#3399 / #3421):
   `model.train_inputs[0]` of a model with an input transform in eval mode is
   transformed, and the posterior over the candidate set transforms it again,
   so the max values are sampled at the wrong points.

   ```python
   Xn = 10 + 5 * torch.rand(8, 2, dtype=torch.float64)
   m = SingleTaskGP(Xn, Y, input_transform=Normalize(d=2)).eval()
   mes = qMaxValueEntropy(m, 10 + 5 * torch.rand(4, 2, dtype=torch.float64))
   mes.candidate_set[-2:]  # in [0, 1]^2, not the training inputs in [10, 15]^2
   ```

7. **`_sample_max_value_Gumbel` rounds its quantiles to float32** (new):
   `quantiles[i, :] = torch.tensor([brentq(...) for p in ...])` builds a
   float32 tensor from the Python floats. The Gumbel samples are off by up to
   2.6e-8 (relative) from the float64 computation (botorch_rs agrees with the
   float64 reference to 8.6e-17).

8. **PathwiseThompsonSampling with a posterior transform and `q = 1` sums
   over the t-batch** (new): the transform drops the output dimension, then
   `IdentityMCObjective` squeezes the `q` dimension, and `sum(-1)` sums over
   the t-batch.

   ```python
   m2 = SingleTaskGP(X, torch.cat([Y, -Y], -1)).eval()
   ts = PathwiseThompsonSampling(m2, posterior_transform=tf)
   ts(torch.rand(3, 1, 2, dtype=torch.float64)).shape  # torch.Size([]), expected (3,)
   ```

9. **MVNXPB: wrong covariance gradients in a batch whose members pivot
   differently** (new), hence wrong gradients of
   `qAnalyticProbabilityOfImprovement` for t-batches (e.g. in
   `optimize_acqf`: 3e-1 relative). `PivotedCholesky.pivot_` swaps entries
   with `swap_along_dim_` on index lists that contain duplicates for the
   members that do not pivot; the backward of the in-place swap then counts
   their gradient twice. One candidate set at a time BoTorch's gradients
   agree with botorch_rs's (3e-14).

   ```python
   from botorch.utils.probability.mvnxpb import MVNXPB
   cov = torch.tensor([[[1.3, 0.6], [0.6, 0.8]]] * 2, dtype=torch.float64, requires_grad=True)
   upper = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64)  # member 0 pivots
   bounds = torch.stack([torch.full_like(upper, -float("inf")), upper], -1)
   (g,) = torch.autograd.grad(MVNXPB(cov, bounds).solve().sum(), cov)
   (g1,) = torch.autograd.grad(MVNXPB(cov[1:], bounds[1:]).solve().sum(), cov)
   g[1], g1[1]  # off-diagonal 0.00422 vs 0.00211
   ```

10. **qAnalyticProbabilityOfImprovement: NaN gradients** (new) where the
    probability of no improvement underflows (value 1), and for strongly
    correlated pairs of points (`|r| >= 0.925`, `_bvnu_taylor`) whose lower
    bound is the clipped `-1e6`: there `Phi(-|b| / a) = 0` multiplies
    `safe_exp(-0.5 skh)`, which is clipped near `exp(709)`, and the backward
    multiplies an infinite partial derivative by zero. botorch_rs's gradients
    agree with central differences there (6.4e-8, rounding-limited).

    ```python
    acqf = qAnalyticProbabilityOfImprovement(model, best_f=-10.0)
    x = torch.rand(1, 3, 2, dtype=torch.float64, requires_grad=True)
    acqf(x).sum().backward()  # value 1.0, x.grad all NaN (botorch_rs: 0)

    from botorch.utils.probability.bvn import bvn
    r, yu = (torch.tensor([v], dtype=torch.float64, requires_grad=True) for v in (0.95, 0.095))
    lo = torch.tensor([-1e6], dtype=torch.float64)
    bvn(r, lo, lo, torch.tensor([-0.085], dtype=torch.float64), yu).sum().backward()
    r.grad, yu.grad  # NaN, NaN
    ```

11. **qJointEntropySearch**: the `"MC"` estimate is not the information gain
    of a batch of `q > 1` points (reported: meta-pytorch/botorch#3428); the
    optimal inputs are transformed twice and the `"MC"` estimate ignores the
    posterior transform of the optimal values (meta-pytorch/botorch#3399, open
    PR); the explicit noise of the conditioning does not enter the
    conditioned posterior mean (meta-pytorch/botorch#3396, open PR).
    botorch_rs implements the corrected behaviour; with the #3399 branch,
    BoTorch's `"MC"` estimate with posterior transforms agrees with
    botorch_rs's.

12. **qPredictiveEntropySearch** ignores the EP factors of the training
    inputs at the candidates (reported: meta-pytorch/botorch#3429); botorch_rs
    keeps BoTorch's semantics.
