//! # botorch_core
//!
//! A Rust re-implementation of [BoTorch](https://botorch.org): the
//! Gaussian-process core (`SingleTaskGP` with BoTorch's default priors,
//! constraints and transforms, single- and multi-output, batched, and
//! `ModelListGP`; the exact marginal log likelihood with analytic gradients;
//! `fit_gpytorch_mll` with a port of L-BFGS-B 3.0), analytic and Monte-Carlo
//! acquisition functions with exact gradients (including
//! `ScalarizedPosteriorTransform`, MC objectives, outcome constraints and
//! analytic constrained expected improvement), `optimize_acqf` with box,
//! linear and nonlinear parameter constraints (SLSQP and hit-and-run polytope
//! sampling of initial conditions), quasi-random sampling and synthetic test
//! functions. All computations are in double precision and are validated
//! against BoTorch itself (see `scripts/generate_fixtures.py` and the
//! `tests/*parity*.rs` files).
//!
//! ## Conventions
//!
//! * [`Matrix`] / [`Vector`] are `nalgebra` `DMatrix<f64>` / `DVector<f64>`.
//!   Point sets are `n × d` matrices whose **rows** are points; a candidate set
//!   `X` of BoTorch's shape `1 × q × d` is a `q × d` matrix, a batch
//!   `b × q × d` is a slice of `b` such matrices, and box bounds are `2 × d`
//!   (lower bounds in row 0).
//! * Inputs passed to a model or an acquisition function are in the *raw*
//!   (original) input space; the model applies its input transform. Posteriors
//!   are reported in the *original* outcome space (un-standardized), like
//!   BoTorch's `posterior`.
//! * Hyperparameters are stored in BoTorch's *raw* parameterization. With the
//!   `SingleTaskGP` defaults the constraints are not enforced through a
//!   transform, so raw values equal actual values and the bounds are imposed
//!   by the optimizer.
//! * GPyTorch materializes prior parameters, constraint bounds and initial
//!   values in float32 before casting the model to double precision. The
//!   BoTorch presets in this crate reproduce that rounding (see
//!   [`Prior::rounded_to_f32`]) so that results agree to ~1e-10. The same holds
//!   for the float32 constants of BoTorch's test functions.
//! * Joint posteriors over `q` points and `m` outputs are flattened in
//!   BoTorch's output-major order (index `output·q + point`, the layout of
//!   `MultitaskMultivariateNormal` with `interleaved=False`); see [`posterior`].
//! * Models and acquisition functions are `Send + Sync`; acquisition
//!   functions share their model through an `Arc<dyn Model>`.
//! * Seeds are `Option<u64>`: `Some(seed)` makes results reproducible, `None`
//!   draws a random seed (BoTorch draws from torch's global RNG).
//!
//! ## API overview
//!
//! | BoTorch / GPyTorch | botorch_core |
//! |---|---|
//! | `SingleTaskGP(train_X, train_Y, train_Yvar, covar_module, likelihood, outcome_transform, input_transform)` | [`SingleTaskGP::new`]`(train_x, train_y, `[`SingleTaskGPOptions`]`)` |
//! | `model.posterior(X, observation_noise)` | [`SingleTaskGP::posterior`]`(&x, observation_noise) -> `[`Posterior`]` { mean, covariance }` |
//! | mean/variance and their gradients for `q = 1` | [`SingleTaskGP::posterior_q1_with_grad`]`(&x) -> (mean, variance, dmean_dx, dvariance_dx)` |
//! | joint mean/covariance and their gradients | [`SingleTaskGP::joint_posterior_with_grad`]`(&x) -> `[`JointPosteriorGrad`] (index `j = point·d + dim`); reverse mode [`SingleTaskGP::joint_posterior_vjp`] |
//! | `SingleTaskGP(train_X, train_Y)` with `m > 1` outputs and/or a batch shape | [`BatchedMultiOutputGP::multi_output`]`(train_x, train_y, `[`SingleTaskGPOptions`]`)`, [`BatchedMultiOutputGP::new`]`(train_x, train_y, batch_shape, `[`BatchedMultiOutputGPOptions`]`)` (one matrix per batch element), `From<SingleTaskGP>` |
//! | `model.posterior(X)` over `m` outputs, batched `X` | [`Model::posterior`] -> [`Posterior`]` { mean, covariance, num_outputs }` (output-major; [`Posterior::mean_matrix`], [`Posterior::select_outputs`]); per batch element [`BatchedMultiOutputGP::posterior_at`]`(batch, &x, observation_noise)` |
//! | `ModelListGP(*models)` | [`ModelListGP::new`]`(vec![...])` (block-diagonal joint posterior over the concatenated outputs), [`ModelListGP::posterior_at`]; for other sub-models (e.g. `LCEMGP`s) [`ModelList::new`]`(vec![Arc<dyn Model>, ...])` |
//! | `LCEMGP(train_X, train_Y, task_feature, context_cat_feature, context_emb_feature, embs_dim_list, output_tasks, all_tasks, ...)` | [`LCEMGP::new`]`(train_x, train_y, task_feature, `[`LCEMGPOptions`]`)`; [`LCEMGP::posterior`] over the output tasks, [`LCEMGP::posterior_with_tasks`], [`LCEMGP::mll_value_and_grad`], `fit_gpytorch_mll` (see [`models::contextual_multioutput`]) |
//! | `ExactMarginalLogLikelihood` of a batched model, `SumMarginalLogLikelihood` | [`BatchedMultiOutputGP::mll_values`] (one value per element of `_aug_batch_shape`), [`BatchedMultiOutputGP::mll_values_and_grad`], [`ModelListGP::mll_values`] |
//! | `botorch.models.model.Model` (what acquisition functions use) | the [`Model`] trait, implemented by [`SingleTaskGP`], [`BatchedMultiOutputGP`] and [`ModelListGP`] (unbatched) and [`acquisition::ScalarizedModel`]; joint multi-output posteriors with input derivatives: [`Model::joint_posterior_with_grad`], [`Model::joint_posterior_vjp`], [`Model::point_posteriors`], [`Model::point_posterior_with_grad`] |
//! | `model.named_parameters()` | [`SingleTaskGP::hyperparameters`], [`SingleTaskGP::set_hyperparameters`], [`SingleTaskGP::raw_parameters`], [`SingleTaskGP::parameter_names`] |
//! | `model.train_inputs`, `model.train_targets` | [`SingleTaskGP::train_inputs`], [`SingleTaskGP::train_targets`] (original space), [`SingleTaskGP::transformed_train_inputs`], [`SingleTaskGP::transformed_train_targets`] |
//! | `model.likelihood.noise` | [`SingleTaskGP::likelihood_noise`] |
//! | `ExactMarginalLogLikelihood(likelihood, model)` | [`ExactMarginalLogLikelihood::new`]`(&model)`, `.value()`, `.value_and_grad(&raw_params)` |
//! | `fit_gpytorch_mll(mll)` | [`fit_gpytorch_mll`]`(&mut model, `[`FitOptions::default()`](FitOptions)`)` for any [`Fittable`] model: [`FitResult`] for a [`SingleTaskGP`], [`BatchedFitResult`] (per-element fits) for a [`BatchedMultiOutputGP`], `Vec<BatchedFitResult>` for a [`ModelListGP`] |
//! | `ExpectedImprovement(model, best_f, maximize=...)`, `LogExpectedImprovement`, `ProbabilityOfImprovement`, `LogProbabilityOfImprovement` | [`acquisition::ExpectedImprovement::new`]`(model, best_f, maximize)`, [`acquisition::LogExpectedImprovement`], [`acquisition::ProbabilityOfImprovement`], [`acquisition::LogProbabilityOfImprovement`] |
//! | `UpperConfidenceBound(model, beta, maximize=...)`, `PosteriorMean`, `PosteriorStandardDeviation` | [`acquisition::UpperConfidenceBound::new`]`(model, beta, maximize)`, [`acquisition::PosteriorMean`], [`acquisition::PosteriorStandardDeviation`] |
//! | `qExpectedImprovement(model, best_f, sampler, X_pending=...)`, `qProbabilityOfImprovement(..., tau)`, `qUpperConfidenceBound(model, beta, ...)`, `qSimpleRegret` | [`acquisition::QExpectedImprovement::new`]`(model, best_f, `[`acquisition::McAcquisitionOptions`]` { sampler, x_pending })`, [`acquisition::QProbabilityOfImprovement`], [`acquisition::QUpperConfidenceBound`], [`acquisition::QSimpleRegret`] |
//! | `qLogExpectedImprovement(model, best_f, sampler, X_pending, fat, tau_max, tau_relu)` | [`acquisition::QLogExpectedImprovement::new`]`(model, best_f, `[`acquisition::QLogExpectedImprovementOptions`]`)` |
//! | `qLogNoisyExpectedImprovement(model, X_baseline, sampler, X_pending, prune_baseline, incremental, ...)` | [`acquisition::QLogNoisyExpectedImprovement::new`]`(model, &x_baseline, `[`acquisition::QLogNoisyExpectedImprovementOptions`]`)` |
//! | `ScalarizedPosteriorTransform(weights, offset)`; `posterior_transform=...` of any acquisition function | [`acquisition::ScalarizedPosteriorTransform::new`]`(weights, offset)`; the acquisition function is given [`acquisition::ScalarizedModel::wrap`]`(model, Some(transform))` |
//! | `LogConstrainedExpectedImprovement(model, best_f, objective_index, constraints, maximize)`, `ConstrainedExpectedImprovement` | [`acquisition::LogConstrainedExpectedImprovement::new`]`(model, best_f, objective_index, constraints, maximize)` with `constraints: Vec<(output, lower, upper)>` (`None`: unbounded), [`acquisition::ConstrainedExpectedImprovement`] |
//! | `IdentityMCObjective`, `LinearMCObjective(weights)`, `GenericMCObjective(f)`, `ConstrainedMCObjective(objective, constraints, infeasible_cost, eta)`; `objective=` of the MC acquisition functions | [`acquisition::IdentityMcObjective`], [`acquisition::LinearMcObjective::new`], [`acquisition::GenericMcObjective::new`] (value and gradient closure), [`acquisition::ConstrainedMcObjective::new`]; any [`acquisition::McObjective`] in `McAcquisitionOptions::objective` |
//! | `constraints=[c(Y) <= 0, ...], eta=...` of qEI, qPI, qLogEI, qLogNEI (and `fat` lists of the LogEI family) | [`utils::objective::OutcomeConstraints::new`]`(constraints, etas)` of [`utils::objective::McConstraint`]s ([`utils::objective::LinearOutcomeConstraint`], [`utils::objective::FnOutcomeConstraint`]) in `constraints` of the options; `constraint_fat` |
//! | `qLogProbabilityOfFeasibility(model, constraints, ...)` | [`acquisition::QLogProbabilityOfFeasibility::new`]`(model, constraints, `[`acquisition::QLogProbabilityOfFeasibilityOptions`]`)` |
//! | `compute_smoothed_feasibility_indicator`, `compute_feasibility_indicator`, `compute_best_feasible_objective`, `get_infeasible_cost` | [`utils::objective::compute_smoothed_feasibility_indicator`], [`utils::objective::compute_feasibility_indicator`], [`acquisition::compute_best_feasible_objective`], [`acquisition::get_infeasible_cost`] |
//! | `acqf(X)`, `torch.autograd.grad(acqf(X), X)`, `acqf(X_batch)` | [`AcquisitionFunction::forward`]`(&x)`, [`AcquisitionFunction::forward_with_grad`]`(&x) -> (value, grad)`, [`AcquisitionFunction::forward_many`]`(&[x])` (parallel) |
//! | `acqf.set_X_pending(X_pending)` | [`AcquisitionFunction::set_x_pending`]`(Some(&x_pending))` |
//! | `prune_inferior_points(model, X, num_samples, max_frac, sampler)` | [`acquisition::prune_inferior_points`]`(&*model, &x, &`[`acquisition::PruneOptions`]`)` |
//! | `SobolQMCNormalSampler(sample_shape=[n], seed=...)` | [`sampling::SobolQMCNormalSampler::new`]`(n, seed)`; explicit base samples via [`sampling::SobolQMCNormalSampler::from_base_samples`] |
//! | `draw_sobol_samples(bounds, n, q, seed)`, `draw_sobol_normal_samples(d, n, seed)` | [`utils::sampling::draw_sobol_samples`], [`utils::sampling::draw_sobol_normal_samples`], [`utils::sampling::SobolEngine`] |
//! | `optimize_acqf(acq_function, bounds, q, num_restarts, raw_samples, options, fixed_features, sequential, ...)` | [`optim::optimize_acqf`]`(&acqf, &bounds, q, num_restarts, raw_samples, &`[`optim::OptimizeAcqfOptions`]`) -> (candidates, value)`; all restarts via [`optim::optimize_acqf_detailed`] |
//! | `gen_batch_initial_conditions`, `initialize_q_batch`, `initialize_q_batch_nonneg` | [`optim::gen_batch_initial_conditions`], [`optim::initialize_q_batch`], [`optim::initialize_q_batch_nonneg`] |
//! | `gen_candidates_scipy(initial_conditions, acqf, lower_bounds, upper_bounds)` | [`optim::gen_candidates_lbfgsb`]; with constraints [`optim::gen_candidates_slsqp`] (one SLSQP problem per chunk of restarts) |
//! | `optimize_acqf(..., inequality_constraints, equality_constraints, nonlinear_inequality_constraints, ic_generator, options={"method": ...})` | [`optim::OptimizeAcqfOptions`]` { constraints: `[`optim::ParameterConstraints`]` { inequality, equality, nonlinear }, ic_generator, method, .. }` with [`optim::LinearConstraint::intra_point`] / [`optim::LinearConstraint::inter_point`] and [`optim::NonlinearInequalityConstraint`]s ([`optim::FnNonlinearConstraint`]), [`optim::IcGenerator`], [`optim::OptimizationMethod`] |
//! | `evaluate_feasibility`, `project_to_feasible_space_via_slsqp`, `transform_constraints`, `sample_q_batches_from_polytope` | [`optim::evaluate_feasibility`], [`optim::project_to_feasible_space`], [`optim::transform_constraints`], [`optim::sample_q_batches_from_polytope`] |
//! | `get_polytope_samples`, `HitAndRunPolytopeSampler`, `find_interior_point`, `sample_polytope`, `sparse_to_dense_constraints` | [`utils::polytope::get_polytope_samples`], [`utils::polytope::HitAndRunPolytopeSampler`], [`utils::polytope::find_interior_point`] (linear program solved with `microlp`), [`utils::polytope::sample_polytope`], [`utils::polytope::sparse_to_dense_constraints`] |
//! | `scipy.optimize.minimize(method="SLSQP")` | `slsqp::minimize_kraft` of the vendored `slsqp` crate (a SciPy-style driver) |
//! | `botorch.test_functions.Hartmann(dim=6, noise_std=..., negate=True)` etc. | [`test_functions::SyntheticTestFunction::hartmann`]`(6)?.with_negate(true)` etc. (20 functions) |
//! | `botorch.test_functions.DTLZ2(dim=6, num_objectives=3, noise_std=[...], negate=True)` etc. (`multi_objective`, `multi_objective_multi_fidelity`) | [`test_functions::MultiObjectiveTestProblem::new`]`(`[`test_functions::MultiObjectiveKind::Dtlz2`]`, Some(6), Some(3))?.with_negate(true)` etc. (30 problems): `evaluate_true`, `forward`, `evaluate_slack`, `is_feasible`, `ref_point`, `max_hv`, `gen_pareto_front` |
//! | `qLogNoisyExpectedHypervolumeImprovement(model, ref_point, X_baseline, sampler, objective, constraints, X_pending, prune_baseline, alpha, cache_pending, max_iep, incremental_nehvi, cache_root, ...)`, `qNoisyExpectedHypervolumeImprovement` | [`acquisition::multi_objective::QLogNoisyExpectedHypervolumeImprovement::new`]`(model, &ref_point, &x_baseline, `[`acquisition::multi_objective::QLogNoisyExpectedHypervolumeImprovementOptions`]`)`, [`acquisition::multi_objective::QNoisyExpectedHypervolumeImprovement`] |
//! | `qLogExpectedHypervolumeImprovement(model, ref_point, partitioning, ...)`, `qExpectedHypervolumeImprovement`, `ExpectedHypervolumeImprovement` | [`acquisition::multi_objective::QLogExpectedHypervolumeImprovement::new`]`(model, &ref_point, &partitioning, `[`acquisition::multi_objective::QLogExpectedHypervolumeImprovementOptions`]`)`, [`acquisition::multi_objective::QExpectedHypervolumeImprovement`], [`acquisition::multi_objective::ExpectedHypervolumeImprovement`] |
//! | `qLogNParEGO(model, X_baseline, scalarization_weights, ...)` | [`acquisition::multi_objective::QLogNParEGO::new`]`(model, &x_baseline, `[`acquisition::multi_objective::QLogNParEGOOptions`]`)` |
//! | `IdentityMCMultiOutputObjective`, `WeightedMCMultiOutputObjective`, `GenericMCMultiOutputObjective`, `FeasibilityWeightedMCMultiOutputObjective` | [`acquisition::multi_objective::IdentityMcMultiOutputObjective`], [`acquisition::multi_objective::WeightedMcMultiOutputObjective`], [`acquisition::multi_objective::GenericMcMultiOutputObjective`], [`acquisition::multi_objective::FeasibilityWeightedMcMultiOutputObjective`] (any [`acquisition::multi_objective::McMultiOutputObjective`]) |
//! | `prune_inferior_points_multi_objective`, `get_default_partitioning_alpha`, `compute_subset_indices` | [`acquisition::multi_objective::prune_inferior_points_multi_objective`], [`acquisition::multi_objective::get_default_partitioning_alpha`], [`acquisition::multi_objective::compute_subset_indices`] |
//! | `optimize_acqf_list(acq_function_list, bounds, num_restarts, raw_samples, options, fixed_features_list, ...)` | [`optim::optimize_acqf_list`]`(&[&acqf, ...], &bounds, num_restarts, raw_samples, &options, fixed_features_list)` |
//! | `draw_matheron_paths(model, sample_shape)`, `draw_kernel_feature_paths`, `gaussian_update`, `get_matheron_path_model` (`botorch.sampling.pathwise`) | [`sampling::pathwise::draw_matheron_paths`]`(&*model, num_samples, &`[`sampling::pathwise::MatheronPathOptions`]`)` (seeded or explicit [`sampling::pathwise::MatheronPathDraws`]), [`sampling::pathwise::draw_kernel_feature_paths`], [`sampling::pathwise::gaussian_update`], [`sampling::pathwise::get_matheron_path_model`] |
//! | `sample_optimal_points`, `random_search_optimizer`, `compute_sample_box_decomposition` | [`acquisition::multi_objective::sample_optimal_points`], [`acquisition::multi_objective::random_search_optimizer`], [`acquisition::multi_objective::compute_sample_box_decomposition`] |
//! | `qLowerBoundMultiObjectiveMaxValueEntropySearch`, `qLowerBoundMultiObjectiveJointEntropySearch(model, pareto_sets, pareto_fronts, hypercell_bounds, X_pending, estimation_type)`, `qMultiObjectivePredictiveEntropySearch(model, pareto_sets, maximize, ...)` | [`acquisition::multi_objective::QLowerBoundMultiObjectiveMaxValueEntropySearch`], [`acquisition::multi_objective::QLowerBoundMultiObjectiveJointEntropySearch`] ([`acquisition::multi_objective::EstimationType`] `"0"`, `"LB"`, `"LB2"`, `"MC"`), [`acquisition::multi_objective::QMultiObjectivePredictiveEntropySearch`] |
//! | `PosteriorMeanModel(model)` (`botorch.models.deterministic`) | [`models::PosteriorMeanModel::new`]`(model)`: a deterministic model ([`Model::is_deterministic`]) usable by every acquisition function |
//! | a `torch.nn.Module` returning `logsumexp(log w_i + acqf_i(X)) - log Σ w_i` | [`acquisition::WeightedLogMeanExp::new`]`(vec![acqf, ...], weights)` (a botorch-rust extension) |
//! | `botorch.utils.safe_math`, `botorch.utils.probability.utils` | [`utils::safe_math`], [`utils::probability`] |
//! | `get_covar_module_with_dim_scaled_prior`, `get_matern_kernel_with_gamma_prior` | [`kernels::get_covar_module_with_dim_scaled_prior`], [`kernels::get_matern_kernel_with_gamma_prior`] |
//! | `get_gaussian_likelihood_with_lognormal_prior`, `..._with_gamma_prior` | [`get_gaussian_likelihood_with_lognormal_prior`], [`get_gaussian_likelihood_with_gamma_prior`] |
//! | `Normalize`, `Standardize` | [`Normalize`], [`Standardize`] |
//! | `LogNormalPrior`, `GammaPrior` | [`Prior`] |
//! | `GreaterThan`, `Positive`, `Interval` | [`GreaterThan`], [`Interval`] (both as a [`Constraint`]) |
//! | `scipy.optimize.minimize(method="L-BFGS-B")` | [`lbfgsb::minimize`]`(fun, &x0, &lower, &upper, &`[`lbfgsb::LbfgsbOptions`]`)` |
//! | `linear_operator.utils.cholesky.psd_safe_cholesky` | [`psd_safe_cholesky`] |
//!
//! ## Example
//!
//! ```
//! use std::sync::Arc;
//!
//! use botorch_core::acquisition::LogExpectedImprovement;
//! use botorch_core::optim::{optimize_acqf, OptimizeAcqfOptions};
//! use botorch_core::{fit_gpytorch_mll, FitOptions, Matrix, Model, SingleTaskGP, SingleTaskGPOptions};
//!
//! let train_x = Matrix::from_fn(12, 2, |i, j| ((i * 7 + j * 3) % 12) as f64 / 11.0);
//! let train_y = Matrix::from_fn(12, 1, |i, _| (6.0 * train_x[(i, 0)]).sin() + train_x[(i, 1)]);
//! let mut model = SingleTaskGP::new(train_x, train_y.clone(), SingleTaskGPOptions::default())?;
//! let fit = fit_gpytorch_mll(&mut model, FitOptions::default())?;
//! assert!(fit.loss.is_finite());
//! let (_mean, variance, dmean, _dvar) = model.posterior_q1_with_grad(&[0.2, 0.4])?;
//! assert!(variance > 0.0 && dmean.len() == 2);
//!
//! // Maximize LogEI over [0, 1]^2 (10 restarts, 512 raw samples, seed 0).
//! let model: Arc<dyn Model> = Arc::new(model);
//! let acqf = LogExpectedImprovement::new(model, train_y.max(), true)?;
//! let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
//! let options = OptimizeAcqfOptions { seed: Some(0), ..Default::default() };
//! let (candidate, value) = optimize_acqf(&acqf, &bounds, 1, 10, 512, &options)?;
//! assert_eq!(candidate.shape(), (1, 2));
//! assert!(value.is_finite());
//! # Ok::<(), botorch_core::BotorchError>(())
//! ```
//!
//! `examples/bo_loop.rs` runs a complete Bayesian optimization loop on
//! Hartmann-6.
//!
//! ## Validation
//!
//! `scripts/generate_fixtures.py` records reference values from BoTorch.
//!
//! *Gaussian processes* (17 models, d ∈ {1, 2, 5}, n ∈ {1, 3, 5, 10, 20}, RBF /
//! Matérn-5/2, `ScaleKernel`, fixed noise, `Normalize` with learned or explicit
//! bounds, with and without `Standardize`, legacy Gamma priors, degenerate
//! data): `tests/parity.rs` requires agreement to a relative 1e-10 for kernels
//! and their derivatives, transforms, MLL values and gradients, posterior means
//! and covariances (with and without observation noise) and their input
//! gradients. The observed errors are at most 6e-14, except 3e-12 for an MLL
//! whose covariance has condition number ~3e6. Fitting from BoTorch's default
//! initialization reaches BoTorch's final loss (worst excess 3.4e-7, typically
//! ~1e-15) with the same number of L-BFGS-B iterations in 16 of 17 problems.
//!
//! *Acquisition functions* (3 models, `tests/acquisition_parity.rs`): values and
//! autograd gradients of all analytic acquisition functions at 8 points
//! (`maximize` true and false, `best_f` up to 10⁹ above the data); MC
//! acquisition functions for `q` = 1, 2, 3 with and without pending points,
//! evaluated with BoTorch's recorded base samples; qLogNEI (incremental or not,
//! full or pruned baseline, and BoTorch's default `cache_root=True` with its
//! effective base samples); `prune_inferior_points` (identical selections).
//! All agree to a relative 1e-10 (observed: at most 6e-11 for EI/PI deep in
//! the tail, where the value's sensitivity to the posterior is about `u²`, and
//! at most 3e-11 for MC gradients; at most 1e-13 for most quantities).
//! `tests/acquisition_gradients.rs` checks every gradient against central
//! finite differences.
//!
//! *Optimization* (`tests/optimize_parity.rs`, a fitted Hartmann-6 model): from
//! BoTorch's own initial conditions, L-BFGS-B reaches BoTorch's final values to
//! ~1e-15 (same iterates and iteration counts); full `optimize_acqf` runs reach
//! BoTorch's best value in 10/10 seeds for LogEI and, for the multimodal q = 2
//! qLogEI problem, about as often as BoTorch does.
//!
//! *Test functions* (`tests/test_functions_parity.rs`, 28 configurations):
//! values, bounds, optima and optimizers agree to 3e-16.
//!
//! *Multi-objective test problems* (`tests/mo_test_functions_parity.rs`, all 30
//! problems in 52 configurations, at random points and at the corners of the
//! domain): objectives, constraint slacks and the deterministic Pareto fronts
//! agree to 7e-16 (Penicillin: relative to BoTorch's own sensitivity to
//! rounding, observed exact), reference points exactly and maximum
//! hypervolumes to the last bit; ten problems implement the corrected
//! definitions of BoTorch pull requests (see
//! [`test_functions::MultiObjectiveKind::deviation`]).
//!
//! *Multi-output, batched and list models* (13 configurations: `m` = 1, 2, 3;
//! batch shapes `3`, `2`, `2 × 2`; shared, per-batch and explicit `Normalize`;
//! fixed noise; legacy Gamma priors; three `ModelListGP`s;
//! `tests/multi_output_parity.rs`): hyperparameter layouts, MLL values and
//! gradients per element, joint posteriors (with and without observation
//! noise, batched and broadcast inputs) and their input derivatives agree to
//! at most 5e-14. Fitting reaches BoTorch's final loss of every output and
//! batch element (largest excess 5.5e-13) with the same number of L-BFGS-B
//! iterations in all 36 elements. Analytic, MC and qLogNEI acquisition
//! functions with a `ScalarizedPosteriorTransform` agree to 4e-13 (values) and
//! 6e-13 (gradients), `prune_inferior_points` selects the same points, and
//! both variants of the constrained expected improvement agree to 4e-12.
//! `tests/multi_output_gradients.rs` checks the new gradients against central
//! finite differences.
//!
//! *MC objectives, outcome constraints and constrained optimization* (five
//! single-output, multi-output and list models, `--phase 4b`;
//! `tests/objective_constraint_parity.rs`, `tests/constrained_optimize_parity.rs`):
//! samples from multi-output base samples agree to 4e-15; MC acquisition
//! values and gradients with linear, generic and constrained objectives and
//! one or two outcome constraints (sigmoid, fatmoid and probability
//! constraints) to 4e-13 and 3e-13; qLogNEI with objectives and constraints,
//! including baselines that are infeasible in some or all samples, to 6e-14
//! (values) and 2e-12 (gradients); `get_infeasible_cost` to 2e-14;
//! `prune_inferior_points` selects the same points. From BoTorch's initial
//! conditions, SLSQP reaches SciPy's optimum in 21 of 22 restarts (value
//! errors at most 4e-7) and the best restart of every problem (shortfall at
//! most 2.1e-8); full constrained `optimize_acqf` runs reach BoTorch's best value
//! (shortfall at most 5e-9 over five seeds). Hit-and-run samples match
//! BoTorch's moments statistically. `tests/objective_constraint_gradients.rs`
//! checks the new gradients against central finite differences.
//!
//! *Pathwise sampling and information-theoretic multi-objective acquisition
//! functions* (five models, `--phase 5c-entropy`; `tests/mo_entropy_parity.rs`,
//! `tests/mo_entropy_properties.rs`): given BoTorch's random numbers, Matheron
//! paths, prior paths and their gradients agree to 3e-15, sampled Pareto sets
//! and box decompositions are identical, MES and JES values and gradients
//! agree to 3e-13 or within ten times BoTorch's own sensitivity to rounding
//! (the entropy estimates are differences of entropies), and PES agrees step
//! by step with BoTorch's EP (factors to 1e-10 relative to their sensitivity
//! to last-bit differences of `exp`); see
//! [`acquisition::multi_objective::QMultiObjectivePredictiveEntropySearch`]
//! for its deviations.
//!
//! ## Scope and known differences
//!
//! * `SingleTaskGP` (single output), [`BatchedMultiOutputGP`] (`m ≥ 1`
//!   independent outputs, any batch shape) and [`ModelListGP`]; RBF and
//!   Matérn-5/2 ARD kernels with optional `ScaleKernel`; LogNormal and Gamma
//!   priors; homoskedastic and fixed-noise Gaussian likelihoods; `Normalize`
//!   (all dimensions, `center = 0.5`) and `Standardize`.
//! * The outputs and batch elements of a [`BatchedMultiOutputGP`] have
//!   independent hyperparameters (BoTorch's default modules with
//!   `batch_shape = _aug_batch_shape`); modules shared by all outputs are not
//!   supported (BoTorch's batched fitting cannot fit them either). A
//!   per-batch `Normalize` of a batched model with several outputs applies
//!   each batch element's bounds to all of its outputs (BoTorch broadcasts the
//!   per-batch bounds against the output dimension instead). Acquisition
//!   functions require unbatched models.
//! * Batched fits run one L-BFGS-B problem per element like BoTorch's
//!   `fmin_l_bfgs_b_batched` (without a `maxfun` limit); a failed element is
//!   retried on its own from prior samples (BoTorch accepts an abnormal
//!   termination of an element and restarts all elements after a numerical
//!   error). A NaN covariance of one element does not affect the others
//!   (BoTorch's batched closure gives all elements a NaN loss with zero
//!   gradients then, a BoTorch bug; see [`fit`]).
//! * [`ModelListGP`] posteriors are the exact block-diagonal concatenation of
//!   the sub-model posteriors. BoTorch (with linear_operator 0.6.1) mixes up
//!   the covariance blocks of multi-output sub-models in some configurations
//!   and fails to scalarize them for `q > 1`; the fixtures evaluate these
//!   code paths with dense covariances.
//! * `SingleTaskGPOptions::use_scale_kernel` wraps the default kernel in a
//!   `ScaleKernel` with BoTorch's legacy `GammaPrior(2.0, 0.15)` outputscale
//!   prior; use [`kernels::get_matern_kernel_with_gamma_prior`] for BoTorch's
//!   complete legacy kernel.
//! * BoTorch's input-scaling *warnings* are not emitted; NaN inputs and
//!   negative noise variances are errors, as in BoTorch. `optimize_acqf`
//!   reports BoTorch's warnings as messages in
//!   [`optim::OptimizeAcqfResult::warnings`].
//! * `fit_gpytorch_mll` retries failed attempts from prior samples like
//!   BoTorch, but with its own seeded random number generator.
//! * Acquisition functions take `best_f`, `beta`, `tau` and constraint
//!   bounds as exact `f64` values. BoTorch stores Python floats passed for
//!   these arguments as float32 tensors (torch's default dtype), i.e. rounded
//!   to about 7 digits; the values here equal BoTorch's when float64 tensors
//!   are passed (as with the common `best_f=train_Y.max()`).
//! * Posterior transforms are limited to `ScalarizedPosteriorTransform`.
//!   qLogNEI implements the plain (`cache_root=False`) computation, which
//!   equals BoTorch's default `cache_root=True` given the same effective base
//!   samples (see [`acquisition::QLogNoisyExpectedImprovement`]); objectives
//!   and constraints are therefore evaluated on the joint samples of the
//!   baseline and the candidates, as with BoTorch's `cache_root=False`. When
//!   some sample has no feasible baseline point, the lower bound of the
//!   objective is estimated at points drawn from a generator seeded by the
//!   sampler (BoTorch: `torch.rand`), or at explicit `lower_bound_points`.
//! * Constraint temperatures `eta` and `ConstrainedMcObjective`'s
//!   `infeasible_cost` are exact `f64` values (BoTorch stores Python floats in
//!   float32), and mismatched numbers of temperatures or `fat` entries are
//!   reported when the acquisition function is created (BoTorch: at the first
//!   evaluation), with BoTorch's messages.
//! * SLSQP follows SciPy's driver (`acc = ftol`, `maxiter = 2000` in
//!   `optimize_acqf`), but its least-squares subproblems are solved by the
//!   NLopt-derived port of Kraft's code, whereas SciPy >= 1.16 uses LAPACK:
//!   iterates agree to about 1e-9 per step, and restarts started on flat parts
//!   of the acquisition function can end in different local optima (in
//!   either direction). Infeasible initial conditions are accepted with
//!   nonlinear constraints, as on BoTorch's main branch (the released 0.18.1
//!   rejects them).
//! * The interior point of a polytope maximizes the same slack as BoTorch's
//!   `find_interior_point`, but the linear program is solved with `microlp`
//!   instead of HiGHS, so a different maximizer may be returned when it is not
//!   unique; hit-and-run chains use their own random streams.
//! * Quasi-random numbers come from an Owen-scrambled Sobol sequence
//!   (`sobol_burley`), not torch's `SobolEngine`, so raw samples and base
//!   samples differ from BoTorch's for the same seed (with the same
//!   distribution); explicit base samples reproduce BoTorch exactly. The
//!   random selection of initial conditions is seeded from `options.seed`
//!   (BoTorch uses torch's global RNG).
//! * The LogEI gradient is accurate to machine precision far in the tail,
//!   where BoTorch's autograd gradient has a relative error of about `ε u²`.
//! * `prune_inferior_points` with `max_frac < 1` keeps the most frequent
//!   maximizers (BoTorch's implementation indexes the wrong points there).
//! * The posterior sample root of a single point is `sqrt(max(v, 0))`
//!   (BoTorch yields NaN for a numerically negative variance); gradients
//!   through the eigendecomposition fallback for covariances that are not
//!   positive definite even with jitter are not available (an error; BoTorch's
//!   are NaN).
//! * `optimize_acqf` supports box bounds, fixed features (scalar values) and
//!   linear and nonlinear parameter constraints; no `post_processing_func`,
//!   timeouts or one-shot acquisition functions. Without constraints, restarts
//!   are independent L-BFGS-B problems (BoTorch's batched fast path), so
//!   `batch_limit` only chunks the work; with constraints, each chunk of
//!   `batch_limit` restarts is one SLSQP problem, as in BoTorch.
//! * Whether a nearly singular covariance block needs jitter in its Cholesky
//!   factorization depends on rounding: in borderline cases (smallest
//!   eigenvalue around 1e-16 of the largest) this crate and LAPACK may decide
//!   differently, which changes the Monte-Carlo samples along the nearly
//!   singular directions.
//! * Python bindings with BoTorch's API are provided by the `botorch_pyo3`
//!   crate (the `botorch_rs` package; see the repository README).

#![warn(missing_docs)]

pub mod acquisition;
pub mod constraints;
pub mod error;
pub mod fit;
pub mod fully_bayesian;
pub mod generation;
pub mod kernels;
pub mod lbfgsb;
pub mod linalg;
pub mod minpack;
pub mod mll;
pub mod models;
pub mod optim;
pub mod posterior;
pub mod priors;
pub mod sampling;
pub mod test_functions;
pub mod transforms;
pub mod utils;

/// Dense double-precision matrix (`nalgebra::DMatrix<f64>`); point sets are
/// stored with one point per row.
pub type Matrix = nalgebra::DMatrix<f64>;

/// Dense double-precision column vector (`nalgebra::DVector<f64>`).
pub type Vector = nalgebra::DVector<f64>;

pub use acquisition::AcquisitionFunction;
pub use constraints::{Constraint, GreaterThan, Interval, LessThan, ParameterTransform};
pub use error::{BotorchError, Result};
pub use fit::{fit_gpytorch_mll, BatchedFitResult, FitOptions, FitResult, FitStatus, Fittable};
pub use kernels::{Kernel, KernelKind, KernelSpec, ScaleKernelSpec};
pub use linalg::{psd_safe_cholesky, CholeskyFactor};
pub use mll::ExactMarginalLogLikelihood;
pub use models::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    BatchedInputTransform, BatchedMultiOutputGP, BatchedMultiOutputGPOptions,
    GaussianLikelihoodSpec, Hyperparameters, LCEMGPOptions, Likelihood, Model, ModelList,
    ModelListGP, ParameterLayout, SingleTaskGP, SingleTaskGPOptions, LCEMGP,
};
pub use optim::{optimize_acqf, OptimizeAcqfOptions};
pub use posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
pub use priors::Prior;
pub use sampling::SobolQMCNormalSampler;
pub use test_functions::SyntheticTestFunction;
pub use test_functions::{MultiObjectiveKind, MultiObjectiveTestProblem};
pub use transforms::{Normalize, OutcomeTransformChoice, Standardize};
