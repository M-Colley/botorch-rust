# Torch acquisition checkpoint

Models selecting `backend="torch"` dispatch supported acquisition constructors
to port-owned `torch.nn.Module` implementations. Inference calls the port's
`model.posterior`; acquisition arithmetic uses Torch tensors on the same device.
Neither acquisition evaluation nor differentiation delegates to BoTorch.
The existing native CPU acquisition route is retained for native CPU models.

Supported families are EI, LogEI, PI, LogPI, UCB, posterior mean and standard
deviation; and qEI, qLogEI, qPI, qUCB, qLCB, simple regret, posterior standard
deviation, qNEI and qLogNEI. Device-resident MC samplers, differentiable scalar
objectives and constraint callables, multiple sample dimensions, t-batches,
pending points, and log smoothing retain their Tensor graphs. Unsupported
families and constructor options raise explicit errors.

Noisy EI currently samples the exact full joint baseline/candidate posterior.
Explicit `cache_root=True` warns and selects `False`. Unbatched baseline pruning
uses 2,048 Sobol samples. Fully Bayesian ensembles use the final MCMC axis;
pruning averages member objectives and rounds member feasibility as in the
reference. Ordinary batched-model pruning remains unsupported. qLogNEI honors
incremental pending points.
By default the incumbent is detached, matching official BoTorch's deliberate
`compute_best_feasible_objective` behavior. The keyword extension
`detach_baseline=False` includes incumbent derivatives in the mathematical
joint objective. Auxiliary infeasible-cost estimation points are generated once
with a seeded device RNG and retained, or supplied explicitly through the port's
existing qLogNEI `lower_bound_points` extension. This keeps repeated evaluations
deterministic; official BoTorch redraws its auxiliary points when needed.

Validation against official revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`: 102 focused tests pass on CPU and an
NVIDIA RTX 4050 Laptop GPU, including actual Rust-dispatched Tensor GPs. Tests
cover reference values and first derivatives, mixed second derivatives through
parameters/training data/queries, inference-plus-acquisition `gradcheck` and
`gradgradcheck`, constraints, pending points, multiple sample dimensions,
batched independent multi-output models, scalarization-weight derivatives,
and forbidden `.cpu()`, `.tolist()`, `.numpy()` and flat-buffer conversions.
Existing native CPU acquisition suites also pass (240 tests, seven existing skips).
This checkpoint does not establish GPU support for other model/acquisition
families or native CPU acquisition optimizers.

## Reference sampled-posterior second derivatives

The pinned reference can produce an incorrect nonlinear second-order query
pullback through a sampled GP posterior. This is a higher-order limitation of
that reference path, not a value or first-gradient discrepancy. For an exact
one-dimensional GP with `X=[.05,.35,.72,.95]`,
`Y=[.12,.62,-.17,.31]`, default kernel/noise/standardization, qLogEI with
`best_f=.25`, 24 CPU float64 normal draws from seed 11, and query `[.21,.56]`, the
port and reference agree in value to `7e-16` and first query gradient to `9e-15`.
With `detach_test_caches(False)` and `fast_pred_var(False)`, differentiation
of the reference gradient in direction `[1,1]` yields approximately
`[-137.4357,7.5905]`; centered differences of that same gradient at step `1e-5`
yield `[-136.6837,7.5307]` (maximum error `0.751963`). The port yields the latter
Hessian-vector product and matches its centered differences to `1.07e-6`.

Reference and port acquisition formulas applied to the same posterior sample
graph have identical second derivatives. Tests therefore separate unmodified
reference-model value/first-gradient parity from acquisition-formula Hessian
parity on the same port posterior, and independently finite-difference check the
complete port inference/acquisition graph. An upstream repair should preserve
the complete differentiable solve/root-decomposition graph in the sampled
posterior backward, or explicitly reject unsupported higher-order derivatives;
no reference code is patched by this implementation.

## Multi-objective checkpoint

qEHVI and qLogEHVI use Tensor inclusion/exclusion over box cells, including
the actual log-fatplus, smooth minimum and signed log-area formulas for
qLogEHVI. Fixed decompositions support any objective dimension; two and three
objectives are tested on CPU and CUDA. Constraints, differentiable multi-output
objectives, pending points, and objective-weight/temperature/training/query
graphs are covered by focused CPU/CUDA tests. Tests forbid inference data transfers
through CPU, NumPy, lists or flat buffers and verify first and second derivatives.

For differentiable observed fronts, the acquisition rebuilds exact cells from
retained caller-owned outcomes. Two objectives use staircase cells; higher
dimensions port the two-pass Lacour local-upper-bound algorithm used by
`FastNondominatedPartitioning`. Tests compare every cell boundary against that
reference for ragged three- and four-objective fronts. Above two objectives,
differentiable observed fronts require `FastNondominatedPartitioning` so that
the log acquisition retains the reference cell decomposition. All boundary
values and graphs remain on device. Integer scalar reads determine the widths
of padded ragged cell arrays. Native CPU box decomposition methods retain their
original numerical behavior. Observed-front and reference-point derivatives pass
`gradcheck` and `gradgradcheck`. Static native partition construction
may still run on CPU; the acquisition's posterior inference and HVI calculations
remain on the model device.

qNEHVI and qLogNEHVI additionally support exact general-dimensional Pareto cells
computed on device from joint posterior samples. They support constraints,
weighted or generic multi-output objectives, model/t-batches, pruning, empty
fronts, and incremental or nonincremental pending points. `cache_root=True`
warns and uses full-joint sampling. The pending-point policy and `max_iep` are
honored while cell geometry is recomputed on each evaluation. Baseline normals
remain shared across pending additions, replacements and removal; no posterior
data is copied to the CPU for partitioning. Initial Sobol sampling follows the
reference's different initialization paths with and without cached pending points.

By default baseline latent samples are detached, matching the reference's
`no_grad` baseline posterior. Objective-weight and reference-point graphs remain
available. `detach_baseline=False` retains the baseline latent graph as well;
the complete training-data/reference/query function passes `gradcheck` and
`gradgradcheck` on CPU and CUDA. The combined acquisition/sampler suite now has
205 passing tests, including 89 hypervolume acquisition tests and six exact
geometry tests. Noisy acquisition values and first/second derivatives are
compared against the pinned reference for two and three objectives on CPU and
CUDA. The geometry algorithm supports higher dimensions, with the usual growth
in the number of Pareto cells; this checkpoint does not establish practical
performance at large objective counts. Approximate `alpha>0` partitioning remains
an explicit boundary in this checkpoint.

## Ensemble acquisition reductions

Analytic acquisitions average the final MCMC axis, using `logmeanexp` for log
acquisitions. Scalar MC acquisitions reduce both sample and MCMC axes after the
pointwise objective and q reduction. Hypervolume acquisitions compute each
member's improvement before averaging. Ordinary query batches remain distinct.
Pruning supports an ensemble with one MCMC batch dimension; the explicit
`marginalize_dim=-3` spelling is accepted for that configuration. Arbitrary
marginalization axes and ensemble pruning with additional model batch dimensions
raise errors.

Fifty-five focused CPU/CUDA tests use three distinct Tensor GPs and the fully Bayesian query
layout, comparing acquisition values, training/input/parameter gradients and
mixed second derivatives against the pinned reference. These tests cover the
scalar families above, noisy EI, constrained analytic acquisitions, weighted
posterior means and all four hypervolume families. Separate tests check member
marginalization during pruning and forbid host conversions during default
sampling and double backward.
The combined acquisition, geometry and sampler suites pass 278 tests.

The pinned reference's `PosteriorStandardDeviation.forward` calls
`std.view(X.shape[:-2])` before its ensemble averaging decorator. For a query
batch `(2,4)`, three MCMC members and `q=1`, it attempts to reshape 24 values into
eight and raises `RuntimeError`. The port preserves the member axis, removes
only the singleton output axis and then averages. Its result and derivatives
are tested against the direct mean of the three posterior standard deviations;
the unmodified reference failure is asserted separately. A minimal upstream
repair is to return `std.squeeze(-1)` instead of reshaping to the input batch,
allowing the existing averaging decorator to retain the MCMC axis and its graph.

## Fully Bayesian model lists

Torch `ModelListGP` passes ordinary query batches to each fully Bayesian child
and combines the component covariances into output-major independent blocks.
The resulting Gaussian law remains a `GaussianMixturePosterior`, including
after output selection and scalarization; mixture covariance includes the
between-component mean covariance. Mixed ordinary and fully Bayesian children
share the MCMC axis. Ordinary model batches are retained before that axis, so an
ordinary batch of size two combined with three MCMC members has model batch
shape `(2,3)`. A one-member mixture broadcasts to the other mixture's size;
incompatible MCMC counts fail ordinary broadcast validation.
An ordinary Gaussian child is evaluated once and its moments are expanded across
MCMC members, avoiding redundant GP inference for those identical components.

For nonlinear ordinary children, a leading component query batch and a Tensor
axis view preserve captured outcome-transform coefficients in their original
batch layout. This produces a `TorchPosteriorList` with differentiable moments
and samples without rebuilding transforms or copying inference values to CPU.
Twenty-four CPU/CUDA tests cover mixture/mixed lists, explicit and inferred
observation noise, samples, ordinary model batches, nonlinear transforms,
training/query/parameter derivatives, selected outputs, scalarization and noisy
hypervolume acquisition graphs.
The combined focused model/acquisition/sampler/geometry run passes 451 tests;
the final ordinary-Gaussian inference optimization also passes all 24 list tests.

The pinned reference additionally omits a noise-axis adjustment in fully Bayesian
prediction: it inserts `X.unsqueeze(-3)` but passes explicit observation noise
of shape `query_batch x q x 1` unchanged. With query batch `(2,4)` and three MCMC
members this raises a broadcast error. Tests assert that unmodified failure and
compare the port against independent reference component laws with the explicit
noise expanded along the MCMC axis. The corresponding upstream repair is to
insert the same singleton MCMC axis in tensor-valued observation noise before
calling the parent posterior implementation.
