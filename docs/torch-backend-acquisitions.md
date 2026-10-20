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
uses 2,048 Sobol samples. Ensemble marginalization and batched-model pruning are
not implemented in this backend. qLogNEI honors incremental pending points.
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
graphs are covered by 32 focused tests. Tests forbid inference data transfers
through CPU, NumPy, lists or flat buffers and verify first and second derivatives.

For differentiable two-objective observed fronts, the acquisition rebuilds exact
staircase cells from retained caller-owned outcomes with device-side comparisons,
sorting and gathering. Native CPU box decomposition methods retain their
original numerical behavior. Observed-front and reference-point derivatives pass
`gradcheck` and `gradgradcheck`. Differentiable observed-front rebuilding above
two objectives and noisy hypervolume acquisitions remain outside this checkpoint;
they raise explicit errors on this backend. Static native partition construction
may still run on CPU; the acquisition's posterior inference and HVI calculations
remain on the model device.
