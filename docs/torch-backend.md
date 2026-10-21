# Torch device and autograd backend

This checkout has two numerical backends. The native Rust backend computes on
the CPU in float64 and accepts lists without importing Torch. The optional
Torch backend computes on the supplied Tensor device, in float32 or float64,
using the Rust extension's graph-preserving Torch operations. It does not call
upstream BoTorch models or acquisition functions. Both first derivatives and
higher derivatives retain the ordinary PyTorch graph for supported operations.
The audited main-package API inventory and tested numerical configurations
are reported separately. Family-specific configuration limits still apply.

Development after the frozen release adds [generic fitting protocols](fitting-protocol.md),
[candidate-generation protocols](candidate-generation-protocols.md),
[analytic batch probability](torch-analytic-batch-probability.md),
[device pruning and constraint utilities](torch-baseline-pruning.md),
[cached sampling and model-fit metrics](torch-low-rank-evaluation.md),
[probit variational observations](torch-variational-probit.md),
[custom acquisition subclasses](torch-acquisition-subclasses.md),
[supplied exact Gaussian likelihoods](torch-exact-gaussian-likelihood.md), and
[binary hypervolume partitioning](torch-hypervolume-partitioning.md).
The newer audit also repairs
[direct input transforms](torch-input-transforms.md),
[direct model/outcome protocols](torch-backend-modules.md), and
[native prediction cache reuse](native-prediction-cache.md).
Caller-defined model modules are covered by the
[custom Torch model protocol](torch-custom-models.md).
These focused results do not replace the frozen release certificate below.
The [equivalence verification notes](equivalence-verification.md) distinguish
strict numerical comparisons, stabilization checks, statistical fitting and
test-specific independent derivative oracles.
The [October 21 development verification](verification-main-development-2026-10-21.json)
records all 11,840 collected CPU/CUDA cases at checkpoint `c156b96`: 11,787
passed, 51 skipped and two learned-objective pruning failures. Both failures
were repaired in `d65f947` and pass focused checks. A separate later checkpoint,
`d50bca8`, passes all 11 CI jobs, including 1,222 pinned-reference equivalence
cases. These checkpoints do not certify subsequent changes or a new release.
The [stricter equivalence audit](verification-equivalence-audit-2026-10-21.json)
at `30c1c95` executed all 1,237 selected cases exactly once: 1,201 passed,
15 skipped and 21 failed. Twenty ensemble cases pass after the device repair;
the remaining inverse-power domain case now classifies undefined values
separately from finite numerical comparisons. The
[fresh frozen equivalence run](verification-equivalence-f943100-2026-10-21.json)
at `f943100` passes 1,222 cases with 15 recorded skips and no failures. Exact
selected-case accounting and all source/native/reference fingerprints remain
stable. The failed baseline remains preserved as failed, and subsequent changes
remain outside the new checkpoint.
The later [cross-platform CI checkpoint](verification-main-ci-4560b50-2026-10-21.json)
at `4560b50` passes all 11 jobs. Its pinned-reference broad phase passes 9,145
cases with 818 skips; its separate equivalence phase passes 1,222 with 15 skips.
The earlier 21 deliberate nonfinite-state/domain comparison failures are repaired
with explicit validity checks and a negative control rejecting shared computed
NaNs. This CI checkpoint precedes the latest direct-transform, structured-call
and native-cache changes and does not certify a new wheel.
The API inventory measures availability rather than configuration equivalence.
Later [variational factory](variational-factory-configurations.md),
[supplied likelihood](torch-variational-likelihood-protocol.md),
[custom model](torch-custom-models.md), [fitting](fitting-protocol.md), and
[acquisition cache](torch-noisy-root-cache.md) checks address configurations
listed as gaps in earlier checkpoints. Family-specific notes distinguish
implemented behavior, native-only limits and source-unsupported calls. A new
release and Unity integration require fresh integrated verification of that
reviewed scope.

The later [frozen development equivalence checkpoint](verification-equivalence-c3e8317-2026-10-21.json)
at `c3e8317` passes 1,222 cases with 15 skips, no failures, and all 1,237
selected IDs accounted for. Python/test inventories, the production v8 native
artifact, worker and pinned reference remain unchanged throughout the run.
This covers the grouped variational, structured likelihood/iterative, and path
module checkpoints at that commit. Subsequent ownership/configuration repairs
remain outside its scope. The [complete frozen suite](verification-full-c3e8317-2026-10-21.json)
now passes 13,683 cases with 51 skips and no failures. All 13,734 collected
cases execute exactly once, with stable source/test/native/reference and
worker fingerprints. The equivalence cases are included in this total.
The [same checkpoint's CI](verification-main-ci-c3e8317-2026-10-21.json)
passes all eleven jobs. These receipts precede the newer transform, sampler,
conditioning and native iterative repairs and do not certify a new release.

Construct supported models with `backend="torch"` to request these semantics
explicitly. `backend="auto"` selects it for CUDA training data, training tensors
requiring gradients, or supported custom Torch module components. Ordinary CPU
data retain the native backend; batched CPU Tensor inputs for `LatentKroneckerGP`
select its Tensor implementation because its native route is unbatched. A later
query requiring gradients does not
switch an already constructed model; select the backend at construction.

```python
import torch
from botorch_rs.models import SingleTaskGP
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.optim.fit import fit_gpytorch_mll_torch
from botorch_rs.acquisition import LogExpectedImprovement
from botorch_rs.generation.gen import gen_candidates_torch
from botorch_rs.optim import optimize_acqf

X = torch.rand(24, 3, device="cuda", dtype=torch.float64)
Y = (X.sum(dim=-1, keepdim=True) * 3).sin()
model = SingleTaskGP(X, Y, backend="torch")
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch,
                 optimizer_kwargs={"step_limit": 50})
acq = LogExpectedImprovement(model, best_f=Y.max())
bounds = torch.stack((torch.zeros_like(X[0]), torch.ones_like(X[0])))
candidate, value = optimize_acqf(
    acq, bounds, q=1, num_restarts=8, raw_samples=128,
    gen_candidates=gen_candidates_torch,
    options={"stopping_criterion_options": {"maxiter": 60}},
)
```

Supported model paths include exact single-task independent outputs and
batches, Gaussian and nonlinear-outcome model lists, long-format `MultiTaskGP`,
`MixedSingleTaskGP`, `SingleTaskMultiFidelityGP`, `KroneckerMultiTaskGP`,
`HigherOrderGP`, `LatentKroneckerGP`, `PairwiseGP` with probit/logit likelihoods,
whitened, unwhitened, batch-decoupled, CIQ and grouped large-batch
`SingleTaskVariationalGP`, compatible caller strategy factories, and loaded-sample fully Bayesian Matérn,
SAAS and linear single-task and latent-task multitask models. Exact single-task and correlated multitask
conditioning and fantasies retain training-data and parameter graphs. See the
detailed model notes for configuration limits. The Kronecker Tensor route
evaluates the exact Gaussian law with dense factorization; a structured or
matrix-free speedup is not implemented. This is a performance distinction,
separate from its supported posterior and likelihood configurations.
`LatentKroneckerGP` has its own
[iterative solver implementation](latent-iterative-methods.md). Loaded MCMC values become registered leaves, matching the
reference loader. Markov-chain choices are not differentiated.

Public outcome transforms participate in actual Torch module graphs, with
registered state, hooks, serialization, and device migration; see
[public outcome module protocols](torch-outcome-modules.md).

The acquisition backend covers EI, LogEI, PI, LogPI, UCB, posterior mean and
standard deviation; constrained EI/LogEI and log feasibility; weighted q means;
the standard scalar MC improvement, regret, confidence-bound and standard
deviation families; noisy EI and noisy LogEI in their MC forms; EHVI, LogEHVI,
NEHVI and LogNEHVI; one-shot KG, hypervolume KG and their multi-fidelity forms; MES, GIBBON and their
multi-fidelity forms; scalar JES with LB and MC estimation; and scalar and
multi-objective PES with device EP; and multi-objective lower-bound MES/JES
with all four entropy estimators; and multistep lookahead fantasy trees. Hypervolume geometry
uses device-resident exact box decomposition, including more than two outputs.
Ensemble acquisitions reduce their MCMC components using means or log means
as appropriate. Feature-fixing and projection wrappers preserve live Tensor
graphs through their supplied values and callables.

KG's `evaluate` method optimizes the inner fantasy decisions with the selected
candidate generator. The returned value retains outer training-data, candidate,
and parameter derivatives while holding the optimizing decisions fixed, as in
the envelope-gradient interface. It does not differentiate the optimizer's
iterations. This is separate from the complete graph of augmented one-shot KG.
KG and hypervolume KG initializers optimize the current model's value function
and use device Boltzmann resampling to seed the inner fantasy decisions.

`gen_candidates_torch` uses actual Torch optimizers on device-resident candidate
leaves. `gen_candidates_scipy` uses L-BFGS-B or constrained SLSQP; only candidate
vectors, acquisition values and derivatives cross the SciPy boundary. Model
training data and posterior algebra stay on their device. `optimize_acqf`
supports joint optimization, best/all restarts, fixed features, postprocessing,
one-shot candidate extraction and sequential pending-point restoration.
Linear-constrained automatic initialization samples the feasible polytope on
device. The automatic box initializer uses
device-resident Boltzmann, nonnegative or top-n restart selection, including
model batches and `init_batch_limit` raw scoring. Candidate `batch_limit`,
low-level aggregation chunks and acquisition sequences now retain their device
execution. Discrete enumeration, lattice local search, mixed combination search,
acquisition lists and cyclic search now have device implementations. Linear
constrained automatic initialization, around-best sampling and KG/hypervolume
initialization heuristics retain their device calculations. Mixed alternating
search now alternates device categorical/ordinal steps with actual bounded
continuous optimization. Homotopy stages, feasibility checks, feature fixing,
candidate selection and equality projection retain their Tensor graphs. See
[optimization verification](torch-optimization.md).

MAP SAAS, additive MAP SAAS and MAP SAAS ensembles now retain their coupled
shrinkage priors and fitting graphs on device. See [MAP SAAS verification](torch-map-saas.md).

Hierarchical scalar and multitask models now use device covariance gates,
including batched transforms, independent scalar-model outputs, conditioning,
fantasies and higher-order graphs. EUBO and pairwise BALD acquisitions retain
their preference-model graphs. Device Fourier priors and exact/whitened
variational Matheron paths support pathwise acquisitions; see
[pathwise verification](pathwise-public-constructors.md).

Heterogeneous multitask models now retain their training-list, imputation,
query, noise and parameter graphs on device. Learned preference objectives
support ordinary and batched preference models with full-joint noisy improvement
and repeated backward through baseline pruning. Sparse and robust GPs use real
device outlier noise and relevance-pursuit fits.

SAC and LCEA contextual models retain shared spatial covariance and live context
embeddings. Independent and correlated empirical curve models retain historical
interpolation and basis graphs. The neural kernel now has a genuine device
backend with consistent coincident self-covariances and higher derivatives.

LCEM and orthogonal additive models now retain their full covariance/component
graphs. Sparse conditioning and fantasies retain the original outlier leaves
and assign zero outlier noise to appended observation labels; search timeouts
follow the source's per-fit allocation.

Empirical multitask models now retain the full historical cross-task covariance,
including conditioning, fantasies and data/parameter higher derivatives. The
integrated Wiener kernel uses consistent polynomial branches at coincident
inputs so its smooth Hessians remain correct.

Iterative empirical EM now has device cold/warm/pretrained inference, full
update graphs, prior fitting, coordinate ascent and shared-model fitting.
Learned preferences retain independent preference and outcome ensemble axes
through nonlinear acquisition reductions, constraints and baseline pruning.
The [standalone kernel catalogue](torch-kernel-catalogue.md) now includes
core, categorical, fidelity and task kernels with device prior graphs.
Direct empirical utility entry points also retain device calculations and
derivative graphs. The installed-wheel verification below includes these paths.
Device-resident NUTS fits scalar and
latent-task multitask Matérn, SAAS and linear fully Bayesian models. Explicit
custom probabilistic hooks use the port's optional NumPyro adapter on the
installed JAX backend, matching the source's inference-device semantics;
they are not translated into Torch NUTS. Scalar and latent-task Tensor models adopt supported
custom Gaussian module loaders, including full correlated likelihood noise,
with device prediction and higher-order graphs. Task-aware loaders preserve
cross-task covariance, mapped labels and physical feature selection. See
[sampler semantics and verification](torch-fully-bayesian.md).
Noisy Tensor acquisitions use baseline Cholesky roots and Schur conditional
sampling for independent Gaussian posteriors. Frozen roots are reused; live
data and parameter graphs rebuild the baseline factor on each call. See
[cache semantics and checks](torch-noisy-root-cache.md). The 776-name API availability audit does not measure
GPU or behavioral parity. New checkpoints receive focused CPU/CUDA and autograd
tests; the current release certificate below records their integrated run.

## Measured prediction costs

The historical October 20 release timings below use one Torch thread, matching hyperparameters,
16 batches of two queries, four input dimensions and
31 shuffled interleaved repetitions. Construction and fitting are excluded.
Reference and Tensor training factors were warmed, but the native snapshot-cache
defect meant native public calls continued to refactor training covariance.
Values below are BoTorch time divided by port time on the
local RTX 4050 Laptop GPU and CPU; values above one favor the port.

| Training observations | Torch CPU float64 | Torch CUDA float64 | Torch CPU float32 | Torch CUDA float32 | Native CPU float64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.57x | 1.61x | 1.65x | 1.49x | 5.27x |
| 256 | 1.20x | 1.46x | 1.23x | 1.65x | 1.33x |
| 1024 | 1.34x | 1.07x | 1.01x | 1.46x | 0.09x |

These timings describe these warmed prediction workloads. They do not establish
an end-to-end or universal speedup. Packed multi-right-hand-side triangular
solves avoid redundant factor copies for shared-factor query batches. Full
samples, numerical checks, exact Python/native fingerprints and reference
revisions are in [float64](benchmarks/tensor-main-prediction-2026-10-20-float64.json)
and [float32](benchmarks/tensor-main-prediction-2026-10-20-float32.json).

That native CPU build took 11.2 times as long as BoTorch in the 1,024-point
public prediction case. A subsequent production cache repair reduces a controlled
repeat of this workload from 50.50 ms to 4.30 ms, an 11.73x improvement over the
preserved build. BoTorch takes 3.67 ms in the candidate run, leaving native
17.3% slower for that case; native is faster at 32 and 256 observations. First
native prediction time is unchanged. The fresh
[cache measurements and validation receipt](native-prediction-cache.md) retain
all raw samples, artifact/source fingerprints and limits. Those candidate
results are separate from the historical table and the release certificate.

The separate native float64 Hartmann-6 benchmark takes 0.207 s versus 13.09 s
for BoTorch (63.2x), measured over three seeded 30-iteration loops after warmup
with one thread. Each iteration includes default GP fitting and LogEI
optimization with ten restarts and 512 raw samples. Optimizer/RNG trajectories
can differ; final objective values and all timings are retained in the
[native BO report](benchmarks/native-main-bo-2026-10-20.json).

Further details: [tensor algebra](tensor-native-algebra.md),
[modules and transforms](torch-backend-modules.md),
[multi-task model](torch-backend-multitask.md),
[mixed and multi-fidelity models](torch-specialized-exact.md),
[Kronecker model](torch-kronecker.md),
[higher-order model](torch-higher-order.md),
[latent Kronecker model](latent-kronecker.md),
[MAP SAAS](torch-map-saas.md),
[fully Bayesian models](torch-fully-bayesian.md),
[fully Bayesian multitask models](torch-fully-bayesian-multitask.md),
[preference model](torch-pairwise-gp.md),
[entropy search](torch-entropy-search.md),
[acquisitions](torch-backend-acquisitions.md),
[public samplers](torch-samplers.md).

## Release verification

The frozen Windows release wheel passes **10,542 Python cases**,
with 49 recorded skips, and **532 Rust tests and
doctests** in 46 groups. The complementary broad and equivalence
phases account for all 10,591 collected case IDs exactly
once against official reference `b0dd01be61d5`.
Execution phases and Python/native/helper/reference fingerprints remain stable
across both runs, with clean tracked source/reference trees. The installed
package and wheel match all 302 Python
source/type-marker files; the wheel contains no bytecode caches or duplicate
archive members.

The combined suite includes specialized/empirical/variational models,
scalar/multitask device NUTS, entropy/search acquisitions, preference ensembles,
pathwise sampling and candidate optimization. CPU/CUDA checks compare independent
Gaussian formulas and reference behavior, complete data/query/parameter
gradients, Hessian products and gradcheck/gradgradcheck. Sampler replay tests
compare recorded NumPyro trajectories. Strict Clippy, rustfmt, Ruff and mypy pass.

The [current machine-readable verification report](verification-main-device-release-2026-10-20.json)
records the wheel, exact hashes, complementary test ledgers, commands and
performance evidence. Main-package public-symbol availability is **776/776**.
This inventory and regression coverage do not certify every possible subclass,
shape, strategy or backend combination. The report's `gpu_coverage_complete`
and `behavioral_parity_complete` flags therefore remain false; configuration
limits are explicit in the linked family notes. Structured Tensor models use
dense exact algebra, and that historical wheel's noisy root-cache requests use
the documented exact joint-sampling performance fallback. Subsequent development
adds actual baseline roots and Schur sampling as described above.

All 11 [cross-platform CI jobs](https://github.com/M-Colley/botorch-rust/actions/runs/39508767912)
pass at test revision `e5a6c0ad3da8`, including Rust on Linux, macOS
and Windows, optional-dependency matrices, and the full Linux Torch 2.14 CPU
BoTorch broad/equivalence suites. That CI revision changed tests and documentation
relative to the frozen wheel; its report verifies unchanged Python payloads and
all 300 Rust/Cargo source inputs. Subsequent development changes add runtime
behavior and are outside that release certificate. Their focused verification is
recorded in the configuration-specific notes; a new full integrated certificate
is pending. Linux CI builds its own platform extension.
Its case counts are separate from the Windows installed-wheel ledger. Local
CUDA validation uses Torch 2.13; Torch 2.14 GPU coverage is not certified.

The [earlier release certificate](verification-torch-release-2026-10-20.json),
its wheel and older measurements remain preserved as historical checkpoints.
Their counts are not added to this release's totals.

The implementation is reviewable in
[port PR #1](https://github.com/M-Colley/botorch-rust/pull/1).

Public input transforms now also expose their actual parameter, prior and
constraint traversal on the Torch module, including Warp prior closures and
initialization, learned-imputation bounds, and custom call hooks. Actual
GPyTorch priors retain caller identity and device/autograd behavior; native
descriptors remain eligible for the native backend until a Tensor operation
requires their live module state. Focused CPU/CUDA integration passes 517 cases,
with 137 subsequent protocol cases covering the final call-hook changes.
Native fully Bayesian loading preserves float64 sampled Warp concentrations
when Torch modules are available, and prior evaluation retains the registered
Warp normalizer. Untouched array feature/perturbation sets retain their native
precision through copies; explicit casts, buffer edits and checkpoint loading
use the live registered state. The 185 native/prior cases and 247 module,
persistence and conditioning cases pass; an additional 110-case run covers
the final buffer-mutation and supplied-model checkpoint controls. Torch 2.14.1
CPU passes 227 cases with 42 device skips. Source-only phase controls skip
cleanly when BoTorch is absent (24 skips).
Conditioned models retain transformed nonleaf training coordinates, as the
pinned source does. Reusing that same conditioning graph for independent
backward/Hessian sweeps requires `retain_graph=True` until the final sweep;
construct a fresh conditioned model for independent graph lifetimes. Source
controls cover ordinary Normalize and a parameterized custom input transform.
Native preference models also retain explicit float64 Warp initializers while
preserving the standalone public Parameter's source dtype and identity. A
Tensor call or explicit migration makes the live Tensor state authoritative;
native fitted normalization survives that transition. The three preference
Warp suites pass all 115 cases, with direct source controls for both precision
and array-to-Tensor normalization. Supplied approximate models preserve the
device of attached transforms, as ordinary source module assignment does;
call `model.to(query)` after attaching CPU-fitted transforms to a CUDA model.
[Upstream submissions](upstream-pull-requests.md) track verified defects and
proposed repairs, including reused PRs and the SciPy human submission step.
