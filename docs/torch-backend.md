# Torch device and autograd backend

This checkout has two numerical backends. The native Rust backend computes on
the CPU in float64 and accepts lists without importing Torch. The optional
Torch backend computes on the supplied Tensor device, in float32 or float64,
using the Rust extension's graph-preserving Torch operations. It does not call
upstream BoTorch models or acquisition functions. Both first derivatives and
higher derivatives retain the ordinary PyTorch graph for supported operations.
This is an implementation checkpoint, not a claim of complete GPU parity for
the main BoTorch package.

Construct supported models with `backend="torch"` to request these semantics
explicitly. `backend="auto"` selects it for CUDA training data, training tensors
requiring gradients, or supported custom Torch module components. Ordinary CPU
data retain the native backend. A later query requiring gradients does not
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
whitened `SingleTaskVariationalGP`, and loaded-sample fully Bayesian Matérn,
SAAS and linear single-task and latent-task multitask models. Exact single-task and correlated multitask
conditioning and fantasies retain training-data and parameter graphs. See the
detailed model notes for configuration limits. The Kronecker implementation
currently uses dense exact factorization; its structured performance work is
still pending. Loaded MCMC values become registered leaves, matching the
reference loader. Markov-chain choices are not differentiated.

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
their preference-model graphs. Device Fourier priors and exact Matheron paths
support pathwise acquisitions; see [pathwise verification](pathwise-public-constructors.md).

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
derivative graphs. Final installed-wheel verification is being completed.
Device-resident NUTS fits scalar and
latent-task multitask Matérn, SAAS and linear fully Bayesian models. Custom
probabilistic overrides remain unsupported. See
[sampler semantics and verification](torch-fully-bayesian.md).
Noisy acquisition root caching currently falls back to exact joint sampling
with an explicit warning. The 776-name API availability audit does not measure
GPU or behavioral parity. New checkpoints receive focused CPU/CUDA and autograd
tests; the older release certificate does not certify these later changes.

## Measured prediction costs

The synchronized release benchmarks use one Torch thread, matching
hyperparameters, warmed cached training factors, 16 batches of two queried
points, four input dimensions and 31 interleaved timing repetitions. They exclude
construction, fitting and first-factor costs. The RTX 4050 results below are
BoTorch time divided by port time; values above one favor the port.

| Training observations | CPU float64 | CUDA float64 | CPU float32 | CUDA float32 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1.47x | 1.41x | 1.59x | 1.48x |
| 256 | 1.26x | 1.70x | 1.25x | 1.60x |
| 1024 | 1.53x | 0.97x | 1.02x | 1.39x |

These are prediction workloads on this machine, not end-to-end BO or universal
speed guarantees. The native CPU path is faster for the 32-observation float64
case. The initial Tensor implementation was much slower at 1024 observations;
packing query batches into a shared-factor multi-right-hand-side triangular
solve removed redundant large factor copies. Full raw timings, numerical error
checks and loaded native binary hashes are in
[float64](benchmarks/tensor-packed-prediction-2026-10-20-float64.json) and
[float32](benchmarks/tensor-packed-prediction-2026-10-20-float32.json).

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
[acquisitions](torch-backend-acquisitions.md).

## Release verification

The 2026-10-20 Windows wheel passed 8,683 Python cases, with 47 documented
skips, against the pinned official reference. The two complementary test phases
account for all 8,730 collected case IDs exactly once. Their execution phases,
installed Python sources, native binary, helper and reference fingerprints
remain consistent before and after both runs. The wheel contains 265 matching
Python source files, no bytecode caches and no duplicate archive members.
The Rust workspace passed 522 tests in 46 groups; strict Clippy, rustfmt,
Ruff and mypy also passed. These results describe the frozen wheel recorded in
that report. Subsequent specialized-model, entropy-search and NUTS changes
have separate focused verification; a fresh combined release is pending.

At source checkpoint `ac07139`, a combined run of all `test_torch*.py` files
passed 2,337 cases with two skips, using the rebuilt release extension on CPU
and the local CUDA GPU. The complete Rust workspace passed 528 tests and doc
tests in 46 groups; its 300 Rust source files were unchanged by the later
Python-only model additions through that checkpoint. The subsequent EM prior
centering, duplicate equality coefficient and rank-zero task-kernel repairs
are now included in a fresh native build whose full workspace passes 532
tests and doctests in 46 groups, with strict Clippy and rustfmt passing. These
checks are separate from an installed-wheel release certificate.

Earlier integrated checks passed 327 Rust core unit tests, 355 kernel/scalar
acquisition cases (11 skips) with the rebuilt release binary, 99 multitask
fully Bayesian CPU/CUDA cases, 48 latent Kronecker cases, 34 hypervolume KG
cases, and 78 PairwiseGP/multistep cases. These focused suites overlap other
checks and are not added to the older wheel's release totals. The source
extension now uses the repaired release binary; the frozen wheel and its
recorded binary remain preserved separately.

The [machine-readable verification report](verification-torch-release-2026-10-20.json)
records the exact local wheel, hashes, commands, test ledger, benchmarks and
remaining limits. Its `gpu_coverage_complete` and `behavioral_parity_complete`
flags remain false. Main-package public-symbol availability is 776/776;
that inventory is separate from behavioral and GPU coverage.

The continuing implementation is reviewable in
[port PR #1](https://github.com/M-Colley/botorch-rust/pull/1).
[Upstream submissions](upstream-pull-requests.md) track verified defects and
their proposed repairs separately from coverage work.
