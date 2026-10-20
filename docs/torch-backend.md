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
whitened `SingleTaskVariationalGP`, and loaded-sample fully Bayesian Matérn,
SAAS and linear single-task models. See the detailed module, variational,
multi-task and fully Bayesian notes for their configuration limits. Exact
single-task conditioning and fantasies preserve training-data and parameter
graphs; multi-task conditioning/fantasies remain a separate missing feature.
Loaded MCMC values become registered leaves, matching the reference loader;
this does not differentiate through an upstream MCMC sampler.

The acquisition backend covers EI, LogEI, PI, LogPI, UCB, posterior mean and
standard deviation; constrained EI/LogEI and log feasibility; weighted q means;
the standard scalar MC improvement, regret, confidence-bound and standard
deviation families; noisy EI and noisy LogEI in their MC forms; EHVI, LogEHVI,
NEHVI and LogNEHVI; and one-shot KG and multi-fidelity KG. Hypervolume geometry
uses device-resident exact box decomposition, including more than two outputs.
Ensemble acquisitions reduce their MCMC components using means or log means
as appropriate. Feature-fixing and projection wrappers preserve live Tensor
graphs through their supplied values and callables.

KG's `evaluate` method optimizes the inner fantasy decisions with the selected
candidate generator. The returned value retains outer training-data, candidate,
and parameter derivatives while holding the optimizing decisions fixed, as in
the envelope-gradient interface. It does not differentiate the optimizer's
iterations. This is separate from the complete graph of augmented one-shot KG.
Initial inner decisions use Sobol points and per-fantasy top-value selection;
this is not BoTorch's current-model-guided initialization heuristic.

`gen_candidates_torch` uses actual Torch optimizers on device-resident candidate
leaves. `gen_candidates_scipy` uses L-BFGS-B or constrained SLSQP; only candidate
vectors, acquisition values and derivatives cross the SciPy boundary. Model
training data and posterior algebra stay on their device. `optimize_acqf`
supports joint optimization, best/all restarts, fixed features, postprocessing,
one-shot candidate extraction and sequential pending-point restoration.
Constrained automatic initialization currently requires feasible supplied
initial conditions or a custom generator. The automatic box initializer uses
device-resident Boltzmann, nonnegative or top-n restart selection and supports
`init_batch_limit` for raw scoring. Candidate `batch_limit`, forced parallel
SciPy mode and aggregation limits raise explicit errors. Batched model restart selection, acquisition sequences,
specialized mixed/discrete/cyclic optimization and remaining initial-condition
heuristics still need the Torch path. Unsupported configurations raise errors.

Specialized model and entropy/search acquisition families remain outside this
backend. Tensor-backend NUTS is explicitly unsupported; native CPU NUTS remains
available. Noisy acquisition root caching currently falls back to exact joint
sampling with an explicit warning. The 776-name API availability audit does not
measure GPU or behavioral parity. The old CPU release certificate remains a
separate artifact and must not be read as a GPU certificate.

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
[fully Bayesian models](torch-fully-bayesian.md),
[acquisitions](torch-backend-acquisitions.md).
