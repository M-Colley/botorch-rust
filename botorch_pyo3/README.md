# botorch-rust (Python package `botorch_rs`)

A fast Rust implementation of [BoTorch](https://botorch.org) with essentially
the same Python API. Typical BoTorch code runs after changing the imports:

```python
import torch
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms import Normalize, Standardize
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.acquisition import LogExpectedImprovement
from botorch_rs.optim import optimize_acqf

train_X = torch.rand(20, 6, dtype=torch.float64)
train_Y = (train_X - 0.5).pow(2).sum(-1, keepdim=True).neg()
model = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=6),
                     outcome_transform=Standardize(m=1))
fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
acqf = LogExpectedImprovement(model, best_f=train_Y.max())
bounds = torch.stack([torch.zeros(6), torch.ones(6)]).double()
candidate, value = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
```

The package has no runtime dependencies: arrays may be torch tensors, NumPy
arrays or nested lists of floats, and results have the type of the
corresponding input. See the repository's README for the supported API,
accuracy, benchmarks and known differences from BoTorch.

Native computation uses CPU float64. Torch/NumPy results preserve the
input dtype and device, with transfers when needed. The audited official
BoTorch core API inventory contains 776 public classes/functions, all
available under matching paths; this measures API availability, not every
configuration or Torch behavior. Supported models also accept
`backend="torch"` for computation on the actual Tensor device, float32/float64,
and full PyTorch autograd graphs through data and model Parameters. CUDA
training data select that backend automatically. Device implementations include
specialized and empirical models, scalar/multiobjective entropy search,
lookahead, mixed/discrete search and scalar/multitask NUTS. Configuration limits,
independent CPU/CUDA checks and measured performance are documented in the
repository's
[Torch backend scope](https://github.com/M-Colley/botorch-rust/blob/main/docs/torch-backend.md).
See the repository's [coverage and configuration limits](https://github.com/M-Colley/botorch-rust#readme)
and [correctness findings](https://github.com/M-Colley/botorch-rust/blob/main/docs/correctness-findings.md).
Performance depends on the workload and should be checked on representative
data with release builds.

## Installation

From a source checkout (requires a Rust toolchain):

```console
$ pip install ./botorch_pyo3
```

## License

MIT License
