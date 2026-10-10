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

## Installation

From a source checkout (requires a Rust toolchain):

```console
$ pip install ./botorch_pyo3
```

## License

MIT License
