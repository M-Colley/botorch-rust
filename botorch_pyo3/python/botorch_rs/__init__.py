"""botorch_rs: a fast Rust implementation of BoTorch with the same Python API.

Typical BoTorch code runs unchanged after replacing ``botorch`` (and the
``gpytorch`` modules it uses) by ``botorch_rs`` in the imports::

    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.transforms import Normalize, Standardize
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.acquisition import LogExpectedImprovement
    from botorch_rs.optim import optimize_acqf

Arrays may be torch tensors, NumPy arrays or nested lists; results are returned
with the type of the corresponding input. Neither torch nor NumPy is required.
"""

from botorch_rs import (
    acquisition,
    constraints,
    exceptions,
    fit,
    kernels,
    likelihoods,
    means,
    mlls,
    models,
    optim,
    posteriors,
    priors,
    sampling,
    test_functions,
    utils,
)
from botorch_rs._botorch_rs import __version__, get_num_threads, set_num_threads
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim import optimize_acqf

__all__ = [
    "SingleTaskGP",
    "__version__",
    "acquisition",
    "constraints",
    "exceptions",
    "fit",
    "fit_gpytorch_mll",
    "get_num_threads",
    "kernels",
    "likelihoods",
    "means",
    "mlls",
    "models",
    "optim",
    "optimize_acqf",
    "posteriors",
    "priors",
    "sampling",
    "set_num_threads",
    "test_functions",
    "utils",
]
