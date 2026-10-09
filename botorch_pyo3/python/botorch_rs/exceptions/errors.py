"""Exceptions raised by botorch_rs (mirroring ``botorch.exceptions.errors``).

Every error raised by the Rust core derives from :class:`BotorchRsError` and
from the matching built-in exception, so that both ``except BotorchRsError``
and, e.g., ``except ValueError`` work.
"""

from __future__ import annotations


class BotorchRsError(Exception):
    """Base class of all botorch_rs errors."""


#: BoTorch's name of the base class.
BotorchError = BotorchRsError


class BotorchTensorDimensionError(BotorchRsError, ValueError):
    """Inputs have inconsistent or unsupported shapes."""


class InputDataError(BotorchRsError, ValueError):
    """Invalid input data, e.g. NaN values or negative noise variances."""


class InvalidArgumentError(BotorchRsError, ValueError):
    """An argument has an invalid value."""


class UnsupportedError(BotorchRsError, NotImplementedError):
    """A BoTorch feature or argument that botorch_rs does not implement."""


class NotPSDError(BotorchRsError, RuntimeError):
    """A covariance matrix is not positive definite even after adding jitter
    (``linear_operator.utils.errors.NotPSDError``)."""


class NanError(BotorchRsError, RuntimeError):
    """A matrix to be factorized contains NaN values
    (``linear_operator.utils.errors.NanError``)."""


class NumericalError(BotorchRsError, RuntimeError):
    """A numerical failure, e.g. non-finite values or gradients."""


class ModelFittingError(BotorchRsError, RuntimeError):
    """All attempts to fit a model failed."""


__all__ = [
    "BotorchError",
    "BotorchRsError",
    "BotorchTensorDimensionError",
    "InputDataError",
    "InvalidArgumentError",
    "ModelFittingError",
    "NanError",
    "NotPSDError",
    "NumericalError",
    "UnsupportedError",
]
