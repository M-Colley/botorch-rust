"""Exceptions and warnings (``botorch.exceptions``)."""

from botorch_rs.exceptions.errors import (
    BotorchError,
    BotorchRsError,
    BotorchTensorDimensionError,
    InputDataError,
    InvalidArgumentError,
    ModelFittingError,
    NanError,
    NotPSDError,
    NumericalError,
    UnsupportedError,
)
from botorch_rs.exceptions.warnings import (
    BadInitialCandidatesWarning,
    BotorchWarning,
    InputDataWarning,
    NumericsWarning,
    OptimizationWarning,
    UserInputWarning,
)

__all__ = [
    "BadInitialCandidatesWarning",
    "BotorchError",
    "BotorchRsError",
    "BotorchTensorDimensionError",
    "BotorchWarning",
    "InputDataError",
    "InputDataWarning",
    "InvalidArgumentError",
    "ModelFittingError",
    "NanError",
    "NotPSDError",
    "NumericalError",
    "NumericsWarning",
    "OptimizationWarning",
    "UnsupportedError",
    "UserInputWarning",
]
