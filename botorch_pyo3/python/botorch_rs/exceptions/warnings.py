"""Warnings emitted by botorch_rs (mirroring ``botorch.exceptions.warnings``)."""

from __future__ import annotations

import warnings


class BotorchWarning(Warning):
    """Base class of all botorch_rs warnings."""


class BadInitialCandidatesWarning(BotorchWarning):
    """The initial conditions of the acquisition optimization are poor."""


class InputDataWarning(BotorchWarning):
    """The training data are not normalized / standardized."""


class NumericsWarning(BotorchWarning):
    """A method with known numerical issues is used."""


class OptimizationWarning(BotorchWarning):
    """An optimization did not terminate successfully."""


class UserInputWarning(BotorchWarning):
    """A potentially problematic argument value."""


class SamplingWarning(BotorchWarning):
    """A sampling-related issue (e.g. i.i.d. instead of quasi-random samples)."""


_LEGACY_TO_LOGEI = {
    "ExpectedImprovement": "LogExpectedImprovement",
    "ConstrainedExpectedImprovement": "LogConstrainedExpectedImprovement",
    "qExpectedImprovement": "qLogExpectedImprovement",
    "qExpectedHypervolumeImprovement": "qLogExpectedHypervolumeImprovement",
    "qNoisyExpectedHypervolumeImprovement": "qLogNoisyExpectedHypervolumeImprovement",
}


def legacy_ei_numerics_warning(legacy_name: str) -> None:
    """BoTorch's recommendation to use the LogEI variant of a legacy EI."""
    if legacy_name in _LEGACY_TO_LOGEI:
        logei_name = _LEGACY_TO_LOGEI[legacy_name]
        msg = (
            f"{legacy_name} has known numerical issues that lead to suboptimal "
            "optimization performance. It is strongly recommended to simply replace"
            f"\n\n\t {legacy_name} \t --> \t {logei_name} \n\n"
            "instead, which fixes the issues and has the same "
            "API. See https://arxiv.org/abs/2310.20708 for details."
        )
        warnings.warn(msg, NumericsWarning, stacklevel=3)


__all__ = [
    "BadInitialCandidatesWarning",
    "BotorchWarning",
    "InputDataWarning",
    "NumericsWarning",
    "OptimizationWarning",
    "SamplingWarning",
    "UserInputWarning",
    "legacy_ei_numerics_warning",
]
