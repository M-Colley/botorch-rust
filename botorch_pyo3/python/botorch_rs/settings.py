r"""
BoTorch settings (``botorch.settings``): ``propagate_grads``,
``validate_input_scaling`` and ``log_level``.

``validate_input_scaling`` controls the input and outcome scaling warnings
of botorch_rs's ``SingleTaskGP``; ``propagate_grads`` is accepted for
compatibility (botorch_rs models do not propagate gradients to their
training data, so it has no effect); ``log_level`` sets the level of the
``botorch_rs`` logger.
"""

from __future__ import annotations

from typing import Any

from botorch_rs.logging import LOG_LEVEL_DEFAULT, logger


class _Flag:
    r"""Base class for context managers for a binary setting."""

    _state: bool = False

    @classmethod
    def on(cls) -> bool:
        return cls._state

    @classmethod
    def off(cls) -> bool:
        return not cls._state

    @classmethod
    def _set_state(cls, state: bool) -> None:
        cls._state = state

    def __init__(self, state: bool = True) -> None:
        self.prev = self.__class__.on()
        self.state = state

    def __enter__(self) -> None:
        self.__class__._set_state(self.state)

    def __exit__(self, *args: Any) -> None:
        self.__class__._set_state(self.prev)


class propagate_grads(_Flag):
    r"""Compatibility flag for BoTorch's training-data gradient setting.

    botorch_rs models do not propagate gradients to training inputs or targets,
    including through fantasy training data. Setting this flag to ``True``
    does not enable those derivatives. Supported query and model-parameter
    derivatives are independent of this flag. The context preserves and
    restores its state for callers that inspect ``on()`` or ``off()``.
    """

    _state: bool = False


class validate_input_scaling(_Flag):
    r"""Flag for validating input normalization/standardization.

    When set to ``True``, standard botorch models will validate (up to reasonable
    tolerance) that
    (i) none of the inputs contain NaN values
    (ii) the training data (``train_X``) is normalized to the unit cube
    (iii) the training targets (``train_Y``) are standardized (zero mean, unit var)
    No checks (other than the NaN check) are performed for observed variances
    (``train_Y_var``) at this point.
    """

    _state: bool = True


class log_level:
    r"""Flag for printing verbose logging statements.

    Applies the given level to logging.getLogger('botorch_rs') calls. For
    instance, when set to logging.INFO, all logger calls of level INFO or
    above will be printed to STDERR
    """

    level: int = LOG_LEVEL_DEFAULT

    @classmethod
    def _set_level(cls, level: int) -> None:
        cls.level = level
        logger.setLevel(level)

    def __init__(self, level: int = LOG_LEVEL_DEFAULT) -> None:
        r"""
        Args:
            level: The log level. Defaults to LOG_LEVEL_DEFAULT.
        """
        self.prev = self.__class__.level
        self.level = level

    def __enter__(self) -> None:
        self.__class__._set_level(self.level)

    def __exit__(self, *args: Any) -> None:
        self.__class__._set_level(self.prev)


__all__ = ["log_level", "propagate_grads", "validate_input_scaling"]
