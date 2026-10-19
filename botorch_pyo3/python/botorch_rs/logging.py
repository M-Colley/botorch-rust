"""Logging (``botorch.logging``): the ``botorch_rs`` logger and
``shape_to_str``."""

from __future__ import annotations

import logging
from collections.abc import Sequence

LOG_LEVEL_DEFAULT = logging.CRITICAL


def _get_logger(name: str = "botorch_rs", level: int = LOG_LEVEL_DEFAULT) -> logging.Logger:
    """Gets a default botorch_rs logger

    Logging level can be tuned via botorch_rs.settings.log_level

    Args:
        name: Name for logger instance
        level: Logging threshold for the given logger. Logs of greater or
            equal severity will be printed to STDERR
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add timestamps to log messages (once per logger).
    if not any(getattr(h, "_botorch_rs", False) for h in logger.handlers):
        console = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
        console.setFormatter(formatter)
        console._botorch_rs = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    logger.propagate = False
    return logger


def shape_to_str(shape: Sequence[int]) -> str:
    """A shape as a string, e.g. ``"`2 x 3`"``."""
    return f"`{' x '.join(str(i) for i in shape)}`"


logger = _get_logger()

__all__ = ["LOG_LEVEL_DEFAULT", "logger", "shape_to_str"]
