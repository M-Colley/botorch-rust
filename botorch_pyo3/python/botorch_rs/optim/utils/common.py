"""General optimization helpers with optional SciPy dependencies."""

from __future__ import annotations

import importlib


def check_scipy_version_at_least(minor: int, major: int = 1) -> bool:
    """Whether the installed SciPy version is at least ``major.minor.0``."""
    scipy = importlib.import_module("scipy")
    return tuple(map(int, scipy.__version__.split(".")[:2])) >= (major, minor)


__all__ = ["check_scipy_version_at_least"]
