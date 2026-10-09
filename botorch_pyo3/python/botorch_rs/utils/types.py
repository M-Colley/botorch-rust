"""Sentinels (``botorch.utils.types``)."""

from __future__ import annotations


class _DefaultType(type):
    r"""Private class whose sole instance ``DEFAULT`` is a special indicator
    representing that a default value should be assigned to an argument."""

    def __repr__(cls) -> str:
        return "DEFAULT"


class DEFAULT(metaclass=_DefaultType):
    """Use the default value of an argument (e.g. ``outcome_transform``)."""


__all__ = ["DEFAULT"]
