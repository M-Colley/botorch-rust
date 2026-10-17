"""Assorted model utilities (``botorch.models.utils.assorted``): the
``fantasize`` flag that input transforms consult in eval mode."""

from __future__ import annotations

from typing import Any


class _Flag:
    """A global boolean flag that can be set in a ``with`` block (GPyTorch's
    ``settings._Flag``)."""

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


class fantasize(_Flag):
    """A flag denoting whether we are currently in a ``fantasize`` context:
    input transforms with ``transform_on_fantasize=False`` are skipped in eval
    mode while it is on."""

    _state: bool = False


__all__ = ["fantasize"]
