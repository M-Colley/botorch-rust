r"""``BufferDict`` (``botorch.utils.torch``): an ordered dictionary of
buffers.

With torch, ``BufferDict`` is (as in BoTorch) a ``torch.nn.Module`` whose
entries are registered buffers, visible to ``state_dict``, ``to`` and the
other ``Module`` methods; the class is created when it is first accessed.
Without torch, it is an ordered dictionary of arrays with the same methods.
"""

from __future__ import annotations

import collections
import importlib
from collections import OrderedDict
from typing import Any


class _BufferDictMixin:
    """The dictionary methods of ``BufferDict`` (on ``self._buffers``)."""

    _buffers: Any
    register_buffer: Any

    def __getitem__(self, key: str) -> Any:
        return self._buffers[key]

    def __setitem__(self, key: str, buffer: Any) -> None:
        self.register_buffer(key, buffer)

    def __delitem__(self, key: str) -> None:
        del self._buffers[key]

    def __len__(self) -> int:
        return len(self._buffers)

    def __iter__(self) -> Any:
        return iter(self._buffers.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._buffers

    def clear(self) -> None:
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key: str) -> Any:
        r"""Remove key from the BufferDict and return its buffer.

        Args:
            key (string): key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Any:
        r"""Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self) -> Any:
        r"""Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self) -> Any:
        r"""Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers: Any) -> None:
        r"""Update the BufferDict with the key-value pairs from a mapping or an
        iterable, overwriting existing keys.

        .. note::
            If :attr:``buffers`` is an ``OrderedDict``, a BufferDict, or an
            iterable of key-value pairs, the order of new elements in it is
            preserved.

        Args:
            buffers (iterable): a mapping (dictionary) from string to
                :class:``~torch.Tensor``, or an iterable of
                key-value pairs of type (string, :class:``~torch.Tensor``)
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError(
                "BuffersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (OrderedDict, _BufferDictMixin)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                p = list(p)
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def __call__(self, input: Any) -> Any:
        raise RuntimeError("BufferDict should not be called.")


class _PlainBufferDict(_BufferDictMixin):
    r"""Holds buffers in a dictionary (without torch: arrays of any kind)."""

    def __init__(self, buffers: Any = None) -> None:
        self._buffers: OrderedDict[str, Any] = OrderedDict()
        if buffers is not None:
            self.update(buffers)

    def register_buffer(self, name: str, tensor: Any) -> None:
        if not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {type(name).__name__}")
        if "." in name:
            raise KeyError('buffer name can\'t contain "."')
        if name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        self._buffers[name] = tensor

    def __repr__(self) -> str:
        lines = [f"  ({k}): Buffer of type {type(v).__name__}" for k, v in self._buffers.items()]
        return "BufferDict(" + ("\n" + "\n".join(lines) + "\n" if lines else "") + ")"


def _build() -> type:
    torch: Any = importlib.import_module("torch")

    class BufferDict(_BufferDictMixin, torch.nn.Module):
        r"""Holds buffers in a dictionary.

        BufferDict can be indexed like a regular Python dictionary, but buffers it
        contains are properly registered, and will be visible by all Module methods.

        :class:``~torch.nn.BufferDict`` is an **ordered** dictionary that respects

        * the order of insertion, and

        * in :meth:``~torch.nn.BufferDict.update``, the order of the merged
          ``OrderedDict`` or another :class:``~torch.nn.BufferDict`` (the argument
          to :meth:``~torch.nn.BufferDict.update``).

        Note that :meth:``~torch.nn.BufferDict.update`` with other unordered mapping
        types (e.g., Python's plain ``dict``) does not preserve the order of the
        merged mapping.

        Args:
            buffers (iterable, optional): a mapping (dictionary) of
                (string : :class:``~torch.Tensor``) or an iterable of key-value pairs
                of type (string, :class:``~torch.Tensor``)
        """

        def __init__(self, buffers: Any = None) -> None:
            r"""
            Args:
                buffers: A mapping (dictionary) from string to
                    :class:``~torch.Tensor``, or an iterable of key-value pairs
                    of type (string, :class:``~torch.Tensor``).
            """
            torch.nn.Module.__init__(self)
            if buffers is not None:
                self.update(buffers)

        def extra_repr(self) -> str:
            child_lines = []
            for k, p in self._buffers.items():
                size_str = "x".join(str(size) for size in p.size())
                device_str = "" if not p.is_cuda else f" (GPU {p.get_device()})"
                parastr = f"Buffer containing: [{torch.typename(p)} of size {size_str}{device_str}]"
                child_lines.append("  (" + k + "): " + parastr)
            return "\n".join(child_lines)

        def __call__(self, input: Any) -> Any:  # type: ignore[override]
            raise RuntimeError("BufferDict should not be called.")

    BufferDict.__module__ = __name__
    BufferDict.__qualname__ = "BufferDict"
    return BufferDict


def __getattr__(name: str) -> Any:
    if name == "BufferDict":
        try:
            cls: Any = _build()
        except ImportError:
            cls = type("BufferDict", (_PlainBufferDict,), {"__module__": __name__})
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BufferDict"]  # noqa: F822 (created lazily by __getattr__)
