"""Multiple dispatch (``botorch.utils.dispatcher``).

BoTorch's ``Dispatcher`` extends ``multipledispatch.Dispatcher``; this module
implements the same behaviour in pure Python (no dependency): methods are
registered for tuples of types, ordered like ``multipledispatch`` (a
topological sort of the signatures by specificity, ties broken by hash) and
looked up from the types of the encoded arguments.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from inspect import getsource, getsourcefile
from typing import Any


class MDNotImplementedError(NotImplementedError):
    """A registered method raises this to pass the call to the next matching
    method (``multipledispatch.MDNotImplementedError``)."""


def _md_not_implemented_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [MDNotImplementedError]
    try:  # the class of multipledispatch, if installed
        from multipledispatch.dispatcher import MDNotImplementedError as _Other

        errors.append(_Other)
    except ImportError:
        pass
    return tuple(errors)


def type_bypassing_encoder(arg: Any) -> type:
    # Allow type variables to be passed as pre-encoded arguments
    return arg if isinstance(arg, type) else type(arg)


def str_signature(sig: tuple[type, ...]) -> str:
    """``"int, float"`` for ``(int, float)``."""
    return ", ".join(cls.__name__ for cls in sig)


def _supercedes(a: tuple[type, ...], b: tuple[type, ...]) -> bool:
    """Whether ``a`` is at least as specific as ``b``."""
    return len(a) == len(b) and all(issubclass(x, y) for x, y in zip(a, b, strict=True))


def _edge(
    a: tuple[type, ...], b: tuple[type, ...], tie_breaker: Callable[[Any], int] = hash
) -> bool:
    return _supercedes(a, b) and (not _supercedes(b, a) or tie_breaker(a) > tie_breaker(b))


def _toposort(edges: dict[Any, list[Any]]) -> list[Any]:
    incoming: OrderedDict[Any, tuple[Any, ...]] = OrderedDict()
    for key in edges:
        for val in edges[key]:
            incoming[val] = incoming.get(val, ()) + (key,)
    incoming_sets = OrderedDict((k, set(v)) for k, v in incoming.items())
    S = OrderedDict.fromkeys(v for v in edges if v not in incoming_sets)
    L = []
    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in edges.get(n, ()):
            incoming_sets[m].remove(n)
            if not incoming_sets[m]:
                S[m] = None
    if any(incoming_sets.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L


def _ordering(signatures: Any) -> list[tuple[type, ...]]:
    """The signatures from the most to the least specific
    (``multipledispatch.conflict.ordering``)."""
    sigs = [tuple(s) for s in signatures]
    grouped: OrderedDict[Any, list[Any]] = OrderedDict()
    for a in sigs:
        for b in sigs:
            if _edge(a, b):
                grouped.setdefault(a, []).append(b)
    for s in sigs:
        grouped.setdefault(s, [])
    return _toposort(dict(grouped))


def _expand_tuples(types: tuple[Any, ...]) -> list[tuple[type, ...]]:
    """Signatures with a tuple of types expanded into one signature each."""
    if not types:
        return [()]
    rest = _expand_tuples(types[1:])
    first = types[0] if isinstance(types[0], tuple) else (types[0],)
    return [(t, *r) for t in first for r in rest]


class Dispatcher:
    r"""Clearing house for multiple dispatch functionality. This class extends
    ``<multipledispatch.Dispatcher>`` by: (i) generalizing the argument encoding
    convention during method lookup, (ii) implementing ``__getitem__`` as a dedicated
    method lookup function.
    """

    def __init__(
        self,
        name: str,
        doc: str | None = None,
        encoder: Callable[[Any], type] = type,
    ) -> None:
        """
        Args:
            name: A string identifier for the ``Dispatcher`` instance.
            doc: A docstring for the multiply dispatched method(s).
            encoder: A callable that individually transforms the arguments passed
                at runtime in order to construct the key used for method lookup as
                ``tuple(map(encoder, args))``. Defaults to ``type``.
        """
        self.name = self.__name__ = name
        self.doc = doc
        self.funcs: dict[tuple[type, ...], Callable[..., Any]] = {}
        self._cache: dict[tuple[type, ...], Callable[..., Any]] = {}
        self._ordering: list[tuple[type, ...]] | None = None
        self._encoder = encoder

    # -- registration ----------------------------------------------------------

    def register(
        self, *types: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a method for the signature ``types``."""

        def _df(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add(types, func, **kwargs)
            return func

        return _df

    def add(self, signature: tuple[Any, ...], func: Callable[..., Any], **kwargs: Any) -> None:
        """Registers ``func`` for ``signature`` (a tuple of types; tuples of
        types are expanded into one signature per type)."""
        for sig in _expand_tuples(tuple(signature)):
            for t in sig:
                if not isinstance(t, type):
                    raise TypeError(
                        f"Tried to dispatch on non-type: {t}\nIn signature: <{str_signature(sig)}>"
                        f"\nIn function: {self.name}"
                    )
            self.funcs[sig] = func
        self._cache.clear()
        self._ordering = None

    @property
    def ordering(self) -> list[tuple[type, ...]]:
        """The registered signatures from the most to the least specific."""
        if self._ordering is None:
            self._ordering = _ordering(self.funcs)
        return self._ordering

    # -- lookup ----------------------------------------------------------------

    def __getitem__(
        self,
        args: Any | None = None,
        types: tuple[type, ...] | None = None,
    ) -> Callable[..., Any]:
        r"""Method lookup.

        Args:
            args: A set of arguments that act as identifiers for a stored method.
            types: A tuple of types that encodes ``args``.

        Returns:
            A callable corresponding to the given ``args`` or ``types``.
        """
        if types is None:
            if args is None:
                raise RuntimeError("One of `args` or `types` must be provided.")
            types = self.encode_args(args)
        elif args is not None:
            raise RuntimeError("Only one of `args` or `types` may be provided.")

        try:
            func = self._cache[types]
        except KeyError:
            found = self.dispatch(*types)
            if not found:
                msg = f"{self.name}: <{', '.join(cls.__name__ for cls in types)}"
                raise NotImplementedError(f"Could not find signature for {msg}") from None
            func = found
            self._cache[types] = func
        return func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Multiply dispatches a call to a collection of methods.

        Args:
            args: A set of arguments that act as identifiers for a stored method.
            kwargs: Optional keyword arguments passed to the retrieved method.

        Returns:
            The result of evaluating ``func(*args, **kwargs)``, where ``func`` is
            the function obtained via method lookup.
        """
        types = self.encode_args(args)
        func = self.__getitem__(None, types)
        errors = _md_not_implemented_errors()
        try:
            return func(*args, **kwargs)
        except errors:
            # Traverses registered methods in order, yields whenever a match is found
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first, same as self.__getitem__(types=types)
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except errors:
                    pass

            raise NotImplementedError(
                f"Matching functions for {self.name:s}: {str_signature(types):s} "
                "found, but none completed successfully"
            ) from None

    def dispatch(self, *types: type) -> Callable[..., Any] | None:
        r"""Method lookup strategy. Checks for an exact match before traversing
        the set of registered methods according to the current ordering.

        Args:
            types: A tuple of types that gets compared with the signatures
                of registered methods to determine compatibility.

        Returns:
            The first method encountered with a matching signature.
        """
        if types in self.funcs:
            return self.funcs[types]
        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def dispatch_iter(self, *types: type) -> Iterator[Callable[..., Any]]:
        """The registered methods whose signature matches ``types``, from the
        most specific one."""
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(
                issubclass(t, s) for t, s in zip(types, signature, strict=True)
            ):
                yield self.funcs[signature]

    def encode_args(self, args: Any) -> tuple[type, ...]:
        r"""Converts arguments into a tuple of types used during method lookup."""
        return tuple(map(self.encoder, args if isinstance(args, tuple) else (args,)))

    def _help(self, *args: Any) -> str | None:
        r"""Returns the retrieved method's docstring."""
        func = self.dispatch(*self.encode_args(args))
        return None if func is None else func.__doc__

    def help(self, *args: Any, **kwargs: Any) -> None:
        r"""Prints the retrieved method's docstring."""
        print(self._help(*args))

    def _source(self, *args: Any) -> str:
        r"""Returns the retrieved method's source types as a string."""
        func = self.dispatch(*self.encode_args(args))
        if not func:
            raise TypeError("No function found")
        return f"File: {getsourcefile(func)}\n\n{getsource(func)}"

    def source(self, *args: Any, **kwargs: Any) -> None:
        r"""Prints the retrieved method's source types."""
        print(self._source(*args))

    @property
    def encoder(self) -> Callable[[Any], type]:
        return self._encoder

    def __repr__(self) -> str:
        return f"<dispatched {self.name}>"


__all__ = ["Dispatcher", "MDNotImplementedError", "type_bypassing_encoder"]
