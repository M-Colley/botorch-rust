"""Utilities for fitting and manipulating models
(``botorch.optim.utils.model_utils``).

botorch_rs models keep their hyperparameters in Rust. Ordinary Torch
SingleTaskGPs expose persistent synchronized parameter leaves; other models
return snapshots. Dictionaries map BoTorch's parameter names to these values.
The native fitting routines use the names to select optimized parameters.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator
from re import Pattern
from typing import Any, NamedTuple

from botorch_rs.exceptions.errors import UnsupportedError


class TorchAttr(NamedTuple):
    """The shape, dtype and device of a tensor (BoTorch's ``TorchAttr``)."""

    shape: Any
    dtype: Any
    device: Any


def _requires_grad(value: Any) -> bool:
    """Only persistent Parameter leaves carry native trainability flags.

    Other native model families return detached array descriptors; those
    still describe trainable Rust parameters, regardless of Tensor flags.
    """
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.nn.Parameter):
        return bool(value.requires_grad)
    return True


def get_data_loader(model: Any, batch_size: int = 1024, **kwargs: Any) -> Any:
    """Create a torch DataLoader over the model's stored training tensors.

    This explicit PyTorch interoperability helper imports torch when called.
    It retains BoTorch's batch-size and TensorDataset semantics; it does not
    enable unsupported minibatch fitting paths in the native model fitters.
    """
    from torch.utils.data import DataLoader, TensorDataset

    inputs = model.train_inputs
    targets = model.train_targets
    dataset = TensorDataset(*inputs, targets)
    return DataLoader(dataset, batch_size=min(batch_size, len(targets)), **kwargs)


def _model_sources(module: Any) -> list[tuple[str, Any]]:
    """``(name prefix, model)`` pairs of a model or an MLL with BoTorch's
    parameter names (an MLL registers the likelihood before the model)."""
    from botorch_rs.mlls import MarginalLogLikelihood, SumMarginalLogLikelihood

    if isinstance(module, SumMarginalLogLikelihood):
        return [(f"model.models.{i}.", m.model) for i, m in enumerate(module.mlls)]
    if hasattr(module, "_param_entries"):
        # An approximate MLL: the parameters with the MLL's names.
        return [("", module)]
    if isinstance(module, MarginalLogLikelihood):
        return [("model.", module.model)]
    return [("", module)]


class _Entry(NamedTuple):
    """A hyperparameter: BoTorch's name, current raw value, raw bounds, the
    model holding it and its position in the model's flat raw vector."""

    name: str
    value: Any
    bounds: tuple[float, float]
    model: Any
    offset: int
    size: int


def _entries(module: Any) -> list[_Entry]:
    """The hyperparameters with BoTorch's names and order."""
    from botorch_rs.mlls import MarginalLogLikelihood, SumMarginalLogLikelihood

    entries: list[_Entry] = []
    for prefix, model in _model_sources(module):
        rs = getattr(model, "_rs", None)
        if rs is None or not hasattr(model, "named_parameters"):
            raise UnsupportedError(
                f"{type(model).__name__} has no botorch_rs hyperparameters to optimize."
            )
        bounds = list(rs.raw_parameter_bounds())
        param_entries = getattr(model, "_param_entries", None)
        if param_entries is not None:
            # Models whose `named_parameters()` order differs from the flat
            # order (an approximate GP: the model's parameters, then the
            # likelihood's).
            for name, value, offset, size in param_entries():
                b = bounds[offset : offset + size]
                lo = min(x[0] for x in b) if b else -math.inf
                hi = max(x[1] for x in b) if b else math.inf
                entries.append(_Entry(prefix + name, value, (lo, hi), model, offset, size))
            continue
        sizes = [size for _, size in rs.parameter_names()]
        offset = 0
        for (name, value), size in zip(model.named_parameters(), sizes, strict=True):
            b = bounds[offset : offset + size]
            lo = min(x[0] for x in b) if b else -math.inf
            hi = max(x[1] for x in b) if b else math.inf
            entries.append(_Entry(prefix + name, value, (lo, hi), model, offset, size))
            offset += size
    if isinstance(module, (MarginalLogLikelihood, SumMarginalLogLikelihood)):
        # The likelihood's parameters come first, under the MLL's likelihood.
        def is_likelihood(name: str) -> bool:
            return ".likelihood." in name or name.startswith("model.likelihood.")

        def rename(name: str) -> str:
            if isinstance(module, SumMarginalLogLikelihood):
                i = name.split(".")[2]
                return f"likelihood.likelihoods.{i}." + name.split(".likelihood.", 1)[1]
            return "likelihood." + name[len("model.likelihood.") :]

        likelihood = [e._replace(name=rename(e.name)) for e in entries if is_likelihood(e.name)]
        rest = [e for e in entries if not is_likelihood(e.name)]
        entries = likelihood + rest
    return entries


def _named_parameters(module: Any) -> Iterator[tuple[str, Any, tuple[float, float]]]:
    """``(name, value, raw bounds)`` with BoTorch's names and order."""
    for e in _entries(module):
        yield e.name, e.value, e.bounds


def get_parameters(
    module: Any,
    requires_grad: bool | None = None,
    name_filter: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    r"""The parameters of a model or marginal log likelihood (BoTorch's
    ``get_parameters``).

    Args:
        module: A botorch_rs model or marginal log likelihood.
        requires_grad: Optional Boolean used to filter parameters based on
            whether they are trainable. Torch leaves retain their actual flags.
        name_filter: Optional Boolean function used to filter parameters by
            name.

    Returns:
        A dictionary of parameters (name to current raw value).
    """
    out: dict[str, Any] = {}
    for name, value, _ in _named_parameters(module):
        if requires_grad is not None and _requires_grad(value) != requires_grad:
            continue
        if name_filter and not name_filter(name):
            continue
        out[name] = value
    return out


def get_parameters_and_bounds(
    module: Any,
    requires_grad: bool | None = None,
    name_filter: Callable[[str], bool] | None = None,
    default_bounds: tuple[float, float] = (-math.inf, math.inf),
) -> tuple[dict[str, Any], dict[str, tuple[float | None, float | None]]]:
    r"""The parameters of a model or marginal log likelihood and the bounds of
    the constrained ones in the raw space (BoTorch's
    ``get_parameters_and_bounds``).

    Returns:
        A dictionary of parameters and a dictionary of parameter bounds.
    """
    params: dict[str, Any] = {}
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for name, value, (lo, hi) in _named_parameters(module):
        if requires_grad is not None and _requires_grad(value) != requires_grad:
            continue
        if name_filter and not name_filter(name):
            continue
        params[name] = value
        if lo == -math.inf and hi == math.inf:
            continue
        bounds[name] = (
            default_bounds[0] if lo == -math.inf else lo,
            default_bounds[1] if hi == math.inf else hi,
        )
    return params, bounds


def get_name_filter(
    patterns: Iterator[Pattern[str] | str],
) -> Callable[[str | tuple[str, Any]], bool]:
    r"""A binary function that filters strings (or iterables whose first
    element is a string) according to a bank of excluded patterns (BoTorch's
    ``get_name_filter``).

    Args:
        patterns: A collection of regular expressions or strings that define
            the set of names to be excluded.
    """
    names = set()
    _patterns = set()
    for pattern in patterns:
        if isinstance(pattern, str):
            names.add(pattern)
        elif isinstance(pattern, Pattern):
            _patterns.add(pattern)
        else:
            raise TypeError(
                "Expected `patterns` to contain `str` or `re.Pattern` typed elements, "
                f"but found {type(pattern)}."
            )

    def name_filter(item: str | tuple[str, Any]) -> bool:
        name = item if isinstance(item, str) else next(iter(item))
        if name in names:
            return False
        for pattern in _patterns:
            if pattern.search(name):
                return False
        return True

    return name_filter


def sample_all_priors(model: Any, max_retries: int = 100, seed: int | None = None) -> None:
    r"""Samples the hyperparameters that have priors from their priors
    (in place; BoTorch's ``sample_all_priors``). ``seed`` (botorch_rs only)
    seeds the draws."""
    from botorch_rs import _botorch_rs as _rs

    models = model.models if hasattr(model, "models") else [model]
    for i, m in enumerate(models):
        rs = getattr(m, "_rs", None)
        if isinstance(rs, _rs.SingleTaskVariationalGP):
            rs.sample_all_priors(None if seed is None else int(seed) + i)
            continue
        if rs is None or not isinstance(rs, _rs.SingleTaskGP):
            raise UnsupportedError("sample_all_priors requires botorch_rs SingleTaskGP models.")
        _rs.sample_all_priors(rs, None if seed is None else int(seed) + i)


__all__ = [
    "TorchAttr",
    "get_data_loader",
    "get_name_filter",
    "get_parameters",
    "get_parameters_and_bounds",
    "sample_all_priors",
]
