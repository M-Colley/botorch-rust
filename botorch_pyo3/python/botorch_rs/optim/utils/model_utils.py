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

from botorch_rs._tensor_mll import is_tensor_mll
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

    if hasattr(module, "_param_entries"):
        # Approximate and globally shared MLLs own their flat parameter layout.
        return [("", module)]
    if isinstance(module, SumMarginalLogLikelihood):
        return [(f"model.models.{i}.", m.model) for i, m in enumerate(module.mlls)]
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
    if isinstance(module, (MarginalLogLikelihood, SumMarginalLogLikelihood)) and not getattr(
        module, "_rs_shared_fit", False
    ):
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


def _named_parameters(module: Any) -> Iterator[tuple[str, Any, tuple[Any, Any]]]:
    """``(name, value, raw bounds)`` with BoTorch's names and order."""
    torch = sys.modules.get("torch")
    if is_tensor_mll(module) or (torch is not None and isinstance(module, torch.nn.Module)):
        from botorch_rs.optim._torch_backend import parameter_bounds

        bounds = parameter_bounds(module)
        for name, value in module.named_parameters():
            lower, upper = bounds.get(name, (None, None))
            yield (
                name,
                value,
                (
                    -math.inf if lower is None else lower,
                    math.inf if upper is None else upper,
                ),
            )
        return
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
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    r"""The parameters of a model or marginal log likelihood and the bounds of
    the constrained ones in the raw space (BoTorch's
    ``get_parameters_and_bounds``).

    Returns:
        A dictionary of parameters and a dictionary of parameter bounds.
    """
    params: dict[str, Any] = {}
    bounds: dict[str, tuple[Any, Any]] = {}
    torch = sys.modules.get("torch")
    traversal = getattr(module, "named_parameters_and_constraints", None)
    if torch is not None and isinstance(module, torch.nn.Module) and callable(traversal):
        for name, value, constraint in traversal():
            if requires_grad is not None and value.requires_grad != requires_grad:
                continue
            if name_filter and not name_filter(name):
                continue
            params[name] = value
            if constraint is not None:
                limits = tuple(
                    default if bound is None else constraint.inverse_transform(bound)
                    for bound, default in zip(
                        (constraint.lower_bound, constraint.upper_bound),
                        default_bounds,
                        strict=True,
                    )
                )
                bounds[name] = limits[0], limits[1]
        return params, bounds
    for name, value, (lo, hi) in _named_parameters(module):
        if requires_grad is not None and _requires_grad(value) != requires_grad:
            continue
        if name_filter and not name_filter(name):
            continue
        params[name] = value
        unbounded_lower = isinstance(lo, (int, float)) and lo == -math.inf
        unbounded_upper = isinstance(hi, (int, float)) and hi == math.inf
        if unbounded_lower and unbounded_upper:
            continue
        bounds[name] = (
            default_bounds[0] if unbounded_lower else lo,
            default_bounds[1] if unbounded_upper else hi,
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

    torch = sys.modules.get("torch")
    if getattr(model, "_tensor_backend", None) == "torch" or (
        torch is not None and isinstance(model, torch.nn.Module)
    ):
        _sample_tensor_priors(model, max_retries, seed)
        return
    custom = getattr(model, "_sample_all_priors", None)
    if callable(custom):
        custom(max_retries=max_retries, seed=seed)
        return
    models = model.models if hasattr(model, "models") else [model]
    visited: set[int] = set()
    for i, m in enumerate(models):
        if id(m) in visited:
            continue
        visited.add(id(m))
        rs = getattr(m, "_rs", None)
        sync = getattr(m, "_sync_torch_parameters", None)
        if sync is not None:
            sync()
        sample = getattr(rs, "sample_all_priors", None)
        if callable(sample):
            sample(None if seed is None else int(seed) + i)
        elif isinstance(rs, _rs.SingleTaskGP):
            _rs.sample_all_priors(rs, None if seed is None else int(seed) + i)
        else:
            raise UnsupportedError(
                f"{type(m).__name__} does not expose a native prior-sampling protocol."
            )
        if sync is not None:
            sync()


def _sample_tensor_priors(model: Any, max_retries: int, seed: int | None) -> None:
    import warnings

    import torch

    from botorch_rs.exceptions.warnings import BotorchWarning
    from botorch_rs.models._torch_backend_modules import tensor_prior_records

    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer.")
    devices = sorted(
        {
            value.device.index
            for value in (*model.parameters(), *model.buffers())
            if value.device.type == "cuda"
        }
    )
    with torch.random.fork_rng(devices=devices, enabled=seed is not None), torch.no_grad():
        if seed is not None:
            torch.random.default_generator.manual_seed(int(seed))
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(int(seed))
        for name, owner, prior, closure, setting_closure in tensor_prior_records(model):
            if setting_closure is None:
                raise RuntimeError(
                    "Must provide inverse transform to be able to sample from prior."
                )
            target = closure(owner)
            shape = prior._extended_shape()
            # Singleton prior batch/event axes still represent one scalar
            # distribution: draw independent values for every target entry.
            scalar = shape.numel() == 1
            sample_shape = target.shape if scalar else target.shape[: -len(shape)]
            saved = [(parameter, parameter.detach().clone()) for parameter in owner.parameters()]
            for attempt in range(max_retries):
                try:
                    draw = prior.sample(sample_shape).to(target)
                    if scalar:
                        draw = draw.reshape(target.shape)
                    if not bool(torch.isfinite(draw).all()):
                        raise ValueError("Prior sample must have finite raw values.")
                    setting_closure(owner, draw)
                    break
                except NotImplementedError:
                    for parameter, value in saved:
                        parameter.copy_(value)
                    warnings.warn(
                        f"Sampling not implemented for {type(prior).__name__} ({name}). Skipping.",
                        BotorchWarning,
                        stacklevel=3,
                    )
                    break
                except (RuntimeError, ValueError) as error:
                    for parameter, value in saved:
                        parameter.copy_(value)
                    message = str(error)
                    if not any(
                        token in message
                        for token in ("out of bounds", "finite raw values", "constraint's support")
                    ):
                        raise
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Failed to sample a feasible parameter value from the prior after {max_retries} attempts."
                        ) from error


__all__ = [
    "TorchAttr",
    "get_data_loader",
    "get_name_filter",
    "get_parameters",
    "get_parameters_and_bounds",
    "sample_all_priors",
]
