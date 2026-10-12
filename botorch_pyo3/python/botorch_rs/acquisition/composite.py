r"""Composite and callable acquisition functions (botorch-rust extensions).

BoTorch users combine acquisition functions in a ``torch.nn.Module`` whose
``forward(X)`` evaluates them and blends the values, and BoTorch's
``optimize_acqf`` optimizes any such module with ``torch.autograd``. botorch_rs
offers two ways to do the same:

* :class:`WeightedLogMeanExp` evaluates the weighted log-mean-exp blend of
  log-scale acquisition functions natively in Rust (exact gradients, parallel
  restarts), e.g. the transfer acquisition function of TAF-style
  meta-learning, ``log((w_t T(x) + Σ_i w_i S_i(x)) / (w_t + Σ_i w_i))`` for
  log-scale terms (Wistuba, Schilling and Schmidt-Thieme, *Machine Learning*
  2018);
* :class:`CallableAcquisitionFunction` wraps any callable on torch tensors
  (``batch x q x d`` candidate sets to ``batch`` values): values come from
  Python and gradients from ``torch.autograd``. ``optimize_acqf`` wraps
  callables automatically. Every evaluation calls Python (holding the GIL),
  so this is slower than a native composite.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.exceptions.errors import UnsupportedError


class CallableAcquisitionFunction(AcquisitionFunction):
    r"""An acquisition function given by a callable on torch tensors, e.g. a
    ``torch.nn.Module`` combining botorch_rs or BoTorch acquisition
    functions: ``acq_function(X)`` maps ``batch x q x d`` float64 tensors to
    ``batch`` values, differentiable with ``torch.autograd``.

    ``optimize_acqf`` (and :class:`WeightedLogMeanExp`) evaluate it from Rust:
    batches of candidate sets in one call, gradients with
    ``torch.autograd.grad``. Pending points are those of the callable
    (``X_pending``, ``set_X_pending``), if it has them.

    Example:
        >>> class Blend(torch.nn.Module):
        ...     def forward(self, X):
        ...         return torch.logsumexp(torch.stack([a(X), b(X)], -1), -1)
        >>> acqf = CallableAcquisitionFunction(Blend(), dim=3)
    """

    def __init__(self, acq_function: Any, dim: int) -> None:
        from botorch_rs._torch_bridge import require_torch

        if not callable(acq_function):
            raise UnsupportedError(
                f"Expected a callable acquisition function, got {type(acq_function).__name__}."
            )
        require_torch()
        torch = importlib.import_module("torch")
        self.acq_function = acq_function
        self.model = getattr(acq_function, "model", None)
        self._pending_kind = ArrayKind("torch", torch.float64)
        self._rs = _rs.callable_acquisition_function(acq_function, int(dim))

    def forward(self, X: Any) -> Any:
        """``acq_function(X)``."""
        return self.acq_function(X)

    def set_X_pending(self, X_pending: Any = None) -> None:
        """``acq_function.set_X_pending(X_pending)``."""
        setter = getattr(self.acq_function, "set_X_pending", None)
        if setter is None:
            raise UnsupportedError(
                f"{type(self.acq_function).__name__} has no `set_X_pending`; pending points are "
                "not supported."
            )
        setter(X_pending)

    @property
    def X_pending(self) -> Any:
        """The callable's ``X_pending`` (``None`` if it has none)."""
        return getattr(self.acq_function, "X_pending", None)

    def __repr__(self) -> str:
        return f"CallableAcquisitionFunction({self.acq_function!r})"


class WeightedLogMeanExp(AcquisitionFunction):
    r"""The weighted log-mean-exp of acquisition functions,

    .. math::

        \alpha(X) = \log \frac{\sum_i w_i \exp(\alpha_i(X))}{\sum_i w_i}
                  = \operatorname{logsumexp}_i(\log w_i + \alpha_i(X)) - \log \sum_i w_i,

    the logarithm of the weighted mean of the linear-scale values when the
    ``acq_functions`` are log-scale (qLogEI, qLogNEI, qLogEHVI, qLogNEHVI,
    LogEI, ...). It is evaluated in Rust exactly like
    ``torch.logsumexp(torch.stack([log(w_i) + acqf_i(X)], -1), -1) -
    log(sum(w))``, with the exact gradient ``Σ_i softmax_i ∇α_i(X)``.

    Args:
        acq_functions: The acquisition functions ``α_i``: botorch_rs
            acquisition functions, or callables on torch tensors (wrapped in
            :class:`CallableAcquisitionFunction`; at least one botorch_rs
            acquisition function must then give the input dimension).
        weights: One non-negative weight per acquisition function (not all
            zero). Terms with weight zero are left out of both sums and are
            not evaluated.

    Pending points are those of the first acquisition function;
    ``set_X_pending`` sets them on every term.

    Example:
        >>> acqf = WeightedLogMeanExp([target_acqf, source_acqf], weights=[1.0, 0.4])
        >>> candidate, value = optimize_acqf(acqf, bounds, q=1, num_restarts=8)
    """

    _log = True

    def __init__(self, acq_functions: Sequence[Any], weights: Any) -> None:
        acqfs = list(acq_functions)
        if not acqfs:
            raise ValueError("WeightedLogMeanExp requires at least one acquisition function.")
        data, shape = to_flat(weights, "weights")
        if len(shape) != 1 or len(data) != len(acqfs):
            raise ValueError(
                f"WeightedLogMeanExp needs one weight per acquisition function, got weights of "
                f"shape {tuple(shape)} for {len(acqfs)} acquisition functions."
            )
        native = [a for a in acqfs if isinstance(a, AcquisitionFunction)]
        if len(native) < len(acqfs):
            if not native:
                raise UnsupportedError(
                    "WeightedLogMeanExp needs at least one botorch_rs acquisition function to "
                    "determine the input dimension; wrap callables in "
                    "CallableAcquisitionFunction(f, dim) instead."
                )
            d = native[0]._rs.dim
            acqfs = [
                a if isinstance(a, AcquisitionFunction) else CallableAcquisitionFunction(a, d)
                for a in acqfs
            ]
        self.acq_functions: list[AcquisitionFunction] = acqfs
        self.weights = weights
        self.model = acqfs[0].model
        self._pending_kind = acqfs[0]._pending_kind
        self._rs = _rs.weighted_log_mean_exp([a._rs for a in acqfs], [float(w) for w in data])

    def _after_evaluation(self) -> None:
        for a in self.acq_functions:
            a._after_evaluation()

    def __repr__(self) -> str:
        names = ", ".join(type(a).__name__ for a in self.acq_functions)
        return f"WeightedLogMeanExp([{names}])"


__all__ = ["CallableAcquisitionFunction", "WeightedLogMeanExp"]
