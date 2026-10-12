r"""Multi-output acquisition functions
(``botorch.acquisition.multioutput_acquisition``): acquisition functions
with one value per output, to be optimized with a multi-objective optimizer
(``botorch_rs.utils.multi_objective.optimize.optimize_with_nsgaii``).

``MultiOutputPosteriorMean`` and ``MultiOutputAcquisitionFunctionWrapper``
(of botorch_rs acquisition functions) are evaluated in Rust; NSGA-II then
runs entirely in Rust unless an ``objective`` or outcome ``constraints``
(callables) are given.
"""

from __future__ import annotations

import importlib
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


class MultiOutputAcquisitionFunction(AcquisitionFunction):
    r"""Abstract base class of the multi-output acquisition functions:
    ``acqf(X)`` maps ``(b) x q x d`` candidate sets to ``(b) x m`` values (all
    maximized). Subclasses implement ``forward``; those computed in Rust set
    ``_rs_mo``."""

    _rs_mo: Any = None

    def forward(self, X: Any) -> Any:
        """The ``(b) x m`` values of the candidate sets ``X``."""
        if self._rs_mo is None:  # pragma: no cover - abstract
            raise NotImplementedError
        if is_torch_tensor(X) and X.requires_grad:
            raise UnsupportedError(
                "Gradients of multi-output acquisition functions are not supported by "
                "botorch_rs (they are optimized with NSGA-II)."
            )
        kind = kind_of(X)
        data, shape = to_flat(X, "X")
        if len(shape) < 2:
            raise BotorchTensorDimensionError(
                f"X is {len(shape)}-dim but must be at least 2-dim (`batch_shape x q x d`)."
            )
        flat_shape = (1, *shape) if len(shape) == 2 else shape
        values, vshape = self._rs_mo.forward((data, flat_shape))
        out_shape = tuple(vshape[1:]) if len(shape) == 2 else tuple(vshape)
        return from_flat(values, out_shape, kind)

    def set_X_pending(self, X_pending: Any = None) -> None:
        raise UnsupportedError("X_pending is not supported for multi-output acquisition functions.")

    @property
    def X_pending(self) -> Any:
        return None


class MultiOutputPosteriorMean(MultiOutputAcquisitionFunction):
    r"""The posterior mean of every output (times ``weights`` if given; a
    negative weight minimizes its output), for ``q = 1``.

    Example:
        >>> acqf = MultiOutputPosteriorMean(model)
        >>> X, Y = optimize_with_nsgaii(acqf, bounds, num_objectives=2)
    """

    def __init__(self, model: Any, weights: Any = None) -> None:
        super().__init__(model)
        if model.num_outputs < 2:
            raise NotImplementedError("MultiPosteriorMean only supports multi-output models.")
        w = None
        if weights is not None:
            data, shape = to_flat(weights, "weights")
            if shape[0] != model.num_outputs:
                raise ValueError(
                    f"weights must have {model.num_outputs} elements, but got {shape[0]}."
                )
            w = [float(v) for v in data]
        self.weights = weights
        self._rs_mo = _rs.MultiOutputAcquisitionFunction.posterior_mean(
            model._acquisition_model(None), w
        )


class MultiOutputAcquisitionFunctionWrapper(MultiOutputAcquisitionFunction):
    r"""Multi-output wrapper around single-output acquisition functions: the
    values of the ``m`` acquisition functions, stacked."""

    def __init__(self, acqfs: list[Any]) -> None:
        if not acqfs:
            raise ValueError("MultiOutputAcquisitionFunctionWrapper needs acquisition functions.")
        super().__init__(acqfs[0].model)
        self.acqfs = list(acqfs)
        rs: list[Any] = [getattr(a, "_rs", None) for a in self.acqfs]
        if all(isinstance(r, _rs.AcquisitionFunction) for r in rs):
            self._rs_mo = _rs.MultiOutputAcquisitionFunction.wrapper(rs)

    def forward(self, X: Any) -> Any:
        if self._rs_mo is not None:
            return super().forward(X)
        torch = importlib.import_module("torch")
        return torch.stack([acqf(X) for acqf in self.acqfs], dim=-1)


__all__ = [
    "MultiOutputAcquisitionFunction",
    "MultiOutputAcquisitionFunctionWrapper",
    "MultiOutputPosteriorMean",
]
