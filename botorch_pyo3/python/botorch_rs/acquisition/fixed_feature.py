r"""A wrapper of acquisition functions that fixes some features
(``botorch.acquisition.fixed_feature``), e.g. for contextual optimization.
Evaluated in Rust with exact gradients with respect to the free features."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import is_torch_tensor, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, pending_arg
from botorch_rs.exceptions.errors import UnsupportedError


def _values_arg(values: Any) -> tuple[list[float], tuple[int, int]]:
    """The fixed values as a ``1 x d_f`` (shared) or ``q x d_f`` array."""
    if not is_torch_tensor(values) and isinstance(values, Sequence):
        parts = list(values)
        if any(is_torch_tensor(v) and v.ndim > 1 for v in parts):
            import torch

            parts = [
                torch.tensor([v], dtype=torch.float64)
                if isinstance(v, Number)
                else (v.unsqueeze(0) if v.ndim == 0 else v).detach().to(torch.float64)
                for v in parts
            ]
            values = torch.cat(torch.broadcast_tensors(*parts), dim=-1)
        else:
            flat: list[float] = []
            for v in parts:
                if isinstance(v, Number):
                    flat.append(float(v))  # type: ignore[arg-type]
                else:
                    flat.extend(float(x) for x in to_flat(v, "values")[0])
            return flat, (1, len(flat))
    data, shape = to_flat(values, "values")
    if len(shape) == 1:
        return [float(x) for x in data], (1, shape[0])
    if any(s != 1 for s in shape[:-2]):
        raise UnsupportedError("t-batched fixed values are not supported by botorch_rs.")
    return [float(x) for x in data], (shape[-2], shape[-1])


class FixedFeatureAcquisitionFunction(AcquisitionFunction):
    """A wrapper around AcquisitionFunctions to fix a subset of features.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)  # d = 5
        >>> qEI = qExpectedImprovement(model, best_f=0.0)
        >>> columns = [2, 4]
        >>> values = X[..., columns]
        >>> qEI_FF = FixedFeatureAcquisitionFunction(qEI, 5, columns, values)
        >>> qei = qEI_FF(test_X)  # d' = 3

    The values are those of the fixed columns in increasing order of the
    columns (as in BoTorch): one value per fixed column (shared by all
    points) or one row per point of the candidate sets (``q x d_f``).
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        d: int,
        columns: list[int],
        values: Any,
    ) -> None:
        if not isinstance(acq_function, AcquisitionFunction):
            raise UnsupportedError("`acq_function` must be a botorch_rs acquisition function.")
        super().__init__(acq_function.model)
        self.acq_func = acq_function
        self.d = int(d)
        self.columns = list(columns)
        self.values = values
        self._log = acq_function._log
        self._rs = _rs.fixed_feature_acquisition_function(
            acq_function._rs,
            self.d,
            [int(c) for c in self.columns],
            _values_arg(values),
        )

    @property
    def X_pending(self) -> Any:
        r"""The ``X_pending`` (full ``m x d`` points) of the base acquisition
        function."""
        return self.acq_func.X_pending

    @X_pending.setter
    def X_pending(self, X_pending: Any) -> None:
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Any = None) -> None:
        """Sets the pending points (``m x d'``, the free features) of the base
        acquisition function with the fixed features filled in."""
        self._rs.set_x_pending(pending_arg(X_pending, self._rs.dim))

    def _construct_X_full(self, X: Any) -> Any:
        r"""The full inputs (``batch_shape x q x d``) of ``X`` (``batch_shape x
        q x d'``)."""
        import torch

        vals, (r, d_f) = _values_arg(self.values)
        d_prime = X.shape[-1]
        if d_prime + d_f != self.d:
            raise ValueError(
                f"Feature dimension d' ({d_prime}) of input must be d - d_f ({self.d - d_f})."
            )
        v = torch.tensor(vals, dtype=X.dtype).reshape(r, d_f).expand(*X.shape[:-1], d_f)
        cols = sorted(self.columns)
        selector = []
        idx_x, idx_f = 0, d_prime
        for i in range(self.d):
            if i in cols:
                selector.append(idx_f)
                idx_f += 1
            else:
                selector.append(idx_x)
                idx_x += 1
        return torch.cat([X, v], dim=-1)[..., selector]

    def _after_evaluation(self) -> None:
        self.acq_func._after_evaluation()


__all__ = ["FixedFeatureAcquisitionFunction"]
