r"""A wrapper of acquisition functions that fixes some features
(``botorch.acquisition.fixed_feature``), e.g. for contextual optimization.
Evaluated in Rust with exact gradients with respect to the free features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Number
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_to, from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, pending_arg
from botorch_rs.exceptions.errors import UnsupportedError


def get_dtype_of_sequence(values: Sequence[Any]) -> Any:
    """Return torch.float32 if every value is a float32 tensor, else float64.

    This explicit torch metadata helper imports torch when called; ordinary
    fixed-feature acquisition functions continue to accept lists without it.
    """
    import torch

    return (
        torch.float32
        if all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 for value in values)
        else torch.float64
    )


def get_device_of_sequence(values: Sequence[Any]) -> Any:
    """Return a CUDA device if any tensor is on CUDA, else the CPU device."""
    import torch

    return (
        torch.device("cuda")
        if any(isinstance(value, torch.Tensor) and value.device.type == "cuda" for value in values)
        else torch.device("cpu")
    )


def _values_flat(values: Any) -> tuple[list[float], tuple[int, ...]]:
    """The fixed values (a tensor, or a sequence of numbers and tensors
    broadcast to a common shape as in BoTorch) as flat data and shape
    (``d_f``, ``q x d_f`` or ``batch_shape x q x d_f``)."""
    if (
        not is_torch_tensor(values)
        and isinstance(values, Sequence)
        and not (values and all(isinstance(v, list) for v in values))
    ):
        # A sequence of numbers and arrays (BoTorch's broadcast and
        # concatenation); a nested list is an array.
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
            return flat, (len(flat),)
    data, shape = to_flat(values, "values")
    return [float(x) for x in data], tuple(shape)


def _is_batched(shape: tuple[int, ...]) -> bool:
    """Whether fixed values of this shape differ between t-batches."""
    return len(shape) >= 3 and math.prod(shape[:-2]) > 1


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
        raise UnsupportedError("t-batched fixed values are evaluated in Python.")
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

    The values follow the supplied order of ``columns``, using the repair
    proposed in BoTorch PR #3391: one value per fixed column (shared by all
    points), one row per point of the candidate sets (``q x d_f``), or
    values that differ between the t-batches (``batch_shape x q x d_f``,
    broadcast against the batch shape of ``X`` as in BoTorch). Shared and
    per-point values are filled in natively; with t-batched values, the full
    inputs are built from ``X`` (BoTorch's ``_construct_X_full``) and the base
    acquisition function is evaluated on them (gradients through torch's
    autograd).
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
        vals, shape = _values_flat(values)
        self._batched = _is_batched(shape)
        if self._batched:
            self._values_flat = (vals, shape)
            if shape[-1] >= self.d:
                raise ValueError(f"Expected fewer than d = {self.d} fixed values per point.")
            return
        native_values, native_shape = _values_arg(values)
        # The Rust helper's explicit contract uses increasing physical columns.
        # Adapt the public columns/value pairs without changing that contract.
        order = sorted(range(len(self.columns)), key=self.columns.__getitem__)
        width = len(self.columns)
        if width:
            native_values = [
                native_values[row + index]
                for row in range(0, len(native_values), width)
                for index in order
            ]
        self._rs = _rs.fixed_feature_acquisition_function(
            acq_function._rs,
            self.d,
            [int(c) for c in self.columns],
            (native_values, native_shape),
        )

    def __getattr__(self, name: str) -> Any:
        # With t-batched values, the extension handle is a callable
        # acquisition function of the full inputs (built on first use).
        if name == "_rs" and self.__dict__.get("_batched"):
            from botorch_rs._torch_bridge import require_torch

            require_torch()
            d_prime = self.d - self._values_flat[1][-1]

            def evaluate(X: Any) -> Any:
                return self.acq_func(self._construct_X_full(X))

            handle = _rs.callable_acquisition_function(evaluate, int(d_prime))
            self.__dict__["_rs"] = handle
            return handle
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def forward(self, X: Any) -> Any:
        """Evaluate base acquisition function under the fixed features."""
        if self._batched:
            return self.acq_func(self._construct_X_full(X))
        return super().forward(X)

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
        if self._batched:
            self.acq_func.set_X_pending(
                None if X_pending is None else self._construct_X_full(X_pending)
            )
            return
        self._rs.set_x_pending(pending_arg(X_pending, self._rs.dim))

    def _construct_X_full(self, X: Any) -> Any:
        r"""The full inputs (``batch_shape x q x d``) of ``X`` (``batch_shape x
        q x d'``): ``X_full[..., columns[k]] = values[..., k]`` for fixed columns,
        the columns of ``X`` otherwise; the values are
        broadcast to the batch shape of ``X`` (BoTorch's ``expand``)."""
        vals, vshape = _values_flat(self.values)
        d_f = vshape[-1]
        cols = {column: position for position, column in enumerate(self.columns)}
        selector = []
        idx_x = 0
        for i in range(self.d):
            if i in cols:
                selector.append(self.d - d_f + cols[i])
            else:
                selector.append(idx_x)
                idx_x += 1
        if is_torch_tensor(X):
            import torch

            d_prime = X.shape[-1]
            if d_prime + d_f != self.d:
                raise ValueError(
                    f"Feature dimension d' ({d_prime}) of input must be d - d_f ({self.d - d_f})."
                )
            v = torch.tensor(vals, dtype=X.dtype, device=X.device).reshape(vshape)
            v = v.expand(*X.shape[:-1], d_f)
            return torch.cat([X, v], dim=-1)[..., selector]
        xdata, xshape = to_flat(X, "X")
        d_prime = xshape[-1]
        if d_prime + d_f != self.d:
            raise ValueError(
                f"Feature dimension d' ({d_prime}) of input must be d - d_f ({self.d - d_f})."
            )
        target = (*xshape[:-1], d_f)
        vfull = broadcast_to(vals, vshape, target)
        rows = len(xdata) // d_prime if d_prime else len(vfull) // d_f
        out: list[float] = []
        for r in range(rows):
            perm = list(xdata[r * d_prime : (r + 1) * d_prime]) + list(
                vfull[r * d_f : (r + 1) * d_f]
            )
            out.extend(perm[j] for j in selector)
        return from_flat(out, (*xshape[:-1], self.d), kind_of(X))

    def _after_evaluation(self) -> None:
        self.acq_func._after_evaluation()


__all__ = ["FixedFeatureAcquisitionFunction", "get_device_of_sequence", "get_dtype_of_sequence"]
