"""Persistent CPU Torch leaves synchronized with an exact native GP.

Only array/state plumbing lives here; posterior parameter derivatives and
optimizer arithmetic remain native. Importing this module does not import Torch.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat

_constraint_function: Any = None


def constrained_parameter(raw: Any, constraint: Any) -> Any:
    """A differentiable logical parameter using native scalar transforms."""
    global _constraint_function
    if constraint is None or not constraint.enforced:
        return raw
    import torch

    if _constraint_function is None:
        from torch.autograd.function import once_differentiable

        class _Constraint(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, value: Any, native: Any) -> Any:  # type: ignore[override]
                data, shape = to_flat(value)
                kind = kind_of(value)
                ctx.save_for_backward(
                    from_flat([native.transform_derivative(v) for v in data], shape, kind)
                )
                return from_flat([native.transform(v) for v in data], shape, kind)

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, adjoint: Any) -> tuple[Any, None]:  # type: ignore[override]
                (derivative,) = ctx.saved_tensors
                return adjoint * derivative, None

        _constraint_function = _Constraint
    return _constraint_function.apply(raw, constraint._rs)


class TorchParameterState:
    def __init__(self, model: Any) -> None:
        import torch

        self.entries: list[tuple[str, Any]] = []
        self.shapes = [(name, tuple(shape)) for name, shape in model._rs.parameter_shapes()]
        raw = model._rs.raw_parameters()
        offset = 0
        for name, shape in self.shapes:
            size = math.prod(shape)
            self.entries.append(
                (
                    name,
                    torch.nn.Parameter(from_flat(raw[offset : offset + size], shape, model._kind)),
                )
            )
            self.entries[-1][1].requires_grad_(
                model.__dict__.get("_torch_parameter_requires_grad", {}).get(name, True)
            )
            offset += size
        self.raw = tuple(raw)
        self.leaf_values = self._values()
        # The caller-visible leaves are authoritative at their logical dtype.
        # Native f64 computation must start from the same rounded parameters.
        if self.leaf_values != self.raw:
            model._rs.set_raw_parameters(self.leaf_values)
            self.raw = self.leaf_values
        self.version = model._rs.parameter_state_version

    def _values(self) -> tuple[float, ...]:
        values: list[float] = []
        for (name, parameter), (_, shape) in zip(self.entries, self.shapes, strict=True):
            if tuple(parameter.shape) != shape or parameter.device.type != "cpu":
                raise ValueError(f"Parameter {name} must retain shape {shape} on the CPU.")
            data, _ = to_flat(parameter, name)
            values.extend(data)
        return tuple(values)

    def sync(self, model: Any) -> None:
        import torch

        values = self._values()
        dirty = values != self.leaf_values
        if model._rs.parameter_state_version != self.version:
            raw = tuple(model._rs.raw_parameters())
            if raw != self.raw:
                if dirty:
                    raise RuntimeError(
                        "Native GP parameters and Torch leaves were both modified since their "
                        "last synchronization. Synchronize with model.named_parameters() after "
                        "native updates before editing the Torch parameters."
                    )
                offset = 0
                with torch.no_grad():
                    for (_, parameter), (_, shape) in zip(self.entries, self.shapes, strict=True):
                        size = math.prod(shape)
                        parameter.copy_(from_flat(raw[offset : offset + size], shape, model._kind))
                        offset += size
                self.raw = raw
                self.leaf_values = self._values()
                # Pulling into a float32 leaf rounds the native update as well.
                if self.leaf_values != raw:
                    model._rs.set_raw_parameters(self.leaf_values)
                    self.raw = self.leaf_values
            self.version = model._rs.parameter_state_version
        if dirty:
            model._rs.set_raw_parameters(values)
            self.raw = self.leaf_values = values
            self.version = model._rs.parameter_state_version

    def parameter(self, name: str) -> Any:
        return dict(self.entries)[name]
