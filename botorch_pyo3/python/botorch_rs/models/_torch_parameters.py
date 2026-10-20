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
    def __init__(self, model: Any, shapes: Any = None) -> None:
        import torch

        self.gradient_error = _shared_transform_error(model)
        self.entries: list[tuple[str, Any]] = []
        if shapes is None:
            shapes = model._rs.parameter_shapes()
        self.shapes = [(name, tuple(shape)) for name, shape in shapes]
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

        self.gradient_error = _shared_transform_error(model)
        values = self._values()
        dirty = values != self.leaf_values
        if model._rs.parameter_state_version != self.version:
            raw = tuple(model._rs.raw_parameters())
            describe = getattr(model._rs, "parameter_shapes", None)
            shapes = (
                [(name, tuple(shape)) for name, shape in describe()]
                if describe is not None
                else self.shapes
            )
            if shapes != self.shapes:
                if dirty:
                    raise RuntimeError(
                        "Native GP parameter layout and Torch leaves were both modified since "
                        "their last synchronization. Synchronize before changing native support."
                    )
                existing = dict(self.entries)
                refreshed = []
                offset = 0
                for name, shape in shapes:
                    size = math.prod(shape)
                    parameter = existing.get(name)
                    if parameter is None or tuple(parameter.shape) != shape:
                        parameter = torch.nn.Parameter(
                            from_flat(raw[offset : offset + size], shape, model._kind),
                            requires_grad=True if parameter is None else parameter.requires_grad,
                        )
                    refreshed.append((name, parameter))
                    offset += size
                if offset != len(raw):
                    raise RuntimeError("Native GP parameter shapes do not match its raw values.")
                self.shapes, self.entries = shapes, refreshed
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


def _shared_transform_error(model: Any) -> str | None:
    """Check metadata that requires a tied native parameter layout."""
    from botorch_rs.models.transforms.input import Warp

    for name in ("mean_module", "covar_module"):
        module = getattr(model, name, None)
        owner = getattr(module, "_model", model)
        if owner is not None and owner is not model:
            return (
                "Automatic parameter gradients do not support shared mean/kernel bindings "
                "with independent native parameter leaves. Use the native shared-model "
                "likelihood/fitter; query-only gradients remain available."
            )

    def check(transform: Any) -> bool:
        if isinstance(transform, Warp):
            return tuple(transform.batch_shape) != tuple(model._rs.aug_batch_shape)
        children = getattr(transform, "_transforms", {})
        if isinstance(children, dict) and any(check(tf) for tf in children.values()):
            return True
        return any(check(tf) for tf in getattr(transform, "transforms", []))

    if check(getattr(model, "input_transform", None)):
        return (
            "Automatic parameter gradients require Warp batch_shape to equal the model's "
            "_aug_batch_shape. The native model currently stores independent concentrations "
            "per output/batch element, while this Warp shares them. Shared learned-transform "
            "parameter training is unsupported. Freeze model parameters or use torch.no_grad() "
            "for forward/query-only evaluation; the explicit native VJP describes its own layout."
        )
    return None
