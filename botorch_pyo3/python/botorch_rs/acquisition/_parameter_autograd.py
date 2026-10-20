"""Native acquisition tapes and immutable persistent-leaf connections."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError

_query_function: Any = None
_parameter_function: Any = None


def _sources(model: Any) -> list[Any]:
    children = getattr(model, "models", None)
    if children is not None:
        return [source for child in children for source in _sources(child)]
    source = getattr(model, "_rs", None)
    return [source] if source is not None and hasattr(source, "parameter_state_version") else []


class ParameterSnapshot:
    def __init__(self, model: Any, state: Any) -> None:
        self.model = model
        self.entries = tuple(state.entries)
        self.registry = tuple((name, id(p), tuple(p.shape)) for name, p in self.entries)
        self.error = state.gradient_error
        self.versions = self._versions()
        self.values = self._values()

    def _versions(self) -> tuple[Any, ...]:
        return tuple(
            (id(source), source.parameter_state_version) for source in _sources(self.model)
        )

    def _values(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(to_flat(p)[0]) for _, p in self.entries)

    def validate(self, model: Any) -> None:
        state = model._sync_torch_parameters() if model is self.model else None
        if (
            model is not self.model
            or state is None
            or state.gradient_error != self.error
            or tuple((name, id(p), tuple(p.shape)) for name, p in state.entries) != self.registry
            or self._versions() != self.versions
            or self._values() != self.values
        ):
            raise RuntimeError(
                "The acquisition's native model snapshot is stale: model state or Torch "
                "parameters changed after construction. Reconstruct the acquisition function "
                "before evaluating it again. Earlier backward graphs retain their original law."
            )


def capture(model: Any) -> ParameterSnapshot | None:
    sync = getattr(model, "_sync_torch_parameters", None)
    state = None if sync is None else sync()
    return None if state is None else ParameterSnapshot(model, state)


def evaluate(acquisition: Any, X: Any, snapshot: ParameterSnapshot) -> Any:
    """One native evaluation; parameter contractions wait for requested backward."""
    global _query_function, _parameter_function
    import torch

    query_grad = X.requires_grad
    native = acquisition._rs
    tape, error = None, snapshot.error
    if native.supports_parameter_gradients:
        values, gradient, tape = native.forward_with_adjoint_tape(to_flat(X), query_grad)
    else:
        error = error or (
            f"{type(acquisition).__name__} does not support automatic model-parameter "
            "gradients yet; query-only gradients remain available."
        )
        if query_grad:
            values, gradient = native.forward_with_grad(to_flat(X))
        else:
            values, gradient = native.forward(to_flat(X)), None
    result = from_flat(values[0], tuple(values[1]), kind_of(X))
    if _query_function is None:
        from torch.autograd.function import once_differentiable

        class _Query(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, query: Any, output: Any, derivative: Any) -> Any:  # type: ignore[override]
                ctx.save_for_backward(derivative)
                return output

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, adjoint: Any) -> tuple[Any, None, None]:  # type: ignore[override]
                (derivative,) = ctx.saved_tensors
                return adjoint[..., None, None] * derivative, None, None

        class _Parameters(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, reverse: Any, failure: Any, output: Any, *leaves: Any) -> Any:  # type: ignore[override]
                ctx.tape, ctx.error = reverse, failure
                ctx.layout = [(tuple(p.shape), p.numel(), kind_of(p)) for p in leaves]
                shape, dtype, device = output
                return torch.zeros((), dtype=dtype, device=device).expand(shape)

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, adjoint: Any) -> tuple[Any, ...]:  # type: ignore[override]
                if ctx.error is not None:
                    raise UnsupportedError(ctx.error)
                count = sum(size for _, size, _ in ctx.layout)
                values_, _ = ctx.tape.parameter_vjp(to_flat(adjoint), count)
                gradients: list[Any] = []
                offset = 0
                for shape, size, kind in ctx.layout:
                    gradients.append(from_flat(values_[offset : offset + size], shape, kind))
                    offset += size
                return None, None, None, *gradients

        _query_function, _parameter_function = _Query, _Parameters
    if query_grad:
        assert gradient is not None
        result = _query_function.apply(
            X, result, from_flat(gradient[0], tuple(X.shape), kind_of(X))
        )
    return result + _parameter_function.apply(
        tape,
        error,
        (tuple(result.shape), result.dtype, result.device),
        *(p for _, p in snapshot.entries),
    )
