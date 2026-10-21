"""Input transforms (``botorch.models.transforms.input``).

Every transform can be used on its own (``tf(X)``, ``tf.transform(X)``,
``tf.untransform(X)``, train and eval mode with BoTorch's learning of
coefficients in train mode) and as the ``input_transform`` of a model, which
evaluates it natively in Rust with exact gradients (with respect to the
inputs, and to learned parameters such as ``Warp``'s concentrations, which
``fit_gpytorch_mll`` fits jointly with the GP). Like BoTorch, a model
applies the transforms whose ``transform_on_train`` is set to its training
inputs, those whose ``transform_on_eval`` is set to the points of a
posterior, and inside ``fantasize`` only those that also have
``transform_on_fantasize`` set (e.g. ``InputPerturbation`` is skipped there).

Callables (``AppendFeatures(f=...)``, the ``encoders`` of
``NumericToCategoricalEncoding``, perturbation functions) are evaluated on
torch tensors through ``torch``; they must act pointwise.
"""

from __future__ import annotations

import copy
import functools
import math
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, ItemsView, Iterator, KeysView, ValuesView
from typing import Any, NamedTuple

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    first_kind,
    from_flat,
    from_pair,
    kind_of,
    to_flat,
)
from botorch_rs._buffers import NativeBufferModule
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import UserInputWarning
from botorch_rs.models.transforms import _torch_input
from botorch_rs.models.utils.assorted import fantasize as _fantasize


class _Step(NamedTuple):
    """A step of a model's chain: the extension object, whether its
    transform was in train mode when the model was built, its
    parameter-name prefix and the Python transform that owns it."""

    rs: Any
    training: bool
    prefix: str
    owner: Any


def _indices_list(indices: Any, name: str = "indices") -> list[int] | None:
    if indices is None:
        return None
    if isinstance(indices, slice):
        return None
    if _torch_input.is_tensor(indices):
        return [int(v) for v in indices.reshape(-1)]
    data, shape = to_flat(indices, name)
    return [int(v) for v in data]


def _check_flat_matrix(x: Any, name: str = "X") -> tuple[Any, tuple[int, ...]]:
    data, shape = to_flat(x, name)
    if len(shape) < 2:
        raise BotorchTensorDimensionError(
            f"`{name}` must have at least 2 dimensions, but has {len(shape)}."
        )
    return data, tuple(shape)


class InputTransform(NativeBufferModule):
    r"""Abstract base class for input transforms.

    Properties:
        is_one_to_many: Whether the transform produces several values for each
            input.
        transform_on_train: Whether to apply the transform in train mode.
        transform_on_eval: Whether to apply the transform in eval mode.
        transform_on_fantasize: Whether to apply the transform when called
            within a ``fantasize`` call.
    """

    training: bool = True
    is_one_to_many: bool = False
    transform_on_train: bool = True
    transform_on_eval: bool = True
    transform_on_fantasize: bool = True

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if sys.modules.get("torch") is not None:
            from botorch_rs.models.transforms._torch_input_protocol import allocate

            return allocate(cls)
        return object.__new__(cls)

    def __getattr__(self, name: str) -> Any:
        # An object created in an array-only process can acquire the optional
        # module protocol later without replacing aliases held by its caller.
        protocols = {
            "to",
            "cpu",
            "cuda",
            "double",
            "float",
            "half",
            "bfloat16",
            "type",
            "state_dict",
            "load_state_dict",
            "children",
            "named_children",
            "modules",
            "named_modules",
            "get_buffer",
            "get_parameter",
            "get_submodule",
            "set_submodule",
            "requires_grad_",
            "zero_grad",
            "register_parameter",
            "register_forward_hook",
            "register_forward_pre_hook",
            "register_full_backward_hook",
            "register_full_backward_pre_hook",
            "register_state_dict_pre_hook",
            "register_state_dict_post_hook",
            "register_load_state_dict_pre_hook",
            "register_load_state_dict_post_hook",
        }
        if name in protocols:
            from botorch_rs.models.transforms._torch_input_protocol import TorchInputModule, promote

            if not isinstance(self, TorchInputModule):
                return getattr(promote(self), name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getstate__(self) -> dict[str, Any]:
        from botorch_rs.models.transforms._input_pickle import get_state

        return get_state(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        from botorch_rs.models.transforms._input_pickle import set_state

        set_state(self, state)

    def train(self, mode: bool = True) -> InputTransform:
        self.training = mode
        return self

    def eval(self) -> InputTransform:
        return self.train(False)

    def forward(self, X: Any) -> Any:
        r"""Transform the inputs to a model (BoTorch's ``forward``): in train
        mode if ``transform_on_train``, in eval mode if ``transform_on_eval``
        (and, inside ``fantasize``, ``transform_on_fantasize``)."""
        if self.training:
            if self.transform_on_train:
                return self.transform(X)
        elif self.transform_on_eval:
            if _fantasize.off() or self.transform_on_fantasize:
                return self.transform(X)
        return X

    def __call__(self, X: Any) -> Any:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(self, torch.nn.Module):
            result = torch.nn.Module.__call__(self, X)
            self._refresh_affine_buffers()
            return result
        return self.forward(X)

    def parameters(self, recurse: bool = True) -> Iterator[Any]:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Any]]:
        seen: set[int] = set()
        for name, parameter in self.__dict__.get("_direct_parameters", {}).items():
            seen.add(id(parameter))
            yield f"{prefix}.{name}" if prefix else name, parameter
        if recurse:
            children = self.__dict__.get("_transforms", self.__dict__.get("transforms", {}))
            items = children.items() if isinstance(children, dict) else enumerate(children)
            for name, child in items:
                for child_name, parameter in child.named_parameters():
                    if not remove_duplicate or id(parameter) not in seen:
                        seen.add(id(parameter))
                        key = f"{name}.{child_name}"
                        yield f"{prefix}.{key}" if prefix else key, parameter

    def transform(self, X: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def untransform(self, X: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method."
        )

    def preprocess_transform(self, X: Any) -> Any:
        r"""Apply the transform for preprocessing inputs (e.g. the training
        inputs of a model in eval mode): ``transform(X)`` if
        ``transform_on_train``, without learning coefficients."""
        if self.transform_on_train:
            learn = getattr(self, "learn_coefficients", None)
            if learn is not None:
                self.learn_coefficients = False
                try:
                    return self.transform(X)
                finally:
                    self.learn_coefficients = learn
            return self.transform(X)
        return X

    def _state(self) -> Any:
        """The configuration and state compared by ``equals``."""
        return None

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent (same type, flags,
        configuration and state)."""
        return (
            type(self) is type(other)
            and self.transform_on_train == other.transform_on_train
            and self.transform_on_eval == other.transform_on_eval
            and self.transform_on_fantasize == other.transform_on_fantasize
            and _close(self._state(), other._state())
        )

    # -- models ----------------------------------------------------------

    def _flags(self) -> tuple[bool, bool, bool]:
        return (
            bool(self.transform_on_train),
            bool(self.transform_on_eval),
            bool(self.transform_on_fantasize),
        )

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        """The steps of a model's chain for ``d``-dimensional inputs (for
        batch element ``batch`` of a per-batch transform)."""
        raise UnsupportedError(
            f"{type(self).__name__} cannot be used as the input transform of a botorch_rs model."
        )

    def _per_batch(self) -> tuple[int, ...] | None:
        """The batch shape of a transform with one state per batch element."""
        return None

    def _batch_params(self) -> list[list[float]] | None:
        """The learned parameters of every batch element of a transform whose
        steps are shared by the batch elements but whose parameters are not
        (a ``Warp`` with ``batch_shape``)."""
        return None

    def _set_state(self, state: Any, batch: int | None, kind: ArrayKind) -> None:
        """Receives the fitted state of its step (``batch``: the batch
        element of a per-batch transform)."""

    def _out_dim(self, d: int) -> int:
        return d

    def _bind(self, model: Any, prefix: str) -> None:
        """Binds learned parameters to a model."""


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-8) or (math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    return bool(a == b)


class ReversibleInputTransform(InputTransform):
    r"""An abstract class for a reversible input transform (``reverse``
    swaps ``transform`` and ``untransform``)."""

    reverse: bool = False

    def transform(self, X: Any) -> Any:
        return self._untransform(X) if self.reverse else self._transform(X)

    def untransform(self, X: Any) -> Any:
        return self._transform(X) if self.reverse else self._untransform(X)

    def _transform(self, X: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _untransform(self, X: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def equals(self, other: InputTransform) -> bool:
        return super().equals(other) and self.reverse == getattr(other, "reverse", None)


# -----------------------------------------------------------------------------
# Affine transforms
# -----------------------------------------------------------------------------


def _row(values: Any, name: str) -> tuple[list[float], tuple[int, ...]]:
    data, shape = to_flat(values, name)
    return [float(v) for v in data], tuple(shape)


class AffineInputTransform(ReversibleInputTransform):
    r"""Apply an affine transformation to the inputs,
    ``output = (input - offset) / coefficient``, of the dimensions
    ``indices`` (all by default). With ``batch_shape`` (or coefficients with
    batch dimensions) every batch element has its own coefficients."""

    _kind_name = "fixed"

    def __init__(
        self,
        d: int,
        coefficient: Any,
        offset: Any,
        indices: Any = None,
        batch_shape: Any = (),
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
    ) -> None:
        idx = _indices_list(indices)
        if idx is not None:
            if len(idx) == 0:
                raise ValueError("`indices` list is empty!")
            if len(idx) > d:
                raise ValueError("Can provide at most `d` indices!")
            if any(i > d - 1 for i in idx):
                raise ValueError("Elements of `indices` have to be smaller than `d`!")
            if len(set(idx)) != len(idx):
                raise ValueError("Elements of `indices` tensor must be unique!")
            idx = [i + d if i < 0 else i for i in idx]
            if any(i < 0 for i in idx):
                raise IndexError(f"indices {idx} are out of range for d = {d}.")
        self._d = int(d)
        self.indices: list[int] | None = idx
        self._idx = idx
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.reverse = bool(reverse)
        self.training = True
        k = d if idx is None else len(idx)
        cshape: tuple[int, ...]
        oshape: tuple[int, ...]
        if _torch_input.is_tensor(coefficient) or _torch_input.is_tensor(offset):
            like = coefficient if _torch_input.is_tensor(coefficient) else offset
            self._coefficient = _torch_input._tensor(coefficient, like)
            self._offset = _torch_input._tensor(offset, like)
            coef, cshape = [1.0] * k, (k,)
            off, oshape = [0.0] * k, (k,)
        else:
            coef, cshape = _row(coefficient, "coefficient")
            off, oshape = _row(offset, "offset")
        self._kind = first_kind(coefficient, offset)
        bshape = tuple(int(b) for b in (batch_shape or ()))
        nb = math.prod(bshape)
        self.batch_shape = bshape
        self._coefs = self._expand(coef, cshape, k, nb, "coefficient")
        self._offs = self._expand(off, oshape, k, nb, "offset")
        if not hasattr(self, "_learn"):
            self._learn = False
        self._steps = [self._make_step(b) for b in range(nb)]

    @staticmethod
    def _expand(
        values: list[float], shape: tuple[int, ...], k: int, nb: int, name: str
    ) -> list[list[float]]:
        if len(values) == k:
            return [list(values) for _ in range(nb)]
        if len(values) == 1:
            return [[values[0]] * k for _ in range(nb)]
        if len(values) == nb * k:
            return [list(values[b * k : (b + 1) * k]) for b in range(nb)]
        raise BotorchTensorDimensionError(
            f"`{name}` of shape {shape} is not broadcastable with the transformed dimension {k}."
        )

    def _kind_args(self) -> dict[str, Any]:
        return {}

    def _make_step(self, b: int) -> Any:
        return _rs.InputStep.affine(
            self._d,
            self._idx,
            self._offs[b],
            self._coefs[b],
            self._kind_name,
            self._learn,
            self.reverse,
            self._flags(),
            **self._kind_args(),
        )

    @property
    def learn_coefficients(self) -> bool:
        return getattr(self, "_learn", False)

    @learn_coefficients.setter
    def learn_coefficients(self, value: bool) -> None:
        self._learn = bool(value)
        if hasattr(self, "_steps"):
            self._steps = [self._make_step_from(s, b) for b, s in enumerate(self._steps)]

    def _make_step_from(self, step: Any, b: int) -> Any:
        off, coef = step.affine_state
        self._offs[b] = list(off)
        self._coefs[b] = list(coef)
        return self._make_step(b)

    def _stat(self, which: int) -> Any:
        tensor = _torch_input.affine_stat(self, which)
        if tensor is not None:
            return tensor
        k = len(self._coefs[0])
        data: list[float] = []
        for s in self._steps:
            data.extend(s.affine_state[which])
        return from_flat(data, (*self.batch_shape, 1, k), self._kind)

    @property
    def coefficient(self) -> Any:
        r"""The linear coefficients, shape ``batch_shape x 1 x k``."""
        return self._stat(1)

    @property
    def offset(self) -> Any:
        r"""The offsets, shape ``batch_shape x 1 x k``."""
        return self._stat(0)

    def _check_shape(self, shape: tuple[int, ...]) -> None:
        if shape[-1] != self._d:
            raise BotorchTensorDimensionError(
                f"Wrong input dimension. Received {shape[-1]}, expected {self._d}."
            )
        n = len(self.batch_shape) + 2
        if self.training and len(shape) < n:
            raise ValueError(
                f"`X` must have at least {n} dimensions, {n - 2} batch and 2 innate , but has "
                f"{len(shape)}."
            )

    def _apply(self, X: Any, inverse: bool) -> Any:
        if _torch_input.is_tensor(X):
            return _torch_input.affine_apply(self, X, inverse)
        _torch_input.sync_affine_native(self)
        kind = kind_of(X)
        data, shape = _check_flat_matrix(X)
        self._check_shape(shape)
        if self._kind.lib == "list" and kind.lib != "list":
            self._kind = kind
        # BoTorch learns the coefficients in `_transform` (the forward map of
        # a transform without `reverse`, the inverse of one with it).
        learn = inverse == self.reverse and self.learn_coefficients and self.training
        if not self.batch_shape:
            if learn:
                self._steps = [self._steps[0].fit((data, shape))]
                _torch_input.clear_affine_tensor_state(self)
            step = self._steps[0]
            out = step.untransform((data, shape)) if inverse else step.transform((data, shape))
            return from_pair(out, kind)
        return from_pair(self._apply_batched(data, shape, inverse, learn), kind)

    def _apply_batched(
        self, data: Any, shape: tuple[int, ...], inverse: bool, learn: bool
    ) -> tuple[list[float], tuple[int, ...]]:
        k = len(self.batch_shape)
        if tuple(shape[len(shape) - 2 - k : -2]) != self.batch_shape:
            raise BotorchTensorDimensionError(
                f"X with shape {shape} does not have the transform's batch shape "
                f"{self.batch_shape} as trailing batch dimensions."
            )
        nb = math.prod(self.batch_shape)
        n, d = shape[-2], shape[-1]
        block = n * d
        lead = len(data) // (nb * block) if nb * block else 0
        values = list(data)
        if learn:
            fitted = []
            for b in range(nb):
                pts: list[float] = []
                for o in range(lead):
                    s = (o * nb + b) * block
                    pts.extend(values[s : s + block])
                fitted.append(self._steps[b].fit((pts, (lead * n, d))))
            self._steps = fitted
            _torch_input.clear_affine_tensor_state(self)
        out: list[float] = [0.0] * len(values)
        for b in range(nb):
            pts = []
            for o in range(lead):
                s = (o * nb + b) * block
                pts.extend(values[s : s + block])
            step = self._steps[b]
            res = (
                step.untransform((pts, (lead, n, d)))
                if inverse
                else step.transform((pts, (lead, n, d)))
            )
            rd = list(res[0])
            for o in range(lead):
                s = (o * nb + b) * block
                out[s : s + block] = rd[o * block : (o + 1) * block]
        return out, shape

    def _transform(self, X: Any) -> Any:
        return self._apply(X, inverse=self.reverse)

    def _untransform(self, X: Any) -> Any:
        return self._apply(X, inverse=not self.reverse)

    def transform(self, X: Any) -> Any:
        # The native step applies ``reverse`` itself.
        return self._apply(X, inverse=False)

    def untransform(self, X: Any) -> Any:
        return self._apply(X, inverse=True)

    def _state(self) -> Any:
        _torch_input.sync_affine_native(self)
        return (
            self._d,
            self._idx,
            [list(s.affine_state[0]) for s in self._steps],
            [list(s.affine_state[1]) for s in self._steps],
            self.learn_coefficients,
            self.reverse,
        )

    def _per_batch(self) -> tuple[int, ...] | None:
        return self.batch_shape or None

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        _torch_input.sync_affine_native(self)
        if d != self._d:
            raise BotorchTensorDimensionError(
                f"Wrong input dimension. Received {d}, expected {self._d}."
            )
        step = self._steps[0 if batch is None else batch]
        return [_Step(step, self.training, prefix, self)]

    def _set_state(self, state: Any, batch: int | None, kind: ArrayKind) -> None:
        _torch_input.clear_affine_tensor_state(self)
        self._steps[0 if batch is None else batch] = state
        if self._kind.lib == "list":
            self._kind = kind


class Normalize(AffineInputTransform):
    r"""Normalize the inputs to have unit range and be centered at 0.5 (by
    default).

    Without explicit ``bounds`` (or with ``learn_bounds=True``) the bounds are
    learned from the inputs in train mode (and from the training inputs when
    a model is built). With a ``batch_shape`` (or bounds of shape
    ``batch_shape x 2 x d``) every batch element has its own bounds;
    otherwise one normalization is learned from all batch elements. With
    ``indices``, only those inputs are normalized (``bounds``, ``offset`` and
    ``coefficient`` then have ``len(indices)`` columns, as in BoTorch).
    """

    def __init__(
        self,
        d: int,
        indices: Any = None,
        bounds: Any = None,
        batch_shape: Any = (),
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_range: float = 1e-8,
        learn_bounds: bool | None = None,
        almost_zero: float = 1e-12,
        center: float = 0.5,
    ) -> None:
        learn = (bounds is None) if learn_bounds is None else bool(learn_bounds)
        idx = _indices_list(indices)
        k = d if idx is None else len(idx)
        self.min_range = float(min_range)
        self.center = float(center)
        self.almost_zero = almost_zero
        shape = tuple(int(b) for b in (batch_shape or ()))
        native_bounds: Any = None
        coefficient: Any
        offset: Any
        if idx is not None:
            idx = [i + d if i < 0 else i for i in idx]
        if _torch_input.is_tensor(bounds):
            selected = bounds if idx is None or bounds.shape[-1] != d else bounds[..., idx]
            if selected.ndim < 2 or selected.shape[-2:] != (2, k):
                raise BotorchTensorDimensionError(
                    f"Dimensions of provided `bounds` are incompatible with transform_dimension = {k}!"
                )
            if selected.ndim > 2:
                shape = tuple(selected.shape[:-2])
            coefficient = (selected[..., 1, :] - selected[..., 0, :]).unsqueeze(-2)
            offset = selected[..., :1, :] + (0.5 - center) * coefficient
        elif bounds is not None:
            bdata, bshape = to_flat(bounds, "bounds")
            if idx is not None and bshape[-1] == d:
                # bounds[..., indices]
                cols = bshape[-1]
                rows = len(bdata) // cols
                bdata = [bdata[r * cols + i] for r in range(rows) for i in idx]
                bshape = (*bshape[:-1], len(idx))
            if bshape[-1] != k:
                raise BotorchTensorDimensionError(
                    "Dimensions of provided `bounds` are incompatible with "
                    f"transform_dimension = {k}!"
                )
            if len(bshape) > 2:
                shape = tuple(bshape[:-2])
            native_bounds = (list(bdata), tuple(bshape))
            nb = math.prod(shape)
            offs: list[float] = []
            coefs: list[float] = []
            for b in range(nb):
                lo = list(bdata[b * 2 * k : b * 2 * k + k])
                hi = list(bdata[b * 2 * k + k : (b + 1) * 2 * k])
                c = [h - lb for lb, h in zip(lo, hi, strict=True)]
                coefs.extend(c)
                offs.extend(lb + (0.5 - center) * ci for lb, ci in zip(lo, c, strict=True))
            coefficient = coefs
            offset = offs
        else:
            coefficient = [1.0] * k
            offset = [0.0] * k
            if not learn:
                warnings.warn(
                    "learn_bounds is False and no bounds were provided. The bounds "
                    "will not be updated and the transform will be a no-op.",
                    UserInputWarning,
                    stacklevel=2,
                )
        self._learn = learn
        # The plain configuration uses the original native Normalize (all
        # dimensions, center 0.5, all flags, no reverse).
        self._plain = (
            idx is None
            and center == 0.5
            and not reverse
            and transform_on_train
            and transform_on_eval
            and transform_on_fantasize
        )
        super().__init__(
            d=d,
            coefficient=coefficient,
            offset=offset,
            indices=indices,
            batch_shape=shape,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            reverse=reverse,
        )
        self._kind = first_kind(bounds)
        # The original native Normalize (of the `k` transformed dimensions):
        # the transform of the plain configuration in all models, and the one
        # a MultiTaskGP uses (also with `indices`, phase 7a).
        self._native = (
            center == 0.5
            and not reverse
            and transform_on_train
            and transform_on_eval
            and transform_on_fantasize
        )
        self._rs: Any = None
        if self._native:
            self._rs = _rs.Normalize(
                k,
                native_bounds,
                learn,
                min_range,
                list(shape),
            )

    def _kind_args(self) -> dict[str, Any]:
        return {"min_range": self.min_range, "center": self.center}

    @property
    def _rs(self) -> Any:
        # Some native models consume the plain descriptor directly rather
        # than chain_spec. Export only at that explicit native boundary.
        _torch_input.sync_affine_native(self)
        return self._native_rs

    @_rs.setter
    def _rs(self, value: Any) -> None:
        self._native_rs = value

    _kind_name = "normalize"

    # -- state -------------------------------------------------------------

    def _set_fitted(self, fitted: Any, kind: ArrayKind) -> None:
        """Receives the native ``Normalize`` fitted by a model (the plain
        configuration, or a ``MultiTaskGP``'s with ``indices``)."""
        _torch_input.clear_affine_tensor_state(self)
        self._rs = fitted
        if kind.lib != "list" or self._kind.lib == "list":
            self._kind = kind
        if not self._plain:
            # The general steps follow the model's fit.
            nb = len(self._steps)
            k = len(self._coefs[0])
            off, coef = fitted.offset[0], fitted.coefficient[0]
            for b in range(nb):
                self._offs[b] = [float(v) for v in off[b * k : (b + 1) * k]]
                self._coefs[b] = [float(v) for v in coef[b * k : (b + 1) * k]]
            self._steps = [self._make_step(b) for b in range(nb)]

    @property
    def learn_bounds(self) -> bool:
        return self.learn_coefficients

    @property
    def bounds(self) -> Any:
        r"""The normalization bounds, shape ``batch_shape x 2 x d``."""
        tensor = _torch_input.affine_stat(self, 0)
        if tensor is not None:
            import torch

            return torch.cat((tensor, tensor + self.coefficient), dim=-2)
        if self._plain:
            return from_pair(self._rs.bounds, self._kind)
        off = to_flat(self.offset, "offset")[0]
        coef = to_flat(self.coefficient, "coefficient")[0]
        k = len(self._coefs[0])
        data: list[float] = []
        for b in range(len(self._steps)):
            o = list(off[b * k : (b + 1) * k])
            c = list(coef[b * k : (b + 1) * k])
            data.extend(o)
            data.extend(oi + ci for oi, ci in zip(o, c, strict=True))
        return from_flat(data, (*self.batch_shape, 2, k), self._kind)

    @property
    def offset(self) -> Any:
        tensor = _torch_input.affine_stat(self, 0)
        if tensor is not None:
            return tensor
        if self._plain:
            return from_pair(self._rs.offset, self._kind)
        return super().offset

    @property
    def coefficient(self) -> Any:
        tensor = _torch_input.affine_stat(self, 1)
        if tensor is not None:
            return tensor
        if self._plain:
            return from_pair(self._rs.coefficient, self._kind)
        return super().coefficient

    @property
    def mins(self) -> Any:
        return self.offset

    @property
    def ranges(self) -> Any:
        return self.coefficient

    def _apply(self, X: Any, inverse: bool) -> Any:
        if _torch_input.is_tensor(X):
            return _torch_input.affine_apply(self, X, inverse)
        _torch_input.sync_affine_native(self)
        if not self._plain:
            return super()._apply(X, inverse)
        kind = first_kind(X)
        flat = to_flat(X, "X")
        if not inverse and self.training and self.learn_bounds:
            self._set_fitted(self._rs.fit(flat), kind if kind.lib != "list" else self._kind)
        out = self._rs.untransform(flat) if inverse else self._rs.transform(flat)
        return from_pair(out, kind)

    def forward(self, X: Any) -> Any:
        if _torch_input.is_tensor(X):
            return super().forward(X)
        if self._plain:
            return self.transform(X)
        return super().forward(X)

    __call__ = InputTransform.__call__

    @property
    def learn_coefficients(self) -> bool:
        return getattr(self, "_learn", False)

    @learn_coefficients.setter
    def learn_coefficients(self, value: bool) -> None:
        self._learn = bool(value)
        if getattr(self, "_plain", False):
            # The native Normalize keeps its bounds; `_apply` checks the flag.
            return
        if hasattr(self, "_steps"):
            self._steps = [self._make_step_from(s, b) for b, s in enumerate(self._steps)]

    def preprocess_transform(self, X: Any) -> Any:
        if _torch_input.is_tensor(X):
            return super().preprocess_transform(X)
        if self._plain:
            if not self.transform_on_train:
                return X
            flat = to_flat(X, "X")
            return from_pair(self._rs.transform(flat), first_kind(X))
        return super().preprocess_transform(X)

    def _state(self) -> Any:
        if self._plain:
            return (to_flat(self.bounds)[0], self.learn_bounds)
        return super()._state()

    def get_init_args(self) -> dict[str, Any]:
        r"""The arguments that construct an exact copy of the transform (the
        un-shifted bounds and ``center``, as with meta-pytorch/botorch#3389)."""
        bounds = self.bounds
        if self.center != 0.5:
            if _torch_input.is_tensor(bounds):
                bounds = bounds - (0.5 - self.center) * self.coefficient
                return {**self._init_args(bounds), "bounds": bounds}
            data, shape = to_flat(bounds, "bounds")
            coef = to_flat(self.coefficient, "coefficient")[0]
            k = shape[-1]
            shift = [(0.5 - self.center) * c for c in coef]
            out: list[float] = []
            for r in range(len(data) // k):
                b = (r // 2) * k
                out.extend(data[r * k + j] - shift[b + j] for j in range(k))
            bounds = from_flat(out, shape, self._kind)
        return self._init_args(bounds)

    def _init_args(self, bounds: Any) -> dict[str, Any]:
        return {
            "d": self._d,
            "indices": self.indices,
            "bounds": bounds,
            "batch_shape": self.batch_shape,
            "transform_on_train": self.transform_on_train,
            "transform_on_eval": self.transform_on_eval,
            "transform_on_fantasize": self.transform_on_fantasize,
            "reverse": self.reverse,
            "min_range": self.min_range,
            "learn_bounds": self.learn_bounds,
            "center": self.center,
        }

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        _torch_input.sync_affine_native(self)
        if self._plain:
            if self.batch_shape:
                raise UnsupportedError(
                    "A Normalize with a batch shape is not supported in a chained or one-to-many "
                    "input transform; use Normalize(d) (bounds shared by all batch elements)."
                )
            if d != self._d:
                raise BotorchTensorDimensionError(
                    f"Wrong input dimension. Received {d}, expected {self._d}."
                )
            return [_Step(self._rs, self.training, prefix, self)]
        return super()._model_steps(d, prefix, batch)

    def _set_state(self, state: Any, batch: int | None, kind: ArrayKind) -> None:
        if self._plain:
            self._set_fitted(state, kind)
            return
        super()._set_state(state, batch, kind)

    def __repr__(self) -> str:
        extra = "" if self.indices is None else f", indices={self.indices}"
        return f"Normalize(d={self._d}{extra}, learn_bounds={self.learn_bounds})"


class InputStandardize(AffineInputTransform):
    r"""Standardize the inputs (zero mean, unit variance), with the means and
    standard deviations learned in train mode. Dimensions whose standard
    deviation is below ``min_std`` (or undefined, for a single point) are not
    standardized."""

    _kind_name = "standardize"

    def __init__(
        self,
        d: int,
        indices: Any = None,
        batch_shape: Any = (),
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_std: float = 1e-8,
    ) -> None:
        idx = _indices_list(indices)
        k = d if idx is None else len(idx)
        self.min_std = float(min_std)
        self._learn = True
        super().__init__(
            d=d,
            coefficient=[1.0] * k,
            offset=[0.0] * k,
            indices=indices,
            batch_shape=batch_shape,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            reverse=reverse,
        )

    def _kind_args(self) -> dict[str, Any]:
        return {"min_std": self.min_std}

    @property
    def stds(self) -> Any:
        return self.coefficient

    @property
    def means(self) -> Any:
        return self.offset

    def __repr__(self) -> str:
        return f"InputStandardize(d={self._d})"


# -----------------------------------------------------------------------------
# Round, Log10, Warp
# -----------------------------------------------------------------------------


def _categorical_pairs(categorical_features: Any) -> list[tuple[int, int]]:
    return [(int(k), int(v)) for k, v in (categorical_features or {}).items()]


class Round(InputTransform):
    r"""A discretization transform for discrete inputs: rounds the
    ``integer_indices`` (half to even, like ``torch.round``, with
    straight-through gradients; or the differentiable ``approximate_round``
    with temperature ``tau``) and maps one-hot encoded categoricals
    (``categorical_features``: start index to cardinality) to the category
    with the largest value (straight-through gradients)."""

    def __init__(
        self,
        integer_indices: Any = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        approximate: bool = False,
        tau: float = 1e-3,
    ) -> None:
        if approximate and categorical_features is not None:
            raise NotImplementedError
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.integer_indices = _indices_list(integer_indices, "integer_indices") or []
        self.categorical_features = dict(categorical_features or {})
        self.approximate = bool(approximate)
        self.tau = float(tau)
        self.training = True

    def _step(self) -> Any:
        return _rs.InputStep.round(
            list(self.integer_indices),
            _categorical_pairs(self.categorical_features),
            self.approximate,
            self.tau,
            self._flags(),
        )

    def transform(self, X: Any) -> Any:
        r"""Discretize the inputs (``batch_shape x n x d``)."""
        if _torch_input.is_tensor(X):
            return _torch_input.round_apply(self, X)
        kind = kind_of(X)
        flat = _check_flat_matrix(X)
        return from_pair(self._step().transform(flat), kind)

    def _state(self) -> Any:
        return (
            list(self.integer_indices),
            self.categorical_features,
            self.approximate,
            self.tau,
        )

    def get_init_args(self) -> dict[str, Any]:
        r"""The arguments that construct an exact copy of the transform."""
        return {
            "integer_indices": list(self.integer_indices),
            "categorical_features": self.categorical_features,
            "transform_on_train": self.transform_on_train,
            "transform_on_eval": self.transform_on_eval,
            "transform_on_fantasize": self.transform_on_fantasize,
            "approximate": self.approximate,
            "tau": self.tau,
        }

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        return [_Step(self._step(), self.training, prefix, self)]

    def __repr__(self) -> str:
        return (
            f"Round(integer_indices={list(self.integer_indices)}, "
            f"categorical_features={self.categorical_features}, approximate={self.approximate})"
        )


class Log10(ReversibleInputTransform):
    r"""A base-10 log transformation of the dimensions ``indices``."""

    def __init__(
        self,
        indices: list[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
    ) -> None:
        self.indices = _indices_list(indices) or []
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.reverse = bool(reverse)
        self.training = True

    def _step(self, reverse: bool) -> Any:
        return _rs.InputStep.log10(list(self.indices), reverse, self._flags())

    def _transform(self, X: Any) -> Any:
        if _torch_input.is_tensor(X):
            return _torch_input.log10_apply(self, X, False)
        return from_pair(self._step(False).transform(_check_flat_matrix(X)), kind_of(X))

    def _untransform(self, X: Any) -> Any:
        if _torch_input.is_tensor(X):
            return _torch_input.log10_apply(self, X, True)
        return from_pair(self._step(False).untransform(_check_flat_matrix(X)), kind_of(X))

    def _state(self) -> Any:
        return list(self.indices)

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        return [_Step(self._step(self.reverse), self.training, prefix, self)]


class Warp(ReversibleInputTransform):
    r"""A transform that uses learned input warping functions: each of the
    dimensions ``indices`` is normalized to the unit cube (by ``bounds``, or
    bounds learned from the training inputs) and warped with the CDF of a
    Kumaraswamy distribution, ``1 - (1 - x^c1)^c0``, whose concentrations
    (``concentration0``, ``concentration1``, initially 1, constrained to be
    at least 1e-4) are fitted jointly with the GP hyperparameters by
    ``fit_gpytorch_mll`` (with the priors ``concentration0_prior`` /
    ``concentration1_prior`` if given). With ``batch_shape`` (the input
    batch shape of a single-output model) every batch element has its own
    concentrations. As in BoTorch, models with several outputs should use a
    ``ModelListGP`` to learn independent warping functions."""

    _min_concentration_level = 1e-4

    def __init__(
        self,
        d: int,
        indices: list[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        eps: float = 1e-7,
        concentration1_prior: Any = None,
        concentration0_prior: Any = None,
        batch_shape: Any = None,
        bounds: Any = None,
    ) -> None:
        from botorch_rs.priors import Prior

        idx = _indices_list(indices) or []
        self.indices = idx
        self._d = int(d)
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.reverse = bool(reverse)
        self.batch_shape = tuple(int(b) for b in (batch_shape or ()))
        self._eps = float(eps)
        for name, p in (
            ("concentration0_prior", concentration0_prior),
            ("concentration1_prior", concentration1_prior),
        ):
            torch = sys.modules.get("torch")
            tensor_prior = (
                torch is not None
                and isinstance(p, torch.nn.Module)
                and callable(getattr(p, "log_prob", None))
            )
            if p is not None and not isinstance(p, Prior) and not tensor_prior:
                raise UnsupportedError(f"`{name}` must be a botorch_rs or Torch module prior.")
        self.concentration0_prior = concentration0_prior
        self.concentration1_prior = concentration1_prior
        self._normalize = Normalize(d=d, indices=idx, bounds=bounds)
        self._bounds = None
        if bounds is not None and not _torch_input.is_tensor(bounds):
            data, shape = to_flat(bounds, "bounds")
            cols = shape[-1]
            lo = [float(v) for v in data[:cols]]
            hi = [float(v) for v in data[cols : 2 * cols]]
            self._bounds = (lo, hi)
        self._kind = first_kind(bounds)
        self.training = True
        nb = math.prod(self.batch_shape)
        self._steps = [self._new_step() for _ in range(nb)]
        self._model: Any = None
        self._prefix = ""

    def _new_step(self) -> Any:
        return _rs.InputStep.warp(
            self._d,
            list(self.indices),
            self._eps,
            self._bounds,
            getattr(self.concentration0_prior, "_rs", None),
            getattr(self.concentration1_prior, "_rs", None),
            self.reverse,
            self._flags(),
            batched=bool(self.batch_shape),
        )

    # -- concentrations ------------------------------------------------------

    def _concentrations(self) -> list[list[float]]:
        if self._model is not None:
            k = len(self.indices)
            c0 = list(self._model._raw_param(self._prefix + "concentration0"))
            c1 = list(self._model._raw_param(self._prefix + "concentration1"))
            nb = len(c0) // k
            return [c0[b * k : (b + 1) * k] + c1[b * k : (b + 1) * k] for b in range(nb)]
        result = [list(s.params) for s in self._steps]
        parameters = self.__dict__.get("_direct_parameters", {})
        k = len(self.indices)
        for i in (0, 1):
            name = f"concentration{i}"
            parameter = parameters.get(name)
            if parameter is not None:
                values = _torch_input.native_warp_values(self, name)
                if values is None:
                    values, _ = to_flat(parameter)
                for b, row in enumerate(result):
                    row[i * k : (i + 1) * k] = values[b * k : (b + 1) * k]
        return result

    def _param_shape(self) -> tuple[int, ...]:
        k = len(self.indices)
        return (*self.batch_shape, 1, k) if self.batch_shape else (k,)

    def _get(self, i: int) -> Any:
        parameter = self.__dict__.get("_direct_parameters", {}).get(f"concentration{i}")
        if self._model is not None:
            name = self._prefix + f"concentration{i}"
            getter = getattr(self._model, "_raw_param_array", None)
            value = (
                getter(name)
                if getter is not None
                else from_flat(list(self._model._raw_param(name)), self._param_shape(), self._kind)
            )
            if parameter is not None and value is not parameter:
                import torch

                # A native fit or MLL trial updates the bound owner. Refresh
                # the caller's existing leaf instead of exposing old values
                # or replacing a Parameter held by its optimizer.
                with torch.no_grad():
                    current = _torch_input._tensor(value, parameter).reshape_as(parameter)
                    if not torch.equal(parameter, current):
                        parameter.copy_(current)
                return parameter
            return value
        if parameter is not None:
            return parameter
        k = len(self.indices)
        data: list[float] = []
        rows = (
            [list(step.params) for step in self._steps]
            if self.__dict__.get("_direct_parameters")
            else self._concentrations()
        )
        for c in rows:
            data.extend(c[i * k : (i + 1) * k])
        return from_flat(data, self._param_shape(), self._kind)

    def _set(self, i: int, value: Any) -> None:
        if self._model is not None:
            self._model._set_raw_param(self._prefix + f"concentration{i}", value)
            return
        parameters = self.__dict__.get("_direct_parameters", {})
        if _torch_input.is_tensor(value) or parameters:
            import torch

            if isinstance(value, torch.nn.Parameter):
                if tuple(value.shape) != self._param_shape():
                    raise ValueError(f"concentration{i} must have shape {self._param_shape()}.")
                self.__dict__.setdefault("_direct_parameters", {})[f"concentration{i}"] = value
            else:
                like = next(iter(parameters.values())) if parameters else value
                parameters = _torch_input.warp_parameters(self, like)
                with torch.no_grad():
                    parameters[f"concentration{i}"].copy_(_torch_input._tensor(value, like))
                name = f"concentration{i}"
                if (
                    self.__dict__.get("_native_default_parameters", {}).get(name)
                    is parameters[name]
                ):
                    from botorch_rs._arrays import broadcast_to

                    data, shape = to_flat(value, name)
                    values = broadcast_to(list(data), tuple(shape), self._param_shape())
                    self.__dict__.setdefault("_native_warp_values", {})[name] = (
                        values,
                        parameters[name],
                    )
            return
        from botorch_rs._arrays import broadcast_to

        data, shape = to_flat(value, f"concentration{i}")
        k = len(self.indices)
        target = self._param_shape()
        vals = broadcast_to(list(data), tuple(shape), target)
        steps = []
        for b, s in enumerate(self._steps):
            p = list(s.params)
            p[i * k : (i + 1) * k] = vals[b * k : (b + 1) * k]
            steps.append(s.with_params(p))
        self._steps = steps

    @property
    def concentration0(self) -> Any:
        r"""The ``concentration0`` parameters, shape ``k`` (``batch_shape x 1 x
        k`` with a batch shape)."""
        return self._get(0)

    @concentration0.setter
    def concentration0(self, value: Any) -> None:
        self._set(0, value)

    @property
    def concentration1(self) -> Any:
        r"""The ``concentration1`` parameters."""
        return self._get(1)

    @concentration1.setter
    def concentration1(self, value: Any) -> None:
        self._set(1, value)

    def _set_concentration(self, i: int, value: Any) -> None:
        self._set(i, value)

    def initialize(self, **kwargs: Any) -> Warp:
        for name, value in kwargs.items():
            if name not in ("concentration0", "concentration1"):
                raise AttributeError(f"Unknown parameter {name!r}.")
            setattr(self, name, value)
        return self

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Any]]:
        for i in (0, 1):
            name = f"concentration{i}"
            yield f"{prefix}.{name}" if prefix else name, self._get(i)

    # -- application ---------------------------------------------------------

    def _current_steps(self) -> list[Any]:
        if _torch_input.is_tensor(getattr(self._normalize, "_coefficient", None)):
            bounds = self._bounds
            if not self._normalize.learn_bounds or not self.training:
                values, shape = to_flat(self._normalize.bounds)
                k = shape[-1]
                bounds = (list(values[:k]), list(values[k : 2 * k]))
            return [
                _rs.InputStep.warp(
                    self._d,
                    list(self.indices),
                    self._eps,
                    bounds,
                    getattr(self.concentration0_prior, "_rs", None),
                    getattr(self.concentration1_prior, "_rs", None),
                    self.reverse,
                    self._flags(),
                    batched=bool(self.batch_shape),
                ).with_params(parameters)
                for parameters in self._concentrations()
            ]
        return [s.with_params(c) for s, c in zip(self._steps, self._concentrations(), strict=True)]

    def _apply(self, X: Any, inverse: bool) -> Any:
        if _torch_input.is_tensor(X):
            return _torch_input.warp_apply(self, X, inverse)
        from botorch_rs._arrays import broadcast_shapes, broadcast_to

        kind = kind_of(X)
        data, shape = _check_flat_matrix(X)
        steps = self._current_steps()
        if self.batch_shape:
            k = len(self.batch_shape)
            if inverse and tuple(shape[len(shape) - 2 - k : -2]) != self.batch_shape:
                raise BotorchTensorDimensionError(
                    "The right most batch dims of X must match self.batch_shape: "
                    f"({self.batch_shape})."
                )
            if not inverse:
                batch = broadcast_shapes(shape[:-2], self.batch_shape)
                if batch is None:
                    raise BotorchTensorDimensionError(
                        f"X batch shape {shape[:-2]} cannot broadcast with {self.batch_shape}."
                    )
                target = (*batch, *shape[-2:])
                data = broadcast_to(data, shape, target)
                shape = target
        if not inverse and self.training:
            # The bounds of the normalization are learned in train mode.
            if self._bounds is None:
                steps = [s.fit((data, shape)) for s in steps]
                self._steps = [
                    s.with_params(list(o.params)) for s, o in zip(steps, self._steps, strict=True)
                ]
                _torch_input.clear_affine_tensor_state(self._normalize)
        if len(steps) == 1:
            step = steps[0]
            out = step.untransform((data, shape)) if inverse else step.transform((data, shape))
            return from_pair(out, kind)
        nb = len(steps)
        n, d = shape[-2], shape[-1]
        block = n * d
        values = list(data)
        out_v: list[float] = [0.0] * len(values)
        member_indices = broadcast_to(list(range(nb)), self.batch_shape, shape[:-2])
        for b in range(nb):
            batches = [i for i, member in enumerate(member_indices) if int(member) == b]
            pts: list[float] = []
            for o in batches:
                s0 = o * block
                pts.extend(values[s0 : s0 + block])
            res = (
                steps[b].untransform((pts, (len(batches), n, d)))
                if inverse
                else steps[b].transform((pts, (len(batches), n, d)))
            )
            rd = list(res[0])
            for local, o in enumerate(batches):
                s0 = o * block
                out_v[s0 : s0 + block] = rd[local * block : (local + 1) * block]
        return from_flat(out_v, shape, kind)

    def _transform(self, X: Any) -> Any:
        return self._apply(X, inverse=False)

    def _untransform(self, X: Any) -> Any:
        return self._apply(X, inverse=True)

    def transform(self, X: Any) -> Any:
        return self._apply(X, inverse=self.reverse)

    def untransform(self, X: Any) -> Any:
        return self._apply(X, inverse=not self.reverse)

    def _state(self) -> Any:
        return (
            list(self.indices),
            self._eps,
            self._concentrations(),
            self._bounds,
            self._normalize._state()
            if _torch_input.is_tensor(getattr(self._normalize, "_coefficient", None))
            else None,
        )

    # -- models ----------------------------------------------------------------

    def _batch_params(self) -> list[list[float]] | None:
        if not self.batch_shape:
            return None
        return self._concentrations()

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        # One step for all batch elements: the normalization is learned from
        # the inputs of all of them (BoTorch's `Normalize` without a batch
        # shape); a batched Warp's concentrations come from `_batch_params`,
        # or, in the chain of batch element `batch` of a model whose chain
        # differs per batch element, from that element.
        if any(
            p is not None and not hasattr(p, "_rs")
            for p in (self.concentration0_prior, self.concentration1_prior)
        ):
            raise UnsupportedError("Native Warp fitting requires port priors; use backend='torch'.")
        steps = self._current_steps()
        index = batch if batch is not None and self.batch_shape else 0
        return [_Step(steps[index], self.training, prefix, self)]

    def _set_state(self, state: Any, batch: int | None, kind: ArrayKind) -> None:
        # The fitted normalization is shared by the batch elements.
        _torch_input.clear_affine_tensor_state(self._normalize)
        self._steps = [state.with_params(list(s.params)) for s in self._steps]

    def _bind(self, model: Any, prefix: str) -> None:
        self._model = model
        self._prefix = prefix
        # A native model owns float64 concentrations. Keep the caller's
        # default leaf identity while refreshing its native precision.
        parameters = self.__dict__.get("_native_default_parameters", {})
        if parameters:
            import torch

            for name, parameter in parameters.items():
                value = list(model._raw_param(prefix + name))
                with torch.no_grad():
                    parameter.data = torch.tensor(value, dtype=torch.double).reshape_as(parameter)

    def __deepcopy__(self, memo: dict[int, Any]) -> Warp:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        return f"Warp(d={self._d}, indices={list(self.indices)})"


# -----------------------------------------------------------------------------
# Feature transforms
# -----------------------------------------------------------------------------


def _one_hot_encoder_classes(encoder: Any) -> int | None:
    """The number of classes of ``partial(torch.nn.functional.one_hot,
    num_classes=c)`` (evaluated natively), else ``None``."""
    if not isinstance(encoder, functools.partial) or encoder.args:
        return None
    func = encoder.func
    if getattr(func, "__name__", None) != "one_hot" or set(encoder.keywords) != {"num_classes"}:
        return None
    module = getattr(func, "__module__", "") or ""
    if not module.startswith("torch"):
        return None
    return int(encoder.keywords["num_classes"])


class AppendFeatures(InputTransform):
    r"""A transform that appends the inputs with a given set of features
    (``feature_set``, ``n_f x d_f``) or features generated by a callable
    ``f`` (``batch_shape x q x d`` to ``batch_shape x q x n_f x d_f``, applied
    to the columns ``indices``). For ``batch_shape x q x d`` inputs the output
    is ``batch_shape x (q * n_f) x (d + d_f)``, the features of ``X[..., i,
    :]`` in rows ``i * n_f`` to ``(i + 1) * n_f``. With ``skip_expand``, the
    inputs are ``batch_shape x (q * n_f) x d`` (e.g. after an
    ``InputPerturbation`` with ``n_f`` perturbations)."""

    is_one_to_many: bool = True

    def __init__(
        self,
        feature_set: Any = None,
        f: Callable[..., Any] | None = None,
        indices: list[int] | None = None,
        fkwargs: dict[str, Any] | None = None,
        skip_expand: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        if feature_set is None and f is None:
            raise ValueError("Either a `feature_set` or a callable `f` has to be provided.")
        if feature_set is not None and f is not None:
            raise ValueError("Only one can be used: either `feature_set` or callable `f`.")
        self._fs: Any = None
        if feature_set is not None:
            data, shape = (
                (None, tuple(feature_set.shape))
                if _torch_input.is_tensor(feature_set)
                else to_flat(feature_set, "feature_set")
            )
            if len(shape) != 2:
                raise ValueError("`feature_set` must be an `n_f x d_f`-dim tensor!")
            self._fs = (data, shape)
            self.feature_set = feature_set
        self._f = f
        self.indices: Any = None
        if f is not None:
            if skip_expand:
                raise ValueError(
                    "`skip_expand` option is not supported in case of using a callable"
                )
            if indices is not None and len(indices) == 0:
                raise ValueError("`indices` list is empty!")
            if indices is not None:
                idx = [int(i) for i in indices]
                if len(set(idx)) != len(idx):
                    raise ValueError("Elements of `indices` tensor must be unique!")
                self.indices = idx
            self.fkwargs = fkwargs or {}
        self.skip_expand = bool(skip_expand)
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.training = True
        self._n_f: int | None = None if self._fs is None else int(self._fs[1][0])
        self._kind = first_kind(feature_set)

    def _indices_for(self, d: int) -> list[int] | None:
        if self.indices is None:
            return None
        return [i + d if i < 0 else i for i in self.indices]

    def _step_for(self, d: int) -> Any:
        features = (
            _torch_input.native_buffer_value(self, "feature_set") if self._fs is not None else None
        )
        return _rs.InputStep.append(
            to_flat(features) if features is not None else self._fs,
            self._f,
            self._indices_for(d),
            getattr(self, "fkwargs", None) if self._f is not None else None,
            self.skip_expand,
            self._flags(),
        )

    def transform(self, X: Any) -> Any:
        r"""Append the features to the inputs (regardless of the mode)."""
        if _torch_input.is_tensor(X):
            return _torch_input.append_apply(self, X)
        kind = kind_of(X)
        data, shape = _check_flat_matrix(X)
        out = self._step_for(shape[-1]).transform((data, shape))
        if self._f is not None and shape[-2] > 0:
            self._n_f = int(out[1][-2]) // int(shape[-2])
        return from_pair(out, kind)

    def _state(self) -> Any:
        source = (
            list(to_flat(self.feature_set)[0])
            if self._fs is not None and _torch_input.is_tensor(self.feature_set)
            else list(self._fs[0])
            if self._fs is not None
            else id(self._f)
        )
        return (source, self.indices, self.skip_expand)

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        return [_Step(self._step_for(d), self.training, prefix, self)]

    def _set_state(self, state: Any, batch: int | None, kind: ArrayKind) -> None:
        if state.n_f is not None:
            self._n_f = int(state.n_f)


class InteractionFeatures(AppendFeatures):
    r"""A transform that appends the first-order interaction terms
    ``x_i * x_j`` (``i < j``) of all inputs or of the inputs ``indices``
    (applied in all modes)."""

    def __init__(self, indices: list[int] | None = None) -> None:
        from botorch_rs.models.transforms.utils import interaction_features

        super().__init__(
            f=interaction_features,
            indices=indices,
            transform_on_train=True,
            transform_on_eval=True,
            transform_on_fantasize=True,
        )
        self._n_f = 1

    def _step_for(self, d: int) -> Any:
        return _rs.InputStep.interaction(self._indices_for(d))

    def transform(self, X: Any) -> Any:
        if _torch_input.is_tensor(X):
            from botorch_rs.models.transforms.utils import interaction_features

            selected = X if self.indices is None else X[..., self.indices]
            import torch

            return torch.cat((X, interaction_features(selected).squeeze(-2)), dim=-1)
        kind = kind_of(X)
        data, shape = _check_flat_matrix(X)
        return from_pair(self._step_for(shape[-1]).transform((data, shape)), kind)

    def _state(self) -> Any:
        return self.indices


class FilterFeatures(InputTransform):
    r"""A transform that keeps the input features ``feature_indices`` (a
    one-dimensional integer tensor)."""

    def __init__(
        self,
        feature_indices: Any,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        data, shape = _int_indices(feature_indices)
        if len(shape) != 1:
            raise ValueError("`feature_indices` must be a one-dimensional tensor!")
        if any(v < 0 for v in data):
            raise ValueError("Elements of `feature_indices` have to be larger/equal to zero!")
        if len(set(data)) != len(data):
            raise ValueError("Elements of `feature_indices` tensor must be unique!")
        self.feature_indices = feature_indices
        self._idx = list(data)
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.training = True

    def _step(self) -> Any:
        return _rs.InputStep.filter(_indices_list(self.feature_indices) or [], self._flags())

    def transform(self, X: Any) -> Any:
        r"""Keep only the features ``feature_indices``."""
        if _torch_input.is_tensor(X):
            return X[..., self.feature_indices]
        kind = kind_of(X)
        flat = _check_flat_matrix(X)
        return from_pair(self._step().transform(flat), kind)

    def equals(self, other: InputTransform) -> bool:
        if len(self._idx) != len(getattr(other, "_idx", [])):
            return False
        return super().equals(other)

    def _state(self) -> Any:
        return list(self._idx)

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        return [_Step(self._step(), self.training, prefix, self)]


def _int_indices(x: Any) -> tuple[list[int], tuple[int, ...]]:
    """The values and shape of an integer index tensor (BoTorch requires
    ``torch.long``)."""
    from botorch_rs._arrays import is_torch_tensor

    if is_torch_tensor(x):
        import importlib

        torch = importlib.import_module("torch")
        if x.dtype != torch.int64:
            raise ValueError("`feature_indices` tensor must be int64/long!")
        return [int(v) for v in x.reshape(-1)], tuple(x.shape)
    data, shape = to_flat(x, "feature_indices")
    if any(float(v) != int(v) for v in data):
        raise ValueError("`feature_indices` tensor must be int64/long!")
    return [int(v) for v in data], tuple(shape)


class OneHotToNumeric(InputTransform):
    r"""Transform categorical parameters from a one-hot to a numeric
    representation (``categorical_features``: start index to cardinality)."""

    def __init__(
        self,
        dim: int,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.dim = int(dim)
        self.categorical_features = OrderedDict(
            sorted((categorical_features or {}).items(), key=lambda x: x[0])
        )
        self.training = True
        self._rs_step = _rs.InputStep.one_hot_to_numeric(
            self.dim, _categorical_pairs(self.categorical_features), self._flags()
        )
        if len(self.categorical_features) > 0:
            onehot = [
                list(range(start, start + card))
                for start, card in self.categorical_features.items()
            ]
            idx = [i for g in onehot for i in g]
            self.numerical_idx = sorted(set(range(self.dim)) - set(idx))
            offset = 0
            self.ordinal_idx = []
            for start, card in self.categorical_features.items():
                self.ordinal_idx.append(start - offset)
                offset += card - 1
            reduced = len(self.ordinal_idx) + len(self.numerical_idx)
            self.new_numerical_idx = sorted(set(range(reduced)) - set(self.ordinal_idx))
            self.numeric_dim = len(self.new_numerical_idx) + len(self.categorical_features)

    def transform(self, X: Any) -> Any:
        r"""Transform the categorical inputs into an integer representation."""
        if _torch_input.is_tensor(X):
            return _torch_input.one_hot_apply(self, X, False)
        kind = kind_of(X)
        flat = _check_flat_matrix(X)
        return from_pair(self._rs_step.transform(flat), kind)

    def untransform(self, X: Any) -> Any:
        r"""Transform the categoricals from an integer to a one-hot
        representation."""
        if _torch_input.is_tensor(X):
            return _torch_input.one_hot_apply(self, X, True)
        kind = kind_of(X)
        flat = _check_flat_matrix(X)
        return from_pair(self._rs_step.untransform(flat), kind)

    def _state(self) -> Any:
        return dict(self.categorical_features)

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        return [_Step(self._rs_step, self.training, prefix, self)]


class NumericToCategoricalEncoding(InputTransform):
    r"""Transform integer-encoded categorical parameters into a vector
    representation (e.g. one-hot or descriptors) inserted at their position
    (``categorical_features``: index to cardinality; ``encoders``: index to a
    callable mapping ``torch.long`` values to ``cardinality``-dimensional
    encodings). ``partial(torch.nn.functional.one_hot, num_classes=c)`` is
    evaluated natively."""

    def __init__(
        self,
        dim: int,
        categorical_features: dict[int, int],
        encoders: dict[int, Callable[[Any], Any]],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self.encoders = encoders
        self.categorical_features = categorical_features
        if len(self.categorical_features) == 0:
            raise ValueError("At least one categorical feature for encoding must be provided.")
        if (num_cat := len(self.categorical_features)) > dim:
            raise ValueError(
                f"The number of categorical features ({num_cat}) exceeds the provided dimension "
                f"({dim})."
            )
        for idx, card in self.categorical_features.items():
            if not isinstance(card, int) or card <= 1:
                raise ValueError(
                    f"Categorical feature at index {idx} has cardinality {card}. All categorical "
                    "features must be an integer and have cardinality greater than 1."
                )
        if (enc_keys := set(self.encoders)) != (cf_keys := set(self.categorical_features)):
            raise ValueError(
                f"The keys of `encoders` ({enc_keys}) must match the keys of "
                f"of `categorical_features`  ({cf_keys})."
            )
        self.dim = int(dim)
        self.ordinal_idx = sorted(self.categorical_features)
        self.numerical_idx = sorted(set(range(dim)) - set(self.ordinal_idx))
        self.new_numerical_idx: list[int] = []
        self.encoded_idx: list[list[int]] = []
        offset = 0
        for idx in range(dim):
            if idx in self.numerical_idx:
                self.new_numerical_idx.append(idx + offset)
            else:
                card = self.categorical_features[idx]
                self.encoded_idx.append(list(range(idx + offset, idx + offset + card)))
                offset += card - 1
        self.training = True
        pairs = sorted(self.categorical_features.items())
        encs = []
        for idx, card in pairs:
            enc = self.encoders[idx]
            encs.append(None if _one_hot_encoder_classes(enc) == card else enc)
        self._rs_step = _rs.InputStep.numeric_to_categorical(
            self.dim, [(int(i), int(c)) for i, c in pairs], encs, self._flags()
        )

    def transform(self, X: Any) -> Any:
        r"""Transform the categorical inputs into a vector representation."""
        if _torch_input.is_tensor(X):
            return _torch_input.categorical_apply(self, X)
        kind = kind_of(X)
        flat = _check_flat_matrix(X)
        return from_pair(self._rs_step.transform(flat), kind)

    def equals(self, other: InputTransform) -> bool:
        return (
            type(self) is type(other)
            and self.transform_on_train == other.transform_on_train
            and self.transform_on_eval == other.transform_on_eval
            and self.transform_on_fantasize == other.transform_on_fantasize
            and self.categorical_features == other.categorical_features
        )

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        return [_Step(self._rs_step, self.training, prefix, self)]


class LearnedFeatureImputation(InputTransform):
    r"""An input transform that learns imputation values for the features a
    task does not observe (``feature_indices``: task value to the observed
    columns; the task column is last). The imputation values
    (``raw_imputation_values``, mapped into ``bounds`` by a sigmoid if given)
    are fitted jointly with the GP. Inputs without the task column get the
    column of ``target_task``."""

    def __init__(
        self,
        feature_indices: dict[int, list[int]],
        d: int,
        task_feature_index: int = -1,
        target_task: int | None = None,
        bounds: Any = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        dtype: Any = None,
        device: Any = None,
    ) -> None:
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        if target_task is not None and target_task not in feature_indices:
            raise ValueError(
                f"target_task={target_task} is not a key in feature_indices. "
                f"Available tasks: {sorted(feature_indices.keys())}."
            )
        self.target_task = target_task
        if task_feature_index != -1:
            raise ValueError(
                "LearnedFeatureImputation requires task_feature_index=-1 (task column last). "
                "Different tasks may have different feature counts, so a fixed non-last "
                "position is ambiguous."
            )
        self.num_tasks = len(feature_indices)
        self.d = int(d)
        self.feature_indices = {int(k): [int(c) for c in v] for k, v in feature_indices.items()}
        self._bounds: tuple[list[float], list[float]] | None = None
        self.bounds = bounds
        if _torch_input.is_tensor(bounds):
            if tuple(bounds.shape) != (2, d):
                raise ValueError(f"bounds must have shape (2, {d}), got {tuple(bounds.shape)}")
            if not bool((bounds[0] == 0).all() & (bounds[1] == 1).all()):
                warnings.warn(
                    "Non-default bounds passed to LearnedFeatureImputation. This transform expects normalized [0, 1] inputs.",
                    UserInputWarning,
                    stacklevel=2,
                )
        elif bounds is not None:
            data, shape = to_flat(bounds, "bounds")
            if tuple(shape) != (2, d):
                raise ValueError(f"bounds must have shape (2, {d}), got {tuple(shape)}")
            lo = [float(v) for v in data[:d]]
            hi = [float(v) for v in data[d:]]
            if not (all(v == 0 for v in lo) and all(v == 1 for v in hi)):
                warnings.warn(
                    "Non-default bounds passed to LearnedFeatureImputation. This transform "
                    "expects normalized [0, 1] inputs -- chain Normalize before this transform "
                    "so that the default bounds=[0, 1]^d are appropriate.",
                    UserInputWarning,
                    stacklevel=2,
                )
            self._bounds = (lo, hi)
        for task_value, cols in self.feature_indices.items():
            if d in cols:
                raise ValueError(
                    f"feature_indices[{task_value}] contains the task column index {d}. Feature "
                    "indices must not include the task column."
                )
        self.training = True
        self._kind = first_kind(bounds)
        self._step = _rs.InputStep.imputation(
            sorted(self.feature_indices.items()), self.d, target_task, self._bounds, self._flags()
        )
        self._model: Any = None
        self._prefix = ""
        if _torch_input.is_tensor(bounds) or (
            (dtype is not None or device is not None) and sys.modules.get("torch") is not None
        ):
            import torch

            like = (
                bounds.new_empty((), dtype=dtype or bounds.dtype, device=device or bounds.device)
                if _torch_input.is_tensor(bounds)
                else torch.empty((), dtype=dtype, device=device)
            )
            _torch_input.imputation_parameter(self, like)

    @property
    def raw_imputation_values(self) -> Any:
        r"""The raw (unconstrained) imputation values, ``num_tasks * (d + 1)``."""
        if self._model is not None:
            return self._model._raw_param_array(self._prefix + "raw_imputation_values")
        parameter = self.__dict__.get("_direct_parameters", {}).get("raw_imputation_values")
        if parameter is not None:
            return parameter
        return from_flat(list(self._step.params), (len(self._step.params),), self._kind)

    @raw_imputation_values.setter
    def raw_imputation_values(self, value: Any) -> None:
        if self._model is not None:
            self._model._set_raw_param(self._prefix + "raw_imputation_values", value)
            return
        if _torch_input.is_tensor(value) or self.__dict__.get("_direct_parameters"):
            import torch

            if isinstance(value, torch.nn.Parameter):
                if value.numel() != len(self._step.params) or value.ndim != 1:
                    raise ValueError(
                        "raw_imputation_values must be a flat Parameter with one entry per task and feature."
                    )
                self.__dict__.setdefault("_direct_parameters", {})["raw_imputation_values"] = value
            else:
                like = value if _torch_input.is_tensor(value) else self.raw_imputation_values
                parameter = _torch_input.imputation_parameter(self, like)
                with torch.no_grad():
                    parameter.copy_(_torch_input._tensor(value, parameter))
            return
        data, _ = to_flat(value, "raw_imputation_values")
        self._step = self._step.with_params([float(v) for v in data])

    def _current_step(self) -> Any:
        if self._model is not None:
            return self._step.with_params(
                list(self._model._raw_param(self._prefix + "raw_imputation_values"))
            )
        if _torch_input.is_tensor(self.bounds) or self.__dict__.get("_direct_parameters"):
            bounds = self._bounds
            if _torch_input.is_tensor(self.bounds):
                data, _ = to_flat(self.bounds)
                bounds = (list(data[: self.d]), list(data[self.d :]))
            step = _rs.InputStep.imputation(
                sorted(self.feature_indices.items()),
                self.d,
                self.target_task,
                bounds,
                self._flags(),
            )
            return step.with_params(list(to_flat(self.raw_imputation_values)[0]))
        return self._step

    @property
    def imputation_values(self) -> Any:
        r"""The imputation values, ``num_tasks x (d + 1)``."""
        raw = self.raw_imputation_values
        if _torch_input.is_tensor(raw):
            return _torch_input.imputation_values(self, raw)
        vals = list(self._current_step().imputation_values)
        return from_flat(vals, (self.num_tasks, self.d + 1), self._kind)

    def transform(self, X: Any) -> Any:
        r"""Impute the missing features with the learned values."""
        if _torch_input.is_tensor(X):
            return _torch_input.imputation_apply(self, X)
        kind = kind_of(X)
        flat = _check_flat_matrix(X)
        return from_pair(self._current_step().transform(flat), kind)

    def _state(self) -> Any:
        return (
            self.feature_indices,
            self.d,
            self.target_task,
            list(self._current_step().params),
            to_flat(self.bounds) if _torch_input.is_tensor(self.bounds) else self._bounds,
        )

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        return [_Step(self._current_step(), self.training, prefix, self)]

    def _set_state(self, state: Any, batch: int | None, kind: ArrayKind) -> None:
        self._step = state

    def _bind(self, model: Any, prefix: str) -> None:
        self._model = model
        self._prefix = prefix


# -----------------------------------------------------------------------------
# InputPerturbation
# -----------------------------------------------------------------------------


class InputPerturbation(InputTransform):
    r"""Replaces every input by ``n_w`` perturbed copies (BoTorch's
    ``InputPerturbation``), the input-noise model of risk-averse and robust
    Bayesian optimization (Cakmak et al., 2020; Daulton et al., 2022).

    For ``batch_shape x q x d`` inputs the output is ``batch_shape x (q * n_w)
    x d``, the perturbations of ``X[..., i, :]`` in rows ``i * n_w`` to
    ``(i + 1) * n_w``. ``perturbation_set`` is an ``n_w x k`` array of
    perturbations added to (or, with ``multiplicative=True``, multiplied
    with) the inputs ``indices`` (all inputs by default), or a callable on
    torch tensors mapping ``n x k`` points to ``n x n_w x k`` perturbations
    (heteroscedastic input noise; it must act pointwise). With ``bounds``
    (``2 x k``), the perturbed inputs are clamped to them.

    By default the transform applies in eval mode only (to the points of a
    posterior), as in BoTorch; ``transform_on_train=True`` is not supported
    by the botorch_rs models (an exact GP cannot have ``n * n_w`` training
    inputs for ``n`` targets). Inside ``fantasize`` it is skipped
    (``transform_on_fantasize=False``), as in BoTorch. Models evaluate it in
    Rust, with exact gradients (through a callable with ``torch.autograd``).

    Example:
        >>> perturbation_set = draw_sobol_normal_samples(d=2, n=16) * 0.05
        >>> model = SingleTaskGP(
        ...     train_X, train_Y,
        ...     input_transform=InputPerturbation(perturbation_set, bounds=bounds),
        ... )
    """

    is_one_to_many = True

    def __init__(
        self,
        perturbation_set: Any,
        bounds: Any = None,
        indices: list[int] | None = None,
        multiplicative: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        self._function: Callable[[Any], Any] | None = None
        self._set: Any = None
        if callable(perturbation_set):
            self._function = perturbation_set
        else:
            flat = (
                (None, tuple(perturbation_set.shape))
                if _torch_input.is_tensor(perturbation_set)
                else to_flat(perturbation_set, "perturbation_set")
            )
            if len(flat[1]) != 2:
                raise ValueError("`perturbation_set` must be an `n_p x d`-dim tensor!")
            self._set = flat
        self._bounds: Any = None
        if bounds is not None:
            self._bounds = (
                (None, tuple(bounds.shape))
                if _torch_input.is_tensor(bounds)
                else to_flat(bounds, "bounds")
            )
            if self._set is not None and self._bounds[1][-1] != self._set[1][-1]:
                raise ValueError(
                    "`bounds` must have the same number of columns (last dimension) as "
                    f"the `perturbation_set`! Got {self._bounds[1][-1]} and "
                    f"{self._set[1][-1]}."
                )
        self.perturbation_set = perturbation_set
        self.bounds = bounds
        self.indices = None if indices is None else [int(i) for i in indices]
        self.multiplicative = bool(multiplicative)
        self.transform_on_train = bool(transform_on_train)
        self.transform_on_eval = bool(transform_on_eval)
        self.transform_on_fantasize = bool(transform_on_fantasize)
        self._n_w: int | None = None if self._set is None else int(self._set[1][0])
        self._kind: ArrayKind = first_kind(perturbation_set, bounds)
        self._rs_cache: dict[int, _rs.InputPerturbation] = {}
        self.training = True

    @property
    def n_w(self) -> int | None:
        """The number of perturbations (``None`` for a callable that has not
        been evaluated yet)."""
        return self._n_w

    def _rs_step(self, d: int) -> _rs.InputPerturbation:
        """The transform for ``d``-dimensional inputs (negative ``indices``
        count from the end)."""
        perturbations = _torch_input.native_buffer_value(self, "perturbation_set")
        bounds = _torch_input.native_buffer_value(self, "bounds")
        tensor_state = _torch_input.is_tensor(perturbations) or _torch_input.is_tensor(bounds)
        step = None if tensor_state else self._rs_cache.get(d)
        if step is None:
            indices = None
            if self.indices is not None:
                indices = [i + d if i < 0 else i for i in self.indices]
                if any(not 0 <= i < d for i in indices):
                    raise IndexError(
                        f"indices {self.indices} are out of range for inputs with d = {d}."
                    )
            step = _rs.InputPerturbation(
                to_flat(perturbations) if self._function is None else self._set,
                self._function,
                to_flat(bounds) if bounds is not None else self._bounds,
                indices,
                self.multiplicative,
                self.transform_on_train,
                self.transform_on_eval,
                self.transform_on_fantasize,
            )
            self._rs_cache[d] = step
        return step

    def transform(self, X: Any) -> Any:
        """The perturbed inputs: ``batch_shape x (q * n_w) x d`` for
        ``batch_shape x q x d`` inputs (regardless of the mode, as BoTorch's
        ``transform``). Tensor calls preserve the device and full graph."""
        if _torch_input.is_tensor(X):
            return _torch_input.perturbation_apply(self, X)
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise ValueError(f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}.")
        out = self._rs_step(flat[1][-1]).transform(flat)
        if self._function is not None and flat[1][-2] > 0:
            self._n_w = out[1][-2] // flat[1][-2]
        if self._n_w is not None:
            # A fixed set broadcasts over the batch dimensions of the inputs.
            batch = tuple(flat[1][:-2]) if self._function is not None else ()
            self._last_batch_shape = (*batch, self._n_w)
        return from_pair(out, kind)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The shape of the perturbations of the last ``transform`` without
        ``n x d`` (BoTorch's ``_perturbations.shape[:-2]``): ``n_w`` for a
        fixed perturbation set, ``batch_shape x n_w`` for a callable;
        ``AttributeError`` before the first ``transform``, as in BoTorch."""
        shape: tuple[int, ...] | None = getattr(self, "_last_batch_shape", None)
        if shape is None:
            raise AttributeError("'NoneType' object has no attribute 'shape'")
        return shape

    def equals(self, other: InputTransform) -> bool:
        r"""Whether another transform is equivalent (the cached perturbations
        of the last inputs are not compared, as with
        meta-pytorch/botorch#3389)."""
        if not (
            type(self) is type(other)
            and self.transform_on_train == other.transform_on_train
            and self.transform_on_eval == other.transform_on_eval
            and self.transform_on_fantasize == other.transform_on_fantasize
            and self.indices == other.indices
            and self.multiplicative == other.multiplicative
        ):
            return False
        assert isinstance(other, InputPerturbation)
        bounds = to_flat(self.bounds) if _torch_input.is_tensor(self.bounds) else self._bounds
        other_bounds = (
            to_flat(other.bounds) if _torch_input.is_tensor(other.bounds) else other._bounds
        )
        if (self._function is None) != (other._function is None):
            return False
        if self._function is not None:
            return self._function is other._function and _close(bounds, other_bounds)
        source = (
            to_flat(self.perturbation_set)
            if _torch_input.is_tensor(self.perturbation_set)
            else self._set
        )
        other_source = (
            to_flat(other.perturbation_set)
            if _torch_input.is_tensor(other.perturbation_set)
            else other._set
        )
        return _close(source, other_source) and _close(bounds, other_bounds)

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        if self.transform_on_train:
            raise UnsupportedError(
                "InputPerturbation with transform_on_train=True is not supported: the "
                "perturbed training inputs (n * n_w points) would not match the n "
                "training targets of the model."
            )
        return [_Step(self._rs_step(d), self.training, prefix, self)]

    def _set_state(self, state: Any, batch: int | None, kind: ArrayKind) -> None:
        if state is not None:
            self._n_w = int(state)

    def __repr__(self) -> str:
        source = "callable" if self._function is not None else f"n_w={self._n_w}"
        return f"InputPerturbation({source}, multiplicative={self.multiplicative})"


# -----------------------------------------------------------------------------
# Containers
# -----------------------------------------------------------------------------


class ChainedInputTransform(InputTransform):
    r"""An input transform representing the chaining of individual
    transforms (BoTorch's ``ChainedInputTransform(**transforms)``): they are
    applied in order, each according to its own flags.

    Example:
        >>> tf = ChainedInputTransform(
        ...     normalize=Normalize(d=2),
        ...     warp=Warp(d=2, indices=[0, 1]),
        ... )
        >>> model = SingleTaskGP(train_X, train_Y, input_transform=tf)
    """

    def __init__(self, **transforms: InputTransform) -> None:
        if not transforms:
            raise ValueError("ChainedInputTransform needs at least one transform.")
        for name, tf in transforms.items():
            if not isinstance(tf, InputTransform):
                raise UnsupportedError(
                    f"botorch_rs chains its own input transforms, got {type(tf).__name__} for "
                    f"{name!r}."
                )
        self._transforms: dict[str, InputTransform] = OrderedDict(transforms)
        tfs = list(transforms.values())
        self.is_one_to_many = any(tf.is_one_to_many for tf in tfs)
        self.transform_on_train = any(tf.transform_on_train for tf in tfs)
        self.transform_on_eval = any(tf.transform_on_eval for tf in tfs)
        self.transform_on_fantasize = any(tf.transform_on_fantasize for tf in tfs)
        self.training = True

    def __getitem__(self, key: str) -> InputTransform:
        return self._transforms[key]

    def __getattr__(self, name: str) -> Any:
        # Attribute access to the transforms (``torch.nn.ModuleDict``).
        transforms = self.__dict__.get("_transforms")
        if transforms is not None and name in transforms:
            return transforms[name]
        return InputTransform.__getattr__(self, name)

    def __contains__(self, key: object) -> bool:
        return key in self._transforms

    def __iter__(self) -> Iterator[str]:
        return iter(self._transforms)

    def __len__(self) -> int:
        return len(self._transforms)

    def keys(self) -> KeysView[str]:
        return self._transforms.keys()

    def values(self) -> ValuesView[InputTransform]:
        return self._transforms.values()

    def items(self) -> ItemsView[str, InputTransform]:
        return self._transforms.items()

    def train(self, mode: bool = True) -> ChainedInputTransform:
        self.training = mode
        for tf in self._transforms.values():
            tf.train(mode)
        return self

    def transform(self, X: Any) -> Any:
        """Applies the transforms in order (each according to the mode)."""
        for tf in self._transforms.values():
            X = tf.forward(X)
        return X

    def untransform(self, X: Any) -> Any:
        """Undoes the transforms in reverse order."""
        for tf in reversed(list(self._transforms.values())):
            X = tf.untransform(X)
        return X

    def preprocess_transform(self, X: Any) -> Any:
        for tf in self._transforms.values():
            X = tf.preprocess_transform(X)
        return X

    def equals(self, other: InputTransform) -> bool:
        if not isinstance(other, ChainedInputTransform) or list(self.keys()) != list(other.keys()):
            return False
        return super().equals(other) and all(
            t1.equals(t2) for t1, t2 in zip(self.values(), other.values(), strict=True)
        )

    def _state(self) -> Any:
        return list(self.keys())

    def _per_batch(self) -> tuple[int, ...] | None:
        shapes = {s for tf in self._transforms.values() if (s := tf._per_batch()) is not None}
        if len(shapes) > 1:
            raise UnsupportedError(
                f"the transforms of the chain have different batch shapes {sorted(shapes)}"
            )
        return shapes.pop() if shapes else None

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        steps: list[_Step] = []
        for name, tf in self._transforms.items():
            per = tf._per_batch() is not None or tf._batch_params() is not None
            sub = tf._model_steps(d, f"{prefix}{name}.", batch if per else None)
            steps.extend(sub)
            if tf.transform_on_train or tf.transform_on_eval:
                d = _out_dim_of(sub, d)
        return steps

    def _bind(self, model: Any, prefix: str) -> None:
        for name, tf in self._transforms.items():
            tf._bind(model, f"{prefix}{name}.")

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._transforms.items())
        return f"ChainedInputTransform({inner})"


def _out_dim_of(steps: list[_Step], d: int) -> int:
    for s in steps:
        if hasattr(s.rs, "out_dim"):
            d = int(s.rs.out_dim(d))
    return d


class BatchBroadcastedInputTransform(InputTransform):
    r"""A list of transforms broadcast across a batch dimension
    (``broadcast_index``): transform ``i`` applies to ``X[i]`` along that
    dimension. In a batched model, every batch element uses its transform
    (registered as submodules, as with meta-pytorch/botorch#3389, so that
    their modes follow the model's)."""

    def __init__(self, transforms: list[InputTransform], broadcast_index: int = -3) -> None:
        if broadcast_index in (-2, -1):
            raise ValueError(
                "The broadcast index cannot be -2 and -1, as these indices are reserved for "
                "non-batch, data and input dimensions."
            )
        self.transforms = list(transforms)
        self.broadcast_index = int(broadcast_index)
        self.is_one_to_many = self.transforms[0].is_one_to_many
        if not all(tf.is_one_to_many == self.is_one_to_many for tf in self.transforms):
            raise ValueError("All transforms must have the same is_one_to_many property.")
        self.transform_on_train = any(tf.transform_on_train for tf in self.transforms)
        self.transform_on_eval = any(tf.transform_on_eval for tf in self.transforms)
        self.transform_on_fantasize = any(tf.transform_on_fantasize for tf in self.transforms)
        self.training = True

    def train(self, mode: bool = True) -> BatchBroadcastedInputTransform:
        self.training = mode
        for tf in self.transforms:
            tf.train(mode)
        return self

    def _split(self, X: Any) -> tuple[list[Any], ArrayKind, tuple[int, ...]]:
        kind = kind_of(X)
        if _torch_input.is_tensor(X):
            if X.shape[self.broadcast_index] != len(self.transforms):
                raise ValueError("The broadcast dimension must match the number of transforms.")
            return list(X.unbind(self.broadcast_index)), kind, tuple(X.shape)
        data, shape = to_flat(X, "X")
        nd = len(shape)
        axis = self.broadcast_index % nd
        if shape[axis] != len(self.transforms):
            raise ValueError(
                f"X has {shape[axis]} entries along the broadcast dimension, but there are "
                f"{len(self.transforms)} transforms."
            )
        outer = math.prod(shape[:axis])
        inner = math.prod(shape[axis + 1 :])
        parts = []
        for i in range(shape[axis]):
            vals: list[float] = []
            for o in range(outer):
                s = (o * shape[axis] + i) * inner
                vals.extend(data[s : s + inner])
            parts.append(from_flat(vals, (*shape[:axis], *shape[axis + 1 :]), kind))
        return parts, kind, shape

    def _stack(self, outs: list[Any], kind: ArrayKind, axis_from_end: int) -> Any:
        if _torch_input.is_tensor(outs[0]):
            import torch

            return torch.stack(outs, dim=axis_from_end)
        flats = [to_flat(o, "X") for o in outs]
        shape = flats[0][1]
        nd = len(shape) + 1
        axis = axis_from_end % nd
        outer = math.prod(shape[:axis])
        inner = math.prod(shape[axis:])
        data: list[float] = []
        for o in range(outer):
            for f in flats:
                data.extend(list(f[0])[o * inner : (o + 1) * inner])
        return from_flat(data, (*shape[:axis], len(flats), *shape[axis:]), kind)

    def transform(self, X: Any) -> Any:
        parts, kind, _ = self._split(X)
        outs = [tf.forward(p) for p, tf in zip(parts, self.transforms, strict=True)]
        return self._stack(outs, kind, self.broadcast_index)

    def untransform(self, X: Any) -> Any:
        parts, kind, _ = self._split(X)
        outs = [tf.untransform(p) for p, tf in zip(parts, self.transforms, strict=True)]
        return self._stack(outs, kind, self.broadcast_index)

    def preprocess_transform(self, X: Any) -> Any:
        parts, kind, _ = self._split(X)
        outs = [tf.preprocess_transform(p) for p, tf in zip(parts, self.transforms, strict=True)]
        return self._stack(outs, kind, self.broadcast_index)

    def equals(self, other: InputTransform) -> bool:
        return (
            super().equals(other)
            and isinstance(other, BatchBroadcastedInputTransform)
            and len(self.transforms) == len(other.transforms)
            and all(t1.equals(t2) for t1, t2 in zip(self.transforms, other.transforms, strict=True))
            and self.broadcast_index == other.broadcast_index
        )

    def _per_batch(self) -> tuple[int, ...] | None:
        return (len(self.transforms),)

    def _model_steps(self, d: int, prefix: str, batch: int | None) -> list[_Step]:
        if batch is None:
            raise UnsupportedError(
                "BatchBroadcastedInputTransform requires a batched model with one transform per "
                "batch element."
            )
        tf = self.transforms[batch]
        steps = tf._model_steps(d, prefix, None)
        if any(s.rs.params if hasattr(s.rs, "params") else False for s in steps):
            raise UnsupportedError(
                "Transforms with learned parameters (e.g. Warp) in a "
                "BatchBroadcastedInputTransform are not supported; use Warp(batch_shape=...)."
            )
        return steps


# -----------------------------------------------------------------------------
# Model integration
# -----------------------------------------------------------------------------


class ChainSpec(NamedTuple):
    """The input transform of a model for the extension: one list of steps
    for all batch elements (``per_batch=False``, with the learned parameters
    of every batch element in ``batch_params`` if they differ) or one per
    batch element."""

    chains: list[list[_Step]]
    per_batch: bool
    batch_params: list[list[float]] | None = None


def _batch_index_map(tf: InputTransform, model_batch: tuple[int, ...]) -> list[int]:
    """For every model batch element (row-major), the index of the per-batch
    state of ``tf``."""
    nb = math.prod(model_batch)
    pb = tf._per_batch()
    if pb is None:
        return [0] * nb
    if isinstance(tf, BatchBroadcastedInputTransform):
        nd = len(model_batch) + 2
        axis = tf.broadcast_index % nd
        if axis >= len(model_batch) or model_batch[axis] != len(tf.transforms):
            raise BotorchTensorDimensionError(
                f"BatchBroadcastedInputTransform with {len(tf.transforms)} transforms along "
                f"dimension {tf.broadcast_index} does not match the model batch shape "
                f"{model_batch}."
            )
        stride = math.prod(model_batch[axis + 1 :])
        return [(b // stride) % model_batch[axis] for b in range(nb)]
    if tuple(pb) != tuple(model_batch):
        raise BotorchTensorDimensionError(
            f"The input transform has batch shape {tuple(pb)} but the training data have batch "
            f"shape {model_batch}."
        )
    return list(range(nb))


def chain_spec(transform: InputTransform, d: int, model_batch: tuple[int, ...]) -> ChainSpec:
    """The steps of a model's input transform (see :class:`ChainSpec`)."""
    if transform._per_batch() is None:
        steps = transform._model_steps(d, "input_transform.", None)
        return ChainSpec([steps], False, _shared_batch_params(steps, model_batch))
    index = _batch_index_map(transform, model_batch)
    return ChainSpec(
        [transform._model_steps(d, "input_transform.", index[b]) for b in range(len(index))],
        True,
    )


def _shared_batch_params(
    steps: list[_Step], model_batch: tuple[int, ...]
) -> list[list[float]] | None:
    """The learned parameters of every model batch element for a shared
    chain (``None`` if they are the same for all)."""
    nb = math.prod(model_batch)
    owners_seen: set[int] = set()
    per_step: list[list[list[float]] | list[float]] = []
    batched = False
    for s in steps:
        params = list(getattr(s.rs, "params", []) or []) if hasattr(s.rs, "params") else []
        bp = s.owner._batch_params() if id(s.owner) not in owners_seen else None
        owners_seen.add(id(s.owner))
        if bp is not None:
            if len(bp) != nb:
                raise BotorchTensorDimensionError(
                    f"{type(s.owner).__name__} has the batch shape {s.owner._per_batch() or ()} "
                    f"but the training data have batch shape {model_batch}."
                )
            batched = True
            per_step.append(bp)
        else:
            per_step.append(params)
    if not batched:
        return None
    out: list[list[float]] = [[] for _ in range(nb)]
    for p in per_step:
        for b in range(nb):
            out[b].extend(p[b] if p and isinstance(p[0], list) else p)  # type: ignore[arg-type]
    return out


def apply_chain_states(
    spec: ChainSpec, states: list[list[Any]], kind: ArrayKind, model_batch: tuple[int, ...]
) -> None:
    """Stores the fitted steps of a model's chain(s) in the Python transforms."""
    for b, (steps, state) in enumerate(zip(spec.chains, states, strict=False)):
        for step, fitted in zip(steps, state, strict=True):
            per = step.owner._per_batch() is not None
            if per or b == 0:
                step.owner._set_state(
                    fitted, _owner_batch(step.owner, b, model_batch) if per else None, kind
                )


def _owner_batch(owner: InputTransform, b: int, model_batch: tuple[int, ...]) -> int:
    return _batch_index_map(owner, model_batch)[b] if owner._per_batch() is not None else 0


def is_basic_transform(transform: InputTransform) -> bool:
    """Whether a transform is one of the plain transforms the models have
    always evaluated (a plain ``Normalize``, an ``InputPerturbation``, or a
    chain of them)."""
    if isinstance(transform, Normalize):
        return transform._plain
    if isinstance(transform, InputPerturbation):
        return True
    if isinstance(transform, ChainedInputTransform):
        return all(is_basic_transform(t) for t in transform.values())
    return False


__all__ = [
    "AffineInputTransform",
    "AppendFeatures",
    "BatchBroadcastedInputTransform",
    "ChainedInputTransform",
    "FilterFeatures",
    "InputPerturbation",
    "InputStandardize",
    "InputTransform",
    "InteractionFeatures",
    "LearnedFeatureImputation",
    "Log10",
    "Normalize",
    "NumericToCategoricalEncoding",
    "OneHotToNumeric",
    "ReversibleInputTransform",
    "Round",
    "Warp",
]
