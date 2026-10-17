"""Input transforms (``botorch.models.transforms.input``): ``Normalize``,
``InputPerturbation`` and ``ChainedInputTransform``."""

from __future__ import annotations

import warnings
from collections.abc import Callable, ItemsView, Iterator, KeysView, ValuesView
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, first_kind, from_flat, from_pair, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import UserInputWarning


class InputTransform:
    """Base class of the input transforms."""

    training: bool = True
    #: Whether the transform maps a point to several points.
    is_one_to_many: bool = False
    transform_on_train: bool = True
    transform_on_eval: bool = True
    transform_on_fantasize: bool = True

    def train(self, mode: bool = True) -> InputTransform:
        self.training = mode
        return self

    def eval(self) -> InputTransform:
        return self.train(False)

    def transform(self, X: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def untransform(self, X: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method."
        )

    def forward(self, X: Any) -> Any:
        """BoTorch's ``InputTransform.forward``: ``transform(X)`` if the
        transform applies in the current mode (``transform_on_train`` in train
        mode, ``transform_on_eval`` in eval mode), else ``X``."""
        if self.training:
            if self.transform_on_train:
                return self.transform(X)
        elif self.transform_on_eval:
            return self.transform(X)
        return X

    def __call__(self, X: Any) -> Any:
        return self.forward(X)


class Normalize(InputTransform):
    r"""Normalizes the inputs to the unit cube, ``(X - offset) / coefficient``.

    With ``bounds=None`` (or ``learn_bounds=True``) the bounds are learned from
    the training inputs when the transform is used by a model (or called in
    train mode). With a ``batch_shape`` (or bounds of shape
    ``batch_shape x 2 x d``) every batch element has its own bounds;
    otherwise one normalization is learned from all batch elements. With
    ``indices``, only those inputs are normalized (``bounds``, ``offset`` and
    ``coefficient`` then have ``len(indices)`` columns, as in BoTorch); a
    ``MultiTaskGP`` uses this to leave its task column alone, the other models
    support only transforms of all inputs. ``center`` must be 0.5.
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
        self._d = int(d)
        self.indices: list[int] | None = None
        transform_dimension = int(d)
        if indices is not None:
            idx = [int(i) for i in to_flat(indices, "indices")[0]]
            # BoTorch's (AffineInputTransform's) checks.
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
            if idx != list(range(d)):
                self.indices = idx
                transform_dimension = len(idx)
                if bounds is not None:
                    bdata, bshape = to_flat(bounds, "bounds")
                    if bshape[-1] == d and len(idx) != d:
                        rows = len(bdata) // d
                        bounds = from_flat(
                            [bdata[r * d + i] for r in range(rows) for i in idx],
                            (*bshape[:-1], len(idx)),
                            first_kind(bounds),
                        )
        if bounds is not None and to_flat(bounds, "bounds")[1][-1] != transform_dimension:
            raise BotorchTensorDimensionError(
                "Dimensions of provided `bounds` are incompatible with "
                f"transform_dimension = {transform_dimension}!"
            )
        if not (transform_on_train and transform_on_eval and transform_on_fantasize):
            raise UnsupportedError("transform_on_train/eval/fantasize=False is not supported.")
        if reverse:
            raise UnsupportedError("reverse=True is not supported.")
        if center != 0.5:
            raise UnsupportedError("Only center=0.5 is supported.")
        learn = (bounds is None) if learn_bounds is None else bool(learn_bounds)
        if not learn and bounds is None:
            warnings.warn(
                "learn_bounds is False and no bounds were provided. The bounds "
                "will not be updated and the transform will be a no-op.",
                UserInputWarning,
                stacklevel=2,
            )
        shape = tuple(int(b) for b in (batch_shape or ()))
        self._kind: ArrayKind = first_kind(bounds)
        self._rs = _rs.Normalize(
            transform_dimension,
            to_flat(bounds, "bounds") if bounds is not None else None,
            learn,
            min_range,
            list(shape),
        )
        self.almost_zero = almost_zero
        self.training = True

    # -- state -------------------------------------------------------------

    def _set_fitted(self, fitted: _rs.Normalize, kind: ArrayKind) -> None:
        self._rs = fitted
        if kind.lib != "list" or self._kind.lib == "list":
            self._kind = kind

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape (empty for one normalization of all inputs)."""
        return tuple(self._rs.batch_shape)

    @property
    def learn_bounds(self) -> bool:
        return self._rs.learn_bounds

    @property
    def min_range(self) -> float:
        return self._rs.min_range

    @property
    def center(self) -> float:
        return 0.5

    @property
    def bounds(self) -> Any:
        """The normalization bounds, shape ``batch_shape x 2 x d``."""
        return from_pair(self._rs.bounds, self._kind)

    @property
    def offset(self) -> Any:
        """The offsets (lower bounds), shape ``batch_shape x 1 x d``."""
        return from_pair(self._rs.offset, self._kind)

    @property
    def coefficient(self) -> Any:
        """The coefficients (ranges), shape ``batch_shape x 1 x d``."""
        return from_pair(self._rs.coefficient, self._kind)

    mins = offset
    ranges = coefficient

    # -- application -------------------------------------------------------

    def _apply(self, X: Any, inverse: bool) -> Any:
        kind = first_kind(X)
        flat = to_flat(X, "X")
        if self.indices is not None:
            # Transform the columns `indices` only.
            data, shape = flat
            d = shape[-1]
            if d != self._d:
                raise BotorchTensorDimensionError(
                    f"Wrong input dimension. Received {d}, expected {self._d}."
                )
            idx = self.indices
            rows = len(data) // d
            sub = ([data[r * d + i] for r in range(rows) for i in idx], (*shape[:-1], len(idx)))
            if not inverse and self.training and self.learn_bounds:
                self._set_fitted(self._rs.fit(sub), kind if kind.lib != "list" else self._kind)
            out = self._rs.untransform(sub) if inverse else self._rs.transform(sub)
            new = list(data)
            k = len(idx)
            for r in range(rows):
                for j, i in enumerate(idx):
                    new[r * d + i] = out[0][r * k + j]
            return from_flat(new, tuple(shape), kind)
        if not inverse and self.training and self.learn_bounds:
            self._set_fitted(self._rs.fit(flat), kind if kind.lib != "list" else self._kind)
        out = self._rs.untransform(flat) if inverse else self._rs.transform(flat)
        return from_pair(out, kind)

    def transform(self, X: Any) -> Any:
        """Normalizes ``X`` (``... x n x d``); in train mode, learnable bounds are
        first fitted to ``X``."""
        return self._apply(X, inverse=False)

    def forward(self, X: Any) -> Any:
        return self.transform(X)

    __call__ = forward

    def untransform(self, X: Any) -> Any:
        """Maps normalized inputs back to the original space."""
        return self._apply(X, inverse=True)

    def __repr__(self) -> str:
        extra = "" if self.indices is None else f", indices={self.indices}"
        return f"Normalize(d={self._d}{extra}, learn_bounds={self.learn_bounds})"


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
    inputs for ``n`` targets). Models evaluate it in Rust, with exact
    gradients (through a callable with ``torch.autograd``).

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
            flat = to_flat(perturbation_set, "perturbation_set")
            if len(flat[1]) != 2:
                raise ValueError("`perturbation_set` must be an `n_p x d`-dim tensor!")
            self._set = flat
        self._bounds: Any = None
        if bounds is not None:
            self._bounds = to_flat(bounds, "bounds")
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
        step = self._rs_cache.get(d)
        if step is None:
            indices = None
            if self.indices is not None:
                indices = [i + d if i < 0 else i for i in self.indices]
                if any(not 0 <= i < d for i in indices):
                    raise IndexError(
                        f"indices {self.indices} are out of range for inputs with d = {d}."
                    )
            step = _rs.InputPerturbation(
                self._set,
                self._function,
                self._bounds,
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
        ``transform``; the result is not connected to torch's autograd)."""
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise ValueError(f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}.")
        out = self._rs_step(flat[1][-1]).transform(flat)
        if self._function is not None and flat[1][-2] > 0:
            self._n_w = out[1][-2] // flat[1][-2]
        return from_pair(out, kind)

    def __repr__(self) -> str:
        source = "callable" if self._function is not None else f"n_w={self._n_w}"
        return f"InputPerturbation({source}, multiplicative={self.multiplicative})"


class ChainedInputTransform(InputTransform):
    r"""Applies its input transforms in order (BoTorch's
    ``ChainedInputTransform(**transforms)``); botorch_rs chains ``Normalize``
    (without a batch shape) and ``InputPerturbation``.

    Example:
        >>> tf = ChainedInputTransform(
        ...     normalize=Normalize(d=2),
        ...     perturbation=InputPerturbation(perturbation_set, bounds=unit_cube),
        ... )
        >>> model = SingleTaskGP(train_X, train_Y, input_transform=tf)
    """

    def __init__(self, **transforms: InputTransform) -> None:
        if not transforms:
            raise ValueError("ChainedInputTransform needs at least one transform.")
        for name, tf in transforms.items():
            if not isinstance(tf, (Normalize, InputPerturbation)):
                raise UnsupportedError(
                    "botorch_rs chains Normalize and InputPerturbation transforms, got "
                    f"{type(tf).__name__} for {name!r}."
                )
        self._transforms: dict[str, InputTransform] = dict(transforms)
        tfs = list(transforms.values())
        self.is_one_to_many = any(tf.is_one_to_many for tf in tfs)
        self.transform_on_train = any(tf.transform_on_train for tf in tfs)
        self.transform_on_eval = any(tf.transform_on_eval for tf in tfs)
        self.transform_on_fantasize = any(tf.transform_on_fantasize for tf in tfs)
        self.training = True

    def __getitem__(self, key: str) -> InputTransform:
        return self._transforms[key]

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

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._transforms.items())
        return f"ChainedInputTransform({inner})"


def _steps_of(transform: InputTransform) -> list[InputTransform]:
    if isinstance(transform, ChainedInputTransform):
        return list(transform.values())
    return [transform]


def chain_steps(transform: InputTransform, d: int) -> list[Any]:
    """The steps of a one-to-many or chained transform for the extension."""
    steps: list[Any] = []
    for tf in _steps_of(transform):
        if isinstance(tf, InputPerturbation):
            if tf.transform_on_train:
                raise UnsupportedError(
                    "InputPerturbation with transform_on_train=True is not supported: the "
                    "perturbed training inputs (n * n_w points) would not match the n "
                    "training targets of the model."
                )
            steps.append(tf._rs_step(d))
        elif isinstance(tf, Normalize):
            if tf.batch_shape:
                raise UnsupportedError(
                    "A Normalize with a batch shape is not supported in a chained or "
                    "one-to-many input transform; use Normalize(d)."
                )
            if tf.indices is not None:
                raise UnsupportedError(
                    "Normalize over a subset of the inputs (`indices`) is not supported in a "
                    "chained or one-to-many input transform."
                )
            steps.append(tf._rs)
        else:  # pragma: no cover - rejected by ChainedInputTransform
            raise UnsupportedError(f"Unsupported input transform {type(tf).__name__}.")
    return steps


def apply_chain_state(transform: InputTransform, state: list[Any], kind: ArrayKind) -> None:
    """Stores the fitted steps of a model's chain in the Python transforms."""
    for tf, fitted in zip(_steps_of(transform), state, strict=True):
        if isinstance(tf, Normalize):
            tf._set_fitted(fitted, kind)
        elif isinstance(tf, InputPerturbation) and fitted is not None:
            tf._n_w = int(fitted)


__all__ = ["ChainedInputTransform", "InputPerturbation", "InputTransform", "Normalize"]
