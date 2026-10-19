"""Base classes of the test problems (``botorch.test_functions.base``).

:class:`BaseTestProblem` provides the shared behaviour of all test problems:
``problem(X)`` (``forward``) evaluates ``X`` (``batch_shape x d``), adds
Gaussian observation noise with standard deviation ``noise_std`` (if set and
``noise=True``) and negates the result if ``negate=True``; ``evaluate_true``
returns the noiseless, un-negated values. :class:`MultiObjectiveTestProblem`
adds the reference point, the maximum hypervolume and Pareto fronts, and
:class:`ConstrainedBaseTestProblem` outcome constraints ``c_i(x) >= 0`` with
``evaluate_slack``, ``evaluate_slack_true`` and ``is_feasible`` (slacks are
never negated).

The evaluations run in Rust (in float64). Results have the array type,
floating dtype and device of the input; the noise is drawn with the caller's
array library (``torch.randn_like`` for torch inputs, NumPy's global generator
for NumPy inputs, a seeded Rust generator for nested lists). Arrays that are
not computed from an input (``bounds``, ``ref_point``, Pareto fronts) use
``dtype``: a torch dtype gives tensors, a NumPy dtype arrays, and ``None``
float64 tensors if torch is imported (else lists); ``to(dtype=...,
device=...)`` changes it like ``torch.nn.Module.to``.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Protocol

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    bool_array,
    from_flat,
    from_pair,
    kind_from_dtype,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import UnsupportedError


def _noise_arg(value: Any, name: str) -> float | list[float] | None:
    """A noise standard deviation for the extension: a float (one for all
    outputs) or a list of floats (one per output). Lists must have one entry
    per output (BoTorch's check); other sequences of length 1 broadcast like
    torch."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    data, _ = to_flat(value, name)
    if not isinstance(value, list) and len(data) == 1:
        return float(data[0])
    return [float(v) for v in data]


def _add_noise(values: Any, std: Any, kind: ArrayKind) -> Any:
    """``values + std * N(0, 1)`` (``std`` a number or one per output) with
    the random number generator of the caller's array library."""
    if kind.lib == "torch":
        import torch

        scale = torch.as_tensor(std, dtype=values.dtype, device=values.device)
        return values + scale * torch.randn_like(values)
    import numpy as np

    noise = np.asarray(std, dtype=np.float64) * np.random.standard_normal(np.shape(values))
    return (values + noise).astype(values.dtype, copy=False)


class BaseTestProblem:
    """Base class of the test problems (BoTorch's ``BaseTestProblem``)."""

    _rs: Any
    _kind: ArrayKind
    noise_std: Any
    negate: bool
    #: Whether the problem is a minimization problem with ``negate=False``.
    _is_minimization_by_default: bool = True
    _check_grad_at_opt: bool = True

    @property
    def dim(self) -> int:
        """The input dimension."""
        return self._rs.dim

    @property
    def continuous_inds(self) -> list[int]:
        """The indices of the continuous inputs (all of them)."""
        return list(range(self.dim))

    @property
    def discrete_inds(self) -> list[int]:
        return []

    @property
    def categorical_inds(self) -> list[int]:
        return []

    @property
    def bounds(self) -> Any:
        """The ``2 x d`` bounds of the domain."""
        return from_pair(self._rs.bounds, self._kind)

    @property
    def _bounds(self) -> list[tuple[float, float]]:
        """BoTorch's ``_bounds``: one ``(lower, upper)`` pair per input."""
        data = getattr(self, "_float64_bounds", self._rs.bounds)[0]
        return list(zip(data[: self.dim], data[self.dim :], strict=True))

    @property
    def is_minimization_problem(self) -> bool:
        """Whether the problem is a minimization problem, after ``negate``."""
        return self._is_minimization_by_default != self.negate

    def evaluate_true(self, X: Any) -> Any:
        """Noiseless, un-negated values at ``X`` (``batch_shape x d``)."""
        return from_pair(self._rs.evaluate_true(to_flat(X, "X")), kind_of(X))

    def forward(self, X: Any, noise: bool = True) -> Any:
        """Values at ``X`` with observation noise (if ``noise``) and negation."""
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if kind.lib == "list" or not noise or self.noise_std is None:
            return from_pair(self._rs.forward(flat, noise), kind)
        f = _add_noise(from_pair(self._rs.evaluate_true(flat), kind), self.noise_std, kind)
        return -f if self.negate else f

    def __call__(self, X: Any, noise: bool = True) -> Any:
        return self.forward(X, noise=noise)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        """Sets the dtype and device of the arrays that are not computed from
        an input, like ``torch.nn.Module.to`` (``dtype=``, ``device=``, or a
        tensor whose dtype and device to use); returns the problem."""
        torch: Any = sys.modules.get("torch")
        dtype, device = kwargs.get("dtype"), kwargs.get("device")
        for arg in args:
            if torch is not None and isinstance(arg, torch.Tensor):
                dtype, device = arg.dtype, arg.device
            elif isinstance(arg, str) or (torch is not None and isinstance(arg, torch.device)):
                device = arg
            else:
                dtype = arg
        if device is not None:
            torch = torch or importlib.import_module("torch")
            device = torch.device(device)
        self._kind = kind_from_dtype(dtype if dtype is not None else self._kind.dtype, device)
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim}, negate={self.negate})"


class ConstrainedBaseTestProblem(BaseTestProblem):
    """Base class of the test problems with outcome constraints ``c_i(x) >=
    0`` (BoTorch's ``ConstrainedBaseTestProblem``)."""

    _check_grad_at_opt: bool = False
    _worst_feasible_value: float | None = None
    _constraint_noise_std: Any = None

    @property
    def num_constraints(self) -> int:
        """The number of constraints."""
        return self._rs.num_constraints

    @property
    def constraint_noise_std(self) -> Any:
        """The standard deviation of the observation noise of the slacks (a
        number or one per constraint)."""
        return self._constraint_noise_std

    @constraint_noise_std.setter
    def constraint_noise_std(self, value: Any) -> None:
        self._constraint_noise_std = value
        self._rebuild()

    def _rebuild(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def evaluate_slack_true(self, X: Any) -> Any:
        """Noiseless constraint slacks at ``X`` (``batch_shape x n_c``;
        feasible iff non-negative)."""
        return from_pair(self._rs.evaluate_slack_true(to_flat(X, "X")), kind_of(X))

    def evaluate_slack(self, X: Any, noise: bool = True) -> Any:
        """Constraint slacks at ``X`` with observation noise (if ``noise``)."""
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if kind.lib == "list" or not noise or self.constraint_noise_std is None:
            return from_pair(self._rs.evaluate_slack(flat, noise), kind)
        c = from_pair(self._rs.evaluate_slack_true(flat), kind)
        return _add_noise(c, self.constraint_noise_std, kind)

    def is_feasible(self, X: Any, noise: bool = True) -> Any:
        """Whether all (noisy, if ``noise``) slacks at each point of ``X`` are
        non-negative (``batch_shape`` booleans)."""
        kind = kind_of(X)
        if kind.lib == "list" or not noise or self.constraint_noise_std is None:
            flags, shape = self._rs.is_feasible(to_flat(X, "X"), noise)
            return bool_array(flags, shape, kind)
        return (self.evaluate_slack(X, noise=noise) >= 0.0).all(-1)

    @property
    def worst_feasible_value(self) -> float:
        """The worst feasible value: 0 (the smallest hypervolume) for
        multi-objective problems."""
        if isinstance(self, MultiObjectiveTestProblem):
            return 0.0
        if self._worst_feasible_value is not None:
            return -self._worst_feasible_value if self.negate else self._worst_feasible_value
        raise NotImplementedError(
            f"Problem {type(self).__name__} does not specify the worst feasible value."
        )


class MultiObjectiveTestProblem(BaseTestProblem):
    """Base class of the multi-objective test problems (BoTorch's
    ``MultiObjectiveTestProblem``), evaluated in Rust.

    Subclasses set ``_name`` (the BoTorch class name) and, if they take them,
    ``_size_args = (dim, num_objectives)`` and ``_constraint_noise_std``
    before calling ``__init__``. ``seed`` (keyword-only) seeds the noise of
    nested-list inputs.
    """

    _name: str = ""
    _size_args: tuple[int | None, int | None] = (None, None)

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        if not self._name:
            raise TypeError(
                f"Can't instantiate abstract class {type(self).__name__} without an "
                "implementation for abstract method '_evaluate_true'"
            )
        self._noise_std = noise_std
        self._negate = bool(negate)
        self._seed = seed
        self._kind: ArrayKind = kind_from_dtype(dtype)
        self._rebuild()
        # Like BoTorch's `ref_point` buffer: fixed at construction.
        self._ref_point_values: list[float] = list(self._rs.ref_point)

    def _rebuild(self) -> None:
        dim, num_objectives = self._size_args
        kwargs: dict[str, Any] = {
            "dim": dim,
            "num_objectives": num_objectives,
            "noise_std": _noise_arg(self._noise_std, "noise_std"),
            "constraint_noise_std": _noise_arg(
                getattr(self, "_constraint_noise_std", None), "constraint_noise_std"
            ),
            "negate": self._negate,
            "seed": self._seed,
        }
        self._rs = _rs.MultiObjectiveTestProblem(self._name, **kwargs)
        self._float64_bounds = self._rs.bounds
        # BoTorch validates inputs against its `bounds` buffer, which has the
        # problem's dtype: validate against the bounds rounded to it.
        rounded = to_flat(from_pair(self._rs.bounds, self._kind), "bounds")
        if list(rounded[0]) != list(self._rs.bounds[0]):
            self._rs = _rs.MultiObjectiveTestProblem(self._name, bounds=rounded, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        super().to(*args, **kwargs)
        self._rebuild()
        return self

    @property
    def noise_std(self) -> Any:
        """The standard deviation of the observation noise (a number or one
        per objective)."""
        return self._noise_std

    @noise_std.setter
    def noise_std(self, value: Any) -> None:
        self._noise_std = value
        self._rebuild()

    @property
    def negate(self) -> bool:  # type: ignore[override]
        """Whether the objectives are negated."""
        return self._negate

    @negate.setter
    def negate(self, value: bool) -> None:
        self._negate = bool(value)
        self._rebuild()

    @property
    def num_objectives(self) -> int:
        """The number of objectives."""
        return self._rs.num_objectives

    @property
    def ref_point(self) -> Any:
        """The reference point for hypervolume computations (negated if the
        problem was created with ``negate=True``)."""
        values = self._ref_point_values
        return from_flat(values, (len(values),), self._kind)

    @property
    def _ref_point(self) -> list[float]:
        return list(self._rs.raw_ref_point)

    @property
    def _max_hv(self) -> float | None:
        return self._rs.max_hv

    @property
    def max_hv(self) -> float:
        """The maximum hypervolume; ``NotImplementedError`` if it is not known."""
        value = self._rs.max_hv
        if value is None:
            raise NotImplementedError(
                f"Problem {type(self).__name__} does not specify maximal hypervolume."
            )
        return value

    @property
    def deviation(self) -> str | None:
        """How botorch_rs deviates from BoTorch main for this problem (the
        corrected definition and the BoTorch pull request), or ``None`` (a
        botorch_rs extension)."""
        return self._rs.deviation

    def gen_pareto_front(self, n: int, *, seed: int | None = None) -> Any:
        """``n`` Pareto optimal points (``n x M``, negated if ``negate``);
        ``NotImplementedError`` if the front is not known. Quasi-random fronts
        (DTLZ) use the keyword-only ``seed`` (random if ``None``)."""
        if not self._rs.has_pareto_front:
            raise NotImplementedError
        return from_pair(self._rs.gen_pareto_front(int(n), seed), self._kind)

    @property
    def is_minimization_problem(self) -> bool:
        raise UnsupportedError(
            "`is_minimization_problem` is not a valid property for multi-objective problems. "
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dim={self.dim}, num_objectives={self.num_objectives}, "
            f"negate={self.negate})"
        )


# -- validation, seeding and corrupted problems (phase 11a) ------------------


def _shape_repr(x: Any) -> str:
    shape = tuple(x.shape) if hasattr(x, "shape") else to_flat(x, "X")[1]
    return f"torch.Size({list(shape)})"


def validate_parameter_indices(
    dim: int,
    bounds: Any,
    continuous_inds: list[int],
    discrete_inds: list[int],
    categorical_inds: list[int],
) -> None:
    r"""Check that the parameter indices are valid.

    Args:
        dim: Number of search space dimensions.
        bounds: A ``2 x d``-dim tensor of lower and upper bounds.
        continuous_inds: List of unique integers corresponding to continuous parameters.
        discrete_inds: List of unique integers corresponding to discrete parameters.
        categorical_inds: List of unique integers corresponding to categorical
            parameters.

    Raises:
        ValueError: If the parameter indices are invalid.
    """
    for inds in [continuous_inds, discrete_inds, categorical_inds]:
        if len(inds) == 0:
            continue  # Nothing to check
        if not (
            isinstance(inds, list)
            and all(isinstance(x, int) for x in inds)
            and len(set(inds)) == len(inds)
            and min(inds) >= 0
            and max(inds) <= dim - 1
        ):
            raise ValueError(
                "All parameter indices must be a list with unique integers between "
                f"0 and dim - 1. Got {inds=}."
            )
    # Let's make sure all parameters are covered.
    all_inds = continuous_inds + discrete_inds + categorical_inds
    if (len(all_inds) != dim) or (set(all_inds) != set(range(dim))):
        raise ValueError(
            f"All parameter indices must be present, got {dim=}, {continuous_inds=}, "
            f"{discrete_inds=}, {categorical_inds=}"
        )
    # Check the shape of the bounds
    data, shape = to_flat(bounds, "bounds")
    if tuple(shape) != (2, dim):
        raise ValueError(
            f"Expected `bounds` to have shape `2 x d`. Got bounds.shape={_shape_repr(bounds)}, "
            f"{dim=}."
        )
    # Check that the bounds are integer valued for discrete and categorical parameters.
    for inds in [discrete_inds, categorical_inds]:
        vals = [float(data[r * dim + i]) for r in range(2) for i in inds]
        if len(inds) > 0 and not all(v == _round_half_even(v) for v in vals):
            raise ValueError(
                "Expected the lower and upper bounds of the discrete and categorical "
                "parameters to be integer-valued."
            )


def _round_half_even(v: float) -> float:
    import math

    if not math.isfinite(v):
        return v
    return float(round(v))


def validate_inputs(
    X: Any,
    dim: int,
    bounds: Any,
    discrete_inds: list[int],
    categorical_inds: list[int],
) -> None:
    r"""Check that the inputs are valid.

    This method checks that the input tensor ``X`` has the correct shape, is within
    the bounds, and that the discrete and categorical parameters are integer-valued.

    Args:
        X: A ``(batch_shape) x n x d``-dim tensor of point(s) at which to evaluate
        dim: Number of search space dimensions.
        bounds: A ``2 x d``-dim tensor of lower and upper bounds.
        discrete_inds: List of unique integers corresponding to discrete parameters.
        categorical_inds: List of unique integers corresponding to categorical
            parameters.

    Raises:
        ValueError: If the parameter indices are invalid.
    """
    data, shape = to_flat(X, "X")
    if not shape or shape[-1] != dim:
        raise ValueError(
            "Expected `X` to have shape `(batch_shape) x n x d`. "
            f"Got X.shape={_shape_repr(X)} and {dim=}"
        )
    bdata, _ = to_flat(bounds, "bounds")
    lower, upper = list(bdata[:dim]), list(bdata[dim:])
    n = len(data) // dim if dim else 0
    for r in range(n):
        for k in range(dim):
            v = float(data[r * dim + k])
            if not (v >= lower[k] and v <= upper[k]):
                raise ValueError("Expected `X` to be within the bounds of the test problem.")
    for inds in [discrete_inds, categorical_inds]:
        for r in range(n):
            for k in inds:
                v = float(data[r * dim + k])
                if v != _round_half_even(v):
                    raise ValueError(
                        "Expected `X` to have integer values for the discrete and "
                        "categorical parameters."
                    )


class SeedingMixin:
    """A mixin of problems that iterate through a sequence of seeds
    (BoTorch's ``SeedingMixin``)."""

    _seeds: Any = None
    _current_seed: int | None = None

    @property
    def has_seeds(self) -> bool:
        return self._seeds is not None

    def increment_seed(self) -> int:
        if self._seeds is None:
            raise ValueError("Unexpected `None` found.")
        self._current_seed = next(self._seeds)
        return self._current_seed  # type: ignore[return-value]

    @property
    def seed(self) -> int | None:
        return self._current_seed


class OutlierGenerator(Protocol):
    def __call__(self, problem: Any, X: Any, bounds: Any) -> Any:
        """Call signature for outlier generators for single-objective problems.

        Args:
            problem: The test problem.
            X: The input tensor.
            bounds: The bounds of the test problem.

        Returns:
            A tensor of outliers with shape X.shape[:-1] (1d if unbatched).
        """
        pass  # pragma: no cover


def constant_outlier_generator(problem: Any, X: Any, bounds: Any, constant: float) -> Any:
    """
    Generates outliers that are all the same constant. To be used in conjunction
    with ``partial`` to fix the constant value and conform to the
    ``OutlierGenerator`` protocol.

    Example:
        >>> generator = partial(constant_outlier_generator, constant=1.0)

    Args:
        problem: Not used.
        X: The ``batch_shape x n x d``-dim inputs. Also determines the number, dtype,
            and device of the returned tensor.
        bounds: Not used.
        constant: The constant value of the outliers.

    Returns:
        Tensor of shape ``batch_shape x n`` (1d if unbatched).
    """
    kind = kind_of(X)
    shape = to_flat(X, "X")[1][:-1]
    import math

    return from_flat([float(constant)] * math.prod(shape), shape, kind)


class CorruptedTestProblem(BaseTestProblem, SeedingMixin):
    def __init__(
        self,
        base_test_problem: Any,
        outlier_generator: Any,
        outlier_fraction: float,
        bounds: list[tuple[float, float]] | None = None,
        seeds: Any = None,
    ) -> None:
        """A problem with outliers.

        NOTE: Both noise_std and negate will be taken from the base test problem.

        Args:
            base_test_problem: The base function to be corrupted.
            outlier_generator: A function that generates outliers. It will be called
                with arguments ``f``, ``X`` and ``bounds``, where ``f`` is the
                ``base_test_problem``, ``X`` is the
                argument passed to the ``forward`` method, and ``bounds``
                are as here, and it returns the values of outliers.
            outlier_fraction: The fraction of outliers.
            bounds: The bounds of the function.
            seeds: The seeds to use for the outlier generator. If seeds are provided,
                the problem will iterate through the list of seeds, changing the seed
                with a call to ``next(seeds)`` with every ``forward`` call. If a list is
                provided, it will first be converted to an iterator.

        The outlier mask is drawn with torch's generator (``torch.rand`` in
        torch's default dtype after ``torch.manual_seed(seed)`` if seeds are
        given, as in BoTorch) for torch inputs, else with Python's
        ``random`` (seeded with the current seed).
        """
        self._dim = base_test_problem.dim
        self.continuous_inds_ = list(base_test_problem.continuous_inds)
        self.discrete_inds_ = list(base_test_problem.discrete_inds)
        self.categorical_inds_ = list(base_test_problem.categorical_inds)
        self._bounds_list: list[tuple[float, float]] = (
            list(bounds) if bounds is not None else list(base_test_problem._bounds)
        )
        if len(self._bounds_list) != self._dim:
            from botorch_rs.exceptions.errors import InputDataError

            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got self.dim={self._dim} and len(self._bounds)={len(self._bounds_list)}."
            )
        self.noise_std = base_test_problem.noise_std
        self.negate = base_test_problem.negate
        self._kind = kind_from_dtype(None)
        self.base_test_problem = base_test_problem
        self.outlier_generator = outlier_generator
        self.outlier_fraction = outlier_fraction
        self._current_seed: int | None = None
        self._seeds = None if seeds is None else iter(seeds)
        validate_parameter_indices(
            dim=self._dim,
            bounds=self.bounds,
            continuous_inds=self.continuous_inds,
            discrete_inds=self.discrete_inds,
            categorical_inds=self.categorical_inds,
        )

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim

    @property
    def continuous_inds(self) -> list[int]:  # type: ignore[override]
        return self.continuous_inds_

    @property
    def discrete_inds(self) -> list[int]:  # type: ignore[override]
        return self.discrete_inds_

    @property
    def categorical_inds(self) -> list[int]:  # type: ignore[override]
        return self.categorical_inds_

    @property
    def _bounds(self) -> list[tuple[float, float]]:  # type: ignore[override]
        return self._bounds_list

    @property
    def bounds(self) -> Any:  # type: ignore[override]
        lo = [float(b[0]) for b in self._bounds_list]
        hi = [float(b[1]) for b in self._bounds_list]
        return from_flat(lo + hi, (2, self._dim), self._kind)

    def _evaluate_true(self, X: Any) -> Any:
        return self.base_test_problem.evaluate_true(X)

    def evaluate_true(self, X: Any) -> Any:
        validate_inputs(X, self.dim, self.bounds, self.discrete_inds, self.categorical_inds)
        return self._evaluate_true(X)

    def forward(self, X: Any, noise: bool = True) -> Any:
        """
        Generate data at X and corrupt it, if noise is True.

        Args:
            X: The ``batch_shape x n x d``-dim inputs.
            noise: Whether to corrupt the data.

        Returns:
            A ``batch_shape x n``-dim tensor.
        """
        kind = kind_of(X)
        f = self.evaluate_true(X)
        if noise and self.noise_std is not None:
            if kind.lib == "list":
                import random

                fl, sh = to_flat(f, "f")
                f = from_flat([v + float(self.noise_std) * random.gauss(0.0, 1.0) for v in fl], sh, kind)
            else:
                f = _add_noise(f, self.noise_std, kind)
        if self.negate:
            fl, sh = to_flat(f, "f")
            f = -f if kind.lib != "list" else from_flat([-v for v in fl], sh, kind)
        if not noise:
            return f
        if self.has_seeds:
            self.increment_seed()
        shape = to_flat(X, "X")[1][:-1]
        if kind.lib == "torch":
            import torch

            if self.has_seeds:
                torch.manual_seed(self.seed)
            corrupt = torch.rand(shape) < self.outlier_fraction
            outliers = self.outlier_generator(problem=self.base_test_problem, X=X, bounds=self.bounds)
            return torch.where(corrupt.to(X.device), outliers, f)
        import math
        import random

        rng = random.Random(self.seed) if self.has_seeds else random
        corrupt_l = [rng.random() < self.outlier_fraction for _ in range(math.prod(shape))]
        outliers = self.outlier_generator(problem=self.base_test_problem, X=X, bounds=self.bounds)
        fo, _ = to_flat(outliers, "outliers")
        fv, sh = to_flat(f, "f")
        vals = [float(o) if c else float(v) for c, o, v in zip(corrupt_l, fo, fv, strict=True)]
        return from_flat(vals, sh, kind)

    def __repr__(self) -> str:
        return f"CorruptedTestProblem(base_test_problem={self.base_test_problem!r})"


__all__ = [
    "BaseTestProblem",
    "ConstrainedBaseTestProblem",
    "CorruptedTestProblem",
    "MultiObjectiveTestProblem",
    "OutlierGenerator",
    "SeedingMixin",
    "constant_outlier_generator",
    "validate_inputs",
    "validate_parameter_indices",
]
