"""Synthetic test functions (``botorch.test_functions.synthetic``).

``problem(X)`` (``forward``) evaluates ``X`` (``batch_shape x d``), adds
Gaussian observation noise with standard deviation ``noise_std`` (if set and
``noise=True``) and negates the result if ``negate=True``; ``evaluate_true``
returns the noiseless, un-negated values. Constants that BoTorch stores in
float32 buffers are rounded identically, so values agree to round-off.

The noise is drawn with the caller's array library (``torch.randn_like`` for
torch inputs, NumPy's global generator for NumPy inputs, a seeded Rust
generator for lists). ``bounds``, ``optimizers`` and ``optimal_value`` use
``dtype``: a torch dtype gives tensors, a NumPy dtype arrays, and ``None``
float64 tensors if torch is imported (else lists).
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, from_flat, from_pair, kind_from_dtype, kind_of, to_flat
from botorch_rs.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem


def _bounds_arg(bounds: Any) -> Any:
    """BoTorch's ``bounds`` argument (``d`` pairs ``(lower, upper)``) as a
    flat ``2 x d`` array."""
    if bounds is None:
        return None
    data, shape = to_flat(bounds, "bounds")
    if len(shape) != 2 or shape[1] != 2:
        raise ValueError("`bounds` must be a list of (lower, upper) pairs.")
    d = shape[0]
    lower = [data[2 * i] for i in range(d)]
    upper = [data[2 * i + 1] for i in range(d)]
    return (lower + upper, (2, d))


class SyntheticTestFunction(BaseTestProblem):
    """Base class of the synthetic test functions."""

    num_objectives = 1
    _name: str = ""
    _fixed_dim: int | None = None

    def __init__(
        self,
        dim: int | None = None,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        m: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._rs = _rs.SyntheticTestFunction(
            self._name,
            dim=None if self._fixed_dim is not None else dim,
            m=m,
            noise_std=None if noise_std is None else float(noise_std),
            negate=bool(negate),
            bounds=_bounds_arg(bounds),
            seed=seed,
        )
        self._kind: ArrayKind = kind_from_dtype(dtype)
        self.noise_std = noise_std
        self.negate = bool(negate)

    @property
    def dim(self) -> int:
        return self._rs.dim

    @property
    def continuous_inds(self) -> list[int]:
        return list(range(self.dim))

    @property
    def bounds(self) -> Any:
        """The ``2 x d`` bounds of the domain."""
        return from_pair(self._rs.bounds, self._kind)

    @property
    def optimal_value(self) -> float:
        """The optimal value (negated if ``negate``); ``NotImplementedError`` if
        it is not known."""
        value = self._rs.optimal_value
        if value is None:
            raise NotImplementedError(
                f"Problem {type(self).__name__} does not specify an optimal value."
            )
        return value

    @property
    def optimizers(self) -> Any:
        """The known global optimizers (one per row); ``NotImplementedError``
        if they are not known."""
        opt = self._rs.optimizers
        if opt is None:
            raise NotImplementedError(f"Problem {type(self).__name__} does not specify optimizers.")
        return from_pair(opt, self._kind)

    @property
    def is_minimization_problem(self) -> bool:
        return self._rs.is_minimization_problem

    def evaluate_true(self, X: Any) -> Any:
        """Noiseless, un-negated values at ``X`` (``batch_shape x d``)."""
        return from_pair(self._rs.evaluate_true(to_flat(X, "X")), kind_of(X))

    def forward(self, X: Any, noise: bool = True) -> Any:
        """Values at ``X`` with observation noise (if ``noise``) and negation."""
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if kind.lib == "list" or not noise or self.noise_std is None:
            return from_pair(self._rs.forward(flat, noise), kind)
        f = from_pair(self._rs.evaluate_true(flat), kind)
        if kind.lib == "torch":
            import torch

            f = f + float(self.noise_std) * torch.randn_like(f)
        else:
            import numpy as np

            f = f + float(self.noise_std) * np.random.standard_normal(np.shape(f))
        return -f if self.negate else f

    def __call__(self, X: Any, noise: bool = True) -> Any:
        return self.forward(X, noise=noise)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim}, negate={self.negate})"


class Ackley(SyntheticTestFunction):
    """``d``-dimensional Ackley function on ``[-32.768, 32.768]^d``; minimum 0 at the origin."""

    _name = "Ackley"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class Beale(SyntheticTestFunction):
    """Beale function on ``[-4.5, 4.5]^2``; minimum 0 at ``(3, 0.5)``."""

    _name = "Beale"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class Branin(SyntheticTestFunction):
    """Branin function on ``[-5, 10] x [0, 15]``; minimum 0.397887 at three points."""

    _name = "Branin"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class Bukin(SyntheticTestFunction):
    """Bukin N.6 function on ``[-15, -5] x [-3, 3]``; minimum 0 at ``(-10, 1)``."""

    _name = "Bukin"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class Cosine8(SyntheticTestFunction):
    """Cosine mixture on ``[-1, 1]^8`` (a maximization problem); maximum 0.8 at the origin."""

    _name = "Cosine8"
    _fixed_dim = 8

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class DixonPrice(SyntheticTestFunction):
    """``d``-dimensional Dixon-Price function on ``[-10, 10]^d``; minimum 0."""

    _name = "DixonPrice"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class DropWave(SyntheticTestFunction):
    """Drop-wave function on ``[-5.12, 5.12]^2``; minimum -1 at the origin."""

    _name = "DropWave"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class EggHolder(SyntheticTestFunction):
    """Eggholder function on ``[-512, 512]^2``; minimum -959.6407."""

    _name = "EggHolder"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class Griewank(SyntheticTestFunction):
    """``d``-dimensional Griewank function on ``[-600, 600]^d``; minimum 0 at the origin."""

    _name = "Griewank"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class Hartmann(SyntheticTestFunction):
    """Hartmann function (``d`` in 3, 4, 6) on ``[0, 1]^d``; minimum -3.32237 for ``d = 6``."""

    _name = "Hartmann"

    def __init__(
        self,
        dim: int = 6,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class HolderTable(SyntheticTestFunction):
    """Hoelder table function on ``[-10, 10]^2``; minimum -19.2085 at four points."""

    _name = "HolderTable"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class Levy(SyntheticTestFunction):
    """``d``-dimensional Levy function on ``[-10, 10]^d``; minimum 0 at ``(1, ..., 1)``."""

    _name = "Levy"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class Michalewicz(SyntheticTestFunction):
    """``d``-dimensional Michalewicz function (``m = 10``) on ``[0, pi]^d``."""

    _name = "Michalewicz"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class Powell(SyntheticTestFunction):
    """``d``-dimensional Powell function on ``[-4, 5]^d``; minimum 0 at the origin."""

    _name = "Powell"

    def __init__(
        self,
        dim: int = 4,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class Rastrigin(SyntheticTestFunction):
    """``d``-dimensional Rastrigin function on ``[-5.12, 5.12]^d``; minimum 0 at the origin."""

    _name = "Rastrigin"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class Rosenbrock(SyntheticTestFunction):
    """``d``-dimensional Rosenbrock function on ``[-5, 10]^d``; minimum 0 at ``(1, ..., 1)``."""

    _name = "Rosenbrock"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class SixHumpCamel(SyntheticTestFunction):
    """Six-hump camel function on ``[-3, 3] x [-2, 2]``; minimum -1.0316 at two points."""

    _name = "SixHumpCamel"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class StyblinskiTang(SyntheticTestFunction):
    """``d``-dimensional Styblinski-Tang function on ``[-5, 5]^d``; minimum -39.166166 d."""

    _name = "StyblinskiTang"

    def __init__(
        self,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class ThreeHumpCamel(SyntheticTestFunction):
    """Three-hump camel function on ``[-5, 5]^2``; minimum 0 at the origin."""

    _name = "ThreeHumpCamel"
    _fixed_dim = 2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, seed=seed)


class Shekel(SyntheticTestFunction):
    """BoTorch's ``Shekel`` test function (4-dimensional, ``m`` terms)."""

    _name = "Shekel"
    _fixed_dim = 4

    def __init__(
        self,
        m: int = 10,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(None, noise_std, negate, bounds, dtype, m=m, seed=seed)
        self.m = m



# -- Problems of phase 11a: mixed, binary, trajectory and constrained ---------


class _NativeProblem(SyntheticTestFunction):
    """Base of the problems evaluated by ``_botorch_rs.SingleObjectiveProblem``
    (BoTorch's ``SyntheticTestFunction`` semantics: ``forward`` adds noise
    with the caller's array library and negates; ``optimal_value`` and
    ``optimizers`` come from ``_optimal_value`` and ``_optimizers``)."""

    _optimal_value: float | None = None
    _optimizers: list[tuple[float, ...]] | None = None
    _is_minimization_by_default: bool = True
    _seed: int | None = None

    def _setup(
        self,
        dim: int,
        bounds: list[tuple[float, float]],
        noise_std: Any,
        negate: bool,
        dtype: Any,
        discrete: list[int] | None = None,
        params: list[float] | None = None,
        custom_bounds: list[tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> None:
        from botorch_rs.exceptions.errors import InputDataError
        from botorch_rs.test_functions.base import validate_parameter_indices

        if custom_bounds is not None:
            bounds = list(custom_bounds)
        if len(bounds) != dim:
            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got self.dim={dim} and len(self._bounds)={len(bounds)}."
            )
        self._bounds_list = [(float(lo), float(hi)) for lo, hi in bounds]
        self.noise_std = noise_std
        self.negate = bool(negate)
        self._kind = kind_from_dtype(dtype)
        lo = [b[0] for b in self._bounds_list]
        hi = [b[1] for b in self._bounds_list]
        self.discrete_inds = list(discrete or [])
        if "continuous_inds" not in self.__dict__:
            self.continuous_inds = [i for i in range(dim) if i not in self.discrete_inds]
        validate_parameter_indices(
            dim=dim,
            bounds=from_flat(lo + hi, (2, dim), kind_from_dtype(None)),
            continuous_inds=self.continuous_inds,
            discrete_inds=self.discrete_inds,
            categorical_inds=self.categorical_inds,
        )
        self._rs = _rs.SingleObjectiveProblem(
            self._name,
            dim,
            (lo + hi, (2, dim)),
            self.discrete_inds,
            [float(v) for v in (params or [])],
            **kwargs,
        )
        if self._optimizers is not None and custom_bounds is not None:
            if not any(
                all(lo_ <= xo <= hi_ for xo, (lo_, hi_) in zip(opt, self._bounds_list, strict=True))
                for opt in self._optimizers
            ):
                raise ValueError(
                    "No global optimum found within custom bounds. Please specify "
                    "bounds which include at least one point in "
                    f"`{type(self).__name__}._optimizers`."
                )

    # BoTorch's class attributes (set per instance by the subclasses).
    continuous_inds: list[int] = []  # type: ignore[assignment]  # noqa: RUF012
    discrete_inds: list[int] = []  # type: ignore[assignment]  # noqa: RUF012

    @property
    def categorical_inds(self) -> list[int]:
        return []

    @property
    def _bounds(self) -> list[tuple[float, float]]:  # type: ignore[override]
        return list(self._bounds_list)

    @property
    def optimal_value(self) -> float:
        value = self._optimal_value
        if value is None:
            raise NotImplementedError(
                f"Problem {type(self).__name__} does not specify an optimal value."
            )
        return -value if self.negate else value

    @property
    def optimizers(self) -> Any:
        if self._optimizers is None:
            raise AttributeError(f"{type(self).__name__} has no attribute 'optimizers'")
        rows = [list(map(float, o)) for o in self._optimizers]
        return from_flat([v for r in rows for v in r], (len(rows), self.dim), self._kind)

    @property
    def is_minimization_problem(self) -> bool:
        return self._is_minimization_by_default != self.negate

    def forward(self, X: Any, noise: bool = True) -> Any:
        """Values at ``X`` with observation noise (if ``noise``) and negation."""
        kind = kind_of(X)
        f = self.evaluate_true(X)
        if noise and self.noise_std is not None:
            if kind.lib == "list":
                import random

                rng = random.Random(self._seed) if self._seed is not None else random
                data, shape = to_flat(f, "f")
                f = from_pair(
                    ([v + float(self.noise_std) * rng.gauss(0.0, 1.0) for v in data], shape), kind
                )
            else:
                from botorch_rs.test_functions.base import _add_noise

                f = _add_noise(f, self.noise_std, kind)
        if self.negate:
            if kind.lib == "list":
                data, shape = to_flat(f, "f")
                return from_pair(([-v for v in data], shape), kind)
            return -f
        return f

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim}, negate={self.negate})"


class AckleyMixed(_NativeProblem):
    r"""Mixed search space version of the Ackley problem.

    This problem has dim-3 binary parameters and 3 continuous parameters in the
    range [0, 1]. This means dim > 3 is required. To make the problem a bit more
    interesting, the optimal value is not at the origin, but rather at ``x_opt``
    which is a randomly generated point.

    The goal is  to minimize f(x) = Ackley(x - x_opt).

    With ``randomize_optimum``, ``x_opt`` is drawn with ``torch.rand`` (torch's
    global generator, as in BoTorch) if torch is available, else with
    Python's ``random``.
    """

    _name = "AckleyMixed"
    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 53,
        noise_std: float | None = None,
        negate: bool = False,
        randomize_optimum: bool = False,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension. Must be > 3.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            randomize_optimum: If True, the optimum is a random point in the domain.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim <= 3:
            raise ValueError(f"Expected dim > 3. Got {dim=}.")
        if randomize_optimum:
            import sys as _sys

            torch = _sys.modules.get("torch")
            if torch is not None:
                tdt = dtype if dtype is not None and isinstance(dtype, torch.dtype) else torch.float64
                x_opt = torch.rand(dim, dtype=tdt).tolist()
            else:
                import random

                x_opt = [random.random() for _ in range(dim)]
            x_opt = [float(round(v)) if i < dim - 3 else float(v) for i, v in enumerate(x_opt)]
        else:
            x_opt = [0.0] * dim
        self._x_opt = x_opt
        self._optimizers = [tuple(x_opt)]
        self.continuous_inds = list(range(dim - 3, dim))
        self._setup(
            dim,
            [(0.0, 1.0)] * dim,
            noise_std,
            negate,
            dtype,
            discrete=list(range(0, dim - 3)),
            params=x_opt,
        )

    @property
    def x_opt(self) -> Any:
        """The optimizer of the problem."""
        return from_flat(list(self._x_opt), (self.dim,), self._kind)


class Labs(_NativeProblem):
    r"""Low Auto-correlation Binary Sequences (LABS) problem.

    This input space is binary and the goal is to maximize the Merit factor.
    [Packebusch2016]_ provides optimal values and optimizers attained through
    brute-force for dim <= 66. We include these for dim = 10, 20, 30, 40, 50, 60.
    """

    _name = "Labs"
    _check_grad_at_opt = False
    _is_minimization_by_default = False

    def __init__(
        self,
        dim: int = 30,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        optvals = {10: 3.846, 20: 7.692, 30: 7.627, 40: 7.407, 50: 8.170, 60: 8.257}
        optimizers = {
            10: "42211",
            20: "5113112321",
            30: "551212111113231",
            40: "44412112131121313131",
            50: "215131311224112241141142",
            60: "761112141111131124211322211222",
        }
        self._optimal_value = optvals.get(dim)
        seq = optimizers.get(dim)
        self._optimizers = None if seq is None else self._optimizer_from_binary_seq(seq)
        self.continuous_inds = []
        self._setup(
            dim, [(0.0, 1.0)] * dim, noise_std, negate, dtype, discrete=list(range(dim))
        )

    def _optimizer_from_binary_seq(self, seq: str) -> list[tuple[float, ...]]:
        """Converts a binary sequence into a an array."""
        arr: list[float] = []
        val = 0
        for c in seq:
            arr += [float(val) for _ in range(int(c))]
            val = 1 - val  # alternate between 0 and 1
        return [tuple(arr)]


class TrajectoryPlanning(_NativeProblem):
    r"""Trajectory optimization benchmark for navigating through obstacle fields.

    This test function optimizes a trajectory from a fixed start point to a goal
    point while minimizing the cost integrated over the trajectory.

    Problem Setup:
        - Domain: [0, 1]^dim hypercube (dim must be even)
        - Start: (0.05, 0.05)
        - Goal: (0.95, 0.95)
        - Obstacles: Grid of axis-aligned squares centered at regular intervals

    Parameterization:
        Parameters specify perturbations to goal-directed steps. This biases
        the search toward trajectories that make progress toward the goal
        while allowing deviations to avoid obstacles.

    Interpolation Modes:
        - Cubic spline (use_smooth_interp=True, default): Smooth C2 trajectories
        - Piecewise linear (use_smooth_interp=False): Straight-line segments

    Cost Function:
        The cost is computed by integrating the arc length of the trajectory
        that passes through obstacles:

        cost = integral over trajectory of (20 if in_obstacle else 0) ds

    The trajectory, the interpolating cubic spline (SciPy's ``splprep`` with
    ``k=3, s=0``: chord-length parameters and FITPACK's interpolation knots,
    evaluated at torch's float32 ``linspace``) and the cost are computed in
    Rust, without SciPy.

    Example:
        >>> problem = TrajectoryPlanning(dim=20)
        >>> x = torch.rand(problem.dim)  # Random trajectory parameters
        >>> cost = problem(x)  # Evaluate obstacle cost
    """

    _name = "TrajectoryPlanning"
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 30,
        use_smooth_interp: bool = True,
        max_step_size: float | None = None,
        negate: bool = False,
        dtype: Any = None,
    ) -> None:
        """Initialize the trajectory planning problem.

        Args:
            dim: Dimensionality of the optimization problem. Must be even since
                each waypoint has 2 coordinates.
            use_smooth_interp: If True, use cubic spline interpolation for
                smooth trajectories. If False, use piecewise linear interpolation.
            max_step_size: Maximum allowed step size. If None, automatically computed.
            negate: If True, negate the cost (for maximization problems).
            dtype: Data type for tensors.

        Raises:
            ValueError: If dim is not even.
        """
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        import math

        self.num_waypoints = dim // 2
        self.use_smooth_interp = use_smooth_interp
        straight_line_dist = math.sqrt(0.9**2 + 0.9**2)
        self.max_step_size = max_step_size or 1.5 * straight_line_dist / self.num_waypoints
        self.obs_cost = 20
        self.continuous_inds = list(range(dim))
        self._setup(
            dim,
            [(0.0, 1.0)] * dim,
            None,
            negate,
            dtype,
            use_smooth_interp=bool(use_smooth_interp),
            max_step_size=float(self.max_step_size),
        )

    def evaluate_true(self, X: Any) -> Any:
        """Noiseless costs at ``X`` (``batch_shape x d``; a single point
        ``d`` gives a 1-element result, as in BoTorch)."""
        kind = kind_of(X)
        data, shape = to_flat(X, "X")
        if len(shape) == 1:
            out = self._rs.evaluate_true((data, (1, *shape)))
            return from_pair(out, kind)
        return from_pair(self._rs.evaluate_true((data, shape)), kind)


class ConstrainedSyntheticTestFunction(_NativeProblem, ConstrainedBaseTestProblem):
    r"""Base class for constrained synthetic test functions."""

    num_constraints: int = 0  # type: ignore[assignment]

    def _validate_constraint_noise(self, constraint_noise_std: Any) -> Any:
        """
        Validates that constraint_noise_std has length equal to
        the number of constraints, if given as a list

        Args:
            constraint_noise_std: Standard deviation of the constraint noise.
                If a list is provided, specifies separate noise standard
                deviations for each constraint.
        """
        if (
            isinstance(constraint_noise_std, list)
            and len(constraint_noise_std) != self.num_constraints
        ):
            from botorch_rs.exceptions.errors import InputDataError

            raise InputDataError(
                "If specified as a list, length of constraint_noise_std "
                f"({len(constraint_noise_std)}) must match the "
                f"number of constraints ({self.num_constraints})"
            )
        return constraint_noise_std

    @property
    def constraint_noise_std(self) -> Any:  # type: ignore[override]
        return self._constraint_noise_std

    @constraint_noise_std.setter
    def constraint_noise_std(self, value: Any) -> None:
        self._constraint_noise_std = value

    def evaluate_slack(self, X: Any, noise: bool = True) -> Any:
        """Constraint slacks at ``X`` with observation noise (if ``noise``)."""
        kind = kind_of(X)
        c = self.evaluate_slack_true(X)
        if not noise or self.constraint_noise_std is None:
            return c
        if kind.lib == "list":
            import random

            std = self.constraint_noise_std
            data, shape = to_flat(c, "slack")
            nc = shape[-1]
            stds = list(std) if isinstance(std, (list, tuple)) else [float(std)] * nc
            vals = [v + stds[i % nc] * random.gauss(0.0, 1.0) for i, v in enumerate(data)]
            return from_pair((vals, shape), kind)
        from botorch_rs.test_functions.base import _add_noise

        return _add_noise(c, self.constraint_noise_std, kind)

    def is_feasible(self, X: Any, noise: bool = True) -> Any:
        """Whether all (noisy, if ``noise``) slacks at each point of ``X`` are
        non-negative (``batch_shape`` booleans)."""
        kind = kind_of(X)
        c = self.evaluate_slack(X, noise=noise)
        if kind.lib == "list":
            data, shape = to_flat(c, "slack")
            nc = shape[-1]
            flags = [all(v >= 0 for v in data[i : i + nc]) for i in range(0, len(data), nc)]
            from botorch_rs._arrays import bool_array

            return bool_array(flags, shape[:-1], kind)
        return (c >= 0.0).all(-1)


class ConstrainedGramacy(ConstrainedSyntheticTestFunction):
    r"""Constrained Gramacy test function.

    This problem comes from [Gramacy2016]_. The problem is defined
    over the unit cube and the goal is to minimize x1+x2 subject to
    1.5 - x1 - 2 * x2 - 0.5 * sin(2*pi*(x1^2 - 2 * x2)) <= 0
    and x1^2 + x2^2 - 1.5 <= 0.
    """

    _name = "ConstrainedGramacy"
    num_objectives = 1
    num_constraints = 2
    _optimizers = [(0.1954, 0.4044)]
    _optimal_value = 0.5998  # approximate from [Gramacy2016]_
    _worst_feasible_value = 1.732051  # Computed from 100 SLSQP restarts

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        self.continuous_inds = [0, 1]
        self._setup(2, [(0.0, 1.0), (0.0, 1.0)], noise_std, negate, dtype, custom_bounds=bounds)
        self.constraint_noise_std = self._validate_constraint_noise(constraint_noise_std)


class ConstrainedHartmann(ConstrainedSyntheticTestFunction):
    r"""Constrained Hartmann test function.

    This is a constrained version of the standard Hartmann test function that
    uses ``||x||_2 <= 1`` as the constraint. This problem comes from [Letham2019]_.
    """

    _name = "ConstrainedHartmann"
    num_constraints = 1

    def __init__(
        self,
        dim: int = 6,
        noise_std: None | float = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            constraint_noise_std: Standard deviation of the constraint noise.
                If a list is provided, specifies separate noise standard
                deviations for each constraint.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim not in (3, 4, 6):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        optvals = {3: -3.86278, 6: -3.32237}
        optimizers = {
            3: [(0.114614, 0.555649, 0.852547)],
            6: [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)],
        }
        self._optimal_value = optvals.get(dim)
        self._optimizers = optimizers.get(dim)
        self.continuous_inds = list(range(dim))
        self._setup(dim, [(0.0, 1.0)] * dim, noise_std, negate, dtype, custom_bounds=bounds)
        self.constraint_noise_std = self._validate_constraint_noise(constraint_noise_std)
        if dim == 3:
            self._worst_feasible_value = -0.0002735  # Computed from 100 SLSQP restarts
        elif dim == 6:
            self._worst_feasible_value = -0.0001346  # Computed from 100 SLSQP restarts

    @property
    def optimizers(self) -> Any:
        if self.dim == 4:
            raise NotImplementedError()
        return super().optimizers


class ConstrainedHartmannSmooth(ConstrainedHartmann):
    r"""Smooth constrained Hartmann test function.

    This is a constrained version of the standard Hartmann test function that
    uses ``||x||_2^2 <= 1`` as the constraint to obtain smoother constraint slack.
    """

    _name = "ConstrainedHartmannSmooth"


class PressureVessel(ConstrainedSyntheticTestFunction):
    r"""Pressure vessel design problem with constraints.

    The four-dimensional pressure vessel design problem with four black-box
    constraints from [CoelloCoello2002constraint]_.
    """

    _name = "PressureVessel"
    num_constraints = 4
    _optimal_value = 6059.946341  # from [CoelloCoello2002constraint]
    _worst_feasible_value = 240526.7248  # Computed from 100 SLSQP restarts

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        self.continuous_inds = list(range(4))
        self._setup(
            4,
            [(0.0, 10.0), (0.0, 10.0), (10.0, 50.0), (150.0, 200.0)],
            noise_std,
            negate,
            dtype,
            custom_bounds=bounds,
        )
        self.constraint_noise_std = self._validate_constraint_noise(constraint_noise_std)


class WeldedBeamSO(ConstrainedSyntheticTestFunction):
    r"""Welded beam design problem with constraints (single-outcome).

    The four-dimensional welded beam design problem with six
    black-box constraints from [CoelloCoello2002constraint]_.

    For a (somewhat modified) multi-objective version, see
    ``botorch.test_functions.multi_objective.WeldedBeam``.
    """

    _name = "WeldedBeamSO"
    num_constraints = 6
    _optimal_value = 1.728226  # from [CoelloCoello2002constraint]
    _worst_feasible_value = 19.01859  # Computed from 100 SLSQP restarts

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        self.continuous_inds = list(range(4))
        self._setup(
            4,
            [(0.125, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)],
            noise_std,
            negate,
            dtype,
            custom_bounds=bounds,
        )
        self.constraint_noise_std = self._validate_constraint_noise(constraint_noise_std)


class TensionCompressionString(ConstrainedSyntheticTestFunction):
    r"""Tension compression string optimization problem with constraints.

    The three-dimensional tension compression string optimization problem with
    four black-box constraints from [Hedar2006derivfree]_.
    """

    _name = "TensionCompressionString"
    num_constraints = 4
    _optimal_value = 0.012681  # from [CoelloCoello2002constraint]
    _worst_feasible_value = 0.306081  # Computed from 100 SLSQP restarts

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        self.continuous_inds = list(range(3))
        self._setup(
            3,
            [(0.01, 1.0), (0.01, 1.0), (0.01, 20.0)],
            noise_std,
            negate,
            dtype,
            custom_bounds=bounds,
        )
        self.constraint_noise_std = self._validate_constraint_noise(constraint_noise_std)


class SpeedReducer(ConstrainedSyntheticTestFunction):
    r"""Speed Reducer design problem with constraints.

    The seven-dimensional speed reducer design problem with eleven black-box
    constraints from [Lemonge2010constrained]_.
    """

    _name = "SpeedReducer"
    num_constraints = 11
    _optimal_value = 2996.3482  # from [Lemonge2010constrained]
    _worst_feasible_value = 6117.3420  # Computed from 100 SLSQP restarts

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        self.continuous_inds = list(range(7))
        self._setup(
            7,
            [
                (2.6, 3.6),
                (0.7, 0.8),
                (17.0, 28.0),
                (7.3, 8.3),
                (7.8, 8.3),
                (2.9, 3.9),
                (5.0, 5.5),
            ],
            noise_std,
            negate,
            dtype,
            custom_bounds=bounds,
        )
        self.constraint_noise_std = self._validate_constraint_noise(constraint_noise_std)


class KeaneBumpFunction(ConstrainedSyntheticTestFunction):
    r"""Keane Bump Function problem with constraints.

    This is a challenging d-dimensional minimization problem with two
    constraints that is evaluated on the domain [0, 10]^d.

    There is no known global optimum for this problem, but the approximate
    global optimal value for a few different dimensionalities can be found in
    [Mishra2007]_.
    """

    _name = "KeaneBumpFunction"
    num_constraints = 2
    _optimal_value_lookup = {
        2: -0.365,
        10: -0.6737,
        15: -0.781647601,
        20: -0.803619104,
        30: -0.818056222,
        40: -0.826624404,
        50: -0.83078783,
    }
    _worst_feasible_value = 0.0  # Computed from 100 SLSQP restarts

    def __init__(
        self,
        dim: int,
        noise_std: None | float = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            constraint_noise_std: Standard deviation of the constraint noise.
                If a list is provided, specifies separate noise standard
                deviations for each constraint.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.continuous_inds = list(range(dim))
        if bounds is None:
            bounds = [(0.0, 10.0) for _ in range(dim)]
        if dim in self._optimal_value_lookup:
            self._optimal_value = self._optimal_value_lookup[dim]
        self._setup(dim, bounds, noise_std, negate, dtype)
        self.constraint_noise_std = self._validate_constraint_noise(constraint_noise_std)

__all__ = [
    "AckleyMixed",
    "ConstrainedGramacy",
    "ConstrainedHartmann",
    "ConstrainedHartmannSmooth",
    "ConstrainedSyntheticTestFunction",
    "KeaneBumpFunction",
    "Labs",
    "PressureVessel",
    "SpeedReducer",
    "TensionCompressionString",
    "TrajectoryPlanning",
    "WeldedBeamSO",
    "Ackley",
    "Beale",
    "Branin",
    "Bukin",
    "Cosine8",
    "DixonPrice",
    "DropWave",
    "EggHolder",
    "Griewank",
    "Hartmann",
    "HolderTable",
    "Levy",
    "Michalewicz",
    "Powell",
    "Rastrigin",
    "Rosenbrock",
    "Shekel",
    "SixHumpCamel",
    "StyblinskiTang",
    "SyntheticTestFunction",
    "ThreeHumpCamel",
]
