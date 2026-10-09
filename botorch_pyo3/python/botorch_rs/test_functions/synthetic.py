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
from botorch_rs._arrays import ArrayKind, from_pair, kind_from_dtype, kind_of, to_flat


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


class BaseTestProblem:
    """Base class of the test problems."""

    num_objectives = 1


class SyntheticTestFunction(BaseTestProblem):
    """Base class of the synthetic test functions."""

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
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class Branin(SyntheticTestFunction):
    """Branin function on ``[-5, 10] x [0, 15]``; minimum 0.397887 at three points."""

    _name = "Branin"
    _fixed_dim = 2

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


class Bukin(SyntheticTestFunction):
    """Bukin N.6 function on ``[-15, -5] x [-3, 3]``; minimum 0 at ``(-10, 1)``."""

    _name = "Bukin"
    _fixed_dim = 2

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


class Cosine8(SyntheticTestFunction):
    """Cosine mixture on ``[-1, 1]^8`` (a maximization problem); maximum 0.8 at the origin."""

    _name = "Cosine8"
    _fixed_dim = 8

    def __init__(
        self,
        dim: int = 8,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


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
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


class EggHolder(SyntheticTestFunction):
    """Eggholder function on ``[-512, 512]^2``; minimum -959.6407."""

    _name = "EggHolder"
    _fixed_dim = 2

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
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


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
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


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
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim, noise_std, negate, bounds, dtype, seed=seed)


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


__all__ = [
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
