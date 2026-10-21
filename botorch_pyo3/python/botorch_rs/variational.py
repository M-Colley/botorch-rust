"""Variational distributions and strategies (``gpytorch.variational``), as
used by ``SingleTaskVariationalGP``.

``CholeskyVariationalDistribution`` and ``VariationalStrategy`` (whitened,
the default of BoTorch's variational GP) describe the model's variational
family; once a model is built with them they are views of its parameters
(the inducing points, the variational mean and the Cholesky factor of the
variational covariance), which live in Rust. For several outputs the model
wraps the strategy in an ``IndependentMultitaskVariationalStrategy``.
The Tensor backend also accepts the diagonal-Gaussian ``MeanField`` and
point-mass ``Delta`` specifications and implements their device operations.
"""

from __future__ import annotations

import math
import random
from typing import Any

from botorch_rs._arrays import LIST, ArrayKind, from_flat, to_flat
from botorch_rs.exceptions.errors import UnsupportedError


def _randn(shape: tuple[int, ...]) -> list[float]:
    """Standard normal draws in GPyTorch's order: torch's global generator in
    double precision (``torch.randn_like`` of a float64 tensor) when torch
    is installed, else Python's ``random`` module."""
    n = math.prod(shape)
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised without torch
        torch = None  # type: ignore[assignment]
    if torch is not None:
        return [float(v) for v in torch.randn(shape, dtype=torch.float64).reshape(-1).tolist()]
    return [random.gauss(0.0, 1.0) for _ in range(n)]


class _VariationalDistribution:
    """Base class of the variational distributions
    (``gpytorch.variational._VariationalDistribution``)."""

    def __init__(
        self,
        num_inducing_points: int,
        batch_shape: Any = (),
        mean_init_std: float = 1e-3,
    ) -> None:
        self.num_inducing_points = int(num_inducing_points)
        self.batch_shape = tuple(int(b) for b in (batch_shape or ()))
        self.mean_init_std = float(mean_init_std)
        self._owner: Any = None
        self._prefix = ""

    def _bind(self, owner: Any, prefix: str) -> None:
        raise NotImplementedError

    def _initialize(self) -> None:
        raise NotImplementedError


class CholeskyVariationalDistribution(_VariationalDistribution):
    """A ``MultivariateNormal`` variational distribution ``q(u) = N(m, L Lᵀ)``
    with a full-rank Cholesky factor ``L`` (the lower triangle of
    ``chol_variational_covar``), GPyTorch's
    ``CholeskyVariationalDistribution``.

    Args:
        num_inducing_points: The number of inducing points ``M``.
        batch_shape: The batch shape (``[m]`` for ``m`` outputs).
        mean_init_std: The standard deviation of the random perturbation of
            the variational mean when it is initialized (at the model's first
            evaluation).
    """

    def __init__(
        self,
        num_inducing_points: int,
        batch_shape: Any = (),
        mean_init_std: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise UnsupportedError(f"Unsupported arguments: {sorted(kwargs)}.")
        super().__init__(num_inducing_points, batch_shape, mean_init_std)
        nb = math.prod(self.batch_shape)
        mi = self.num_inducing_points
        self._local_mean = [0.0] * (nb * mi)
        eye = [1.0 if i == j else 0.0 for i in range(mi) for j in range(mi)]
        self._local_chol = eye * nb

    def _kind(self) -> ArrayKind:
        return self._owner._kind if self._owner is not None else LIST

    @property
    def variational_mean(self) -> Any:
        """The variational mean ``m`` (``batch_shape x M``)."""
        shape = (*self.batch_shape, self.num_inducing_points)
        if self._owner is None:
            return from_flat(self._local_mean, shape, LIST)
        return self._owner._raw_param_array(self._prefix + "variational_mean")

    @variational_mean.setter
    def variational_mean(self, value: Any) -> None:
        if self._owner is None:
            self._local_mean = _broadcast(value, len(self._local_mean), "variational_mean")
            return
        self._owner._set_raw_param(self._prefix + "variational_mean", value)

    @property
    def chol_variational_covar(self) -> Any:
        """The Cholesky factor of the variational covariance
        (``batch_shape x M x M``; only its lower triangle is used)."""
        mi = self.num_inducing_points
        shape = (*self.batch_shape, mi, mi)
        if self._owner is None:
            return from_flat(self._local_chol, shape, LIST)
        return self._owner._raw_param_array(self._prefix + "chol_variational_covar")

    @chol_variational_covar.setter
    def chol_variational_covar(self, value: Any) -> None:
        if self._owner is None:
            self._local_chol = _broadcast(value, len(self._local_chol), "chol_variational_covar")
            return
        self._owner._set_raw_param(self._prefix + "chol_variational_covar", value)

    def _bind(self, owner: Any, prefix: str) -> None:
        if self._owner is not None and self._owner is not owner:
            raise UnsupportedError(
                "This variational distribution is already part of another model; create a "
                "new one for each model."
            )
        self._owner = owner
        self._prefix = prefix

    def _initialize(self) -> None:
        """GPyTorch's ``initialize_variational_distribution`` from the
        whitened prior ``N(0, I)``: the mean is ``mean_init_std`` times
        standard normal draws, the Cholesky factor the identity."""
        mi = self.num_inducing_points
        shape = (*self.batch_shape, mi)
        draws = _randn(shape)
        self.variational_mean = from_flat(
            [0.0 + self.mean_init_std * v for v in draws], shape, LIST
        )
        nb = math.prod(self.batch_shape)
        eye = [1.0 if i == j else 0.0 for i in range(mi) for j in range(mi)]
        self.chol_variational_covar = from_flat(eye * nb, (*self.batch_shape, mi, mi), LIST)

    def __repr__(self) -> str:
        return f"CholeskyVariationalDistribution(num_inducing_points={self.num_inducing_points})"


class DeltaVariationalDistribution(_VariationalDistribution):
    """Point-mass variational variables for Tensor-backend MAP inference."""

    def __init__(
        self,
        num_inducing_points: int,
        batch_shape: Any = (),
        mean_init_std: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise UnsupportedError(f"Unsupported arguments: {sorted(kwargs)}.")
        super().__init__(num_inducing_points, batch_shape, mean_init_std)
        self._local_mean = [0.0] * (math.prod(self.batch_shape) * self.num_inducing_points)

    @property
    def variational_mean(self) -> Any:
        return from_flat(self._local_mean, (*self.batch_shape, self.num_inducing_points), LIST)

    @variational_mean.setter
    def variational_mean(self, value: Any) -> None:
        self._local_mean = _broadcast(value, len(self._local_mean), "variational_mean")

    def _bind(self, owner: Any, prefix: str) -> None:
        raise UnsupportedError("Delta variational distributions require backend='torch'.")

    def _initialize(self) -> None:
        self._local_mean = [
            self.mean_init_std * v for v in _randn((*self.batch_shape, self.num_inducing_points))
        ]


class MeanFieldVariationalDistribution(DeltaVariationalDistribution):
    """Diagonal Gaussian variational variables for the Tensor backend.

    The signed raw standard deviations have the reference's unconstrained
    ``_variational_stddev`` layout; covariance entries are their squares.
    """

    def __init__(
        self,
        num_inducing_points: int,
        batch_shape: Any = (),
        mean_init_std: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_inducing_points, batch_shape, mean_init_std, **kwargs)
        self._local_stddev = [1.0] * len(self._local_mean)

    @property
    def _variational_stddev(self) -> Any:
        return from_flat(self._local_stddev, (*self.batch_shape, self.num_inducing_points), LIST)

    @_variational_stddev.setter
    def _variational_stddev(self, value: Any) -> None:
        self._local_stddev = _broadcast(value, len(self._local_stddev), "_variational_stddev")

    @property
    def variational_stddev(self) -> Any:
        return from_flat(
            [max(abs(v), 1e-8) for v in self._local_stddev],
            (*self.batch_shape, self.num_inducing_points),
            LIST,
        )

    def _bind(self, owner: Any, prefix: str) -> None:
        raise UnsupportedError("Mean-field variational distributions require backend='torch'.")

    def _initialize(self) -> None:
        super()._initialize()
        self._local_stddev = [1.0] * len(self._local_mean)


def _broadcast(value: Any, size: int, name: str) -> list[float]:
    data, shape = to_flat(value, name)
    if len(data) == size:
        return [float(v) for v in data]
    if len(data) == 1:
        return [float(data[0])] * size
    raise RuntimeError(f"{name} has {size} values, got a value of shape {tuple(shape)}.")


class _VariationalStrategy:
    """Base class of the variational strategies
    (``gpytorch.variational._VariationalStrategy``)."""


class _Flag:
    """GPyTorch's ``variational_params_initialized`` buffer (a 0-dim integer
    tensor) of a strategy: ``item()``, ``fill_(value)``, ``int()``."""

    def __init__(self, strategy: VariationalStrategy) -> None:
        self._strategy = strategy

    def item(self) -> int:
        return self._strategy._initialized

    def fill_(self, value: Any) -> _Flag:
        self._strategy._initialized = int(value)
        return self

    def __int__(self) -> int:
        return self._strategy._initialized

    def __bool__(self) -> bool:
        return bool(self._strategy._initialized)

    def __eq__(self, other: object) -> bool:
        try:
            return self._strategy._initialized == int(other)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self._strategy._initialized)

    def __repr__(self) -> str:
        return f"tensor({self._strategy._initialized})"


class VariationalStrategy(_VariationalStrategy):
    """GPyTorch's whitened ``VariationalStrategy`` (Hensman et al. 2015): the
    variational distribution is over ``K_ZZ^{-1/2} u``, the predictive
    distribution ``q(f(X)) = N(k_XZ K_ZZ^{-1/2} m, K_XX + k_XZ K_ZZ^{-1/2}
    (S - I) K_ZZ^{-1/2} k_ZX)`` with the variational jitter ``1e-6`` (double
    precision) on ``K_ZZ`` and ``K_XX``.

    Pass the class (``variational_strategy=VariationalStrategy``) to
    ``SingleTaskVariationalGP``; the model builds the strategy, which is then
    a view of its inducing points and variational parameters.
    """

    def __init__(
        self,
        model: Any,
        inducing_points: Any,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: float | None = None,
    ) -> None:
        if not getattr(model, "_builds_variational_strategy", False):
            raise UnsupportedError(
                "botorch_rs builds the variational strategy inside SingleTaskVariationalGP: "
                "pass `variational_strategy=VariationalStrategy` (the class)."
            )
        self.model = model
        self._initial_inducing_points = inducing_points
        self._variational_distribution = variational_distribution
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.jitter_val = 1e-6 if jitter_val is None else float(jitter_val)
        self._owner: Any = None
        self._prefix = ""
        self._initialized = 0

    def _bind(self, owner: Any, prefix: str) -> None:
        self._owner = owner
        self._prefix = prefix
        self._variational_distribution._bind(owner, prefix + "_variational_distribution.")

    @property
    def inducing_points(self) -> Any:
        """The inducing point locations (``M x d``, in the space of the
        transformed inputs)."""
        self._owner._sync_torch_parameters()
        if self.learn_inducing_locations:
            return self._owner._raw_param_array(self._prefix + "inducing_points")
        data, shape = self._owner._rs.inducing_points()
        return from_flat(data, tuple(shape), self._owner._kind)

    @inducing_points.setter
    def inducing_points(self, value: Any) -> None:
        self._owner._sync_torch_parameters()
        self._owner._rs.set_inducing_points(to_flat(value, "inducing_points"))
        self._owner._sync_torch_parameters()

    @property
    def variational_params_initialized(self) -> _Flag:
        """Whether the variational parameters have been initialized (GPyTorch's
        buffer: ``0`` until the model's first evaluation; ``.fill_(0)``
        re-initializes them at the next evaluation)."""
        return _Flag(self)

    @variational_params_initialized.setter
    def variational_params_initialized(self, value: Any) -> None:
        self._initialized = int(value)

    def _ensure_initialized(self) -> None:
        """GPyTorch's lazy initialization at the strategy's first call."""
        if not self._initialized:
            self._variational_distribution._initialize()
            self._initialized = 1

    @property
    def variational_distribution(self) -> _VariationalDistribution:
        """The variational distribution module."""
        return self._variational_distribution

    def kl_divergence(self) -> Any:
        """``KL(q(u) ‖ p(u))`` with the whitened prior ``N(0, I)`` (one value per
        batch element)."""
        return self._owner._kl_divergence(summed=False)

    def __repr__(self) -> str:
        return "VariationalStrategy()"


class UnwhitenedVariationalStrategy(VariationalStrategy):
    """Tensor-backend inducing variables in original, unwhitened GP coordinates.

    Pass this class to ``SingleTaskVariationalGP(backend="torch")``. Native
    CPU variational models retain the whitened strategy only.
    """


class IndependentMultitaskVariationalStrategy(_VariationalStrategy):
    """GPyTorch's ``IndependentMultitaskVariationalStrategy``: independent
    outputs as the batch dimension of a base strategy (``task_dim=-1``)."""

    def __init__(
        self,
        base_variational_strategy: VariationalStrategy,
        num_tasks: int,
        task_dim: int = -1,
    ) -> None:
        self.base_variational_strategy = base_variational_strategy
        self.num_tasks = int(num_tasks)
        self.task_dim = int(task_dim)

    @property
    def variational_params_initialized(self) -> Any:
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self) -> Any:
        """The KL divergences of all outputs, summed."""
        return self.base_variational_strategy._owner._kl_divergence(summed=True)

    def __repr__(self) -> str:
        return f"IndependentMultitaskVariationalStrategy(num_tasks={self.num_tasks})"


__all__ = [
    "CholeskyVariationalDistribution",
    "DeltaVariationalDistribution",
    "IndependentMultitaskVariationalStrategy",
    "MeanFieldVariationalDistribution",
    "UnwhitenedVariationalStrategy",
    "VariationalStrategy",
]
