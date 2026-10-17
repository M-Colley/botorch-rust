"""Gaussian likelihoods (``gpytorch.likelihoods``).

A model's ``likelihood`` exposes the noise variance in the model's
(transformed) outcome space, like GPyTorch: ``noise`` has shape
``batch_shape x 1`` for a homoskedastic likelihood and ``batch_shape x n`` for
fixed observation noise, where ``batch_shape`` is the model's
``_aug_batch_shape`` (empty for a single-output model without batch shape).
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs._module import GPModule
from botorch_rs.constraints import Interval
from botorch_rs.constraints import _from_rs as _constraint_from_rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import _check_batch_shape, _check_constraint, _check_kwargs, _check_prior
from botorch_rs.priors import Prior
from botorch_rs.priors import _from_rs as _prior_from_rs


class Likelihood(GPModule):
    """Base class of the likelihoods."""

    @property
    def noise_covar(self) -> Likelihood:
        """GPyTorch's noise module (``likelihood.noise_covar.noise``)."""
        return self


class GaussianLikelihood(Likelihood):
    """Homoskedastic Gaussian noise with a learned noise variance."""

    def __init__(
        self,
        noise_prior: Prior | None = None,
        noise_constraint: Interval | None = None,
        batch_shape: Any = (),
        **kwargs: Any,
    ) -> None:
        self.batch_shape = _check_batch_shape(batch_shape)
        _check_kwargs(kwargs, "GaussianLikelihood")
        _check_prior(noise_prior, "noise_prior")
        _check_constraint(noise_constraint, "noise_constraint")
        self.noise_prior = noise_prior
        self._spec = _rs.GaussianLikelihoodSpec(
            noise_prior._rs if noise_prior is not None else None,
            noise_constraint._rs if noise_constraint is not None else None,
        )
        self.noise_constraint = (
            noise_constraint
            if noise_constraint is not None
            else _constraint_from_rs(self._spec.noise_constraint)
        )

    @classmethod
    def _from_spec(
        cls, spec: _rs.GaussianLikelihoodSpec, batch_shape: tuple[int, ...] = ()
    ) -> GaussianLikelihood:
        obj = cls.__new__(cls)
        obj._spec = spec
        obj.noise_prior = _prior_from_rs(spec.noise_prior)
        obj.noise_constraint = _constraint_from_rs(spec.noise_constraint)
        obj.batch_shape = batch_shape
        return obj

    @property
    def noise(self) -> Any:
        """The noise variance, shape ``batch_shape x 1``."""
        return self._param_out(self._rs_model.noise, (1,))

    @noise.setter
    def noise(self, value: Any) -> None:
        self._rs_model.set_noise(to_flat(value, "noise"))

    @property
    def raw_noise(self) -> Any:
        return self._param_out(self._model._raw_param("raw_noise"), (1,))

    @raw_noise.setter
    def raw_noise(self, value: Any) -> None:
        self._model._set_raw_param("raw_noise", value)

    @property
    def raw_noise_constraint(self) -> Interval:
        return self.noise_constraint

    def __repr__(self) -> str:
        return "GaussianLikelihood()"


class FixedNoiseGaussianLikelihood(Likelihood):
    """Fixed observation noise, created by models given ``train_Yvar``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise UnsupportedError(
            "Pass `train_Yvar` to the model instead of a FixedNoiseGaussianLikelihood."
        )

    @classmethod
    def _create(cls, batch_shape: tuple[int, ...] = ()) -> FixedNoiseGaussianLikelihood:
        obj = cls.__new__(cls)
        obj.batch_shape = batch_shape
        return obj

    @property
    def noise(self) -> Any:
        """The (transformed, clamped at 1e-6) noise variances, shape
        ``batch_shape x n``."""
        return self._param_out(self._rs_model.noise, (-1,))

    def __repr__(self) -> str:
        return "FixedNoiseGaussianLikelihood()"


class HadamardGaussianLikelihood(Likelihood):
    """Per-task noise levels (``gpytorch.likelihoods.HadamardGaussianLikelihood``),
    the likelihood of a ``MultiTaskGP`` and of an ``LCEMGP``: ``noise`` has
    shape ``num_tasks`` (in the model's transformed outcome space); the
    models' default has BoTorch's ``LogNormal(-4, 1)`` prior and
    ``GreaterThan(1e-4)`` constraint. A ``MultiTaskGP`` also accepts one
    built by the user (``batch_shape=()``), looking up the noise of every
    observation by its task index."""

    _spec: _rs.GaussianLikelihoodSpec
    noise_prior: Prior | None
    noise_constraint: Interval
    num_tasks: int
    task_feature_index: int | None

    def __init__(
        self,
        num_tasks: int,
        noise_prior: Prior | None = None,
        noise_constraint: Interval | None = None,
        batch_shape: Any = (),
        task_feature_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        if _check_batch_shape(batch_shape):
            raise UnsupportedError(
                "A HadamardGaussianLikelihood with a batch shape is not supported."
            )
        _check_kwargs(kwargs, "HadamardGaussianLikelihood")
        _check_prior(noise_prior, "noise_prior")
        _check_constraint(noise_constraint, "noise_constraint")
        self.num_tasks = int(num_tasks)
        self.task_feature_index = task_feature_index
        self.noise_prior = noise_prior
        self._spec = _rs.GaussianLikelihoodSpec(
            noise_prior._rs if noise_prior is not None else None,
            noise_constraint._rs if noise_constraint is not None else None,
        )
        self.noise_constraint = (
            noise_constraint
            if noise_constraint is not None
            else _constraint_from_rs(self._spec.noise_constraint)
        )

    @classmethod
    def _create(
        cls, model: Any, spec: _rs.GaussianLikelihoodSpec, num_tasks: int, task_feature_index: int
    ) -> HadamardGaussianLikelihood:
        obj = cls.__new__(cls)
        obj._spec = spec
        obj.noise_prior = _prior_from_rs(spec.noise_prior)
        obj.noise_constraint = _constraint_from_rs(spec.noise_constraint)
        obj.num_tasks = num_tasks
        obj.task_feature_index = task_feature_index
        obj._bind(model)
        return obj

    @property
    def noise(self) -> Any:
        """The noise variance of every task, shape ``num_tasks``."""
        return self._out(self._model._rs.noise, (self.num_tasks,))

    @noise.setter
    def noise(self, value: Any) -> None:
        self._model._rs.set_noise(list(to_flat(value, "noise")[0]))

    @property
    def raw_noise(self) -> Any:
        return self._out(
            self._model._raw_param("likelihood.noise_covar.raw_noise"), (self.num_tasks,)
        )

    @raw_noise.setter
    def raw_noise(self, value: Any) -> None:
        self._model._set_raw_param("likelihood.noise_covar.raw_noise", value)

    @property
    def raw_noise_constraint(self) -> Interval:
        return self.noise_constraint

    def __repr__(self) -> str:
        return f"HadamardGaussianLikelihood(num_tasks={self.num_tasks})"


class MultitaskGaussianLikelihood(Likelihood):
    r"""The likelihood of a ``KroneckerMultiTaskGP``
    (``gpytorch.likelihoods.MultitaskGaussianLikelihood``): the noise
    covariance ``I_n kron N`` of the ``n`` points and ``m`` tasks, with
    ``N = diag(task_noises) + noise I`` (``rank=0``) or ``N = F F^T + noise
    I`` (``rank > 0``, ``task_noise_covar_factor``, ``m x rank``). The noise
    prior applies to the task noises and the global noise. Only the default
    ``has_global_noise=True`` and ``has_task_noise=True`` are supported."""

    def __init__(
        self,
        num_tasks: int,
        rank: int = 0,
        batch_shape: Any = (),
        task_prior: Prior | None = None,
        noise_prior: Prior | None = None,
        noise_constraint: Interval | None = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
        **kwargs: Any,
    ) -> None:
        _check_kwargs(kwargs, "MultitaskGaussianLikelihood")
        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )
        if not (has_global_noise and has_task_noise):
            raise UnsupportedError(
                "MultitaskGaussianLikelihood without global or task noise is not supported."
            )
        if task_prior is not None:
            if rank == 0:
                raise RuntimeError("Cannot set a `task_prior` if rank=0")
            raise UnsupportedError("A `task_prior` of the task noise is not supported.")
        if _check_batch_shape(batch_shape):
            raise UnsupportedError("A likelihood with a batch shape is not supported.")
        _check_prior(noise_prior, "noise_prior")
        _check_constraint(noise_constraint, "noise_constraint")
        self.num_tasks = int(num_tasks)
        self.rank = int(rank)
        self.has_global_noise = True
        self.has_task_noise = True
        self.noise_prior = noise_prior
        self._spec = _rs.GaussianLikelihoodSpec(
            noise_prior._rs if noise_prior is not None else None,
            noise_constraint._rs if noise_constraint is not None else None,
        )
        self.noise_constraint = (
            noise_constraint
            if noise_constraint is not None
            else _constraint_from_rs(self._spec.noise_constraint)
        )
        self._local_factor: list[float] | None = None
        if self.rank > 0:
            from botorch_rs.kernels import _randn

            self._local_factor = _randn((self.num_tasks, self.rank))

    @classmethod
    def _from_spec(
        cls, spec: _rs.GaussianLikelihoodSpec, num_tasks: int, rank: int
    ) -> MultitaskGaussianLikelihood:
        obj = cls.__new__(cls)
        obj._spec = spec
        obj.noise_prior = _prior_from_rs(spec.noise_prior)
        obj.noise_constraint = _constraint_from_rs(spec.noise_constraint)
        obj.num_tasks = num_tasks
        obj.rank = rank
        obj.has_global_noise = True
        obj.has_task_noise = True
        obj._local_factor = None
        if rank > 0:
            from botorch_rs.kernels import _randn

            obj._local_factor = _randn((num_tasks, rank))
        return obj

    def _bind(self, model: Any) -> None:
        super()._bind(model)
        if self._local_factor is not None:
            model._set_raw_param("likelihood.task_noise_covar_factor", self._local_factor)
            self._local_factor = None

    @property
    def noise(self) -> Any:
        """The global noise variance (shape ``1``)."""
        return self._out([self._model._rs.noises()[1]], (1,))

    @noise.setter
    def noise(self, value: Any) -> None:
        data, _ = to_flat(value, "noise")
        self._model._set_raw_param(
            "likelihood.raw_noise", [self.noise_constraint.inverse_transform(data[0])]
        )

    @property
    def raw_noise(self) -> Any:
        return self._out(self._model._raw_param("likelihood.raw_noise"), (1,))

    @raw_noise.setter
    def raw_noise(self, value: Any) -> None:
        self._model._set_raw_param("likelihood.raw_noise", value)

    @property
    def task_noises(self) -> Any:
        """The task noise variances (``num_tasks``; ``rank=0`` only)."""
        if self.rank != 0:
            raise AttributeError(
                "Cannot set diagonal task noises when covariance has ", self.rank, ">0"
            )
        return self._out(list(self._model._rs.noises()[0]), (self.num_tasks,))

    @task_noises.setter
    def task_noises(self, value: Any) -> None:
        if self.rank != 0:
            raise AttributeError(
                "Cannot set diagonal task noises when covariance has ", self.rank, ">0"
            )
        data, shape = to_flat(value, "task_noises")
        raw = [self.noise_constraint.inverse_transform(v) for v in data]
        if len(raw) == 1:
            raw = raw * self.num_tasks
        self._model._set_raw_param("likelihood.raw_task_noises", raw)

    @property
    def raw_task_noises(self) -> Any:
        return self._out(self._model._raw_param("likelihood.raw_task_noises"), (self.num_tasks,))

    @raw_task_noises.setter
    def raw_task_noises(self, value: Any) -> None:
        self._model._set_raw_param("likelihood.raw_task_noises", value)

    @property
    def task_noise_covar_factor(self) -> Any:
        """The factor of the task noise covariance (``num_tasks x rank``)."""
        return self._out(
            self._model._raw_param("likelihood.task_noise_covar_factor"),
            (self.num_tasks, self.rank),
        )

    @task_noise_covar_factor.setter
    def task_noise_covar_factor(self, value: Any) -> None:
        self._model._set_raw_param("likelihood.task_noise_covar_factor", value)

    @property
    def task_noise_covar(self) -> Any:
        """``F F^T`` (``rank > 0``)."""
        if self.rank == 0:
            raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")
        f = self._model._raw_param("likelihood.task_noise_covar_factor")
        m, r = self.num_tasks, self.rank
        out = [
            sum(f[i * r + k] * f[j * r + k] for k in range(r)) for i in range(m) for j in range(m)
        ]
        return self._out(out, (m, m))

    @property
    def raw_noise_constraint(self) -> Interval:
        return self.noise_constraint

    @property
    def raw_task_noises_constraint(self) -> Interval:
        return self.noise_constraint

    def __repr__(self) -> str:
        return f"MultitaskGaussianLikelihood(num_tasks={self.num_tasks}, rank={self.rank})"


class LikelihoodList(Likelihood):
    """The likelihoods of the sub-models of a ``ModelListGP``
    (``gpytorch.likelihoods.LikelihoodList``)."""

    def __init__(self, *likelihoods: Likelihood) -> None:
        self.likelihoods = list(likelihoods)

    def __repr__(self) -> str:
        return f"LikelihoodList({', '.join(repr(lik) for lik in self.likelihoods)})"


__all__ = [
    "FixedNoiseGaussianLikelihood",
    "GaussianLikelihood",
    "HadamardGaussianLikelihood",
    "Likelihood",
    "LikelihoodList",
    "MultitaskGaussianLikelihood",
]
