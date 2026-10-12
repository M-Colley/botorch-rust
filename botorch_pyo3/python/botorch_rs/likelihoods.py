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
    the likelihood of an ``LCEMGP``: ``noise`` has shape ``num_tasks`` (in the
    model's transformed outcome space), with BoTorch's ``LogNormal(-4, 1)``
    prior and ``GreaterThan(1e-4)`` constraint."""

    _spec: _rs.GaussianLikelihoodSpec
    noise_prior: Prior | None
    noise_constraint: Interval
    num_tasks: int
    task_feature_index: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise UnsupportedError(
            "HadamardGaussianLikelihood is created by LCEMGP; custom likelihoods are not supported."
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
]
