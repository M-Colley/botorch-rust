"""Gaussian likelihoods (``gpytorch.likelihoods``).

A model's ``likelihood`` exposes the noise variance in the model's
(transformed) outcome space, like GPyTorch: ``noise`` has shape ``(1,)`` for a
homoskedastic likelihood and ``(n,)`` for fixed observation noise.
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
        _check_batch_shape(batch_shape)
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
    def _from_spec(cls, spec: _rs.GaussianLikelihoodSpec) -> GaussianLikelihood:
        obj = cls.__new__(cls)
        obj._spec = spec
        obj.noise_prior = _prior_from_rs(spec.noise_prior)
        obj.noise_constraint = _constraint_from_rs(spec.noise_constraint)
        return obj

    @property
    def noise(self) -> Any:
        """The noise variance, shape ``(1,)``."""
        return self._out(self._rs_model.noise, (1,))

    @noise.setter
    def noise(self, value: Any) -> None:
        data = to_flat(value, "noise")[0]
        if len(data) != 1:
            raise ValueError("The noise of a GaussianLikelihood is a single value.")
        self._rs_model.set_noise(data[0])

    @property
    def raw_noise(self) -> Any:
        return self._out(self._model._raw_param("raw_noise"), (1,))

    @raw_noise.setter
    def raw_noise(self, value: Any) -> None:
        self._model._set_raw_param("raw_noise", to_flat(value, "raw_noise")[0])

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
    def _create(cls) -> FixedNoiseGaussianLikelihood:
        return cls.__new__(cls)

    @property
    def noise(self) -> Any:
        """The (transformed, clamped at 1e-6) noise variances, shape ``(n,)``."""
        values = self._rs_model.noise
        return self._out(values, (len(values),))

    def __repr__(self) -> str:
        return "FixedNoiseGaussianLikelihood()"


__all__ = ["FixedNoiseGaussianLikelihood", "GaussianLikelihood", "Likelihood"]
