"""Samplers that are not base-sample driven
(``botorch.sampling.stochastic_samplers``): ``StochasticSampler``.

BoTorch's ``StochasticSampler`` calls ``posterior.rsample`` at every
evaluation, for stochastic optimization of acquisition functions. Its main use
with ``optimize_acqf`` is with deterministic models (``PosteriorMeanModel``,
``GenericDeterministicModel``), whose samples are the mean whatever the random
numbers: there, botorch_rs's acquisition functions accept it and are exact.
With any other model they raise ``UnsupportedError``, because botorch_rs's
Monte-Carlo acquisition functions are deterministic sample-average
approximations with fixed base samples.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import MCSampler, SobolQMCNormalSampler, _sample_shape


class StochasticSampler(MCSampler):
    r"""A sampler that simply calls ``posterior.rsample`` to generate the
    samples (BoTorch's ``StochasticSampler``). It ignores the ``seed``.

    In botorch_rs acquisition functions it is supported for deterministic
    models only (see the module documentation).

    Example:
        >>> sampler = StochasticSampler(sample_shape=torch.Size([1]))
        >>> acqf = qLogExpectedHypervolumeImprovement(
        ...     PosteriorMeanModel(model), ref_point, partitioning, sampler=sampler
        ... )
    """

    def __init__(self, sample_shape: Any, seed: int | None = None) -> None:
        self.sample_shape = _sample_shape(sample_shape)
        # Sized base samples for the acquisition functions; a deterministic
        # model's samples do not depend on them.
        self._rs = _rs.SobolQMCNormalSampler(
            math.prod(self.sample_shape), None if seed is None else int(seed)
        )

    @property
    def seed(self) -> int:
        """The seed (drawn at random if not given, as in BoTorch; not used for
        sampling)."""
        return self._rs.seed

    @property
    def num_samples(self) -> int:
        return self._rs.num_samples

    def forward(self, posterior: Any) -> Any:
        """Draws MC samples from the posterior (``posterior.rsample``)."""
        return posterior.rsample(sample_shape=self.sample_shape)

    __call__ = forward

    def __repr__(self) -> str:
        return f"StochasticSampler(sample_shape={self.sample_shape})"


def _is_botorch_stochastic_sampler(sampler: Any) -> bool:
    cls = type(sampler)
    return cls.__name__ == "StochasticSampler" and cls.__module__.startswith("botorch.")


def as_stochastic_sampler(sampler: Any) -> StochasticSampler | None:
    """``sampler`` as a botorch_rs ``StochasticSampler`` if it is one of
    botorch_rs or BoTorch (``None`` otherwise)."""
    if isinstance(sampler, StochasticSampler):
        return sampler
    if _is_botorch_stochastic_sampler(sampler):
        return StochasticSampler(sampler.sample_shape, seed=getattr(sampler, "seed", None))
    return None


def mc_sampler_arg(
    sampler: Any, rs_model: Any, default_shape: tuple[int, ...]
) -> SobolQMCNormalSampler | StochasticSampler:
    """The sampler of a Monte-Carlo acquisition function on the model handle
    ``rs_model``: a botorch_rs ``SobolQMCNormalSampler`` (default: one with
    ``default_shape`` samples), or a ``StochasticSampler`` (of botorch_rs or
    BoTorch) for a deterministic model."""
    if sampler is None:
        return SobolQMCNormalSampler(sample_shape=default_shape)
    if isinstance(sampler, SobolQMCNormalSampler):
        return sampler
    stochastic = as_stochastic_sampler(sampler)
    if stochastic is not None:
        if not rs_model.is_deterministic:
            raise UnsupportedError(
                "A StochasticSampler draws new samples at every evaluation; botorch_rs "
                "supports it only for deterministic models (e.g. PosteriorMeanModel), whose "
                "samples equal the mean. Use a SobolQMCNormalSampler."
            )
        return stochastic
    raise UnsupportedError(
        "`sampler` must be a botorch_rs SobolQMCNormalSampler (or a StochasticSampler with a "
        "deterministic model)."
    )


__all__ = ["StochasticSampler"]
