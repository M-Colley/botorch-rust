r"""``get_sampler`` (``botorch.sampling.get_sampler``): the default sampler of
a posterior, chosen by multiple dispatch on the type of the posterior (on the
type of its distribution for normal posteriors), as in BoTorch:

* a ``SobolQMCNormalSampler`` for a multivariate normal posterior
  (``IIDNormalSampler`` if the collapsed base samples have more than
  ``SobolEngine.MAXDIM`` dimensions),
* the sampler of the underlying posterior of a ``TransformedPosterior``,
* a ``ListSampler`` of ``IIDNormalSampler``\ s for a ``PosteriorList``,
* an ``IndexSampler`` for an ``EnsemblePosterior``.

New samplers are registered with ``GetSampler.register``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from botorch_rs.posteriors.ensemble import EnsemblePosterior
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior, MultivariateNormal, Posterior
from botorch_rs.posteriors.higher_order import HigherOrderGPPosterior
from botorch_rs.posteriors.latent_kronecker import LatentKroneckerGPPosterior
from botorch_rs.posteriors.posterior_list import PosteriorList
from botorch_rs.posteriors.torch import TorchPosterior
from botorch_rs.posteriors.transformed import TransformedPosterior
from botorch_rs.sampling.index_sampler import IndexSampler
from botorch_rs.sampling.list_sampler import ListSampler
from botorch_rs.sampling.normal import (
    IIDNormalSampler,
    MCSampler,
    NormalMCSampler,
    SobolQMCNormalSampler,
    _sample_shape,
)
from botorch_rs.utils.dispatcher import Dispatcher

logger = logging.getLogger("botorch")

#: torch's ``SobolEngine.MAXDIM``.
SOBOL_MAXDIM = 21201


def _posterior_to_distribution_encoder(posterior: Any) -> type:
    r"""An encoder returning the type of the distribution for normal
    posteriors and the type of the posterior for the rest.
    """
    if isinstance(posterior, (GPyTorchPosterior, TorchPosterior)):
        return type(posterior.distribution)
    return type(posterior)


GetSampler = Dispatcher("get_sampler", encoder=_posterior_to_distribution_encoder)


def get_sampler(
    posterior: Any,
    sample_shape: Any,
    *,
    seed: int | None = None,
) -> MCSampler:
    r"""Get the sampler for the given posterior.

    The sampler can be used as ``sampler(posterior)`` to produce samples
    suitable for use in acquisition function optimization via SAA.

    Args:
        posterior: A ``Posterior`` to get the sampler for.
        sample_shape: The sample shape of the samples produced by the
            given sampler. The full shape of the resulting samples is
            given by ``posterior._extended_shape(sample_shape)``.
        seed: Seed used to initialize sampler.

    Returns:
        The ``MCSampler`` object for the given posterior.
    """
    return GetSampler(posterior, sample_shape=sample_shape, seed=seed)


def _collapsed_size(posterior: GPyTorchPosterior) -> int:
    """The number of base samples per sample with the t-batches collapsed."""
    shape = tuple(posterior.base_sample_shape)
    start, end = posterior.batch_range
    end = len(shape) + end if end <= 0 else end
    start = len(shape) + start if start < 0 else start
    return math.prod((*shape[:start], *(1 for _ in shape[start:end]), *shape[end:]))


@GetSampler.register(MultivariateNormal)
@GetSampler.register(HigherOrderGPPosterior)
@GetSampler.register(LatentKroneckerGPPosterior)
def _get_sampler_mvn(
    posterior: Any,
    sample_shape: Any,
    *,
    seed: int | None = None,
) -> NormalMCSampler:
    r"""The Sobol normal sampler for the ``MultivariateNormal`` posterior.

    If the output dim is too large, falls back to ``IIDNormalSampler``.
    """
    sampler: NormalMCSampler = SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)
    size = _collapsed_size(posterior)
    if size > SOBOL_MAXDIM:
        logger.warning(
            f"Output dim {size} is too large for the Sobol engine. Using IIDNormalSampler instead."
        )
        sampler = IIDNormalSampler(sample_shape=sample_shape, seed=seed)
    return sampler


@GetSampler.register(TransformedPosterior)
def _get_sampler_derived(
    posterior: TransformedPosterior,
    sample_shape: Any,
    *,
    seed: int | None = None,
) -> MCSampler:
    r"""Get the sampler for the underlying posterior."""
    return get_sampler(
        posterior=posterior._posterior,
        sample_shape=sample_shape,
        seed=seed,
    )


@GetSampler.register(PosteriorList)
def _get_sampler_list(
    posterior: PosteriorList, sample_shape: Any, *, seed: int | None = None
) -> MCSampler:
    r"""Get the ``ListSampler`` with the appropriate list of samplers.

    NOTE: Does not dispatch to Sobol sampling for normal posteriors due to
    correlations between samplers. Instead uses ``IIDNormalSampler``. See the following
    issue for details: https://github.com/meta-pytorch/botorch/issues/2658
    """
    samplers: list[Any] = []
    for p in posterior.posteriors:
        sampler = get_sampler(posterior=p, sample_shape=sample_shape, seed=seed)
        if isinstance(sampler, SobolQMCNormalSampler):
            sampler = IIDNormalSampler(sample_shape=sample_shape, seed=seed)
        samplers.append(sampler)

    return ListSampler(*samplers)


@GetSampler.register(EnsemblePosterior)
def _get_sampler_ensemble(
    posterior: EnsemblePosterior,
    sample_shape: Any,
    seed: int | None = None,
) -> MCSampler:
    r"""Get the ``IndexSampler`` for the ``EnsemblePosterior``."""
    return IndexSampler(sample_shape=_sample_shape(sample_shape), seed=seed)


@GetSampler.register(object)
def _not_found_error(
    posterior: Posterior,
    sample_shape: Any,
    seed: int | None = None,
) -> None:
    raise NotImplementedError(
        f"A registered `MCSampler` for posterior {posterior} is not found. You can "
        "implement and register one using `@GetSampler.register`."
    )


__all__ = ["GetSampler", "get_sampler"]
