"""Posterior defaults must not fall through to distribution-only attributes."""

from __future__ import annotations

import pytest

from botorch_rs.posteriors import TorchPosterior
from botorch_rs.posteriors.gpytorch import Posterior
from helpers import KINDS, as_kind, assert_close


@pytest.mark.parametrize("member", ["base_sample_shape", "batch_range"])
def test_torch_posterior_preserves_unsupported_base_protocol(member):
    torch = pytest.importorskip("torch")
    distribution = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    # Distribution attributes do not override the Posterior-level API.
    setattr(distribution, member, (99,))
    with pytest.raises(NotImplementedError, match=member):
        getattr(TorchPosterior(distribution), member)


def test_torch_posterior_does_not_forward_base_sample_method():
    torch = pytest.importorskip("torch")
    distribution = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    distribution.rsample_from_base_samples = lambda *args: "not the posterior protocol"
    with pytest.raises(NotImplementedError, match="rsample_from_base_samples"):
        TorchPosterior(distribution).rsample_from_base_samples((1,), torch.zeros(1, 2))


@pytest.mark.parametrize("member", ["device", "dtype", "base_sample_shape", "batch_range"])
def test_base_posterior_metadata_requires_implementation(member):
    with pytest.raises(NotImplementedError, match=member):
        getattr(Posterior(), member)


@pytest.mark.parametrize("kind", KINDS)
def test_structured_posterior_metadata_and_mvn_aliases(kind):
    from botorch_rs.models import HigherOrderGP, LatentKroneckerGP

    x = as_kind([[0.1], [0.4], [0.8]], kind)
    y = as_kind([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], kind)
    t = as_kind([[0.0], [1.0]], kind)
    query = as_kind([[0.3], [0.6]], kind)
    models = [HigherOrderGP(x, y, outcome_transform=None), LatentKroneckerGP(x, t, y)]
    for model in models:
        posterior = model.posterior(query)
        assert posterior.dtype == getattr(query, "dtype", None)
        if kind == "torch":
            assert posterior.device == query.device
        else:
            assert posterior.device in (None, "cpu")
        assert_close(posterior.mvn.mean, posterior.distribution.mean)
        assert_close(posterior.mvn.variance, posterior.distribution.variance)
