"""Gaussian marginal evaluation and scalarization against live BoTorch."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.posteriors.gpytorch import GPyTorchPosterior as BtPosterior  # noqa: E402
from botorch.posteriors.gpytorch import scalarize_posterior as bt_scalarize  # noqa: E402
from botorch.posteriors.gpytorch import scalarize_posterior_gpytorch as bt_moments  # noqa: E402
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal  # noqa: E402

from botorch_rs._arrays import kind_of, to_flat  # noqa: E402
from botorch_rs.posteriors.gpytorch import (  # noqa: E402
    GPyTorchPosterior,
    scalarize_posterior,
    scalarize_posterior_gpytorch,
)


@pytest.mark.parametrize("q,m", [(1, 1), (3, 1), (1, 2), (3, 2)])
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("interleaved", [False, True])
def test_gaussian_marginals_and_scalarization_match_botorch(
    q: int, m: int, batch: tuple[int, ...], interleaved: bool
) -> None:
    generator = torch.Generator().manual_seed(202)
    mean = torch.randn(*batch, q, m, generator=generator, dtype=torch.float64)
    factor = torch.randn(*batch, q * m, q * m, generator=generator, dtype=torch.float64)
    covariance = factor @ factor.transpose(-1, -2) + 0.2 * torch.eye(q * m, dtype=torch.float64)
    if m == 1:
        distribution = MultivariateNormal(mean.squeeze(-1), covariance)
    else:
        # The Rust posterior keeps the raw covariance in output-major order;
        # a displayed interleaved distribution permutes those rows/columns.
        permutation = torch.tensor([t * q + i for i in range(q) for t in range(m)])
        displayed_covariance = (
            covariance[..., permutation, :][..., :, permutation] if interleaved else covariance
        )
        distribution = MultitaskMultivariateNormal(
            mean, displayed_covariance, interleaved=interleaved
        )
    bt = BtPosterior(distribution)
    rs = GPyTorchPosterior(
        None, ([], (*batch, q, 0)), kind_of(mean), False, m, interleaved=interleaved
    )
    rs._joint = (to_flat(mean), to_flat(covariance))
    variance = covariance.diagonal(dim1=-2, dim2=-1).reshape(*batch, m, q).transpose(-1, -2)
    rs._marginals = (list(to_flat(mean)[0]), list(to_flat(variance)[0]))
    for probabilities in [
        torch.tensor(0.3, dtype=torch.float64),
        torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64),
    ]:
        torch.testing.assert_close(
            rs.quantile(probabilities), bt.quantile(probabilities), rtol=1e-12, atol=1e-12
        )
    values = torch.tensor([-0.7, 0.0, 1.3], dtype=torch.float64)
    torch.testing.assert_close(rs.density(values), bt.density(values), rtol=1e-12, atol=1e-12)
    weights = torch.linspace(-0.3, 0.7, m, dtype=torch.float64)
    mu, cov = scalarize_posterior_gpytorch(rs, weights, offset=0.4)
    mu_ref, cov_ref = bt_moments(bt, weights, offset=0.4)
    if hasattr(cov_ref, "to_dense"):
        cov_ref = cov_ref.to_dense()
    torch.testing.assert_close(mu, mu_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cov, cov_ref, rtol=1e-12, atol=1e-12)
    actual = scalarize_posterior(rs, weights, offset=0.4)
    expected = bt_scalarize(bt, weights, offset=0.4)
    base = torch.randn(4, *batch, q, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        actual.rsample_from_base_samples((4,), base),
        expected.rsample_from_base_samples(torch.Size([4]), base),
        rtol=1e-11,
        atol=1e-12,
    )
