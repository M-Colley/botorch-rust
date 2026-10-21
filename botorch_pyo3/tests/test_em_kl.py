"""Native KL derivatives and genuine kernel fitting to an empirical prior."""

from __future__ import annotations

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models import SingleTaskGP as BtGP  # noqa: E402
from botorch.models.empirical_gps.em_coordinate_ascent import KLPriorFitMLL as BtMLL  # noqa: E402
from gpytorch.kernels import RBFKernel as BtRBF  # noqa: E402
from gpytorch.kernels import ScaleKernel as BtScale  # noqa: E402

from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll  # noqa: E402
from botorch_rs.kernels import RBFKernel, ScaleKernel  # noqa: E402
from botorch_rs.means import ConstantMean, ZeroMean  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.models.empirical_gps.em_coordinate_ascent import (  # noqa: E402
    KLPriorFitMLL,
    fit_kernel_to_em_prior,
    kl_divergence_mvn,
)

DT = torch.float64


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("size", [1, 3, 7])
def test_native_gaussian_kl_values(kind, size):
    generator = torch.Generator().manual_seed(218)
    means = [torch.randn(size, dtype=DT, generator=generator) for _ in range(2)]
    factors = [torch.randn(size, size, dtype=DT, generator=generator) for _ in range(2)]
    covariances = [f @ f.T + 0.4 * torch.eye(size, dtype=DT) for f in factors]
    expected = torch.distributions.kl_divergence(
        torch.distributions.MultivariateNormal(means[0], covariances[0]),
        torch.distributions.MultivariateNormal(means[1], covariances[1]),
    )
    args = [
        as_kind(value.tolist(), kind)
        for value in (means[0], covariances[0], means[1], covariances[1])
    ]
    assert_close(kl_divergence_mvn(*args), expected.item(), rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("size", [1, 3, 7])
def test_gaussian_kl_all_four_input_gradients(size):
    generator = torch.Generator().manual_seed(812)
    mean_p, mean_q = [
        torch.randn(size, dtype=DT, generator=generator).requires_grad_() for _ in range(2)
    ]
    a, b = [torch.randn(size, size, dtype=DT, generator=generator) for _ in range(2)]
    sigma_p, sigma_q = [(f @ f.T + torch.eye(size, dtype=DT)).requires_grad_() for f in (a, b)]
    inputs = (mean_p, sigma_p, mean_q, sigma_q)
    actual = kl_divergence_mvn(*inputs)
    expected = torch.distributions.kl_divergence(
        torch.distributions.MultivariateNormal(mean_p, sigma_p),
        torch.distributions.MultivariateNormal(mean_q, sigma_q),
    )
    gradients = torch.autograd.grad(actual, inputs)
    reference = torch.autograd.grad(expected, inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    for value, other in zip(gradients, reference, strict=True):
        torch.testing.assert_close(value, other, rtol=2e-10, atol=2e-10)


def test_kl_raw_parameter_order_excludes_likelihood_and_priors():
    x = torch.tensor([[0.0, 0.1], [0.3, 0.4], [0.8, 0.7]], dtype=DT)
    y = torch.tensor([[0.2], [0.7], [0.4]], dtype=DT)
    kernel, other = ScaleKernel(RBFKernel(ard_num_dims=2)), BtScale(BtRBF(ard_num_dims=2)).double()
    actual, reference = (
        SingleTaskGP(x, y, covar_module=kernel, outcome_transform=None),
        BtGP(x, y, covar_module=other, outcome_transform=None),
    )
    mu = torch.tensor([0.4, 0.6, 0.3], dtype=DT)
    sigma = torch.tensor([[0.8, 0.2, 0.1], [0.2, 0.9, 0.3], [0.1, 0.3, 1.0]], dtype=DT)
    mll, bt = (
        KLPriorFitMLL(actual.likelihood, actual, x, mu, sigma),
        BtMLL(reference.likelihood, reference, x, mu, sigma),
    )
    value, gradient = mll.value_and_grad()
    result = bt.forward(None, None)
    result.backward()
    expected = [
        v
        for _, param in reference.named_parameters()
        for v in (torch.zeros_like(param) if param.grad is None else param.grad)
        .reshape(-1)
        .tolist()
    ]
    assert_close(value, result.item(), rtol=2e-10, atol=2e-10)
    assert_close(gradient, expected, rtol=2e-9, atol=2e-9)
    assert gradient[0] == 0.0
    assert_close(mll(), value)
    before = value
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 15}})
    assert mll.value_and_grad()[0] > before
    assert mll._fit_result["status"] in ("SUCCESS", "STOPPED")
    assert not mll.training


@pytest.mark.parametrize("constant", [False, True])
def test_standalone_kernel_fit_recovers_exact_gaussian_prior(constant):
    x = torch.tensor([[0.0], [0.23], [0.58], [0.91]], dtype=DT)
    target = ScaleKernel(RBFKernel())
    target.outputscale = torch.tensor(0.8, dtype=DT)
    target.base_kernel.lengthscale = torch.tensor(0.3, dtype=DT)
    covariance = target(x, x)
    mu = torch.full((4,), 1.2 if constant else 0.0, dtype=DT)
    kernel = ScaleKernel(RBFKernel())
    kernel.outputscale = torch.tensor(1.4, dtype=DT)
    kernel.base_kernel.lengthscale = torch.tensor(0.65, dtype=DT)
    mean = ConstantMean() if constant else ZeroMean()
    fit_kernel_to_em_prior(mean, kernel, x, mu, covariance)
    fitted_mu = mean.constant.expand(4) if constant else torch.zeros_like(mu)
    loss = kl_divergence_mvn(mu, covariance, fitted_mu, kernel(x, x))
    assert loss.item() < 1e-8
    torch.testing.assert_close(kernel(x, x), covariance, rtol=2e-4, atol=2e-5)


def test_invalid_kl_shape_and_reference_gradient_are_explicit():
    x = torch.tensor([[0.0], [0.5]], dtype=DT)
    with pytest.raises((ValueError, RuntimeError)):
        kl_divergence_mvn(
            torch.ones(3, dtype=DT),
            torch.eye(2, dtype=DT),
            torch.zeros(2, dtype=DT),
            torch.eye(2, dtype=DT),
        )
    with pytest.raises(UnsupportedError):
        fit_kernel_to_em_prior(
            ZeroMean(),
            RBFKernel(),
            x.requires_grad_(),
            torch.zeros(2, dtype=DT),
            torch.eye(2, dtype=DT),
            backend="native",
        )
