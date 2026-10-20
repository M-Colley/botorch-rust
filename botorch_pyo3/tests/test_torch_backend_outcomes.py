"""Full Torch graphs through custom priors and nonlinear outcome adapters."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch.models.transforms import outcome as reference_outcomes

from botorch_rs.models._torch_backend_modules import (
    TorchStandardize,
    build_outcome_transform,
    module_log_prior,
)
from botorch_rs.models.transforms.outcome import (
    Bilog,
    ChainedOutcomeTransform,
    Log,
    Power,
    Standardize,
)
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior, independent_covariance

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _compare_gradients(first, second, variables):
    gradients = [
        torch.autograd.grad(value.sum(), variables, create_graph=True, retain_graph=True)
        for value in (first, second)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b)
    hvps = [
        torch.autograd.grad(sum(grad.square().sum() for grad in row), variables, retain_graph=True)
        for row in gradients
    ]
    for a, b in zip(*hvps, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_custom_registered_priors_keep_batch_axes_and_gradient_graph(device, batch_shape):
    like = torch.zeros(3, 2, dtype=torch.double, device=device)
    prior = gpytorch.priors.LogNormalPrior(like.new_tensor(0.2), like.new_tensor(0.7))
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            ard_num_dims=2, batch_shape=torch.Size(batch_shape), lengthscale_prior=prior
        ),
        batch_shape=torch.Size(batch_shape),
        outputscale_prior=gpytorch.priors.GammaPrior(like.new_tensor(2), like.new_tensor(0.3)),
    ).to(like)
    actual = module_log_prior(kernel, batch_shape)
    expected = prior.log_prob(kernel.base_kernel.lengthscale).sum(
        (-1, -2)
    ) + kernel.outputscale_prior.log_prob(kernel.outputscale)
    assert actual.shape == batch_shape
    torch.testing.assert_close(actual, expected)
    _compare_gradients(actual.square(), expected.square(), tuple(kernel.parameters()))
    assert module_log_prior(torch.nn.Identity(), (2,)) == 0.0
    # A shared kernel has one prior term, regardless of its two tree positions.
    alias = gpytorch.kernels.AdditiveKernel(kernel.base_kernel, kernel.base_kernel)
    torch.testing.assert_close(
        module_log_prior(alias, batch_shape),
        prior.log_prob(kernel.base_kernel.lengthscale).sum((-1, -2)),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["Log", "Bilog", "Power"])
@pytest.mark.parametrize("outputs", [None, [1]])
def test_elementwise_training_values_noise_first_and_second_derivatives(device, kind, outputs):
    y = torch.tensor(
        [[0.2, 0.7], [0.9, 0.4], [0.6, 0.8]], dtype=torch.double, device=device, requires_grad=True
    )
    kwargs = {"outputs": outputs}
    if kind == "Power":
        kwargs["power"] = 2.5
    source = {"Log": Log, "Bilog": Bilog, "Power": Power}[kind](**kwargs)
    module = build_outcome_transform(source, 2, y)
    reference = getattr(reference_outcomes, kind)(**kwargs).to(y)
    actual, _ = module(y)
    expected, _ = reference(y)
    torch.testing.assert_close(actual, expected)
    weights = y.new_tensor([[0.1, -0.4], [0.7, 0.8], [-0.2, 0.5]])
    _compare_gradients(actual * weights, expected * weights, (y,))
    torch.testing.assert_close(module.untransform(actual)[0], y)
    noise = torch.full_like(y, 0.03, requires_grad=True)
    if kind == "Log":
        actual_y, actual_noise = module(y, noise)
        ref_y, ref_noise = reference(y, noise)
        torch.testing.assert_close(actual_noise, ref_noise)
        _compare_gradients(actual_y.square() + actual_noise, ref_y.square() + ref_noise, (y, noise))
    else:
        with pytest.raises(NotImplementedError, match="observation noise"):
            module(y, noise)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["Log", "Bilog", "Power"])
def test_nonlinear_posterior_samples_and_moments_keep_graphs(device, kind):
    mean = torch.tensor(
        [[0.2, 0.7], [0.9, 0.4], [0.6, 0.8]], dtype=torch.double, device=device, requires_grad=True
    )
    scale = mean.new_tensor([0.4, 0.7]).requires_grad_()
    blocks = torch.eye(3, dtype=mean.dtype, device=device).expand(2, 3, 3) * scale.square().reshape(
        2, 1, 1
    )
    base = TorchGPyTorchPosterior(mean, independent_covariance(blocks))
    source = {"Log": Log, "Bilog": Bilog, "Power": lambda: Power(2)}[kind]()
    transform = build_outcome_transform(source, 2, mean)
    posterior = transform.untransform_posterior(base)
    normals = torch.full((2, 3, 2), 0.2, dtype=mean.dtype, device=device)
    observed = posterior.rsample_from_base_samples(torch.Size([2]), normals)
    samples = base.rsample_from_base_samples(torch.Size([2]), normals)
    expected = transform.untransform(samples)[0]
    torch.testing.assert_close(observed, expected)
    _compare_gradients(observed.square(), expected.square(), (mean, scale))
    if kind == "Log":
        expected_mean = (mean + 0.5 * base.variance).exp()
        expected_variance = base.variance.expm1() * (2 * mean + base.variance).exp()
        torch.testing.assert_close(posterior.mean, expected_mean)
        torch.testing.assert_close(posterior.variance, expected_variance)
        _compare_gradients(
            posterior.mean.square() + posterior.variance,
            expected_mean.square() + expected_variance,
            (mean, scale),
        )
    else:
        with pytest.raises(NotImplementedError, match="mean transform"):
            _ = posterior.mean


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["Log", "Bilog", "Power"])
def test_chained_outcome_training_graph_and_posterior_samples(device, kind):
    y = torch.tensor(
        [[0.2, 0.7], [0.9, 0.4], [0.6, 0.8]], dtype=torch.double, device=device, requires_grad=True
    )
    elementwise = {"Log": Log, "Bilog": Bilog, "Power": lambda: Power(2)}[kind]()
    source = ChainedOutcomeTransform(nonlinear=elementwise, standardize=Standardize(2))
    module = build_outcome_transform(source, 2, y)
    module.train()
    actual, _ = module(y)
    first, _ = getattr(reference_outcomes, kind)(**({"power": 2} if kind == "Power" else {}))(y)
    reference = reference_outcomes.Standardize(2).to(y)
    expected, _ = reference(first)
    torch.testing.assert_close(actual, expected)
    _compare_gradients(actual.pow(3), expected.pow(3), (y,))
    module.eval()
    gaussian = TorchGPyTorchPosterior(
        y, independent_covariance(torch.eye(3, dtype=y.dtype, device=device).expand(2, 3, 3))
    )
    posterior = module.untransform_posterior(gaussian)
    normals = torch.zeros(2, 3, 2, dtype=y.dtype, device=device)
    actual_samples = posterior.rsample_from_base_samples(torch.Size([2]), normals)
    expected_samples = module.untransform(
        gaussian.rsample_from_base_samples(torch.Size([2]), normals)
    )[0]
    torch.testing.assert_close(actual_samples, expected_samples)


@pytest.mark.parametrize("device", DEVICES)
def test_standardize_attribute_protocol_and_frozen_posterior_statistics(device):
    y = torch.tensor([[0.2, 0.7], [0.9, 0.4], [0.6, 0.8]], dtype=torch.double, device=device)
    reference = reference_outcomes.Standardize(2).to(y)
    module = build_outcome_transform(reference, 2, y)
    assert isinstance(module, TorchStandardize) and module is not reference
    covariance = independent_covariance(torch.eye(3, dtype=y.dtype, device=device).expand(2, 3, 3))
    base = TorchGPyTorchPosterior(y, covariance)
    frozen = module.untransform_posterior(base)
    mean, cov = frozen.mean.clone(), frozen.distribution.covariance_matrix.clone()
    module.train()
    module(y * 4 + 5)
    torch.testing.assert_close(frozen.mean, mean)
    torch.testing.assert_close(frozen.distribution.covariance_matrix, cov)


@pytest.mark.parametrize("device", DEVICES)
def test_bilog_origin_derivative_and_unselected_log_domain(device):
    y = torch.zeros(3, 1, dtype=torch.double, device=device, requires_grad=True)
    module = build_outcome_transform(Bilog(), 1, y)
    derivative = torch.autograd.grad(module(y)[0].sum(), y)[0]
    torch.testing.assert_close(derivative, torch.ones_like(y))
    h = 1e-6
    finite = (module(y + h)[0] - module(y - h)[0]) / (2 * h)
    torch.testing.assert_close(derivative, finite, rtol=1e-6, atol=1e-6)
    mixed = y.new_tensor([[0.0, 0.4], [-1.0, 0.8]], requires_grad=True)
    log = build_outcome_transform(Log(outputs=[1]), 2, mixed)
    transformed, _ = log(mixed)
    grad = torch.autograd.grad(transformed.sum(), mixed, create_graph=True)[0]
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad[..., 0], torch.ones_like(grad[..., 0]))
