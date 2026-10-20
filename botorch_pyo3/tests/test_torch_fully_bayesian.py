"""Loaded fully Bayesian components and mixture laws on CPU/CUDA."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("jax")
pytest.importorskip("numpyro")

from botorch import settings as reference_settings
from botorch.models import fully_bayesian as reference
from botorch.models.transforms.input import Normalize as ReferenceNormalize
from botorch.models.transforms.outcome import Standardize as ReferenceStandardize
from botorch.posteriors.gpytorch import GPyTorchPosterior as ReferencePosterior

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import fully_bayesian as port
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]
FAMILIES = [
    "FullyBayesianSingleTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "FullyBayesianLinearSingleTaskGP",
]


def _mixture_loss(posterior):
    return (
        posterior.mixture_mean.square().sum()
        + 0.4 * posterior.mixture_variance.sum()
        + 0.2 * posterior.mixture_covariance_matrix.to_dense().square().sum()
    )


def _pair(
    device, family, fixed=False, warp=False, transforms=False, gradients=False, sample_count=3
):
    x = torch.tensor(
        [
            [0.12, 0.2, 0.7],
            [0.27, 0.45, 0.24],
            [0.55, 0.7, 0.6],
            [0.72, 0.3, 0.4],
            [0.88, 0.8, 0.9],
        ],
        dtype=torch.double,
        device=device,
    )
    y = (x[:, :1] * 4).sin() + x[:, 1:2] * 0.3
    variance = torch.full_like(y, 0.03) if fixed else None
    data = [
        [a.clone().requires_grad_(gradients) if a is not None else None for a in (x, y, variance)]
        for _ in range(2)
    ]
    kwargs = {"use_input_warping": warp, "indices_to_warp": [0, 2] if warp else None}
    models = []
    for i, namespace in enumerate((port, reference)):
        options = dict(kwargs)
        if transforms:
            options["input_transform"] = (Normalize if i == 0 else ReferenceNormalize)(
                d=3, bounds=x.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            )
            options["outcome_transform"] = (Standardize if i == 0 else ReferenceStandardize)(m=1)
        if i == 0:
            options["backend"] = "torch"
        models.append(getattr(namespace, family)(*data[i], **options))
    s = sample_count
    samples = {
        "mean": torch.linspace(-0.1, 0.2, s, device=device, dtype=x.dtype),
        "noise": torch.linspace(0.025, 0.055, s, device=device, dtype=x.dtype),
    }
    if family == "FullyBayesianLinearSingleTaskGP":
        samples["weight_variance"] = torch.linspace(
            0.4, 1.2, s * 3, device=device, dtype=x.dtype
        ).reshape(s, 3)
    else:
        samples["lengthscale"] = torch.linspace(
            0.3, 0.85, s * 3, device=device, dtype=x.dtype
        ).reshape(s, 3)
        samples["outputscale"] = torch.linspace(0.8, 1.2, s, device=device, dtype=x.dtype)
    if warp:
        samples["c0"] = torch.linspace(0.8, 1.3, s * 2, device=device, dtype=x.dtype).reshape(s, 2)
        samples["c1"] = torch.linspace(0.9, 1.4, s * 2, device=device, dtype=x.dtype).reshape(s, 2)
    for model in models:
        model.load_mcmc_samples(samples)
        model.eval()
    return models, data, samples


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("warp", [False, True])
@pytest.mark.parametrize("transforms", [False, True])
def test_loaded_mcmc_posterior_mixture_moments_and_shared_samples(
    device, family, fixed, warp, transforms
):
    (model, ref), data, _ = _pair(device, family, fixed, warp, transforms)
    assert isinstance(model, getattr(port, family))
    assert model.num_mcmc_samples == 3
    assert model.batch_shape == torch.Size([3])
    query = data[0][0][[0, 2]].unsqueeze(0).expand(2, 2, 3).clone()
    query[1] += 0.01
    for noisy in (False, True):
        actual, expected = (
            model.posterior(query, observation_noise=noisy),
            ref.posterior(query, observation_noise=noisy),
        )
        assert isinstance(actual, GaussianMixturePosterior)
        assert actual.device.type == device
        assert actual.mean.shape == (2, 3, 2, 1)
        for name in ("mean", "variance", "mixture_mean", "mixture_variance"):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), atol=3e-10, rtol=2e-9
            )
        torch.testing.assert_close(
            actual.distribution.covariance_matrix,
            expected.distribution.covariance_matrix,
            atol=3e-10,
            rtol=2e-9,
        )
        torch.testing.assert_close(
            actual.mixture_covariance_matrix,
            expected.mixture_covariance_matrix.to_dense(),
            atol=3e-10,
            rtol=2e-9,
        )
        z = torch.linspace(-1, 1, 12, dtype=query.dtype, device=device).reshape(6, 1, 1, 2)
        # LinearKernel can retain a q x d root with d > q; the official
        # sampler then fails with its advertised q-dimensional base samples.
        # Densifying the official covariance retains exactly the same law
        # and selects the canonical Cholesky root for this comparison.
        dense_reference = ReferencePosterior(
            gpytorch.distributions.MultivariateNormal(
                expected.distribution.mean, expected.distribution.covariance_matrix
            )
        )
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            expected_samples = dense_reference.rsample_from_base_samples(
                torch.Size([6]), z.expand(6, 2, 3, 2).contiguous()
            )
        torch.testing.assert_close(
            actual.rsample_from_base_samples(torch.Size([6]), z),
            expected_samples,
            atol=5e-10,
            rtol=3e-9,
        )
    median_name = "median_weight_variance" if "Linear" in family else "median_lengthscale"
    torch.testing.assert_close(
        getattr(model, median_name), getattr(ref, median_name), atol=1e-14, rtol=1e-14
    )
    sampler = SobolQMCNormalSampler((4,), seed=13)
    samples = sampler(model.posterior(query))
    assert samples.shape == (4, 2, 3, 2, 1)
    assert samples.device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_loaded_mcmc_full_first_graph_matches_reference_and_has_second_graph(device, family):
    (model, ref), data, _ = _pair(device, family, gradients=True)
    queries = [values[0][[1, 3]].detach().clone().requires_grad_() for values in data]
    parameters = [dict(m.named_parameters()) for m in (model, ref)]
    assert parameters[0].keys() == parameters[1].keys()
    gradients = []
    variables = []
    for i, m in enumerate((model, ref)):
        if i == 1 and hasattr(m, "input_transform"):
            # BoTorch eval stores transformed training inputs under no_grad.
            # Restore the identical transformed values with their functional
            # graph to compare training-input derivatives of the same law.
            m.train_inputs = (m.transform_inputs(data[i][0]),)
        with reference_settings.propagate_grads(True):
            posterior = m.posterior(queries[i])
        loss = _mixture_loss(posterior)
        variables.append([*data[i][:2], queries[i], *parameters[i].values()])
        gradients.append(torch.autograd.grad(loss, variables[i], create_graph=True))
    for a, b in zip(gradients[0][:3], gradients[1][:3], strict=True):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=3e-7)
    expected = dict(zip(parameters[1], gradients[1][3:], strict=True))
    for name, derivative in zip(parameters[0], gradients[0][3:], strict=True):
        torch.testing.assert_close(derivative, expected[name], atol=2e-8, rtol=3e-7)
    directions = [
        [
            torch.linspace(-0.2, 0.3, v.numel(), dtype=v.dtype, device=v.device).reshape_as(v)
            for v in values
        ]
        for values in variables
    ]
    second = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(gradients[0], directions[0], strict=True)), variables[0]
    )
    assert all(torch.isfinite(g).all() for g in second)
    # Independent reference first-gradient differences check every mixed
    # data/query/hyperparameter HVP even where GPyTorch's cdist backward
    # has no second derivative implementation.
    original = [value.detach().clone() for value in variables[1]]
    finite_difference = []
    step = 1e-5
    for sign in (-1, 1):
        with torch.no_grad():
            for value, saved, direction in zip(variables[1], original, directions[1], strict=True):
                value.copy_(saved + sign * step * direction)
        ref.prediction_strategy = None
        if hasattr(ref, "input_transform"):
            ref.train_inputs = (ref.transform_inputs(data[1][0]),)
        with reference_settings.propagate_grads(True):
            loss = _mixture_loss(ref.posterior(queries[1]))
        finite_difference.append(torch.autograd.grad(loss, variables[1]))
    expected_second = [
        (plus - minus) / (2 * step) for minus, plus in zip(*finite_difference, strict=True)
    ]
    for actual, expected in zip(second[:3], expected_second[:3], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-5)
    expected = dict(zip(parameters[1], expected_second[3:], strict=True))
    for name, actual in zip(parameters[0], second[3:], strict=True):
        torch.testing.assert_close(actual, expected[name], atol=3e-6, rtol=5e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_loaded_mcmc_reset_reload_state_and_immutable_laws(device):
    (model, _), data, draws = _pair(device, "SaasFullyBayesianSingleTaskGP", warp=True)
    query = data[0][0][:2]
    old = model.posterior(query)
    old_mean, old_cov = old.mean.clone(), old.distribution.covariance_matrix.clone()
    model.mean_module.constant = model.mean_module.constant + 0.5
    torch.testing.assert_close(old.mean, old_mean, atol=0, rtol=0)
    torch.testing.assert_close(old.distribution.covariance_matrix, old_cov, atol=0, rtol=0)
    state = copy.deepcopy(model.state_dict())
    model.train(reset=False)
    assert model.num_mcmc_samples == 3
    model.train()
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.posterior(query)
    model.load_state_dict(state)
    model.eval()
    restored = model.posterior(query)
    assert not torch.equal(restored.mean, old_mean)
    model.load_mcmc_samples(draws)
    torch.testing.assert_close(model.posterior(query).mean, old_mean, atol=1e-12, rtol=1e-12)
    model.load_mcmc_samples({name: value[:2] for name, value in draws.items()})
    assert model.posterior(query).mean.shape == (2, 2, 1)
    with pytest.raises(UnsupportedError, match="NUTS"):
        fit_fully_bayesian_model_nuts(model, warmup_steps=1, num_samples=1)
    assert model.num_mcmc_samples == 2
    with pytest.raises(UnsupportedError, match="not MAP fitting"):
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    assert model.num_mcmc_samples == 2


@pytest.mark.parametrize("device", DEVICES)
def test_loaded_mcmc_quantile_values_and_first_second_derivatives(device):
    (model, ref), data, _ = _pair(device, "FullyBayesianSingleTaskGP")
    queries = [values[0][[1, 3]].detach().clone().requires_grad_() for values in data]
    actual = model.posterior(queries[0]).quantile(
        torch.tensor([0.1, 0.5, 0.9], device=device, dtype=torch.double)
    )
    expected = ref.posterior(queries[1]).quantile(
        torch.tensor([0.1, 0.5, 0.9], device=device, dtype=torch.double)
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    first = torch.autograd.grad(actual.sum(), queries[0], create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), queries[0])[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    h = 1e-5
    for index in range(queries[0].numel()):
        delta = torch.zeros_like(queries[0]).flatten()
        delta[index] = h
        delta = delta.reshape_as(queries[0])
        finite_difference = (
            model.posterior(queries[0] + delta).quantile(0.5).sum()
            - model.posterior(queries[0] - delta).quantile(0.5).sum()
        ) / (2 * h)
        middle = torch.autograd.grad(model.posterior(queries[0]).quantile(0.5).sum(), queries[0])[
            0
        ].flatten()[index]
        torch.testing.assert_close(middle, finite_difference, atol=3e-7, rtol=3e-6)


def test_loaded_mcmc_validates_shapes_values_and_auto_backend():
    (model, _), data, draws = _pair("cpu", "FullyBayesianSingleTaskGP")
    for key, value in (
        ("lengthscale", torch.ones(3, 2, dtype=torch.double)),
        ("mean", torch.full((3,), float("nan"), dtype=torch.double)),
        ("noise", torch.ones(3, dtype=torch.float)),
    ):
        with pytest.raises(ValueError, match=f"MCMC {key}"):
            model.load_mcmc_samples({**draws, key: value})
    auto = port.SaasFullyBayesianSingleTaskGP(data[0][0].requires_grad_(), data[0][1])
    assert auto._tensor_backend == "torch"
    assert isinstance(auto, port.SaasFullyBayesianSingleTaskGP)


def test_tensor_mixture_preserves_type_and_output_major_moments_when_selected_or_scalarized():
    from botorch_rs.models._torch_backend_fully_bayesian import TorchGaussianMixturePosterior

    mean = torch.arange(12, dtype=torch.double).reshape(3, 2, 2).requires_grad_()
    covariance = torch.eye(4, dtype=torch.double).expand(3, 4, 4).clone().requires_grad_()
    posterior = TorchGaussianMixturePosterior(mean, covariance)
    selected = posterior.subset_output([1, 0])
    scalar = posterior.scalarize(torch.tensor([0.2, 0.8], dtype=mean.dtype))
    assert isinstance(selected, GaussianMixturePosterior)
    assert isinstance(scalar, GaussianMixturePosterior)
    assert selected._is_gaussian_mixture and scalar._is_gaussian_mixture
    torch.testing.assert_close(selected.mixture_mean, posterior.mixture_mean[..., [1, 0]])
    torch.testing.assert_close(
        scalar.mixture_mean, (posterior.mixture_mean @ mean.new_tensor([0.2, 0.8])).unsqueeze(-1)
    )
    variance = posterior.mixture_covariance_matrix.diagonal().reshape(2, 2).transpose(-1, -2)
    torch.testing.assert_close(variance, posterior.mixture_variance)
    derivative = torch.autograd.grad(
        scalar.mixture_variance.sum(), (mean, covariance), create_graph=True
    )
    assert all(torch.isfinite(value).all() for value in derivative)
