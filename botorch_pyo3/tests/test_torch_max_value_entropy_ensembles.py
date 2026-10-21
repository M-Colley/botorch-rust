"""MES keeps implicit ensemble axes distinct from both Monte Carlo sample axes."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch_rs.models import fully_bayesian
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FAMILIES = [
    "FullyBayesianSingleTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "FullyBayesianLinearSingleTaskGP",
]


class Component(torch.nn.Module):
    _tensor_backend = "torch"

    def __init__(self, model, index, source=False):
        super().__init__()
        self.base_model, self.index, self.source = model, index, source

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size()

    @property
    def train_inputs(self):
        return self.base_model.train_inputs

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        posterior = self.base_model.posterior(X, observation_noise=observation_noise)
        mean = posterior.mean[..., self.index, :, :]
        covariance = posterior.distribution.covariance_matrix[..., self.index, :, :]
        if self.source:
            from botorch.posteriors.gpytorch import GPyTorchPosterior
            from gpytorch.distributions import MultivariateNormal
            from linear_operator.operators import DenseLinearOperator

            posterior = GPyTorchPosterior(
                MultivariateNormal(mean.squeeze(-1), DenseLinearOperator(covariance))
            )
        else:
            posterior = TorchGPyTorchPosterior(mean, covariance)
        return posterior if posterior_transform is None else posterior_transform(posterior)


def make_model(device, family, source=False):
    X = torch.tensor(
        [[0.1, 0.2], [0.3, 0.6], [0.6, 0.4], [0.9, 0.8]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    Y = (4 * X[:, :1].detach()).sin().requires_grad_()
    namespace = fully_bayesian
    if source:
        namespace = pytest.importorskip("botorch.models.fully_bayesian")
    model = getattr(namespace, family)(
        X, Y, outcome_transform=None, **({} if source else {"backend": "torch"})
    )
    samples = {
        "mean": X.new_tensor([-0.1, 0.05, 0.2]),
        "noise": X.new_tensor([0.025, 0.04, 0.055]),
    }
    if family == "FullyBayesianLinearSingleTaskGP":
        samples["weight_variance"] = X.new_tensor([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    else:
        samples["lengthscale"] = X.new_tensor([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        samples["outputscale"] = X.new_tensor([0.8, 1.0, 1.2])
    model.load_mcmc_samples(samples)
    return model.eval(), X, Y


def identity_cost(X, deltas, **kwargs):
    return deltas


def acquisition(model, multi, count=4, source=False):
    classes = (qMaxValueEntropy, qMultiFidelityMaxValueEntropy)
    sampler = SobolQMCNormalSampler
    if source:
        from botorch.acquisition.max_value_entropy_search import (
            qMaxValueEntropy as SourceMES,
        )
        from botorch.acquisition.max_value_entropy_search import (
            qMultiFidelityMaxValueEntropy as SourceMF,
        )
        from botorch.sampling.normal import SobolQMCNormalSampler as sampler

        classes = (SourceMES, SourceMF)
    candidate = next(model.parameters()).new_tensor([[0.15, 0.7], [0.45, 0.3], [0.8, 0.9]])
    result = classes[multi](
        model,
        candidate,
        num_mv_samples=count,
        num_y_samples=8,
        use_gumbel=False,
        **({"cost_aware_utility": identity_cost} if multi else {}),
    )
    result.sampler = sampler(torch.Size([8]), seed=13)
    return result


def maxima(like, count=4):
    return torch.linspace(0.9, 1.8, count, dtype=like.dtype, device=like.device).unsqueeze(
        -1
    ) + like.new_tensor([0.0, 0.2, 0.4])


def component_mean(model, X, values, multi, source=False):
    parts = []
    for index in range(3):
        item = acquisition(Component(model, index, source), multi, len(values), source)
        item.posterior_max_values = values[:, index : index + 1]
        parts.append(item(X))
    return torch.stack(parts).mean(0)


def derivatives(value, leaves):
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(
        sum(
            (
                gradient
                * torch.linspace(0.2, 0.8, leaf.numel(), device=leaf.device)
                .to(leaf)
                .reshape_as(leaf)
            ).sum()
            for gradient, leaf in zip(first, leaves, strict=True)
        ),
        leaves,
    )
    return first, second


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("multi", [False, True])
def test_ensemble_mes_values_all_active_leaves_and_hvp_equal_independent_components(
    device, family, multi
):
    model, X, Y = make_model(device, family)
    values = maxima(X).requires_grad_()
    result = acquisition(model, multi)
    result.posterior_max_values = values
    query = torch.linspace(0.2, 0.7, 12, dtype=X.dtype, device=device).reshape(2, 3, 1, 2)
    query.requires_grad_()
    actual = result(query)
    expected = component_mean(model, query, values, multi)
    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
    leaves = (X, Y, query, values, *model.parameters())
    for left, right in zip(derivatives(actual, leaves), derivatives(expected, leaves), strict=True):
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", [FAMILIES[1], FAMILIES[2]])
@pytest.mark.parametrize("multi", [False, True])
def test_ensemble_mes_matches_unmodified_source_component_values_and_query_gradients(
    device, family, multi
):
    actual, X, _ = make_model(device, family)
    source, _, _ = make_model(device, family, source=True)
    result = acquisition(actual, multi)
    values = maxima(X)
    result.posterior_max_values = values
    query = X.new_tensor([[[0.2, 0.35]], [[0.55, 0.7]]], requires_grad=True)
    expected = component_mean(source, query, values, multi, source=True)
    value = result(query)
    torch.testing.assert_close(value, expected, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        torch.autograd.grad(value.sum(), query)[0],
        torch.autograd.grad(expected.sum(), query)[0],
        atol=3e-7,
        rtol=3e-6,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("count", [4, 8])
def test_original_source_implicit_axis_bug_unequal_draws_raise_equal_draws_are_wrong(
    device, multi, count
):
    source, X, _ = make_model(device, FAMILIES[1], source=True)
    result = acquisition(source, multi, count, source=True)
    values = maxima(X, count)
    result.posterior_max_values = values
    query = X.new_tensor([[[0.2, 0.35]], [[0.55, 0.7]]], requires_grad=True)
    expected = component_mean(source, query, values, multi, source=True)
    assert bool(torch.isfinite(expected).all())
    if count == 4:
        with pytest.raises(RuntimeError, match="size of tensor"):
            result(query)
    else:
        value = result(query)
        assert bool(torch.isfinite(value).all())
        assert not torch.allclose(value, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_ensemble_mes_has_no_inference_host_copy_and_sampler_module_state(device, monkeypatch):
    model, X, Y = make_model(device, FAMILIES[1])
    result = acquisition(model, False)
    result.posterior_max_values = maxima(X).requires_grad_()
    query = X.new_tensor([[[0.2, 0.35]], [[0.55, 0.7]]], requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Entropy inference copied model data to the host")

    with monkeypatch.context() as guard:
        for name in ("cpu", "numpy", "tolist"):
            guard.setattr(torch.Tensor, name, forbidden)
        derivatives(result(query), (X, Y, query, result.posterior_max_values, *model.parameters()))
    assert "sampler.base_samples" in result.state_dict()
