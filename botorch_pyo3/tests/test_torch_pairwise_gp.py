"""Independent law, optimizer and higher-order checks for device preference GPs."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models._torch_backend_pairwise import _ImplicitMode
from botorch_rs.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch_rs.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch_rs.models.transforms.input import Normalize

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device, batch=False, gradients=False, dtype=None):
    x = torch.tensor(
        [[0.1, 0.2], [0.2, 0.6], [0.8, 0.5], [0.6, 0.9]],
        dtype=dtype or torch.double,
        device=device,
    )
    comparisons = torch.tensor([[1, 0], [2, 1], [2, 3], [3, 0], [2, 0]], device=device)
    if batch:
        x = torch.stack((x, x * 0.8 + 0.05))
        comparisons = comparisons.expand(2, -1, -1).clone()
    return x.requires_grad_(gradients), comparisons


def _pair(device, kind, batch=False, gradients=False, transforms=False):
    pytest.importorskip("botorch")
    from botorch.models.likelihoods import pairwise as reference_likelihood
    from botorch.models.pairwise_gp import PairwiseGP as Reference
    from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood as ReferenceMLL
    from botorch.models.transforms.input import Normalize as ReferenceNormalize

    models, mlls, inputs = [], [], []
    for i in range(2):
        x, comparisons = _data(device, batch, gradients)
        kwargs = {}
        if transforms:
            cls = ReferenceNormalize if i else Normalize
            kwargs["input_transform"] = cls(2, bounds=x.new_tensor([[0.0, 0.0], [1.0, 1.0]]))
        likelihood_class = PairwiseProbitLikelihood if kind == "probit" else PairwiseLogitLikelihood
        likelihood = (
            getattr(reference_likelihood, likelihood_class.__name__)() if i else likelihood_class()
        )
        model = (Reference if i else PairwiseGP)(
            x, comparisons, likelihood=likelihood, **kwargs, **({} if i else {"backend": "torch"})
        )
        mll = (ReferenceMLL if i else PairwiseLaplaceMarginalLogLikelihood)(model.likelihood, model)
        models.append(model)
        mlls.append(mll)
        inputs.append(x)
    return models, mlls, inputs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["probit", "logit"])
@pytest.mark.parametrize("batch", [False, True])
def test_mode_evidence_posterior_and_all_first_gradients_match_official(device, kind, batch):
    models, mlls, data = _pair(device, kind, batch, gradients=True, transforms=True)
    gradients, laws = [], []
    for model, mll, x in zip(models, mlls, data, strict=True):
        model.train()
        output = model(x)
        value = mll(output, model.unconsolidated_comparisons)
        parameters = dict(model.named_parameters())
        trainable = tuple(
            parameters[name] for name in sorted(parameters) if parameters[name].requires_grad
        )
        gradients.append(torch.autograd.grad(value, (x, *trainable)))
        query = x.new_tensor([[[0.25, 0.35], [0.6, 0.4]]], requires_grad=True)
        # The pinned reference's rank-only batch helper cannot broadcast
        # training batch2 against query batch1; expand that oracle explicitly.
        evaluated_query = query.expand(2, -1, -1) if batch else query
        posterior = model.posterior(evaluated_query if model is models[1] else query)
        derivative = torch.autograd.grad(
            posterior.mean.square().sum() + posterior.variance.sum(), query
        )[0]
        laws.append(
            (
                value,
                model.utility,
                posterior.mean,
                posterior.distribution.covariance_matrix,
                derivative,
            )
        )
    for actual, expected in zip(*laws, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-7)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["probit", "logit"])
@pytest.mark.parametrize("saturated", [False, True])
def test_implicit_mode_gradcheck_gradgradcheck_and_third_derivative(device, kind, saturated):
    x, comparisons = _data(device)
    model = PairwiseGP(x, comparisons, backend="torch")
    covariance = model.covar.detach().requires_grad_()
    mean = x.new_tensor(
        [9.0, -9.0, 9.0, -9.0] if saturated else [0.1, -0.2, 0.3, 0.1], requires_grad=True
    )
    limit = 3.0 if kind == "probit" else 8.0

    def function(K, mu):
        return _ImplicitMode.apply(K, mu, model.D, kind, limit, 1e-12, 100)

    assert torch.autograd.gradcheck(function, (covariance, mean), eps=1e-6, atol=3e-6, rtol=2e-5)
    assert torch.autograd.gradgradcheck(
        function, (covariance, mean), eps=1e-6, atol=8e-6, rtol=4e-5
    )
    utility = function(covariance, mean)
    first = torch.autograd.grad(utility.square().sum(), (covariance, mean), create_graph=True)
    second = torch.autograd.grad(
        sum(value.square().sum() for value in first), (covariance, mean), create_graph=True
    )
    third = torch.autograd.grad(sum(value.square().sum() for value in second), (covariance, mean))
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second, *third))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["probit", "logit"])
def test_complete_prediction_data_parameter_hvp_by_finite_gradients(device, kind):
    x, comparisons = _data(device, gradients=True)
    model = PairwiseGP(
        x,
        comparisons,
        likelihood=(PairwiseProbitLikelihood() if kind == "probit" else PairwiseLogitLikelihood()),
        input_transform=Normalize(2),
        backend="torch",
    )
    query = x.new_tensor([[0.25, 0.35], [0.6, 0.4]], requires_grad=True)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    targets = (x, query, *parameters)

    def gradient(create_graph):
        posterior = model.posterior(query)
        value = (
            posterior.mean.square().sum()
            + 0.3 * posterior.distribution.covariance_matrix.square().sum()
        )
        return torch.autograd.grad(value, targets, create_graph=create_graph)

    first = gradient(True)
    directions = tuple(
        torch.linspace(0.07, 0.19, t.numel(), dtype=t.dtype, device=t.device).reshape(t.shape)
        for t in targets
    )
    actual = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(first, directions, strict=True)), targets
    )
    originals = tuple(t.detach().clone() for t in targets)
    differences = []
    epsilon = 2e-5
    for sign in (-1, 1):
        with torch.no_grad():
            for target, original, direction in zip(targets, originals, directions, strict=True):
                target.copy_(original + sign * epsilon * direction)
        differences.append(gradient(False))
    with torch.no_grad():
        for target, original in zip(targets, originals, strict=True):
            target.copy_(original)
    for result, left, right in zip(actual, *differences, strict=True):
        torch.testing.assert_close(result, (right - left) / (2 * epsilon), atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_duplicates_conditioning_saved_graphs_and_real_adam(device):
    x, comparisons = _data(device, gradients=True)
    x = torch.cat((x, x[1:2]), -2).detach().requires_grad_()
    comparisons = torch.cat((comparisons, comparisons.new_tensor([[4, 0]])))
    model = PairwiseGP(x, comparisons, backend="torch")
    assert model.n == 4
    assert model.consolidated_indices.tolist() == [0, 1, 2, 3, 1]
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    old_output = model(model.unconsolidated_datapoints)
    old_value = mll(old_output, comparisons).detach().clone()
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.01)
    before = tuple(p.detach().clone() for p in model.parameters())
    for _ in range(3):
        optimizer.zero_grad()
        (-mll()).backward()
        optimizer.step()
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before, strict=True))
    torch.testing.assert_close(mll(old_output, comparisons).detach(), old_value, atol=0, rtol=0)
    # Every new call reconstructs data selections rather than reusing a freed
    # index_select graph from the first consolidated training table.
    posterior = model.posterior(x.new_tensor([[0.3, 0.4]]))
    torch.autograd.grad(posterior.mean.sum() + posterior.variance.sum(), x)
    new_x = x.new_tensor([[0.3, 0.4], [0.7, 0.6]], requires_grad=True)
    conditioned = model.condition_on_observations(new_x, comparisons.new_tensor([[1, 0]]))
    assert conditioned.n == 6
    assert all(
        a is not b for a, b in zip(model.parameters(), conditioned.parameters(), strict=True)
    )
    value = conditioned.posterior(new_x).mean.sum()
    assert bool(torch.isfinite(torch.autograd.grad(value, new_x)[0]).all())
    state = copy.deepcopy(model.state_dict())
    restored = PairwiseGP(x.detach(), comparisons, backend="torch")
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.posterior(new_x).mean, model.posterior(new_x).mean)


@pytest.mark.parametrize("device", DEVICES)
def test_self_comparison_is_constant_likelihood_after_consolidation(device):
    pytest.importorskip("botorch")
    from botorch.models.pairwise_gp import PairwiseGP as Reference

    x = torch.tensor(
        [[0.2, 0.3], [0.2, 0.3]], device=device, dtype=torch.double, requires_grad=True
    )
    comparisons = torch.tensor([[0, 1]], device=device)
    model = PairwiseGP(x, comparisons, backend="torch")
    assert model.n == 1
    torch.testing.assert_close(model.D, x.new_zeros(1, 1))
    torch.testing.assert_close(model.utility, x.new_zeros(1))
    torch.testing.assert_close(model.likelihood.p(model.utility, model.D), x.new_full((1,), 0.5))
    reference = Reference(x.detach(), comparisons)
    torch.testing.assert_close(reference.D, x.new_full((1, 1), -1.0))
    assert reference.utility.abs().item() > 0.4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["scipy", "torch"])
def test_public_fitting_and_loss_closure_optimize_actual_leaves(device, method):
    if method == "scipy":
        pytest.importorskip("scipy")
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.optim.closures.model_closures import get_loss_closure_with_grads
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    x, comparisons = _data(device)
    model = PairwiseGP(x, comparisons, backend="torch")
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    assert isinstance(mll, PairwiseLaplaceMarginalLogLikelihood)
    assert isinstance(model.likelihood, PairwiseProbitLikelihood)
    parameters = {name: p for name, p in mll.named_parameters() if p.requires_grad}
    closure = get_loss_closure_with_grads(mll, parameters=parameters)
    loss, gradients = closure()
    assert loss.device == x.device
    assert all(gradient.device == x.device for gradient in gradients)
    initial = mll().detach()
    if method == "scipy":
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 3}})
    else:
        fit_gpytorch_mll(
            mll,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={"step_limit": 3, "optimizer": lambda p: torch.optim.Adam(p, lr=0.01)},
        )
    assert mll().detach() > initial


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prior_only_and_float_device_graphs(device, dtype):
    from botorch_rs.exceptions.errors import UnsupportedError

    model = PairwiseGP(None, None, backend="torch").to(device=device, dtype=dtype)
    query = torch.tensor([[0.2, 0.3], [0.7, 0.4]], device=device, dtype=dtype, requires_grad=True)
    posterior = model.posterior(query)
    assert posterior.mean.shape == (2, 1)
    assert posterior.mean.device == query.device
    saved_mean = posterior.mean.clone()
    model.mean_module.constant = 0.2
    torch.testing.assert_close(posterior.mean, saved_mean, atol=0, rtol=0)
    first = torch.autograd.grad(posterior.variance.sum(), query, create_graph=True)[0]
    assert bool(torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all())
    with pytest.raises(RuntimeError, match="Training requires"):
        PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)()
    x, comparisons = _data(device, dtype=dtype)
    model = PairwiseGP(x, comparisons, backend="torch")
    posterior = model.posterior(query)
    value = posterior.mean.square().sum() + posterior.distribution.covariance_matrix.square().sum()
    targets = (query, *(p for p in model.parameters() if p.requires_grad))
    first = torch.autograd.grad(value, targets, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(torch.isfinite(g).all()) for g in (*first, *second))
    with pytest.raises(UnsupportedError, match="training output"):
        PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)(
            posterior.distribution, comparisons
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["probit", "logit"])
def test_evidence_mixed_hvp_against_independent_official_gradient_differences(device, kind):
    models, mlls, inputs = _pair(device, kind, gradients=True)
    all_targets, originals, directions = [], [], []
    for model, x in zip(models, inputs, strict=True):
        parameters = dict(model.named_parameters())
        targets = (
            x,
            *(parameters[name] for name in sorted(parameters) if parameters[name].requires_grad),
        )
        all_targets.append(targets)
        originals.append(tuple(t.detach().clone() for t in targets))
        directions.append(
            tuple(
                torch.linspace(0.07, 0.19, t.numel(), dtype=t.dtype, device=t.device).reshape(
                    t.shape
                )
                for t in targets
            )
        )
    value = mlls[0]()
    gradient = torch.autograd.grad(value, all_targets[0], create_graph=True)
    actual = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(gradient, directions[0], strict=True)), all_targets[0]
    )
    expected_parts = []
    epsilon = 2e-5
    for sign in (-1, 1):
        with torch.no_grad():
            for target, original, direction in zip(
                all_targets[1], originals[1], directions[1], strict=True
            ):
                target.copy_(original + sign * epsilon * direction)
        models[1].train()
        output = models[1](inputs[1])
        loss = mlls[1](output, models[1].unconsolidated_comparisons)
        expected_parts.append(torch.autograd.grad(loss, all_targets[1]))
    for result, left, right in zip(actual, *expected_parts, strict=True):
        torch.testing.assert_close(result, (right - left) / (2 * epsilon), atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_reference_batch_broadcast_reproduction_and_exact_solution(device):
    models, _, inputs = _pair(device, "probit", batch=True)
    query = inputs[0].new_tensor([[[0.25, 0.35], [0.6, 0.4]]])
    with pytest.raises(RuntimeError, match="expanded size of the tensor"):
        models[1].posterior(query)
    actual = models[0].posterior(query)
    expected = models[1].posterior(query.expand(2, -1, -1))
    torch.testing.assert_close(actual.mean, expected.mean, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_deepcopy_fitted_transform_and_independent_conditioning_parameters(device):
    x, comparisons = _data(device, gradients=True)
    model = PairwiseGP(x, comparisons, input_transform=Normalize(2), backend="torch")
    query = x.new_tensor([[0.2, 0.3]], requires_grad=True)
    expected = model.posterior(query).mean.detach()
    cloned = copy.deepcopy(model)
    torch.testing.assert_close(cloned.posterior(query).mean, expected)
    first = torch.autograd.grad(cloned.posterior(query).mean.sum(), (x, query), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (x, query))
    assert all(bool(torch.isfinite(g).all()) for g in (*first, *second))
