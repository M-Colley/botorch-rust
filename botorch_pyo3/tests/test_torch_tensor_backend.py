"""Public exact-GP Tensor backend: real device execution and full autograd."""

# ruff: noqa: E402

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.posteriors.gpytorch import scalarize_posterior
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device="cpu", dtype=torch.double, batch=(), outputs=1):
    generator = torch.Generator(device=device).manual_seed(42)
    X = torch.rand(*batch, 8, 2, generator=generator, device=device, dtype=dtype)
    Y = (X[..., :1] * 6).sin() + X[..., 1:].square()
    Y = torch.cat([Y + 0.25 * j for j in range(outputs)], dim=-1)
    query = torch.rand(*batch, 3, 2, generator=generator, device=device, dtype=dtype)
    return X, Y, query


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.double, torch.float32])
@pytest.mark.parametrize("batch,outputs", [((), 1), ((), 2), ((2,), 1), ((2,), 2)])
@pytest.mark.parametrize("fixed", [False, True])
def test_exact_posterior_and_mll_reference(device, dtype, batch, outputs, fixed):
    from botorch.models import SingleTaskGP as ReferenceGP
    from gpytorch.mlls import ExactMarginalLogLikelihood as ReferenceMLL

    X, Y, query = data(device, dtype, batch, outputs)
    Yvar = torch.full_like(Y, 0.015) if fixed else None
    model = SingleTaskGP(X, Y, Yvar, backend="torch")
    reference = ReferenceGP(X, Y, Yvar)
    model.covar_module.lengthscale = reference.covar_module.lengthscale.detach()
    model.mean_module.constant = reference.mean_module.constant.detach()
    if not fixed:
        model.likelihood.noise = reference.likelihood.noise.detach()
    actual = model.posterior(query, observation_noise=True)
    expected = reference.posterior(query, observation_noise=True)
    tolerance = 3e-4 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(actual.mean, expected.mean, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual.variance, expected.variance, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=tolerance,
        rtol=tolerance,
    )
    model.train()
    reference.train()
    actual_mll = ExactMarginalLogLikelihood(model.likelihood, model)
    expected_mll = ReferenceMLL(reference.likelihood, reference)
    expected_value = expected_mll(reference(*reference.train_inputs), reference.train_targets)
    torch.testing.assert_close(actual_mll(), expected_value, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        actual_mll(model(*model.train_inputs), model.train_targets),
        actual_mll(),
        atol=tolerance,
        rtol=tolerance,
    )
    assert actual.mean.device == X.device and actual.mean.dtype == X.dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("standardize", [False, True])
def test_training_data_query_gradcheck_and_gradgradcheck(device, standardize):
    X, Y, query = data(device)
    X, Y, query = (x[:4].clone().requires_grad_() for x in (X, Y, query))

    def evaluate(x, y, q):
        kwargs = {} if standardize else {"outcome_transform": None}
        model = SingleTaskGP(x, y, backend="torch", **kwargs)
        posterior = model.posterior(q)
        return (
            posterior.mean.square().sum()
            + posterior.variance.sum()
            + ExactMarginalLogLikelihood(model.likelihood, model)().sum()
        )

    assert torch.autograd.gradcheck(evaluate, (X, Y, query), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (X, Y, query), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_device_forward_backward_does_not_copy_tensor_data_to_host(device, outputs):
    X, Y, query = data(device, outputs=outputs)
    X.requires_grad_()
    Y.requires_grad_()
    query.requires_grad_()
    model = SingleTaskGP(X, Y, backend="torch")
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("list copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy copy")),
    ):
        posterior = model.posterior(query)
        samples = posterior.rsample_from_base_samples(
            (2,), torch.ones((2,) + posterior.base_sample_shape, dtype=X.dtype, device=X.device)
        )
        loss = posterior.mean.square().sum() + samples.square().sum()
        loss = loss + ExactMarginalLogLikelihood(model.likelihood, model)().sum()
        leaves = (X, Y, query, *model.parameters())
        gradients = torch.autograd.grad(loss, leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves)
    assert all(torch.isfinite(g).all() and g.device == X.device for g in second)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler_type", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize("outputs", [1, 2])
def test_sampler_scalarization_and_repeated_backward(device, sampler_type, outputs):
    X, Y, query = data(device, batch=(2,), outputs=outputs)
    model = SingleTaskGP(X, Y, backend="torch")
    query.requires_grad_()
    sampler = sampler_type((4,), seed=11)
    for _ in range(2):
        posterior = model.posterior(query)
        if outputs > 1:
            weights = torch.tensor([0.25, 0.75], dtype=X.dtype, device=X.device, requires_grad=True)
            posterior = scalarize_posterior(posterior, weights, offset=0.2)
        samples = sampler(posterior)
        assert samples.shape == (4, 2, 3, 1)
        gradients = torch.autograd.grad(
            samples.square().sum(), (query, *model.parameters()), create_graph=True
        )
        assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_optimizer_updates_live_parameters_and_respects_frozen_leaves(device):
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    X, Y, _ = data(device)
    model = SingleTaskGP(X, Y, backend="torch")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.mean_module.raw_constant.requires_grad_(False)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = fit_gpytorch_mll_torch(mll, step_limit=4, stopping_criterion=None)
    after = dict(model.named_parameters())
    assert result.step == 4
    assert torch.equal(before["mean_module.raw_constant"], after["mean_module.raw_constant"])
    assert any(not torch.equal(before[name], p) for name, p in after.items() if p.requires_grad)
    assert all(p.device == X.device for p in after.values())


@pytest.mark.parametrize("device", DEVICES)
def test_conditioning_and_fantasy_training_graphs(device):
    X, Y, query = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    model = SingleTaskGP(X, Y, backend="torch")
    query.requires_grad_()
    observed = torch.ones(3, 1, dtype=X.dtype, device=X.device, requires_grad=True)
    conditioned = model.condition_on_observations(query, observed)
    prediction = conditioned.posterior(query[:2] + 0.03)
    gradients = torch.autograd.grad(
        prediction.mean.square().sum(), (X, Y, query, observed), create_graph=True
    )
    assert all(torch.isfinite(g).all() for g in gradients)
    fantasy = model.fantasize(query, IIDNormalSampler((2,), seed=4), propagate_grads=True)
    assert fantasy.batch_shape == (2,)
    result = fantasy.posterior(query[:1] + 0.05).mean.sum()
    gradient = torch.autograd.grad(result, query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), query)[0]).all()


def test_backend_selection_and_invalid_output_indices():
    X, Y, query = data()
    native = SingleTaskGP(X, Y)
    assert getattr(native, "_tensor_backend", None) is None
    assert SingleTaskGP(X.requires_grad_(), Y)._tensor_backend == "torch"
    model = SingleTaskGP(X, Y, backend="torch")
    with pytest.raises(IndexError):
        model.posterior(query, output_indices=[5])
    with pytest.raises(ValueError, match="backend"):
        SingleTaskGP(X, Y, backend="invalid")
