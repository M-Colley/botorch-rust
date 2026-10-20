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
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskGP as ReferenceGP

    pytest.importorskip("gpytorch")
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


@pytest.mark.parametrize("device", DEVICES)
def test_model_list_joint_law_mll_and_all_training_gradients(device):
    from botorch_rs.mlls import SumMarginalLogLikelihood
    from botorch_rs.models import ModelListGP

    X, Y, query = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    query.requires_grad_()
    first = SingleTaskGP(X, Y, backend="torch")
    other_X = (X.detach()[:6] * 0.9).requires_grad_()
    other_Y = Y.detach()[:6].cos().requires_grad_()
    second = SingleTaskGP(other_X, other_Y, backend="torch")
    model = ModelListGP(first, second)
    posterior = model.posterior(query)
    parts = [child.posterior(query) for child in model.models]
    assert posterior.mean.shape == (3, 2)
    torch.testing.assert_close(posterior.mean, torch.cat([p.mean for p in parts], -1))
    torch.testing.assert_close(
        posterior.distribution.covariance_matrix,
        torch.block_diag(*(p.distribution.covariance_matrix for p in parts)),
    )
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    value = mll()
    torch.testing.assert_close(
        value,
        sum(ExactMarginalLogLikelihood(child.likelihood, child)() for child in model.models) / 2,
    )
    loss = value + posterior.mean.square().sum() + posterior.variance.sum()
    leaves = (X, Y, other_X, other_Y, query, *model.parameters())
    gradients = torch.autograd.grad(loss, leaves, create_graph=True)
    second_derivatives = torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves)
    assert all(torch.isfinite(g).all() and g.device == X.device for g in second_derivatives)
    selected = model.posterior(query, output_indices=[1, 0])
    torch.testing.assert_close(selected.mean, posterior.mean.flip(-1))


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_sampler_normals_broadcast_and_remain_differentiable(device):
    X, Y, query = data(device, batch=(2,))
    model = SingleTaskGP(X, Y, backend="torch")
    posterior = model.posterior(query)
    base = torch.ones(4, 3, dtype=X.dtype, device=X.device, requires_grad=True)
    sampler = SobolQMCNormalSampler((4,), seed=3)
    sampler.base_samples = base
    samples = sampler(posterior)
    assert samples.shape == (4, 2, 3, 1)
    gradient = torch.autograd.grad(samples.square().sum(), base, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), base)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("indices", [[1], [1, 0]])
def test_output_subsets_preserve_moments_parameters_and_training_graph(device, fixed, indices):
    X, Y, query = data(device, batch=(2,), outputs=2)
    Y.requires_grad_()
    variance = torch.full_like(Y, 0.01) if fixed else None
    model = SingleTaskGP(X, Y, variance, backend="torch")
    expected = model.posterior(query, output_indices=indices)
    subset = model.subset_output(indices)
    actual = subset.posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )
    assert subset.num_outputs == len(indices)
    assert subset.covar_module.raw_lengthscale.shape == (
        (2, 1, 2) if len(indices) == 1 else (2, 2, 1, 2)
    )
    assert subset.covar_module.raw_lengthscale is not model.covar_module.raw_lengthscale
    gradient = torch.autograd.grad(actual.mean.square().sum(), Y, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), Y)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_custom_torch_module_selects_backend_and_keeps_network_parameter_graph(device):
    class NetworkKernel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Linear(2, 3, dtype=torch.double, device=device)

        def forward(self, a, b):
            distance = (
                (self.network(a).unsqueeze(-2) - self.network(b).unsqueeze(-3)).square().sum(-1)
            )
            return (-distance / 2).exp()

    X, Y, query = data(device)
    model = SingleTaskGP(X, Y, covar_module=NetworkKernel())
    assert model._tensor_backend == "torch"
    loss = model.posterior(query).mean.square().sum()
    parameters = tuple(model.covar_module.parameters())
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), parameters)
    assert all(torch.isfinite(g).all() for g in second)


@pytest.mark.parametrize("device", DEVICES)
def test_training_forward_does_not_factor_noiseless_covariance(device):
    X = torch.linspace(0, 1, 80, dtype=torch.double, device=device).unsqueeze(-1)
    model = SingleTaskGP(X, (X * 6).sin(), backend="torch")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with patch.object(torch.linalg, "cholesky", side_effect=AssertionError("eager latent factor")):
        output = model(*model.train_inputs)
    assert torch.isfinite(mll(output, model.train_targets)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_prediction_factor_cache_reuses_and_invalidates_actual_state(device):
    from botorch_rs import _botorch_rs as native

    X, Y, query = data(device)
    model = SingleTaskGP(X, Y, backend="torch")
    with patch.object(
        native, "tensor_exact_gp_factor", wraps=native.tensor_exact_gp_factor
    ) as factor:
        with torch.no_grad():
            model.posterior(query)
            model.posterior(query + 0.01)
            model.posterior(query + 0.02)
            assert factor.call_count == 1
            model.covar_module.raw_lengthscale.add_(0.01)
            actual = model.posterior(query).mean
            assert factor.call_count == 2
            model._prediction_cache = None
            torch.testing.assert_close(actual, model.posterior(query).mean)
            assert factor.call_count == 3
            Y.add_(0.01)
            model.posterior(query)
            assert factor.call_count == 4
            model.train()
            model.posterior(query)
            assert factor.call_count == 5
        # A full parameter graph must be fresh on each independent backward.
        for _ in range(2):
            model.posterior(query).mean.sum().backward()
        assert factor.call_count == 7


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_training_factor_preserves_query_double_backward(device):
    from botorch_rs import _botorch_rs as native

    X, Y, query = data(device)
    query.requires_grad_()
    model = SingleTaskGP(X, Y, backend="torch").requires_grad_(False)
    with patch.object(
        native, "tensor_exact_gp_factor", wraps=native.tensor_exact_gp_factor
    ) as factor:
        for _ in range(2):
            loss = model.posterior(query).mean.square().sum()
            gradient = torch.autograd.grad(loss, query, create_graph=True)[0]
            assert torch.isfinite(torch.autograd.grad(gradient.sum(), query)[0]).all()
        assert factor.call_count == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("on_train,on_eval", [(True, True), (False, True), (True, False)])
def test_public_input_transform_flags_and_training_call_protocol(device, on_train, on_eval):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.models.transforms.input import Normalize as ReferenceNormalize

    from botorch_rs.models.transforms.input import Normalize

    X, Y, query = data(device)
    bounds = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=X.dtype, device=X.device)
    flags = {"transform_on_train": on_train, "transform_on_eval": on_eval}
    model = SingleTaskGP(
        X, Y, backend="torch", input_transform=Normalize(2, bounds=bounds, **flags)
    )
    reference = ReferenceGP(X, Y, input_transform=ReferenceNormalize(2, bounds=bounds, **flags))
    torch.testing.assert_close(model.train_inputs[0], X)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    torch.testing.assert_close(mll(model(*model.train_inputs), model.train_targets), mll())
    actual = model.posterior(query)
    expected = reference.posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(actual.variance, expected.variance, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["log", "power", "bilog", "chain"])
def test_nonlinear_posterior_sampling_and_training_graphs(device, kind):
    from botorch_rs.models.transforms.outcome import (
        Bilog,
        ChainedOutcomeTransform,
        Log,
        Power,
        Standardize,
    )

    X, Y, query = data(device, outputs=2)
    X.requires_grad_()
    Y = Y.exp().detach().requires_grad_()
    query.requires_grad_()
    transforms = {
        "log": Log(),
        "power": Power(0.5),
        "bilog": Bilog(),
        "chain": ChainedOutcomeTransform(log=Log(), standardize=Standardize(2)),
    }
    model = SingleTaskGP(X, Y, outcome_transform=transforms[kind], backend="torch")
    posterior = model.posterior(query, output_indices=[1])
    sampler = IIDNormalSampler((4,), seed=13)
    samples = sampler(posterior)
    assert samples.shape == (4, 3, 1)
    assert samples.device == X.device
    loss = samples.square().sum() + ExactMarginalLogLikelihood(model.likelihood, model)().sum()
    leaves = (X, Y, query, *model.parameters())
    gradients = torch.autograd.grad(loss, leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves)
    assert all(torch.isfinite(g).all() for g in second)
    subset = model.subset_output([1])
    expected = model.posterior(query, output_indices=[1])
    normals = torch.ones((2,) + expected.base_sample_shape, dtype=X.dtype, device=X.device)
    torch.testing.assert_close(
        subset.posterior(query).rsample_from_base_samples((2,), normals),
        expected.rsample_from_base_samples((2,), normals),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_nonlinear_model_list_samples_keep_duplicate_outputs_identical(device):
    from botorch_rs.models import ModelListGP
    from botorch_rs.models.transforms.outcome import Log, Power

    X, Y, query = data(device)
    Y = Y.exp().detach().requires_grad_()
    first = SingleTaskGP(X, Y, backend="torch", outcome_transform=Log())
    second = SingleTaskGP(X[:6], Y[:6], backend="torch", outcome_transform=Power(0.5))
    posterior = ModelListGP(first, second).posterior(query, output_indices=[1, 0, 1])
    samples = IIDNormalSampler((3,), seed=31)(posterior)
    assert samples.shape == (3, 3, 3)
    torch.testing.assert_close(samples[..., 0], samples[..., 2], atol=0, rtol=0)
    gradients = torch.autograd.grad(
        samples.square().sum(), (Y, *first.parameters(), *second.parameters()), create_graph=True
    )
    assert all(torch.isfinite(g).all() for g in gradients)
