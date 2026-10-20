"""Coupled MAP SAAS priors, optimizers and ensemble laws on CPU/CUDA."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import functools
import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch import settings as reference_settings
from botorch.models import map_saas as reference
from gpytorch.mlls import ExactMarginalLogLikelihood as ReferenceMLL

from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import MaternKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import map_saas as port
from botorch_rs.models._torch_backend_modules import build_kernel
from botorch_rs.optim.fit import fit_gpytorch_mll_torch
from botorch_rs.optim.utils.model_utils import sample_all_priors

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device, batched=False, multi=False):
    x = torch.tensor(
        [[0.1, 0.7], [0.3, 0.2], [0.6, 0.6], [0.9, 0.9]], dtype=torch.double, device=device
    )
    y = torch.tensor([[0.1], [0.9], [0.4], [-0.2]], dtype=x.dtype, device=device)
    if multi:
        y = torch.cat([y, y.square() + 0.2], -1)
    if batched:
        x = torch.stack([x, x * 0.9])
        y = torch.stack([y, y + 0.1])
    return x.requires_grad_(), y.requires_grad_(), torch.full_like(y, 0.03, requires_grad=True)


def _pair(device, family="single", fixed=False, tau=None, batched=False, multi=False):
    data = _data(device, batched, multi)
    models = []
    for index, namespace in enumerate((port, reference)):
        x, y, noise = (value.detach().clone().requires_grad_() for value in data)
        kwargs = dict(outcome_transform=None)
        if index == 0:
            kwargs["backend"] = "torch"
        if family == "single":
            model = namespace.get_map_saas_model(x, y, noise if fixed else None, tau=tau, **kwargs)
        elif family == "additive":
            model = namespace.AdditiveMapSaasSingleTaskGP(
                x, y, noise if fixed else None, num_taus=2, **kwargs
            )
        else:
            model = namespace.EnsembleMapSaasSingleTaskGP(
                x,
                y,
                noise if fixed else None,
                num_taus=3,
                taus=x.new_tensor([0.03, 0.1, 0.5]),
                **kwargs,
            )
        models.append((model, (x, y, noise)))
    (actual, _), (expected, _) = models
    # Match every parameter explicitly; additive sparsity draws differ between
    # device RNGs, so install the same fixed hyperparameters separately.
    with torch.no_grad():
        expected_params = dict(expected.named_parameters())
        assert dict(actual.named_parameters()).keys() == expected_params.keys()
        for name, value in actual.named_parameters():
            expected_params[name].copy_(value)
        if family == "additive":
            for a, b in zip(
                actual.covar_module.kernels, expected.covar_module.kernels, strict=True
            ):
                b.base_kernel._priors["inv_lengthscale_prior"][
                    1
                ].__self__._tau = a.base_kernel.tau.detach().clone()
    return models


def _loss(model, x, y, index):
    model.train()
    if index == 0:
        return ExactMarginalLogLikelihood(model.likelihood, model)()
    return ReferenceMLL(model.likelihood, model)(model(model.train_inputs[0]), model.train_targets)


def _prediction_gradient(model, data, query, create_graph):
    x, y, noise = data
    model.eval()
    with reference_settings.propagate_grads(True), gpytorch.settings.fast_pred_var(False):
        posterior = model.posterior(query)
        value = (
            posterior.mean.square().sum()
            + 0.3 * posterior.distribution.covariance_matrix.square().sum()
        )
        parameters = dict(model.named_parameters())
        targets = (x, y, noise, query, *(parameters[name] for name in sorted(parameters)))
        gradient = torch.autograd.grad(
            value, targets, create_graph=create_graph, allow_unused=True, materialize_grads=True
        )
    return posterior, targets, gradient


def _reference_prediction_gradient_at_shift(family, original, data, query, shift):
    x, y, noise = ((value.detach() + shift).requires_grad_() for value in data)
    kwargs = dict(outcome_transform=None)
    if family == "single":
        model = reference.get_map_saas_model(x, y, noise, **kwargs)
    elif family == "additive":
        model = reference.AdditiveMapSaasSingleTaskGP(x, y, noise, num_taus=2, **kwargs)
    else:
        model = reference.EnsembleMapSaasSingleTaskGP(
            x, y, noise, num_taus=3, taus=x.new_tensor([0.03, 0.1, 0.5]), **kwargs
        )
    with torch.no_grad():
        parameters = dict(original.named_parameters())
        for name, value in model.named_parameters():
            value.copy_(parameters[name] + shift)
    return _prediction_gradient(
        model, (x, y, noise), (query.detach() + shift).requires_grad_(), False
    )[2]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("shape", ["single", "batch", "multi"])
def test_saas_prior_mll_complete_gradients_and_hessian(device, fixed, learned, shape):
    batch = shape != "single"
    tau = (
        None
        if learned
        else torch.tensor([0.1, 0.3] if batch else 0.1, dtype=torch.double, device=device)
    )
    pair = _pair(device, fixed=fixed, tau=tau, batched=shape == "batch", multi=shape == "multi")
    values, gradients, hvps = [], [], []
    for index, (model, (x, y, noise)) in enumerate(pair):
        value = _loss(model, x, y, index)
        parameters = dict(model.named_parameters())
        targets = (x, y, *(parameters[name] for name in sorted(parameters)))
        if fixed:
            targets = (*targets, noise)
        grad = torch.autograd.grad(value.sum(), targets, create_graph=True)
        direction = [torch.full_like(value, 0.2 + 0.03 * i) for i, value in enumerate(targets)]
        hvp = torch.autograd.grad(
            sum((a * b).sum() for a, b in zip(grad, direction, strict=True)), targets
        )
        values.append(value)
        gradients.append(grad)
        hvps.append(hvp)
        assert all(item.device.type == device for item in (*grad, *hvp))
    torch.testing.assert_close(values[0], values[1], atol=3e-8, rtol=3e-9)
    for groups in (gradients, hvps):
        for actual, expected in zip(*groups, strict=True):
            torch.testing.assert_close(actual, expected, atol=8e-8, rtol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["single", "additive", "ensemble"])
def test_map_family_prediction_device_and_query_data_parameter_graphs(device, family):
    pair = _pair(device, family=family, fixed=True)
    outputs = []
    query = None
    for index, (model, (x, y, noise)) in enumerate(pair):
        query = x.new_tensor(
            [[[0.25, 0.4], [0.72, 0.8]], [[0.31, 0.2], [0.8, 0.4]]], requires_grad=True
        )
        # Tau couples the prior, not prediction at a fixed lengthscale, so
        # its genuine zero derivative may be absent from the posterior graph.
        posterior, targets, grad = _prediction_gradient(model, (x, y, noise), query, index == 0)
        if index == 0:
            hvp = torch.autograd.grad(
                sum(item.sum() for item in grad), targets, allow_unused=True, materialize_grads=True
            )
        else:
            # GPyTorch's distinct-point Matérn uses cdist, whose backward has
            # no higher derivative in this runtime. Check the complete port
            # HVP against independent reference first-gradient differences.
            epsilon = 2e-5
            plus = _reference_prediction_gradient_at_shift(
                family, model, (x, y, noise), query, epsilon
            )
            minus = _reference_prediction_gradient_at_shift(
                family, model, (x, y, noise), query, -epsilon
            )
            hvp = tuple((a - b) / (2 * epsilon) for a, b in zip(plus, minus, strict=True))
        outputs.append((posterior.mean, posterior.distribution.covariance_matrix, grad, hvp))
    for a, b in zip(outputs[0][:2], outputs[1][:2], strict=True):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-8)
    for groups in zip(outputs[0][2:], outputs[1][2:], strict=True):
        for a, b in zip(*groups, strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_converted_native_saas_prior_is_retained_and_public_helper_graphs(device):
    x, y, _ = _data(device)
    source = MaternKernel(ard_num_dims=2)
    port.add_saas_prior(source)
    kernel = build_kernel(source, 2, (), x)
    assert hasattr(kernel, "raw_tau")
    # Preserve the descriptor's existing raw value and its construction
    # precision, rather than reinitializing a converted parameter.
    torch.testing.assert_close(kernel.raw_tau, x.new_tensor(source.raw_tau), atol=0, rtol=0)
    torch.testing.assert_close(kernel.tau, kernel.raw_tau.exp(), atol=0, rtol=0)
    helper = port.SaasPriorHelper()
    torch.testing.assert_close(
        helper.inv_lengthscale_prior_param_or_closure(kernel), kernel.inv_lengthscale
    )
    leaves = tuple(kernel.parameters())
    helper.tau_prior_setting_closure(kernel, x.new_tensor(0.2))
    helper.inv_lengthscale_prior_setting_closure(kernel, x.new_full((1, 2), 0.4))
    assert tuple(kernel.parameters()) == leaves
    torch.testing.assert_close(kernel.lengthscale, x.new_full((1, 2), 0.5).sqrt())
    torch.autograd.grad(kernel.log_prior(), leaves, create_graph=True)
    sample_all_priors(kernel)
    assert tuple(kernel.parameters()) == leaves
    assert all(item.device == x.device for item in kernel.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_real_adam_and_scipy_fits_saas_parameters(device):
    (model, (x, y, _)), (reference_model, (rx, ry, _)) = _pair(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    reference_optimizer = torch.optim.Adam(reference_model.parameters(), lr=0.02)
    for _ in range(4):
        for index, (fitted, a, b, opt) in enumerate(
            ((model, x, y, optimizer), (reference_model, rx, ry, reference_optimizer))
        ):
            opt.zero_grad()
            loss = -_loss(fitted, a, b, index).sum()
            loss.backward()
            opt.step()
    for name, actual in model.named_parameters():
        torch.testing.assert_close(
            actual, dict(reference_model.named_parameters())[name], atol=2e-8, rtol=3e-8
        )
    initial = -mll().detach().sum()
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 5}})
    assert bool(-mll().sum() < initial)
    assert model.covar_module.base_kernel.raw_tau.device.type == device
    previous = model.covar_module.base_kernel.raw_tau.detach().clone()
    fit_gpytorch_mll(
        mll,
        optimizer=fit_gpytorch_mll_torch,
        optimizer_kwargs={
            "step_limit": 2,
            "optimizer": functools.partial(torch.optim.SGD, lr=0.002),
        },
    )
    assert not torch.equal(previous, model.covar_module.base_kernel.raw_tau)


@pytest.mark.parametrize("device", DEVICES)
def test_ensemble_state_dict_restores_sparsity_and_law(device):
    (model, (x, _, _)), _ = _pair(device, family="ensemble", fixed=True)
    query = x.detach()[:2]
    original = model.posterior(query)
    state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.taus.mul_(2)
        model.covar_module.base_kernel.raw_lengthscale.add_(0.3)
    model.load_state_dict(state)
    restored = model.posterior(query)
    torch.testing.assert_close(original.mean, restored.mean, atol=0, rtol=0)
    torch.testing.assert_close(model.covar_module.base_kernel.tau, model.taus, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log_scale", [False, True])
def test_public_saas_configuration_constraints_and_prior_closures(device, log_scale):
    x, _, _ = _data(device)
    kernel = build_kernel(MaternKernel(ard_num_dims=2, batch_shape=(2,)), 2, (2,), x)
    expected = gpytorch.kernels.MaternKernel(ard_num_dims=2, batch_shape=torch.Size([2])).to(x)
    port.add_saas_prior(kernel, log_scale=log_scale)
    reference.add_saas_prior(expected, log_scale=log_scale)
    # Newly registered GPyTorch constraints are default-dtype/CPU modules.
    # Match the actual MAP model factory's final .to(train_X), giving both
    # complete modules the declared device/dtype before comparing graphs.
    expected.to(x)
    with torch.no_grad():
        expected.raw_tau.copy_(kernel.raw_tau)
        expected.raw_lengthscale.copy_(kernel.raw_lengthscale)
    actual = kernel.log_prior()
    ref = sum(
        prior.log_prob(get(owner)).reshape(2, -1).sum(-1)
        for _, owner, prior, get, _ in expected.named_priors()
    )
    torch.testing.assert_close(actual, ref, atol=2e-14, rtol=2e-14)
    first = torch.autograd.grad(
        actual.sum(), (kernel.raw_tau, kernel.raw_lengthscale), create_graph=True
    )
    second = torch.autograd.grad(
        sum(value.sum() for value in first), (kernel.raw_tau, kernel.raw_lengthscale)
    )
    rfirst = torch.autograd.grad(
        ref.sum(), (expected.raw_tau, expected.raw_lengthscale), create_graph=True
    )
    rsecond = torch.autograd.grad(
        sum(value.sum() for value in rfirst), (expected.raw_tau, expected.raw_lengthscale)
    )
    for group in ((first, rfirst), (second, rsecond)):
        for a, b in zip(*group, strict=True):
            torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    bounds = kernel.raw_parameter_bounds()
    assert math.isfinite(bounds["raw_tau"][0]) == log_scale


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_tau_external_tensor_retains_prior_derivatives(device):
    results = []
    for namespace in (port, reference):
        x, y, noise = _data(device, batched=True)
        tau = x.new_tensor([0.1, 0.3], requires_grad=True)
        kwargs = {"backend": "torch"} if namespace is port else {}
        model = namespace.get_map_saas_model(x, y, noise, tau=tau, outcome_transform=None, **kwargs)
        if namespace is port:
            value = model.covar_module.log_prior()
        else:
            value = sum(
                prior.log_prob(get(owner)).reshape(2, -1).sum(-1)
                for _, owner, prior, get, _ in model.covar_module.named_priors()
            )
        (first,) = torch.autograd.grad(value.sum(), tau, create_graph=True)
        (second,) = torch.autograd.grad(first.sum(), tau)
        results.append((value, first, second))
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_additive_covariance_factory_respects_dtype_device_and_active_columns(device):
    actual = port.get_additive_map_saas_covar_module(
        2, num_taus=2, active_dims=(0, 2), dtype=torch.double, device=torch.device(device)
    )
    expected = reference.get_additive_map_saas_covar_module(
        2, num_taus=2, active_dims=(0, 2), dtype=torch.double, device=torch.device(device)
    ).to(device=device, dtype=torch.double)
    # The pinned helper applies dtype/device only to each base kernel; its
    # ScaleKernel is otherwise created with default-dtype CPU parameters.
    # Explicitly promote the independent reference before matching leaves.
    with torch.no_grad():
        parameters = dict(expected.named_parameters())
        for name, value in actual.named_parameters():
            parameters[name].copy_(value)
    x = torch.tensor(
        [[0.1, 0.7, 0.4], [0.3, 0.2, 0.8]], device=device, dtype=torch.double, requires_grad=True
    )
    value, reference_value = actual(x, x + 0.03), expected(x, x + 0.03).to_dense()
    torch.testing.assert_close(value, reference_value, atol=3e-12, rtol=3e-12)
    assert all(buffer.device == x.device for buffer in actual.buffers())
    (gradient,) = torch.autograd.grad(value.sum(), x, create_graph=True)
    assert bool((gradient[:, 1] == 0).all())
    assert bool(torch.isfinite(torch.autograd.grad(gradient.sum(), x)[0]).all())
