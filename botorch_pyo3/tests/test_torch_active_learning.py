"""Feasibility and active-learning laws keep complete device graphs."""

# ruff: noqa: E402
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.acquisition.active_learning import (
    PairwiseMCPosteriorVariance,
    qNegIntegratedPosteriorVariance,
)
from botorch_rs.acquisition.logei import qLogProbabilityOfFeasibility
from botorch_rs.acquisition.objective import GenericMCObjective, ScalarizedPosteriorTransform
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _model(device, outputs=1, fixed_noise=False):
    X = torch.tensor([[0.04], [0.27], [0.56], [0.81]], dtype=torch.double, device=device)
    Y = torch.cat(((X * 5.3).sin(), (X * 3.1).cos()), dim=-1)[..., :outputs]
    X, Y = X.requires_grad_(), Y.requires_grad_()
    noise = torch.full_like(Y, 0.04, requires_grad=True) if fixed_noise else None
    return SingleTaskGP(X, Y, noise, backend="torch", outcome_transform=None), (X, Y, noise)


def _derivatives(value, leaves):
    weights = torch.linspace(0.3, 0.9, value.numel(), dtype=value.dtype, device=value.device)
    gradient = torch.autograd.grad(
        (value.reshape(-1) * weights).sum(),
        leaves,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    gradient = tuple(leaf * 0 if g is None else g for leaf, g in zip(leaves, gradient, strict=True))
    hvp = torch.autograd.grad(
        sum(g.square().sum() for g in gradient), leaves, retain_graph=True, allow_unused=True
    )
    return gradient, tuple(
        leaf * 0 if g is None else g for leaf, g in zip(leaves, hvp, strict=True)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fat", [False, True])
@pytest.mark.parametrize("outputs,pending", [(1, False), (2, True)])
def test_log_feasibility_reference_values_weighted_gradients_and_hvp(device, fat, outputs, pending):
    reference = pytest.importorskip("botorch.acquisition.logei")
    model, data = _model(device, outputs)
    query = data[0].new_tensor([[[0.18], [0.63]], [[0.35], [0.72]]], requires_grad=True)
    eta = data[0].new_tensor([0.13, 0.2], requires_grad=True)
    constraints = [lambda samples: samples[..., 0] - 0.3, lambda samples: -samples[..., -1] - 0.4]
    arguments = dict(
        constraints=constraints,
        eta=eta,
        fat=fat,
        tau_max=0.08,
        X_pending=data[0].new_tensor([[0.94]]) if pending else None,
    )
    own = qLogProbabilityOfFeasibility(model, sampler=IIDNormalSampler((32,), seed=17), **arguments)
    expected_acq = reference.qLogProbabilityOfFeasibility(
        model, sampler=IIDNormalSampler((32,), seed=17), **arguments
    )
    actual, expected = own(query), expected_acq(query)
    leaves = (query, data[0], data[1], eta, *model.parameters())
    actual_derivatives, expected_derivatives = (
        _derivatives(actual, leaves),
        _derivatives(expected, leaves),
    )
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    for actual_order, expected_order in zip(actual_derivatives, expected_derivatives, strict=True):
        for a, b in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
    assert actual.device == query.device and actual.shape == (2,)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("q", [2, 4])
def test_pairwise_variance_reference_link_and_all_second_derivatives(device, outputs, q):
    reference = pytest.importorskip("botorch.acquisition.active_learning")
    model, data = _model(device, outputs)
    query = torch.linspace(0.12, 0.76, 2 * q, device=device, dtype=torch.double).reshape(2, q, 1)
    query.requires_grad_()
    objective = GenericMCObjective(lambda differences, X=None: differences.sum(-1).sigmoid())
    own = PairwiseMCPosteriorVariance(model, objective, IIDNormalSampler((24,), seed=37))
    expected_acq = reference.PairwiseMCPosteriorVariance(
        model, objective, IIDNormalSampler((24,), seed=37)
    )
    actual, expected = own(query), expected_acq(query)
    leaves = (query, data[0], data[1], *model.parameters())
    a, b = _derivatives(actual, leaves), _derivatives(expected, leaves)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for actual_order, expected_order in zip(a, b, strict=True):
        for actual_gradient, expected_gradient in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-9, rtol=2e-9)
    assert actual.device == query.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("fixed_noise,pending", [(False, False), (True, True)])
def test_integrated_variance_fantasy_protocol_reference_values_and_hvp(
    device, outputs, fixed_noise, pending
):
    reference = pytest.importorskip("botorch.acquisition.active_learning")
    model, data = _model(device, outputs, fixed_noise)
    points = data[0].new_tensor([[0.1], [0.45], [0.9]], requires_grad=True)
    query = data[0].new_tensor([[[0.19], [0.67]], [[0.31], [0.73]]], requires_grad=True)
    transform = (
        ScalarizedPosteriorTransform(data[0].new_tensor([0.7, -0.3])) if outputs > 1 else None
    )
    arguments = dict(
        mc_points=points,
        posterior_transform=transform,
        X_pending=data[0].new_tensor([[0.96]]) if pending else None,
    )
    own = qNegIntegratedPosteriorVariance(
        model, sampler=IIDNormalSampler((1,), seed=11), **arguments
    )
    expected_acq = reference.qNegIntegratedPosteriorVariance(
        model, sampler=IIDNormalSampler((1,), seed=11), **arguments
    )
    actual, expected = own(query), expected_acq(query)
    leaves = (
        query,
        points,
        data[0],
        data[1],
        *(() if data[2] is None else (data[2],)),
        *model.parameters(),
    )
    a, b = _derivatives(actual, leaves), _derivatives(expected, leaves)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    for ao, bo in zip(a, b, strict=True):
        for ag, bg in zip(ao, bo, strict=True):
            torch.testing.assert_close(ag, bg, atol=3e-9, rtol=3e-9)
    assert actual.shape == (2,) and actual.device == query.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["feasibility", "pairwise", "integrated"])
def test_active_learning_host_transfer_guards_and_repeated_backward(device, kind):
    model, data = _model(device)
    query = data[0].new_tensor([[[0.2], [0.65]]], requires_grad=True)
    if kind == "feasibility":
        acq = qLogProbabilityOfFeasibility(
            model, [lambda samples: samples[..., 0] - 0.3], sampler=IIDNormalSampler((12,), seed=7)
        )
    elif kind == "pairwise":
        acq = PairwiseMCPosteriorVariance(
            model,
            GenericMCObjective(lambda d, X=None: d[..., 0].sigmoid()),
            IIDNormalSampler((12,), seed=7),
        )
    else:
        acq = qNegIntegratedPosteriorVariance(model, data[0].new_tensor([[0.1], [0.5], [0.9]]))
    with ExitStack() as stack:
        for name in ("cpu", "numpy", "tolist"):
            stack.enter_context(
                patch.object(torch.Tensor, name, side_effect=AssertionError("host transfer"))
            )
        for _ in range(2):
            value = acq(query)
            gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
            second = torch.autograd.grad(gradient.square().sum(), query)[0]
            assert value.device == gradient.device == second.device == query.device
            assert bool(
                value.isfinite().all() & gradient.isfinite().all() & second.isfinite().all()
            )


@pytest.mark.parametrize("device", DEVICES)
def test_pairwise_q_validation_and_unused_feasibility_objective(device):
    model, data = _model(device, outputs=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("The unused feasibility objective was evaluated")

    acq = qLogProbabilityOfFeasibility(model, [lambda s: s[..., 0]], objective=forbidden)
    assert bool(acq(data[0][:2]).isfinite().all())
    pair = PairwiseMCPosteriorVariance(model, GenericMCObjective(lambda d, X=None: d.sum(-1)))
    for q in (0, 1, 3):
        with pytest.raises(RuntimeError, match="multiple of 2"):
            pair(data[0][:q])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_integrated_variance_matches_independent_joint_schur_law(device, outputs):
    model, data = _model(device, outputs, fixed_noise=True)
    query = data[0].new_tensor([[[0.2], [0.65]]], requires_grad=True)
    grid = data[0].new_tensor([[0.1], [0.5], [0.9]], requires_grad=True)
    weights = data[0].new_tensor([0.7, -0.3]) if outputs > 1 else None
    transform = ScalarizedPosteriorTransform(weights) if weights is not None else None
    acquisition = qNegIntegratedPosteriorVariance(model, grid, posterior_transform=transform)
    actual = acquisition(query)
    n, q = grid.shape[-2], query.shape[-2]
    joined = torch.cat((grid.expand(*query.shape[:-2], n, grid.shape[-1]), query), -2)
    joint = model.posterior(joined).distribution.covariance_matrix
    grid_indices = torch.cat([torch.arange(n, device=device) + o * (n + q) for o in range(outputs)])
    query_indices = torch.cat(
        [torch.arange(n, n + q, device=device) + o * (n + q) for o in range(outputs)]
    )
    grid_cov = joint.index_select(-2, grid_indices).index_select(-1, grid_indices)
    cross_cov = joint.index_select(-2, grid_indices).index_select(-1, query_indices)
    noisy_cov = model.posterior(query, observation_noise=True).distribution.covariance_matrix
    conditional = grid_cov - cross_cov @ torch.linalg.solve(noisy_cov, cross_cov.transpose(-1, -2))
    variance = conditional.diagonal(dim1=-2, dim2=-1).reshape(*query.shape[:-2], outputs, n)
    if weights is not None:
        variance = (variance * weights.square().unsqueeze(-1)).sum(-2)
    expected = -variance.mean(-1).squeeze(-1) if outputs == 1 else -variance.mean(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    leaves = (query, grid, data[0], data[2], *model.parameters())
    a, b = _derivatives(actual, leaves), _derivatives(expected, leaves)
    for ao, bo in zip(a, b, strict=True):
        for ag, bg in zip(ao, bo, strict=True):
            torch.testing.assert_close(ag, bg, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_actual_pairwise_gp_matches_reference_link_samples_and_query_gradients(device):
    pytest.importorskip("botorch")
    from botorch.acquisition.active_learning import PairwiseMCPosteriorVariance as ReferenceAcq
    from botorch.sampling.normal import IIDNormalSampler as ReferenceSampler
    from test_torch_pairwise_gp import _pair

    models, _, inputs = _pair(device, "probit")
    values, gradients = [], []
    for i, model in enumerate(models):
        query = inputs[i].new_tensor([[[0.25, 0.35], [0.6, 0.4]]], requires_grad=True)
        objective = GenericMCObjective(lambda differences, X=None: differences[..., 0].sigmoid())
        sampler = (ReferenceSampler if i else IIDNormalSampler)(torch.Size([32]), seed=23)
        acquisition = (ReferenceAcq if i else PairwiseMCPosteriorVariance)(
            model, objective, sampler
        )
        value = acquisition(query)
        values.append(value)
        gradients.append(torch.autograd.grad(value.sum(), query)[0])
    torch.testing.assert_close(values[0], values[1], atol=2e-8, rtol=2e-7)
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("kind", ["feasibility", "integrated"])
def test_independent_official_gp_values_query_gradients_and_hvp(device, dtype, outputs, kind):
    pytest.importorskip("botorch")
    from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance as ReferenceIPV
    from botorch.acquisition.logei import qLogProbabilityOfFeasibility as ReferencePF
    from botorch.acquisition.objective import ScalarizedPosteriorTransform as ReferenceTransform
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.sampling.normal import IIDNormalSampler as ReferenceSampler
    from gpytorch import settings as gp_settings

    X = torch.tensor([[0.04], [0.27], [0.56], [0.81]], dtype=dtype, device=device)
    Y = torch.cat(((X * 5.3).sin(), (X * 3.1).cos()), dim=-1)[..., :outputs]
    noise = torch.full_like(Y, 0.04)
    models = (
        SingleTaskGP(X, Y, noise, backend="torch", outcome_transform=None),
        ReferenceGP(X, Y, noise, outcome_transform=None),
    )
    results = []
    reference_acq = None
    for i, model in enumerate(models):
        query = X.new_tensor([[[0.18], [0.63]], [[0.35], [0.72]]], requires_grad=True)
        sampler = (ReferenceSampler if i else IIDNormalSampler)(torch.Size([16]), seed=47)
        if kind == "feasibility":
            acq = (ReferencePF if i else qLogProbabilityOfFeasibility)(
                model,
                [lambda samples: samples[..., 0] - 0.3],
                sampler=sampler,
                eta=0.15,
                tau_max=0.08,
                X_pending=X.new_tensor([[0.96]]),
            )
        else:
            transform = (
                (ReferenceTransform if i else ScalarizedPosteriorTransform)(
                    X.new_tensor([0.7, -0.3])
                )
                if outputs > 1
                else None
            )
            acq = (ReferenceIPV if i else qNegIntegratedPosteriorVariance)(
                model,
                X.new_tensor([[0.1], [0.5], [0.9]]),
                sampler=(ReferenceSampler if i else IIDNormalSampler)(torch.Size([1]), seed=11),
                posterior_transform=transform,
                X_pending=X.new_tensor([[0.96]]),
            )
        with gp_settings.fast_pred_var(False), gp_settings.detach_test_caches(False):
            value = acq(query)
            results.append((value, *_derivatives(value, (query,))))
        if i:
            reference_acq = acq
    tolerance = 3e-4 if dtype == torch.float32 else 3e-8
    torch.testing.assert_close(results[0][0], results[1][0], atol=tolerance, rtol=tolerance)
    for order, (actual_order, expected_order) in enumerate(
        zip(results[0][1:], results[1][1:], strict=True), start=1
    ):
        # Source sampled-posterior double backward is incorrect even with
        # exact prediction settings (torch-backend-acquisitions.md). The
        # common-posterior tests above compare the acquisition's full graph;
        # below we independently difference the source's correct first grad.
        if kind == "feasibility" and order == 2:
            continue
        torch.testing.assert_close(
            actual_order[0],
            expected_order[0],
            atol=tolerance,
            rtol=tolerance,
            msg=lambda message, order=order: f"Derivative order {order}: {message}",
        )
    if kind == "feasibility" and dtype == torch.float64:
        direction = 2 * results[0][1][0].detach()
        step = 1e-4
        finite_gradients = []
        for sign in (-1, 1):
            variable = (query.detach() + sign * step * direction).requires_grad_()
            with gp_settings.fast_pred_var(False), gp_settings.detach_test_caches(False):
                sampled_value = reference_acq(variable)
                weights = torch.linspace(
                    0.3, 0.9, sampled_value.numel(), device=device, dtype=dtype
                )
                finite_gradients.append(
                    torch.autograd.grad((sampled_value.reshape(-1) * weights).sum(), variable)[0]
                )
        finite_difference = (finite_gradients[1] - finite_gradients[0]) / (2 * step)
        torch.testing.assert_close(results[0][2][0], finite_difference, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("posterior_kind", ["gaussian", "transformed", "list"])
def test_public_sampler_selection_for_tensor_posteriors_preserves_device_and_graph(
    device, posterior_kind
):
    from botorch_rs.posteriors.posterior_list import PosteriorList
    from botorch_rs.posteriors.transformed import TransformedPosterior
    from botorch_rs.sampling.get_sampler import get_sampler

    model, data = _model(device)
    query = data[0].new_tensor([[0.2], [0.65]], requires_grad=True)
    posterior = model.posterior(query)
    if posterior_kind == "transformed":
        posterior = TransformedPosterior(posterior, sample_transform=lambda value: value.exp())
    elif posterior_kind == "list":
        posterior = PosteriorList(posterior, posterior)
    with ExitStack() as stack:
        for name in ("cpu", "numpy", "tolist"):
            stack.enter_context(
                patch.object(torch.Tensor, name, side_effect=AssertionError("host transfer"))
            )
        sampler = get_sampler(posterior, sample_shape=(8,), seed=31)
        samples = sampler(posterior)
        first = torch.autograd.grad(samples.square().sum(), query, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), query)[0]
    assert samples.shape == (8, 2, 2 if posterior_kind == "list" else 1)
    assert samples.device == query.device
    assert bool(first.isfinite().all() & second.isfinite().all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["feasibility", "pairwise"])
def test_active_learning_default_sampler_is_selected_lazily_and_reused(device, kind):
    from botorch_rs.sampling.normal import SobolQMCNormalSampler

    model, data = _model(device)
    query = data[0].new_tensor([[0.2], [0.65]])
    acquisition = (
        qLogProbabilityOfFeasibility(model, [lambda samples: samples[..., 0]])
        if kind == "feasibility"
        else PairwiseMCPosteriorVariance(
            model, GenericMCObjective(lambda values, X=None: values[..., 0])
        )
    )
    assert acquisition.sampler is None
    assert tuple(acquisition.sample_shape) == (512,)
    first = acquisition(query)
    sampler = acquisition.sampler
    assert isinstance(sampler, SobolQMCNormalSampler)
    torch.testing.assert_close(first, acquisition(query))
    assert acquisition.sampler is sampler
