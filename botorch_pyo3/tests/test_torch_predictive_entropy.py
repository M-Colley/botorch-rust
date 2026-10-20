"""Device EP/PES value parity and derivatives within fixed damping branches."""

import pytest
import torch

from botorch_rs.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch_rs.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch_rs.models import ModelListGP, SingleTaskGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_pair(
    device,
    outputs=1,
    pending=False,
    model_list=False,
    normalized=False,
    maximize=True,
    dtype=torch.double,
):
    from botorch.acquisition.multi_objective.predictive_entropy_search import (
        qMultiObjectivePredictiveEntropySearch as ReferenceMOPES,
    )
    from botorch.acquisition.predictive_entropy_search import (
        qPredictiveEntropySearch as ReferencePES,
    )
    from botorch.models import ModelListGP as ReferenceList
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from mo_entropy_helpers import raw_train_inputs

    from botorch_rs.models.transforms.input import Normalize

    X = torch.tensor([[0.05], [0.3], [0.65], [0.9]], dtype=dtype, device=device)
    Y = 1 - 4 * (X - 0.45).square()
    if outputs == 2:
        Y = torch.cat((Y, 1 - 4 * (X - 0.65).square()), -1)
    if not maximize:
        Y = -Y
    pareto = X.new_tensor([[[0.39], [0.59]], [[0.43], [0.63]]])
    query = X.new_tensor([[[0.17], [0.77]], [[0.23], [0.83]]])
    pending_X = X.new_tensor([[0.92]]) if pending else None
    if normalized:
        X, pareto, query = X * 3 + 2, pareto * 3 + 2, query * 3 + 2
        if pending_X is not None:
            pending_X = pending_X * 3 + 2
    kwargs, refkwargs = {}, {}
    if normalized:
        kwargs["input_transform"] = Normalize(1)
        refkwargs["input_transform"] = ReferenceNormalize(1)
    if model_list:
        actual_model = ModelListGP(
            *[
                SingleTaskGP(
                    X, Y[..., i : i + 1], backend="torch", outcome_transform=None, **kwargs
                )
                for i in range(outputs)
            ]
        )
        reference_model = ReferenceList(
            *[
                ReferenceGP(X, Y[..., i : i + 1], outcome_transform=None, **refkwargs)
                for i in range(outputs)
            ]
        )
    else:
        actual_model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None, **kwargs)
        reference_model = ReferenceGP(X, Y, outcome_transform=None, **refkwargs)
    if outputs == 1:
        actual_class, reference_class = qPredictiveEntropySearch, ReferencePES
        optima = pareto[:, 0]
    else:
        actual_class, reference_class = qMultiObjectivePredictiveEntropySearch, ReferenceMOPES
        optima = pareto
    options = dict(X_pending=pending_X, max_ep_iterations=16, threshold=0.02, maximize=maximize)
    with raw_train_inputs(reference_model):
        reference = reference_class(reference_model, optima, **options)
    actual = actual_class(actual_model, optima, **options)
    return actual, reference, query


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,model_list", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("pending", [False, True])
def test_pes_values_and_query_first_second_derivatives(device, outputs, model_list, pending):
    actual, reference, query = make_pair(device, outputs, pending, model_list)
    query.requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=2e-6, atol=2e-7)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-4, atol=3e-5)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    assert value.device == query.device


@pytest.mark.parametrize("device", DEVICES)
def test_normalize_raw_training_locations_and_batched_queries(device):
    actual, reference, query = make_pair(device, 1, normalized=True)
    query = query[:, None].expand(2, 3, 2, 1).clone().requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=2e-6, atol=2e-7)
    rows = torch.stack([actual(row) for row in query.flatten(0, 1)]).reshape(2, 3)
    torch.testing.assert_close(value, rows, rtol=2e-8, atol=2e-9)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_pes_training_and_optimal_input_gradgradcheck(device):
    X = torch.tensor([[0.05], [0.4], [0.9]], dtype=torch.double, device=device, requires_grad=True)
    Y = (1 - 4 * (X.detach() - 0.45).square()).requires_grad_()
    optima = X.new_tensor([[0.43]], requires_grad=True)
    query = X.new_tensor([[0.21]], requires_grad=True)

    def evaluate(train_X, train_Y, optimal, points):
        model = SingleTaskGP(train_X, train_Y, backend="torch", outcome_transform=None)
        acquisition = qPredictiveEntropySearch(model, optimal, max_ep_iterations=4)
        return acquisition(points)

    arguments = X, Y, optima, query
    assert torch.autograd.gradcheck(evaluate, arguments, fast_mode=True, rtol=4e-3, atol=4e-4)
    assert torch.autograd.gradgradcheck(evaluate, arguments, fast_mode=True, rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_complete_parameter_graphs_repeated_backwards_without_host_data(
    device, outputs, monkeypatch
):
    actual, _, query = make_pair(device, outputs)
    query.requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference tensors were transferred to host data.")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    for _ in range(2):
        value = actual(query)
        parameters = (query, *actual.model.parameters())
        gradient = torch.autograd.grad(value.sum(), parameters, create_graph=True)
        hessian_product = torch.autograd.grad(sum(g.square().sum() for g in gradient), parameters)
        assert all(torch.isfinite(g).all() for g in (*gradient, *hessian_product))
        assert actual.ep_iterations > 0
        assert all(torch.isfinite(f).all() for f in actual.cached_factors)


@pytest.mark.parametrize("device", DEVICES)
def test_first_pivot_failure_requires_damping_and_reproduces_reference_error(device):
    from botorch.acquisition.multi_objective.predictive_entropy_search import (
        _update_damping as reference_damping,
    )

    from botorch_rs.acquisition._torch_backend_predictive_entropy import _update_damping

    old = torch.eye(2, dtype=torch.double, device=device)
    proposal = old.new_tensor([[-1.0, 0.0], [0.0, 1.0]]).requires_grad_()
    damping = old.new_tensor(1.0)
    original_damping, original_root = reference_damping(old, proposal, damping, 1e-4)
    if device == "cpu":
        # CPU Cholesky returns a finite invalid factor for info=1, so the
        # source's additional NaN check does not rescue the missed failure.
        assert original_damping == 1
        assert not torch.allclose(original_root @ original_root.T, proposal)
    else:
        # cuSOLVER may mark the failed factor with NaNs, independently
        # triggering the source's other fallback condition.
        assert original_damping == 0.5
    corrected_damping, corrected_root = _update_damping(old, proposal, damping, 1e-4)
    assert corrected_damping == 0.5
    expected = 0.5 * (old + proposal) + 1e-4 * old
    torch.testing.assert_close(corrected_root @ corrected_root.T, expected)
    first = torch.autograd.grad(corrected_root.square().sum(), proposal, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), proposal)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_log_cdf_extreme_tail_has_finite_matching_clamped_derivatives(device):
    from botorch.acquisition.multi_objective.predictive_entropy_search import (
        log_cdf_robust as reference_log_cdf,
    )

    from botorch_rs.acquisition._torch_backend_predictive_entropy import log_cdf_robust

    x = torch.tensor(
        [-100.0, -12.0, -3.0, 0.0, 4.0, 8.0, 100.0],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    value = log_cdf_robust(x)
    torch.testing.assert_close(value, reference_log_cdf(x), rtol=1e-14, atol=1e-14)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_minimization_values(device, outputs):
    actual, reference, query = make_pair(device, outputs, maximize=False)
    torch.testing.assert_close(actual(query), reference(query), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_values_and_finite_query_hvp(device):
    actual, reference, query = make_pair(device, outputs=2, dtype=torch.float32)
    query.requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=3e-3, atol=3e-4)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    assert value.dtype == torch.float32
