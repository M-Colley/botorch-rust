"""Public optimization utilities retain device tensors and their selected graphs."""

import numpy as np
import pytest
import torch
from test_torch_acq_discrete import DEVICES, no_host

from botorch_rs.generation.gen import gen_candidates_torch, get_best_candidates
from botorch_rs.optim.parameter_constraints import (
    evaluate_feasibility,
    nonlinear_constraint_is_feasible,
)
from botorch_rs.optim.utils.acquisition_utils import columnwise_clamp, fix_features


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("insert", [False, True])
def test_public_clamp_fix_selected_graphs_match_source(device, insert):
    from botorch.generation.gen import get_best_candidates as reference_best
    from botorch.optim.utils.acquisition_utils import columnwise_clamp as reference_clamp
    from botorch.optim.utils.acquisition_utils import fix_features as reference_fix

    points = (
        torch.linspace(-0.2, 1.2, 12, dtype=torch.double, device=device)
        .reshape(2, 2, 3)
        .requires_grad_()
    )
    fixed = points.new_tensor([0.2, 0.8], requires_grad=True)
    X = points[..., :2] if insert else points
    values = points.new_tensor([0.1, 0.8])

    def apply(clamp, fix, best):
        return best(clamp(fix(X, {1: fixed}, replace_current_value=not insert), 0.0, 1.0), values)

    expected = apply(reference_clamp, reference_fix, reference_best)
    expected_grad = torch.autograd.grad(expected.square().sum(), (points, fixed), create_graph=True)
    with no_host():
        actual = apply(columnwise_clamp, fix_features, get_best_candidates)
        gradients = torch.autograd.grad(actual.square().sum(), (points, fixed), create_graph=True)
        second = torch.autograd.grad(sum(g.sum() for g in gradients), (points, fixed))
    torch.testing.assert_close(actual, expected)
    for actual_g, expected_g in zip(gradients, expected_grad, strict=True):
        torch.testing.assert_close(actual_g, expected_g)
    assert all(torch.isfinite(g).all() for g in second)


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_none_detaches_only_fixed_column_and_checks_keys(device):
    X = torch.rand(2, 3, 2, dtype=torch.double, device=device, requires_grad=True)
    with no_host():
        result = fix_features(X, {1: None})
        derivative = torch.autograd.grad(result.sum(), X)[0]
    torch.testing.assert_close(derivative[..., 0], torch.ones_like(X[..., 0]))
    assert not derivative[..., 1].any()
    for fixed in ({2: 0.2}, {1: 0.2, -1: 0.3}):
        with pytest.raises(ValueError):
            fix_features(X, fixed)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_public_feasibility_matches_independent_reference_all_constraint_types(device, batch):
    from botorch.optim.parameter_constraints import evaluate_feasibility as reference
    from botorch.optim.parameter_constraints import (
        nonlinear_constraint_is_feasible as reference_nonlinear,
    )

    generator = torch.Generator(device=device).manual_seed(37)
    X = torch.rand(*batch, 2, 3, dtype=torch.double, device=device, generator=generator)
    inequalities = [
        (torch.tensor([0, 2], device=device), X.new_tensor([1.0, -0.2]), 0.1),
        (torch.tensor([[0, 1], [1, 2]], device=device), X.new_tensor([1.0, 0.5]), 0.4),
    ]
    equalities = [(torch.tensor([1], device=device), X.new_tensor([1.0]), 0.5)]
    nonlinear = [(lambda x: x.sum() - 0.7, True), (lambda x: 1.1 - x[:, 0].sum(), False)]
    for eq in (None, equalities):
        expected = reference(X, inequalities, eq, nonlinear, tolerance=0.2)
        with no_host():
            actual = evaluate_feasibility(X, inequalities, eq, nonlinear, tolerance=0.2)
        torch.testing.assert_close(actual, expected)
    for fn, intra in nonlinear:
        expected = reference_nonlinear(fn, intra, X, tolerance=0.2)
        with no_host():
            actual = nonlinear_constraint_is_feasible(fn, intra, X, tolerance=0.2)
        torch.testing.assert_close(actual, expected)
    assert actual.device == X.device


@pytest.mark.parametrize("device", DEVICES)
def test_empty_batch_feasibility_and_model_batch_feature_values(device):
    X = torch.empty(0, 2, 3, dtype=torch.double, device=device)
    with no_host():
        assert evaluate_feasibility(X).shape == (0,)
        assert nonlinear_constraint_is_feasible(lambda x: x.sum(), True, X).shape == (0,)
        points = torch.ones(2, 3, 4, 2, dtype=torch.double, device=device, requires_grad=True)
        fixed = torch.arange(6, dtype=torch.double, device=device).reshape(2, 3).requires_grad_()
        result = fix_features(points, {1: fixed})
        derivative = torch.autograd.grad(result.sum(), fixed)[0]
    torch.testing.assert_close(result[..., 1], fixed[..., None].expand(2, 3, 4))
    torch.testing.assert_close(derivative, torch.full_like(fixed, 4.0))


@pytest.mark.parametrize("optimizer", [torch.optim.SGD, torch.optim.LBFGS])
@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_native_acquisition_custom_torch_optimizer_handles_attached_starts(optimizer, kind):
    from botorch_rs.acquisition.analytic import PosteriorMean
    from botorch_rs.models import SingleTaskGP

    model = SingleTaskGP([[0.0], [0.3], [0.8], [1.0]], [[0.0], [0.7], [0.2], [0.0]])
    acquisition = PosteriorMean(model)
    leaf = torch.tensor([[[0.1]], [[0.8]]], dtype=torch.double, requires_grad=True)
    initial = leaf * 1.0
    if kind == "list":
        initial = initial.detach().tolist()
    elif kind == "numpy":
        initial = initial.detach().numpy()
    options = {
        "optimizer_options": {"lr": 0.1},
        "stopping_criterion_options": {"maxiter": 12, "n_window": 12},
    }
    candidate, value = gen_candidates_torch(
        initial, acquisition, 0.0, 1.0, optimizer=optimizer, options=options
    )
    assert isinstance(candidate, {"list": list, "numpy": np.ndarray, "torch": torch.Tensor}[kind])
    torch.testing.assert_close(torch.as_tensor(value), torch.as_tensor(acquisition(candidate)))
    assert torch.isfinite(torch.as_tensor(candidate)).all()
    assert leaf.grad is None
