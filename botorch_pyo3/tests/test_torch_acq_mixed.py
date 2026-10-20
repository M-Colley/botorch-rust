"""Mixed alternating device algorithms and public search building blocks."""

import pytest
import torch
from test_torch_acq_discrete import DEVICES, no_host
from test_torch_acq_optimization_configs import Quadratic

from botorch_rs.optim import optimize_mixed as mixed
from botorch_rs.optim.optimize import OptimizeAcqfInputs


def inputs(bounds, acquisition=None, options=None, fixed=None, inequality=None, equality=None):
    return OptimizeAcqfInputs(
        acq_function=acquisition or Quadratic(),
        bounds=bounds,
        q=1,
        num_restarts=3,
        raw_samples=16,
        options=options or {},
        inequality_constraints=inequality,
        equality_constraints=equality,
        nonlinear_inequality_constraints=None,
        fixed_features=fixed,
        post_processing_func=None,
        batch_initial_conditions=None,
        return_best_only=False,
        gen_candidates=None,
        sequential=False,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_device_round_neighbors_and_complements_match_reference(device):
    from botorch.optim import optimize_mixed as reference

    X = torch.tensor([[0.1, 0.4, 0.7], [0.9, 0.6, 0.3]], dtype=torch.double, device=device)
    bounds = X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    dims = {0: [0.0, 0.4, 1.0], 2: [0.0, 0.5, 1.0]}
    categories = {1: [0.1, 0.2, 0.8]}
    expected_round = reference.round_discrete_dims(X, dims)
    expected_neighbors = reference.get_nearest_neighbors(expected_round[0], bounds, dims)
    expected_categories = reference.get_categorical_neighbors(expected_round[0], categories)
    with no_host():
        rounded = mixed.round_discrete_dims(X, dims)
        neighbors = mixed.get_nearest_neighbors(rounded[0], bounds, dims)
        categorical = mixed.get_categorical_neighbors(rounded[0], categories)
        complement = mixed.complement_indices_like(torch.tensor([0, 2], device=device), 3)
    torch.testing.assert_close(rounded, expected_round)
    torch.testing.assert_close(neighbors, expected_neighbors)
    torch.testing.assert_close(categorical, expected_categories)
    torch.testing.assert_close(complement, torch.tensor([1], device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_public_discrete_and_continuous_steps_match_actual_reference(device):
    from botorch.optim import optimize_mixed as reference

    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double, device=device)
    current = bounds.new_tensor([[0.0, 0.0, 0.2], [0.5, 0.0, 0.4], [1.0, 1.0, 0.9]])
    dd = {0: [0.0, 0.5, 1.0]}
    cd = {1: [0.0, 1.0]}
    options = {
        "maxiter_discrete": 4,
        "maxiter_continuous": 8,
        "init_batch_limit": 2,
        "num_spray_points": 0,
    }
    actual_inputs = inputs(bounds, options=options)
    reference_inputs = reference.OptimizeAcqfInputs(
        acq_function=Quadratic(),
        bounds=bounds,
        q=1,
        num_restarts=3,
        raw_samples=16,
        options=options,
        inequality_constraints=None,
        equality_constraints=None,
        nonlinear_inequality_constraints=None,
        fixed_features=None,
        post_processing_func=None,
        batch_initial_conditions=None,
        return_best_only=False,
        gen_candidates=reference.gen_candidates_scipy,
        sequential=False,
    )
    expected = reference.discrete_step(reference_inputs, dd, cd, current.clone())
    expected = reference.continuous_step(
        reference_inputs,
        torch.tensor([0], device=device),
        torch.tensor([1], device=device),
        expected[0],
    )
    with no_host():
        actual = mixed.discrete_step(actual_inputs, dd, cd, current.clone())
        actual = mixed.continuous_step(
            actual_inputs,
            torch.tensor([0], device=device),
            torch.tensor([1], device=device),
            actual[0],
        )
    torch.testing.assert_close(actual[0], expected[0], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("strategy", ["random", "equally_spaced", "continuous_relaxation"])
def test_real_device_mixed_alternating_strategies_and_pending_restore(device, strategy):
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double, device=device)
    acquisition = Quadratic()
    original = bounds.new_tensor([[0.95, 0.95, 0.95]])
    acquisition.set_X_pending(original)
    with no_host():
        candidate, value = mixed.optimize_acqf_mixed_alternating(
            acquisition,
            bounds,
            discrete_dims={0: [0.0, 1.0], 1: [0.0, 1.0]},
            q=2,
            num_restarts=3,
            raw_samples=16,
            options={
                "initialization_strategy": strategy,
                "num_spray_points": 0,
                "maxiter_init": 8,
                "maxiter_continuous": 8,
                "maxiter_alternating": 4,
                "init_batch_limit": 3,
                "batch_limit": 2,
            },
            seed=17,
        )
    torch.testing.assert_close(candidate[:, :2], candidate.new_ones(2, 2))
    torch.testing.assert_close(candidate[:, 2], candidate.new_full((2,), 0.73), atol=1e-6, rtol=0)
    torch.testing.assert_close(value, acquisition(candidate))
    assert acquisition.X_pending is original


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_large_subset_spray_and_feasible_initials(device):
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double, device=device)
    grid = torch.linspace(0, 1, 31, dtype=bounds.dtype, device=device)
    current = bounds.new_tensor([0.5, 0.2, 0.3])
    baseline = bounds.new_tensor([[0.5, 0.0, 0.2], [1.0, 1.0, 0.8]])
    dd = {0: [0.0, 0.5, 1.0]}
    cd = {1: grid}
    with no_host():
        neighbors = mixed.get_categorical_neighbors(current, cd, 5, seed=13)
        repeated = mixed.get_categorical_neighbors(current, cd, 5, seed=13)
        spray = mixed.get_spray_points(
            baseline, torch.tensor([2], device=device), dd, {1: [0.0, 1.0]}, bounds, 8, seed=13
        )
        starts = mixed.sample_feasible_points(inputs(bounds), dd, {1: [0.0, 1.0]}, 12, seed=13)
    torch.testing.assert_close(neighbors, repeated)
    assert neighbors.shape == (5, 3)
    assert spray.shape == (16, 3) and starts.shape == (12, 3)
    assert bool((spray[:, 0, None] == bounds.new_tensor([0.0, 0.5, 1.0])).any(-1).all())
    assert bool((starts[:, 1, None] == bounds.new_tensor([0.0, 1.0])).any(-1).all())


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_relaxes_large_ordinal_domain_and_preserves_fixed_features(device):
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double, device=device)
    values = torch.linspace(0, 1, 101, dtype=bounds.dtype, device=device)
    acquisition = Quadratic()
    with no_host():
        candidate, value = mixed.optimize_acqf_mixed_alternating(
            acquisition,
            bounds,
            discrete_dims={0: values},
            cat_dims={1: [0.0, 1.0]},
            fixed_features={-2: 1.0},
            raw_samples=8,
            num_restarts=2,
            options={
                "num_spray_points": 0,
                "maxiter_init": 10,
                "maxiter_continuous": 10,
                "maxiter_alternating": 3,
            },
            seed=13,
        )
    torch.testing.assert_close(candidate, bounds.new_tensor([[0.73, 1.0, 0.73]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(value, acquisition(candidate))


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_linear_constraints_and_continuous_equalities(device):
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double, device=device)
    inequality = [(torch.tensor([0, 1], device=device), bounds.new_tensor([-1.0, -1.0]), -1.0)]
    equality = [(torch.tensor([2], device=device), bounds.new_ones(1), 0.4)]
    candidate, value = mixed.optimize_acqf_mixed_alternating(
        Quadratic(),
        bounds,
        discrete_dims={0: [0.0, 0.5, 1.0]},
        cat_dims={1: [0.0, 1.0]},
        inequality_constraints=inequality,
        equality_constraints=equality,
        raw_samples=8,
        num_restarts=2,
        options={
            "num_spray_points": 0,
            "maxiter_init": 20,
            "maxiter_continuous": 20,
            "maxiter_alternating": 3,
        },
        seed=13,
    )
    assert bool((candidate[:, 0] + candidate[:, 1] <= 1.0 + 1e-7).all())
    torch.testing.assert_close(candidate[:, 2], bounds.new_tensor([0.4]), atol=1e-7, rtol=0)
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_exception_restores_original_pending(device):
    from test_torch_acq_discrete import FailAfterPending

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    acquisition = FailAfterPending()
    original = bounds.new_tensor([[0.95, 0.95]])
    acquisition.set_X_pending(original)
    with pytest.raises(RuntimeError, match="pending-dependent failure"), no_host():
        mixed.optimize_acqf_mixed_alternating(
            acquisition,
            bounds,
            discrete_dims={0: [0.0, 1.0]},
            q=2,
            num_restarts=2,
            raw_samples=8,
            options={
                "initialization_strategy": "random",
                "num_spray_points": 0,
                "maxiter_alternating": 2,
                "maxiter_continuous": 4,
            },
            seed=13,
        )
    assert acquisition.X_pending is original


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_real_gp_matches_reference_quality_and_keeps_training_grads_empty(device):
    from botorch.acquisition.analytic import PosteriorMean as RefMean
    from botorch.models import SingleTaskGP as RefGP
    from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating as reference

    from botorch_rs.acquisition.analytic import PosteriorMean
    from botorch_rs.models import SingleTaskGP

    X = torch.tensor(
        [[0.0, 0.1], [0.0, 0.5], [0.0, 0.9], [1.0, 0.1], [1.0, 0.5], [1.0, 0.9]],
        dtype=torch.double,
        device=device,
    )
    Y = (-(X[:, 0] - 1).square() - (X[:, 1] - 0.45).square()).unsqueeze(-1)
    actual_model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    reference_model = RefGP(X, Y, outcome_transform=None)
    actual = PosteriorMean(actual_model)
    expected = RefMean(reference_model)
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    options = {
        "initialization_strategy": "random",
        "num_spray_points": 0,
        "maxiter_alternating": 4,
        "maxiter_continuous": 12,
        "tol": 1e-9,
    }
    reference_candidate, reference_value = reference(
        expected,
        bounds,
        discrete_dims={0: [0.0, 1.0]},
        raw_samples=16,
        num_restarts=3,
        options=options,
    )
    with no_host():
        candidate, value = mixed.optimize_acqf_mixed_alternating(
            actual,
            bounds,
            discrete_dims={0: [0.0, 1.0]},
            raw_samples=16,
            num_restarts=3,
            options=options,
            seed=13,
        )
    torch.testing.assert_close(candidate, reference_candidate, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value.reshape(-1), reference_value.reshape(-1), atol=1e-8, rtol=1e-8)
    assert all(parameter.grad is None for parameter in actual_model.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_postprocessing_cannot_return_infeasible_candidates(device):
    from botorch_rs.exceptions.errors import CandidateGenerationError

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    with pytest.raises(CandidateGenerationError), no_host():
        mixed.optimize_acqf_mixed_alternating(
            Quadratic(),
            bounds,
            discrete_dims={0: [0.0, 1.0]},
            raw_samples=8,
            num_restarts=2,
            post_processing_func=lambda X: X.new_full(X.shape, 2.0),
            options={
                "initialization_strategy": "random",
                "num_spray_points": 0,
                "maxiter_alternating": 2,
                "maxiter_continuous": 4,
            },
            seed=13,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target", [0.73, -0.3, 1.3])
@pytest.mark.parametrize("restarts", [1, 3])
def test_box_lbfgs_does_not_stop_at_overshoot_false_stationary_point(device, target, restarts):
    from botorch_rs.generation.gen import gen_candidates_torch

    initial = torch.full((restarts, 1, 1), 0.2, dtype=torch.double, device=device)
    acquisition = Quadratic(target)
    with no_host():
        candidate, value = gen_candidates_torch(
            initial,
            acquisition,
            0.0,
            1.0,
            optimizer=torch.optim.LBFGS,
            options={
                "max_optimization_problem_aggregation_size": 1,
                "optimizer_options": {
                    "lr": 1.0,
                    "max_iter": 12,
                    "line_search_fn": "strong_wolfe",
                    "tolerance_grad": 1e-10,
                    "tolerance_change": 1e-12,
                },
                "stopping_criterion_options": {"maxiter": 1},
            },
        )
    expected = max(0.0, min(1.0, target))
    torch.testing.assert_close(
        candidate, initial.new_full(initial.shape, expected), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(value, acquisition(candidate))
