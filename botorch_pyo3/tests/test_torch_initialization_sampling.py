# ruff: noqa: E402
"""Device affine-polytope laws, feasibility and real constrained initialization."""

import pytest

pytest.importorskip("torch")


import torch
from test_torch_acq_discrete import DEVICES, no_host
from test_torch_acq_optimization_configs import Quadratic

from botorch_rs.exceptions.errors import InfeasibilityError
from botorch_rs.optim._torch_initialization_sampling import sample_polytope
from botorch_rs.optim.initializers import (
    gen_batch_initial_conditions,
    sample_q_batches_from_polytope,
    transform_constraints,
)
from botorch_rs.optim.optimize import optimize_acqf


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [1, 2])
def test_triangle_uniform_moments_and_feasibility(device, q):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    constraint = (torch.tensor([0, 1], device=device), bounds.new_tensor([-1.0, -1.0]), -1.0)
    with no_host():
        points = sample_q_batches_from_polytope(
            3072, q, bounds, 96, 8, seed=17, inequality_constraints=[constraint]
        )
    assert points.shape == (3072, q, 2)
    assert bool((points >= -1e-9).all() & (points.sum(-1) <= 1 + 1e-9).all())
    torch.testing.assert_close(
        points.mean((0, 1)), bounds.new_full((2,), 1 / 3), atol=0.015, rtol=0
    )
    torch.testing.assert_close(
        points.square().mean((0, 1)), bounds.new_full((2,), 1 / 6), atol=0.015, rtol=0
    )
    # The independent reference samples the same stationary law with its own RNG.
    pytest.importorskip("botorch")
    from botorch.optim.initializers import sample_q_batches_from_polytope as reference

    expected = reference(3072, q, bounds, 96, 8, seed=17, inequality_constraints=[constraint])
    torch.testing.assert_close(
        points.mean((0, 1)), expected.mean((0, 1)).to(points), atol=0.025, rtol=0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_affine_equalities_interpoint_fixed_and_degenerate_features(device):
    bounds = torch.tensor([[0.0, 0.0, 2.0], [1.0, 1.0, 2.0]], dtype=torch.double, device=device)
    index = torch.tensor([0, 1], device=device)
    inter = torch.tensor([[0, 0], [1, 0]], device=device)
    kwargs = dict(
        n=128,
        q=2,
        bounds=bounds,
        n_burnin=16,
        n_thinning=2,
        seed=13,
        equality_constraints=[(index, bounds.new_ones(2), 1.0)],
        inequality_constraints=[(inter, bounds.new_tensor([1.0, 1.0]), 0.5)],
        fixed_features={0: 0.3},
    )
    with no_host():
        points = sample_polytope(**kwargs)
        same = sample_polytope(**kwargs)
    torch.testing.assert_close(points, same)
    torch.testing.assert_close(points[..., :2], bounds.new_tensor([0.3, 0.7]).expand(128, 2, 2))
    torch.testing.assert_close(points[..., 2], bounds.new_full((128, 2), 2.0))


@pytest.mark.parametrize("device", DEVICES)
def test_joint_q_inequality_and_equality_laws(device):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    index = torch.tensor([[0, 0], [1, 0]], device=device)
    constraints = [(index, bounds.new_ones(2), 1.0)]
    with no_host():
        points = sample_polytope(2048, 2, bounds, 64, 2, 9, equality_constraints=constraints)
    torch.testing.assert_close(points.sum(-2), bounds.new_ones(2048, 1), atol=1e-10, rtol=0)
    torch.testing.assert_close(points[:, 0].mean(), bounds.new_tensor(0.5), atol=0.025, rtol=0)
    assert points[:, 0].std() > 0.25


@pytest.mark.parametrize("device", DEVICES)
def test_constraint_transforms_match_reference_without_host_conversion(device):
    pytest.importorskip("botorch")
    from botorch.optim.initializers import transform_constraints as reference

    constraints = [
        (torch.tensor([0, 2], device=device), torch.tensor([1.0, -2.0], device=device), 0.3),
        (
            torch.tensor([[0, 1], [1, 2]], device=device),
            torch.tensor([0.2, 0.5], device=device),
            0.1,
        ),
    ]
    expected = reference(constraints, 2, 3)
    with no_host():
        actual = transform_constraints(constraints, 2, 3)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left[0], right[0])
        torch.testing.assert_close(left[1], right[1])
        assert left[2] == right[2]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("inter_point", [False, True])
def test_constraint_invalid_feature_error_includes_actual_dimension(device, dimension, inter_point):
    indices = torch.tensor([[0, dimension]] if inter_point else [0, dimension], device=device)
    coefficients = torch.ones(indices.shape[0], dtype=torch.double, device=device)
    with pytest.raises(ValueError) as error:
        transform_constraints([(indices, coefficients, 0.0)], q=2, d=dimension)
    assert str(error.value) == (
        f"Constraint indices cannot exceed the problem dimension d={dimension}."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_constrained_initialization_and_real_slsqp_search(device):
    pytest.importorskip("scipy")
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    constraint = (torch.tensor([0, 1], device=device), bounds.new_tensor([-1.0, -1.0]), -0.8)
    acquisition = Quadratic()
    options = {"seed": 13, "n_burnin": 16, "n_thinning": 2, "init_batch_limit": 5, "maxiter": 40}
    with no_host():
        initial = gen_batch_initial_conditions(
            acquisition, bounds, 1, 4, 32, options=options, inequality_constraints=[constraint]
        )
    assert bool((initial.sum(-1) <= 0.8 + 1e-9).all())
    candidate, value = optimize_acqf(
        acquisition, bounds, 1, 4, 32, options=options, inequality_constraints=[constraint]
    )
    torch.testing.assert_close(candidate, bounds.new_full((1, 2), 0.4), atol=1e-5, rtol=0)
    torch.testing.assert_close(value, acquisition(candidate))


@pytest.mark.parametrize("device", DEVICES)
def test_inconsistent_affine_constraints_fail_explicitly(device):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    index = torch.tensor([0], device=device)
    with no_host(), pytest.raises(InfeasibilityError):
        sample_polytope(2, 1, bounds, 0, 1, equality_constraints=[(index, bounds.new_ones(1), 2.0)])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("qmc", [False, True])
def test_truncated_public_helper_matches_reference_and_preserves_graph(device, qmc):
    pytest.importorskip("botorch")
    from botorch.utils.sampling import sample_truncated_normal_perturbations as reference

    from botorch_rs.utils.sampling import sample_truncated_normal_perturbations

    centers = torch.tensor(
        [[0.2, 0.4], [0.8, 0.6]], dtype=torch.double, device=device, requires_grad=True
    )
    bounds = centers.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    torch.manual_seed(19)
    expected = reference(centers, 16, 0.15, bounds, qmc)
    torch.manual_seed(19)
    with no_host():
        actual = sample_truncated_normal_perturbations(centers, 16, 0.15, bounds, qmc)
        gradient = torch.autograd.grad(actual.square().sum(), centers, create_graph=True)[0]
        second = torch.autograd.grad(gradient.sum(), centers)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert bool(torch.isfinite(gradient).all() & torch.isfinite(second).all())


@pytest.mark.parametrize("device", DEVICES)
def test_seeded_perturbation_gradcheck_and_subset_center_law(device):
    from botorch_rs.utils.sampling import (
        sample_perturbed_subset_dims,
        sample_truncated_normal_perturbations,
    )

    centers = torch.tensor([[0.3, 0.6]], dtype=torch.double, device=device, requires_grad=True)
    bounds = centers.new_tensor([[0.0, 0.0], [1.0, 1.0]])

    def function(X):
        return sample_truncated_normal_perturbations(X, 3, 0.15, bounds, seed=17)

    with no_host():
        assert torch.autograd.gradcheck(function, (centers,))
        assert torch.autograd.gradgradcheck(function, (centers,))
        wide_centers = centers.detach().new_tensor([[0.1] * 24, [0.9] * 24])
        wide_bounds = torch.stack((wide_centers.new_zeros(24), wide_centers.new_ones(24)))
        samples = sample_perturbed_subset_dims(
            wide_centers, wide_bounds, 128, sigma=1e-5, prob_perturb=0.1, seed=17
        )
        same = sample_perturbed_subset_dims(
            wide_centers, wide_bounds, 128, sigma=1e-5, prob_perturb=0.1, seed=17
        )
    torch.testing.assert_close(samples, same)
    # All dimensions stay close to the same selected center, including changed ones.
    assert bool(
        ((samples - 0.1).abs().amax(-1) < 1e-3)
        .logical_or((samples - 0.9).abs().amax(-1) < 1e-3)
        .all()
    )


@pytest.mark.parametrize("device", DEVICES)
def test_around_best_reference_selection_seed_and_constrained_raw_feasibility(device):
    from types import SimpleNamespace

    pytest.importorskip("botorch")
    from botorch.optim.initializers import sample_points_around_best as reference

    from botorch_rs.optim.initializers import sample_points_around_best
    from botorch_rs.optim.utils.acquisition_utils import get_X_baseline

    X = torch.tensor(
        [[0.1, 0.2], [0.3, 0.5], [0.6, 0.3], [0.8, 0.9]], dtype=torch.double, device=device
    )
    mean = X[:, 0:1]
    acquisition = SimpleNamespace(
        _tensor_backend="torch",
        X_baseline=X,
        model=SimpleNamespace(posterior=lambda X: SimpleNamespace(mean=mean)),
        objective=None,
        maximize=True,
        constraints=None,
    )
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    torch.manual_seed(19)
    expected = reference(acquisition, 16, 0.01, bounds)
    torch.manual_seed(19)
    with no_host():
        actual = sample_points_around_best(acquisition, 16, 0.01, bounds)
        assert get_X_baseline(acquisition) is X
        first = sample_points_around_best(acquisition, 16, 0.01, bounds, seed=13)
        second = sample_points_around_best(acquisition, 16, 0.01, bounds, seed=13)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(first, second)
    # The raw initializer must not choose infeasible around-best draws.
    quadratic = Quadratic()
    quadratic.model = acquisition.model
    quadratic.X_baseline = X
    constraint = (torch.tensor([0, 1], device=device), bounds.new_tensor([-1.0, -1.0]), -0.8)
    with no_host():
        starts = gen_batch_initial_conditions(
            quadratic,
            bounds,
            1,
            4,
            32,
            options={"seed": 13, "n_burnin": 16, "n_thinning": 2, "sample_around_best": True},
            inequality_constraints=[constraint],
        )
    assert bool((starts.sum(-1) <= 0.8 + 1e-9).all())


@pytest.mark.parametrize("device", DEVICES)
def test_around_best_dictionary_constraints_and_single_subset_draw(device):
    from types import SimpleNamespace

    from botorch_rs.optim.initializers import sample_points_around_best

    X = torch.tensor([[0.1] * 24, [0.8] * 24], dtype=torch.double, device=device)
    bounds = torch.stack((X.new_zeros(24), X.new_ones(24)))
    # The unconstrained objective favors the second point; its auxiliary output is infeasible.
    acquisition = SimpleNamespace(
        _tensor_backend="torch",
        X_baseline=X,
        model=SimpleNamespace(
            posterior=lambda X: SimpleNamespace(mean=X.new_tensor([[0.1, 0.1], [0.8, 0.9]]))
        ),
        objective=lambda Y: Y[..., 0],
        maximize=True,
        constraints={1: (None, 0.2)},
    )
    with no_host():
        sample = sample_points_around_best(
            acquisition, 1, 1e-4, bounds, subset_sigma=1e-4, prob_perturb=0.1, seed=17
        )
    assert sample.shape == (1, 24)
    torch.testing.assert_close(sample, X[0:1], atol=1e-3, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_narrow_polytope_adapts_interior_margin(device):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    index = torch.tensor([0], device=device)
    with no_host():
        points = sample_polytope(
            128,
            1,
            bounds,
            4,
            1,
            13,
            inequality_constraints=[
                (index, bounds.new_tensor([1.0]), 0.5000001),
                (index, bounds.new_tensor([-1.0]), -0.5000009),
            ],
        )
    assert bool((points >= 0.5000001 - 1e-10).all() & (points <= 0.5000009 + 1e-10).all())
    assert points.std() > 1e-7
