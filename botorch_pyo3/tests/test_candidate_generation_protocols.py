"""Numerical-gradient and independent parallel candidate-generation protocols."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from botorch_rs.generation.gen import gen_candidates_scipy, gen_candidates_torch

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


class Quadratic:
    def __init__(self, tensor_backend=True, detach=False):
        self._tensor_backend = "torch" if tensor_backend else None
        self.detach = detach
        self.batch_sizes = []

    def __call__(self, X):
        self.batch_sizes.append(X.shape[0])
        values = -(X - 0.73).square().sum((-2, -1))
        return values.detach() if self.detach else values


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["L-BFGS-B", "SLSQP", "Powell", "Nelder-Mead"])
def test_numerical_gradients_support_nondifferentiable_tensor_acquisitions(device, method):
    starts = torch.tensor([[[0.1]], [[0.9]]], device=device, dtype=torch.double)
    candidates, values = gen_candidates_scipy(
        starts,
        Quadratic(detach=True),
        lower_bounds=0,
        upper_bounds=1,
        options={
            "method": method,
            "with_grad": False,
            "maxiter": 600,
            **({"ftol": 1e-12} if method != "Nelder-Mead" else {}),
        },
    )
    torch.testing.assert_close(candidates, torch.full_like(candidates, 0.73), atol=2e-4, rtol=0)
    torch.testing.assert_close(values, torch.zeros_like(values), atol=1e-7, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed_shape", [(3,), (3, 2)])
def test_parallel_native_states_match_official_batched_driver(device, fixed_shape):
    pytest.importorskip("botorch")
    from botorch.generation.gen import gen_candidates_scipy as reference

    starts = torch.linspace(0.05, 0.95, 12, device=device, dtype=torch.double).reshape(3, 2, 2)
    fixed = torch.linspace(
        0.2, 0.6, torch.Size(fixed_shape).numel(), device=device, dtype=torch.double
    ).reshape(fixed_shape)
    own_acquisition = Quadratic()
    kwargs = {
        "lower_bounds": starts.new_zeros(2),
        "upper_bounds": starts.new_ones(2),
        "fixed_features": {0: fixed},
        "use_parallel_mode": True,
    }
    own = gen_candidates_scipy(starts, own_acquisition, **kwargs)
    expected = reference(starts, Quadratic(), **kwargs)
    torch.testing.assert_close(own[0], expected[0], atol=3e-9, rtol=3e-9)
    torch.testing.assert_close(own[1], expected[1], atol=3e-10, rtol=3e-10)
    assert max(own_acquisition.batch_sizes) == 3
    assert own[0].device == starts.device


@pytest.mark.parametrize("device", DEVICES)
def test_scipy_intermediate_result_callback_protocol(device):
    starts = torch.tensor([[[0.1]], [[0.9]]], device=device, dtype=torch.double)
    seen = []

    def callback(*, intermediate_result):
        seen.append(intermediate_result)

    candidates, _ = gen_candidates_scipy(
        starts,
        Quadratic(),
        lower_bounds=0,
        upper_bounds=1,
        options={"callback": callback},
        use_parallel_mode=False,
    )
    assert seen and all(hasattr(item, "x") and hasattr(item, "fun") for item in seen)
    torch.testing.assert_close(candidates, torch.full_like(candidates, 0.73))


@pytest.mark.parametrize("device", DEVICES)
def test_parallel_multifeature_bounds_follow_q_major_variable_order(device):
    starts = torch.full((3, 2, 3), 0.4, dtype=torch.double, device=device)
    lower, upper = starts.new_tensor([0, 0.6, -2]), starts.new_tensor([0.5, 0.8, 0.9])
    candidates, _ = gen_candidates_scipy(
        starts, Quadratic(), lower_bounds=lower, upper_bounds=upper, use_parallel_mode=True
    )
    expected = starts.new_tensor([0.5, 0.73, 0.73]).expand_as(starts)
    torch.testing.assert_close(candidates, expected)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_native_generation_supports_shaped_fixed_features_and_preserves_array_kind(kind):
    from helpers import as_kind, check_kind

    starts = as_kind([[[0.1, 0.2]], [[0.9, 0.8]]], kind)
    fixed = as_kind([0.3, 0.6], kind)
    candidates, _ = gen_candidates_scipy(
        starts,
        Quadratic(tensor_backend=False),
        lower_bounds=0,
        upper_bounds=1,
        fixed_features={0: fixed},
        use_parallel_mode=True,
    )
    check_kind(candidates, kind)
    expected = torch.tensor([[[0.3, 0.73]], [[0.6, 0.73]]], dtype=torch.double)
    torch.testing.assert_close(torch.as_tensor(candidates, dtype=torch.double), expected)
    candidates, _ = gen_candidates_torch(
        starts,
        Quadratic(tensor_backend=False),
        lower_bounds=0,
        upper_bounds=1,
        fixed_features={0: fixed},
        options={"stopping_criterion_options": {"maxiter": 120, "n_window": 200}},
    )
    check_kind(candidates, kind)
    torch.testing.assert_close(
        torch.as_tensor(candidates, dtype=torch.double), expected, atol=2e-3, rtol=0
    )


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_native_initializer_calls_custom_raw_generator(kind):
    from botorch_rs.optim.initializers import gen_batch_initial_conditions
    from helpers import as_kind, check_kind

    bounds = as_kind([[0.0], [1.0]], kind)
    calls = []

    def generator(n, q, seed):
        calls.append((n, q, seed))
        return as_kind([[[i / (n - 1)]] for i in range(n)], kind)

    candidates = gen_batch_initial_conditions(
        Quadratic(tensor_backend=False),
        bounds,
        q=1,
        num_restarts=2,
        raw_samples=10,
        generator=generator,
        options={"seed": 17, "topn": True},
    )
    assert calls == [(10, 1, 17)]
    check_kind(candidates, kind)
    actual = torch.as_tensor(candidates, dtype=torch.double).flatten().sort().values
    torch.testing.assert_close(actual, torch.tensor([2 / 3, 7 / 9], dtype=torch.double))


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("method", ["L-BFGS-B", "Powell", "trust-constr"])
def test_high_level_native_optimization_uses_general_method_and_numerical_gradients(kind, method):
    from botorch_rs.optim import optimize_acqf
    from helpers import as_kind, check_kind

    bounds = as_kind([[0.0], [1.0]], kind)
    candidates, values = optimize_acqf(
        Quadratic(tensor_backend=False, detach=True),
        bounds,
        q=1,
        num_restarts=2,
        batch_initial_conditions=as_kind([[[0.1]], [[0.9]]], kind),
        return_best_only=False,
        options={"method": method, "with_grad": False, "maxiter": 200},
    )
    check_kind(candidates, kind)
    expected = torch.full((2, 1, 1), 0.73, dtype=torch.double)
    torch.testing.assert_close(
        torch.as_tensor(candidates, dtype=torch.double), expected, atol=2e-4, rtol=0
    )
    torch.testing.assert_close(
        torch.as_tensor(values, dtype=torch.double),
        torch.zeros(2, dtype=torch.double),
        atol=1e-7,
        rtol=0,
    )
