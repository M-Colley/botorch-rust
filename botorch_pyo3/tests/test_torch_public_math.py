"""Public utility Tensor paths preserve device and complete derivative graphs."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _arrays, _ops  # noqa: E402
from botorch_rs.utils import safe_math as sm  # noqa: E402
from botorch_rs.utils.probability import utils as probability  # noqa: E402

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
UNARY = [
    "exp",
    "log",
    "log1mexp",
    "log1pexp",
    "logexpit",
    "cauchy",
    "log_softplus",
    "fatplus",
    "log_fatplus",
    "fatmoid",
    "log_fatmoid",
    "sigmoid",
    "ndtr",
    "phi",
    "log_phi",
    "log_ndtr",
    "log_erfc",
    "log_erfcx",
    "standard_normal_log_hazard",
]
PROBABILITY = set(UNARY[12:])


def _unary(name, x, module):
    argument = x.exp() if name == "log" else -x.exp() if name == "log1mexp" else x
    return getattr(module, name)(argument)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", UNARY)
def test_public_unary_values_and_first_second_gradients_match_reference(device, dtype, name):
    reference = pytest.importorskip(
        "botorch.utils.probability.utils" if name in PROBABILITY else "botorch.utils.safe_math"
    )
    actual = probability if name in PROBABILITY else sm
    x = torch.tensor([-2.1, -0.3, 0.2, 1.4], dtype=dtype, device=device, requires_grad=True)
    weights = torch.tensor([0.3, -0.7, 1.2, 0.4], dtype=dtype, device=device)
    value, expected = _unary(name, x, actual), _unary(name, x, reference)
    gradient = torch.autograd.grad((value * weights).sum(), x, create_graph=True)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), x, create_graph=True)[0]
    direction = weights.square()
    second = torch.autograd.grad((gradient * direction).sum(), x)[0]
    expected_second = torch.autograd.grad((expected_gradient * direction).sum(), x)[0]
    tolerance = 3e-5 if dtype == torch.float32 else 2e-11
    for a, b in ((value, expected), (gradient, expected_gradient), (second, expected_second)):
        assert a.device == x.device and a.dtype == x.dtype
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name", ["log1mexp", "log_softplus", "fatplus", "fatmoid", "log_ndtr", "log_erfcx"]
)
def test_public_unary_gradcheck_and_gradgradcheck(device, name):
    module = probability if name in PROBABILITY else sm
    x = torch.tensor([-1.7, 0.3, 1.2], dtype=torch.double, device=device, requires_grad=True)
    function = lambda value: _unary(name, value, module)  # noqa: E731
    assert torch.autograd.gradcheck(function, (x,))
    assert torch.autograd.gradgradcheck(function, (x,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["smooth_amax", "smooth_amin", "fatmax", "fatmin"])
@pytest.mark.parametrize("dim,keepdim", [(-1, False), ((0, 2), True)])
def test_temperature_reductions_keep_input_temperature_and_hessian_graphs(
    device, name, dim, keepdim
):
    reference = pytest.importorskip("botorch.utils.safe_math")
    x = torch.tensor(
        [[[-0.9, 0.4], [0.2, 1.4]], [[0.7, -0.2], [-0.4, 0.8]]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    tau = torch.tensor(0.7, dtype=torch.double, device=device, requires_grad=True)
    f = lambda a, t: getattr(sm, name)(a, dim=dim, keepdim=keepdim, tau=t)  # noqa: E731
    expected = getattr(reference, name)(x, dim=dim, keepdim=keepdim, tau=tau)
    torch.testing.assert_close(f(x, tau), expected, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(f(x, tau).sum(), (x, tau), create_graph=True)
    reference_gradients = torch.autograd.grad(expected.sum(), (x, tau), create_graph=True)
    for a, b in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    assert torch.autograd.gradcheck(f, (x, tau))
    assert torch.autograd.gradgradcheck(f, (x, tau))


@pytest.mark.parametrize("device", DEVICES)
def test_log_softplus_has_real_temperature_mixed_derivatives(device):
    x = torch.tensor([-1.8, 0.2, 2.1], dtype=torch.double, device=device, requires_grad=True)
    tau = torch.tensor(0.6, dtype=torch.double, device=device, requires_grad=True)
    f = lambda a, t: sm.log_softplus(a, tau=t)  # noqa: E731
    # Unlike F.softplus(beta=1/tau), this equivalent expression retains beta's
    # dependence on tau. It is an independent oracle for the complete function.
    expected = (tau * torch.nn.functional.softplus(x / tau)).log()
    a = torch.autograd.grad(f(x, tau).sum(), (x, tau), create_graph=True)
    b = torch.autograd.grad(expected.sum(), (x, tau), create_graph=True)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(f, (x, tau))
    assert torch.autograd.gradgradcheck(f, (x, tau))


@pytest.mark.parametrize("device", DEVICES)
def test_log_erfcx_large_tail_has_correct_first_and_second_derivatives(device):
    x = torch.tensor(1e50, dtype=torch.double, device=device, requires_grad=True)
    value = probability.log_erfcx(x)
    first = torch.autograd.grad(value, x, create_graph=True)[0]
    second = torch.autograd.grad(first, x)[0]
    # Corrections to these limits are O(x^-3) and O(x^-4), far below a double ULP.
    torch.testing.assert_close(first, -x.detach().reciprocal(), rtol=1e-14, atol=0)
    torch.testing.assert_close(second, x.detach().reciprocal().square(), rtol=1e-14, atol=0)
    torch.testing.assert_close(value, -x.detach().log() - 0.5 * torch.log(x.new_tensor(torch.pi)))


@pytest.mark.parametrize("device", DEVICES)
def test_interval_and_constraint_bound_gradients_stay_on_device(device):
    reference = pytest.importorskip("botorch.utils.probability.utils")
    means = torch.tensor(
        [[0.2, -0.4, 0.1], [-0.2, 0.3, 0.5]], device=device, dtype=torch.double, requires_grad=True
    )
    sigmas = torch.tensor([[0.7, 0.8, 0.9]], device=device, dtype=torch.double, requires_grad=True)
    lower = torch.tensor([-0.3], device=device, dtype=torch.double, requires_grad=True)
    upper = torch.tensor([0.9], device=device, dtype=torch.double, requires_grad=True)
    both = torch.tensor([[-0.8, 1.1]], device=device, dtype=torch.double, requires_grad=True)
    indices = [torch.tensor([i], device=device) for i in range(3)]
    f = lambda m, s, lo, hi, interval: probability.compute_log_prob_feas_from_bounds(  # noqa: E731
        *indices, lo, hi, interval, m, s
    )
    expected = reference.compute_log_prob_feas_from_bounds(
        *indices, lower, upper, both, means, sigmas
    )
    torch.testing.assert_close(f(means, sigmas, lower, upper, both), expected)
    assert torch.autograd.gradcheck(f, (means, sigmas, lower, upper, both))
    assert torch.autograd.gradgradcheck(f, (means, sigmas, lower, upper, both))


@pytest.mark.parametrize("device", DEVICES)
def test_public_math_never_flattens_or_transfers_numeric_tensors(device):
    x = torch.tensor([[-1.2, 0.3, 1.1]], device=device, dtype=torch.double, requires_grad=True)
    tau = torch.tensor(0.6, device=device, dtype=torch.double, requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor utility attempted a host/flat-buffer conversion")

    with ExitStack() as stack:
        for module in (_arrays, _ops, sm, probability):
            stack.enter_context(patch.object(module, "to_flat", forbidden))
        for name in ("cpu", "numpy", "tolist"):
            stack.enter_context(patch.object(torch.Tensor, name, forbidden))
        value = sum(
            _unary(name, x, probability if name in PROBABILITY else sm).sum() for name in UNARY
        )
        for name in ("smooth_amax", "smooth_amin", "fatmax", "fatmin"):
            value = value + getattr(sm, name)(x, dim=-1, tau=tau).sum()
        value = value + probability.log_prob_normal_in(x, x + 0.8).sum()
        first = torch.autograd.grad(value, (x, tau), create_graph=True)
        second = torch.autograd.grad(sum(g.sum() for g in first), (x, tau))
    assert all(g.device == x.device and bool(g.isfinite().all()) for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_infinite_reductions_and_unbounded_intervals_keep_finite_limit_gradients(device):
    x = torch.tensor(
        [[1.0, -torch.inf, 2.0]], device=device, dtype=torch.double, requires_grad=True
    )
    tau = torch.tensor(0.7, device=device, dtype=torch.double, requires_grad=True)
    value = sm.fatmax(x, dim=-1, tau=tau)
    gradients = torch.autograd.grad(value.sum(), (x, tau), create_graph=True)
    assert all(bool(g.isfinite().all()) for g in gradients)
    assert gradients[0][0, 1] == 0
    a = torch.tensor(
        [-torch.inf, -0.3, -torch.inf], device=device, dtype=torch.double, requires_grad=True
    )
    b = torch.tensor(
        [0.4, torch.inf, torch.inf], device=device, dtype=torch.double, requires_grad=True
    )
    gradients = torch.autograd.grad(probability.log_prob_normal_in(a, b).sum(), (a, b))
    assert all(bool(g.isfinite().all()) for g in gradients)
    assert gradients[0][0] == gradients[0][2] == gradients[1][1] == gradients[1][2] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "logplusexp"])
def test_binary_scalar_tensor_promotion_and_mixed_derivatives(device, name):
    reference = pytest.importorskip("botorch.utils.safe_math")
    x = torch.tensor([[0.3, 0.8]], device=device, dtype=torch.float32, requires_grad=True)
    scalar = torch.tensor(0.7, device=device, dtype=torch.double, requires_grad=True)
    actual = getattr(sm, name)(x, scalar)
    expected = getattr(reference, name)(x, scalar)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.square().sum(), (x, scalar), create_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (x, scalar), create_graph=True)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right)
    aa = torch.autograd.grad(sum(g.sum() for g in a), (x, scalar))
    bb = torch.autograd.grad(sum(g.sum() for g in b), (x, scalar))
    for left, right in zip(aa, bb, strict=True):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["log_softplus", "fatplus", "fatmoid", "fatmax", "smooth_amax"])
def test_temperature_scalar_tensor_dtype_and_broadcasting_match_reference(device, name):
    reference = pytest.importorskip("botorch.utils.safe_math")
    x = torch.tensor([[0.3, 0.8], [-0.4, 1.2]], device=device, dtype=torch.float32)
    tau = torch.tensor(0.7, device=device, dtype=torch.double, requires_grad=True)
    kwargs = {"dim": -1} if name in {"fatmax", "smooth_amax"} else {}
    actual = getattr(sm, name)(x, tau=tau, **kwargs)
    # Reference log_softplus casts tau to x.dtype and then passes it as the
    # Number-only softplus beta: a tracked cast is rejected by Torch 2.13.
    # Compare its value with an untracked temperature; the independent graph
    # oracle and gradgradcheck above cover the full temperature derivative.
    expected = getattr(reference, name)(x, tau=tau.detach(), **kwargs)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.grad(actual.sum(), tau)[0].isfinite()


@pytest.mark.parametrize("device", DEVICES)
def test_percentile_ties_nans_and_broadcasting_stay_on_device(device):
    data = torch.tensor([[1.0, 2.0, 2.0, 4.0], [1.0, torch.nan, 3.0, 4.0]], device=device)
    score = torch.tensor([[2.0]], device=device)
    actual = probability.percentile_of_score(data, score)
    expected = torch.tensor([62.5, torch.nan], device=device)
    torch.testing.assert_close(actual, expected, equal_nan=True)
    assert actual.device == data.device
