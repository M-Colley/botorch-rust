"""Forward/backward closures must leave gradients usable by Torch optimizers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.optim.closures import (  # noqa: E402
    ForwardBackwardClosure,
    get_loss_closure_with_grads,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_forward_backward_closure_drives_sgd_and_retains_unused_gradient(device):
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.7], dtype=torch.double, device=device))
    unused = torch.nn.Parameter(torch.ones(1, dtype=torch.double, device=device))
    unused.grad = torch.full_like(unused, 17)
    initial = parameter.detach().clone()
    closure = ForwardBackwardClosure(
        lambda: parameter.square() / 2, {"p": parameter, "unused": unused}
    )
    optimizer = torch.optim.SGD([parameter, unused], lr=0.1)
    for _ in range(3):
        before = parameter.detach().clone()
        value, gradients = closure()
        torch.testing.assert_close(value, before.square().sum() / 2)
        torch.testing.assert_close(gradients[0], before)
        assert gradients[0] is parameter.grad
        assert gradients[1] is unused.grad
        torch.testing.assert_close(unused.grad, torch.zeros_like(unused))
        optimizer.step()
    torch.testing.assert_close(parameter, initial * 0.9**3)
    torch.testing.assert_close(unused, torch.ones_like(unused))


def test_native_exact_gp_gradient_closure_populates_persistent_parameter_grads():
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.models import SingleTaskGP
    from botorch_rs.optim.utils import get_parameters

    x = torch.linspace(0, 1, 7, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(x, (5 * x).sin(), backend="native")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    parameters = get_parameters(mll, requires_grad=True)
    closure = get_loss_closure_with_grads(mll, parameters)
    optimizer = torch.optim.SGD(list(parameters.values()), lr=1e-4)
    initial = tuple(model._rs.raw_parameters())
    before, gradients = closure()
    for parameter, gradient in zip(parameters.values(), gradients, strict=True):
        assert parameter.grad is gradient
    optimizer.step()
    after, _ = closure()
    assert tuple(model._rs.raw_parameters()) != initial
    assert float(after) < float(before)
