"""External exact and custom MLL closures preserve the public data protocol."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskGP as ReferenceGP
from botorch.optim.closures.model_closures import get_loss_closure as reference_closure

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim.closures import get_loss_closure, get_loss_closure_with_grads
from botorch_rs.optim.fit import fit_gpytorch_mll_torch
from botorch_rs.optim.utils import get_parameters

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_external_exact_batches_cycle_with_reference_values_and_gradients(device, fixed):
    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    y = (5 * x).sin()
    noise = torch.full_like(y, 0.03) if fixed else None
    own_model = SingleTaskGP(x, y, train_Yvar=noise, backend="torch", outcome_transform=None)
    ref_model = ReferenceGP(x, y, train_Yvar=noise, outcome_transform=None)
    own_mll = ExactMarginalLogLikelihood(own_model.likelihood, own_model).train()
    ref_mll = gpytorch.mlls.ExactMarginalLogLikelihood(ref_model.likelihood, ref_model).train()
    targets = [y.squeeze(-1).clone().requires_grad_(), (y.squeeze(-1) + 0.1).requires_grad_()]
    batches = [(x, target) for target in targets]
    own = get_loss_closure(own_mll, batches)
    reference = reference_closure(ref_mll, batches)
    for index in (0, 1, 0):
        actual, expected = own(), reference()
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
        own_leaves = (
            targets[index],
            own_model.mean_module.raw_constant,
            own_model.covar_module.raw_lengthscale,
        )
        ref_leaves = (
            targets[index],
            ref_model.mean_module.raw_constant,
            ref_model.covar_module.raw_lengthscale,
        )
        if not fixed:
            own_leaves += (own_model.likelihood.raw_noise,)
            ref_leaves += (ref_model.likelihood.raw_noise,)
        g = torch.autograd.grad(actual.sum(), own_leaves, create_graph=True)
        rg = torch.autograd.grad(expected.sum(), ref_leaves, create_graph=True)
        for a, b in zip(g, rg, strict=True):
            torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-7)
        directions = [torch.full_like(leaf, 0.3) for leaf in own_leaves]
        second = torch.autograd.grad(
            sum((a * d).sum() for a, d in zip(g, directions, strict=True)), own_leaves
        )
        epsilon = 2e-5
        # The reference RBF fast path caches a detached lengthscale derivative.
        # Its first derivatives are correct; central differences of those
        # derivatives independently check the full loss Hessian here.
        originals = [leaf.detach().clone() for leaf in ref_leaves]
        displaced = []
        try:
            for sign in (-1, 1):
                with torch.no_grad():
                    for leaf, base, direction in zip(
                        ref_leaves, originals, directions, strict=True
                    ):
                        leaf.copy_(base + sign * epsilon * direction)
                loss = -ref_mll(ref_model(x), targets[index])
                displaced.append(torch.autograd.grad(loss.sum(), ref_leaves))
        finally:
            with torch.no_grad():
                for leaf, base in zip(ref_leaves, originals, strict=True):
                    leaf.copy_(base)
        for a, minus, plus in zip(second, *displaced, strict=True):
            torch.testing.assert_close(a, (plus - minus) / (2 * epsilon), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_external_exact_gradient_closure_runs_a_real_optimizer(device):
    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    y = (5 * x).sin()
    model = SingleTaskGP(x, y, backend="torch", outcome_transform=None)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    parameters = get_parameters(mll, requires_grad=True)
    batches = [(x, y.squeeze(-1))]
    closure = get_loss_closure_with_grads(mll, parameters, batches)
    before = float(closure()[0].detach())
    result = fit_gpytorch_mll_torch(mll, closure=closure, step_limit=3, stopping_criterion=None)
    assert result.fval < before
    assert all(value.grad is not None for value in parameters.values())


@pytest.mark.parametrize("device", DEVICES)
def test_external_custom_model_inputs_targets_and_keyword_arguments(device):
    weight = torch.tensor(0.7, device=device, dtype=torch.double, requires_grad=True)
    x = weight.new_tensor([0.1, 0.3])

    class Model:
        train_inputs = (x, x)

        def __call__(self, a, b):
            return weight * (a + b)

    class MLL:
        model = Model()

        def __call__(self, output, target, offset, *, scale=1):
            return -scale * (output - target - offset).square().sum()

    batches = [(x, x + 0.1, x.square(), x / 2)]
    own = get_loss_closure(MLL(), batches)
    reference = reference_closure(MLL(), batches)
    torch.testing.assert_close(own(scale=2), reference(scale=2))
    gradient = torch.autograd.grad(own(scale=2), weight, create_graph=True)[0]
    assert torch.autograd.grad(gradient, weight)[0] > 0
    with pytest.raises(TypeError, match="batch of tensors"):
        get_loss_closure(MLL(), [object()])()
    with pytest.raises(ValueError, match="at least one target"):
        get_loss_closure(MLL(), [(x, x)])()
    with pytest.raises(ValueError, match="at least one minibatch"):
        get_loss_closure(MLL(), [])()
