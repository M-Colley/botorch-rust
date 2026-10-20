"""Explicit query gradients follow the same single native call as autograd."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.acquisition import LogExpectedImprovement  # noqa: E402
from botorch_rs.acquisition.logei import (  # noqa: E402
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO  # noqa: E402
from botorch_rs.acquisition.objective import GenericMCObjective  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.sampling.normal import SobolQMCNormalSampler  # noqa: E402


class CountedNative:
    def __init__(self, native):
        self.native = native
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def forward(self, *args):
        self.calls.append("forward")
        return self.native.forward(*args)

    def forward_with_grad(self, *args):
        self.calls.append("query")
        return self.native.forward_with_grad(*args)

    def forward_with_adjoint_tape(self, *args):
        self.calls.append("tape")
        return self.native.forward_with_adjoint_tape(*args)


def make_acquisition(name, cache_root=False, objective=None):
    train = torch.tensor([[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]], dtype=torch.double)
    targets = (train[:, :1] * 3).sin()
    if name == "parego":
        targets = torch.cat([targets, train[:, 1:].cos()], dim=-1)
    model = SingleTaskGP(train, targets)
    if name == "analytic":
        return LogExpectedImprovement(model, best_f=0.4)
    kwargs = dict(sampler=SobolQMCNormalSampler((8,), seed=19), objective=objective)
    if name == "mc":
        return qLogExpectedImprovement(model, best_f=0.4, **kwargs)
    kwargs.update(X_baseline=train[:3], prune_baseline=False, cache_root=cache_root)
    if name == "parego":
        kwargs["scalarization_weights"] = torch.tensor([0.3, 0.7], dtype=torch.double)
        return qLogNParEGO(model, **kwargs)
    return qLogNoisyExpectedImprovement(model, **kwargs)


@pytest.mark.parametrize(
    ("name", "cache_root"),
    [
        ("analytic", False),
        ("mc", False),
        ("nei", False),
        ("nei", True),
        ("parego", False),
        ("parego", True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_value_and_grad_matches_active_parameter_graph_in_one_call(
    name, cache_root, dtype, batched
):
    acquisition = make_acquisition(name, cache_root)
    native = CountedNative(acquisition._rs)
    acquisition._rs = native
    q = 1 if name == "analytic" else 2
    query = torch.linspace(0.2, 0.8, (6 if batched else 1) * q * 2, dtype=dtype)
    query = query.reshape((2, 3, q, 2) if batched else (q, 2)).requires_grad_()
    actual = acquisition(query)
    (derivative,) = torch.autograd.grad(actual.sum(), query)
    values, gradient = acquisition.value_and_grad(query.detach())
    assert native.calls == ["tape", "tape"]
    assert torch.equal(values, actual.detach())
    assert torch.equal(gradient, derivative)
    assert gradient.shape == query.shape and gradient.dtype == query.dtype
    assert not values.requires_grad and not gradient.requires_grad


@pytest.mark.parametrize("mode", ["frozen", "no_grad", "list"])
def test_value_and_grad_preserves_query_only_dispatch(mode):
    acquisition = make_acquisition("analytic")
    if mode == "frozen":
        for parameter in acquisition.model.parameters():
            parameter.requires_grad_(False)
    native = CountedNative(acquisition._rs)
    acquisition._rs = native
    query = torch.tensor([[[0.2, 0.6]]], dtype=torch.double, requires_grad=True)
    if mode == "list":
        query = query.detach().tolist()
    with torch.no_grad() if mode == "no_grad" else nullcontext():
        acquisition(query)
        acquisition.value_and_grad(query)
    assert native.calls == (["query", "query"] if mode == "frozen" else ["forward", "query"])


@pytest.mark.parametrize(("name", "cache_root"), [("mc", False), ("nei", False), ("nei", True)])
def test_explicit_gradient_does_not_repeat_objective_evaluation(name, cache_root):
    calls = []

    def objective(samples, X=None):
        calls.append(tuple(samples.shape))
        return samples[..., 0] + 0.2 * X[..., 0]

    acquisition = make_acquisition(name, cache_root, GenericMCObjective(objective))
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double, requires_grad=True)
    calls.clear()
    actual = acquisition(query)
    (derivative,) = torch.autograd.grad(actual.sum(), query)
    forward_calls = list(calls)
    assert forward_calls
    calls.clear()
    values, gradient = acquisition.value_and_grad(query.detach())
    assert calls == forward_calls
    assert torch.equal(values, actual.detach())
    assert torch.equal(gradient, derivative)
