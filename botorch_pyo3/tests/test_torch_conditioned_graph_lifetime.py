"""Cached conditioning graphs obey the source's normal autograd lifetime."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_heterogeneous import FEATURES, training

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.heterogeneous_mtgp import HeterogeneousMTGP
from botorch_rs.models.transforms.input import Normalize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_source_cached_conditioning_graph_requires_retention_for_repeated_backward(
    device, retained, custom
):
    source = pytest.importorskip("botorch.models.heterogeneous_mtgp")
    source_input = pytest.importorskip("botorch.models.transforms.input")
    Xs, Ys = training(device)
    if custom:

        class Gain(source_input.InputTransform):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(Xs[0].new_tensor(0.8))
                self.transform_on_train = self.transform_on_eval = self.transform_on_fantasize = (
                    True
                )

            def transform(self, X):
                return torch.cat((X[..., :-1] * self.gain, X[..., -1:]), -1)

        transform = Gain()
    else:
        transform = source_input.Normalize(4, indices=[0, 1, 2])
    model = source.HeterogeneousMTGP(
        Xs, Ys, None, FEATURES, 3, input_transform=transform, outcome_transform=None
    ).to(Xs[0])
    query = Xs[0].new_tensor([[0.65, 0.6, 0.0]], requires_grad=True)
    model.posterior(query)
    observed = Xs[0].new_tensor([[0.35, 0.25]], requires_grad=True)
    target = Ys[0].new_tensor([[0.25]], requires_grad=True)
    child = model.condition_on_observations(model.map_to_full_tensor(observed, 0), target)

    def backward(keep):
        # Clear GP prediction factors so this isolates the preprocessing graph.
        child.prediction_strategy = None
        posterior = child.posterior(query)
        return torch.autograd.grad(
            posterior.mean.sum() + posterior.variance.sum(),
            (observed, query),
            retain_graph=keep,
        )

    first = backward(retained)
    assert all(torch.isfinite(g).all() for g in first)
    if retained:
        second = backward(False)
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    else:
        with pytest.raises(RuntimeError, match="backward through the graph a second time"):
            backward(False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_fresh_conditioning_repeats_complete_data_and_parameter_hessians(device, fixed):
    Xs, Ys = training(device, True)
    variances = [torch.full_like(Y, 0.03, requires_grad=True) for Y in Ys] if fixed else None
    model = HeterogeneousMTGP(
        Xs,
        Ys,
        variances,
        FEATURES,
        3,
        input_transform=Normalize(4, indices=[0, 1, 2]),
        backend="torch",
    )
    observed = Xs[0].new_tensor([[0.35, 0.25, 0.0]], requires_grad=True)
    target = Ys[0].new_tensor([[0.25]], requires_grad=True)
    query = Xs[0].new_tensor([[0.65, 0.6, 0.0]], requires_grad=True)
    noise = torch.full_like(target, 0.04, requires_grad=True) if fixed else None
    for _ in range(2):
        child = model.condition_on_observations(observed, target, noise=noise)
        leaves = (
            *Xs,
            *Ys,
            observed,
            target,
            query,
            *(variances or []),
            *((noise,) if fixed else ()),
            *child.parameters(),
        )
        posterior = child.posterior(query)
        loss = posterior.mean.sum() + posterior.variance.sum()
        loss = loss + ExactMarginalLogLikelihood(child.likelihood, child)()
        first = torch.autograd.grad(loss, leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
        assert all(torch.isfinite(g).all() for g in (*first, *second))
