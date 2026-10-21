"""Independent reference checks for compositional device acquisition graphs."""

# ruff: noqa: E402
from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.analytic import PosteriorMean as ReferenceMean
from botorch.acquisition.logei import qLogExpectedImprovement as ReferenceLogEI
from botorch.acquisition.monte_carlo import qExpectedImprovement as ReferenceEI
from botorch.acquisition.penalized import PenalizedAcquisitionFunction as ReferencePenalized
from botorch.acquisition.prior_guided import PriorGuidedAcquisitionFunction as ReferencePrior
from botorch.acquisition.proximal import ProximalAcquisitionFunction as ReferenceProximal
from test_torch_backend_acquisition import FixedSampler, TensorModel

from botorch_rs.acquisition.analytic import PosteriorMean
from botorch_rs.acquisition.logei import qLogExpectedImprovement
from botorch_rs.acquisition.monte_carlo import qExpectedImprovement
from botorch_rs.acquisition.penalized import L2Penalty, PenalizedAcquisitionFunction
from botorch_rs.acquisition.prior_guided import PriorGuidedAcquisitionFunction
from botorch_rs.acquisition.proximal import ProximalAcquisitionFunction

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _compare_graphs(actual, expected, leaves):
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-11)
    gradients = [
        torch.autograd.grad(
            value.square().sum(),
            leaves,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        for value in (actual, expected)
    ]
    for left, right in zip(*gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=3e-8, atol=3e-9)
    seconds = [
        torch.autograd.grad(
            sum(g.square().sum() for g in gradient),
            leaves,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        for gradient in gradients
    ]
    for left, right in zip(*seconds, strict=True):
        torch.testing.assert_close(left, right, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_penalized_keyword_dispatch_and_complete_graph(device):
    model = TensorModel(device)
    query = model.train_X.new_tensor([[[0.23]], [[0.61]]], requires_grad=True)
    origin = model.train_X.new_tensor([0.4], requires_grad=True)
    strength = model.train_X.new_tensor(0.13, requires_grad=True)
    penalty = L2Penalty(origin)
    own = PenalizedAcquisitionFunction(
        raw_acqf=PosteriorMean(model), penalty_func=penalty, regularization_parameter=strength
    )
    reference = ReferencePenalized(ReferenceMean(model), penalty, strength)
    leaves = (query, origin, strength, *model.parameters())
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        _compare_graphs(own(query), reference(query), leaves)
    assert isinstance(own, PenalizedAcquisitionFunction)
    assert own._tensor_backend == "torch"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_prior_weighting_precedes_mc_reductions_and_retains_graph(device, log, pending):
    model = TensorModel(device)
    query = model.train_X.new_tensor([[[0.21], [0.48]], [[0.59], [0.81]]], requires_grad=True)
    location = model.train_X.new_tensor(0.3, requires_grad=True)
    exponent = model.train_X.new_tensor(0.7, requires_grad=True)

    def prior(X):
        value = -((X[..., 0] - location) ** 2)
        return value if log else value.exp()

    cls, ref_cls = (
        (qLogExpectedImprovement, ReferenceLogEI) if log else (qExpectedImprovement, ReferenceEI)
    )
    count = 3 if pending else 2
    own_base = cls(model, best_f=0.2, sampler=FixedSampler(count, device))
    ref_base = ref_cls(model, best_f=0.2, sampler=FixedSampler(count, device))
    extra = model.train_X.new_tensor([[0.9]]) if pending else None
    own = PriorGuidedAcquisitionFunction(own_base, prior, log, exponent, extra)
    reference = ReferencePrior(ref_base, prior, log, exponent, extra)
    leaves = (query, location, exponent, *model.parameters())
    _compare_graphs(own(query), reference(query), leaves)
    assert own_base.X_pending is None
    own.set_X_pending(None)
    assert own.X_pending is None


class _AffineTransform(torch.nn.Module):
    def __init__(self, like):
        super().__init__()
        self.scale = torch.nn.Parameter(like.new_tensor(0.7))

    def transform(self, X):
        return X / self.scale

    def untransform(self, X):
        return X * self.scale


class _ProximityModel(TensorModel):
    @property
    def _train_X(self):
        return self.train_X

    @property
    def train_inputs(self):
        return (self.input_transform.transform(self.train_X),)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transformed", [False, True])
def test_proximal_raw_and_transformed_spaces_match_reference(device, transformed):
    model = _ProximityModel(device)
    model.input_transform = _AffineTransform(model.train_X)
    query = model.train_X.new_tensor([[[0.23]], [[0.61]]], requires_grad=True)
    weights = model.train_X.new_tensor([0.6], requires_grad=True)
    own = ProximalAcquisitionFunction(PosteriorMean(model), weights, transformed, beta=1.1)
    reference = ReferenceProximal(ReferenceMean(model), weights, transformed, beta=1.1)
    leaves = (query, weights, *model.parameters())
    _compare_graphs(own(query), reference(query), leaves)


@pytest.mark.parametrize("device", DEVICES)
def test_regularized_wrappers_work_with_actual_port_model(device):
    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.transforms.input import Normalize

    x = torch.tensor([[0.1], [0.3], [0.7], [0.9]], device=device, dtype=torch.double)
    model = SingleTaskGP(x, (4 * x).sin(), backend="torch", input_transform=Normalize(d=1))
    query = x.new_tensor([[0.42]], requires_grad=True)
    base = PosteriorMean(model)
    own = ProximalAcquisitionFunction(base, x.new_tensor([0.4]), beta=1.0)
    prior = PriorGuidedAcquisitionFunction(base, lambda X: (-X.square().sum(-1)).exp())
    penalized = PenalizedAcquisitionFunction(base, L2Penalty(x.new_tensor([0.2])), 0.1)
    for acquisition in (own, prior, penalized):
        value, gradient = acquisition.value_and_grad(query)
        assert value.device == query.device
        assert torch.isfinite(gradient).all()
        second = torch.autograd.grad(gradient.sum(), query, retain_graph=True)[0]
        assert torch.isfinite(second).all()
