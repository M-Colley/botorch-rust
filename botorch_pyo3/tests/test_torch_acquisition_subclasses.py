"""Public subclass hooks keep their identities and complete Tensor graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.acquisition.analytic import AnalyticAcquisitionFunction, LogExpectedImprovement
from botorch_rs.acquisition.logei import qLogExpectedImprovement
from botorch_rs.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    SampleReducingMCAcquisitionFunction,
    qExpectedImprovement,
)
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _model(device):
    x = torch.tensor([[0.12], [0.34], [0.65], [0.88]], dtype=torch.double, device=device)
    y = (x * 4).sin().detach().requires_grad_()
    x.requires_grad_()
    model = SingleTaskGP(x, y, outcome_transform=None)
    assert model._tensor_backend == "torch"  # CPU graph / CUDA auto dispatch.
    return model, (x, y)


class _AffineLogHooks:
    def __init__(self, model, best_f):
        super().__init__(model, best_f)
        self.shift = nn.Parameter(model.train_inputs[0].new_tensor(0.12))
        self.register_buffer("offset", self.shift.new_tensor(0.07))
        self.init_calls, self.forward_calls, self.posterior_calls = 1, 0, 0

    def _mean_and_sigma(self, X, compute_sigma=True, min_var=1e-12):
        mean, sigma = super()._mean_and_sigma(X, compute_sigma, min_var)
        return mean + self.shift, sigma

    def _posterior(self, X):
        self.posterior_calls += 1
        return super()._posterior(X)

    def forward(self, X):
        self.forward_calls += 1
        return super().forward(X) + self.offset


class AffineLogEI(_AffineLogHooks, LogExpectedImprovement):
    __init__ = _AffineLogHooks.__init__


class CallerAcquisition(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(model)
        self.scale = nn.Parameter(model.train_inputs[0].new_tensor(0.7))

    def forward(self, X):
        if self.X_pending is not None:
            X = torch.cat((X, self.X_pending.expand(*X.shape[:-2], -1, -1)), -2)
        return (self.model.posterior(X).mean.sin() * self.scale).sum((-1, -2))


class PositiveUtility(SampleReducingMCAcquisitionFunction):
    def _sample_forward(self, obj):
        return obj.square() + 0.2


class CallerAnalytic(AnalyticAcquisitionFunction):
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        return (mean + sigma * 0.3).squeeze(-1)


class CallerMC(MCAcquisitionFunction):
    def forward(self, X):
        _, objective = self._get_samples_and_objectives(X)
        return objective.square().mean(0).sum(-1)


class HookedLogEI(qLogExpectedImprovement):
    def _q_reduction(self, values):
        return super()._q_reduction(values) + 0.1

    def _sample_reduction(self, values):
        return super()._sample_reduction(values) + 0.2


class ScaledSampleEI(qExpectedImprovement):
    def __init__(self, model, best_f, sampler):
        super().__init__(model, best_f, sampler)
        self.scale = nn.Parameter(model.train_inputs[0].new_tensor(0.7))
        self.init_calls = 1

    def _sample_forward(self, obj):
        return super()._sample_forward(obj) * self.scale + 0.4


def _compare_graphs(actual, expected, actual_targets, expected_targets):
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-10)
    actual_first = torch.autograd.grad(actual.sum(), actual_targets, create_graph=True)
    expected_first = torch.autograd.grad(expected.sum(), expected_targets, create_graph=True)
    actual_second = torch.autograd.grad(sum(g.square().sum() for g in actual_first), actual_targets)
    expected_second = torch.autograd.grad(
        sum(g.square().sum() for g in expected_first), expected_targets
    )
    for a, b in zip(
        (*actual_first, *actual_second), (*expected_first, *expected_second), strict=True
    ):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_cooperative_analytic_constructor_posterior_hook_state_and_all_graphs(device):
    model, data = _model(device)
    best = data[0].new_tensor(0.2, requires_grad=True)
    query = data[0].new_tensor([[[0.25]], [[0.61]]], requires_grad=True)
    acq = AffineLogEI(model, best)
    assert isinstance(acq, AffineLogEI) and type(acq).__name__ == "TorchAffineLogEI"
    assert acq.init_calls == 1 and "model" in dict(acq.named_children())
    assert dict(acq.named_parameters())["shift"] is acq.shift
    expected = LogExpectedImprovement(model, best - acq.shift)(query) + acq.offset
    leaves = (*data, best, query, *model.parameters(), acq.shift)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host conversion")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host conversion")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host conversion")),
    ):
        _compare_graphs(acq(query), expected, leaves, leaves)
    assert acq.forward_calls == 1 and acq.posterior_calls == 1
    state = copy.deepcopy(acq.state_dict())
    with torch.no_grad():
        acq.shift.add_(0.13)
    changed = acq(query)
    acq.load_state_dict(state)
    assert not torch.allclose(changed, acq(query))
    for clone in (copy.deepcopy(acq), pickle.loads(pickle.dumps(acq))):
        assert isinstance(clone, AffineLogEI)
        torch.testing.assert_close(clone(query), acq(query), atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_generic_custom_acquisition_module_pending_and_full_graph(device):
    model, data = _model(device)
    query = data[0].new_tensor([[[0.25], [0.61]]], requires_grad=True)
    acq = CallerAcquisition(model)
    pending = query.new_tensor([[0.45]], requires_grad=True)
    with pytest.warns(BotorchWarning, match="Pending points"):
        acq.set_X_pending(pending)
    assert not acq.X_pending.requires_grad and acq.X_pending.data_ptr() != pending.data_ptr()
    full = torch.cat((query, pending.detach().unsqueeze(0)), -2)
    expected = (model.posterior(full).mean.sin() * acq.scale).sum((-1, -2))
    leaves = (*data, query, *model.parameters(), acq.scale)
    _compare_graphs(acq(query), expected, leaves, leaves)
    acq.set_X_pending(None)
    value, gradient = acq.value_and_grad(query)
    torch.testing.assert_close(value, acq(query))
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sample_shape", [(6,), (2, 3)])
def test_custom_sample_utility_and_actual_reduction_callables(device, sample_shape):
    model, data = _model(device)
    query = data[0].new_tensor([[[0.25], [0.61]], [[0.35], [0.71]]], requires_grad=True)
    calls = []

    def sample_fn(values, *, dim):
        calls.append(dim)
        return values.tanh().sum(dim=dim)

    sampler = IIDNormalSampler(sample_shape, seed=29)
    acq = PositiveUtility(
        model, sampler=sampler, sample_reduction=sample_fn, q_reduction=torch.mean
    )
    assert isinstance(acq, PositiveUtility)
    assert acq._sample_reduction_fn is sample_fn and acq._q_reduction_fn is torch.mean
    samples = sampler(model.posterior(query)).squeeze(-1)
    expected = (samples.square() + 0.2).mean(-1).tanh().sum(tuple(range(len(sample_shape))))
    leaves = (*data, query, *model.parameters())
    _compare_graphs(acq(query), expected, leaves, leaves)
    assert calls == [tuple(range(len(sample_shape)))]


@pytest.mark.parametrize("device", DEVICES)
def test_cooperative_mc_subclass_on_fully_bayesian_model_preserves_sample_hook(device):
    _, data = _model(device)
    x, y = data
    model = FullyBayesianSingleTaskGP(
        x, y, torch.full_like(y, 0.03), backend="torch", outcome_transform=None
    )
    model.load_mcmc_samples(
        {"mean": x.new_tensor([0.1, 0.3]), "lengthscale": x.new_tensor([[0.3], [0.7]])}
    )
    sampler = IIDNormalSampler((8,), seed=29)
    acq = ScaledSampleEI(model, x.new_tensor(0.2), sampler)
    assert acq.init_calls == 1
    query = x.new_tensor([[[0.25], [0.61]]], requires_grad=True)
    expected = qExpectedImprovement(model, x.new_tensor(0.2), sampler)(query) * acq.scale + 0.4
    leaves = (x, y, query, *model.parameters(), acq.scale)
    _compare_graphs(acq(query), expected, leaves, leaves)


def test_registry_uses_actual_base_identity_and_preserves_abstract_requirements():
    model, data = _model("cpu")
    # A coincident name is a genuine custom acquisition, not an EI formula.
    named = type("ExpectedImprovement", (CallerAcquisition,), {})
    acq = named(model)
    query = data[0].new_tensor([[[0.25]]])
    expected = (model.posterior(query).mean.sin() * acq.scale).sum((-1, -2))
    torch.testing.assert_close(acq(query), expected)

    class MissingForward(AcquisitionFunction):
        pass

    class MissingUtility(SampleReducingMCAcquisitionFunction):
        pass

    with pytest.raises(TypeError, match="abstract method forward"):
        MissingForward(model)
    with pytest.raises(TypeError, match="abstract method _sample_forward"):
        MissingUtility(model)

    class FalseyReduction:
        def __bool__(self):
            return False

        def __call__(self, values, *, dim):
            return values.sum(dim=dim)

    reducer = FalseyReduction()
    sampler = IIDNormalSampler((4,), seed=29)
    acq = PositiveUtility(model, sampler=sampler, sample_reduction=reducer, q_reduction=reducer)
    samples = sampler(model.posterior(query)).squeeze(-1)
    torch.testing.assert_close(acq(query), (samples.square() + 0.2).sum((-1, 0)))


@pytest.mark.parametrize("device", DEVICES)
def test_custom_analytic_mc_bases_and_inherited_log_reduction_hooks(device):
    model, data = _model(device)
    query = data[0].new_tensor([[[0.25], [0.61]]], requires_grad=True)
    analytic = CallerAnalytic(model)
    posterior = model.posterior(query[..., :1, :])
    expected = (posterior.mean + 0.3 * posterior.variance.sqrt()).squeeze(-1).squeeze(-1)
    torch.testing.assert_close(analytic(query[..., :1, :]), expected)
    sampler = IIDNormalSampler((8,), seed=29)
    mc = CallerMC(model, sampler=sampler)
    expected = sampler(model.posterior(query)).squeeze(-1).square().mean(0).sum(-1)
    leaves = (*data, query, *model.parameters())
    _compare_graphs(mc(query), expected, leaves, leaves)
    hooked = HookedLogEI(model, data[0].new_tensor(0.2), sampler=sampler)
    expected = qLogExpectedImprovement(model, data[0].new_tensor(0.2), sampler=sampler)(query) + 0.3
    _compare_graphs(hooked(query), expected, leaves, leaves)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_hooks_and_constrained_reductions_match_official_protocol(device):
    pytest.importorskip("botorch")
    from botorch.acquisition.analytic import LogExpectedImprovement as ReferenceLogEI
    from botorch.acquisition.monte_carlo import SampleReducingMCAcquisitionFunction as ReferenceSRC

    class ReferenceAffine(_AffineLogHooks, ReferenceLogEI):
        pass

    class ReferencePositive(ReferenceSRC):
        def _sample_forward(self, obj):
            return obj.square() + 0.2

    model, data = _model(device)
    best = data[0].new_tensor(0.2, requires_grad=True)
    query = data[0].new_tensor([[[0.25]]], requires_grad=True)
    port, reference = AffineLogEI(model, best), ReferenceAffine(model, best)
    leaves = (*data, best, query, *model.parameters())
    _compare_graphs(
        port(query), reference(query), (*leaves, port.shift), (*leaves, reference.shift)
    )
    query = data[0].new_tensor([[[0.25], [0.61]]], requires_grad=True)

    def sample_fn(values, *, dim):
        return values.tanh().sum(dim=dim)

    def constraint(samples):
        return 0.4 - samples.squeeze(-1)

    results = []
    for cls in (PositiveUtility, ReferencePositive):
        acq = cls(
            model,
            sampler=IIDNormalSampler((8,), seed=29),
            sample_reduction=sample_fn,
            q_reduction=torch.mean,
            constraints=[constraint],
            eta=0.2,
        )
        results.append(acq(query))
    leaves = (*data, query, *model.parameters())
    _compare_graphs(*results, leaves, leaves)
