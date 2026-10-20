# ruff: noqa: E402
"""Mixture reductions preserve ordinary batches, graphs, and shared normals."""

import pytest

pytest.importorskip("torch")


import importlib
import warnings
from types import SimpleNamespace

import torch
from test_torch_backend_acquisition import ANALYTIC, MC, FixedSampler, acquisition, native_data

from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class EnsembleGP(torch.nn.Module):
    """Three distinct native Tensor GPs with the fully Bayesian query layout."""

    _tensor_backend = "torch"
    _is_ensemble = _is_fully_bayesian = True
    batch_shape = torch.Size((3,))

    def __init__(self, device, outcomes=1):
        super().__init__()
        X, Y = native_data(device)
        self.X = X.requires_grad_()
        self.Y = torch.cat([(j + 1) * Y + 0.1 * j for j in range(outcomes)], -1).requires_grad_()
        self.num_outputs = outcomes
        member = torch.arange(3, dtype=Y.dtype, device=device).reshape(3, 1, 1)
        self.gp = SingleTaskGP(
            self.X.expand(3, -1, -1),
            self.Y.unsqueeze(0) * (1 + 0.15 * member) + 0.05 * member,
            backend="torch",
        )

    def posterior(self, X, **kwargs):
        return self.gp.posterior(X.unsqueeze(-3), **kwargs)


def compare_graphs(actual, expected, leaves):
    torch.testing.assert_close(actual, expected, rtol=3e-8, atol=3e-9)
    first = torch.autograd.grad(
        actual.sum(), leaves, create_graph=True, retain_graph=True, allow_unused=True
    )
    reference_first = torch.autograd.grad(
        expected.sum(), leaves, create_graph=True, retain_graph=True, allow_unused=True
    )
    assert [g is None for g in first] == [g is None for g in reference_first]
    first = [leaf * 0 if g is None else g for leaf, g in zip(leaves, first, strict=True)]
    reference_first = [
        leaf * 0 if g is None else g for leaf, g in zip(leaves, reference_first, strict=True)
    ]
    for a, b in zip(first, reference_first, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    reference_second = torch.autograd.grad(
        sum(g.square().sum() for g in reference_first), leaves, retain_graph=True
    )
    for a, b in zip(second, reference_second, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,arguments", ANALYTIC + MC)
def test_ensemble_analytic_and_scalar_mc_match_reference(device, name, arguments):
    model = EnsembleGP(device)
    q = 2 if name.startswith("q") else 1
    query = model.X.new_tensor([[0.21], [0.56]][:q]).expand(2, 4, q, 1).clone().requires_grad_()
    kwargs = dict(arguments)
    if q == 2:
        kwargs["sampler"] = FixedSampler(q, device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(name)(model, **kwargs)(query)
        reference = acquisition(name, True)(model, **kwargs)
        if name == "PosteriorStandardDeviation":
            # The pinned reference calls std.view(X.shape[:-2]) before its
            # ensemble decorator, which cannot retain an extra MCMC axis.
            with pytest.raises(RuntimeError, match="invalid for input"):
                reference(query)
            expected = model.posterior(query).variance.sqrt().squeeze(-1).squeeze(-1).mean(-1)
        else:
            expected = reference(query)
    assert actual.shape == (2, 4)
    compare_graphs(actual, expected, (query, model.X, model.Y, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
def test_ensemble_noisy_ei_shared_normals_and_member_incumbents(device, log):
    model = EnsembleGP(device)
    name = "qLogNoisyExpectedImprovement" if log else "qNoisyExpectedImprovement"
    baseline = model.X.detach()[::2]
    query = model.X.new_tensor([[[0.21], [0.56]], [[0.31], [0.64]]], requires_grad=True)
    kwargs = dict(
        X_baseline=baseline,
        sampler=FixedSampler(4, device),
        cache_root=False,
        prune_baseline=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(name)(model, **kwargs)(query)
        expected = acquisition(name, True)(model, **kwargs)(query)
    assert actual.shape == (2,)
    compare_graphs(actual, expected, (query, model.X, model.Y, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name",
    [
        "ConstrainedExpectedImprovement",
        "LogConstrainedExpectedImprovement",
        "LogProbabilityOfFeasibility",
        "ScalarizedPosteriorMean",
    ],
)
def test_extra_analytic_ensemble_reductions(device, name):
    scalarized = name == "ScalarizedPosteriorMean"
    model = EnsembleGP(device, outcomes=1 if scalarized else 2)
    query = (
        model.X.new_tensor([[0.21], [0.56]][: 2 if scalarized else 1])
        .expand(4, -1, -1)
        .clone()
        .requires_grad_()
    )
    kwargs = (
        {"weights": query.new_tensor([0.3, -0.2])}
        if scalarized
        else {"constraints": {1: (None, 0.7)}}
    )
    if "ExpectedImprovement" in name:
        kwargs["best_f"] = 0.25
        kwargs["objective_index"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls = getattr(importlib.import_module("botorch_rs.acquisition.analytic"), name)
        reference_cls = getattr(importlib.import_module("botorch.acquisition.analytic"), name)
        actual, expected = cls(model, **kwargs)(query), reference_cls(model, **kwargs)(query)
    assert actual.shape == (4,)
    compare_graphs(actual, expected, (query, model.X, model.Y, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("noisy", [False, True])
def test_ensemble_hypervolume_averages_after_member_hvi(device, log, noisy):
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning as ReferencePartition,
    )
    from test_torch_backend_hypervolume import acquisition as hv_acquisition

    from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    model = EnsembleGP(device, outcomes=2)
    ref = model.X.new_tensor([-0.3, -0.3])
    query = model.X.new_tensor([[[0.21], [0.56]], [[0.31], [0.64]]], requires_grad=True)
    kwargs = (
        dict(X_baseline=model.X.detach()[::2], cache_root=False, prune_baseline=False)
        if noisy
        else {}
    )
    observed = model.X.new_tensor([[0.1, 0.4], [0.3, 0.2]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = hv_acquisition(log, noisy=noisy)(
            model,
            ref,
            sampler=SobolQMCNormalSampler((12,), seed=13),
            **kwargs,
            **({} if noisy else {"partitioning": FastNondominatedPartitioning(ref, observed)}),
        )(query)
        expected = hv_acquisition(log, reference=True, noisy=noisy)(
            model,
            ref,
            sampler=ReferenceSampler(torch.Size((12,)), seed=13),
            **kwargs,
            **({} if noisy else {"partitioning": ReferencePartition(ref, observed)}),
        )(query)
    assert actual.shape == (2,)
    compare_graphs(actual, expected, (query, model.X, model.Y, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
def test_ensemble_scalar_pruning_matches_reference_with_same_seed(device):
    from botorch.acquisition.utils import prune_inferior_points
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    model = EnsembleGP(device)

    def constraint(y):
        return y[..., 0] - 0.5

    reference = prune_inferior_points(
        model,
        model.X.detach(),
        constraints=[constraint],
        sampler=ReferenceSampler(torch.Size((2048,)), seed=19),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition("qNoisyExpectedImprovement")(
            model,
            X_baseline=model.X.detach(),
            constraints=[constraint],
            sampler=SobolQMCNormalSampler((12,), seed=19),
            cache_root=False,
            marginalize_dim=-3,
        )
    torch.testing.assert_close(actual.X_baseline, reference)


@pytest.mark.parametrize("device", DEVICES)
def test_ensemble_front_pruning_matches_reference_member_marginalization(device):
    from botorch.acquisition.utils import _prune_inferior_shared_processing
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from test_torch_backend_hypervolume import acquisition as hv_acquisition

    model = EnsembleGP(device, outcomes=2)
    ref = model.X.new_tensor([-0.3, -0.3])

    def constraint(y):
        return y[..., 1] - 0.7

    _, obj, infeasible = _prune_inferior_shared_processing(
        model,
        model.X.detach(),
        is_moo=True,
        constraints=[constraint],
        sampler=ReferenceSampler(torch.Size((2048,)), seed=19),
    )
    obj = torch.where(infeasible.unsqueeze(-1), ref, obj)
    possible = (is_non_dominated(obj, deduplicate=False) & (obj > ref).all(-1)).any(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = hv_acquisition(False, noisy=True)(
            model,
            ref,
            X_baseline=model.X.detach(),
            constraints=[constraint],
            sampler=SobolQMCNormalSampler((12,), seed=19),
            cache_root=False,
            marginalize_dim=-3,
            prune_baseline=True,
        )
    torch.testing.assert_close(actual.X_baseline, model.X.detach()[possible])


@pytest.mark.parametrize("device", DEVICES)
def test_ensemble_default_sampler_full_graph_has_no_host_conversion(device, monkeypatch):
    import botorch_rs._arrays as arrays

    model = EnsembleGP(device)
    query = model.X.new_tensor([[[0.21], [0.56]], [[0.31], [0.64]]], requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Ensemble acquisition copied Tensor data to the host")

    for name in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(arrays, "to_flat", forbidden)
    actual = acquisition("qLogExpectedImprovement")(model, best_f=0.25)
    value, gradient = actual.value_and_grad(query)
    assert value.shape == (2,) and gradient.device == query.device
    second = torch.autograd.grad(gradient.square().sum(), query)[0]
    assert torch.isfinite(second).all()


def test_ensemble_detection_recurses_into_module_lists_without_importing_torch():
    from botorch_rs.utils.transforms import is_ensemble, is_fully_bayesian

    child = torch.nn.Module()
    child._is_ensemble = child._is_fully_bayesian = True
    model = SimpleNamespace(models=torch.nn.ModuleList([child]))
    assert is_ensemble(model) and is_fully_bayesian(model)
