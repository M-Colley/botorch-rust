# ruff: noqa: E402
"""Batched pending decisions preserve sample coordinates and posterior graphs."""

import warnings

import pytest

torch = pytest.importorskip("torch")

from test_torch_backend_hypervolume import acquisition, data

from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_acq(model, ref, baseline, pending, log, cache, incremental):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return acquisition(log, noisy=True)(
            model,
            ref,
            X_baseline=baseline,
            X_pending=pending,
            sampler=SobolQMCNormalSampler((12,), seed=11),
            cache_root=False,
            prune_baseline=False,
            cache_pending=cache,
            incremental_nehvi=incremental,
            max_iep=1,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_pending_batches_match_independent_decisions_and_full_graphs(
    device, log, cache, incremental
):
    X, Y, ref, _ = data(device, 2)
    model = SingleTaskGP(X, Y, backend="torch")
    baseline = X.detach()[::2]
    query = X.new_tensor([[[0.21]], [[0.56]]], requires_grad=True)
    pending = X.new_tensor([[[0.72]], [[0.89]]])
    actual_acq = make_acq(model, ref, baseline, pending, log, cache, incremental)
    independent = [
        make_acq(model, ref, baseline, pending[row], log, cache, incremental) for row in range(2)
    ]
    # Append beyond max_iep, replace below it, then clear pending decisions.
    transitions = [pending, torch.cat((pending, pending - 0.09), -2), pending + 0.02, None]
    for index, current in enumerate(transitions):
        if index:
            actual_acq.set_X_pending(current)
            for row, reference in enumerate(independent):
                reference.set_X_pending(None if current is None else current[row])
        actual = actual_acq(query)
        expected = torch.cat([reference(query[row]) for row, reference in enumerate(independent)])
        assert actual.shape == (2,)
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
        if index == 1:
            leaves = (query, X, Y, *model.parameters())
            first = torch.autograd.grad(actual.sum(), leaves, create_graph=True, retain_graph=True)
            oracle_first = torch.autograd.grad(
                expected.sum(), leaves, create_graph=True, retain_graph=True
            )
            second = torch.autograd.grad(
                sum(g.square().sum() for g in first), leaves, retain_graph=True
            )
            oracle_second = torch.autograd.grad(sum(g.square().sum() for g in oracle_first), leaves)
            for a, b in zip((*first, *second), (*oracle_first, *oracle_second), strict=True):
                torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.usefixtures("reference_python_logehvi")
def test_uncached_batched_pending_matches_official_decisions_and_higher_graphs(device, log):
    pytest.importorskip("botorch")
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    X, Y, ref, _ = data(device, 2)
    model = SingleTaskGP(X, Y, backend="torch")
    query = X.new_tensor([[[0.21]], [[0.56]]], requires_grad=True)
    pending = X.new_tensor([[[0.72]], [[0.89]]])
    kwargs = dict(
        X_baseline=X.detach()[::2],
        X_pending=pending,
        cache_root=False,
        prune_baseline=False,
        cache_pending=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(log, noisy=True)(
            model,
            ref,
            sampler=SobolQMCNormalSampler((12,), seed=11),
            **kwargs,
        )(query)
        expected = torch.cat(
            [
                acquisition(log, True, noisy=True)(
                    model,
                    ref,
                    sampler=ReferenceSampler(torch.Size((12,)), seed=11),
                    **{**kwargs, "X_pending": pending[row]},
                )(query[row])
                for row in range(2)
            ]
        )
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    leaves = (query, X, Y, *model.parameters())
    first = torch.autograd.grad(actual.sum(), leaves, create_graph=True, retain_graph=True)
    oracle_first = torch.autograd.grad(expected.sum(), leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    oracle_second = torch.autograd.grad(sum(g.square().sum() for g in oracle_first), leaves)
    for a, b in zip((*first, *second), (*oracle_first, *oracle_second), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)
