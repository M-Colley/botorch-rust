# ruff: noqa: E402
"""Reference binary cells and approximate hypervolume retain device graphs."""

import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning as ReferencePartition,
)
from test_torch_backend_acquisition import FixedSampler
from test_torch_backend_hypervolume import acquisition, data

from botorch_rs.acquisition._torch_box_decomposition import partition_binary
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def canonical(bounds):
    values = bounds.transpose(0, 1).reshape(bounds.shape[1], 2 * bounds.shape[-1])
    order = sorted(range(values.shape[0]), key=lambda i: tuple(values[i].detach().cpu().tolist()))
    return values[order]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("outcomes", [3, 4])
@pytest.mark.parametrize("alpha", [0.0, 0.02, 0.2, 0.8])
def test_ragged_binary_cells_match_source(device, dtype, outcomes, alpha):
    generator = torch.Generator().manual_seed(173)
    Y = torch.rand(2, 3, 7, outcomes, generator=generator, dtype=dtype).to(device)
    Y[0, 0] = -1  # Empty front in one sample.
    Y[0, 1, 1] = Y[0, 1, 0]  # Duplicate coordinates.
    feasible = torch.ones(2, 3, 7, dtype=torch.bool, device=device)
    feasible[1, 0, 1:] = False  # A singleton front.
    feasible[1, 1] = False  # Empty feasibility set.
    ref = torch.full((outcomes,), 0.1, dtype=Y.dtype, device=device)
    lower, upper, active = partition_binary(Y, ref, feasible, alpha)
    assert lower.device == Y.device and upper.device == Y.device
    for row in range(2):
        for column in range(3):
            source = ReferencePartition(ref, Y=Y[row, column, feasible[row, column]], alpha=alpha)
            expected = source.get_hypercell_bounds()
            actual = torch.stack((lower[row, column], upper[row, column]))[:, active[row, column]]
            torch.testing.assert_close(canonical(actual), canonical(expected), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_binary_cell_coordinate_higher_derivatives(device, alpha):
    Y = torch.tensor(
        [[0.2, 0.7, 0.5], [0.4, 0.4, 0.6], [0.7, 0.2, 0.4]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    ref = torch.tensor([-0.3, -0.4, -0.5], dtype=Y.dtype, device=device, requires_grad=True)
    lower, upper, active = partition_binary(Y, ref, alpha=alpha)
    actual = canonical(torch.stack((lower, upper))[:, active])
    expected = canonical(ReferencePartition(ref, Y=Y, alpha=alpha).get_hypercell_bounds())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for result in (actual, expected):
        finite = result.masked_fill(~torch.isfinite(result), 0)
        loss = finite.sin().square().sum()
        first = torch.autograd.grad(loss, (Y, ref), create_graph=True, retain_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), (Y, ref), retain_graph=True
        )
        if result is actual:
            actual_derivatives = (*first, *second)
        else:
            for a, b in zip(actual_derivatives, (*first, *second), strict=True):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("alpha", [0.02, 0.3])
@pytest.mark.usefixtures("reference_python_logehvi")
def test_approximate_noisy_hypervolume_value_and_full_graph(device, log, alpha):
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    X, Y, ref, _ = data(device, 3)
    model = SingleTaskGP(X, Y, backend="torch")
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()
    kwargs = dict(X_baseline=X.detach()[::2], cache_root=False, prune_baseline=False, alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(log, noisy=True)(
            model, ref, sampler=SobolQMCNormalSampler((12,), seed=11), **kwargs
        )(query)
        expected = acquisition(log, True, noisy=True)(
            model, ref, sampler=ReferenceSampler(torch.Size((12,)), seed=11), **kwargs
        )(query)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    leaves = (query, X, Y, *model.parameters())
    first = torch.autograd.grad(actual.sum(), leaves, create_graph=True, retain_graph=True)
    source_first = torch.autograd.grad(expected.sum(), leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    source_second = torch.autograd.grad(sum(g.square().sum() for g in source_first), leaves)
    for a, b in zip((*first, *second), (*source_first, *source_second), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.usefixtures("reference_python_logehvi")
def test_binary_observed_front_live_graphs_without_host_copies(device, log, monkeypatch):
    X, Y, ref, observed = data(device, 3)
    ref.requires_grad_()
    observed.requires_grad_()
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()
    model = SingleTaskGP(X, Y, backend="torch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_acq = acquisition(log)(
            model,
            ref,
            NondominatedPartitioning(ref, Y=observed, alpha=0.1),
            sampler=FixedSampler(6, device),
        )
        expected = acquisition(log, True)(
            model,
            ref,
            ReferencePartition(ref, Y=observed, alpha=0.1),
            sampler=FixedSampler(6, device),
        )(query)

    def forbidden(*args, **kwargs):
        raise AssertionError("Binary hypervolume copied numerical tensors to host")

    for name in ("cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    actual = actual_acq(query)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    leaves = (observed, ref, query, X, Y, *model.parameters())
    first = torch.autograd.grad(actual.sum(), leaves, create_graph=True, retain_graph=True)
    source_first = torch.autograd.grad(expected.sum(), leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    source_second = torch.autograd.grad(sum(g.square().sum() for g in source_first), leaves)
    for a, b in zip((*first, *second), (*source_first, *source_second), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("marginalize", [-3, 1])
def test_noisy_pruning_marginalizes_an_ordinary_model_batch(device, marginalize, monkeypatch):
    import botorch.acquisition.utils as reference_utils
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    X, Y, ref, _ = data(device, 3)
    model = SingleTaskGP(X.detach().repeat(2, 1, 1), Y.detach().repeat(2, 1, 1), backend="torch")
    baseline = X.detach()
    monkeypatch.setattr(
        reference_utils,
        "get_sampler",
        lambda posterior, sample_shape: ReferenceSampler(sample_shape, seed=11),
    )
    kwargs = dict(
        X_baseline=baseline,
        cache_root=False,
        prune_baseline=True,
        marginalize_dim=marginalize,
        constraints=[lambda y: y[..., 2] - 0.8],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(False, noisy=True)(
            model, ref, sampler=SobolQMCNormalSampler((12,), seed=11), **kwargs
        )
        expected = acquisition(False, True, noisy=True)(
            model, ref, sampler=ReferenceSampler(torch.Size((12,)), seed=11), **kwargs
        )
    torch.testing.assert_close(actual.X_baseline, expected.X_baseline, rtol=0, atol=0)
    torch.testing.assert_close(actual(baseline[:1]), expected(baseline[:1]), rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
def test_pruning_only_dimension_is_ignored_when_disabled(device, log):
    X, Y, ref, _ = data(device, 3)
    model = SingleTaskGP(X, Y, backend="torch")
    kwargs = dict(X_baseline=X.detach(), cache_root=False, prune_baseline=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ordinary = acquisition(log, noisy=True)(
            model, ref, sampler=SobolQMCNormalSampler((12,), seed=11), **kwargs
        )
        ignored = acquisition(log, noisy=True)(
            model,
            ref,
            sampler=SobolQMCNormalSampler((12,), seed=11),
            marginalize_dim=999,
            **kwargs,
        )
    torch.testing.assert_close(ignored(X.detach()[:1]), ordinary(X.detach()[:1]), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_binary_front_retains_reference_graph(device):
    Y = torch.empty(2, 0, 4, device=device, dtype=torch.double)
    ref = torch.full((4,), -0.3, device=device, dtype=torch.double, requires_grad=True)
    lower, upper, active = partition_binary(Y, ref, alpha=0.2)
    assert active.shape == (2, 1) and active.all() and upper.isposinf().all()
    torch.testing.assert_close(lower, ref.expand(2, 1, 4))
    torch.testing.assert_close(torch.autograd.grad(lower.sum(), ref)[0], torch.full_like(ref, 2))
