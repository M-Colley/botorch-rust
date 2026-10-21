"""Heterogeneous-domain empirical multi-task inference and official parity."""

# ruff: noqa: F811
from __future__ import annotations

import math

import pytest
from test_empirical_gp_models import official, restore_torch_dtype  # noqa: F401

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.empirical_gps.multitask_empirical_1d_gp import (
    MultiTaskEmpiricalOneDimensionalGP,
    MultiTaskEmpiricalOneDimensionalKernel,
    MultiTaskEmpiricalOneDimensionalMean,
)
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy
from helpers import KINDS, as_kind, assert_close, check_kind, shape, to_list


def data() -> tuple[list, list, list, list]:
    Xs = [[[0.0], [0.2], [0.5], [0.8], [1.0]], [[0.5], [0.9], [1.15], [1.5]]]
    Ys = [
        [[1 + 0.1 * r + math.sin((r + 1) * p[0] * 0.8 + t) for p in X] for r in range(6)]
        for t, X in enumerate(Xs)
    ]
    X = [[0.1, 0.0], [0.4, 0.0], [0.7, 0.0], [0.95, 0.0]]
    Y = [[1.5 + math.sin(3 * p[0])] for p in X]
    return Xs, Ys, X, Y


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("task_feature", [0, 1])
def test_task_domain_selection_fitting_and_adapters(kind: str, task_feature: int) -> None:
    Xs, Ys, X, Y = data()
    if task_feature == 0:
        X = [row[::-1] for row in X]
    Xs, Ys = [as_kind(x, kind) for x in Xs], [as_kind(y, kind) for y in Ys]
    model = MultiTaskEmpiricalOneDimensionalGP(
        as_kind(X, kind), as_kind(Y, kind), task_feature, Xs, Ys, output_tasks=[1]
    )
    assert model.num_tasks == 2 and model.num_outputs == 1
    # Predict task one where task zero has no progression data.
    Q = as_kind([[1.2], [1.4]], kind)
    p = model.posterior(Q)
    assert shape(p.mean) == (2, 1)
    check_kind(p.mean, kind)
    explicit = [[1.2, 1], [0.35, 0]]
    if task_feature == 0:
        explicit = [row[::-1] for row in explicit]
    pe = model.posterior(as_kind(explicit, kind))
    assert shape(pe.mean) == (2, 1)
    assert_close(to_list(pe.mean)[0], to_list(p.mean)[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = float(to_list(mll()))
    result = fit_gpytorch_mll_scipy(mll, options={"maxiter": 8})
    assert math.isfinite(result.fval) and float(to_list(mll())) >= before
    for bad in (0.4, -1.0, 2.0, math.nan):
        row = [0.8, bad]
        if task_feature == 0:
            row.reverse()
        with pytest.raises(ValueError):
            _ = model.posterior(as_kind([row], kind)).mean
    with pytest.raises(ValueError, match="outside"):
        _ = model.posterior(as_kind([[0.2]], kind)).mean


@pytest.mark.parametrize("task_feature", [0, 1])
def test_official_standalone_task_mean_kernel_and_gradients(
    official: object, task_feature: int
) -> None:
    import torch

    Xs, Ys, _, _ = data()
    Xs = [torch.tensor(x, dtype=torch.double) for x in Xs]
    Ys = [torch.tensor(y, dtype=torch.double) for y in Ys]
    source = official.multitask_empirical_1d_gp
    a = MultiTaskEmpiricalOneDimensionalKernel(Xs, Ys, task_feature, 6, correction=1)
    b = source.MultiTaskEmpiricalOneDimensionalKernel(Xs, Ys, task_feature, 6, correction=1)
    ma = MultiTaskEmpiricalOneDimensionalMean(Xs, Ys, task_feature)
    mb = source.MultiTaskEmpiricalOneDimensionalMean(Xs, Ys, task_feature)
    rows = [[0.15, 0], [1.21, 1], [0.64, 0], [0.86, 1]]
    if task_feature == 0:
        rows = [r[::-1] for r in rows]
    x = torch.tensor(rows, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    ka, kb = a(x), b(y, y).to_dense()
    torch.testing.assert_close(ka, kb, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(ma(x), mb(y), atol=2e-12, rtol=2e-12)
    ga = torch.autograd.grad(ka.sum() + ma(x).sum(), x)[0]
    gb = torch.autograd.grad(kb.sum() + mb(y).sum(), y)[0]
    torch.testing.assert_close(ga, gb, atol=3e-10, rtol=3e-10)
    q = x.detach().reshape(2, 2, 2)
    torch.testing.assert_close(a(q), b(q, q).to_dense(), atol=2e-12, rtol=2e-12)
    near = x.detach().clone()
    near[:, task_feature] += 5e-7
    torch.testing.assert_close(a(near), ka.detach())


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("task_feature", [0, 1])
def test_official_task_model_mll_and_posteriors(
    official: object, fixed: bool, task_feature: int
) -> None:
    import torch

    pytest.importorskip("gpytorch")
    from gpytorch.mlls import ExactMarginalLogLikelihood as OfficialMLL

    torch.set_default_dtype(torch.double)
    Xs, Ys, X, Y = data()
    if fixed:
        X.append([0.8, 1.0])
        Y.append([1.8])
    if task_feature == 0:
        X = [row[::-1] for row in X]
    Xs = [torch.tensor(v) for v in Xs]
    Ys = [torch.tensor(v) for v in Ys]
    X, Y = torch.tensor(X), torch.tensor(Y)
    variance = torch.tensor([[0.01], [0.02], [0.03], [0.04], [0.12]]) if fixed else None
    a = MultiTaskEmpiricalOneDimensionalGP(X, Y, task_feature, Xs, Ys, train_Yvar=variance)
    b = official.multitask_empirical_1d_gp.MultiTaskEmpiricalOneDimensionalGP(
        X, Y, task_feature, Xs, Ys, train_Yvar=variance
    )
    torch.testing.assert_close(a.train_inputs[0], b.train_inputs[0], rtol=0, atol=0)
    torch.testing.assert_close(a.train_targets, b.train_targets, rtol=0, atol=0)
    va, ga = ExactMarginalLogLikelihood(a.likelihood, a).value_and_grad()
    vb = OfficialMLL(b.likelihood, b)(b(*b.train_inputs), b.train_targets)
    torch.testing.assert_close(torch.as_tensor(va), vb, atol=3e-10, rtol=3e-10)
    if not fixed:
        vb.backward()
        torch.testing.assert_close(
            torch.tensor(ga), b.likelihood.raw_noise.grad.flatten(), atol=3e-10, rtol=3e-10
        )
    else:
        assert ga == []
    b.eval()
    q = torch.tensor([[0.6], [0.9]])
    for noise in (False, True):
        pa, pb = a.posterior(q, observation_noise=noise), b.posterior(q, observation_noise=noise)
        torch.testing.assert_close(pa.mean, pb.mean, atol=3e-10, rtol=3e-10)
        torch.testing.assert_close(
            pa.distribution.covariance_matrix,
            pb.distribution.covariance_matrix,
            atol=3e-10,
            rtol=3e-10,
        )
    explicit = torch.tensor([[1.2, 1.0], [0.36, 0.0]])
    if task_feature == 0:
        explicit = explicit.flip(-1)
    torch.testing.assert_close(
        a.posterior(explicit).mean, b.posterior(explicit).mean, atol=3e-10, rtol=3e-10
    )


def test_unobserved_task_fixed_noise_fallback_and_upstream_reproduction(official: object) -> None:
    import torch

    torch.set_default_dtype(torch.double)
    Xs, Ys, X, Y = data()
    Xs = [torch.tensor(v) for v in Xs]
    Ys = [torch.tensor(v) for v in Ys]
    X, Y = torch.tensor(X), torch.tensor(Y)
    variance = torch.full_like(Y, 0.023)
    a = MultiTaskEmpiricalOneDimensionalGP(X, Y, 1, Xs, Ys, train_Yvar=variance)
    b = official.multitask_empirical_1d_gp.MultiTaskEmpiricalOneDimensionalGP(
        X, Y, 1, Xs, Ys, train_Yvar=variance
    )
    Q = torch.tensor([[0.6], [0.9]])
    p, n = a.posterior(Q), a.posterior(Q, observation_noise=True)
    torch.testing.assert_close(n.variance - p.variance, torch.full((2, 2), 0.023))
    assert torch.isfinite(n.distribution.covariance_matrix).all()
    b.eval()
    assert b.posterior(Q, observation_noise=True).variance[:, 1].isnan().all()


@pytest.mark.parametrize("kind", KINDS)
def test_wide_and_prebuilt_module_reuse(kind: str) -> None:
    Xs, Ys, _, _ = data()
    Xs = [as_kind(v, kind) for v in Xs]
    Ys = [as_kind(v, kind) for v in Ys]
    X = as_kind([[0.6], [0.9]], kind)
    Y = as_kind([[1.4, 1.8], [1.3, 1.7]], kind)
    a = MultiTaskEmpiricalOneDimensionalGP.from_wide_format(X, Y, Xs, Ys)
    b = MultiTaskEmpiricalOneDimensionalGP(
        a.train_inputs[0],
        as_kind([[1.4], [1.8], [1.3], [1.7]], kind),
        1,
        mean_module=a.mean_module,
        covar_module=a.covar_module,
    )
    assert_close(a.posterior(X).mean, b.posterior(X).mean)
    assert shape(a.train_targets) == (4,)
    before, tasks, after = a._split_inputs(a.train_inputs[0])
    assert shape(before) == (4, 1) and shape(tasks) == (4, 1) and shape(after) == (4, 0)
