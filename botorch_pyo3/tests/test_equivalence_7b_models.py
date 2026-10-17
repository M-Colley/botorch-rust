"""Randomized equivalence sweep of the models of phase 7b against BoTorch:
random ``SingleTaskMultiFidelityGP`` (fidelity columns, linear truncated or
downsampling kernels, ``nu``, outputs, fixed noise), ``MixedSingleTaskGP``
(categorical columns and categories, outputs) and MAP SAAS models (learned,
fixed or random taus, additive, ensembles) with random data and
hyperparameters: structure, MLL and its gradient, posteriors and the input
gradients of ``PosteriorMean`` / ``PosteriorStandardDeviation`` (values
1e-9, gradients 1e-8)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.models.map_saas as bt_saas  # noqa: E402
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.models import MixedSingleTaskGP as BtMixed  # noqa: E402
from botorch.models import SingleTaskMultiFidelityGP as BtMF  # noqa: E402
from models7b_helpers import (  # noqa: E402
    DT,
    VALUE_TOL,
    check_grad,
    compare_mll,
    compare_posteriors,
    compare_structure,
    randomize,
    values_and_grads,
)

import botorch_rs.models.map_saas as rs_saas  # noqa: E402
from botorch_rs.acquisition import analytic as rs_an  # noqa: E402
from botorch_rs.models import MixedSingleTaskGP as RsMixed  # noqa: E402
from botorch_rs.models import SingleTaskMultiFidelityGP as RsMF  # noqa: E402
from parity_helpers import Checker  # noqa: E402

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

CHECK = Checker("models (7b) sweep")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Any:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _targets(X: Any, m: int, g: Any) -> Any:
    cols = []
    for t in range(m):
        f = torch.sin((2.0 + t) * X[:, 0]) + (0.5 + t) * X[:, 1:].sum(-1)
        cols.append(f + 0.05 * torch.randn(X.shape[0], generator=g, dtype=DT))
    return torch.stack(cols, -1)


def _mf_case(r: random.Random, g: Any) -> tuple[Any, Any, Any]:
    d = r.randint(3, 5)
    n = r.randint(6, 14)
    m = r.choice([1, 1, 2])
    X = torch.rand(n, d, generator=g, dtype=DT)
    Y = _targets(X, m, g)
    lt = r.random() < 0.5
    kw: dict[str, Any] = {"linear_truncated": lt}
    fid = r.sample(range(1, d), r.choice([1, 2]) if d > 3 else 1)
    if r.random() < 0.5 or len(fid) == 1:
        kw["data_fidelities"] = fid if r.random() < 0.7 else [f - d for f in fid]
        if r.random() < 0.4 and len(fid) == 1:
            others = [i for i in range(1, d) if i not in fid]
            kw["iteration_fidelity"] = r.choice(others)
    else:
        kw["iteration_fidelity"] = fid[0]
        kw["data_fidelities"] = fid[1:]
    if lt:
        kw["nu"] = r.choice([0.5, 1.5, 2.5])
    if r.random() < 0.3:
        kw["train_Yvar"] = 0.01 + 0.02 * torch.rand(n, m, generator=g, dtype=DT)
    return BtMF(X, Y, **kw), RsMF(X, Y, **kw), X


def _mixed_case(r: random.Random, g: Any) -> tuple[Any, Any, Any]:
    d_cont = r.randint(0, 3)
    n_cat = r.randint(1, 2)
    n = r.randint(6, 14)
    m = r.choice([1, 1, 2])
    cont = torch.rand(n, d_cont, generator=g, dtype=DT)
    cats = torch.stack(
        [torch.randint(0, r.randint(2, 4), (n,), generator=g) for _ in range(n_cat)], -1
    ).to(DT)
    cols = list(range(d_cont + n_cat))
    perm = r.sample(cols, len(cols))
    X = torch.cat([cont, cats], -1)[:, perm]
    cat_dims = [perm.index(d_cont + j) for j in range(n_cat)]
    if r.random() < 0.3:
        cat_dims = [c - X.shape[-1] for c in cat_dims]
    Y = _targets(torch.cat([X, X], -1) if X.shape[-1] == 1 else X, m, g)
    return BtMixed(X, Y, cat_dims=cat_dims), RsMixed(X, Y, cat_dims=cat_dims), X


def _saas_case(r: random.Random, g: Any, seed: int) -> tuple[Any, Any, Any]:
    d = r.randint(2, 5)
    n = r.randint(6, 14)
    X = torch.rand(n, d, generator=g, dtype=DT)
    kind = r.choice(["map", "map_tau", "additive", "ensemble"])
    m = 1 if kind == "ensemble" else r.choice([1, 1, 2])
    Y = _targets(torch.cat([X, X], -1) if d == 1 else X, m, g)
    torch.manual_seed(seed)
    if kind == "map":
        bt = bt_saas.get_map_saas_model(X, Y)
        torch.manual_seed(seed)
        rs = rs_saas.get_map_saas_model(X, Y)
    elif kind == "map_tau":
        tau = torch.rand(1, generator=g, dtype=DT).item() + 0.05
        bt = bt_saas.get_map_saas_model(X, Y, tau=torch.tensor(tau, dtype=DT))
        rs = rs_saas.get_map_saas_model(X, Y, tau=torch.tensor(tau, dtype=DT))
    elif kind == "additive":
        nt = r.randint(1, 4)
        bt = bt_saas.AdditiveMapSaasSingleTaskGP(X, Y, num_taus=nt)
        torch.manual_seed(seed)
        rs = rs_saas.AdditiveMapSaasSingleTaskGP(X, Y, num_taus=nt)
    else:
        nt = r.randint(2, 4)
        bt = bt_saas.EnsembleMapSaasSingleTaskGP(X, Y, num_taus=nt)
        torch.manual_seed(seed)
        rs = rs_saas.EnsembleMapSaasSingleTaskGP(X, Y, num_taus=nt)
    return bt, rs, X


@pytest.mark.parametrize("scenario", range(45))
def test_random_models_sweep(scenario: int) -> None:
    r = random.Random(7000 + scenario)
    g = torch.Generator().manual_seed(7000 + scenario)
    family = ("mf", "mixed", "saas")[scenario % 3]
    if family == "mf":
        bt, rs, X = _mf_case(r, g)
    elif family == "mixed":
        bt, rs, X = _mixed_case(r, g)
    else:
        bt, rs, X = _saas_case(r, g, 7000 + scenario)
    compare_structure(CHECK, bt, rs)
    compare_mll(CHECK, f"sweep {family} default", bt, rs)
    randomize(bt, rs, 7100 + scenario, scale=0.4)
    compare_mll(CHECK, f"sweep {family} random", bt, rs)
    Xt = X[: min(3, X.shape[0])].clone()
    if family != "mixed":
        Xt = torch.rand(3, X.shape[-1], generator=g, dtype=DT)
    if getattr(bt, "_is_ensemble", False):
        bt.eval()
        with torch.no_grad():
            pb, pr = bt.posterior(Xt), rs.posterior(Xt)
            CHECK.check(
                "sweep saas: ensemble mixture mean", pr.mixture_mean, pb.mixture_mean, VALUE_TOL
            )
            CHECK.check(
                "sweep saas: ensemble mixture variance",
                pr.mixture_variance,
                pb.mixture_variance,
                VALUE_TOL,
            )
        X1 = Xt.unsqueeze(-2)
        best = torch.tensor(1.0, dtype=DT)
        vb, gb = values_and_grads(bt_an.LogExpectedImprovement(bt, best), X1)
        vr, gr = values_and_grads(rs_an.LogExpectedImprovement(rs, best), X1)
        CHECK.check("sweep saas: LogEI on ensemble", vr, vb, VALUE_TOL)
        check_grad(CHECK, "sweep saas: LogEI on ensemble gradients", gr, gb)
    else:
        compare_posteriors(CHECK, f"sweep {family}", bt, rs, Xt)
