"""Parity of the models of phase 7b with BoTorch, through the public APIs of
both libraries side by side:

* ``SingleTaskMultiFidelityGP`` (linear truncated and downsampling /
  exponential decay kernels, one or two fidelities, ``nu``, a custom
  covariance module, several outputs, fixed noise, ``Normalize``): parameter
  names, shapes, default values and optimizer bounds; the MLL and its
  gradient at the default and at random parameters; posteriors and the
  input gradients of ``PosteriorMean`` / ``PosteriorStandardDeviation``;
  fits; ``subset_output``; ``qMultiFidelityKnowledgeGradient``,
  ``qMultiFidelityMaxValueEntropy`` and GIBBON (BoTorch's max values and
  base samples) and ``qMultiFidelityHypervolumeKnowledgeGradient`` on a list
  of multi-fidelity GPs;
* ``MixedSingleTaskGP`` (categorical columns, all-categorical inputs, a
  ``cont_kernel_factory``, several outputs): structure, MLL, posteriors,
  fits, ``qLogNoisyExpectedImprovement`` and ``optimize_acqf_mixed`` /
  ``optimize_acqf_mixed_alternating`` from identical initial conditions;
* the MAP SAAS models (``get_map_saas_model`` with a learned, fixed or
  batched ``tau``, ``AdditiveMapSaasSingleTaskGP``,
  ``EnsembleMapSaasSingleTaskGP``; the random taus drawn from torch's
  generator in both): structure, MLL, posteriors, the mixture posterior,
  fits, the acquisition functions on the ensemble (averaged over the
  members) and the ensemble pruning of the noisy expected improvement;
* ``AffineDeterministicModel``, ``FixedSingleSampleModel``, ``ModelList``
  (of GPs and deterministic models), ``EnsembleModel`` /
  ``EnsemblePosterior`` (with BoTorch's ``IndexSampler`` draws), and
  ``construct_inputs``.

Tolerances: values 1e-9, gradients 1e-8 (relative); fits as in
``models7b_helpers.compare_fits`` (twins with training inputs perturbed at
1e-14 .. 1e-11). Python floats that BoTorch stores in float32 (``beta``,
``tau`` of qPI, ...) are passed as float64 tensors. The randomized sweep is
in ``test_equivalence_7b_models.py``.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.acquisition.knowledge_gradient as bt_kg  # noqa: E402
import botorch.models.map_saas as bt_saas  # noqa: E402
import botorch.optim.optimize as bto  # noqa: E402
import botorch.optim.optimize_mixed as btm  # noqa: E402
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.utils import project_to_target_fidelity as bt_project  # noqa: E402
from botorch.acquisition.utils import prune_inferior_points as bt_prune  # noqa: E402
from botorch.models import GenericDeterministicModel as BtGeneric  # noqa: E402
from botorch.models import MixedSingleTaskGP as BtMixed  # noqa: E402
from botorch.models import ModelList as BtModelList  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models import SingleTaskMultiFidelityGP as BtMF  # noqa: E402
from botorch.models.deterministic import AffineDeterministicModel as BtAffine  # noqa: E402
from botorch.models.deterministic import FixedSingleSampleModel as BtFixedSample  # noqa: E402
from botorch.models.ensemble import EnsembleModel as BtEnsembleModel  # noqa: E402
from botorch.models.transforms.input import Normalize as BtNormalize  # noqa: E402
from botorch.sampling.index_sampler import IndexSampler as BtIndexSampler  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.utils.datasets import SupervisedDataset as BtDataset  # noqa: E402
from gpytorch.kernels import MaternKernel as BtMatern  # noqa: E402
from hvkg_helpers import HvkgCase, run_hvkg_case  # noqa: E402
from models7b_helpers import (  # noqa: E402
    DT,
    GRAD_TOL,
    VALUE_TOL,
    check_grad,
    compare_fits,
    compare_mll,
    compare_posteriors,
    compare_structure,
    randomize,
    values_and_grads,
)
from optim9_helpers import botorch_parallel_lbfgsb, patched, starting_points  # noqa: E402

import botorch_rs.acquisition.knowledge_gradient as rs_kg  # noqa: E402
import botorch_rs.models.map_saas as rs_saas  # noqa: E402
import botorch_rs.optim.optimize as rso  # noqa: E402
import botorch_rs.optim.optimize_mixed as rsm  # noqa: E402
from botorch_rs.acquisition import analytic as rs_an  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.utils import project_to_target_fidelity as rs_project  # noqa: E402
from botorch_rs.acquisition.utils import prune_inferior_points as rs_prune  # noqa: E402
from botorch_rs.kernels import MaternKernel as RsMatern  # noqa: E402
from botorch_rs.models import AffineDeterministicModel as RsAffine  # noqa: E402
from botorch_rs.models import GenericDeterministicModel as RsGeneric  # noqa: E402
from botorch_rs.models import MixedSingleTaskGP as RsMixed  # noqa: E402
from botorch_rs.models import ModelList as RsModelList  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models import SingleTaskMultiFidelityGP as RsMF  # noqa: E402
from botorch_rs.models.deterministic import FixedSingleSampleModel as RsFixedSample  # noqa: E402
from botorch_rs.models.ensemble import EnsembleModel as RsEnsembleModel  # noqa: E402
from botorch_rs.models.transforms import Normalize as RsNormalize  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.sampling.index_sampler import IndexSampler as RsIndexSampler  # noqa: E402
from botorch_rs.utils.datasets import SupervisedDataset as RsDataset  # noqa: E402
from parity_helpers import Checker, rel_err  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

CHECK = Checker("models (7b)")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    yield
    torch.set_num_threads(threads)


def _gen(seed: int) -> Any:
    return torch.Generator().manual_seed(seed)


def _data(n: int, d: int, m: int, seed: int) -> tuple[Any, Any, Any]:
    g = _gen(seed)
    X = torch.rand(n, d, generator=g, dtype=DT)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * X[:, 0]) + (1.0 + t) * (X[:, 1:] ** 2).sum(-1)
        cols.append(1.0 + t + f + 0.05 * torch.randn(n, generator=g, dtype=DT))
    Y = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(n, m, generator=g, dtype=DT)
    return X, Y, Yvar


def _sampler_pair(n: int, bt_shape: tuple[int, ...], seed: int) -> tuple[Any, Any]:
    Z = torch.randn(n, *bt_shape, generator=_gen(seed), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([n]), seed=0)
    s_bt.base_samples = Z.clone()
    s_rs = RsSampler(sample_shape=torch.Size([n]), seed=0)
    s_rs.base_samples = Z.reshape(n, -1).clone()
    return s_bt, s_rs


def _compare_acqf(name: str, a_rs: Any, a_bt: Any, X: Any, grads: bool = True) -> None:
    vr, gr = values_and_grads(a_rs, X)
    vb, gb = values_and_grads(a_bt, X)
    assert tuple(vr.shape) == tuple(vb.shape), (name, vr.shape, vb.shape)
    CHECK.check(f"{name}: values", vr, vb, VALUE_TOL)
    if grads:
        check_grad(CHECK, f"{name}: gradients", gr, gb)


# ---------------------------------------------------------------------------
# SingleTaskMultiFidelityGP
# ---------------------------------------------------------------------------

#: name -> (keyword arguments, d, m, fixed noise, Normalize)
MF_CASES: dict[str, tuple[dict[str, Any], int, int, bool, bool]] = {
    "ltf_data": ({"data_fidelities": [3]}, 4, 1, False, False),
    "ltf_iteration": ({"iteration_fidelity": 2}, 3, 1, False, False),
    "ltf_both": ({"iteration_fidelity": 2, "data_fidelities": [3]}, 4, 1, False, False),
    "ltf_two_data": ({"data_fidelities": [2, 3]}, 4, 1, False, False),
    "ltf_nu15_negative": ({"data_fidelities": [-1], "nu": 1.5}, 3, 1, False, False),
    "ltf_nu05_fixed": ({"iteration_fidelity": 1, "nu": 0.5}, 3, 1, True, False),
    "ltf_m2_normalize": ({"data_fidelities": [3]}, 4, 2, False, True),
    "ds_data": ({"data_fidelities": [3], "linear_truncated": False}, 4, 1, False, False),
    "ds_iteration": ({"iteration_fidelity": 2, "linear_truncated": False}, 3, 1, False, False),
    "ds_both": (
        {"iteration_fidelity": 1, "data_fidelities": [2, 3], "linear_truncated": False},
        4,
        1,
        False,
        False,
    ),
    "ds_m2_fixed": ({"data_fidelities": [2], "linear_truncated": False}, 3, 2, True, False),
}


def mf_pair(name: str, X: Any = None, seed: int = 0) -> tuple[Any, Any, Any]:
    kw, d, m, fixed, normalize = MF_CASES[name]
    X0, Y, Yvar = _data(12, d, m, 700 + seed)
    X = X0 if X is None else X
    bkw: dict[str, Any] = dict(kw)
    rkw: dict[str, Any] = dict(kw)
    if fixed:
        bkw["train_Yvar"] = rkw["train_Yvar"] = Yvar
    if normalize:
        bkw["input_transform"] = BtNormalize(d=d)
        rkw["input_transform"] = RsNormalize(d=d)
    return BtMF(X, Y, **bkw), RsMF(X, Y, **rkw), X


@pytest.mark.parametrize("name", list(MF_CASES))
def test_multi_fidelity_models_match_botorch(name: str) -> None:
    seed = list(MF_CASES).index(name)
    bt, rs, X = mf_pair(name, seed=seed)
    compare_structure(CHECK, bt, rs)
    compare_mll(CHECK, "MF default", bt, rs)
    randomize(bt, rs, 900 + seed)
    compare_mll(CHECK, "MF random", bt, rs)
    Xt = torch.rand(4, X.shape[-1], generator=_gen(seed), dtype=DT)
    compare_posteriors(CHECK, "MF", bt, rs, Xt)


def test_multi_fidelity_custom_covar_module_matches_botorch() -> None:
    X, Y, _ = _data(12, 4, 1, 31)
    bt = BtMF(
        X,
        Y,
        data_fidelities=[3],
        iteration_fidelity=2,
        linear_truncated=False,
        covar_module=BtMatern(nu=1.5, ard_num_dims=2, active_dims=[0, 1]),
    )
    rs = RsMF(
        X,
        Y,
        data_fidelities=[3],
        iteration_fidelity=2,
        linear_truncated=False,
        covar_module=RsMatern(nu=1.5, ard_num_dims=2, active_dims=[0, 1]),
    )
    compare_structure(CHECK, bt, rs)
    randomize(bt, rs, 32)
    compare_mll(CHECK, "MF custom covar", bt, rs)
    compare_posteriors(CHECK, "MF custom covar", bt, rs, torch.rand(3, 4, dtype=DT))


def test_multi_fidelity_empty_data_fidelities_is_none() -> None:
    """BoTorch builds an empty ``ProductKernel`` for ``data_fidelities=[]``
    with an iteration fidelity, whose evaluation raises IndexError;
    botorch_rs treats the empty list like ``None`` (the documented intent),
    equal to BoTorch's model without data fidelities."""
    X, Y, _ = _data(10, 3, 1, 41)
    broken = BtMF(X, Y, iteration_fidelity=2, data_fidelities=[])
    with pytest.raises(IndexError):
        _ = broken.posterior(X[:2]).mean
    rs = RsMF(X, Y, iteration_fidelity=2, data_fidelities=[])
    bt = BtMF(X, Y, iteration_fidelity=2)
    compare_structure(CHECK, bt, rs)
    randomize(bt, rs, 42)
    compare_mll(CHECK, "MF data_fidelities=[]", bt, rs)


@pytest.mark.parametrize("name", ["ltf_data", "ds_both", "ltf_m2_normalize"])
def test_multi_fidelity_fits_match_botorch(name: str) -> None:
    seed = 50 + list(MF_CASES).index(name)
    _, _, X0 = mf_pair(name, seed=seed)

    def make(X: Any) -> tuple[Any, Any]:
        bt, rs, _ = mf_pair(name, X=X, seed=seed)
        return bt, rs

    compare_fits(CHECK, f"MF fit ({name})", make, X0)


@pytest.mark.parametrize("name", ["ltf_m2_normalize", "ds_m2_fixed"])
def test_multi_fidelity_subset_output_matches_botorch(name: str) -> None:
    seed = 60 + list(MF_CASES).index(name)
    bt, rs, X = mf_pair(name, seed=seed)
    randomize(bt, rs, seed)
    Xt = torch.rand(3, X.shape[-1], generator=_gen(seed), dtype=DT)
    for idcs in ([1], [0], [1, 0], [0, 1]):
        sb, sr = bt.subset_output(idcs), rs.subset_output(idcs)
        compare_structure(CHECK, sb, sr)
        compare_posteriors(CHECK, f"subset_output {idcs}", sb, sr, Xt)
        compare_mll(CHECK, f"subset_output {idcs}", sb, sr)


@pytest.mark.parametrize("variant", ["project", "cost", "valfunc_ei"])
@pytest.mark.parametrize("name", ["ltf_both", "ds_both"])
def test_mfkg_on_multi_fidelity_gp_matches_botorch(name: str, variant: str) -> None:
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility as BtICW
    from botorch.models.cost import AffineFidelityCostModel as BtAffineCost

    from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility as RsICW
    from botorch_rs.models.cost import AffineFidelityCostModel as RsAffineCost

    seed = 70 + list(MF_CASES).index(name)
    bt, rs, X = mf_pair(name, seed=seed)
    randomize(bt, rs, seed, scale=0.3)
    d = X.shape[-1]
    q, n_f = 2, 3
    tf = {d - 1: 1.0}
    kb: dict[str, Any] = {"project": functools.partial(bt_project, target_fidelities=tf, d=d)}
    kr: dict[str, Any] = {"project": functools.partial(rs_project, target_fidelities=tf, d=d)}
    if variant == "cost":
        kb["cost_aware_utility"] = BtICW(BtAffineCost(fidelity_weights={d - 1: 0.75}))
        kr["cost_aware_utility"] = RsICW(RsAffineCost(fidelity_weights={d - 1: 0.75}))
        kb["current_value"] = kr["current_value"] = torch.tensor(0.25, dtype=DT)
    if variant == "valfunc_ei":
        best = {"best_f": torch.tensor(1.5, dtype=DT)}
        kb.update(valfunc_cls=bt_an.ExpectedImprovement, valfunc_argfac=lambda model: best)
        kr.update(valfunc_cls=rs_an.ExpectedImprovement, valfunc_argfac=lambda model: best)
    s_bt, s_rs = _sampler_pair(n_f, (1, q), seed)
    a_bt = bt_kg.qMultiFidelityKnowledgeGradient(bt, num_fantasies=n_f, sampler=s_bt, **kb)
    a_rs = rs_kg.qMultiFidelityKnowledgeGradient(rs, num_fantasies=n_f, sampler=s_rs, **kr)
    Xt = torch.rand(2, q + n_f, d, generator=_gen(seed + 1), dtype=DT)
    Xt[..., -1] = 0.3 + 0.7 * Xt[..., -1]
    from so_acquisition_helpers import botorch_has_kg_fix

    exact = variant != "valfunc_ei" or botorch_has_kg_fix()
    _compare_acqf(f"qMFKG on MF GP ({variant})", a_rs, a_bt, Xt, grads=exact)


@pytest.mark.parametrize(
    "cls", ["qMultiFidelityMaxValueEntropy", "qMultiFidelityLowerBoundMaxValueEntropy"]
)
def test_mf_max_value_entropy_on_multi_fidelity_gp_matches_botorch(cls: str) -> None:
    from so_acquisition_helpers import mes_pair

    bt, rs, X = mf_pair("ds_both", seed=81)
    randomize(bt, rs, 81, scale=0.3)
    d = X.shape[-1]
    cs = torch.rand(20, d, generator=_gen(82), dtype=DT)
    tf = {d - 1: 1.0}
    kb = {"project": functools.partial(bt_project, target_fidelities=tf, d=d)}
    kr = {"project": functools.partial(rs_project, target_fidelities=tf, d=d)}
    a_bt, a_rs = mes_pair(cls, bt, rs, cs, None, kb, kr)
    Xt = torch.rand(4, 1, d, generator=_gen(83), dtype=DT)
    Xt[..., -1] = 0.3 + 0.7 * Xt[..., -1]
    _compare_acqf(f"{cls} on MF GP", a_rs, a_bt, Xt)


@pytest.mark.parametrize("variant", ["posterior_mean", "samples"])
def test_mfhvkg_on_multi_fidelity_gps_matches_botorch(variant: str) -> None:
    bts, rss = [], []
    for i, name in enumerate(("ltf_data", "ds_data")):
        bt, rs, X = mf_pair(name, seed=90 + i)
        randomize(bt, rs, 91 + i, scale=0.3)
        bts.append(bt)
        rss.append(rs)
    bt_list, rs_list = BtModelListGP(*bts), RsModelListGP(*rss)
    case = HvkgCase(
        f"MF GPs ({variant})",
        n_f=2,
        num_pareto=2,
        q=1,
        mf=True,
        use_posterior_mean=variant == "posterior_mean",
        inner=8,
    )
    pair, _, ev_rs, ev_bt = run_hvkg_case(case, bt_list, rs_list, X, 95)
    err_v, err_g = pair.errors(ev_rs, ev_bt)
    CHECK.record(f"qMFHVKG on MF GPs ({variant}): values", err_v, VALUE_TOL)
    CHECK.record(f"qMFHVKG on MF GPs ({variant}): gradients", err_g, GRAD_TOL)
    assert err_v <= VALUE_TOL and err_g <= GRAD_TOL, (err_v, err_g)


# ---------------------------------------------------------------------------
# MixedSingleTaskGP
# ---------------------------------------------------------------------------


def _mixed_data(n: int, d_cont: int, cats: list[int], m: int, seed: int) -> tuple[Any, Any, Any]:
    g = _gen(seed)
    cont = torch.rand(n, d_cont, generator=g, dtype=DT)
    cat = torch.stack([torch.randint(0, c, (n,), generator=g) for c in cats], -1).to(DT)
    X = torch.cat([cont, cat], -1)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * X[:, 0]) + (1.0 + t) * cat.sum(-1) / len(cats)
        if d_cont > 1:
            f = f + X[:, 1] ** 2
        cols.append(f + 0.05 * torch.randn(n, generator=g, dtype=DT))
    Y = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(n, m, generator=g, dtype=DT)
    return X, Y, Yvar


#: name -> (continuous dims, categories per categorical dim, cat_dims, m, fixed, factory)
MIXED_CASES: dict[str, tuple[int, list[int], list[int], int, bool, str | None]] = {
    "last": (2, [3], [-1], 1, False, None),
    "two_cats": (2, [3, 2], [2, 3], 1, False, None),
    "all_categorical": (0, [3, 2], [0, 1], 1, False, None),
    "matern_factory": (2, [3], [2], 1, False, "matern"),
    "m2_fixed": (2, [3], [-1], 2, True, None),
}


def mixed_pair(name: str, X: Any = None, seed: int = 0) -> tuple[Any, Any, Any]:
    d_cont, cats, cat_dims, m, fixed, factory = MIXED_CASES[name]
    X0, Y, Yvar = _mixed_data(14, d_cont, cats, m, 1100 + seed)
    X = X0 if X is None else X
    bkw: dict[str, Any] = {}
    rkw: dict[str, Any] = {}
    if fixed:
        bkw["train_Yvar"] = rkw["train_Yvar"] = Yvar
    if factory == "matern":
        bkw["cont_kernel_factory"] = lambda **kw: BtMatern(nu=2.5, **kw)
        rkw["cont_kernel_factory"] = lambda **kw: RsMatern(nu=2.5, **kw)
    return BtMixed(X, Y, cat_dims=cat_dims, **bkw), RsMixed(X, Y, cat_dims=cat_dims, **rkw), X


def _mixed_points(name: str, n: int, seed: int) -> Any:
    d_cont, cats, _, _, _, _ = MIXED_CASES[name]
    return _mixed_data(n, d_cont, cats, 1, seed)[0]


@pytest.mark.parametrize("name", list(MIXED_CASES))
def test_mixed_models_match_botorch(name: str) -> None:
    seed = list(MIXED_CASES).index(name)
    bt, rs, X = mixed_pair(name, seed=seed)
    compare_structure(CHECK, bt, rs)
    compare_mll(CHECK, "mixed default", bt, rs)
    randomize(bt, rs, 1200 + seed)
    compare_mll(CHECK, "mixed random", bt, rs)
    compare_posteriors(CHECK, "mixed", bt, rs, _mixed_points(name, 4, 1300 + seed))


@pytest.mark.parametrize("name", ["last", "all_categorical"])
def test_mixed_fits_match_botorch(name: str) -> None:
    seed = 20 + list(MIXED_CASES).index(name)
    d_cont = MIXED_CASES[name][0]
    _, _, X0 = mixed_pair(name, seed=seed)

    def make(X: Any) -> tuple[Any, Any]:
        bt, rs, _ = mixed_pair(name, X=X, seed=seed)
        return bt, rs

    # Twins perturb the continuous columns only (categories stay integers).
    cols = list(range(d_cont)) if d_cont else None
    if cols is None:
        # All-categorical inputs: no twins (nothing to perturb); one fit per
        # library is compared (stage 1 only).
        compare_fits(CHECK, f"mixed fit ({name})", make, X0, cols=[], twins=0, max_twins=0)
    else:
        compare_fits(CHECK, f"mixed fit ({name})", make, X0, cols=cols)


def test_mixed_acquisition_and_optimize_acqf_mixed_match_botorch() -> None:
    bt, rs, X = mixed_pair("last", seed=40)
    randomize(bt, rs, 41, scale=0.3)
    d = X.shape[-1]
    s_bt, s_rs = _sampler_pair(16, (1, X.shape[0] + 2), 42)
    a_bt = bt_logei.qLogNoisyExpectedImprovement(
        bt, X, sampler=s_bt, prune_baseline=False, cache_root=False
    )
    a_rs = rs_logei.qLogNoisyExpectedImprovement(
        rs, X, cache_root=False, sampler=s_rs, prune_baseline=False
    )
    Xq = _mixed_points("last", 2, 43).unsqueeze(0)
    _compare_acqf("qLogNEI on mixed GP", a_rs, a_bt, Xq)
    # optimize_acqf_mixed over the categories from identical initial conditions.
    e_rs = rs_an.LogExpectedImprovement(rs, best_f=torch.tensor(2.0, dtype=DT))
    e_bt = bt_an.LogExpectedImprovement(bt, best_f=torch.tensor(2.0, dtype=DT))
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=DT)
    ffl = [{d - 1: float(c)} for c in range(3)]
    ics = torch.rand(4, 1, d, generator=_gen(44), dtype=DT)
    rb = bto.optimize_acqf_mixed(e_bt, bounds, 1, 4, ffl, batch_initial_conditions=ics)
    rr = rso.optimize_acqf_mixed(e_rs, bounds, 1, 4, ffl, batch_initial_conditions=ics)
    if botorch_parallel_lbfgsb():
        CHECK.check("optimize_acqf_mixed on mixed GP: candidates", rr[0], rb[0], 1e-7)
        CHECK.check("optimize_acqf_mixed on mixed GP: values", rr[1], rb[1], VALUE_TOL)
    else:  # BoTorch 0.18.1: SciPy's gtol on the serial path (see test_parity_9_optim)
        CHECK.check_bound(
            "optimize_acqf_mixed on mixed GP (0.18.1): shortfall",
            (float(rb[1].max()) - float(rr[1].max())) / max(abs(float(rb[1].max())), 1.0),
            1e-9,
        )
    # optimize_acqf_mixed_alternating with BoTorch's starting points.
    start = torch.rand(4, d, generator=_gen(45), dtype=DT)
    start[:, -1] = torch.tensor([0.0, 1.0, 2.0, 1.0], dtype=DT)
    kw = dict(cat_dims={d - 1: [0.0, 1.0, 2.0]}, num_restarts=4, raw_samples=16, q=1)
    with (
        patched(btm, "generate_starting_points", starting_points(start, e_bt)),
        patched(rsm, "generate_starting_points", starting_points(start, e_bt)),
    ):
        ab = btm.optimize_acqf_mixed_alternating(e_bt, bounds, **kw)
        ar = rsm.optimize_acqf_mixed_alternating(e_rs, bounds, **kw)
    if botorch_parallel_lbfgsb():
        CHECK.check("mixed alternating on mixed GP: candidates", ar[0], ab[0], 1e-7)
        CHECK.check("mixed alternating on mixed GP: values", ar[1], ab[1], VALUE_TOL)
    else:  # BoTorch 0.18.1: SciPy's gtol on the serial path (see test_parity_9_optim)
        CHECK.check_bound(
            "mixed alternating on mixed GP (0.18.1): shortfall",
            (float(ab[1].max()) - float(ar[1].max())) / max(abs(float(ab[1].max())), 1.0),
            1e-9,
        )


# ---------------------------------------------------------------------------
# MAP SAAS
# ---------------------------------------------------------------------------


def saas_pair(kind: str, X: Any = None, seed: int = 0) -> tuple[Any, Any, Any]:
    m = 2 if kind.endswith("_m2") else 1
    X0, Y, Yvar = _data(12, 4, m, 1500 + seed)
    X = X0 if X is None else X
    torch.manual_seed(seed)
    if kind.startswith("map"):
        kw: dict[str, Any] = {}
        if kind == "map_tau":
            kw["tau"] = 0.3
        if kind == "map_tau_tensor":
            kw["tau"] = torch.tensor(0.25, dtype=DT)
        if kind == "map_fixed":
            kw["train_Yvar"] = Yvar
        if kind == "map_normalize":
            kw["input_transform"] = BtNormalize(d=4)
        bt = bt_saas.get_map_saas_model(X, Y, **kw)
        if kind == "map_normalize":
            kw["input_transform"] = RsNormalize(d=4)
        torch.manual_seed(seed)
        rs = rs_saas.get_map_saas_model(X, Y, **kw)
    elif kind.startswith("additive"):
        bt = bt_saas.AdditiveMapSaasSingleTaskGP(X, Y, num_taus=3)
        torch.manual_seed(seed)
        rs = rs_saas.AdditiveMapSaasSingleTaskGP(X, Y, num_taus=3)
    else:
        kw = {"train_Yvar": Yvar} if kind == "ensemble_fixed" else {}
        bt = bt_saas.EnsembleMapSaasSingleTaskGP(X, Y, num_taus=3, **kw)
        torch.manual_seed(seed)
        rs = rs_saas.EnsembleMapSaasSingleTaskGP(X, Y, num_taus=3, **kw)
    return bt, rs, X


SAAS_KINDS = [
    "map",
    "map_tau",
    "map_tau_tensor",
    "map_fixed",
    "map_normalize",
    "map_m2",
    "additive",
    "additive_m2",
    "ensemble",
    "ensemble_fixed",
]


@pytest.mark.parametrize("kind", SAAS_KINDS)
def test_map_saas_models_match_botorch(kind: str) -> None:
    seed = SAAS_KINDS.index(kind)
    bt, rs, X = saas_pair(kind, seed=seed)
    compare_structure(CHECK, bt, rs)
    compare_mll(CHECK, "MAP SAAS default", bt, rs)
    randomize(bt, rs, 1600 + seed)
    compare_mll(CHECK, "MAP SAAS random", bt, rs)
    Xt = torch.rand(3, 4, generator=_gen(seed), dtype=DT)
    if kind.startswith("ensemble"):
        bt.eval()
        with torch.no_grad():
            for noise in (False, True):
                pb, pr = (
                    bt.posterior(Xt, observation_noise=noise),
                    rs.posterior(Xt, observation_noise=noise),
                )
                CHECK.check("ensemble: member posterior mean", pr.mean, pb.mean, VALUE_TOL)
                CHECK.check(
                    "ensemble: member posterior variance", pr.variance, pb.variance, VALUE_TOL
                )
                CHECK.check("ensemble: mixture mean", pr.mixture_mean, pb.mixture_mean, VALUE_TOL)
                CHECK.check(
                    "ensemble: mixture variance",
                    pr.mixture_variance,
                    pb.mixture_variance,
                    VALUE_TOL,
                )
                CHECK.check(
                    "ensemble: mixture covariance",
                    pr.mixture_covariance_matrix,
                    pb.mixture_covariance_matrix.to_dense(),
                    VALUE_TOL,
                )
            q = torch.tensor([0.1, 0.5, 0.9], dtype=DT)
            CHECK.check("ensemble: mixture quantiles", pr.quantile(q), pb.quantile(q), 1e-6)
    else:
        compare_posteriors(CHECK, "MAP SAAS", bt, rs, Xt)


@pytest.mark.parametrize("kind", ["map", "map_tau", "additive", "ensemble"])
def test_map_saas_fits_match_botorch(kind: str) -> None:
    seed = 30 + SAAS_KINDS.index(kind)
    _, _, X0 = saas_pair(kind, seed=seed)

    def make(X: Any) -> tuple[Any, Any]:
        bt, rs, _ = saas_pair(kind, X=X, seed=seed)
        return bt, rs

    compare_fits(CHECK, f"MAP SAAS fit ({kind})", make, X0)


def _ensemble(seed: int) -> tuple[Any, Any, Any]:
    bt, rs, X = saas_pair("ensemble", seed=seed)
    randomize(bt, rs, seed + 1, scale=0.4)
    bt.eval()
    return bt, rs, X


@pytest.mark.parametrize(
    "name",
    [
        "PosteriorMean",
        "LogExpectedImprovement",
        "ExpectedImprovement",
        "ProbabilityOfImprovement",
        "LogProbabilityOfImprovement",
        "UpperConfidenceBound",
    ],
)
def test_analytic_acquisition_on_ensemble_matches_botorch(name: str) -> None:
    bt, rs, X = _ensemble(70)
    best = torch.tensor(float(bt.train_targets.max()) * 0 + 2.5, dtype=DT)
    kw: dict[str, Any] = {}
    if name == "UpperConfidenceBound":
        kw["beta"] = torch.tensor(0.4, dtype=DT)
    elif name != "PosteriorMean":
        kw["best_f"] = best
    a_bt = getattr(bt_an, name)(bt, **kw)
    a_rs = getattr(rs_an, name)(rs, **kw)
    Xt = torch.rand(5, 1, 4, generator=_gen(71), dtype=DT)
    _compare_acqf(f"{name} on ensemble", a_rs, a_bt, Xt)


def test_posterior_standard_deviation_on_ensemble_is_the_member_average() -> None:
    """BoTorch's ``PosteriorStandardDeviation`` fails on ensemble models
    (``std.view(X.shape[:-2])`` of the ``b x num_models`` values inside
    ``average_over_ensemble_models``); botorch_rs averages the members'
    values like every other analytic acquisition function, compared with
    the average of BoTorch's values on the members."""
    bt, rs, X = _ensemble(72)
    Xt = torch.rand(5, 1, 4, generator=_gen(73), dtype=DT)
    with pytest.raises(RuntimeError):
        bt_an.PosteriorStandardDeviation(bt)(Xt)
    a_rs = rs_an.PosteriorStandardDeviation(rs)
    vr, gr = values_and_grads(a_rs, Xt)
    with torch.no_grad():
        post = bt.posterior(Xt)
    Xg = Xt.clone().requires_grad_(True)
    ref = bt.posterior(Xg).variance.sqrt().squeeze(-1).squeeze(-1).mean(-1)
    (gref,) = torch.autograd.grad(ref.sum(), Xg)
    CHECK.check("PosteriorStandardDeviation on ensemble (reference)", vr, ref.detach(), VALUE_TOL)
    check_grad(CHECK, "PosteriorStandardDeviation on ensemble (reference) gradients", gr, gref)
    del post


@pytest.mark.parametrize(
    "name",
    [
        "qLogExpectedImprovement",
        "qExpectedImprovement",
        "qProbabilityOfImprovement",
        "qSimpleRegret",
        "qUpperConfidenceBound",
        "qLogNoisyExpectedImprovement",
        "qNoisyExpectedImprovement",
    ],
)
def test_mc_acquisition_on_ensemble_matches_botorch(name: str) -> None:
    bt, rs, X = _ensemble(74)
    q = 2
    noisy = "Noisy" in name
    k = X.shape[0] + q if noisy else q
    # BoTorch's base samples of the ensemble posterior: batch dimensions
    # (t-batch, ensemble) collapsed.
    Z = torch.randn(32, 1, 1, k, generator=_gen(75), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([32]), seed=0)
    s_bt.base_samples = Z.clone()
    s_rs = RsSampler(sample_shape=torch.Size([32]), seed=0)
    s_rs.base_samples = Z.reshape(32, -1).clone()
    kw: dict[str, Any] = {}
    if name in ("qLogExpectedImprovement", "qExpectedImprovement", "qProbabilityOfImprovement"):
        kw["best_f"] = torch.tensor(2.5, dtype=DT)
    if name == "qProbabilityOfImprovement":
        kw["tau"] = torch.tensor(0.05, dtype=DT)
    if name == "qUpperConfidenceBound":
        kw["beta"] = torch.tensor(0.4, dtype=DT)
    module = bt_logei if name.startswith("qLog") else bt_mc
    rmodule = rs_logei if name.startswith("qLog") else rs_mc
    bkw, rkw = dict(kw), dict(kw)
    if noisy:
        bkw.update(X_baseline=X, prune_baseline=False, cache_root=False)
        rkw.update(X_baseline=X, prune_baseline=False, cache_root=False)
    a_bt = getattr(module, name)(bt, sampler=s_bt, **bkw)
    a_rs = getattr(rmodule, name)(rs, sampler=s_rs, **rkw)
    Xq = torch.rand(1 if noisy else 3, q, 4, generator=_gen(76), dtype=DT)
    _compare_acqf(f"{name} on ensemble", a_rs, a_bt, Xq)


def test_prune_inferior_points_on_ensemble_matches_botorch() -> None:
    bt, rs, X = _ensemble(77)
    n = X.shape[0]
    Z = torch.randn(256, 1, 1, n, generator=_gen(78), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([256]), seed=0)
    s_bt.base_samples = Z.clone()
    s_rs = RsSampler(sample_shape=torch.Size([256]), seed=0)
    s_rs.base_samples = Z.reshape(256, -1).clone()
    pb = bt_prune(bt, X, sampler=s_bt)
    pr = rs_prune(rs, X, sampler=s_rs)
    CHECK.check("prune_inferior_points on ensemble: kept points", pr, pb, 0.0)


# ---------------------------------------------------------------------------
# Deterministic models, ModelList, EnsembleModel
# ---------------------------------------------------------------------------


def test_affine_deterministic_model_matches_botorch() -> None:
    a = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75]], dtype=DT)
    for b in (0.01, torch.tensor([0.3, -0.2], dtype=DT)):
        mb, mr = BtAffine(a, b), RsAffine(a, b)
        X = torch.rand(2, 3, 3, generator=_gen(1), dtype=DT)
        CHECK.check(
            "AffineDeterministicModel: posterior mean",
            mr.posterior(X).mean,
            mb.posterior(X).mean,
            1e-15,
        )
        sb, sr = mb.subset_output([1]), mr.subset_output([1])
        CHECK.check(
            "AffineDeterministicModel: subset", sr.posterior(X).mean, sb.posterior(X).mean, 1e-15
        )
        s_bt, s_rs = BtIndexSampler(torch.Size([4]), seed=0), RsSampler((4,), seed=0)
        from botorch.acquisition.objective import LinearMCObjective as BtLin

        from botorch_rs.acquisition.objective import LinearMCObjective as RsLin

        w = torch.tensor([0.5, -1.5], dtype=DT)
        _compare_acqf(
            "qSimpleRegret on AffineDeterministicModel",
            rs_mc.qSimpleRegret(mr, sampler=s_rs, objective=RsLin(w)),
            bt_mc.qSimpleRegret(mb, sampler=s_bt, objective=BtLin(w)),
            X,
        )


@pytest.mark.parametrize("m", [1, 2])
def test_fixed_single_sample_model_matches_botorch(m: int) -> None:
    X, Y, _ = _data(10, 3, m, 3)
    gb, gr = BtSingleTaskGP(X, Y), RsSingleTaskGP(X, Y)
    randomize(gb, gr, 4)
    gb.eval()
    torch.manual_seed(5)
    fb = BtFixedSample(gb)
    torch.manual_seed(5)
    fr = RsFixedSample(gr)
    CHECK.check("FixedSingleSampleModel: w (torch.randn)", fr.w, fb.w, 0.0)
    Xt = torch.rand(2, 4, 3, generator=_gen(6), dtype=DT)
    with torch.no_grad():
        CHECK.check(
            "FixedSingleSampleModel: values",
            fr.posterior(Xt).mean,
            fb.posterior(Xt).mean,
            VALUE_TOL,
        )
    s_bt, s_rs = BtIndexSampler(torch.Size([4]), seed=0), RsSampler((4,), seed=0)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if m > 1:
        from botorch.acquisition.objective import LinearMCObjective as BtLin

        from botorch_rs.acquisition.objective import LinearMCObjective as RsLin

        w = torch.tensor([0.5, -1.5], dtype=DT)
        kb["objective"], kr["objective"] = BtLin(w), RsLin(w)
    _compare_acqf(
        "qSimpleRegret on FixedSingleSampleModel",
        rs_mc.qSimpleRegret(fr, sampler=s_rs, **kr),
        bt_mc.qSimpleRegret(fb, sampler=s_bt, **kb),
        Xt,
    )
    if m == 1:
        _compare_acqf(
            "PosteriorMean on FixedSingleSampleModel",
            rs_an.PosteriorMean(fr),
            bt_an.PosteriorMean(fb),
            Xt[:, :1],
        )


def test_model_list_matches_botorch() -> None:
    X, Y, _ = _data(10, 3, 2, 8)
    g1b, g1r = BtSingleTaskGP(X, Y[:, :1]), RsSingleTaskGP(X, Y[:, :1])
    g2b, g2r = BtSingleTaskGP(X, Y), RsSingleTaskGP(X, Y)
    randomize(g1b, g1r, 9)
    randomize(g2b, g2r, 10)

    def f(x: Any) -> Any:
        return (x**2).sum(-1, keepdim=True)

    lb = BtModelList(g1b, g2b, BtGeneric(f))
    lr = RsModelList(g1r, g2r, RsGeneric(f, dim=3))
    assert lr.num_outputs == lb.num_outputs == 4
    lb.eval()
    Xt = torch.rand(2, 3, 3, generator=_gen(11), dtype=DT)
    with torch.no_grad():
        pb, pr = lb.posterior(Xt), lr.posterior(Xt)
        CHECK.check("ModelList: posterior mean", pr.mean, pb.mean, VALUE_TOL)
        CHECK.check("ModelList: posterior variance", pr.variance, pb.variance, VALUE_TOL)
        # (BoTorch main fails for a strict subset of the outputs of a
        # multi-output SingleTaskGP; such subsets are compared in
        # test_multi_fidelity_subset_output_matches_botorch.)
        for idcs in ([0, 3], [1, 2], [3]):
            pb, pr = lb.posterior(Xt, output_indices=idcs), lr.posterior(Xt, output_indices=idcs)
            CHECK.check("ModelList: output_indices mean", pr.mean, pb.mean, VALUE_TOL)
    sb, sr = lb.subset_output([0, 3]), lr.subset_output([0, 3])
    assert type(sr).__name__ == type(sb).__name__ == "ModelList"
    single_b, single_r = lb.subset_output([2]), lr.subset_output([2])
    with torch.no_grad():
        CHECK.check(
            "ModelList: subset_output to one sub-model",
            single_r.posterior(Xt).mean,
            single_b.posterior(Xt).mean,
            VALUE_TOL,
        )
    # An MC acquisition function on the list (GP outputs and a deterministic one).
    from botorch.acquisition.objective import LinearMCObjective as BtLin
    from botorch.sampling.list_sampler import ListSampler as BtListSampler

    from botorch_rs.acquisition.objective import LinearMCObjective as RsLin

    w = torch.tensor([1.0, -0.5], dtype=DT)
    # BoTorch samples the PosteriorList with a ListSampler (Sobol base
    # samples for the GP, an IndexSampler for the deterministic model);
    # botorch_rs samples the block-diagonal joint posterior with the same
    # base samples for the GP's output (index ``p * m + t``).
    Z = torch.randn(8, 1, 3, generator=_gen(12), dtype=DT)
    s_gp = BtSampler(sample_shape=torch.Size([8]), seed=0)
    s_gp.base_samples = Z.clone()
    s_bt = BtListSampler(s_gp, BtIndexSampler(torch.Size([8]), seed=0))
    a_bt = bt_mc.qSimpleRegret(BtModelList(g1b, BtGeneric(f)), sampler=s_bt, objective=BtLin(w))
    # (The deterministic output ignores its base samples, which are random
    # here: its zero covariance block has a zero root, not a jittered one.)
    z_rs = torch.randn(8, 3, 2, generator=_gen(13), dtype=DT)
    z_rs[..., 0] = Z.reshape(8, 3)
    s_rs = RsSampler(sample_shape=torch.Size([8]), seed=0)
    s_rs.base_samples = z_rs.reshape(8, -1)
    a_rs = rs_mc.qSimpleRegret(
        RsModelList(g1r, RsGeneric(f, dim=3)), sampler=s_rs, objective=RsLin(w)
    )
    _compare_acqf("qSimpleRegret on ModelList(GP, deterministic)", a_rs, a_bt, Xt)


class _BtEnsemble(BtEnsembleModel):
    def __init__(self, W: Any, weights: Any = None) -> None:
        super().__init__(weights=weights)
        self.W = W
        self._num_outputs = W.shape[-1]

    def forward(self, X: Any) -> Any:
        # ``batch x s x n x m``: member k is sin(X @ W[k]).
        return torch.sin(torch.einsum("...nd,sdm->...snm", X, self.W))


class _RsEnsemble(RsEnsembleModel):
    def __init__(self, W: Any, weights: Any = None) -> None:
        super().__init__(weights=weights)
        self.W = W
        self._num_outputs = W.shape[-1]
        self.dim = W.shape[-2]

    def forward(self, X: Any) -> Any:
        return torch.sin(torch.einsum("...nd,sdm->...snm", X, self.W))


@pytest.mark.parametrize("weighted", [False, True])
def test_ensemble_model_matches_botorch(weighted: bool) -> None:
    W = torch.randn(4, 3, 1, generator=_gen(13), dtype=DT)
    weights = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=DT) if weighted else None
    mb, mr = _BtEnsemble(W, weights), _RsEnsemble(W, weights)
    Xt = torch.rand(2, 3, 3, generator=_gen(14), dtype=DT)
    pb, pr = mb.posterior(Xt), mr.posterior(Xt)
    for attr in (
        "mean",
        "variance",
        "mixture_mean",
        "mixture_variance",
        "weights",
        "mixture_weights",
    ):
        CHECK.check(f"EnsemblePosterior.{attr}", getattr(pr, attr), getattr(pb, attr), 1e-14)
    idx = torch.randint(0, 4, (5, 2), generator=_gen(15))
    CHECK.check(
        "EnsemblePosterior.rsample_from_base_samples",
        pr.rsample_from_base_samples(torch.Size([5]), idx),
        pb.rsample_from_base_samples(torch.Size([5]), idx),
        0.0,
    )
    # IndexSampler: BoTorch's draws (same seed) and MC acquisition values.
    sb, sr = BtIndexSampler(sample_shape=torch.Size([64]), seed=3), RsIndexSampler((64,), seed=3)
    X1 = torch.rand(1, 2, 3, generator=_gen(16), dtype=DT)
    CHECK.check("IndexSampler samples (same seed)", sr(mr.posterior(X1)), sb(mb.posterior(X1)), 0.0)
    for name, kw in (
        ("qSimpleRegret", {}),
        ("qExpectedImprovement", {"best_f": torch.tensor(0.2, dtype=DT)}),
        ("qLogExpectedImprovement", {"best_f": torch.tensor(0.2, dtype=DT)}),
    ):
        module = bt_logei if name.startswith("qLog") else bt_mc
        rmodule = rs_logei if name.startswith("qLog") else rs_mc
        a_bt = getattr(module, name)(mb, sampler=BtIndexSampler(torch.Size([64]), seed=3), **kw)
        a_rs = getattr(rmodule, name)(mr, sampler=RsIndexSampler((64,), seed=3), **kw)
        _compare_acqf(f"{name} on EnsembleModel (IndexSampler)", a_rs, a_bt, X1)


@pytest.mark.parametrize("members", ["distinct", "weighted", "identical"])
def test_analytic_acquisition_on_ensemble_model_matches_botorch(members: str) -> None:
    """Analytic acquisition functions of an ``EnsembleModel`` use the mean and
    the unbiased weighted variance of its ``EnsemblePosterior`` (not averaged
    over the members); identical members have zero variance (``sigma`` is
    clamped at ``1e-6`` by ``_mean_and_sigma`` only). There, ``|u|`` is up
    to about 1e6 and LogEI's gradient is compared with an exact reference
    (mpmath, 60 digits) instead of BoTorch's autograd, which loses accuracy
    (relative error about ``eps * u**2``; see ``log_ei_helper_grad``)."""
    W = torch.randn(4, 3, 2, generator=_gen(18), dtype=DT)
    if members == "identical":
        W = W[:1].expand(4, 3, 2).clone()
    weights = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=DT) if members == "weighted" else None
    mb1, mr1 = _BtEnsemble(W[..., :1], weights), _RsEnsemble(W[..., :1], weights)
    mb2, mr2 = _BtEnsemble(W, weights), _RsEnsemble(W, weights)
    X = torch.rand(6, 1, 3, generator=_gen(19), dtype=DT)
    best = torch.tensor(0.3, dtype=DT)
    cons = {1: (None, torch.tensor(0.1, dtype=DT))}
    exact_log_ei_grad = None
    if members == "identical":
        mpmath = pytest.importorskip("mpmath")
        mpmath.mp.dps = 60
        Xg = X.clone().requires_grad_(True)
        mu = mb1.posterior(Xg).mean.reshape(-1)
        (dmu,) = torch.autograd.grad(mu.sum(), Xg)
        sigma = mpmath.sqrt(mpmath.mpf(1e-12))
        exact = []
        for i in range(X.shape[0]):
            u = (mpmath.mpf(float(mu[i])) - mpmath.mpf(float(best))) / sigma
            ei = mpmath.npdf(u) + u * mpmath.ncdf(u)
            exact.append(float(mpmath.ncdf(u) / ei / sigma) * dmu[i])
        a_bt = bt_an.LogExpectedImprovement(mb1, best_f=best)
        a_rs = rs_an.LogExpectedImprovement(mr1, best_f=best)
        vr, gr = values_and_grads(a_rs, X)
        vb, gb = values_and_grads(a_bt, X)
        CHECK.check(
            "LogExpectedImprovement on EnsembleModel (identical): values", vr, vb, VALUE_TOL
        )
        CHECK.record(
            "LogEI on EnsembleModel (identical): BoTorch's gradient error vs exact",
            rel_err(gb, torch.stack(exact)),
            None,
        )
        exact_log_ei_grad = torch.stack(exact)
        CHECK.check(
            "LogExpectedImprovement on EnsembleModel (identical): gradient vs exact",
            gr,
            exact_log_ei_grad,
            1e-10,
        )
    for name, kw in (
        ("ExpectedImprovement", {"best_f": best}),
        ("LogExpectedImprovement", {"best_f": best}),
        ("ProbabilityOfImprovement", {"best_f": best, "maximize": False}),
        ("LogProbabilityOfImprovement", {"best_f": best}),
        ("UpperConfidenceBound", {"beta": torch.tensor(0.7, dtype=DT)}),
        ("PosteriorMean", {}),
        ("PosteriorStandardDeviation", {}),
    ):
        if members == "identical" and name == "LogExpectedImprovement":
            continue
        a_bt = getattr(bt_an, name)(mb1, **kw)
        a_rs = getattr(rs_an, name)(mr1, **kw)
        _compare_acqf(f"{name} on EnsembleModel ({members})", a_rs, a_bt, X)
    for name, kw in (
        ("ConstrainedExpectedImprovement", {"best_f": best, "objective_index": 0}),
        ("LogProbabilityOfFeasibility", {}),
        ("LogConstrainedExpectedImprovement", {"best_f": best, "objective_index": 0}),
    ):
        a_bt = getattr(bt_an, name)(mb2, constraints=cons, **kw)
        a_rs = getattr(rs_an, name)(mr2, constraints=cons, **kw)
        label = f"{name} on EnsembleModel ({members})"
        if exact_log_ei_grad is None or name != "LogConstrainedExpectedImprovement":
            _compare_acqf(label, a_rs, a_bt, X)
            continue
        # log EI of the objective (exact gradient, see above) + log PoF.
        _, g_pof = values_and_grads(rs_an.LogProbabilityOfFeasibility(mr2, constraints=cons), X)
        vr, gr = values_and_grads(a_rs, X)
        CHECK.check(f"{label}: values", vr, values_and_grads(a_bt, X)[0], VALUE_TOL)
        CHECK.check(f"{label}: gradient vs exact", gr, exact_log_ei_grad + g_pof, 1e-10)


# ---------------------------------------------------------------------------
# condition_on_observations and fantasize
# ---------------------------------------------------------------------------

COND_CASES = [
    "mf:ltf_both",
    "mf:ds_both",
    "mf:ltf_m2_normalize",
    "mixed:last",
    "mixed:m2_fixed",
    "saas:map",
    "saas:additive",
    "saas:ensemble",
]


@pytest.mark.parametrize("case", COND_CASES)
def test_conditioning_and_fantasies_match_botorch(case: str) -> None:
    """``condition_on_observations`` and ``fantasize`` (BoTorch's base
    samples) of the models of phase 7b: posteriors of the conditioned and
    fantasy models. BoTorch's ``EnsembleMapSaasSingleTaskGP.fantasize``
    fails for every shape of ``X`` (its posterior adds the ensemble dimension
    in front of ``X``'s batch, see the draft's BoTorch bugs); the reference is
    BoTorch's ``fantasize`` with the posterior of the ``SingleTaskGP`` base
    class (the members' posteriors, a ``num_taus x n' x d`` batch of
    ``X``)."""
    family, name = case.split(":")
    seed = COND_CASES.index(case)
    g = _gen(1600 + seed)
    if family == "mf":
        bt, rs, X = mf_pair(name, seed=seed)
        Xn = torch.rand(2, X.shape[-1], generator=g, dtype=DT)
        Xt = torch.rand(3, X.shape[-1], generator=g, dtype=DT)
        fixed = MF_CASES[name][3]
    elif family == "mixed":
        bt, rs, X = mixed_pair(name, seed=seed)
        Xn, Xt = _mixed_points(name, 2, 1610 + seed), _mixed_points(name, 3, 1620 + seed)
        fixed = MIXED_CASES[name][4]
    else:
        bt, rs, X = saas_pair(name, seed=seed)
        Xn = torch.rand(2, X.shape[-1], generator=g, dtype=DT)
        Xt = torch.rand(3, X.shape[-1], generator=g, dtype=DT)
        fixed = False
    randomize(bt, rs, 1630 + seed, scale=0.3)
    m = rs.num_outputs
    bt.eval()
    rs.eval()
    with torch.no_grad():
        bt.posterior(Xt)
        rs.posterior(Xt)
    Yn = torch.randn(2, m, generator=g, dtype=DT)
    kw = {"noise": torch.full((2, m), 0.02, dtype=DT)} if fixed else {}
    cb = bt.condition_on_observations(Xn, Yn, **kw)
    cr = rs.condition_on_observations(Xn, Yn, **kw)
    sb = BtSampler(sample_shape=torch.Size([3]), seed=1640 + seed)
    if rs.batch_shape:
        with patched(bt_saas.EnsembleMapSaasSingleTaskGP, "posterior", BtSingleTaskGP.posterior):
            fb = bt.fantasize(Xn.expand(*rs.batch_shape, *Xn.shape), sb)
    else:
        fb = bt.fantasize(Xn, sb)
    sr = RsSampler(sample_shape=torch.Size([3]), seed=0)
    sr.base_samples = sb.base_samples.clone()
    fr = rs.fantasize(Xn, sr)
    with torch.no_grad():
        for label, mb, mr in (("conditioned", cb, cr), ("fantasy", fb, fr)):
            pb, pr = mb.posterior(Xt), mr.posterior(Xt)
            assert tuple(pr.mean.shape) == tuple(pb.mean.shape), (label, pr.mean.shape)
            CHECK.check(f"{family}: {label} posterior mean", pr.mean, pb.mean, VALUE_TOL)
            CHECK.check(
                f"{family}: {label} posterior variance", pr.variance, pb.variance, VALUE_TOL
            )


# ---------------------------------------------------------------------------
# construct_inputs
# ---------------------------------------------------------------------------


def _same_inputs(label: str, rs: dict[str, Any], bt: dict[str, Any]) -> None:
    assert set(rs) == set(bt), (label, set(rs), set(bt))
    for key, value in bt.items():
        if isinstance(value, torch.Tensor):
            CHECK.check(f"construct_inputs ({label}): {key}", rs[key], value, 0.0)
        else:
            assert rs[key] == value, (label, key, rs[key], value)


def test_construct_inputs_match_botorch() -> None:
    X, Y, Yvar = _data(6, 3, 1, 17)
    names = {"feature_names": ["a", "b", "c"], "outcome_names": ["y"]}
    for yvar in (None, Yvar):
        db = BtDataset(X, Y, Yvar=yvar, **names)
        dr = RsDataset(X, Y, Yvar=yvar, **names)
        _same_inputs(
            "SingleTaskGP", RsSingleTaskGP.construct_inputs(dr), BtSingleTaskGP.construct_inputs(db)
        )
        _same_inputs(
            "SingleTaskMultiFidelityGP",
            RsMF.construct_inputs(dr, fidelity_features=[2]),
            BtMF.construct_inputs(db, fidelity_features=[2]),
        )
        _same_inputs(
            "MixedSingleTaskGP",
            RsMixed.construct_inputs(dr, categorical_features=[2]),
            BtMixed.construct_inputs(db, categorical_features=[2]),
        )
        _same_inputs(
            "EnsembleMapSaasSingleTaskGP",
            rs_saas.EnsembleMapSaasSingleTaskGP.construct_inputs(dr, num_taus=3),
            bt_saas.EnsembleMapSaasSingleTaskGP.construct_inputs(db, num_taus=3),
        )
        # BoTorch's dataset is accepted by botorch_rs (interoperability).
        _same_inputs(
            "BoTorch dataset",
            RsSingleTaskGP.construct_inputs(db),
            BtSingleTaskGP.construct_inputs(db),
        )
    m = RsMF(**RsMF.construct_inputs(RsDataset(X, Y, **names), fidelity_features=[2]))
    assert m.num_outputs == 1
    assert math.isfinite(float(rel_err(torch.zeros(1), torch.zeros(1))))


# ---------------------------------------------------------------------------
# OrthogonalAdditiveGP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("second_order", [False, True])
def test_orthogonal_additive_gp_matches_botorch(second_order: bool) -> None:
    from botorch.models.additive_gp import OrthogonalAdditiveGP as BtOAG

    from botorch_rs.models.additive_gp import OrthogonalAdditiveGP as RsOAG

    X, Y, _ = _data(10, 3, 1, 18)
    bt = BtOAG(X, Y, second_order=second_order)
    rs = RsOAG(X, Y, second_order=second_order)
    compare_structure(CHECK, bt, rs)
    compare_mll(CHECK, "OrthogonalAdditiveGP default", bt, rs)
    randomize(bt, rs, 19, scale=0.3)
    compare_mll(CHECK, "OrthogonalAdditiveGP random", bt, rs)
    compare_posteriors(CHECK, "OrthogonalAdditiveGP", bt, rs, torch.rand(3, 3, dtype=DT))
