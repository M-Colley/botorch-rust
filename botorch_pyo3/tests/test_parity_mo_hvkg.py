"""Parity of fantasies, cost models and the hypervolume knowledge gradient
with BoTorch, through the public APIs of both libraries side by side:

* ``SingleTaskGP.condition_on_observations`` (unbatched, batched targets,
  batched inputs; inferred and fixed noise, ``Normalize`` /
  ``Standardize``, one and two outputs) and ``SingleTaskGP.fantasize``
  (unbatched and batched, explicit observation noise) with BoTorch's base
  samples: posterior means, variances and joint covariances (with and
  without observation noise) of the conditioned / fantasy models;
* ``ModelListGP.fantasize`` (``ListSampler``, evaluation masks, a shared
  sampler) and ``ModelListGP.condition_on_observations``;
* qHypervolumeKnowledgeGradient and qMultiFidelityHypervolumeKnowledgeGradient
  (posterior mean and posterior samples, ``log``, ``current_value``, pending
  points, decoupled evaluation masks, objectives, cost-aware utilities,
  target-fidelity projections, ``q = 0``): values and gradients with
  BoTorch's base samples, and t-batches;
* the value function (``_get_hv_value_function``) and
  ``_split_hvkg_fantasy_points``;
* the cost models (``AffineFidelityCostModel``, ``FixedCostModel``), the
  cost-aware utilities (``InverseCostWeightedUtility`` with deterministic
  and GP cost models, ``log``, evaluation masks; ``GenericCostAwareUtility``)
  and ``project_to_target_fidelity`` / ``expand_trace_observations``;
* ``optimize_acqf`` of qHVKG from identical initial conditions and, end to
  end, with each library's ``gen_one_shot_hvkg_initial_conditions``;
* error types and messages.

Version differences. The posterior mean of a model conditioned with
explicit noise on a homoskedastic likelihood follows
meta-pytorch/botorch#3396, and ``ModelListGP.condition_on_observations``
of sub-models with transforms #3399: BoTorch versions without #3396 are
compared after applying the fix to the conditioned model
(``hvkg_helpers.use_fantasy_mean_cache``); lists with transforms are
compared only with #3399.

Deviations (see docs/drafts/mo-hvkg.md). botorch_rs evaluates every t-batch
on its own fantasy designs (BoTorch's ``_split_hvkg_fantasy_points`` mixes
the designs of different t-batches): t-batched values are compared with
BoTorch's values of the single t-batches. With posterior samples
(``use_posterior_mean=False``) BoTorch's default gradients miss the
dependence of the fantasy models' covariances on the candidates (GPyTorch's
``detach_test_caches``); the reference gradients are BoTorch's with the
caches attached.

Tolerances: values 1e-9 (relative; log values absolute below 1),
gradients 1e-8 (relative to the largest component), posterior moments 1e-9,
deterministic quantities (costs, projections, splits) 1e-15 or identical.
Skipped unless BoTorch is importable (required in CI's parity job).
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import gpytorch  # noqa: E402
from botorch.acquisition import cost_aware as bt_cost_aware  # noqa: E402
from botorch.acquisition import utils as bt_acq_utils  # noqa: E402
from botorch.acquisition.multi_objective import (  # noqa: E402
    hypervolume_knowledge_gradient as bt_hvkg,
)
from botorch.models import cost as bt_cost  # noqa: E402
from botorch.optim import initializers as bt_init  # noqa: E402
from botorch.optim import optimize as bt_optimize  # noqa: E402
from botorch.sampling.list_sampler import ListSampler as BtListSampler  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from hvkg_helpers import (  # noqa: E402
    HvkgCase,
    HvkgPair,
    botorch_has_fantasy_mean_cache_fix,
    botorch_has_model_list_conditioning_fix,
    points,
    posterior_moments,
    ref_point,
    run_hvkg_case,
    use_fantasy_mean_cache,
    values_and_grads,
)
from mo_acquisition_helpers import DT, objective_pair  # noqa: E402

from botorch_rs.acquisition import cost_aware as rs_cost_aware  # noqa: E402
from botorch_rs.acquisition import utils as rs_acq_utils  # noqa: E402
from botorch_rs.acquisition.multi_objective import (  # noqa: E402
    hypervolume_knowledge_gradient as rs_hvkg,
)
from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.models import cost as rs_cost  # noqa: E402
from botorch_rs.optim import initializers as rs_init  # noqa: E402
from botorch_rs.optim import optimize as rs_optimize  # noqa: E402
from botorch_rs.sampling import ListSampler as RsListSampler  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker, ModelSpec, list_pair, model_pair  # noqa: E402

CHECK = Checker("hypervolume knowledge gradient")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
MOMENT_TOL = 1e-9

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def check_moments(label: str, rs_model: Any, bt_model: Any, X: Any) -> None:
    """Posterior means, variances and covariances of both models at ``X``
    (with and without observation noise)."""
    for noise in (False, True):
        tag = f"{label}{' (observation noise)' if noise else ''}"
        m_rs, v_rs, c_rs = posterior_moments(rs_model, X, noise)
        m_bt, v_bt, c_bt = posterior_moments(bt_model, X, noise)
        CHECK.check(f"{tag}: mean", m_rs, m_bt, MOMENT_TOL)
        CHECK.check(f"{tag}: variance", v_rs, v_bt, MOMENT_TOL)
        CHECK.check(f"{tag}: covariance", c_rs, c_bt, MOMENT_TOL)


def same_error(fn_bt: Any, fn_rs: Any) -> None:
    """Both calls raise the same error type (by name, or a botorch_rs
    subclass of BoTorch's built-in type) with the same message."""
    with pytest.raises(Exception) as e_bt:
        fn_bt()
    with pytest.raises(Exception) as e_rs:
        fn_rs()
    a, e = e_rs.value, e_bt.value
    assert type(a).__name__ == type(e).__name__ or isinstance(a, type(e)), (a, e)
    assert str(a) == str(e)


# ---------------------------------------------------------------------------
# Conditioning and fantasies of SingleTaskGP
# ---------------------------------------------------------------------------

SINGLE = {
    "m1_norm": ModelSpec(n=8, d=2, normalize=True),
    "m1_fixed": ModelSpec(n=7, d=2, fixed_noise=True, normalize=True),
    "m2_matern": ModelSpec(n=9, d=3, m=2, kernel="matern52", normalize=True, lo=-1.0, hi=2.0),
    "m2_plain": ModelSpec(n=8, d=2, m=2, standardize=False),
    "m2_fixed": ModelSpec(n=8, d=2, m=2, fixed_noise=True),
}


@pytest.fixture(scope="module", params=list(SINGLE), ids=list(SINGLE))
def single(request: pytest.FixtureRequest) -> tuple[str, Any, Any, Any]:
    spec = SINGLE[request.param]
    bt, rs, X = model_pair(spec, 40 + list(SINGLE).index(request.param))
    return request.param, bt, rs, X


def test_condition_on_observations(single: tuple) -> None:
    label, bt, rs, X = single
    m, d = rs.num_outputs, X.shape[-1]
    fixed = "fixed" in label
    Xt = points(X, (4, d), 1)
    bt.posterior(Xt)  # GPyTorch needs the prediction caches
    rs.eval()
    g = torch.Generator().manual_seed(2)
    for x_shape, y_shape in [((3,), (3,)), ((3,), (2, 3)), ((2, 3), (2, 3))]:
        if fixed and len(y_shape) > len(x_shape):
            # GPyTorch fails to concatenate batched noise of unbatched inputs.
            continue
        Xn = points(X, (*x_shape, d), 3 + len(y_shape) + len(x_shape))
        Yn = 2.0 + torch.randn(*y_shape, m, generator=g, dtype=DT)
        kw: dict[str, Any] = {}
        if fixed:
            kw["noise"] = (0.01 + 0.05 * torch.rand(*y_shape, m, generator=g, dtype=DT)) / 4
        cm_bt = bt.condition_on_observations(X=Xn, Y=Yn, **kw)
        cm_rs = rs.condition_on_observations(X=Xn, Y=Yn, **kw)
        check_moments("condition_on_observations", cm_rs, cm_bt, Xt)
    if not fixed:
        # Explicit noise of a homoskedastic model (meta-pytorch/botorch#3396).
        Xn = points(X, (3, d), 21)
        Yn = 2.0 + torch.randn(3, m, generator=g, dtype=DT)
        noise = torch.tensor([1e-6, 0.3, 0.05], dtype=DT).unsqueeze(-1).expand(3, m)
        cm_bt = bt.condition_on_observations(X=Xn, Y=Yn, noise=noise)
        if not botorch_has_fantasy_mean_cache_fix():
            use_fantasy_mean_cache(cm_bt)
        cm_rs = rs.condition_on_observations(X=Xn, Y=Yn, noise=noise)
        check_moments("condition_on_observations, explicit noise (#3396)", cm_rs, cm_bt, Xt)


def test_fantasize(single: tuple) -> None:
    label, bt, rs, X = single
    m, d = rs.num_outputs, X.shape[-1]
    Xt = points(X, (4, d), 1)
    bt.posterior(Xt)
    cases: list[tuple[str, tuple[int, ...], int, Any]] = [
        ("fantasize", (2,), 3, None),
        ("fantasize, batched", (2, 2), 4, None),
        ("fantasize, one point", (1,), 2, None),
        ("fantasize, explicit noise", (2,), 3, torch.full((1, m), 0.02, dtype=DT)),
    ]
    for k, (name, x_shape, n_f, noise) in enumerate(cases):
        Xn = points(X, (*x_shape, d), 11 + k)
        s_bt = BtSampler(sample_shape=torch.Size([n_f]), seed=13 + k)
        fm_bt = bt.fantasize(Xn, s_bt, observation_noise=noise)
        if noise is not None and "fixed" not in label and not botorch_has_fantasy_mean_cache_fix():
            use_fantasy_mean_cache(fm_bt)
        s_rs = RsSampler(sample_shape=(n_f,), seed=0)
        s_rs.base_samples = s_bt.base_samples.clone()
        fm_rs = rs.fantasize(Xn, s_rs, observation_noise=noise)
        assert tuple(fm_rs.batch_shape) == tuple(fm_bt.batch_shape), name
        check_moments(name, fm_rs, fm_bt, Xt)


def test_fantasize_errors() -> None:
    bt, rs, X = model_pair(SINGLE["m1_norm"], 40)
    s_rs = RsSampler(sample_shape=(2,), seed=0)
    s_bt = BtSampler(sample_shape=torch.Size([2]), seed=0)
    Xn = points(X, (2, 2), 3)
    same_error(
        lambda: bt.fantasize(Xn, s_bt, observation_noise=True),
        lambda: rs.fantasize(Xn, s_rs, observation_noise=True),
    )
    # Fixed noise without explicit noise when conditioning.
    btf, rsf, Xf = model_pair(SINGLE["m1_fixed"], 41)
    btf.posterior(Xf[:2])
    rsf.eval()
    Yn = torch.ones(2, 1, dtype=DT)
    same_error(
        lambda: btf.condition_on_observations(X=Xn, Y=Yn),
        lambda: rsf.condition_on_observations(X=Xn, Y=Yn),
    )
    # Conditioning in train mode.
    bt2, rs2, _ = model_pair(SINGLE["m1_norm"], 40)
    bt2.train()
    rs2.train()
    same_error(
        lambda: bt2.condition_on_observations(X=Xn, Y=Yn),
        lambda: rs2.condition_on_observations(X=Xn, Y=Yn),
    )


# ---------------------------------------------------------------------------
# ModelListGP
# ---------------------------------------------------------------------------

LISTS = {
    "list3": (
        ModelSpec(n=9, d=2, normalize=True),
        ModelSpec(n=7, d=2),
        ModelSpec(n=8, d=2, fixed_noise=True, normalize=True),
    ),
    "list2_d3": (
        ModelSpec(n=10, d=3, kernel="matern52", normalize=True, lo=-1.0, hi=2.0),
        ModelSpec(n=9, d=3, standardize=False, lo=-1.0, hi=2.0),
    ),
    "list2_plain": (
        ModelSpec(n=8, d=2, standardize=False),
        ModelSpec(n=7, d=2, standardize=False, kernel="matern52"),
    ),
}


def make_list(name: str) -> tuple[Any, Any, Any]:
    return list_pair(LISTS[name], 60 + 7 * list(LISTS).index(name))


@pytest.fixture(scope="module", params=list(LISTS), ids=list(LISTS))
def lst(request: pytest.FixtureRequest) -> tuple[str, Any, Any, Any]:
    bt, rs, X = make_list(request.param)
    return request.param, bt, rs, X


def test_model_list_fantasize(lst: tuple) -> None:
    label, bt, rs, X = lst
    m, d = rs.num_outputs, X.shape[-1]
    Xt = points(X, (4, d), 1)
    Xn = points(X, (3, d), 2)
    n_f = 3
    masks = [None, [[True] * m, [True] + [False] * (m - 1), [False] * (m - 1) + [True]]]
    for mask in masks:
        bts = [BtSampler(sample_shape=torch.Size([n_f]), seed=5 + i) for i in range(m)]
        em = None if mask is None else torch.tensor(mask)
        fm_bt = bt.fantasize(Xn, BtListSampler(*bts), evaluation_mask=em)
        rss = []
        for s in bts:
            r = RsSampler(sample_shape=(n_f,), seed=0)
            if s.base_samples is not None:
                r.base_samples = s.base_samples.clone()
            rss.append(r)
        fm_rs = rs.fantasize(Xn, RsListSampler(*rss), evaluation_mask=em)
        name = "ModelListGP.fantasize" + ("" if mask is None else ", evaluation mask")
        check_moments(name, fm_rs, fm_bt, Xt)
    shared_bt = BtSampler(sample_shape=torch.Size([n_f]), seed=9)
    fm_bt = bt.fantasize(Xn, shared_bt)
    shared_rs = RsSampler(sample_shape=(n_f,), seed=0)
    shared_rs.base_samples = shared_bt.base_samples.clone()
    fm_rs = rs.fantasize(Xn, shared_rs)
    check_moments("ModelListGP.fantasize, shared sampler", fm_rs, fm_bt, Xt)


def test_model_list_condition_on_observations(lst: tuple) -> None:
    label, bt, rs, X = lst
    plain = label == "list2_plain"
    if not plain and not botorch_has_model_list_conditioning_fix():
        pytest.skip("BoTorch without meta-pytorch/botorch#3399 ignores the transforms")
    from botorch.models import ModelListGP as BtModelListGP

    m, d = rs.num_outputs, X.shape[-1]
    Xt = points(X, (4, d), 1)
    bt.posterior(Xt)
    rs.eval()
    g = torch.Generator().manual_seed(3)
    Xs = [points(X, (2, d), 30 + k) for k in range(m)]
    Yn = 2.0 + torch.randn(2, m, generator=g, dtype=DT)
    noise = 0.01 + 0.05 * torch.rand(2, m, generator=g, dtype=DT)
    cm_bt = BtModelListGP.condition_on_observations(bt, X=Xs, Y=Yn, noise=noise)
    # The explicit noise of the homoskedastic sub-models (#3396; a no-op if
    # BoTorch applied it: without #3399 the list bypasses the sub-models'
    # condition_on_observations, which #3396 fixes).
    use_fantasy_mean_cache(cm_bt)
    cm_rs = rs.condition_on_observations(X=Xs, Y=Yn, noise=noise)
    check_moments("ModelListGP.condition_on_observations", cm_rs, cm_bt, Xt)


# ---------------------------------------------------------------------------
# The hypervolume knowledge gradient
# ---------------------------------------------------------------------------

HVKG_CASES = [
    HvkgCase("default", n_f=4, num_pareto=3, q=1),
    HvkgCase("posterior samples", n_f=3, num_pareto=2, q=2, use_posterior_mean=False, inner=8),
    HvkgCase("log", n_f=3, num_pareto=2, q=1, log=True, current_value=True),
    HvkgCase("current value", n_f=3, num_pareto=3, q=2, current_value=True),
    HvkgCase(
        "decoupled, pending",
        n_f=3,
        num_pareto=2,
        q=2,
        use_posterior_mean=False,
        inner=8,
        pending=2,
        pending_mask="alternate",
        mask="alternate",
    ),
    HvkgCase("unobserved output", n_f=2, num_pareto=2, q=2, mask="first_only"),
    HvkgCase("pending, all outputs", n_f=2, num_pareto=2, q=1, pending=1),
    HvkgCase("objective outcomes", n_f=3, num_pareto=2, q=1, objective="outcomes"),
    HvkgCase(
        "objective weighted",
        n_f=2,
        num_pareto=2,
        q=1,
        objective="weighted",
        use_posterior_mean=False,
        inner=8,
    ),
    HvkgCase("objective generic", n_f=2, num_pareto=2, q=1, objective="generic"),
    HvkgCase(
        "cost (fixed)", n_f=3, num_pareto=2, q=1, current_value=True, cost="fixed", mask="same_rows"
    ),
    HvkgCase(
        "cost (log)", n_f=2, num_pareto=2, q=2, current_value=True, log=True, cost="fixed_log"
    ),
    HvkgCase(
        "multi-fidelity", n_f=3, num_pareto=2, q=1, current_value=True, cost="affine", mf=True
    ),
    HvkgCase(
        "multi-fidelity, samples",
        n_f=2,
        num_pareto=2,
        q=1,
        mf=True,
        use_posterior_mean=False,
        inner=8,
    ),
    HvkgCase("q = 0", n_f=3, num_pareto=2, q=0),
]

HVKG_LISTS = ["list3", "list2_d3"]


@pytest.fixture(scope="module", params=HVKG_LISTS, ids=HVKG_LISTS)
def hvkg_list(request: pytest.FixtureRequest) -> tuple[str, Any, Any, Any]:
    bt, rs, X = make_list(request.param)
    return request.param, bt, rs, X


@pytest.mark.parametrize("case", HVKG_CASES, ids=[c.name for c in HVKG_CASES])
def test_hvkg(hvkg_list: tuple, case: HvkgCase) -> None:
    label, bt, rs, X = hvkg_list
    seed = 100 + 17 * HVKG_CASES.index(case) + 5 * HVKG_LISTS.index(label)
    pair, Xs, ev_rs, ev_bt = run_hvkg_case(case, bt, rs, X, seed)
    err_v, err_g = pair.errors(ev_rs, ev_bt)
    name = "qMFHVKG" if case.mf else "qHVKG"
    kind = "posterior mean" if case.use_posterior_mean else "posterior samples"
    CHECK.record(f"{name} ({kind}{', log' if case.log else ''}) value", err_v, VALUE_TOL)
    CHECK.record(f"{name} ({kind}) gradient", err_g, GRAD_TOL)
    assert err_v <= VALUE_TOL, (case.name, err_v)
    assert err_g <= GRAD_TOL, (case.name, err_g)
    # t-batches are evaluated independently: the values of a stacked batch
    # are those of the single t-batches (BoTorch mixes their fantasy designs).
    with torch.no_grad():
        stacked = torch.as_tensor(pair.rs_acqf(torch.cat(Xs, dim=0)))
    single = torch.cat([v.reshape(1) for v in ev_rs.values])
    if case.log:
        CHECK.check_log("qHVKG t-batch = single t-batches (botorch_rs)", stacked, single, 1e-12)
    else:
        CHECK.check("qHVKG t-batch = single t-batches (botorch_rs)", stacked, single, 1e-12)


def test_hvkg_t_batches_differ_from_botorch() -> None:
    """Documents the deviation: BoTorch's values of a stacked batch of
    candidate sets use the fantasy designs of other t-batches (unless its
    reshape is fixed), so they differ from the values of the single
    t-batches; botorch_rs's equal them."""
    bt, rs, X = make_list("list3")
    case = HvkgCase("t-batches", n_f=3, num_pareto=2, q=1)
    pair, Xs, ev_rs, ev_bt = run_hvkg_case(case, bt, rs, X, 7, count=2)
    Xb = torch.cat(Xs, dim=0)
    with torch.no_grad():
        bt_stacked = pair.bt_acqf(Xb)
        rs_stacked = torch.as_tensor(pair.rs_acqf(Xb))
    single = torch.cat([v.reshape(1) for v in ev_bt.values])
    CHECK.check("qHVKG t-batch vs BoTorch's single t-batches", rs_stacked, single, VALUE_TOL)
    _, fant = bt_hvkg._split_hvkg_fantasy_points(Xb, 3, 2)
    per_batch = torch.stack([Xb[b, 1:].reshape(3, 2, -1) for b in range(2)], dim=1)
    if not torch.equal(fant, per_batch):
        assert not torch.allclose(bt_stacked, single)


def test_value_function(hvkg_list: tuple) -> None:
    label, bt, rs, X = hvkg_list
    m, d = rs.num_outputs, X.shape[-1]
    target = {d - 1: 1.0}
    configs = [
        (True, False, False, "identity"),
        (False, False, False, "identity"),
        (True, True, False, "weighted"),
        (False, True, True, "identity"),
        (False, False, False, "generic"),
    ]
    for k, (upm, log, proj, obj) in enumerate(configs):
        obj_bt, obj_rs, _ = objective_pair(obj, m, bt, rs, X)
        ref = ref_point(bt, obj_bt, X, 70 + k)
        s_bt = BtSampler(sample_shape=torch.Size([8]), seed=80 + k)
        kw_bt: dict[str, Any] = dict(use_posterior_mean=upm, log=log)
        kw_rs: dict[str, Any] = dict(use_posterior_mean=upm, log=log)
        if proj:
            kw_bt["project"] = functools.partial(
                bt_acq_utils.project_to_target_fidelity, target_fidelities=target, d=d
            )
            kw_rs["project"] = functools.partial(
                rs_acq_utils.project_to_target_fidelity, target_fidelities=target, d=d
            )
        vf_bt = bt_hvkg._get_hv_value_function(
            model=bt, ref_point=ref, objective=obj_bt, sampler=s_bt, **kw_bt
        )
        x = points(X, (2, 3, d), 90 + k)
        with gpytorch.settings.detach_test_caches(False):
            v_bt, g_bt = values_and_grads(vf_bt, x)
        s_rs = RsSampler(sample_shape=(8,), seed=0)
        if s_bt.base_samples is not None:
            s_rs.base_samples = s_bt.base_samples.clone()
        vf_rs = rs_hvkg._get_hv_value_function(
            model=rs, ref_point=ref, objective=obj_rs, sampler=s_rs, **kw_rs
        )
        v_rs, g_rs = values_and_grads(vf_rs, x)
        tag = f"value function ({'mean' if upm else 'samples'}{', log' if log else ''})"
        if log:
            CHECK.check_log(f"{tag}: value", v_rs, v_bt, VALUE_TOL)
        else:
            CHECK.check(f"{tag}: value", v_rs, v_bt, VALUE_TOL, atol=1e-300)
        CHECK.check(f"{tag}: gradient", g_rs, g_bt, GRAD_TOL, atol=1e-14)


def test_split_hvkg_fantasy_points() -> None:
    g = torch.Generator().manual_seed(0)
    for q, n_f, n_p in [(1, 3, 2), (0, 2, 2), (2, 1, 3)]:
        X = torch.rand(1, q + n_f * n_p, 3, generator=g, dtype=DT)
        a_bt, f_bt = bt_hvkg._split_hvkg_fantasy_points(X, n_f, n_p)
        a_rs, f_rs = rs_hvkg._split_hvkg_fantasy_points(X, n_f, n_p)
        assert torch.equal(a_rs, a_bt) and torch.equal(f_rs, f_bt)
        # Lists and differentiable views.
        a_l, f_l = rs_hvkg._split_hvkg_fantasy_points(X.tolist(), n_f, n_p)
        assert torch.equal(torch.tensor(a_l, dtype=DT).reshape(a_bt.shape), a_bt)
        assert torch.equal(torch.tensor(f_l, dtype=DT), f_bt)
    # Several t-batches: X_fantasies[i, b] are the designs of t-batch b.
    X = torch.rand(2, 2, 1 + 2 * 2, 3, generator=g, dtype=DT)
    _, f_rs = rs_hvkg._split_hvkg_fantasy_points(X, 2, 2)
    for i in range(2):
        for b in range(2):
            for c in range(2):
                assert torch.equal(f_rs[i, b, c], X[b, c, 1 + 2 * i : 1 + 2 * (i + 1)])
    with pytest.raises(ValueError) as e_bt:
        bt_hvkg._split_hvkg_fantasy_points(X, 3, 2)
    with pytest.raises(ValueError) as e_rs:
        rs_hvkg._split_hvkg_fantasy_points(X, 3, 2)
    assert str(e_rs.value) == str(e_bt.value)


def test_hvkg_errors() -> None:
    bt_l, rs_l, X = make_list("list3")
    ref = torch.zeros(3, dtype=DT)
    bt_s, rs_s, _ = model_pair(SINGLE["m2_plain"], 44)

    same_error(
        lambda: bt_hvkg.qHypervolumeKnowledgeGradient(bt_s, ref_point=ref[:2]),
        lambda: rs_hvkg.qHypervolumeKnowledgeGradient(rs_s, ref_point=ref[:2]),
    )
    same_error(
        lambda: bt_hvkg.qHypervolumeKnowledgeGradient(
            bt_l,
            ref_point=ref,
            num_fantasies=4,
            sampler=BtListSampler(*[BtSampler(torch.Size([3])) for _ in range(3)]),
        ),
        lambda: rs_hvkg.qHypervolumeKnowledgeGradient(
            rs_l,
            ref_point=ref,
            num_fantasies=4,
            sampler=RsListSampler(*[RsSampler((3,)) for _ in range(3)]),
        ),
    )
    cm = bt_cost.FixedCostModel(torch.ones(3, dtype=DT))
    same_error(
        lambda: bt_hvkg.qHypervolumeKnowledgeGradient(
            bt_l, ref_point=ref, cost_aware_utility=bt_cost_aware.InverseCostWeightedUtility(cm)
        ),
        lambda: rs_hvkg.qHypervolumeKnowledgeGradient(
            rs_l,
            ref_point=ref,
            cost_aware_utility=rs_cost_aware.InverseCostWeightedUtility(
                rs_cost.FixedCostModel(torch.ones(3, dtype=DT))
            ),
        ),
    )
    same_error(
        lambda: bt_hvkg.qMultiFidelityHypervolumeKnowledgeGradient(
            bt_l, ref_point=ref, target_fidelities={1: 1.0}, expand=lambda X: X
        ),
        lambda: rs_hvkg.qMultiFidelityHypervolumeKnowledgeGradient(
            rs_l, ref_point=ref, target_fidelities={1: 1.0}, expand=lambda X: X
        ),
    )
    # One-shot acquisition functions cannot be optimized sequentially.
    acq_bt = bt_hvkg.qHypervolumeKnowledgeGradient(
        bt_l, ref_point=ref, num_fantasies=2, num_pareto=1
    )
    acq_rs = rs_hvkg.qHypervolumeKnowledgeGradient(
        rs_l, ref_point=ref, num_fantasies=2, num_pareto=1
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    same_error(
        lambda: bt_optimize.optimize_acqf(
            acq_bt, bounds, q=2, num_restarts=2, raw_samples=4, sequential=True
        ),
        lambda: rs_optimize.optimize_acqf(
            acq_rs, bounds, q=2, num_restarts=2, raw_samples=4, sequential=True
        ),
    )
    # botorch_rs requires its own samplers.
    with pytest.raises(UnsupportedError):
        rs_hvkg.qHypervolumeKnowledgeGradient(
            rs_l, ref_point=ref, sampler=BtListSampler(*[BtSampler(torch.Size([8]))] * 3)
        )


def test_hvkg_set_x_pending(hvkg_list: tuple) -> None:
    """Pending points set after construction (with and without an
    evaluation mask) give the values of pending points at construction."""
    label, bt, rs, X = hvkg_list
    case = HvkgCase("pending", n_f=2, num_pareto=2, q=1, pending=2, pending_mask="alternate")
    pair, Xs, ev_rs, ev_bt = run_hvkg_case(case, bt, rs, X, 300)
    acq = pair.rs_acqf
    pending, pmask = pair.pending, pair.pmask
    acq.set_X_pending(None)
    with torch.no_grad():
        v_none = torch.as_tensor(acq(Xs[0]))
    acq.set_X_pending(pending, pmask)
    with torch.no_grad():
        v_again = torch.as_tensor(acq(Xs[0]))
    CHECK.check("qHVKG set_X_pending = pending at construction", v_again, ev_rs.values[0], 1e-14)
    assert not torch.allclose(v_none, v_again)
    # BoTorch: the evaluation mask of the candidates and pending points.
    mask_bt = pair.bt_acqf.construct_evaluation_mask(Xs[0][..., :1, :])
    mask_rs = acq.construct_evaluation_mask(Xs[0][..., :1, :])
    assert torch.equal(torch.as_tensor(mask_rs), mask_bt)


# ---------------------------------------------------------------------------
# Cost models, cost-aware utilities and fidelity helpers
# ---------------------------------------------------------------------------


def test_cost_models() -> None:
    g = torch.Generator().manual_seed(0)
    X = torch.rand(2, 3, 4, generator=g, dtype=DT)
    weights = {1: 1.5, -1: 0.5}
    a_bt = bt_cost.AffineFidelityCostModel(fidelity_weights=weights, fixed_cost=0.25)
    a_rs = rs_cost.AffineFidelityCostModel(fidelity_weights=weights, fixed_cost=0.25)
    assert a_rs.fidelity_dims == a_bt.fidelity_dims
    CHECK.check("AffineFidelityCostModel", a_rs(X), a_bt(X), 1e-15)
    a_bt = bt_cost.AffineFidelityCostModel()
    a_rs = rs_cost.AffineFidelityCostModel()
    CHECK.check("AffineFidelityCostModel (default)", a_rs(X), a_bt(X), 1e-15)
    costs = torch.tensor([1.0, 2.5], dtype=DT)
    f_bt = bt_cost.FixedCostModel(costs)
    f_rs = rs_cost.FixedCostModel(costs)
    CHECK.check("FixedCostModel", f_rs(X), f_bt(X), 0.0)
    # Gradients of a cost model (autograd).
    Xg = X.clone().requires_grad_(True)
    (g_rs,) = torch.autograd.grad(a_rs(Xg).pow(2).sum(), Xg)
    Xg = X.clone().requires_grad_(True)
    (g_bt,) = torch.autograd.grad(a_bt(Xg).pow(2).sum(), Xg)
    CHECK.check("AffineFidelityCostModel gradient", g_rs, g_bt, 1e-15)


def grads(out: Any, inputs: tuple[Any, ...]) -> list[Any]:
    """Gradients of ``out`` (zeros for inputs it does not depend on, e.g.
    the candidates of a fixed cost model)."""
    gs = torch.autograd.grad(out, inputs, allow_unused=True)
    return [torch.zeros_like(x) if g is None else g for g, x in zip(gs, inputs, strict=True)]


@pytest.mark.parametrize("log", [False, True], ids=["linear", "log"])
def test_inverse_cost_weighted_utility(log: bool) -> None:
    g = torch.Generator().manual_seed(1)
    X = torch.rand(2, 3, 2, generator=g, dtype=DT)
    deltas = torch.randn(4, 2, generator=g, dtype=DT)
    costs = torch.tensor([1.0, 2.5], dtype=DT)
    mask = torch.tensor([[True, False]] * 3)
    mixed = torch.tensor([[True, False], [True, True], [False, True]])
    models = [
        (
            "fixed",
            bt_cost.FixedCostModel(costs),
            rs_cost.FixedCostModel(costs),
        ),
        (
            "affine",
            bt_cost.AffineFidelityCostModel({1: 2.0}, fixed_cost=0.5),
            rs_cost.AffineFidelityCostModel({1: 2.0}, fixed_cost=0.5),
        ),
    ]
    bt_gp, rs_gp, _ = model_pair(ModelSpec(n=8, d=2, normalize=True), 50)
    models.append(("GP posterior mean", bt_gp, rs_gp))
    for name, cm_bt, cm_rs in models:
        u_bt = bt_cost_aware.InverseCostWeightedUtility(cost_model=cm_bt, log=log)
        u_rs = rs_cost_aware.InverseCostWeightedUtility(cost_model=cm_rs, log=log)
        for em in [None, mask] if name == "fixed" else [None]:
            Xg_bt = X.clone().requires_grad_(True)
            d_bt = deltas.clone().requires_grad_(True)
            out_bt = u_bt(Xg_bt, d_bt, X_evaluation_mask=em)
            gx_bt, gd_bt = grads(out_bt.pow(2).sum(), (Xg_bt, d_bt))
            Xg_rs = X.clone().requires_grad_(True)
            d_rs = deltas.clone().requires_grad_(True)
            out_rs = u_rs(Xg_rs, d_rs, X_evaluation_mask=em)
            gx_rs, gd_rs = grads(out_rs.pow(2).sum(), (Xg_rs, d_rs))
            tag = f"InverseCostWeightedUtility ({name}{', log' if log else ''})"
            CHECK.check(f"{tag}: value", out_rs, out_bt, 1e-14)
            CHECK.check(f"{tag}: gradient (deltas)", gd_rs, gd_bt, 1e-14)
            CHECK.check(f"{tag}: gradient (X)", gx_rs, gx_bt, 1e-12, atol=1e-300)
    # Errors.
    u_bt = bt_cost_aware.InverseCostWeightedUtility(models[0][1], log=log)
    u_rs = rs_cost_aware.InverseCostWeightedUtility(models[0][2], log=log)
    same_error(
        lambda: u_bt(X, deltas, X_evaluation_mask=mixed),
        lambda: u_rs(X, deltas, X_evaluation_mask=mixed),
    )
    u_bt = bt_cost_aware.InverseCostWeightedUtility(models[0][1], use_mean=False)
    u_rs = rs_cost_aware.InverseCostWeightedUtility(models[0][2], use_mean=False)
    with pytest.raises(RuntimeError) as e_bt:
        u_bt(X, deltas)
    with pytest.raises(RuntimeError) as e_rs:
        u_rs(X, deltas)
    assert str(e_rs.value) == str(e_bt.value)


def test_generic_cost_aware_utility() -> None:
    g = torch.Generator().manual_seed(2)
    X = torch.rand(2, 3, 2, generator=g, dtype=DT)
    deltas = torch.randn(4, 2, generator=g, dtype=DT)

    def cost(X: Any, deltas: Any) -> Any:
        return deltas / (1.0 + X[..., -1].sum(dim=-1))

    u_bt = bt_cost_aware.GenericCostAwareUtility(cost)
    u_rs = rs_cost_aware.GenericCostAwareUtility(cost)
    CHECK.check("GenericCostAwareUtility", u_rs(X, deltas), u_bt(X, deltas), 0.0)


def test_fidelity_helpers() -> None:
    g = torch.Generator().manual_seed(3)
    X = torch.rand(2, 3, 4, generator=g, dtype=DT)
    for tf in [{-1: 1.0}, {1: 0.5, 3: 0.75}]:
        p_bt = bt_acq_utils.project_to_target_fidelity(X, target_fidelities=tf, d=4)
        p_rs = rs_acq_utils.project_to_target_fidelity(X, target_fidelities=tf, d=4)
        assert torch.equal(p_rs, p_bt)
        p_l = rs_acq_utils.project_to_target_fidelity(X.tolist(), target_fidelities=tf, d=4)
        assert torch.equal(torch.tensor(p_l, dtype=DT), p_bt)
        # Insertion of the fidelity columns.
        Xr = X[..., : 4 - len(tf)]
        p_bt = bt_acq_utils.project_to_target_fidelity(Xr, target_fidelities=tf, d=4)
        p_rs = rs_acq_utils.project_to_target_fidelity(Xr.tolist(), target_fidelities=tf, d=4)
        assert torch.equal(torch.tensor(p_rs, dtype=DT), p_bt)
    same_error(
        lambda: bt_acq_utils.project_to_target_fidelity(X, target_fidelities={5: 1.0}, d=4),
        lambda: rs_acq_utils.project_to_target_fidelity(X, target_fidelities={5: 1.0}, d=4),
    )
    for dims, k in [(None, 0), (None, 2), ([0, 2], 3)]:
        e_bt_ = bt_acq_utils.expand_trace_observations(X, fidelity_dims=dims, num_trace_obs=k)
        e_rs_ = rs_acq_utils.expand_trace_observations(X, fidelity_dims=dims, num_trace_obs=k)
        assert torch.equal(e_rs_, e_bt_)
        e_l = rs_acq_utils.expand_trace_observations(
            X.tolist(), fidelity_dims=dims, num_trace_obs=k
        )
        CHECK.check("expand_trace_observations (Rust)", torch.tensor(e_l, dtype=DT), e_bt_, 1e-15)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_optimize_from_identical_initial_conditions() -> None:
    """optimize_acqf of qHVKG from the same initial conditions (BoTorch with
    ``batch_limit=1``: every restart is optimized by its own L-BFGS-B run, as
    in botorch_rs). Restarts converge to the same local optima: values agree
    to 1e-6 for most restarts (L-BFGS-B paths may part at flat directions of
    dominated fantasy designs), and botorch_rs's best is not worse."""
    bt, rs, X = make_list("list3")
    case = HvkgCase("optimize", n_f=2, num_pareto=2, q=1)
    pair = HvkgPair(case, bt, rs, X, 500)
    d = X.shape[-1]
    bounds = torch.stack([X.min(0).values, X.max(0).values])
    R = 6
    ics = points(X, (R, 1 + case.n_f * case.num_pareto, d), 501)
    with torch.no_grad():
        pair.bt_acqf(ics[:1])  # draws BoTorch's base samples
    acq_rs = pair.build_rs()
    opts = {"batch_limit": 1, "maxiter": 200}
    c_bt, v_bt = bt_optimize.optimize_acqf(
        pair.bt_acqf,
        bounds,
        q=1,
        num_restarts=R,
        batch_initial_conditions=ics,
        options=opts,
        return_best_only=False,
        return_full_tree=True,
    )
    c_rs, v_rs = rs_optimize.optimize_acqf(
        acq_rs,
        bounds,
        q=1,
        num_restarts=R,
        batch_initial_conditions=ics,
        options=opts,
        return_best_only=False,
        return_full_tree=True,
    )
    c_rs, v_rs = torch.as_tensor(c_rs), torch.as_tensor(v_rs)
    assert c_rs.shape == c_bt.shape == ics.shape
    # Both libraries' values of their solutions (same base samples).
    with torch.no_grad():
        v_rs_at_bt = torch.as_tensor(acq_rs(c_bt))
    CHECK.check("qHVKG at BoTorch's optimizers", v_rs_at_bt, v_bt, VALUE_TOL)
    agree = ((v_rs - v_bt).abs() <= 1e-6 * v_bt.abs().clamp_min(1.0)).sum().item()
    CHECK.record("optimize_acqf (qHVKG): restarts that disagree (fraction)", 1 - agree / R, 0.5)
    assert agree >= R / 2, (v_rs, v_bt)
    CHECK.check_bound(
        "optimize_acqf (qHVKG): best value shortfall (relative)",
        (v_bt.max() - v_rs.max()).item() / v_bt.max().abs().item(),
        1e-6,
    )
    # extract_candidates of the best restart.
    c_best, v_best = rs_optimize.optimize_acqf(
        acq_rs, bounds, q=1, num_restarts=R, batch_initial_conditions=ics, options=opts
    )
    best = int(v_rs.argmax())
    assert torch.equal(torch.as_tensor(c_best), c_rs[best, :1])
    assert float(torch.as_tensor(v_best)) == float(v_rs[best])


def test_optimize_end_to_end() -> None:
    """optimize_acqf with each library's own one-shot initial conditions
    (gen_one_shot_hvkg_initial_conditions, different random numbers): a
    gross-failure check of the best values (evaluated with the same
    acquisition function), and the structure of the initial conditions."""
    bt, rs, X = make_list("list2_d3")
    case = HvkgCase("end to end", n_f=2, num_pareto=2, q=1)
    pair = HvkgPair(case, bt, rs, X, 600)
    d = X.shape[-1]
    bounds = torch.stack([X.min(0).values, X.max(0).values])
    with torch.no_grad():
        pair.bt_acqf(points(X, (1, 5, d), 601))
    acq_rs = pair.build_rs()
    small = {"num_inner_restarts": 4, "raw_inner_samples": 64, "batch_limit": 4}
    torch.manual_seed(0)
    c_bt, _ = bt_optimize.optimize_acqf(
        pair.bt_acqf,
        bounds,
        q=1,
        num_restarts=4,
        raw_samples=64,
        options=dict(small),
        return_full_tree=True,
    )
    c_rs, _ = rs_optimize.optimize_acqf(
        acq_rs,
        bounds,
        q=1,
        num_restarts=4,
        raw_samples=64,
        options=dict(small, seed=0),
        return_full_tree=True,
    )
    with torch.no_grad():
        best_bt = float(torch.as_tensor(acq_rs(c_bt.unsqueeze(0))))
        best_rs = float(torch.as_tensor(acq_rs(torch.as_tensor(c_rs).unsqueeze(0))))
    CHECK.check_bound(
        "optimize_acqf (qHVKG, own initial conditions): shortfall vs BoTorch (relative)",
        (best_bt - best_rs) / abs(best_bt),
        0.1,
    )
    # The initial conditions.
    ics = torch.as_tensor(
        rs_init.gen_one_shot_hvkg_initial_conditions(
            acq_rs, bounds, q=1, num_restarts=5, raw_samples=64, options=dict(small, seed=1)
        )
    )
    torch.manual_seed(1)
    ics_bt = bt_init.gen_one_shot_hvkg_initial_conditions(
        pair.bt_acqf, bounds, q=1, num_restarts=5, raw_samples=64, options=dict(small)
    )
    assert ics.shape == ics_bt.shape == (5, 1 + 2 * 2, d)
    assert bool(((ics >= bounds[0]) & (ics <= bounds[1])).all())
    with torch.no_grad():
        mean_rs = torch.as_tensor(acq_rs(ics)).mean().item()
        mean_bt = torch.as_tensor(acq_rs(ics_bt)).mean().item()
    CHECK.check_bound(
        "one-shot initial conditions: mean value shortfall vs BoTorch (relative)",
        (mean_bt - mean_rs) / abs(mean_bt),
        0.5,
    )
    # extract_candidates and the augmented q.
    assert acq_rs.get_augmented_q_batch_size(2) == pair.bt_acqf.get_augmented_q_batch_size(2)
    assert torch.equal(
        torch.as_tensor(acq_rs.extract_candidates(ics)), pair.bt_acqf.extract_candidates(ics)
    )
