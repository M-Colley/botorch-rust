"""Parity of the multi-task GPs (phase 7a: ``MultiTaskGP``,
``KroneckerMultiTaskGP``) with BoTorch, through the public APIs of both
libraries side by side.

Every model is built from the same case (``multitask_helpers.py``) in both
libraries, with the same random initial task factors (both draw them from
torch's generator), and compared:

* ``MultiTaskGP``: parameter names, shapes, default values and optimizer
  bounds; the MLL and its gradient with respect to all raw parameters
  (including the task factors and the task prior) at the default and at
  random parameters; ``eval()`` with tasks without data; posteriors over
  the output tasks and with task columns (also batched, other
  ``output_indices``, with observation noise), samples from identical base
  samples; the input gradients of ``PosteriorMean`` / ``qSimpleRegret`` with
  a scalarizing posterior transform; ``condition_on_observations`` and
  ``fantasize`` (identical base samples); fits from identical initial
  values; qLogNEI, qEI, qUCB, LogEI, ``PosteriorMean``,
  ``PosteriorStandardDeviation``; a ``ModelListGP`` of two models with
  qLogNEHVI;
* ``KroneckerMultiTaskGP``: parameters, bounds, MLL and gradients; the
  posterior mean (BoTorch's), the exact joint posterior (computed densely in
  torch from BoTorch's hyperparameters, which is also the distribution of
  BoTorch's Matheron samples, measured through their linear map), the
  distribution of botorch_rs's Matheron samples, the base sample shape;
  qEI and qLogNEHVI against BoTorch's acquisition code on the exact joint
  posterior with identical base samples; fits.

BoTorch's bugs are fixed in the reference (``multitask_helpers``), and
``test_botorch_bugs`` measures what they change. Tolerances: values 1e-9,
gradients 1e-8 (relative); MC acquisition values against the regime of
BoTorch's outcomes on twins (training inputs perturbed at 1e-14 .. 1e-11);
fits as in the 6a kernel tests (two stages: 4 twins, then 64). The
randomized sweep is marked ``equivalence``; it evaluates BoTorch with exact
kernel products (``multitask_helpers.exact_kernel_products``: GPyTorch's
jittered Cholesky roots of the singular task covariance change the MLL by up
to 4.8e-9 in the sweep) and zeroed self-distances (6a).
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from typing import Any

import pytest
from kernel_tree_helpers import random_spec, zero_diagonal_distances

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import multitask_helpers as h  # noqa: E402
from botorch.acquisition import LogExpectedImprovement as BtLogEI  # noqa: E402
from botorch.acquisition import PosteriorMean as BtPM  # noqa: E402
from botorch.acquisition import PosteriorStandardDeviation as BtPSD  # noqa: E402
from botorch.acquisition import qExpectedImprovement as BtQEI  # noqa: E402
from botorch.acquisition import qSimpleRegret as BtQSR  # noqa: E402
from botorch.acquisition import qUpperConfidenceBound as BtQUCB  # noqa: E402
from botorch.acquisition.logei import qLogNoisyExpectedImprovement as BtQLogNEI  # noqa: E402
from botorch.acquisition.multi_objective.logei import (  # noqa: E402
    qLogNoisyExpectedHypervolumeImprovement as BtQLogNEHVI,
)
from botorch.acquisition.objective import LinearMCObjective as BtLinear  # noqa: E402
from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtSPT  # noqa: E402
from botorch.fit import fit_gpytorch_mll as bt_fit  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models.multitask import MultiTaskGP as BtMTGP  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from mo_acquisition_helpers import bt_sampler, rs_sampler, synced_draws  # noqa: E402
from so_acquisition_helpers import check_against_twins  # noqa: E402

from botorch_rs.acquisition import LogExpectedImprovement as RsLogEI  # noqa: E402
from botorch_rs.acquisition import PosteriorMean as RsPM  # noqa: E402
from botorch_rs.acquisition import PosteriorStandardDeviation as RsPSD  # noqa: E402
from botorch_rs.acquisition import qExpectedImprovement as RsQEI  # noqa: E402
from botorch_rs.acquisition import qSimpleRegret as RsQSR  # noqa: E402
from botorch_rs.acquisition import qUpperConfidenceBound as RsQUCB  # noqa: E402
from botorch_rs.acquisition.logei import qLogNoisyExpectedImprovement as RsQLogNEI  # noqa: E402
from botorch_rs.acquisition.multi_objective.logei import (  # noqa: E402
    qLogNoisyExpectedHypervolumeImprovement as RsQLogNEHVI,
)
from botorch_rs.acquisition.objective import LinearMCObjective as RsLinear  # noqa: E402
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform as RsSPT  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import MultiTaskGP as RsMTGP  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker, FitOutcome, compare_twin_fits, rel_err  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

DT = torch.float64
CHECK = Checker("multi-task GPs (7a)")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
N_TWINS = 4
NUM_SAMPLES = 64

MTGP = {c["name"]: c for c in h.mtgp_cases()}
KRON = {c["name"]: c for c in h.kron_cases()}


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _grad_err(a: Any, e: Any) -> float:
    a = torch.as_tensor(a, dtype=DT).reshape(-1)
    e = torch.as_tensor(e, dtype=DT).reshape(-1)
    assert a.shape == e.shape
    return (a - e).abs().max().item() / max(e.abs().max().item(), 1e-300)


def check_grad(quantity: str, a: Any, e: Any, tol: float = GRAD_TOL) -> None:
    err = _grad_err(a, e)
    CHECK.record(quantity, err, tol)
    assert err <= tol, (quantity, err)


def _vg(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.clone().requires_grad_(True)
    v = acqf(Xg)
    # (Zero where the values do not depend on X, e.g. a CategoricalKernel.)
    (g,) = torch.autograd.grad(v.sum(), Xg, allow_unused=True)
    return v.detach(), torch.zeros_like(X) if g is None else g


def _seed(name: str, table: dict[str, Any]) -> int:
    return 9000 + 31 * list(table).index(name)


def mtgp_pair(name: str, X: Any = None) -> tuple[Any, Any, dict[str, Any], tuple[Any, Any, Any]]:
    case = MTGP[name]
    seed = _seed(name, MTGP)
    X0, Y, Yvar = h.make_data(case, seed)
    bt, rs = h.build_pair(case, X0 if X is None else X, Y, Yvar, seed)
    return bt, rs, case, (X0, Y, Yvar)


def randomize(bt: Any, rs: Any, seed: int) -> list[float]:
    raw = h.random_raw(bt, seed)
    h.set_flat(bt, raw)
    rs._rs.set_raw_parameters(raw)
    return raw


def rs_mll(rs: Any) -> tuple[float, Any]:
    value, grad = RsMLL(rs.likelihood, rs).value_and_grad()
    return float(value), torch.as_tensor(grad, dtype=DT)


# ---------------------------------------------------------------------------
# MultiTaskGP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MTGP))
def test_mtgp_structure_and_mll(name: str) -> None:
    bt, rs, _, _ = mtgp_pair(name)
    nb = [(n, tuple(p.shape)) for n, p in bt.named_parameters()]
    nr = [(n, tuple(torch.as_tensor(p).shape)) for n, p in rs.named_parameters()]
    assert nb == nr
    for (_, pb), (_, pr) in zip(bt.named_parameters(), rs.named_parameters(), strict=True):
        CHECK.check(
            "MultiTaskGP: default raw parameters (incl. random task factors)", pr, pb.detach(), 0.0
        )
    assert [list(b) for b in rs._rs.raw_parameter_bounds()] == h.bounds_of(bt)
    vb, gb = h.bt_mll_value_and_grad(bt)
    vr, gr = rs_mll(rs)
    CHECK.check("MultiTaskGP: default MLL", vr, vb, VALUE_TOL)
    check_grad("MultiTaskGP: default MLL gradient", gr, gb)
    randomize(bt, rs, _seed(name, MTGP) + 1)
    vb, gb = h.bt_mll_value_and_grad(bt)
    vr, gr = rs_mll(rs)
    CHECK.check("MultiTaskGP: MLL", vr, vb, VALUE_TOL)
    check_grad("MultiTaskGP: MLL gradient", gr, gb)
    # Attribute access like BoTorch's.
    assert rs.num_tasks == bt.num_tasks and rs._output_tasks == list(bt._output_tasks)
    assert rs._task_feature == bt._task_feature
    assert rs.num_non_task_features == bt.num_non_task_features
    # (BoTorch's model is in eval mode after its MLL evaluation.)
    rs.eval()
    tk_b, tk_r = bt.covar_module.kernels[-1], rs.covar_module.kernels[-1]
    CHECK.check(
        "MultiTaskGP: task covariance matrix",
        tk_r.covar_matrix,
        tk_b.covar_matrix.detach(),
        VALUE_TOL,
    )
    if not case_is_fixed(name):
        CHECK.check(
            "MultiTaskGP: likelihood noise",
            rs.likelihood.noise,
            bt.likelihood.noise.detach().reshape(torch.as_tensor(rs.likelihood.noise).shape),
            VALUE_TOL,
        )


def case_is_fixed(name: str) -> bool:
    return bool(MTGP[name].get("fixed_noise"))


@pytest.mark.parametrize("name", list(MTGP))
def test_mtgp_posteriors(name: str) -> None:
    bt, rs, case, _ = mtgp_pair(name)
    seed = _seed(name, MTGP)
    randomize(bt, rs, seed + 1)
    bt.eval()
    rs.eval()
    # eval(): the task factors of tasks without data.
    CHECK.check(
        "MultiTaskGP: raw parameters after eval()",
        torch.cat([torch.as_tensor(p).reshape(-1) for p in rs.parameters()]),
        torch.tensor(h.flat(bt), dtype=DT),
        1e-15,
    )
    m = rs.num_outputs
    for label, X in (
        ("output tasks", h.test_points(case, 3, seed + 2)),
        ("task inputs", h.test_points(case, 3, seed + 3, with_tasks=True)),
        ("batched", h.test_points(case, 6, seed + 4).reshape(2, 3, -1)),
    ):
        for noise in (False, True):
            with torch.no_grad():
                pb = bt.posterior(X, observation_noise=noise)
            pr = rs.posterior(X, observation_noise=noise)
            tag = f"MultiTaskGP: posterior ({label}{', noise' if noise else ''})"
            CHECK.check(f"{tag} mean", pr.mean, pb.mean, VALUE_TOL)
            CHECK.check(f"{tag} variance", pr.variance, pb.variance, VALUE_TOL)
            CHECK.check(
                f"{tag} covariance",
                pr.distribution.covariance_matrix,
                pb.distribution.covariance_matrix,
                VALUE_TOL,
            )
    # Other output tasks (reversed order of all tasks).
    tasks = list(reversed(rs.all_tasks))
    X = h.test_points(case, 2, seed + 5)
    with torch.no_grad():
        pb = bt.posterior(X, output_indices=tasks)
    pr = rs.posterior(X, output_indices=tasks)
    CHECK.check("MultiTaskGP: posterior (output_indices) mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "MultiTaskGP: posterior (output_indices) covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    # Samples from identical base samples (correlated outputs).
    Z = torch.randn(8, 2, len(tasks), generator=torch.Generator().manual_seed(seed + 6), dtype=DT)
    with torch.no_grad():
        sb = pb.rsample_from_base_samples(torch.Size([8]), Z)
    sr = pr.rsample_from_base_samples(torch.Size([8]), Z)
    CHECK.check("MultiTaskGP: samples from base samples", sr, sb, VALUE_TOL)
    # Input gradients (scalarized over the output tasks).
    w = torch.linspace(0.6, -0.4, m, dtype=DT)
    X1 = h.test_points(case, 4, seed + 7).unsqueeze(-2)
    vb, gb = _vg(BtPM(bt, posterior_transform=BtSPT(w, offset=0.2)), X1)
    vr, gr = _vg(RsPM(rs, posterior_transform=RsSPT(w, offset=0.2)), X1)
    CHECK.check("MultiTaskGP: PosteriorMean (scalarized) values", vr, vb, VALUE_TOL)
    check_grad("MultiTaskGP: PosteriorMean (scalarized) gradients", gr, gb)
    # (The scalarized posterior's base samples: S x 1 x q, shared by the batch.)
    Z = torch.randn(16, 1, 2, generator=torch.Generator().manual_seed(seed + 8), dtype=DT)
    s_b, s_r = BtSampler(torch.Size([16]), seed=0), RsSampler(torch.Size([16]), seed=0)
    s_b.base_samples = Z.clone()
    s_r.base_samples = Z.clone()
    Xq = h.test_points(case, 4, seed + 9).reshape(2, 2, -1)
    vb, gb = _vg(BtQSR(bt, sampler=s_b, posterior_transform=BtSPT(w)), Xq)
    vr, gr = _vg(RsQSR(rs, sampler=s_r, posterior_transform=RsSPT(w)), Xq)
    CHECK.check("MultiTaskGP: qSimpleRegret (scalarized) values", vr, vb, VALUE_TOL)
    check_grad("MultiTaskGP: qSimpleRegret (scalarized) gradients", gr, gb)


@pytest.mark.parametrize("name", list(MTGP))
def test_mtgp_conditioning_and_fantasies(name: str) -> None:
    bt, rs, case, _ = mtgp_pair(name)
    seed = _seed(name, MTGP)
    randomize(bt, rs, seed + 1)
    Xt = h.test_points(case, 3, seed + 2)
    bt.posterior(Xt)
    rs.posterior(Xt)
    Xn = h.test_points(case, 2, seed + 3, with_tasks=True)
    Yn = torch.tensor([[0.4], [-0.3]], dtype=DT) + 2.0
    kw: dict[str, Any] = {}
    if case.get("fixed_noise"):
        kw["noise"] = torch.full((2, 1), 0.05, dtype=DT)
    cb = bt.condition_on_observations(Xn, Yn, **kw)
    cr = rs.condition_on_observations(Xn, Yn, **kw)
    with torch.no_grad():
        pb, pr = cb.posterior(Xt), cr.posterior(Xt)
    CHECK.check("MultiTaskGP: conditioned posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "MultiTaskGP: conditioned posterior covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    # Batched conditioning (broadcast over the observations). (With fixed
    # noise, a batched `noise` for non-batched `X` fails in BoTorch: GPyTorch's
    # FixedNoiseGaussianLikelihood cannot concatenate it with the training
    # noise; see the draft.)
    Yb = Yn.unsqueeze(0) + torch.tensor([0.0, 0.7], dtype=DT).reshape(2, 1, 1)
    cb = bt.condition_on_observations(Xn, Yb, **kw)
    cr = rs.condition_on_observations(Xn, Yb, **kw)
    assert tuple(cr.batch_shape) == tuple(cb.batch_shape)
    with torch.no_grad():
        pb, pr = cb.posterior(Xt), cr.posterior(Xt)
    CHECK.check("MultiTaskGP: conditioned posterior mean (batched)", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "MultiTaskGP: conditioned posterior covariance (batched)",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    if case.get("fixed_noise"):
        # Batched inputs, observations and noise.
        Xb = Xn.unsqueeze(0).expand(2, *Xn.shape)
        nb = torch.full((2, 2, 1), 0.05, dtype=DT) + torch.tensor([0.0, 0.02], dtype=DT).reshape(
            2, 1, 1
        )
        cb = bt.condition_on_observations(Xb, Yb, noise=nb)
        cr = rs.condition_on_observations(Xb, Yb, noise=nb)
        with torch.no_grad():
            pb, pr = cb.posterior(Xt), cr.posterior(Xt)
        CHECK.check(
            "MultiTaskGP: conditioned posterior covariance (batched noise)",
            pr.distribution.covariance_matrix,
            pb.distribution.covariance_matrix,
            VALUE_TOL,
        )
        # Fantasies of fixed-noise models: NotImplementedError (BoTorch: with
        # several output tasks, an IndexError first; see the draft).
        bt_error = IndexError if bt.num_outputs > 1 else NotImplementedError
        with pytest.raises(bt_error):
            bt.fantasize(Xn, BtSampler(sample_shape=torch.Size([3]), seed=0))
        with pytest.raises(NotImplementedError):
            rs.fantasize(Xn, RsSampler(sample_shape=torch.Size([3]), seed=0))
        return
    # Fantasies with BoTorch's base samples.
    sb = BtSampler(sample_shape=torch.Size([3]), seed=seed)
    fb = bt.fantasize(Xn, sb)
    sr = RsSampler(sample_shape=torch.Size([3]), seed=0)
    sr.base_samples = sb.base_samples.clone()
    fr = rs.fantasize(Xn, sr)
    assert tuple(fr.batch_shape) == tuple(fb.batch_shape)
    with torch.no_grad():
        pb, pr = fb.posterior(Xt), fr.posterior(Xt)
    CHECK.check("MultiTaskGP: fantasy posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "MultiTaskGP: fantasy posterior covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )


def _dense_conditioned_posterior(bt: Any, Xn: Any, Yn: Any, noise: Any, Xt: Any) -> tuple[Any, Any]:
    """The exact posterior over the output tasks at ``Xt`` of BoTorch's model
    conditioned on ``(Xn, Yn)`` with the noise variances ``noise``
    (transformed space) of the new observations, computed densely."""
    from botorch.models.utils.assorted import _make_X_full

    def covar_inputs(X: Any) -> Any:
        xb, ti, xa = bt._split_inputs(X)
        return torch.cat([xb, ti.to(X.dtype), xa], -1), ti.reshape(-1).long()

    Xtr = bt.train_inputs[0]
    Xall = torch.cat([Xtr, Xn])
    xc, tasks = covar_inputs(Xall)
    K = bt.covar_module(xc).to_dense()
    task_noise = bt.likelihood.noise.reshape(-1)
    n = Xtr.shape[0]
    nz = torch.cat([task_noise[tasks[:n]], noise.reshape(-1)])
    K = K + torch.diag(nz)
    consts = torch.stack([bm.constant.reshape(()) for bm in bt.mean_module.base_means])
    ot = bt.outcome_transform
    y_new = ((Yn - ot.means) / ot.stdvs).reshape(-1)
    y = torch.cat([bt.train_targets, y_new]) - consts[tasks]
    Xf = _make_X_full(Xt, bt._output_tasks, bt._task_feature)
    xs, ts = covar_inputs(Xf)
    Ks = bt.covar_module(xs, xc).to_dense()
    Kss = bt.covar_module(xs).to_dense()
    mean = consts[ts] + Ks @ torch.linalg.solve(K, y)
    cov = Kss - Ks @ torch.linalg.solve(K, Ks.T)
    sd, mu = ot.stdvs.reshape(()), ot.means.reshape(())
    q, m = Xt.shape[0], len(bt._output_tasks)
    return (mu + sd * mean).reshape(m, q).T, sd * sd * cov


def test_mtgp_conditioning_with_explicit_noise_matches_the_exact_posterior() -> None:
    """Explicit noise of new observations of a model with learned noise
    enters the posterior mean and covariance (BoTorch main uses it for the
    covariance only; meta-pytorch/botorch#3396)."""
    bt, rs, case, _ = mtgp_pair("default_t3_d2")
    randomize(bt, rs, 17)
    Xt = h.test_points(case, 3, 18)
    bt.posterior(Xt)
    rs.posterior(Xt)
    Xn = h.test_points(case, 2, 19, with_tasks=True)
    Yn = torch.tensor([[2.4], [1.3]], dtype=DT)
    noise = torch.tensor([[0.02], [0.3]], dtype=DT)
    cr = rs.condition_on_observations(Xn, Yn, noise=noise)
    pr = cr.posterior(Xt)
    with torch.no_grad():
        mean, cov = _dense_conditioned_posterior(bt, Xn, Yn, noise, Xt)
    CHECK.check("MultiTaskGP: conditioned (explicit noise) mean vs exact", pr.mean, mean, VALUE_TOL)
    CHECK.check(
        "MultiTaskGP: conditioned (explicit noise) covariance vs exact",
        pr.distribution.covariance_matrix,
        cov,
        VALUE_TOL,
    )


def _twin_x(X0: Any, case: dict[str, Any], t: int) -> Any:
    if t == 0:
        return X0
    g = torch.Generator().manual_seed(t)
    rel = TWIN_SCALES[(t - 1) % len(TWIN_SCALES)]
    X = X0.clone()
    tf = case["task_feature"] % (case["d"] + 1)
    cols = [c for c in range(X.shape[-1]) if c != tf]
    X[:, cols] = X[:, cols] * (1 + rel * torch.randn(X[:, cols].shape, generator=g, dtype=DT))
    return X


def _bt_fit(model: Any) -> tuple[float, int]:
    import botorch.optim.fit as bt_optim_fit

    record: dict[str, Any] = {"attempts": 0}
    orig = bt_optim_fit.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    bt_fit(BtMLL(model.likelihood, model), optimizer=recorder)
    return float(record["r"].fval), record["attempts"]


def _rs_fit(model: Any) -> tuple[float, int]:
    mll = RsMLL(model.likelihood, model)
    rs_fit(mll)
    return float(mll._fit_result["fval"]), int(mll._fit_result["attempts"])


@pytest.mark.parametrize("name", [n for n, c in MTGP.items() if c.get("fit")])
def test_mtgp_fit_from_identical_initial_values(name: str) -> None:
    case = MTGP[name]
    X0 = h.make_data(case, _seed(name, MTGP))[0]

    def fit_twin(t: int) -> tuple[FitOutcome, FitOutcome]:
        bt, rs, _, _ = mtgp_pair(name, X=_twin_x(X0, case, t))
        return _bt_fit(bt), _rs_fit(rs)

    compare_twin_fits(CHECK, "MultiTaskGP fit", fit_twin, N_TWINS)


def _mtgp_twin(bt: Any, case: dict[str, Any], k: int) -> Any:
    if k == 0:
        return bt
    tw = copy.deepcopy(bt)
    X = _twin_x(tw.train_inputs[0], case, k)
    tw.set_train_data(inputs=(X,), strict=False)
    tw.prediction_strategy = None
    tw.eval()
    return tw


@pytest.mark.parametrize("q", [1, 2])
def test_mtgp_acquisition_functions(q: int) -> None:
    bt, rs, case, (X, _, _) = mtgp_pair("single_output")
    seed = _seed("single_output", MTGP)
    randomize(bt, rs, seed + 1)
    bt.eval()
    rs.eval()
    tf = case["task_feature"] % (case["d"] + 1)
    Xb = torch.cat([X[:, :tf], X[:, tf + 1 :]], -1)
    Xc = torch.rand(3, q, case["d"], generator=torch.Generator().manual_seed(seed + q), dtype=DT)
    n_dim = Xb.shape[0] + q

    def outcome(k: int) -> tuple[Any, Any]:
        with synced_draws(seed + 10 + q):
            acqf = BtQLogNEI(
                _mtgp_twin(bt, case, k),
                X_baseline=Xb,
                sampler=bt_sampler(NUM_SAMPLES),
                prune_baseline=False,
                cache_root=False,
            )
            return _vg(acqf, Xc)

    acqf_r = RsQLogNEI(
        rs,
        X_baseline=Xb,
        sampler=rs_sampler(NUM_SAMPLES, seed + 10 + q, n_dim),
        prune_baseline=False,
    )
    check_against_twins(CHECK, f"MultiTaskGP: qLogNEI q={q}", _vg(acqf_r, Xc), outcome)
    best = torch.tensor(2.0, dtype=DT)
    with synced_draws(seed + 20 + q):
        vb, gb = _vg(BtQEI(bt, best_f=best, sampler=bt_sampler(NUM_SAMPLES)), Xc)
    vr, gr = _vg(RsQEI(rs, best_f=best, sampler=rs_sampler(NUM_SAMPLES, seed + 20 + q, q)), Xc)
    CHECK.check("MultiTaskGP: qEI values", vr, vb, VALUE_TOL)
    check_grad("MultiTaskGP: qEI gradients", gr, gb)
    with synced_draws(seed + 30 + q):
        vb, gb = _vg(BtQUCB(bt, beta=0.4, sampler=bt_sampler(NUM_SAMPLES)), Xc)
    vr, gr = _vg(RsQUCB(rs, beta=0.4, sampler=rs_sampler(NUM_SAMPLES, seed + 30 + q, q)), Xc)
    CHECK.check("MultiTaskGP: qUCB values", vr, vb, VALUE_TOL)
    check_grad("MultiTaskGP: qUCB gradients", gr, gb)
    X1 = Xc.reshape(-1, 1, case["d"])
    for name_, ab, ar in (
        ("LogEI", BtLogEI(bt, best_f=best), RsLogEI(rs, best_f=best)),
        ("PosteriorMean", BtPM(bt), RsPM(rs)),
        ("PosteriorStandardDeviation", BtPSD(bt), RsPSD(rs)),
    ):
        vb, gb = _vg(ab, X1)
        vr, gr = _vg(ar, X1)
        CHECK.check(f"MultiTaskGP: {name_} values", vr, vb, VALUE_TOL)
        check_grad(f"MultiTaskGP: {name_} gradients", gr, gb)


def test_mtgp_model_list_and_qlognehvi() -> None:
    """A ``ModelListGP`` of two (single-output) multi-task GPs with different
    targets: posterior and qLogNEHVI."""
    bt1, rs1, case, (X, Y, Yvar) = mtgp_pair("single_output")
    seed = _seed("single_output", MTGP)
    randomize(bt1, rs1, seed + 1)
    torch.manual_seed(seed + 5)
    bt2 = BtMTGP(X, -Y + 0.3 * X[:, :1], **h.model_kwargs("bt", case, Yvar))
    torch.manual_seed(seed + 5)
    rs2 = RsMTGP(X, -Y + 0.3 * X[:, :1], **h.model_kwargs("rs", case, Yvar))
    randomize(bt2, rs2, seed + 2)
    lb, lr = BtModelListGP(bt1, bt2), RsModelListGP(rs1, rs2)
    lb.eval()
    Xt = h.test_points(case, 3, seed + 3)
    with torch.no_grad():
        pb = lb.posterior(Xt)
    pr = lr.posterior(Xt)
    CHECK.check("MultiTaskGP: ModelListGP posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "MultiTaskGP: ModelListGP posterior covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    tf = case["task_feature"] % (case["d"] + 1)
    Xb = torch.cat([X[:, :tf], X[:, tf + 1 :]], -1)
    with torch.no_grad():
        ref = (lb.posterior(Xb).mean.min(0).values - 0.1).tolist()
    Xc = torch.rand(3, 1, case["d"], generator=torch.Generator().manual_seed(seed + 4), dtype=DT)

    def outcome(k: int) -> tuple[Any, Any]:
        model = BtModelListGP(_mtgp_twin(bt1, case, k), _mtgp_twin(bt2, case, k))
        with synced_draws(seed + 40):
            acqf = BtQLogNEHVI(
                model,
                ref_point=ref,
                X_baseline=Xb,
                sampler=bt_sampler(NUM_SAMPLES),
                prune_baseline=False,
            )
            return _vg(acqf, Xc)

    acqf_r = RsQLogNEHVI(
        lr,
        ref_point=ref,
        X_baseline=Xb,
        sampler=rs_sampler(NUM_SAMPLES, seed + 40, 2 * (Xb.shape[0] + 1)),
        prune_baseline=False,
    )
    check_against_twins(CHECK, "MultiTaskGP: qLogNEHVI (ModelListGP)", _vg(acqf_r, Xc), outcome)


# ---------------------------------------------------------------------------
# KroneckerMultiTaskGP
# ---------------------------------------------------------------------------


def kron_pair(name: str, X: Any = None) -> tuple[Any, Any, dict[str, Any], tuple[Any, Any]]:
    case = KRON[name]
    seed = _seed(name, KRON)
    X0, Y = h.kron_data(case, seed)
    bt, rs = h.build_kron_pair(case, X0 if X is None else X, Y, seed)
    return bt, rs, case, (X0, Y)


@pytest.mark.parametrize("name", list(KRON))
def test_kronecker_structure_mll_and_posterior(name: str) -> None:
    bt, rs, case, _ = kron_pair(name)
    seed = _seed(name, KRON)
    nb = [(n, tuple(p.shape)) for n, p in bt.named_parameters()]
    nr = [(n, tuple(torch.as_tensor(p).shape)) for n, p in rs.named_parameters()]
    assert nb == nr
    for (_, pb), (_, pr) in zip(bt.named_parameters(), rs.named_parameters(), strict=True):
        CHECK.check(
            "Kronecker: default raw parameters (incl. random factors)", pr, pb.detach(), 0.0
        )
    assert [list(b) for b in rs._rs.raw_parameter_bounds()] == h.bounds_of(bt)
    vb, gb = h.bt_mll_value_and_grad(bt)
    vr, gr = rs_mll(rs)
    CHECK.check("Kronecker: default MLL", vr, vb, VALUE_TOL)
    check_grad("Kronecker: default MLL gradient", gr, gb)
    randomize(bt, rs, seed + 1)
    vb, gb = h.bt_mll_value_and_grad(bt)
    vr, gr = rs_mll(rs)
    CHECK.check("Kronecker: MLL", vr, vb, VALUE_TOL)
    check_grad("Kronecker: MLL gradient", gr, gb)
    bt.eval()
    X = torch.rand(3, case["d"], generator=torch.Generator().manual_seed(seed + 2), dtype=DT)
    q, m = 3, case["m"]
    for noise in (False, True):
        with torch.no_grad():
            pb = bt.posterior(X, observation_noise=noise)
            mean, cov = h.kron_exact_posterior(bt, X, noise)
        pr = rs.posterior(X, observation_noise=noise)
        tag = f"Kronecker: posterior{' (noise)' if noise else ''}"
        CHECK.check(f"{tag} mean (BoTorch)", pr.mean, pb.mean, VALUE_TOL)
        CHECK.check(f"{tag} mean (exact)", pr.mean, mean, VALUE_TOL)
        CHECK.check(f"{tag} variance (exact)", pr.variance, cov.diagonal().reshape(q, m), VALUE_TOL)
        assert tuple(pr.base_sample_shape) == tuple(pb.base_sample_shape)
        # BoTorch's Matheron samples have the exact covariance; so do ours.
        with torch.no_grad():
            _, cb = h.matheron_covariance(
                pb,
                lambda Z, p=pb: p.rsample_from_base_samples(
                    torch.Size([Z.shape[0]]), Z.unsqueeze(-1)
                ),
            )
        mr, cr = h.matheron_covariance(
            pr, lambda Z, p=pr: p.rsample_from_base_samples(torch.Size([Z.shape[0]]), Z)
        )
        CHECK.check(f"{tag}: BoTorch's Matheron covariance vs exact", cb, cov, VALUE_TOL)
        CHECK.check(f"{tag}: botorch_rs's Matheron covariance vs exact", cr, cov, VALUE_TOL)
        CHECK.check(
            f"{tag}: botorch_rs's Matheron mean vs exact", mr.reshape(q, m), mean, VALUE_TOL
        )
        if not noise or case.get("likelihood_rank", 0) == 0:
            d = torch.as_tensor(pr.distribution.covariance_matrix)
            assert torch.equal(d, torch.diag(d.diagonal())), "a diagonal (marginal) covariance"
        assert pr.distribution._interleaved
    # Samples drawn by rsample have the posterior's shape.
    s = torch.as_tensor(rs.posterior(X).rsample(torch.Size([5])))
    assert tuple(s.shape) == (5, q, m)


def test_kronecker_variance_with_equal_task_noises_matches_botorch() -> None:
    """With equal task noises (homoskedastic noise), BoTorch's closed-form
    posterior variance is exact and agrees with botorch_rs's."""
    bt, rs, case, _ = kron_pair("default_m3_d2")
    randomize(bt, rs, 3)
    with torch.no_grad():
        bt.likelihood.raw_task_noises.fill_(0.07)
    rs.initialize(**{"likelihood.raw_task_noises": torch.full((case["m"],), 0.07, dtype=DT)})
    bt.eval()
    X = torch.rand(4, case["d"], generator=torch.Generator().manual_seed(4), dtype=DT)
    with torch.no_grad():
        pb = bt.posterior(X)
    CHECK.check(
        "Kronecker: variance (equal task noises) vs BoTorch",
        rs.posterior(X).variance,
        pb.variance,
        VALUE_TOL,
    )


@pytest.mark.parametrize("name", [n for n, c in KRON.items() if c.get("fit")])
def test_kronecker_fit_from_identical_initial_values(name: str) -> None:
    X0 = kron_pair(name)[3][0]

    def fit_twin(t: int) -> tuple[FitOutcome, FitOutcome]:
        X = X0
        if t > 0:
            g = torch.Generator().manual_seed(t)
            X = X0 * (1 + TWIN_SCALES[(t - 1) % 4] * torch.randn(X0.shape, generator=g, dtype=DT))
        bt, rs, _, _ = kron_pair(name, X=X)
        return _bt_fit(bt), _rs_fit(rs)

    compare_twin_fits(CHECK, "Kronecker fit", fit_twin, N_TWINS)


@pytest.mark.parametrize("name", list(KRON))
def test_kronecker_acquisition_functions(name: str) -> None:
    """qEI with a linear objective and qLogNEHVI against BoTorch's
    acquisition code on the exact joint posterior, with identical base
    samples (botorch_rs samples the exact joint posterior with its Cholesky
    root; BoTorch samples by Matheron's rule from base samples of another
    shape)."""
    bt, rs, case, (X, _) = kron_pair(name)
    seed = _seed(name, KRON)
    randomize(bt, rs, seed + 1)
    bt.eval()
    exact = h.ExactKroneckerModel(bt)
    m, d = case["m"], case["d"]
    w = torch.tensor([0.6, -0.3, 0.5][:m], dtype=DT)
    Xc = torch.rand(3, 2, d, generator=torch.Generator().manual_seed(seed + 2), dtype=DT)
    best = torch.tensor(0.2, dtype=DT)
    with synced_draws(seed + 3):
        vb, gb = _vg(
            BtQEI(exact, best_f=best, sampler=bt_sampler(NUM_SAMPLES), objective=BtLinear(w)), Xc
        )
    vr, gr = _vg(
        RsQEI(
            rs, best_f=best, sampler=rs_sampler(NUM_SAMPLES, seed + 3, 2 * m), objective=RsLinear(w)
        ),
        Xc,
    )
    CHECK.check("Kronecker: qEI (linear objective) values", vr, vb, VALUE_TOL)
    check_grad("Kronecker: qEI (linear objective) gradients", gr, gb)
    if m != 2:
        return
    with torch.no_grad():
        ref = (exact.posterior(X).mean.min(0).values - 0.1).tolist()
    X1 = Xc[:, :1]
    with synced_draws(seed + 4):
        acqf = BtQLogNEHVI(
            exact,
            ref_point=ref,
            X_baseline=X,
            sampler=bt_sampler(NUM_SAMPLES),
            prune_baseline=False,
        )
        vb, gb = _vg(acqf, X1)
    acqf_r = RsQLogNEHVI(
        rs,
        ref_point=ref,
        X_baseline=X,
        sampler=rs_sampler(NUM_SAMPLES, seed + 4, m * (X.shape[0] + 1)),
        prune_baseline=False,
    )
    vr, gr = _vg(acqf_r, X1)
    CHECK.check_log("Kronecker: qLogNEHVI values", vr, vb, VALUE_TOL)
    check_grad("Kronecker: qLogNEHVI gradients", gr, gb)


# ---------------------------------------------------------------------------
# BoTorch's bugs
# ---------------------------------------------------------------------------


def test_botorch_bugs() -> None:
    """The bugs fixed in the reference: BoTorch differs from botorch_rs (and
    the fixed reference) by the recorded amounts."""
    # 1. Per-task noise looked up by raw task value.
    case = MTGP["noncontiguous_tf1"]
    X, Y, Yvar = h.make_data(case, 1)
    torch.manual_seed(2)
    plain = BtMTGP(X, Y, **h.model_kwargs("bt", case, Yvar))
    torch.manual_seed(2)
    rs = RsMTGP(X, Y, **h.model_kwargs("rs", case, Yvar))
    vb, _ = h.bt_mll_value_and_grad(plain)
    vr, _ = rs_mll(rs)
    diff = abs(vb - vr)
    CHECK.record("bug: MultiTaskGP noise by raw task value (MLL change)", diff, None)
    assert diff > 1e-3
    # 2. LKJCovariancePrior on the correlations: BoTorch fails.
    case = MTGP["lkj_prior"]
    X, Y, Yvar = h.make_data(case, 3)
    from gpytorch.priors import GammaPrior, LKJCovariancePrior

    bad = BtMTGP(
        X, Y, task_feature=-1, task_covar_prior=LKJCovariancePrior(3, 1.5, GammaPrior(2.0, 1.5))
    )
    with pytest.raises(IndexError):
        h.bt_mll_value_and_grad(bad)
    # 3. A ProductKernel covar_module: its active_dims are flattened away.
    case = MTGP["product_tf0"]
    X, Y, Yvar = h.make_data(case, 4)
    torch.manual_seed(5)
    plain = BtMTGP(X, Y, **h.model_kwargs("bt", case, Yvar))
    torch.manual_seed(5)
    rs = RsMTGP(X, Y, **h.model_kwargs("rs", case, Yvar))
    vb, _ = h.bt_mll_value_and_grad(plain)
    vr, _ = rs_mll(rs)
    CHECK.record("bug: MultiTaskGP ProductKernel active_dims (MLL change)", abs(vb - vr), None)
    assert abs(vb - vr) > 1e-6
    # 4. KroneckerMultiTaskGP's variance with task-dependent noise.
    bt, rs, kcase, _ = kron_pair("default_m3_d2")
    randomize(bt, rs, 6)
    bt.eval()
    Xt = torch.rand(3, kcase["d"], generator=torch.Generator().manual_seed(7), dtype=DT)
    with torch.no_grad():
        pb = bt.posterior(Xt)
        _, cov = h.kron_exact_posterior(bt, Xt)
    err = rel_err(pb.variance, cov.diagonal().reshape(pb.variance.shape))
    CHECK.record("bug: Kronecker variance with task-dependent noise (relative)", err, None)
    assert err > 1e-3
    # 5. construct_inputs with an LKJ prior_config counts the values of a row.
    from botorch.utils.datasets import SupervisedDataset

    case = MTGP["default_t3_d2"]
    X, Y, _ = h.make_data(case, 8)
    X = torch.cat([X[:, :-1], X[:, :-1] ** 2, X[:, -1:]], -1)  # d = 4: a row has 5 values
    ds = SupervisedDataset(X, Y, feature_names=[f"x{i}" for i in range(5)], outcome_names=["y"])
    kb = BtMTGP.construct_inputs(ds, task_feature=-1, prior_config={"use_LKJ_prior": True})
    kr = RsMTGP.construct_inputs(ds, task_feature=-1, prior_config={"use_LKJ_prior": True})
    assert kr["task_covar_prior"].n == 3
    assert kb["task_covar_prior"].correlation_prior.n != 3


# ---------------------------------------------------------------------------
# construct_inputs
# ---------------------------------------------------------------------------


def test_construct_inputs_match_botorch() -> None:
    from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset

    case = MTGP["default_t3_d2"]
    X, Y, Yvar = h.make_data(case, 9)
    ds = SupervisedDataset(X, Y, Yvar=Yvar, feature_names=["a", "b", "t"], outcome_names=["y"])
    kb = BtMTGP.construct_inputs(ds, task_feature=-1, rank=2, output_tasks=[1])
    kr = RsMTGP.construct_inputs(ds, task_feature=-1, rank=2, output_tasks=[1])
    assert set(kb) == set(kr)
    for key in kb:
        if torch.is_tensor(kb[key]):
            assert torch.equal(kb[key], kr[key])
        else:
            assert kb[key] == kr[key]
    # A multi-task dataset (one dataset per task, the task as a feature).
    sets = []
    for t in range(3):
        rows = X[:, -1] == t
        sets.append(
            SupervisedDataset(
                X[rows], Y[rows], feature_names=["a", "b", "t"], outcome_names=[f"y{t}"]
            )
        )
    mds = MultiTaskDataset(sets, target_outcome_name="y0", task_feature_index=-1)
    kb = BtMTGP.construct_inputs(mds, task_feature=-1)
    kr = RsMTGP.construct_inputs(mds, task_feature=-1)
    assert set(kb) == set(kr)
    rs = RsMTGP(**kr)
    bt = BtMTGP(**kb)
    assert rs.all_tasks == sorted(set(int(v) for v in bt.train_inputs[0][:, -1].tolist()))


# ---------------------------------------------------------------------------
# Randomized sweep
# ---------------------------------------------------------------------------


def _random_mtgp_case(r: random.Random) -> dict[str, Any]:
    d = r.randint(1, 3)
    k = r.randint(2, 4)
    values = sorted(r.sample(range(0, 8), k)) if r.random() < 0.4 else list(range(k))
    observed = values if r.random() < 0.6 else values[:-1] + values[-1:][:0]
    if len(observed) < 2:
        observed = values
    case: dict[str, Any] = {
        "name": "sweep",
        "n": r.randint(8, 14),
        "d": d,
        "tasks": observed,
        "task_feature": r.choice([-1, 0]),
        "prior": r.choice(["default", "default", "none", "gamma"]),
    }
    if observed != values:
        case["all_tasks"] = values
    if r.random() < 0.5:
        case["rank"] = r.randint(1, k)
    if r.random() < 0.4:
        outs = r.sample(values, r.randint(1, k))
        case["output_tasks"] = outs
    if r.random() < 0.3:
        case["fixed_noise"] = True
    if r.random() < 0.5:
        case["covar"] = random_spec(r, d)
    return case


@pytest.mark.equivalence
@pytest.mark.parametrize("scenario", range(30))
def test_random_multitask_gp_sweep(scenario: int) -> None:
    r = random.Random(7700 + scenario)
    case = _random_mtgp_case(r)
    with zero_diagonal_distances(), h.exact_kernel_products():
        _mtgp_sweep_case(case, 7800 + scenario)


def _mtgp_sweep_case(case: dict[str, Any], seed: int) -> None:
    X, Y, Yvar = h.make_data(case, seed)
    try:
        bt, rs = h.build_pair(case, X, Y, Yvar, seed)
    except Exception as e:  # pragma: no cover - reported
        pytest.fail(f"construction failed: {e!r} for {case}")
    randomize(bt, rs, seed + 1)
    try:
        vb, gb = h.bt_mll_value_and_grad(bt)
    except RuntimeError:  # NotPSDError: botorch_rs must fail alike
        with pytest.raises(RuntimeError):
            rs_mll(rs)
        return
    vr, gr = rs_mll(rs)
    CHECK.check("sweep: MultiTaskGP MLL", vr, vb, VALUE_TOL)
    check_grad("sweep: MultiTaskGP MLL gradient", gr, gb)
    bt.eval()
    rs.eval()
    # BoTorch's observation noise of fixed-noise models is NaN for tasks
    # without data (the mean of no noise levels; see the draft).
    unobserved = set(rs.all_tasks) - set(case["tasks"])
    outputs = case.get("output_tasks", rs.all_tasks)
    noise = not (case.get("fixed_noise") and unobserved & set(outputs))
    for X_ in (h.test_points(case, 3, seed + 2), h.test_points(case, 3, seed + 3, with_tasks=True)):
        with torch.no_grad():
            pb = bt.posterior(X_, observation_noise=noise)
        pr = rs.posterior(X_, observation_noise=noise)
        CHECK.check("sweep: MultiTaskGP posterior mean", pr.mean, pb.mean, VALUE_TOL)
        CHECK.check(
            "sweep: MultiTaskGP posterior covariance",
            pr.distribution.covariance_matrix,
            pb.distribution.covariance_matrix,
            VALUE_TOL,
        )
    m = rs.num_outputs
    w = torch.linspace(0.7, -0.5, m, dtype=DT)
    X1 = h.test_points(case, 3, seed + 4).unsqueeze(-2)
    vb, gb = _vg(BtPM(bt, posterior_transform=BtSPT(w)), X1)
    vr, gr = _vg(RsPM(rs, posterior_transform=RsSPT(w)), X1)
    CHECK.check("sweep: MultiTaskGP PosteriorMean values", vr, vb, VALUE_TOL)
    check_grad("sweep: MultiTaskGP PosteriorMean gradients", gr, gb)


@pytest.mark.equivalence
@pytest.mark.parametrize("scenario", range(16))
def test_random_kronecker_sweep(scenario: int) -> None:
    r = random.Random(8800 + scenario)
    m = r.randint(2, 4)
    d = r.randint(1, 3)
    case: dict[str, Any] = {
        "name": "sweep",
        "n": r.randint(6, 10),
        "d": d,
        "m": m,
        "rank": r.randint(1, m),
        "likelihood_rank": r.choice([0, 0, r.randint(1, m)]),
        "standardize": r.random() < 0.5,
        "normalize": r.random() < 0.3,
    }
    if r.random() < 0.4:
        case["covar"] = random_spec(r, d)
    seed = 8900 + scenario
    X, Y = h.kron_data(case, seed)
    bt, rs = h.build_kron_pair(case, X, Y, seed)
    randomize(bt, rs, seed + 1)
    try:
        vb, gb = h.bt_mll_value_and_grad(bt)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            rs_mll(rs)
        return
    vr, gr = rs_mll(rs)
    CHECK.check("sweep: Kronecker MLL", vr, vb, VALUE_TOL)
    check_grad("sweep: Kronecker MLL gradient", gr, gb)
    bt.eval()
    Xt = torch.rand(3, d, generator=torch.Generator().manual_seed(seed + 2), dtype=DT)
    with torch.no_grad():
        pb = bt.posterior(Xt)
        mean, cov = h.kron_exact_posterior(bt, Xt)
    pr = rs.posterior(Xt)
    CHECK.check("sweep: Kronecker posterior mean (BoTorch)", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "sweep: Kronecker posterior variance (exact)",
        pr.variance,
        cov.diagonal().reshape(3, m),
        VALUE_TOL,
    )
