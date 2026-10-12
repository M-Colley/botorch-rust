"""Parity of ``Interval`` constraints and of the Unity project's contextual
models (``LCEMGP``, ``ProvidedEmbeddingLCEMGP``) with BoTorch, through the
public APIs of both libraries side by side.

The references are the contextual models of the Unity project BOforUnity on
BoTorch, built exactly as its ``context_support.build_contextual_model``
builds them (``lcemgp_reference.py``, a copy of its classes and of that
function): ``learned`` embeddings (BoTorch's ``LCEMGP`` with the Unity
project's ``eval()``), ``provided`` embeddings (its
``_ProvidedEmbeddingLCEMGP``: no learned embedding, a context-specific
weight) and, less important, ``stock`` (learned plus fixed embeddings with
the task kernel resized to both).

* ``Interval``: values, inverses, derivatives, raw initial values, stored and
  optimizer bounds, errors; ``SingleTaskGP``s with ``Interval`` constraints on
  the lengthscales (box bounds or sigmoid), the outputscale and the noise:
  initialization, MLL values and gradients, fits from identical initial
  values.
* Unity models with K = 2, 3, 5 contexts (some without data), m = 1, 3 fixed
  embedding dimensions and d = 1, ..., 4 inputs: parameter names, shapes and
  values; MLL values and gradients; the in-place renormalization of the
  embedding weights by forward passes; ``eval()`` (constant means of
  unobserved contexts); posteriors over the output task (with observation
  noise), over all tasks, at inputs with a task column and batched;
  ``ModelListGP`` with ``SumMarginalLogLikelihood``; fits from identical
  initial values; qLogNEI and qLogNEHVI values and gradients with identical
  base samples; ``optimize_acqf`` from identical initial conditions.

Tolerances: posteriors and MLL values 1e-9, gradients 1e-8; fits: the final
loss must not exceed BoTorch's by more than ``max(1e-6, 10 x`` BoTorch's own
change of its final loss when the training inputs are perturbed by 1e-14 or
L-BFGS-B's variables are reversed``)`` (long L-BFGS-B runs on flat
likelihoods amplify rounding, and botorch_rs's L-BFGS-B port and SciPy's,
which uses BLAS, agree only to rounding); MC acquisition values / gradients 1e-9 / 1e-8 or 10x BoTorch's own
change when the training inputs are perturbed by 1e-14 (errors scaled to the
nominal tolerance). The worst errors are listed in the terminal summary
(section "LCEMGP (Unity contextual models)"). Skipped unless BoTorch is
importable; the randomized sweep is marked ``equivalence``.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import gpytorch  # noqa: E402
from botorch.acquisition.logei import (  # noqa: E402
    qLogNoisyExpectedImprovement as BtQLogNEI,
)
from botorch.acquisition.multi_objective.logei import (  # noqa: E402
    qLogNoisyExpectedHypervolumeImprovement as BtQLogNEHVI,
)
from botorch.fit import fit_gpytorch_mll as bt_fit  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.optim import optimize as bt_optimize  # noqa: E402
from botorch.optim.utils import get_parameters_and_bounds  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from lcemgp_reference import build_contextual_model, build_model, mll_parameter_names  # noqa: E402
from mo_acquisition_helpers import bt_sampler, rs_sampler, synced_draws  # noqa: E402

from botorch_rs import constraints as rs_constraints  # noqa: E402
from botorch_rs import kernels as rs_kernels  # noqa: E402
from botorch_rs.acquisition import qLogNoisyExpectedImprovement as RsQLogNEI  # noqa: E402
from botorch_rs.acquisition.multi_objective import (  # noqa: E402
    qLogNoisyExpectedHypervolumeImprovement as RsQLogNEHVI,
)
from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.likelihoods import GaussianLikelihood as RsGaussianLikelihood  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models.contextual_multioutput import (  # noqa: E402
    LCEMGP,
    ProvidedEmbeddingLCEMGP,
)
from botorch_rs.models.multitask import MultiTaskGP  # noqa: E402
from botorch_rs.optim import optimize_acqf as rs_optimize_acqf  # noqa: E402
from parity_helpers import Checker, log_err, rel_err  # noqa: E402

DT = torch.float64
CHECK = Checker("LCEMGP (Unity contextual models)")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
FIT_TOL = 1e-6
F32_ULP = 2.4e-7
NUM_SAMPLES = 16


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

INTERVALS = [
    (0.0, 2.0, None, 1.0),
    (0.1, 3.0, "sigmoid", 1.0),
    (0.05, 4.0, "sigmoid", None),
    (-1.0, 1.5, "softplus", 0.2),
    (1e-4, 0.5, None, None),
]


def _gp_interval(lo: float, hi: float, transform: Any, init: Any) -> tuple[Any, Any]:
    kw: dict[str, Any] = {"initial_value": init}
    if transform is None:
        kw.update(transform=None)
        rs_kw = dict(kw)
    elif transform == "softplus":
        kw.update(transform=torch.nn.functional.softplus)
        rs_kw = {"initial_value": init, "transform": rs_constraints.softplus}
    else:
        rs_kw = dict(kw)
    return gpytorch.constraints.Interval(lo, hi, **kw), rs_constraints.Interval(lo, hi, **rs_kw)


@pytest.mark.parametrize("case", INTERVALS, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}-{c[3]}")
def test_interval_values(case: tuple) -> None:
    lo, hi, transform, init = case
    bt, rs = _gp_interval(lo, hi, transform, init)
    assert rs.lower_bound == bt.lower_bound.item() and rs.upper_bound == bt.upper_bound.item()
    assert rs.enforced == bt.enforced
    bt = bt.to(DT)
    raws = [-3.0, -0.4, 0.0, 0.7, 2.5]
    x = torch.tensor(raws, dtype=DT, requires_grad=True)
    v = bt.transform(x)
    (dv,) = torch.autograd.grad(v.sum(), x)
    CHECK.check("Interval transform", [rs.transform(r) for r in raws], v.detach(), 1e-15)
    CHECK.check("Interval derivative", [rs._rs.transform_derivative(r) for r in raws], dv, 1e-15)
    vals = [lo + f * (hi - lo) for f in (0.1, 0.5, 0.93)]
    CHECK.check(
        "Interval inverse",
        [rs.inverse_transform(t) for t in vals],
        bt.inverse_transform(torch.tensor(vals, dtype=DT)),
        1e-12,
    )
    if init is None:
        assert rs.initial_value is None and bt.initial_value is None
    else:
        e = bt.initial_value.item()
        CHECK.check("Interval raw initial value (float32 ulps)", [rs.initial_value], [e], F32_ULP)
    lob, hib = bt.inverse_transform(bt.lower_bound), bt.inverse_transform(bt.upper_bound)
    assert rs._rs.raw_bounds == (lob.item(), hib.item())
    assert rs.check(vals[1]) and not rs.check(hi + 1.0)
    assert repr(rs).startswith("Interval(")


def test_interval_errors() -> None:
    for lo, hi in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError, match="empty intervals"):
            gpytorch.constraints.Interval(lo, hi)
        with pytest.raises(ValueError, match="empty intervals"):
            rs_constraints.Interval(lo, hi)
    with pytest.raises(ValueError, match="non-finite bounds"):
        rs_constraints.Interval(0.0, math.inf)
    with pytest.raises(UnsupportedError):
        rs_constraints.Interval(0.0, 1.0, transform=torch.exp)
    assert isinstance(rs_constraints.GreaterThan(0.1), rs_constraints.Interval)


INTERVAL_MODELS = ["lengthscale_box", "lengthscale_sigmoid", "scale_noise"]


def _interval_gp_pair(kind: str, seed: int) -> tuple[Any, Any]:
    g = torch.Generator().manual_seed(seed)
    d = 2 + INTERVAL_MODELS.index(kind)
    X = torch.rand(14, d, generator=g, dtype=DT)
    Y = torch.sin(4 * X[:, :1]) + X[:, 1:].pow(2).sum(-1, keepdim=True)
    Y = Y + 0.05 * torch.randn(14, 1, generator=g, dtype=DT)
    modules = []
    for lib in (gpytorch, rs_kernels):
        C = gpytorch.constraints if lib is gpytorch else rs_constraints
        RBF = gpytorch.kernels.RBFKernel if lib is gpytorch else rs_kernels.RBFKernel
        Scale = gpytorch.kernels.ScaleKernel if lib is gpytorch else rs_kernels.ScaleKernel
        Lik = gpytorch.likelihoods.GaussianLikelihood if lib is gpytorch else RsGaussianLikelihood
        if kind == "lengthscale_box":
            covar = RBF(
                ard_num_dims=d,
                lengthscale_constraint=C.Interval(0.05, 2.0, transform=None, initial_value=0.7),
            )
            lik = None
        elif kind == "lengthscale_sigmoid":
            covar = RBF(
                ard_num_dims=d, lengthscale_constraint=C.Interval(0.05, 3.0, initial_value=0.9)
            )
            lik = None
        else:
            covar = Scale(
                RBF(ard_num_dims=d, lengthscale_constraint=C.Interval(0.02, 4.0)),
                outputscale_constraint=C.Interval(0.1, 5.0, transform=None, initial_value=1.0),
            )
            lik = Lik(noise_constraint=C.Interval(1e-4, 0.5, transform=None, initial_value=0.05))
        modules.append((covar, lik))
    (cb, lb), (cr, lr) = modules
    bt = BtSingleTaskGP(X, Y, covar_module=cb, likelihood=lb)
    rs = RsSingleTaskGP(X, Y, covar_module=cr, likelihood=lr)
    return bt, rs


@pytest.mark.parametrize("kind", INTERVAL_MODELS)
def test_interval_single_task_gp(kind: str) -> None:
    bt, rs = _interval_gp_pair(kind, 3 + INTERVAL_MODELS.index(kind))
    names_b = [n for n, _ in bt.named_parameters()]
    assert names_b == [n for n, _ in rs.named_parameters()]
    for (n, pb), (_, pr) in zip(bt.named_parameters(), rs.named_parameters(), strict=True):
        CHECK.check(
            f"SingleTaskGP with Interval: initial {n.split('.')[-1]}", pr, pb.detach(), F32_ULP
        )
    _, bounds = get_parameters_and_bounds(BtMLL(bt.likelihood, bt))
    expected = []
    for n, p in bt.named_parameters():
        key = n if n.startswith("likelihood") else f"model.{n}"
        lo, hi = bounds.get(key, (-math.inf, math.inf))
        expected += [(float(lo), float(hi))] * p.numel()
    assert [tuple(b) for b in rs._rs.raw_parameter_bounds()] == expected
    # MLL at the (copied) initial values and fits from them.
    rs.initialize(**{n: p.detach() for n, p in bt.named_parameters()})
    bt.train()
    mll_b = BtMLL(bt.likelihood, bt)
    out = mll_b(bt(*bt.train_inputs), bt.train_targets)
    out.backward()
    grad_b = [g for _, p in bt.named_parameters() for g in p.grad.reshape(-1).tolist()]
    bt.zero_grad()
    value_r, grad_r = RsMLL(rs.likelihood, rs).value_and_grad()
    CHECK.check("SingleTaskGP with Interval: MLL", [value_r], [out.item()], VALUE_TOL)
    CHECK.check("SingleTaskGP with Interval: MLL gradient", grad_r, grad_b, GRAD_TOL)
    record: dict[str, Any] = {}
    import botorch.optim.fit as bt_optim_fit

    orig = bt_optim_fit.fit_gpytorch_mll_scipy

    def recorder(*args: Any, **kwargs: Any) -> Any:
        record["r"] = orig(*args, **kwargs)
        return record["r"]

    twins = []
    for rep in range(3):
        twin = copy.deepcopy(bt)
        g = torch.Generator().manual_seed(rep + 1)
        X0 = twin.train_inputs[0]
        twin.train_inputs = (X0 * (1 + 1e-14 * torch.randn(X0.shape, generator=g, dtype=DT)),)
        twin.prediction_strategy = None
        twins.append(twin)
    reordered = copy.deepcopy(bt)
    bt_fit(BtMLL(bt.likelihood, bt), optimizer=recorder)
    loss_b = float(record["r"].fval)
    floor = max(abs(_bt_fit_loss(t) - loss_b) for t in twins)
    floor = max(floor, abs(_bt_fit_loss(reordered, reverse=True) - loss_b))
    mll_r = RsMLL(rs.likelihood, rs)
    rs_fit(mll_r)
    excess = mll_r._fit_result["fval"] - loss_b
    CHECK.record("SingleTaskGP with Interval: fit, BoTorch's noise floor", floor, None)
    CHECK.check_bound(
        "SingleTaskGP with Interval: fit, loss above BoTorch's (scaled)",
        excess / max(FIT_TOL, 10 * floor),
        1.0,
    )
    # Box-bounded values stay within the interval.
    if kind == "lengthscale_box":
        ls = torch.as_tensor(rs.covar_module.lengthscale)
        assert bool(((ls >= 0.05 - 1e-9) & (ls <= 2.0 + 1e-9)).all())


# ---------------------------------------------------------------------------
# The Unity models
# ---------------------------------------------------------------------------


def make_data(
    seed: int, K: int, d: int, n_per: int, unobserved: list[int], shift: float = 0.0
) -> tuple[Any, Any]:
    g = torch.Generator().manual_seed(seed)
    observed = [k for k in range(K) if k not in unobserved]
    rows = [k for k in observed for _ in range(n_per)]
    X = torch.rand(len(rows), d, generator=g, dtype=DT)
    t = torch.tensor(rows, dtype=DT)
    offsets = torch.rand(K, generator=g, dtype=DT)
    Y = torch.sin(3 * X[:, 0] + shift) + (X[:, 1:] ** 2).sum(-1) + offsets[t.long()]
    Y = (Y + 0.05 * torch.randn(len(rows), generator=g, dtype=DT)).unsqueeze(-1)
    return torch.cat([X, t.unsqueeze(-1)], -1), Y


class Case:
    """A Unity model in BoTorch and botorch_rs with identical initial values."""

    def __init__(
        self,
        variant: str,
        K: int,
        m: int,
        d: int,
        n_per: int,
        unobserved: list[int],
        seed: int,
        Y: Any = None,
    ) -> None:
        self.variant, self.K, self.m, self.d = variant, K, m, d
        self.unobserved = unobserved
        self.current = unobserved[0] if unobserved else K - 1
        self.X, Y0 = make_data(seed, K, d, n_per, unobserved)
        self.Y = Y0 if Y is None else Y
        g = torch.Generator().manual_seed(seed + 1)
        self.emb = torch.rand(K, m, generator=g, dtype=DT)
        self.w0 = 1.2 * torch.randn(K, 1, generator=g, dtype=DT)
        self.bt = build_model("bt", variant, self.X, self.Y, K, self.emb, self.current)
        self.rs = build_model("rs", variant, self.X, self.Y, K, self.emb, self.current)
        if variant != "provided":
            with torch.no_grad():
                self.bt.emb_layers[0].weight.copy_(self.w0)
            self.rs.emb_layers[0].weight = self.w0
        self.label = f"{variant}_K{K}_m{m}_d{d}"

    def randomize(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in self.bt.named_parameters():
                k = p.shape
                if "raw_noise" in name:
                    v = 0.01 + 0.08 * torch.rand(k, generator=g, dtype=DT)
                elif "raw_constant" in name:
                    v = 0.4 * torch.randn(k, generator=g, dtype=DT)
                elif name.startswith("covar_module"):
                    v = 0.2 + 0.9 * torch.rand(k, generator=g, dtype=DT)
                elif name.startswith("task_covar_module_base"):
                    v = 0.3 + 1.5 * torch.rand(k, generator=g, dtype=DT)
                elif name.startswith("emb_layers"):
                    v = 1.3 * torch.randn(k, generator=g, dtype=DT)
                else:
                    v = 0.5 * torch.rand(k, generator=g, dtype=DT)
                p.copy_(v)
        self.rs.initialize(**{n: p.detach() for n, p in self.bt.named_parameters()})


CASES = [
    ("learned", 2, 1, 1, 5, [1]),
    ("learned", 3, 1, 2, 4, []),
    ("learned", 5, 1, 3, 3, [3, 4]),
    ("learned", 3, 1, 4, 5, [2]),
    ("provided", 2, 1, 2, 5, [1]),
    ("provided", 3, 3, 1, 4, []),
    ("provided", 5, 3, 4, 3, [4]),
    ("provided", 3, 1, 3, 4, [0]),
    ("stock", 3, 3, 2, 4, [2]),
]


def _ids(c: tuple) -> str:
    return f"{c[0]}-K{c[1]}-m{c[2]}-d{c[3]}-u{len(c[5])}"


def compare_named_parameters(label: str, bt: Any, rs: Any, tol: float = 0.0) -> None:
    pb = list(bt.named_parameters())
    pr = list(rs.named_parameters())
    assert [n for n, _ in pb] == [n for n, _ in pr], (label, [n for n, _ in pb], [n for n, _ in pr])
    for (n, b), (_, r) in zip(pb, pr, strict=True):
        assert tuple(b.shape) == tuple(torch.as_tensor(r).shape), (label, n)
        err = rel_err(r, b.detach())
        CHECK.record("named parameters", err, tol)
        assert err <= tol, (label, n, r, b)


def bt_mll_value_and_grad(bt: Any) -> tuple[float, dict[str, Any]]:
    mll = BtMLL(bt.likelihood, bt)
    bt.train()
    bt.zero_grad()
    out = mll(bt(*bt.train_inputs), bt.train_targets, *bt.train_inputs)
    out.backward()
    grads = {
        n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in bt.named_parameters()
    }
    bt.zero_grad()
    return out.item(), grads


def rs_grads_by_name(rs: Any, grad: list[float]) -> dict[str, Any]:
    out, offset = {}, 0
    for name, shape in rs._rs.parameter_shapes():
        k = math.prod(shape)
        out[name] = torch.tensor(grad[offset : offset + k], dtype=DT).reshape(shape)
        offset += k
    return out


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_structure_and_initialization(case: tuple) -> None:
    c = Case(*case, seed=11)
    compare_named_parameters(c.label, c.bt, c.rs)
    assert mll_parameter_names(c.bt) == [n for n, _ in c.rs._rs.parameter_names()]
    _, bounds = get_parameters_and_bounds(BtMLL(c.bt.likelihood, c.bt))
    params = dict(c.bt.named_parameters())
    expected = []
    for n in mll_parameter_names(c.bt):
        key = n if n.startswith("likelihood") else f"model.{n}"
        lo, hi = bounds.get(key, (-math.inf, math.inf))
        expected += [(float(lo), float(hi))] * params[n].numel()
    assert [tuple(b) for b in c.rs._rs.raw_parameter_bounds()] == expected
    for attr in (
        "num_tasks",
        "_output_tasks",
        "_num_outputs",
        "_task_feature",
        "num_non_task_features",
    ):
        assert getattr(c.rs, attr) == getattr(c.bt, attr), attr
    assert c.rs.num_outputs == c.bt.num_outputs == 1
    assert c.rs._unobserved_task_indices == c.bt._unobserved_task_indices.tolist()
    if c.variant == "provided":
        assert c.rs.emb_layers == [] and len(c.bt.emb_layers) == 0
        assert float(c.rs.context_specific_weight[0]) == c.bt.context_specific_weight.item() == 0.0
        assert isinstance(c.rs, ProvidedEmbeddingLCEMGP)
    else:
        assert c.rs.emb_dims == c.bt.emb_dims
        assert c.rs.emb_layers[0].max_norm == c.bt.emb_layers[0].max_norm == 1.0
    lb = c.bt.task_covar_module_base.raw_lengthscale_constraint
    lr = c.rs.task_covar_module_base.raw_lengthscale_constraint
    assert (lr.lower_bound, lr.upper_bound, lr.enforced) == (
        lb.lower_bound.item(),
        lb.upper_bound.item(),
        lb.enforced,
    )
    assert isinstance(c.rs, MultiTaskGP)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_mll_and_renormalization(case: tuple) -> None:
    c = Case(*case, seed=21)
    c.randomize(22)
    value_b, grads_b = bt_mll_value_and_grad(c.bt)
    mll_r = RsMLL(c.rs.likelihood, c.rs)
    value_r, grad_r = mll_r.value_and_grad()
    CHECK.check("MLL", [value_r], [value_b], VALUE_TOL)
    grads_r = rs_grads_by_name(c.rs, grad_r)
    gb = torch.cat([grads_b[n].reshape(-1) for n in grads_r])
    gr = torch.cat([g.reshape(-1) for g in grads_r.values()])
    CHECK.check("MLL gradient", gr, gb, GRAD_TOL)
    # Both forward passes renormalized the embedding weights in place.
    compare_named_parameters(c.label, c.bt, c.rs)
    CHECK.check("mll()", [float(mll_r())], [value_b], VALUE_TOL)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_posteriors(case: tuple) -> None:
    c = Case(*case, seed=31)
    c.randomize(32)
    g = torch.Generator().manual_seed(33)
    X = torch.rand(3, c.d, generator=g, dtype=DT)
    with torch.no_grad():
        pb = c.bt.posterior(X)
    pr = c.rs.posterior(X)
    # eval(): the constant means of contexts without data are the average.
    compare_named_parameters(c.label, c.bt, c.rs)
    assert c.rs.training is False
    assert tuple(pr.mean.shape) == tuple(pb.mean.shape) == (3, 1)
    CHECK.check("posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check("posterior variance", pr.variance, pb.variance, VALUE_TOL)
    CHECK.check(
        "posterior covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    with torch.no_grad():
        pbn = c.bt.posterior(X, observation_noise=True)
    prn = c.rs.posterior(X, observation_noise=True)
    CHECK.check("posterior variance (observation noise)", prn.variance, pbn.variance, VALUE_TOL)
    CHECK.check(
        "posterior covariance (observation noise)",
        prn.distribution.covariance_matrix,
        pbn.distribution.covariance_matrix,
        VALUE_TOL,
    )
    tasks = list(range(c.K))
    with torch.no_grad():
        pba = c.bt.posterior(X, output_indices=tasks)
    pra = c.rs.posterior(X, output_indices=tasks)
    assert tuple(pra.mean.shape) == tuple(pba.mean.shape) == (3, c.K)
    CHECK.check("posterior mean (all tasks)", pra.mean, pba.mean, VALUE_TOL)
    CHECK.check(
        "posterior covariance (all tasks)",
        pra.distribution.covariance_matrix,
        pba.distribution.covariance_matrix,
        VALUE_TOL,
    )
    t = torch.tensor([[float(k % c.K)] for k in range(3)], dtype=DT)
    Xf = torch.cat([X, t], -1)
    with torch.no_grad():
        pbf = c.bt.posterior(Xf)
    prf = c.rs.posterior(Xf)
    CHECK.check("posterior mean (task column)", prf.mean, pbf.mean, VALUE_TOL)
    CHECK.check(
        "posterior covariance (task column)",
        prf.distribution.covariance_matrix,
        pbf.distribution.covariance_matrix,
        VALUE_TOL,
    )
    Xb = torch.rand(2, 4, c.d, generator=g, dtype=DT)
    with torch.no_grad():
        pbb = c.bt.posterior(Xb)
    prb = c.rs.posterior(Xb)
    CHECK.check("posterior mean (batched)", prb.mean, pbb.mean, VALUE_TOL)
    CHECK.check(
        "posterior covariance (batched)",
        prb.distribution.covariance_matrix,
        pbb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    # The task covariance and the embeddings.
    with torch.no_grad():
        kb = (
            c.bt._eval_context_covar().to_dense()
            if hasattr(c.bt._eval_context_covar(), "to_dense")
            else c.bt._eval_context_covar()
        )
        eb = c.bt._task_embeddings()
    CHECK.check("context covariance", c.rs._eval_context_covar(), kb, VALUE_TOL)
    CHECK.check("context embeddings", c.rs._task_embeddings(), eb, 0.0)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_task_covariance_renormalizes_first(case: tuple) -> None:
    """``task_covar_module(task_idcs)`` before any other forward pass: the
    embedding lookup renormalizes the weights before the covariance."""
    c = Case(*case, seed=35)
    c.randomize(36)
    g = torch.Generator().manual_seed(37)
    idx = torch.randint(c.K, (2, 4, 1), generator=g)
    with torch.no_grad():
        kb = c.bt.task_covar_module(idx)
    kr = c.rs.task_covar_module(idx)
    assert tuple(torch.as_tensor(kr).shape) == tuple(kb.shape) == (2, 4, 4)
    CHECK.check("task_covar_module(task_idcs)", kr, kb, VALUE_TOL)
    compare_named_parameters(c.label, c.bt, c.rs)


def test_task_values_are_validated() -> None:
    c = Case("learned", 3, 1, 2, 4, [2], seed=41)
    X = torch.rand(2, 3, dtype=DT)
    X[:, -1] = torch.tensor([0.0, 5.0], dtype=DT)
    with pytest.raises(ValueError, match="between 0 and"):
        c.bt.posterior(X)
    with pytest.raises(ValueError, match="between 0 and"):
        c.rs.posterior(X)
    with pytest.raises(ValueError, match="output_indices"):
        c.rs.posterior(X[:, :], output_indices=[0])
    with pytest.raises(UnsupportedError):
        MultiTaskGP(c.X, c.Y, task_feature=-1)
    with pytest.raises(ValueError, match="context_emb_feature"):
        ProvidedEmbeddingLCEMGP(c.X, c.Y, task_feature=-1)
    with pytest.raises(RuntimeError, match="All output tasks"):
        LCEMGP(c.X, c.Y, task_feature=-1, output_tasks=[7])
    with pytest.raises(Exception, match="all_tasks"):
        LCEMGP(c.X, c.Y, task_feature=-1, all_tasks=[0])
    with pytest.raises(Exception, match="ard_num_dims"):
        c.rs.task_covar_module_base = rs_kernels.RBFKernel(ard_num_dims=3)
    lst = RsModelListGP(c.rs, copy.deepcopy(c.rs))
    with pytest.raises(UnsupportedError):
        lst.condition_on_observations([X[:, :2]], torch.rand(2, 2, dtype=DT))


def test_task_kernel_replacement() -> None:
    """Assigning ``task_covar_module_base`` resets its hyperparameters (the
    Unity project's earlier workaround)."""
    c = Case("stock", 3, 3, 2, 4, [2], seed=43)
    with torch.no_grad():
        c.bt.task_covar_module_base.lengthscale = 0.4
    c.rs.task_covar_module_base.lengthscale = 0.4
    kb = gpytorch.kernels.RBFKernel(
        ard_num_dims=4,
        lengthscale_constraint=gpytorch.constraints.Interval(
            0.0, 2.0, transform=None, initial_value=1.0
        ),
    ).to(c.X)
    kr = rs_kernels.RBFKernel(
        ard_num_dims=4,
        lengthscale_constraint=rs_constraints.Interval(0.0, 2.0, transform=None, initial_value=1.0),
    ).to(c.X)
    c.bt.task_covar_module_base = kb
    c.rs.task_covar_module_base = kr
    compare_named_parameters("replaced task kernel", c.bt, c.rs)
    assert c.rs.task_covar_module_base is kr


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _bt_fit_loss(model: Any, reverse: bool = False) -> float:
    """BoTorch's final loss of ``fit_gpytorch_mll``; with ``reverse``, the
    optimizer's variables are in reverse order (the same problem with other
    rounding in L-BFGS-B)."""
    import botorch.optim.fit as bt_optim_fit

    record: dict[str, Any] = {}
    orig = bt_optim_fit.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        if reverse:
            params, bounds = get_parameters_and_bounds(mll)
            kwargs.update(parameters=dict(list(params.items())[::-1]), bounds=bounds)
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    bt_fit(BtMLL(model.likelihood, model), optimizer=recorder)
    return float(record["r"].fval)


FIT_CASES = [CASES[0], CASES[1], CASES[3], CASES[4], CASES[5], CASES[7]]


@pytest.mark.parametrize("case", FIT_CASES, ids=_ids)
def test_fit_from_identical_initial_values(case: tuple) -> None:
    c = Case(*case, seed=51)
    twins = [perturbed_twin(c.bt, rep + 1) for rep in range(3)]
    reordered = copy.deepcopy(c.bt)
    loss_b = _bt_fit_loss(c.bt)
    floor = max(abs(_bt_fit_loss(t) - loss_b) for t in twins)
    floor = max(floor, abs(_bt_fit_loss(reordered, reverse=True) - loss_b))
    mll_r = RsMLL(c.rs.likelihood, c.rs)
    rs_fit(mll_r)
    excess = mll_r._fit_result["fval"] - loss_b
    tol = max(FIT_TOL, 10 * floor)
    CHECK.record("fit: BoTorch's noise floor (final loss)", floor, None)
    CHECK.record("fit: loss above BoTorch's (unscaled)", max(excess, 0.0), None)
    CHECK.check_bound("fit: loss above BoTorch's (scaled)", excess / tol, 1.0)
    # After fit_gpytorch_mll both are in eval mode without the eval() rule
    # applied yet (BoTorch's mll.eval() does not call model.eval()).
    assert c.rs.training is False


def test_model_list_with_sum_mll() -> None:
    """The Unity project's ``build_contextual_model`` with two objectives."""
    X, Y1 = make_data(61, 3, 2, 4, [2])
    _, Y2 = make_data(61, 3, 2, 4, [2], shift=1.1)
    Y = torch.cat([Y1, Y2], -1)
    emb = torch.rand(3, 2, generator=torch.Generator().manual_seed(62), dtype=DT)
    mll_b, model_b = build_contextual_model("bt", X, Y, 3, emb, 2)
    mll_r, model_r = build_contextual_model("rs", X, Y, 3, emb, 2)
    assert (
        type(model_r).__name__ == "ModelListGP"
        and type(mll_r).__name__ == "SumMarginalLogLikelihood"
    )
    compare_named_parameters("model list", model_b, model_r)

    def sum_mll(model: Any) -> float:
        values = [bt_mll_value_and_grad(sub)[0] for sub in model.models]
        return sum(values) / len(values)

    CHECK.check("SumMarginalLogLikelihood", [float(mll_r())], [sum_mll(model_b)], VALUE_TOL)
    bt_fit(mll_b)
    rs_fit(mll_r)
    value_b = sum_mll(model_b)
    CHECK.check_bound("fit (ModelListGP): MLL below BoTorch's", value_b - float(mll_r()), FIT_TOL)
    Xt = torch.rand(4, 2, generator=torch.Generator().manual_seed(63), dtype=DT)
    assert tuple(model_r.posterior(Xt).mean.shape) == (4, 2)


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def perturbed_twin(model: Any, seed: int, rel: float = 1e-14) -> Any:
    """A BoTorch model whose training inputs (not the task column) are
    perturbed by ``rel``."""
    twin = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    X = twin.train_inputs[0].clone()
    cols = [j for j in range(X.shape[-1]) if j != twin._task_feature]
    X[:, cols] = X[:, cols] * (1 + rel * torch.randn(X[:, cols].shape, generator=g, dtype=DT))
    twin.train_inputs = (X,)
    twin.prediction_strategy = None
    return twin


def _values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.clone().requires_grad_(True)
    v = acqf(Xg)
    (g,) = torch.autograd.grad(v.sum(), Xg)
    return v.detach(), g


def _check_mc(name: str, rs_vg: tuple, bt_vg: tuple, twins: list[tuple]) -> None:
    floor_v = max(log_err(v, bt_vg[0]) for v, _ in twins)
    floor_g = max(rel_err(g, bt_vg[1]) for _, g in twins)
    tol_v, tol_g = max(VALUE_TOL, 10 * floor_v), max(GRAD_TOL, 10 * floor_g)
    ev, eg = log_err(rs_vg[0], bt_vg[0]), rel_err(rs_vg[1], bt_vg[1])
    CHECK.record(f"{name} values (unscaled)", ev, None)
    CHECK.record(f"{name} gradients (unscaled)", eg, None)
    CHECK.record(f"{name}: BoTorch noise floor, values", floor_v, None)
    CHECK.check_bound(f"{name} values (scaled)", ev / tol_v, 1.0)
    CHECK.check_bound(f"{name} gradients (scaled)", eg / tol_g, 1.0)


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_qlognei(case: tuple, q: int) -> None:
    c = Case(*case, seed=71)
    c.randomize(72)
    c.bt.eval()
    Xb = c.X[:, :-1]
    Xc = torch.rand(3, q, c.d, generator=torch.Generator().manual_seed(73), dtype=DT)
    n_dim = Xb.shape[0] + q
    results = []
    for model in [c.bt, perturbed_twin(c.bt, 1), perturbed_twin(c.bt, 2)]:
        with synced_draws(74):
            acqf = BtQLogNEI(
                model, X_baseline=Xb, sampler=bt_sampler(NUM_SAMPLES), prune_baseline=False
            )
            results.append(_values_and_grads(acqf, Xc))
    acqf_r = RsQLogNEI(
        c.rs, X_baseline=Xb, sampler=rs_sampler(NUM_SAMPLES, 74, n_dim), prune_baseline=False
    )
    _check_mc("qLogNEI", _values_and_grads(acqf_r, Xc), results[0], results[1:])


def _list_pair(c: Case, seed: int) -> tuple[Any, Any]:
    """A second model on the inputs of ``c`` with other targets (the second
    objective of the Unity project's model list)."""
    Y2 = (c.X[:, :1] - 0.4) ** 2 + 0.1 * torch.sin(5 * c.X[:, -1:])
    bt = build_model("bt", c.variant, c.X, Y2, c.K, c.emb, c.current)
    rs = build_model("rs", c.variant, c.X, Y2, c.K, c.emb, c.current)
    if c.variant != "provided":
        w = torch.randn(c.K, 1, generator=torch.Generator().manual_seed(seed), dtype=DT)
        with torch.no_grad():
            bt.emb_layers[0].weight.copy_(w)
        rs.emb_layers[0].weight = w
    twin = copy.copy(c)
    twin.bt, twin.rs = bt, rs
    twin.randomize(seed + 1)
    return bt, rs


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("case", CASES[::2], ids=_ids)
def test_qlognehvi(case: tuple, q: int) -> None:
    c = Case(*case, seed=81)
    c.randomize(82)
    bt2, rs2 = _list_pair(c, 83)
    lb, lr = BtModelListGP(c.bt, bt2), RsModelListGP(c.rs, rs2)
    lb.eval()
    Xb = c.X[:, :-1]
    with torch.no_grad():
        ref = (lb.posterior(Xb).mean.min(0).values - 0.1).tolist()
    Xc = torch.rand(3, q, c.d, generator=torch.Generator().manual_seed(84), dtype=DT)
    n_dim = (Xb.shape[0] + q) * 2
    results = []
    lists = [lb] + [BtModelListGP(perturbed_twin(c.bt, s), perturbed_twin(bt2, s)) for s in (1, 2)]
    for model in lists:
        model.eval()
        with synced_draws(85):
            acqf = BtQLogNEHVI(
                model,
                ref_point=ref,
                X_baseline=Xb,
                sampler=bt_sampler(NUM_SAMPLES),
                prune_baseline=False,
            )
            results.append(_values_and_grads(acqf, Xc))
    acqf_r = RsQLogNEHVI(
        lr,
        ref_point=ref,
        X_baseline=Xb,
        sampler=rs_sampler(NUM_SAMPLES, 85, n_dim),
        prune_baseline=False,
    )
    _check_mc("qLogNEHVI", _values_and_grads(acqf_r, Xc), results[0], results[1:])


@pytest.mark.parametrize("variant", ["learned", "provided"])
def test_optimize_acqf_from_identical_initial_conditions(variant: str) -> None:
    c = Case(variant, 3, 2, 2, 5, [2], seed=91)
    c.randomize(92)
    c.bt.eval()
    Xb = c.X[:, :-1]
    bounds = torch.stack([torch.zeros(c.d, dtype=DT), torch.ones(c.d, dtype=DT)])
    ics = torch.rand(4, 1, c.d, generator=torch.Generator().manual_seed(93), dtype=DT)
    with synced_draws(94):
        a_bt = BtQLogNEI(c.bt, X_baseline=Xb, sampler=bt_sampler(64), prune_baseline=False)
        _, v_bt = bt_optimize.optimize_acqf(
            a_bt, bounds, q=1, num_restarts=4, batch_initial_conditions=ics, return_best_only=False
        )
    a_rs = RsQLogNEI(
        c.rs, X_baseline=Xb, sampler=rs_sampler(64, 94, Xb.shape[0] + 1), prune_baseline=False
    )
    _, v_rs = rs_optimize_acqf(
        a_rs, bounds, q=1, num_restarts=4, batch_initial_conditions=ics, return_best_only=False
    )
    agree = [
        abs(float(v_rs[i]) - float(v_bt[i])) <= 1e-6 * max(1.0, abs(float(v_bt[i])))
        for i in range(4)
    ]
    CHECK.record(
        "optimize_acqf (qLogNEI): restarts not reaching BoTorch's value", 4 - sum(agree), None
    )
    assert sum(agree) >= 3, (v_rs, v_bt)


# ---------------------------------------------------------------------------
# Randomized sweep
# ---------------------------------------------------------------------------


@pytest.mark.equivalence
@pytest.mark.parametrize("scenario", range(8))
def test_equivalence_sweep(scenario: int) -> None:
    r = random.Random(1000 + scenario)
    variant = r.choice(["learned", "provided", "provided", "learned", "stock"])
    K = r.choice([2, 3, 5])
    m = r.choice([1, 3])
    d = r.randint(1, 4)
    unobserved = sorted(r.sample(range(K), r.randint(0, K - 1)))
    c = Case(variant, K, m, d, r.randint(2, 5), unobserved, seed=2000 + scenario)
    c.randomize(3000 + scenario)
    value_b, grads_b = bt_mll_value_and_grad(c.bt)
    value_r, grad_r = RsMLL(c.rs.likelihood, c.rs).value_and_grad()
    CHECK.check("sweep: MLL", [value_r], [value_b], VALUE_TOL)
    grads_r = rs_grads_by_name(c.rs, grad_r)
    CHECK.check(
        "sweep: MLL gradient",
        torch.cat([g.reshape(-1) for g in grads_r.values()]),
        torch.cat([grads_b[n].reshape(-1) for n in grads_r]),
        GRAD_TOL,
    )
    X = torch.rand(4, d, generator=torch.Generator().manual_seed(scenario), dtype=DT)
    with torch.no_grad():
        pb = c.bt.posterior(X, observation_noise=bool(scenario % 2))
    pr = c.rs.posterior(X, observation_noise=bool(scenario % 2))
    CHECK.check("sweep: posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "sweep: posterior covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    q = 1 + scenario % 2
    Xb = c.X[:, :-1]
    Xc = torch.rand(2, q, d, generator=torch.Generator().manual_seed(7 + scenario), dtype=DT)
    results = []
    for model in [c.bt, perturbed_twin(c.bt, 1), perturbed_twin(c.bt, 2)]:
        model.eval()
        with synced_draws(scenario):
            acqf = BtQLogNEI(
                model, X_baseline=Xb, sampler=bt_sampler(NUM_SAMPLES), prune_baseline=False
            )
            results.append(_values_and_grads(acqf, Xc))
    acqf_r = RsQLogNEI(
        c.rs,
        X_baseline=Xb,
        sampler=rs_sampler(NUM_SAMPLES, scenario, Xb.shape[0] + q),
        prune_baseline=False,
    )
    _check_mc("sweep: qLogNEI", _values_and_grads(acqf_r, Xc), results[0], results[1:])
