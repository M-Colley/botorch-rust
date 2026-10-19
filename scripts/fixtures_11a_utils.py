"""Fixtures of phase 11a (utilities): the native kernels behind
``botorch.utils.safe_math``, ``botorch.utils.probability``, the samplers'
random numbers and QMC transforms, and the single-objective test functions.

Written to ``<out>/11a-utils/`` by ``generate_fixtures.py --phase 11a-utils``
(BoTorch main):

* ``elementwise.json``: every element-wise function of
  ``botorch_core::utils::elementwise`` (the standard-normal helpers, the
  safe-math functions with their temperatures, ``log_prob_normal_in``,
  ``approximate_round``, ``bvnu`` / ``bvn`` / ``bvnmom``) on random and edge
  inputs: BoTorch's values and autograd gradients (``null`` where BoTorch's
  gradient is not finite);
* ``reductions.json``: ``logsumexp``, ``logmeanexp``, ``smooth_amax`` /
  ``smooth_amin``, ``fatmax`` / ``fatmin`` over rows, values and gradients;
* ``probability.json``: ``compute_log_prob_feas_from_bounds`` and
  ``percentile_of_score``;
* ``mvnxpb.json``: ``MVNXPB`` log probabilities, their gradients and the
  final pivot order; ``augment_cholesky``;
* ``lin_ess.json``: ``LinearEllipticalSliceSampler`` chains with the random
  numbers BoTorch draws (torch's generator), with BoTorch's wrapping constant
  2 pi in float64 (BoTorch rounds it to float32; see
  ``docs/drafts/11a-utils.md``);
* ``rng_qmc.json``: ``torch.randn`` / ``torch.rand`` after
  ``torch.manual_seed`` (float64 and float32), and ``NormalQMCEngine`` draws
  with their scrambled Sobol points;
* ``test_functions.json``: the configurations of the multi-fidelity,
  sensitivity-analysis and synthetic single-objective problems of phase 11a,
  points, values, slacks and costs.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Any

import torch

DT = torch.float64


def _list(x: Any) -> Any:
    return x.detach().to(DT).tolist()


def _finite_or_none(values: Any) -> Any:
    """Nested lists with non-finite entries replaced by ``None``."""
    if isinstance(values, list):
        return [_finite_or_none(v) for v in values]
    return values if math.isfinite(values) else None


def _grads(f: Any, inputs: list[Any]) -> tuple[list[list[float]], list[list[list[float | None]]]]:
    xs = [x.clone().requires_grad_(True) for x in inputs]
    out = f(*xs)
    outs = list(out) if isinstance(out, tuple) else [out]
    values = [_list(o) for o in outs]
    grads = []
    for o in outs:
        gs = torch.autograd.grad(o.sum(), xs, retain_graph=True, allow_unused=True)
        grads.append(
            [_finite_or_none(_list(torch.zeros_like(x) if g is None else g)) for g, x in zip(gs, xs, strict=True)]
        )
    return values, grads


def _elementwise() -> list[dict[str, Any]]:
    import botorch.utils.probability.utils as pu
    import botorch.utils.safe_math as sm
    from botorch.utils.probability.bvn import bvn, bvnmom, bvnu
    from botorch.utils.rounding import approximate_round

    g = torch.Generator().manual_seed(11)
    x = torch.cat(
        [
            torch.randn(12, generator=g, dtype=DT) * 3,
            torch.tensor([-40.0, -12.0, -1e-6, 0.0, 1e-6, 7.0, 25.0], dtype=DT),
        ]
    )
    y = torch.randn(x.numel(), generator=g, dtype=DT) * 3
    pos = x.abs() + 1e-3
    cases: list[dict[str, Any]] = []

    def add(name: str, f: Any, inputs: list[Any], params: list[float] | None = None) -> None:
        values, grads = _grads(f, inputs)
        cases.append(
            {
                "name": name,
                "inputs": [_list(v) for v in inputs],
                "params": params or [],
                "values": values,
                "grads": grads,
            }
        )

    for name in ("ndtr", "phi", "log_phi", "log_ndtr", "log_erfc", "log_erfcx", "standard_normal_log_hazard"):
        add(name, getattr(pu, name), [x])
    for name in ("exp", "log1pexp", "logexpit", "cauchy"):
        add(name, getattr(sm, name), [x])
    add("log", sm.log, [pos])
    add("log1mexp", sm.log1mexp, [-pos])
    add("sigmoid", lambda t: sm.sigmoid(t), [x])
    add("log_sigmoid", lambda t: sm.sigmoid(t, log=True), [x])
    add("fat_sigmoid", lambda t: sm.sigmoid(t, fat=True), [x])
    add("log_fat_sigmoid", lambda t: sm.sigmoid(t, log=True, fat=True), [x])
    for name in ("add", "sub", "mul", "logplusexp"):
        add(name, getattr(sm, name), [x, y])
    add("div", sm.div, [x, y.abs() + 0.5])
    lo, hi = torch.minimum(x, y), torch.maximum(x, y) + 0.25
    add("logdiffexp", sm.logdiffexp, [lo, hi])
    add("log_prob_normal_in", pu.log_prob_normal_in, [lo, hi])
    for tau, alpha in ((1.0, 2.0), (0.3, 0.5)):
        for name in ("fatmaximum", "fatminimum"):
            add(name, lambda a, b, n=name, t=tau, al=alpha: getattr(sm, n)(a, b, tau=t, alpha=al), [x, y], [tau, alpha])
    taus = torch.rand(x.numel(), generator=g, dtype=DT) + 0.05
    for name in ("fatplus", "log_fatplus", "fatmoid", "log_fatmoid"):
        add(name, lambda a, t, n=name: getattr(sm, n)(a, tau=t), [x, taus])
    add("approximate_round", lambda a, t: approximate_round(a, tau=t), [x, taus])
    # BoTorch's log_softplus takes a scalar temperature (softplus's beta):
    # the gradients with respect to tau are not compared (null).
    for tau in (1.0, 0.1):
        tt = torch.full_like(x, tau)
        add("log_softplus", lambda a, t, v=tau: sm.log_softplus(a, tau=v) + 0 * t, [x, tt])
        cases[-1]["grads"][0][1] = [None] * x.numel()
    n = 9
    r = torch.rand(n, generator=g, dtype=DT) * 1.96 - 0.98
    xl = torch.randn(n, generator=g, dtype=DT) * 1.5
    yl = torch.randn(n, generator=g, dtype=DT) * 1.5
    xu = xl + torch.rand(n, generator=g, dtype=DT) * 3 + 0.05
    yu = yl + torch.rand(n, generator=g, dtype=DT) * 3 + 0.05
    add("bvnu", bvnu, [r, xl, yl])
    add("bvn", bvn, [r, xl, yl, xu, yu])
    p = bvn(r, xl, yl, xu, yu)
    add("bvnmom", lambda *a: bvnmom(*a[:5], p=a[5]), [r, xl, yl, xu, yu, p])
    return cases


def _reductions() -> list[dict[str, Any]]:
    import botorch.utils.safe_math as sm

    g = torch.Generator().manual_seed(12)
    x = torch.randn(4, 6, generator=g, dtype=DT) * 4
    x[1, 2] = x[1, 3] = x[1].max()
    cases = []
    for name, params in (
        ("logsumexp", []),
        ("logmeanexp", []),
        ("smooth_amax", [0.5]),
        ("smooth_amin", [0.1]),
        ("fatmax", [0.3, 2.0]),
        ("fatmin", [1.0, 0.5]),
    ):
        kw = {}
        if name.startswith("smooth"):
            kw["tau"] = params[0]
        if name.startswith("fat"):
            kw.update(tau=params[0], alpha=params[1])
        values, grads = _grads(lambda t, n=name, k=kw: getattr(sm, n)(t, dim=-1, **k), [x])
        cases.append({"name": name, "x": _list(x), "params": params, "values": values[0], "grads": grads[0][0]})
    return cases


def _probability() -> dict[str, Any]:
    import botorch.utils.probability.utils as pu

    g = torch.Generator().manual_seed(13)
    means = torch.randn(5, 4, generator=g, dtype=DT)
    sigmas = torch.rand(5, 4, generator=g, dtype=DT) + 0.2
    kw = {
        "con_lower_inds": torch.tensor([0, 3]),
        "con_upper_inds": torch.tensor([1]),
        "con_both_inds": torch.tensor([2]),
        "con_lower": torch.tensor([-0.5, 0.2], dtype=DT),
        "con_upper": torch.tensor([0.7], dtype=DT),
        "con_both": torch.tensor([[-1.0, 0.5]], dtype=DT),
    }
    values, grads = _grads(lambda m, s: pu.compute_log_prob_feas_from_bounds(means=m, sigmas=s, **kw), [means, sigmas])
    data = torch.randn(3, 7, generator=g, dtype=DT).round(decimals=1)
    score = data[:, :1] + torch.tensor([[0.0], [0.05], [-3.0]], dtype=DT)
    return {
        "feasibility": {
            "lower": [[0, -0.5], [3, 0.2]],
            "upper": [[1, 0.7]],
            "both": [[2, -1.0, 0.5]],
            "means": _list(means),
            "sigmas": _list(sigmas),
            "values": values[0],
            "grad_means": grads[0][0],
            "grad_sigmas": grads[0][1],
        },
        "percentile": {
            "data": _list(data),
            "scores": _list(score.squeeze(-1)),
            "values": _list(pu.percentile_of_score(data, score, dim=-1).squeeze(-1).to(DT)),
        },
    }


def _mvnxpb() -> dict[str, Any]:
    from botorch.utils.probability.linalg import augment_cholesky
    from botorch.utils.probability.mvnxpb import MVNXPB

    cases = []
    for seed, d in ((0, 2), (1, 3), (2, 5), (3, 7)):
        g = torch.Generator().manual_seed(100 + seed)
        L = torch.randn(d, d, generator=g, dtype=DT)
        cov = L @ L.T + d * torch.eye(d, dtype=DT)
        lo = torch.randn(d, generator=g, dtype=DT) - 1.0
        hi = lo + torch.rand(d, generator=g, dtype=DT) * 3 + 0.2
        if d > 2:
            lo[0] = -math.inf
            hi[-1] = math.inf
        bounds = torch.stack([lo, hi], -1)
        solver = MVNXPB(cov, bounds)
        log_prob = solver.solve()
        c = cov.clone().requires_grad_(True)
        bd = bounds.clone().requires_grad_(True)
        lp = MVNXPB(c, bd).solve()
        gc, gb = torch.autograd.grad(lp, [c, bd])
        cases.append(
            {
                "cov": _list(cov),
                # Infinite bounds as null (lower column: -inf, upper: +inf).
                "bounds": _finite_or_none(_list(bounds)),
                "log_prob": log_prob.item(),
                "grad_cov": _list(gc),
                "grad_bounds": _finite_or_none(_list(gb)),
                "perm": solver.perm.tolist(),
            }
        )
    g = torch.Generator().manual_seed(200)
    L = torch.randn(5, 5, generator=g, dtype=DT)
    K = L @ L.T + torch.eye(5, dtype=DT)
    Laa = torch.linalg.cholesky(K[:3, :3])
    aug = augment_cholesky(Laa=Laa, Kbb=K[3:, 3:], Kba=K[3:, :3])
    return {"cases": cases, "augment": {"K": _list(K), "na": 3, "L": _list(aug)}}


def _lin_ess() -> list[dict[str, Any]]:
    import botorch.utils.probability.lin_ess as lin_ess

    lin_ess._twopi = torch.tensor(2.0 * math.pi, dtype=DT)
    A = torch.tensor([[1.0, 1.0, 0.0], [-1.0, 0.5, 0.2], [0.0, -0.3, 1.0]], dtype=DT)
    b = torch.tensor([[1.0], [0.8], [0.9]], dtype=DT)
    x0 = torch.tensor([[0.1], [0.0], [-0.2]], dtype=DT)
    cov = torch.tensor([[1.0, 0.3, 0.1], [0.3, 0.5, -0.1], [0.1, -0.1, 0.8]], dtype=DT)
    cases = []
    for label, kw in (
        ("standard", {}),
        ("covariance", {"covariance_matrix": cov, "mean": torch.tensor([[0.2], [-0.1], [0.05]], dtype=DT)}),
        ("chains", {"num_chains": 3}),
        ("fixed", {"fixed_indices": [1], "covariance_root": torch.diag(torch.tensor([1.0, 2.0, 0.5], dtype=DT))}),
    ):
        steps = 40
        torch.manual_seed(7)
        s = lin_ess.LinearEllipticalSliceSampler(inequality_constraints=(A, b), interior_point=x0, **kw)
        state = torch.random.get_rng_state()
        k, c = s._z.shape
        nu, u = [], []
        for _ in range(steps):
            nu.append(torch.randn(k, c, dtype=DT).reshape(-1).tolist())
            u.append(torch.rand(c, dtype=DT).tolist())
        torch.random.set_rng_state(state)
        xs = [s.step().T.reshape(-1).tolist() for _ in range(steps)]
        case: dict[str, Any] = {
            "label": label,
            "A": _list(A),
            "b": _list(b.squeeze(-1)),
            "x0": _list(x0.squeeze(-1)),
            "num_chains": kw.get("num_chains", 1),
            "nu": nu,
            "u": u,
            "samples": xs,
        }
        if "covariance_matrix" in kw:
            case["covariance_matrix"] = _list(kw["covariance_matrix"])
            case["mean"] = _list(kw["mean"].squeeze(-1))
        if "fixed_indices" in kw:
            case["fixed_indices"] = kw["fixed_indices"]
            case["root_diag"] = _list(torch.diagonal(kw["covariance_root"]))
        cases.append(case)
    return cases


def _rng_qmc() -> dict[str, Any]:
    from botorch.sampling.qmc import NormalQMCEngine

    rng = []
    for seed in (0, 1, 1234, 2**40 + 3):
        for n in (1, 3, 16, 17, 100):
            torch.manual_seed(seed)
            randn = torch.randn(n, dtype=DT).tolist()
            torch.manual_seed(seed)
            rand = torch.rand(n, dtype=DT).tolist()
            torch.manual_seed(seed)
            rand32 = torch.rand(n, dtype=torch.float32).double().tolist()
            rng.append({"seed": seed, "n": n, "randn": randn, "rand": rand, "rand32": rand32})
    qmc = []
    for d, inv in ((1, False), (3, False), (4, True), (5, True)):
        e = NormalQMCEngine(d, seed=3, inv_transform=inv)
        sobol_dim = d if inv else 2 * math.ceil(d / 2)
        sob = torch.quasirandom.SobolEngine(dimension=sobol_dim, scramble=True, seed=3)
        n = 8
        uniform = sob.draw(n, dtype=DT)
        qmc.append(
            {
                "d": d,
                "inv_transform": inv,
                "sobol_dim": sobol_dim,
                "uniform": _list(uniform),
                "samples": _list(e.draw(n, dtype=DT)),
            }
        )
    return {"rng": rng, "qmc": qmc}


def _test_functions() -> list[dict[str, Any]]:
    import botorch.test_functions.multi_fidelity as mf
    import botorch.test_functions.sensitivity_analysis as sa
    import botorch.test_functions.synthetic as sy

    problems: list[tuple[Any, str, dict[str, Any]]] = [
        (mf, "AugmentedBranin", {}),
        (mf, "AugmentedHartmann", {}),
        (mf, "AugmentedRosenbrock", {"dim": 5}),
        (mf, "WingWeightMultiFidelity", {}),
        (mf, "BoreholeMultiFidelity", {}),
        (sa, "Ishigami", {"b": 0.05}),
        (sa, "Gsobol", {"dim": 6}),
        (sa, "Morris", {}),
        (sy, "AckleyMixed", {"dim": 7}),
        (sy, "Labs", {"dim": 12}),
        (sy, "TrajectoryPlanning", {"dim": 8}),
        (sy, "TrajectoryPlanning", {"dim": 10, "use_smooth_interp": False, "max_step_size": 0.2}),
        (sy, "ConstrainedGramacy", {}),
        (sy, "ConstrainedHartmann", {"dim": 6}),
        (sy, "ConstrainedHartmannSmooth", {"dim": 3}),
        (sy, "PressureVessel", {}),
        (sy, "WeldedBeamSO", {}),
        (sy, "TensionCompressionString", {}),
        (sy, "SpeedReducer", {}),
        (sy, "KeaneBumpFunction", {"dim": 5}),
    ]
    out = []
    for i, (mod, name, kw) in enumerate(problems):
        p = getattr(mod, name)(**kw)
        g = torch.Generator().manual_seed(300 + i)
        lo, hi = p.bounds
        X = lo + (hi - lo) * torch.rand(6, p.dim, generator=g, dtype=DT)
        inds = list(p.discrete_inds) + list(p.categorical_inds)
        if inds:
            X[:, inds] = X[:, inds].round()
        params: list[float] = []
        if name == "Ishigami":
            params = [float(p.b)]
        elif name == "Gsobol":
            params = [float(v) for v in p.a]
        elif name == "AckleyMixed":
            params = [float(v) for v in p._optimizers[0]]
        case: dict[str, Any] = {
            "name": name,
            "dim": p.dim,
            "bounds": _list(p.bounds),
            "discrete": list(p.discrete_inds),
            "params": params,
            "X": _list(X),
            "values": _list(p.evaluate_true(X)),
        }
        if name == "TrajectoryPlanning":
            case["use_smooth_interp"] = bool(p.use_smooth_interp)
            case["max_step_size"] = float(p.max_step_size)
        if hasattr(p, "evaluate_slack_true"):
            case["slack"] = _list(p.evaluate_slack_true(X))
        if hasattr(p, "cost"):
            case["cost"] = _list(p.cost(X))
        out.append(case)
    return out


def generate_11a_utils(out: str) -> None:
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    d = os.path.join(out, "11a-utils")
    os.makedirs(d, exist_ok=True)
    files = {
        "elementwise.json": _elementwise(),
        "reductions.json": _reductions(),
        "probability.json": _probability(),
        "mvnxpb.json": _mvnxpb(),
        "lin_ess.json": _lin_ess(),
        "rng_qmc.json": _rng_qmc(),
        "test_functions.json": _test_functions(),
    }
    for name, data in files.items():
        with open(os.path.join(d, name), "w") as f:
            json.dump(data, f, allow_nan=False)
        print(f"wrote {os.path.join(d, name)}", file=sys.stderr)
