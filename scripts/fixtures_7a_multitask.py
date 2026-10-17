"""Fixtures of the multi-task GPs (phase 7a): ``MultiTaskGP`` and
``KroneckerMultiTaskGP``.

Written to ``<out>/7a_multitask/`` by ``generate_fixtures.py --phase
7a-multitask`` (BoTorch main). Every ``MultiTaskGP`` case (see
``botorch_pyo3/tests/multitask_helpers.py``; the Rust test builds the same
model from the recorded configuration and the data kernel's spec) records:

* the parameter names and shapes, the optimizer bounds and the default raw
  parameters (``PositiveIndexKernel`` draws its initial values from torch's
  generator; the Rust test starts from the recorded values);
* the MLL and its gradient at the default and at random raw parameters
  (BoTorch's bugs fixed in the reference, see ``multitask_helpers``), and the
  raw parameters after ``eval()`` (tasks without data);
* point posteriors (``b x 1 x d``) over the output tasks with their output
  covariances, with and without observation noise, and the gradients of the
  sums of the means and variances; joint ``q = 3`` posteriors over the
  output tasks and with task columns (mean, output-major covariance, with
  observation noise) and the gradient of a weighted sum of mean and
  covariance;
* ``condition_on_observations`` at two new points and ``fantasize`` with
  recorded base samples: the posteriors of the resulting models;
* for some cases ``fit_gpytorch_mll`` from the default parameters with
  BoTorch's outcomes on 16 twins (training inputs perturbed at 1e-14 ..
  1e-11), and acquisition values and gradients (qLogNEI with recorded base
  samples on the model and its twins, qUCB / qSimpleRegret with a
  scalarizing posterior transform, LogEI, PosteriorMean).

Non-finite numbers are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

DT = torch.float64
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
N_TWINS = 16
NUM_SAMPLES = 64
TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "botorch_pyo3", "tests")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures_6a_kernels as f6a  # noqa: E402


def enc(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return enc(x.detach().tolist())
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return x


def _helpers() -> tuple[Any, Any]:
    sys.path.insert(0, os.path.normpath(TESTS))
    import kernel_tree_helpers
    import multitask_helpers

    return kernel_tree_helpers, multitask_helpers


@contextmanager
def recorded_draws(seed: int, record: dict[int, Any]) -> Iterator[None]:
    """BoTorch's Sobol normal draws replaced by seeded i.i.d. normal draws,
    recorded per dimension."""
    import botorch.sampling.normal as bt_normal

    orig = bt_normal.draw_sobol_normal_samples

    def draw(d: int, n: int, device: Any = None, dtype: Any = None, seed_: Any = None, **kw: Any):
        if d not in record:
            g = torch.Generator().manual_seed(seed * 7919 + d)
            record[d] = torch.randn(n, d, generator=g, dtype=DT)
        return record[d].clone().to(dtype=dtype or DT)

    bt_normal.draw_sobol_normal_samples = draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig


# ----------------------------------------------------------------------
# MultiTaskGP
# ----------------------------------------------------------------------


def mtgp_cases() -> list[dict[str, Any]]:
    _, h = _helpers()
    return h.mtgp_cases()


def mtgp_case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    from botorch.sampling.normal import SobolQMCNormalSampler

    _, h = _helpers()
    X, Y, Yvar = h.make_data(case, seed)
    bt, _ = h.build_pair(case, X, Y, Yvar, seed)
    out: dict[str, Any] = {
        "name": case["name"],
        "case": case,
        "train_x": X,
        "train_y": Y,
        "train_yvar": Yvar if case.get("fixed_noise") else None,
        "param_names": [(n, list(p.shape)) for n, p in bt.named_parameters()],
        "bounds": h.bounds_of(bt),
        "default_raw": h.flat(bt),
    }
    v, g = h.bt_mll_value_and_grad(bt)
    out["default_mll"] = {"value": v, "grad": g}
    if case.get("fit"):
        out["fit"] = fit_record(case, bt, X, Y, Yvar, seed)
    raw = h.random_raw(bt, seed + 1)
    h.set_flat(bt, raw)
    out["raw"] = raw
    v, g = h.bt_mll_value_and_grad(bt)
    out["mll"] = {"value": v, "grad": g}
    bt.eval()
    out["eval_raw"] = h.flat(bt)
    Xq = h.test_points(case, 4, seed + 2)
    out["marginal"] = h.marginal_record(bt, Xq)
    out["marginal_noisy"] = h.marginal_record(bt, Xq, noise=True)
    out["joint"] = h.joint_record(bt, h.test_points(case, 3, seed + 3), seed + 4)
    out["joint_tasks"] = h.joint_record(
        bt, h.test_points(case, 3, seed + 5, with_tasks=True), seed + 6
    )
    # Conditioning and fantasies.
    Xn = h.test_points(case, 2, seed + 7, with_tasks=True)
    g = torch.Generator().manual_seed(seed + 8)
    Yn = 1.0 + torch.randn(2, 1, generator=g, dtype=DT)
    Xc = h.test_points(case, 3, seed + 9)
    cond = {"x": Xn, "y": Yn}
    if case.get("fixed_noise"):
        cond["noise"] = torch.full((2, 1), 0.05, dtype=DT)
        cm = bt.condition_on_observations(Xn, Yn, noise=cond["noise"])
    else:
        cm = bt.condition_on_observations(Xn, Yn)
    with torch.no_grad():
        pc = cm.posterior(Xc)
        cond["test_x"] = Xc
        cond["mean"] = pc.mean
        cond["covariance"] = pc.distribution.covariance_matrix
    out["conditioning"] = cond
    if not case.get("fixed_noise"):
        draws: dict[int, Any] = {}
        with recorded_draws(seed + 10, draws):
            sampler = SobolQMCNormalSampler(torch.Size([3]), seed=0)
            fm = bt.fantasize(Xn, sampler)
        with torch.no_grad():
            pf = fm.posterior(Xc)
        out["fantasy"] = {
            "x": Xn,
            "base_samples": draws[2],
            "test_x": Xc,
            "mean": pf.mean,
            "covariance": pf.distribution.covariance_matrix,
        }
    if case.get("acquisition"):
        out["acquisition"] = acquisition_record(case, bt, X, seed + 11)
    return out


def twin(model: Any, case: dict[str, Any], seed: int, rel: float) -> Any:
    """A deep copy of ``model`` with its non-task training inputs perturbed."""
    tw = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    X = tw.train_inputs[0].clone()
    tf = case["task_feature"] % (case["d"] + 1)
    cols = [c for c in range(X.shape[-1]) if c != tf]
    X[..., cols] = X[..., cols] * (1 + rel * torch.randn(X[..., cols].shape, generator=g, dtype=DT))
    tw.set_train_data(inputs=(X,), strict=False)
    tw.prediction_strategy = None
    return tw


def fit_record(
    case: dict[str, Any], bt: Any, X: Any, Y: Any, Yvar: Any, seed: int
) -> dict[str, Any]:
    _, h = _helpers()

    def fit(model: Any) -> dict[str, Any]:
        import botorch.optim.fit as of
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        record: dict[str, Any] = {"attempts": 0}
        orig = of.fit_gpytorch_mll_scipy

        def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
            record["attempts"] += 1
            record["r"] = orig(mll, *args, **kwargs)
            return record["r"]

        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), optimizer=recorder)
        r = record["r"]
        return {
            "loss": float(r.fval),
            "iterations": int(r.step),
            "status": r.status.name,
            "attempts": record["attempts"],
            "raw_params": h.flat(model),
        }

    default = h.flat(bt)
    fitted = fit(copy.deepcopy(bt))
    twins = []

    def add_twins(ks: range) -> None:
        for kk in ks:
            tw = twin(bt, case, kk, TWIN_SCALES[(kk - 1) % len(TWIN_SCALES)])
            h.set_flat(tw, default)
            r = fit(tw)
            twins.append((tw.train_inputs[0].clone(), r["loss"], r["attempts"]))

    add_twins(range(1, N_TWINS + 1))
    if f6a.is_chaotic(fitted, twins):
        add_twins(range(N_TWINS + 1, f6a.CHAOTIC_TWINS + 1))
    fitted["twin_train_x"] = [t for t, _, _ in twins]
    fitted["twin_losses"] = [loss for _, loss, _ in twins]
    fitted["twin_attempts"] = [a for _, _, a in twins]
    return fitted


def acquisition_record(case: dict[str, Any], bt: Any, X: Any, seed: int) -> dict[str, Any]:
    from botorch.acquisition.analytic import LogExpectedImprovement, PosteriorMean
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound
    from botorch.acquisition.objective import ScalarizedPosteriorTransform
    from botorch.models.multitask import MultiTaskGP
    from botorch.sampling.normal import SobolQMCNormalSampler

    _, h = _helpers()
    g = torch.Generator().manual_seed(seed)
    d = case["d"]
    tf = case["task_feature"] % (d + 1)
    Xb = torch.cat([X[:, :tf], X[:, tf + 1 :]], -1)
    rec: dict[str, Any] = {"x_baseline": Xb}

    def vg(acqf: Any, Xc: Any) -> tuple[Any, Any]:
        Xg = Xc.clone().requires_grad_(True)
        v = acqf(Xg)
        (gr,) = torch.autograd.grad(v.sum(), Xg)
        return v.detach(), gr

    twins = [twin(bt, case, kk, TWIN_SCALES[(kk - 1) % len(TWIN_SCALES)]) for kk in range(1, 9)]
    for t in twins:
        t.eval()
    rec["qlognei"] = []
    for q in (1, 2):
        Xc = torch.rand(3, q, d, generator=g, dtype=DT)
        draws: dict[int, Any] = {}
        evals = []
        for mdl in [bt] + twins:
            with recorded_draws(seed + q, draws):
                sampler = SobolQMCNormalSampler(torch.Size([NUM_SAMPLES]), seed=0)
                acqf = qLogNoisyExpectedImprovement(
                    mdl, X_baseline=Xb, sampler=sampler, prune_baseline=False, cache_root=False
                )
                evals.append(vg(acqf, Xc))
        rec["qlognei"].append(
            {
                "q": q,
                "x": Xc,
                "draws": dict(draws),
                "values": evals[0][0],
                "grads": evals[0][1],
                "twin_values": [v for v, _ in evals[1:]],
                "twin_grads": [gg for _, gg in evals[1:]],
            }
        )
    # Multi-output model (all tasks) with a scalarizing posterior transform.
    mo = copy.deepcopy(bt)
    mo._output_tasks = sorted(set(case["tasks"]))
    mo._num_outputs = len(mo._output_tasks)
    assert isinstance(mo, MultiTaskGP)
    w = torch.tensor([0.7, -0.4][: mo._num_outputs], dtype=DT)
    pt = ScalarizedPosteriorTransform(weights=w, offset=0.1)
    Xc = torch.rand(3, 2, d, generator=g, dtype=DT)
    draws = {}
    with recorded_draws(seed + 5, draws):
        sampler = SobolQMCNormalSampler(torch.Size([NUM_SAMPLES]), seed=0)
        ucb = qUpperConfidenceBound(mo, beta=0.3, sampler=sampler, posterior_transform=pt)
        v, gr = vg(ucb, Xc)
    rec["qucb_scalarized"] = {
        "weights": w,
        "offset": 0.1,
        "x": Xc,
        "draws": dict(draws),
        "values": v,
        "grads": gr,
    }
    X1 = torch.rand(4, 1, d, generator=g, dtype=DT)
    # (A best_f in the middle of the data: far in the tail, BoTorch's LogEI is
    # an asymptotic approximation.)
    best = float(bt.train_targets.median()) * float(bt.outcome_transform.stdvs) + float(
        bt.outcome_transform.means
    )
    # (A tensor: BoTorch stores a float best_f in torch's default dtype, float32.)
    v, gr = vg(LogExpectedImprovement(bt, best_f=torch.tensor(best, dtype=DT)), X1)
    rec["logei"] = {"best_f": best, "x": X1, "values": v, "grads": gr}
    v, gr = vg(PosteriorMean(mo, posterior_transform=pt), X1)
    rec["posterior_mean_scalarized"] = {"x": X1, "values": v, "grads": gr}
    return rec


# ----------------------------------------------------------------------
# KroneckerMultiTaskGP
# ----------------------------------------------------------------------


def kron_cases() -> list[dict[str, Any]]:
    _, h = _helpers()
    return h.kron_cases()


def kron_posterior_record(bt: Any, X: Any, seed: int) -> dict[str, Any]:
    _, h = _helpers()
    g = torch.Generator().manual_seed(seed)
    q, m = X.shape[-2], bt.num_outputs
    bt.eval()
    with torch.no_grad():
        pb = bt.posterior(X)
        pbn = bt.posterior(X, observation_noise=True)
        mean, cov = h.kron_exact_posterior(bt, X)
        _, cov_n = h.kron_exact_posterior(bt, X, noise=True)
        mm, mc = h.matheron_covariance(
            pb, lambda Z: pb.rsample_from_base_samples(torch.Size([Z.shape[0]]), Z.unsqueeze(-1))
        )
        _, mcn = h.matheron_covariance(
            pbn, lambda Z: pbn.rsample_from_base_samples(torch.Size([Z.shape[0]]), Z.unsqueeze(-1))
        )
    Xg = X.clone().requires_grad_(True)
    em, ec = h.kron_exact_posterior(bt, Xg)
    wm = torch.randn(q, m, generator=g, dtype=DT)
    wc = torch.randn(q * m, q * m, generator=g, dtype=DT)
    (dx,) = torch.autograd.grad((em * wm).sum() + (ec * wc).sum(), Xg)
    return {
        "x": X,
        "botorch_mean": pb.mean,
        "botorch_variance": pb.variance,
        "base_sample_shape": list(pb.base_sample_shape),
        "matheron_mean": mm.reshape(q, m),
        "matheron_covariance": mc,
        "matheron_covariance_noisy": mcn,
        "exact_mean": mean,
        "exact_covariance": cov,
        "exact_covariance_noisy": cov_n,
        "weights_mean": wm,
        "weights_cov": wc,
        "dx": dx,
    }


def kron_case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.acquisition.objective import LinearMCObjective
    from botorch.sampling.normal import SobolQMCNormalSampler

    _, h = _helpers()
    X, Y = h.kron_data(case, seed)
    bt, _ = h.build_kron_pair(case, X, Y, seed)
    out: dict[str, Any] = {
        "name": case["name"],
        "case": case,
        "train_x": X,
        "train_y": Y,
        "param_names": [(n, list(p.shape)) for n, p in bt.named_parameters()],
        "bounds": h.bounds_of(bt),
        "default_raw": h.flat(bt),
    }
    v, g = h.bt_mll_value_and_grad(bt)
    out["default_mll"] = {"value": v, "grad": g}
    if case.get("fit"):
        out["fit"] = kron_fit_record(case, bt)
    raw = h.random_raw(bt, seed + 1)
    h.set_flat(bt, raw)
    out["raw"] = raw
    v, g = h.bt_mll_value_and_grad(bt)
    out["mll"] = {"value": v, "grad": g}
    gx = torch.Generator().manual_seed(seed + 2)
    out["posterior"] = kron_posterior_record(
        bt, torch.rand(3, case["d"], generator=gx, dtype=DT), seed + 3
    )
    if case.get("acquisition"):
        exact = h.ExactKroneckerModel(bt)
        w = torch.tensor([0.6, -0.3, 0.5][: case["m"]], dtype=DT)
        Xc = torch.rand(3, 2, case["d"], generator=gx, dtype=DT)
        draws: dict[int, Any] = {}
        with recorded_draws(seed + 4, draws):
            sampler = SobolQMCNormalSampler(torch.Size([NUM_SAMPLES]), seed=0)
            acqf = qExpectedImprovement(
                exact,
                best_f=torch.tensor(0.2, dtype=DT),
                sampler=sampler,
                objective=LinearMCObjective(weights=w),
            )
            Xg = Xc.clone().requires_grad_(True)
            vals = acqf(Xg)
            (gr,) = torch.autograd.grad(vals.sum(), Xg)
        out["qei"] = {
            "weights": w,
            "best_f": 0.2,
            "x": Xc,
            "draws": dict(draws),
            "values": vals.detach(),
            "grads": gr,
        }
    return out


def kron_fit_record(case: dict[str, Any], bt: Any) -> dict[str, Any]:
    _, h = _helpers()

    def fit(model: Any) -> dict[str, Any]:
        import botorch.optim.fit as of
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        record: dict[str, Any] = {"attempts": 0}
        orig = of.fit_gpytorch_mll_scipy

        def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
            record["attempts"] += 1
            record["r"] = orig(mll, *args, **kwargs)
            return record["r"]

        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), optimizer=recorder)
        r = record["r"]
        return {
            "loss": float(r.fval),
            "iterations": int(r.step),
            "status": r.status.name,
            "attempts": record["attempts"],
            "raw_params": h.flat(model),
        }

    default = h.flat(bt)
    fitted = fit(copy.deepcopy(bt))
    twins = []

    def add_twins(ks: range) -> None:
        for kk in ks:
            tw = copy.deepcopy(bt)
            gg = torch.Generator().manual_seed(kk)
            Xt = tw.train_inputs[0].clone()
            rel = TWIN_SCALES[(kk - 1) % len(TWIN_SCALES)]
            Xt = Xt * (1 + rel * torch.randn(Xt.shape, generator=gg, dtype=DT))
            tw.set_train_data(inputs=(Xt,), strict=False)
            h.set_flat(tw, default)
            r = fit(tw)
            twins.append((Xt, r["loss"], r["attempts"]))

    add_twins(range(1, N_TWINS + 1))
    if f6a.is_chaotic(fitted, twins):
        add_twins(range(N_TWINS + 1, f6a.CHAOTIC_TWINS + 1))
    fitted["twin_train_x"] = [t for t, _, _ in twins]
    fitted["twin_losses"] = [loss for _, loss, _ in twins]
    fitted["twin_attempts"] = [a for _, _, a in twins]
    return fitted


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def generate_7a_multitask(out: str) -> None:
    out = os.path.join(out, "7a_multitask")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    for i, case in enumerate(mtgp_cases()):
        fixture = mtgp_case(case, seed=7000 + 17 * i)
        path = os.path.join(out, f"mtgp_{case['name']}.json")
        with open(path, "w") as f:
            json.dump(enc(fixture), f)
        fit_s = ""
        if "fit" in fixture:
            fl = fixture["fit"]["twin_losses"]
            fit_s = (
                f" fit loss={fixture['fit']['loss']:+.8f} twins=[{min(fl):+.8f}, "
                f"{max(fl):+.8f}] iters={fixture['fit']['iterations']}"
            )
        print(f"phase 7a-multitask: mtgp {case['name']:26s}{fit_s}", file=sys.stderr)
    for i, case in enumerate(kron_cases()):
        fixture = kron_case(case, seed=7500 + 17 * i)
        path = os.path.join(out, f"kron_{case['name']}.json")
        with open(path, "w") as f:
            json.dump(enc(fixture), f)
        fit_s = ""
        if "fit" in fixture:
            fl = fixture["fit"]["twin_losses"]
            fit_s = (
                f" fit loss={fixture['fit']['loss']:+.8f} twins=[{min(fl):+.8f}, "
                f"{max(fl):+.8f}] iters={fixture['fit']['iterations']}"
            )
        print(f"phase 7a-multitask: kron {case['name']:26s}{fit_s}", file=sys.stderr)
