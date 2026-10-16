"""Fixtures of kernel trees, GPyTorch's and BoTorch's kernels, priors and
constraints in ``SingleTaskGP`` (phase 6a).

Written to ``<out>/6a_kernels/`` by ``generate_fixtures.py --phase
6a-kernels`` (BoTorch main). Every case is a ``SingleTaskGP`` whose
``covar_module`` is built from a JSON *spec* (see
``botorch_pyo3/tests/kernel_tree_helpers.py``; the Rust test builds the same
kernel tree from it) and records:

* the parameter names and shapes in ``named_parameters()`` order, the
  optimizer bounds (``get_parameters_and_bounds``) and the default raw
  parameters (``IndexKernel`` / ``PositiveIndexKernel`` draw theirs from
  torch's generator; the Rust test starts from the recorded values);
* the marginal log likelihood and its gradient (BoTorch's autograd) at the
  default and at random raw parameters, and the prior kernel matrix
  ``covar_module(X, X)`` at the random parameters;
* posteriors at random points: ``q = 1`` means and variances (BoTorch's
  ``diag=True`` mode), with and without observation noise, and the
  gradients of the sums of the means and variances with respect to the
  points; a joint ``q = 3`` posterior (mean, covariance, with observation
  noise) and the gradient of a weighted sum of its mean and covariance;
* models with a Matérn-1/2 kernel are evaluated with GPyTorch's ``sq_dist``
  patched to zero the diagonal of a set's distances to itself also when the
  inputs require gradients (GPyTorch computes these distances by a quadratic
  expansion and zeroes the diagonal only without gradients, so the
  diagonal of the Matérn-1/2 kernel, which is not differentiable at zero,
  is ``exp(-sqrt(rounding noise)) = 1 - O(1e-8)`` instead of 1 in training
  mode), and ``HalfCauchyPrior`` is evaluated in double precision (torch
  casts its argument to the float32 dtype of the base ``Cauchy``);
  ``botorch_noise`` records the change of the MLL by these fixes;
* ``fit_gpytorch_mll`` from the default parameters (single-output cases):
  the final loss, iterations and raw parameters, and BoTorch's final losses
  on 16 twins whose training inputs are perturbed at 1e-14 .. 1e-11 (with
  their inputs, so that botorch_rs fits the same twins), and the number of
  attempts of every fit (retries start from random prior samples).

Non-finite numbers are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import warnings
from typing import Any

import torch

DT = torch.float64
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
N_TWINS = 16
TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "botorch_pyo3", "tests")


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


def _helpers() -> Any:
    sys.path.insert(0, os.path.normpath(TESTS))
    import kernel_tree_helpers

    return kernel_tree_helpers


def cases() -> list[dict[str, Any]]:
    h = _helpers()
    k, sub, prior, con = h.kernel, h.sub, h.prior, h.constraint
    f64 = {"__dtype__": "float64"}
    rbf2 = k("RBFKernel", ard_num_dims=2, active_dims=[0, 1])
    out = [
        ("rbf_shared_d3", k("RBFKernel"), 3, {}),
        ("matern12_ard_d2", k("MaternKernel", nu=0.5, ard_num_dims=2), 2, {"exact_diagonal": True}),
        (
            "matern32_scale_d2",
            k("ScaleKernel", sub(k("MaternKernel", nu=1.5, ard_num_dims=2))),
            2,
            {},
        ),
        ("matern52_active_d3", k("MaternKernel", ard_num_dims=2, active_dims=[0, 2]), 3, {}),
        ("rq_ard_d2", k("RQKernel", ard_num_dims=2), 2, {}),
        ("rq_interval_alpha_d2", k("RQKernel", alpha_constraint=con("Interval", 0.1, 5.0)), 2, {}),
        ("periodic_ard_d2", k("PeriodicKernel", ard_num_dims=2), 2, {}),
        ("periodic_shared_d2", k("ScaleKernel", sub(k("PeriodicKernel"))), 2, {}),
        ("linear_ard_d2", k("LinearKernel", ard_num_dims=2), 2, {}),
        (
            "linear_plus_rbf_d2",
            k("AdditiveKernel", sub(k("LinearKernel")), sub(k("RBFKernel", ard_num_dims=2))),
            2,
            {},
        ),
        # (BoTorch's fit of this model fails: no fit case.)
        (
            "polynomial2_d2",
            k("PolynomialKernel", power=2, offset_constraint=con("LessThan", 2.0)),
            2,
            {"fit": False},
        ),
        ("cosine_d1", k("CosineKernel", period_length_prior=prior("GammaPrior", 2.0, 1.0)), 1, {}),
        (
            "additive_active_d3",
            k(
                "AdditiveKernel",
                sub(k("RBFKernel", active_dims=[0])),
                sub(k("MaternKernel", nu=2.5, ard_num_dims=2, active_dims=[1, 2])),
            ),
            3,
            {},
        ),
        (
            "product_scale_d3",
            k(
                "ScaleKernel",
                sub(
                    k(
                        "ProductKernel",
                        sub(k("RBFKernel", ard_num_dims=2, active_dims=[0, 2])),
                        sub(k("PeriodicKernel", active_dims=[1])),
                    )
                ),
            ),
            3,
            {},
        ),
        (
            "index_d3",
            k(
                "ProductKernel",
                sub(rbf2),
                sub(k("IndexKernel", num_tasks=3, rank=2, active_dims=[2])),
            ),
            3,
            {"tasks": True},
        ),
        (
            "positive_index_d3",
            k(
                "ProductKernel",
                sub(rbf2),
                sub(
                    k(
                        "PositiveIndexKernel",
                        num_tasks=3,
                        rank=1,
                        active_dims=[2],
                        task_prior=prior("BetaPrior", 2.5, 1.5),
                    )
                ),
            ),
            3,
            {"tasks": True, "n": 18},
        ),
        (
            "categorical_d2",
            k("CategoricalKernel", ard_num_dims=2, lengthscale_prior=prior("GammaPrior", 2.0, 2.0)),
            2,
            {"categorical": True},
        ),
        (
            "downsampling_d1",
            k("DownsamplingKernel", power_prior=prior("GammaPrior", 2.0, 1.0)),
            1,
            {},
        ),
        (
            "expdecay_d1",
            k("ExponentialDecayKernel", offset_prior=prior("HalfNormalPrior", 1.0)),
            1,
            {},
        ),
        ("bnn_d2", k("InfiniteWidthBNNKernel", depth=3), 2, {}),
        ("ltf1_d3", k("LinearTruncatedFidelityKernel", fidelity_dims=[2], dimension=3), 3, {}),
        (
            "ltf2_d3",
            k("LinearTruncatedFidelityKernel", fidelity_dims=[1, 2], dimension=3, nu=1.5),
            3,
            {},
        ),
        ("oak_d3", k("OrthogonalAdditiveKernel", dim=3, quad_deg=16, dtype=f64), 3, {}),
        (
            "oak2_shared_d2",
            k(
                "OrthogonalAdditiveKernel",
                dim=2,
                quad_deg=8,
                second_order=True,
                per_dim_lengthscales=False,
                dtype=f64,
                coeffs_1_prior=prior("LogNormalPrior", -1.0, 1.0),
            ),
            2,
            {},
        ),
        (
            "priors_mix_d2",
            k(
                "AdditiveKernel",
                sub(
                    k(
                        "ScaleKernel",
                        sub(
                            k(
                                "RBFKernel",
                                ard_num_dims=2,
                                lengthscale_prior=prior("NormalPrior", 0.6, 0.5),
                                lengthscale_constraint=con(
                                    "GreaterThan", 0.05, transform=None, initial_value=0.4
                                ),
                            )
                        ),
                        outputscale_prior=prior("HalfNormalPrior", 2.0),
                    )
                ),
                sub(
                    k(
                        "LinearKernel",
                        variance_prior=prior("HalfCauchyPrior", 0.5),
                        variance_constraint=con("Interval", 1e-3, 4.0),
                    )
                ),
            ),
            2,
            {"double_half_cauchy": True},
        ),
        (
            "rbf_fixed_noise_d2",
            k("ScaleKernel", sub(k("RBFKernel", ard_num_dims=2, active_dims=[1, 0]))),
            2,
            {"fixed_noise": True},
        ),
        (
            "tree_m2_d2",
            k(
                "ScaleKernel",
                sub(
                    k(
                        "AdditiveKernel",
                        sub(k("RBFKernel", active_dims=[0])),
                        sub(k("MaternKernel", nu=0.5, active_dims=[1])),
                    )
                ),
            ),
            2,
            {"m": 2, "exact_diagonal": True},
        ),
        (
            "tree_nostd_d2",
            k(
                "ProductKernel",
                sub(k("RBFKernel", active_dims=[0])),
                sub(k("RQKernel", active_dims=[1])),
            ),
            2,
            {"standardize": False},
        ),
    ]
    return [{"name": n, "spec": s, "d": d, **o} for n, s, d, o in out]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def make_data(case: dict[str, Any], seed: int) -> tuple[Any, Any, Any]:
    g = torch.Generator().manual_seed(seed)
    n, d, m = case.get("n", 10), case["d"], case.get("m", 1)
    X = torch.rand(n, d, generator=g, dtype=DT)
    if case.get("tasks"):
        X[:, -1] = (torch.arange(n, dtype=DT) % 3) + 0.25
    if case.get("categorical"):
        X = torch.randint(0, 3, (n, d), generator=g).to(DT)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * X[:, 0]) + (1.0 + t) * (X[:, 1:] ** 2).sum(-1)
        cols.append(f + 0.05 * torch.randn(n, generator=g, dtype=DT))
    Y = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(n, m, generator=g, dtype=DT)
    return X, Y, Yvar


def test_points(case: dict[str, Any], q: int, seed: int) -> Any:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(q, case["d"], generator=g, dtype=DT)
    if case.get("tasks"):
        X[:, -1] = torch.tensor([float(i % 3) for i in range(q)], dtype=DT)
    if case.get("categorical"):
        X = torch.randint(0, 3, (q, case["d"]), generator=g).to(DT)
    return X


def build_model(case: dict[str, Any], X: Any, Y: Any, Yvar: Any, seed: int) -> Any:
    from botorch.models import SingleTaskGP

    h = _helpers()
    spec = case["spec"]
    if case.get("m", 1) > 1:
        spec = h.with_batch(spec, [case["m"]])
    torch.manual_seed(seed)
    kernel = h.build_kernel(spec, "bt")
    kw: dict[str, Any] = {}
    if not case.get("standardize", True):
        kw["outcome_transform"] = None
    return SingleTaskGP(
        X, Y, train_Yvar=Yvar if case.get("fixed_noise") else None, covar_module=kernel, **kw
    )


def flat(model: Any) -> list[float]:
    return [v for _, p in model.named_parameters() for v in p.detach().reshape(-1).tolist()]


def set_flat(model: Any, values: list[float]) -> None:
    offset = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            n = p.numel()
            p.copy_(torch.tensor(values[offset : offset + n], dtype=DT).reshape(p.shape))
            offset += n


def mll_value_and_grad(model: Any) -> tuple[float, list[float]]:
    from gpytorch.mlls import ExactMarginalLogLikelihood

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.zero_grad()
    out = mll(model(*model.train_inputs), model.train_targets)
    out.sum().backward()
    grad = []
    for _, p in model.named_parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grad.extend(g.reshape(-1).tolist())
    model.zero_grad()
    model.eval()
    return out.detach().reshape(-1).tolist(), grad


def bounds_of(model: Any) -> list[list[float]]:
    from botorch.optim.utils import get_parameters_and_bounds
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _, bounds = get_parameters_and_bounds(ExactMarginalLogLikelihood(model.likelihood, model))
    # The MLL's parameter names have a "model." prefix (except those of its
    # likelihood).
    bounds = {k.removeprefix("model."): v for k, v in bounds.items()}
    out = []
    for name, p in model.named_parameters():
        lo, hi = bounds.get(name, (-math.inf, math.inf))
        lo = -math.inf if lo is None else float(lo)
        hi = math.inf if hi is None else float(hi)
        out.extend([[lo, hi]] * p.numel())
    return out


def random_raw(model: Any, seed: int) -> list[float]:
    """Random raw parameters within the optimizer bounds."""
    g = torch.Generator().manual_seed(seed)
    bounds = bounds_of(model)
    values = []
    i = 0
    for name, p in model.named_parameters():
        for _ in range(p.numel()):
            lo, hi = bounds[i]
            if "raw_noise" in name:
                v = 0.02 + 0.06 * torch.rand(1, generator=g, dtype=DT).item()
                v = max(v, lo) if math.isfinite(lo) else math.log(math.expm1(v))
            elif "raw_constant" in name:
                v = 0.4 * torch.randn(1, generator=g, dtype=DT).item()
            else:
                v = 0.5 * torch.randn(1, generator=g, dtype=DT).item()
                if math.isfinite(lo):
                    v = lo + 0.1 + abs(v)
                if math.isfinite(hi):
                    v = min(v, hi - 0.05)
            values.append(v)
            i += 1
    return values


def posterior_record(model: Any, case: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    Xq = test_points(case, 4, 777).unsqueeze(-2)
    Xg = Xq.clone().requires_grad_(True)
    post = model.posterior(Xg)
    mean, var = post.mean, post.variance
    # (Categorical kernels are not differentiable: zero gradients.)
    (dm,) = torch.autograd.grad(mean.sum(), Xg, retain_graph=True, allow_unused=True)
    (dv,) = torch.autograd.grad(var.sum(), Xg, allow_unused=True)
    dm = torch.zeros_like(Xq) if dm is None else dm
    dv = torch.zeros_like(Xq) if dv is None else dv
    with torch.no_grad():
        noisy = model.posterior(Xq, observation_noise=True).variance
    Xj = test_points(case, 3, 778)
    Xjg = Xj.clone().requires_grad_(True)
    m_out = mean.shape[-1]
    g = torch.Generator().manual_seed(5)
    wm = torch.randn(m_out, 3, generator=g, dtype=DT)
    wc = torch.randn(m_out, 3, 3, generator=g, dtype=DT)
    s = torch.zeros((), dtype=DT)
    means, covs, covs_noisy = [], [], []
    # (BoTorch main's `posterior(X, output_indices=[t])` raises for models
    # with several outputs, fixed in M-Colley/botorch fix/models-posteriors:
    # the blocks of the joint posterior are used instead.)
    pj = model.posterior(Xjg)
    with torch.no_grad():
        pn = model.posterior(Xj, observation_noise=True)

    def block(p: Any, t: int) -> Any:
        cov = p.distribution.covariance_matrix
        if m_out == 1:
            return cov.reshape(3, 3)
        if getattr(p.distribution, "_interleaved", True):
            return cov[t::m_out, t::m_out]
        return cov[t * 3 : (t + 1) * 3, t * 3 : (t + 1) * 3]

    for t in range(m_out):
        mt = pj.mean.reshape(3, m_out)[:, t]
        ct = block(pj, t)
        s = s + (wm[t] * mt).sum() + (wc[t] * ct).sum()
        means.append(mt.detach())
        covs.append(ct.detach())
        covs_noisy.append(block(pn, t))
    (dj,) = torch.autograd.grad(s, Xjg, allow_unused=True)
    dj = torch.zeros_like(Xj) if dj is None else dj
    return {
        "x": Xq.squeeze(-2),
        "mean": mean.detach().reshape(4, -1),
        "variance": var.detach().reshape(4, -1),
        "variance_noisy": noisy.reshape(4, -1),
        "dmean_dx": dm.squeeze(-2),
        "dvariance_dx": dv.squeeze(-2),
        "joint_x": Xj,
        "joint_mean": means,
        "joint_covariance": covs,
        "joint_covariance_noisy": covs_noisy,
        "weights_mean": wm,
        "weights_cov": wc,
        "djoint_dx": dj,
    }


def perturbed_twin(model: Any, case: dict[str, Any], seed: int, rel: float) -> Any:
    import copy

    twin = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    X = twin.train_inputs[0].clone()
    cols = list(range(X.shape[-1]))
    if case.get("tasks"):
        cols = cols[:-1]
    if not case.get("categorical"):
        X[..., cols] = X[..., cols] * (
            1 + rel * torch.randn(X[..., cols].shape, generator=g, dtype=DT)
        )
    twin.train_inputs = (X,)
    twin.prediction_strategy = None
    return twin


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
        "raw_params": flat(model),
    }


def model_case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    """The fixture of ``case``; with BoTorch's numerical quirks fixed for the
    cases that need it (``exact_diagonal``, ``double_half_cauchy``), and the
    change of the MLL by the fix as ``botorch_noise``."""
    h = _helpers()
    fixes = []
    if case.get("exact_diagonal"):
        fixes.append(h.zero_diagonal_distances)
    if case.get("double_half_cauchy"):
        fixes.append(h.double_precision_half_cauchy)
    if not fixes:
        return _model_case(case, seed)
    X, Y, Yvar = make_data(case, seed)
    model = build_model(case, X, Y, Yvar, seed)
    set_flat(model, random_raw(model, seed + 1))
    plain, _ = mll_value_and_grad(model)
    with contextlib.ExitStack() as stack:
        for fix in fixes:
            stack.enter_context(fix())
        out = _model_case(case, seed)
    out["botorch_noise"] = max(abs(a - b) for a, b in zip(plain, out["mll"]["value"], strict=True))
    return out


def _model_case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    X, Y, Yvar = make_data(case, seed)
    model = build_model(case, X, Y, Yvar, seed)
    names = [(n, list(p.shape)) for n, p in model.named_parameters()]
    default_raw = flat(model)
    dv, dg = mll_value_and_grad(model)
    raw = random_raw(model, seed + 1)
    set_flat(model, raw)
    v, g = mll_value_and_grad(model)
    with torch.no_grad():
        model.train()
        k_train = model.covar_module(model.train_inputs[0]).to_dense()
        model.eval()
    out: dict[str, Any] = {
        "name": case["name"],
        "spec": case["spec"],
        "d": case["d"],
        "m": case.get("m", 1),
        "fixed_noise": bool(case.get("fixed_noise")),
        "standardize": case.get("standardize", True),
        "train_x": X,
        "train_y": Y,
        "train_yvar": Yvar if case.get("fixed_noise") else None,
        "param_names": names,
        "bounds": bounds_of(model),
        "default_raw": default_raw,
        "default_mll": {"value": dv, "grad": dg},
        "raw": raw,
        "mll": {"value": v, "grad": g},
        "k_train": k_train,
        "posterior": posterior_record(model, case),
    }
    if case.get("m", 1) == 1 and case.get("fit", True):
        set_flat(model, default_raw)
        fitted = fit(model)
        twins = []
        for kk in range(1, N_TWINS + 1):
            m0 = build_model(case, X, Y, Yvar, seed)
            set_flat(m0, default_raw)
            twin = perturbed_twin(m0, case, kk, TWIN_SCALES[(kk - 1) % len(TWIN_SCALES)])
            r = fit(twin)
            twins.append((twin.train_inputs[0].clone(), r["loss"], r["attempts"]))
        fitted["twin_train_x"] = [t for t, _, _ in twins]
        fitted["twin_losses"] = [loss for _, loss, _ in twins]
        fitted["twin_attempts"] = [a for _, _, a in twins]
        out["fit"] = fitted
    return out


def generate_6a_kernels(out: str) -> None:
    out = os.path.join(out, "6a_kernels")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    for i, case in enumerate(cases()):
        fixture = model_case(case, seed=6000 + 17 * i)
        path = os.path.join(out, f"{case['name']}.json")
        with open(path, "w") as f:
            json.dump(enc(fixture), f)
        fit_s = ""
        if "fit" in fixture:
            fl = fixture["fit"]["twin_losses"]
            fit_s = (
                f" fit loss={fixture['fit']['loss']:+.8f} twins=[{min(fl):+.8f}, "
                f"{max(fl):+.8f}] iters={fixture['fit']['iterations']}"
            )
        print(f"phase 6a-kernels: {case['name']:24s}{fit_s}", file=sys.stderr)
