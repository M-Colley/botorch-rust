"""Fixtures of the models of phase 7b: ``SingleTaskMultiFidelityGP``,
``MixedSingleTaskGP`` and the MAP SAAS models (``get_map_saas_model``,
``AdditiveMapSaasSingleTaskGP``, ``EnsembleMapSaasSingleTaskGP``).

Written to ``<out>/7b_models/`` by ``generate_fixtures.py --phase 7b-models``
(BoTorch main). Every case is a BoTorch model whose covariance module is
described by a JSON kernel *spec* (the format of
``botorch_pyo3/tests/kernel_tree_helpers.py``, extended by the pseudo-kernel
``DimScaledRBF`` for BoTorch's ``get_covar_module_with_dim_scaled_prior``, the
``LogTransformedInterval`` constraint, double-precision priors and the SAAS
prior ``saas`` of a ``MaternKernel``), together with the mean and likelihood
modules, the batch shape of an ensemble and the fixed SAAS taus of every
element; the Rust test builds the same model from it. Recorded:

* the parameter names and shapes in ``named_parameters()`` order, the
  optimizer bounds and the default raw parameters;
* the MLL of every element and its gradient (BoTorch's autograd) at the
  default and at random raw parameters, and the prior kernel matrices;
* posteriors of every element and output at random points: ``q = 1`` means
  and variances (with and without observation noise) and the gradients of
  their sums over points and elements with respect to the points; joint
  ``q = 3`` means and covariances (with observation noise) and the gradient
  of a weighted sum of them;
* ``fit_gpytorch_mll`` from the default parameters (single-output,
  unbatched cases), with BoTorch's final losses on 16 twins whose training
  inputs are perturbed at 1e-14 .. 1e-11 (first attempts are compared).

Matérn-1/2 kernels (``nu=0.5``) are evaluated with GPyTorch's ``sq_dist``
zeroing the diagonal also with gradients (``exact_diagonal``, see
``fixtures_6a_kernels.py``); ``botorch_noise`` records the change.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures_6a_kernels as f6a  # noqa: E402


def _h() -> Any:
    return f6a._helpers()


# ---------------------------------------------------------------------------
# Specs of the covariance modules
# ---------------------------------------------------------------------------


def _k(type_: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return _h().kernel(type_, *args, **kwargs)


def _sub(spec: dict[str, Any]) -> dict[str, Any]:
    return {"__kernel__": spec}


def _prior(name: str, *args: Any, double: bool = False) -> dict[str, Any]:
    p: dict[str, Any] = {"__prior__": [name, list(args)]}
    if double:
        p["__prior__"].append({"double": True})
    return p


def _con(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"__constraint__": [name, list(args), kwargs]}


def _batch(kw: dict[str, Any], batch: list[int]) -> dict[str, Any]:
    if batch:
        kw["batch_shape"] = list(batch)
    return kw


def mf_spec(
    d: int,
    batch: list[int],
    iteration_fidelity: int | None,
    data_fidelities: list[int] | None,
    linear_truncated: bool,
    nu: float,
) -> dict[str, Any]:
    """The spec of BoTorch's ``_setup_multifidelity_covar_module``."""
    if iteration_fidelity is not None and iteration_fidelity < 0:
        iteration_fidelity += d
    if data_fidelities is not None:
        data_fidelities = [f + d if f < 0 else f for f in data_fidelities]
    if linear_truncated:
        lead = [iteration_fidelity] if iteration_fidelity is not None else []
        trailing = [[i] for i in data_fidelities] if data_fidelities else [[]]
        kernels = [
            _sub(
                _k(
                    "LinearTruncatedFidelityKernel",
                    **_batch(
                        {
                            "fidelity_dims": lead + t,
                            "dimension": d,
                            "nu": nu,
                            "power_prior": _prior("GammaPrior", 3.0, 3.0),
                        },
                        batch,
                    ),
                )
            )
            for t in trailing
        ]
        return _k(
            "ScaleKernel",
            _sub(_k("ProductKernel", *kernels)),
            **_batch({"outputscale_prior": _prior("GammaPrior", 2.0, 0.15)}, batch),
        )
    non_active = set(data_fidelities or [])
    if iteration_fidelity is not None:
        non_active.add(iteration_fidelity)
    active = sorted(set(range(d)) - non_active)
    kernels = [
        _sub(
            _k(
                "DimScaledRBF",
                **_batch({"ard_num_dims": len(active), "active_dims": active}, batch),
            )
        )
    ]
    g36 = _prior("GammaPrior", 3.0, 6.0)
    if iteration_fidelity is not None:
        kernels.append(
            _sub(
                _k(
                    "ExponentialDecayKernel",
                    **_batch(
                        {
                            "lengthscale_prior": g36,
                            "offset_prior": g36,
                            "power_prior": g36,
                            "active_dims": [iteration_fidelity],
                        },
                        batch,
                    ),
                )
            )
        )
    for f in data_fidelities or []:
        kernels.append(
            _sub(
                _k(
                    "DownsamplingKernel",
                    **_batch({"offset_prior": g36, "power_prior": g36, "active_dims": [f]}, batch),
                )
            )
        )
    return _k("ProductKernel", *kernels)


def mixed_spec(d: int, cat_dims: list[int], batch: list[int]) -> dict[str, Any]:
    """The spec of ``MixedSingleTaskGP``'s covariance module (default
    ``cont_kernel_factory``)."""
    cat_dims = [c + d if c < 0 else c for c in cat_dims]
    ord_dims = sorted(set(range(d)) - set(cat_dims))
    gt = _con("GreaterThan", 1e-06)

    def cat(active: bool) -> dict[str, Any]:
        kw: dict[str, Any] = {"ard_num_dims": len(cat_dims), "lengthscale_constraint": gt}
        if active:
            kw["active_dims"] = cat_dims
        return _k("CategoricalKernel", **_batch(kw, batch))

    if not ord_dims:
        return _k("ScaleKernel", _sub(cat(False)))
    cont = _k(
        "DimScaledRBF", **_batch({"ard_num_dims": len(ord_dims), "active_dims": ord_dims}, batch)
    )
    sum_kernel = _k(
        "ScaleKernel",
        _sub(_k("AdditiveKernel", _sub(cont), _sub(_k("ScaleKernel", _sub(cat(True)))))),
    )
    prod_kernel = _k("ScaleKernel", _sub(_k("ProductKernel", _sub(cont), _sub(cat(True)))))
    return _k("AdditiveKernel", _sub(sum_kernel), _sub(prod_kernel))


def saas_matern(d: int, batch: list[int], tau: float | None) -> dict[str, Any]:
    """A Matérn-5/2 kernel with BoTorch's SAAS prior (``add_saas_prior``)."""
    kw: dict[str, Any] = {
        "nu": 2.5,
        "ard_num_dims": d,
        "lengthscale_constraint": _con("LogTransformedInterval", 0.01, 1e4, initial_value=1.0),
        "saas": {"tau": tau},
    }
    return _k("MaternKernel", **_batch(kw, batch))


def saas_scale(base: dict[str, Any], batch: list[int]) -> dict[str, Any]:
    kw = {"outputscale_constraint": _con("LogTransformedInterval", 1e-2, 1e4, initial_value=10.0)}
    return _k("ScaleKernel", _sub(base), **_batch(kw, batch))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def cases() -> list[dict[str, Any]]:
    """``name``, the model ``kind`` and its options, input dimension ``d``,
    outputs ``m``, fixed noise and whether to fit."""
    return [
        {"name": "mf_ltf_data", "kind": "mf", "d": 4, "mf": {"data_fidelities": [3]}},
        {
            "name": "mf_ltf_iteration_nu15",
            "kind": "mf",
            "d": 3,
            "mf": {"iteration_fidelity": 2, "nu": 1.5},
        },
        {
            "name": "mf_ltf_both_nu05",
            "kind": "mf",
            "d": 4,
            "mf": {"iteration_fidelity": 2, "data_fidelities": [3], "nu": 0.5},
            "exact_diagonal": True,
        },
        {
            "name": "mf_ltf_two_data_m2",
            "kind": "mf",
            "d": 4,
            "m": 2,
            "mf": {"data_fidelities": [2, 3]},
        },
        {
            "name": "mf_ds_both",
            "kind": "mf",
            "d": 4,
            "mf": {"iteration_fidelity": 1, "data_fidelities": [2, 3], "linear_truncated": False},
        },
        {
            "name": "mf_ds_data_fixed",
            "kind": "mf",
            "d": 3,
            "fixed_noise": True,
            "mf": {"data_fidelities": [-1], "linear_truncated": False},
        },
        {"name": "mixed_last", "kind": "mixed", "d": 3, "cat_dims": [-1], "cats": [3]},
        {
            "name": "mixed_two_cats_m2",
            "kind": "mixed",
            "d": 4,
            "m": 2,
            "cat_dims": [1, 3],
            "cats": [3, 2],
        },
        {
            "name": "mixed_all_categorical",
            "kind": "mixed",
            "d": 2,
            "cat_dims": [0, 1],
            "cats": [3, 3],
        },
        {"name": "saas_map", "kind": "saas_map", "d": 4},
        {"name": "saas_map_tau", "kind": "saas_map", "d": 3, "tau": 0.25},
        {"name": "saas_map_fixed_m2", "kind": "saas_map", "d": 3, "m": 2, "fixed_noise": True},
        {"name": "saas_additive", "kind": "saas_additive", "d": 3, "num_taus": 3},
        {"name": "saas_ensemble", "kind": "saas_ensemble", "d": 3, "num_taus": 3},
    ]


def make_data(case: dict[str, Any], seed: int) -> tuple[Any, Any, Any]:
    g = torch.Generator().manual_seed(seed)
    n, d, m = case.get("n", 10), case["d"], case.get("m", 1)
    X = torch.rand(n, d, generator=g, dtype=DT)
    if case["kind"] == "mixed":
        cat_dims = [c + d if c < 0 else c for c in case["cat_dims"]]
        for c, k in zip(cat_dims, case["cats"], strict=True):
            X[:, c] = torch.randint(0, k, (n,), generator=g).to(DT)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * X[:, 0]) + (1.0 + t) * (X[:, 1:] ** 2).sum(-1)
        cols.append(f + 0.05 * torch.randn(n, generator=g, dtype=DT))
    Y = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(n, m, generator=g, dtype=DT)
    return X, Y, Yvar


def build_model(case: dict[str, Any], X: Any, Y: Any, Yvar: Any, seed: int) -> Any:
    from botorch.models import MixedSingleTaskGP, SingleTaskMultiFidelityGP, map_saas

    yvar = Yvar if case.get("fixed_noise") else None
    kind = case["kind"]
    torch.manual_seed(seed)
    if kind == "mf":
        return SingleTaskMultiFidelityGP(X, Y, train_Yvar=yvar, **case["mf"])
    if kind == "mixed":
        return MixedSingleTaskGP(X, Y, cat_dims=case["cat_dims"], train_Yvar=yvar)
    if kind == "saas_map":
        tau = case.get("tau")
        return map_saas.get_map_saas_model(X, Y, train_Yvar=yvar, tau=tau)
    if kind == "saas_additive":
        return map_saas.AdditiveMapSaasSingleTaskGP(
            X, Y, train_Yvar=yvar, num_taus=case["num_taus"]
        )
    return map_saas.EnsembleMapSaasSingleTaskGP(X, Y, train_Yvar=yvar, num_taus=case["num_taus"])


def spec_of(case: dict[str, Any], model: Any) -> dict[str, Any]:
    """The spec of the model's covariance, mean and likelihood modules and
    the fixed SAAS taus of every element (from the BoTorch model)."""
    d, m = case["d"], case.get("m", 1)
    kind = case["kind"]
    batch = list(model._aug_batch_shape)
    out: dict[str, Any] = {"batch": [], "mean": None, "likelihood": None, "saas_taus": None}
    if kind == "mf":
        mf = case["mf"]
        out["spec"] = mf_spec(
            d,
            batch,
            mf.get("iteration_fidelity"),
            mf.get("data_fidelities"),
            mf.get("linear_truncated", True),
            mf.get("nu", 2.5),
        )
        return out
    if kind == "mixed":
        out["spec"] = mixed_spec(d, case["cat_dims"], batch)
        return out
    normal_mean = {
        "prior": _prior("NormalPrior", 0.0, 1.0),
        "constraint": _con("Interval", -10.0, 10.0, initial_value=0.0, transform=None),
    }
    gamma_lik = {
        "prior": _prior("GammaPrior", 0.9, 10.0),
        "constraint": _con("LogTransformedInterval", 1e-4, 1.0, initial_value=1e-2),
    }
    out["mean"] = normal_mean
    n_el = max(math.prod(batch), 1)
    if kind == "saas_map":
        tau = case.get("tau")
        if tau is not None:
            tau = float(torch.tensor(tau).item())  # BoTorch: float32
        out["spec"] = saas_scale(saas_matern(d, batch, tau), batch)
        if not case.get("fixed_noise"):
            out["likelihood"] = gamma_lik
        return out
    if kind == "saas_additive":
        taus = [_saas_taus(k.base_kernel, n_el) for k in model.covar_module.kernels]
        out["spec"] = _k(
            "AdditiveKernel",
            *[_sub(saas_scale(saas_matern(d, batch, t[0]), batch)) for t in taus],
        )
        out["saas_taus"] = [[t[e] for t in taus] for e in range(n_el)]
        return out
    # Ensemble: a batched model of num_taus repeated data sets.
    taus = [float(v) for v in model.taus.reshape(-1).tolist()]
    out["batch"] = [len(taus)]
    out["spec"] = saas_scale(saas_matern(d, batch, taus[0]), batch)
    out["saas_taus"] = [[t] for t in taus]
    if not case.get("fixed_noise"):
        out["likelihood"] = gamma_lik
    del m
    return out


def _saas_taus(kernel: Any, n_el: int) -> list[float]:
    """The fixed taus of a SAAS kernel (``SaasPriorHelper``), one per element."""
    for _, _, _, closure, _ in kernel.named_priors():
        helper = getattr(closure, "__self__", None)
        if helper is not None and getattr(helper, "_tau", None) is not None:
            t = helper._tau.to(DT).reshape(-1)
            values = [float(v) for v in t.tolist()]
            return values * n_el if len(values) == 1 else values
    raise RuntimeError("no fixed tau")


def posterior_record(model: Any, case: dict[str, Any], batched: bool) -> dict[str, Any]:
    """Posteriors of every element (``SingleTaskGP.posterior``, without the
    ensemble's mixture), see the module documentation."""
    from botorch.models import SingleTaskGP

    model.eval()
    g = torch.Generator().manual_seed(777)
    d = case["d"]

    def pts(q: int) -> Any:
        X = torch.rand(q, d, generator=g, dtype=DT)
        if case["kind"] == "mixed":
            cat_dims = [c + d if c < 0 else c for c in case["cat_dims"]]
            for c, k in zip(cat_dims, case["cats"], strict=True):
                X[:, c] = torch.randint(0, k, (q,), generator=g).to(DT)
        return X

    Xq = pts(4)
    # One point at a time (``[1 x] 1 x d``), against every element: BoTorch
    # fails on a batch of single points for some models (see the draft:
    # linear_operator's MulLinearOperator with batched outputscales).
    means, variances, noisies, dms, dvs = [], [], [], [], []
    for i in range(4):
        Xs = Xq[i : i + 1].reshape(*([1] if batched else []), 1, d)
        Xg = Xs.clone().requires_grad_(True)
        post = SingleTaskGP.posterior(model, Xg)
        mean_i, var_i = post.mean, post.variance  # [K] x 1 x m
        (dm_i,) = torch.autograd.grad(mean_i.sum(), Xg, retain_graph=True, allow_unused=True)
        (dv_i,) = torch.autograd.grad(var_i.sum(), Xg, allow_unused=True)
        dms.append(torch.zeros_like(Xs) if dm_i is None else dm_i)
        dvs.append(torch.zeros_like(Xs) if dv_i is None else dv_i)
        with torch.no_grad():
            noisies.append(SingleTaskGP.posterior(model, Xs, observation_noise=True).variance)
        means.append(mean_i.detach())
        variances.append(var_i.detach())
    mean, var, noisy = torch.stack(means), torch.stack(variances), torch.stack(noisies)
    dm, dv = torch.stack(dms), torch.stack(dvs)
    m = mean.shape[-1]
    k = model.batch_shape[0] if batched else 1

    # Elements in the model's order (batch element b, output t): e = b m + t.
    def per_element(v: Any) -> list[list[float]]:
        v = v.detach().reshape(4, k, m)
        return [v[:, b, t].tolist() for b in range(k) for t in range(m)]

    Xj = pts(3)
    Xjg = Xj.clone().requires_grad_(True)
    gw = torch.Generator().manual_seed(5)
    wm = torch.randn(k * m, 3, generator=gw, dtype=DT)
    wc = torch.randn(k * m, 3, 3, generator=gw, dtype=DT)
    pj = SingleTaskGP.posterior(model, Xjg)
    with torch.no_grad():
        pn = SingleTaskGP.posterior(model, Xj, observation_noise=True)

    def block(p: Any, b: int, t: int) -> Any:
        cov = p.distribution.covariance_matrix
        if batched:
            cov = cov[b]
        if m == 1:
            return cov.reshape(3, 3)
        if getattr(p.distribution, "_interleaved", True):
            return cov[t::m, t::m]
        return cov[t * 3 : (t + 1) * 3, t * 3 : (t + 1) * 3]

    s = torch.zeros((), dtype=DT)
    means, covs, covs_noisy = [], [], []
    for b in range(k):
        for t in range(m):
            e = b * m + t
            mt = (pj.mean[b] if batched else pj.mean).reshape(3, m)[:, t]
            ct = block(pj, b, t)
            s = s + (wm[e] * mt).sum() + (wc[e] * ct).sum()
            means.append(mt.detach())
            covs.append(ct.detach())
            covs_noisy.append(block(pn, b, t))
    (dj,) = torch.autograd.grad(s, Xjg, allow_unused=True)
    dj = torch.zeros_like(Xj) if dj is None else dj
    return {
        "x": Xq,
        "mean": per_element(mean),
        "variance": per_element(var),
        "variance_noisy": per_element(noisy),
        "dmean_dx": dm.reshape(4, d),
        "dvariance_dx": dv.reshape(4, d),
        "joint_x": Xj,
        "joint_mean": means,
        "joint_covariance": covs,
        "joint_covariance_noisy": covs_noisy,
        "weights_mean": wm,
        "weights_cov": wc,
        "djoint_dx": dj,
    }


def _posterior_record_eager(model: Any, case: dict[str, Any], batched: bool) -> dict[str, Any]:
    """``posterior_record``, with eagerly evaluated kernels where BoTorch's
    lazy evaluation fails at single points for a multi-output model with a
    product of two LinearTruncatedFidelityKernels (linear_operator's
    ``MulLinearOperator._mul_constant`` compares a batched outputscale with
    0, see the draft)."""
    import gpytorch

    try:
        return posterior_record(model, case, batched)
    except RuntimeError:
        with gpytorch.settings.lazily_evaluate_kernels(False):
            return posterior_record(model, case, batched)


def _case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    X, Y, Yvar = make_data(case, seed)
    model = build_model(case, X, Y, Yvar, seed)
    batched = case["kind"] == "saas_ensemble"
    names = [(n, list(p.shape)) for n, p in model.named_parameters()]
    default_raw = f6a.flat(model)
    dv, dg = f6a.mll_value_and_grad(model)
    raw = f6a.random_raw(model, seed + 1)
    f6a.set_flat(model, raw)
    v, g = f6a.mll_value_and_grad(model)
    import gpytorch

    # (Eagerly: linear_operator's MulLinearOperator._mul_constant fails for
    # the batched outputscales of a ScaleKernel around a product of
    # kernels, ``if other > 0`` on a tensor.)
    with torch.no_grad(), gpytorch.settings.lazily_evaluate_kernels(False):
        model.train()
        k_train = model.covar_module(model.train_inputs[0]).to_dense()
        model.eval()
    out: dict[str, Any] = {
        "name": case["name"],
        **spec_of(case, model),
        "d": case["d"],
        "m": case.get("m", 1),
        "fixed_noise": bool(case.get("fixed_noise")),
        # (``get_map_saas_model`` has no outcome transform by default.)
        "standardize": hasattr(model, "outcome_transform"),
        "train_x": X,
        "train_y": Y,
        "train_yvar": Yvar if case.get("fixed_noise") else None,
        "param_names": names,
        "bounds": f6a.bounds_of(model),
        "default_raw": default_raw,
        "default_mll": {"value": dv, "grad": dg},
        "raw": raw,
        "mll": {"value": v, "grad": g},
        "k_train": k_train,
        "posterior": _posterior_record_eager(model, case, batched),
    }
    if case.get("m", 1) == 1 and not batched and case.get("fit", True):
        f6a.set_flat(model, default_raw)
        fitted = f6a.fit(model)
        twins = []
        for kk in range(1, N_TWINS + 1):
            m0 = build_model(case, X, Y, Yvar, seed)
            f6a.set_flat(m0, default_raw)
            twin = f6a.perturbed_twin(
                m0,
                {"categorical": case["kind"] == "mixed"},
                kk,
                TWIN_SCALES[(kk - 1) % len(TWIN_SCALES)],
            )
            if case["kind"] == "mixed":
                # Perturb the continuous columns only.
                d = case["d"]
                cat_dims = {c + d if c < 0 else c for c in case["cat_dims"]}
                gg = torch.Generator().manual_seed(kk)
                Xt = m0.train_inputs[0].clone()
                rel = TWIN_SCALES[(kk - 1) % len(TWIN_SCALES)]
                for c in range(d):
                    if c not in cat_dims:
                        Xt[:, c] = Xt[:, c] * (
                            1 + rel * torch.randn(Xt.shape[0], generator=gg, dtype=DT)
                        )
                twin = m0
                twin.train_inputs = (Xt,)
                twin.prediction_strategy = None
            r = f6a.fit(twin)
            twins.append((twin.train_inputs[0].clone(), r["loss"], r["attempts"]))
        fitted["twin_train_x"] = [t for t, _, _ in twins]
        fitted["twin_losses"] = [loss for _, loss, _ in twins]
        fitted["twin_attempts"] = [a for _, _, a in twins]
        out["fit"] = fitted
    return out


def model_case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    if not case.get("exact_diagonal"):
        return _case(case, seed)
    X, Y, Yvar = make_data(case, seed)
    model = build_model(case, X, Y, Yvar, seed)
    f6a.set_flat(model, f6a.random_raw(model, seed + 1))
    plain, _ = f6a.mll_value_and_grad(model)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_h().zero_diagonal_distances())
        out = _case(case, seed)
    out["botorch_noise"] = max(abs(a - b) for a, b in zip(plain, out["mll"]["value"], strict=True))
    return out


def generate_7b_models(out: str) -> None:
    out = os.path.join(out, "7b_models")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    for i, case in enumerate(cases()):
        fixture = model_case(case, seed=7700 + 17 * i)
        path = os.path.join(out, f"{case['name']}.json")
        with open(path, "w") as f:
            json.dump(f6a.enc(fixture), f)
        fit_s = ""
        if "fit" in fixture:
            fl = fixture["fit"]["twin_losses"]
            fit_s = (
                f" fit loss={fixture['fit']['loss']:+.8f} twins=[{min(fl):+.8f}, "
                f"{max(fl):+.8f}] iters={fixture['fit']['iterations']}"
            )
        print(f"phase 7b-models: {case['name']:24s}{fit_s}", file=sys.stderr)
