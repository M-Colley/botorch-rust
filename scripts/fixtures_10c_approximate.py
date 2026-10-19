"""Fixtures of phase 10c (approximate GPs): BoTorch's
``SingleTaskVariationalGP`` with GPyTorch's ``VariationalELBO``, and the
inducing point allocators of ``botorch.models.utils.inducing_point_allocators``.

Written to ``<out>/10c_approximate/`` by ``generate_fixtures.py --phase
10c-approximate`` (BoTorch main):

* ``allocators.json``: ``_pivoted_cholesky_init`` on kernel matrices of
  random inputs (with unit and random quality scores, early stops on
  duplicated inputs) and the expected-improvement quality scores of a
  previous ``SingleTaskGP``'s posterior.
* ``models.json``: the cases of ``botorch_pyo3/tests/approximate_helpers.py``
  (kernels as JSON specs of ``kernel_tree_helpers.py``, BoTorch's default as
  the pseudo-kernel ``DimScaledRBF``): the data (raw and transformed), the
  inducing points chosen at construction, parameter names, sizes, bounds and
  initial values; at random parameters the ELBO terms and the autograd
  gradient of the ELBO (several ``num_data`` / ``beta``), and posteriors at
  test points (``q = 1`` batches: means, variances and the autograd
  gradient of their sum; a joint set: mean, covariance, with observation
  noise, and the gradient of a weighted sum).
* ``fits.json``: ``fit_gpytorch_mll`` (L-BFGS-B) on 64 twins whose training
  inputs are perturbed at 1e-14 .. 1e-11 (each twin's initial parameters,
  final loss and attempts).
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
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
FIT_TWINS = 64
TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "botorch_pyo3", "tests")


def _tests() -> Any:
    sys.path.insert(0, os.path.normpath(TESTS))
    import approximate_helpers

    return approximate_helpers


def _flat(t: Any) -> list[float]:
    return [float(v) for v in torch.as_tensor(t, dtype=DT).reshape(-1).tolist()]


def _rows(t: Any) -> list[list[float]]:
    t = torch.as_tensor(t, dtype=DT)
    return [[float(v) for v in row] for row in t.reshape(t.shape[0], -1).tolist()]


# ---------------------------------------------------------------------------
# Allocators
# ---------------------------------------------------------------------------


def allocator_fixture() -> dict[str, Any]:
    from botorch.models import SingleTaskGP
    from botorch.models.utils import inducing_point_allocators as ipa
    from gpytorch.kernels import MaternKernel, RBFKernel

    cases = []
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        n = 10 + 4 * seed
        X = torch.rand(n, 2, generator=g, dtype=DT)
        if seed % 3 == 2:
            X = torch.cat([X, X[:4]])
        k = (MaternKernel(nu=1.5) if seed % 2 else RBFKernel()).to(DT)
        k.raw_lengthscale.data.fill_(-0.5 - 0.2 * seed)
        with torch.no_grad():
            K = k(X).to_dense()
        q = torch.rand(X.shape[0], generator=g, dtype=DT) + 0.05
        for max_length in (1, 4, X.shape[0]):
            for quality in (torch.ones(X.shape[0], dtype=DT), q):
                sel = ipa._pivoted_cholesky_init(X, K, max_length, quality)
                idx = [int((X == row).all(-1).nonzero()[0]) for row in sel]
                cases.append(
                    {
                        "kernel_matrix": _rows(K),
                        "max_length": max_length,
                        "quality": _flat(quality),
                        "selected_points": _rows(sel) if len(sel) else [],
                        "selected": idx,
                        "x": _rows(X),
                    }
                )
    g = torch.Generator().manual_seed(42)
    X = torch.rand(12, 2, generator=g, dtype=DT)
    Y = torch.sin(4 * X[:, :1]) + X[:, 1:]
    gp = SingleTaskGP(X, Y)
    Xn = torch.rand(15, 2, generator=g, dtype=DT)
    post = gp.posterior(Xn)
    ei = []
    for maximize in (True, False):
        ei.append(
            {
                "maximize": maximize,
                "mean": _flat(post.mean.detach()),
                "variance": _flat(post.variance.detach()),
                "quality": _flat(ipa.ExpectedImprovementQualityFunction(gp, maximize)(Xn)),
            }
        )
    return {"pivoted": cases, "ei_quality": ei}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _kernel_rs(case: Any) -> dict[str, Any]:
    if case.kernel is not None:
        return case.kernel
    return {"type": "DimScaledRBF", "args": [], "kwargs": {"ard_num_dims": case.d}}


def _description(case: Any, bt: Any, X: Any, Y: Any) -> dict[str, Any]:
    from gpytorch.mlls import VariationalELBO

    h = _tests()
    mll = VariationalELBO(bt.likelihood, bt.model, num_data=case.n)
    from botorch.optim.utils import get_parameters_and_bounds

    _, bounds = get_parameters_and_bounds(mll)
    names, sizes, lower, upper, raw = [], [], [], [], []
    for name, p in mll.named_parameters():
        names.append(name)
        sizes.append(p.numel())
        lo, hi = bounds.get(name, (None, None))
        lower += [-math.inf if lo is None else float(lo)] * p.numel()
        upper += [math.inf if hi is None else float(hi)] * p.numel()
        raw += _flat(p.detach())
    out: dict[str, Any] = {
        "name": case.name,
        "m": case.m,
        "learn_inducing_points": case.learn,
        "kernel": _kernel_rs(case),
        "mean": case.mean,
        "likelihood": case.likelihood,
        "input_tf": case.input_tf,
        "outcome_tf": case.outcome_tf,
        "train_x": _rows(X),
        "train_y": _rows(Y),
        "train_x_tf": _rows(bt.model.train_inputs[0]),
        "train_y_tf": _rows(bt.model.train_targets.reshape(case.n, -1)),
        "inducing": _rows(h.strategy(bt).inducing_points.detach()),
        "names": names,
        "sizes": sizes,
        "lower": [str(v) if math.isinf(v) else v for v in lower],
        "upper": [str(v) if math.isinf(v) else v for v in upper],
        "raw_default": raw,
    }
    if case.outcome_tf:
        out["outcome_means"] = _flat(bt.outcome_transform.means)
        out["outcome_stdvs"] = _flat(bt.outcome_transform.stdvs)
    return out


def model_fixture(case: Any) -> dict[str, Any]:
    h = _tests()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt, rs, X, Y = h.build_pair(case, with_rs=False)
    out = _description(case, bt, X, Y)
    evals = []
    for seed in range(2):
        h.set_random_parameters(bt, rs, seed)
        from gpytorch.mlls import VariationalELBO

        raw = _flat(torch.cat([p.detach().reshape(-1) for p in VariationalELBO(bt.likelihood, bt.model, num_data=1).parameters()]))
        terms_list = []
        for num_data, beta in ((case.n, 1.0), (case.n + 9, 0.4)):
            mll = VariationalELBO(bt.likelihood, bt.model, num_data=num_data, beta=beta, combine_terms=False)
            bt.train()
            ll, kl, lp = mll(bt.model(*bt.model.train_inputs), bt.model.train_targets)
            value, grad = h.bt_elbo_and_grad(
                VariationalELBO(bt.likelihood, bt.model, num_data=num_data, beta=beta)
            )
            terms_list.append(
                {
                    "num_data": num_data,
                    "beta": beta,
                    "terms": [float(ll), float(kl), float(lp)],
                    "value": value,
                    "grad": _flat(grad),
                }
            )
        bt.eval()
        g = torch.Generator().manual_seed(50 + seed)
        Xt = case.scale * torch.rand(5, case.d, generator=g, dtype=DT) + (1.0 if case.input_tf else 0.0)
        # q = 1 batches.
        Xq = Xt.unsqueeze(1).clone().requires_grad_(True)
        post = bt.posterior(Xq)
        q1_mean, q1_var = post.mean, post.variance
        (q1_grad,) = torch.autograd.grad(q1_mean.sum() + q1_var.sum(), Xq)
        # Joint posterior.
        Xj = Xt.clone().requires_grad_(True)
        pj = bt.posterior(Xj)
        cov = pj.distribution.covariance_matrix
        mean = pj.mean
        wm = torch.linspace(-1.0, 1.0, mean.numel(), dtype=DT).reshape(mean.shape)
        wc = torch.cos(torch.arange(cov.numel(), dtype=DT)).reshape(cov.shape)
        (j_grad,) = torch.autograd.grad((wm * mean).sum() + (wc * cov).sum(), Xj)
        pn = bt.posterior(Xt, observation_noise=True)
        evals.append(
            {
                "raw": raw,
                "elbo": terms_list,
                "x_test": _rows(Xt),
                "q1_mean": _flat(q1_mean.detach()),
                "q1_variance": _flat(q1_var.detach()),
                "q1_grad": _flat(q1_grad),
                "joint_mean": _flat(mean.detach()),
                "joint_covariance": _rows(cov.detach()),
                "joint_covariance_noise": _rows(pn.distribution.covariance_matrix.detach()),
                "joint_variance_noise": _flat(pn.variance.detach()),
                "joint_grad": _flat(j_grad),
                "weights_mean": _flat(wm),
                "weights_cov": _rows(wc),
            }
        )
    out["evaluations"] = evals
    return out


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _twin(X: Any, t: int) -> Any:
    if t == 0:
        return X
    g = torch.Generator().manual_seed(t)
    rel = TWIN_SCALES[(t - 1) % len(TWIN_SCALES)]
    return X * (1 + rel * torch.randn(X.shape, generator=g, dtype=X.dtype))


def fit_fixture(case: Any) -> dict[str, Any]:
    import botorch.optim.fit as of
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import VariationalELBO

    h = _tests()
    X, Y = h.make_data(case)
    twins = []
    desc: dict[str, Any] | None = None
    for t in range(FIT_TWINS + 1):
        Xt = _twin(X, t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt, _, _, _ = h.build_pair(case, Xt, Y, with_rs=False)
        if desc is None:
            desc = _description(case, bt, Xt, Y)
        # GPyTorch's initialization of the variational parameters (as at the
        # first evaluation), with fixed draws.
        torch.manual_seed(1000 + t)
        vs = h.strategy(bt)
        vs._variational_distribution.initialize_variational_distribution(vs.prior_distribution)
        vs.variational_params_initialized.fill_(1)
        mll = VariationalELBO(bt.likelihood, bt.model, num_data=case.n)
        raw0 = _flat(torch.cat([p.detach().reshape(-1) for p in mll.parameters()]))
        inducing0 = _rows(vs.inducing_points.detach())
        record: dict[str, Any] = {"attempts": 0}
        orig = of.fit_gpytorch_mll_scipy

        def recorder(
            m: Any, *args: Any, _record: dict[str, Any] = record,
            _optimizer: Any = orig, **kwargs: Any,
        ) -> Any:
            _record["attempts"] += 1
            _record["r"] = _optimizer(m, *args, **kwargs)
            return _record["r"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, optimizer=recorder)
        twins.append(
            {
                "x": _rows(Xt),
                "raw0": raw0,
                "inducing0": inducing0,
                "loss": float(record["r"].fval),
                "nit": int(record["r"].step),
                "attempts": int(record["attempts"]),
            }
        )
        print(
            f"phase 10c-approximate: {case.name} twin {t}: loss={twins[-1]['loss']:+.10f} "
            f"attempts={twins[-1]['attempts']}",
            file=sys.stderr,
        )
    assert desc is not None
    desc["twins"] = twins
    return desc


FIT_CASE_NAMES = ("fit_fixed_inducing", "fit_default")


def generate_10c_approximate(out: str) -> None:
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    h = _tests()
    d = os.path.join(out, "10c_approximate")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "allocators.json"), "w") as f:
        json.dump(allocator_fixture(), f)
    print("phase 10c-approximate: wrote allocators.json", file=sys.stderr)
    models = [model_fixture(c) for c in h.CASES]
    with open(os.path.join(d, "models.json"), "w") as f:
        json.dump(models, f)
    print("phase 10c-approximate: wrote models.json", file=sys.stderr)
    fits = [fit_fixture(c) for c in h.FIT_CASES if c.name in FIT_CASE_NAMES]
    with open(os.path.join(d, "fits.json"), "w") as f:
        json.dump(fits, f)
    print("phase 10c-approximate: wrote fits.json", file=sys.stderr)


if __name__ == "__main__":
    generate_10c_approximate(sys.argv[1] if len(sys.argv) > 1 else ".")
