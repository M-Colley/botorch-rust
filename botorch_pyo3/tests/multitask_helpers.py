"""Multi-task GPs built as the same model in BoTorch and in botorch_rs
(``test_parity_7a_multitask.py`` and ``scripts/fixtures_7a_multitask.py``).

A *case* is a JSON-compatible dict: ``n``, ``d``, ``tasks`` (the task values
cycled over the training rows), ``task_feature`` and the keyword arguments of
``MultiTaskGP`` (``output_tasks``, ``all_tasks``, ``rank``, ``fixed_noise``,
``prior`` = ``"default"`` / ``"none"`` / ``"lkj"`` / ``"gamma"``,
``covar`` = a kernel spec of ``kernel_tree_helpers``, ``mean`` =
``"multitask"`` / ``"constant"``, ``likelihood`` = ``None`` / ``"gaussian"``,
``normalize`` = indices of a ``Normalize`` input transform, ``standardize``,
``validate``).

BoTorch's bugs are fixed in the reference model (``fix_reference``) where
botorch_rs implements the correct behaviour:

* non-contiguous task values: the per-task noise is looked up by task index
  (M-Colley/botorch ``fix/models-posteriors``,
  ``_TaskMappedHadamardGaussianLikelihood``, meta-pytorch/botorch#3396);
* an ``LKJCovariancePrior`` task prior is evaluated on the task covariance
  matrix (BoTorch evaluates it on the vector of correlations and fails), in
  double precision, with the correlation term counted once (GPyTorch's
  ``log_prob`` with a univariate ``sd_prior`` counts it ``n`` times);
* a ``ProductKernel`` ``covar_module`` without ``active_dims``: its
  sub-kernels act on the non-task columns (GPyTorch's ``*`` flattens the
  product and drops the ``active_dims`` BoTorch sets on it).

Numerical quirks are fixed for comparisons by ``exact_kernel_products``.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import torch

DT = torch.float64


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------


def mtgp_cases() -> list[dict[str, Any]]:
    """The ``MultiTaskGP`` cases of the fixtures and the live tests."""
    from kernel_tree_helpers import kernel as k
    from kernel_tree_helpers import sub

    return [
        {
            "name": "default_t3_d2",
            "n": 12,
            "d": 2,
            "tasks": [0, 1, 2],
            "task_feature": -1,
            "fit": True,
        },
        {
            "name": "rank1_tf0_outputs",
            "n": 10,
            "d": 2,
            "tasks": [0, 1, 2],
            "task_feature": 0,
            "rank": 1,
            "output_tasks": [2, 0],
        },
        {
            "name": "unobserved_t4",
            "n": 9,
            "d": 2,
            "tasks": [0, 1, 3],
            "task_feature": -1,
            "all_tasks": [0, 1, 2, 3],
            "output_tasks": [2, 3],
        },
        {
            "name": "noncontiguous_tf1",
            "n": 10,
            "d": 3,
            "tasks": [3, 7],
            "task_feature": 1,
            "fit": True,
        },
        {
            "name": "fixed_noise",
            "n": 10,
            "d": 2,
            "tasks": [0, 1],
            "task_feature": -1,
            "fixed_noise": True,
            "fit": True,
        },
        {
            "name": "matern_scale_d3",
            "n": 12,
            "d": 3,
            "tasks": [0, 1],
            "task_feature": -1,
            "covar": k("ScaleKernel", sub(k("MaternKernel", nu=1.5, ard_num_dims=3))),
        },
        {
            "name": "additive_tf0",
            "n": 10,
            "d": 2,
            "tasks": [0, 1, 2],
            "task_feature": 0,
            "covar": k(
                "AdditiveKernel",
                sub(k("RBFKernel", active_dims=[0])),
                sub(k("MaternKernel", nu=2.5, active_dims=[1])),
            ),
        },
        {
            "name": "product_tf0",
            "n": 10,
            "d": 2,
            "tasks": [0, 1],
            "task_feature": 0,
            "covar": k("ProductKernel", sub(k("RBFKernel")), sub(k("PeriodicKernel"))),
        },
        {
            "name": "lkj_prior",
            "n": 12,
            "d": 2,
            "tasks": [0, 1, 2],
            "task_feature": -1,
            "prior": "lkj",
        },
        {
            "name": "gamma_constant_gaussian",
            "n": 10,
            "d": 2,
            "tasks": [0, 1],
            "task_feature": -1,
            "prior": "gamma",
            "mean": "constant",
            "likelihood": "gaussian",
        },
        {
            "name": "normalize_nostd",
            "n": 10,
            "d": 2,
            "tasks": [0, 1],
            "task_feature": -1,
            "prior": "none",
            "normalize": [0, 1],
            "standardize": False,
        },
        {
            "name": "single_output",
            "n": 12,
            "d": 2,
            "tasks": [0, 1],
            "task_feature": -1,
            "output_tasks": [1],
            "acquisition": True,
        },
    ]


def kron_cases() -> list[dict[str, Any]]:
    """The ``KroneckerMultiTaskGP`` cases of the fixtures and the live tests."""
    from kernel_tree_helpers import kernel as k
    from kernel_tree_helpers import sub

    return [
        {"name": "default_m3_d2", "n": 9, "d": 2, "m": 3, "fit": True, "acquisition": True},
        {
            "name": "rank1_lr2_std_norm",
            "n": 8,
            "d": 2,
            "m": 3,
            "rank": 1,
            "likelihood_rank": 2,
            "standardize": True,
            "normalize": True,
        },
        {
            "name": "matern_eta_box_m2",
            "n": 10,
            "d": 3,
            "m": 2,
            "eta": 0.5,
            "sd_box": [0.1, 3.0, 0.1],
            "covar": k("ScaleKernel", sub(k("MaternKernel", nu=2.5, ard_num_dims=3))),
        },
        {
            "name": "active_dims_m2",
            "n": 8,
            "d": 3,
            "m": 2,
            "covar": k("RBFKernel", ard_num_dims=2, active_dims=[0, 2]),
        },
    ]


# ----------------------------------------------------------------------
# Data and construction
# ----------------------------------------------------------------------


def make_data(case: dict[str, Any], seed: int) -> tuple[Any, Any, Any]:
    """Long-format training data ``n x (d + 1)`` with the task column at
    ``task_feature``, targets ``n x 1`` and noise variances ``n x 1``."""
    g = torch.Generator().manual_seed(seed)
    n, d = case["n"], case["d"]
    tasks = case["tasks"]
    X = torch.rand(n, d, generator=g, dtype=DT)
    t = torch.tensor([float(tasks[i % len(tasks)]) for i in range(n)], dtype=DT)
    f = torch.sin(3.0 * X.sum(-1)) + (X[:, 0] - 0.4) ** 2 * (1.0 + 0.2 * t)
    Y = 1.5 + 2.0 * f + 0.3 * t + 0.05 * torch.randn(n, generator=g, dtype=DT)
    cols = list(X.unbind(-1))
    tf = case["task_feature"] % (d + 1)
    cols.insert(tf, t)
    Xf = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(n, 1, generator=g, dtype=DT)
    return Xf, Y.unsqueeze(-1), Yvar


def test_points(case: dict[str, Any], q: int, seed: int, with_tasks: bool = False) -> Any:
    """``q x d`` points (``q x (d + 1)`` with task values cycled over the
    tasks of the model)."""
    g = torch.Generator().manual_seed(seed)
    d = case["d"]
    X = torch.rand(q, d, generator=g, dtype=DT)
    if not with_tasks:
        return X
    tasks = case.get("all_tasks") or sorted(set(case["tasks"]))
    t = torch.tensor([float(tasks[(i + 1) % len(tasks)]) for i in range(q)], dtype=DT)
    cols = list(X.unbind(-1))
    cols.insert(case["task_feature"] % (d + 1), t)
    return torch.stack(cols, -1)


def _prior(lib: str, case: dict[str, Any], num_tasks: int) -> Any:
    kind = case.get("prior", "default")
    if kind == "default":
        return "DEFAULT"
    if kind == "none":
        return None
    if kind == "gamma":
        mod = "gpytorch.priors" if lib == "bt" else "botorch_rs.priors"
        return __import__(mod, fromlist=["GammaPrior"]).GammaPrior(2.0, 2.0)
    if kind == "lkj":
        mod = "gpytorch.priors" if lib == "bt" else "botorch_rs.priors"
        m = __import__(mod, fromlist=["LKJCovariancePrior", "GammaPrior"])
        eta = torch.tensor(1.5) if lib == "bt" else 1.5
        return m.LKJCovariancePrior(num_tasks, eta, m.GammaPrior(2.0, 1.5))
    raise ValueError(kind)


def model_kwargs(lib: str, case: dict[str, Any], Yvar: Any) -> dict[str, Any]:
    """The keyword arguments of ``MultiTaskGP`` in BoTorch (``"bt"``) or
    botorch_rs (``"rs"``)."""
    import importlib

    from kernel_tree_helpers import build_kernel

    kw: dict[str, Any] = {"task_feature": case["task_feature"]}
    for key in ("output_tasks", "all_tasks", "rank"):
        if case.get(key) is not None:
            kw[key] = case[key]
    if case.get("fixed_noise"):
        kw["train_Yvar"] = Yvar
    num_tasks = len(case.get("all_tasks") or set(case["tasks"]))
    prior = _prior(lib, case, num_tasks)
    if prior != "DEFAULT" and case.get("prior") != "lkj":
        kw["task_covar_prior"] = prior
    if case.get("prior") == "lkj" and lib == "rs":
        kw["task_covar_prior"] = prior
    if case.get("prior") == "lkj" and lib == "bt":
        # BoTorch fails with an LKJ prior; the fixed prior is registered by
        # `fix_reference`.
        kw["task_covar_prior"] = None
    if case.get("covar") is not None:
        kw["covar_module"] = build_kernel(case["covar"], lib)
    if case.get("mean") == "constant":
        mod = "gpytorch.means" if lib == "bt" else "botorch_rs.means"
        kw["mean_module"] = importlib.import_module(mod).ConstantMean()
    if case.get("likelihood") == "gaussian":
        if lib == "bt":
            from botorch.models.utils.gpytorch_modules import (
                get_gaussian_likelihood_with_lognormal_prior,
            )
        else:
            from botorch_rs.models.utils.gpytorch_modules import (
                get_gaussian_likelihood_with_lognormal_prior,
            )
        kw["likelihood"] = get_gaussian_likelihood_with_lognormal_prior()
    if case.get("normalize") is not None:
        mod = (
            "botorch.models.transforms.input"
            if lib == "bt"
            else "botorch_rs.models.transforms.input"
        )
        kw["input_transform"] = importlib.import_module(mod).Normalize(
            d=case["d"] + 1, indices=case["normalize"]
        )
    if not case.get("standardize", True):
        kw["outcome_transform"] = None
    if case.get("validate") is False:
        kw["validate_task_values"] = False
    return kw


def build_pair(case: dict[str, Any], X: Any, Y: Any, Yvar: Any, seed: int) -> tuple[Any, Any]:
    """The same ``MultiTaskGP`` in BoTorch (with ``fix_reference`` applied)
    and botorch_rs, with the same random initial task factors."""
    from botorch.models.multitask import MultiTaskGP as BtMTGP

    from botorch_rs.models import MultiTaskGP as RsMTGP

    torch.manual_seed(seed)
    bt = BtMTGP(X, Y, **model_kwargs("bt", case, Yvar))
    torch.manual_seed(seed)
    rs = RsMTGP(X, Y, **model_kwargs("rs", case, Yvar))
    fix_reference(bt, case)
    return bt, rs


# ----------------------------------------------------------------------
# BoTorch's bugs, fixed in the reference
# ----------------------------------------------------------------------


def _mapped_hadamard_class() -> Any:
    """M-Colley/botorch ``fix/models-posteriors``: a
    ``HadamardGaussianLikelihood`` that maps the raw task column through the
    model's task mapper (``_rs_task_mapper``) before looking up the per-task
    noise."""
    from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood

    class _TaskMappedHadamardGaussianLikelihood(HadamardGaussianLikelihood):
        def _shaped_noise_covar(self, base_shape: Any, *params: Any, **kwargs: Any) -> Any:
            if len(params) > 0 and len(params[0]) > 0:
                tv = params[0] if torch.is_tensor(params[0]) else params[0][0]
                if tv.shape[-1] > 1:
                    tv = tv[..., [self.task_feature_index]]
                params = (self._rs_task_mapper[tv.long()], *params[1:])
            return super()._shaped_noise_covar(base_shape, *params, **kwargs)

    return _TaskMappedHadamardGaussianLikelihood


def _fixed_eval_class(base: Any) -> Any:
    """``MultiTaskGP`` whose ``eval()`` finds the task kernel as the last
    sub-kernel of ``covar_module`` (BoTorch uses ``kernels[1]``)."""

    class _FixedEvalMultiTaskGP(base):  # type: ignore[misc, valid-type]
        def eval(self) -> Any:
            if len(self._unobserved_task_indices) > 0:
                tk = self.covar_module.kernels[-1]
                cf = tk.covar_factor
                mean = cf[self._observed_task_indices].mean(dim=0)
                new = cf.clone()
                new[self._unobserved_task_indices] = mean
                tk._set_covar_factor(new)
            return super(base, self).eval()

    return _FixedEvalMultiTaskGP


def lkj_in_double(bt: Any) -> None:
    """LKJ normalizing constants in double precision (torch keeps the
    concentration of ``LKJCholesky`` in float32)."""
    for _, _, prior_, _, _ in bt.named_priors():
        corr = getattr(prior_, "correlation_prior", None)
        if corr is not None:
            corr.concentration = corr.concentration.to(DT)


def fix_reference(bt: Any, case: dict[str, Any]) -> None:
    """Applies the fixes of BoTorch's bugs (see the module documentation);
    they survive ``copy.deepcopy``."""
    from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood

    lik = bt.likelihood
    if bt._task_mapper is not None and type(lik) is HadamardGaussianLikelihood:
        lik.__class__ = _mapped_hadamard_class()
        lik._rs_task_mapper = bt._task_mapper
    task_kernel = bt.covar_module.kernels[-1]
    if case.get("prior") == "lkj":
        from gpytorch.priors import GammaPrior, LKJCovariancePrior

        prior = LKJCovariancePrior(
            task_kernel.num_tasks, torch.tensor(1.5, dtype=DT), GammaPrior(2.0, 1.5)
        )
        task_kernel.register_prior(
            "IndexKernelPrior", prior, lambda m: m.covar_matrix, lambda m, v: None
        )
        bt.to(DT)
        lkj_in_double(bt)
    if case.get("covar") is not None and case["covar"]["type"] == "ProductKernel":
        base = bt._base_idxr.tolist()
        for kern in bt.covar_module.kernels[:-1]:
            if kern.active_dims is None:
                kern.active_dims = torch.tensor(base, dtype=torch.long)
            else:
                kern.active_dims = torch.tensor([base[i] for i in kern.active_dims.tolist()])
    if bt.covar_module.kernels[1] is not task_kernel:
        bt.__class__ = _fixed_eval_class(type(bt))


@contextlib.contextmanager
def exact_kernel_products() -> Iterator[None]:
    """Exact (dense) elementwise products of kernel matrices in GPyTorch.

    GPyTorch multiplies two non-dense kernel matrices (e.g. the low-rank
    output of a ``LinearKernel``, or the sum of an ``AdditiveKernel``, by the
    task kernel's ``InterpolatedLinearOperator``) through a
    ``MulLinearOperator``, which takes Cholesky roots of both factors. The
    ``n x n`` task covariance of ``n`` data from fewer tasks is singular, so
    the Cholesky factorization often fails and is retried with jitter
    (``1e-8`` in double precision) on the diagonal; botorch_rs multiplies
    the matrices exactly."""
    from linear_operator.operators import DenseLinearOperator
    from linear_operator.operators._linear_operator import LinearOperator

    orig = LinearOperator._mul_matrix

    def _mul_matrix(self: Any, other: Any) -> Any:
        self, other = self.evaluate_kernel(), other.evaluate_kernel()
        return DenseLinearOperator(self.to_dense() * other.to_dense())

    LinearOperator._mul_matrix = _mul_matrix
    try:
        yield
    finally:
        LinearOperator._mul_matrix = orig


def botorch_prior_offset(bt: Any) -> Any:
    """The part of BoTorch's MLL from GPyTorch's ``LKJCovariancePrior``
    counting its correlation term ``n`` times (``n - 1`` too many) with a
    univariate ``sd_prior``, divided by the number of data."""
    n_data = bt.train_targets.numel()
    total = torch.zeros((), dtype=DT)
    for _, module, prior_, closure, _ in bt.named_priors():
        corr_prior = getattr(prior_, "correlation_prior", None)
        if corr_prior is not None:
            value = closure(module)
            if prior_.log_prob(value).dim() <= value.dim() - 2:
                # One value per matrix (e.g. a SmoothedBoxPrior sd_prior of
                # event shape 1): GPyTorch's density is correct.
                continue
            sd = value.diagonal(dim1=-2, dim2=-1).sqrt()
            corr = value / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
            total = total + (corr_prior.n - 1) * corr_prior.log_prob(corr).sum() / n_data
    return total


# ----------------------------------------------------------------------
# Parameters and the marginal log likelihood
# ----------------------------------------------------------------------


def flat(model: Any) -> list[float]:
    return [v for _, p in model.named_parameters() for v in p.detach().reshape(-1).tolist()]


def set_flat(model: Any, values: list[float]) -> None:
    offset = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            n = p.numel()
            p.copy_(torch.tensor(values[offset : offset + n], dtype=DT).reshape(p.shape))
            offset += n


def bounds_of(model: Any) -> list[list[float]]:
    from botorch.optim.utils import get_parameters_and_bounds
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _, bounds = get_parameters_and_bounds(ExactMarginalLogLikelihood(model.likelihood, model))
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
            elif "covar_factor" in name:
                v = 0.3 + 0.6 * torch.rand(1, generator=g, dtype=DT).item()
            else:
                v = 0.5 * torch.randn(1, generator=g, dtype=DT).item()
                if math.isfinite(lo):
                    v = lo + 0.1 + abs(v)
                if math.isfinite(hi):
                    v = min(v, hi - 0.05)
            values.append(v)
            i += 1
    return values


def bt_mll_value_and_grad(model: Any) -> tuple[float, list[float]]:
    """BoTorch's MLL (``mll(model(*train_inputs), train_targets,
    *train_inputs)`` as in ``fit_gpytorch_mll``'s closure) with the
    LKJ fix, and its gradient."""
    from gpytorch.mlls import ExactMarginalLogLikelihood

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.zero_grad()
    out = mll(model(*model.train_inputs), model.train_targets, *model.train_inputs)
    out = out - botorch_prior_offset(model)
    out.sum().backward()
    grad = []
    for _, p in model.named_parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grad.extend(g.reshape(-1).tolist())
    model.zero_grad()
    model.eval()
    return float(out.detach()), grad


# ----------------------------------------------------------------------
# Posteriors
# ----------------------------------------------------------------------


def marginal_record(model: Any, X: Any, noise: bool = False) -> dict[str, Any]:
    """Point posteriors of the rows of ``X`` (``b x 1 x d``): means
    (``b x m``), output covariances (``b x m x m``) and the gradients of the
    sums of the means and variances with respect to ``X``."""
    Xg = X.unsqueeze(-2).clone().requires_grad_(True)
    post = model.posterior(Xg, observation_noise=noise)
    mean = post.mean
    var = post.variance
    m = mean.shape[-1]
    cov = post.distribution.covariance_matrix.reshape(X.shape[0], m, m)
    (dm,) = torch.autograd.grad(mean.sum(), Xg, retain_graph=True, allow_unused=True)
    (dv,) = torch.autograd.grad(var.sum(), Xg, allow_unused=True)
    dm = torch.zeros_like(Xg) if dm is None else dm
    dv = torch.zeros_like(Xg) if dv is None else dv
    return {
        "x": X,
        "mean": mean.detach().reshape(X.shape[0], m),
        "covariance": cov.detach(),
        "dmean_dx": dm.squeeze(-2),
        "dvariance_dx": dv.squeeze(-2),
    }


def joint_record(model: Any, X: Any, seed: int, **kwargs: Any) -> dict[str, Any]:
    """The joint posterior at ``X`` (``q x d`` or ``q x (d + 1)``): mean
    (``q x m``), covariance (output-major ``qm x qm``), the covariance with
    observation noise, and the gradient of a weighted sum of the mean and
    covariance with respect to ``X``."""
    g = torch.Generator().manual_seed(seed)
    Xg = X.clone().requires_grad_(True)
    p = model.posterior(Xg, **kwargs)
    mean = p.mean
    cov = p.distribution.covariance_matrix
    wm = torch.randn(mean.shape, generator=g, dtype=DT)
    wc = torch.randn(cov.shape, generator=g, dtype=DT)
    s = (mean * wm).sum() + (cov * wc).sum()
    (dx,) = torch.autograd.grad(s, Xg)
    with torch.no_grad():
        noisy = model.posterior(X, observation_noise=True, **kwargs)
    return {
        "x": X,
        "mean": mean.detach(),
        "covariance": cov.detach(),
        "covariance_noisy": noisy.distribution.covariance_matrix,
        "weights_mean": wm,
        "weights_cov": wc,
        "dx": dx,
    }


# ----------------------------------------------------------------------
# KroneckerMultiTaskGP
# ----------------------------------------------------------------------


def kron_kwargs(lib: str, case: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments of ``KroneckerMultiTaskGP`` for a case
    (``rank``, ``likelihood_rank``, ``standardize``, ``normalize``,
    ``covar`` (a kernel spec), ``eta``, ``sd_box`` (``[a, b, sigma]``))."""
    import importlib

    from kernel_tree_helpers import build_kernel

    kw: dict[str, Any] = {}
    for key in ("rank", "likelihood_rank", "eta"):
        if case.get(key) is not None:
            kw[key] = case[key]
    if case.get("sd_box") is not None:
        mod = "gpytorch.priors" if lib == "bt" else "botorch_rs.priors"
        kw["sd_prior"] = importlib.import_module(mod).SmoothedBoxPrior(*case["sd_box"])
    if case.get("covar") is not None:
        kw["data_covar_module"] = build_kernel(case["covar"], lib)
    m = case["m"]
    if case.get("standardize"):
        mod = (
            "botorch.models.transforms.outcome"
            if lib == "bt"
            else "botorch_rs.models.transforms.outcome"
        )
        kw["outcome_transform"] = importlib.import_module(mod).Standardize(m=m)
    if case.get("normalize"):
        mod = (
            "botorch.models.transforms.input"
            if lib == "bt"
            else "botorch_rs.models.transforms.input"
        )
        kw["input_transform"] = importlib.import_module(mod).Normalize(d=case["d"])
    return kw


def kron_data(case: dict[str, Any], seed: int) -> tuple[Any, Any]:
    g = torch.Generator().manual_seed(seed)
    n, d, m = case["n"], case["d"], case["m"]
    X = torch.rand(n, d, generator=g, dtype=DT)
    cols = []
    for t in range(m):
        f = torch.sin((2.0 + t) * X.sum(-1)) + (0.5 + 0.3 * t) * (X[:, 0] - 0.3) ** 2
        cols.append((1.0 + t) * f + 0.05 * torch.randn(n, generator=g, dtype=DT))
    return X, torch.stack(cols, -1)


def build_kron_pair(case: dict[str, Any], X: Any, Y: Any, seed: int) -> tuple[Any, Any]:
    """The same ``KroneckerMultiTaskGP`` in BoTorch and botorch_rs, with the
    same random initial values. The reference applies the ``active_dims``
    of the data kernel (on the ``MultitaskKernel``, which GPyTorch applies
    to the data kernel's inputs)."""
    from botorch.models.multitask import KroneckerMultiTaskGP as Bt

    from botorch_rs.models import KroneckerMultiTaskGP as Rs

    torch.manual_seed(seed)
    bt = Bt(X, Y, **kron_kwargs("bt", case))
    torch.manual_seed(seed)
    rs = Rs(X, Y, **kron_kwargs("rs", case))
    dc = bt.covar_module.data_covar_module
    if dc.active_dims is not None:
        bt.covar_module.active_dims = dc.active_dims
    return bt, rs


def kron_noise(bt: Any) -> Any:
    lik = bt.likelihood
    m = lik.num_tasks
    if lik.rank == 0:
        n = torch.diag(lik.task_noises.reshape(-1))
    else:
        f = lik.task_noise_covar_factor
        n = f @ f.transpose(-1, -2)
    return n + lik.noise.reshape(()) * torch.eye(m, dtype=DT)


def kron_exact_posterior(bt: Any, X: Any, noise: bool = False) -> tuple[Any, Any]:
    """The exact joint posterior of a BoTorch ``KroneckerMultiTaskGP`` at
    ``X`` (``q x d``), computed densely (differentiable): the mean (``q x
    m``) and the interleaved covariance (``qm x qm``) in the original outcome
    space."""
    bt.eval()
    m = bt.num_outputs
    Xt = bt.transform_inputs(X)
    Xtr = bt.train_inputs[0]
    data = bt.covar_module.data_covar_module
    # (`data.__call__` applies its own active_dims once.)
    kxx_train = data(Xtr).to_dense()
    kx_star = data(Xt, Xtr).to_dense()
    kx_ss = data(Xt).to_dense()
    B = bt.covar_module.task_covar_module.covar_matrix.to_dense()
    N = kron_noise(bt)
    n = Xtr.shape[-2]
    K = torch.kron(kxx_train, B) + torch.kron(torch.eye(n, dtype=DT), N)
    c = torch.stack([bm.constant.reshape(()) for bm in bt.mean_module.base_means])
    resid = (bt.train_targets - c).reshape(-1)
    Ks = torch.kron(kx_star, B)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(resid.unsqueeze(-1), L).squeeze(-1)
    q = Xt.shape[-2]
    mean = (Ks @ alpha).reshape(q, m) + c
    V = torch.linalg.solve_triangular(L, Ks.transpose(-1, -2), upper=False)
    cov = torch.kron(kx_ss, B) - V.transpose(-1, -2) @ V
    if noise:
        cov = cov + torch.kron(torch.eye(q, dtype=DT), N)
    ot = getattr(bt, "outcome_transform", None)
    if ot is not None:
        mu = ot.means.reshape(-1)
        sd = ot.stdvs.reshape(-1)
        mean = mu + sd * mean
        s = sd.repeat(q)
        cov = s.unsqueeze(-1) * cov * s.unsqueeze(-2)
    return mean, cov


def interleaved_to_output_major(cov: Any, q: int, m: int) -> Any:
    """The covariance of interleaved outputs (``p m + t``) in output-major
    order (``t q + p``)."""
    perm = [p * m + t for t in range(m) for p in range(q)]
    return cov[perm][:, perm]


def matheron_covariance(post: Any, sample: Any) -> tuple[Any, Any]:
    """The mean and covariance (interleaved ``qm x qm``) of the samples of a
    Matheron posterior, from its linear map: ``sample(z)`` evaluated at
    ``z = 0`` and at the unit vectors of the base samples."""
    tot = post.base_sample_shape[-1]
    Z = torch.cat([torch.zeros(1, tot, dtype=DT), torch.eye(tot, dtype=DT)])
    S = torch.as_tensor(sample(Z), dtype=DT).reshape(tot + 1, -1)
    A = S[1:] - S[0]
    return S[0], A.T @ A


class ExactKroneckerModel:
    """A BoTorch model whose posterior is the exact joint posterior of a
    ``KroneckerMultiTaskGP`` (a ``MultitaskMultivariateNormal`` over the
    output-major flattening): the reference for acquisition functions, which
    botorch_rs evaluates on the exact joint posterior."""

    def __new__(cls, bt: Any) -> Any:
        from botorch.models.model import Model
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        from gpytorch.distributions import MultitaskMultivariateNormal

        class _Exact(Model):
            def __init__(self) -> None:
                super().__init__()
                self.bt = bt

            @property
            def num_outputs(self) -> int:
                return bt.num_outputs

            @property
            def batch_shape(self) -> Any:
                return torch.Size()

            def posterior(
                self,
                X: Any,
                output_indices: Any = None,
                observation_noise: Any = False,
                posterior_transform: Any = None,
                **kw: Any,
            ) -> Any:
                batch = X.shape[:-2]
                Xf = X.reshape(-1, *X.shape[-2:])
                means, covs = [], []
                q, m = X.shape[-2], bt.num_outputs
                for b in range(Xf.shape[0]):
                    mean, cov = kron_exact_posterior(bt, Xf[b], bool(observation_noise))
                    means.append(mean)
                    covs.append(interleaved_to_output_major(cov, q, m))
                mean = torch.stack(means).reshape(*batch, q, m)
                cov = torch.stack(covs).reshape(*batch, q * m, q * m)
                mvn = MultitaskMultivariateNormal(mean, cov, interleaved=False)
                post = GPyTorchPosterior(mvn)
                if posterior_transform is not None:
                    return posterior_transform(post)
                return post

        return _Exact()
