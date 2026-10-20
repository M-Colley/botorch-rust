"""Shared cases of the phase-10c tests (approximate GPs): BoTorch's and
botorch_rs's ``SingleTaskVariationalGP`` built from the same description,
with identical parameters, and their variational ELBOs.

Import only after ``importorskip_botorch()``: this module requires torch,
GPyTorch and BoTorch. The cases are shared with
``scripts/fixtures_10c_approximate.py``.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from kernel_tree_helpers import build_kernel, with_batch

DT = torch.float64
#: Nominal tolerances (relative): values and gradients.
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8


@dataclass(frozen=True)
class Case:
    """A ``SingleTaskVariationalGP``: ``n`` points in ``d`` dimensions, ``m``
    outputs, ``inducing`` points (a count, ``None`` for BoTorch's 25 %
    default, or ``"train"`` for the training inputs), a kernel spec of
    ``kernel_tree_helpers`` (``None`` for BoTorch's default), the mean
    (``"default"``, ``"prior"`` with a ``NormalPrior``, ``"zero"``), the
    likelihood (``"default"``, ``"gaussian"`` = ``GaussianLikelihood()``,
    ``"gamma"`` with a Gamma noise prior), ``learn`` (inducing points
    learned), transforms (``Normalize`` / ``Standardize``), the data scale and
    seed."""

    name: str
    n: int = 12
    d: int = 2
    m: int = 1
    inducing: Any = 4
    kernel: dict[str, Any] | None = None
    mean: str = "default"
    likelihood: str = "default"
    learn: bool = True
    input_tf: bool = False
    outcome_tf: bool = False
    scale: float = 1.0
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


CASES = [
    Case("default"),
    Case("default_d3_n20", n=20, d=3, inducing=None, seed=1),
    Case("fixed_inducing", learn=False, seed=2),
    Case(
        "matern_scale",
        kernel={
            "type": "ScaleKernel",
            "args": [
                {
                    "__kernel__": {
                        "type": "MaternKernel",
                        "args": [],
                        "kwargs": {"nu": 2.5, "ard_num_dims": 2},
                    }
                }
            ],
            "kwargs": {},
        },
        mean="prior",
        likelihood="gamma",
        seed=3,
    ),
    Case("zero_mean_gaussian", mean="zero", likelihood="gaussian", seed=4),
    Case("two_outputs", m=2, inducing=5, seed=5),
    Case("three_outputs_fixed", m=3, n=15, inducing=4, learn=False, seed=6),
    Case("transforms", input_tf=True, outcome_tf=True, scale=3.0, n=14, seed=7),
    Case("transforms_two_outputs", m=2, input_tf=True, outcome_tf=True, scale=2.0, seed=8),
    Case("inducing_train", n=8, inducing="train", seed=9),
]

#: Fits compared on twins (live) and recorded on 64 twins (fixtures: the
#: first two).
FIT_CASES = [
    Case("fit_fixed_inducing", n=16, inducing=4, learn=False, seed=21),
    Case("fit_default", n=12, inducing=3, seed=22),
    Case("fit_two_outputs", n=12, m=2, inducing=3, learn=False, seed=23),
]


def make_data(case: Case, seed: int | None = None) -> tuple[Any, Any]:
    g = torch.Generator().manual_seed(case.seed if seed is None else seed)
    X = case.scale * torch.rand(case.n, case.d, generator=g, dtype=DT)
    if case.input_tf:
        X = X + 1.0
    cols = []
    for t in range(case.m):
        y = torch.sin((3.0 + t) * X[:, :1] / case.scale) + 0.5 * X[:, -1:] / case.scale
        cols.append(y + 0.1 * torch.randn(case.n, 1, generator=g, dtype=DT))
    Y = torch.cat(cols, -1)
    if case.outcome_tf:
        Y = 4.0 * Y + 2.0
    return X, Y


def _modules(case: Case, lib: str, X: Any) -> dict[str, Any]:
    """The constructor keyword arguments of ``case`` in BoTorch (``"bt"``) or
    botorch_rs (``"rs"``)."""
    if lib == "bt":
        import gpytorch
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize

        likelihoods, priors, means = gpytorch.likelihoods, gpytorch.priors, gpytorch.means
    else:
        import botorch_rs.likelihoods as likelihoods
        import botorch_rs.means as means
        import botorch_rs.priors as priors
        from botorch_rs.models.transforms.input import Normalize  # type: ignore[assignment]
        from botorch_rs.models.transforms.outcome import Standardize  # type: ignore[assignment]
    kw: dict[str, Any] = {}
    aug = [case.m] if case.m > 1 else []
    if case.kernel is not None:
        spec = with_batch(case.kernel, aug) if aug else case.kernel
        # BoTorch evaluates the kernel (to allocate the inducing points)
        # before casting the model to the data's dtype.
        k = build_kernel(spec, lib)
        kw["covar_module"] = k.to(DT) if lib == "bt" else k
    if case.mean == "prior":
        kw["mean_module"] = means.ConstantMean(
            constant_prior=priors.NormalPrior(0.1, 2.0),
            batch_shape=torch.Size(aug) if lib == "bt" else tuple(aug),
        )
    elif case.mean == "zero":
        kw["mean_module"] = means.ZeroMean()
    if case.likelihood == "gaussian":
        kw["likelihood"] = likelihoods.GaussianLikelihood()
    elif case.likelihood == "gamma":
        if case.m > 1:
            kw["likelihood"] = likelihoods.MultitaskGaussianLikelihood(
                num_tasks=case.m, noise_prior=priors.GammaPrior(1.1, 0.05)
            )
        else:
            kw["likelihood"] = likelihoods.GaussianLikelihood(
                noise_prior=priors.GammaPrior(1.1, 0.05)
            )
    if case.input_tf:
        kw["input_transform"] = Normalize(d=case.d)
    if case.outcome_tf:
        kw["outcome_transform"] = Standardize(m=case.m)
    if case.inducing == "train":
        kw["inducing_points"] = X.clone() if case.input_tf is False else None
    else:
        kw["inducing_points"] = case.inducing
    kw["learn_inducing_points"] = case.learn
    kw.update(case.extra)
    return kw


def build_pair(
    case: Case, X: Any = None, Y: Any = None, with_rs: bool = True
) -> tuple[Any, Any, Any, Any]:
    """BoTorch's and botorch_rs's models of ``case`` (data from
    :func:`make_data` unless given) and the data (botorch_rs's model is
    ``None`` without ``with_rs``, e.g. in the fixture generator)."""
    from botorch.models import SingleTaskVariationalGP as BtSVGP

    if X is None:
        X, Y = make_data(case)
    rs = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt = BtSVGP(X, Y, **_modules(case, "bt", X))
        if with_rs:
            from botorch_rs.models import SingleTaskVariationalGP as RsSVGP

            rs = RsSVGP(X, Y, **_modules(case, "rs", X))
    return bt, rs, X, Y


def strategy(model: Any) -> Any:
    """The (base) variational strategy of a model of either library."""
    vs = model.model.variational_strategy
    return getattr(vs, "base_variational_strategy", vs)


def elbo_pair(bt: Any, rs: Any, num_data: int, beta: float = 1.0, combine: bool = True) -> Any:
    from gpytorch.mlls import VariationalELBO as BtELBO

    from botorch_rs.mlls import VariationalELBO as RsELBO

    return (
        BtELBO(bt.likelihood, bt.model, num_data=num_data, beta=beta, combine_terms=combine),
        RsELBO(rs.likelihood, rs.model, num_data=num_data, beta=beta, combine_terms=combine),
    )


def set_random_parameters(bt: Any, rs: Any, seed: int, scale: float = 0.3) -> None:
    """Random parameters (BoTorch's initial values perturbed: positive
    parameters multiplicatively) in BoTorch, copied into botorch_rs; the
    variational parameters count as initialized in both."""
    from gpytorch.mlls import VariationalELBO as BtELBO

    strategy(bt).variational_params_initialized.fill_(1)
    if rs is not None:
        strategy(rs).variational_params_initialized.fill_(1)
    g = torch.Generator().manual_seed(seed)
    mll = BtELBO(bt.likelihood, bt.model, num_data=1)
    with torch.no_grad():
        for name, p in mll.named_parameters():
            noise = scale * torch.randn(p.shape, generator=g, dtype=DT)
            positive = any(
                key in name for key in ("lengthscale", "raw_noise", "outputscale", "task_noises")
            )
            if positive and not name.endswith(("task_noises", "likelihood.raw_noise")):
                p.mul_(torch.exp(noise))
            else:
                p.add_(noise)
    if rs is not None:
        copy_parameters(bt, rs)


def copy_parameters(bt: Any, rs: Any) -> None:
    """BoTorch's parameters (and inducing points) copied into botorch_rs."""
    from gpytorch.mlls import VariationalELBO as BtELBO

    mll = BtELBO(bt.likelihood, bt.model, num_data=1)
    rs.initialize(**{name: p.detach().clone() for name, p in mll.named_parameters()})
    strategy(rs).inducing_points = strategy(bt).inducing_points.detach().clone()


def bt_elbo_and_grad(mll: Any) -> tuple[float, Any]:
    """BoTorch's ELBO on the training data and its gradient (flat, in the
    order of ``mll.named_parameters()``)."""
    model = mll.model
    model.train()
    for p in mll.parameters():
        p.grad = None
    value = mll(model(*model.train_inputs), model.train_targets)
    value.backward()
    grad = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in mll.parameters()
        ]
    )
    return float(value), grad.detach()


@contextlib.contextmanager
def fixed_allocator() -> Iterator[None]:
    """BoTorch's ``_pivoted_cholesky_init`` on the detached kernel matrix.

    BoTorch evaluates the kernel matrix of the allocation under
    ``torch.no_grad()``, but lazily: a ``ScaleKernel`` whose
    ``outputscale_constraint`` has no transform multiplies the (lazy) matrix
    of e.g. a ``ProductKernel`` by its raw parameter itself, so the
    evaluated matrix requires gradients and ``_pivoted_cholesky_init`` raises
    ``UnsupportedError`` (a BoTorch bug, see the phase-10c draft)."""
    import botorch.models.utils.inducing_point_allocators as ipa

    orig = ipa._pivoted_cholesky_init

    def patched(train_inputs: Any, kernel_matrix: Any, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            dense = (
                kernel_matrix.to_dense() if hasattr(kernel_matrix, "to_dense") else kernel_matrix
            )
        return orig(train_inputs, dense.detach(), *args, **kwargs)

    ipa._pivoted_cholesky_init = patched
    try:
        yield
    finally:
        ipa._pivoted_cholesky_init = orig
