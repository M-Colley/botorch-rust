"""Parity of kernel trees, GPyTorch's and BoTorch's kernels, priors and
constraints (phase 6a) with GPyTorch and BoTorch, through the public APIs of
both libraries side by side.

Every model is built from the same JSON-style kernel spec in both libraries
(``kernel_tree_helpers.py``), with the same random initial values (index
kernels draw theirs from torch's generator in both), and compared:

* kernel matrices ``k(x1, x2)``, ``k(x1)`` and ``k(x1, diag=True)`` of
  stand-alone kernels (every kernel class, batched kernels, trees);
* ``SingleTaskGP`` models: parameter names, shapes, default values and
  optimizer bounds; the MLL and its gradient with respect to all raw
  parameters at the default and at random parameters; posteriors (``q = 1``
  and joint, with observation noise); the input gradients of
  ``PosteriorMean``, ``PosteriorStandardDeviation`` and ``qSimpleRegret``
  (with identical base samples); fits from identical initial values;
* multi-output and batched models, ``ModelListGP``, conditioning on
  observations and fantasies, Matheron paths and prior paths (identical
  random draws) with the kernels BoTorch's pathwise sampling supports;
* the priors (univariate, LKJ, BoTorch's ``BetaPrior``) and ``LessThan``.

Tolerances: values 1e-9, gradients 1e-8 (relative); fits as in the Rust
fixture test (``kernels_6a_parity.rs``): botorch_rs's best outcome on the
model and 4 twins (training inputs perturbed at 1e-14 .. 1e-11) must reach
BoTorch's best and its own fit must not end above BoTorch's worst, within
``max(1e-6, 10 x`` the spread of BoTorch's outcomes at its best optimum``)``;
on chaotic likelihoods (BoTorch's outcomes spread over more than 1e-3),
statistically on 64 twins (``parity_helpers.compare_chaotic_fits``: a
one-sided Mann-Whitney U test at level 1e-3 and botorch_rs's best within
BoTorch's best ``k`` outcomes, each failing by chance with probability at
most 1e-3).

BoTorch's numerical quirks are fixed for the comparisons (see the deviations
in ``docs/drafts/6a-kernels.md``): Matérn-1/2 models are compared with
GPyTorch's ``sq_dist`` zeroing the diagonal also with gradients, and
``HalfCauchyPrior`` is evaluated in double precision; the differences these
fixes make are recorded as BoTorch's noise. The randomized sweep over random
kernel trees is marked ``equivalence``.
"""

from __future__ import annotations

import contextlib
import functools
import math
import random
import statistics
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import ParityLog, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import gpytorch  # noqa: E402
from botorch.acquisition import PosteriorMean as BtPosteriorMean  # noqa: E402
from botorch.acquisition import PosteriorStandardDeviation as BtPSD  # noqa: E402
from botorch.acquisition import qSimpleRegret as BtQSR  # noqa: E402
from botorch.fit import fit_gpytorch_mll as bt_fit  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.optim.utils import get_parameters_and_bounds  # noqa: E402
from botorch.sampling import pathwise as bt_pw  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from kernel_tree_helpers import (  # noqa: E402
    build_kernel,
    constraint,
    double_precision_half_cauchy,
    kernel,
    prior,
    random_spec,
    sub,
    with_batch,
    zero_diagonal_distances,
)
from mo_entropy_helpers import bt_path_draws, recorded_randn  # noqa: E402

from botorch_rs.acquisition import PosteriorMean as RsPosteriorMean  # noqa: E402
from botorch_rs.acquisition import PosteriorStandardDeviation as RsPSD  # noqa: E402
from botorch_rs.acquisition import qSimpleRegret as RsQSR  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.sampling import pathwise as rs_pw  # noqa: E402
from parity_helpers import (  # noqa: E402
    CHAOTIC_TWINS,
    Checker,
    compare_chaotic_fits,
    perturbed_copy,
    rel_err,
)

pytestmark = pytest.mark.filterwarnings("ignore")

DT = torch.float64
CHECK = Checker("kernel trees, kernels, priors and constraints (6a)")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
FIT_TOL = 1e-6
FIT_CLUSTER = 1e-4
CHAOTIC_RANGE = 1e-3
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
N_TWINS = 4
F64 = {"__dtype__": "float64"}


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _grad_err(a: Any, e: Any) -> float:
    """Gradient error relative to the largest component."""
    a, e = torch.as_tensor(a, dtype=DT).reshape(-1), torch.as_tensor(e, dtype=DT).reshape(-1)
    assert a.shape == e.shape
    scale = max(e.abs().max().item(), 1e-300)
    return (a - e).abs().max().item() / scale


def check_grad(quantity: str, a: Any, e: Any, tol: float = GRAD_TOL) -> None:
    err = _grad_err(a, e)
    CHECK.record(quantity, err, tol)
    assert err <= tol, (quantity, err)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

k, s_, p_, c_ = kernel, sub, prior, constraint
RBF2 = k("RBFKernel", ard_num_dims=2, active_dims=[0, 1])

#: Stand-alone kernels: (spec, input dimension, inputs are task indices in
#: the last column).
KERNELS: dict[str, tuple[dict[str, Any], int, bool]] = {
    "rbf_ard": (k("RBFKernel", ard_num_dims=3), 3, False),
    "rbf_shared": (k("RBFKernel"), 3, False),
    "matern12": (k("MaternKernel", nu=0.5, ard_num_dims=2), 2, False),
    "matern32_active": (k("MaternKernel", nu=1.5, active_dims=[2, 0]), 3, False),
    "matern52_scale": (k("ScaleKernel", s_(k("MaternKernel", ard_num_dims=2))), 2, False),
    "rq": (k("RQKernel", ard_num_dims=2), 2, False),
    "periodic": (k("PeriodicKernel", ard_num_dims=2), 2, False),
    "periodic_shared": (k("PeriodicKernel"), 2, False),
    "linear": (k("LinearKernel", ard_num_dims=2), 2, False),
    "polynomial3": (k("PolynomialKernel", power=3), 2, False),
    "cosine": (k("CosineKernel"), 2, False),
    "index": (k("IndexKernel", num_tasks=3, rank=2), 1, True),
    "positive_index": (k("PositiveIndexKernel", num_tasks=4, rank=2, target_task_index=1), 1, True),
    "categorical": (k("CategoricalKernel", ard_num_dims=3), 3, False),
    "downsampling": (k("DownsamplingKernel"), 1, False),
    "expdecay": (k("ExponentialDecayKernel", ard_num_dims=2), 2, False),
    "bnn": (k("InfiniteWidthBNNKernel", depth=3), 3, False),
    "ltf2": (k("LinearTruncatedFidelityKernel", fidelity_dims=[0, 2], dimension=4), 4, False),
    "oak2": (
        k("OrthogonalAdditiveKernel", dim=3, quad_deg=10, second_order=True, dtype=F64),
        3,
        False,
    ),
    "oak_matern": (
        k(
            "OrthogonalAdditiveKernel",
            dim=2,
            quad_deg=8,
            base_kernel=s_(k("MaternKernel", nu=1.5, batch_shape=[2])),
            dtype=F64,
        ),
        2,
        False,
    ),
    "tree": (
        k(
            "ScaleKernel",
            s_(
                k(
                    "AdditiveKernel",
                    s_(k("ProductKernel", s_(RBF2), s_(k("PeriodicKernel", active_dims=[2])))),
                    s_(k("LinearKernel", active_dims=[1])),
                )
            ),
        ),
        3,
        False,
    ),
}


def _points(n: int, d: int, seed: int, tasks: bool = False, num_tasks: int = 3) -> Any:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=DT)
    if tasks:
        X[..., -1] = torch.randint(0, num_tasks, (n,), generator=g).to(DT) + 0.3
    return X


def _copy_kernel_params(kb: Any, kr: Any) -> None:
    kr.initialize(**{name: p.detach() for name, p in kb.named_parameters()})


def _randomize(module: Any, seed: int, scale: float = 0.5) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in module.named_parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g, dtype=DT))


@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("name", list(KERNELS))
def test_kernel_values_match_gpytorch(name: str, batch: tuple[int, ...]) -> None:
    spec, d, tasks = KERNELS[name]
    if batch:
        if name.startswith("oak"):
            pytest.skip("BoTorch's OrthogonalAdditiveKernel base kernels carry the input dimension")
        spec = with_batch(spec, list(batch))
    torch.manual_seed(3)
    kb = build_kernel(spec, "bt").to(DT)
    torch.manual_seed(3)
    kr = build_kernel(spec, "rs")
    # Index kernels: identical draws from torch's generator.
    for (nb, pb), (nr, pr) in zip(kb.named_parameters(), kr.named_parameters(), strict=True):
        assert nb == nr
        CHECK.check("kernel: default raw parameters", pr, pb.detach(), 0.0)
    _randomize(kb, 11)
    _copy_kernel_params(kb, kr)
    nt = 4 if name == "positive_index" else 3
    x1 = _points(5, d, 1, tasks, nt)
    x2 = _points(4, d, 2, tasks, nt)
    if name == "categorical":
        x1, x2 = x1.round(decimals=0), x2.round(decimals=0)
    with torch.no_grad():
        CHECK.check("kernel: k(x1, x2)", kr(x1, x2), kb(x1, x2).to_dense(), VALUE_TOL)
        CHECK.check("kernel: k(x1)", kr(x1), kb(x1).to_dense(), VALUE_TOL)
        expected = kb(x1, diag=True)
        if not torch.is_tensor(expected):
            # GPyTorch's Kernel.__call__ takes the diagonal of kernels that
            # ignore ``diag`` only if the result has the inputs' batch
            # dimensions: batched index kernels on non-batched inputs return
            # the full matrix.
            expected = expected.to_dense().diagonal(dim1=-2, dim2=-1)
        CHECK.check("kernel: k(x1, diag=True)", kr(x1, diag=True), expected, VALUE_TOL)


@pytest.mark.parametrize("raw", [-30.0, -46.0, -60.0, -69.0, -700.0, -800.0])
def test_periodic_kernel_at_vanishing_lengthscales_matches_gpytorch(raw: float) -> None:
    """GPyTorch clamps the per-dimension distances of the periodic kernel at
    1e-15, also on the diagonal of ``k(x, x)``, which therefore vanishes
    when the lengthscale underflows (L-BFGS-B steps there in fits)."""
    x = _points(5, 2, 9)
    x[3, 0] = x[1, 0]
    kb = gpytorch.kernels.PeriodicKernel(ard_num_dims=2).to(DT)
    kr = build_kernel(k("PeriodicKernel", ard_num_dims=2), "rs")
    with torch.no_grad():
        kb.raw_lengthscale.fill_(raw)
    _copy_kernel_params(kb, kr)
    with torch.no_grad():
        CHECK.check("kernel: periodic, vanishing lengthscale", kr(x), kb(x).to_dense(), VALUE_TOL)
        CHECK.check(
            "kernel: periodic, vanishing lengthscale",
            kr(x, diag=True),
            kb(x, diag=True),
            VALUE_TOL,
        )


def test_downsampling_diag_with_several_inputs_is_the_full_diagonal() -> None:
    """``DownsamplingKernel(x, diag=True)`` is the diagonal of ``k(x, x)``,
    ``offset + sum_d ((1 - x_d)^2)^(1 + power)``. BoTorch computes
    ``offset + (sum_d (1 - x_d)^2)^(1 + power)``, which differs for more than
    one input (a documented deviation)."""
    from botorch.models.kernels.downsampling import DownsamplingKernel as BtDownsampling

    from botorch_rs.models.kernels import DownsamplingKernel as RsDownsampling

    kb = BtDownsampling().to(DT)
    kr = RsDownsampling()
    _randomize(kb, 21)
    _copy_kernel_params(kb, kr)
    x = _points(5, 3, 22)
    with torch.no_grad():
        full = kb(x).to_dense().diagonal()
        ParityLog.record(
            "DownsamplingKernel: BoTorch's diag=True error (d = 3)",
            rel_err(kb(x, diag=True), full),
            section=CHECK.section,
        )
    CHECK.check("kernel: Downsampling diag=True (d = 3)", kr(x, diag=True), full, VALUE_TOL)
    CHECK.check("kernel: Downsampling k(x1) (d = 3)", kr(x), kb(x).to_dense().detach(), VALUE_TOL)


def test_kernel_arithmetic_matches_gpytorch() -> None:
    a = build_kernel(k("RBFKernel", active_dims=[0]), "rs")
    b = build_kernel(k("MaternKernel", nu=0.5, active_dims=[1]), "rs")
    c = build_kernel(k("PeriodicKernel"), "rs")
    ab = gpytorch.kernels.RBFKernel(active_dims=[0])
    bb = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=[1])
    cb = gpytorch.kernels.PeriodicKernel()
    for kr, kb in (
        (a + b + c, ab + bb + cb),
        (a * b * c, ab * bb * cb),
        ((a + b) * c, (ab + bb) * cb),
    ):
        assert [n for n, _ in kr.named_parameters()] == [n for n, _ in kb.named_parameters()]
        _randomize(kb, 5)
        _copy_kernel_params(kb, kr)
        x = _points(6, 2, 7)
        CHECK.check("kernel: arithmetic trees", kr(x), kb.to(DT)(x).to_dense().detach(), VALUE_TOL)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

#: A tree with priors on every parameter kind (and a fit).
PRIORS_FIT = k(
    "AdditiveKernel",
    s_(
        k(
            "ScaleKernel",
            s_(
                k(
                    "RBFKernel",
                    ard_num_dims=2,
                    lengthscale_prior=p_("NormalPrior", 0.6, 0.5),
                    lengthscale_constraint=c_(
                        "GreaterThan", 0.05, transform=None, initial_value=0.4
                    ),
                )
            ),
            outputscale_prior=p_("HalfNormalPrior", 1.5),
        )
    ),
    s_(
        k(
            "PeriodicKernel",
            period_length_prior=p_("UniformPrior", 0.1, 3.0),
            lengthscale_prior=p_("LogNormalPrior", 0.0, 1.5),
        )
    ),
    s_(k("LinearKernel", variance_prior=p_("HalfCauchyPrior", 0.5))),
)

#: Models: (spec, d, options). Options: ``tasks`` (task indices in the last
#: column), ``fixed`` (fixed noise), ``nostd`` (no outcome transform), ``mc``
#: (full-rank joint posteriors, for qSimpleRegret), ``exact_diagonal`` /
#: ``double_half_cauchy`` / ``double_lkj`` (BoTorch's quirks fixed), ``fit``.
MODELS: dict[str, tuple[dict[str, Any], int, dict[str, Any]]] = {
    "rbf_shared": (k("RBFKernel"), 3, {"mc": True, "fit": True}),
    "matern12": (
        k("MaternKernel", nu=0.5, ard_num_dims=2),
        2,
        {"mc": True, "exact_diagonal": True, "fit": True},
    ),
    "matern32_active": (
        k("ScaleKernel", s_(k("MaternKernel", nu=1.5, active_dims=[0, 2], ard_num_dims=2))),
        3,
        {"mc": True},
    ),
    "rq": (
        k("RQKernel", ard_num_dims=2, alpha_constraint=c_("Interval", 0.1, 5.0)),
        2,
        {"mc": True},
    ),
    "periodic": (k("ScaleKernel", s_(k("PeriodicKernel", ard_num_dims=2))), 2, {"mc": True}),
    "linear_rbf": (
        k("AdditiveKernel", s_(k("LinearKernel")), s_(k("RBFKernel", ard_num_dims=2))),
        2,
        {"mc": True},
    ),
    "polynomial": (k("PolynomialKernel", power=2), 2, {}),
    "cosine": (k("CosineKernel"), 1, {"mc": True}),
    "additive_active": (
        k(
            "AdditiveKernel",
            s_(k("RBFKernel", active_dims=[0])),
            s_(k("MaternKernel", ard_num_dims=2, active_dims=[1, 2])),
        ),
        3,
        {"mc": True, "fit": True},
    ),
    "product_scale": (
        k(
            "ScaleKernel",
            s_(
                k(
                    "ProductKernel",
                    s_(k("RBFKernel", ard_num_dims=2, active_dims=[0, 2])),
                    s_(k("PeriodicKernel", active_dims=[1])),
                )
            ),
        ),
        3,
        {"mc": True, "fit": True},
    ),
    "index": (
        k("ProductKernel", s_(RBF2), s_(k("IndexKernel", num_tasks=3, rank=2, active_dims=[2]))),
        3,
        {"tasks": True},
    ),
    "index_lkj": (
        k(
            "ProductKernel",
            s_(RBF2),
            s_(
                k(
                    "IndexKernel",
                    num_tasks=3,
                    rank=1,
                    active_dims=[2],
                    prior=p_("LKJCovariancePrior", 3, 1.5, ["GammaPrior", [2.0, 1.5]]),
                )
            ),
        ),
        3,
        {"tasks": True, "double_lkj": True},
    ),
    "positive_index": (
        k(
            "ProductKernel",
            s_(RBF2),
            s_(
                k(
                    "PositiveIndexKernel",
                    num_tasks=3,
                    rank=1,
                    active_dims=[2],
                    task_prior=p_("BetaPrior", 2.5, 1.5),
                )
            ),
        ),
        3,
        {"tasks": True, "fit": True},
    ),
    "categorical": (k("CategoricalKernel", ard_num_dims=2), 2, {"categorical": True}),
    "downsampling": (k("DownsamplingKernel", power_prior=p_("GammaPrior", 2.0, 1.0)), 1, {}),
    "expdecay": (k("ExponentialDecayKernel"), 1, {"mc": True}),
    "bnn": (k("InfiniteWidthBNNKernel", depth=2), 2, {"mc": True}),
    "ltf1": (
        k("LinearTruncatedFidelityKernel", fidelity_dims=[2], dimension=3),
        3,
        {"mc": True, "fit": True},
    ),
    "ltf2": (
        k("LinearTruncatedFidelityKernel", fidelity_dims=[1, 2], dimension=3, nu=0.5),
        3,
        {"mc": True, "exact_diagonal": True},
    ),
    "oak": (
        k("OrthogonalAdditiveKernel", dim=3, quad_deg=12, dtype=F64),
        3,
        {"mc": True, "fit": True},
    ),
    "oak2": (
        k(
            "OrthogonalAdditiveKernel",
            dim=2,
            quad_deg=8,
            second_order=True,
            dtype=F64,
            coeffs_2_prior=p_("HalfCauchyPrior", 1.0),
        ),
        2,
        {"mc": True, "double_half_cauchy": True},
    ),
    "priors": (
        k(
            "AdditiveKernel",
            s_(
                k(
                    "ScaleKernel",
                    s_(
                        k(
                            "RBFKernel",
                            ard_num_dims=2,
                            lengthscale_prior=p_("NormalPrior", 0.6, 0.5),
                            lengthscale_constraint=c_(
                                "GreaterThan", 0.05, transform=None, initial_value=0.4
                            ),
                        )
                    ),
                    outputscale_prior=p_("HalfNormalPrior", 1.5),
                )
            ),
            s_(
                k(
                    "PeriodicKernel",
                    period_length_prior=p_("UniformPrior", 0.1, 3.0),
                    lengthscale_prior=p_("LogNormalPrior", 0.0, 1.5),
                )
            ),
            s_(
                k(
                    "LinearKernel",
                    variance_prior=p_("HalfCauchyPrior", 0.5),
                    variance_constraint=c_("LessThan", 5.0),
                )
            ),
        ),
        2,
        # (No fit: the variance of the linear kernel turns negative, and
        # GPyTorch's LinearKernel NaN.)
        {"mc": True, "double_half_cauchy": True},
    ),
    "priors_fit": (PRIORS_FIT, 2, {"mc": True, "double_half_cauchy": True, "fit": True}),
    "fixed_noise": (
        k("ScaleKernel", s_(k("RBFKernel", ard_num_dims=2, active_dims=[1, 0]))),
        2,
        {"fixed": True, "mc": True},
    ),
    "nostd": (
        k("ProductKernel", s_(k("RBFKernel", active_dims=[0])), s_(k("RQKernel", active_dims=[1]))),
        2,
        {"nostd": True, "mc": True},
    ),
}


def _data(
    d: int, n: int, seed: int, opts: dict[str, Any], m: int = 1, batch: tuple[int, ...] = ()
) -> tuple[Any, Any, Any]:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(*batch, n, d, generator=g, dtype=DT)
    if opts.get("tasks"):
        X[..., -1] = (torch.arange(n, dtype=DT) % 3) + 0.25
    if opts.get("categorical"):
        X = torch.randint(0, 3, (*batch, n, d), generator=g).to(DT)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * X[..., 0]) + (1.0 + t) * (X[..., 1:] ** 2).sum(-1)
        cols.append(f + 0.05 * torch.randn(*batch, n, generator=g, dtype=DT))
    Y = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(*batch, n, m, generator=g, dtype=DT)
    return X, Y, Yvar


def _fixes(opts: dict[str, Any]) -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    if opts.get("exact_diagonal"):
        stack.enter_context(zero_diagonal_distances())
    if opts.get("double_half_cauchy"):
        stack.enter_context(double_precision_half_cauchy())
    return stack


def lkj_in_double(bt: Any) -> None:
    """The normalizing constants of the LKJ priors of ``bt`` in double
    precision: torch's ``LKJCholesky`` evaluates them in the dtype of its
    concentration, which is not a buffer and stays in float32 when the model
    is cast to double precision."""
    for _, _, prior_, _, _ in bt.named_priors():
        corr = getattr(prior_, "correlation_prior", None)
        if corr is not None:
            corr.concentration = corr.concentration.to(DT)


def model_pair(
    spec: dict[str, Any],
    d: int,
    opts: dict[str, Any],
    seed: int,
    n: int = 10,
    m: int = 1,
    batch: tuple[int, ...] = (),
    X: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """The same SingleTaskGP in both libraries (default parameters), the
    training data and the BoTorch model's default raw parameters."""
    X0, Y, Yvar = _data(d, n, seed, opts, m, batch)
    X = X0 if X is None else X
    aug = list(batch) + ([m] if m > 1 else [])
    if aug:
        spec = with_batch(spec, aug)
    kw: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if opts.get("nostd"):
        kw["outcome_transform"] = kr["outcome_transform"] = None
    yvar = Yvar if opts.get("fixed") else None
    torch.manual_seed(seed)
    bt = BtSingleTaskGP(X, Y, train_Yvar=yvar, covar_module=build_kernel(spec, "bt"), **kw)
    torch.manual_seed(seed)
    rs = RsSingleTaskGP(X, Y, train_Yvar=yvar, covar_module=build_kernel(spec, "rs"), **kr)
    if opts.get("double_lkj"):
        lkj_in_double(bt)
    return bt, rs, X, {n_: p.detach().clone() for n_, p in bt.named_parameters()}


def _bounds(bt: Any) -> dict[str, tuple[float, float]]:
    _, bounds = get_parameters_and_bounds(BtMLL(bt.likelihood, bt))
    out = {}
    for name, (lo, hi) in bounds.items():
        out[name.removeprefix("model.")] = (
            -math.inf if lo is None else float(lo),
            math.inf if hi is None else float(hi),
        )
    return out


def randomize_model(bt: Any, rs: Any, seed: int) -> None:
    """Random raw parameters (within the optimizer bounds) in both models."""
    g = torch.Generator().manual_seed(seed)
    bounds = _bounds(bt)
    with torch.no_grad():
        for name, p in bt.named_parameters():
            lo, hi = bounds.get(name, (-math.inf, math.inf))
            if "raw_noise" in name:
                v = torch.full(p.shape, 0.03, dtype=DT)
                if not math.isfinite(lo):
                    v = v.expm1().log()
            elif "raw_constant" in name:
                v = 0.4 * torch.randn(p.shape, generator=g, dtype=DT)
            else:
                v = 0.5 * torch.randn(p.shape, generator=g, dtype=DT)
                if math.isfinite(lo):
                    v = lo + 0.1 + v.abs()
                if math.isfinite(hi):
                    v = v.clamp(max=hi - 0.05)
            p.copy_(v)
    rs.initialize(**{name: p.detach() for name, p in bt.named_parameters()})


def bt_mll_value_and_grad(bt: Any) -> tuple[Any, Any]:
    mll = BtMLL(bt.likelihood, bt)
    bt.train()
    bt.zero_grad()
    values = mll(bt(*bt.train_inputs), bt.train_targets)
    values.sum().backward()
    grad = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for _, p in bt.named_parameters()
        ]
    )
    bt.zero_grad()
    bt.eval()
    return values.detach().reshape(-1), grad


def rs_mll_value_and_grad(rs: Any) -> tuple[Any, Any]:
    value, grad = RsMLL(rs.likelihood, rs).value_and_grad()
    return torch.as_tensor(value, dtype=DT).reshape(-1), torch.as_tensor(grad, dtype=DT)


def compare_structure(bt: Any, rs: Any, random_init: bool) -> None:
    nb = [(n, tuple(p.shape)) for n, p in bt.named_parameters()]
    nr = [(n, tuple(torch.as_tensor(p).shape)) for n, p in rs.named_parameters()]
    assert nb == nr
    for (name, pb), (_, pr) in zip(bt.named_parameters(), rs.named_parameters(), strict=True):
        if random_init or not (name.endswith("covar_factor") or name.endswith(".raw_var")):
            CHECK.check("model: default raw parameters", pr, pb.detach(), 0.0)
    bounds = _bounds(bt)
    expected = []
    for name, p in bt.named_parameters():
        expected.extend([bounds.get(name, (-math.inf, math.inf))] * p.numel())
    assert [tuple(b) for b in rs._rs.raw_parameter_bounds()] == expected


def botorch_prior_errors(bt: Any) -> dict[str, Any]:
    """The parts of BoTorch's MLL from its prior bugs (documented deviations
    of botorch_rs), divided by the number of data like the MLL:

    * ``OrthogonalAdditiveKernel``: a prior on the second-order coefficients
      evaluated also at the zeros of the ``dim x dim`` coefficient matrix
      outside its strict upper triangle (a constant);
    * ``LKJCovariancePrior`` with a univariate ``sd_prior``: GPyTorch's
      ``log_prob`` has ``n`` values ``log p_LKJ(R) + log p(sd_i)``, whose sum
      counts the correlation term ``n`` times (``n - 1`` too many; this term
      has a gradient)."""
    n_data = bt.train_targets.shape[-1]
    out: dict[str, Any] = {}
    for name, module, prior_, closure, _ in bt.named_priors():
        value = closure(module)
        if name.endswith("coeffs_2_prior"):
            c = value.detach()
            d = c.shape[-1]
            zeros = ~torch.ones(d, d, dtype=torch.bool).triu(1)
            out["OAK second-order prior on zeros"] = prior_.log_prob(c)[..., zeros].sum() / n_data
        corr_prior = getattr(prior_, "correlation_prior", None)
        if corr_prior is not None:
            var = value.diagonal(dim1=-2, dim2=-1)
            sd = var.sqrt()
            corr = value / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
            extra = (corr_prior.n - 1) * corr_prior.log_prob(corr)
            out["LKJCovariancePrior correlation term counted n times"] = extra.sum() / n_data
    return out


def compare_mll(name: str, bt: Any, rs: Any) -> None:
    errors = botorch_prior_errors(bt)
    bt.zero_grad()
    offset = sum(errors.values(), torch.zeros((), dtype=DT))
    if offset.requires_grad:
        offset.backward()
    offset_grad = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for _, p in bt.named_parameters()
        ]
    )
    vb, gb = bt_mll_value_and_grad(bt)
    vr, gr = rs_mll_value_and_grad(rs)
    for label, err in errors.items():
        ParityLog.record(f"{label} (BoTorch's MLL error)", abs(err.item()), section=CHECK.section)
    vb, gb = vb - offset.detach(), gb - offset_grad
    CHECK.check(f"{name} MLL", vr, vb, VALUE_TOL)
    check_grad(f"{name} MLL gradient", gr, gb)


def _values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.detach().clone().requires_grad_(True)
    v = acqf(Xg)
    (g,) = torch.autograd.grad(v.sum(), Xg, allow_unused=True)
    return v.detach(), torch.zeros_like(X) if g is None else g


def compare_posteriors(bt: Any, rs: Any, d: int, opts: dict[str, Any], seed: int) -> None:
    bt.eval()
    Xt = _data(d, 3, seed + 100, opts)[0]
    with torch.no_grad():
        for noise in (False, True):
            pb = bt.posterior(Xt, observation_noise=noise)
            pr = rs.posterior(Xt, observation_noise=noise)
            CHECK.check("posterior mean", pr.mean, pb.mean, VALUE_TOL)
            CHECK.check("posterior variance", pr.variance, pb.variance, VALUE_TOL)
            CHECK.check(
                "posterior covariance",
                pr.distribution.covariance_matrix,
                pb.distribution.covariance_matrix,
                VALUE_TOL,
            )
    X1 = Xt.unsqueeze(-2)
    for bt_acqf, rs_acqf, label in (
        (BtPosteriorMean(bt), RsPosteriorMean(rs), "PosteriorMean"),
        (BtPSD(bt), RsPSD(rs), "PosteriorStandardDeviation"),
    ):
        vb, gb = _values_and_grads(bt_acqf, X1)
        vr, gr = _values_and_grads(rs_acqf, X1)
        CHECK.check(f"{label} values", vr, vb, VALUE_TOL)
        check_grad(f"{label} gradients", gr, gb)
    if opts.get("mc"):
        Z = torch.randn(16, 1, 2, generator=torch.Generator().manual_seed(seed), dtype=DT)
        sb = BtSampler(sample_shape=torch.Size([16]), seed=0)
        sb.base_samples = Z.clone()
        sr = RsSampler(sample_shape=torch.Size([16]), seed=0)
        sr.base_samples = Z.clone()
        Xq = _data(d, 4, seed + 200, opts)[0].reshape(2, 2, d)
        vb, gb = _values_and_grads(BtQSR(bt, sampler=sb), Xq)
        vr, gr = _values_and_grads(RsQSR(rs, sampler=sr), Xq)
        CHECK.check("qSimpleRegret values", vr, vb, VALUE_TOL)
        check_grad("qSimpleRegret gradients", gr, gb)


@pytest.mark.parametrize("name", list(MODELS))
def test_models_match_botorch(name: str) -> None:
    spec, d, opts = MODELS[name]
    seed = 100 + list(MODELS).index(name)
    with _fixes(opts):
        bt, rs, _, _ = model_pair(spec, d, opts, seed)
        compare_structure(bt, rs, random_init=True)
        compare_mll("default", bt, rs)
        randomize_model(bt, rs, seed + 1)
        compare_mll("random", bt, rs)
        compare_posteriors(bt, rs, d, opts, seed)


def test_botorch_quirk_sizes() -> None:
    """Records the effect of the quirks that the comparisons fix (Matérn-1/2
    diagonal in training mode, float32 HalfCauchyPrior, float32 LKJ
    normalizing constants)."""
    for name in ("matern12", "oak2", "priors", "index_lkj"):
        spec, d, opts = MODELS[name]
        seed = 100 + list(MODELS).index(name)
        bt, rs, _, _ = model_pair(spec, d, {**opts, "double_lkj": False}, seed)
        randomize_model(bt, rs, seed + 1)
        plain, _ = bt_mll_value_and_grad(bt)
        if opts.get("double_lkj"):
            lkj_in_double(bt)
        with _fixes(opts):
            fixed, _ = bt_mll_value_and_grad(bt)
        ParityLog.record(
            f"BoTorch's MLL change by the fixed quirk ({name})",
            (plain - fixed).abs().max().item(),
            section=CHECK.section,
        )


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _bt_fit(bt: Any) -> tuple[float, int]:
    import botorch.optim.fit as bt_optim_fit

    record: dict[str, Any] = {"attempts": 0}
    orig = bt_optim_fit.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    bt_fit(BtMLL(bt.likelihood, bt), optimizer=recorder)
    return float(record["r"].fval), record["attempts"]


def _rs_fit(rs: Any) -> tuple[float, int]:
    mll = RsMLL(rs.likelihood, rs)
    rs_fit(mll)
    return float(mll._fit_result["fval"]), int(mll._fit_result["attempts"])


def _twin(X0: Any, t: int, opts: dict[str, Any]) -> Any:
    """Twin ``t`` of the training inputs ``X0`` (``t = 0``: ``X0``)."""
    if t == 0:
        return X0
    g = torch.Generator().manual_seed(t)
    rel = TWIN_SCALES[(t - 1) % len(TWIN_SCALES)]
    X = X0.clone()
    cols = list(range(X.shape[-1] - 1)) if opts.get("tasks") else list(range(X.shape[-1]))
    X[:, cols] = X[:, cols] * (1 + rel * torch.randn(X[:, cols].shape, generator=g, dtype=DT))
    return X


@pytest.mark.parametrize("name", [n for n, (_, _, o) in MODELS.items() if o.get("fit")])
def test_fit_from_identical_initial_values(name: str) -> None:
    spec, d, opts = MODELS[name]
    seed = 300 + list(MODELS).index(name)
    with _fixes(opts):
        X0 = _data(d, 10, seed, opts)[0]
        bt_out: list[tuple[float, int]] = []
        rs_out: list[tuple[float, int]] = []

        def run(twins: range) -> None:
            for t in twins:
                bt, rs, _, defaults = model_pair(spec, d, opts, seed, X=_twin(X0, t, opts))
                rs.initialize(**defaults)
                bt_out.append(_bt_fit(bt))
                rs_out.append(_rs_fit(rs))

        run(range(N_TWINS + 1))
        first = [loss for loss, a in bt_out if a == 1]
        if max(first) - min(first) > CHAOTIC_RANGE:
            # Chaotic: more twins for the statistical comparison.
            run(range(N_TWINS + 1, CHAOTIC_TWINS + 1))
    # Outcomes of first attempts only: retries start from prior samples,
    # which the two libraries draw from different random number generators.
    bt_first = [loss for loss, a in bt_out if a == 1]
    rs_first = [loss for loss, a in rs_out if a == 1]
    CHECK.record(
        "fit: first attempts failed (botorch_rs - BoTorch)",
        max(len(bt_first) - len(rs_first), 0),
        None,
    )
    best, worst = min(bt_first), max(bt_first)
    spread = max(v for v in bt_first if v <= best + FIT_CLUSTER) - best
    tol = max(FIT_TOL, 10 * spread)
    CHECK.record("fit: BoTorch's spread at its best optimum", spread, None)
    if worst - best > CHAOTIC_RANGE:
        compare_chaotic_fits(CHECK, "fit", bt_first, rs_first, FIT_TOL)
    else:
        CHECK.check_bound(
            "fit: best loss above BoTorch's best (scaled)", (min(rs_first) - best) / tol, 1.0
        )
        if rs_out[0][1] == 1:
            CHECK.check_bound(
                "fit: loss above BoTorch's worst (scaled)", (rs_out[0][0] - worst) / tol, 1.0
            )


def test_fit_leaving_a_prior_support_raises_like_botorch() -> None:
    """BoTorch retries only ``NotPSDError``: a period length leaving the
    support of its ``UniformPrior`` during the fit raises ``ValueError`` in
    both libraries, and the models keep their hyperparameters."""
    opts = MODELS["priors_fit"][2]
    with _fixes(opts):
        for t in range(3):
            X = _twin(_data(2, 10, 321, opts)[0], t, opts)
            bt, rs, _, defaults = model_pair(PRIORS_FIT, 2, opts, 321, X=X)
            rs.initialize(**defaults)
            with pytest.raises(ValueError, match="support"):
                _bt_fit(bt)
            with pytest.raises(ValueError, match="support"):
                _rs_fit(rs)
            for (_, pb), (_, pr) in zip(bt.named_parameters(), rs.named_parameters(), strict=True):
                CHECK.check("failed fit: hyperparameters kept", pr, pb.detach(), 0.0)


# ---------------------------------------------------------------------------
# Multi-output, batched and list models, conditioning and fantasies
# ---------------------------------------------------------------------------

TREE = k(
    "ScaleKernel",
    s_(
        k(
            "AdditiveKernel",
            s_(k("RBFKernel", active_dims=[0])),
            s_(k("MaternKernel", nu=1.5, active_dims=[1])),
        )
    ),
)


@pytest.mark.parametrize("m,batch", [(2, ()), (1, (2,)), (2, (2,))])
def test_multi_output_and_batched_trees_match_botorch(m: int, batch: tuple[int, ...]) -> None:
    bt, rs, _, _ = model_pair(TREE, 2, {}, 41, m=m, batch=batch)
    compare_structure(bt, rs, random_init=True)
    randomize_model(bt, rs, 42)
    compare_mll("multi-output / batched", bt, rs)
    Xt = _data(2, 3, 43, {})[0]
    with torch.no_grad():
        pb = bt.posterior(Xt)
        pr = rs.posterior(Xt)
        CHECK.check("multi-output / batched posterior mean", pr.mean, pb.mean, VALUE_TOL)
        CHECK.check(
            "multi-output / batched posterior variance", pr.variance, pb.variance, VALUE_TOL
        )


def test_model_list_conditioning_and_fantasies_match_botorch() -> None:
    a_b, a_r, X, _ = model_pair(TREE, 2, {}, 51)
    b_b, b_r, _, _ = model_pair(MODELS["periodic"][0], 2, {}, 52)
    randomize_model(a_b, a_r, 53)
    randomize_model(b_b, b_r, 54)
    lb, lr = BtModelListGP(a_b, b_b), RsModelListGP(a_r, b_r)
    lb.eval()
    Xt = _data(2, 3, 55, {})[0]
    with torch.no_grad():
        pb, pr = lb.posterior(Xt), lr.posterior(Xt)
    CHECK.check("ModelListGP posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check("ModelListGP posterior variance", pr.variance, pb.variance, VALUE_TOL)
    # Conditioning on observations.
    a_b.posterior(Xt)
    a_r.posterior(Xt)
    Xn = _data(2, 2, 56, {})[0]
    Yn = torch.tensor([[0.3], [-0.2]], dtype=DT)
    cb, cr = a_b.condition_on_observations(Xn, Yn), a_r.condition_on_observations(Xn, Yn)
    with torch.no_grad():
        CHECK.check(
            "conditioned posterior mean", cr.posterior(Xt).mean, cb.posterior(Xt).mean, VALUE_TOL
        )
        CHECK.check(
            "conditioned posterior covariance",
            cr.posterior(Xt).distribution.covariance_matrix,
            cb.posterior(Xt).distribution.covariance_matrix,
            VALUE_TOL,
        )
    # Fantasies with BoTorch's base samples.
    sb = BtSampler(sample_shape=torch.Size([3]), seed=57)
    fb = a_b.fantasize(Xn, sb)
    sr = RsSampler(sample_shape=torch.Size([3]), seed=0)
    sr.base_samples = sb.base_samples.clone()
    fr = a_r.fantasize(Xn, sr)
    with torch.no_grad():
        CHECK.check(
            "fantasy posterior mean", fr.posterior(Xt).mean, fb.posterior(Xt).mean, VALUE_TOL
        )
        CHECK.check(
            "fantasy posterior variance",
            fr.posterior(Xt).variance,
            fb.posterior(Xt).variance,
            VALUE_TOL,
        )


# ---------------------------------------------------------------------------
# Pathwise sampling
# ---------------------------------------------------------------------------

FEATURES = 64
PATH_MODELS: dict[str, tuple[dict[str, Any], int]] = {
    "rbf_shared": (k("RBFKernel"), 3),
    "matern12_scale": (k("ScaleKernel", s_(k("MaternKernel", nu=0.5, ard_num_dims=2))), 2),
    "matern32_active": (
        k("ScaleKernel", s_(k("MaternKernel", nu=1.5, ard_num_dims=2, active_dims=[0, 2]))),
        3,
    ),
    "rbf_scale_active": (k("ScaleKernel", s_(k("RBFKernel", active_dims=[1]))), 3),
}


def _bt_matheron_paths(model: Any, noises: list[Any] | None = None) -> Any:
    """BoTorch's Matheron paths of ``model`` with the draws of seed 63
    (recorded in ``noises``)."""
    torch.manual_seed(63)
    with recorded_randn(noises) if noises is not None else contextlib.nullcontext():
        return bt_pw.draw_matheron_paths(
            model,
            torch.Size([3]),
            prior_sampler=functools.partial(bt_pw.draw_kernel_feature_paths, num_features=FEATURES),
        )


@pytest.mark.parametrize("name", list(PATH_MODELS))
def test_matheron_paths_match_botorch(name: str) -> None:
    """Path values and gradients with BoTorch's draws. The Matheron update
    solves with the training covariance, which is ill-conditioned for the
    non-smooth Matérn-1/2 kernel: BoTorch's own paths change by a median of
    2e-9 when the training inputs are perturbed at 1e-14 .. 1e-11 (the other
    kernels: below 3e-12), so the tolerances are at least 10 times BoTorch's
    median change on 8 such twins."""
    spec, d = PATH_MODELS[name]
    bt, rs, X, _ = model_pair(spec, d, {}, 61)
    randomize_model(bt, rs, 62)
    bt.eval()
    noises: list[Any] = []
    bp = _bt_matheron_paths(bt, noises)
    rp = rs_pw.draw_matheron_paths(
        rs,
        torch.Size([3]),
        functools.partial(rs_pw.draw_kernel_feature_paths, num_features=FEATURES),
        draws=bt_path_draws(bp, noises),
    )
    Xt = _data(d, 5, 64, {})[0]
    Xb = Xt.clone().requires_grad_(True)
    (gb,) = torch.autograd.grad(bp(Xb).sum(), Xb)
    floor_v, floor_g = [], []
    for k_ in range(1, 9):
        twin = perturbed_copy(bt, seed=k_, rel=(1e-14, 1e-13, 1e-12, 1e-11)[(k_ - 1) % 4])
        twin.eval()
        tp = _bt_matheron_paths(twin)
        floor_v.append(rel_err(tp(Xt).detach(), bp(Xt).detach()))
        Xk = Xt.clone().requires_grad_(True)
        (gk,) = torch.autograd.grad(tp(Xk).sum(), Xk)
        floor_g.append(_grad_err(gk, gb))
    tol_v = max(VALUE_TOL, 10 * statistics.median(floor_v))
    tol_g = max(GRAD_TOL, 10 * statistics.median(floor_g))
    CHECK.record(
        "Matheron paths: BoTorch's median change on twins, values", statistics.median(floor_v), None
    )
    CHECK.check("Matheron paths: values", rp(Xt), bp(Xt).detach(), tol_v)
    Xr = Xt.clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(rp(Xr).sum(), Xr)
    check_grad("Matheron paths: gradients", gr, gb, tol_g)
    torch.manual_seed(65)
    bq = bt_pw.draw_kernel_feature_paths(bt, torch.Size([2]), num_features=FEATURES)
    rq = rs_pw.draw_kernel_feature_paths(
        rs, (2,), num_features=FEATURES, draws=bt_path_draws(bq, None)
    )
    CHECK.check("prior paths: values", rq(Xt), bq(Xt).detach(), VALUE_TOL)


def test_pathwise_errors_for_unsupported_kernels_match_botorch() -> None:
    bt, rs, _, _ = model_pair(MODELS["linear_rbf"][0], 2, {}, 66)
    with pytest.raises(NotImplementedError):
        bt_pw.draw_matheron_paths(bt, torch.Size([2]))
    with pytest.raises(NotImplementedError):
        rs_pw.draw_matheron_paths(rs, torch.Size([2]))


# ---------------------------------------------------------------------------
# Priors and constraints
# ---------------------------------------------------------------------------

PRIORS = [
    ("NormalPrior", (0.3, 1.7), [0.05, 0.37, 2.5]),
    ("UniformPrior", (0.1, 2.3), [0.15, 1.0, 2.2]),
    ("HalfNormalPrior", (0.7,), [0.05, 0.37, 2.5]),
    ("HalfNormalPrior", (1.5,), [0.05, 0.37, 2.5]),
    ("HalfCauchyPrior", (0.5,), [0.05, 0.37, 2.5]),
    ("LogNormalPrior", (0.0, 1.5), [0.05, 0.37, 2.5]),
    ("GammaPrior", (2.5, 1.5), [0.05, 0.37, 2.5]),
    ("BetaPrior", (2.5, 1.5), [0.05, 0.37, 0.95]),
]


@pytest.mark.parametrize("name,args,xs", PRIORS)
def test_priors_match_gpytorch(name: str, args: tuple[float, ...], xs: list[float]) -> None:
    """The log densities of priors registered on a module cast to double
    precision (as in BoTorch's models)."""
    pb = (
        build_kernel(k("RBFKernel", lengthscale_prior=p_(name, *args)), "bt")
        .to(DT)
        .lengthscale_prior
    )
    pr = build_kernel(k("RBFKernel", lengthscale_prior=p_(name, *args)), "rs").lengthscale_prior
    x = torch.tensor(xs, dtype=DT)
    expected = pb.log_prob(x).to(DT)
    if name == "HalfCauchyPrior":
        with double_precision_half_cauchy():
            fixed = pb.log_prob(x)
        ParityLog.record(
            "HalfCauchyPrior: BoTorch's float32 error",
            rel_err(expected, fixed),
            section=CHECK.section,
        )
        expected = fixed
    CHECK.check(f"prior log_prob ({name})", pr.log_prob(x), expected, VALUE_TOL)


def test_prior_support_errors_match_gpytorch() -> None:
    for name, args, bad in (("UniformPrior", (0.1, 2.3), 3.0), ("HalfNormalPrior", (1.0,), -0.5)):
        pb = build_kernel(k("RBFKernel", lengthscale_prior=p_(name, *args)), "bt").lengthscale_prior
        pr = build_kernel(k("RBFKernel", lengthscale_prior=p_(name, *args)), "rs").lengthscale_prior
        with pytest.raises(ValueError):
            pb.log_prob(torch.tensor(bad, dtype=DT))
        with pytest.raises(ValueError):
            pr.log_prob(bad)


def test_lkj_priors_match_gpytorch() -> None:
    """LKJ priors with ``eta`` in double precision (botorch_rs keeps a
    float64 tensor's precision, like torch). ``LKJCovariancePrior`` is
    compared with the correct density, ``log p_LKJ(R) + sum_i log p(sd_i)``:
    GPyTorch returns ``log p_LKJ(R) + log p(sd_i)`` for every ``i`` (a
    documented deviation)."""
    from gpytorch.priors import GammaPrior, LKJCholeskyFactorPrior, LKJCovariancePrior, LKJPrior

    from botorch_rs import priors as rp

    g = torch.Generator().manual_seed(7)
    A = torch.randn(3, 3, generator=g, dtype=DT)
    cov = A @ A.T + 0.5 * torch.eye(3, dtype=DT)
    sd = cov.diagonal().sqrt()
    corr = cov / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
    for eta in (0.5, 0.7, 1.5, 2.0):
        e = torch.tensor(eta, dtype=DT)
        L = torch.linalg.cholesky(corr)
        CHECK.check(
            "LKJCholeskyFactorPrior log_prob",
            rp.LKJCholeskyFactorPrior(3, e).log_prob(L),
            LKJCholeskyFactorPrior(3, e).log_prob(L),
            VALUE_TOL,
        )
        CHECK.check(
            "LKJPrior log_prob",
            rp.LKJPrior(3, e).log_prob(corr),
            LKJPrior(3, e).log_prob(corr),
            VALUE_TOL,
        )
        bt = LKJCovariancePrior(3, e, GammaPrior(2.0, 1.5).to(DT))
        values = bt.log_prob(cov)
        assert values.shape == (3,)  # GPyTorch: one value per standard deviation
        expected = bt.correlation_prior.log_prob(corr) + bt.sd_prior.log_prob(sd).sum()
        ParityLog.record(
            "LKJCovariancePrior: GPyTorch's sum - correct density",
            abs((values.sum() - expected).item()),
            section=CHECK.section,
        )
        CHECK.check(
            "LKJCovariancePrior log_prob",
            rp.LKJCovariancePrior(3, e, rp.GammaPrior(2.0, 1.5)).log_prob(cov),
            expected,
            VALUE_TOL,
        )


LESS_THAN = [(2.0, True, None), (1.5, True, 0.2), (0.3, False, 0.1), (-1.0, False, -2.0)]


@pytest.mark.parametrize("upper,softplus,init", LESS_THAN)
def test_less_than_matches_gpytorch(upper: float, softplus: bool, init: Any) -> None:
    from gpytorch.constraints import LessThan as BtLessThan

    from botorch_rs.constraints import LessThan as RsLessThan

    kw: dict[str, Any] = {} if softplus else {"transform": None}
    if init is not None:
        kw["initial_value"] = init
    cb = BtLessThan(upper, **kw).to(DT)
    cr = RsLessThan(upper, **kw)
    raw = torch.tensor([-3.0, -0.5, 0.0, 0.8, 4.0], dtype=DT)
    CHECK.check(
        "LessThan transform", [cr.transform(v) for v in raw.tolist()], cb.transform(raw), VALUE_TOL
    )
    vals = torch.tensor([upper - 2.0, upper - 0.3, upper - 1e-3], dtype=DT)
    CHECK.check(
        "LessThan inverse_transform",
        [cr.inverse_transform(v) for v in vals.tolist()],
        cb.inverse_transform(vals),
        VALUE_TOL,
    )
    if init is not None:
        CHECK.check("LessThan initial raw value", cr.initial_value, cb.initial_value, VALUE_TOL)
    assert cr.upper_bound == float(cb.upper_bound)
    assert cr.enforced == cb.enforced
    if upper <= 0:
        return  # a variance below zero
    # Raw parameters, optimizer bounds and the MLL of a model using it.
    spec = k("LinearKernel", variance_constraint=c_("LessThan", upper, **kw))
    bt, rs, _, _ = model_pair(spec, 2, {}, 71)
    compare_structure(bt, rs, random_init=True)
    compare_mll("LessThan", bt, rs)


# ---------------------------------------------------------------------------
# Randomized sweep
# ---------------------------------------------------------------------------


@pytest.mark.equivalence
@pytest.mark.parametrize("scenario", range(40))
def test_random_kernel_trees_sweep(scenario: int) -> None:
    """Random trees of all kernels that take arbitrary inputs, with
    ``active_dims``, ARD, priors, constraints and ``ScaleKernel``s."""
    r = random.Random(4000 + scenario)
    d = r.randint(1, 3)
    spec = random_spec(r, d)
    seed = 5000 + scenario
    with zero_diagonal_distances():
        _sweep_case(spec, d, seed)


def _sweep_case(spec: dict[str, Any], d: int, seed: int) -> None:
    bt, rs, _, _ = model_pair(spec, d, {}, seed)
    compare_structure(bt, rs, random_init=True)
    randomize_model(bt, rs, seed + 1)
    try:
        expected = bt_mll_value_and_grad(bt)
    except RuntimeError:  # e.g. NotPSDError: botorch_rs must fail alike
        with pytest.raises(RuntimeError):
            rs_mll_value_and_grad(rs)
        return
    vr, gr = rs_mll_value_and_grad(rs)
    CHECK.check("sweep: MLL", vr, expected[0], VALUE_TOL)
    check_grad("sweep: MLL gradient", gr, expected[1])
    bt.eval()
    Xt = _data(d, 3, seed + 2, {})[0]
    with torch.no_grad():
        pb, pr = bt.posterior(Xt), rs.posterior(Xt)
    CHECK.check("sweep: posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "sweep: posterior covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix,
        VALUE_TOL,
    )
    X1 = Xt.unsqueeze(-2)
    vb, gb = _values_and_grads(BtPosteriorMean(bt), X1)
    vr2, gr2 = _values_and_grads(RsPosteriorMean(rs), X1)
    CHECK.check("sweep: PosteriorMean", vr2, vb, VALUE_TOL)
    check_grad("sweep: PosteriorMean gradient", gr2, gb)
    vb, gb = _values_and_grads(BtPSD(bt), X1)
    vr2, gr2 = _values_and_grads(RsPSD(rs), X1)
    CHECK.check("sweep: PosteriorStandardDeviation", vr2, vb, VALUE_TOL)
    check_grad("sweep: PosteriorStandardDeviation gradient", gr2, gb)
