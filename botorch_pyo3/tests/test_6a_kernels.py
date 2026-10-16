"""Kernel trees, GPyTorch's and BoTorch's kernels, priors and ``LessThan``
(phase 6a): values against closed forms, GPyTorch's parameter names, shapes
and defaults, kernel arithmetic, models with kernel trees, and the errors
(GPyTorch's and BoTorch's exception types). Runs without torch and BoTorch;
``test_parity_6a_kernels.py`` compares with GPyTorch and BoTorch."""

from __future__ import annotations

import math
import struct

import pytest

from botorch_rs.constraints import GreaterThan, Interval, LessThan
from botorch_rs.exceptions import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import (
    AdditiveKernel,
    CosineKernel,
    IndexKernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PolynomialKernel,
    ProductKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.kernels import (
    CategoricalKernel,
    DownsamplingKernel,
    ExponentialDecayKernel,
    InfiniteWidthBNNKernel,
    LinearTruncatedFidelityKernel,
    OrthogonalAdditiveKernel,
    PositiveIndexKernel,
)
from botorch_rs.models.utils.priors import BetaPrior
from botorch_rs.priors import (
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    LKJCovariancePrior,
    LKJPrior,
    LogNormalPrior,
    NormalPrior,
    UniformPrior,
)
from botorch_rs.sampling import pathwise
from helpers import as_kind, assert_close, check_kind, flatten, make_data, shape

LN2 = math.log(2.0)


def _f32(v: float) -> float:
    return float(struct.unpack("f", struct.pack("f", v))[0])


X1 = [[0.1, 0.7], [0.4, 0.2], [0.9, 0.5]]
X2 = [[0.3, 0.3], [0.8, 0.9]]


def _dist(x: list[float], y: list[float], ls: list[float]) -> float:
    return math.sqrt(sum(((a - b) / s) ** 2 for a, b, s in zip(x, y, ls, strict=True)))


def _closed_form(name: str, x: list[float], y: list[float]) -> float:
    """The kernels of ``_kernels()`` in closed form (GPyTorch's formulas)."""
    ls = [0.5, 0.8]
    r = _dist(x, y, ls)
    if name == "rbf":
        return math.exp(-0.5 * r * r)
    if name == "matern12":
        return math.exp(-max(r, 1e-15))
    if name == "matern32":
        a = math.sqrt(3) * r
        return (1 + a) * math.exp(-a)
    if name == "matern52":
        a = math.sqrt(5) * r
        return (1 + a + a * a / 3) * math.exp(-a)
    if name == "rq":
        return (1 + r * r / (2 * 1.5)) ** -1.5
    if name == "periodic":
        p = [1.3, 0.6]
        e = sum(
            math.sin(math.pi * max(abs(a - b), 1e-15) / per) ** 2 / s
            for a, b, per, s in zip(x, y, p, ls, strict=True)
        )
        return math.exp(-2.0 * e)
    if name == "linear":
        return sum(v * a * b for v, a, b in zip([0.7, 1.9], x, y, strict=True))
    if name == "polynomial":
        return (sum(a * b for a, b in zip(x, y, strict=True)) + 0.4) ** 3
    if name == "cosine":
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y, strict=True)))
        return math.cos(math.pi * max(d, 1e-15) / 0.9)
    if name == "scaled_sum":
        return 2.5 * _closed_form("rbf", x, y) + _closed_form("linear", x, y)
    if name == "product":
        return _closed_form("matern52", x, y) * _closed_form("periodic", x, y)
    raise AssertionError(name)


def _kernels() -> dict[str, object]:
    def rbf() -> RBFKernel:
        k = RBFKernel(ard_num_dims=2)
        k.lengthscale = [[0.5, 0.8]]
        return k

    def matern(nu: float) -> MaternKernel:
        k = MaternKernel(nu=nu, ard_num_dims=2)
        k.lengthscale = [[0.5, 0.8]]
        return k

    def periodic() -> PeriodicKernel:
        k = PeriodicKernel(ard_num_dims=2)
        k.lengthscale = [[0.5, 0.8]]
        k.period_length = [[1.3, 0.6]]
        return k

    def linear() -> LinearKernel:
        k = LinearKernel(ard_num_dims=2)
        k.variance = [[0.7, 1.9]]
        return k

    rq = RQKernel(ard_num_dims=2)
    rq.lengthscale = [[0.5, 0.8]]
    rq.alpha = 1.5
    poly = PolynomialKernel(power=3)
    poly.offset = 0.4
    cos = CosineKernel()
    cos.period_length = 0.9
    scaled = ScaleKernel(rbf())
    scaled.outputscale = 2.5
    return {
        "rbf": rbf(),
        "matern12": matern(0.5),
        "matern32": matern(1.5),
        "matern52": matern(2.5),
        "rq": rq,
        "periodic": periodic(),
        "linear": linear(),
        "polynomial": poly,
        "cosine": cos,
        "scaled_sum": scaled + linear(),
        "product": matern(2.5) * periodic(),
    }


@pytest.mark.parametrize("name", list(_kernels()))
def test_kernel_values_match_closed_forms(name: str, kind: str) -> None:
    k = _kernels()[name]
    K = k(as_kind(X1, kind), as_kind(X2, kind))  # type: ignore[operator]
    check_kind(K, kind)
    assert shape(K) == (3, 2)
    expected = [[_closed_form(name, x, y) for y in X2] for x in X1]
    assert_close(flatten(K), flatten(expected), rtol=1e-13, atol=1e-15)
    sym = k(as_kind(X1, kind))  # type: ignore[operator]
    expected = [[_closed_form(name, x, y) for y in X1] for x in X1]
    assert_close(flatten(sym), flatten(expected), rtol=1e-13, atol=1e-15)
    diag = k(as_kind(X1, kind), diag=True)  # type: ignore[operator]
    assert shape(diag) == (3,)
    assert_close(flatten(diag), [expected[i][i] for i in range(3)], rtol=1e-13, atol=1e-15)


def test_batched_kernels_evaluate_every_batch_element(kind: str) -> None:
    k = RBFKernel(batch_shape=[2])
    k.lengthscale = [[[0.5]], [[2.0]]]
    K = k(as_kind(X1, kind), as_kind(X2, kind))
    assert shape(K) == (2, 3, 2)
    for b, ls in enumerate((0.5, 2.0)):
        expected = [[math.exp(-0.5 * _dist(x, y, [ls, ls]) ** 2) for y in X2] for x in X1]
        assert_close(flatten(K)[6 * b : 6 * (b + 1)], flatten(expected), rtol=1e-13)
    # Batched inputs broadcast with the kernel's batch shape.
    Kb = k(as_kind([X1, X1], kind))
    assert shape(Kb) == (2, 3, 3)


def test_active_dims_select_inputs(kind: str) -> None:
    k = RBFKernel(active_dims=[1])
    X = [[0.1, 0.2, 0.9], [0.5, 0.6, 0.3]]
    K = k(as_kind(X, kind))
    assert_close(flatten(K)[1], math.exp(-0.5 * (0.4 / LN2) ** 2), rtol=1e-14)
    # ScaleKernel takes over the active_dims of its base kernel (GPyTorch).
    s = ScaleKernel(MaternKernel(nu=0.5, active_dims=[0, 2], ard_num_dims=2))
    assert s.active_dims == (0, 2)


def test_parameter_names_shapes_and_defaults() -> None:
    tree = ScaleKernel(
        AdditiveKernel(
            ProductKernel(
                RBFKernel(ard_num_dims=2, active_dims=[0, 1]), PeriodicKernel(active_dims=[2])
            ),
            LinearKernel(active_dims=[1]),
            IndexKernel(num_tasks=3, rank=2, active_dims=[3]),
        )
    )
    params = dict(tree.named_parameters())
    assert list(params) == [
        "raw_outputscale",
        "base_kernel.kernels.0.kernels.0.raw_lengthscale",
        "base_kernel.kernels.0.kernels.1.raw_lengthscale",
        "base_kernel.kernels.0.kernels.1.raw_period_length",
        "base_kernel.kernels.1.raw_variance",
        "base_kernel.kernels.2.covar_factor",
        "base_kernel.kernels.2.raw_var",
    ]
    shapes = {name: shape(v) for name, v in params.items()}
    assert shapes["raw_outputscale"] == ()
    assert shapes["base_kernel.kernels.0.kernels.0.raw_lengthscale"] == (1, 2)
    assert shapes["base_kernel.kernels.0.kernels.1.raw_period_length"] == (1, 1)
    assert shapes["base_kernel.kernels.1.raw_variance"] == (1, 1)
    assert shapes["base_kernel.kernels.2.covar_factor"] == (3, 2)
    assert shapes["base_kernel.kernels.2.raw_var"] == (3,)
    # GPyTorch's initial values: raw 0 (softplus(0) = log 2).
    assert_close(flatten(tree.outputscale), [LN2], rtol=1e-15)
    rbf = tree.base_kernel.kernels[0].kernels[0]
    assert_close(flatten(rbf.lengthscale), [LN2, LN2], rtol=1e-15)
    assert_close(flatten(tree.base_kernel.kernels[1].variance), [LN2], rtol=1e-15)
    # Batch shapes lead every parameter.
    batched = RQKernel(ard_num_dims=3, batch_shape=[2])
    assert {n: shape(v) for n, v in batched.named_parameters()} == {
        "raw_lengthscale": (2, 1, 3),
        "raw_alpha": (2, 1),
    }
    # Constraints with initial values set the raw initial value (GPyTorch).
    c = RBFKernel(lengthscale_constraint=GreaterThan(0.05, transform=None, initial_value=0.4))
    assert_close(flatten(c.raw_lengthscale), [0.4], rtol=1e-6)
    lt = LinearKernel(variance_constraint=LessThan(2.0))
    assert_close(flatten(lt.variance), [2.0 - LN2], rtol=1e-6)


def test_initialize_and_shape_errors() -> None:
    k = AdditiveKernel(RBFKernel(), MaternKernel(nu=1.5, ard_num_dims=2))
    k.initialize(**{"kernels.0.lengthscale": 0.5, "kernels.1.raw_lengthscale": [[0.1, -0.2]]})
    assert_close(flatten(k.kernels[0].lengthscale), [0.5], rtol=1e-14)
    assert_close(flatten(k.kernels[1].raw_lengthscale), [0.1, -0.2], rtol=1e-15)
    with pytest.raises(AttributeError):
        k.initialize(**{"kernels.0.alpha": 1.0})
    with pytest.raises(BotorchTensorDimensionError):
        k.kernels[1].lengthscale = [0.1, 0.2, 0.3]


def test_kernel_arithmetic_flattens_like_gpytorch() -> None:
    a, b, c = RBFKernel(), PeriodicKernel(), LinearKernel()
    s = a + b + c
    assert isinstance(s, AdditiveKernel) and list(s.kernels) == [a, b, c]
    p = a * b * c
    assert isinstance(p, ProductKernel) and list(p.kernels) == [a, b, c]
    mixed = (a + b) * c
    assert isinstance(mixed, ProductKernel) and isinstance(mixed.kernels[0], AdditiveKernel)
    assert [n for n, _ in mixed.named_parameters()][0] == "kernels.0.kernels.0.raw_lengthscale"
    assert repr(MaternKernel(nu=0.5, ard_num_dims=2)).startswith("MaternKernel(")


def _tree_models() -> dict[str, tuple[object, int, bool]]:
    """Kernel trees of every kernel class: (kernel, d, task column)."""
    return {
        "scale_additive": (
            ScaleKernel(RBFKernel(active_dims=[0]) + MaternKernel(nu=0.5, active_dims=[1])),
            2,
            False,
        ),
        "product_periodic": (
            RBFKernel(active_dims=[0]) * PeriodicKernel(active_dims=[1]),
            2,
            False,
        ),
        "rq_linear": (RQKernel(ard_num_dims=2) + LinearKernel(), 2, False),
        "polynomial_cosine": (PolynomialKernel(power=2) + CosineKernel(active_dims=[0]), 2, False),
        "index": (
            RBFKernel(active_dims=[0]) * IndexKernel(num_tasks=2, rank=1, active_dims=[1]),
            2,
            True,
        ),
        "positive_index": (
            RBFKernel(active_dims=[0])
            * PositiveIndexKernel(
                num_tasks=2, rank=1, active_dims=[1], task_prior=BetaPrior(2.0, 2.0)
            ),
            2,
            True,
        ),
        "categorical": (CategoricalKernel(), 2, False),
        "downsampling": (DownsamplingKernel(), 1, False),
        "expdecay": (ExponentialDecayKernel(), 1, False),
        "bnn": (InfiniteWidthBNNKernel(depth=2), 2, False),
        "ltf": (LinearTruncatedFidelityKernel(fidelity_dims=[1], dimension=2), 2, False),
        "oak": (OrthogonalAdditiveKernel(dim=2, quad_deg=8, second_order=True), 2, False),
    }


@pytest.mark.parametrize("name", list(_tree_models()))
def test_models_with_kernel_trees(name: str, kind: str) -> None:
    covar, d, tasks = _tree_models()[name]
    X, Y = make_data(n=10, d=d, seed=3)
    if tasks:
        X = [x[:-1] + [float(i % 2)] for i, x in enumerate(X)]
    if name == "categorical":
        X = [[float(round(2 * v)) for v in x] for x in X]
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), covar_module=covar)
    assert model.covar_module is covar
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    value, grad = mll.value_and_grad()
    assert math.isfinite(value) and all(math.isfinite(g) for g in grad)
    assert len(grad) == sum(len(flatten(p)) for _, p in model.named_parameters())
    fit_gpytorch_mll(mll)
    assert mll.value_and_grad()[0] >= value - 1e-9
    post = model.posterior(as_kind(X[:3], kind))
    check_kind(post.mean, kind)
    assert shape(post.mean) == (3, 1)
    assert all(v >= 0 for v in flatten(post.variance))
    # The bound kernel reads and writes the model's hyperparameters.
    first, raw = next(iter(covar.named_parameters()))  # type: ignore[attr-defined]
    assert_close(flatten(raw), flatten(dict(model.named_parameters())[f"covar_module.{first}"]))


def test_multi_output_and_batched_trees(kind: str) -> None:
    X, Y = make_data(n=10, d=2, seed=4)
    Y2 = [[y[0], 2 * y[0] + x[1]] for x, y in zip(X, Y, strict=True)]
    covar = ScaleKernel(
        RBFKernel(active_dims=[0], batch_shape=[2]) + LinearKernel(batch_shape=[2]), batch_shape=[2]
    )
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y2, kind), covar_module=covar)
    assert shape(covar.outputscale) == (2,)
    assert shape(model.posterior(as_kind(X[:4], kind)).mean) == (4, 2)
    with pytest.raises(UnsupportedError):  # shared by the outputs: not fitted by BoTorch
        SingleTaskGP(as_kind(X, kind), as_kind(Y2, kind), covar_module=RBFKernel())


def test_pathwise_sampling_with_kernel_trees(kind: str) -> None:
    X, Y = make_data(n=8, d=3, seed=5)
    covar = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 2]))
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), covar_module=covar)
    paths = pathwise.draw_matheron_paths(model, (4,), seed=0)
    values = paths(as_kind(X[:5], kind))
    assert shape(values) == (4, 5)
    unsupported = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), covar_module=PeriodicKernel())
    with pytest.raises(NotImplementedError):
        pathwise.draw_matheron_paths(unsupported, (2,))


# ---------------------------------------------------------------------------
# Priors and constraints
# ---------------------------------------------------------------------------


def test_prior_log_densities() -> None:
    x = 0.75
    cases = [
        (
            NormalPrior(0.5, 1.5),
            -0.5 * ((x - 0.5) / 1.5) ** 2 - math.log(1.5 * math.sqrt(2 * math.pi)),
        ),
        # (Parameters in single precision without torch, as GPyTorch's
        # in torch's default dtype: -log(b - a) is float32's log 2.)
        (UniformPrior(0.25, 2.25), -_f32(math.log(2.0))),
        (
            HalfNormalPrior(1.5),
            math.log(2) - 0.5 * (x / 1.5) ** 2 - math.log(1.5 * math.sqrt(2 * math.pi)),
        ),
        (HalfCauchyPrior(0.5), math.log(2 / (math.pi * 0.5 * (1 + (x / 0.5) ** 2)))),
        (
            BetaPrior(2.0, 3.0),
            math.log(x) + 2 * math.log(1 - x) - (math.lgamma(2) + math.lgamma(3) - math.lgamma(5)),
        ),
        (GammaPrior(2.0, 4.0), 2 * math.log(4.0) - math.lgamma(2.0) + math.log(x) - 4.0 * x),
    ]
    for p, expected in cases:
        # rtol: without torch, parameters (and some logarithms) are in single
        # precision, like GPyTorch's in torch's default dtype.
        assert_close(p.log_prob(x), expected, rtol=1e-7)
    with pytest.raises(ValueError):
        UniformPrior(0.25, 2.25).log_prob(3.0)
    with pytest.raises(ValueError):
        HalfNormalPrior(1.0).log_prob(-0.5)
    # BoTorch's BetaPrior does not validate its argument (torch's Beta: NaN).
    assert math.isnan(BetaPrior(2.0, 3.0).log_prob(1.5))


def test_lkj_priors() -> None:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # LKJ(eta = 1) is uniform over correlation matrices: for n = 3 the
    # density is 1 / (pi^2 / 2).
    assert_close(LKJPrior(3, 1.0).log_prob(eye), -math.log(math.pi**2 / 2), rtol=1e-12)
    cov = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.25]]
    sd = GammaPrior(2.0, 2.0)
    value = LKJCovariancePrior(3, 1.0, sd).log_prob(cov)
    # log p_LKJ(R) + sum_i log p(sd_i) (GPyTorch: log p_LKJ(R) + log p(sd_i)
    # for every i, a documented deviation).
    corr = -math.log(math.pi**2 / 2)
    assert_close(value, corr + sum(sd.log_prob(s) for s in (2.0, 1.0, 0.5)), rtol=1e-12)
    with pytest.raises(ValueError):
        LKJPrior(3, 1.0).log_prob(cov)
    with pytest.raises(ValueError):
        LKJCovariancePrior(3, 1.0, sd).log_prob([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])


def test_priors_enter_the_mll() -> None:
    X, Y = make_data(n=8, d=2, seed=6)
    priors = {
        "lengthscale_prior": NormalPrior(0.5, 0.5),
    }
    plain = SingleTaskGP(X, Y, covar_module=RBFKernel(ard_num_dims=2))
    with_prior = SingleTaskGP(X, Y, covar_module=RBFKernel(ard_num_dims=2, **priors))
    v0 = ExactMarginalLogLikelihood(plain.likelihood, plain).value_and_grad()[0]
    v1 = ExactMarginalLogLikelihood(with_prior.likelihood, with_prior).value_and_grad()[0]
    expected = 2 * NormalPrior(0.5, 0.5).log_prob(LN2) / len(X)
    assert_close(v1 - v0, expected, rtol=1e-10)
    # A prior on the task covariance (GPyTorch's IndexKernel(prior=...)).
    task = IndexKernel(
        num_tasks=2,
        rank=1,
        active_dims=[1],
        prior=LKJCovariancePrior(2, 2.0, LogNormalPrior(0.0, 1.0)),
    )
    Xt = [[x[0], float(i % 2)] for i, x in enumerate(X)]
    m = SingleTaskGP(Xt, Y, covar_module=RBFKernel(active_dims=[0]) * task)
    assert math.isfinite(ExactMarginalLogLikelihood(m.likelihood, m).value_and_grad()[0])


def test_less_than() -> None:
    c = LessThan(2.0)
    assert c.upper_bound == 2.0
    assert c.enforced
    assert_close(c.transform(0.0), 2.0 - LN2, rtol=1e-15)
    assert_close(c.inverse_transform(c.transform(0.3)), 0.3, rtol=1e-12)
    box = LessThan(2.0, transform=None)
    assert not box.enforced
    model = SingleTaskGP(*make_data(n=6, d=2), covar_module=LinearKernel(variance_constraint=box))
    bounds = model._rs.raw_parameter_bounds()
    assert tuple(bounds[-1]) == (-math.inf, 2.0)
    with pytest.raises(UnsupportedError):
        LessThan(2.0, transform="sigmoid")


# ---------------------------------------------------------------------------
# Errors (GPyTorch's and BoTorch's exception types)
# ---------------------------------------------------------------------------


def test_construction_errors() -> None:
    with pytest.raises(RuntimeError, match="nu expected"):
        MaternKernel(nu=1.0)
    with pytest.raises(RuntimeError, match="larger than the number of tasks"):
        IndexKernel(num_tasks=2, rank=3)
    with pytest.raises(RuntimeError, match="more than one power"):
        PolynomialKernel(power=[1, 2])
    with pytest.raises(ValueError, match="target_task_index"):
        PositiveIndexKernel(num_tasks=3, rank=1, target_task_index=3)
    with pytest.raises(UnsupportedError, match="dimension"):
        LinearTruncatedFidelityKernel(fidelity_dims=[0])
    with pytest.raises(UnsupportedError):
        LinearTruncatedFidelityKernel(fidelity_dims=[0, 1, 2], dimension=4)
    with pytest.raises(ValueError, match="nu must be"):
        LinearTruncatedFidelityKernel(fidelity_dims=[0], dimension=3, nu=1.0)
    with pytest.raises(AttributeError):
        OrthogonalAdditiveKernel(dim=2, coeffs_2_prior=GammaPrior(2.0, 1.0))
    with pytest.raises(UnsupportedError):  # GPyTorch ignores unknown arguments
        RBFKernel(foo=1)
    with pytest.raises(TypeError):
        RBFKernel(lengthscale_prior=object())
    with pytest.raises(UnsupportedError):
        AdditiveKernel()
    with pytest.raises(RuntimeError, match="do not broadcast"):
        _ = (RBFKernel(batch_shape=[2]) + RBFKernel(batch_shape=[3])).batch_shape
    with pytest.raises(UnsupportedError):
        RBFKernel()(X1, last_dim_is_batch=True)


def test_model_errors() -> None:
    X, Y = make_data(n=8, d=3, seed=7)
    with pytest.raises(RuntimeError):  # GPyTorch: a size mismatch of torch
        SingleTaskGP(X, Y, covar_module=RBFKernel(ard_num_dims=2))
    with pytest.raises(RuntimeError):  # GPyTorch: index out of range
        SingleTaskGP(X, Y, covar_module=RBFKernel(active_dims=[3]))
    with pytest.raises(RuntimeError):
        SingleTaskGP(X, Y, covar_module=ExponentialDecayKernel(ard_num_dims=2))
    with pytest.raises(RuntimeError):  # an index kernel needs one input
        SingleTaskGP(X, Y, covar_module=IndexKernel(num_tasks=2))
    tasks = [x[:2] + [float(i % 4)] for i, x in enumerate(X)]
    bad = SingleTaskGP(tasks, Y, covar_module=IndexKernel(num_tasks=3, active_dims=[2]))
    with pytest.raises(RuntimeError, match="task index"):  # when evaluated, as in GPyTorch
        ExactMarginalLogLikelihood(bad.likelihood, bad).value_and_grad()
    k = RBFKernel()
    with pytest.raises(UnsupportedError, match="twice"):
        SingleTaskGP(X, Y, covar_module=k + k)
    with pytest.raises(BotorchTensorDimensionError):
        SingleTaskGP(X, Y, covar_module=RBFKernel(batch_shape=[2]))
    outside = [[3.0 * v for v in x] for x in X]
    oak = SingleTaskGP(outside, Y, covar_module=OrthogonalAdditiveKernel(dim=3))
    with pytest.raises(ValueError, match="hypercube"):  # BoTorch's check_hypercube
        ExactMarginalLogLikelihood(oak.likelihood, oak).value_and_grad()
    model = SingleTaskGP(
        outside, Y, covar_module=OrthogonalAdditiveKernel(dim=3, check_hypercube=False)
    )
    assert math.isfinite(ExactMarginalLogLikelihood(model.likelihood, model).value_and_grad()[0])
    shared = SingleTaskGP(X, Y, covar_module=_interval_kernel())
    assert shape(shared.posterior(X[:2]).mean) == (2, 1)


def _interval_kernel() -> RBFKernel:
    """An RBF kernel with an interval-constrained lengthscale."""
    return RBFKernel(lengthscale_constraint=Interval(0.05, 4.0))
