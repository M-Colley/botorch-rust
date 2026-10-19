"""API and error tests of phase 10c (approximate GPs): ``SingleTaskVariationalGP``,
``ApproximateGPyTorchModel``, ``VariationalELBO``, the variational modules and
the inducing point allocators, for every array type (without torch and
BoTorch too). Comparisons with BoTorch are in
``test_parity_10c_approximate.py``.
"""

from __future__ import annotations

import copy
import math
import warnings

import pytest

from botorch_rs.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import UserInputWarning
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from botorch_rs.means import ConstantMean, ZeroMean
from botorch_rs.mlls import ExactMarginalLogLikelihood, VariationalELBO
from botorch_rs.models import ApproximateGPyTorchModel, SingleTaskGP, SingleTaskVariationalGP
from botorch_rs.models.transforms import Normalize, Standardize
from botorch_rs.models.transforms.outcome import Log
from botorch_rs.models.utils.inducing_point_allocators import (
    ExpectedImprovementQualityFunction,
    GreedyImprovementReduction,
    GreedyVarianceReduction,
    InducingPointAllocator,
    QualityFunction,
    UnitQualityFunction,
    _pivoted_cholesky_init,
)
from botorch_rs.optim.utils import get_parameters_and_bounds
from botorch_rs.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from helpers import KINDS, as_kind, check_kind, make_data, shape, to_list

pytestmark = pytest.mark.filterwarnings("ignore::botorch_rs.exceptions.warnings.InputDataWarning")


def _data(kind: str, m: int = 1, n: int = 12) -> tuple[object, object]:
    X, Y = make_data(n=n, d=2, seed=3)
    Y2 = [[y[0], math.cos(3 * x[0])] for x, y in zip(X, Y)][:n]
    return as_kind(X, kind), as_kind([row[:m] for row in Y2], kind)


@pytest.mark.parametrize("kind", KINDS)
def test_array_kinds_posterior_and_elbo(kind: str) -> None:
    X, Y = _data(kind)
    model = SingleTaskVariationalGP(X, Y, inducing_points=4)
    vs = model.model.variational_strategy
    assert isinstance(vs, VariationalStrategy)
    assert isinstance(vs._variational_distribution, CholeskyVariationalDistribution)
    assert shape(vs.inducing_points) == (4, 2)
    check_kind(vs.inducing_points, kind)
    assert int(vs.variational_params_initialized) == 0
    post = model.posterior(as_kind([[0.2, 0.3], [0.7, 0.1]], kind))
    assert int(vs.variational_params_initialized) == 1
    assert shape(post.mean) == (2, 1)
    check_kind(post.mean, kind)
    assert all(v > 0 for v in to_list(post.variance)[0])
    noisy = model.posterior(as_kind([[0.2, 0.3]], kind), observation_noise=True)
    assert to_list(noisy.variance)[0][0] > to_list(model.posterior(as_kind([[0.2, 0.3]], kind)).variance)[0][0]
    mll = VariationalELBO(model.likelihood, model.model, num_data=12)
    value = mll()
    check_kind(value, kind)
    assert shape(value) == ()
    terms = VariationalELBO(model.likelihood, model.model, num_data=12, combine_terms=False)()
    assert len(terms) == 3
    assert abs(float(to_list(terms[0])) - float(to_list(terms[1])) + float(to_list(terms[2])) - float(to_list(value))) < 1e-12
    v, g = mll.value_and_grad()
    assert len(g) == sum(size for _, size in model._rs.parameter_names())
    out = model.model(model.model.train_inputs[0])
    assert shape(out.mean) == (12,)
    assert shape(out.covariance_matrix) == (12, 12)


@pytest.mark.parametrize("kind", KINDS)
def test_fit_and_acquisition(kind: str) -> None:
    X, Y = _data(kind)
    model = SingleTaskVariationalGP(X, Y, inducing_points=4)
    mll = VariationalELBO(model.likelihood, model.model, num_data=12)
    before = float(to_list(mll()))
    fit_gpytorch_mll(mll)
    after = float(to_list(mll()))
    assert after > before
    assert mll._fit_result["status"] in ("SUCCESS", "STOPPED")
    assert not mll.training
    Xq = as_kind([[[0.3, 0.4]], [[0.8, 0.2]]], kind)
    best = max(row[0] for row in to_list(Y))
    ei = ExpectedImprovement(model, best_f=best)
    assert shape(ei(Xq)) == (2,)
    qei = qExpectedImprovement(model, best_f=best)
    vals = qei(as_kind([[[0.3, 0.4], [0.5, 0.5]]], kind))
    assert shape(vals) == (1,)


@pytest.mark.parametrize("kind", KINDS)
def test_multi_output_and_transforms(kind: str) -> None:
    X, Y = _data(kind, m=2)
    Xs = as_kind([[2 * v + 1 for v in row] for row in to_list(X)], kind)
    Ys = as_kind([[3 * v - 1 for v in row] for row in to_list(Y)], kind)
    with pytest.warns(UserInputWarning, match="Using an outcome transform"):
        model = SingleTaskVariationalGP(
            Xs, Ys, inducing_points=5, input_transform=Normalize(d=2), outcome_transform=Standardize(m=2)
        )
    assert model.num_outputs == 2
    assert isinstance(model.model.variational_strategy, IndependentMultitaskVariationalStrategy)
    assert isinstance(model.likelihood, MultitaskGaussianLikelihood)
    names = [n for n, _ in VariationalELBO(model.likelihood, model.model, num_data=12).named_parameters()]
    assert names[:2] == ["likelihood.raw_task_noises", "likelihood.raw_noise"]
    assert names[2] == "model.variational_strategy.base_variational_strategy.inducing_points"
    post = model.posterior(as_kind([[1.2, 2.1], [2.5, 1.5]], kind))
    assert shape(post.mean) == (2, 2)
    assert shape(post.distribution.covariance_matrix) == (4, 4)
    # Inducing points live in the transformed (unit cube) space.
    z = to_list(model.model.variational_strategy.base_variational_strategy.inducing_points)
    assert all(-1e-12 <= v <= 1 + 1e-12 for row in z for v in row)
    params, bounds = get_parameters_and_bounds(VariationalELBO(model.likelihood, model.model, 12))
    assert "model.covar_module.raw_lengthscale" in bounds
    assert shape(params["model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar"]) == (2, 5, 5)


def test_inducing_point_options() -> None:
    X, Y = _data("list", n=12)
    assert shape(SingleTaskVariationalGP(X, Y).model.variational_strategy.inducing_points) == (3, 2)
    Z = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]]
    model = SingleTaskVariationalGP(X, Y, inducing_points=Z, learn_inducing_points=False)
    assert to_list(model.model.variational_strategy.inducing_points) == Z
    names = [n for n, _ in model.named_parameters()]
    assert not any("inducing_points" in n for n in names)
    # Duplicated inputs: the greedy selection stops early.
    Xd = X[:3] * 4
    m2 = SingleTaskVariationalGP(Xd, Y, inducing_points=6)
    assert shape(m2.model.variational_strategy.inducing_points)[0] == 3
    with pytest.warns(UserWarning, match="set `learn_inducing_points` to False"):
        SingleTaskVariationalGP(X, Y, inducing_point_allocator=GreedyVarianceReduction())
    model = SingleTaskVariationalGP(X, Y, inducing_points=4)
    z = model.init_inducing_points(X[::-1])
    assert shape(z) == (4, 2)
    assert int(model.model.variational_strategy.variational_params_initialized) == 0


def test_custom_modules() -> None:
    X, Y = _data("list")
    model = SingleTaskVariationalGP(
        X,
        Y,
        covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2)),
        mean_module=ZeroMean(),
        likelihood=GaussianLikelihood(),
        variational_distribution=CholeskyVariationalDistribution(3, mean_init_std=0.0),
        inducing_points=3,
    )
    assert model._is_custom_likelihood
    names = [n for n, _ in model.named_parameters()]
    assert "model.covar_module.raw_outputscale" in names
    assert not any("mean_module" in n for n in names)
    model.posterior([[0.2, 0.2]])
    assert to_list(model.model.variational_strategy._variational_distribution.variational_mean) == [0.0] * 3
    model.model.covar_module.outputscale = 2.0
    assert abs(to_list(model.model.covar_module.outputscale) - 2.0) < 1e-12
    model.likelihood.noise = 0.05
    assert abs(to_list(model.likelihood.noise)[0] - 0.05) < 1e-12
    mean = ConstantMean()
    m2 = SingleTaskVariationalGP(X, Y, mean_module=mean, inducing_points=3)
    mean.constant = 0.7
    assert to_list(m2.model.mean_module.constant) == 0.7


def test_copy_and_approximate_model() -> None:
    X, Y = _data("list")
    model = SingleTaskVariationalGP(X, Y, inducing_points=3)
    model.posterior([[0.5, 0.5]])
    clone = copy.deepcopy(model)
    clone.likelihood.noise = 0.2
    assert to_list(model.likelihood.noise)[0] != pytest.approx(0.2)
    approx = ApproximateGPyTorchModel(train_X=X, inducing_points=3)
    assert isinstance(approx.likelihood, GaussianLikelihood)
    assert shape(approx.posterior([[0.3, 0.3]]).mean) == (1, 1)
    with pytest.raises(AttributeError):
        approx.model.train_targets  # noqa: B018


def test_allocators_api() -> None:
    X, Y = _data("list")
    assert to_list(UnitQualityFunction()(X)) == [1.0] * 12
    gp = SingleTaskGP(X, Y)
    q = ExpectedImprovementQualityFunction(gp, maximize=True)(X)
    assert shape(q) == (12,) and all(v >= 0 for v in to_list(q))
    assert issubclass(GreedyImprovementReduction, InducingPointAllocator)
    assert issubclass(UnitQualityFunction, QualityFunction)
    alloc = GreedyImprovementReduction(gp, maximize=False)
    z = alloc.allocate_inducing_points(X, RBFKernel(), 4, ())
    assert shape(z) == (4, 2)
    with pytest.raises(TypeError):
        InducingPointAllocator()  # type: ignore[abstract]
    K = RBFKernel()(X)
    assert len(to_list(_pivoted_cholesky_init(X, K, 0, [1.0] * 12))) == 0


def test_errors() -> None:
    X, Y = _data("list")
    model = SingleTaskVariationalGP(X, Y, inducing_points=3)
    with pytest.raises(NotImplementedError, match="does not support output indices"):
        model.posterior([[0.1, 0.1]], output_indices=[0])
    with pytest.raises(AttributeError, match="get_fantasy_model"):
        model.condition_on_observations([[0.1, 0.1]], [[1.0]])
    with pytest.raises(AttributeError):
        model.fantasize  # noqa: B018
    with pytest.raises(BotorchTensorDimensionError):
        model.posterior([[0.1, 0.1, 0.3]])
    with pytest.raises(UnsupportedError, match="Standardize"):
        SingleTaskVariationalGP(X, [[abs(y[0]) + 1] for y in Y], outcome_transform=Log())
    with pytest.raises(UnsupportedError, match="MultitaskGaussianLikelihood"):
        SingleTaskVariationalGP(X, [[y[0], y[0]] for y in Y], likelihood=GaussianLikelihood())
    with pytest.raises(UnsupportedError, match="unbatched"):
        SingleTaskVariationalGP([X, X], [Y, Y])
    with pytest.raises(UnsupportedError, match="VariationalStrategy"):
        SingleTaskVariationalGP(X, Y, variational_strategy=IndependentMultitaskVariationalStrategy)
    with pytest.raises(UnsupportedError):
        VariationalELBO(model.likelihood, model, num_data=12)
    with pytest.raises(UnsupportedError):
        ExactMarginalLogLikelihood(model.likelihood, model)
    with pytest.raises(ValueError, match="requires a quality score"):
        _pivoted_cholesky_init(X, RBFKernel()(X), 3, [1.0] * 5)
    with pytest.raises(NotImplementedError, match="Multi-output"):
        ExpectedImprovementQualityFunction(SingleTaskGP(X, [[y[0], y[0]] for y in Y]), True)
    with pytest.raises(RuntimeError, match="inducing points"):
        SingleTaskVariationalGP(
            X, Y, inducing_points=3, variational_distribution=CholeskyVariationalDistribution(4)
        )
    no_targets = SingleTaskVariationalGP(X, inducing_points=3)
    with pytest.raises(AttributeError):
        fit_gpytorch_mll(VariationalELBO(no_targets.likelihood, no_targets.model, num_data=12))
    mll = VariationalELBO(model.likelihood, model.model, num_data=12)
    with pytest.raises(UnsupportedError, match="data_loader"):
        fit_gpytorch_mll(mll, data_loader=object())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(UnsupportedError, match="closures"):
            fit_gpytorch_mll(mll, closure=lambda: None)
