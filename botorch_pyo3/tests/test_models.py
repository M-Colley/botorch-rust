"""SingleTaskGP, its modules, transforms and posteriors."""

from __future__ import annotations

import copy
import math
import warnings

import pytest

from botorch_rs.constraints import GreaterThan, Positive
from botorch_rs.exceptions import (
    BotorchRsError,
    BotorchTensorDimensionError,
    InputDataError,
    InputDataWarning,
    UnsupportedError,
)
from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms import Normalize, Standardize
from botorch_rs.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
)
from botorch_rs.priors import GammaPrior, LogNormalPrior
from botorch_rs.utils.types import DEFAULT
from helpers import as_kind, assert_close, check_kind, flatten, make_data, shape, to_list


@pytest.fixture
def data() -> tuple[list, list]:
    return make_data(n=12, d=2, seed=0)


def test_default_model(kind: str, data: tuple[list, list]) -> None:
    X, Y = data
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    assert model.num_outputs == 1 and model.batch_shape == ()
    (train_x,) = model.train_inputs
    check_kind(train_x, kind)
    assert_close(train_x, X, rtol=0, atol=0)
    targets = model.train_targets
    check_kind(targets, kind)
    assert shape(targets) == (12,)
    t = flatten(targets)
    assert abs(sum(t) / len(t)) < 1e-12  # standardized
    # BoTorch's default modules and their initial values.
    assert isinstance(model.likelihood, GaussianLikelihood)
    assert isinstance(model.covar_module, RBFKernel)
    assert isinstance(model.mean_module, ConstantMean)
    noise = model.likelihood.noise
    check_kind(noise, kind)
    assert shape(noise) == (1,)
    assert_close(noise, [math.exp(-4.0 - 1.0)], rtol=1e-6)  # LogNormal(-4, 1) mode
    ls = model.covar_module.lengthscale
    assert shape(ls) == (1, 2)
    prior_mode = math.exp(math.sqrt(2) + math.log(2) / 2 - 3.0)
    assert_close(ls, [[prior_mode, prior_mode]], rtol=1e-6)
    assert float(model.mean_module.constant) == 0.0
    assert isinstance(model.outcome_transform, Standardize)
    assert not hasattr(model, "input_transform")
    names = [name for name, _ in model.named_parameters()]
    assert names == [
        "likelihood.noise_covar.raw_noise",
        "mean_module.raw_constant",
        "covar_module.raw_lengthscale",
    ]
    assert [shape(v) for v in model.parameters()] == [(1,), (), (1, 2)]


def test_hyperparameter_setters(kind: str, data: tuple[list, list]) -> None:
    X, Y = data
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    model.covar_module.lengthscale = as_kind([[0.3, 0.7]], kind)
    model.likelihood.noise = 0.01
    model.mean_module.constant = 0.25
    assert_close(model.covar_module.lengthscale, [[0.3, 0.7]], rtol=1e-15)
    assert_close(model.likelihood.noise, [0.01], rtol=1e-15)
    assert_close(model.mean_module.constant, 0.25, rtol=1e-15)
    model.covar_module.lengthscale = 0.5  # broadcast
    assert_close(model.covar_module.lengthscale, [[0.5, 0.5]], rtol=1e-15)
    model.initialize(
        **{
            "likelihood.noise_covar.raw_noise": as_kind([0.02], kind),
            "covar_module.raw_lengthscale": as_kind([[0.4, 0.6]], kind),
            "mean_module.raw_constant": 0.1,
        }
    )
    params = {name: to_list(v) for name, v in model.named_parameters()}
    assert_close(params["likelihood.noise_covar.raw_noise"], [0.02], rtol=1e-15)
    assert_close(params["covar_module.raw_lengthscale"], [[0.4, 0.6]], rtol=1e-15)
    with pytest.raises(AttributeError):
        model.initialize(**{"covar_module.unknown": 1.0})
    with pytest.raises(BotorchRsError):  # lengthscales must be positive
        model.covar_module.lengthscale = [[-1.0, 1.0]]


def test_size_one_values_for_scalar_hyperparameters(kind: str, data: tuple[list, list]) -> None:
    """Like GPyTorch's ``Module.initialize`` (``expand_as``, else
    ``view_as``), a parameter can be set from a value of another shape with
    as many elements, e.g. a 0-dim outputscale from a tensor of shape
    ``(1,)``."""
    X, Y = data
    model = SingleTaskGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)),
    )
    model.covar_module.outputscale = as_kind([1.7], kind)
    model.likelihood.noise = as_kind([[0.03]], kind)
    model.mean_module.constant = as_kind([-0.2], kind)
    model.covar_module.base_kernel.lengthscale = as_kind([[0.4], [0.6]], kind)
    assert shape(model.covar_module.outputscale) == ()
    assert_close(model.covar_module.outputscale, 1.7, rtol=1e-15)
    assert shape(model.likelihood.noise) == (1,)
    assert_close(model.likelihood.noise, [0.03], rtol=1e-15)
    assert shape(model.mean_module.constant) == ()
    assert_close(model.mean_module.constant, -0.2, rtol=1e-15)
    assert_close(model.covar_module.base_kernel.lengthscale, [[0.4, 0.6]], rtol=1e-15)
    model.initialize(**{"covar_module.raw_outputscale": as_kind([0.5], kind)})
    params = {name: to_list(v) for name, v in model.named_parameters()}
    assert_close(params["covar_module.raw_outputscale"], 0.5, rtol=1e-15)
    with pytest.raises(BotorchRsError):
        model.covar_module.outputscale = as_kind([1.0, 2.0], kind)
    with pytest.raises(BotorchRsError):
        model.covar_module.base_kernel.lengthscale = as_kind([0.1, 0.2, 0.3], kind)
    with pytest.raises(BotorchRsError):
        model.initialize(**{"covar_module.raw_outputscale": as_kind([0.5, 0.1], kind)})


def test_transforms_are_fitted(kind: str) -> None:
    X, Y = make_data(n=10, d=3, seed=1)
    Xs = [[10.0 + 3.0 * v for v in x] for x in X]
    nz = Normalize(d=3)
    st = Standardize(m=1)
    model = SingleTaskGP(
        as_kind(Xs, kind), as_kind(Y, kind), input_transform=nz, outcome_transform=st
    )
    assert model.input_transform is nz and model.outcome_transform is st
    lower = [min(x[j] for x in Xs) for j in range(3)]
    upper = [max(x[j] for x in Xs) for j in range(3)]
    check_kind(nz.bounds, kind)
    assert_close(nz.bounds, [lower, upper], rtol=1e-15)
    assert shape(nz.offset) == (1, 3) and shape(nz.coefficient) == (1, 3)
    y = [v[0] for v in Y]
    mean = sum(y) / len(y)
    std = math.sqrt(sum((v - mean) ** 2 for v in y) / (len(y) - 1))
    assert_close(st.means, [[mean]], rtol=1e-14)
    assert_close(st.stdvs, [[std]], rtol=1e-14)
    tx = model.input_transform.transform(as_kind(Xs, kind))
    assert min(flatten(tx)) == pytest.approx(0.0, abs=1e-15)
    assert max(flatten(tx)) == pytest.approx(1.0, abs=1e-15)
    assert_close(nz.untransform(tx), Xs, rtol=1e-14)
    yt, yv = st.untransform(as_kind([[0.0], [1.0]], kind))
    assert yv is None
    assert_close(yt, [[mean], [mean + std]], rtol=1e-14)


def test_standalone_transforms(kind: str) -> None:
    X = [[1.0, 5.0], [3.0, 5.0], [2.0, 6.0]]
    nz = Normalize(d=2)
    out = nz(as_kind(X, kind))  # train mode learns the bounds
    check_kind(out, kind)
    assert_close(out, [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], rtol=1e-15)
    nz.eval()
    assert_close(nz(as_kind([[2.0, 5.5]], kind)), [[0.5, 0.5]], rtol=1e-15)
    fixed = Normalize(d=2, bounds=as_kind([[0.0, 0.0], [2.0, 4.0]], kind))
    assert not fixed.learn_bounds
    assert_close(fixed(as_kind([[1.0, 1.0]], kind)), [[0.5, 0.25]], rtol=1e-15)
    st = Standardize(m=2)
    Y = as_kind([[1.0, 10.0], [3.0, 30.0]], kind)
    Yt, Yvt = st(Y, as_kind([[2.0, 2.0], [2.0, 2.0]], kind))
    assert_close(Yt, [[-math.sqrt(0.5), -math.sqrt(0.5)], [math.sqrt(0.5), math.sqrt(0.5)]])
    assert_close(Yvt, [[1.0, 0.01], [1.0, 0.01]])
    # Centering and subsets of the dimensions (phase 6b).
    centered = Normalize(d=2, center=0.0)
    assert_close(centered(as_kind(X, kind)), [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]], rtol=1e-15)
    sub = Normalize(d=3, indices=[0, 1])
    X3 = [[1.0, 5.0, 7.0], [3.0, 5.0, 8.0], [2.0, 6.0, 9.0]]
    assert_close(
        sub(as_kind(X3, kind)), [[0.0, 0.0, 7.0], [1.0, 0.0, 8.0], [0.5, 1.0, 9.0]], rtol=1e-15
    )
    # (Normalize over a subset of the inputs is supported since phase 7a.)
    assert list(Normalize(d=3, indices=[0, 1]).indices) == [0, 1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Normalize(d=2, learn_bounds=False)
    assert any("no-op" in str(x.message) for x in w)


def test_posterior_shapes(kind: str, data: tuple[list, list]) -> None:
    X, Y = data
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    Xt = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.3]]
    post = model.posterior(as_kind(Xt, kind))
    check_kind(post.mean, kind)
    assert shape(post.mean) == (3, 1) and shape(post.variance) == (3, 1)
    assert all(v > 0 for v in flatten(post.variance))
    cov = post.distribution.covariance_matrix
    assert shape(cov) == (3, 3)
    assert_close([cov_row[i] for i, cov_row in enumerate(to_list(cov))], flatten(post.variance))
    assert shape(post.distribution.mean) == (3,)
    assert_close([v * v for v in flatten(post.distribution.stddev)], flatten(post.variance))
    batched = model.posterior(as_kind([[[0.1, 0.2]], [[0.5, 0.5]]], kind))
    assert shape(batched.mean) == (2, 1, 1)
    assert shape(batched.distribution.covariance_matrix) == (2, 1, 1)
    deep = model.posterior(as_kind([[[[0.1, 0.2], [0.3, 0.4]]] * 3] * 2, kind))
    assert shape(deep.mean) == (2, 3, 2, 1)
    # Joint and marginal posteriors agree; observation noise adds the noise.
    assert_close(flatten(batched.mean), flatten(post.mean)[:2])
    noisy = model.posterior(as_kind(Xt, kind), observation_noise=True)
    noise = flatten(model.likelihood.noise)[0] * flatten(model.outcome_transform.stdvs)[0] ** 2
    assert_close(flatten(noisy.variance), [v + noise for v in flatten(post.variance)])
    assert_close(flatten(noisy.mean), flatten(post.mean))
    with pytest.raises(BotorchTensorDimensionError):
        model.posterior(as_kind([0.1, 0.2], kind))
    with pytest.raises(UnsupportedError):
        model.posterior(as_kind(Xt, kind), posterior_transform=object())


def test_posterior_sampling(kind: str, data: tuple[list, list]) -> None:
    X, Y = data
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    post = model.posterior(as_kind([[[0.1, 0.2], [0.6, 0.4]], [[0.3, 0.3], [0.8, 0.1]]], kind))
    samples = post.rsample(sample_shape=[5])
    check_kind(samples, kind)
    assert shape(samples) == (5, 2, 2, 1)
    zero = post.rsample_from_base_samples([3], as_kind([[[0.0, 0.0]]] * 3, kind))
    assert shape(zero) == (3, 2, 2, 1)
    for s in to_list(zero):
        assert_close(s, to_list(post.mean))
    # Explicit base samples z give mean + L z with L the Cholesky factor.
    z = [[[1.0, -0.5]]]
    one = to_list(post.rsample_from_base_samples([1], as_kind(z, kind)))[0]
    cov = to_list(post.distribution.covariance_matrix)
    mean = to_list(post.mean)
    for b in range(2):
        c = cov[b]
        l00 = math.sqrt(c[0][0])
        l10 = c[1][0] / l00
        l11 = math.sqrt(c[1][1] - l10 * l10)
        expected = [mean[b][0][0] + l00 * 1.0, mean[b][1][0] + l10 * 1.0 - 0.5 * l11]
        assert_close([one[b][0][0], one[b][1][0]], expected, rtol=1e-7)
    many = flatten(model.posterior(as_kind([[0.5, 0.5]], kind)).rsample([4000]))
    m = flatten(model.posterior(as_kind([[0.5, 0.5]], kind)).mean)[0]
    v = flatten(model.posterior(as_kind([[0.5, 0.5]], kind)).variance)[0]
    assert abs(sum(many) / len(many) - m) < 5 * math.sqrt(v / len(many))


def test_fixed_noise_model(kind: str, data: tuple[list, list]) -> None:
    X, Y = data
    Yvar = [[0.01 * (1 + i % 3)] for i in range(len(Y))]
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), as_kind(Yvar, kind))
    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    noise = model.likelihood.noise
    assert shape(noise) == (12,)
    std = flatten(model.outcome_transform.stdvs)[0]
    assert_close(noise, [v[0] / std**2 for v in Yvar])
    assert [n for n, _ in model.named_parameters()] == [
        "mean_module.raw_constant",
        "covar_module.raw_lengthscale",
    ]
    with pytest.raises(UnsupportedError):
        FixedNoiseGaussianLikelihood(noise=Yvar)
    with pytest.raises(InputDataError):
        SingleTaskGP(X, Y, [[-1.0]] * len(Y))


def test_custom_modules(kind: str, data: tuple[list, list]) -> None:
    X, Y = data
    legacy = SingleTaskGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        covar_module=get_matern_kernel_with_gamma_prior(ard_num_dims=2),
        likelihood=get_gaussian_likelihood_with_gamma_prior(),
    )
    assert isinstance(legacy.covar_module, ScaleKernel)
    assert isinstance(legacy.covar_module.base_kernel, MaternKernel)
    assert shape(legacy.covar_module.outputscale) == ()
    assert shape(legacy.covar_module.base_kernel.lengthscale) == (1, 2)
    assert [n for n, _ in legacy.named_parameters()] == [
        "likelihood.noise_covar.raw_noise",
        "mean_module.raw_constant",
        "covar_module.raw_outputscale",
        "covar_module.base_kernel.raw_lengthscale",
    ]
    legacy.covar_module.outputscale = 2.0
    assert_close(legacy.covar_module.outputscale, 2.0, rtol=1e-14)
    custom = ScaleKernel(
        RBFKernel(ard_num_dims=2, lengthscale_prior=GammaPrior(3.0, 6.0)),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    model = SingleTaskGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        covar_module=custom,
        likelihood=GaussianLikelihood(noise_prior=LogNormalPrior(-4.0, 1.0)),
        mean_module=ConstantMean(),
        outcome_transform=None,
    )
    assert model.covar_module is custom
    assert not hasattr(model, "outcome_transform")
    # GPyTorch defaults: softplus constraints with raw values 0.
    assert_close(custom.base_kernel.lengthscale, [[math.log(2.0)] * 2], rtol=1e-14)
    assert_close(custom.outputscale, math.log(2.0), rtol=1e-14)
    matern = SingleTaskGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        covar_module=get_covar_module_with_dim_scaled_prior(2, use_rbf_kernel=False),
    )
    assert isinstance(matern.covar_module, MaternKernel)
    with pytest.raises(RuntimeError):  # as GPyTorch
        MaternKernel(nu=1.0)
    SingleTaskGP(X, Y, covar_module=RBFKernel())  # a shared lengthscale (kernel tree)
    with pytest.raises(RuntimeError):  # GPyTorch: a RuntimeError of torch
        SingleTaskGP(X, Y, covar_module=RBFKernel(ard_num_dims=3))
    nested = ScaleKernel(custom)  # ScaleKernel around any kernel, as in GPyTorch
    with pytest.raises(UnsupportedError):
        SingleTaskGP(X, Y, covar_module=custom)  # already part of a model
    with pytest.raises(UnsupportedError):
        SingleTaskGP(X, Y, covar_module=nested)  # contains a kernel of another model
    c = GreaterThan(1e-3, transform=None, initial_value=0.5)
    assert not c.enforced and c.initial_value == pytest.approx(0.5)
    assert Positive().enforced
    assert GammaPrior(2.0, 3.0).mode == pytest.approx(1.0 / 3.0)


def test_invalid_inputs(data: tuple[list, list]) -> None:
    X, Y = data
    with pytest.raises(BotorchTensorDimensionError, match="explicit output dimension"):
        SingleTaskGP(X, [y[0] for y in Y])
    with pytest.raises(BotorchTensorDimensionError):  # batch shapes differ
        SingleTaskGP([X], [Y, Y])
    with pytest.raises(BotorchTensorDimensionError):
        SingleTaskGP(X, Y[:-1])
    with pytest.raises(InputDataError):
        SingleTaskGP([[float("nan"), 0.0]] + X[1:], Y)
    with pytest.raises(UnsupportedError):
        SingleTaskGP(X, Y, mean_module=object())
    with pytest.raises(UnsupportedError):
        SingleTaskGP(X, Y, likelihood=object())
    with pytest.raises(UnsupportedError):
        SingleTaskGP(X, Y, outcome_transform=object())
    with pytest.raises(ValueError):  # all errors are also builtin exceptions
        SingleTaskGP(X, Y[:-1])


def test_input_scaling_warnings(data: tuple[list, list]) -> None:
    X, Y = data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SingleTaskGP(X, Y)
        SingleTaskGP(X, Y, outcome_transform=DEFAULT)
    assert not [x for x in w if issubclass(x.category, InputDataWarning)]
    with pytest.warns(InputDataWarning, match="unit cube"):
        SingleTaskGP([[5.0 * v for v in x] for x in X], Y)
    with pytest.warns(InputDataWarning, match="not standardized"):
        SingleTaskGP(X, [[10.0 + y[0]] for y in Y], outcome_transform=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SingleTaskGP([[5.0 * v for v in x] for x in X], Y, covar_module=RBFKernel(ard_num_dims=2))
    assert not [x for x in w if issubclass(x.category, InputDataWarning)]


def test_deepcopy(data: tuple[list, list]) -> None:
    X, Y = data
    model = SingleTaskGP(X, Y, input_transform=Normalize(d=2))
    before = flatten(model.posterior([[0.3, 0.3]]).mean)
    clone = copy.deepcopy(model)
    assert_close(clone.posterior([[0.3, 0.3]]).mean, before, rtol=0, atol=0)
    clone.covar_module.lengthscale = 0.123
    assert flatten(model.covar_module.lengthscale) != flatten(clone.covar_module.lengthscale)
    assert clone.covar_module._model is clone and model.covar_module._model is model
    assert clone.input_transform is not model.input_transform
    assert flatten(model.posterior([[0.3, 0.3]]).mean) == before
    assert flatten(clone.posterior([[0.3, 0.3]]).mean) != before
