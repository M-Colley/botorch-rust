"""``MultiTaskGP``, ``KroneckerMultiTaskGP`` and their modules through the
API, for every array type (BoTorch is not needed; see
``test_parity_7a_multitask.py`` for the comparisons with BoTorch)."""

from __future__ import annotations

import copy
import math
import random
from types import SimpleNamespace

import pytest

from botorch_rs.acquisition import qExpectedImprovement, qLogNoisyExpectedImprovement
from botorch_rs.acquisition.objective import LinearMCObjective, ScalarizedPosteriorTransform
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, DeprecationError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import MaternKernel, MultitaskKernel, RBFKernel, ScaleKernel
from botorch_rs.likelihoods import (
    GaussianLikelihood,
    HadamardGaussianLikelihood,
    MultitaskGaussianLikelihood,
)
from botorch_rs.means import ConstantMean
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import KroneckerMultiTaskGP, ModelListGP, MultiTaskGP, SingleTaskGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.posteriors import MultitaskGPPosterior
from botorch_rs.priors import GammaPrior, LKJCovariancePrior, SmoothedBoxPrior
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import KINDS, as_kind, check_kind, flatten, shape


def long_data(
    tasks: tuple[int, ...] = (0, 1, 2), n_per: int = 4, d: int = 2, tf: int = -1, seed: int = 3
) -> tuple[list, list, list]:
    """Long-format data: rows of every task (task column ``tf``), targets
    and noise variances."""
    rng = random.Random(seed)
    X, Y, Yvar = [], [], []
    for t in tasks:
        for _ in range(n_per):
            x = [rng.random() for _ in range(d)]
            Y.append([math.sin(3 * x[0]) + sum(v * v for v in x[1:]) + 0.2 * t])
            Yvar.append([0.01 + 0.01 * rng.random()])
            row = list(x)
            row.insert(tf % (d + 1), float(t))
            X.append(row)
    return X, Y, Yvar


def block_data(n: int = 8, d: int = 2, m: int = 2, seed: int = 5) -> tuple[list, list]:
    rng = random.Random(seed)
    X = [[rng.random() for _ in range(d)] for _ in range(n)]
    Y = [
        [math.sin((3 + t) * x[0]) + t * x[-1] + 0.05 * rng.gauss(0, 1) for t in range(m)] for x in X
    ]
    return X, Y


def points(q: int, d: int, seed: int = 11) -> list:
    rng = random.Random(seed)
    return [[rng.random() for _ in range(d)] for _ in range(q)]


# ---------------------------------------------------------------------------
# MultiTaskGP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_multitask_gp_api(kind: str) -> None:
    X, Y, _ = long_data()
    model = MultiTaskGP(as_kind(X, kind), as_kind(Y, kind), task_feature=-1)
    assert model.num_tasks == 3 and model.all_tasks == [0, 1, 2]
    assert model.num_outputs == 3 and model._output_tasks == [0, 1, 2]
    assert model._task_feature == 2 and model.num_non_task_features == 2
    assert isinstance(model.likelihood, HadamardGaussianLikelihood)
    names = [n for n, _ in model.named_parameters()]
    assert names == [
        "likelihood.noise_covar.raw_noise",
        "mean_module.base_means.0.raw_constant",
        "mean_module.base_means.1.raw_constant",
        "mean_module.base_means.2.raw_constant",
        "covar_module.kernels.0.raw_lengthscale",
        "covar_module.kernels.1.raw_var",
        "covar_module.kernels.1.raw_covar_factor",
    ]
    shapes = {n: shape(p) for n, p in model.named_parameters()}
    assert shapes["likelihood.noise_covar.raw_noise"] == (3,)
    assert shapes["covar_module.kernels.1.raw_covar_factor"] == (3, 3)
    assert shapes["covar_module.kernels.0.raw_lengthscale"] == (1, 2)
    for _, p in model.named_parameters():
        check_kind(p, kind)
    check_kind(model.train_inputs[0], kind)
    assert shape(model.train_inputs[0]) == (12, 3) and shape(model.train_targets) == (12,)
    # Posteriors: over the output tasks without the task column, single-output
    # with it, batched.
    Xt = as_kind(points(4, 2), kind)
    post = model.posterior(Xt)
    assert shape(post.mean) == (4, 3) and shape(post.variance) == (4, 3)
    check_kind(post.mean, kind)
    assert shape(post.distribution.covariance_matrix) == (12, 12)
    noisy = model.posterior(Xt, observation_noise=True)
    assert all(a > b for a, b in zip(flatten(noisy.variance), flatten(post.variance), strict=True))
    with_tasks = [[*x, float(t)] for x, t in zip(points(4, 2), (0, 2, 1, 2), strict=True)]
    p1 = model.posterior(as_kind(with_tasks, kind))
    assert shape(p1.mean) == (4, 1)
    # The single-output posterior at (x, t) is the multi-output one's task t.
    cols = [flatten(post.mean)[i * 3 + t] for i, t in enumerate((0, 2, 1, 2))]
    assert max(abs(a - b) for a, b in zip(flatten(p1.mean), cols, strict=True)) < 1e-12
    batched = model.posterior(as_kind([points(2, 2, 1), points(2, 2, 2)], kind))
    assert shape(batched.mean) == (2, 2, 3)
    sub = model.posterior(Xt, output_indices=[2, 0])
    assert shape(sub.mean) == (4, 2)
    assert abs(flatten(sub.mean)[0] - flatten(post.mean)[2]) < 1e-12
    samples = post.rsample_from_base_samples([5], as_kind([[[0.1] * 3] * 4] * 5, kind))
    assert shape(samples) == (5, 4, 3)
    # MLL and fit.
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = float(mll.value_and_grad()[0])
    fit_gpytorch_mll(mll)
    assert float(mll.value_and_grad()[0]) > before
    # Copies are independent.
    other = copy.deepcopy(model)
    other.initialize(**{"likelihood.noise": 0.3})
    assert abs(flatten(other.likelihood.noise)[0] - 0.3) < 1e-12
    assert abs(flatten(model.likelihood.noise)[0] - 0.3) > 1e-6
    assert repr(model).startswith("MultiTaskGP(n=12, d=2, num_tasks=3")


@pytest.mark.parametrize("kind", KINDS)
def test_multitask_gp_options(kind: str) -> None:
    X, Y, Yvar = long_data(tasks=(1, 3, 6), tf=0, d=2)
    # Non-contiguous task values (task column 0), output tasks, rank, all_tasks
    # with a task without data, a custom kernel and fixed noise.
    model = MultiTaskGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        task_feature=0,
        train_Yvar=as_kind(Yvar, kind),
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)),
        output_tasks=[6, 1],
        rank=1,
        all_tasks=[1, 3, 6, 7],
    )
    assert model.num_tasks == 4 and model.all_tasks == [1, 3, 6, 7]
    assert model._observed_task_indices == [0, 1, 2] and model._unobserved_task_indices == [3]
    assert model.num_outputs == 2
    shapes = dict((n, shape(p)) for n, p in model.named_parameters())
    assert shapes["covar_module.kernels.1.raw_covar_factor"] == (4, 1)
    assert not any(n.startswith("likelihood") for n in shapes)
    post = model.posterior(as_kind(points(3, 2), kind), observation_noise=True)
    assert shape(post.mean) == (3, 2)
    # eval(): the factor of the task without data is the mean of the others
    # (of the positive factors, softplus of the raw values).
    raw = flatten(dict(model.named_parameters())["covar_module.kernels.1.raw_covar_factor"])
    factor = [math.log1p(math.exp(v)) for v in raw]
    assert abs(factor[3] - sum(factor[:3]) / 3) < 1e-12
    # Unknown task values.
    bad = [[5.0, 0.2, 0.3]]
    with pytest.raises(ValueError, match="invalid raw task values"):
        model.posterior(as_kind(bad, kind))
    with pytest.raises(ValueError, match="output_indices"):
        model.posterior(as_kind([[1.0, 0.2, 0.3]], kind), output_indices=[1])
    # A constant mean, a Gaussian likelihood and Normalize of some columns.
    X2, Y2, _ = long_data()
    model2 = MultiTaskGP(
        as_kind(X2, kind),
        as_kind(Y2, kind),
        task_feature=-1,
        mean_module=ConstantMean(),
        likelihood=GaussianLikelihood(),
        outcome_transform=None,
        input_transform=Normalize(d=3, indices=[0]),
        task_covar_prior=LKJCovariancePrior(3, 1.5, GammaPrior(2.0, 1.5)),
    )
    names = [n for n, _ in model2.named_parameters()]
    assert names[:2] == ["likelihood.noise_covar.raw_noise", "mean_module.raw_constant"]
    assert not hasattr(model2, "outcome_transform")
    mll = ExactMarginalLogLikelihood(model2.likelihood, model2)
    assert math.isfinite(float(mll.value_and_grad()[0]))
    assert shape(model2.posterior(as_kind(points(2, 2), kind)).mean) == (2, 3)


@pytest.mark.parametrize("kind", KINDS)
def test_multitask_gp_conditioning_and_acquisition(kind: str) -> None:
    X, Y, _ = long_data()
    model = MultiTaskGP(as_kind(X, kind), as_kind(Y, kind), task_feature=-1)
    Xn = as_kind([[0.3, 0.4, 1.0], [0.7, 0.1, 2.0]], kind)
    Yn = as_kind([[0.5], [0.9]], kind)
    with pytest.raises(RuntimeError, match="Call the model on some data first"):
        model.condition_on_observations(Xn, Yn)
    model.eval()
    cond = model.condition_on_observations(Xn, Yn)
    assert shape(cond.train_inputs[0]) == (14, 3)
    p0 = model.posterior(as_kind([[0.3, 0.4]], kind))
    p1 = cond.posterior(as_kind([[0.3, 0.4]], kind))
    assert flatten(p1.variance)[1] < flatten(p0.variance)[1]
    sampler = SobolQMCNormalSampler(sample_shape=[3], seed=0)
    fant = model.fantasize(Xn, sampler)
    assert fant.batch_shape == (3,)
    assert shape(fant.posterior(as_kind([[0.3, 0.4]], kind)).mean) == (3, 1, 3)
    with pytest.raises(RuntimeError, match="task feature"):
        model.fantasize(as_kind([[0.3, 0.4]], kind), sampler)
    with pytest.raises(DeprecationError):
        model.fantasize(Xn, sampler, observation_noise=True)
    with pytest.raises(NotImplementedError):
        model.posterior(as_kind([[0.3, 0.4]], kind), observation_noise=as_kind([[0.1]], kind))
    with pytest.raises(UnsupportedError):
        model.subset_output([0])
    # Acquisition functions take candidates without the task column.
    acqf = qLogNoisyExpectedImprovement(
        model,
        as_kind(points(3, 2), kind),
        sampler=SobolQMCNormalSampler([16], seed=1),
        posterior_transform=ScalarizedPosteriorTransform(as_kind([0.5, 0.3, 0.2], kind)),
        prune_baseline=False,
    )
    values = acqf(as_kind([points(2, 2, 7), points(2, 2, 8)], kind))
    assert shape(values) == (2,) and all(math.isfinite(v) for v in flatten(values))
    # Fixed noise: conditioning needs the noise; fantasies are not supported.
    _, _, Yvar = long_data()
    fixed = MultiTaskGP(
        as_kind(X, kind), as_kind(Y, kind), task_feature=-1, train_Yvar=as_kind(Yvar, kind)
    ).eval()
    with pytest.raises(Exception, match="noise"):
        fixed.condition_on_observations(Xn, Yn)
    cond = fixed.condition_on_observations(Xn, Yn, noise=as_kind([[0.01], [0.02]], kind))
    assert shape(cond.train_inputs[0]) == (14, 3)
    with pytest.raises(NotImplementedError):
        fixed.fantasize(Xn, sampler)
    # In a ModelListGP.
    lst = ModelListGP(model, copy.deepcopy(model))
    assert lst.num_outputs == 6
    assert shape(lst.posterior(as_kind(points(2, 2), kind)).mean) == (2, 6)


def test_multitask_gp_construction_errors() -> None:
    X, Y, Yvar = long_data()
    with pytest.raises(ValueError, match="task_feature"):
        MultiTaskGP(X, Y, task_feature=5)
    with pytest.raises(ValueError, match="Unsupported shape"):
        MultiTaskGP([X, X], [Y, Y], task_feature=-1)
    with pytest.raises(RuntimeError, match="All output tasks"):
        MultiTaskGP(X, Y, task_feature=-1, output_tasks=[7])
    with pytest.raises(Exception, match="all_tasks"):
        MultiTaskGP(X, Y, task_feature=-1, all_tasks=[0, 1])
    with pytest.raises(UnsupportedError, match="StratifiedStandardize"):
        MultiTaskGP(X, Y, task_feature=-1, outcome_transform=object())
    with pytest.raises(BotorchTensorDimensionError):
        MultiTaskGP(X, Y, task_feature=-1, outcome_transform=Standardize(m=2))
    with pytest.raises(UnsupportedError, match="train_Yvar"):
        MultiTaskGP(X, Y, task_feature=-1, train_Yvar=Yvar, likelihood=GaussianLikelihood())
    with pytest.raises(UnsupportedError, match="covar_module"):
        MultiTaskGP(X, Y, task_feature=-1, covar_module="rbf")
    with pytest.raises(BotorchTensorDimensionError, match="tasks"):
        MultiTaskGP(X, Y, task_feature=-1, likelihood=HadamardGaussianLikelihood(num_tasks=2))
    kernel = RBFKernel()
    MultiTaskGP(X, Y, task_feature=-1, covar_module=kernel)
    with pytest.raises(UnsupportedError, match="already part of another model"):
        MultiTaskGP(X, Y, task_feature=-1, covar_module=kernel)
    # Task values that are not task indices without validation.
    X2, Y2, _ = long_data(tasks=(0, 2))
    model = MultiTaskGP(X2, Y2, task_feature=-1, validate_task_values=False)
    assert model.all_tasks == [0, 2]


def test_multitask_gp_construct_inputs() -> None:
    X, Y, Yvar = long_data()
    data = SimpleNamespace(X=X, Y=Y, Yvar=None)
    kw = MultiTaskGP.construct_inputs(data, task_feature=-1, rank=2)
    assert kw["task_feature"] == -1 and kw["rank"] == 2 and "train_Yvar" not in kw
    model = MultiTaskGP(**kw)
    assert model._rank == 2
    kw = MultiTaskGP.construct_inputs(
        SimpleNamespace(X=X, Y=Y, Yvar=Yvar),
        task_feature=-1,
        prior_config={"use_LKJ_prior": True, "eta": 0.6},
    )
    assert isinstance(kw["task_covar_prior"], LKJCovariancePrior)
    assert kw["task_covar_prior"].n == 3 and kw["train_Yvar"] is Yvar
    with pytest.raises(ValueError, match="Only one of"):
        MultiTaskGP.construct_inputs(
            data, task_feature=-1, task_covar_prior=GammaPrior(1.0, 1.0), prior_config={}
        )
    with pytest.raises(ValueError, match="only config for LKJ"):
        MultiTaskGP.construct_inputs(data, task_feature=-1, prior_config={"use_LKJ_prior": False})
    with pytest.raises(ValueError, match="eta must be a real number"):
        MultiTaskGP.construct_inputs(
            data, task_feature=-1, prior_config={"use_LKJ_prior": True, "eta": "x"}
        )
    with pytest.raises(TypeError, match="SupervisedDataset"):
        MultiTaskGP.construct_inputs(object(), task_feature=-1)


# ---------------------------------------------------------------------------
# KroneckerMultiTaskGP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_kronecker_api(kind: str) -> None:
    X, Y = block_data(m=3)
    model = KroneckerMultiTaskGP(as_kind(X, kind), as_kind(Y, kind), rank=2)
    assert model.num_outputs == 3 and isinstance(model.likelihood, MultitaskGaussianLikelihood)
    names = [n for n, _ in model.named_parameters()]
    assert names == [
        "likelihood.raw_task_noises",
        "likelihood.raw_noise",
        "mean_module.base_means.0.raw_constant",
        "mean_module.base_means.1.raw_constant",
        "mean_module.base_means.2.raw_constant",
        "covar_module.task_covar_module.covar_factor",
        "covar_module.task_covar_module.raw_var",
        "covar_module.data_covar_module.raw_lengthscale",
    ]
    shapes = dict((n, shape(p)) for n, p in model.named_parameters())
    assert shapes["covar_module.task_covar_module.covar_factor"] == (3, 2)
    check_kind(model.likelihood.task_noises, kind)
    assert shape(model.train_targets) == (8, 3)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = float(mll.value_and_grad()[0])
    fit_gpytorch_mll(mll)
    assert float(mll.value_and_grad()[0]) > before
    post = model.posterior(as_kind(points(4, 2), kind))
    assert isinstance(post, MultitaskGPPosterior)
    assert shape(post.mean) == (4, 3) and shape(post.variance) == (4, 3)
    check_kind(post.mean, kind)
    assert post.base_sample_shape == ((8 + 4) * 3 + 8 * 3,)
    assert shape(post.rsample([5])) == (5, 4, 3)
    z = as_kind([[0.3] * post.base_sample_shape[0]] * 2, kind)
    s1, s2 = post.rsample_from_base_samples([2], z), post.rsample_from_base_samples([2], z)
    assert flatten(s1) == flatten(s2)
    # Base samples of the output shape (BoTorch draws the rest).
    assert shape(post.rsample_from_base_samples([2], as_kind([[[0.1] * 3] * 4] * 2, kind))) == (
        2,
        4,
        3,
    )
    noisy = model.posterior(as_kind(points(4, 2), kind), observation_noise=True)
    assert noisy.base_sample_shape == ((8 + 4) * 3 + 8 * 3 + 4 * 3,)
    assert all(a > b for a, b in zip(flatten(noisy.variance), flatten(post.variance), strict=True))
    batched = model.posterior(as_kind([points(2, 2, 1), points(2, 2, 2)], kind))
    assert shape(batched.mean) == (2, 2, 3)
    acqf = qExpectedImprovement(
        model,
        best_f=0.5,
        sampler=SobolQMCNormalSampler([16], seed=0),
        objective=LinearMCObjective(as_kind([0.5, 0.3, 0.2], kind)),
    )
    values = acqf(as_kind([points(2, 2, 3), points(2, 2, 4)], kind))
    assert shape(values) == (2,)
    other = copy.deepcopy(model)
    other.likelihood.noise = 0.2
    assert abs(flatten(other.likelihood.noise)[0] - 0.2) < 1e-12
    assert abs(flatten(model.likelihood.noise)[0] - 0.2) > 1e-6
    assert repr(model) == "KroneckerMultiTaskGP(n=8, d=2, num_tasks=3, rank=2)"


@pytest.mark.parametrize("kind", KINDS)
def test_kronecker_options(kind: str) -> None:
    X, Y = block_data(m=2, d=3)
    model = KroneckerMultiTaskGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        data_covar_module=MaternKernel(nu=1.5, ard_num_dims=3),
        likelihood_rank=1,
        eta=2.0,
        sd_prior=SmoothedBoxPrior(0.01, 3.0, 0.1),
        outcome_transform=Standardize(m=2),
        input_transform=Normalize(d=3),
    )
    names = [n for n, _ in model.named_parameters()]
    assert names[:2] == ["likelihood.task_noise_covar_factor", "likelihood.raw_noise"]
    with pytest.raises(AttributeError):
        model.likelihood.task_noises  # noqa: B018
    post = model.posterior(as_kind(points(3, 3), kind))
    assert shape(post.mean) == (3, 2)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    assert math.isfinite(float(mll.value_and_grad()[0]))


def test_kronecker_errors() -> None:
    X, Y = block_data()
    model = KroneckerMultiTaskGP(X, Y)
    Xt = points(2, 2)
    with pytest.raises(NotImplementedError, match="Posterior transforms"):
        model.posterior(Xt, posterior_transform=ScalarizedPosteriorTransform([1.0, 1.0]))
    with pytest.raises(NotImplementedError, match="output_indices"):
        model.posterior(Xt, output_indices=[1])
    assert shape(model.posterior(Xt, output_indices=[0, 1]).mean) == (2, 2)
    with pytest.raises(NotImplementedError, match="observation noise"):
        model.posterior(Xt, observation_noise=[[0.1, 0.1]] * 2)
    with pytest.raises(BotorchTensorDimensionError):
        model.posterior(points(2, 3))
    with pytest.raises(RuntimeError, match="Call the model on some data first"):
        model.condition_on_observations(Xt, [[0.1, 0.2]] * 2)
    with pytest.raises(RuntimeError, match="Call the model on some data first"):
        model.fantasize(Xt, SobolQMCNormalSampler([2], seed=0))
    with pytest.raises(UnsupportedError):
        model.subset_output([0])
    with pytest.raises(UnsupportedError, match="Batched"):
        KroneckerMultiTaskGP([X, X], [Y, Y])
    with pytest.raises(UnsupportedError, match="MultitaskGaussianLikelihood"):
        KroneckerMultiTaskGP(X, Y, likelihood=GaussianLikelihood())
    with pytest.raises(BotorchTensorDimensionError, match="tasks"):
        KroneckerMultiTaskGP(X, Y, likelihood=MultitaskGaussianLikelihood(num_tasks=3))
    with pytest.raises(UnsupportedError, match="Unsupported keyword"):
        KroneckerMultiTaskGP(X, Y, foo=1)
    with pytest.raises(TypeError, match="Prior"):
        KroneckerMultiTaskGP(X, Y, task_covar_prior=1.0)
    with pytest.raises(BotorchTensorDimensionError):
        KroneckerMultiTaskGP(X, Y, outcome_transform=Standardize(m=3))


# ---------------------------------------------------------------------------
# Likelihoods, priors and transforms
# ---------------------------------------------------------------------------


def test_multitask_likelihoods() -> None:
    lik = HadamardGaussianLikelihood(num_tasks=3)
    assert lik.num_tasks == 3
    with pytest.raises(ValueError, match="no noise terms"):
        MultitaskGaussianLikelihood(num_tasks=2, has_global_noise=False, has_task_noise=False)
    with pytest.raises(UnsupportedError):
        MultitaskGaussianLikelihood(num_tasks=2, has_global_noise=False)
    with pytest.raises(RuntimeError, match="rank=0"):
        MultitaskGaussianLikelihood(num_tasks=2, task_prior=GammaPrior(1.0, 1.0))
    X, Y = block_data()
    lik = MultitaskGaussianLikelihood(num_tasks=2, rank=1)
    model = KroneckerMultiTaskGP(X, Y, likelihood=lik)
    assert model.likelihood is lik and lik.rank == 1
    assert dict(model.named_parameters())["likelihood.task_noise_covar_factor"] is not None


def test_smoothed_box_prior() -> None:
    p = SmoothedBoxPrior(1.0, 3.0, 0.5)
    norm = math.log(1 + 2.0 / (math.sqrt(2 * math.pi) * 0.5))
    inside = -math.log(0.5 * math.sqrt(2 * math.pi)) - norm
    for x, dist in ((2.0, 0.0), (1.0, 0.0), (0.2, 0.8), (4.0, 1.0)):
        expected = inside - dist**2 / (2 * 0.25)
        assert abs(flatten(p.log_prob([x]))[0] - expected) < 1e-6
    with pytest.raises(ValueError, match="a < b"):
        SmoothedBoxPrior(2.0, 1.0)
    assert repr(p) == "SmoothedBoxPrior(a=1.0, b=3.0, sigma=0.5)"


def test_normalize_with_indices() -> None:
    with pytest.raises(ValueError, match="empty"):
        Normalize(d=3, indices=[])
    with pytest.raises(ValueError, match="unique"):
        Normalize(d=3, indices=[0, 0])
    with pytest.raises(ValueError, match="smaller than"):
        Normalize(d=3, indices=[3])
    X, Y, _ = long_data()
    X = [[10 * r[0], r[1], r[2]] for r in X]
    tf = Normalize(d=3, indices=[0])
    model = MultiTaskGP(X, Y, task_feature=-1, input_transform=tf)
    assert list(tf.indices) == [0]
    assert shape(tf.bounds) == (2, 1)
    out = tf(X)
    assert max(r[0] for r in out) <= 1.0 + 1e-12 and [r[1:] for r in out] == [r[1:] for r in X]
    assert shape(model.posterior([[5.0, 0.5]]).mean) == (1, 3)
    with pytest.raises(BotorchTensorDimensionError):
        MultiTaskGP(X, Y, task_feature=-1, input_transform=Normalize(d=2, indices=[0]))
    Xs, Ys = block_data()
    # A SingleTaskGP normalizes a subset of the inputs since phase 6b.
    tf_s = Normalize(d=2, indices=[0])
    model_s = SingleTaskGP(Xs, [y[:1] for y in Ys], input_transform=tf_s)
    assert shape(tf_s.bounds) == (2, 1)
    assert shape(model_s.posterior(points(2, 2)).mean) == (2, 1)


@pytest.mark.parametrize("kind", KINDS)
def test_multitask_kernel(kind: str) -> None:
    k = MultitaskKernel(RBFKernel(), num_tasks=2, rank=1)
    x = as_kind(points(3, 2), kind)
    K = k(x)
    check_kind(K, kind)
    assert shape(K) == (6, 6)
    kx = flatten(RBFKernel()(x))
    B = flatten(k.task_covar_module.covar_matrix)
    full = flatten(K)
    # Interleaved: entry (i m + s, j m + t) = k(x_i, x_j) B[s, t].
    for i, s, j, t in ((0, 0, 1, 1), (2, 1, 0, 0), (1, 1, 1, 0)):
        assert abs(full[(i * 2 + s) * 6 + j * 2 + t] - kx[i * 3 + j] * B[s * 2 + t]) < 1e-12
    assert shape(k(x, diag=True)) == (6,)
    with pytest.raises(RuntimeError, match="last_dim_is_batch"):
        k(x, last_dim_is_batch=True)
    with pytest.raises(UnsupportedError):
        MultitaskKernel("rbf", num_tasks=2)
