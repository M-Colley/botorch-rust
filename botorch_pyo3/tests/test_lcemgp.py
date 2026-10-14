"""``LCEMGP``, ``ProvidedEmbeddingLCEMGP`` and ``Interval`` through the API,
for every array type (BoTorch is not needed; see ``test_parity_lcemgp.py``
for the comparisons with BoTorch)."""

from __future__ import annotations

import copy
import math
import random

import pytest

from botorch_rs.acquisition import qLogNoisyExpectedImprovement
from botorch_rs.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch_rs.constraints import GreaterThan, Interval, Positive
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood, HadamardGaussianLikelihood
from botorch_rs.means import MultitaskMean
from botorch_rs.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch_rs.models import (
    LCEMGP,
    ModelListGP,
    MultiTaskGP,
    ProvidedEmbeddingLCEMGP,
    SingleTaskGP,
)
from botorch_rs.models.contextual_multioutput import HAS_PROVIDED_EMBEDDING_LCEMGP
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import as_kind, check_kind, flatten, shape


def contextual_data(n_per: int = 4, contexts: tuple[int, ...] = (0, 1), d: int = 2) -> tuple:
    """Rows of the given contexts (task column last) and smooth targets."""
    rng = random.Random(3)
    X, Y = [], []
    for k in contexts:
        for _ in range(n_per):
            x = [rng.random() for _ in range(d)]
            X.append([*x, float(k)])
            Y.append([math.sin(3 * x[0]) + sum(v * v for v in x[1:]) + 0.3 * k])
    return X, Y


EMB = [[0.9, 0.1, 0.3], [0.85, 0.15, 0.35], [0.8, 0.2, 0.3]]


def build(kind: str, provided: bool, Y: list | None = None) -> LCEMGP:
    X, Y0 = contextual_data()
    kw = dict(
        train_X=as_kind(X, kind),
        train_Y=as_kind(Y if Y is not None else Y0, kind),
        task_feature=2,
        context_cat_feature=as_kind([[0.0], [1.0], [2.0]], kind),
        output_tasks=[2],
        all_tasks=[0, 1, 2],
    )
    if provided:
        return ProvidedEmbeddingLCEMGP(context_emb_feature=as_kind(EMB, kind), **kw)
    return LCEMGP(embs_dim_list=[1], seed=5, **kw)


def test_interval_api() -> None:
    c = Interval(0.0, 2.0, transform=None, initial_value=1.0)
    assert not c.enforced and (c.lower_bound, c.upper_bound) == (0.0, 2.0)
    assert c.initial_value == 1.0 and c.transform(0.7) == 0.7 and list(c) == [0.0, 2.0]
    s = Interval(0.1, 3.0)
    assert s.enforced and abs(s.transform(0.0) - 1.55) < 1e-7
    assert abs(s.inverse_transform(s.transform(0.4)) - 0.4) < 1e-12
    assert s.check(1.0) and not s.check(3.5)
    with pytest.raises(ValueError):
        Interval(1.0, 1.0)
    with pytest.raises(ValueError):
        Interval(0.0, math.inf)
    assert isinstance(GreaterThan(0.1), Interval) and isinstance(Positive(), Interval)
    with pytest.raises(UnsupportedError):
        GreaterThan(0.0, transform="sigmoid")


def test_interval_constraints_in_a_single_task_gp(kind: str) -> None:
    X = [[0.1 * i, (0.37 * i) % 1.0] for i in range(10)]
    Y = [[math.sin(4 * x[0]) + x[1]] for x in X]
    covar = ScaleKernel(
        RBFKernel(
            ard_num_dims=2,
            lengthscale_constraint=Interval(0.05, 1.5, transform=None, initial_value=0.6),
        ),
        outputscale_constraint=Interval(0.1, 5.0),
    )
    lik = GaussianLikelihood(
        noise_constraint=Interval(1e-4, 0.2, transform=None, initial_value=0.05)
    )
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), covar_module=covar, likelihood=lik)
    assert abs(flatten(model.covar_module.base_kernel.lengthscale)[0] - 0.6) < 1e-7
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    ls = flatten(model.covar_module.base_kernel.lengthscale)
    assert all(0.05 * (1 - 1e-7) <= v <= 1.5 + 1e-12 for v in ls)
    noise = flatten(model.likelihood.noise)[0]
    assert 1e-4 * (1 - 1e-7) <= noise <= 0.2 + 1e-12  # float32 bounds, like GPyTorch
    assert 0.1 < flatten(model.covar_module.outputscale)[0] < 5.0


@pytest.mark.parametrize("provided", [False, True], ids=["learned", "provided"])
def test_contextual_model(kind: str, provided: bool) -> None:
    model = build(kind, provided)
    assert isinstance(model, MultiTaskGP) and HAS_PROVIDED_EMBEDDING_LCEMGP
    assert model.num_tasks == 3 and model._output_tasks == [2] and model.num_outputs == 1
    assert model._unobserved_task_indices == [2]
    assert isinstance(model.likelihood, HadamardGaussianLikelihood)
    assert isinstance(model.mean_module, MultitaskMean)
    noise = model.likelihood.noise
    check_kind(noise, kind)
    assert shape(noise) == (3,)
    names = [n for n, _ in model.named_parameters()]
    if provided:
        assert names[0] == "raw_context_specific_weight" and model.emb_layers == []
        assert flatten(model.context_specific_weight) == [0.0]
        assert shape(model.task_covar_module_base.lengthscale) == (1, 3)
    else:
        assert "emb_layers.0.weight" in names and shape(model.emb_layers[0].weight) == (3, 1)
        assert shape(model.task_covar_module_base.lengthscale) == (1, 1)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = float(flatten(mll())[0])
    fit_gpytorch_mll(mll)
    after = float(flatten(mll())[0])
    assert after >= before
    # eval(): the unobserved context's constant is the observed average.
    model.eval()
    c = [float(flatten(m.constant)[0]) for m in model.mean_module.base_means]
    assert abs(c[2] - (c[0] + c[1]) / 2) < 1e-15
    test_X = as_kind([[0.2, 0.4], [0.7, 0.1], [0.5, 0.5]], kind)
    post = model.posterior(test_X)
    check_kind(post.mean, kind)
    assert shape(post.mean) == (3, 1) and shape(post.variance) == (3, 1)
    assert shape(post.distribution.covariance_matrix) == (3, 3)
    all_tasks = model.posterior(test_X, output_indices=[0, 1, 2])
    assert shape(all_tasks.mean) == (3, 3)
    full = model.posterior(as_kind([[0.2, 0.4, 0.0], [0.7, 0.1, 2.0]], kind))
    assert shape(full.mean) == (2, 1)
    # The posterior of the current context at task-free inputs equals the one
    # with the task column.
    p2 = model.posterior(as_kind([[0.2, 0.4, 2.0]], kind))
    assert abs(flatten(p2.mean)[0] - flatten(post.mean)[0]) < 1e-12
    with pytest.raises(ValueError):
        model.posterior(as_kind([[0.2, 0.4, 7.0]], kind))
    copied = copy.deepcopy(model)
    assert flatten(copied.posterior(test_X).mean) == flatten(post.mean)


def test_model_list_and_acquisition(kind: str) -> None:
    X, Y = contextual_data()
    Y2 = [[(x[0] - 0.4) ** 2] for x in X]
    m1, m2 = build(kind, True), build(kind, True, Y2)
    model = ModelListGP(m1, m2)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    Xb = as_kind([x[:2] for x in X], kind)
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model,
        ref_point=[-1.0, -1.0],
        X_baseline=Xb,
        sampler=SobolQMCNormalSampler(sample_shape=[32], seed=0),
        prune_baseline=True,
    )
    bounds = as_kind([[0.0, 0.0], [1.0, 1.0]], kind)
    cand, _ = optimize_acqf(acqf, bounds, q=1, num_restarts=2, raw_samples=16, options={"seed": 0})
    assert shape(cand) == (1, 2)
    single = build(kind, False)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(single.likelihood, single))
    nei = qLogNoisyExpectedImprovement(
        single, X_baseline=Xb, sampler=SobolQMCNormalSampler(sample_shape=[32], seed=0)
    )
    cand, value = optimize_acqf(
        nei, bounds, q=1, num_restarts=2, raw_samples=16, options={"seed": 0}
    )
    assert shape(cand) == (1, 2) and math.isfinite(float(flatten(value)[0]))
    with pytest.raises(UnsupportedError):
        model.fantasize(Xb, SobolQMCNormalSampler(sample_shape=[2], seed=0))


def test_construction_errors() -> None:
    X, Y = contextual_data()
    with pytest.raises(UnsupportedError):
        MultiTaskGP(X, Y, task_feature=-1)
    with pytest.raises(UnsupportedError):
        LCEMGP(X, Y, task_feature=-1, train_Yvar=[[0.1]] * len(Y))
    with pytest.raises(ValueError, match="task_feature"):
        LCEMGP(X, Y, task_feature=5)
    with pytest.raises(ValueError):
        ProvidedEmbeddingLCEMGP(X, Y, task_feature=-1)
    model = LCEMGP(X, Y, task_feature=-1, seed=1)
    assert model.all_tasks == [0, 1] and model._output_tasks == [0, 1]
    assert model.num_outputs == 2
    a = LCEMGP(X, Y, task_feature=-1, seed=1)
    assert [list(flatten(p)) for _, p in a.named_parameters()] == [
        list(flatten(p)) for _, p in model.named_parameters()
    ]
