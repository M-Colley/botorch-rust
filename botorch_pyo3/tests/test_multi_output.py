"""Multi-output and batched ``SingleTaskGP``, ``ModelListGP``, posterior
transforms and the constrained analytic EI: shapes, array types, module
attributes, fitting, acquisition functions and errors."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from botorch_rs.acquisition import (
    ConstrainedExpectedImprovement,
    LogConstrainedExpectedImprovement,
    LogExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
    prune_inferior_points,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qSimpleRegret,
)
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.exceptions import (
    BotorchTensorDimensionError,
    NumericsWarning,
    OptimizationWarning,
    UnsupportedError,
)
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, LikelihoodList
from botorch_rs.means import ConstantMean
from botorch_rs.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.transforms import Normalize, Standardize
from botorch_rs.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
)
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import as_kind, assert_close, check_kind, flatten, make_data, shape, to_list


def multi_data(n: int = 12, d: int = 2, m: int = 3, seed: int = 0) -> tuple[list, list]:
    X, _ = make_data(n=n, d=d, seed=seed)
    Y = [
        [math.sin((3 + t) * x[0]) + (1 + t) * sum(x[1:]) ** 2 + 0.3 * t for t in range(m)]
        for x in X
    ]
    return X, Y


def batched_data(b: int = 2, n: int = 8, d: int = 2, m: int = 2) -> tuple[list, list]:
    Xs, Ys = [], []
    for i in range(b):
        X, Y = multi_data(n=n, d=d, m=m, seed=10 + i)
        Xs.append([[v * (1 + 0.5 * i) for v in x] for x in X])
        Ys.append(Y)
    return Xs, Ys


def test_multi_output_model(kind: str) -> None:
    X, Y = multi_data()
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    assert model.num_outputs == 3 and model.batch_shape == ()
    assert model._aug_batch_shape == (3,)
    (train_x,) = model.train_inputs
    check_kind(train_x, kind)
    assert shape(train_x) == (3, 12, 2)  # expanded over the outputs, like BoTorch
    assert shape(model.train_targets) == (3, 12)
    names = [(n, shape(v)) for n, v in model.named_parameters()]
    assert names == [
        ("likelihood.noise_covar.raw_noise", (3, 1)),
        ("mean_module.raw_constant", (3,)),
        ("covar_module.raw_lengthscale", (3, 1, 2)),
    ]
    assert shape(model.covar_module.lengthscale) == (3, 1, 2)
    assert shape(model.likelihood.noise) == (3, 1)
    assert shape(model.mean_module.constant) == (3,)
    assert shape(model.outcome_transform.means) == (1, 3)
    # Setters broadcast like torch.
    model.covar_module.lengthscale = 0.5
    assert_close(model.covar_module.lengthscale, [[[0.5, 0.5]]] * 3, rtol=1e-15)
    model.covar_module.lengthscale = as_kind([[0.3, 0.6]], kind)
    assert_close(model.covar_module.lengthscale, [[[0.3, 0.6]]] * 3, rtol=1e-15)
    model.likelihood.noise = as_kind([[0.01], [0.02], [0.03]], kind)
    assert_close(model.likelihood.noise, [[0.01], [0.02], [0.03]], rtol=1e-15)
    model.mean_module.constant = as_kind([0.1, 0.2, 0.3], kind)
    assert_close(model.mean_module.constant, [0.1, 0.2, 0.3], rtol=1e-15)
    with pytest.raises(ValueError):
        model.covar_module.lengthscale = as_kind([[0.3, 0.6, 0.9]], kind)
    # MLL: one value per output.
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    values = mll()
    check_kind(values, kind)
    assert shape(values) == (3,)
    value, grad = mll.value_and_grad()
    assert len(value) == 3 and len(grad) == 3 + 3 + 6
    # Each output equals a single-output model of that output.
    for t in range(3):
        single = SingleTaskGP(X, [[y[t]] for y in Y])
        single.covar_module.lengthscale = [[0.3, 0.6]]
        single.likelihood.noise = 0.01 * (t + 1)
        single.mean_module.constant = 0.1 * (t + 1)
        assert flatten(ExactMarginalLogLikelihood(single.likelihood, single)()) == [
            pytest.approx(flatten(values)[t], rel=1e-14)
        ]


def test_multi_output_posterior(kind: str) -> None:
    X, Y = multi_data(m=2)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    Xt = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.3]]
    post = model.posterior(as_kind(Xt, kind))
    check_kind(post.mean, kind)
    assert shape(post.mean) == (3, 2) and shape(post.variance) == (3, 2)
    mvn = post.distribution
    assert type(mvn).__name__ == "MultitaskMultivariateNormal" and not mvn._interleaved
    cov = to_list(mvn.covariance_matrix)
    assert shape(cov) == (6, 6)
    # Output-major layout: independent outputs, variances on the diagonal.
    assert all(cov[i][j] == 0.0 for i in range(3) for j in range(3, 6))
    var = to_list(post.variance)
    for t in range(2):
        for p in range(3):
            assert cov[t * 3 + p][t * 3 + p] == pytest.approx(var[p][t], rel=1e-14)
    assert shape(mvn.mean) == (3, 2) and mvn.event_shape == (3, 2)
    assert shape(model.posterior(as_kind([Xt] * 4, kind)).mean) == (4, 3, 2)
    # Observation noise, output selection, samples.
    noisy = model.posterior(as_kind(Xt, kind), observation_noise=True)
    assert all(a > b for a, b in zip(flatten(noisy.variance), flatten(post.variance), strict=True))
    sub = model.posterior(as_kind(Xt, kind), output_indices=[1])
    assert shape(sub.mean) == (3, 1)
    assert_close(flatten(sub.mean), [row[1] for row in to_list(post.mean)])
    samples = post.rsample(sample_shape=[5])
    check_kind(samples, kind)
    assert shape(samples) == (5, 3, 2)
    zero = post.rsample_from_base_samples([2], as_kind([[[0.0, 0.0]] * 3] * 2, kind))
    assert_close(to_list(zero)[1], to_list(post.mean))
    sampler = SobolQMCNormalSampler(sample_shape=[16], seed=0)
    assert shape(sampler(post)) == (16, 3, 2)
    with pytest.raises(BotorchTensorDimensionError):
        model.posterior(as_kind(Xt, kind), output_indices=[2])


def test_posterior_transform(kind: str) -> None:
    X, Y = multi_data(m=3)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    w = as_kind([0.5, -1.0, 2.0], kind)
    pt = ScalarizedPosteriorTransform(w, offset=0.25)
    Xt = [[0.1, 0.2], [0.5, 0.5]]
    post = model.posterior(as_kind(Xt, kind))
    scal = model.posterior(as_kind(Xt, kind), posterior_transform=pt)
    assert shape(scal.mean) == (2, 1)
    assert type(scal.distribution).__name__ == "MultivariateNormal"
    means = to_list(post.mean)
    variances = to_list(post.variance)
    for p in range(2):
        mu = 0.25 + 0.5 * means[p][0] - means[p][1] + 2.0 * means[p][2]
        var = 0.25 * variances[p][0] + variances[p][1] + 4.0 * variances[p][2]
        assert flatten(scal.mean)[p] == pytest.approx(mu, rel=1e-13)
        assert flatten(scal.variance)[p] == pytest.approx(var, rel=1e-12)
    assert_close(flatten(pt(post).mean), flatten(scal.mean), rtol=1e-15)
    assert_close(flatten(pt.evaluate(post.mean)), flatten(scal.mean), rtol=1e-13)
    with pytest.raises(RuntimeError):
        ScalarizedPosteriorTransform([1.0, 2.0])(post)
    with pytest.raises(RuntimeError, match="weights"):
        model.posterior(Xt, posterior_transform=ScalarizedPosteriorTransform([1.0, 2.0]))
    with pytest.raises(RuntimeError, match="weights"):
        LogExpectedImprovement(model, 0.0, posterior_transform=ScalarizedPosteriorTransform([1.0]))
    with pytest.raises(ValueError):
        ScalarizedPosteriorTransform([[1.0, 2.0]])


def test_fixed_noise_and_custom_modules(kind: str) -> None:
    X, Y = multi_data(m=2)
    Yvar = [[0.01, 0.02 + 0.001 * i] for i in range(len(Y))]
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), as_kind(Yvar, kind))
    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    assert shape(model.likelihood.noise) == (2, 12)
    legacy = SingleTaskGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        covar_module=get_matern_kernel_with_gamma_prior(2, batch_shape=[2]),
        likelihood=get_gaussian_likelihood_with_gamma_prior(batch_shape=[2]),
    )
    assert shape(legacy.covar_module.outputscale) == (2,)
    assert shape(legacy.covar_module.base_kernel.lengthscale) == (2, 1, 2)
    legacy.covar_module.outputscale = as_kind([1.5, 2.5], kind)
    assert_close(legacy.covar_module.outputscale, [1.5, 2.5], rtol=1e-14)
    custom = SingleTaskGP(
        X,
        Y,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=2, batch_shape=[2]), batch_shape=[2]),
        mean_module=ConstantMean(batch_shape=[2]),
    )
    assert shape(custom.covar_module.base_kernel.lengthscale) == (2, 1, 2)
    # Modules shared by all outputs are not supported (independent hyperparameters).
    with pytest.raises(UnsupportedError):
        SingleTaskGP(X, Y, covar_module=get_matern_kernel_with_gamma_prior(2))
    with pytest.raises(UnsupportedError):
        SingleTaskGP(X, Y, mean_module=ConstantMean())
    with pytest.raises(BotorchTensorDimensionError):
        SingleTaskGP(X, Y, covar_module=get_covar_module_with_dim_scaled_prior(2, batch_shape=[3]))


def test_batched_model(kind: str) -> None:
    Xs, Ys = batched_data(b=2, m=2)
    model = SingleTaskGP(as_kind(Xs, kind), as_kind(Ys, kind), input_transform=Normalize(d=2))
    assert model.batch_shape == (2,) and model._aug_batch_shape == (2, 2)
    assert shape(model.train_inputs[0]) == (2, 2, 8, 2)
    assert shape(model.train_targets) == (2, 2, 8)
    assert shape(model.covar_module.lengthscale) == (2, 2, 1, 2)
    assert shape(model.outcome_transform.means) == (2, 1, 2)
    assert shape(model.input_transform.bounds) == (2, 2)  # shared by the batch
    assert shape(ExactMarginalLogLikelihood(model.likelihood, model)()) == (2, 2)
    Xt = [[0.2, 0.4], [0.6, 0.1], [0.5, 0.9]]
    assert shape(model.posterior(as_kind(Xt, kind)).mean) == (2, 3, 2)
    assert shape(model.posterior(as_kind([Xt, Xt], kind)).mean) == (2, 3, 2)
    post = model.posterior(as_kind([[Xt], [Xt], [Xt]], kind))
    assert shape(post.mean) == (3, 2, 3, 2)
    assert shape(post.distribution.covariance_matrix) == (3, 2, 6, 6)
    assert shape(post.rsample([4])) == (4, 3, 2, 3, 2)
    with pytest.raises(BotorchTensorDimensionError):
        model.posterior(as_kind([Xt] * 3, kind))  # batch 3 vs model batch 2
    # Batch elements equal unbatched models of their data.
    plain = SingleTaskGP(as_kind(Xs, kind), as_kind(Ys, kind))
    single = SingleTaskGP(Xs[1], Ys[1])
    assert_close(to_list(plain.posterior(Xt).mean)[1], to_list(single.posterior(Xt).mean))
    with pytest.raises(UnsupportedError):
        LogExpectedImprovement(model, best_f=0.0)


def test_batched_single_output_per_batch_normalize(kind: str) -> None:
    Xs, Ys = batched_data(b=3, m=1)
    model = SingleTaskGP(
        as_kind(Xs, kind), as_kind(Ys, kind), input_transform=Normalize(d=2, batch_shape=[3])
    )
    assert model._aug_batch_shape == (3,)
    assert shape(model.input_transform.bounds) == (3, 2, 2)
    bounds = to_list(model.input_transform.bounds)
    for b in range(3):
        lower = [min(x[j] for x in Xs[b]) for j in range(2)]
        assert_close(bounds[b][0], lower, rtol=1e-15)
    post = model.posterior(as_kind([[0.5, 0.5]], kind))
    assert shape(post.mean) == (3, 1, 1)
    with pytest.raises(UnsupportedError):  # BoTorch misaligns per-batch bounds for m > 1
        Xm, Ym = batched_data(b=2, m=2)
        SingleTaskGP(Xm, Ym, input_transform=Normalize(d=2, batch_shape=[2]))
    with pytest.raises(RuntimeError):
        SingleTaskGP(Xs, Ys, outcome_transform=Standardize(m=1))


def test_fit_multi_output_and_batched(kind: str) -> None:
    X, Y = multi_data(m=2)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = flatten(mll())
    fit_gpytorch_mll(mll)
    after = flatten(mll())
    assert all(a > b for a, b in zip(after, before, strict=True))
    info = mll._fit_result
    assert info["message"] == "Batched L-BFGS-B: 2/2 outputs converged."
    assert len(info["elements"]) == 2
    assert info["fval"] == pytest.approx(-sum(after), rel=1e-12)
    with pytest.warns(OptimizationWarning, match="maxfun"):
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxfun": 10}})
    Xs, Ys = batched_data(b=2, m=1)
    batched = SingleTaskGP(Xs, Ys)
    bmll = ExactMarginalLogLikelihood(batched.likelihood, batched)
    fit_gpytorch_mll(bmll)
    assert len(bmll._fit_result["elements"]) == 2


def test_model_list(kind: str) -> None:
    X1, Y1 = multi_data(n=10, m=2, seed=1)
    X2, Y2 = make_data(n=7, d=2, seed=2)
    m1 = SingleTaskGP(as_kind(X1, kind), as_kind(Y1, kind))
    m2 = SingleTaskGP(as_kind(X2, kind), as_kind(Y2, kind), input_transform=Normalize(d=2))
    model = ModelListGP(m1, m2)
    assert model.num_outputs == 3 and model.models == [m1, m2]
    assert isinstance(model.likelihood, LikelihoodList)
    assert model.likelihood.likelihoods == [m1.likelihood, m2.likelihood]
    names = [n for n, _ in model.named_parameters()]
    assert names[0] == "models.0.likelihood.noise_covar.raw_noise"
    assert names[-1] == "models.1.covar_module.raw_lengthscale"
    Xt = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.3]]
    post = model.posterior(as_kind(Xt, kind))
    check_kind(post.mean, kind)
    assert shape(post.mean) == (3, 3)
    assert shape(post.distribution.covariance_matrix) == (9, 9)
    assert_close([r[:2] for r in to_list(post.mean)], to_list(m1.posterior(Xt).mean))
    assert_close([r[2:] for r in to_list(post.mean)], to_list(m2.posterior(Xt).mean))
    assert shape(post.rsample([2])) == (2, 3, 3)
    assert shape(model.posterior(as_kind([Xt] * 2, kind), output_indices=[2]).mean) == (2, 3, 1)
    # Fitting through the SumMarginalLogLikelihood updates the sub-models.
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    before = flatten(mll())
    fit_gpytorch_mll(mll)
    assert not mll.training and len(mll._fit_result) == 2
    assert all(a > b for a, b in zip(flatten(mll()), before, strict=True))
    assert shape(mll()) == (2,)  # GPyTorch broadcasts the sub-model MLLs
    model.initialize(**{"models.1.likelihood.noise": 0.02})
    assert_close(m2.likelihood.noise, [0.02], rtol=1e-15)
    clone = copy.deepcopy(model)
    assert (
        clone.models[0] is not m1 and clone.likelihood.likelihoods[0] is clone.models[0].likelihood
    )
    with pytest.raises(UnsupportedError):
        ModelListGP(m1, object())  # type: ignore[arg-type]
    with pytest.raises(UnsupportedError):
        SumMarginalLogLikelihood(m1.likelihood, model)


def test_acquisition_with_posterior_transform(kind: str) -> None:
    X, Y = multi_data(m=2)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    pt = ScalarizedPosteriorTransform(as_kind([0.7, 0.3], kind))
    points = as_kind([[[0.1, 0.9]], [[0.5, 0.5]], [[0.95, 0.05]]], kind)
    best = max(0.7 * y[0] + 0.3 * y[1] for y in Y)
    logei = LogExpectedImprovement(model, best_f=best, posterior_transform=pt)
    values = logei(points)
    check_kind(values, kind)
    assert shape(values) == (3,)
    # Equal to LogEI of the scalarized posterior, evaluated by hand.
    mean = flatten(model.posterior(points, posterior_transform=pt).mean)
    pm = flatten(PosteriorMean(model, posterior_transform=pt)(points))
    assert_close(pm, mean, rtol=1e-14)
    _, grad = logei.value_and_grad(points)
    assert shape(grad) == (3, 1, 2)
    sets = as_kind([[[0.1, 0.9], [0.4, 0.2]]], kind)
    sampler = SobolQMCNormalSampler(sample_shape=[64], seed=0)
    qlogei = qLogExpectedImprovement(model, best_f=best, sampler=sampler, posterior_transform=pt)
    assert shape(qlogei(sets)) == (1,)
    nei = qLogNoisyExpectedImprovement(
        model, X_baseline=as_kind(X, kind), sampler=sampler, posterior_transform=pt
    )
    assert all(math.isfinite(v) for v in flatten(nei(sets)))
    kept = prune_inferior_points(model, as_kind(X, kind), posterior_transform=pt)
    assert 1 <= shape(kept)[0] <= len(X)
    candidate, value = optimize_acqf(
        logei, as_kind([[0.0, 0.0], [1.0, 1.0]], kind), q=1, num_restarts=4, raw_samples=64
    )
    assert shape(candidate) == (1, 2)
    # Multi-output models need a posterior transform.
    with pytest.raises(UnsupportedError, match="posterior transform"):
        LogExpectedImprovement(model, best_f=0.0)
    with pytest.raises(UnsupportedError, match="objective or a posterior transform"):
        qSimpleRegret(model)
    with pytest.raises(UnsupportedError):
        prune_inferior_points(model, X)
    with pytest.raises(UnsupportedError):
        UpperConfidenceBound(model, beta=0.1, posterior_transform=object())
    # A model list works the same way.
    m2 = SingleTaskGP(*make_data(n=6, d=2, seed=4))
    lst = ModelListGP(model, m2)
    pt3 = ScalarizedPosteriorTransform([0.5, 0.3, 0.2])
    assert shape(UpperConfidenceBound(lst, beta=0.2, posterior_transform=pt3)(points)) == (3,)


def test_constrained_expected_improvement(kind: str) -> None:
    X, Y = multi_data(m=3)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    points = as_kind([[[0.1, 0.9]], [[0.5, 0.5]], [[0.95, 0.05]]], kind)
    constraints: dict[int, tuple[Any, Any]] = {1: (0.5, None), 2: (None, 2.5)}
    log_cei = LogConstrainedExpectedImprovement(
        model, best_f=1.0, objective_index=0, constraints=constraints
    )
    values = log_cei(points)
    check_kind(values, kind)
    assert shape(values) == (3,) and log_cei._log
    with pytest.warns(NumericsWarning, match="LogConstrainedExpectedImprovement"):
        cei = ConstrainedExpectedImprovement(
            model, best_f=1.0, objective_index=0, constraints=constraints
        )
    assert_close(flatten(cei(points)), [math.exp(v) for v in flatten(values)], rtol=1e-12)
    _, grad = log_cei.value_and_grad(points)
    assert shape(grad) == (3, 1, 2)
    # Both bounds, minimization, negative indices.
    both = LogConstrainedExpectedImprovement(
        model, best_f=0.0, objective_index=-1, constraints={0: (0.0, 3.0)}, maximize=False
    )
    assert all(math.isfinite(v) for v in flatten(both(points)))
    for kwargs in (
        {"objective_index": 0, "constraints": {}},
        {"objective_index": 0, "constraints": {0: (0.0, None)}},
        {"objective_index": 0, "constraints": {1: (1.0, 0.5)}},
        {"objective_index": 0, "constraints": {5: (0.0, None)}},
    ):
        with pytest.raises(ValueError):
            LogConstrainedExpectedImprovement(model, best_f=0.0, **kwargs)
    with pytest.raises(UnsupportedError):
        log_cei.set_X_pending(as_kind([[0.5, 0.5]], kind))
