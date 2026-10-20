"""API and error tests of the fully Bayesian GPs of phase 10a
(``SaasFullyBayesianSingleTaskGP``, ``FullyBayesianSingleTaskGP``,
``FullyBayesianLinearSingleTaskGP``, ``SaasFullyBayesianMultiTaskGP``, their
NumPyro models, ``fit_fully_bayesian_model_nuts``, BALD, Thompson sampling,
qLogNParEGO, qJES and lists of fully Bayesian GPs in acquisition
functions), without BoTorch, for every array type.

The exceptions and messages are BoTorch's.
"""

from __future__ import annotations

import math
import random
import warnings
from typing import Any

import pytest

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.bayesian_active_learning import (
    FULLY_BAYESIAN_ERROR_MSG,
    NEGATIVE_INFOGAIN_WARNING,
    check_negative_info_gain,
    qBayesianActiveLearningByDisagreement,
)
from botorch_rs.acquisition.logei import qLogNoisyExpectedImprovement
from botorch_rs.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch_rs.exceptions.errors import InvalidArgumentError, UnsupportedError
from botorch_rs.fit import fit_fully_bayesian_model_nuts, get_fitted_map_saas_model
from botorch_rs.models import ModelList, ModelListGP, SingleTaskGP
from botorch_rs.models import fully_bayesian as fb
from botorch_rs.models.fully_bayesian_multitask import (
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)
from botorch_rs.models.transforms import Normalize, Standardize
from botorch_rs.models.transforms.outcome import Log
from botorch_rs.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.utils.transforms import is_ensemble, is_fully_bayesian
from helpers import as_kind, check_kind, flatten, shape

MODELS = {
    "saas": fb.SaasFullyBayesianSingleTaskGP,
    "matern": fb.FullyBayesianSingleTaskGP,
    "linear": fb.FullyBayesianLinearSingleTaskGP,
}
FIT = {"warmup_steps": 24, "num_samples": 8, "thinning": 2}

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _data(n: int = 10, d: int = 2, seed: int = 0) -> tuple[list, list, list]:
    r = random.Random(seed)
    X = [[r.random() for _ in range(d)] for _ in range(n)]
    f = [math.sin(5 * x[0]) + 0.3 * sum(v * v for v in x[1:]) for x in X]
    m = sum(f) / n
    s = math.sqrt(sum((v - m) ** 2 for v in f) / (n - 1))
    Y = [[(v - m) / s + 0.01 * r.gauss(0, 1)] for v in f]
    Yvar = [[0.01 + 0.01 * r.random()] for _ in range(n)]
    return X, Y, Yvar


def _mt_data(n: int = 12, d: int = 2, seed: int = 0) -> tuple[list, list]:
    r = random.Random(seed)
    X = [[r.random() for _ in range(d)] + [float(i % 3)] for i in range(n)]
    Y = [[math.sin(4 * x[0]) + 0.2 * x[-1] + 0.01 * r.gauss(0, 1)] for x in X]
    m = sum(y[0] for y in Y) / n
    s = math.sqrt(sum((y[0] - m) ** 2 for y in Y) / (n - 1))
    return X, [[(y[0] - m) / s] for y in Y]


# ---------------------------------------------------------------------------
# Construction and errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MODELS))
def test_construction_errors(name: str) -> None:
    cls = MODELS[name]
    X, Y, Yvar = _data()
    with pytest.raises(ValueError, match="Expected train_X to have shape n x d"):
        cls([X], [Y])
    with pytest.raises(ValueError, match="Expected train_X to have shape n x d"):
        cls(X, [y + y for y in Y])
    with pytest.raises(ValueError, match="Expected train_Yvar to be None"):
        cls(X, Y, Yvar[:-1])
    with pytest.raises(UnsupportedError, match="Standardize"):
        cls(X, [[abs(y[0]) + 1] for y in Y], outcome_transform=Log())


@pytest.mark.parametrize("name", list(MODELS))
def test_unfitted_model(name: str) -> None:
    X, Y, _ = _data()
    model = MODELS[name](X, Y)
    msg = "Model has not been fitted"
    with pytest.raises(RuntimeError, match=msg):
        model.posterior([[0.5, 0.5]])
    with pytest.raises(RuntimeError, match=msg):
        _ = model.num_mcmc_samples
    with pytest.raises(RuntimeError, match=msg):
        _ = model.batch_shape
    assert model._is_fully_bayesian and model._is_ensemble
    assert is_fully_bayesian(model) and is_ensemble(model)
    assert not hasattr(model, "fantasize")
    # NumPyro tracing is an optional public interface, separately tested
    # with seeded effect handlers. Native model construction needs no JAX.


# ---------------------------------------------------------------------------
# Fits and posteriors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("fixed", [False, True])
def test_fit_and_posterior(kind: str, name: str, fixed: bool) -> None:
    X, Y, Yvar = _data(seed=1)
    model = MODELS[name](as_kind(X, kind), as_kind(Y, kind), as_kind(Yvar, kind) if fixed else None)
    fit_fully_bayesian_model_nuts(model, **FIT, seed=3)
    assert model.num_mcmc_samples == 4
    assert tuple(model.batch_shape) == (4,)
    Xt = as_kind([[0.2, 0.4], [0.7, 0.1], [0.5, 0.5]], kind)
    post = model.posterior(Xt)
    assert isinstance(post, GaussianMixturePosterior)
    check_kind(post.mean, kind)
    assert shape(post.mean) == (4, 3, 1)
    assert shape(post.mixture_mean) == (3, 1)
    assert shape(post.mixture_variance) == (3, 1)
    assert shape(post.mixture_covariance_matrix) == (3, 3)
    assert all(v > 0 for v in flatten(post.variance))
    noisy = model.posterior(Xt, observation_noise=True)
    assert all(a >= b for a, b in zip(flatten(noisy.variance), flatten(post.variance), strict=True))
    batched = model.posterior(as_kind([[[0.2, 0.4]], [[0.5, 0.5]]], kind))
    assert shape(batched.mean) == (2, 4, 1, 1)
    attr = "median_weight_variance" if name == "linear" else "median_lengthscale"
    med = getattr(model, attr)
    check_kind(med, kind)
    assert shape(med) == (2,)
    assert all(v > 0 for v in flatten(med))
    # The noise samples are the noise variance of the likelihood (>= 1e-4).
    if not fixed:
        assert all(v >= 1e-4 for v in flatten(model.likelihood.noise))


def test_fit_is_deterministic_for_a_seed() -> None:
    X, Y, _ = _data(seed=2)
    out = []
    for seed in (5, 5, 6):
        model = fb.SaasFullyBayesianSingleTaskGP(X, Y)
        fit_fully_bayesian_model_nuts(model, **FIT, seed=seed)
        out.append(flatten(model.median_lengthscale))
    assert out[0] == out[1]
    assert out[0] != out[2]


def test_refit_and_reload_do_not_chain_transforms() -> None:
    X, Y, _ = _data(seed=3)
    Xs = [[3 * v - 1 for v in x] for x in X]
    model = fb.SaasFullyBayesianSingleTaskGP(
        Xs,
        [[2 * y[0] + 1] for y in Y],
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
        use_input_warping=True,
    )
    for _ in range(2):
        fit_fully_bayesian_model_nuts(model, **FIT)
        names = [n for n, _ in model.named_parameters() if n.startswith("input_transform")]
        assert names == ["input_transform.tf1.concentration0", "input_transform.tf1.concentration1"]
    model.train()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.posterior([[0.0, 0.0]])


def test_replay_arguments() -> None:
    X, Y, _ = _data(seed=4)
    model = fb.SaasFullyBayesianSingleTaskGP(X, Y)
    with pytest.raises(ValueError, match="`draws` and `init_params` must be given together"):
        fit_fully_bayesian_model_nuts(model, draws=[("n", [0.0])])
    dim = model.pyro_model._potential().dim
    with pytest.raises(InvalidArgumentError, match="recorded draw"):
        fit_fully_bayesian_model_nuts(model, draws=[("u", 0.5, 0.5)], init_params=[0.0] * dim)


def test_load_mcmc_samples_and_helpers(kind: str) -> None:
    X, Y, _ = _data(seed=5)
    model = fb.SaasFullyBayesianSingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    dummy = model.pyro_model.get_dummy_mcmc_samples(3)
    assert sorted(dummy) == ["lengthscale", "mean", "noise", "outputscale"]
    assert shape(dummy["lengthscale"]) == (3, 2)
    samples = {
        "mean": as_kind([0.1, -0.2, 0.0], kind),
        "noise": as_kind([1e-5, 0.01, 0.1], kind),
        "outputscale": as_kind([1.0, 2.0, 0.5], kind),
        "lengthscale": as_kind([[0.5, 2.0], [1.0, 1.0], [3.0, 0.3]], kind),
    }
    model.load_mcmc_samples(samples)
    model.eval()
    assert model.num_mcmc_samples == 3
    # BoTorch clamps the loaded noise at 1e-4.
    noise = flatten(model.likelihood.noise)
    assert abs(noise[0] - 1e-4) < 1e-12 and abs(noise[2] - 0.1) < 1e-12
    med = flatten(model.median_lengthscale)
    assert med == pytest.approx([1.0, 1.0])
    # Post-processing of NumPyro-style samples (lengthscale from tausq and
    # the inverse squared lengthscales; noise + 1e-4).
    pm = fb.SaasPyroModel()
    pm.set_inputs(X, Y)
    out = pm.postprocess_mcmc_samples(
        {
            "kernel_tausq": [0.25],
            "_kernel_inv_length_sq": [[4.0, 1.0]],
            "noise": [0.01],
            "mean": [0.0],
            "outputscale": [1.0],
        }
    )
    assert sorted(out) == ["lengthscale", "mean", "noise", "outputscale"]
    assert flatten(out["lengthscale"]) == pytest.approx([1.0, 2.0])
    assert flatten(out["noise"]) == pytest.approx([0.0101])
    lp = fb.LinearPyroModel()
    lp.set_inputs(X, Y)
    out = lp.postprocess_mcmc_samples(
        {"tau_sq": [4.0], "_weight_variance_sq": [[1.0, 0.25]], "mean": [0.0], "noise": [0.0]}
    )
    assert flatten(out["weight_variance"]) == pytest.approx([2.0, 1.0])


def test_kernel_helpers(kind: str) -> None:
    X = as_kind([[0.0, 0.0], [1.0, 0.0]], kind)
    ls = as_kind([0.5, 1.0], kind)
    d = fb.compute_dists(X, ls)
    check_kind(d, kind)
    assert flatten(d) == pytest.approx([1e-15, 2.0, 2.0, 1e-15])
    k = flatten(fb.matern52_kernel(X, ls))
    s5 = math.sqrt(5) * 2.0
    assert k[1] == pytest.approx((1 + s5 + 5 / 3 * 4.0) * math.exp(-s5))
    assert flatten(fb.linear_kernel(X, as_kind([4.0, 1.0], kind))) == pytest.approx(
        [0.0, 0.0, 0.0, 4.0]
    )
    assert shape(fb.reshape_and_detach(as_kind([[0.0, 0.0]], kind), [1.0, 2.0])) == (1, 2)
    with pytest.raises(RuntimeError):
        fb.reshape_and_detach([[0.0, 0.0]], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("kernel", ["compute_dists", "matern52_kernel", "linear_kernel"])
def test_kernel_helpers_broadcast_batches(kind: str, kernel: str) -> None:
    X = [[[[0.0, 0.1], [0.3, 0.7], [1.0, 0.4]]], [[[0.2, 0.5], [0.5, 0.8], [0.9, 0.0]]]]
    scales = [[[0.5, 1.0]], [[2.0, 0.3]], [[0.7, 1.8]], [[1.1, 1.2]]]
    result = getattr(fb, kernel)(as_kind(X, kind), as_kind(scales, kind))
    check_kind(result, kind)
    assert shape(result) == (2, 4, 3, 3)
    expected = []
    for batch in X:
        for (scale,) in scales:
            if kernel == "linear_kernel":
                expected.extend(
                    sum(a[k] * b[k] * scale[k] for k in range(2))
                    for a in batch[0]
                    for b in batch[0]
                )
            else:
                for a in batch[0]:
                    for b in batch[0]:
                        r = math.sqrt(
                            max(sum(((a[k] - b[k]) / scale[k]) ** 2 for k in range(2)), 1e-30)
                        )
                        expected.append(
                            r
                            if kernel == "compute_dists"
                            else (1 + math.sqrt(5) * r + 5 / 3 * r * r)
                            * math.exp(-math.sqrt(5) * r)
                        )
    assert flatten(result) == pytest.approx(expected)
    # A scalar scale broadcasts to every feature; incompatible feature
    # counts must raise instead of silently cycling the scale values.
    assert shape(getattr(fb, kernel)(as_kind(X, kind), 0.5)) == (2, 1, 3, 3)
    with pytest.raises(ValueError, match="cannot be broadcast"):
        getattr(fb, kernel)(as_kind([[0.1, 0.2, 0.3]], kind), as_kind([0.5, 1.0], kind))


@pytest.mark.parametrize("name", list(MODELS))
def test_subset_warp_samples_and_custom_eps(kind: str, name: str) -> None:
    X, Y, _ = _data(d=3)
    model = MODELS[name](
        as_kind(X, kind), as_kind(Y, kind), use_input_warping=True, indices_to_warp=[1]
    )
    pm = model.pyro_model
    pm._eps = 0.02
    samples = pm.get_dummy_mcmc_samples(2)
    assert shape(samples["c0"]) == (2, 1)
    assert shape(samples["c1"]) == (2, 1)
    samples["c0"] = [[2.0], [0.4]]
    samples["c1"] = [[0.8], [2.0]]
    model.load_mcmc_samples(samples)
    model.eval()
    actual = model.transform_inputs(as_kind(X, kind))
    expected = []
    for c0, c1 in zip(samples["c0"], samples["c1"], strict=True):
        expected.extend(flatten(pm.warp(as_kind(X, kind), c0=c0, c1=c1)))
    if name == "linear":
        expected = [v - 0.5 for v in expected]
    assert flatten(actual) == pytest.approx(expected)
    assert shape(model.posterior(as_kind(X[:2], kind)).mean) == (2, 2, 1)


def test_model_transform_inputs(kind: str) -> None:
    X, Y, _ = _data()
    Xt = as_kind(X[:2], kind)
    model = fb.SaasFullyBayesianSingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    assert model.transform_inputs(Xt) is Xt
    tf = Normalize(d=2, bounds=[[0.0, 0.0], [2.0, 4.0]])
    transformed = model.transform_inputs(Xt, input_transform=tf)
    check_kind(transformed, kind)
    assert flatten(transformed) == pytest.approx(
        [v / (2.0 if i % 2 == 0 else 4.0) for i, v in enumerate(flatten(Xt))]
    )
    model.input_transform = Normalize(d=2, bounds=[[0.0, 0.0], [3.0, 5.0]])
    assert flatten(model.transform_inputs(Xt)) != flatten(transformed)
    assert flatten(model.transform_inputs(Xt, input_transform=tf)) == flatten(transformed)


def test_construct_inputs() -> None:
    from botorch_rs.utils.datasets import SupervisedDataset

    X, Y, Yvar = _data(n=5)
    ds = SupervisedDataset(X, Y, feature_names=["a", "b"], outcome_names=["y"], Yvar=Yvar)
    kw = fb.SaasFullyBayesianSingleTaskGP.construct_inputs(ds, use_input_warping=True)
    assert kw["use_input_warping"] is True and kw["indices_to_warp"] is None
    assert flatten(kw["train_Yvar"]) == flatten(Yvar)


# ---------------------------------------------------------------------------
# Multi-task
# ---------------------------------------------------------------------------


def test_multitask_model(kind: str) -> None:
    X, Y = _mt_data()
    model = SaasFullyBayesianMultiTaskGP(
        as_kind(X, kind), as_kind(Y, kind), task_feature=-1, rank=2
    )
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.posterior(as_kind([[0.5, 0.5]], kind))
    with pytest.raises(NotImplementedError, match="Fantasize is not implemented"):
        model.fantasize(X, None)
    fit_fully_bayesian_model_nuts(model, **FIT)
    assert model.num_mcmc_samples == 4
    post = model.posterior(as_kind([[0.5, 0.5], [0.1, 0.9]], kind))
    check_kind(post.mean, kind)
    assert shape(post.mean) == (4, 2, 3)
    assert shape(post.mixture_mean) == (2, 3)
    assert shape(post.mixture_covariance_matrix) == (6, 6)
    single = model.posterior(as_kind([[0.5, 0.5, 1.0]], kind))
    assert shape(single.mean) == (4, 1, 1)
    assert shape(model.median_lengthscale) == (2,)
    with pytest.raises(ValueError, match="Expected train_X to have shape n x d"):
        SaasFullyBayesianMultiTaskGP([X], [Y], task_feature=-1)
    with pytest.raises(UnsupportedError, match="warping"):
        SaasFullyBayesianMultiTaskGP(
            X, Y, task_feature=-1, pyro_model=MultitaskSaasPyroModel(use_input_warping=True)
        )


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def _fitted(name: str = "saas", seed: int = 0, num: int = 8) -> tuple[Any, list, list]:
    X, Y, _ = _data(seed=seed)
    model = MODELS[name](X, Y)
    fit_fully_bayesian_model_nuts(model, warmup_steps=24, num_samples=num, thinning=2, seed=seed)
    return model, X, Y


def test_bald(kind: str) -> None:
    model, X, _ = _fitted()
    with pytest.raises(RuntimeError, match="Fully Bayesian acquisition functions require"):
        qBayesianActiveLearningByDisagreement(SingleTaskGP(X, [[x[0]] for x in X]))
    assert "SaasFullyBayesianSingleTaskGP" in FULLY_BAYESIAN_ERROR_MSG
    acqf = qBayesianActiveLearningByDisagreement(
        model, sampler=SobolQMCNormalSampler(sample_shape=(64,), seed=0)
    )
    Xt = as_kind([[[0.1, 0.2], [0.8, 0.9]], [[0.5, 0.5], [0.4, 0.6]]], kind)
    v = acqf(Xt)
    check_kind(v, kind)
    assert shape(v) == (2,)
    assert all(math.isfinite(a) for a in flatten(v))
    vals, grads = acqf.value_and_grad(Xt)
    assert shape(grads) == (2, 2, 2)
    acqf.set_X_pending(as_kind([[0.3, 0.3]], kind))
    assert shape(acqf.X_pending) == (1, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_negative_info_gain([-1.0, 0.5])
    assert any(NEGATIVE_INFOGAIN_WARNING in str(x.message) for x in w)


def test_acquisition_on_fully_bayesian_lists() -> None:
    m1, X, Y = _fitted("saas", seed=1)
    m2, _, _ = _fitted("matern", seed=1)
    m3, _, _ = _fitted("matern", seed=1, num=4)
    ref = [-3.0, -3.0]
    lst = ModelListGP(m1, m2)
    sampler = SobolQMCNormalSampler(sample_shape=(8,), seed=0)
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        lst, ref_point=ref, X_baseline=X, sampler=sampler, marginalize_dim=-3, prune_baseline=True
    )
    assert shape(acqf([[[0.5, 0.5]]])) == (1,)
    with pytest.raises(UnsupportedError, match="marginalize_dim=-3"):
        qLogNoisyExpectedHypervolumeImprovement(
            lst, ref_point=ref, X_baseline=X, sampler=sampler, marginalize_dim=-1
        )
    with pytest.raises(NotImplementedError, match="same `batch_shape`"):
        ModelListGP(m1, m3)
    with pytest.raises(UnsupportedError, match="different sizes"):
        qLogNoisyExpectedHypervolumeImprovement(
            ModelList(m1, m3), ref_point=ref, X_baseline=X, sampler=sampler
        )
    # Single-objective acquisition functions average over the samples too.
    nei = qLogNoisyExpectedImprovement(m1, X_baseline=X, sampler=sampler)
    assert shape(nei([[[0.5, 0.5]]])) == (1,)


def test_thompson_sampling_parego_and_jes_on_fully_bayesian_gps(kind: str) -> None:
    from botorch_rs.acquisition.joint_entropy_search import qJointEntropySearch
    from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
    from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling

    m1, X, Y = _fitted("saas", seed=2)
    m2, _, _ = _fitted("matern", seed=2)
    # Thompson sampling: paths of the samples selected by ensemble indices.
    ts = PathwiseThompsonSampling(m1, seed=0)
    assert ts.ensemble_indices is None
    Xt = as_kind([[[0.1, 0.2], [0.8, 0.9]], [[0.5, 0.5], [0.4, 0.6]]], kind)
    v = ts(Xt)
    check_kind(v, kind)
    assert shape(v) == (2,)
    idx = ts.ensemble_indices
    assert shape(idx) == (2, 1, 1)
    assert all(0 <= int(i) < m1.num_mcmc_samples for i in flatten(idx))
    ts.redraw(2, ensemble_indices=[[[0]], [[3]]])
    assert [int(i) for i in flatten(ts.ensemble_indices)] == [0, 3]
    with pytest.raises(InvalidArgumentError, match="ensemble indices"):
        ts.redraw(2, ensemble_indices=[[[0]], [[m1.num_mcmc_samples]]])
    single = PathwiseThompsonSampling(SingleTaskGP(X, Y), seed=0)
    single(Xt)
    assert single.ensemble_indices is None
    with pytest.raises(ValueError, match="only used by ensemble models"):
        single.redraw(2, ensemble_indices=[[[0]], [[0]]])
    lst = ModelListGP(m1, m2)
    w = ScalarizedPosteriorTransform(as_kind([0.5, 0.5], kind))
    ts_list = PathwiseThompsonSampling(lst, posterior_transform=w, seed=1)
    assert shape(ts_list(Xt)) == (2,)
    assert shape(ts_list.ensemble_indices) == (2, 1, 2)
    # qLogNParEGO: one scalarization normalized by the ensemble's Y_baseline.
    sampler = SobolQMCNormalSampler(sample_shape=(8,), seed=0)
    par = qLogNParEGO(lst, X, scalarization_weights=[0.3, 0.7], sampler=sampler)
    assert shape(par.Y_baseline) == (len(X), 2)
    assert shape(par(Xt)) == (2,)
    pruned = qLogNParEGO(
        lst, X, scalarization_weights=[0.3, 0.7], sampler=sampler, prune_baseline=True
    )
    assert 1 <= shape(pruned.X_baseline)[0] <= len(X)
    with pytest.raises(UnsupportedError, match="objective or constraints"):
        qLogNParEGO(
            lst,
            X,
            sampler=sampler,
            prune_baseline=True,
            constraints=[lambda Z: Z[..., 0]],
        )
    # qJointEntropySearch: BoTorch's NotImplementedError.
    with pytest.raises(NotImplementedError, match="JES is not yet available"):
        qJointEntropySearch(m1, X[:2], [[0.0], [0.0]])


def test_bald_input_constructor_and_per_sample_best_f(kind: str) -> None:
    from botorch_rs.acquisition.input_constructors import (
        construct_inputs_BALD,
        get_acqf_input_constructor,
    )
    from botorch_rs.acquisition.logei import qLogExpectedImprovement

    model, X, Y = _fitted(seed=3)
    constructor = get_acqf_input_constructor(qBayesianActiveLearningByDisagreement)
    assert constructor.__closure__ is not None
    assert constructor.__closure__[1].cell_contents is construct_inputs_BALD
    Xp = as_kind([[0.2, 0.3]], kind)
    kw = constructor(model=model, X_pending=Xp)
    assert set(kw) == {"model", "X_pending", "sampler", "posterior_transform"}
    acqf = qBayesianActiveLearningByDisagreement(**kw)
    Xt = as_kind([[[0.1, 0.2]], [[0.5, 0.5]]], kind)
    assert shape(acqf(Xt)) == (2,)
    # A best_f with one value per MCMC sample applies sample by sample.
    n = model.num_mcmc_samples
    best = as_kind([0.1 * i for i in range(n)], kind)
    sampler = SobolQMCNormalSampler(sample_shape=(16,), seed=0)
    ei = qLogExpectedImprovement(model, best_f=best, sampler=sampler)
    assert [float(flatten(m.best_f)[0]) for m in ei._ensemble_acqfs] == pytest.approx(
        [0.1 * i for i in range(n)]
    )
    check_kind(ei.best_f, kind)
    assert shape(ei(Xt)) == (2,)


def test_get_fitted_map_saas_model() -> None:
    X, Y, _ = _data(seed=6)
    model = get_fitted_map_saas_model(X, Y)
    assert isinstance(model, SingleTaskGP)
    assert shape(model.posterior([[0.5, 0.5]]).mean) == (1, 1)


def test_nuts_samples_a_known_posterior() -> None:
    """Without data, the samples follow the priors: the mean ~ N(0, 1)."""
    model = _rs.PyroModel("matern", ([], [0, 2]), [])
    out = model.sample_nuts(200, 2000, seed=0)
    offsets, k = {}, 0
    for name, site_shape, _ in model.sites():
        offsets[name] = k
        k += math.prod(site_shape)
    # The mean ~ N(0, 1); the log lengthscales ~ N(sqrt(2) + log(2) / 2, 3).
    for site, mu0, var0 in (
        ("mean", 0.0, 1.0),
        ("lengthscale", math.sqrt(2) + math.log(2) / 2, 3.0),
    ):
        values = [z[offsets[site]] for z in out["samples"]]
        mu = sum(values) / len(values)
        var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
        assert abs(mu - mu0) < 0.15 * math.sqrt(var0), (site, mu)
        assert abs(var / var0 - 1.0) < 0.2, (site, var)
