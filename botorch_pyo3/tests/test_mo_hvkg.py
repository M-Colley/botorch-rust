"""Fantasies, cost models and the hypervolume knowledge gradient through the
public API, for every array type: result types and shapes, conditioning
(equal to the model built on the augmented data), fantasy models, the
cost models and cost-aware utilities, the fidelity helpers, qHVKG / qMFHVKG
(values, gradients by finite differences and torch.autograd, pending points,
evaluation masks, one-shot helpers), ``optimize_acqf`` with one-shot
initial conditions, and errors. The comparisons with BoTorch are in
``test_parity_mo_hvkg.py`` and ``test_equivalence_mo_hvkg.py``.
"""

from __future__ import annotations

import copy
import functools
import math
import random
from typing import Any

import pytest

from botorch_rs.acquisition.cost_aware import (
    GenericCostAwareUtility,
    InverseCostWeightedUtility,
)
from botorch_rs.acquisition.multi_objective import (
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    _split_hvkg_fantasy_points,
)
from botorch_rs.acquisition.utils import expand_trace_observations, project_to_target_fidelity
from botorch_rs.exceptions import BotorchError, DeprecationError, UnsupportedError
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.cost import AffineFidelityCostModel, FixedCostModel
from botorch_rs.optim import gen_one_shot_hvkg_initial_conditions, optimize_acqf
from botorch_rs.sampling import ListSampler, SobolQMCNormalSampler
from helpers import HAS_TORCH, as_kind, assert_close, check_kind, flatten, shape

# Models without outcome transforms (to compare with refitted models).
pytestmark = pytest.mark.filterwarnings("ignore::botorch_rs.exceptions.InputDataWarning")


def close(a: Any, b: Any, tol: float) -> None:
    assert_close(a, b, rtol=tol, atol=tol)


def _data(n: int, d: int, seed: int, offset: float = 0.0) -> tuple[list, list, list]:
    rng = random.Random(seed)
    X = [[rng.random() for _ in range(d)] for _ in range(n)]
    Y = [[math.sin(3.0 * x[0] + offset) + 0.5 * sum(x[1:]) + 0.02 * rng.gauss(0, 1)] for x in X]
    Yvar = [[0.01 + 0.01 * rng.random()] for _ in X]
    return X, Y, Yvar


def _gp(seed: int, d: int = 2, fixed: bool = False, offset: float = 0.0) -> SingleTaskGP:
    X, Y, Yvar = _data(9, d, seed, offset)
    gp = SingleTaskGP(X, Y, train_Yvar=Yvar if fixed else None)
    kwargs: dict[str, Any] = {
        "covar_module.lengthscale": [[0.4 + 0.1 * k for k in range(d)]],
    }
    if not fixed:
        kwargs["likelihood.noise"] = [0.02]
    gp.initialize(**kwargs)
    gp.eval()
    return gp


def _list(d: int = 2) -> ModelListGP:
    return ModelListGP(_gp(1, d), _gp(2, d, fixed=True, offset=1.5))


def _points(n: int, d: int, seed: int) -> list:
    rng = random.Random(seed)
    return [[rng.random() for _ in range(d)] for _ in range(n)]


def _moments(model: Any, X: Any) -> tuple[list[float], list[float]]:
    post = model.posterior(X)
    return flatten(post.mean), flatten(post.distribution.covariance_matrix)


# ---------------------------------------------------------------------------
# Conditioning and fantasies
# ---------------------------------------------------------------------------


def test_condition_on_observations_equals_refit(kind: str) -> None:
    X, Y, Yvar = _data(9, 2, 1)
    gp = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), train_Yvar=as_kind(Yvar, kind))
    gp.initialize(**{"covar_module.lengthscale": [[0.4, 0.5]]})
    gp.eval()
    Xn, Yn, Vn = _points(3, 2, 5), [[0.3], [1.1], [-0.2]], [[0.02], [0.01], [0.03]]
    cm = gp.condition_on_observations(as_kind(Xn, kind), as_kind(Yn, kind), noise=as_kind(Vn, kind))
    assert isinstance(cm, SingleTaskGP)
    Xt = as_kind(_points(4, 2, 6), kind)
    mean, cov = _moments(cm, Xt)
    check_kind(cm.posterior(Xt).mean, kind)
    # The Standardize transform of the conditioned model is the original one;
    # without it, conditioning equals building the model on all points.
    plain = SingleTaskGP(X, Y, train_Yvar=Yvar, outcome_transform=None)
    plain.initialize(**{"covar_module.lengthscale": [[0.4, 0.5]], "mean_module.constant": 0.7})
    plain.eval()
    cm_plain = plain.condition_on_observations(Xn, Yn, noise=Vn)
    refit = SingleTaskGP(X + Xn, Y + Yn, train_Yvar=Yvar + Vn, outcome_transform=None)
    refit.initialize(**{"covar_module.lengthscale": [[0.4, 0.5]], "mean_module.constant": 0.7})
    refit.eval()
    a, e = _moments(cm_plain, Xt), _moments(refit, Xt)
    close(a[0], e[0], tol=1e-10)
    close(a[1], e[1], tol=1e-10)
    assert len(mean) == 4 and len(cov) == 16


def test_condition_on_observations_batched(kind: str) -> None:
    gp = _gp(3)
    Xn = as_kind([_points(2, 2, 7), _points(2, 2, 8)], kind)
    Yn = as_kind([[[0.1], [0.2]], [[0.4], [-0.3]]], kind)
    cm = gp.condition_on_observations(Xn, Yn)
    assert tuple(cm.batch_shape) == (2,)
    post = cm.posterior(as_kind(_points(3, 2, 9), kind))
    assert shape(post.mean) == (2, 3, 1)
    # Batched targets of unbatched inputs.
    cm = gp.condition_on_observations(as_kind(_points(2, 2, 7), kind), Yn)
    assert tuple(cm.batch_shape) == (2,)


def test_fantasize_shapes_and_mean(kind: str) -> None:
    gp = _gp(4)
    Xn = as_kind(_points(2, 2, 10), kind)
    sampler = SobolQMCNormalSampler(sample_shape=(5,), seed=1)
    fm = gp.fantasize(Xn, sampler)
    assert tuple(fm.batch_shape) == (5,)
    Xt = as_kind(_points(3, 2, 11), kind)
    post = fm.posterior(Xt)
    check_kind(post.mean, kind)
    assert shape(post.mean) == (5, 3, 1)
    # Batched inputs: n_f x batch.
    Xb = as_kind([_points(2, 2, 12), _points(2, 2, 13)], kind)
    assert tuple(gp.fantasize(Xb, sampler).batch_shape) == (5, 2)
    # Antithetic base samples: the fantasy means average to the posterior mean.
    s = SobolQMCNormalSampler(sample_shape=(2,), seed=0)
    z = [[0.3, -1.2], [-0.3, 1.2]]
    s.base_samples = as_kind(z, kind) if kind != "list" else z
    fm = gp.fantasize(Xn, s)
    means = flatten(fm.posterior(Xt).mean)
    avg = [(means[i] + means[3 + i]) / 2 for i in range(3)]
    close(avg, flatten(gp.posterior(Xt).mean), tol=1e-10)


def test_fantasize_errors() -> None:
    gp = _gp(4)
    with pytest.raises(DeprecationError):
        gp.fantasize(_points(2, 2, 1), SobolQMCNormalSampler((2,)), observation_noise=True)
    with pytest.raises(UnsupportedError):
        gp.fantasize(_points(2, 2, 1), object())
    fixed = _gp(5, fixed=True)
    with pytest.raises(RuntimeError, match="noise"):
        fixed.condition_on_observations(_points(1, 2, 1), [[0.0]])
    gp.train()
    with pytest.raises(RuntimeError, match="Fantasy observations"):
        gp.condition_on_observations(_points(1, 2, 1), [[0.0]])


def test_model_list_fantasize(kind: str) -> None:
    lst = _list()
    Xn = as_kind(_points(3, 2, 14), kind)
    samplers = ListSampler(*[SobolQMCNormalSampler((4,), seed=k) for k in range(2)])
    fm = lst.fantasize(Xn, samplers)
    assert isinstance(fm, ModelListGP)
    post = fm.posterior(as_kind(_points(2, 2, 15), kind))
    assert shape(post.mean) == (4, 2, 2)
    mask = [[True, False], [True, True], [False, True]]
    fm = lst.fantasize(
        Xn, samplers, evaluation_mask=as_kind(mask, kind) if kind == "torch" else mask
    )
    assert shape(fm.posterior(as_kind(_points(2, 2, 15), kind)).mean) == (4, 2, 2)
    # A shared sampler.
    fm = lst.fantasize(Xn, SobolQMCNormalSampler((3,), seed=2))
    assert shape(fm.posterior(as_kind(_points(2, 2, 15), kind)).mean) == (3, 2, 2)
    # Conditioning on per-output inputs.
    cm = lst.condition_on_observations(
        [as_kind(_points(2, 2, 16), kind), as_kind(_points(2, 2, 17), kind)],
        as_kind([[0.1, 0.2], [0.3, 0.4]], kind),
        noise=as_kind([[0.01, 0.01], [0.02, 0.02]], kind),
    )
    assert shape(cm.posterior(as_kind(_points(2, 2, 15), kind)).mean) == (2, 2)


def test_list_sampler() -> None:
    with pytest.raises(UnsupportedError):
        ListSampler(SobolQMCNormalSampler((2,)), SobolQMCNormalSampler((3,)))
    with pytest.raises(UnsupportedError):
        ListSampler(object())  # type: ignore[arg-type]
    s = ListSampler(SobolQMCNormalSampler((2,)), SobolQMCNormalSampler((2,)))
    assert s.sample_shape == (2,)
    assert "ListSampler" in repr(s)


# ---------------------------------------------------------------------------
# Cost models, utilities and fidelity helpers
# ---------------------------------------------------------------------------


def test_cost_models(kind: str) -> None:
    X = as_kind([[[0.1, 0.5], [0.3, 1.0]]], kind)
    affine = AffineFidelityCostModel(fidelity_weights={1: 2.0}, fixed_cost=0.5)
    out = affine(X)
    check_kind(out, kind)
    close(flatten(out), [1.5, 2.5], tol=1e-15)
    assert affine.num_outputs == 1 and affine.fidelity_dims == [1]
    fixed = FixedCostModel(as_kind([1.0, 3.0], kind))
    close(flatten(fixed(X)), [1.0, 3.0, 1.0, 3.0], tol=0.0)
    assert fixed.num_outputs == 2
    close(flatten(affine.posterior(X).mean), [1.5, 2.5], tol=1e-15)
    assert "AffineFidelityCostModel" in repr(affine)


def test_inverse_cost_weighted_utility(kind: str) -> None:
    X = as_kind([[[0.1, 0.5], [0.3, 1.0]]], kind)
    cm = AffineFidelityCostModel(fidelity_weights={1: 2.0}, fixed_cost=0.5)
    deltas = as_kind([[2.0], [-1.0]], kind)
    u = InverseCostWeightedUtility(cm)
    out = u(X, deltas)
    check_kind(out, kind)
    # Cost of the candidate set: 1.5 + 2.5 = 4.
    close(flatten(out), [0.5, -4.0], tol=1e-15)
    ulog = InverseCostWeightedUtility(cm, log=True)
    close(flatten(ulog(X, deltas)), [2.0 - math.log(4.0), -1.0 - math.log(4.0)], tol=1e-15)
    with pytest.raises(RuntimeError, match="sampler"):
        InverseCostWeightedUtility(cm, use_mean=False)(X, deltas)
    # GP cost model (posterior mean).
    gp = _gp(6)
    out = InverseCostWeightedUtility(gp)(X, deltas)
    assert shape(out) == (2, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_cost_utilities_autograd() -> None:
    import torch

    X = torch.tensor([[[0.1, 0.5], [0.3, 1.0]]], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([[2.0], [-1.0]], dtype=torch.float64, requires_grad=True)
    cm = AffineFidelityCostModel(fidelity_weights={1: 2.0}, fixed_cost=0.5)
    out = InverseCostWeightedUtility(cm)(X, d)
    gx, gd = torch.autograd.grad(out.sum(), (X, d))
    # d/dX of 2/c - 1*c with c = 1 + 2 (x01 + x11): -2/c^2 * 2 - 2.
    close(gx[0, :, 1].tolist(), [-2 * 2 / 16 - 2] * 2, tol=1e-14)
    close(gx[0, :, 0].tolist(), [0.0, 0.0], tol=0.0)
    close(gd.reshape(-1).tolist(), [0.25, 4.0], tol=1e-15)
    generic = GenericCostAwareUtility(lambda X, deltas: deltas / X[..., -1].sum(-1))
    close(flatten(generic(X.detach(), d.detach())), [2.0 / 1.5, -1.0 / 1.5], tol=1e-15)


def test_fidelity_helpers(kind: str) -> None:
    X = as_kind([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], kind)
    out = project_to_target_fidelity(X, target_fidelities={-1: 1.0}, d=3)
    check_kind(out, kind)
    assert flatten(out) == [0.1, 0.2, 1.0, 0.4, 0.5, 1.0]
    out = project_to_target_fidelity(as_kind([[0.1, 0.2]], kind), target_fidelities={1: 0.5}, d=3)
    assert flatten(out) == [0.1, 0.5, 0.2]
    with pytest.raises(Exception, match="out of bounds"):
        project_to_target_fidelity(X, target_fidelities={3: 1.0}, d=3)
    out = expand_trace_observations(X, num_trace_obs=2)
    assert shape(out) == (1, 6, 3)
    close(flatten(out)[-1], [0.6 / 3], tol=1e-15)
    assert expand_trace_observations(X) is X


# ---------------------------------------------------------------------------
# The hypervolume knowledge gradient
# ---------------------------------------------------------------------------

REF = [-1.5, -1.0]


def _hvkg(**kwargs: Any) -> qHypervolumeKnowledgeGradient:
    kwargs.setdefault("num_fantasies", 3)
    kwargs.setdefault("num_pareto", 2)
    kwargs.setdefault(
        "sampler",
        ListSampler(*[SobolQMCNormalSampler((kwargs["num_fantasies"],), seed=k) for k in range(2)]),
    )
    return qHypervolumeKnowledgeGradient(_list(), ref_point=REF, **kwargs)


def test_hvkg_values(kind: str) -> None:
    acq = _hvkg()
    assert acq.get_augmented_q_batch_size(2) == 2 + 6
    X = as_kind([_points(1 + 6, 2, 20), _points(1 + 6, 2, 21)], kind)
    out = acq(X)
    check_kind(out, kind)
    assert shape(out) == (2,)
    assert all(v >= 0 for v in flatten(out))
    # t-batches are independent.
    single = flatten(acq(as_kind([_points(1 + 6, 2, 21)], kind)))
    close(flatten(out)[1:], single, tol=1e-12)
    cands = acq.extract_candidates(X)
    assert shape(cands) == (2, 1, 2)
    actual, fantasies = _split_hvkg_fantasy_points(X, 3, 2)
    assert shape(actual) == (2, 1, 2) and shape(fantasies) == (3, 2, 2, 2)
    assert flatten(actual) == flatten(cands)
    # log and current value: log(exp(value) - exp(current_value)).
    X1 = as_kind([_points(7, 2, 20)], kind)
    log_value = flatten(_hvkg(log=True)(X1))[0]
    cv = log_value - 1.0
    expected = log_value + math.log1p(-math.exp(cv - log_value))
    close(flatten(_hvkg(log=True, current_value=cv)(X1)), [expected], tol=1e-12)
    # The log value function smooths the hypervolume (qLogEHVI): close to log.
    close([log_value], [math.log(flatten(acq(X1))[0])], tol=1e-2)
    assert "num_fantasies=3" in repr(acq)


def test_hvkg_gradients_finite_differences() -> None:
    acq = _hvkg(use_posterior_mean=False, inner_sampler=SobolQMCNormalSampler((8,), seed=4))
    X = _points(2 + 6, 2, 22)
    if HAS_TORCH:
        import torch

        Xt = torch.tensor([X], dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(acq(Xt).sum(), Xt)
        g = grad.reshape(-1).tolist()
        h = 1e-6
        flat = [v for row in X for v in row]
        for j in range(0, len(flat), 3):
            up, down = list(flat), list(flat)
            up[j] += h
            down[j] -= h
            fu = flatten(acq([[up[i : i + 2] for i in range(0, len(up), 2)]]))[0]
            fd = flatten(acq([[down[i : i + 2] for i in range(0, len(down), 2)]]))[0]
            assert abs((fu - fd) / (2 * h) - g[j]) <= 1e-5 * max(1.0, max(map(abs, g)))


def test_hvkg_pending_and_masks(kind: str) -> None:
    X = as_kind([_points(2 + 6, 2, 23)], kind)
    mask = [[True, False], [False, True]]
    acq = _hvkg(X_evaluation_mask=as_kind(mask, kind) if kind == "torch" else mask)
    out_masked = flatten(acq(X))
    full = flatten(_hvkg()(X))
    assert out_masked[0] != full[0]
    em = acq.construct_evaluation_mask(X)
    assert flatten(em) == [1.0, 0.0, 0.0, 1.0]
    pending = as_kind(_points(1, 2, 24), kind)
    acq_p = _hvkg(X_pending=pending)
    v_p = flatten(acq_p(X))
    acq.set_X_pending(None)
    acq_p.set_X_pending(None)
    close(flatten(acq_p(X)), full, tol=1e-14)
    acq_p.set_X_pending(pending)
    close(flatten(acq_p(X)), v_p, tol=1e-14)
    acq_p.set_X_pending(pending, [[True, False]])
    assert acq_p.X_pending_evaluation_mask == [[True, False]]


def test_hvkg_cost_and_multi_fidelity(kind: str) -> None:
    cm = AffineFidelityCostModel(fidelity_weights={-1: 1.0}, fixed_cost=0.2)
    project = functools.partial(project_to_target_fidelity, target_fidelities={-1: 1.0}, d=2)
    samplers = ListSampler(*[SobolQMCNormalSampler((2,), seed=k) for k in range(2)])
    acq = qMultiFidelityHypervolumeKnowledgeGradient(
        _list(),
        ref_point=REF,
        target_fidelities={-1: 1.0},
        num_fantasies=2,
        num_pareto=2,
        sampler=samplers,
        current_value=0.1,
        cost_aware_utility=InverseCostWeightedUtility(cm),
        project=project,
    )
    X = as_kind([_points(1 + 4, 2, 25)], kind)
    out = acq(X)
    check_kind(out, kind)
    assert acq.target_fidelities == {-1: 1.0}
    with pytest.raises(NotImplementedError, match="Trace observations"):
        qMultiFidelityHypervolumeKnowledgeGradient(
            _list(), ref_point=REF, target_fidelities={-1: 1.0}, expand=lambda X: X
        )
    with pytest.raises(UnsupportedError, match="current_value"):
        _hvkg(cost_aware_utility=InverseCostWeightedUtility(cm))


def test_hvkg_value_function(kind: str) -> None:
    vf = _get_hv_value_function(_list(), ref_point=REF, use_posterior_mean=True)
    out = vf(as_kind([_points(3, 2, 26)], kind))
    check_kind(out, kind)
    vf_s = _get_hv_value_function(
        _list(), ref_point=REF, sampler=SobolQMCNormalSampler((16,), seed=1)
    )
    assert shape(vf_s(as_kind([_points(3, 2, 26)], kind))) == (1,)


def test_hvkg_errors() -> None:
    with pytest.raises(ValueError, match="requires using a ModelList"):
        qHypervolumeKnowledgeGradient(_gp(1), ref_point=REF)
    with pytest.raises(ValueError, match="sampler shape"):
        _hvkg(
            num_fantasies=3,
            sampler=ListSampler(SobolQMCNormalSampler((2,)), SobolQMCNormalSampler((2,))),
        )
    with pytest.raises(UnsupportedError):
        qHypervolumeKnowledgeGradient(_list(), ref_point=REF, inner_sampler=object())
    with pytest.raises(ValueError, match="must be less than"):
        _split_hvkg_fantasy_points(_points(3, 2, 1), 2, 2)
    with pytest.raises(BotorchError, match="sequential"):
        optimize_acqf(
            _hvkg(), [[0.0, 0.0], [1.0, 1.0]], q=2, num_restarts=2, raw_samples=4, sequential=True
        )


def test_hvkg_samplers_copy() -> None:
    acq = _hvkg()
    cs = acq.cost_sampler
    assert cs is not acq.sampler and cs.sample_shape == acq.sampler.sample_shape
    assert copy.deepcopy(acq.sampler).sample_shape == (3,)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_optimize_hvkg(kind: str) -> None:
    acq = _hvkg(num_fantasies=2, num_pareto=2)
    bounds = as_kind([[0.0, 0.0], [1.0, 1.0]], kind)
    opts = {"seed": 0, "num_inner_restarts": 3, "raw_inner_samples": 32, "maxiter": 50}
    cand, value = optimize_acqf(acq, bounds, q=1, num_restarts=3, raw_samples=32, options=opts)
    check_kind(cand, kind)
    assert shape(cand) == (1, 2)
    full, value_full = optimize_acqf(
        acq, bounds, q=1, num_restarts=3, raw_samples=32, options=opts, return_full_tree=True
    )
    assert shape(full) == (5, 2)
    close(flatten(value_full), flatten(value), tol=0.0)
    assert flatten(full)[:2] == flatten(cand)
    ics = gen_one_shot_hvkg_initial_conditions(
        acq, bounds, q=1, num_restarts=4, raw_samples=32, options=opts
    )
    check_kind(ics, kind)
    assert shape(ics) == (4, 5, 2)
    assert all(0.0 <= v <= 1.0 for v in flatten(ics))
    # The same seed gives the same initial conditions.
    again = gen_one_shot_hvkg_initial_conditions(
        acq, bounds, q=1, num_restarts=4, raw_samples=32, options=opts
    )
    assert flatten(again) == flatten(ics)
    with pytest.raises(ValueError, match="frac_random"):
        gen_one_shot_hvkg_initial_conditions(
            acq, bounds, q=1, num_restarts=4, raw_samples=32, options={"frac_random": 1.5}
        )
