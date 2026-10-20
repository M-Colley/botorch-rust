"""Actual cached-root / joint NEI samples, incumbents and native gradients."""

import warnings

import pytest
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement as RefLogNEI
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement as RefNEI
from botorch.acquisition.multi_objective.parego import qLogNParEGO as RefParEGO
from botorch.acquisition.objective import GenericMCObjective as RefObjective
from botorch.models import SingleTaskGP as RefGP
from botorch.sampling.normal import SobolQMCNormalSampler as RefSampler
from so_acquisition_helpers import rs_sampler, seeded_draws

from botorch_rs.acquisition.logei import qLogNoisyExpectedImprovement as LogNEI
from botorch_rs.acquisition.monte_carlo import qNoisyExpectedImprovement as NEI
from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO as ParEGO
from botorch_rs.acquisition.objective import GenericMCObjective as Objective
from botorch_rs.models import SingleTaskGP as GP
from botorch_rs.sampling.normal import SobolQMCNormalSampler as Sampler

DT = torch.float64
S = 32
pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def torch_factory_and_threads():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def models(outputs=1):
    x = torch.tensor([[0.0], [0.2], [0.55], [0.85], [1.0]], dtype=DT)
    y = torch.sin(5 * x)
    if outputs > 1:
        y = torch.cat([y, torch.cos(4 * x)], dim=-1)
    noise = torch.full_like(y, 0.03)
    return (
        RefGP(x, y, noise, outcome_transform=None).eval(),
        GP(x, y, noise, outcome_transform=None),
        x,
    )


def values_gradients(acqf, x):
    point = x.clone().requires_grad_(True)
    values = acqf(point)
    return values.detach(), torch.autograd.grad(values.sum(), point)[0]


@pytest.mark.parametrize("name", ["nei", "lognei", "parego"])
@pytest.mark.parametrize("cache_root", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_best_f_actual_forward_state_matches_reference_parallel_batches(
    name, cache_root, incremental
):
    multi = name == "parego"
    reference_model, model, baseline = models(outputs=2 if multi else 1)
    pending = torch.tensor([[0.42]], dtype=DT)
    if name == "nei" and incremental:
        pending = None
    full_baseline = len(baseline) + (
        len(pending) if pending is not None and incremental and name != "nei" else 0
    )
    q = 2
    total = len(baseline) + q + (0 if pending is None else len(pending))
    m = 2 if multi else 1
    kw = dict(X_baseline=baseline, prune_baseline=False, cache_root=cache_root, X_pending=pending)
    if name != "nei":
        kw["incremental"] = incremental
    if multi:
        kw["scalarization_weights"] = torch.tensor([0.35, 0.65], dtype=DT)
    ref_cls, cls = {
        "nei": (RefNEI, NEI),
        "lognei": (RefLogNEI, LogNEI),
        "parego": (RefParEGO, ParEGO),
    }[name]
    with seeded_draws(), warnings.catch_warnings():
        ref_sampler = RefSampler(torch.Size([S]), seed=0)
        reference = ref_cls(reference_model, sampler=ref_sampler, **kw)
        sampler = rs_sampler(S, [full_baseline * m, total * m])
        actual = cls(model, sampler=sampler, **kw)
        x = torch.linspace(0.06, 0.94, 12, dtype=DT).reshape(2, 3, q, 1)
        if cache_root:
            obj = torch.zeros(S, 2, 3, q, dtype=torch.float32)
            torch.testing.assert_close(
                actual.compute_best_f(obj), reference.compute_best_f(obj), atol=2e-6, rtol=2e-6
            )
        else:
            with pytest.raises(RuntimeError, match="completed forward"):
                actual.compute_best_f(torch.zeros(S, 2, 3, q, dtype=DT))
        values, gradients = values_gradients(actual, x)
        ref_values, ref_gradients = values_gradients(reference, x)
        torch.testing.assert_close(values, ref_values, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(gradients, ref_gradients, atol=2e-7, rtol=2e-7)
        obj = torch.zeros(S, 2, 3, q, dtype=DT)
        best, ref_best = actual.compute_best_f(obj), reference.compute_best_f(obj)
        torch.testing.assert_close(best, ref_best, atol=2e-10, rtol=2e-9)
        assert tuple(best.shape) == ((S, 1, 1) if cache_root else (S, 2, 3))
        assert actual._rs.cache_root is cache_root
        # A new pending set must rebuild the incremental cached root and
        # clear any previous uncached forward state.
        if name != "nei":
            new_pending = torch.tensor([[0.31], [0.69]], dtype=DT)
            new_n = len(baseline) + len(new_pending) + q
            sampler._rs.set_base_samples(
                (torch.randn(S, new_n * m).reshape(-1).tolist(), (S, new_n * m))
            )
            actual.set_X_pending(new_pending)
            if not cache_root:
                with pytest.raises(RuntimeError, match="completed forward"):
                    actual.compute_best_f(obj)
            assert torch.isfinite(actual(x)).all()


@pytest.mark.parametrize("cache_root", [False, True])
@pytest.mark.parametrize("fat", [False, True])
def test_lognei_constraints_and_changed_base_samples(cache_root, fat):
    ref_model, model, baseline = models(outputs=2)

    def objective(y, X=None):
        return y[..., 0]

    def constraint(y):
        return y[..., 1] - 0.1

    x = torch.tensor([[[0.34], [0.7]], [[0.44], [0.8]]], dtype=DT)
    dims = [len(baseline) * 2, (len(baseline) + 2) * 2]
    kwargs = dict(
        X_baseline=baseline,
        prune_baseline=False,
        cache_root=cache_root,
        constraints=[constraint],
        eta=0.125,
        fat=fat,
    )
    with seeded_draws():
        reference = RefLogNEI(
            ref_model,
            sampler=RefSampler(torch.Size([S]), seed=0),
            objective=RefObjective(objective),
            **kwargs,
        )
        sampler = rs_sampler(S, dims)
        actual = LogNEI(model, sampler=sampler, objective=Objective(objective), **kwargs)
        for changed in (False, True):
            if changed:
                base = reference.sampler.base_samples.detach().clone()
                base[..., -2:, :] *= -1.0
                reference.sampler.base_samples = base
                sampler.base_samples = base.reshape(S, -1)
            value, gradient = values_gradients(actual, x)
            expected, expected_gradient = values_gradients(reference, x)
            torch.testing.assert_close(value, expected, atol=2e-8, rtol=2e-8)
            torch.testing.assert_close(gradient, expected_gradient, atol=2e-7, rtol=2e-7)
            obj = torch.zeros(S, 2, 2, dtype=DT)
            torch.testing.assert_close(
                actual.compute_best_f(obj), reference.compute_best_f(obj), atol=2e-10, rtol=2e-9
            )


def test_uncached_contextual_objective_keeps_each_parallel_batch_incumbent():
    ref_model, model, baseline = models()

    def objective(samples, X=None):
        return samples[..., 0] + X[..., 0].mean(dim=-1, keepdim=True)

    x = torch.tensor([[[0.11], [0.23]], [[0.71], [0.93]]], dtype=DT)
    kwargs = dict(X_baseline=baseline, prune_baseline=False, cache_root=False)
    with seeded_draws():
        reference = RefLogNEI(
            ref_model,
            sampler=RefSampler(torch.Size([S]), seed=0),
            objective=RefObjective(objective),
            **kwargs,
        )
        actual = LogNEI(
            model,
            sampler=rs_sampler(S, [len(baseline) + 2]),
            objective=Objective(objective),
            **kwargs,
        )
        actual(x)
        reference(x)
        obj = torch.zeros(S, 2, 2, dtype=DT)
        best = actual.compute_best_f(obj)
        torch.testing.assert_close(best, reference.compute_best_f(obj), atol=2e-10, rtol=2e-9)
        assert not torch.equal(best[:, 0], best[:, 1])


def test_lognei_cache_default_and_explicit_disable_have_actual_different_draws():
    _, model, baseline = models()
    x = torch.tensor([[[0.43], [0.74]]], dtype=DT)
    cached = LogNEI(model, baseline, sampler=rs_sampler(S, [5, 7]), prune_baseline=False)
    joint = LogNEI(
        model, baseline, sampler=rs_sampler(S, [7]), prune_baseline=False, cache_root=False
    )
    assert cached._cache_root and cached._rs.cache_root
    assert not joint._cache_root and not joint._rs.cache_root
    assert not torch.equal(cached(x), joint(x))


def test_nonincremental_pending_preserves_cached_baseline_after_sampler_edits():
    _, model, baseline = models()
    sampler = rs_sampler(S, [5, 7])
    actual = LogNEI(model, baseline, sampler=sampler, prune_baseline=False, incremental=False)
    obj = torch.zeros(S, 1, 2, dtype=DT)
    original = actual.compute_best_f(obj)
    sampler.base_samples = torch.full((S, 5), 5.0, dtype=DT)
    actual.set_X_pending(torch.tensor([[0.49]], dtype=DT))
    torch.testing.assert_close(actual.compute_best_f(obj), original, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("cache_root", [False, True])
def test_best_f_restores_multidimensional_sample_shape(cache_root):
    ref_model, model, baseline = models()
    sample_shape = torch.Size([4, 8])
    sampler = Sampler(sample_shape, seed=0)
    source = rs_sampler(S, [5, 7])
    for dimension in (5, 7):
        sampler._rs.set_base_samples(source._rs.base_samples(dimension))
    kwargs = dict(X_baseline=baseline, prune_baseline=False, cache_root=cache_root)
    with seeded_draws():
        reference = RefLogNEI(ref_model, sampler=RefSampler(sample_shape, seed=0), **kwargs)
        actual = LogNEI(model, sampler=sampler, **kwargs)
        x = torch.tensor([[[0.34], [0.7]]], dtype=DT)
        actual(x)
        try:
            reference(x)
        except RuntimeError as error:
            # Official _update_base_samples subtracts one sample axis
            # regardless of sample_shape rank. Cached incumbents were
            # already computed correctly at construction, and remain the
            # exact public compute_best_f reference for this shape.
            assert cache_root and "shape" in str(error) and "invalid" in str(error)
        obj = torch.zeros(4, 8, 1, 2, dtype=DT)
        torch.testing.assert_close(
            actual.compute_best_f(obj), reference.compute_best_f(obj), atol=2e-10, rtol=2e-9
        )
