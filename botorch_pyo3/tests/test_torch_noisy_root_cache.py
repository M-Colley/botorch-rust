"""Cached Schur draws retain the complete device Gaussian probability law."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import importlib
import warnings

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as _rs
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FAMILIES = [
    "qNoisyExpectedImprovement",
    "qLogNoisyExpectedImprovement",
    "qNoisyExpectedHypervolumeImprovement",
    "qLogNoisyExpectedHypervolumeImprovement",
]


def _acquisition(name):
    prefix = "multi_objective." if "Hypervolume" in name else ""
    module = "logei" if name.startswith("qLog") else "monte_carlo"
    return getattr(importlib.import_module(f"botorch_rs.acquisition.{prefix}{module}"), name)


def _model(device, outputs=1, live=True):
    X = torch.tensor([[0.04], [0.32], [0.69], [0.94]], dtype=torch.double, device=device)
    Y = torch.cat((X.sin(), (2 * X).cos()), -1)[..., :outputs].clone()
    noise = torch.full_like(Y, 0.025)
    X.requires_grad_(live)
    Y.requires_grad_(live)
    noise.requires_grad_(live)
    model = SingleTaskGP(X, Y, noise, outcome_transform=None, backend="torch")
    if not live:
        model.requires_grad_(False)
    return model, (X, Y, noise)


def _make(name, model, baseline, cached=True, **kwargs):
    options = dict(
        X_baseline=baseline,
        cache_root=cached,
        prune_baseline=False,
        sampler=IIDNormalSampler((12,), seed=43),
        **kwargs,
    )
    if "Hypervolume" in name:
        options["ref_point"] = [-0.2, -0.2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _acquisition(name)(model=model, **options)


class DenseNormalSampler(IIDNormalSampler):
    """Independent dense Cholesky oracle, with common explicit normal draws."""

    def forward(self, posterior):
        self._construct_tensor_base_samples(posterior)
        q, m = posterior.mean.shape[-2:]
        batch = posterior.mean.shape[:-2]
        z = self.base_samples.reshape(*self.sample_shape, q, m).transpose(-1, -2)
        z = z.reshape(*self.sample_shape, *((1,) * len(batch)), m * q, 1)
        root = torch.linalg.cholesky(posterior.distribution.covariance_matrix)
        draw = (root @ z).squeeze(-1).reshape(*self.sample_shape, *batch, m, q)
        return posterior.mean + draw.transpose(-1, -2)

    __call__ = forward


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", FAMILIES)
def test_cached_values_all_first_and_second_gradients_match_dense_law(device, name):
    outputs = 2 if "Hypervolume" in name else 1
    model, data = _model(device, outputs)
    baseline = data[0].detach()[[0, 2]].clone().requires_grad_()
    query = data[0].new_tensor([[[0.21], [0.53]], [[0.45], [0.84]]], requires_grad=True)
    actual = _make(name, model, baseline, detach_baseline=False)
    expected = _make(name, model, baseline, False, detach_baseline=False)
    expected.sampler = DenseNormalSampler((12,), seed=43)
    if outputs == 2:
        expected._observed_sampler = expected._new_sampler()
    leaves = (query, baseline, *data, *model.parameters())
    # Repeat complete independent backward calls against the same acquisition.
    # A cached non-leaf baseline Cholesky would fail on the second call.
    for _ in range(2):
        value = actual(query)
        expected.sampler.base_samples = actual.sampler.base_samples.clone()
        oracle = expected(query)
        assert actual._last_root_cache_used and actual._baseline_L is not None
        torch.testing.assert_close(value, oracle, atol=2e-10, rtol=2e-9)
        first = torch.autograd.grad(value.sum(), leaves, create_graph=True)
        oracle_first = torch.autograd.grad(oracle.sum(), leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
        oracle_second = torch.autograd.grad(sum(g.square().sum() for g in oracle_first), leaves)
        for a, b in zip((*first, *second), (*oracle_first, *oracle_second), strict=True):
            assert a.device == query.device and bool(torch.isfinite(a).all())
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
        assert not actual._tensor_root_cache.entries  # no retained live parameter/data graph


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", FAMILIES)
def test_frozen_root_reuse_public_mutations_and_pending_prefixes(device, name, monkeypatch):
    outputs = 2 if "Hypervolume" in name else 1
    model, data = _model(device, outputs, live=False)
    baseline = data[0][[0, 2]].clone()
    kwargs = (
        (dict(incremental=True) if name == "qLogNoisyExpectedImprovement" else {})
        if outputs == 1
        else dict(cache_pending=True, max_iep=0)
    )
    actual = _make(name, model, baseline, **kwargs)
    actual.sampler = SobolQMCNormalSampler((12,), seed=43)
    if outputs == 1:
        from botorch_rs.acquisition._torch_root_cache import new_sampler

        actual.base_sampler = new_sampler(actual.sampler)
    else:
        actual._observed_sampler = actual._new_sampler()
    query = data[0].new_tensor([[0.24], [0.55]], requires_grad=True)
    calls = []
    original = _rs.tensor_psd_safe_cholesky

    def record(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(_rs, "tensor_psd_safe_cholesky", record)
    value = actual(query)
    root = actual._baseline_L
    first_calls = len(calls)
    again = actual(query)
    assert actual._baseline_L.data_ptr() == root.data_ptr()
    torch.testing.assert_close(again, value)
    # Only the candidate Schur block is factored on another query.
    assert [shape[-2:] for shape in calls[first_calls:]] == [(2, 2)]
    torch.autograd.grad(again.sum(), query)
    # Public hyperparameter/data writes must invalidate even frozen caches.
    for mutate in (
        lambda: setattr(model.covar_module, "lengthscale", model.covar_module.lengthscale * 1.1),
        lambda: setattr(model.mean_module, "constant", model.mean_module.constant + 0.03),
        lambda: model._train_Y.add_(0.01),
        lambda: baseline.add_(0.005),
    ):
        with torch.no_grad():
            mutate()
        actual(query)
        assert actual._baseline_L.data_ptr() != root.data_ptr()
        root = actual._baseline_L
    initial_normals = actual.base_sampler.base_samples.clone()
    for pending in (
        data[0].new_tensor([[0.77]]),
        data[0].new_tensor([[0.77], [0.86]]),
        data[0].new_tensor([[0.61]]),
        None,
    ):
        actual.set_X_pending(pending)
        result = actual(query)
        assert bool(torch.isfinite(result).all())
        torch.autograd.grad(result.sum(), query)
        axis = -2 if outputs > 1 else -1
        retained = actual.base_sampler.base_samples.narrow(axis, 0, 2)
        torch.testing.assert_close(retained.reshape_as(initial_normals), initial_normals)
        assert len(actual._tensor_root_cache.entries) <= 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.usefixtures("reference_python_logehvi")
def test_cached_reference_values_query_gradient_and_directional_hessian(device, name):
    pytest.importorskip("botorch")
    import gpytorch
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.sampling.normal import IIDNormalSampler as ReferenceSampler

    outputs = 2 if "Hypervolume" in name else 1
    _, (X, Y, noise) = _model(device, outputs, live=False)
    batch = torch.Size((outputs,)) if outputs > 1 else torch.Size()
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1, batch_shape=batch).to(X)
    mean = gpytorch.means.ConstantMean(batch_shape=batch).to(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = ReferenceGP(
            X, Y, noise, outcome_transform=None, covar_module=kernel, mean_module=mean
        )
        model = SingleTaskGP(
            X,
            Y,
            noise,
            outcome_transform=None,
            backend="torch",
            covar_module=copy.deepcopy(kernel),
            mean_module=copy.deepcopy(mean),
        )
    model.requires_grad_(False)
    reference.requires_grad_(False)
    baseline = X[[0, 2]]
    actual = _make(name, model, baseline)
    prefix = "multi_objective." if outputs > 1 else ""
    module = "logei" if name.startswith("qLog") else "monte_carlo"
    cls = getattr(importlib.import_module(f"botorch.acquisition.{prefix}{module}"), name)
    options = dict(
        X_baseline=baseline,
        cache_root=True,
        prune_baseline=False,
        sampler=ReferenceSampler(torch.Size((12,)), seed=43),
    )
    if outputs > 1:
        options["ref_point"] = [-0.2, -0.2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = cls(reference, **options)
    query = X.new_tensor([[[0.21], [0.53]], [[0.45], [0.84]]], requires_grad=True)
    value, oracle = actual(query), expected(query)
    torch.testing.assert_close(value, oracle, rtol=2e-8, atol=2e-9)
    first = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    oracle_first = torch.autograd.grad(oracle.sum(), query)[0]
    torch.testing.assert_close(first, oracle_first, rtol=2e-7, atol=2e-8)
    direction = torch.linspace(
        -0.3, 0.6, query.numel(), device=device, dtype=query.dtype
    ).reshape_as(query)
    hessian = torch.autograd.grad((first * direction).sum(), query)[0]
    # The pinned linear_operator triangular custom VJP detaches matrix Hessians.
    # Differentiate the unmodified source query gradient by finite differences.
    epsilon = 1e-5
    sides = []
    for sign in (-1, 1):
        point = (query.detach() + sign * epsilon * direction).requires_grad_()
        sides.append(torch.autograd.grad(expected(point).sum(), point)[0])
    difference = (sides[1] - sides[0]) / (2 * epsilon)
    torch.testing.assert_close(hessian, difference, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_scalarization_weights_keep_a_fresh_root_graph(device):
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform

    model, data = _model(device, 2, live=False)
    weights = data[0].new_tensor([0.6, -0.3], requires_grad=True)
    transform = ScalarizedPosteriorTransform(weights)
    baseline = data[0][[0, 2]]
    actual = _make(
        "qLogNoisyExpectedImprovement",
        model,
        baseline,
        posterior_transform=transform,
        detach_baseline=False,
    )
    expected = _make(
        "qLogNoisyExpectedImprovement",
        model,
        baseline,
        False,
        posterior_transform=transform,
        detach_baseline=False,
    )
    query = data[0].new_tensor([[0.21], [0.53]], requires_grad=True)
    for _ in range(2):
        value = actual(query)
        expected.sampler.base_samples = actual.sampler.base_samples.clone()
        oracle = expected(query)
        gradients = torch.autograd.grad(value.sum(), (query, weights), create_graph=True)
        reference = torch.autograd.grad(oracle.sum(), (query, weights), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), (query, weights))
        reference_second = torch.autograd.grad(
            sum(g.square().sum() for g in reference), (query, weights)
        )
        for a, b in zip(
            (value, *gradients, *second), (oracle, *reference, *reference_second), strict=True
        ):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", FAMILIES)
def test_failed_schur_update_falls_back_to_full_samples(device, name, monkeypatch):
    from botorch_rs.acquisition import _torch_root_cache
    from botorch_rs.exceptions.errors import NotPSDError
    from botorch_rs.exceptions.warnings import BotorchWarning

    outputs = 2 if "Hypervolume" in name else 1
    model, data = _model(device, outputs, live=False)
    baseline = data[0][[0, 2]]
    actual = _make(name, model, baseline)
    expected = _make(name, model, baseline, False)
    query = data[0].new_tensor([[0.21], [0.53]], requires_grad=True)

    def fail(*args, **kwargs):
        raise NotPSDError("Deliberately ill-conditioned Schur block")

    monkeypatch.setattr(_torch_root_cache, "sample_cached_cholesky", fail)
    with pytest.warns(BotorchWarning, match="falling back to full-joint"):
        value = actual(query)
    expected.sampler.base_samples = actual.sampler.base_samples.clone()
    torch.testing.assert_close(value, expected(query), rtol=2e-9, atol=2e-10)
    assert actual._cache_root and not actual._last_root_cache_used
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert bool(torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name,incremental",
    [
        (FAMILIES[1], True),
        (FAMILIES[2], True),
        (FAMILIES[2], False),
        (FAMILIES[3], True),
        (FAMILIES[3], False),
    ],
)
def test_cached_pending_batch_transitions_match_independent_decisions(device, name, incremental):
    outputs = 2 if "Hypervolume" in name else 1
    model, data = _model(device, outputs)
    baseline = data[0].detach()[[0, 2]]
    query = data[0].new_tensor([[[0.23]], [[0.52]]], requires_grad=True)
    pending = data[0].new_tensor([[[0.77]], [[0.88]]])
    options = (
        dict(incremental=True)
        if outputs == 1
        else dict(cache_pending=True, max_iep=0, incremental_nehvi=incremental)
    )
    actual = _make(name, model, baseline, X_pending=pending, **options)
    independent = [
        _make(name, model, baseline, X_pending=pending[row], **options) for row in range(2)
    ]
    transitions = [pending, torch.cat((pending, pending - 0.1), -2), pending + 0.01, None]
    for index, points in enumerate(transitions):
        if index:
            actual.set_X_pending(points)
            for row, acq in enumerate(independent):
                acq.set_X_pending(None if points is None else points[row])
        value = actual(query)
        oracle = torch.cat([acq(query[row]) for row, acq in enumerate(independent)])
        torch.testing.assert_close(value, oracle, rtol=2e-8, atol=2e-9)
        first = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
        reference = torch.autograd.grad(oracle.sum(), query, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), query)[0]
        reference_second = torch.autograd.grad(reference.square().sum(), query)[0]
        for a, b in zip((first, second), (reference, reference_second), strict=True):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_dtype_migration_rebuilds_roots_and_normal_coordinates(device):
    model, data = _model(device, live=False)
    baseline = data[0][[0, 2]]
    actual = _make("qLogNoisyExpectedImprovement", model, baseline)
    query = data[0].new_tensor([[0.21]], requires_grad=True)
    actual(query)
    actual.to(dtype=torch.float32)
    point = query.detach().float().requires_grad_()
    result = actual(point)
    assert actual._baseline_L.dtype == actual.base_sampler.base_samples.dtype == torch.float32
    assert actual._baseline_L.device == point.device
    first = torch.autograd.grad(result.sum(), point, create_graph=True)[0]
    assert bool(torch.isfinite(torch.autograd.grad(first.square().sum(), point)[0]).all())


@pytest.mark.parametrize("device", DEVICES)
def test_correlated_outputs_warn_and_disable_unsupported_root_cache(device):
    from botorch_rs.models._torch_backend import TorchSingleTaskGP
    from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

    class CorrelatedModel(TorchSingleTaskGP):
        def posterior(self, *args, **kwargs):
            posterior = super().posterior(*args, **kwargs)
            covariance = posterior.distribution.covariance_matrix
            return TorchGPyTorchPosterior(posterior.mean, covariance + 0.003)

    _, (X, Y, noise) = _model(device, 2, live=False)
    model = CorrelatedModel(X, Y, noise, outcome_transform=None)
    actual = _make("qLogNoisyExpectedHypervolumeImprovement", model, X[[0, 2]])
    query = X.new_tensor([[0.21], [0.53]], requires_grad=True)
    with pytest.warns(RuntimeWarning, match="independent Gaussian outputs"):
        value = actual(query)
    assert not actual._cache_root
    first = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert bool(torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("listed", [False, True])
def test_independent_model_and_t_batches_keep_exact_sample_coordinates(device, listed):
    from botorch_rs.models import ModelListGP

    _, (X, Y, noise) = _model(device, 2, live=False)
    X = X.unsqueeze(0).expand(2, -1, -1).clone().requires_grad_()
    Y = Y.unsqueeze(0).expand(2, -1, -1).clone().requires_grad_()
    noise = noise.unsqueeze(0).expand(2, -1, -1).clone().requires_grad_()
    if listed:
        model = ModelListGP(
            *[
                SingleTaskGP(
                    X,
                    Y[..., j : j + 1],
                    noise[..., j : j + 1],
                    outcome_transform=None,
                    backend="torch",
                )
                for j in range(2)
            ]
        )
    else:
        model = SingleTaskGP(X, Y, noise, outcome_transform=None, backend="torch")
    baseline = X.detach()[0, [0, 2]]
    actual = _make(FAMILIES[-1], model, baseline, detach_baseline=False)
    expected = _make(FAMILIES[-1], model, baseline, False, detach_baseline=False)
    expected.sampler = DenseNormalSampler((12,), seed=43)
    expected._observed_sampler = expected._new_sampler()
    query = X.new_tensor(
        [[[[0.21], [0.53]]], [[[0.45], [0.84]]], [[[0.1], [0.8]]]], requires_grad=True
    )
    value = actual(query)
    expected.sampler.base_samples = actual.sampler.base_samples.clone()
    oracle = expected(query)
    assert value.shape == (3, 2)
    torch.testing.assert_close(value, oracle, rtol=2e-8, atol=2e-9)
    leaves = (query, X, Y, noise, *model.parameters())
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True)
    reference = torch.autograd.grad(oracle.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    reference_second = torch.autograd.grad(sum(g.square().sum() for g in reference), leaves)
    for a, b in zip((*first, *second), (*reference, *reference_second), strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_public_cached_mixin_uses_tensor_graphs_and_fallback(device, outputs, monkeypatch):
    from botorch_rs.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin

    model, data = _model(device, outputs)
    baseline = data[0].detach()[[0, 2]]
    query = data[0].new_tensor([[0.21], [0.53]], requires_grad=True)
    sampler = IIDNormalSampler((4,), seed=43)
    mixin = CachedCholeskyMCSamplerMixin(model, cache_root=True, sampler=sampler)
    mixin.model = model
    prefix = model.posterior(baseline)
    joint = model.posterior(torch.cat((baseline, query), -2))
    mixin._baseline_L = mixin._compute_root_decomposition(prefix)
    sampler._construct_base_samples(joint)
    cached = mixin._get_f_X_samples(joint, 2)
    leaves = (query, *data, *model.parameters())
    first = torch.autograd.grad(cached.square().sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    assert all(bool(torch.isfinite(g).all()) for g in (*first, *second))
    mixin._cache_root = False

    def forbidden(*args, **kwargs):
        raise AssertionError("Device fallback must not flatten samples just to read their shape.")

    monkeypatch.setattr("botorch_rs.acquisition.cached_cholesky.to_flat", forbidden)
    fallback = mixin._get_f_X_samples(model.posterior(torch.cat((baseline, query), -2)), 2)
    assert fallback.device == query.device and fallback.shape == cached.shape
