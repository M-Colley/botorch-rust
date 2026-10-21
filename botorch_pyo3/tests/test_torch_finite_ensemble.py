"""Finite device ensembles retain weighted moment and selected-member graphs."""

# ruff: noqa: E402
from __future__ import annotations

import importlib
import warnings

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models.ensemble import EnsembleModel
from botorch_rs.posteriors.ensemble import EnsemblePosterior
from botorch_rs.sampling.get_sampler import get_sampler
from botorch_rs.sampling.index_sampler import IndexSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("weighted", [False, True])
def test_finite_ensemble_moments_and_full_hessians_reference(device, dtype, batch, weighted):
    pytest.importorskip("botorch")
    from botorch.posteriors.ensemble import EnsemblePosterior as Reference

    count = 4 * 3 * 2 * (torch.Size(batch).numel())
    values = (
        torch.linspace(-0.4, 1.2, count, device=device, dtype=dtype)
        .reshape(*batch, 4, 3, 2)
        .requires_grad_()
    )
    weights = (
        torch.tensor([0.1, 0.3, 0.4, 0.2], device=device, dtype=dtype, requires_grad=True)
        if weighted
        else None
    )
    actual, expected = EnsemblePosterior(values, weights), Reference(values, weights)
    assert isinstance(actual, EnsemblePosterior) and actual._tensor_backend == "torch"
    assert actual.batch_shape == torch.Size(batch)
    assert actual._extended_shape((2, 3)) == (2, 3, *batch, 3, 2)
    with pytest.raises(NotImplementedError):
        _ = actual.base_sample_shape
    leaves = (values, weights) if weighted else (values,)
    actual_properties = [
        getattr(actual, name) for name in ("mean", "variance", "mixture_mean", "mixture_variance")
    ]
    reference_properties = [
        getattr(expected, name) for name in ("mean", "variance", "mixture_mean", "mixture_variance")
    ]
    for a, b in zip(actual_properties, reference_properties, strict=True):
        torch.testing.assert_close(a, b)
        assert a.device == values.device and a.dtype == dtype
    first = torch.autograd.grad(
        sum(p.square().sum() for p in actual_properties), leaves, create_graph=True
    )
    reference_first = torch.autograd.grad(
        sum(p.square().sum() for p in reference_properties), leaves, create_graph=True
    )
    second = torch.autograd.grad(sum(p.square().sum() for p in first), leaves)
    reference_second = torch.autograd.grad(sum(p.square().sum() for p in reference_first), leaves)
    for a, b in zip((*first, *second), (*reference_first, *reference_second), strict=True):
        # Independent normalization graphs change the order of the many
        # weight-gradient/Hessian reductions in float32. Bound that ordinary
        # reduction roundoff, while retaining strict float64 comparisons.
        tolerance = 64 * torch.finfo(dtype).eps if dtype == torch.float32 else 1e-9
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    if weighted:
        # Constructor-time cached normalization must not retain a freed graph.
        for _ in range(2):
            gradient = torch.autograd.grad(
                actual.mean.square().sum() + actual.variance.sum(), leaves, create_graph=True
            )
            assert all(
                torch.isfinite(g).all()
                for g in torch.autograd.grad(sum(p.square().sum() for p in gradient), leaves)
            )


@pytest.mark.parametrize("device", DEVICES)
def test_batch_specific_weights_use_per_ensemble_reliability_denominator(device):
    values = (
        torch.arange(2 * 3 * 2, device=device, dtype=torch.double)
        .reshape(2, 3, 2, 1)
        .requires_grad_()
    )
    weights = values.new_tensor([[3.0, 3.0, 0.0], [6.0, 6.0, 0.0]], requires_grad=True)
    actual = EnsemblePosterior(values, weights)
    # Each batch's first two members have equal weight. Their unbiased
    # variance is half their squared difference, with finite full derivatives.
    expected = (values[:, 0] - values[:, 1]).square() / 2
    torch.testing.assert_close(actual.variance, expected)
    assert torch.isfinite(actual.variance).all()
    first = torch.autograd.grad(actual.variance.sum(), (values, weights), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (values, weights))
    assert all(torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_indices_match_source_without_host_copies_or_global_rng_changes(device, batch, monkeypatch):
    pytest.importorskip("botorch")
    from botorch.posteriors.ensemble import EnsemblePosterior as ReferencePosterior
    from botorch.sampling.index_sampler import IndexSampler as ReferenceSampler

    values = (
        torch.linspace(
            -0.4, 1.2, 4 * 3 * 2 * torch.Size(batch).numel(), device=device, dtype=torch.double
        )
        .reshape(*batch, 4, 3, 2)
        .requires_grad_()
    )
    weights = values.new_tensor([0.1, 0.3, 0.4, 0.2])

    def forbidden(*args, **kwargs):
        raise AssertionError("Device ensemble calculations must not flatten to the host.")

    monkeypatch.setattr("botorch_rs.posteriors.ensemble.to_flat", forbidden)
    monkeypatch.setattr("botorch_rs.sampling.index_sampler.to_flat", forbidden)
    posterior = EnsemblePosterior(values, weights)
    sampler = get_sampler(posterior, (2, 3), seed=67)
    assert isinstance(sampler, IndexSampler)
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    sampled = sampler(posterior)
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    if cuda_state is not None:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))
    with torch.random.fork_rng(devices=[torch.device(device)] if device == "cuda" else []):
        reference = ReferenceSampler(torch.Size((2, 3)), seed=67)
        expected = reference(ReferencePosterior(values, weights))
    torch.testing.assert_close(sampler.base_samples, reference.base_samples)
    torch.testing.assert_close(sampled, expected)
    first = torch.autograd.grad(sampled.square().sum(), values, create_graph=True)[0]
    reference_first = torch.autograd.grad(expected.square().sum(), values, create_graph=True)[0]
    torch.testing.assert_close(first, reference_first)
    second = torch.autograd.grad(first.square().sum(), values)[0]
    reference_second = torch.autograd.grad(reference_first.square().sum(), values)[0]
    torch.testing.assert_close(second, reference_second)
    indices = sampler.base_samples
    sampler(EnsemblePosterior(values[..., :1, :], weights))
    assert sampler.base_samples is indices


class FiniteModel(EnsembleModel):
    def __init__(self, device="cpu", outcomes=1, weights=None):
        super().__init__(weights)
        self._num_outputs = outcomes
        self.slopes = torch.nn.Parameter(
            torch.tensor([[0.6, -0.1], [1.2, 0.4], [-0.7, 0.5]], device=device, dtype=torch.double)
        )

    def forward(self, X):
        first = (X.unsqueeze(-3) * self.slopes[:, None, :]).sum(-1)
        second = first.sin() + 0.2
        return torch.stack((first, second), -1)[..., : self.num_outputs]


def _acquisition(name, reference=False):
    prefix = "botorch" if reference else "botorch_rs"
    mo = "multi_objective." if "Hypervolume" in name else ""
    module = (
        ("logei" if name.startswith("qLog") else "monte_carlo")
        if name.startswith("q")
        else "analytic"
    )
    return getattr(importlib.import_module(f"{prefix}.acquisition.{mo}{module}"), name)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name,options",
    [
        ("ExpectedImprovement", {"best_f": 0.2}),
        ("LogExpectedImprovement", {"best_f": 0.2}),
        ("UpperConfidenceBound", {"beta": 0.3}),
        ("PosteriorMean", {}),
        ("PosteriorStandardDeviation", {}),
    ],
)
def test_ensemble_analytic_moments_query_parameter_weight_graphs(device, name, options):
    pytest.importorskip("botorch")
    from botorch.models.ensemble import EnsembleModel as ReferenceModel

    weights = torch.tensor([0.2, 0.3, 0.5], device=device, dtype=torch.double, requires_grad=True)
    model = FiniteModel(device, weights=weights)

    class Reference(ReferenceModel):
        _num_outputs = 1

        def forward(self, X):
            return model.forward(X)

    reference = Reference(weights)
    query = weights.new_tensor([[[0.2, 0.7]], [[0.6, 0.4]]], requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = _acquisition(name)(model, **options)(query)
        expected = _acquisition(name, True)(reference, **options)(query)
    torch.testing.assert_close(actual, expected)
    leaves = (query, model.slopes, weights)
    first = torch.autograd.grad(actual.sum(), leaves, create_graph=True)
    oracle_first = torch.autograd.grad(expected.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, allow_unused=True)
    oracle_second = torch.autograd.grad(
        sum(g.square().sum() for g in oracle_first), leaves, allow_unused=True
    )
    for a, b in zip((*first, *second), (*oracle_first, *oracle_second), strict=True):
        if a is not None:
            torch.testing.assert_close(a, b)


MC = [
    ("qExpectedImprovement", {"best_f": 0.2}),
    ("qLogExpectedImprovement", {"best_f": 0.2}),
    ("qProbabilityOfImprovement", {"best_f": 0.2, "tau": 0.1}),
    ("qUpperConfidenceBound", {"beta": 0.3}),
    ("qLowerConfidenceBound", {"beta": 0.3}),
    ("qSimpleRegret", {}),
    ("qPosteriorStandardDeviation", {}),
    ("qNoisyExpectedImprovement", {}),
    ("qLogNoisyExpectedImprovement", {}),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,options", MC)
def test_index_sampled_scalar_acquisitions_match_source_full_graphs(device, name, options):
    pytest.importorskip("botorch")
    from botorch.models.ensemble import EnsembleModel as ReferenceModel
    from botorch.sampling.index_sampler import IndexSampler as ReferenceSampler

    model = FiniteModel(
        device, weights=torch.tensor([0.2, 0.3, 0.5], dtype=torch.double, device=device)
    )

    class Reference(ReferenceModel):
        _num_outputs = 1

        def forward(self, X):
            return model.forward(X)

    reference = Reference(model.ensemble_weights)
    query = model.slopes.new_tensor([[0.2, 0.7], [0.6, 0.4]], requires_grad=True)
    options = dict(options)
    if "Noisy" in name:
        options.update(
            X_baseline=query.detach().new_tensor([[0.1, 0.3], [0.7, 0.8]]),
            cache_root=False,
            prune_baseline=False,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = _acquisition(name)(model, sampler=IndexSampler((24,), seed=67), **options)
        expected = _acquisition(name, True)(
            reference, sampler=ReferenceSampler(torch.Size((24,)), seed=67), **options
        )
    value, oracle = actual(query), expected(query)
    torch.testing.assert_close(value, oracle, rtol=1e-8, atol=1e-9)
    leaves = (query, model.slopes)
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True)
    reference_first = torch.autograd.grad(oracle.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    reference_second = torch.autograd.grad(sum(g.square().sum() for g in reference_first), leaves)
    for a, b in zip((*first, *second), (*reference_first, *reference_second), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name",
    [
        "qExpectedHypervolumeImprovement",
        "qLogExpectedHypervolumeImprovement",
        "qNoisyExpectedHypervolumeImprovement",
        "qLogNoisyExpectedHypervolumeImprovement",
    ],
)
@pytest.mark.usefixtures("reference_python_logehvi")
def test_index_sampled_hypervolume_acquisitions_match_source_full_graphs(device, name):
    pytest.importorskip("botorch")
    from botorch.models.ensemble import EnsembleModel as ReferenceModel
    from botorch.sampling.index_sampler import IndexSampler as ReferenceSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning as RefPartition,
    )

    from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    model = FiniteModel(device, outcomes=2)

    class Reference(ReferenceModel):
        _num_outputs = 2

        def forward(self, X):
            return model.forward(X)

    reference = Reference()
    query = model.slopes.new_tensor([[0.2, 0.7], [0.6, 0.4]], requires_grad=True)
    ref = query.new_tensor([-0.3, -0.2])
    actual_options, reference_options = {}, {}
    if "Noisy" in name:
        options = dict(
            X_baseline=query.new_tensor([[0.1, 0.3], [0.7, 0.8]]),
            cache_root=False,
            prune_baseline=False,
        )
        actual_options.update(options)
        reference_options.update(options)
    else:
        observed = query.new_tensor([[0.1, 0.2], [0.3, 0.1]])
        actual_options["partitioning"] = FastNondominatedPartitioning(ref, Y=observed)
        reference_options["partitioning"] = RefPartition(ref, Y=observed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = _acquisition(name)(
            model, ref, sampler=IndexSampler((24,), seed=67), **actual_options
        )
        expected = _acquisition(name, True)(
            reference,
            ref,
            sampler=ReferenceSampler(torch.Size((24,)), seed=67),
            **reference_options,
        )
    value, oracle = actual(query), expected(query)
    torch.testing.assert_close(value, oracle, rtol=1e-8, atol=1e-9)
    leaves = (query, model.slopes)
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True)
    reference_first = torch.autograd.grad(oracle.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    reference_second = torch.autograd.grad(sum(g.square().sum() for g in reference_first), leaves)
    for a, b in zip((*first, *second), (*reference_first, *reference_second), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_default_index_sampling_modes_output_subsets_and_transforms(device):
    from botorch_rs.exceptions.errors import UnsupportedError

    class InputTransform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.8, device=device, dtype=torch.double))

        def forward(self, X):
            return X * self.scale

    class OutcomeTransform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.double))

        def untransform(self, Y, X=None):
            return Y + self.shift, None

    model = FiniteModel(device, outcomes=2)
    model.input_transform = InputTransform()
    model.outcome_transform = OutcomeTransform()
    model.train()
    query = model.slopes.new_tensor([[0.2, 0.7]], requires_grad=True)
    posterior = model.posterior(query, output_indices=[1], observation_noise=True)
    assert not model.training and not model.input_transform.training
    torch.testing.assert_close(
        posterior.values,
        (model.forward(query * model.input_transform.scale) + model.outcome_transform.shift)[
            ..., [1]
        ],
    )
    first = torch.autograd.grad(
        posterior.mean.square().sum(), (query, *model.parameters()), create_graph=True
    )
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(
            sum(g.square().sum() for g in first), (query, *model.parameters())
        )
    )
    with pytest.raises(UnsupportedError, match="observation noise"):
        model.posterior(query, observation_noise=torch.zeros_like(query))
    scalar = FiniteModel(device)
    acq = _acquisition("qSimpleRegret")(scalar)
    assert isinstance(acq.sampler, IndexSampler)
    value = acq(query)
    gradient = torch.autograd.grad(value.sum(), (query, scalar.slopes), create_graph=True)
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(
            sum(g.square().sum() for g in gradient), (query, scalar.slopes)
        )
    )
    called = []
    scalar.register_forward_hook(lambda *args: called.append(True))
    scalar(query)
    assert called == [True]


@pytest.mark.parametrize("device", DEVICES)
def test_singleton_empty_shapes_and_explicit_index_validation(device):
    value = torch.tensor([[[0.3]]], device=device, dtype=torch.double, requires_grad=True)
    posterior = EnsemblePosterior(value)
    torch.testing.assert_close(posterior.variance, torch.zeros_like(posterior.mean))
    assert torch.autograd.grad(posterior.variance.sum(), value)[0].item() == 0
    assert posterior.rsample((2, 3)).shape == (2, 3, 1, 1)
    for shape in ((0, 1, 3, 2), (2, 3, 0, 2)):
        empty = EnsemblePosterior(value.new_empty(shape))
        assert empty.rsample((4,)).shape == (4, *shape[:-3], *shape[-2:])
    with pytest.raises(ValueError, match="Sample_shape"):
        posterior.rsample_from_base_samples((3,), torch.zeros(2, device=device, dtype=torch.long))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name",
    [
        "qNoisyExpectedImprovement",
        "qLogNoisyExpectedImprovement",
        "qNoisyExpectedHypervolumeImprovement",
        "qLogNoisyExpectedHypervolumeImprovement",
    ],
)
def test_default_noisy_index_pruning_and_pending_keep_fresh_graphs(device, name):
    outcomes = 2 if "Hypervolume" in name else 1
    model = FiniteModel(device, outcomes=outcomes)
    query = model.slopes.new_tensor([[[0.2, 0.7]], [[0.6, 0.4]]], requires_grad=True)
    baseline = model.slopes.new_tensor([[0.1, 0.3], [0.7, 0.8], [0.4, 0.2]], requires_grad=True)
    kwargs = dict(X_baseline=baseline)
    if outcomes == 2:
        kwargs.update(ref_point=[-0.3, -0.2], cache_pending=True, max_iep=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = _acquisition(name)(model, **kwargs)
    assert isinstance(actual.sampler, IndexSampler)
    assert not actual._cache_root
    for pending in (None, baseline.detach()[[0]], baseline.detach()[[0, 2]], None):
        actual.set_X_pending(pending)
        value = actual(query)
        first = torch.autograd.grad(value.sum(), (query, model.slopes), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), (query, model.slopes))
        assert all(torch.isfinite(g).all() for g in (value, *first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("point", [-6.0, -30.0, -1000.0, -1e5, -1e6])
def test_log_ei_negative_tail_first_through_fourth_derivatives(device, point):
    mp = pytest.importorskip("mpmath")
    from botorch_rs.acquisition._torch_backend_acquisition import _log_ei

    with mp.workdps(80):

        def exact(x):
            return mp.log(
                mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi) + x * mp.erfc(-x / mp.sqrt(2)) / 2
            )

        reference = [float(mp.diff(exact, mp.mpf(point), order)) for order in range(5)]
    variable = torch.tensor(point, device=device, dtype=torch.double, requires_grad=True)
    value = _log_ei(variable)
    torch.testing.assert_close(value, variable.new_tensor(reference[0]), rtol=2e-15, atol=1e-11)
    for order in range(1, 5):
        value = torch.autograd.grad(value, variable, create_graph=True)[0]
        torch.testing.assert_close(
            value, variable.new_tensor(reference[order]), rtol=2e-10, atol=1e-30
        )
