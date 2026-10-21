"""Public sampler module state and caller-defined strategies on CPU and CUDA."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import io
import pickle

import pytest

torch = pytest.importorskip("torch")

from test_normal_sampler_public_protocol import PublicPosterior

from botorch_rs.acquisition._torch_backend_entropy import _clone_sampler
from botorch_rs.acquisition.monte_carlo import qSimpleRegret
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models import SingleTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.ensemble import EnsemblePosterior
from botorch_rs.posteriors.posterior_list import PosteriorList
from botorch_rs.sampling.index_sampler import IndexSampler
from botorch_rs.sampling.list_sampler import ListSampler
from botorch_rs.sampling.normal import (
    IIDNormalSampler,
    MCSampler,
    NormalMCSampler,
    SobolQMCNormalSampler,
)
from botorch_rs.sampling.pairwise_samplers import (
    PairwiseIIDNormalSampler,
    PairwiseSobolQMCNormalSampler,
)
from botorch_rs.sampling.stochastic_samplers import ForkedRNGSampler, StochasticSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CONCRETE = [
    IIDNormalSampler,
    SobolQMCNormalSampler,
    IndexSampler,
    ForkedRNGSampler,
    StochasticSampler,
    PairwiseIIDNormalSampler,
    PairwiseSobolQMCNormalSampler,
]


class CustomSampler(MCSampler):
    def __init__(self, sample_shape, gain):
        super().__init__(sample_shape, seed=19)
        self.gain = torch.nn.Parameter(gain)
        self.shift = torch.nn.Linear(1, 1, bias=False).to(gain)
        with torch.no_grad():
            self.shift.weight.fill_(0.2)

    def forward(self, posterior):
        return (self.gain * posterior.mean + self.shift(posterior.mean)).expand(
            *self.sample_shape, *posterior.mean.shape
        )


class CustomNormals(NormalMCSampler):
    def __init__(self, sample_shape, gain):
        super().__init__(sample_shape, seed=23)
        self.gain = torch.nn.Parameter(gain)
        self.calls = 0

    def _construct_base_samples(self, posterior):
        self.calls += 1
        shape = self._get_collapsed_shape(posterior)
        self.register_buffer("base_samples", self.gain.expand(shape))


class ConfiguredSobol(SobolQMCNormalSampler):
    def __init__(self, sample_shape, gain, seed=29):
        super().__init__(sample_shape, seed)
        self.gain = torch.nn.Parameter(gain)

    def forward(self, posterior):
        return self.gain * super().forward(posterior)


def posterior(device="cpu", dtype=torch.double, points=3):
    mean = torch.linspace(0.1, 0.8, points * 4, dtype=dtype, device=device).reshape(2, 2, points, 1)
    return PublicPosterior(mean, mean.new_tensor(0.7))


@pytest.mark.parametrize("sampler_class", CONCRETE)
@pytest.mark.parametrize("device", DEVICES)
def test_concrete_module_registration_modes_hooks_and_migration(sampler_class, device):
    pytest.importorskip("botorch")
    from botorch.sampling import index_sampler, normal, pairwise_samplers, stochastic_samplers

    source_class = next(
        getattr(module, sampler_class.__name__)
        for module in (normal, index_sampler, stochastic_samplers, pairwise_samplers)
        if hasattr(module, sampler_class.__name__)
    )
    actual = sampler_class(torch.Size((4,)), seed=13)
    source = source_class(torch.Size((4,)), seed=13)
    assert isinstance(actual, (torch.nn.Module, sampler_class))
    assert actual._buffers.keys() == source._buffers.keys() == {"base_samples"}
    assert actual.state_dict().keys() == source.state_dict().keys()
    assert tuple(actual.sample_shape) == tuple(source.sample_shape)
    parent = torch.nn.ModuleDict({"sampler": actual})
    parent.eval()
    assert not actual.training
    parent.train()
    assert actual.training
    marker = torch.tensor(0.3, dtype=torch.double)
    actual.register_buffer("marker", marker)
    parent.to(device=device, dtype=torch.float)
    assert actual.marker.device.type == device and actual.marker.dtype == torch.float
    assert dict(parent.named_buffers())["sampler.marker"] is actual.marker


@pytest.mark.parametrize("sampler_class", CONCRETE)
def test_default_seed_draw_and_rng_state_match_source(sampler_class):
    pytest.importorskip("botorch")
    from botorch.sampling import index_sampler, normal, pairwise_samplers, stochastic_samplers

    source_class = next(
        getattr(module, sampler_class.__name__)
        for module in (normal, index_sampler, stochastic_samplers, pairwise_samplers)
        if hasattr(module, sampler_class.__name__)
    )
    with torch.random.fork_rng():
        torch.manual_seed(733)
        source = source_class(torch.Size((4,)))
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(733)
        actual = sampler_class(torch.Size((4,)))
        assert actual.seed == source.seed
        torch.testing.assert_close(torch.random.get_rng_state(), expected_state)


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize("device", DEVICES)
def test_cache_buffers_source_values_migration_loading_and_pickle(sampler_class, device):
    pytest.importorskip("botorch")
    from botorch.sampling import normal

    actual = sampler_class(torch.Size((2, 3)), seed=31)
    source = getattr(normal, sampler_class.__name__)(torch.Size((2, 3)), seed=31)
    inputs = posterior()
    first, expected = actual(inputs), source(inputs)
    torch.testing.assert_close(first, expected, rtol=3e-14, atol=3e-15)
    assert dict(actual.named_buffers())["base_samples"] is actual.base_samples
    assert actual.state_dict().keys() == source.state_dict().keys() == {"base_samples"}
    old = actual.base_samples.clone()
    actual.to(device=device, dtype=torch.float)
    source.to(device=device, dtype=torch.float)
    inputs = posterior(device, torch.float)
    torch.testing.assert_close(actual(inputs), source(inputs), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.base_samples, old.to(device=device, dtype=torch.float))
    restored = sampler_class((2, 3), seed=31)
    restored(inputs)
    restored.load_state_dict(actual.state_dict())
    torch.testing.assert_close(restored(inputs), actual(inputs))
    for copied in (copy.deepcopy(actual), pickle.loads(pickle.dumps(actual))):
        assert type(copied) is type(actual)
        assert copied.base_samples is not actual.base_samples
        torch.testing.assert_close(copied(inputs), actual(inputs))
    stream = io.BytesIO()
    torch.save(torch.nn.ModuleDict({"sampler": actual}), stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=False)["sampler"]
    torch.testing.assert_close(loaded(inputs), actual(inputs))
    # Source and port both require materializing a None buffer before strict
    # state loading, following the ordinary nn.Module loading protocol.
    for sampler in (sampler_class((2, 3), seed=31), type(source)(torch.Size((2, 3)), seed=31)):
        with pytest.raises(RuntimeError, match="Unexpected key"):
            sampler.load_state_dict(actual.state_dict())


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize("device", DEVICES)
def test_caller_base_parameter_keeps_identity_state_and_full_autograd(sampler_class, device):
    pytest.importorskip("botorch")
    from botorch.sampling import normal

    inputs = posterior(device)
    shape = (4, 1, 1, 3)
    base = torch.nn.Parameter(
        torch.linspace(-0.4, 0.7, 12, dtype=inputs.dtype, device=device).reshape(shape)
    )
    sampler = sampler_class(torch.Size((4,)), seed=13)
    sampler.base_samples = base
    reference = getattr(normal, sampler_class.__name__)(torch.Size((4,)), seed=13)
    expected_base = torch.nn.Parameter(base.detach().clone())
    reference.base_samples = expected_base
    assert sampler.base_samples is base and dict(sampler.named_parameters())["base_samples"] is base
    actual, expected = sampler(inputs), reference(inputs)
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual.sin().sum(), base, create_graph=True)[0]
    expected_first = torch.autograd.grad(expected.sin().sum(), expected_base, create_graph=True)[0]
    torch.testing.assert_close(first, expected_first)
    second = torch.autograd.grad(first.square().sum(), base)[0]
    expected_second = torch.autograd.grad(expected_first.square().sum(), expected_base)[0]
    torch.testing.assert_close(second, expected_second)
    sampler.to(dtype=torch.float)
    assert sampler.base_samples is base and base.dtype == torch.float


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler_class", [CustomSampler, CustomNormals, ConfiguredSobol])
def test_user_sampler_parameters_callbacks_hooks_and_complete_graph(
    device, sampler_class, monkeypatch
):
    gain = torch.tensor(0.8, dtype=torch.double, device=device)
    sampler = sampler_class(torch.Size((4,)), gain)
    inputs = posterior(device)
    inputs.mean.requires_grad_()
    events = []
    hook = sampler.register_forward_hook(lambda module, args, output: events.append(output.shape))

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampler inference converted device data to host.")

    for method in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    value = sampler(inputs)
    assert events == [value.shape]
    if sampler_class is CustomNormals:
        assert sampler.calls == 1
        expected = inputs.mean + inputs.scale * sampler.gain
        torch.testing.assert_close(value, expected.expand_as(value))
    parameters = (inputs.mean, *sampler.parameters())
    gradient = torch.autograd.grad(value.sin().sum(), parameters, create_graph=True)
    second = torch.autograd.grad(sum(item.square().sum() for item in gradient), parameters)
    assert all(torch.isfinite(item).all() for item in (*gradient, *second))
    assert dict(sampler.named_parameters())["gain"] is sampler.gain
    hook.remove()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler_class", [CustomSampler, CustomNormals, ConfiguredSobol])
def test_custom_base_normal_and_sobol_protocols_match_source_derivatives(device, sampler_class):
    pytest.importorskip("botorch")
    from botorch.sampling.base import MCSampler as SourceBase
    from botorch.sampling.normal import NormalMCSampler as SourceNormal
    from botorch.sampling.normal import SobolQMCNormalSampler as SourceSobol

    class SourceCustom(SourceBase):
        def __init__(self, shape, gain):
            super().__init__(shape, seed=19)
            self.gain = torch.nn.Parameter(gain)
            self.shift = torch.nn.Linear(1, 1, bias=False).to(gain)
            with torch.no_grad():
                self.shift.weight.fill_(0.2)

        forward = CustomSampler.forward

    class SourceNormals(SourceNormal):
        def __init__(self, shape, gain):
            super().__init__(shape, seed=23)
            self.gain = torch.nn.Parameter(gain)
            self.calls = 0

        _construct_base_samples = CustomNormals._construct_base_samples

    class SourceConfigured(SourceSobol):
        def __init__(self, shape, gain):
            super().__init__(shape, seed=29)
            self.gain = torch.nn.Parameter(gain)

        def forward(self, posterior):
            return self.gain * super().forward(posterior)

    source_class = {
        CustomSampler: SourceCustom,
        CustomNormals: SourceNormals,
        ConfiguredSobol: SourceConfigured,
    }[sampler_class]
    gain = torch.tensor(0.8, dtype=torch.double, device=device)
    actual, source = (
        sampler_class(torch.Size((4,)), gain.clone()),
        source_class(torch.Size((4,)), gain.clone()),
    )
    graphs = []
    values = []
    for sampler in (actual, source):
        inputs = posterior(device)
        inputs.mean.requires_grad_()
        value = sampler(inputs)
        variables = (inputs.mean, *sampler.parameters())
        gradient = torch.autograd.grad(value.sin().sum(), variables, create_graph=True)
        second = torch.autograd.grad(sum(item.square().sum() for item in gradient), variables)
        values.append(value)
        graphs.append((*gradient, *second))
    torch.testing.assert_close(values[0], values[1], rtol=2e-14, atol=2e-15)
    for actual_gradient, expected_gradient in zip(*graphs, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-13, atol=3e-14)
    assert actual.state_dict().keys() == source.state_dict().keys()


@pytest.mark.parametrize("device", DEVICES)
def test_custom_call_and_setattr_overrides_retain_module_hooks(device):
    class CustomCall(IIDNormalSampler):
        def __init__(self, gain):
            super().__init__((4,), seed=11)
            self.gain = gain
            self.calls = 0

        def __setattr__(self, name, value):
            if name == "gain":
                value = torch.nn.Parameter(value + 0.2)
            super().__setattr__(name, value)

        def __call__(self, posterior):
            self.calls += 1
            return self.gain * super().__call__(posterior)

    gain = torch.tensor(0.8, dtype=torch.double, device=device)
    sampler = CustomCall(gain)
    events = []
    sampler.register_forward_hook(lambda *args: events.append(True))
    inputs = posterior(device)
    value = sampler(inputs)
    assert events == [True] and sampler.calls == 1
    assert sampler.gain.item() == 1.0
    torch.testing.assert_close(value, IIDNormalSampler((4,), seed=11)(inputs))


@pytest.mark.parametrize("device", DEVICES)
def test_custom_sampler_retained_by_acquisition_and_independently_cloned(device):
    X = torch.tensor([[0.05], [0.34], [0.67], [0.9]], dtype=torch.double, device=device)
    model = SingleTaskGP(X, X.sin(), outcome_transform=None, backend="torch")
    sampler = ConfiguredSobol(torch.Size((8,)), X.new_tensor(0.8))
    acquisition = qSimpleRegret(model, sampler=sampler)
    query = X.new_tensor([[[0.2]], [[0.6]]], requires_grad=True)
    value = acquisition(query)
    assert acquisition.sampler is sampler
    assert dict(acquisition.named_parameters())["sampler.gain"] is sampler.gain
    variables = (query, sampler.gain, *model.parameters())
    first = torch.autograd.grad(value.sin().sum(), variables, create_graph=True)
    second = torch.autograd.grad(sum(item.square().sum() for item in first), variables)
    assert all(torch.isfinite(item).all() for item in (*first, *second))
    clone = _clone_sampler(sampler)
    assert isinstance(clone, ConfiguredSobol) and clone.gain is not sampler.gain
    torch.testing.assert_close(clone.gain, sampler.gain)
    torch.testing.assert_close(clone.base_samples, sampler.base_samples)
    loaded = pickle.loads(pickle.dumps(sampler))
    torch.testing.assert_close(loaded(model.posterior(query)), sampler(model.posterior(query)))


@pytest.mark.parametrize("device", DEVICES)
def test_list_sampler_registers_actual_children_and_updates_each_baseline(device):
    pytest.importorskip("botorch")
    from botorch.sampling.normal import IIDNormalSampler as SourceSampler

    children = [IIDNormalSampler((4,), seed=7), SourceSampler(torch.Size((4,)), seed=11)]
    sampler = ListSampler(*children)
    assert isinstance(sampler, torch.nn.Module)
    assert isinstance(sampler.samplers, torch.nn.ModuleList)
    assert all(
        actual is expected for actual, expected in zip(sampler.samplers, children, strict=True)
    )
    inputs = PosteriorList(posterior(device), posterior(device))
    value = sampler(inputs)
    expected = torch.cat(
        [child(part) for child, part in zip(children, inputs.posteriors, strict=True)], -1
    )
    torch.testing.assert_close(value, expected)
    assert sampler.state_dict().keys() == {"samplers.0.base_samples", "samplers.1.base_samples"}
    sampler.eval().to(dtype=torch.float)
    assert all(not child.training and child.base_samples.dtype == torch.float for child in children)
    loaded = pickle.loads(pickle.dumps(sampler))
    torch.testing.assert_close(
        loaded(PosteriorList(posterior(device, torch.float), posterior(device, torch.float))),
        sampler(PosteriorList(posterior(device, torch.float), posterior(device, torch.float))),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_list_prefix_updates_and_actual_posterior_list_match_source(device):
    pytest.importorskip("botorch")
    from botorch.posteriors.posterior_list import PosteriorList as SourceList
    from botorch.sampling.list_sampler import ListSampler as SourceSampler
    from botorch.sampling.normal import IIDNormalSampler as SourceIID
    from botorch.sampling.normal import SobolQMCNormalSampler as SourceSobol

    def gaussian(q):
        mean = torch.linspace(0.1, 0.7, q, dtype=torch.double, device=device).reshape(q, 1)
        return TorchGPyTorchPosterior(mean, torch.eye(q, dtype=mean.dtype, device=device))

    baseline, extended = gaussian(2), gaussian(4)
    actual = ListSampler(IIDNormalSampler((4,), seed=7), SobolQMCNormalSampler((4,), seed=11))
    previous = ListSampler(IIDNormalSampler((4,), seed=17), SobolQMCNormalSampler((4,), seed=19))
    source = SourceSampler(
        SourceIID(torch.Size((4,)), seed=7), SourceSobol(torch.Size((4,)), seed=11)
    )
    source_previous = SourceSampler(
        SourceIID(torch.Size((4,)), seed=17), SourceSobol(torch.Size((4,)), seed=19)
    )
    previous(PosteriorList(baseline, baseline))
    source_previous(SourceList(baseline, baseline))
    actual._update_base_samples(PosteriorList(extended, extended), previous)
    source._update_base_samples(SourceList(extended, extended), source_previous)
    for own_child, source_child in zip(actual.samplers, source.samplers, strict=True):
        torch.testing.assert_close(
            own_child.base_samples, source_child.base_samples, rtol=2e-14, atol=2e-15
        )
    own_value = actual(SourceList(extended, extended))
    torch.testing.assert_close(
        own_value, source(SourceList(extended, extended)), rtol=2e-14, atol=2e-15
    )


@pytest.mark.parametrize("device", DEVICES)
def test_index_sampler_actual_source_posterior_cache_is_registered(device):
    pytest.importorskip("botorch")
    from botorch.posteriors.ensemble import EnsemblePosterior as SourcePosterior
    from botorch.sampling.index_sampler import IndexSampler as SourceSampler

    values = torch.arange(24, dtype=torch.double, device=device).reshape(2, 3, 2, 2)
    own_posterior = EnsemblePosterior(values)
    source_posterior = SourcePosterior(values)
    sampler = IndexSampler((4,), seed=17)
    reference = SourceSampler(torch.Size((4,)), seed=17)
    torch.testing.assert_close(sampler(own_posterior), reference(source_posterior))
    assert dict(sampler.named_buffers())["base_samples"] is sampler.base_samples
    torch.testing.assert_close(sampler(source_posterior), reference(source_posterior))
    restored = pickle.loads(pickle.dumps(sampler))
    assert restored.base_samples.dtype == torch.long
    torch.testing.assert_close(restored(own_posterior), sampler(own_posterior))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler_class", [PairwiseIIDNormalSampler, PairwiseSobolQMCNormalSampler])
def test_pairwise_comparisons_stay_on_device_and_follow_utility_order(
    device, sampler_class, monkeypatch
):
    inputs = posterior(device)
    sampler = sampler_class((4,), seed=37, max_num_comparisons=3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Pairwise comparisons copied sampled utilities to host.")

    for method in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    comparisons = sampler(inputs)
    assert comparisons.device.type == device and comparisons.dtype == torch.long
    assert comparisons.shape == (4, 2, 2, 3, 2)
    utilities = sampler._base_forward(inputs).squeeze(-1)
    preferred = utilities.gather(-1, comparisons[..., 0])
    other = utilities.gather(-1, comparisons[..., 1])
    assert (preferred >= other).all()
    assert dict(sampler.named_buffers())["base_samples"] is sampler.base_samples


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize(
    "customization", ["subclass", "forward", "parameter", "hook", "base", "batch"]
)
def test_native_acquisitions_reject_callback_or_parameter_erasure(sampler_class, customization):
    X = torch.tensor([[0.05], [0.34], [0.67], [0.9]], dtype=torch.double)
    model = SingleTaskGP(X, X.sin(), outcome_transform=None, backend="native")
    if customization == "subclass":

        class Custom(sampler_class):
            def forward(self, posterior):
                raise AssertionError("The native bridge silently ignored a caller method.")

        sampler = Custom((4,), seed=13)
    else:
        sampler = sampler_class((4,), seed=13)
        if customization == "forward":
            sampler.forward = lambda posterior: posterior.mean.expand(4, *posterior.mean.shape)
        elif customization == "parameter":
            sampler.gain = torch.nn.Parameter(X.new_tensor(0.8))
        elif customization == "hook":
            sampler.register_forward_hook(lambda *args: None)
        elif customization == "base":
            sampler.base_samples = torch.nn.Parameter(X.new_zeros(4, 1))
        else:
            sampler.batch_range_override = (0, -1)
    with pytest.raises(UnsupportedError, match="backend='torch'"):
        qSimpleRegret(model, sampler=sampler)


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize("source_sampler", [False, True])
def test_native_canonical_normals_honor_explicit_public_cache(sampler_class, source_sampler):
    if source_sampler:
        pytest.importorskip("botorch")
        from botorch.sampling import normal

        sampler_class = getattr(normal, sampler_class.__name__)
    X = torch.tensor([[0.05], [0.34], [0.67], [0.9]], dtype=torch.double)
    model = SingleTaskGP(X, X.sin(), outcome_transform=None, backend="native")
    sampler = sampler_class(torch.Size((4,)), seed=13)
    base = X.new_tensor([0.3, 0.7, 1.1, 1.5]).reshape(4, 1, 1)
    sampler.base_samples = base
    acquisition = qSimpleRegret(model, sampler=sampler)
    query = X.new_tensor([[0.2]])
    value = acquisition(query)
    marginal = model.posterior(query)
    expected = marginal.mean.squeeze() + marginal.variance.squeeze().sqrt() * base.mean()
    torch.testing.assert_close(value.squeeze(), expected, rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
def test_native_list_cache_can_be_used_with_a_tensor_posterior(sampler_class):
    native = SingleTaskGP([[0.1], [0.4], [0.8]], [[0.2], [0.7], [0.5]], outcome_transform=None)
    sampler = sampler_class((4,), seed=13)
    sampler(native.posterior([[0.2], [0.6]]))
    assert isinstance(sampler.base_samples, list)
    mean = torch.tensor([[0.1], [0.4]], dtype=torch.double, requires_grad=True)
    covariance = torch.eye(2, dtype=mean.dtype)
    inputs = TorchGPyTorchPosterior(mean, covariance)
    value = sampler(inputs)
    assert isinstance(sampler.base_samples, torch.Tensor)
    assert dict(sampler.named_buffers())["base_samples"] is sampler.base_samples
    assert value.shape == (4, 2, 1)
    first = torch.autograd.grad(value.sin().sum(), mean, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), mean)[0]).all()
