"""Caller path classes retain cooperative overrides, leaves and module state."""

from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from botorch_rs.sampling.pathwise import MatheronPath  # noqa: E402
from botorch_rs.sampling.pathwise import paths as port_paths  # noqa: E402
from botorch_rs.sampling.pathwise import utils as port_utils  # noqa: E402
from botorch_rs.sampling.pathwise.features import maps as port_maps  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
BASES = {
    "linear": port_paths.GeneralizedLinearPath,
    "dict": port_paths.PathDict,
    "list": port_paths.PathList,
    "matheron": MatheronPath,
    "affine": port_maps.KernelFeatureMap,
    "evaluation": port_maps.KernelEvaluationMap,
    "sincos": port_utils.SineCosineTransform,
    "selector": port_utils.FeatureSelector,
    "transform": port_utils.TensorTransform,
    "feature": port_maps.FeatureMap,
    "path": port_paths.SamplePath,
}


class Gain:
    def __init__(self, *args, like, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = nn.Parameter(like.new_tensor(0.7))

    def forward(self, values, **kwargs):
        result = super().forward(values, **kwargs)
        if not isinstance(result, torch.Tensor):
            result = result.to_dense()
        return result.sin() * self.gain


class Primitive(Gain):
    def forward(self, values, **kwargs):
        return values.sin() * self.gain

    def set_ensemble_as_batch(self, value):
        self.ensemble_as_batch = value


PORT_CLASSES = {}
for _kind, _base in BASES.items():
    _mixin = Primitive if _kind in {"transform", "feature", "path"} else Gain
    _name = "Custom" + _base.__name__
    PORT_CLASSES[_kind] = type(_name, (_mixin, _base), {"__module__": __name__})
    globals()[_name] = PORT_CLASSES[_kind]


def construct(kind, constructor, like, reference=False):
    from botorch_rs.kernels import RBFKernel
    from botorch_rs.models._torch_backend_modules import TorchStationaryKernel

    kernel = TorchStationaryKernel(RBFKernel(ard_num_dims=2), like)
    linear = port_paths.GeneralizedLinearPath
    if reference:
        from botorch.sampling.pathwise.paths import GeneralizedLinearPath as linear

    feature = nn.Linear(2, 2, bias=False, device=like.device, dtype=like.dtype)
    with torch.no_grad():
        feature.weight.copy_(like.new_tensor([[0.2, -0.3], [0.4, 0.1]]))
    weight = nn.Parameter(like.new_tensor([0.4, 0.8]))
    if kind == "linear":
        return constructor(feature, weight, like=like)
    if kind in {"dict", "list", "matheron"}:
        child = linear(feature, weight)
        if kind == "dict":
            return constructor({"a": child, "b": child}, join=sum, like=like)
        if kind == "list":
            return constructor([child, child], join=sum, like=like)
        return constructor(child, child, like=like)
    if kind in {"affine", "evaluation"}:
        values = nn.Parameter(like.new_tensor([[0.2, -0.3], [0.4, 0.1]]))
        return constructor(kernel, values, like=like)
    if kind == "sincos":
        return constructor(nn.Parameter(like.new_tensor(0.4)), like=like)
    if kind == "selector":
        return constructor([1, 0, 1], like=like)
    return constructor(like=like)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", BASES)
def test_subclass_module_leaves_overrides_state_and_serialization(kind, device):
    like = torch.zeros((), dtype=torch.double, device=device)
    path = construct(kind, PORT_CLASSES[kind], like)
    assert isinstance(path, nn.Module)
    assert isinstance(path, PORT_CLASSES[kind])
    assert dict(path.named_parameters())["gain"] is path.gain
    query = like.new_tensor([[0.2, 0.5], [0.7, 0.1]]).requires_grad_()
    first = torch.autograd.grad(
        path(query).square().sum(),
        (query, *path.parameters()),
        create_graph=True,
        allow_unused=True,
    )
    active = [g for g in first if g is not None]
    second = torch.autograd.grad(
        sum(g.square().sum() for g in active), (query, *path.parameters()), allow_unused=True
    )
    assert all(torch.isfinite(g).all() for g in (*first, *second) if g is not None)
    stream = io.BytesIO()
    torch.save(path, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert isinstance(restored, PORT_CLASSES[kind])
    torch.testing.assert_close(restored(query), path(query))
    restored.load_state_dict(path.state_dict())
    restored.float().cpu()
    torch.testing.assert_close(restored(query.float().cpu()), path(query).float().cpu())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", BASES)
def test_subclass_values_and_all_active_mixed_derivatives_match_unmodified_source(kind, device):
    pytest.importorskip("botorch")
    from botorch.sampling.pathwise import paths, utils
    from botorch.sampling.pathwise.features import maps
    from botorch.sampling.pathwise.posterior_samplers import MatheronPath as ReferenceMatheron

    bases = {
        **{
            name: getattr(paths, base.__name__)
            for name, base in BASES.items()
            if name in {"linear", "dict", "list", "path"}
        },
        **{
            name: getattr(maps, base.__name__)
            for name, base in BASES.items()
            if name in {"affine", "evaluation", "feature"}
        },
        **{
            name: getattr(utils, base.__name__)
            for name, base in BASES.items()
            if name in {"sincos", "selector", "transform"}
        },
        "matheron": ReferenceMatheron,
    }
    mixin = Primitive if kind in {"transform", "feature", "path"} else Gain
    reference_type = type("Reference", (mixin, bases[kind]), {})
    like = torch.zeros((), dtype=torch.double, device=device)
    actual = construct(kind, PORT_CLASSES[kind], like).to(like)
    reference = construct(kind, reference_type, like, reference=True).to(like)
    query = like.new_tensor([[0.2, 0.5], [0.7, 0.1]]).requires_grad_()
    other = query.detach().clone().requires_grad_()
    result, expected = actual(query), reference(other)
    torch.testing.assert_close(result, expected)
    actual_parameters = dict(actual.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    assert actual_parameters.keys() == reference_parameters.keys()
    variables = (query, *actual_parameters.values())
    other_variables = (other, *reference_parameters.values())
    first = torch.autograd.grad(
        result.square().sum(), variables, create_graph=True, allow_unused=True
    )
    expected_first = torch.autograd.grad(
        expected.square().sum(), other_variables, create_graph=True, allow_unused=True
    )
    for a, b in zip(first, expected_first, strict=True):
        assert (a is None) is (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first if g is not None), variables, allow_unused=True
    )
    expected_second = torch.autograd.grad(
        sum(g.square().sum() for g in expected_first if g is not None),
        other_variables,
        allow_unused=True,
    )
    for a, b in zip(second, expected_second, strict=True):
        assert (a is None) is (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)
