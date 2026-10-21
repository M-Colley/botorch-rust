"""Actual module registration, hook, state and graph contracts for Tensor paths."""

from __future__ import annotations

import copy
import io

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from botorch_rs.sampling.pathwise import GeneralizedLinearPath, MatheronPath  # noqa: E402
from botorch_rs.sampling.pathwise.features import (  # noqa: E402
    KernelEvaluationMap,
    KernelFeatureMap,
)
from botorch_rs.sampling.pathwise.paths import PathDict, PathList  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("constructor", [PathDict, PathList])
def test_empty_compositions_support_module_state_and_later_registered_children(constructor):
    composition = constructor()
    assert isinstance(composition, nn.Module)
    assert composition.state_dict() == {}
    child = make_path("cpu")
    if constructor is PathDict:
        composition["path"] = child
    else:
        composition.paths.append(child)
    assert list(composition.parameters())[0] is next(child.parameters())


class Shift(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.shift = nn.Parameter(value)

    def forward(self, values):
        return values + self.shift


class Bias(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(value)

    def forward(self, values):
        return values.sum(-1) * self.value


def make_path(device, parameter=True, reference=False):
    constructor = GeneralizedLinearPath
    if reference:
        pytest.importorskip("botorch")
        from botorch.sampling.pathwise.paths import GeneralizedLinearPath as constructor

    feature = nn.Linear(2, 3, bias=False, dtype=torch.double, device=device)
    feature.weight.data.copy_(torch.arange(6, dtype=torch.double, device=device).reshape(3, 2) / 9)
    weight = torch.tensor([0.3, -0.2, 0.7], dtype=torch.double, device=device)
    if parameter:
        weight = nn.Parameter(weight)
    return constructor(
        feature,
        weight,
        bias_module=Bias(weight.new_tensor(0.13)),
        input_transform=Shift(weight.new_tensor([0.1, -0.2])),
        output_transform=Shift(weight.new_tensor(0.2)),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("parameter", [False, True])
def test_linear_path_registers_actual_modules_and_weight_and_matches_source(device, parameter):
    actual, reference = make_path(device, parameter), make_path(device, parameter, reference=True)
    assert isinstance(actual, nn.Module)
    assert isinstance(actual, GeneralizedLinearPath)
    assert set(actual.state_dict()) == set(reference.state_dict())
    assert set(dict(actual.named_parameters())) == set(dict(reference.named_parameters()))
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name])
    query = torch.tensor([[0.2, 0.5], [0.7, -0.1]], dtype=torch.double, device=device)
    calls = []
    handle = actual.register_forward_hook(lambda module, args, out: calls.append((args[0], out)))
    torch.testing.assert_close(actual(query), reference(query))
    handle.remove()
    torch.testing.assert_close(calls[0][0], query + actual.input_transform.shift)
    torch.testing.assert_close(calls[0][1], actual.forward(calls[0][0]))
    assert len(calls) == 1
    assert ("weight" in dict(actual.named_parameters())) is parameter
    assert ("weight" in dict(actual.named_buffers())) is not parameter


@pytest.mark.parametrize("device", DEVICES)
def test_path_all_registered_leaves_have_dense_first_and_mixed_second_derivatives(device):
    path = make_path(device)
    query = torch.tensor(
        [[0.2, 0.5], [0.7, -0.1]], dtype=torch.double, device=device, requires_grad=True
    )
    leaves = (query, *path.parameters())
    z = query + path.input_transform.shift
    dense = (
        (z @ path.feature_map.weight.T) @ path.weight
        + z.sum(-1) * path.bias_module.value
        + path.output_transform.shift
    )
    actual = path(query)
    torch.testing.assert_close(actual, dense)
    first = torch.autograd.grad(actual.sin().sum(), leaves, create_graph=True)
    expected = torch.autograd.grad(dense.sin().sum(), leaves, create_graph=True)
    for a, b in zip(first, expected, strict=True):
        torch.testing.assert_close(a, b)
    hessian = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    other = torch.autograd.grad(sum(g.square().sum() for g in expected), leaves)
    for a, b in zip(hessian, other, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("composition", ["dict", "list", "matheron"])
def test_compositions_register_children_preserve_aliases_and_roundtrip(device, composition):
    child = make_path(device)
    if composition == "dict":
        path = PathDict({"a": child, "b": child}, join=sum)
    elif composition == "list":
        path = PathList((p for p in (child, child)), join=sum)
    else:
        path = MatheronPath(child, child)

    def children(p):
        return list(p.paths) if composition == "list" else list(p.values())

    assert isinstance(path, nn.Module)
    assert children(path)[0] is child and children(path)[1] is child
    assert len(list(path.parameters())) == len(list(child.parameters()))
    query = torch.tensor([[0.2, 0.5]], dtype=torch.double, device=device)
    torch.testing.assert_close(path(query), 2 * child(query))
    stream = io.BytesIO()
    torch.save(path, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert children(restored)[0] is children(restored)[1]
    torch.testing.assert_close(restored(query), path(query))
    cloned = copy.deepcopy(path)
    assert children(cloned)[0] is children(cloned)[1]
    assert children(cloned)[0] is not child
    cloned.load_state_dict(path.state_dict())
    cloned.float().cpu()
    assert all(p.dtype == torch.float and p.device.type == "cpu" for p in cloned.parameters())
    torch.testing.assert_close(cloned(query.float().cpu()), path(query).float().cpu())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("feature_kind", ["affine", "evaluation"])
def test_feature_maps_register_weights_points_kernel_and_transforms(device, feature_kind):
    from botorch_rs.kernels import RBFKernel
    from botorch_rs.models._torch_backend_modules import TorchStationaryKernel

    like = torch.tensor(0.0, dtype=torch.double, device=device)
    kernel = TorchStationaryKernel(RBFKernel(ard_num_dims=2), like)
    values = nn.Parameter(like.new_tensor([[0.2, 0.4], [0.7, 0.8], [0.9, 0.1]]))
    cls = KernelFeatureMap if feature_kind == "affine" else KernelEvaluationMap
    feature = cls(kernel, values, input_transform=Shift(like.new_tensor([0.1, -0.2])))
    assert isinstance(feature, nn.Module)
    field = "weight" if feature_kind == "affine" else "points"
    assert dict(feature.named_parameters())[field] is values
    assert feature.kernel is kernel
    assert dict(feature.named_parameters())["kernel.raw_lengthscale"] is kernel.raw_lengthscale
    query = like.new_tensor([[0.3, 0.5], [0.8, 0.2]]).requires_grad_()
    result = feature(query)
    z = query + feature.input_transform.shift
    dense = z @ values.T if feature_kind == "affine" else kernel(z, values)
    torch.testing.assert_close(result, dense)
    all_leaves = (query, *feature.parameters())
    if feature_kind == "affine":
        # The descriptor's kernel metadata is intentionally absent from this affine law.
        all_leaves = tuple(x for x in all_leaves if x is not kernel.raw_lengthscale)
    first = torch.autograd.grad(result.square().sum(), all_leaves, create_graph=True)
    assert all(torch.isfinite(v).all() for v in first)
    second = torch.autograd.grad(sum(v.square().sum() for v in first), all_leaves)
    assert all(torch.isfinite(v).all() for v in second)
    feature.float().cpu()
    assert all(p.dtype == torch.float and p.device.type == "cpu" for p in feature.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("matheron", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_generated_paths_are_registered_picklable_modules_with_migrating_draws(
    device, matheron, standardize
):
    from botorch_rs.kernels import RBFKernel
    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.transforms.outcome import Standardize
    from botorch_rs.sampling.pathwise import draw_kernel_feature_paths, draw_matheron_paths

    X = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double, device=device)
    Y = (X * 3).sin()
    model = SingleTaskGP(
        X,
        Y,
        covar_module=RBFKernel(ard_num_dims=1),
        outcome_transform=Standardize(1) if standardize else None,
        backend="torch",
    ).eval()
    if matheron:
        import functools

        path = draw_matheron_paths(
            model,
            (2,),
            prior_sampler=functools.partial(draw_kernel_feature_paths, num_features=8),
            seed=31,
        )
    else:
        path = draw_kernel_feature_paths(model, (2,), num_features=8, seed=31)
    assert isinstance(path, nn.Module)
    assert any("raw_lengthscale" in name for name, _ in path.named_parameters())
    query = X.new_tensor([[0.2], [0.7]])
    expected = path(query)
    stream = io.BytesIO()
    torch.save(path, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(restored(query), expected)
    path.float().cpu()
    torch.testing.assert_close(
        path(query.float().cpu()), expected.float().cpu(), atol=2e-5, rtol=2e-5
    )
    if matheron:
        assert all(
            value.dtype == torch.float and value.device.type == "cpu"
            for draw in path.draws
            for value in draw
        )
