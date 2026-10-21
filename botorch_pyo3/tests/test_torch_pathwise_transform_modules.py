"""Module state and graphs of the six public path-transform families."""

from __future__ import annotations

import copy
import io

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from botorch_rs.kernels import RBFKernel, ScaleKernel  # noqa: E402
from botorch_rs.models._torch_backend_modules import (  # noqa: E402
    TorchScaleKernel,
    TorchStandardize,
    TorchStationaryKernel,
)
from botorch_rs.sampling.pathwise.utils import (  # noqa: E402
    ChainedTransform,
    FeatureSelector,
    InverseLengthscaleTransform,
    OutcomeUntransformer,
    OutputscaleTransform,
    SineCosineTransform,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FAMILIES = ["chain", "sincos", "lengthscale", "outputscale", "selector", "outcome"]


def make_transform(family, like):
    kernel = TorchStationaryKernel(RBFKernel(ard_num_dims=2), like)
    if family == "sincos":
        return SineCosineTransform(nn.Parameter(like.new_tensor(0.4)))
    if family == "lengthscale":
        return InverseLengthscaleTransform(kernel)
    if family == "outputscale":
        return OutputscaleTransform(TorchScaleKernel(ScaleKernel(RBFKernel()), kernel, like))
    if family == "selector":
        return FeatureSelector(torch.tensor([1, 0, 1], device=like.device))
    if family == "outcome":
        outcome = TorchStandardize(1, (), like)
        outcome(like.new_tensor([[0.2], [0.7], [0.9]]))
        return OutcomeUntransformer(outcome.eval(), 1)
    return ChainedTransform(SineCosineTransform(), InverseLengthscaleTransform(kernel))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_transform_module_hooks_checkpoint_serialization_and_device_dtype(family, device):
    like = torch.zeros((), dtype=torch.double, device=device)
    transform = make_transform(family, like)
    assert isinstance(transform, nn.Module)
    query = like.new_tensor([[0.2, 0.5], [0.7, 0.1]])
    expected = transform.forward(query)
    calls = []
    hook = transform.register_forward_hook(lambda module, args, out: calls.append(out))
    torch.testing.assert_close(transform(query), expected)
    hook.remove()
    assert len(calls) == 1
    state = copy.deepcopy(transform.state_dict())
    transform.load_state_dict(state)
    stream = io.BytesIO()
    torch.save(transform, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(restored(query), expected)
    restored.eval().float().cpu()
    assert all(p.dtype == torch.float and p.device.type == "cpu" for p in restored.parameters())
    assert all(b.device.type == "cpu" for b in restored.buffers())
    torch.testing.assert_close(restored(query.float().cpu()), expected.float().cpu())
    if family == "chain":
        assert isinstance(transform.transforms, nn.ModuleList)
        assert next(transform.parameters()) is transform.transforms[1].kernel.raw_lengthscale
    elif family == "selector":
        assert set(dict(transform.named_buffers())) == {"dim", "indices"}
    elif family == "outcome":
        assert "num_outputs" in dict(transform.named_buffers())
        assert transform.transform.stdvs is dict(transform.named_buffers())["transform.stdvs"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_module_transforms_retain_all_query_and_registered_parameter_mixed_derivatives(
    family, device
):
    like = torch.zeros((), dtype=torch.double, device=device)
    transform = make_transform(family, like)
    query = like.new_tensor([[0.2, 0.5], [0.7, 0.1]]).requires_grad_()
    if family == "sincos":
        dense = torch.cat((query.sin(), query.cos()), -1) * transform.scale
    elif family == "lengthscale":
        dense = query / transform.kernel.lengthscale
    elif family == "outputscale":
        dense = query * transform.kernel.outputscale.sqrt()
    elif family == "selector":
        dense = query[..., [1, 0, 1]]
    elif family == "outcome":
        dense = query * transform.transform.stdvs.squeeze(-1) + transform.transform.means.squeeze(
            -1
        )
    else:
        phase = query / transform.transforms[1].kernel.lengthscale
        dense = torch.cat((phase.sin(), phase.cos()), -1)
    actual = transform(query)
    torch.testing.assert_close(actual, dense)
    leaves = (query, *transform.parameters())
    # A ScaleKernel's base lengthscale does not enter output scaling.
    if family == "outputscale":
        leaves = (query, transform.kernel.raw_outputscale)
    first = torch.autograd.grad(actual.sin().sum(), leaves, create_graph=True)
    reference = torch.autograd.grad(dense.sin().sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    other = torch.autograd.grad(sum(g.square().sum() for g in reference), leaves)
    for a, b in zip((*first, *second), (*reference, *other), strict=True):
        torch.testing.assert_close(a, b)
