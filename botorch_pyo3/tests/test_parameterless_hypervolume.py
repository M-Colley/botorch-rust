# ruff: noqa: E402
"""Hypervolume acquisitions use geometry metadata for parameterless models."""

import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
from test_torch_backend_hypervolume import acquisition

from botorch_rs.models.model import Model
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class ParameterlessGaussian(Model):
    num_outputs = 2
    batch_shape = torch.Size()

    def __init__(self, location=None):
        super().__init__()
        if location is not None:
            self.register_buffer("location", location)

    def posterior(self, X, **kwargs):
        location = getattr(self, "location", 0.2)
        x = X[..., 0]
        mean = torch.stack(((2 * x).sin() + location, x.cos() + 0.5 * location), -1)
        distance = (X.unsqueeze(-2) - X.unsqueeze(-3)).square().sum(-1)
        spatial = 0.07 * (-distance).exp() + 0.005 * torch.eye(
            X.shape[-2], dtype=X.dtype, device=X.device
        )
        tasks = X.new_tensor([[1.0, 0.1], [0.1, 0.7]])
        covariance = torch.einsum("tu,...ij->...tiuj", tasks, spatial).reshape(
            *X.shape[:-2], 2 * X.shape[-2], 2 * X.shape[-2]
        )
        return TorchGPyTorchPosterior(mean, covariance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.usefixtures("reference_python_logehvi")
def test_parameterless_hypervolume_values_and_higher_graphs(device, buffered, noisy, log):
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double, requires_grad=True)
    location = query.new_tensor(0.2, requires_grad=True) if buffered else None
    model = ParameterlessGaussian(location)
    assert not tuple(model.parameters())
    baseline = query.new_tensor([[0.05], [0.4], [0.8]])
    observed = query.new_tensor([[0.4, 1.1], [0.9, 0.8]])
    ref = [-0.1, -0.2]
    if noisy:
        kwargs = dict(X_baseline=baseline, cache_root=False, prune_baseline=False)
    else:
        kwargs = dict(partitioning=NondominatedPartitioning(baseline.new_tensor(ref), Y=observed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_acq = acquisition(log, noisy=noisy)(
            model, ref, sampler=SobolQMCNormalSampler((12,), seed=91), **kwargs
        )
        reference_acq = acquisition(log, reference=True, noisy=noisy)(
            model, ref, sampler=ReferenceSampler(torch.Size([12]), seed=91), **kwargs
        )
        actual, expected = actual_acq(query), reference_acq(query)
    assert actual_acq.ref_point.device == query.device
    assert actual_acq.ref_point.dtype == query.dtype
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)
    leaves = (query, location) if buffered else (query,)
    first = torch.autograd.grad(actual.sum(), leaves, create_graph=True, retain_graph=True)
    source_first = torch.autograd.grad(expected.sum(), leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    source_second = torch.autograd.grad(sum(g.square().sum() for g in source_first), leaves)
    for a, b in zip((*first, *second), (*source_first, *source_second), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("tensor_reference", [False, True])
@pytest.mark.usefixtures("reference_python_logehvi")
def test_reference_metadata_follows_geometry_and_preserves_input_graph(
    device, noisy, tensor_reference
):
    # A model's first buffer need not describe its posterior's dtype/device.
    model = ParameterlessGaussian(torch.tensor(0.2, dtype=torch.float))
    baseline = torch.tensor([[0.05], [0.4]], device=device, dtype=torch.double)
    ref = torch.tensor([-0.1, -0.2], dtype=torch.float, requires_grad=True)
    if not tensor_reference:
        ref = [-0.1, -0.2]
    if noisy:
        kwargs = dict(X_baseline=baseline, cache_root=False, prune_baseline=False)
    else:
        kwargs = dict(
            partitioning=NondominatedPartitioning(
                baseline.new_tensor([-0.1, -0.2]), Y=baseline.new_tensor([[0.4, 1.1], [0.9, 0.8]])
            )
        )
    actual = acquisition(True, noisy=noisy)(
        model, ref, sampler=SobolQMCNormalSampler((4,), seed=91), **kwargs
    )
    expected = acquisition(True, reference=True, noisy=noisy)(
        model, ref, sampler=ReferenceSampler(torch.Size([4]), seed=91), **kwargs
    )
    assert actual.ref_point.dtype == expected.ref_point.dtype == baseline.dtype
    assert actual.ref_point.device == expected.ref_point.device == baseline.device
    torch.testing.assert_close(actual.ref_point, expected.ref_point, rtol=0, atol=0)
    if tensor_reference:
        (gradient,) = torch.autograd.grad(actual.ref_point.square().sum(), ref)
        torch.testing.assert_close(gradient, 2 * ref)
