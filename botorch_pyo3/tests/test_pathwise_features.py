"""Callable feature-map values and adjoints against official BoTorch."""

from __future__ import annotations

import subprocess
import sys

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from botorch.sampling.pathwise.features.maps import KernelFeatureMap as BtMap  # noqa: E402
from botorch.sampling.pathwise.utils import SineCosineTransform as BtSinCos  # noqa: E402
from gpytorch.kernels import RBFKernel as BtRBF  # noqa: E402

from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel  # noqa: E402
from botorch_rs.sampling.pathwise.features import (  # noqa: E402
    KernelEvaluationMap,
    KernelFeatureMap,
    gen_kernel_features,
)
from botorch_rs.sampling.pathwise.utils import FeatureSelector, SineCosineTransform  # noqa: E402


def as_kind(x, kind):
    return x if kind == "torch" else x.numpy() if kind == "numpy" else x.tolist()


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("batch", [(), (2,), (2, 1)])
@pytest.mark.parametrize("sincos", [False, True])
def test_affine_map_values(kind, batch, sincos):
    g = torch.Generator().manual_seed(913)
    x = torch.randn(*batch, 5, 3, generator=g, dtype=torch.float64)
    w = torch.randn(*batch, 4, 3, generator=g, dtype=x.dtype)
    bias = torch.randn(4, generator=g, dtype=x.dtype)
    kernel = RBFKernel(batch_shape=batch)
    ref_kernel = BtRBF(batch_shape=torch.Size(batch)).double()
    actual = KernelFeatureMap(
        kernel,
        as_kind(w, kind),
        as_kind(bias, kind),
        output_transform=SineCosineTransform(0.7) if sincos else None,
    )
    reference = BtMap(ref_kernel, w, bias, output_transform=BtSinCos(0.7) if sincos else None)
    np.testing.assert_allclose(actual(as_kind(x, kind)), reference(x), rtol=2e-14, atol=2e-14)
    assert actual.num_outputs == (8 if sincos else 4)
    assert actual.batch_shape == batch


def test_affine_map_broadcast_input_weight_bias_gradients():
    g = torch.Generator().manual_seed(74)
    x = torch.randn(2, 1, 5, 3, generator=g, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1, 3, 4, 3, generator=g, dtype=x.dtype, requires_grad=True)
    bias = torch.randn(3, 1, 4, generator=g, dtype=x.dtype, requires_grad=True)
    feature_map = KernelFeatureMap(RBFKernel(batch_shape=(2, 3)), weight, bias)
    actual = feature_map(x)
    expected = x @ weight.transpose(-1, -2) + bias
    torch.testing.assert_close(actual, expected)
    coefficients = torch.randn(actual.shape, generator=g, dtype=x.dtype)
    ag = torch.autograd.grad((actual * coefficients).sum(), (x, weight, bias))
    eg = torch.autograd.grad((expected * coefficients).sum(), (x, weight, bias))
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_kernel_evaluation_map_values(kind, batch):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(*batch, 5, 3, generator=g, dtype=torch.float64)
    points = torch.randn(4, 3, generator=g, dtype=x.dtype)
    kernel = RBFKernel(ard_num_dims=3, batch_shape=batch)
    kernel.lengthscale = [0.3, 0.6, 1.2]
    ref_kernel = BtRBF(ard_num_dims=3, batch_shape=torch.Size(batch)).double()
    ref_kernel.lengthscale = torch.tensor([0.3, 0.6, 1.2], dtype=x.dtype)
    actual = KernelEvaluationMap(
        kernel, as_kind(points, kind), output_transform=FeatureSelector([2, 0, 2])
    )
    np.testing.assert_allclose(
        actual(as_kind(x, kind)),
        ref_kernel(x, points).to_dense()[..., [2, 0, 2]],
        rtol=2e-14,
        atol=2e-14,
    )
    assert actual.num_outputs == 3
    assert actual.batch_shape == batch


def test_kernel_evaluation_map_broadcast_input_point_gradients():
    g = torch.Generator().manual_seed(531)
    x = torch.randn(2, 1, 5, 3, generator=g, dtype=torch.float64, requires_grad=True)
    points = torch.randn(3, 4, 3, generator=g, dtype=x.dtype, requires_grad=True)
    kernel = RBFKernel(ard_num_dims=2, active_dims=[2, 0], batch_shape=(2, 3))
    kernel.lengthscale = [0.3, 0.8]
    ref_kernel = BtRBF(ard_num_dims=2, active_dims=[2, 0], batch_shape=torch.Size([2, 3])).double()
    ref_kernel.lengthscale = torch.tensor([0.3, 0.8], dtype=x.dtype)
    actual = KernelEvaluationMap(kernel, points)(x)
    expected = ref_kernel(x, points).to_dense()
    torch.testing.assert_close(actual, expected)
    coefficients = torch.randn(actual.shape, generator=g, dtype=x.dtype)
    ag = torch.autograd.grad((actual * coefficients).sum(), (x, points))
    eg = torch.autograd.grad((expected * coefficients).sum(), (x, points))
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("family", ["rbf", "matern12", "matern32", "matern52"])
@pytest.mark.parametrize("scaled", [False, True])
def test_fourier_generation_covariance_and_seed(family, scaled):
    kernel = (
        RBFKernel(ard_num_dims=2, active_dims=[2, 0])
        if family == "rbf"
        else MaternKernel(
            nu={"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}[family],
            ard_num_dims=2,
            active_dims=[2, 0],
        )
    )
    kernel.lengthscale = [0.6, 1.2]
    if scaled:
        kernel = ScaleKernel(kernel)
        kernel.outputscale = 1.7
    feature_map = gen_kernel_features(kernel, 3, 16384, seed=19)
    other = gen_kernel_features(kernel, 3, 16384, seed=19)
    np.testing.assert_array_equal(feature_map.weight, other.weight)
    x = np.array([[0.1, 0.2, 0.3], [0.3, -0.7, 0.2], [0.6, 0.5, 0.1]])
    features = feature_map(x)
    np.testing.assert_allclose(features @ features.T, kernel(x), rtol=0.025, atol=0.025)
    assert feature_map.num_outputs == 16384


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_fourier_batched_and_input_derivatives(kind):
    kernel = RBFKernel(ard_num_dims=3, batch_shape=(2,))
    kernel.lengthscale = [[[0.3, 0.6, 1.2]], [[0.6, 1.2, 2.4]]]
    fmap = gen_kernel_features(kernel, 3, 16, seed=23)
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    actual = fmap(as_kind(x, kind))
    weight = torch.as_tensor(fmap.weight, dtype=x.dtype)
    lengthscale = torch.as_tensor(kernel.lengthscale, dtype=x.dtype)
    u = (x / lengthscale) @ weight.transpose(-1, -2)
    expected = torch.cat([u.sin(), u.cos()], dim=-1) * (2 / 16) ** 0.5
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    if kind == "torch":
        x.requires_grad_()
        assert torch.autograd.gradcheck(fmap, (x,))


def test_feature_validation_and_dependency_free_use():
    with pytest.raises(RuntimeError, match="Shape mismatch"):
        KernelEvaluationMap(RBFKernel(batch_shape=(2,)), [[[1.0]]] * 3)
    with pytest.raises(NotImplementedError):
        gen_kernel_features(RBFKernel(), 1, 3)
    code = "from botorch_rs.kernels import RBFKernel; from botorch_rs.sampling.pathwise.features import gen_kernel_features; import sys; f=gen_kernel_features(RBFKernel(),1,4,seed=1); assert len(f([[0.]] )[0])==4; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
