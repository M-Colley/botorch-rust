"""Live parity for independently composable pathwise transforms and data access."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from botorch_rs.models import ModelListGP, SingleTaskGP, SingleTaskVariationalGP  # noqa: E402
from botorch_rs.models.transforms.input import Normalize  # noqa: E402
from botorch_rs.models.transforms.outcome import Standardize  # noqa: E402
from botorch_rs.sampling.pathwise import utils as rs  # noqa: E402

bt = pytest.importorskip("botorch.sampling.pathwise.utils")


def as_kind(x, kind):
    return x if kind == "torch" else x.numpy() if kind == "numpy" else x.tolist()


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_sine_cosine_values_and_composition(kind, batch):
    x = torch.linspace(-2, 2, 12, dtype=torch.float64).reshape(3, 4)
    x = x.expand(*batch, 3, 4)
    native = rs.SineCosineTransform(scale=0.7)
    expected = bt.SineCosineTransform(scale=0.7)(x)
    actual = native(as_kind(x, kind))
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
    chain = rs.ChainedTransform(native, rs.FeatureSelector([2, 0]))
    np.testing.assert_allclose(chain(as_kind(x, kind)), bt.SineCosineTransform(0.7)(x[..., [2, 0]]))


def test_sine_cosine_native_autograd():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(rs.SineCosineTransform(scale), (x,))
    actual = rs.SineCosineTransform(scale)(x).square().sum()
    expected = bt.SineCosineTransform(scale)(x).square().sum()
    ag = torch.autograd.grad(actual, (x, scale))
    eg = torch.autograd.grad(expected, (x, scale))
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_feature_selector(kind, dim):
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    indices = [1, 0, 1]
    actual = rs.FeatureSelector(indices, dim)(as_kind(x, kind))
    expected = bt.FeatureSelector(indices, dim)(x)
    np.testing.assert_array_equal(actual, expected)
    empty = rs.FeatureSelector([], dim)(as_kind(x, kind))
    expected_shape = list(x.shape)
    expected_shape[dim] = 0
    # Nested lists cannot encode axes following an empty axis.
    if kind != "list":
        assert tuple(empty.shape) == tuple(expected_shape)


def test_feature_selector_validation_and_autograd():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    out = rs.FeatureSelector([1, 1, 0])(x)
    torch.testing.assert_close(
        torch.autograd.grad(out.sum(), x)[0], torch.tensor([[1.0, 2.0, 0.0]] * 2, dtype=x.dtype)
    )
    with pytest.raises(TypeError):
        rs.FeatureSelector([1.2])
    with pytest.raises(IndexError):
        rs.FeatureSelector([-1])(x)
    with pytest.raises(IndexError):
        rs.FeatureSelector([3])(x)
    with pytest.raises(IndexError):
        rs.FeatureSelector([0], 3)(x)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_lengthscale_outputscale_values(kind):
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    lengthscale = torch.tensor([[[0.2, 0.4, 0.8, 1.6]], [[0.3, 0.6, 0.9, 1.8]]], dtype=x.dtype)
    kernel = SimpleNamespace(has_lengthscale=True, lengthscale=as_kind(lengthscale, kind))
    np.testing.assert_allclose(
        rs.InverseLengthscaleTransform(kernel)(as_kind(x, kind)), x / lengthscale
    )
    outputscale = torch.tensor([0.4, 1.2], dtype=x.dtype)
    kernel = SimpleNamespace(outputscale=as_kind(outputscale, kind), batch_shape=(2,))
    np.testing.assert_allclose(
        rs.OutputscaleTransform(kernel)(as_kind(x, kind)), outputscale[:, None, None].sqrt() * x
    )


def test_lengthscale_outputscale_gradients():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    lengthscale = torch.tensor([[0.3, 0.8, 1.2]], dtype=x.dtype, requires_grad=True)
    outputscale = torch.tensor(0.7, dtype=x.dtype, requires_grad=True)
    kernel = SimpleNamespace(
        has_lengthscale=True, lengthscale=lengthscale, outputscale=outputscale, batch_shape=()
    )
    out = rs.OutputscaleTransform(kernel)(rs.InverseLengthscaleTransform(kernel)(x)).sum()
    expected = (outputscale.sqrt() * x / lengthscale).sum()
    for a, e in zip(
        torch.autograd.grad(out, (x, lengthscale, outputscale)),
        torch.autograd.grad(expected, (x, lengthscale, outputscale)),
        strict=True,
    ):
        torch.testing.assert_close(a, e)
    with pytest.raises(RuntimeError, match="does not implement"):
        rs.InverseLengthscaleTransform(SimpleNamespace(has_lengthscale=False))


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("m", [1, 2])
def test_outcome_untransformer(kind, m):
    y = torch.tensor([[1.0, 3.0], [2.0, 6.0], [4.0, 9.0]], dtype=torch.float64)[:, :m]
    transform = Standardize(m)
    transform(as_kind(y, kind))
    bt_transform = (
        importlib.import_module("botorch.models.transforms.outcome").Standardize(m).double()
    )
    bt_transform(y)
    values = torch.linspace(-2, 2, 3 * m * 5, dtype=y.dtype).reshape(3, m, 5)
    if m == 1:
        values = values.squeeze(-2)
    actual = rs.OutcomeUntransformer(transform, m)(as_kind(values, kind))
    expected = bt.OutcomeUntransformer(bt_transform, m)(values)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("m", [1, 2])
def test_model_training_data_spaces(training, m):
    x = torch.tensor([[2.0, 5.0], [3.0, 6.0], [4.0, 8.0]], dtype=torch.float64)
    y = torch.tensor([[1.0, 4.0], [2.0, 9.0], [5.0, 6.0]], dtype=x.dtype)[:, :m]
    model = SingleTaskGP(x, y, input_transform=Normalize(2), outcome_transform=Standardize(m))
    model.train(training)
    expected_x = x if m == 1 else x.expand(m, *x.shape)
    expected_y = y.squeeze(-1) if m == 1 else y.T
    torch.testing.assert_close(rs.get_train_inputs(model)[0], expected_x)
    torch.testing.assert_close(
        rs.get_train_inputs(model, transformed=True)[0], model.input_transform(expected_x)
    )
    torch.testing.assert_close(rs.get_train_targets(model), expected_y)
    torch.testing.assert_close(rs.get_train_targets(model, transformed=True), model.train_targets)
    assert rs.get_input_transform(model) is model.input_transform
    assert rs.get_output_transform(model).transform is model.outcome_transform


@pytest.mark.parametrize("m", [1, 2])
def test_variational_training_data_spaces(m):
    x = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float64)
    y = torch.tensor([[1.0, 4.0], [2.0, 9.0], [5.0, 6.0]], dtype=x.dtype)[:, :m]
    model = SingleTaskVariationalGP(
        x, y, input_transform=Normalize(1), outcome_transform=Standardize(m)
    )
    for training in (True, False):
        model.train(training)
        torch.testing.assert_close(rs.get_train_inputs(model)[0], x)
        torch.testing.assert_close(
            rs.get_train_inputs(model, transformed=True)[0], model.model.train_inputs[0]
        )
        torch.testing.assert_close(rs.get_train_targets(model), y.squeeze(-1) if m == 1 else y)
        torch.testing.assert_close(
            rs.get_train_targets(model, transformed=True), model.model.train_targets
        )


def test_dispatch_and_nested_model_lists():
    x = torch.tensor([[0.0], [0.3], [1.0]], dtype=torch.float64)
    y = torch.tensor([[1.0], [2.0], [4.0]], dtype=x.dtype)
    models = [SingleTaskGP(x, y, outcome_transform=None) for _ in range(2)]
    container = ModelListGP(*models)
    for a in rs.get_train_inputs(container):
        torch.testing.assert_close(a[0], x)
    for a in rs.get_train_targets(container):
        torch.testing.assert_close(a, y.squeeze(-1))
    assert rs.get_input_transform(models[0]) is None
    assert rs.get_output_transform(models[0]) is None


def test_transform_mixin_and_abstract_base():
    class Identity(rs.TensorTransform):
        def forward(self, values, **kwargs):
            return values

    class Wrapped(rs.TransformedModuleMixin, Identity):
        input_transform = staticmethod(lambda x: x + 1)
        output_transform = staticmethod(lambda x: x * 2)

    torch.testing.assert_close(Wrapped()(torch.tensor([1.0])), torch.tensor([4.0]))
    with pytest.raises(TypeError):
        rs.TensorTransform()


def test_no_eager_torch_import():
    code = "from botorch_rs.sampling.pathwise.utils import FeatureSelector; import sys; assert 'torch' not in sys.modules; assert FeatureSelector([1])([[1., 2.]]) == [[2.]]"
    subprocess.run([sys.executable, "-c", code], check=True)
