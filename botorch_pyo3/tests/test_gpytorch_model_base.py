"""Public native GP base protocol and BoTorch-compatible tensor validation."""

from __future__ import annotations

import pytest

from botorch_rs.exceptions.errors import BotorchTensorDimensionError, InputDataError
from botorch_rs.exceptions.warnings import BotorchTensorDimensionWarning, InputDataWarning
from botorch_rs.models import SACGP, SingleTaskGP
from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel
from botorch_rs.models.gpytorch import GPyTorchModel
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from helpers import KINDS, as_kind


@pytest.mark.parametrize("kind", KINDS)
def test_native_gp_inheritance_and_valid_arrays(kind: str) -> None:
    x, y = as_kind([[0.1], [0.4], [0.8]], kind), as_kind([[0.2], [0.7], [-0.3]], kind)
    GPyTorchModel._validate_tensor_args(x, y)
    model = SingleTaskGP(x, y)
    assert isinstance(model, GPyTorchModel)
    assert model.batch_shape == () and model.num_outputs == 1
    assert isinstance(SACGP(x, y, None, {"context": [0]}), GPyTorchModel)
    for cls in (ApproximateGPyTorchModel, HigherOrderGP, MultiTaskGP, KroneckerMultiTaskGP):
        assert issubclass(cls, GPyTorchModel)


def test_base_batch_and_output_properties() -> None:
    class NativeGP(GPyTorchModel):
        train_inputs = ([[[0.1], [0.3]]],)
        _num_outputs = 2

    model = NativeGP()
    assert model.batch_shape == (1,) and model.num_outputs == 2
    with pytest.raises(NotImplementedError):
        model.posterior([[0.1]])


def test_strict_shape_validation_and_non_strict_warning() -> None:
    x, y = [[0.1], [0.3]], [0.2, 0.7]
    with pytest.raises(BotorchTensorDimensionError, match="explicit output dimension"):
        GPyTorchModel._validate_tensor_args(x, y)
    with pytest.warns(BotorchTensorDimensionWarning, match="Non-strict enforcement"):
        GPyTorchModel._validate_tensor_args(x, y, strict=False)
    with pytest.raises(BotorchTensorDimensionError, match="observation noise"):
        GPyTorchModel._validate_tensor_args(x, [[0.2], [0.7]], [0.01, 0.01])


def test_dtype_validation_and_single_precision_warning() -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0.1], [0.3]], dtype=torch.float64)
    y = torch.tensor([[0.2], [0.7]], dtype=torch.float32)
    with pytest.raises(InputDataError, match="same dtype"):
        GPyTorchModel._validate_tensor_args(x, y)
    with pytest.warns(InputDataWarning, match="double precision"):
        GPyTorchModel._validate_tensor_args(x.float(), y)
