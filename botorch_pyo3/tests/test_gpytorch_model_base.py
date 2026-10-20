"""Public native GP base protocol and BoTorch-compatible tensor validation."""

from __future__ import annotations

import pytest

from botorch_rs.exceptions.errors import BotorchTensorDimensionError, InputDataError
from botorch_rs.exceptions.warnings import BotorchTensorDimensionWarning, InputDataWarning
from botorch_rs.models import SACGP, ModelListGP, SingleTaskGP
from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel
from botorch_rs.models.gpytorch import (
    BatchedMultiOutputGPyTorchModel,
    GPyTorchModel,
    ModelListGPyTorchModel,
    MultiTaskGPyTorchModel,
)
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape


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


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("outputs", [1, 2])
def test_independent_output_dimensions_match_official(
    kind: str, batch: tuple, outputs: int
) -> None:
    importorskip_botorch()
    import torch
    from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel as BtBase

    generator = torch.Generator().manual_seed(731)
    x = torch.rand(*batch, 3, 2, generator=generator, dtype=torch.float64)
    y = torch.rand(*batch, 3, outputs, generator=generator, dtype=torch.float64)
    noise = torch.full_like(y, 0.03)
    rs, bt = BatchedMultiOutputGPyTorchModel(), BtBase()
    rs._set_dimensions(as_kind(x.tolist(), kind), as_kind(y.tolist(), kind))
    bt._set_dimensions(x, y)
    assert rs.num_outputs == bt.num_outputs == outputs
    assert rs.batch_shape == tuple(bt.batch_shape) == batch
    assert rs.__dict__["_aug_batch_shape"] == tuple(bt._aug_batch_shape)
    actual = rs._transform_tensor_args(
        as_kind(x.tolist(), kind), as_kind(y.tolist(), kind), as_kind(noise.tolist(), kind)
    )
    expected = bt._transform_tensor_args(x, y, noise)
    for a, b in zip(actual, expected, strict=True):
        check_kind(a, kind)
        assert shape(a) == tuple(b.shape)
        assert_close(a, b, rtol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
def test_real_native_base_list_posterior_and_selection(kind: str) -> None:
    x = as_kind([[0.1], [0.4], [0.8]], kind)
    first = SingleTaskGP(x, as_kind([[0.2, 0.4], [0.7, 0.5], [-0.3, 0.1]], kind))
    second = SingleTaskGP(x, as_kind([[0.5], [0.1], [0.9]], kind))
    base, concrete = ModelListGPyTorchModel(first, second), ModelListGP(first, second)
    assert isinstance(concrete, ModelListGPyTorchModel)
    from botorch_rs.models.model import ModelList

    assert isinstance(concrete, ModelList)
    assert base._get_group_subset_indices([2, 0]) == {1: [0], 0: [0]}
    for indices in [None, [2, 0], [-1]]:
        expected = concrete.posterior(x, output_indices=indices, observation_noise=True)
        actual = base.posterior(x, output_indices=indices, observation_noise=True)
        assert_close(actual.mean, expected.mean)
        assert_close(actual.distribution.covariance_matrix, expected.distribution.covariance_matrix)
    assert not base.training and not first.training and not second.training
    assert isinstance(first, BatchedMultiOutputGPyTorchModel)
    assert issubclass(HigherOrderGP, BatchedMultiOutputGPyTorchModel)
    assert issubclass(MultiTaskGP, MultiTaskGPyTorchModel)


@pytest.mark.parametrize("kind", KINDS)
def test_nonlinear_list_selection_preserves_first_selected_model_order(kind: str) -> None:
    from botorch_rs.models.transforms.outcome import Log
    from botorch_rs.posteriors.posterior_list import PosteriorList

    x = as_kind([[0.1], [0.4], [0.8]], kind)
    first = SingleTaskGP(
        x,
        as_kind([[1.2, 1.4], [1.7, 1.5], [0.7, 1.1]], kind),
        outcome_transform=Log(),
    )
    second = SingleTaskGP(x, as_kind([[0.5], [0.1], [0.9]], kind))
    query = as_kind([[0.3], [0.6]], kind)
    # Selecting the second model first must also place its posterior first.
    # Physical model iteration would silently reverse these output blocks.
    for cls in (ModelListGPyTorchModel, ModelListGP):
        posterior = cls(first, second).posterior(query, output_indices=[2, 1, 0])
        assert isinstance(posterior, PosteriorList)
        assert len(posterior.posteriors) == 2
        expected = (second.posterior(query), first.posterior(query, output_indices=[1, 0]))
        for actual, reference in zip(posterior.posteriors, expected, strict=True):
            check_kind(actual.mean, kind)
            assert_close(actual.mean, reference.mean)
            assert_close(actual.variance, reference.variance)


def test_base_list_broadcast_native_gaussian_views() -> None:
    torch = pytest.importorskip("torch")
    from botorch_rs._arrays import kind_of, to_flat
    from botorch_rs.posteriors.gpytorch import _MomentPosterior

    x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    gp = SingleTaskGP(x, torch.tensor([[0.2], [0.7], [-0.3]], dtype=torch.float64))
    base = ModelListGPyTorchModel(gp)
    p0 = gp.posterior(torch.tensor([[[0.2], [0.6]]], dtype=torch.float64))
    p1 = gp.posterior(torch.tensor([[[0.2], [0.6]], [[0.3], [0.7]]], dtype=torch.float64))
    # Use explicit moments to verify the batch expansion leaves Gaussian
    # means/covariances intact and usable by the native sampler.
    moments = _MomentPosterior(
        to_flat(p0.mean.squeeze(-1)), to_flat(p0.distribution.covariance_matrix), kind_of(x)
    )
    result = base._broadcast_mvns([moments.distribution, p1.distribution])
    assert result[0].batch_shape == result[1].batch_shape == (2,)
    torch.testing.assert_close(result[0].mean, p0.mean.squeeze(-1).expand(2, 2))
    torch.testing.assert_close(
        result[0].covariance_matrix, p0.distribution.covariance_matrix.expand(2, 2, 2)
    )
    assert tuple(result[0].rsample((3,)).shape) == (3, 2, 2)


@pytest.mark.parametrize("fixed_noise", [False, True])
def test_multitask_base_targets_noise_and_sources(fixed_noise: bool) -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0.1, 0], [0.4, 1], [0.8, 0], [0.6, 1]], dtype=torch.float64)
    y = torch.tensor([[0.2], [0.7], [-0.3], [0.5]], dtype=torch.float64)
    noise = torch.full_like(y, 0.03) if fixed_noise else None
    model = MultiTaskGP(x, y, task_feature=1, train_Yvar=noise)
    targets, observed = model._extract_targets_and_noise()
    torch.testing.assert_close(targets.squeeze(-1), model.train_targets)
    assert (observed is not None) == fixed_noise
    if fixed_noise:
        torch.testing.assert_close(observed.squeeze(-1), model.likelihood.noise)
    query = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    implicit = model.posterior(query, output_indices=[1])
    explicit = model.posterior(torch.cat([query, torch.ones_like(query)], dim=-1))
    torch.testing.assert_close(implicit.mean, explicit.mean)
    torch.testing.assert_close(
        implicit.distribution.covariance_matrix, explicit.distribution.covariance_matrix
    )
