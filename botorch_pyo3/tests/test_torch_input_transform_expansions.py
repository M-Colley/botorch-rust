"""Direct callable, categorical, imputation and one-to-many device transforms."""

# ruff: noqa: E402
from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")
from test_torch_input_transforms import _X, DEVICES, _derivatives

from botorch_rs.models.transforms.input import (
    AppendFeatures,
    BatchBroadcastedInputTransform,
    ChainedInputTransform,
    FilterFeatures,
    InputPerturbation,
    LearnedFeatureImputation,
    Log10,
    Normalize,
    NumericToCategoricalEncoding,
    OneHotToNumeric,
    Warp,
)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize("skip_expand", [False, True])
def test_append_features_source_values_and_full_input_feature_parameter_graphs(
    device, callback, skip_expand
):
    source = pytest.importorskip("botorch.models.transforms.input")
    if callback and skip_expand:
        with pytest.raises(ValueError):
            AppendFeatures(f=lambda x: x, skip_expand=True)
        return
    X = _X(device, (2,))
    features = X.new_tensor([[0.2, 0.4], [0.3, 0.7]], requires_grad=True)
    scale = torch.nn.Parameter(X.new_tensor(0.7))
    if callback:

        def fn(x, scale):
            return torch.stack((x.sin() * scale, x.square() * scale), dim=-2)

        options = dict(f=fn, indices=[2, 0], fkwargs={"scale": scale})
        leaves = (X, scale)
    else:
        options = dict(feature_set=features, skip_expand=skip_expand)
        leaves = (X, features)
    actual, expected = AppendFeatures(**options), source.AppendFeatures(**options)
    result, reference = actual.transform(X), expected.transform(X)
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    assert actual._n_f == 2
    for a, e in zip(_derivatives(result, leaves), _derivatives(reference, leaves), strict=True):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize("multiplicative", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_perturbation_source_layout_bounds_batches_full_graphs(
    device, callback, multiplicative, selected
):
    source = pytest.importorskip("botorch.models.transforms.input")
    X = _X(device, (2,))
    indices = [2, 0] if selected else None
    width = 2 if selected else 3
    perturbations = X.new_full((2, width), 1.05 if multiplicative else 0.02, requires_grad=True)
    bounds = X.new_tensor([[0.1] * width, [0.9] * width], requires_grad=True)
    if callback:

        def fn(x):
            return x.unsqueeze(-2).square() * perturbations

        definition = fn
    else:
        definition = perturbations
    options = dict(
        perturbation_set=definition, bounds=bounds, indices=indices, multiplicative=multiplicative
    )
    actual, expected = InputPerturbation(**options), source.InputPerturbation(**options)
    assert actual(X) is X  # source defaults: skip expansion during training
    result, reference = actual.transform(X), expected.transform(X)
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    assert actual.batch_shape == expected.batch_shape
    assert result.shape == (2, 8, 3)
    leaves = (X, perturbations, bounds)
    for a, e in zip(_derivatives(result, leaves), _derivatives(reference, leaves), strict=True):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_reduction_inverse_and_user_encoder_full_graphs(device):
    source = pytest.importorskip("botorch.models.transforms.input")
    X = _X(device, (2,))
    one_hot = OneHotToNumeric(3, {1: 2})
    reference = source.OneHotToNumeric(3, {1: 2})
    numeric, expected_numeric = one_hot(X), reference(X)
    torch.testing.assert_close(numeric, expected_numeric, atol=0, rtol=0)
    for a, e in zip(_derivatives(numeric, (X,)), _derivatives(expected_numeric, (X,)), strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    torch.testing.assert_close(
        one_hot.untransform(numeric), reference.untransform(expected_numeric), atol=0, rtol=0
    )
    embedding = torch.nn.Embedding(3, 3, dtype=X.dtype, device=device)
    options = dict(dim=2, categorical_features={1: 3}, encoders={1: embedding})
    actual_encoder = NumericToCategoricalEncoding(**options)
    source_encoder = source.NumericToCategoricalEncoding(**options)
    actual, expected = actual_encoder(numeric), source_encoder(expected_numeric)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, e in zip(
        _derivatives(actual, (X, embedding.weight)),
        _derivatives(expected, (X, embedding.weight)),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("include_tasks", [False, True])
def test_imputation_actual_parameter_identity_source_values_and_complete_graphs(
    device, bounded, include_tasks
):
    source = pytest.importorskip("botorch.models.transforms.input")
    X = _X(device, (2,))
    if include_tasks:
        X = (
            torch.cat((X, X.new_tensor([[2], [5], [2], [5]]).expand(2, 4, 1)), -1)
            .detach()
            .requires_grad_()
        )
    bounds = (
        X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True) if bounded else None
    )
    options = dict(
        feature_indices={2: [2, 0], 5: [1]},
        d=3,
        target_task=2,
        bounds=bounds,
        dtype=X.dtype,
        device=device,
    )
    # GPyTorch Interval constructs its bounds in the global default dtype.
    # Use the source's supported double-construction configuration so a
    # float32 intermediate does not quantize the explicit double bounds VJP.
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(X.dtype)
        actual = LearnedFeatureImputation(**options)
        expected = source.LearnedFeatureImputation(**options)
    finally:
        torch.set_default_dtype(default_dtype)
    parameter = torch.nn.Parameter(X.new_tensor([0.2, -0.3, 0.4, 0.0, -0.2, 0.6, 0.1, 0.0]))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    actual.raw_imputation_values = parameter
    expected.raw_imputation_values = reference_parameter
    result, reference = actual(X), expected(X)
    assert actual.raw_imputation_values is parameter
    assert tuple(actual.parameters()) == (parameter,)
    torch.testing.assert_close(actual.imputation_values, expected.imputation_values, atol=0, rtol=0)
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    left = (X, parameter, *((bounds,) if bounded else ()))
    right = (X, reference_parameter, *((bounds,) if bounded else ()))
    for a, e in zip(_derivatives(result, left), _derivatives(reference, right), strict=True):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_chains_batch_wrappers_and_all_expansions_do_not_copy_to_host(device, monkeypatch):
    X = _X(device, (2,))
    features = X.new_tensor([[0.2], [0.5]], requires_grad=True)
    bounds = X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct Tensor expansion must not copy to CPU/NumPy/list")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        append = AppendFeatures(feature_set=features)
        perturb = InputPerturbation(X.new_full((2, 3), 0.02), bounds=bounds)
        impute = LearnedFeatureImputation(
            {2: [0, 2], 5: [1]}, 3, target_task=2, bounds=bounds, dtype=X.dtype, device=device
        )
        categorical = NumericToCategoricalEncoding(
            3, {1: 2}, {1: functools.partial(torch.nn.functional.one_hot, num_classes=2)}
        )
        one_hot = OneHotToNumeric(3, {1: 2})
        for transform in (append, perturb, impute, categorical, one_hot):
            result = transform.transform(X)
            assert result.device == X.device and result.dtype == X.dtype
            assert torch.isfinite(result).all()
            _derivatives(result, (X,))
        chain = ChainedInputTransform(
            normalize=Normalize(3, bounds=bounds),
            warp=Warp(3, [2, 0], bounds=bounds),
            append=AppendFeatures(feature_set=features, transform_on_train=True),
        )
        result = chain(X)
        leaves = (X, features, *chain.parameters())
        _derivatives(result, leaves)
        broadcast = BatchBroadcastedInputTransform([Log10([2, 0]), FilterFeatures([0, 1, 2])])
        actual = broadcast(X)
        expected = torch.stack((Log10([2, 0])(X[0]), X[1]))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for a, e in zip(_derivatives(actual, (X,)), _derivatives(expected, (X,)), strict=True):
            torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_array_export_and_equality_read_actual_features_perturbations_imputation(device):
    X = _X(device)
    array = X.detach().cpu().tolist()
    features = X.new_tensor([[0.2], [0.5]])
    appended = AppendFeatures(feature_set=features)
    torch.testing.assert_close(
        torch.tensor(appended.transform(array), device=device, dtype=X.dtype), appended.transform(X)
    )
    perturbations = X.new_full((2, 3), 0.02)
    perturbed = InputPerturbation(perturbations)
    different = InputPerturbation(perturbations + 0.01)
    assert not perturbed.equals(different)
    torch.testing.assert_close(
        torch.tensor(perturbed.transform(array), device=device, dtype=X.dtype),
        perturbed.transform(X),
    )
    perturbations.add_(0.01)
    assert perturbed.equals(different)
    torch.testing.assert_close(
        torch.tensor(perturbed.transform(array), device=device, dtype=X.dtype),
        perturbed.transform(X),
    )
    imputed = LearnedFeatureImputation(
        {2: [0, 2], 5: [1]}, 3, target_task=2, bounds=X.new_tensor([[0, 0, 0], [1, 1, 1]])
    )
    parameter = torch.nn.Parameter(X.new_tensor([0.2, -0.3, 0.4, 0.0, -0.2, 0.6, 0.1, 0.0]))
    imputed.raw_imputation_values = parameter
    torch.testing.assert_close(
        torch.tensor(imputed.transform(array), device=device, dtype=X.dtype), imputed.transform(X)
    )
    assert imputed.raw_imputation_values is parameter
