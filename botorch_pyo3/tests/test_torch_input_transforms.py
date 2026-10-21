"""Direct public input transforms preserve device operations and source graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.input import (
    AffineInputTransform,
    FilterFeatures,
    InputStandardize,
    InteractionFeatures,
    Log10,
    Normalize,
    Round,
    Warp,
)
from botorch_rs.models.utils.assorted import fantasize

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _X(device, batch=()):
    values = torch.tensor(
        [[0.21, 0.71, 0.31], [0.43, 0.27, 0.62], [0.64, 0.83, 0.48], [0.82, 0.41, 0.79]],
        dtype=torch.double,
        device=device,
    )
    return values.expand(*batch, 4, 3).clone().requires_grad_()


def _derivatives(output, leaves):
    loss = output.sin().sum() + 0.2 * output.square().sum()
    first = torch.autograd.grad(loss, leaves, create_graph=True, retain_graph=True)
    return (
        *first,
        *torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["normalize", "standardize", "affine"])
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("reverse", [False, True])
def test_direct_affine_source_values_stats_first_and_second_graphs(device, family, batch, reverse):
    source_module = pytest.importorskip("botorch.models.transforms.input")
    X = _X(device, (3, *batch))
    options = {"d": 3, "indices": [2, 0], "batch_shape": batch, "reverse": reverse}
    cls, source_cls = {
        "normalize": (Normalize, source_module.Normalize),
        "standardize": (InputStandardize, source_module.InputStandardize),
        "affine": (AffineInputTransform, source_module.AffineInputTransform),
    }[family]
    if family == "affine":
        options.update(
            coefficient=X.new_tensor([[0.7, 1.3]]).expand(*batch, 1, 2).clone(),
            offset=X.new_tensor([[0.1, -0.3]]).expand(*batch, 1, 2).clone(),
        )
    actual, expected = cls(**options), source_cls(**options)
    for method in ("transform", "untransform"):
        result, reference = getattr(actual, method)(X), getattr(expected, method)(X)
        torch.testing.assert_close(result, reference, atol=3e-13, rtol=3e-12)
        torch.testing.assert_close(actual.offset, expected.offset, atol=3e-13, rtol=3e-12)
        torch.testing.assert_close(actual.coefficient, expected.coefficient, atol=3e-13, rtol=3e-12)
        for a, e in zip(_derivatives(result, (X,)), _derivatives(reference, (X,)), strict=True):
            torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)
    actual.eval()
    expected.eval()
    query = (_X(device, (3, *batch)) + 0.03).requires_grad_()
    result, reference = actual(query), expected(query)
    torch.testing.assert_close(result, reference, atol=3e-13, rtol=3e-12)
    assert actual.offset.requires_grad is expected.offset.requires_grad is False
    assert torch.autograd.grad(result.sum(), X, allow_unused=True)[0] is None
    for a, e in zip(_derivatives(result, (query,)), _derivatives(reference, (query,)), strict=True):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("reverse", [False, True])
def test_direct_warp_live_parameters_source_values_all_graphs_and_inverse(device, batch, reverse):
    source_module = pytest.importorskip("botorch.models.transforms.input")
    X = _X(device, batch)
    bounds = X.new_tensor([[0.1, -0.3, 0.1], [0.9, 1.2, 0.9]])
    options = dict(
        d=3, indices=[2, 0], batch_shape=torch.Size(batch), bounds=bounds, eps=0.02, reverse=reverse
    )
    actual, expected = Warp(**options), source_module.Warp(**options).to(X)
    parameters, reference_parameters = [], []
    for i, value in ((0, 1.3), (1, 0.8)):
        shape = (*batch, 1, 2) if batch else (2,)
        parameter = torch.nn.Parameter(X.new_full(shape, value))
        setattr(actual, f"concentration{i}", parameter)
        parameters.append(parameter)
        expected._set_concentration(i, X.new_full(shape, value))
        reference_parameters.append(getattr(expected, f"concentration{i}"))
    result, reference = actual(X), expected(X)
    assert actual.concentration0 is parameters[0] and actual.concentration1 is parameters[1]
    assert list(actual.parameters()) == parameters
    torch.testing.assert_close(result, reference, atol=3e-13, rtol=3e-12)
    for a, e in zip(
        _derivatives(result, (X, *parameters)),
        _derivatives(reference, (X, *reference_parameters)),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)
    torch.testing.assert_close(actual.untransform(result), X, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["round", "approximate_round", "log", "filter", "interaction"])
def test_stateless_direct_source_values_and_first_second_graphs(device, family):
    source = pytest.importorskip("botorch.models.transforms.input")
    X = _X(device)
    cls, source_cls, options = {
        "round": (Round, source.Round, dict(integer_indices=[0], categorical_features={1: 2})),
        "approximate_round": (
            Round,
            source.Round,
            dict(integer_indices=[2, 0], approximate=True, tau=0.2),
        ),
        "log": (Log10, source.Log10, dict(indices=[2, 0])),
        "filter": (
            FilterFeatures,
            source.FilterFeatures,
            dict(feature_indices=torch.tensor([2, 0], device=device)),
        ),
        "interaction": (InteractionFeatures, source.InteractionFeatures, dict(indices=[2, 0])),
    }[family]
    actual, expected = cls(**options), source_cls(**options)
    result, reference = actual(X), expected(X)
    torch.testing.assert_close(result, reference, atol=3e-13, rtol=3e-12)
    for a, e in zip(_derivatives(result, (X,)), _derivatives(reference, (X,)), strict=True):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_constructor_forward_inverse_preprocess_and_warp_model_never_copy_inputs_to_host(
    device, monkeypatch
):
    X = _X(device)
    bounds = X.new_tensor([[0.1, -0.3, 0.1], [0.9, 1.2, 0.9]])

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor input transforms must not copy data to CPU/NumPy/list")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        normalize = Normalize(3, bounds=bounds, indices=[2, 0])
        standardize = InputStandardize(3, indices=[2, 0])
        warp = Warp(3, [2, 0], bounds=bounds, eps=0.02)
        transforms = [
            normalize,
            standardize,
            warp,
            Round([0]),
            Log10([2, 0]),
            FilterFeatures(torch.tensor([2, 0], device=device)),
            InteractionFeatures([2, 0]),
        ]
        for transform in transforms:
            result = transform.transform(X)
            assert result.device == X.device and result.dtype == X.dtype
            _derivatives(result, (X, *list(transform.parameters())))
        for transform in (normalize, standardize, warp):
            result = transform.transform(X)
            torch.testing.assert_close(transform.untransform(result), X, atol=3e-12, rtol=3e-11)
            transform.preprocess_transform(X)
        model = SingleTaskGP(
            X, X.new_tensor([[-0.4], [0.8], [0.1], [-0.2]]), input_transform=warp, backend="torch"
        )
        assert model.input_transform.concentration0 is warp.concentration0
        assert model.input_transform.concentration1 is warp.concentration1
        posterior = model.posterior(X[1:3])
        _derivatives(posterior.mean, (X, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
def test_flags_preprocess_constant_standardize_and_native_array_state(device):
    X = _X(device)
    transform = Normalize(3, transform_on_train=False)
    assert transform(X) is X and transform.preprocess_transform(X) is X
    transform.eval()
    transform.transform_on_fantasize = False
    with fantasize():
        assert transform(X) is X
    standardized = InputStandardize(3)
    constant = torch.ones(4, 3, dtype=X.dtype, device=device, requires_grad=True)
    result = standardized(constant)
    torch.testing.assert_close(result, constant)
    assert all(torch.isfinite(value).all() for value in _derivatives(result, (constant,)))
    normalize = Normalize(3)
    trained = normalize(X)
    normalize.eval()
    # Array calls deliberately export their fitted state to the native path.
    actual = normalize(X.detach().cpu().tolist())
    torch.testing.assert_close(torch.tensor(actual, device=device, dtype=X.dtype), trained)
    duplicate = Normalize(**normalize.get_init_args()).eval()
    torch.testing.assert_close(duplicate(X), trained)
    clone = copy.deepcopy(normalize)
    torch.testing.assert_close(clone(X), trained)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("plain", [False, True])
def test_list_bounds_materialize_both_tensor_statistics_atomically(device, plain):
    X = _X(device)
    bounds = [[0.1, -0.3, 0.1], [0.9, 1.2, 0.9]]
    options = {} if plain else {"indices": [2, 0]}
    transform = Normalize(3, bounds=bounds, **options).eval()
    result = transform(X)
    expected = X.clone()
    indices = [0, 1, 2] if plain else [2, 0]
    expected[..., indices] = (X[..., indices] - X.new_tensor(bounds[0])[indices]) / (
        X.new_tensor(bounds[1])[indices] - X.new_tensor(bounds[0])[indices]
    )
    torch.testing.assert_close(result, expected, atol=3e-13, rtol=3e-12)
    for a, e in zip(_derivatives(result, (X,)), _derivatives(expected, (X,)), strict=True):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-11)
    Y = X.new_tensor([[-0.4], [0.8], [0.1], [-0.2]])
    model = SingleTaskGP(
        X, Y, input_transform=Normalize(3, bounds=bounds, **options), backend="torch"
    )
    torch.testing.assert_close(model.transform_inputs(X), expected, atol=3e-13, rtol=3e-12)
    native = SingleTaskGP(
        X.detach(), Y, input_transform=Normalize(3, bounds=bounds, **options), backend="native"
    )
    torch.testing.assert_close(native.transform_inputs(X), expected, atol=3e-13, rtol=3e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_selected_tensor_normalize_exports_current_native_descriptor(device):
    from botorch_rs._arrays import from_pair, to_flat

    X = _X(device).detach()
    bounds = X.new_tensor([[0.1, -0.2, 0.15], [0.9, 1.2, 0.95]])
    transform = Normalize(3, indices=[2, 0], bounds=bounds)
    expected = (X[..., [2, 0]] - bounds[0, [2, 0]]) / (bounds[1, [2, 0]] - bounds[0, [2, 0]])
    native = from_pair(transform._rs.transform(to_flat(X[..., [2, 0]])), transform._kind)
    torch.testing.assert_close(native, expected)
    # Selected-column native descriptors must receive direct Tensor updates.
    transform._coefficient = transform._coefficient * 1.4
    native = from_pair(transform._rs.transform(to_flat(X[..., [2, 0]])), transform._kind)
    torch.testing.assert_close(native, expected / 1.4)


@pytest.mark.parametrize("device", DEVICES)
def test_native_pairwise_owner_refreshes_existing_warp_leaf(device):
    from botorch_rs.models.pairwise_gp import PairwiseGP

    X = _X(device).detach()
    warp = Warp(3, [2, 0])
    warp(X)
    held0, held1 = warp.concentration0, warp.concentration1
    model = PairwiseGP(X, [[0, 1], [2, 1], [2, 3]], input_transform=warp, backend="native")
    raw = list(model._rs.raw_parameters())
    raw[-4:] = [1.3, 0.8, 0.9, 1.2]
    model._rs.set_raw_parameters(raw)
    assert warp.concentration0 is held0 and warp.concentration1 is held1
    torch.testing.assert_close(held0, X.new_tensor([1.3, 0.8]))
    torch.testing.assert_close(held1, X.new_tensor([0.9, 1.2]))
    saved_loss = held0.square().sum() + held1.square().sum()
    versions = held0._version, held1._version
    assert warp.concentration0 is held0 and warp.concentration1 is held1
    assert (held0._version, held1._version) == versions
    gradients = torch.autograd.grad(saved_loss, (held0, held1))
    torch.testing.assert_close(gradients[0], 2 * held0)
    torch.testing.assert_close(gradients[1], 2 * held1)
