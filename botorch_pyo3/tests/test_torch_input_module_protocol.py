"""Actual public modules retain source state, hooks, migration and aliasing."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import functools
import io

import pytest

torch = pytest.importorskip("torch")

from test_torch_input_model_persistence import FAMILIES
from test_torch_input_models import DEVICES, _case, _data

from botorch_rs.models import SingleTaskGP
from botorch_rs.models._backend_selection import requires_tensor_backend
from botorch_rs.models.transforms import input as port


def _roundtrip(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_all_public_input_families_actual_module_protocol(device, family):
    X, _, Q = _data(device)
    actual, X, Q, _ = _case(family, X, Q)
    if family == "categorical":
        actual.encoders[1] = functools.partial(torch.nn.functional.one_hot, num_classes=2)
    assert isinstance(actual, torch.nn.Module)
    public = type(actual)._native_port_descriptor
    assert isinstance(actual, public)
    assert type(actual).forward is public.forward
    seen = []
    hook = actual.register_forward_hook(lambda module, inputs, output: seen.append(module))
    expected = actual(Q)
    assert seen == [actual]
    hook.remove()
    parameters = dict(actual.named_parameters())
    assert all(isinstance(p, torch.nn.Parameter) for p in parameters.values())
    if family == "warp":
        assert parameters["concentration0"] is actual.concentration0
    elif family == "impute":
        assert parameters["raw_imputation_values"] is actual.raw_imputation_values
    # A trained affine buffer can be non-leaf, matching source deepcopy's
    # restriction. Eval state snapshots deliberately contain detached buffers.
    actual.eval()
    with torch.no_grad():
        expected = actual(Q)
    state = {name: value.detach().clone() for name, value in actual.state_dict().items()}
    restored = _roundtrip(actual)
    torch.testing.assert_close(restored(Q), expected)
    restored.load_state_dict(state)
    assert restored.load_state_dict(state).missing_keys == []
    assert restored.load_state_dict(state).unexpected_keys == []
    restored.to(dtype=torch.float32)
    assert restored(Q.float()).dtype == torch.float32
    assert all(not b.is_floating_point() or b.dtype == torch.float32 for b in restored.buffers())
    restored.train()
    assert all(module.training for module in restored.modules())
    restored.eval()
    assert not any(module.training for module in restored.modules())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family",
    [
        "normalize",
        "standardize",
        "affine",
        "warp",
        "round",
        "log",
        "filter",
        "append",
        "perturb",
        "impute",
    ],
)
def test_input_module_source_state_dict_keys_values_and_load(device, family):
    source = pytest.importorskip("botorch.models.transforms.input")
    X, _, _ = _data(device)
    constructors = {
        "normalize": (
            "Normalize",
            dict(d=3, indices=[2, 0], bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
        ),
        "standardize": ("InputStandardize", dict(d=3, indices=[2, 0])),
        "affine": (
            "AffineInputTransform",
            dict(
                d=3,
                coefficient=X.new_tensor([[0.7, 1.3, 0.9]]),
                offset=X.new_tensor([[0.1, -0.2, 0.3]]),
            ),
        ),
        "warp": (
            "Warp",
            dict(d=3, indices=[2, 0], bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
        ),
        "round": ("Round", dict(integer_indices=[0], approximate=True)),
        "log": ("Log10", dict(indices=[2, 0])),
        "filter": ("FilterFeatures", dict(feature_indices=torch.tensor([2, 0], device=device))),
        "append": ("AppendFeatures", dict(feature_set=X.new_tensor([[0.2]]))),
        "perturb": (
            "InputPerturbation",
            dict(perturbation_set=X.new_tensor([[0.02, -0.01, 0.03]])),
        ),
        "impute": (
            "LearnedFeatureImputation",
            dict(
                feature_indices={0: [0, 2], 1: [1, 2]},
                d=3,
                bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                dtype=X.dtype,
                device=device,
            ),
        ),
    }
    name, options = constructors[family]
    actual, expected = getattr(port, name)(**options), getattr(source, name)(**options)
    actual.to(X)
    expected.to(X)
    if family == "impute":
        X = torch.cat((X, X.new_tensor([[0.0], [1.0], [0.0], [1.0]])), -1)
    actual(X)
    expected(X)
    astate, estate = actual.state_dict(), expected.state_dict()
    assert set(astate) == set(estate)
    for key in astate:
        torch.testing.assert_close(astate[key], estate[key])
    actual.load_state_dict(estate)
    actual.eval()
    expected.eval()
    torch.testing.assert_close(actual(X), expected(X))


def test_default_modules_keep_native_cpu_auto_and_caller_leaves_select_tensor():
    X, Y, _ = (x.detach() for x in _data("cpu"))
    for transform in (
        port.Normalize(3),
        port.Warp(3, [0, 2]),
        port.ChainedInputTransform(n=port.Normalize(3), w=port.Warp(3, [0, 2])),
    ):
        assert not requires_tensor_backend(transform)
        model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None)
        assert not isinstance(model, torch.nn.Module)
    transform = port.Warp(3, [0, 2])
    leaf = torch.nn.Parameter(torch.tensor([1.3, 0.8], dtype=torch.double))
    transform.concentration0 = leaf
    assert dict(transform.named_parameters())["concentration0"] is leaf
    assert requires_tensor_backend(transform)
    model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None)
    assert isinstance(model, torch.nn.Module)
    assert model.input_transform.concentration0 is leaf
    transform = port.Normalize(3)
    handle = transform.register_forward_hook(lambda module, inputs, output: output + 0.01)
    assert requires_tensor_backend(transform)
    model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None)
    assert isinstance(model, torch.nn.Module)
    handle.remove()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
def test_warp_source_initialization_before_migration_retains_factory_precision(device, batch):
    source = pytest.importorskip("botorch.models.transforms.input")
    actual = port.Warp(3, [0, 2], batch_shape=batch)
    expected = source.Warp(3, [0, 2], batch_shape=torch.Size(batch))
    actual.initialize(concentration0=0.9, concentration1=1.3)
    expected.initialize(concentration0=0.9, concentration1=1.3)
    actual.to(device=device, dtype=torch.double)
    expected.to(device=device, dtype=torch.double)
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter, dict(expected.named_parameters())[name], atol=0, rtol=0
        )


def test_native_affine_snapshot_and_double_migration_preserve_original_precision():
    bounds = [[0.1234567890123, -0.2345678901234], [1.2345678901234, 2.3456789012345]]
    actual = port.Normalize(2, bounds=bounds)
    X = [[0.321987654321, 0.876543210987]]
    expected = actual(X)
    assert dict(actual.named_buffers())["_coefficient"].dtype == torch.get_default_dtype()
    assert actual(X) == expected
    actual.double()
    torch.testing.assert_close(
        actual(torch.tensor(X, dtype=torch.double)),
        torch.tensor(expected, dtype=torch.double),
        atol=2e-16,
        rtol=2e-16,
    )
    detached = copy.deepcopy(actual)
    torch.testing.assert_close(
        detached(torch.tensor(X, dtype=torch.double)), actual(torch.tensor(X, dtype=torch.double))
    )


@pytest.mark.parametrize("family", ["append", "perturbation"])
def test_fixed_array_buffers_keep_native_precision_and_live_module_updates(family):
    value = 0.123456789012345
    actual = (
        port.AppendFeatures([[value]])
        if family == "append"
        else port.InputPerturbation([[value]], bounds=[[0.0], [1.0]])
    ).eval()
    name = "feature_set" if family == "append" else "perturbation_set"
    expected = [[0.2, value]] if family == "append" else [[0.2 + value]]
    assert actual([[0.2]]) == expected
    assert getattr(actual, name).dtype == torch.get_default_dtype()
    assert copy.deepcopy(actual)([[0.2]]) == expected
    assert _roundtrip(actual)([[0.2]]) == expected
    actual.double()
    torch.testing.assert_close(
        getattr(actual, name), torch.tensor([[value]], dtype=torch.double), atol=0, rtol=0
    )
    assert actual([[0.2]]) == expected
    with torch.no_grad():
        getattr(actual, name).fill_(0.3)
    updated = [[0.2, 0.3]] if family == "append" else [[0.5]]
    assert actual([[0.2]]) == updated
    state = actual.state_dict()
    state[name] = torch.tensor([[0.4]], dtype=torch.double)
    actual.load_state_dict(state)
    loaded = [[0.2, 0.4]] if family == "append" else [[0.2 + 0.4]]
    assert actual([[0.2]]) == loaded


def test_native_numpy_model_dtype_adoption_retains_affine_array_kind():
    numpy = pytest.importorskip("numpy")
    X = numpy.array([[0.1234567890123], [0.7345678901234], [1.3456789012345]])
    transform = port.Normalize(1)
    SingleTaskGP(X, numpy.sin(X), outcome_transform=None, input_transform=transform)
    assert isinstance(transform.bounds, numpy.ndarray)
    numpy.testing.assert_array_equal(transform.bounds, [[X.min()], [X.max()]])
    assert dict(transform.named_buffers())["_coefficient"].dtype == torch.double
    transform.double()
    assert isinstance(transform.bounds, torch.Tensor)
    torch.testing.assert_close(
        transform.bounds, torch.from_numpy(numpy.array([[X.min()], [X.max()]])), atol=0, rtol=0
    )
