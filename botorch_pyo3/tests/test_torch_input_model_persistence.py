"""Public transform model serialization preserves live state and aliases."""

# ruff: noqa: E402
from __future__ import annotations

import functools
import io
import pickle

import pytest

torch = pytest.importorskip("torch")

from test_torch_input_models import DEVICES, _case, _data, _derivatives

from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.input import AppendFeatures, ChainedInputTransform, Warp
from botorch_rs.priors import LogNormalPrior

FAMILIES = [
    "normalize",
    "standardize",
    "affine",
    "warp",
    "round",
    "log",
    "filter",
    "interaction",
    "append",
    "perturb",
    "impute",
    "onehot",
    "categorical",
    "chain",
    "broadcast",
]


class AppendedLinear(torch.nn.Module):
    def __init__(self, like):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1, dtype=like.dtype, device=like.device)

    def forward(self, value):
        return self.linear(value).unsqueeze(-2)


def _roundtrip(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_whole_model_pickle_all_public_transform_families(device, family):
    X, Y, Q = _data(device)
    transform, X, Q, _ = _case(family, X, Q)
    if family == "categorical":
        transform.encoders[1] = functools.partial(torch.nn.functional.one_hot, num_classes=2)
    if X.ndim == 3:
        Y = Y.expand(*X.shape[:-2], *Y.shape).clone().detach().requires_grad_()
    model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None, backend="torch")
    expected = model.posterior(Q)
    restored = _roundtrip(model)
    actual = restored.posterior(Q)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )
    leaves = (restored._train_X, restored._train_Y, Q, *restored.parameters())
    first, second = _derivatives(actual.mean, actual.distribution.covariance_matrix, leaves)
    assert all(torch.isfinite(value).all() for value in (*first, *second))
    if family in ("warp", "impute"):
        for name, parameter in restored.input_transform._source.named_parameters():
            assert dict(restored.input_transform.named_parameters())[name] is parameter
    restored.to(dtype=torch.float32)
    assert restored.posterior(Q.float()).mean.dtype == torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_whole_model_pickle_keeps_caller_module_and_parameter_aliases(device):
    X, Y, Q = _data(device)
    feature = AppendedLinear(X)
    warp = Warp(4, [0, 2], concentration0_prior=LogNormalPrior(0.0, 0.7))
    source = ChainedInputTransform(
        append=AppendFeatures(f=feature, transform_on_train=True), first=warp, second=warp
    )
    model = SingleTaskGP(X, Y, input_transform=source, outcome_transform=None, backend="torch")
    expected = model.posterior(Q)
    restored, module, leaf = _roundtrip((model, feature, warp.concentration0))
    assert restored.input_transform.append._f is module
    assert restored.input_transform.first is restored.input_transform.second
    assert restored.input_transform.first.concentration0 is leaf
    assert restored.input_transform._source.first is restored.input_transform._source.second
    assert len(list(restored.input_transform.named_priors())) == 1
    actual = restored.posterior(Q)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )
    _derivatives(actual.mean, actual.distribution.covariance_matrix, (Q, *restored.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family", ["normalize", "warp", "impute", "onehot", "categorical", "perturb"]
)
def test_public_descriptor_pickle_keeps_tensor_and_array_values(device, family):
    X, _, Q = _data(device)
    transform, X, Q, _ = _case(family, X, Q)
    if family == "categorical":
        transform.encoders[1] = functools.partial(torch.nn.functional.one_hot, num_classes=2)
    transform(X)
    transform.eval()
    expected = transform(Q)
    restored = pickle.loads(pickle.dumps(transform))
    torch.testing.assert_close(restored(Q), expected)
    torch.testing.assert_close(
        torch.as_tensor(restored(Q.detach().cpu().tolist()), device=device, dtype=Q.dtype),
        expected.detach(),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family", ["normalize", "filter", "interaction", "append", "impute", "onehot", "categorical"]
)
def test_unmodified_source_model_whole_object_pickle(device, family):
    inputs = pytest.importorskip("botorch.models.transforms.input")
    source_gp = pytest.importorskip("botorch.models.gp_regression").SingleTaskGP
    X, Y, Q = _data(device)
    transform, X, Q, _ = _case(family, X, Q)
    if family == "normalize":
        source = inputs.Normalize(3)
    elif family == "filter":
        source = inputs.FilterFeatures(torch.tensor([2, 0], device=device))
    elif family == "interaction":
        source = inputs.InteractionFeatures([2, 0])
    elif family == "append":
        source = inputs.AppendFeatures(feature_set=transform.feature_set, transform_on_train=True)
    elif family == "impute":
        source = inputs.LearnedFeatureImputation(
            {0: [0, 2], 1: [1, 2]}, 3, dtype=X.dtype, device=X.device
        )
    elif family == "onehot":
        source = inputs.OneHotToNumeric(3, {1: 2})
    else:
        source = inputs.NumericToCategoricalEncoding(
            2, {1: 2}, {1: functools.partial(torch.nn.functional.one_hot, num_classes=2)}
        )
    model = source_gp(X, Y, input_transform=source, outcome_transform=None).to(X)
    expected = model.posterior(Q)
    restored = _roundtrip(model)
    actual = restored.posterior(Q)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )


@pytest.mark.parametrize("factory_dtype", [torch.float32, torch.float64])
def test_prior_native_descriptor_roundtrip_preserves_factory_precision(factory_dtype):
    from botorch_rs.models.utils.priors import BetaPrior
    from botorch_rs.priors import (
        GammaPrior,
        HalfCauchyPrior,
        HalfNormalPrior,
        NormalPrior,
        SmoothedBoxPrior,
        UniformPrior,
    )

    value = torch.tensor(0.731234567891, dtype=factory_dtype)
    priors = [
        LogNormalPrior(value, value),
        GammaPrior(value, value),
        NormalPrior(value, value),
        UniformPrior(value / 2, value * 3),
        HalfCauchyPrior(value),
        HalfNormalPrior(value),
        BetaPrior(value, value),
        SmoothedBoxPrior(value / 2, value * 3, sigma=0.1),
    ]
    for prior in priors:
        restored = pickle.loads(pickle.dumps(prior))
        assert restored._rs.kind == prior._rs.kind
        assert restored._rs.parameters == prior._rs.parameters
        for point in (0.47, 0.73, 0.92):
            assert restored._rs.log_prob(point) == prior._rs.log_prob(point)
            assert restored._rs.grad_log_prob(point) == prior._rs.grad_log_prob(point)
