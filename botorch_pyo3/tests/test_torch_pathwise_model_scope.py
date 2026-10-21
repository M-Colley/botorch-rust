"""Source-supported ensemble paths and structured callback construction."""

# ruff: noqa: E402
from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.sampling.pathwise import draw_kernel_feature_paths as source_prior
from botorch.sampling.pathwise import draw_matheron_paths as source_paths
from botorch.sampling.pathwise.paths import SamplePath as SourcePath
from mo_entropy_helpers import recorded_randn
from test_torch_fully_bayesian import _pair

from botorch_rs.sampling.pathwise import draw_kernel_feature_paths, draw_matheron_paths
from botorch_rs.sampling.pathwise.paths import SamplePath

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["FullyBayesianSingleTaskGP", "SaasFullyBayesianSingleTaskGP"])
@pytest.mark.parametrize("ensemble_as_batch", [False, True])
def test_loaded_matern_paths_match_unmodified_source(device, family, ensemble_as_batch):
    (actual_model, source_model), data, _ = _pair(device, family, transforms=True)
    noises = []
    with recorded_randn(noises):
        expected = source_paths(
            source_model,
            torch.Size([2]),
            prior_sampler=functools.partial(source_prior, num_features=16),
        )
    actual = draw_matheron_paths(
        actual_model,
        torch.Size([2]),
        prior_sampler=functools.partial(draw_kernel_feature_paths, num_features=16),
        draws=[
            (expected["prior_paths"].feature_map.weight, expected["prior_paths"].weight, noises[0])
        ],
    )
    for path in (actual, expected):
        path.set_ensemble_as_batch(ensemble_as_batch)
    query = data[0][0][[1, 3]].clone()
    if ensemble_as_batch:
        query = query.unsqueeze(0).expand(3, -1, -1).clone()
    query.requires_grad_()
    values = [path(query) for path in (actual, expected)]
    assert values[0].shape == (2, 3, 2)
    torch.testing.assert_close(*values, atol=3e-8, rtol=3e-8)
    gradients = [torch.autograd.grad(value.square().sum(), query)[0] for value in values]
    torch.testing.assert_close(*gradients, atol=3e-7, rtol=3e-7)
    assert actual["prior_paths"].is_ensemble and actual["update_paths"].is_ensemble


def _structured_pair(device, family):
    if family == "kronecker":
        from test_torch_kronecker import data, paired

        X, Y, query = data(device)
        actual, source = paired(X, Y)
    elif family == "higher":
        from test_torch_higher_order import data, paired

        X, Y, query = data(device)
        actual, source = paired(X, Y, outcome_transform=None)
    else:
        from test_torch_latent_kronecker import data, paired

        X, T, Y, query, _ = data(device)
        actual, source = paired(X, T, Y)
    for model in (actual, source):
        model.eval()
    return actual, source, query


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kronecker", "higher"])
def test_structured_custom_callbacks_match_source_and_retain_leaves(device, family):
    actual, source, query = _structured_pair(device, family)
    rows = []
    # Run the unmodified source first: these callbacks are a valid public call,
    # independent of whether the model has a default Fourier feature generator.
    for model, constructor, path_base in (
        (source, source_paths, SourcePath),
        (actual, draw_matheron_paths, SamplePath),
    ):
        calls = []
        gain = torch.nn.Parameter(query.new_tensor(0.7))
        update_gain = torch.nn.Parameter(query.new_tensor(0.3))

        class Path(path_base):
            def __init__(self, parameter, shift, shape):
                super().__init__()
                self.gain = parameter
                self.register_buffer("shift", shift)
                self.shape = shape

            def forward(self, X):
                result = (X.sum(-1) * self.gain).sin() + self.shift
                return result.expand(*self.shape, *result.shape)

            def set_ensemble_as_batch(self, value):
                self.ensemble_as_batch = value

        def prior(model, sample_shape, calls=calls, gain=gain, path_type=Path):
            calls.append(("prior", model))
            return path_type(gain, query.new_zeros(()), sample_shape)

        def update(model, sample_values, target_values, calls=calls, gain=update_gain, path_type=Path):
            calls.append(("update", model, sample_values, target_values))
            return path_type(gain, sample_values.mean() + target_values.mean(), (2,))

        path = constructor(model, torch.Size([2]), prior_sampler=prior, update_strategy=update)
        assert calls[0] == ("prior", model)
        assert calls[1][0:2] == ("update", model)
        torch.testing.assert_close(calls[1][3], model.train_targets)
        assert any(parameter is gain for parameter in path.parameters())
        assert any(parameter is update_gain for parameter in path.parameters())
        point = query.clone().requires_grad_()
        value = path(point)
        first = torch.autograd.grad(
            value.square().sum(), (point, gain, update_gain), create_graph=True
        )
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), (point, gain, update_gain)
        )
        rows.append((value, first, second))
    for actual_row, source_row in zip(rows[1], rows[0], strict=True):
        if isinstance(actual_row, tuple):
            for a, b in zip(actual_row, source_row, strict=True):
                torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
        else:
            torch.testing.assert_close(actual_row, source_row, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_source_latent_kronecker_two_input_dispatch_boundary(device):
    _, source, _ = _structured_pair(device, "latent")
    # Even explicit samplers cannot override the ExactGP host's single-input
    # destructuring. This is not a source-supported custom-callback call.
    with pytest.raises(ValueError, match="too many values to unpack"):
        source_paths(
            source,
            torch.Size([2]),
            prior_sampler=lambda **kwargs: None,
            update_strategy=lambda **kwargs: None,
        )
