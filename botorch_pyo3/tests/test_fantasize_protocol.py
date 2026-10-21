"""Generic fantasy conditioning preserves noise, context and array semantics."""

from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from botorch_rs import _ops, settings
from botorch_rs.exceptions.errors import DeprecationError
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.model import FantasizeMixin, Model
from botorch_rs.models.utils.assorted import fantasize
from helpers import as_kind, assert_close


class _RecordingModel(Model, FantasizeMixin):
    def __init__(self, outputs: int = 1, likelihood: Any = None) -> None:
        self._outputs = outputs
        self.likelihood = likelihood
        self.calls: list[tuple[Any, bool, bool]] = []

    @property
    def num_outputs(self) -> int:
        return self._outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    def posterior(self, X: Any, **kwargs: Any) -> Any:
        self.calls.append(
            (kwargs["observation_noise"], fantasize.on(), settings.propagate_grads.on())
        )
        return SimpleNamespace(mean=X)

    def condition_on_observations(self, X: Any, Y: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(X=X, Y=Y, kwargs=kwargs, in_fantasize=fantasize.on())


class _Sampler:
    def __init__(self, shape: tuple[int, ...], outputs: int) -> None:
        self.sample_shape = shape
        self.outputs = outputs
        self.called = False

    def __call__(self, posterior: Any) -> Any:
        self.called = True
        shape = _ops.shape_of(posterior.mean)
        return _ops.expand(
            posterior.mean,
            (*self.sample_shape, *shape[:-1], self.outputs),
        )


@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("sample_shape", [(3,), (2, 3)])
def test_fixed_noise_uses_average_and_expands_without_fantasy_axes(
    kind: str, outputs: int, sample_shape: tuple[int, ...]
) -> None:
    x = as_kind([[0.0], [1.0]], kind)
    y = as_kind([[0.1] * outputs, [0.2] * outputs], kind)
    noise = as_kind([[0.1, 0.3][:outputs], [0.3, 0.7][:outputs]], kind)
    native = SingleTaskGP(x, y, train_Yvar=noise, outcome_transform=None)
    model = _RecordingModel(outputs, native.likelihood)
    query = as_kind([[0.2], [0.4], [0.8]], kind)
    result = model.fantasize(query, _Sampler(sample_shape, outputs), sentinel="kept")
    assert _ops.shape_of(result.Y) == (*sample_shape, 3, outputs)
    assert _ops.shape_of(result.kwargs["noise"]) == (3, outputs)
    assert_close(result.kwargs["noise"], [[0.2, 0.5][:outputs]] * 3)
    assert_close(model.calls[0][0], [[0.2, 0.5][:outputs]])
    assert model.calls[0][1:] == (True, False)
    assert result.in_fantasize and result.kwargs["sentinel"] == "kept"
    assert fantasize.off() and settings.propagate_grads.off()


def test_generic_fantasize_preserves_graph_and_scopes_settings() -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0.2], [0.8]], dtype=torch.double, requires_grad=True)
    noise = torch.tensor([[0.04]], dtype=torch.double, requires_grad=True)
    model = _RecordingModel()
    result = model.fantasize(x, _Sampler((2, 3), 1), observation_noise=noise, propagate_grads=True)
    assert model.calls[0][1:] == (True, True)
    assert "propagate_grads" not in result.kwargs
    (result.Y.sum() + result.kwargs["noise"].sum()).backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 6.0))
    torch.testing.assert_close(noise.grad, torch.full_like(noise, 2.0))
    assert fantasize.off() and settings.propagate_grads.off()


@pytest.mark.parametrize("empty", [False, True])
def test_generic_fantasy_matches_official_conditioning_arguments(empty: bool) -> None:
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("botorch.models.model")
    x = torch.empty(0, 1, dtype=torch.double) if empty else torch.tensor([[0.2], [0.8]])
    noise = torch.tensor([[0.04]])
    actual = _RecordingModel().fantasize(x, _Sampler((3,), 1), observation_noise=noise)
    expected = reference.FantasizeMixin.fantasize(
        _RecordingModel(), x, _Sampler((3,), 1), observation_noise=noise
    )
    torch.testing.assert_close(actual.X, expected.X)
    torch.testing.assert_close(actual.Y, expected.Y)
    torch.testing.assert_close(actual.kwargs["noise"], expected.kwargs["noise"])


@pytest.mark.parametrize("empty", [False, True])
def test_multidimensional_noise_matches_corrected_official_protocol(empty: bool) -> None:
    torch = pytest.importorskip("torch")
    reference = pytest.importorskip("botorch.models.model")
    method = reference.FantasizeMixin.fantasize
    source = textwrap.dedent(inspect.getsource(method))
    source = source.replace("Y.shape[1:]", "Y.shape[len(sampler.sample_shape):]")
    source = source.replace(
        "Y_fantasized.shape[1:]", "Y_fantasized.shape[len(sampler.sample_shape):]"
    )
    namespace = dict(method.__globals__)
    exec(compile(source, "<corrected official fantasy method>", "exec"), namespace)
    x = torch.empty(0, 1, dtype=torch.double) if empty else torch.tensor([[0.2], [0.8]])
    noise = torch.tensor([[0.04]])
    actual = _RecordingModel().fantasize(x, _Sampler((2, 3), 1), observation_noise=noise)
    expected = namespace["fantasize"](
        _RecordingModel(), x, _Sampler((2, 3), 1), observation_noise=noise
    )
    torch.testing.assert_close(actual.Y, expected.Y)
    torch.testing.assert_close(actual.kwargs["noise"], expected.kwargs["noise"])
    assert actual.kwargs["noise"].shape == (x.shape[-2], 1)


def test_empty_fantasies_skip_sampling_and_preserve_shape() -> None:
    torch = pytest.importorskip("torch")
    x = torch.empty(4, 0, 2, dtype=torch.double)
    sampler = _Sampler((2, 3), 2)
    model = _RecordingModel(2)
    result = model.fantasize(x, sampler, observation_noise=torch.ones(1, 2))
    assert result.Y.shape == (2, 3, 4, 0, 2)
    assert result.kwargs["noise"].shape == (4, 0, 2)
    assert not sampler.called and not model.calls


def test_generic_fantasy_errors_restore_contexts() -> None:
    model = _RecordingModel()
    with pytest.raises(DeprecationError):
        model.fantasize([[0.2]], _Sampler((2,), 1), observation_noise=True)

    class FailingSampler(_Sampler):
        def __call__(self, posterior: Any) -> Any:
            raise RuntimeError("sampling failed")

    with pytest.raises(RuntimeError, match="sampling failed"):
        model.fantasize([[0.2]], FailingSampler((2,), 1), propagate_grads=True)
    assert fantasize.off() and settings.propagate_grads.off()


def test_fantasy_mixin_requires_implementations_and_model_batch_shape_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        FantasizeMixin()  # type: ignore[abstract]
    with pytest.raises(NotImplementedError, match="Model does not define batch_shape"):
        _ = Model().batch_shape
