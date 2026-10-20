"""Public generalized-linear and Matheron path constructor parity."""

from __future__ import annotations

import copy
import functools
import pickle

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch, shape

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.sampling.pathwise.features import KernelFeatureMap as BtMap  # noqa:E402
from botorch.sampling.pathwise.paths import GeneralizedLinearPath as BtLinear  # noqa:E402
from botorch.sampling.pathwise.utils import InverseLengthscaleTransform as BtInverse  # noqa:E402
from botorch.sampling.pathwise.utils import SineCosineTransform as BtSineCosine  # noqa:E402
from gpytorch.kernels import RBFKernel as BtRBF  # noqa:E402

from botorch_rs.kernels import RBFKernel  # noqa:E402
from botorch_rs.models import SingleTaskGP  # noqa:E402
from botorch_rs.sampling.pathwise import (  # noqa:E402
    GeneralizedLinearPath,
    MatheronPath,
    draw_kernel_feature_paths,
    draw_matheron_paths,
)
from botorch_rs.sampling.pathwise.features import KernelFeatureMap  # noqa:E402
from botorch_rs.sampling.pathwise.utils import (  # noqa:E402
    InverseLengthscaleTransform,
    SineCosineTransform,
)


def features(batch=(), kind="torch"):
    kernel = RBFKernel(ard_num_dims=2, batch_shape=batch)
    reference = BtRBF(ard_num_dims=2, batch_shape=torch.Size(batch)).double()
    ls = (
        torch.tensor([0.6, 0.9], dtype=torch.float64)
        .reshape(*(1 for _ in batch), 1, 2)
        .expand(*batch, 1, 2)
    )
    kernel.lengthscale = as_kind(ls, kind)
    reference.lengthscale = ls
    frequency = (
        torch.arange(max(1, torch.Size(batch).numel()) * 6, dtype=torch.float64).reshape(
            *batch, 3, 2
        )
        / 7
        - 0.5
    )
    actual = KernelFeatureMap(
        kernel,
        as_kind(frequency, kind),
        input_transform=InverseLengthscaleTransform(kernel),
        output_transform=SineCosineTransform(0.4),
    )
    expected = BtMap(
        reference,
        frequency,
        input_transform=BtInverse(reference),
        output_transform=BtSineCosine(0.4),
    )
    return actual, expected


@pytest.mark.parametrize("kind", KINDS)
def test_native_feature_map_and_weight_constructor_arrays(kind):
    actual, reference = features(kind=kind)
    weight = torch.arange(18, dtype=torch.float64).reshape(3, 6) / 13 - 0.2
    x = torch.tensor([[0.2, 0.7], [0.8, 0.1]], dtype=torch.float64)
    a = GeneralizedLinearPath(feature_map=actual, weight=as_kind(weight, kind))
    b = BtLinear(reference, weight)
    assert_close(a(as_kind(x, kind)), b(x), rtol=2e-8, atol=2e-9)
    assert shape(a.weight) == (3, 6) and a.feature_map is actual


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("ensemble_as_batch", [False, True])
def test_batched_ensemble_layout_and_native_query_weight_feature_gradients(
    ensemble, ensemble_as_batch
):
    actual, reference = features(batch=(3,))
    weight = torch.nn.Parameter(torch.arange(36, dtype=torch.float64).reshape(2, 3, 6) / 31 - 0.2)
    other = weight.detach().clone().requires_grad_()
    actual.weight = actual.weight.detach().clone().requires_grad_()
    reference.weight = actual.weight.detach().clone().requires_grad_()
    query = torch.linspace(0.1, 0.9, 5 * 1 * 3 * 2 * 2, dtype=torch.float64).reshape(5, 1, 3, 2, 2)
    if ensemble and not ensemble_as_batch:
        query = query[:, :, 0]
    query.requires_grad_()
    other_query = query.detach().clone().requires_grad_()
    a = GeneralizedLinearPath(
        actual, weight, is_ensemble=ensemble, ensemble_as_batch=ensemble_as_batch
    )
    b = BtLinear(reference, other, is_ensemble=ensemble, ensemble_as_batch=ensemble_as_batch)
    av, bv = a(query), b(other_query)
    torch.testing.assert_close(av, bv, rtol=2e-8, atol=2e-9)
    ag = torch.autograd.grad(av.square().sum(), (query, weight, actual.weight))
    bg = torch.autograd.grad(bv.square().sum(), (other_query, other, reference.weight))
    for x, y in zip(ag, bg, strict=True):
        torch.testing.assert_close(x, y, rtol=3e-7, atol=3e-8)
    a.set_ensemble_as_batch(not ensemble_as_batch)
    assert a.ensemble_as_batch is not ensemble_as_batch


def test_explicit_callable_bias_transforms_kwargs_and_parameter_graphs():
    phase = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[0.2, 0.7], [0.8, 0.1]], dtype=torch.float64, requires_grad=True)

    class Features:
        def __call__(self, values, *, gain=1):
            return gain * values.sin()

    a = GeneralizedLinearPath(
        Features(),
        weight,
        bias_module=lambda values: phase * values.sum(-1),
        input_transform=lambda values: values * 2,
        output_transform=lambda values: values.exp(),
    )
    expected = ((2 * x).sin().matmul(weight) * 1.7 + phase * (2 * x).sum(-1)).exp()
    torch.testing.assert_close(a(x, gain=1.7), expected)
    assert all(
        torch.isfinite(v).all()
        for v in torch.autograd.grad(a(x, gain=1.7).sum(), (x, weight, phase))
    )
    a.weight = weight * 2
    assert a.weight.requires_grad


@pytest.mark.parametrize("kind", KINDS)
def test_matheron_public_constructor_composes_native_paths_and_dictionary_behavior(kind):
    from botorch_rs.models.transforms.input import Normalize

    x = as_kind([[0.1, 0.2], [0.3, 0.8], [0.8, 0.4]], kind)
    y = as_kind([[0.2], [0.7], [0.4]], kind)
    model = SingleTaskGP(x, y, outcome_transform=None, input_transform=Normalize(2))
    native = draw_matheron_paths(
        model,
        (3,),
        prior_sampler=functools.partial(draw_kernel_feature_paths, num_features=8),
        seed=17,
    )
    p = MatheronPath(prior_paths=native["prior_paths"], update_paths=native["update_paths"])
    query = as_kind([[0.25, 0.7], [0.6, 0.3]], kind)
    assert list(p.keys()) == ["prior_paths", "update_paths"]
    assert_close(p(query), native(query), rtol=2e-8, atol=2e-9)
    p["update_paths"] = native["prior_paths"]
    from botorch_rs import _ops

    assert_close(p(query), _ops.apply("mul_raw", native["prior_paths"](query), 2))
    del p["update_paths"]
    assert len(p) == 1
    assert_close(p(query), native["prior_paths"](query))
    restored = copy.deepcopy(p)
    assert restored is not p and restored.paths is not p.paths
    assert_close(restored(query), p(query))


def test_matheron_native_path_composition_query_graph_and_transforms():
    x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    model = SingleTaskGP(x, x.sin(), outcome_transform=None)
    native = draw_matheron_paths(
        model,
        (3,),
        prior_sampler=functools.partial(draw_kernel_feature_paths, num_features=8),
        seed=21,
    )
    p = MatheronPath(
        native["prior_paths"],
        native["update_paths"],
        input_transform=lambda x: x * 1.1,
        output_transform=lambda values: values.square(),
    )
    query = torch.tensor([[0.2], [0.6]], dtype=torch.float64, requires_grad=True)
    av = p(query)
    expected = native(query * 1.1).square()
    torch.testing.assert_close(av, expected)
    ag = torch.autograd.grad(av.sum(), query, retain_graph=True)[0]
    eg = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(ag, eg)


def test_public_matheron_constructor_matches_official_composition_and_graphs():
    from botorch.sampling.pathwise.posterior_samplers import MatheronPath as BtMatheron

    af, bf = features()
    w1 = torch.tensor([[0.1, 0.3, -0.2, 0.5, 0.8, 0.4]], dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor([[0.3, -0.1, 0.5, 0.2, 0.7, 0.6]], dtype=torch.float64, requires_grad=True)
    bw1, bw2 = (w.detach().clone().requires_grad_() for w in (w1, w2))
    a = MatheronPath(
        GeneralizedLinearPath(af, w1),
        GeneralizedLinearPath(af, w2),
        input_transform=lambda x: x * 1.2,
        output_transform=lambda x: x.exp(),
    )
    b = BtMatheron(
        BtLinear(bf, bw1),
        BtLinear(bf, bw2),
        input_transform=lambda x: x * 1.2,
        output_transform=lambda x: x.exp(),
    )
    x = torch.tensor([[0.2, 0.7], [0.8, 0.1]], dtype=torch.float64, requires_grad=True)
    bx = x.detach().clone().requires_grad_()
    av, bv = a(x), b(bx)
    torch.testing.assert_close(av, bv, rtol=2e-8, atol=2e-9)
    for ag, bg in zip(
        torch.autograd.grad(av.sum(), (x, w1, w2)),
        torch.autograd.grad(bv.sum(), (bx, bw1, bw2)),
        strict=True,
    ):
        torch.testing.assert_close(ag, bg, rtol=3e-7, atol=3e-8)


def test_explicit_torch_path_serialization():
    p = GeneralizedLinearPath(torch.nn.Identity(), torch.tensor([0.2, 0.7], dtype=torch.float64))
    x = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
    for restored in (copy.deepcopy(p), pickle.loads(pickle.dumps(p))):
        torch.testing.assert_close(restored(x), p(x))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("outputs", [1, 2])
def test_factory_dictionary_edits_and_standardize_snapshot_preserve_query_graph(kind, outputs):
    from botorch_rs import _ops

    x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    y = torch.cat([1 + 3 * x, 4 - x], dim=-1)[..., :outputs]
    model = SingleTaskGP(as_kind(x, kind), as_kind(y, kind))
    native = draw_matheron_paths(
        model,
        (3,),
        prior_sampler=functools.partial(draw_kernel_feature_paths, num_features=8),
        seed=43,
    )
    query = torch.tensor([[0.2], [0.6]], dtype=torch.float64, requires_grad=kind == "torch")
    if outputs == 2:
        query = query.unsqueeze(-3)
    query = as_kind(query, kind) if kind != "torch" else query
    expected = native(query)
    composed = MatheronPath(
        native["prior_paths"], native["update_paths"], output_transform=native.output_transform
    )
    assert_close(composed(query), expected, rtol=2e-8, atol=2e-9)
    assert_close(native.forward(query), composed.forward(query), rtol=2e-8, atol=2e-9)
    native["update_paths"] = native["prior_paths"]
    replacement = native.output_transform(_ops.apply("mul_raw", native["prior_paths"](query), 2))
    edited = native(query)
    assert_close(edited, replacement, rtol=2e-8, atol=2e-9)
    # Updating the original model's transform cannot mutate the path snapshot.
    model.outcome_transform.train()(as_kind(y * 10, kind))
    assert_close(native(query), replacement, rtol=2e-8, atol=2e-9)
    assert_close(copy.deepcopy(native)(query), replacement, rtol=2e-8, atol=2e-9)
    if kind == "torch":
        ag = torch.autograd.grad(edited.sum(), query, retain_graph=True)[0]
        bg = torch.autograd.grad(replacement.sum(), query)[0]
        torch.testing.assert_close(ag, bg, rtol=3e-7, atol=3e-8)
