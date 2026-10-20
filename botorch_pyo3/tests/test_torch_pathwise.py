"""GPU RFF/Matheron paths compared with explicit reference draws and full graphs."""

import functools
from unittest.mock import patch

import pytest
import torch

from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.sampling.pathwise import (
    MatheronPath,
    draw_kernel_feature_paths,
    draw_matheron_paths,
    gaussian_update,
)
from botorch_rs.sampling.pathwise.features import gen_kernel_features

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FEATURES = 16


def data(device):
    X = torch.tensor(
        [[0.04, 0.12], [0.27, 0.35], [0.56, 0.62], [0.81, 0.89]], dtype=torch.double, device=device
    )
    return X, (X[:, :1] * 5.3).sin()


def model(X, Y, kernel="rbf", transformed=False, fixed=False, reference=False):
    if reference:
        from botorch.models import SingleTaskGP as GP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.kernels import MaternKernel as Matern
        from gpytorch.kernels import RBFKernel as RBF
        from gpytorch.kernels import ScaleKernel as Scale
    else:
        GP = SingleTaskGP
        from botorch_rs.models.transforms.input import Normalize
        from botorch_rs.models.transforms.outcome import Standardize

        Matern, RBF, Scale = MaternKernel, RBFKernel, ScaleKernel
    batch = Y.shape[:-2] + ((Y.shape[-1],) if Y.shape[-1] > 1 else ())
    base = (
        RBF(ard_num_dims=2, batch_shape=batch)
        if kernel == "rbf"
        else Matern(ard_num_dims=2, batch_shape=batch, nu=1.5)
    )
    covar = Scale(base, batch_shape=batch) if kernel == "scale" else base
    return (
        GP(
            X,
            Y,
            train_Yvar=torch.full_like(Y, 0.01) if fixed else None,
            covar_module=covar,
            input_transform=Normalize(2) if transformed else None,
            outcome_transform=Standardize(Y.shape[-1]) if transformed else None,
            **({} if reference else {"backend": "torch"}),
        )
        .to(X)
        .eval()
    )


def paths(gp, shape=(2,), draws=None, seed=23):
    return draw_matheron_paths(
        gp,
        shape,
        functools.partial(draw_kernel_feature_paths, num_features=FEATURES),
        draws=draws,
        seed=seed,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel", ["rbf", "matern", "scale"])
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("outputs", [1, 2])
def test_device_matheron_paths_match_reference_draws(device, kernel, transformed, outputs):
    from botorch.sampling.pathwise import draw_kernel_feature_paths as ReferencePrior
    from botorch.sampling.pathwise import draw_matheron_paths as ReferencePaths
    from mo_entropy_helpers import bt_path_draws, recorded_randn

    X, Y = data(device)
    if outputs == 2:
        Y = torch.cat((Y, (X[:, :1] * 3.1).cos()), -1)
    actual_gp, reference_gp = (
        model(X, Y, kernel, transformed),
        model(X, Y, kernel, transformed, reference=True),
    )
    noises = []
    torch.manual_seed(31)
    with recorded_randn(noises):
        expected = ReferencePaths(
            reference_gp,
            torch.Size([2]),
            prior_sampler=functools.partial(ReferencePrior, num_features=FEATURES),
        )
    actual = paths(actual_gp, draws=bt_path_draws(expected, noises))
    torch.testing.assert_close(
        actual["update_paths"].weight, expected["update_paths"].weight, atol=3e-8, rtol=3e-8
    )
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]], requires_grad=True)
    av, bv = actual(query), expected(query)
    for component in ("prior_paths", "update_paths"):
        torch.testing.assert_close(
            actual[component](query),
            expected[component](query),
            atol=3e-8,
            rtol=3e-8,
            msg=component,
        )
    torch.testing.assert_close(av, bv, atol=3e-8, rtol=3e-8)
    ag = torch.autograd.grad(av.sum(), query, create_graph=True)[0]
    bg = torch.autograd.grad(bv.sum(), query)[0]
    torch.testing.assert_close(ag, bg, atol=3e-7, rtol=3e-7)
    assert torch.isfinite(torch.autograd.grad(ag.square().sum(), query)[0]).all()
    assert isinstance(actual, MatheronPath) and av.device == query.device
    torch.testing.assert_close(paths(actual_gp, draws=actual.draws)(query), av)
    torch.testing.assert_close(
        actual["update_paths"].weight, expected["update_paths"].weight, atol=3e-8, rtol=3e-8
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_pathwise_all_data_parameter_draw_graphs_and_no_host(device, fixed):
    X, Y = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]], requires_grad=True)
    frequency = (
        torch.linspace(-0.8, 1.0, 16, dtype=X.dtype, device=X.device).reshape(8, 2).requires_grad_()
    )
    weight = (
        torch.linspace(-0.7, 0.6, 32, dtype=X.dtype, device=X.device)
        .reshape(2, 16)
        .requires_grad_()
    )
    noise = (
        torch.linspace(-0.6, 0.7, 8, dtype=X.dtype, device=X.device).reshape(2, 4).requires_grad_()
    )

    def make(a, b, c, d, e):
        return paths(model(a, b, "scale", True, fixed), draws=[(c, d, e)])

    gp = model(X, Y, "scale", True, fixed)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        p = paths(gp, draws=[(frequency, weight, noise)])
        leaves = (X, Y, query, frequency, weight, noise, *gp.parameters())
        gradients = torch.autograd.grad(p(query).sum(), leaves, create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in gradients), leaves, allow_unused=True
        )
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in second)

    def function(a, b, c, d, e, f):
        return make(a, b, d, e, f)(c)

    inputs = (X, Y, query, frequency, weight, noise)
    assert torch.autograd.gradcheck(function, inputs, atol=3e-5, rtol=3e-4)
    assert torch.autograd.gradgradcheck(function, inputs, atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_prior_features_update_list_and_model_batch(device):
    X, Y = data(device)
    X = X.expand(2, 4, 2).clone()
    Y = Y.expand(2, 4, 1).clone()
    gp = model(X, Y, "matern")
    p = paths(gp, shape=(2, 3))
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]])
    assert p(query).shape == (2, 3, 2, 2)
    prior = draw_kernel_feature_paths(gp, (2, 3), num_features=FEATURES, seed=23)
    assert prior(query).shape == (2, 3, 2, 2)
    feature = gen_kernel_features(gp.covar_module, num_inputs=2, num_outputs=16, seed=31)
    torch.testing.assert_close(
        feature(query),
        gen_kernel_features(gp.covar_module, num_inputs=2, num_outputs=16, seed=31)(query),
    )
    update = gaussian_update(gp, prior.forward(gp._training_values()[0]), seed=23)
    assert update(query).shape == (2, 3, 2, 2)
    collection = ModelListGP(gp, model(X, Y, "rbf"))
    ps = paths(collection, shape=(2, 3))
    assert len(ps) == 2
    for i, out in enumerate(ps(query)):
        torch.testing.assert_close(
            out, paths(collection.models[i], shape=(2, 3), seed=23 + i)(query)
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_active_dimensions_and_public_path_composition(device, dtype):
    from botorch_rs.sampling.pathwise import GeneralizedLinearPath
    from botorch_rs.sampling.pathwise.features import KernelEvaluationMap, KernelFeatureMap
    from botorch_rs.sampling.pathwise.utils import InverseLengthscaleTransform, SineCosineTransform

    X, Y = data(device)
    X, Y = X.to(dtype), Y.to(dtype)
    gp = SingleTaskGP(
        X,
        Y,
        backend="torch",
        outcome_transform=None,
        covar_module=MaternKernel(ard_num_dims=1, active_dims=[1], nu=2.5),
    )
    feature = gen_kernel_features(gp.covar_module, num_inputs=2, num_outputs=16, seed=13)
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]], requires_grad=True)
    phase = query[:, 1:2] / gp.covar_module.lengthscale @ feature.weight.transpose(-1, -2)
    torch.testing.assert_close(
        feature(query), torch.cat((phase.sin(), phase.cos()), -1) * (2 / 16) ** 0.5
    )
    p = paths(gp, shape=())
    assert p(query).shape == (2,)
    assert torch.isfinite(torch.autograd.grad(p(query).sum(), query)[0]).all()

    # The public constructors also preserve graphs for caller-supplied maps.
    gp = model(X, Y)
    frequencies = X.new_tensor([[0.2, 0.4], [-0.3, 0.7]]).requires_grad_()
    fmap = KernelFeatureMap(
        gp.covar_module,
        frequencies,
        input_transform=InverseLengthscaleTransform(gp.covar_module),
        output_transform=SineCosineTransform(0.5),
    )
    weights = X.new_tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).requires_grad_()
    prior = GeneralizedLinearPath(fmap, weights)
    points = X.detach().clone().requires_grad_()
    update = GeneralizedLinearPath(KernelEvaluationMap(gp.covar_module, points), weights)
    composed = MatheronPath(prior, update)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        leaves = (query, frequencies, weights, points, *gp.covar_module.parameters())
        first = torch.autograd.grad(composed(query).sum(), leaves, create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), leaves, allow_unused=True
        )
    assert all(torch.isfinite(g).all() for g in first)
    assert all(g is None or torch.isfinite(g).all() for g in second)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["scalar", "multioutput", "list"])
@pytest.mark.parametrize("shape", [(), (2,)])
def test_get_matheron_path_model_public_shapes_posterior_and_conversion(device, kind, shape):
    from botorch_rs.models.deterministic import MatheronPathModel
    from botorch_rs.sampling.pathwise import get_matheron_path_model

    X, Y = data(device)
    gp = (
        model(X, Y, transformed=True)
        if kind == "scalar"
        else (
            model(X, torch.cat((Y, 0.5 + Y.square()), -1), transformed=True)
            if kind == "multioutput"
            else ModelListGP(
                model(X, Y, transformed=True), model(X, 0.5 + Y.square(), transformed=True)
            )
        )
    )
    path_model = get_matheron_path_model(gp, sample_shape=shape, num_features=FEATURES, seed=13)
    assert isinstance(path_model, MatheronPathModel) and isinstance(path_model, torch.nn.Module)
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]], requires_grad=True)
    result = path_model(query)
    assert result.shape == (*shape, 2, gp.num_outputs)
    torch.testing.assert_close(path_model.posterior(query).mean, result)
    assert path_model.batch_shape == shape
    first = torch.autograd.grad(result.sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all()
    # Explicit .to rebuilds the exact update for the same random path.
    before = result.detach()
    path_model = path_model.float()
    after = path_model(query.detach().float())
    assert after.dtype == torch.float32 and after.device == query.device
    torch.testing.assert_close(after, before.float(), atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_path_model_scalar_mc_acquisition_full_graph(device):
    from botorch_rs.acquisition.monte_carlo import qSimpleRegret
    from botorch_rs.sampling.pathwise import get_matheron_path_model

    X, Y = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    gp = model(X, Y, "scale", True)
    path_model = get_matheron_path_model(gp, num_features=FEATURES, seed=13)
    acq = qSimpleRegret(path_model)
    query = X.new_tensor([[[0.19, 0.23], [0.68, 0.74]]], requires_grad=True)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        value = acq(query)
        torch.testing.assert_close(value, path_model(query).squeeze(-1).amax(-1))
        leaves = (X, Y, query, *gp.parameters())
        first = torch.autograd.grad(value.sum(), leaves, create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), leaves, allow_unused=True
        )
    assert all(torch.isfinite(g).all() for g in first)
    assert all(g is None or torch.isfinite(g).all() for g in second)
