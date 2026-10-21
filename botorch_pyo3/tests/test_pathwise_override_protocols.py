"""Independent public-oracle checks for configurable pathwise construction."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskGP as ReferenceGP
from botorch.models.transforms.outcome import Standardize as ReferenceStandardize
from botorch.sampling.pathwise import draw_kernel_feature_paths as reference_prior
from botorch.sampling.pathwise import draw_matheron_paths as reference_matheron
from botorch.sampling.pathwise import gaussian_update as reference_update
from botorch.sampling.pathwise.features import KernelFeatureMap as ReferenceFeatureMap
from botorch.sampling.pathwise.features import gen_kernel_features as reference_features

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.kernels import RBFKernel
from botorch_rs.models import SingleTaskGP
from botorch_rs.models._pathwise_context import training_outcome_transform
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.sampling.pathwise import (
    draw_kernel_feature_paths,
    draw_matheron_paths,
    gaussian_update,
)
from botorch_rs.sampling.pathwise.features import KernelFeatureMap, gen_kernel_features
from botorch_rs.utils.context_managers import delattr_ctx

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _models(device="cpu", backend="torch", standardized=False):
    x = torch.linspace(0.1, 0.9, 12, device=device, dtype=torch.double).reshape(6, 2)
    y = (3 + 2 * x.sum(-1).sin()).unsqueeze(-1)
    actual = SingleTaskGP(
        x,
        y,
        covar_module=RBFKernel(ard_num_dims=2),
        outcome_transform=Standardize(1) if standardized else None,
        backend=backend,
    )
    expected = ReferenceGP(
        x,
        y,
        covar_module=gpytorch.kernels.RBFKernel(ard_num_dims=2).to(x),
        outcome_transform=ReferenceStandardize(1) if standardized else None,
    ).to(x)
    for model in (actual, expected):
        model.covar_module.lengthscale = x.new_tensor(0.6)
        model.mean_module.constant = x.new_tensor(0.2)
        model.likelihood.noise = x.new_tensor(0.08)
        model.eval()
    return actual, expected, x


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("provided_root", [False, True])
def test_exact_tensor_update_overrides_match_reference_values_and_full_graphs(
    device, provided_root, monkeypatch
):
    actual, expected, x = _models(device)
    leaves, paths = [], []
    for index, model in enumerate((actual, expected)):
        points = x[[0, 2, 5]].clone().requires_grad_()
        target = x.new_tensor([0.3, -0.2, 0.6], requires_grad=True)
        prior = (
            torch.linspace(-0.4, 0.7, 18, device=device, dtype=x.dtype)
            .reshape(2, 3, 3)
            .requires_grad_()
        )
        noise = torch.linspace(-0.5, 0.6, 18, device=device, dtype=x.dtype).reshape(2, 3, 3)
        covariance = (
            0.08 * torch.eye(3, device=device, dtype=x.dtype) + x.new_full((3, 3), 0.02)
        ).requires_grad_()
        root = (
            torch.linalg.cholesky(model.covar_module(points).to_dense() + covariance)
            .detach()
            .requires_grad_()
        )
        query = x[[1, 4]].clone().requires_grad_()
        options = dict(points=points, target_values=target, noise_covariance=covariance)
        if provided_root:
            options["scale_tril"] = root
        if index:
            with monkeypatch.context() as patch:
                patch.setattr(
                    torch, "randn_like", lambda value, noise=noise: noise.expand_as(value)
                )
                path = reference_update(model, prior, **options)
        else:
            path = gaussian_update(model, prior, noise=noise, **options)
        paths.append(path(query))
        leaves.append(
            (points, target, prior, covariance, query) + ((root,) if provided_root else ())
        )
    torch.testing.assert_close(*paths, atol=3e-11, rtol=3e-10)
    gradients = [
        torch.autograd.grad(value.square().sum(), targets, create_graph=True)
        for value, targets in zip(paths, leaves, strict=True)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-8)
    # The reference kernel's cdist second derivative is unavailable. The new
    # root/covariance/target algebra itself admits an independent Hessian oracle.
    selected = (1, 2, 3) + ((5,) if provided_root else ())
    seconds = [
        torch.autograd.grad(
            sum(gradients[i][j].sum() for j in selected),
            [leaves[i][j] for j in selected],
        )
        for i in (0, 1)
    ]
    for a, b in zip(*seconds, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("provided_root", [False, True])
def test_native_exact_update_overrides_match_independent_reference(
    kind, provided_root, monkeypatch
):
    actual, expected, x = _models(backend="native")
    points = x[[0, 2, 5]]
    target = x.new_tensor([0.3, -0.2, 0.6])
    prior = torch.linspace(-0.4, 0.7, 18, dtype=x.dtype).reshape(2, 3, 3)
    noise = torch.linspace(-0.5, 0.6, 18, dtype=x.dtype).reshape(2, 3, 3)
    covariance = 0.08 * torch.eye(3, dtype=x.dtype) + x.new_full((3, 3), 0.02)
    root = torch.linalg.cholesky(expected.covar_module(points).to_dense() + covariance).detach()

    def convert(value):
        return value.tolist() if kind == "list" else value.numpy() if kind == "numpy" else value

    options = dict(points=points, target_values=target, noise_covariance=covariance)
    if provided_root:
        options["scale_tril"] = root
    with monkeypatch.context() as patch:
        patch.setattr(torch, "randn_like", lambda value: noise.expand_as(value))
        reference = reference_update(expected, prior, **options)(x[[1, 4]])
    path = gaussian_update(
        actual,
        convert(prior),
        noise=convert(noise),
        **{key: convert(value) for key, value in options.items()},
    )
    values, shape = to_flat(path(convert(x[[1, 4]])))
    torch.testing.assert_close(
        x.new_tensor(list(values)).reshape(shape), reference, atol=2e-11, rtol=2e-10
    )


@pytest.mark.parametrize("device", DEVICES)
def test_custom_prior_and_update_callbacks_preserve_transformed_training_law(device):
    actual, expected, x = _models(device, standardized=True)
    results, targets = [], []
    for index, model in enumerate((actual, expected)):
        calls = []
        frequency = x.new_tensor([[0.3, -0.5], [0.2, 0.7]], requires_grad=True)
        weights = x.new_tensor([[0.1, 0.3], [0.2, -0.4]], requires_grad=True)
        query = x[[1, 4]].clone().requires_grad_()
        prior_function = reference_prior if index else draw_kernel_feature_paths
        update_function = reference_update if index else gaussian_update
        constructor = ReferenceFeatureMap if index else KernelFeatureMap
        saved_outcome = model.outcome_transform
        transformed_target = model.train_targets.clone()

        def map_generator(
            kernel,
            num_inputs,
            num_outputs,
            calls=calls,
            constructor=constructor,
            frequency=frequency,
        ):
            calls.append((num_inputs, num_outputs))
            return constructor(
                kernel, frequency, input_transform=lambda value: value / kernel.lengthscale
            )

        def prior_sampler(
            model,
            sample_shape,
            transformed_target=transformed_target,
            prior_function=prior_function,
            map_generator=map_generator,
            weights=weights,
        ):
            assert not hasattr(model, "outcome_transform")
            torch.testing.assert_close(model.train_targets, transformed_target, atol=0, rtol=0)
            return prior_function(
                model,
                sample_shape,
                num_features=8,
                map_generator=map_generator,
                weight_generator=lambda shape: weights.expand(shape),
            )

        def update_strategy(
            model,
            sample_values,
            target_values,
            transformed_target=transformed_target,
            update_function=update_function,
        ):
            assert not hasattr(model, "outcome_transform")
            torch.testing.assert_close(target_values, transformed_target, atol=0, rtol=0)
            # An explicit zero covariance avoids random updates and is supported
            # by the reference exact-GP protocol.
            from linear_operator.operators import ZeroLinearOperator

            return update_function(
                model,
                sample_values,
                target_values=target_values,
                noise_covariance=ZeroLinearOperator(len(x), len(x), dtype=x.dtype, device=x.device),
            )

        function = reference_matheron if index else draw_matheron_paths
        path = function(model, torch.Size([2]), prior_sampler, update_strategy)
        assert calls == [(2, 8)]
        assert model.outcome_transform is saved_outcome
        results.append(path(query))
        targets.append((frequency, weights, query))
    torch.testing.assert_close(*results, atol=3e-9, rtol=3e-8)
    gradients = [
        torch.autograd.grad(value.square().sum(), leaves, create_graph=True)
        for value, leaves in zip(results, targets, strict=True)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-7)
    second = torch.autograd.grad(sum(value.sum() for value in gradients[0]), targets[0])
    assert all(bool(torch.isfinite(value).all()) for value in second)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_callback_exception_restores_outcome_and_scoped_training_metadata(device):
    model, _, _ = _models(device, standardized=True)
    original = model.outcome_transform

    def forbidden(model, sample_shape):
        assert not hasattr(model, "outcome_transform")
        assert training_outcome_transform(model) is original
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        draw_matheron_paths(model, (2,), prior_sampler=forbidden)
    assert model.outcome_transform is original
    with delattr_ctx(model, "outcome_transform"):
        assert training_outcome_transform(model) is None


@pytest.mark.parametrize("bad", ["length", "shape", "nan", "diagonal"])
def test_native_update_solve_rejects_malformed_raw_arrays(bad):
    factor = ([1.0, 0.0, 0.2, 0.8], [2, 2])
    errors = ([0.2, 0.3], [2])
    if bad == "length":
        factor = ([1.0], [2, 2])
    elif bad == "shape":
        errors = ([0.2, 0.3], [1, 2])
    elif bad == "nan":
        errors = ([float("nan"), 0.3], [2])
    else:
        factor = ([0.0, 0.0, 0.2, 0.8], [2, 2])
    with pytest.raises(ValueError):
        _rs.pathwise_cholesky_solve(factor, errors)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["rbf", "matern", "scaled"])
def test_actual_gpytorch_kernel_feature_modules_preserve_first_and_second_graphs(device, family):
    """The feature formula, including active dimensions, has its own oracle."""
    maps, inputs = [], []
    like = torch.empty((), device=device, dtype=torch.double)
    for reference in (False, True):
        base = (
            gpytorch.kernels.RBFKernel(
                ard_num_dims=2, active_dims=[2, 0], batch_shape=torch.Size([3])
            )
            if family == "rbf"
            else gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=2, active_dims=[2, 0], batch_shape=torch.Size([3])
            )
        ).to(like)
        base.lengthscale = like.new_tensor([[[0.6, 0.9]], [[0.8, 1.1]], [[0.3, 1.4]]])
        kernel = gpytorch.kernels.ScaleKernel(base).to(like) if family == "scaled" else base
        if family == "scaled":
            kernel.outputscale = like.new_tensor([0.7, 1.2, 1.6])
        function = reference_features if reference else gen_kernel_features
        feature_map = function(kernel, num_inputs=3, num_outputs=16)
        # Independent feature constructions use identical explicit frequencies.
        frequency = like.new_tensor(range(48)).reshape(3, 8, 2).div(29).sub(0.8)
        feature_map.weight = frequency.requires_grad_()
        x = like.new_tensor([[0.12, 0.25, 0.43], [0.57, 0.73, 0.91]]).requires_grad_()
        maps.append(feature_map(x.unsqueeze(-3)))
        inputs.append((x, feature_map.weight, *tuple(kernel.parameters())))
    torch.testing.assert_close(*maps, atol=2e-14, rtol=2e-14)
    gradients = [
        torch.autograd.grad(value.square().sum(), leaves, create_graph=True)
        for value, leaves in zip(maps, inputs, strict=True)
    ]
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-13, rtol=5e-13)
    second = [
        torch.autograd.grad(sum(g.square().sum() for g in grad), leaves)
        for grad, leaves in zip(gradients, inputs, strict=True)
    ]
    for actual, expected in zip(*second, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize(
    "backend,kind,device",
    [("native", kind, "cpu") for kind in ("list", "numpy", "torch")]
    + [
        ("torch", "torch", "cpu"),
        pytest.param(
            "torch",
            "torch",
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("family", ["saas", "matern"])
def test_loaded_fully_bayesian_prior_and_matheron_batches(
    backend, kind, device, family, monkeypatch
):
    from botorch.models import fully_bayesian as source

    from botorch_rs.models import fully_bayesian as port

    cls = "SaasFullyBayesianSingleTaskGP" if family == "saas" else "FullyBayesianSingleTaskGP"
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.double, device=device).reshape(6, 2)
    y = (x.sum(-1).sin() + 1.7).unsqueeze(-1)

    def convert(value):
        return value.tolist() if kind == "list" else value.numpy() if kind == "numpy" else value

    actual = getattr(port, cls)(convert(x), convert(y), outcome_transform=None, backend=backend)
    expected = getattr(source, cls)(x, y, outcome_transform=None).to(x)
    draws = {
        "mean": x.new_tensor([0.1, -0.2, 0.3]),
        "noise": x.new_tensor([0.02, 0.04, 0.06]),
        "lengthscale": x.new_tensor([[0.3, 0.6], [0.7, 0.5], [1.2, 0.9]]),
    }
    if family == "saas":
        draws["outputscale"] = x.new_tensor([0.7, 1.1, 1.4])
    actual.load_mcmc_samples({key: convert(value) for key, value in draws.items()})
    expected.load_mcmc_samples(draws)
    expected.eval()
    reference = reference_prior(expected, torch.Size([2]), num_features=16)
    # Force the public generic feature protocol for this comparison; the
    # default Matheron factory below separately exercises its optimized route.
    prior = draw_kernel_feature_paths(
        actual, (2,), num_features=16, seed=17, map_generator=gen_kernel_features
    )
    prior.feature_map.weight = convert(reference.feature_map.weight.detach())
    prior.weight = convert(reference.weight.detach())
    query = x.new_tensor([[0.24, 0.31], [0.64, 0.78]]).expand(4, 1, 2, 2).clone() + x.new_tensor(
        [0.0, 0.01, -0.01, 0.02]
    ).reshape(4, 1, 1, 1)
    result, shape = to_flat(prior(convert(query)))
    torch.testing.assert_close(
        x.new_tensor(list(result)).reshape(shape), reference(query), atol=3e-13, rtol=3e-13
    )
    assert tuple(shape) == (4, 2, 3, 2)
    noise = torch.linspace(-0.6, 0.7, 36, dtype=x.dtype, device=x.device).reshape(2, 3, 6)

    def prior_sampler(model, sample_shape):
        assert not hasattr(model, "outcome_transform")
        return prior

    def update(model, sample_values, **options):
        return gaussian_update(model, sample_values, noise=convert(noise), **options)

    path = draw_matheron_paths(actual, (2,), prior_sampler, update)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "randn_like", lambda value: noise.expand_as(value))
        expected_update = reference_update(
            expected, reference.forward(x), target_values=expected.train_targets
        )
    result, shape = to_flat(path(convert(query)))
    oracle = reference(query) + expected_update(query)
    torch.testing.assert_close(
        x.new_tensor(list(result)).reshape(shape), oracle, atol=2e-11, rtol=2e-11
    )
    # Default composition must also work without injecting callbacks or draws.
    default = draw_matheron_paths(actual, (2,), seed=29)
    values, shape = to_flat(default(convert(query)))
    assert tuple(shape) == (4, 2, 3, 2)
    assert bool(torch.isfinite(x.new_tensor(list(values))).all())


@pytest.mark.parametrize("backend", ["native", "torch"])
def test_fixed_noise_update_at_different_points_uses_reference_zero_noise_protocol(backend):
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.double).reshape(6, 2)
    y = x[:, :1].sin()
    actual = SingleTaskGP(
        x,
        y,
        train_Yvar=torch.full_like(y, 0.05),
        covar_module=RBFKernel(ard_num_dims=2),
        outcome_transform=None,
        backend=backend,
    )
    expected = ReferenceGP(
        x,
        y,
        train_Yvar=torch.full_like(y, 0.05),
        covar_module=gpytorch.kernels.RBFKernel(ard_num_dims=2).to(x),
        outcome_transform=None,
    ).to(x)
    actual.covar_module.lengthscale = expected.covar_module.lengthscale = x.new_tensor(0.7)
    points, targets, samples = x[[0, 2, 5]], y[[0, 2, 5], 0], x.new_zeros(2, 3)
    actual.eval()
    expected.eval()
    result = gaussian_update(actual, samples, points=points, target_values=targets)(x[[1, 4]])
    oracle = reference_update(expected, samples, points=points, target_values=targets)(x[[1, 4]])
    torch.testing.assert_close(result, oracle, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_gaussian_likelihood_noise_parameter_graph_matches_reference(device, monkeypatch):
    actual, expected, x = _models(device)
    results, parameters = [], []
    normals = x.new_tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]])
    for reference, model in enumerate((actual, expected)):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(x)
        likelihood.noise = x.new_tensor(0.07)
        targets = x.new_tensor([0.3, 0.4, -0.1], requires_grad=True)
        options = dict(points=x[[0, 2, 5]], target_values=targets)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "randn_like", lambda value: normals.expand_as(value))
            path = (
                reference_update(model, x.new_zeros(2, 3), likelihood, **options)
                if reference
                else gaussian_update(model, x.new_zeros(2, 3), likelihood, noise=normals, **options)
            )
        results.append(path(x[[1, 4]]))
        parameters.append((likelihood.noise_covar.raw_noise, targets))
    torch.testing.assert_close(*results, atol=2e-12, rtol=2e-12)
    gradients = [
        torch.autograd.grad(value.square().sum(), leaves, create_graph=True)
        for value, leaves in zip(results, parameters, strict=True)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    hessians = [
        torch.autograd.grad(sum(g.sum() for g in gradient), leaves)
        for gradient, leaves in zip(gradients, parameters, strict=True)
    ]
    for a, b in zip(*hessians, strict=True):
        torch.testing.assert_close(a, b, atol=5e-12, rtol=5e-12)
