"""Whitened variational Matheron paths preserve device values and complete graphs."""

# ruff: noqa: E402
import functools
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.sampling.pathwise import (
    draw_kernel_feature_paths,
    draw_matheron_paths,
    gaussian_update,
)
from botorch_rs.sampling.pathwise.posterior_samplers import get_matheron_path_model

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FEATURES = 16


def data(device, batch=False):
    X = torch.tensor(
        [[0.04, 0.12], [0.27, 0.35], [0.56, 0.62], [0.81, 0.89]], dtype=torch.double, device=device
    )
    Y = (X[:, :1] * 5.3).sin()
    if batch:
        X = X.expand(2, *X.shape).clone()
        Y = torch.stack((Y, Y * 1.2 + 0.3))
    return X, Y


def make_model(X, Y, transformed=False, reference=False, kernel="rbf", learn=True):
    if reference:
        from botorch.models import SingleTaskVariationalGP as GP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
    else:
        GP = SingleTaskVariationalGP
        from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel
        from botorch_rs.models.transforms.input import Normalize
        from botorch_rs.models.transforms.outcome import Standardize
    batch = X.shape[:-2]
    base = (
        RBFKernel(ard_num_dims=2, batch_shape=batch)
        if kernel == "rbf"
        else MaternKernel(ard_num_dims=2, batch_shape=batch, nu=1.5)
    )
    Z = X.new_tensor([[0.08, 0.16], [0.45, 0.55], [0.85, 0.92]])
    gp = (
        GP(
            X,
            Y,
            covar_module=ScaleKernel(base, batch_shape=batch),
            inducing_points=Z,
            learn_inducing_points=learn,
            input_transform=Normalize(2) if transformed else None,
            outcome_transform=Standardize(1, batch_shape=batch) if transformed else None,
            **({} if reference else {"backend": "torch"}),
        )
        .to(X)
        .eval()
    )
    gp.model.covar_module.base_kernel.lengthscale = X.new_tensor([0.3, 0.4])
    gp.model.covar_module.outputscale = X.new_tensor(1.2)
    gp.model.mean_module.constant = X.new_tensor(0.2)
    strategy = gp.model.variational_strategy
    distribution = strategy._variational_distribution
    with torch.no_grad():
        distribution.variational_mean.copy_(X.new_tensor([-0.1, 0.15, 0.2]))
        distribution.chol_variational_covar.copy_(
            torch.eye(3, dtype=X.dtype, device=X.device) * 0.8
            + torch.tril(torch.ones(3, 3, dtype=X.dtype, device=X.device), -1) * 0.1
        )
        strategy.variational_params_initialized.fill_(1)
    return gp


def paths(model, shape=(2,), draws=None, seed=23):
    return draw_matheron_paths(
        model,
        shape,
        functools.partial(draw_kernel_feature_paths, num_features=FEATURES),
        draws=draws,
        seed=seed,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("kernel", ["rbf", "matern"])
def test_variational_paths_match_reference_current_q_draws_and_query_gradients(
    device, transformed, batch, kernel
):
    from botorch.sampling.pathwise import draw_kernel_feature_paths as ReferencePrior
    from botorch.sampling.pathwise import draw_matheron_paths as ReferencePaths

    X, Y = data(device, batch)
    actual_gp, ref_gp = (
        make_model(X, Y, transformed, kernel=kernel),
        make_model(X, Y, transformed, True, kernel),
    )
    normal = torch.linspace(
        -0.8, 0.7, 2 * (2 if batch else 1) * 3, dtype=X.dtype, device=X.device
    ).reshape(2, *((2,) if batch else ()), 3)
    distribution = ref_gp.model.variational_strategy.variational_distribution
    rsample = distribution.rsample
    with patch.object(
        distribution, "rsample", side_effect=lambda shape: rsample(shape, base_samples=normal)
    ) as synchronized:
        expected = ReferencePaths(
            ref_gp,
            torch.Size([2]),
            prior_sampler=functools.partial(ReferencePrior, num_features=FEATURES),
        )
    assert synchronized.call_count == 1
    prior = expected["prior_paths"]
    frequency, weight = prior.feature_map.weight.detach(), prior.weight.detach()
    actual = paths(actual_gp, draws=[(frequency, weight, normal)])
    torch.testing.assert_close(
        actual["update_paths"].weight, expected["update_paths"].weight, atol=3e-8, rtol=3e-8
    )
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]], requires_grad=True)
    value, ref_value = actual(query), expected(query)
    torch.testing.assert_close(value, ref_value, atol=3e-8, rtol=3e-8)
    first = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_first = torch.autograd.grad(ref_value.sum(), query)[0]
    torch.testing.assert_close(first, expected_first, atol=3e-7, rtol=3e-7)
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all()
    torch.testing.assert_close(paths(actual_gp, draws=actual.draws)(query), value)


@pytest.mark.parametrize("device", DEVICES)
def test_source_multioutput_prior_supported_but_matheron_update_is_not_registered(device):
    from botorch.models import SingleTaskVariationalGP as ReferenceGP
    from botorch.sampling.pathwise import draw_kernel_feature_paths as ReferencePrior
    from botorch.sampling.pathwise import draw_matheron_paths as ReferencePaths
    from mo_entropy_helpers import bt_path_draws

    X, Y = data(device)
    Y = torch.cat((Y, Y.cos()), -1)
    actual_gp = SingleTaskVariationalGP(X, Y, inducing_points=X[:3], backend="torch").eval()
    ref_gp = ReferenceGP(X, Y, inducing_points=X[:3]).to(X).eval()
    prior = ReferencePrior(ref_gp, torch.Size([2]), num_features=FEATURES)
    actual = draw_kernel_feature_paths(
        actual_gp, (2,), num_features=FEATURES, draws=bt_path_draws(prior, None)
    )
    torch.testing.assert_close(actual(X), prior(X), atol=3e-8, rtol=3e-8)
    with pytest.raises(
        (AttributeError, NotImplementedError), match="inducing_points|signature|not.*found"
    ):
        ReferencePaths(
            ref_gp,
            torch.Size([2]),
            prior_sampler=functools.partial(ReferencePrior, num_features=FEATURES),
        )
    with pytest.raises(NotImplementedError, match="IndependentMultitask"):
        paths(actual_gp)


def independent_rbf_matheron(gp, train_X, train_Y, query, frequency, weight, normal):
    """Canonical one-output law; no port kernel, path or solve helper calls."""
    lower, upper = train_X.amin(-2, keepdim=True), train_X.amax(-2, keepdim=True)
    query = (query - lower) / (upper - lower)
    kernel = gp.model.covar_module
    lengthscale, scale = kernel.base_kernel.lengthscale, kernel.outputscale
    points = gp.model.variational_strategy.inducing_points
    constant = gp.model.mean_module.constant

    def features(X):
        phase = (X / lengthscale) @ frequency.mT
        return torch.cat((phase.sin(), phase.cos()), -1) * (2 / FEATURES) ** 0.5 * scale.sqrt()

    def covariance(left, right):
        difference = left.unsqueeze(-2) / lengthscale - right.unsqueeze(-3) / lengthscale
        return scale * (-0.5 * difference.square().sum(-1)).exp()

    prior_Z = (features(points) @ weight.mT).mT + constant
    prior_X = (features(query) @ weight.mT).mT + constant
    factor = torch.linalg.cholesky(covariance(points, points))
    distribution = gp.model.variational_strategy._variational_distribution
    whitened = (
        distribution.variational_mean + (distribution.chol_variational_covar.tril() @ normal.mT).mT
    )
    target = constant + (factor @ whitened.mT).mT
    update_weight = torch.cholesky_solve((target - prior_Z).mT, factor).mT
    transformed = prior_X + (covariance(query, points) @ update_weight.mT).mT
    return transformed * train_Y.std(-2, correction=1).squeeze(-1) + train_Y.mean(-2).squeeze(-1)


@pytest.mark.parametrize("device", DEVICES)
def test_variational_path_all_data_parameter_draw_hvps_match_independent_formula(device):
    from test_torch_acq_discrete import no_host

    X, Y = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]], requires_grad=True)
    frequency = (
        torch.linspace(-0.8, 1.0, 16, dtype=X.dtype, device=device).reshape(8, 2).requires_grad_()
    )
    weight = (
        torch.linspace(-0.7, 0.6, 32, dtype=X.dtype, device=device).reshape(2, 16).requires_grad_()
    )
    normal = (
        torch.linspace(-0.6, 0.7, 6, dtype=X.dtype, device=device).reshape(2, 3).requires_grad_()
    )
    gp = make_model(X, Y, transformed=True)
    leaves = (X, Y, query, frequency, weight, normal, *gp.parameters())
    with no_host():
        actual = paths(gp, draws=[(frequency, weight, normal)])(query)
        expected = independent_rbf_matheron(gp, X, Y, query, frequency, weight, normal)
        actual_gradients = torch.autograd.grad(
            actual.square().sum(), leaves, create_graph=True, allow_unused=True
        )
        expected_gradients = torch.autograd.grad(
            expected.square().sum(), leaves, create_graph=True, allow_unused=True
        )
        direction = [
            torch.linspace(-0.3, 0.4, leaf.numel(), dtype=leaf.dtype, device=device).reshape_as(
                leaf
            )
            for leaf in leaves
        ]
        actual_hvp = torch.autograd.grad(
            sum(
                (gradient * vector).sum()
                for gradient, vector in zip(actual_gradients, direction, strict=True)
                if gradient is not None
            ),
            leaves,
            allow_unused=True,
        )
        expected_hvp = torch.autograd.grad(
            sum(
                (gradient * vector).sum()
                for gradient, vector in zip(expected_gradients, direction, strict=True)
                if gradient is not None
            ),
            leaves,
            allow_unused=True,
        )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    for ag, eg, ah, eh in zip(
        actual_gradients, expected_gradients, actual_hvp, expected_hvp, strict=True
    ):
        assert (ag is None) == (eg is None)
        assert (ah is None) == (eh is None)
        if ag is not None:
            torch.testing.assert_close(ag, eg, atol=3e-8, rtol=3e-7)
        if ah is not None:
            torch.testing.assert_close(ah, eh, atol=3e-7, rtol=3e-6)
    # Observation noise is deliberately absent from the latent variational law.
    noise = dict(gp.named_parameters())["likelihood.noise_covar.raw_noise"]
    assert actual_gradients[next(i for i, leaf in enumerate(leaves) if leaf is noise)] is None


@pytest.mark.parametrize("device", DEVICES)
def test_variational_path_complete_data_query_draw_gradcheck_and_gradgradcheck(device):
    X, Y = data(device)
    arguments = (
        X.requires_grad_(),
        Y.requires_grad_(),
        X.new_tensor([[0.19, 0.23]], requires_grad=True),
        torch.linspace(-0.8, 1.0, 16, dtype=X.dtype, device=device).reshape(8, 2).requires_grad_(),
        torch.linspace(-0.7, 0.6, 16, dtype=X.dtype, device=device).reshape(1, 16).requires_grad_(),
        X.new_tensor([[0.2, -0.1, 0.4]], requires_grad=True),
    )

    def function(x, y, query, frequency, weight, normal):
        return paths(
            make_model(x, y, transformed=True), shape=(1,), draws=[(frequency, weight, normal)]
        )(query)

    assert torch.autograd.gradcheck(function, arguments, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, arguments, fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_variational_public_prior_update_targets_path_model_and_acquisition(device, shape):
    from test_torch_acq_discrete import no_host

    from botorch_rs.acquisition.monte_carlo import qSimpleRegret
    from botorch_rs.sampling.normal import IIDNormalSampler

    X, Y = data(device)
    gp = make_model(X, Y, transformed=True)
    query = X.new_tensor([[0.19, 0.23], [0.68, 0.74]], requires_grad=True)
    with no_host():
        prior = draw_kernel_feature_paths(gp, shape, num_features=FEATURES, seed=23)
        Z = gp.model.variational_strategy.inducing_points
        sample_values = prior.forward(Z)
        update = gaussian_update(gp, sample_values, seed=29)
        update_repeated = gaussian_update(gp, sample_values, likelihood=None, seed=29)
        torch.testing.assert_close(update(query), update_repeated(query))
        target = X.new_full(sample_values.shape, 0.7)
        explicit = gaussian_update(gp, sample_values, target_values=target)
        # Z is already normalized. Bypass the feature map's raw-input
        # transform, just as the sampler bypasses the prior path's transform.
        at_inducing = (explicit.feature_map.forward(Z) @ explicit.weight.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(prior.forward(Z) + at_inducing, target, atol=1e-10, rtol=1e-10)
        with pytest.raises(NotImplementedError, match="noise covariance"):
            gaussian_update(gp, sample_values, noise_covariance=torch.eye(3, device=device))
        with pytest.raises(ValueError, match="num_inducing"):
            gaussian_update(gp, sample_values[..., :-1])
        path_model = get_matheron_path_model(gp, sample_shape=shape, num_features=FEATURES, seed=23)
        value = path_model(query)
        assert value.shape == (*shape, 2, 1)
        acquisition = qSimpleRegret(path_model, sampler=IIDNormalSampler((1,), seed=13))
        result = acquisition(query.unsqueeze(0))
        first = torch.autograd.grad(result.sum(), query, create_graph=True)[0]
        assert torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all()
    converted = path_model.to(dtype=torch.float32)
    assert converted(query.float()).dtype == torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_variational_prior_without_targets_and_current_uninitialized_q_are_preserved(device):
    X, _ = data(device)
    gp = SingleTaskVariationalGP(X, inducing_points=X[:3], backend="torch").eval()
    before = gp.model.variational_strategy._variational_distribution.variational_mean.clone()
    assert not gp.model.variational_strategy.variational_params_initialized
    first = paths(gp, seed=23)
    second = paths(gp, seed=23)
    torch.testing.assert_close(first(X), second(X))
    torch.testing.assert_close(
        before, gp.model.variational_strategy._variational_distribution.variational_mean
    )
    assert not gp.model.variational_strategy.variational_params_initialized
