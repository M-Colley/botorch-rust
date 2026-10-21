# ruff: noqa: E402
"""Structured conditioning preserves actual Gaussian modules and noise graphs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from test_torch_structured_gaussian_likelihoods import (
    CorrelatedGaussian,
    CorrelatedMultitask,
    gradients,
    independent_observed,
    independent_posterior,
    make_model,
    point_major_posterior,
    prior_blocks,
)

from botorch_rs.models import HigherOrderGP, KroneckerMultiTaskGP
from botorch_rs.models.transforms.outcome import Standardize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("standardized", [False, True])
def test_actual_fantasy_likelihood_and_optional_noise_complete_gaussian_hessians(
    device, family, explicit, standardized
):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device, requires_grad=True)
    Y = torch.sin(4 * X.detach() + X.new_tensor([0.0, 0.6])).requires_grad_()
    W = X.new_tensor([[0.25]], requires_grad=True)
    targets = X.new_tensor([[0.8, 0.3]], requires_grad=True)
    noise = X.new_tensor([[0.08, 0.17]], requires_grad=True) if explicit else None
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    likelihood = (CorrelatedMultitask() if family == "kron" else CorrelatedGaussian()).to(X)
    cls = KroneckerMultiTaskGP if family == "kron" else HigherOrderGP
    transform = None
    if standardized:
        if family == "higher":
            from botorch_rs.models.higher_order_gp import FlattenedStandardize

            transform = FlattenedStandardize([2])
        else:
            transform = Standardize(m=2)
    model = cls(X, Y, likelihood=likelihood, outcome_transform=transform, backend="torch")
    model.eval()
    conditioned = model.condition_on_observations(W, targets, noise=noise)
    assert conditioned.likelihood is not likelihood
    assert type(conditioned.likelihood) is type(likelihood)
    leaves = (X, Y, W, targets, Q, *(() if noise is None else (noise,)), *conditioned.parameters())
    for _ in range(2):
        posterior = conditioned.posterior(Q)
        actual_mean, actual_covariance = point_major_posterior(posterior, family)
        train_mean, query_mean, train, cross, test = prior_blocks(conditioned, family, Q)
        observed_mean, observed = independent_observed(conditioned, family, train_mean, train)
        if explicit:
            variance = noise
            if conditioned.outcome_transform is not None and family == "higher":
                variance = conditioned.outcome_transform(torch.zeros_like(variance), variance)[1]
            keep = train.new_tensor([1.0] * 6 + [0.0] * 2)
            observed = train + (observed - train) * keep[:, None] * keep[None, :]
            observed = observed + torch.diag_embed(
                torch.cat((train.new_zeros(6), variance.flatten(-2)))
            )
        values = conditioned.train_targets
        values = values.flatten(-2) if family == "kron" else values
        mean = query_mean + (
            cross.T @ torch.linalg.solve(observed, (values - observed_mean).unsqueeze(-1))
        ).squeeze(-1)
        covariance = test - cross.T @ torch.linalg.solve(observed, cross)
        if conditioned.outcome_transform is not None:
            transform = conditioned.outcome_transform
            if family == "kron":
                stdvs = transform.stdvs.squeeze(-2).repeat(Q.shape[-2])
                location = transform.means.squeeze(-2).repeat(Q.shape[-2])
            else:
                stdvs, location = transform.stdvs.squeeze(), transform.means.squeeze()
            mean = mean * stdvs + location
            covariance = (
                covariance * stdvs[..., None] * stdvs[..., None, :]
                if family == "kron"
                else covariance * stdvs.square()
            )
        torch.testing.assert_close(actual_mean, mean, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(actual_covariance, covariance, atol=2e-9, rtol=2e-9)
        aa = gradients(actual_mean.square().sum() + actual_covariance.square().sum(), leaves, True)
        bb = gradients(mean.square().sum() + covariance.square().sum(), leaves, True)
        for a, b in zip(aa, bb, strict=True):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)
        for a, b in zip(
            gradients(sum(g.square().sum() for g in aa), leaves),
            gradients(sum(g.square().sum() for g in bb), leaves),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
    assert model._train_X.shape[-2] == 3
    assert conditioned._train_X.shape[-2] == 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("standardized", [False, True])
@pytest.mark.parametrize("additional", [False, True])
def test_fixed_noise_higher_conditioning_repeated_calibration_graphs(
    device, standardized, additional
):
    from botorch_rs.models.higher_order_gp import FlattenedStandardize

    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device, requires_grad=True)
    Y = torch.sin(4 * X.detach() + X.new_tensor([0.0, 0.6])).requires_grad_()
    original = X.new_tensor([0.08, 0.07, 0.09, 0.1, 0.13, 0.14], requires_grad=True)
    likelihood = FixedNoiseGaussianLikelihood(noise=original, learn_additional_noise=additional).to(
        X
    )
    model = HigherOrderGP(
        X,
        Y,
        likelihood=likelihood,
        outcome_transform=FlattenedStandardize([2]) if standardized else None,
        backend="torch",
    )
    W, targets = X.new_tensor([[0.25]], requires_grad=True), X.new_tensor([[0.8, 0.3]])
    noise = X.new_tensor([[0.08, 0.17]], requires_grad=True)
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    model.eval()
    with pytest.raises(RuntimeError, match="noise.*kwarg"):
        model.condition_on_observations(W, targets)
    conditioned = model.condition_on_observations(W, targets, noise=noise)
    assert isinstance(conditioned.likelihood, FixedNoiseGaussianLikelihood)
    assert conditioned.likelihood is not likelihood
    leaves = (X, Y, W, original, noise, Q, *conditioned.parameters())
    for _ in range(2):
        mean_train, mean_query, train, cross, test = prior_blocks(conditioned, "higher", Q)
        variance = noise
        if standardized:
            before_std = conditioned.outcome_transform.stdvs.clone()
            variance = conditioned.outcome_transform(torch.zeros_like(noise), noise)[1]
        variance = torch.cat((original, variance.flatten(-2)))
        if additional:
            variance = variance + conditioned.likelihood.second_noise
        observed = train + torch.diag_embed(variance)
        values = conditioned.train_targets
        if standardized:
            torch.testing.assert_close(conditioned.outcome_transform.stdvs, before_std)
        mean = mean_query + (
            cross.T @ torch.linalg.solve(observed, (values - mean_train).unsqueeze(-1))
        ).squeeze(-1)
        covariance = test - cross.T @ torch.linalg.solve(observed, cross)
        expected_mll = (
            torch.distributions.MultivariateNormal(mean_train, observed).log_prob(values)
            + conditioned._log_prior()
        ) / values.shape[-1]
        actual_mll = conditioned._exact_mll()
        torch.testing.assert_close(actual_mll, expected_mll, atol=2e-9, rtol=2e-9)
        if standardized:
            stdvs, location = (
                conditioned.outcome_transform.stdvs,
                conditioned.outcome_transform.means,
            )
            mean = mean * stdvs.squeeze() + location.squeeze()
            covariance = covariance * stdvs.squeeze().square()
        actual_mean, actual_covariance = point_major_posterior(conditioned.posterior(Q), "higher")
        torch.testing.assert_close(actual_mean, mean, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(actual_covariance, covariance, atol=2e-9, rtol=2e-9)
        aa = gradients(actual_mll + actual_mean.square().sum(), leaves, True)
        bb = gradients(expected_mll + mean.square().sum(), leaves, True)
        for a, b in zip(aa, bb, strict=True):
            torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-7)
        for a, b in zip(
            gradients(sum(g.square().sum() for g in aa), leaves),
            gradients(sum(g.square().sum() for g in bb), leaves),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=4e-6, rtol=4e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher"])
def test_source_valid_supplied_likelihood_conditioning_values_and_query_gradients(device, family):
    from botorch.models import HigherOrderGP as ReferenceHigher
    from botorch.models import KroneckerMultiTaskGP as ReferenceKron
    from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
    from gpytorch.settings import detach_test_caches, fast_pred_var

    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device)
    Y = torch.sin(4 * X + X.new_tensor([0.0, 0.6]))
    W, targets = X.new_tensor([[0.25]]), X.new_tensor([[0.8, 0.3]])
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    cls, reference_cls = (
        (KroneckerMultiTaskGP, ReferenceKron)
        if family == "kron"
        else (HigherOrderGP, ReferenceHigher)
    )
    factory = (
        (lambda: MultitaskGaussianLikelihood(num_tasks=2))
        if family == "kron"
        else GaussianLikelihood
    )
    actual = cls(X, Y, likelihood=factory().to(X), outcome_transform=None, backend="torch")
    reference = reference_cls(X, Y, likelihood=factory().to(X), outcome_transform=None).to(X)
    aa, bb = dict(actual.named_parameters()), dict(reference.named_parameters())
    assert aa.keys() == bb.keys()
    with torch.no_grad():
        for name in aa:
            bb[name].copy_(aa[name])
    actual.eval()
    reference.eval()
    with detach_test_caches(False), fast_pred_var(False):
        reference(Q)
        expected = reference.condition_on_observations(W, targets)
        source = expected(Q)
        source_mean = source.mean.flatten(-2) if family == "kron" else source.mean
        source_covariance = source.covariance_matrix
        model = actual.condition_on_observations(W, targets)
        mean, covariance = point_major_posterior(model.posterior(Q), family)
        torch.testing.assert_close(mean, source_mean, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(covariance, source_covariance, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(
            gradients(mean.square().sum() + covariance.square().sum(), (Q,))[0],
            gradients(source_mean.square().sum() + source_covariance.square().sum(), (Q,))[0],
            atol=2e-7,
            rtol=2e-7,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher", "latent"])
def test_fantasies_sample_actual_correlated_observation_marginal(device, family):
    model, X, Y = make_model(family, device)
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    latent_mean, covariance = independent_posterior(model, family, Q)
    mean, covariance = independent_observed(model, family, latent_mean, covariance)

    def sampler(posterior):
        base = getattr(posterior, "_flat", posterior)
        actual_mean, actual_covariance = point_major_posterior(base, "kron")
        torch.testing.assert_close(actual_mean, mean, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(actual_covariance, covariance, atol=2e-9, rtol=2e-9)
        return posterior.mean.unsqueeze(0).expand(2, *posterior.mean.shape)

    fantasy = model.fantasize(Q, sampler, propagate_grads=True)
    torch.testing.assert_close(fantasy._train_Y[..., -2:, :], mean.reshape(1, 2, 2).expand(2, 2, 2))
    loss = fantasy.posterior(Q).mean.square().sum()
    leaves = (X, Y, Q, *fantasy.parameters())
    first = gradients(loss, leaves, True)
    second = gradients(sum(value.square().sum() for value in first), leaves)
    assert all(torch.isfinite(value).all() for value in first + second)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher"])
def test_conditioning_noise_buffers_follow_copy_state_and_migration(device, family):
    import copy
    import io

    model, X, _ = make_model(family, device)
    W, targets = X.new_tensor([[0.25]]), X.new_tensor([[0.8, 0.3]])
    noise = X.new_tensor([[0.08, 0.17]], requires_grad=True)
    query = X.new_tensor([[0.2], [0.7]])
    conditioned = model.condition_on_observations(W, targets, noise=noise)
    baseline = conditioned.posterior(query).mean.detach().cpu()
    state = conditioned.state_dict()
    assert "_botorch_conditioning_noise_0" in state
    cloned = copy.deepcopy(conditioned)
    cloned.load_state_dict(state)
    torch.testing.assert_close(cloned.posterior(query).mean, conditioned.posterior(query).mean)
    serialized = io.BytesIO()
    torch.save(conditioned, serialized)
    serialized.seek(0)
    restored = torch.load(serialized, weights_only=False)
    torch.testing.assert_close(restored.posterior(query).mean, conditioned.posterior(query).mean)
    conditioned.to(device="cpu", dtype=torch.float32)
    assert conditioned._botorch_conditioning_noise_0.dtype == torch.float32
    assert conditioned._botorch_conditioning_noise_0.device.type == "cpu"
    result = conditioned.posterior(query.detach().to(device="cpu", dtype=torch.float32)).mean
    torch.testing.assert_close(result.double(), baseline, atol=4e-5, rtol=4e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher", "latent"])
def test_conditioned_models_own_independent_parameter_leaves(device, family):
    model, X, _ = make_model(family, device)
    query = X.new_tensor([[0.2], [0.7]])
    baseline = model.posterior(query).mean.detach().clone()
    conditioned = model.condition_on_observations(
        X.new_tensor([[0.25]]), X.new_tensor([[0.8, 0.3]])
    )
    original, updated = dict(model.named_parameters()), dict(conditioned.named_parameters())
    assert original.keys() == updated.keys()
    for name in original:
        assert original[name] is not updated[name]
        assert updated[name].is_leaf
    with torch.no_grad():
        for value in updated.values():
            value.add_(0.15)
    torch.testing.assert_close(model.posterior(query).mean, baseline)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher", "latent"])
@pytest.mark.parametrize("propagate", [False, True])
def test_fantasy_query_path_and_original_future_parameter_partials_and_hessians(
    device, family, propagate
):
    model, X, Y = make_model(family, device)
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    query = X.new_tensor([[0.35]], requires_grad=True)
    shifts = X.new_tensor([-0.05, 0.08]).reshape(2, 1, 1)

    def sampler(posterior):
        return posterior.mean.unsqueeze(0) + shifts

    future = model.fantasize(Q, sampler, propagate_grads=propagate)
    mtrain, mquery, train, cross, _ = prior_blocks(model, family, Q)
    observed_mean, observed_covariance = independent_observed(model, family, mtrain, train)
    residual = Y.flatten(-2) - observed_mean
    if not propagate:
        observed_covariance, residual = observed_covariance.detach(), residual.detach()
    sampled = mquery + (
        cross.transpose(-1, -2) @ torch.linalg.solve(observed_covariance, residual.unsqueeze(-1))
    ).squeeze(-1)
    sampled = sampled + model.likelihood.offset
    sampled = sampled.reshape(2, 2).unsqueeze(0) + shifts
    targets = torch.cat((Y.expand(2, *Y.shape), sampled), dim=-2).flatten(-2)
    train_mean, query_mean, train, cross, test = prior_blocks(future, family, query)
    train_mean, train = independent_observed(future, family, train_mean, train)
    mean = query_mean + (
        cross.transpose(-1, -2) @ torch.linalg.solve(train, (targets - train_mean).unsqueeze(-1))
    ).squeeze(-1)
    covariance = test - cross.transpose(-1, -2) @ torch.linalg.solve(train, cross)
    actual_mean, actual_covariance = point_major_posterior(future.posterior(query), family)
    torch.testing.assert_close(actual_mean, mean, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(actual_covariance, covariance, atol=2e-9, rtol=2e-9)
    leaves = (X, Y, Q, query, *model.parameters(), *future.parameters())
    aa = gradients(actual_mean.square().sum() + actual_covariance.square().sum(), leaves, True)
    bb = gradients(mean.square().sum() + covariance.square().sum(), leaves, True)
    for a, b in zip(aa, bb, strict=True):
        torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-7)
    for a, b in zip(
        gradients(sum(g.square().sum() for g in aa), leaves),
        gradients(sum(g.square().sum() for g in bb), leaves),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=5e-6, rtol=5e-6)
