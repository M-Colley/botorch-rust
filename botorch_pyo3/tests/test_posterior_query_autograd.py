"""Native first-order GP posterior query gradients and snapshot semantics."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtScalarize  # noqa: E402
from botorch.models import SingleTaskGP as BtGP  # noqa: E402
from botorch.models.transforms.outcome import Log as BtLog  # noqa: E402
from botorch.models.transforms.outcome import Standardize as BtStandardize  # noqa: E402

from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform  # noqa: E402
from botorch_rs.models import ModelListGP, MultiTaskGP, SingleTaskGP  # noqa: E402
from botorch_rs.models.transforms.outcome import Log, Standardize  # noqa: E402
from botorch_rs.sampling import IIDNormalSampler, SobolQMCNormalSampler  # noqa: E402


def pair(outputs=1, batch=(), outcome="none", dtype=torch.float64):
    generator = torch.Generator().manual_seed(762)
    x = torch.rand(*batch, 6, 2, generator=generator, dtype=dtype)
    y = torch.rand(*batch, 6, outputs, generator=generator, dtype=dtype) + 0.3
    noise = torch.full_like(y, 0.025)
    transforms = {
        "none": (None, None),
        "standardize": (
            Standardize(outputs, batch_shape=batch),
            BtStandardize(outputs, batch_shape=torch.Size(batch)),
        ),
        "log": (Log(), BtLog()),
    }
    actual_t, reference_t = transforms[outcome]
    actual = SingleTaskGP(x, y, noise, outcome_transform=actual_t)
    reference = BtGP(x, y, noise, outcome_transform=reference_t)
    return actual, reference


def weighted_gradient(values, query):
    weights = torch.linspace(0.2, 0.8, values.numel(), dtype=values.dtype).reshape(values.shape)
    return torch.autograd.grad((values * weights).sum(), query)[0]


@pytest.mark.parametrize("outputs,batch", [(1, ()), (2, ()), (1, (2,)), (2, (2,))])
@pytest.mark.parametrize("mode", ["mean", "variance", "covariance", "samples"])
@pytest.mark.parametrize("noisy", [False, True])
def test_native_gaussian_query_gradients_match_official(outputs, batch, mode, noisy):
    actual, reference = pair(outputs, batch)
    query = torch.tensor([[[0.25, 0.55], [0.68, 0.33]]], dtype=torch.float64, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    ap, bp = (
        actual.posterior(query, observation_noise=noisy),
        reference.posterior(other, observation_noise=noisy),
    )
    if mode == "covariance":
        av, bv = ap.distribution.covariance_matrix, bp.distribution.covariance_matrix
    elif mode == "samples":
        shape = ap.base_sample_shape
        z = torch.linspace(-0.8, 0.9, 3 * torch.Size(shape).numel(), dtype=torch.float64).reshape(
            3, *shape
        )
        av, bv = (
            ap.rsample_from_base_samples((3,), z),
            bp.rsample_from_base_samples(torch.Size([3]), z),
        )
    else:
        av, bv = getattr(ap, mode), getattr(bp, mode)
    assert av.requires_grad
    torch.testing.assert_close(av, bv, rtol=3e-9, atol=3e-10)
    ag, bg = weighted_gradient(av, query), weighted_gradient(bv, other)
    torch.testing.assert_close(ag, bg, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("mode", ["mean", "variance", "samples"])
@pytest.mark.parametrize("outcome", ["standardize", "log"])
@pytest.mark.parametrize("noisy", [False, True])
def test_native_transformed_posterior_query_gradients(outcome, mode, noisy):
    actual, reference = pair(outcome=outcome)
    query = torch.tensor([[0.25, 0.55], [0.68, 0.33]], dtype=torch.float64, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    ap, bp = (
        actual.posterior(query, observation_noise=noisy),
        reference.posterior(other, observation_noise=noisy),
    )
    if mode == "samples":
        z = torch.tensor([[-0.5, 0.2], [0.1, 0.8]], dtype=torch.float64)
        av, bv = (
            ap.rsample_from_base_samples((2,), z),
            bp.rsample_from_base_samples(torch.Size([2]), z),
        )
    else:
        av, bv = getattr(ap, mode), getattr(bp, mode)
    torch.testing.assert_close(av, bv, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(
        weighted_gradient(av, query), weighted_gradient(bv, other), rtol=3e-8, atol=3e-9
    )


@pytest.mark.parametrize("selection", [[1], [1, 0]])
@pytest.mark.parametrize("scalarize", [False, True])
@pytest.mark.parametrize("mode", ["mean", "variance", "covariance", "samples"])
def test_selected_scalarized_output_query_gradients(selection, scalarize, mode):
    actual, reference = pair(outputs=2, batch=(2,))
    query = torch.tensor([[0.25, 0.55], [0.68, 0.33]], dtype=torch.float64, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    weights = torch.linspace(-0.3, 0.7, len(selection), dtype=torch.float64)
    at = ScalarizedPosteriorTransform(weights, offset=0.2) if scalarize else None
    bt = BtScalarize(weights, offset=0.2) if scalarize else None
    ap = actual.posterior(query, output_indices=selection, posterior_transform=at)
    if len(selection) == 1:
        # Official b0dd constructs an MTMVN from only one selected MVN and
        # raises. Subsetting the reference model gives the intended law.
        with pytest.raises(ValueError, match="at least 2 MVNs"):
            reference.posterior(other, output_indices=selection, posterior_transform=bt)
        bp = reference.subset_output(selection).posterior(other, posterior_transform=bt)
    else:
        bp = reference.posterior(other, output_indices=selection, posterior_transform=bt)
    if mode == "covariance":
        av, bv = ap.distribution.covariance_matrix, bp.distribution.covariance_matrix
    elif mode == "samples":
        shape = ap.base_sample_shape
        z = torch.linspace(-0.8, 0.9, 3 * torch.Size(shape).numel(), dtype=torch.float64).reshape(
            3, *shape
        )
        av, bv = (
            ap.rsample_from_base_samples((3,), z),
            bp.rsample_from_base_samples(torch.Size([3]), z),
        )
    else:
        av, bv = getattr(ap, mode), getattr(bp, mode)
    torch.testing.assert_close(av, bv, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(
        weighted_gradient(av, query), weighted_gradient(bv, other), rtol=3e-8, atol=3e-9
    )


@pytest.mark.parametrize("sampler", [IIDNormalSampler, SobolQMCNormalSampler])
def test_sampler_preserves_query_graph_and_sample_no_grad(sampler):
    model, _ = pair()
    query = torch.tensor([[0.25, 0.55], [0.68, 0.33]], dtype=torch.float64, requires_grad=True)
    p = model.posterior(query)
    values = sampler((4,), seed=71)(p)
    assert values.requires_grad
    assert torch.isfinite(weighted_gradient(values, query)).all()
    assert not p.sample((4,)).requires_grad
    with torch.no_grad():
        assert not p.mean.requires_grad
        assert not p.rsample((4,)).requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distribution_views_dtype_and_parameter_snapshot(dtype):
    model, _ = pair(dtype=dtype)
    query = torch.tensor([[0.25, 0.55], [0.68, 0.33]], dtype=dtype, requires_grad=True)
    posterior = model.posterior(query)
    expected = weighted_gradient(posterior.mean.square() + posterior.variance, query)
    model.covar_module.lengthscale = 2.0
    values = posterior.distribution.mean.square() + posterior.distribution.variance
    actual = weighted_gradient(values, query)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    assert posterior.distribution.stddev.requires_grad
    assert posterior.distribution.rsample((2,)).requires_grad


def test_model_list_and_multitask_query_gradients_match_finite_differences():
    a, _ = pair()
    b, _ = pair()
    models = [
        ModelListGP(a, b),
        MultiTaskGP(
            torch.tensor([[0.1, 0], [0.3, 1], [0.7, 0], [0.9, 1]], dtype=torch.float64),
            torch.tensor([[0.2], [0.7], [-0.3], [0.8]], dtype=torch.float64),
            task_feature=-1,
            outcome_transform=None,
        ),
    ]
    for model in models:
        d = 2 if isinstance(model, ModelListGP) else 1
        query = (
            torch.tensor([[0.25, 0.55], [0.68, 0.33]], dtype=torch.float64)[:, :d]
            .clone()
            .requires_grad_()
        )
        posterior = model.posterior(query)
        loss = (
            posterior.mean.square().sum()
            + posterior.variance.sum()
            + posterior.distribution.covariance_matrix.square().sum()
        )
        gradient = torch.autograd.grad(loss, query)[0]
        for i in range(2):
            for j in range(d):

                def value(step, query=query, i=i, j=j, model=model):
                    x = query.detach().clone()
                    x[i, j] += step
                    p = model.posterior(x)
                    return (
                        p.mean.square().sum()
                        + p.variance.sum()
                        + p.distribution.covariance_matrix.square().sum()
                    )

                finite = (value(1e-6) - value(-1e-6)) / 2e-6
                torch.testing.assert_close(gradient[i, j], finite, rtol=2e-6, atol=2e-7)


def test_float32_query_gradients_match_official_precision():
    actual, reference = pair(outputs=2, dtype=torch.float32)
    query = torch.tensor([[0.25, 0.55], [0.68, 0.33]], dtype=torch.float32, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    a, b = actual.posterior(query), reference.posterior(other)
    av = a.mean.square().sum() + a.variance.sum() + a.distribution.covariance_matrix.square().sum()
    bv = b.mean.square().sum() + b.variance.sum() + b.distribution.covariance_matrix.square().sum()
    torch.testing.assert_close(av, bv, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        torch.autograd.grad(av, query)[0], torch.autograd.grad(bv, other)[0], rtol=2e-4, atol=2e-5
    )


@pytest.mark.parametrize("transform", ["normalize", "warp"])
def test_raw_query_gradient_chain_through_native_input_transform(transform):
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.models.transforms.input import Warp as BtWarp

    from botorch_rs.models.transforms.input import Normalize, Warp

    generator = torch.Generator().manual_seed(61)
    x = torch.rand(6, 2, generator=generator, dtype=torch.float64)
    y = torch.rand(6, 1, generator=generator, dtype=torch.float64)
    if transform == "normalize":
        bounds = torch.tensor([[-0.2, 0.0], [1.2, 2.0]], dtype=torch.float64)
        at, bt = Normalize(2, bounds=bounds), BtNormalize(2, bounds=bounds)
    else:
        at, bt = Warp(2, indices=[0, 1]), BtWarp(2, indices=[0, 1])
        c0, c1 = torch.tensor([1.2, 0.8]), torch.tensor([0.9, 1.1])
        at.concentration0, at.concentration1 = c0, c1
        with torch.no_grad():
            bt.concentration0.copy_(c0)
            bt.concentration1.copy_(c1)
    actual, reference = SingleTaskGP(x, y, input_transform=at), BtGP(x, y, input_transform=bt)
    query = torch.tensor([[0.25, 0.55], [0.68, 0.33]], dtype=torch.float64, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    ap, bp = actual.posterior(query), reference.posterior(other)
    av = (
        ap.mean.square().sum()
        + ap.variance.sum()
        + ap.distribution.covariance_matrix.square().sum()
    )
    bv = (
        bp.mean.square().sum()
        + bp.variance.sum()
        + bp.distribution.covariance_matrix.square().sum()
    )
    torch.testing.assert_close(av, bv, rtol=3e-8, atol=3e-9)
    torch.testing.assert_close(
        torch.autograd.grad(av, query)[0], torch.autograd.grad(bv, other)[0], rtol=3e-7, atol=3e-8
    )


def test_structured_posterior_query_gradients_fail_explicitly():
    from botorch_rs.exceptions.errors import UnsupportedError
    from botorch_rs.models import HigherOrderGP, LatentKroneckerGP

    x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    y = torch.tensor([[0.2, 0.3], [0.7, 0.9], [0.1, 0.4]], dtype=torch.float64)
    query = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    for model in (
        HigherOrderGP(x, y),
        LatentKroneckerGP(x, torch.tensor([[0.0], [1.0]], dtype=torch.float64), y),
    ):
        with pytest.raises(UnsupportedError, match="structured"):
            model.posterior(query)
        with torch.no_grad():
            assert torch.isfinite(model.posterior(query).mean).all()


@pytest.mark.parametrize("multitask", [False, True])
@pytest.mark.parametrize("mode", ["mean", "variance", "covariance", "samples"])
def test_fully_bayesian_component_query_gradients_match_finite_differences(multitask, mode):
    from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch_rs.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

    x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    y = torch.tensor([[0.2], [0.7], [0.1]], dtype=torch.float64)
    if multitask:
        x = torch.cat(
            [torch.cat([x, torch.zeros_like(x)], -1), torch.cat([x, torch.ones_like(x)], -1)]
        )
        y = torch.cat([y, y + 0.2])
        model = SaasFullyBayesianMultiTaskGP(x, y, task_feature=-1, rank=1)
    else:
        model = SaasFullyBayesianSingleTaskGP(x, y)
    samples = model.pyro_model.get_dummy_mcmc_samples(2)
    if multitask:
        samples["latent_features"] = [[[0.2], [0.9]], [[0.3], [0.8]]]
    model.load_mcmc_samples(samples)
    query = torch.tensor([[0.3], [0.6]], dtype=torch.float64, requires_grad=True)

    def value(at):
        p = model.posterior(at)
        if mode == "covariance":
            values = p.distribution.covariance_matrix
        elif mode == "samples":
            z = torch.linspace(-0.4, 0.7, 2 * p._q * p._m, dtype=torch.float64)
            values = p.rsample_from_base_samples((2,), z.reshape(2, 1, p._q, p._m))
        else:
            values = getattr(p, mode)
        return values.square().sum()

    gradient = torch.autograd.grad(value(query), query)[0]
    for i in range(2):
        left, right = query.detach().clone(), query.detach().clone()
        left[i, 0] -= 1e-6
        right[i, 0] += 1e-6
        finite = (value(right) - value(left)) / 2e-6
        torch.testing.assert_close(gradient[i, 0], finite, rtol=3e-6, atol=3e-7)
