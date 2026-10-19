"""Efficient LOO parity with BoTorch, including fixed noise and ensembles."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.cross_validation as bt_cv  # noqa: E402
from botorch.models import SingleTaskGP as BtGP  # noqa: E402
from botorch.models.transforms import Log as BtLog  # noqa: E402
from botorch.models.transforms import Normalize as BtNormalize  # noqa: E402
from botorch.models.transforms import Standardize as BtStandardize  # noqa: E402

from botorch_rs.cross_validation import loo_cv  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsGP  # noqa: E402
from botorch_rs.models.transforms import Log as RsLog  # noqa: E402
from botorch_rs.models.transforms import Normalize as RsNormalize  # noqa: E402
from botorch_rs.models.transforms import Standardize as RsStandardize  # noqa: E402


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("transform", ["none", "standardize", "log"])
@pytest.mark.parametrize("observation_noise", [False, True])
@pytest.mark.parametrize("untransform", [False, True])
def test_loo_parity(batch, m, fixed, transform, observation_noise, untransform) -> None:
    if not hasattr(bt_cv, "loo_cv"):
        pytest.skip("requires BoTorch's efficient LOO API")
    generator = torch.Generator().manual_seed(72)
    x = 2.0 + 3.0 * torch.rand(*batch, 6, 2, generator=generator, dtype=torch.float64)
    y = torch.randn(*batch, 6, m, generator=generator, dtype=torch.float64)
    if transform == "log":
        y = y.exp()
    noise = 0.01 + 0.02 * torch.rand(y.shape, generator=generator, dtype=torch.float64)
    models = []
    for GP, ST, Log, Normalize in (
        (BtGP, BtStandardize, BtLog, BtNormalize),
        (RsGP, RsStandardize, RsLog, RsNormalize),
    ):
        outcome = ST(m=m, batch_shape=batch) if transform == "standardize" else None
        if transform == "log":
            outcome = Log()
        model = GP(
            x,
            y,
            train_Yvar=noise if fixed else None,
            outcome_transform=outcome,
            input_transform=Normalize(d=2, batch_shape=batch),
        )
        model.covar_module.lengthscale = 0.6
        model.mean_module.constant = 0.15
        if not fixed:
            model.likelihood.noise = 0.04
        models.append(model)
    expected = bt_cv.loo_cv(models[0], observation_noise, untransform)
    actual = loo_cv(models[1], observation_noise, untransform)
    for a, e in (
        (actual.posterior.mean, expected.posterior.mean),
        (actual.posterior.variance, expected.posterior.variance),
        (actual.observed_Y, expected.observed_Y),
    ):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    if fixed:
        torch.testing.assert_close(
            actual.observed_Yvar, expected.observed_Yvar, rtol=1e-9, atol=1e-10
        )
    sample_shape = (3,)
    base = torch.randn(
        *sample_shape,
        *expected.posterior.base_sample_shape,
        generator=generator,
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        actual.posterior.rsample_from_base_samples(sample_shape, base),
        expected.posterior.rsample_from_base_samples(torch.Size(sample_shape), base),
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("untransform", [False, True])
def test_fully_bayesian_loo(fixed, untransform) -> None:
    if not hasattr(bt_cv, "loo_cv"):
        pytest.skip("requires BoTorch's efficient LOO API")
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP as BtFB

    from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP as RsFB

    x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.float64)
    y = torch.tensor([[1.5], [2.1], [1.3], [2.8]], dtype=torch.float64)
    samples = {
        "lengthscale": torch.tensor([[0.2], [0.7], [1.3]], dtype=torch.float64),
        "outputscale": torch.tensor([0.7, 1.0, 1.5], dtype=torch.float64),
        "mean": torch.tensor([0.1, -0.3, 0.2], dtype=torch.float64),
    }
    if not fixed:
        samples["noise"] = torch.tensor([0.02, 0.04, 0.08], dtype=torch.float64)
    models = []
    for cls, ST in ((BtFB, BtStandardize), (RsFB, RsStandardize)):
        model = cls(
            x, y, train_Yvar=torch.full_like(y, 0.03) if fixed else None, outcome_transform=ST(m=1)
        )
        model.load_mcmc_samples({k: v.clone() for k, v in samples.items()})
        model.eval()
        models.append(model)
    expected = bt_cv.loo_cv(models[0], untransform=untransform)
    actual = loo_cv(models[1], untransform=untransform)
    for name in (
        "mean",
        "variance",
        "mixture_mean",
        "mixture_variance",
        "mixture_covariance_matrix",
    ):
        torch.testing.assert_close(
            getattr(actual.posterior, name),
            getattr(expected.posterior, name),
            rtol=1e-9,
            atol=1e-10,
        )
    torch.testing.assert_close(actual.observed_Y, expected.observed_Y, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("transform", ["none", "standardize", "log"])
def test_ensemble_batch_layout_and_sample_transform(transform) -> None:
    from botorch_rs.cross_validation import efficient_loo_cv

    x = torch.linspace(0.1, 0.9, 4, dtype=torch.float64).reshape(4, 1).expand(2, 3, 4, 1)
    y = (1.0 + x.sin()).expand(2, 3, 4, 2).contiguous()
    outcome = RsStandardize(m=2, batch_shape=(2, 3)) if transform == "standardize" else None
    if transform == "log":
        outcome = RsLog()
    model = RsGP(x, y, outcome_transform=outcome)
    model._is_ensemble = True
    actual = loo_cv(model)
    raw = efficient_loo_cv(model, untransform=False)
    mu = raw.posterior.mean.movedim(-4, -3)
    var = raw.posterior.variance.movedim(-4, -3)
    assert actual.posterior.mean.shape == (2, 4, 3, 1, 2)
    if transform == "log":
        expected_mean = (mu + var / 2).exp()
        expected_variance = (2 * mu + var).exp() * var.expm1()
    elif transform == "standardize":
        expected = efficient_loo_cv(model)
        expected_mean = expected.posterior.mean.movedim(-4, -3)
        expected_variance = expected.posterior.variance.movedim(-4, -3)
    else:
        expected_mean, expected_variance = mu, var
    torch.testing.assert_close(actual.posterior.mean, expected_mean)
    torch.testing.assert_close(actual.posterior.variance, expected_variance)
    if transform == "log":
        # Zero Gaussian base samples map to exp(mu), not to the lognormal
        # mean. This catches replacing transformed components with Gaussians.
        samples = actual.posterior.rsample_from_base_samples(
            (1,), torch.zeros(1, *actual.posterior.base_sample_shape, dtype=torch.float64)
        )
        torch.testing.assert_close(samples[0], mu.exp())


def test_ensemble_rejects_different_training_targets() -> None:
    x = torch.tensor([[[0.1], [0.8]], [[0.1], [0.8]]], dtype=torch.float64)
    y = torch.tensor([[[1.0], [2.0]], [[1.0], [3.0]]], dtype=torch.float64)
    model = RsGP(x, y, outcome_transform=None)
    model._is_ensemble = True
    with pytest.raises(NotImplementedError, match="different train_Y"):
        loo_cv(model)
