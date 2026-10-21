# ruff: noqa: E402
"""ExactGP eval calls consume transformed coordinates and existing output axes."""

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskGP as ReferenceGP
from botorch.models.transforms.input import Normalize as ReferenceNormalize
from botorch.models.transforms.input import Warp as ReferenceWarp
from botorch.models.transforms.outcome import Standardize as ReferenceStandardize
from test_torch_variational_external_likelihood import DenseRBF

from botorch_rs.kernels import RBFKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.input import Normalize, Warp
from botorch_rs.models.transforms.outcome import Standardize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def pair(device, outputs, batch, warp, external):
    x = torch.tensor([[0.2], [0.6], [1.0], [1.4], [1.8]], device=device, dtype=torch.double)
    y = (x.sin() + 0.3).expand(5, outputs).clone()
    if outputs > 1:
        y[:, 1] += 0.4
    if batch:
        x, y = x.expand(2, 5, 1).clone(), y.expand(2, 5, outputs).clone()
        x[1] += 0.01
        y[1] -= 0.2
    x.requires_grad_()
    y.requires_grad_()
    augmented = torch.Size((*x.shape[:-2], outputs)) if outputs > 1 else x.shape[:-2]
    models = []
    for reference in (False, True):
        if warp:
            transform = (ReferenceWarp if reference else Warp)(
                1, [0], bounds=x.new_tensor([[0.0], [2.0]])
            )
        else:
            transform = (ReferenceNormalize if reference else Normalize)(
                1, bounds=x.new_tensor([[0.0], [2.0]])
            )
        likelihood = (
            gpytorch.likelihoods.GaussianLikelihood(batch_shape=augmented).to(x)
            if reference or external
            else GaussianLikelihood(batch_shape=augmented)
        )
        model = (ReferenceGP if reference else SingleTaskGP)(
            x,
            y,
            input_transform=transform,
            outcome_transform=(ReferenceStandardize if reference else Standardize)(
                outputs, batch_shape=x.shape[:-2]
            ),
            covar_module=(DenseRBF if reference else RBFKernel)(
                ard_num_dims=1, batch_shape=augmented
            ),
            mean_module=(gpytorch.means.ConstantMean if reference else ConstantMean)(
                batch_shape=augmented
            ),
            likelihood=likelihood,
            **({} if reference else {"backend": "torch"}),
        ).to(x)
        model.covar_module.lengthscale = x.new_tensor(0.6)
        model.mean_module.constant = x.new_tensor(0.2)
        model.likelihood.noise = x.new_tensor(0.06)
        if warp:
            with torch.no_grad():
                model.input_transform.concentration0.fill_(1.3)
                model.input_transform.concentration1.fill_(0.8)
        models.append(model.eval())
    return models, x, y


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("warp", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_exact_eval_call_preserves_transformed_query_and_source_event_batch_axes(
    device, outputs, batch, warp, external
):
    (model, source), x, y = pair(device, outputs, batch, warp, external)
    raw = x.new_tensor([[0.44], [1.23]])
    # Query and existing model/output axes are explicit in an ExactGP call.
    # posterior(raw) has a separate protocol that inserts the output axis.
    raw = (
        raw.reshape(*(1,) * len(model._aug_batch_shape), 2, 1)
        .expand(3, *(1,) * len(model._aug_batch_shape), 2, 1)
        .clone()
        .requires_grad_()
    )
    query = model.transform_inputs(raw)
    reference_query = source.transform_inputs(raw)
    actual = model(query)
    with gpytorch.settings.detach_test_caches(False):
        expected = source(reference_query)
        source_gradient = torch.autograd.grad(
            expected.mean.sin().sum() + expected.variance.sum(), raw
        )[0]
    assert actual.mean.shape == (3, *model._aug_batch_shape, 2)
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual.covariance_matrix, expected.covariance_matrix, atol=3e-12, rtol=3e-11
    )
    query_gradient = torch.autograd.grad(
        actual.mean.sin().sum() + actual.variance.sum(), raw, retain_graph=True
    )[0]
    torch.testing.assert_close(query_gradient, source_gradient, atol=3e-10, rtol=3e-9)

    train, targets, _ = model._training_values()
    lengthscale = model.covar_module.lengthscale

    def kernel(left, right):
        distance = (left.unsqueeze(-2) - right.unsqueeze(-3)) / lengthscale.unsqueeze(-2)
        return (-0.5 * distance.square().sum(-1)).exp()

    observed = kernel(train, train) + torch.diag_embed(model.likelihood.noise.expand_as(targets))
    cross = kernel(train, query)
    train_mean, query_mean = model.mean_module(train), model.mean_module(query)
    mean = query_mean + (
        cross.mT @ torch.linalg.solve(observed, (targets - train_mean).unsqueeze(-1))
    ).squeeze(-1)
    covariance = kernel(query, query) - cross.mT @ torch.linalg.solve(observed, cross)
    torch.testing.assert_close(actual.mean, mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(actual.covariance_matrix, covariance, atol=3e-12, rtol=3e-11)
    leaves = (x, y, raw, *model.parameters())

    def derivatives(mean, covariance):
        first = torch.autograd.grad(
            mean.sin().sum() + covariance.square().sum(),
            leaves,
            create_graph=True,
            retain_graph=True,
        )
        return (
            *first,
            *torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True),
        )

    for actual_gradient, expected_gradient in zip(
        derivatives(actual.mean, actual.covariance_matrix),
        derivatives(mean, covariance),
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-9, rtol=3e-8)
    prior, source_prior = model.forward(query), source.forward(reference_query)
    torch.testing.assert_close(prior.mean, source_prior.mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        prior.covariance_matrix, source_prior.covariance_matrix, atol=3e-12, rtol=3e-11
    )
    assert not torch.allclose(prior.covariance_matrix, actual.covariance_matrix)
    ordinary_raw = x.new_tensor([[0.44], [1.23]])
    actual_posterior, source_posterior = (
        model.posterior(ordinary_raw),
        source.posterior(ordinary_raw),
    )
    torch.testing.assert_close(actual_posterior.mean, source_posterior.mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual_posterior.variance, source_posterior.variance, atol=3e-12, rtol=3e-11
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [False, True])
def test_fully_bayesian_direct_call_keeps_actual_loaded_transform_noise_and_mcmc_axes(
    device, correlated
):
    if correlated:
        from test_torch_fully_bayesian_likelihood import _load as create
        from test_torch_fully_bayesian_likelihood import _oracle
    else:
        from test_torch_fully_bayesian_loader import _model as create
        from test_torch_fully_bayesian_loader import _oracle

    model, x, y, _ = create(device)
    model.eval()
    raw = x.new_tensor([[0.24], [0.63]], requires_grad=True)
    query = model.transform_inputs(raw.unsqueeze(-3))
    actual = model(query)
    mean, covariance = _oracle(model, x, y, raw, False)
    assert actual.mean.shape == (2, 2)
    torch.testing.assert_close(actual.mean, mean.squeeze(-1), atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(actual.covariance_matrix, covariance, atol=3e-12, rtol=3e-11)
    leaves = (x, y, raw, *model.parameters())

    def derivatives(mean, covariance):
        first = torch.autograd.grad(
            mean.sin().sum() + covariance.square().sum(),
            leaves,
            create_graph=True,
            retain_graph=True,
        )
        return (
            *first,
            *torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True),
        )

    for actual_gradient, expected_gradient in zip(
        derivatives(actual.mean, actual.covariance_matrix),
        derivatives(mean.squeeze(-1), covariance),
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-9, rtol=3e-8)
