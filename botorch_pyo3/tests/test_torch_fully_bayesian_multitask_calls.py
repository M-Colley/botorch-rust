# ruff: noqa: E402
"""Direct eval calls retain ExactGP conditioning and the existing MCMC axis."""

import pytest

torch = pytest.importorskip("torch")
from test_fully_bayesian_multitask_subclasses import TaskSubclass
from test_torch_fully_bayesian_multitask import DEVICES, _data, _pair, _samples

from botorch_rs.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("transforms", [False, True])
@pytest.mark.parametrize("batch", ["unbatched", "mcmc", "query_and_mcmc"])
def test_eval_call_conditions_matches_source_and_retains_explicit_mcmc_axis(
    device, fixed, transforms, batch
):
    gpytorch = pytest.importorskip("gpytorch")
    (model, source), _ = _pair(device, fixed=fixed, transforms=transforms)
    raw = model._train_X.new_tensor([[0.22, 3.0, 0.41], [0.63, 7.0, 0.71]])
    if batch != "unbatched":
        raw = raw.expand(3, 2, 3).clone()
        raw[:, :, 0] += torch.arange(3, dtype=raw.dtype, device=raw.device).unsqueeze(-1) * 0.02
    if batch == "query_and_mcmc":
        raw = raw.unsqueeze(0).expand(2, 3, 2, 3).clone()
        raw[1, :, :, 0] += 0.03
    raw.requires_grad_()
    query = model.transform_inputs(raw)
    source_query = source.transform_inputs(raw)
    actual = model(query)
    with gpytorch.settings.detach_test_caches(False):
        expected = source(source_query)
        reference_gradient = torch.autograd.grad(
            expected.mean.sum() + expected.variance.sum(), raw
        )[0]
    expected_batch = (2, 3) if batch == "query_and_mcmc" else (3,)
    assert actual.mean.shape == (*expected_batch, 2)
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual.covariance_matrix, expected.covariance_matrix, atol=3e-12, rtol=3e-11
    )
    gradient = torch.autograd.grad(actual.mean.sum() + actual.variance.sum(), raw)[0]
    torch.testing.assert_close(gradient, reference_gradient, atol=3e-10, rtol=3e-9)
    assert not torch.allclose(model.forward(query).covariance_matrix, actual.covariance_matrix)
    if batch == "unbatched":
        posterior = model.posterior(raw)
        direct_mean = actual.mean.unsqueeze(-1)
        if transforms:
            direct_mean = (
                direct_mean * model.outcome_transform.stdvs + model.outcome_transform.means
            )
        torch.testing.assert_close(posterior.mean, direct_mean, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("subclass", [False, True])
def test_eval_call_independent_dense_all_leaf_hessians_and_training_virtual_prior(
    device, subclass, monkeypatch
):
    X, Y, noise = _data(device, gradients=True)
    cls = TaskSubclass if subclass else SaasFullyBayesianMultiTaskGP
    model = cls(
        X,
        Y,
        1,
        noise,
        backend="torch",
        input_transform=Normalize(3, indices=[0, 2]),
        outcome_transform=Standardize(1),
    )
    model.load_mcmc_samples(_samples(X))
    model.eval()
    query = X.new_tensor([[0.22, 3.0, 0.41], [0.63, 7.0, 0.71]], requires_grad=True)
    transformed = model.transform_inputs(query)
    train, targets, fixed = model._training_values()
    mapped_train = model._mapped_inputs(train)
    mapped_query = model._mapped_inputs(transformed.expand(3, 2, 3))
    scale = model.prior_scale.square() if subclass else 1
    train_mean = model._mean(mapped_train) + (0.1 * train[..., 0].sin() if subclass else 0)
    query_mean = model._mean(mapped_query) + (0.1 * transformed[..., 0].sin() if subclass else 0)
    observed = scale * model._covariance(mapped_train, mapped_train) + torch.diag_embed(
        model._training_noise(targets, fixed)
    )
    cross = scale * model._covariance(mapped_train, mapped_query)
    expected_mean = query_mean + (
        cross.mT @ torch.linalg.solve(observed, (targets - train_mean).unsqueeze(-1))
    ).squeeze(-1)
    expected_covariance = scale * model._covariance(
        mapped_query, mapped_query
    ) - cross.mT @ torch.linalg.solve(observed, cross)

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct device inference must not flatten data")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        actual = model(transformed)
        torch.testing.assert_close(actual.mean, expected_mean, atol=3e-12, rtol=3e-11)
        torch.testing.assert_close(
            actual.covariance_matrix, expected_covariance, atol=3e-12, rtol=3e-11
        )
        leaves = (X, Y, noise, query, *model.parameters())

        def derivatives(mean, covariance):
            first = torch.autograd.grad(
                mean.sin().sum() + covariance.square().sum(),
                leaves,
                create_graph=True,
                retain_graph=True,
            )
            return (
                *first,
                *torch.autograd.grad(
                    sum(g.square().sum() for g in first), leaves, retain_graph=True
                ),
            )

        for actual_gradient, expected_gradient in zip(
            derivatives(actual.mean, actual.covariance_matrix),
            derivatives(expected_mean, expected_covariance),
            strict=True,
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-9, rtol=3e-8)
    model.train(reset=False)
    torch.testing.assert_close(model(query).mean, model.forward(query).mean)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_eval_call_retains_actual_task_loader_likelihood_and_all_data_graphs(device, fixed):
    pytest.importorskip("gpytorch")
    from test_torch_fully_bayesian_multitask_loader import _dense_task_oracle, _models

    model, _, (X, Y, noise), program, _ = _models(device, 1, fixed=fixed, gradients=True)
    assert model.likelihood is program.loaded[2]
    query = X.new_tensor([[0.22, 3.0, 0.41], [0.63, 7.0, 0.71]], requires_grad=True)
    actual = model(model.transform_inputs(query))
    expected_mean, expected_covariance = _dense_task_oracle(model, query)
    torch.testing.assert_close(actual.mean, expected_mean.squeeze(-1), atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual.covariance_matrix, expected_covariance, atol=3e-12, rtol=3e-11
    )
    leaves = (X, Y, query, *((noise,) if fixed else ()), *model.parameters())

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
        derivatives(expected_mean.squeeze(-1), expected_covariance),
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-9, rtol=3e-8)
