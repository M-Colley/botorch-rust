# ruff: noqa: E402
"""Structured ExactGP calls distinguish priors and transformed conditional laws."""

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import KroneckerMultiTaskGP as ReferenceKronecker
from botorch.models.higher_order_gp import HigherOrderGP as ReferenceHigherOrder
from botorch.models.latent_kronecker_gp import LatentKroneckerGP as ReferenceLatent
from botorch.models.transforms.input import Normalize as ReferenceNormalize
from botorch.models.transforms.outcome import Standardize as ReferenceStandardize
from test_torch_direct_standardize import _compare_graphs

from botorch_rs.models import KroneckerMultiTaskGP, LatentKroneckerGP
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def pair(family, device, batch, normalize):
    x = torch.tensor([[0.2], [0.6], [1.0], [1.4]], dtype=torch.double, device=device)
    time = x.new_tensor([[0.0], [1.0]])
    y = torch.cat(((x * 2).sin() + 1.4, (x * 3).cos() + 0.5), -1)
    if batch:
        x, y = x.expand(2, 4, 1).clone(), y.expand(2, 4, 2).clone()
        y[1] = y[1] * 1.3 - 0.2
    x.requires_grad_()
    y.requires_grad_()
    time.requires_grad_()
    result = []
    for reference in (False, True):
        options = {} if reference else {"backend": "torch"}
        if normalize:
            options["input_transform"] = (ReferenceNormalize if reference else Normalize)(
                1, bounds=x.new_tensor([[0.0], [2.0]])
            )
        if family == "kronecker":
            options["outcome_transform"] = (ReferenceStandardize if reference else Standardize)(
                2, batch_shape=x.shape[:-2]
            )
            cls = ReferenceKronecker if reference else KroneckerMultiTaskGP
            model = cls(x, y, **options).to(x)
        elif family == "higher_order":
            cls = ReferenceHigherOrder if reference else HigherOrderGP
            model = cls(x, y, **options).to(x)
        else:
            cls = ReferenceLatent if reference else LatentKroneckerGP
            model = cls(x, time, y, **options).to(x)
        result.append(model)
    actual, source = result
    a, b = dict(actual.named_parameters()), dict(source.named_parameters())
    assert a.keys() == b.keys()
    with torch.no_grad():
        for name in a:
            a[name].copy_(b[name])
    return actual.eval(), source.eval(), x, y, time


def canonical(distribution, family):
    mean, covariance = distribution.mean, distribution.covariance_matrix
    if family == "kronecker":
        q, m = mean.shape[-2:]
        indices = torch.arange(q * m, device=mean.device).reshape(q, m).T.flatten()
        covariance = covariance.index_select(-2, indices).index_select(-1, indices)
    return mean, covariance


def kronecker_blocks(model, x, query):
    # A batched source MultitaskKernel repeats an already batched task factor,
    # so evaluate its actual data and task modules separately for this oracle.
    data = model.covar_module.data_covar_module(x, query).to_dense()
    task = model.covar_module.task_covar_module.covar_matrix.to_dense()
    return torch.einsum("...ij,...kl->...ikjl", data, task).reshape(
        *torch.broadcast_shapes(data.shape[:-2], task.shape[:-2]),
        data.shape[-2] * task.shape[-2],
        data.shape[-1] * task.shape[-1],
    )


def independent_kronecker_conditional(model, query):
    train = model.train_inputs[0]
    noise = model.likelihood._shaped_noise_covar(
        torch.Size((*train.shape[:-2], train.shape[-2], model.num_outputs))
    ).to_dense()
    observed = kronecker_blocks(model, train, train) + noise
    cross = kronecker_blocks(model, train, query)
    mean_train, mean_query = model.mean_module(train), model.mean_module(query)
    residual = (model.train_targets - mean_train).flatten(-2).unsqueeze(-1)
    mean = mean_query.flatten(-2) + (
        cross.transpose(-1, -2) @ torch.linalg.solve(observed, residual)
    ).squeeze(-1)
    covariance = kronecker_blocks(model, query, query) - cross.transpose(
        -1, -2
    ) @ torch.linalg.solve(observed, cross)
    mean = mean.reshape(*mean.shape[:-1], query.shape[-2], model.num_outputs)
    index = (
        torch.arange(mean.shape[-2] * mean.shape[-1], device=query.device)
        .reshape(mean.shape[-2], mean.shape[-1])
        .T.flatten()
    )
    return mean, covariance.index_select(-2, index).index_select(-1, index)


def higher_order_blocks(model, x, query):
    value = model.covar_modules[0](x, query).to_dense()
    for module, latent in zip(model.covar_modules[1:], model.latent_parameters, strict=True):
        other = module(latent, latent).to_dense()
        value = torch.einsum("...ij,...kl->...ikjl", value, other).reshape(
            *torch.broadcast_shapes(value.shape[:-2], other.shape[:-2]),
            value.shape[-2] * other.shape[-2],
            value.shape[-1] * other.shape[-1],
        )
    return value


def independent_higher_order_conditional(model, query):
    train = model.train_inputs[0]
    targets = model.train_targets
    covariance = higher_order_blocks(model, train, train)
    observed = covariance + torch.diag_embed(model.likelihood.noise.expand_as(targets))
    cross = higher_order_blocks(model, train, query)
    mean = (cross.transpose(-1, -2) @ torch.linalg.solve(observed, targets.unsqueeze(-1))).squeeze(
        -1
    )
    return mean, higher_order_blocks(model, query, query) - cross.transpose(
        -1, -2
    ) @ torch.linalg.solve(observed, cross)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kronecker", "higher_order", "latent"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_structured_eval_calls_use_transformed_coordinates_and_model_outcome_units(
    device, family, batch, normalize
):
    model, source, x, _, time = pair(family, device, batch, normalize)
    raw = x.new_tensor([[0.43], [1.13]])
    if batch:
        raw = raw.expand(3, 1, 2, 1).clone()
    else:
        raw = raw.expand(3, 2, 1).clone()
    raw.requires_grad_()
    query, reference_query = model.transform_inputs(raw), source.transform_inputs(raw)
    args = (time,) if family == "latent" else ()
    actual = model(query, *args)
    with gpytorch.settings.fast_pred_var(False), gpytorch.settings.detach_test_caches(False):
        if family in ("kronecker", "higher_order") and batch:
            with pytest.raises(RuntimeError, match="incompatible for a Kronecker product"):
                source(reference_query)
            oracle = (
                independent_kronecker_conditional
                if family == "kronecker"
                else independent_higher_order_conditional
            )
            expected_mean, expected_covariance = oracle(source, reference_query)
        else:
            reference = source(reference_query, *args)
            expected_mean, expected_covariance = canonical(reference, family)
        torch.testing.assert_close(actual.mean, expected_mean, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(
            actual.covariance_matrix, expected_covariance, atol=2e-7, rtol=2e-7
        )
        score = actual.mean.square().sum() + actual.covariance_matrix.square().sum()
        expected_score = expected_mean.square().sum() + expected_covariance.square().sum()
        gradient = torch.autograd.grad(score, raw, create_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected_score, raw)[0]
        torch.testing.assert_close(gradient, expected_gradient, atol=4e-6, rtol=4e-6)
        assert torch.isfinite(torch.autograd.grad(gradient.sum(), raw)[0]).all()

    # Calling forward in eval mode must still evaluate a prior. LatentKronecker
    # applies the observed training mask to the training prefix of this joint grid.
    train = model._training_values()[0]
    joint_batch = torch.broadcast_shapes(train.shape[:-2], query.shape[:-2])
    joint = torch.cat(
        (
            train.expand(*joint_batch, *train.shape[-2:]),
            query.expand(*joint_batch, *query.shape[-2:]),
        ),
        -2,
    )
    prior = model.forward(joint, *args)
    if family == "kronecker" and batch:
        expected_mean = source.mean_module(joint)
        covariance = kronecker_blocks(source, joint, joint)
        q, m = joint.shape[-2], source.num_outputs
        index = torch.arange(q * m, device=joint.device).reshape(q, m).T.flatten()
        expected_covariance = covariance.index_select(-2, index).index_select(-1, index)
    elif family == "higher_order" and batch:
        expected_covariance = higher_order_blocks(source, joint, joint)
        expected_mean = expected_covariance.new_zeros(expected_covariance.shape[:-1])
    else:
        reference_prior = source.forward(joint, *args)
        expected_mean, expected_covariance = canonical(reference_prior, family)
    torch.testing.assert_close(prior.mean, expected_mean, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(prior.covariance_matrix, expected_covariance, atol=2e-7, rtol=2e-7)

    posterior = model.posterior(raw, *args)
    means, stdvs = model.outcome_transform.means, model.outcome_transform.stdvs
    expected = actual.mean
    if family != "kronecker":
        expected = expected.reshape(*expected.shape[:-1], raw.shape[-2], -1)
    torch.testing.assert_close(posterior.mean, expected * stdvs + means, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
def test_public_flattened_standardize_preserves_noise_and_complete_training_graph(device, batch):
    from unittest.mock import patch

    from botorch.models.higher_order_gp import FlattenedStandardize as ReferenceFlattened

    from botorch_rs.models.higher_order_gp import FlattenedStandardize

    y = torch.linspace(0.2, 2.3, 24, device=device, dtype=torch.double).reshape(4, 2, 3)
    y = y.expand(*batch, 4, 2, 3).clone().requires_grad_()
    noise = (y.detach().square() * 0.02 + 0.06).requires_grad_()
    actual = FlattenedStandardize((2, 3), batch_shape=batch)
    source = ReferenceFlattened(torch.Size((2, 3)), batch_shape=torch.Size(batch)).to(y)
    with patch(
        "botorch_rs.models.higher_order_gp.to_flat", side_effect=AssertionError("host copy")
    ):
        transformed, transformed_noise = actual(y, noise)
        expected, expected_noise = source(y, noise)
        _compare_graphs(
            transformed.sin().sum() + transformed_noise.square().sum(),
            expected.sin().sum() + expected_noise.square().sum(),
            (y, noise),
        )
        actual.eval()
        source.eval()
        restored, restored_noise = actual.untransform(transformed, transformed_noise)
        torch.testing.assert_close(restored, y)
        torch.testing.assert_close(restored_noise, noise)

        mean = y.detach()[..., :2, :, :].clone().requires_grad_()
        variance = (mean.detach().square() * 0.03 + 0.04).requires_grad_()
        sample = mean.expand(3, *mean.shape).clone().requires_grad_()
        from types import SimpleNamespace

        base = SimpleNamespace(mean=mean, variance=variance)
        posterior = actual.untransform_posterior(base)
        reference = source.untransform_posterior(base)
        _compare_graphs(
            posterior.mean.square().sum()
            + posterior.variance.sin().sum()
            + posterior._sample_transform(sample).sin().sum(),
            reference.mean.square().sum()
            + reference.variance.sin().sum()
            + reference._sample_transform(sample).sin().sum(),
            (y, mean, variance, sample),
        )
        saved = posterior.mean.clone()
        actual.train()(y.detach() + 8)
        torch.testing.assert_close(posterior.mean, saved)
