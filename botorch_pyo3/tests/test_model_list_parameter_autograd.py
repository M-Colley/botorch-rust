"""Joint GP-list native parameter graphs and true child batch broadcasting."""

from __future__ import annotations

import inspect

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtScalarize  # noqa: E402
from botorch.models import ModelListGP as BtList  # noqa: E402
from botorch.models import SingleTaskGP as BtGP  # noqa: E402
from botorch.posteriors.gpytorch import GPyTorchPosterior as BtPosterior  # noqa: E402
from botorch.settings import propagate_grads  # noqa: E402
from gpytorch.distributions import MultitaskMultivariateNormal  # noqa: E402
from gpytorch.settings import fast_pred_var  # noqa: E402

from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform  # noqa: E402
from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.models import ModelListGP, SingleTaskGP  # noqa: E402


def model_pair(batches=((), ())):
    actual, expected = [], []
    generator = torch.Generator().manual_seed(4412)
    for i, batch in enumerate(batches):
        n, m = (5, 2) if i == 0 else (7, 1)
        x = torch.rand(*batch, n, 2, generator=generator, dtype=torch.double)
        y = torch.rand(*batch, n, m, generator=generator, dtype=torch.double)
        noise = torch.full_like(y, 0.015) if i == 1 else None
        actual.append(SingleTaskGP(x, y, noise))
        expected.append(BtGP(x, y, noise))
    return ModelListGP(*actual), BtList(*expected)


def reference_posterior(
    model, query, *, observation_noise=False, output_indices=None, posterior_transform=None
):
    """Corrected Gaussian conditional from actual official module state.

    Official mixed multioutput ModelListGP currently gets incorrect task blocks
    from lazy BlockDiag slicing; some augmented GP batches also fail in ExactGP
    cache reshaping. Use the same exact conditional with official kernel/mean/
    noise/outcome modules, retaining their parameter graph and full training solve.
    A separate regression verifies this against ordinary official child posteriors.
    """
    batch = torch.broadcast_shapes(query.shape[:-2], *(child.batch_shape for child in model.models))
    q, d = query.shape[-2:]
    points = query.expand(*batch, q, d)
    means, covariances = [], []
    for child in model.models:
        child.eval()
        train = child.train_inputs[0]
        outputs = child.num_outputs
        target = child.train_targets
        x = points.unsqueeze(-3) if outputs > 1 else points
        k = child.covar_module(train).to_dense() + torch.diag_embed(
            child.likelihood.noise.expand_as(target)
        )
        cross = child.covar_module(train, x).to_dense()
        covariance = child.covar_module(x).to_dense()
        factor = torch.linalg.cholesky(k)
        alpha = torch.cholesky_solve((target - child.mean_module(train)).unsqueeze(-1), factor)
        mean = child.mean_module(x) + (cross.transpose(-2, -1) @ alpha).squeeze(-1)
        covariance = covariance - cross.transpose(-2, -1) @ torch.cholesky_solve(cross, factor)
        if observation_noise:
            noise = child.likelihood.noise
            if type(child.likelihood).__name__ == "FixedNoiseGaussianLikelihood":
                noise = noise.mean(-1, keepdim=True)
            covariance = covariance + torch.diag_embed(noise.expand_as(mean))
        if outputs > 1:
            mean = mean.transpose(-2, -1)
        else:
            mean = mean.unsqueeze(-1)
            covariance = covariance.unsqueeze(-3)
        transform = child.outcome_transform
        mean = mean * transform.stdvs + transform.means
        covariance = covariance * transform.stdvs.squeeze(-2)[..., :, None, None].square()
        means.append(mean)
        covariances.extend(covariance[..., t, :, :] for t in range(outputs))
    mean = torch.cat(means, dim=-1)
    m = mean.shape[-1]
    covariance = mean.new_zeros(*batch, q * m, q * m)
    for t, block in enumerate(covariances):
        covariance[..., t * q : (t + 1) * q, t * q : (t + 1) * q] = block
    if output_indices is not None:
        mean = mean[..., output_indices]
        indices = torch.tensor([t * q + p for t in output_indices for p in range(q)])
        covariance = covariance.index_select(-2, indices).index_select(-1, indices)
    posterior = BtPosterior(MultitaskMultivariateNormal(mean, covariance, interleaved=False))
    return posterior if posterior_transform is None else posterior_transform(posterior)


@pytest.mark.parametrize("noisy", [False, True])
def test_corrected_reference_matches_official_individual_conditionals(noisy):
    _, reference = model_pair()
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double)
    for child in reference.models:
        with propagate_grads(True), fast_pred_var(False):
            ordinary = child.posterior(query, observation_noise=noisy)
            corrected = reference_posterior(BtList(child), query, observation_noise=noisy)
        torch.testing.assert_close(corrected.mean, ordinary.mean, rtol=2e-9, atol=2e-10)
        torch.testing.assert_close(
            corrected.distribution.covariance_matrix,
            ordinary.distribution.covariance_matrix,
            rtol=2e-9,
            atol=2e-10,
        )
        actual = corrected.mean.square().sum() + corrected.distribution.covariance_matrix.sum()
        expected = ordinary.mean.square().sum() + ordinary.distribution.covariance_matrix.sum()
        actual_grads = torch.autograd.grad(actual, tuple(child.parameters()))
        expected_grads = torch.autograd.grad(expected, tuple(child.parameters()))
        for result, want in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(result, want, rtol=2e-8, atol=2e-9)


def test_official_dense_task_block_workaround_preserves_joint_covariance(monkeypatch):
    import botorch.models.gpytorch as reference_models
    import botorch.utils.multitask as reference_multitask

    _, reference = model_pair()
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double)
    with propagate_grads(True), fast_pred_var(False):
        expected = reference_posterior(reference, query, observation_noise=True)
        unpatched = reference.posterior(query, observation_noise=True)
    assert not torch.allclose(
        unpatched.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )
    namespace = dict(vars(reference_multitask))
    source = inspect.getsource(reference_multitask.separate_mtmvn).replace(
        "full_covar = mvn.lazy_covariance_matrix", "full_covar = mvn.covariance_matrix"
    )
    exec(compile(source, "<dense-task-block-workaround>", "exec"), namespace)
    monkeypatch.setattr(reference_models, "separate_mtmvn", namespace["separate_mtmvn"])
    with propagate_grads(True), fast_pred_var(False):
        fixed = reference.posterior(query, observation_noise=True)
    torch.testing.assert_close(fixed.mean, expected.mean)
    torch.testing.assert_close(
        fixed.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )
    actual_gradients = torch.autograd.grad(
        fixed.distribution.covariance_matrix.sum() + fixed.mean.sum(), tuple(reference.parameters())
    )
    expected_gradients = torch.autograd.grad(
        expected.distribution.covariance_matrix.sum() + expected.mean.sum(),
        tuple(reference.parameters()),
    )
    for result, want in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, atol=5e-8, rtol=5e-7)


@pytest.mark.parametrize("batches", [((), ()), ((2, 1), (3,))])
@pytest.mark.parametrize("mode", ["mean", "variance", "covariance", "samples"])
@pytest.mark.parametrize("view", ["all", "select", "scalar"])
def test_joint_posterior_parameter_and_query_gradients_match_official(batches, mode, view):
    model, reference = model_pair(batches)
    query = torch.tensor([0.23, 0.61, 0.71, 0.32], dtype=torch.double).reshape(1, 1, 2, 2)
    query = query.expand(4, 1, 1, 2, 2).clone().requires_grad_()
    expected_query = query.detach().clone().requires_grad_()
    kwargs, expected_kwargs = {}, {}
    if view == "select":
        kwargs["output_indices"] = expected_kwargs["output_indices"] = [2, 0]
    elif view == "scalar":
        weights = torch.tensor([0.3, -0.6, 0.4], dtype=torch.double)
        kwargs["posterior_transform"] = ScalarizedPosteriorTransform(weights, offset=0.2)
        expected_kwargs["posterior_transform"] = BtScalarize(weights, offset=0.2)
    with propagate_grads(True), fast_pred_var(False):
        expected = reference_posterior(
            reference, expected_query, observation_noise=True, **expected_kwargs
        )
    actual = model.posterior(query, observation_noise=True, **kwargs)
    if mode == "samples":
        shape = (2, 3, *expected.base_sample_shape)
        normals = torch.linspace(-0.7, 0.8, torch.Size(shape).numel(), dtype=torch.double).reshape(
            shape
        )
        value = actual.rsample_from_base_samples(torch.Size([2, 3]), normals)
        want = expected.rsample_from_base_samples(torch.Size([2, 3]), normals)
    elif mode == "covariance":
        value, want = actual.distribution.covariance_matrix, expected.distribution.covariance_matrix
    else:
        value, want = getattr(actual, mode), getattr(expected, mode)
    torch.testing.assert_close(value, want, rtol=3e-7, atol=3e-8)
    weights = torch.linspace(-0.3, 0.7, value.numel(), dtype=torch.double).reshape(value.shape)
    gradients = torch.autograd.grad((value * weights).sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        (want * weights).sum(),
        (*reference.parameters(), expected_query),
        allow_unused=True,
        materialize_grads=True,
    )
    for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, expected_gradient, rtol=6e-7, atol=6e-8)


@pytest.mark.parametrize("mode", ["mean", "samples"])
def test_parent_broadcast_dimensions_are_reduced_into_original_child_leaves(mode):
    model, _ = model_pair(((2, 1), (3,)))
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double)
    post = model.posterior(query)
    assert tuple(post.mean.shape) == (2, 3, 2, 3)
    if mode == "mean":
        value = post.mean
    else:
        z = torch.linspace(-0.5, 0.9, 2 * 2 * 3 * 2 * 3, dtype=torch.double).reshape(2, 2, 3, 2, 3)
        value = post.rsample_from_base_samples(torch.Size([2]), z)
    gradients = torch.autograd.grad(value.sum(), tuple(model.parameters()))
    direction = [torch.full_like(p, 0.01) for p in model.parameters()]
    analytic = sum((g * d).sum().item() for g, d in zip(gradients, direction, strict=True))
    originals = [p.detach().clone() for p in model.parameters()]

    def evaluate(sign):
        with torch.no_grad():
            for p, initial, d in zip(model.parameters(), originals, direction, strict=True):
                p.copy_(initial + sign * 1e-5 * d)
            posterior = model.posterior(query)
            values = (
                posterior.mean
                if mode == "mean"
                else posterior.rsample_from_base_samples(torch.Size([2]), z)
            )
            return values.sum().item()

    finite = (evaluate(1) - evaluate(-1)) / 2e-5
    assert analytic == pytest.approx(finite, rel=5e-6, abs=5e-7)


def test_child_source_and_parameter_graph_snapshots_are_immutable():
    model, _ = model_pair()
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double)
    post = model.posterior(query)
    mean = post.mean
    expected = torch.autograd.grad(mean.sum(), tuple(model.parameters()), retain_graph=True)
    for child in model.models:
        child.mean_module.constant = 0.8
        tuple(child.parameters())
    actual = torch.autograd.grad(mean.sum(), tuple(model.parameters()))
    torch.testing.assert_close(post.mean, mean)
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)


def test_aliased_child_leaves_accumulate_both_output_group_contributions():
    model, reference = model_pair()
    child, ref_child = model.models[0], reference.models[0]
    aliased, expected = ModelListGP(child, child), BtList(ref_child, ref_child)
    assert len(tuple(aliased.parameters())) == len(tuple(child.parameters()))
    assert len(tuple(aliased.named_parameters(remove_duplicate=False))) == 2 * len(
        tuple(child.parameters())
    )
    assert not tuple(aliased.parameters(recurse=False))
    assert all(
        name.startswith("outer.models.0.") for name, _ in aliased.named_parameters(prefix="outer")
    )
    query = torch.tensor([[0.2, 0.6]], dtype=torch.double)
    with propagate_grads(True), fast_pred_var(False):
        want = expected.posterior(query).mean
    value = aliased.posterior(query).mean
    gradients = torch.autograd.grad(value.square().sum(), tuple(child.parameters()))
    expected_gradients = torch.autograd.grad(want.square().sum(), tuple(ref_child.parameters()))
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=6e-7, atol=6e-8)


def test_unsupported_child_parameter_backward_explicit_but_query_grad_works():
    class SpecializedGP(SingleTaskGP):
        pass

    model, _ = model_pair()
    child = model.models[1]
    specialized = SpecializedGP(child.train_inputs[0], child.train_targets.unsqueeze(-1))
    combined = ModelListGP(model.models[0], specialized)
    query = torch.tensor([[0.2, 0.6]], dtype=torch.double, requires_grad=True)
    value = combined.posterior(query).mean
    assert torch.isfinite(torch.autograd.grad(value.sum(), query, retain_graph=True)[0]).all()
    with pytest.raises(UnsupportedError, match="Unsupported child: models.1"):
        value.sum().backward()
