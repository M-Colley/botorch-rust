"""Device contextual multitask inference and full data/embedding graphs."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models.contextual_multioutput import LCEMGP as ReferenceLCEM
from botorch.settings import propagate_grads

from botorch_rs.kernels import MaternKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ZeroMean
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.contextual_multioutput import LCEMGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device, task_feature, dtype=torch.double, labels=(0, 1)):
    torch.manual_seed(52)
    spatial = torch.rand(6, 2, dtype=dtype, device=device)
    task = torch.tensor(labels * 3, dtype=dtype, device=device).unsqueeze(-1)
    x = torch.cat([spatial[:, :task_feature], task, spatial[:, task_feature:]], -1).requires_grad_()
    y = torch.tensor(
        [[-0.7], [0.1], [0.9], [-0.4], [0.6], [-0.5]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    return x, y


def _copy(reference, port):
    params = dict(port.named_parameters())
    with torch.no_grad():
        for name, value in reference.named_parameters():
            value.copy_(params[name])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("task_feature", [0, 1, 2])
@pytest.mark.parametrize("fixed", [False, True])
def test_reference_moments_parameter_and_query_gradients(device, task_feature, fixed):
    x, y = _data(device, task_feature)
    noise = torch.full_like(y, 0.04) if fixed else None
    kwargs = dict(
        context_cat_feature=torch.tensor([[0, 1], [1, 0]], device=device),
        embs_dim_list=[1, 2],
        outcome_transform=None,
    )
    model = LCEMGP(x, y, task_feature, noise, backend="torch", seed=13, **kwargs)
    with torch.no_grad():
        for layer in model.emb_layers:
            layer.weight.copy_(
                torch.linspace(-0.2, 0.3, layer.weight.numel(), device=device).reshape_as(
                    layer.weight
                )
            )
    reference = ReferenceLCEM(x.detach(), y.detach(), task_feature, noise, **kwargs).to(x)
    _copy(reference, model)
    query = torch.rand(2, 3, 2, dtype=x.dtype, device=device, requires_grad=True)
    actual = model.posterior(query)
    with propagate_grads(True), gpytorch.settings.detach_test_caches(False):
        expected = reference.posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean, atol=2e-10, rtol=2e-9)
    # Align coordinate layouts explicitly; pinned long-format multitask
    # posteriors currently use output-major blocks, like the Tensor law.
    q = query.shape[-2]
    permutation = torch.arange(q * 2, device=device).reshape(q, 2).T.flatten()
    ref_cov = expected.distribution.covariance_matrix
    if expected.distribution._interleaved:
        ref_cov = ref_cov.index_select(-1, permutation).index_select(-2, permutation)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, ref_cov, atol=3e-10, rtol=2e-9
    )
    params = dict(reference.named_parameters())
    left = (query, *model.parameters())
    right = (query, *(params[name] for name, _ in model.named_parameters()))
    actual_grads = torch.autograd.grad(
        actual.mean.square().sum() + actual.variance.sum(), left, retain_graph=True
    )
    expected_grads = torch.autograd.grad(
        expected.mean.square().sum() + expected.variance.sum(), right
    )
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-7)
    model.train()
    reference.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    ref_mll = gpytorch.mlls.ExactMarginalLogLikelihood(reference.likelihood, reference)
    actual_value = mll(model(*model.train_inputs), model.train_targets)
    expected_value = ref_mll(
        reference(*reference.train_inputs), reference.train_targets, reference.train_inputs
    )
    torch.testing.assert_close(actual_value, expected_value, atol=3e-8, rtol=3e-8)
    grad = torch.autograd.grad(actual_value, tuple(model.parameters()))
    ref_grad = torch.autograd.grad(
        expected_value, tuple(params[name] for name, _ in model.named_parameters())
    )
    for a, b in zip(grad, ref_grad, strict=True):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-7)


def _rbf(a, b, lengthscale):
    return (
        -0.5 * ((a / lengthscale).unsqueeze(-2) - (b / lengthscale).unsqueeze(-3)).square().sum(-1)
    ).exp()


def _manual(model, query):
    # Independent Gaussian conditioning and context covariance, with no
    # call to model._covariance or native factor/predict.
    query_full, outputs, q = model._query_inputs(query, None)
    train = model._mapped_inputs(model._train_X)
    test = model._mapped_inputs(query_full)
    embeddings = torch.cat(
        [
            layer.weight.index_select(0, model._context_category_indices[:, i])
            for i, layer in enumerate(model.emb_layers)
        ]
        + ([] if model.context_emb_feature is None else [model.context_emb_feature]),
        -1,
    )
    context = _rbf(embeddings, embeddings, model.task_covar_module_base.lengthscale)

    def covariance(a, b):
        space = _rbf(
            a.index_select(-1, model._base_idxr),
            b.index_select(-1, model._base_idxr),
            model.covar_module.lengthscale,
        )
        return (
            space
            * context[
                a[..., model._task_feature].long().unsqueeze(-1),
                b[..., model._task_feature].long().unsqueeze(-2),
            ]
        )

    kernel = covariance(train, train)
    cross = covariance(train, test)
    noise = model._train_Yvar.squeeze(-1)
    factor = torch.linalg.cholesky(kernel + noise.diag_embed())
    centered = model._train_Y.squeeze(-1) - model._mean(train)
    alpha = torch.cholesky_solve(centered.unsqueeze(-1), factor)
    mean = (cross.transpose(-1, -2) @ alpha).squeeze(-1) + model._mean(test)
    cov = covariance(test, test) - cross.transpose(-1, -2) @ torch.cholesky_solve(cross, factor)
    return mean.reshape(outputs, q).T, cov


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("labels", [(0, 1), (3, 8)])
def test_fixed_embedding_data_parameter_hvp_and_repeated_graphs(device, labels):
    x, y = _data(device, 1, labels=labels)
    noise = torch.full_like(y, 0.05, requires_grad=True)
    features = torch.tensor(
        [[0.1, 0.4], [-0.3, 0.6]], dtype=x.dtype, device=device, requires_grad=True
    )
    categories = torch.tensor([[5, 20], [12, 5]], device=device)
    model = LCEMGP(
        x,
        y,
        1,
        noise,
        context_cat_feature=categories,
        context_emb_feature=features,
        embs_dim_list=[1, 2],
        outcome_transform=None,
        backend="torch",
        seed=11,
    )
    with torch.no_grad():
        for layer in model.emb_layers:
            layer.weight.mul_(0.1)
    query = torch.rand(2, 2, dtype=x.dtype, device=device, requires_grad=True)
    assert model.task_covar_module_base.ard_num_dims == 5
    leaves = (x, y, noise, features, query, *model.parameters())
    for _ in range(2):
        actual = model.posterior(query)
        mean, covariance = _manual(model, query)
        torch.testing.assert_close(actual.mean, mean, atol=3e-12, rtol=2e-11)
        torch.testing.assert_close(
            actual.distribution.covariance_matrix, covariance, atol=3e-12, rtol=2e-11
        )
        objective = (
            actual.mean.square().sum() + actual.distribution.covariance_matrix.square().sum()
        )
        expected = mean.square().sum() + covariance.square().sum()
        first = torch.autograd.grad(objective, leaves, create_graph=True, retain_graph=True)
        ref_first = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
        for a, b in zip(first, ref_first, strict=True):
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), leaves, retain_graph=True
        )
        ref_second = torch.autograd.grad(sum(g.square().sum() for g in ref_first), leaves)
        for a, b in zip(second, ref_second, strict=True):
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-7)
        assert torch.equal(first[0][:, 1], torch.zeros_like(x[:, 1]))
        assert first[3].abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_transforms_fitting_unobserved_contexts_and_fantasies(device):
    x, y = _data(device, 2, torch.float32, labels=(2, 7))
    model = LCEMGP(
        x,
        y,
        2,
        all_tasks=[2, 7, 9],
        context_cat_feature=torch.tensor([[0], [1], [2]], device=device),
        context_emb_feature=torch.tensor([[0.1], [0.3], [0.7]], device=device),
        input_transform=Normalize(3, indices=[0, 1]),
        outcome_transform=Standardize(1),
        seed=9,
        backend="torch",
    )
    query = torch.rand(2, 2, dtype=x.dtype, device=device, requires_grad=True)
    posterior = model.posterior(query)
    assert posterior.mean.shape == (2, 3)
    selected = model.posterior(query, output_indices=[9])
    torch.testing.assert_close(selected.mean, posterior.mean[..., 2:3])
    torch.testing.assert_close(model._eval_context_covar().diagonal(), torch.ones(3, device=device))
    indices = torch.tensor([[[0], [2]], [[1], [0]]], device=device)
    cov = model.task_covar_module(indices)
    assert cov.shape == (2, 2, 2)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.002)
    before = model.emb_layers[0].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        model.train()
        loss = -mll(model(*model.train_inputs), model.train_targets).sum()
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, model.emb_layers[0].weight)
    # Conditioning retains full-label inputs and supports fantasy batches.
    full = torch.cat([query, query.new_full((2, 1), 9)], -1)
    conditioned = model.condition_on_observations(full, y[:2])
    assert conditioned.posterior(query).mean.shape == (2, 3)
    model.task_covar_module_base = MaternKernel(nu=2.5, ard_num_dims=2)
    assert model.task_covar_module_base.kind == "matern"
    assert torch.isfinite(model.posterior(query).mean).all()
    with pytest.raises(IndexError, match="indices"):
        model.task_covar_module(torch.tensor([[3]], device=device))


def test_configuration_validation_and_seed_isolation():
    x, y = _data("cpu", 2)
    kwargs = dict(backend="torch", seed=919)
    left, right = LCEMGP(x, y, 2, **kwargs), LCEMGP(x, y, 2, **kwargs)
    torch.testing.assert_close(left.emb_layers[0].weight, right.emb_layers[0].weight)
    with pytest.raises(ValueError, match="positive dimension"):
        LCEMGP(x, y, 2, embs_dim_list=[0], **kwargs)
    with pytest.raises(ValueError, match="finite integers"):
        LCEMGP(x, y, 2, context_cat_feature=torch.tensor([[0.2], [1.0]]), **kwargs)
    with pytest.raises(ValueError, match="num_tasks"):
        LCEMGP(x, y, 2, context_emb_feature=torch.zeros(3, 1), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_modules_real_bounded_fit_and_state_reload(device):
    x, y = _data(device, 1)
    kwargs = dict(
        mean_module=ZeroMean(),
        covar_module=MaternKernel(nu=2.5, ard_num_dims=2),
        likelihood=GaussianLikelihood(),
        input_transform=Normalize(3, indices=[0, 2]),
        outcome_transform=None,
        seed=23,
        backend="torch",
    )
    model = LCEMGP(x, y, 1, **kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = model.task_covar_module_base.raw_lengthscale.detach().clone()
    result = fit_gpytorch_mll_scipy(mll, options={"maxiter": 2})
    assert result.step > 0
    assert not torch.equal(before, model.task_covar_module_base.raw_lengthscale)
    assert bool(
        (
            (model.task_covar_module_base.lengthscale >= 0)
            & (model.task_covar_module_base.lengthscale <= 2)
        ).all()
    )
    clone = LCEMGP(x.detach().clone(), y.detach().clone(), 1, **kwargs)
    clone.load_state_dict(model.state_dict())
    query = torch.rand(2, 2, dtype=x.dtype, device=device, requires_grad=True)
    torch.testing.assert_close(clone.posterior(query).mean, model.posterior(query).mean)
    selected = model.posterior(query, output_indices=[1], observation_noise=True)
    gradient = torch.autograd.grad(
        selected.mean.sum() + selected.variance.sum(), (query, x, y), create_graph=True
    )
    assert all(torch.isfinite(value).all() for value in gradient)
