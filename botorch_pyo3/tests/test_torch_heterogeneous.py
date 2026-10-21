# ruff: noqa: E402
"""Device transfer, Gaussian conditioning and complete heterogeneous graphs."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.kernels import ScaleKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.heterogeneous_mtgp import HeterogeneousMTGP
from botorch_rs.models.kernels.heterogeneous_multitask import (
    CombinatorialCovarModule,
    DeltaKernel,
    MultiTaskConditionalKernel,
)
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FEATURES = [[2, 0], [0, 1], [1, 2]]


@pytest.mark.parametrize("device", DEVICES)
def test_public_auxiliary_kernels_and_native_conversion(device):
    pytest.importorskip("botorch")
    from botorch.models.kernels.heterogeneous_multitask import CombinatorialCovarModule as Reference

    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    delta = DeltaKernel(active_dims=[0], backend="torch").double().to(device)
    expected_delta = ((x[:, :1] == 1) & (x[:, :1].transpose(-2, -1) == 1)).to(x)
    torch.testing.assert_close(delta(x), expected_delta)
    torch.testing.assert_close(delta(x, diag=True), expected_delta.diagonal())
    combinatorial = CombinatorialCovarModule(2, backend="torch").double().to(device)
    reference = Reference(2).double().to(device)
    with torch.no_grad():
        for a, b in zip(combinatorial.parameters(), reference.parameters(), strict=True):
            b.copy_(a)
    torch.testing.assert_close(combinatorial(x), reference(x).to_dense())
    full = torch.cat(
        (x, x[:, :1], torch.tensor([[0.0], [1.0], [2.0]], device=device, dtype=x.dtype)), -1
    )
    source = MultiTaskConditionalKernel(FEATURES)
    model = SingleTaskGP(
        full, x[:, :1], covar_module=ScaleKernel(source), outcome_transform=None, backend="torch"
    )
    assert isinstance(model.covar_module, torch.nn.Module)
    assert torch.isfinite(model.posterior(full).variance).all()


def training(device, requires_grad=False):
    Xs = [
        torch.tensor([[0.2, 0.1], [0.7, 0.8], [0.45, 0.4]], dtype=torch.double, device=device)
        .add(i * 0.05)
        .requires_grad_(requires_grad)
        for i in range(3)
    ]
    Ys = [X.detach().sum(-1, keepdim=True).sin().requires_grad_(requires_grad) for X in Xs]
    return Xs, Ys


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_official_model_values_likelihood_and_query_derivatives(device, fixed):
    pytest.importorskip("botorch")
    from botorch.models.heterogeneous_mtgp import HeterogeneousMTGP as Reference
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from botorch.models.transforms.outcome import Standardize as ReferenceStandardize

    pytest.importorskip("gpytorch")
    from gpytorch.mlls import ExactMarginalLogLikelihood as ReferenceMLL

    Xs, Ys = training(device)
    variances = [torch.full_like(Y, 0.03) for Y in Ys] if fixed else None
    actual = HeterogeneousMTGP(
        Xs,
        Ys,
        variances,
        FEATURES,
        3,
        input_transform=Normalize(4, indices=[0, 1, 2]),
        outcome_transform=Standardize(1),
        backend="torch",
    )
    expected = Reference(
        Xs,
        Ys,
        variances,
        FEATURES,
        3,
        input_transform=ReferenceNormalize(4, indices=[0, 1, 2]),
        outcome_transform=ReferenceStandardize(1),
    ).to(Xs[0])
    with torch.no_grad():
        for model in (actual, expected):
            for name, parameter in model.named_parameters():
                parameter.fill_(-2.0 if name.endswith("raw_tau") else 0.23)
    a = ExactMarginalLogLikelihood(actual.likelihood, actual)()
    b = ReferenceMLL(expected.likelihood, expected)(
        expected(*expected.train_inputs), expected.train_targets, *expected.train_inputs
    )
    torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-7)
    query = Xs[0].new_tensor([[[0.35, 0.25, 0.0], [0.65, 0.6, 0.0]]], requires_grad=True)
    for noise in (False, True):
        ap, bp = (
            actual.posterior(query, observation_noise=noise),
            expected.posterior(query, observation_noise=noise),
        )
        torch.testing.assert_close(ap.mean, bp.mean)
        torch.testing.assert_close(
            ap.distribution.covariance_matrix, bp.distribution.covariance_matrix
        )
        ag = torch.autograd.grad(ap.mean.sum() + ap.variance.sum(), query)[0]
        bg = torch.autograd.grad(bp.mean.sum() + bp.variance.sum(), query)[0]
        torch.testing.assert_close(ag, bg)


@pytest.mark.parametrize("device", DEVICES)
def test_data_query_gradgradcheck(device):
    X = torch.tensor([[0.15], [0.65]], dtype=torch.double, device=device, requires_grad=True)
    Y = torch.tensor([[0.3], [0.5]], dtype=torch.double, device=device, requires_grad=True)
    query = torch.tensor([[0.35]], dtype=torch.double, device=device, requires_grad=True)
    other_X, other_Y = X.detach().clone() + 0.1, Y.detach().clone() + 0.1

    def moments(data, targets, test):
        torch.manual_seed(83)
        model = HeterogeneousMTGP(
            [data, other_X],
            [targets, other_Y],
            None,
            [[1], [0]],
            2,
            use_saas_prior=False,
            backend="torch",
        )
        public = torch.cat((test, torch.zeros_like(test)), -1)
        p = model.posterior(public)
        return p.mean + p.variance

    assert torch.autograd.gradcheck(moments, (X, Y, query), fast_mode=True)
    assert torch.autograd.gradgradcheck(moments, (X, Y, query), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
def test_no_host_conversion_and_source_migration(device, monkeypatch):
    Xs, Ys = training(device, True)
    model = HeterogeneousMTGP(Xs, Ys, None, FEATURES, 3)
    assert model._tensor_backend == "torch"
    query = Xs[0].new_tensor([[0.35, 0.25, 0.0]], requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Heterogeneous algebra crossed a host-array boundary")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        posterior = model.posterior(query)
        first = torch.autograd.grad(
            posterior.mean.sum() + posterior.variance.sum(), (*Xs, query), create_graph=True
        )
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), (*Xs, query), allow_unused=True
        )
        assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))
    model.float().cpu()
    assert model.posterior(query.float().cpu()).mean.dtype == torch.float32


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("saas", [False, True])
@pytest.mark.parametrize("combinatorial", [False, True])
def test_conditional_kernel_values_gradients_and_diagonals(device, saas, combinatorial):
    pytest.importorskip("botorch")
    from botorch.models.kernels.heterogeneous_multitask import (
        MultiTaskConditionalKernel as Reference,
    )

    actual = (
        MultiTaskConditionalKernel(
            FEATURES, use_saas_prior=saas, use_combinatorial_kernel=combinatorial, backend="torch"
        )
        .double()
        .to(device)
    )
    expected = (
        Reference(FEATURES, use_saas_prior=saas, use_combinatorial_kernel=combinatorial)
        .double()
        .to(device)
    )
    with torch.no_grad():
        for a, b in zip(actual.parameters(), expected.parameters(), strict=True):
            b.copy_(a)
    X = torch.tensor(
        [[0.2, 0.3, 0.5, 0.0], [0.7, 0.8, 0.4, 1.0], [0.4, 0.5, 0.9, 2.0]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    Y = X.detach().flip(-2)
    value, reference = actual(X, Y), expected(X, Y).to_dense()
    torch.testing.assert_close(value, reference)
    torch.testing.assert_close(actual(X, Y, diag=True), reference.diagonal(dim1=-2, dim2=-1))
    gradient = torch.autograd.grad(value.sum(), X, create_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(reference.sum(), X)[0])
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), X)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_model_full_graph_repeated_backwards_and_transforms(device, fixed):
    Xs, Ys = training(device, True)
    variances = [torch.full_like(Y, 0.03, requires_grad=True) for Y in Ys] if fixed else None
    model = HeterogeneousMTGP(
        Xs,
        Ys,
        variances,
        FEATURES,
        3,
        input_transform=Normalize(4, indices=[0, 1, 2]),
        outcome_transform=Standardize(1),
        backend="torch",
    )
    query = Xs[0].new_tensor([[0.35, 0.25, 0.0], [0.65, 0.6, 0.0]], requires_grad=True)
    inputs = (*Xs, *Ys, query, *(variances or []), *model.parameters())
    for _ in range(2):
        posterior = model.posterior(query, observation_noise=True)
        loss = (
            posterior.mean.sum()
            + posterior.variance.sum()
            + ExactMarginalLogLikelihood(model.likelihood, model)()
        )
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), inputs, allow_unused=True
        )
        assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_conditioning_uses_target_embedding_and_fantasy_batches(device, fixed):
    Xs, Ys = training(device)
    variances = [torch.full_like(Y, 0.03) for Y in Ys] if fixed else None
    model = HeterogeneousMTGP(Xs, Ys, variances, FEATURES, 3, use_saas_prior=False, backend="torch")
    observed = Xs[0].new_tensor([[0.35, 0.25, 0.0]])
    query = Xs[0].new_tensor([[0.65, 0.6, 0.0], [0.5, 0.3, 0.0]], requires_grad=True)
    target = Ys[0].new_tensor([[0.25]], requires_grad=True)
    joint = model.posterior(torch.cat((observed, query)))
    mean, covariance = joint.mean.squeeze(-1), joint.distribution.covariance_matrix
    noise = (
        Ys[0].new_tensor([[0.03]])
        if fixed
        else model._noise_at(model._target_full_inputs(observed)).unsqueeze(-1)
    )
    gain = covariance[1:, :1] / (covariance[:1, :1] + noise)
    expected_mean = mean[1:] + gain @ (target.flatten() - mean[:1])
    expected_covariance = covariance[1:, 1:] - gain @ covariance[:1, 1:]
    conditioned = model.condition_on_observations(observed, target, noise=noise if fixed else None)
    for _ in range(2):
        actual = conditioned.posterior(query)
        torch.testing.assert_close(actual.mean.squeeze(-1), expected_mean)
        torch.testing.assert_close(actual.distribution.covariance_matrix, expected_covariance)
    fantasy = model.fantasize(
        observed, IIDNormalSampler(torch.Size([2, 3]), seed=41), propagate_grads=True
    )
    assert fantasy.posterior(query).mean.shape == (2, 3, 2, 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_conditioned_data_noise_and_parameter_graphs_repeat(device, fixed):
    Xs, Ys = training(device, True)
    variances = [torch.full_like(Y, 0.03, requires_grad=True) for Y in Ys] if fixed else None
    model = HeterogeneousMTGP(
        Xs,
        Ys,
        variances,
        FEATURES,
        3,
        input_transform=Normalize(4, indices=[0, 1, 2]),
        outcome_transform=Standardize(1),
        backend="torch",
    )
    observed = Xs[0].new_tensor([[0.35, 0.25, 0.0]], requires_grad=True)
    target = Ys[0].new_tensor([[0.25]], requires_grad=True)
    noise = Ys[0].new_tensor([[0.04]], requires_grad=True) if fixed else None
    query = Xs[0].new_tensor([[0.65, 0.6, 0.0]], requires_grad=True)
    conditioned = model.condition_on_observations(observed, target, noise=noise)
    original, child = dict(model.named_parameters()), dict(conditioned.named_parameters())
    assert original.keys() == child.keys()
    assert all(child[name] is not original[name] for name in original)
    inputs = (
        *Xs,
        *Ys,
        observed,
        target,
        query,
        *(variances or []),
        *((noise,) if fixed else ()),
        *child.values(),
    )
    for _ in range(2):
        p = conditioned.posterior(query)
        loss = (
            p.mean.sum()
            + p.variance.sum()
            + ExactMarginalLogLikelihood(conditioned.likelihood, conditioned)()
        )
        assert torch.autograd.grad(
            loss, tuple(original.values()), allow_unused=True, retain_graph=True
        ) == (None,) * len(original)
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), inputs)
        assert all(torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_source_heterogeneous_conditioning_owns_new_parameter_leaves(device):
    source = pytest.importorskip("botorch.models.heterogeneous_mtgp")
    Xs, Ys = training(device)
    model = source.HeterogeneousMTGP(Xs, Ys, None, FEATURES, 3, outcome_transform=None).to(Xs[0])
    query = Xs[0].new_tensor([[0.65, 0.6, 0.0]])
    model.posterior(query)
    # The inherited source conditioning method accepts the full feature space.
    observed = model.map_to_full_tensor(Xs[0].new_tensor([[0.35, 0.25]]), task_index=0)
    child = model.condition_on_observations(observed, Ys[0].new_tensor([[0.25]]))
    before, after = dict(model.named_parameters()), dict(child.named_parameters())
    assert before.keys() == after.keys()
    assert all(before[name] is not after[name] for name in before)
    child.prediction_strategy = None
    posterior = child.posterior(query)
    value = posterior.mean.square().sum() + posterior.variance.square().sum()
    assert torch.autograd.grad(
        value, tuple(before.values()), allow_unused=True, retain_graph=True
    ) == (None,) * len(before)
    # raw_tau enters the source SAAS prior rather than its covariance formula.
    names = [
        name
        for name in after
        if name.startswith(("covar_module.", "mean_module.")) and not name.endswith("raw_tau")
    ]
    assert all(
        torch.isfinite(item).all()
        for item in torch.autograd.grad(value, tuple(after[name] for name in names))
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("empty_task", [0, 1])
def test_empty_source_or_target_task_and_fixed_noise_fallback(device, fixed, empty_task):
    Xs, Ys = training(device)
    Xs[empty_task], Ys[empty_task] = Xs[empty_task][:0], Ys[empty_task][:0]
    variances = [torch.full_like(Y, 0.01 * (i + 1)) for i, Y in enumerate(Ys)] if fixed else None
    model = HeterogeneousMTGP(
        Xs,
        Ys,
        variances,
        FEATURES,
        3,
        input_transform=Normalize(4, indices=[0, 1, 2]),
        outcome_transform=Standardize(1),
        backend="torch",
    )
    query = Xs[0].new_tensor([[0.35, 0.25, 0.0]])
    latent, observed = model.posterior(query), model.posterior(query, observation_noise=True)
    assert torch.isfinite(observed.variance).all()
    if fixed:
        expected_noise = torch.cat(variances).mean() if empty_task == 0 else variances[0].mean()
        torch.testing.assert_close(
            observed.variance - latent.variance, expected_noise.expand_as(latent.variance)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_full_space_training_replacement(device):
    Xs, Ys = training(device)
    model = HeterogeneousMTGP(Xs, Ys, None, FEATURES, 3, backend="torch")
    query = Xs[0].new_tensor([[0.35, 0.25, 0.0]])
    original = model.posterior(query).mean
    full_X, full_Y = model._train_X.clone(), model._train_Y.clone()
    model.set_train_data(full_X, full_Y + 2, strict=True)
    assert (model.posterior(query).mean - original).min() > 1
