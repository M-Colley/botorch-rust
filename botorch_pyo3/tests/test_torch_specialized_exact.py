# ruff: noqa: E402
"""Official mixed/fidelity covariance laws, devices and complete derivative graphs."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


from unittest.mock import patch

import torch

from botorch_rs.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch_rs.models.gp_regression_mixed import MixedSingleTaskGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SPECS = [
    "mixed",
    "categorical",
    "truncated",
    "truncated_two",
    "downsampling",
    "decay",
    "combined",
    "multiple_truncated",
    "multiple_downsampling",
]


def data(spec, device, dtype=torch.double, batch=False):
    X = torch.tensor(
        [[0.1, 0.2, 0.3], [0.3, 0.6, 0.5], [0.55, 0.35, 0.8], [0.8, 0.85, 0.6]],
        dtype=dtype,
        device=device,
    )
    Q = X.new_tensor([[0.25, 0.45, 0.65], [0.65, 0.7, 0.4]])
    if spec in ("mixed", "categorical"):
        X[:, 2] = X.new_tensor([0, 1, 0, 2])
        Q[:, 2] = Q.new_tensor([0, 2])
        if spec == "categorical":
            X, Q = X[:, 2:], Q[:, 2:]
    Y = torch.sin(X[:, :1] * 3)
    if batch:
        X = X.unsqueeze(0).expand(2, *X.shape).clone()
        Y = torch.cat((Y, Y.cos()), -1).unsqueeze(0).expand(2, -1, -1).clone()
        Y[1] = Y[1] * 1.3 + 0.2
        Q = Q.unsqueeze(0).expand(2, *Q.shape).clone()
    return X, Y, Q


def classes_and_options(spec):
    pytest.importorskip("botorch")
    from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP as ReferenceFidelity
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP as ReferenceMixed

    if spec in ("mixed", "categorical"):
        return MixedSingleTaskGP, ReferenceMixed, {"cat_dims": [-1]}
    options = {
        "truncated": {"data_fidelities": [2]},
        "truncated_two": {"iteration_fidelity": 1, "data_fidelities": [2]},
        "downsampling": {"data_fidelities": [2], "linear_truncated": False},
        "decay": {"iteration_fidelity": 2, "linear_truncated": False},
        "combined": {"iteration_fidelity": 1, "data_fidelities": [2], "linear_truncated": False},
        "multiple_truncated": {"data_fidelities": [1, 2]},
        "multiple_downsampling": {"data_fidelities": [1, 2], "linear_truncated": False},
    }[spec]
    return SingleTaskMultiFidelityGP, ReferenceFidelity, options


def copy_parameters(source, target):
    expected = dict(source.named_parameters())
    actual = dict(target.named_parameters())
    assert actual.keys() == expected.keys()
    with torch.no_grad():
        for name, parameter in actual.items():
            parameter.copy_(expected[name])


def parameters(model):
    values = dict(model.named_parameters())
    return tuple(values[name] for name in sorted(values))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("spec", SPECS)
def test_official_posterior_covariance_mll_and_parameter_query_gradients(
    device, dtype, batch, spec
):
    pytest.importorskip("botorch")
    from botorch.settings import propagate_grads

    pytest.importorskip("gpytorch")
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.settings import detach_test_caches

    X, Y, Q = data(spec, device, dtype, batch)
    port, reference, options = classes_and_options(spec)
    actual = port(X, Y, backend="torch", **options)
    expected = reference(X, Y, **options).to(X)
    assert isinstance(actual, port)
    copy_parameters(expected, actual)
    Qa, Qb = Q.clone().requires_grad_(), Q.clone().requires_grad_()
    pa = actual.posterior(Qa)
    with propagate_grads(True), detach_test_caches(False):
        pb = expected.posterior(Qb)
    atol = 5e-5 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(pa.mean, pb.mean, atol=atol, rtol=atol)
    torch.testing.assert_close(
        pa.distribution.covariance_matrix, pb.distribution.covariance_matrix, atol=atol, rtol=atol
    )
    assert pa.mean.device == X.device
    loss_a = pa.mean.square().sum() + pa.distribution.covariance_matrix.square().sum()
    loss_b = pb.mean.square().sum() + pb.distribution.covariance_matrix.square().sum()
    ga = torch.autograd.grad(loss_a, (Qa, *parameters(actual)), allow_unused=True)
    gb = torch.autograd.grad(loss_b, (Qb, *parameters(expected)), allow_unused=True)
    for a, b in zip(ga, gb, strict=True):
        if a is None or b is None:
            assert a is b is None
        else:
            torch.testing.assert_close(a, b, atol=atol * 5, rtol=atol * 5)
    expected.train()
    mll = ExactMarginalLogLikelihood(expected.likelihood, expected)
    torch.testing.assert_close(
        actual._exact_mll(),
        mll(expected(*expected.train_inputs), expected.train_targets),
        atol=atol * 5,
        rtol=atol * 5,
    )


def moments_loss(mean, covariance):
    return 0.4 * mean.square().sum() + 0.3 * covariance.square().sum()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", ["mixed", "truncated_two", "combined"])
def test_training_data_noise_and_parameter_graphs_and_hvp(device, spec):
    X, Y, Q = data(spec, device)
    X, Y, Q = (t.clone().requires_grad_() for t in (X, Y, Q))
    noise = torch.full_like(Y, 0.025, requires_grad=True)
    port, reference, options = classes_and_options(spec)
    actual = port(X, Y, train_Yvar=noise, outcome_transform=None, backend="torch", **options)
    expected = reference(X, Y, train_Yvar=noise, outcome_transform=None, **options).to(X)
    copy_parameters(expected, actual)
    pa = actual.posterior(Q)
    # Build the conditional law using actual official kernel/mean modules. This
    # avoids GPyTorch eval caches detaching the stored training-input graph.
    ktt = expected.covar_module(X, X).to_dense() + torch.diag_embed(noise.squeeze(-1))
    ktq = expected.covar_module(X, Q).to_dense()
    kqq = expected.covar_module(Q, Q).to_dense()
    residual = Y.squeeze(-1) - expected.mean_module(X)
    mean = expected.mean_module(Q) + (
        ktq.transpose(-1, -2) @ torch.linalg.solve(ktt, residual[..., None])
    ).squeeze(-1)
    covariance = kqq - ktq.transpose(-1, -2) @ torch.linalg.solve(ktt, ktq)
    arguments_a = (X, Y, noise, Q, *parameters(actual))
    arguments_b = (X, Y, noise, Q, *parameters(expected))
    first = torch.autograd.grad(
        moments_loss(pa.mean, pa.distribution.covariance_matrix),
        arguments_a,
        create_graph=True,
        retain_graph=True,
    )
    reference_first = torch.autograd.grad(moments_loss(mean[..., None], covariance), arguments_b)
    for a, b in zip(first, reference_first, strict=True):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)
    directions = [torch.ones_like(t) * 0.13 for t in arguments_a]
    if spec == "mixed":
        directions[0][..., -1] = 0
        directions[3][..., -1] = 0
    hvp = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(first, directions, strict=True)), arguments_a
    )
    differences = []
    step = 2e-5
    for sign in (1, -1):
        with torch.no_grad():
            for value, direction in zip(arguments_a, directions, strict=True):
                value.add_(direction, alpha=sign * step)
        posterior = actual.posterior(Q)
        differences.append(
            torch.autograd.grad(
                moments_loss(posterior.mean, posterior.distribution.covariance_matrix), arguments_a
            )
        )
        with torch.no_grad():
            for value, direction in zip(arguments_a, directions, strict=True):
                value.add_(direction, alpha=-sign * step)
    for derivative, positive, negative in zip(hvp, *differences, strict=True):
        assert bool(torch.isfinite(derivative).all())
        torch.testing.assert_close(
            derivative, (positive - negative) / (2 * step), atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", ["mixed", "truncated", "combined"])
def test_conditioning_samples_automatic_dispatch_and_native_boundary(device, spec):
    X, Y, Q = data(spec, device)
    Q.requires_grad_()
    port, _, options = classes_and_options(spec)
    actual = port(X.requires_grad_(), Y, **options)
    assert actual._tensor_backend == "torch"
    conditioned = actual.condition_on_observations(Q, Q[..., :1].sin())
    assert isinstance(conditioned, port)
    posterior = conditioned.posterior(Q)
    samples = posterior.rsample(torch.Size([3]))
    gradient = torch.autograd.grad(samples.square().sum(), Q, create_graph=True)
    assert gradient[0] is not None
    second = torch.autograd.grad(gradient[0].sum(), Q)[0]
    assert bool(torch.isfinite(second).all())
    assert samples.device == X.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", ["mixed", "downsampling"])
def test_custom_torch_continuous_kernels_keep_live_parameters_and_priors(device, spec):
    pytest.importorskip("gpytorch")
    from gpytorch.kernels import RBFKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.priors import GammaPrior

    X, Y, Q = data(spec, device)
    port, reference, options = classes_and_options(spec)
    captured = []

    def factory(**kwargs):
        result = RBFKernel(lengthscale_prior=GammaPrior(3.0, 6.0), **kwargs).to(X)
        captured.append(result)
        return result

    if spec == "mixed":
        actual = port(X, Y, cont_kernel_factory=factory, backend="torch", **options)
        expected = reference(X, Y, cont_kernel_factory=factory, **options).to(X)
    else:
        actual = port(
            X,
            Y,
            covar_module=factory(ard_num_dims=2, active_dims=[0, 1]),
            backend="torch",
            **options,
        )
        expected = reference(
            X, Y, covar_module=factory(ard_num_dims=2, active_dims=[0, 1]), **options
        ).to(X)
    actual_params = {
        name.replace(".module.", "."): value for name, value in actual.named_parameters()
    }
    with torch.no_grad():
        for name, value in expected.named_parameters():
            actual_params[name].copy_(value)
    torch.testing.assert_close(
        actual.posterior(Q).mean, expected.posterior(Q).mean, atol=2e-9, rtol=2e-9
    )
    expected.train()
    torch.testing.assert_close(
        actual._exact_mll(),
        ExactMarginalLogLikelihood(expected.likelihood, expected)(
            expected(*expected.train_inputs), expected.train_targets
        ),
        atol=2e-9,
        rtol=2e-9,
    )
    leaf = captured[0].raw_lengthscale
    assert any(value is leaf for value in actual.parameters())
    first = torch.autograd.grad(actual.posterior(Q).mean.square().sum(), leaf, create_graph=True)[0]
    assert bool(torch.isfinite(torch.autograd.grad(first.sum(), leaf)[0]).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", ["mixed", "truncated_two", "combined"])
def test_subset_and_fantasies_preserve_law_and_device_without_flat_conversion(device, spec):
    from botorch_rs.sampling.normal import IIDNormalSampler

    X, Y, Q = data(spec, device, batch=True)
    port, _, options = classes_and_options(spec)
    model = port(X, Y, backend="torch", **options)
    with torch.no_grad():
        selected = model.subset_output([1])
    Q.requires_grad_()
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host copy")),
    ):
        full = model.posterior(Q)
        reduced = selected.posterior(Q)
        torch.testing.assert_close(reduced.mean, full.mean[..., 1:])
        fantasies = model.fantasize(Q, IIDNormalSampler((2,), seed=18), propagate_grads=True)
        posterior = fantasies.posterior(Q)
        loss = posterior.mean.square().sum() + posterior.variance.sum()
        gradient = torch.autograd.grad(loss, Q, create_graph=True)[0]
        assert bool(torch.isfinite(torch.autograd.grad(gradient.sum(), Q)[0]).all())
    assert posterior.mean.device == Q.device
