"""Long-format multitask exact inference stays on device and fully differentiable."""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood  # noqa: E402
from botorch_rs.models import MultiTaskGP  # noqa: E402
from botorch_rs.posteriors.gpytorch import scalarize_posterior  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device, dtype=torch.double, task_feature=-1, labels=(0, 1)):
    continuous = torch.tensor(
        [[0.1], [0.2], [0.4], [0.65], [0.7], [0.9]], dtype=dtype, device=device
    )
    tasks = torch.tensor(
        [labels[0], labels[1], labels[0], labels[1], labels[0], labels[1]],
        dtype=dtype,
        device=device,
    ).unsqueeze(-1)
    X = torch.cat((continuous, tasks) if task_feature == -1 else (tasks, continuous), dim=-1)
    Y = torch.sin(5 * continuous) + (tasks == labels[1]) * 0.3
    query = torch.tensor([[[0.25], [0.55]], [[0.35], [0.8]]], dtype=dtype, device=device)
    return X, Y, query


def paired_models(X, Y, task_feature=-1, noise=None, **kwargs):
    from botorch.models import MultiTaskGP as ReferenceGP

    torch.manual_seed(23)
    actual = MultiTaskGP(X, Y, task_feature, noise, backend="torch", **kwargs)
    torch.manual_seed(23)
    expected = ReferenceGP(X, Y, task_feature, noise, **kwargs)
    return actual, expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("task_feature", [-1, 0])
@pytest.mark.parametrize("fixed", [False, True])
def test_default_multitask_reference_moments_mll_and_parameter_gradients(
    device, dtype, task_feature, fixed
):
    from botorch.settings import propagate_grads
    from gpytorch.mlls import ExactMarginalLogLikelihood as ReferenceMLL
    from gpytorch.settings import detach_test_caches

    X, Y, query = data(device, dtype, task_feature)
    noise = torch.full_like(Y, 0.025) if fixed else None
    actual, expected = paired_models(X, Y, task_feature, noise)
    tol = 2e-4 if dtype == torch.float32 else 2e-8
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter, dict(expected.named_parameters())[name], atol=tol, rtol=tol
        )
    with propagate_grads(True), detach_test_caches(False):
        a = actual.posterior(query, output_indices=[1, 0], observation_noise=True)
        e = expected.posterior(query, output_indices=[1, 0], observation_noise=True)
        for value, reference in [
            (a.mean, e.mean),
            (a.variance, e.variance),
            (a.distribution.covariance_matrix, e.distribution.covariance_matrix),
        ]:
            torch.testing.assert_close(value, reference, atol=tol, rtol=tol)
        assert a.dtype == dtype and a.device == X.device
        actual.train()
        expected.train()
        amll = ExactMarginalLogLikelihood(actual.likelihood, actual)
        emll = ReferenceMLL(expected.likelihood, expected)
        av, ev = (
            amll(),
            emll(
                expected(*expected.train_inputs), expected.train_targets, expected.train_inputs[0]
            ),
        )
        torch.testing.assert_close(av, ev, atol=tol, rtol=tol)
        torch.testing.assert_close(
            amll(actual(*actual.train_inputs), actual.train_targets), av, atol=tol, rtol=tol
        )
        ag = torch.autograd.grad(
            av + a.mean.square().sum() + a.variance.sum(), tuple(actual.parameters())
        )
        ep = dict(expected.named_parameters())
        eg = torch.autograd.grad(
            ev + e.mean.square().sum() + e.variance.sum(),
            tuple(ep[name] for name, _ in actual.named_parameters()),
        )
        for value, reference in zip(ag, eg, strict=True):
            torch.testing.assert_close(value, reference, atol=tol * 5, rtol=tol * 5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("task_feature", [-1, 0])
def test_noncontiguous_labels_selected_tasks_and_full_task_queries(device, task_feature):
    X, Y, query = data(device, task_feature=task_feature, labels=(2, 5))
    noise = torch.full_like(Y, 0.03)
    actual, expected = paired_models(
        X, Y, task_feature, noise, output_tasks=[5], all_tasks=[5, 2], rank=1
    )
    assert actual.num_outputs == 1 and actual.num_tasks == 2
    torch.testing.assert_close(actual._task_mapper, expected._task_mapper, equal_nan=True)
    for selected in (None, [2], [5, 2], [5, 2, 5]):
        a, e = (
            actual.posterior(query, output_indices=selected, observation_noise=True),
            expected.posterior(query, output_indices=selected, observation_noise=True),
        )
        torch.testing.assert_close(a.mean, e.mean, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(
            a.distribution.covariance_matrix, e.distribution.covariance_matrix, atol=1e-9, rtol=1e-9
        )
    q = query[0]
    labels = q.new_tensor([[5.0], [2.0]])
    full = torch.cat((q, labels) if task_feature == -1 else (labels, q), dim=-1).requires_grad_()
    a, e = actual.posterior(full), expected.posterior(full)
    torch.testing.assert_close(a.mean, e.mean, atol=1e-9, rtol=1e-9)
    gradient = torch.autograd.grad(a.mean.sum() + a.variance.sum(), full, create_graph=True)[0]
    assert torch.equal(
        gradient[..., actual._task_feature], torch.zeros_like(gradient[..., actual._task_feature])
    )
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), full)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("standardize", [False, True])
def test_training_inputs_targets_noise_queries_gradcheck_and_gradgradcheck(device, standardize):
    X, Y, query = data(device)
    continuous = X[:, :1].clone().requires_grad_()
    Y = Y.clone().requires_grad_()
    query = query[0].clone().requires_grad_()
    noise = torch.full_like(Y, 0.04, requires_grad=True)

    def loss(x, y, q, n):
        torch.manual_seed(42)
        full = torch.cat((x, X[:, 1:]), dim=-1)
        model = MultiTaskGP(
            full, y, -1, n, backend="torch", **({} if standardize else {"outcome_transform": None})
        )
        posterior = model.posterior(q, output_indices=[1, 0])
        return (
            posterior.mean.square().sum()
            + posterior.variance.sum()
            + ExactMarginalLogLikelihood(model.likelihood, model)()
        )

    arguments = (continuous, Y, query, noise)
    assert torch.autograd.gradcheck(loss, arguments, fast_mode=True)
    assert torch.autograd.gradgradcheck(loss, arguments, fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
def test_correlated_samples_scalarization_and_parameter_double_backward(device):
    X, Y, query = data(device)
    model = MultiTaskGP(X, Y, -1, backend="torch")
    query = query[0].requires_grad_()
    normals = torch.randn(3, 2, 2, dtype=X.dtype, device=device, requires_grad=True)
    for _ in range(2):
        posterior = model.posterior(query)
        scalar = scalarize_posterior(posterior, weights=X.new_tensor([0.3, -0.7]))
        covariance = posterior.distribution.covariance_matrix.reshape(2, 2, 2, 2)
        oracle = torch.einsum(
            "tiuj,t,u->ij", covariance, X.new_tensor([0.3, -0.7]), X.new_tensor([0.3, -0.7])
        )
        torch.testing.assert_close(scalar.distribution.covariance_matrix, oracle)
        samples = posterior.rsample_from_base_samples(torch.Size([3]), normals)
        loss = samples.square().sum() + scalar.mean.square().sum()
        inputs = (query, normals, *tuple(model.parameters()))
        first = torch.autograd.grad(loss, inputs, create_graph=True, allow_unused=True)
        active = [g for g in first if g is not None]
        second = torch.autograd.grad(
            sum(g.square().sum() for g in active), inputs, allow_unused=True
        )
        assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))
    assert not torch.equal(
        posterior.distribution.covariance_matrix[:2, 2:],
        torch.zeros(2, 2, device=device, dtype=X.dtype),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_custom_kernel_mean_gaussian_likelihood_and_input_transform(device):
    from botorch.models import MultiTaskGP as ReferenceGP
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from gpytorch.kernels import MaternKernel as ReferenceMatern
    from gpytorch.likelihoods import GaussianLikelihood as ReferenceLikelihood
    from gpytorch.means import ConstantMean as ReferenceMean

    from botorch_rs.kernels import MaternKernel
    from botorch_rs.likelihoods import GaussianLikelihood
    from botorch_rs.means import ConstantMean
    from botorch_rs.models.transforms.input import Normalize

    X, Y, query = data(device, labels=(2, 5))
    torch.manual_seed(25)
    actual = MultiTaskGP(
        X,
        Y,
        -1,
        backend="torch",
        covar_module=MaternKernel(nu=2.5, ard_num_dims=1),
        likelihood=GaussianLikelihood(),
        mean_module=ConstantMean(),
        input_transform=Normalize(2, indices=[0]),
        task_covar_prior=None,
    )
    torch.manual_seed(25)
    expected = ReferenceGP(
        X,
        Y,
        -1,
        covar_module=ReferenceMatern(nu=2.5, ard_num_dims=1),
        likelihood=ReferenceLikelihood(),
        mean_module=ReferenceMean(),
        input_transform=ReferenceNormalize(2, indices=[0]),
        task_covar_prior=None,
    )
    a, e = actual.posterior(query), expected.posterior(query)
    torch.testing.assert_close(a.mean, e.mean, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(
        a.distribution.covariance_matrix, e.distribution.covariance_matrix, atol=1e-8, rtol=1e-8
    )
    loss = (
        a.mean.square().sum()
        + a.variance.sum()
        + ExactMarginalLogLikelihood(actual.likelihood, actual)()
    )
    gradients = torch.autograd.grad(loss, tuple(actual.parameters()), create_graph=True)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in gradients), tuple(actual.parameters())
    )
    assert all(torch.isfinite(g).all() for g in (*gradients, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_unobserved_tasks_initialization_matches_reference_and_repeatable_backward(device):
    X, Y, query = data(device)
    actual, expected = paired_models(X, Y, all_tasks=[2, 0, 1], output_tasks=[2, 0])
    for _ in range(2):
        a, e = actual.posterior(query), expected.posterior(query)
        torch.testing.assert_close(a.mean, e.mean, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(a.variance, e.variance, atol=1e-8, rtol=1e-8)
        loss = a.mean.sum() + a.variance.sum()
        assert all(
            torch.isfinite(g).all() for g in torch.autograd.grad(loss, tuple(actual.parameters()))
        )
    assert actual._unobserved_task_indices.device == X.device


@pytest.mark.parametrize("device", DEVICES)
def test_backend_validation_and_no_training_tensor_host_roundtrip(device):
    X, Y, query = data(device)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy copy")),
        patch.object(torch.Tensor, "detach", side_effect=AssertionError("Detached graph")),
    ):
        model = MultiTaskGP(X.requires_grad_(), Y.requires_grad_(), -1)
        assert getattr(model, "_tensor_backend", None) == "torch"
        posterior = model.posterior(query)
        assert torch.isfinite(torch.autograd.grad(posterior.mean.sum(), X)[0]).all()
    with pytest.raises(ValueError, match="task"):
        model.posterior(torch.tensor([[0.2, 9.0]], dtype=X.dtype, device=device))
    with pytest.raises(ValueError, match="output_indices"):
        model.posterior(X, output_indices=[0])
    with pytest.raises(NotImplementedError, match="Tensor observation_noise"):
        model.posterior(query, observation_noise=Y)
    with pytest.raises(UnsupportedError, match="Conditioning"):
        model.condition_on_observations(query, Y)


@pytest.mark.parametrize("device", DEVICES)
def test_every_raw_parameter_gradcheck_and_gradgradcheck(device):
    X, Y, query = data(device)

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MultiTaskGP(X, Y, -1, backend="torch", outcome_transform=None)

        def forward(self):
            posterior = self.model.posterior(query)
            return (
                posterior.mean.square().sum() + posterior.variance.sum() + self.model._exact_mll()
            )

    loss = Loss()
    names, parameters = zip(*loss.named_parameters(), strict=True)

    def evaluate(*arguments):
        return torch.func.functional_call(loss, dict(zip(names, arguments, strict=True)), ())

    assert torch.autograd.gradcheck(evaluate, parameters, fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, parameters, fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
def test_torch_custom_kernel_likelihood_and_live_training_optimizer(device):
    from gpytorch.likelihoods import GaussianLikelihood

    X, Y, query = data(device)

    class CustomKernel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_lengthscale = torch.nn.Parameter(X.new_tensor(-0.5))

        def forward(self, x1, x2):
            # The module receives continuous columns only. Its parameter is
            # consumed directly by the GP graph, including higher derivatives.
            assert x1.shape[-1] == 1
            return (
                -0.5
                * (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(-1)
                / (2 * self.log_lengthscale).exp()
            ).exp()

    kernel = CustomKernel()
    likelihood = GaussianLikelihood().to(X)
    model = MultiTaskGP(X, Y, -1, covar_module=kernel, likelihood=likelihood)
    assert model._tensor_backend == "torch"
    assert model.covar_module.kernels[0].base_kernel is kernel and model.likelihood is likelihood
    objective = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = float(objective().detach())
    for _ in range(4):
        optimizer.zero_grad()
        loss = -objective()
        loss.backward()
        optimizer.step()
    assert float(objective().detach()) > before
    first = torch.autograd.grad(
        model.posterior(query).mean.square().sum(), kernel.log_lengthscale, create_graph=True
    )[0]
    assert torch.isfinite(torch.autograd.grad(first, kernel.log_lengthscale)[0])


@pytest.mark.parametrize("device", DEVICES)
def test_dtype_move_and_task_validation_flag(device):
    X, Y, query = data("cpu", labels=(2, 5))
    model = MultiTaskGP(X, Y, -1, backend="torch", validate_task_values=False).to(
        device=device, dtype=torch.float32
    )
    result = model.posterior(query.to(device=device, dtype=torch.float32))
    assert result.device.type == device and result.dtype == torch.float32
    assert model._task_labels.dtype == torch.long and model._task_labels.device.type == device
    # Disabling validation exposes upstream's raw mapper (including NaNs for
    # absent labels) rather than silently aliasing an unexpected task.
    assert model._map_tasks(result.mean.new_tensor([3])).isnan().all()
    with pytest.raises(IndexError):
        model.posterior(result.mean.new_tensor([[0.3, 3.0]]))


def test_cpu_native_default_and_cuda_auto_selection():
    X, Y, _ = data("cpu")
    assert getattr(MultiTaskGP(X, Y, -1), "_tensor_backend", None) != "torch"
    if torch.cuda.is_available():
        assert MultiTaskGP(X.cuda(), Y.cuda(), -1)._tensor_backend == "torch"


@pytest.mark.parametrize("device", DEVICES)
def test_input_transform_task_feature_and_invalid_configuration_are_explicit(device):
    from botorch_rs.models.transforms.input import Normalize

    X, Y, _ = data(device, labels=(2, 5))
    with pytest.raises(UnsupportedError, match="preserve categorical"):
        MultiTaskGP(X, Y, -1, backend="torch", input_transform=Normalize(2))
    with pytest.raises(UnsupportedError, match="all_tasks"):
        MultiTaskGP(X, Y, -1, backend="torch", all_tasks=[2])
    with pytest.raises(ValueError, match="rank"):
        MultiTaskGP(X, Y, -1, backend="torch", rank=3)
    with pytest.raises(ValueError, match="unbatched"):
        MultiTaskGP(X.unsqueeze(0), Y.unsqueeze(0), -1, backend="torch")
    from botorch_rs.priors import GammaPrior, LKJCovariancePrior

    with pytest.raises(UnsupportedError, match="matrix-prior"):
        MultiTaskGP(
            X,
            Y,
            -1,
            backend="torch",
            task_covar_prior=LKJCovariancePrior(2, 1.0, GammaPrior(2.0, 1.0)),
        )

    class NonGaussian(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("noise", X.new_tensor([0.1]))

    with pytest.raises(UnsupportedError, match="attribute alone"):
        MultiTaskGP(X, Y, -1, backend="torch", likelihood=NonGaussian())


@pytest.mark.parametrize("device", DEVICES)
def test_nonlinear_outcome_chain_broadcasts_long_format_statistics_over_tasks(device):
    from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize

    X, Y, query = data(device)
    Y = (Y + 2).requires_grad_()
    query = query[0].requires_grad_()
    torch.manual_seed(13)
    model = MultiTaskGP(
        X,
        Y,
        -1,
        backend="torch",
        outcome_transform=ChainedOutcomeTransform(log=Log(), standardize=Standardize(1)),
    )
    log_y = Y.log()
    offset, scale = log_y.mean(0, keepdim=True), log_y.std(0, keepdim=True)
    torch.manual_seed(13)
    gaussian = MultiTaskGP(X, (log_y - offset) / scale, -1, backend="torch", outcome_transform=None)
    normals = torch.randn(3, 2, 2, dtype=X.dtype, device=device)
    posterior = model.posterior(query)
    samples = posterior.rsample_from_base_samples(torch.Size([3]), normals)
    expected = (
        gaussian.posterior(query).rsample_from_base_samples(torch.Size([3]), normals) * scale
        + offset
    ).exp()
    torch.testing.assert_close(samples, expected, atol=1e-10, rtol=1e-10)
    first = torch.autograd.grad(samples.square().sum(), (Y, query), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (Y, query))
    assert all(torch.isfinite(g).all() for g in (*first, *second))
