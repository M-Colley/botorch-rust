"""Public categorical/fidelity/task kernel Tensor factories and tree conversion."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.models.kernels.categorical import CategoricalKernel as RefCategorical
from botorch.models.kernels.downsampling import DownsamplingKernel as RefDownsampling
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel as RefDecay
from botorch.models.kernels.linear_truncated_fidelity import (
    LinearTruncatedFidelityKernel as RefTruncated,
)
from botorch.models.kernels.positive_index import PositiveIndexKernel as RefPositive
from botorch.models.utils.priors import BetaPrior as RefBeta

from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models._torch_backend_modules import build_kernel
from botorch_rs.models.kernels import (
    CategoricalKernel,
    DownsamplingKernel,
    ExponentialDecayKernel,
    LinearTruncatedFidelityKernel,
    PositiveIndexKernel,
)
from botorch_rs.models.utils.priors import BetaPrior
from botorch_rs.optim.utils.model_utils import sample_all_priors
from botorch_rs.priors import GammaPrior, LKJCovariancePrior

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _copy(reference, port):
    params = dict(port.named_parameters())
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            parameter.copy_(params[name])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["categorical", "downsampling", "decay", "truncated"])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_standalone_reference_laws_parameter_graphs_and_correct_diagonal(device, family, batch):
    torch.manual_seed(932)
    like = torch.empty((), dtype=torch.double, device=device)
    classes = {
        "categorical": (CategoricalKernel, RefCategorical),
        "downsampling": (DownsamplingKernel, RefDownsampling),
        "decay": (ExponentialDecayKernel, RefDecay),
        "truncated": (LinearTruncatedFidelityKernel, RefTruncated),
    }
    cls, ref_cls = classes[family]
    kwargs = {"batch_shape": batch}
    if family in ("categorical", "decay"):
        kwargs["ard_num_dims"] = 3
    if family == "truncated":
        kwargs.update(fidelity_dims=[1, 2], dimension=3)
    kernel = cls(**kwargs, backend="torch").to(like)
    reference = ref_cls(**kwargs).to(like)
    _copy(reference, kernel)
    x = torch.rand(*batch, 3, 3, dtype=torch.double, device=device) * 0.6 + 0.2
    z = torch.rand(*((1,) if batch else ()), 4, 3, dtype=torch.double, device=device) * 0.6 + 0.2
    if family == "categorical":
        x, z = (x * 5).floor(), (z * 5).floor()
    else:
        x.requires_grad_()
        z.requires_grad_()
    actual, expected = kernel(x, z), reference(x, z).to_dense()
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-10)
    torch.testing.assert_close(
        kernel(x, z[..., :3, :], diag=True), actual[..., :3].diagonal(dim1=-2, dim2=-1)
    )
    variables = (
        tuple(kernel.parameters()) if family == "categorical" else (x, z, *kernel.parameters())
    )
    ref_params = dict(reference.named_parameters())
    ref_variables = tuple(ref_params[name] for name, _ in kernel.named_parameters())
    if family != "categorical":
        ref_variables = (x, z, *ref_variables)
    gradients = torch.autograd.grad(actual.square().sum(), variables, create_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), ref_variables)
    for a, b in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-11, rtol=2e-9)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), variables)
    assert all(torch.isfinite(g).all() for g in second)
    assert isinstance(kernel, cls)
    assert all(parameter.device.type == device for parameter in kernel.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["categorical", "downsampling", "decay"])
def test_feature_batches_match_independent_real_feature_kernels(device, family):
    cls = {
        "categorical": CategoricalKernel,
        "downsampling": DownsamplingKernel,
        "decay": ExponentialDecayKernel,
    }[family]
    kwargs = {"batch_shape": (2,)}
    if family != "downsampling":
        kwargs["ard_num_dims"] = 3
    kernel = cls(**kwargs, backend="torch").double().to(device)
    x = torch.rand(2, 3, 3, dtype=torch.double, device=device) * 0.5 + 0.2
    z = torch.rand(1, 4, 3, dtype=torch.double, device=device) * 0.5 + 0.2
    if family == "categorical":
        x, z = (x * 5).floor(), (z * 5).floor()
    for diag in (False, True):
        b = z[..., :3, :] if diag else z
        actual = kernel(x, b, diag=diag, last_dim_is_batch=True)
        expected = []
        for feature in range(3):
            child_kwargs = {"batch_shape": (2,)}
            if family != "downsampling":
                child_kwargs["ard_num_dims"] = 1
            child = cls(**child_kwargs, backend="torch").double().to(device)
            with torch.no_grad():
                for name, parameter in child.named_parameters():
                    original = dict(kernel.named_parameters())[name]
                    parameter.copy_(
                        original[..., feature : feature + 1]
                        if name == "raw_lengthscale"
                        else original
                    )
            expected.append(
                child(x[..., feature : feature + 1], b[..., feature : feature + 1], diag=diag)
            )
        torch.testing.assert_close(
            actual, torch.stack(expected, -2 if diag else -3), atol=2e-13, rtol=2e-11
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "batch,rank,unit,target", [((), 1, True, 0), ((2,), 3, True, 2), ((2,), 2, False, 1)]
)
def test_positive_index_reference_all_parameter_hessians_and_priors(
    device, batch, rank, unit, target
):
    kernel = (
        PositiveIndexKernel(
            3,
            rank=rank,
            batch_shape=batch,
            target_task_index=target,
            unit_scale_for_target=unit,
            task_prior=BetaPrior(2.5, 1.5),
            backend="torch",
        )
        .double()
        .to(device)
    )
    reference = (
        RefPositive(
            3,
            rank=rank,
            batch_shape=torch.Size(batch),
            target_task_index=target,
            unit_scale_for_target=unit,
            task_prior=RefBeta(2.5, 1.5),
        )
        .double()
        .to(device)
    )
    _copy(reference, kernel)
    x = torch.tensor([[0.0], [2.0], [1.0]], dtype=torch.double, device=device)
    z = torch.tensor([[1.0], [0.0], [2.0], [0.0]], dtype=torch.double, device=device)
    actual, expected = kernel(x, z), reference(x, z).to_dense()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        kernel(x, z[:3], diag=True), actual[..., :3].diagonal(dim1=-2, dim2=-1)
    )
    torch.testing.assert_close(
        kernel.log_prior(),
        reference.IndexKernelPrior.log_prob(reference._lower_triangle_corr).sum(-1),
    )
    value = actual.square().sum() + kernel.log_prior().sum()
    ref_value = (
        expected.square().sum()
        + reference.IndexKernelPrior.log_prob(reference._lower_triangle_corr).sum()
    )
    params = dict(reference.named_parameters())
    gradients = torch.autograd.grad(value, tuple(kernel.parameters()), create_graph=True)
    expected_gradients = torch.autograd.grad(
        ref_value, tuple(params[name] for name, _ in kernel.named_parameters()), create_graph=True
    )
    for a, b in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(a, b)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in gradients), tuple(kernel.parameters())
    )
    ref_second = torch.autograd.grad(
        sum(g.square().sum() for g in expected_gradients),
        tuple(params[name] for name, _ in kernel.named_parameters()),
    )
    for a, b in zip(second, ref_second, strict=True):
        torch.testing.assert_close(a, b, atol=3e-11, rtol=2e-9)
    held = kernel.raw_covar_factor
    sample_all_priors(kernel)
    assert kernel.raw_covar_factor is held
    assert torch.isfinite(kernel.covar_matrix).all()
    features = x.expand(3, 2)
    torch.testing.assert_close(
        kernel(features, features, last_dim_is_batch=True),
        kernel(x, x).unsqueeze(-3).expand(*batch, 2, 3, 3),
    )
    with pytest.raises(IndexError, match="outside"):
        kernel(x + 4)


@pytest.mark.parametrize("device", DEVICES)
def test_common_tree_conversion_aliases_dotted_initialize_and_training(device):
    like = torch.empty((), dtype=torch.double, device=device)
    shared = RBFKernel(ard_num_dims=1)
    fidelity = LinearTruncatedFidelityKernel(
        [1], dimension=2, covar_module_unbiased=shared, covar_module_biased=shared
    )
    tree = build_kernel(ScaleKernel(fidelity), 2, (), like)
    assert tree.base_kernel.covar_module_unbiased is tree.base_kernel.covar_module_biased
    held = tree.base_kernel.covar_module_unbiased.raw_lengthscale
    tree.initialize(
        **{"base_kernel.covar_module_unbiased.lengthscale": 0.7, "base_kernel.power": 1.2}
    )
    assert tree.base_kernel.covar_module_unbiased.raw_lengthscale is held
    torch.testing.assert_close(
        tree.base_kernel.covar_module_unbiased.lengthscale, torch.full_like(held, 0.7)
    )
    x = torch.rand(6, 2, dtype=like.dtype, device=device, requires_grad=True) * 0.5 + 0.2
    y = torch.randn(6, 1, dtype=x.dtype, device=device, requires_grad=True)
    model = SingleTaskGP(x, y, covar_module=tree, backend="torch")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    query = torch.rand(2, 2, dtype=x.dtype, device=device, requires_grad=True) * 0.5 + 0.2
    leaves = (x, y, query, *model.parameters())
    for _ in range(2):
        posterior = model.posterior(query)
        objective = (
            posterior.mean.square().sum()
            + posterior.variance.sum()
            + mll(model(*model.train_inputs), model.train_targets).sum()
        )
        first = torch.autograd.grad(objective, leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
        assert all(torch.isfinite(g).all() for g in (*first, *second))
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.003)
    before = held.detach().clone()
    optimizer.zero_grad()
    (-mll(model(*model.train_inputs), model.train_targets)).backward()
    optimizer.step()
    assert not torch.equal(before, held)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 3])
def test_lkj_covariance_prior_true_density_full_gradients_and_projected_draws(device, rank):
    prior = LKJCovariancePrior(3, 1.7, GammaPrior(2.0, 1.3))
    kernel = (
        PositiveIndexKernel(3, rank=rank, batch_shape=(2,), task_prior=prior, backend="torch")
        .double()
        .to(device)
    )
    actual = kernel.log_prior()
    # The existing Rust scalar covariance density is independent of the
    # device Tensor implementation and counts the correlation term once.
    expected = prior.log_prob(kernel.covar_matrix.detach().cpu())
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-13, rtol=2e-11)
    assert actual.shape == (2,)
    first = torch.autograd.grad(actual.sum(), tuple(kernel.parameters()), create_graph=True)
    assert all(torch.isfinite(value).all() for value in first)
    second = torch.autograd.grad(
        sum(value.square().sum() for value in first), tuple(kernel.parameters())
    )
    assert all(torch.isfinite(value).all() for value in second)
    parameters = tuple(kernel.parameters())
    original = [value.detach().clone() for value in parameters]
    sample_all_priors(kernel, seed=28)
    assert tuple(kernel.parameters()) == parameters
    assert any(
        not torch.equal(left, right) for left, right in zip(original, parameters, strict=True)
    )
    assert torch.isfinite(kernel.log_prior()).all()
    assert (torch.linalg.eigvalsh(kernel.covar_matrix) > 0).all()
    assert kernel.IndexKernelPrior.concentration.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_custom_tensor_fidelity_children_and_registered_external_priors(device):
    # A user-supplied Torch/GpyTorch module is accepted as a real differentiable
    # subkernel. Its registered priors must contribute to the fitting objective.
    like = torch.empty((), dtype=torch.double, device=device)
    child = (
        pytest.importorskip("gpytorch")
        .kernels.RBFKernel(
            ard_num_dims=1,
            lengthscale_prior=pytest.importorskip("gpytorch").priors.GammaPrior(2.0, 3.0),
        )
        .to(like)
    )
    kernel = LinearTruncatedFidelityKernel(
        [1],
        dimension=2,
        covar_module_unbiased=child,
        covar_module_biased=child,
        dtype=like.dtype,
        device=like.device,
        backend="torch",
    )
    assert kernel.covar_module_unbiased is child and kernel.covar_module_biased is child
    x = torch.rand(4, 2, dtype=like.dtype, device=device, requires_grad=True) * 0.5 + 0.2
    value = kernel(x, x).sum() + kernel.log_prior()
    expected_prior = child.lengthscale_prior.log_prob(child.lengthscale).sum()
    torch.testing.assert_close(kernel.log_prior(), expected_prior)
    first = torch.autograd.grad(value, (x, child.raw_lengthscale), create_graph=True)
    assert all(torch.isfinite(value).all() for value in first)
    # GPyTorch's cdist second derivative is outside the custom module contract;
    # native converted children are already covered by complete HVP tests.
    kernel.initialize(**{"covar_module_unbiased.lengthscale": 0.8})
    torch.testing.assert_close(child.lengthscale, torch.full_like(child.lengthscale, 0.8))


@pytest.mark.parametrize("device", DEVICES)
def test_lkj_task_covariance_fitting_uses_device_density_and_preserves_graphs(device):
    kernel = (
        PositiveIndexKernel(
            3,
            rank=2,
            task_prior=LKJCovariancePrior(3, 1.7, GammaPrior(2.0, 1.3)),
            unit_scale_for_target=False,
            backend="torch",
        )
        .double()
        .to(device)
    )
    x = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0]], dtype=torch.double, device=device)
    y = torch.tensor(
        [[0.3], [0.8], [-0.2], [0.4], [0.6]], dtype=x.dtype, device=device, requires_grad=True
    )
    model = SingleTaskGP(x, y, covar_module=kernel, outcome_transform=None, backend="torch")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    variables = (y, *model.parameters())
    value = mll(model(*model.train_inputs), model.train_targets).sum()
    first = torch.autograd.grad(value, variables, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), variables)
    assert all(torch.isfinite(g).all() for g in (*first, *second))
    held = kernel.raw_covar_factor
    before = held.detach().clone()
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.003)
    optimizer.zero_grad()
    (-mll(model(*model.train_inputs), model.train_targets).sum()).backward()
    optimizer.step()
    assert kernel.raw_covar_factor is held and not torch.equal(before, held)
