"""Standalone public Tensor kernel constructors and task covariance conversion."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from test_torch_backend_modules import DEVICES

from botorch_rs import kernels
from botorch_rs.constraints import GreaterThan
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models._torch_backend_modules import build_kernel
from botorch_rs.optim.utils.model_utils import sample_all_priors
from botorch_rs.priors import GammaPrior, LKJCovariancePrior


def _copy(reference, port):
    values = dict(port.named_parameters())
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            parameter.copy_(values[name])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "family", ["RBF", "Matern", "RQ", "Periodic", "Linear", "Polynomial", "Cosine"]
)
def test_explicit_elementary_factory_device_values_full_graphs_and_state(device, dtype, family):
    cls = getattr(kernels, family + "Kernel")
    ref_cls = getattr(gpytorch.kernels, family + "Kernel")
    args = (2,) if family == "Polynomial" else ()
    kwargs = dict(batch_shape=torch.Size([2]), active_dims=[2, 0])
    if family not in {"Polynomial", "Cosine"}:
        kwargs["ard_num_dims"] = 2
    kernel = cls(*args, **kwargs, backend="torch", dtype=dtype, device=device)
    reference = ref_cls(*args, **kwargs).to(dtype=dtype, device=device)
    _copy(reference, kernel)
    assert isinstance(kernel, cls) and isinstance(kernel, torch.nn.Module)
    assert all(p.dtype == dtype and p.device.type == device for p in kernel.parameters())
    x = torch.rand(3, 1, 4, 3, dtype=dtype, device=device, requires_grad=True)
    z = torch.rand(1, 2, 3, 3, dtype=dtype, device=device, requires_grad=True)
    actual, expected = kernel(x, z), reference(x, z).to_dense()
    tolerance = 2e-5 if dtype == torch.float else 2e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        kernel(x[..., :3, :], z, diag=True),
        actual[..., :3, :].diagonal(dim1=-2, dim2=-1),
        atol=tolerance,
        rtol=tolerance,
    )
    values = (x, z, *kernel.parameters())
    first = torch.autograd.grad(actual.square().sum(), values, create_graph=True)
    ref_params = dict(reference.named_parameters())
    oracle = torch.autograd.grad(
        expected.square().sum(), (x, z, *(ref_params[n] for n, _ in kernel.named_parameters()))
    )
    for a, b in zip(first, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=5 * tolerance, rtol=5 * tolerance)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), values)
    assert all(torch.isfinite(g).all() for g in (*first, *second))
    restored = copy.deepcopy(kernel)
    restored.load_state_dict(kernel.state_dict())
    torch.testing.assert_close(restored(x, z), actual)
    restored = pickle.loads(pickle.dumps(kernel))
    torch.testing.assert_close(restored(x, z), actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_generic_index_batches_active_columns_feature_batches_and_reference_graphs(device, rank):
    kernel = kernels.IndexKernel(
        3,
        rank=rank,
        batch_shape=(2,),
        active_dims=[1],
        backend="torch",
        dtype=torch.double,
        device=device,
    )
    reference = (
        gpytorch.kernels.IndexKernel(3, rank=rank, batch_shape=torch.Size([2]), active_dims=[1])
        .double()
        .to(device)
    )
    _copy(reference, kernel)
    x = torch.tensor(
        [[0.1, 0.0], [0.2, 2.0], [0.3, 1.0], [0.4, 0.0]], dtype=torch.double, device=device
    ).expand(3, 1, 4, 2)
    z = torch.tensor([[0.5, 1.0], [0.6, 0.0], [0.7, 2.0]], dtype=x.dtype, device=device).expand(
        1, 2, 3, 2
    )
    actual, expected = kernel(x, z), reference(x, z).to_dense()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        kernel(x[..., :3, :], z, diag=True), actual[..., :3, :].diagonal(dim1=-2, dim2=-1)
    )
    actual_loss = actual.square().sum()
    first = torch.autograd.grad(actual_loss, tuple(kernel.parameters()), create_graph=True)
    oracle = torch.autograd.grad(
        expected.square().sum(), tuple(reference.parameters()), create_graph=True
    )
    for a, b in zip(first, oracle, strict=True):
        torch.testing.assert_close(a, b)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), tuple(kernel.parameters()))
    reference_second = torch.autograd.grad(
        sum(g.square().sum() for g in oracle), tuple(reference.parameters())
    )
    for a, b in zip(second, reference_second, strict=True):
        torch.testing.assert_close(a, b)
    simple = kernels.IndexKernel(3, rank=rank, backend="torch", dtype=x.dtype, device=device)
    labels = x[0, 0, :, 1:].expand(4, 2)
    torch.testing.assert_close(
        simple(labels, last_dim_is_batch=True), simple(labels[:, :1]).unsqueeze(0).expand(2, 4, 4)
    )
    with pytest.raises(IndexError, match="outside"):
        simple(labels[:, :1] + 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
def test_multitask_interleaved_covariance_diagonal_and_all_derivatives(device, batch):
    base = kernels.MaternKernel(
        ard_num_dims=2, batch_shape=batch, backend="torch", dtype=torch.double, device=device
    )
    kernel = kernels.MultitaskKernel(base, 3, rank=2, batch_shape=batch, backend="torch")
    reference = (
        gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=2, batch_shape=torch.Size(batch)),
            3,
            rank=2,
            batch_shape=torch.Size(batch),
        )
        .double()
        .to(device)
    )
    _copy(reference, kernel)
    x = torch.rand(4, 2, dtype=torch.double, device=device, requires_grad=True)
    z = torch.rand(3, 2, dtype=x.dtype, device=device, requires_grad=True)
    actual, expected = kernel(x, z), reference(x, z).to_dense()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        kernel(x[:3], z, diag=True), actual[..., :9, :].diagonal(dim1=-2, dim2=-1)
    )
    first = torch.autograd.grad(
        actual.square().sum(), (x, z, *kernel.parameters()), create_graph=True
    )
    oracle = torch.autograd.grad(expected.square().sum(), (x, z, *reference.parameters()))
    for a, b in zip(first, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-9)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (x, z, *kernel.parameters()))
    assert all(torch.isfinite(g).all() for g in second)
    assert kernel.data_covar_module is base
    with pytest.raises(NotImplementedError, match="last_dim"):
        kernel(x, last_dim_is_batch=True)


@pytest.mark.parametrize("device", DEVICES)
def test_core_tree_aliases_frozen_leaves_constraints_and_gp_fitting(device):
    shared = kernels.RBFKernel(
        ard_num_dims=1,
        active_dims=[0],
        lengthscale_prior=GammaPrior(2.0, 3.0),
        lengthscale_constraint=GreaterThan(0.1),
        backend="torch",
        dtype=torch.double,
        device=device,
    )
    held = shared.raw_lengthscale
    shared.requires_grad_(False)
    tree = kernels.ScaleKernel(shared + shared * shared, outputscale_prior=GammaPrior(2.0, 3.0))
    assert isinstance(tree, kernels.ScaleKernel) and isinstance(
        tree.base_kernel, kernels.AdditiveKernel
    )
    assert tree.base_kernel.kernels[0] is shared
    assert tree.base_kernel.kernels[1].kernels[0] is shared
    tree.initialize(**{"base_kernel.kernels.1.kernels.0.lengthscale": 0.8})
    assert shared.raw_lengthscale is held and not held.requires_grad
    torch.testing.assert_close(shared.lengthscale, torch.full_like(held, 0.8))
    x = torch.rand(5, 1, dtype=torch.double, device=device, requires_grad=True)
    y = torch.randn(5, 1, dtype=x.dtype, device=device, requires_grad=True)
    model = SingleTaskGP(x, y, covar_module=tree, outcome_transform=None, backend="torch")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    loss = -mll(model(*model.train_inputs), model.train_targets).sum()
    variables = (x, y, *(p for p in model.parameters() if p.requires_grad))
    first = torch.autograd.grad(loss, variables, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), variables)
    assert all(torch.isfinite(g).all() for g in (*first, *second))
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.002)
    before = tree.raw_outputscale.detach().clone()
    optimizer.zero_grad()
    (-mll(model(*model.train_inputs), model.train_targets).sum()).backward()
    optimizer.step()
    assert not torch.equal(before, tree.raw_outputscale) and shared.raw_lengthscale is held


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tasks,rank", [(1, 1), (3, 0), (3, 1), (3, 3)])
def test_index_conversion_lkj_density_sampling_and_training(device, tasks, rank):
    native = kernels.IndexKernel(
        tasks,
        rank=rank,
        batch_shape=(2,),
        prior=LKJCovariancePrior(tasks, 1.7, GammaPrior(2.0, 1.3)),
    )
    like = torch.empty((), dtype=torch.double, device=device)
    kernel = build_kernel(native, 1, (2,), like)
    assert isinstance(kernel, kernels.IndexKernel)
    torch.testing.assert_close(
        kernel.log_prior().cpu(),
        native.IndexKernelPrior.log_prob(kernel.covar_matrix.detach().cpu()),
    )
    held = tuple(kernel.parameters())
    sample_all_priors(kernel, seed=37)
    assert all(a is b for a, b in zip(held, kernel.parameters(), strict=True))
    assert torch.isfinite(kernel.log_prior()).all()
    desired = kernel.IndexKernelPrior.sample((2,))
    kernel.covar_matrix = desired
    if rank == tasks:
        torch.testing.assert_close(kernel.covar_matrix, desired, atol=3e-14, rtol=3e-12)
    x = torch.arange(tasks, dtype=like.dtype, device=device).reshape(tasks, 1).repeat(2, 1)
    y = (
        torch.linspace(0.1, 0.8, len(x), dtype=like.dtype, device=device)
        .reshape(-1, 1)
        .requires_grad_()
    )
    model = SingleTaskGP(x, y, covar_module=kernel, outcome_transform=None, backend="torch")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    loss = mll(model(*model.train_inputs), model.train_targets).sum()
    gradients = torch.autograd.grad(loss, (y, *model.parameters()), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), (y, *model.parameters()))
    assert all(torch.isfinite(g).all() for g in (*gradients, *second))


def test_native_default_and_invalid_backend_remain_explicit():
    assert not isinstance(kernels.RBFKernel(), torch.nn.Module)
    assert not isinstance(kernels.IndexKernel(2, backend="native"), torch.nn.Module)
    with pytest.raises(ValueError, match="backend"):
        kernels.RBFKernel(backend="invalid")


@pytest.mark.parametrize("device", DEVICES)
def test_composite_own_batch_active_dims_and_multitask_source_conversion(device):
    like = torch.empty((), dtype=torch.double, device=device)
    shared = kernels.RBFKernel(ard_num_dims=1, backend="torch", dtype=like.dtype, device=device)
    tree = kernels.AdditiveKernel(
        shared, shared, backend="torch", batch_shape=(3,), active_dims=[1], ard_num_dims=1
    )
    x = torch.rand(4, 2, dtype=like.dtype, device=device, requires_grad=True)
    z = torch.rand(3, 2, dtype=like.dtype, device=device, requires_grad=True)
    expected = (shared(x[:, 1:], z[:, 1:]) * 2).expand(3, 4, 3)
    torch.testing.assert_close(tree(x, z), expected)
    assert tree.kernels[0] is tree.kernels[1] is shared
    with pytest.raises(ValueError, match="Expected"):
        kernels.AdditiveKernel(shared, backend="torch", ard_num_dims=3)(x, z)
    native_shared = kernels.MaternKernel(ard_num_dims=2)
    source = kernels.MultitaskKernel(kernels.ScaleKernel(native_shared + native_shared), 3, rank=2)
    converted = build_kernel(source, 2, (), like)
    assert isinstance(converted, kernels.MultitaskKernel)
    assert (
        converted.data_covar_module.base_kernel.kernels[0]
        is converted.data_covar_module.base_kernel.kernels[1]
    )
    torch.testing.assert_close(
        converted(x, z).cpu(), source(x.detach().cpu(), z.detach().cpu()), atol=2e-12, rtol=2e-10
    )
    held = converted.task_covar_module.raw_var
    converted.initialize(**{"task_covar_module.var": 0.4})
    assert converted.task_covar_module.raw_var is held
    torch.testing.assert_close(converted.task_covar_module.var, torch.full_like(held, 0.4))
    covariance = converted(x, z)
    first = torch.autograd.grad(
        covariance.square().sum(), (x, z, *converted.parameters()), create_graph=True
    )
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), (x, z, *converted.parameters())
    )
    assert all(torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_device_factory_auto_selection_and_cpu_migration():
    kernel = kernels.ScaleKernel(
        kernels.RBFKernel(ard_num_dims=1), device="cuda", dtype=torch.double
    )
    assert isinstance(kernel, torch.nn.Module)
    x = torch.rand(4, 1, dtype=torch.double, device="cuda", requires_grad=True)
    value = kernel(x).detach().cpu()
    kernel.cpu()
    torch.testing.assert_close(kernel(x.detach().cpu()), value)
    assert kernel.base_kernel.raw_lengthscale.device.type == "cpu"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("configuration", ["active_dims", "batched_inputs"])
def test_multitask_known_upstream_failures_use_independent_kronecker_oracle(device, configuration):
    # GPyTorch2809 bypasses the child active_dims;2817 repeats existing task
    # batches instead of broadcasting them. Compare the mathematical law,
    # rather than retaining either faulty source configuration as an oracle.
    batch = (2,) if configuration == "batched_inputs" else ()
    base = kernels.RBFKernel(
        ard_num_dims=1 if not batch else 2,
        active_dims=[1] if not batch else None,
        batch_shape=batch,
        backend="torch",
        dtype=torch.double,
        device=device,
    )
    kernel = kernels.MultitaskKernel(base, 2, rank=1, batch_shape=batch, backend="torch")
    x = torch.rand(*batch, 4, 2, dtype=torch.double, device=device, requires_grad=True)
    z = torch.rand(*((1,) if batch else ()), 3, 2, dtype=x.dtype, device=device, requires_grad=True)
    spatial = base(x, z)
    task = kernel.task_covar_module.covar_matrix
    expected = (
        torch.stack([torch.kron(spatial[i].contiguous(), task[i].contiguous()) for i in range(2)])
        if batch
        else torch.kron(spatial.contiguous(), task.contiguous())
    )
    actual = kernel(x, z)
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-11)
    torch.testing.assert_close(
        kernel(x[..., :3, :], z, diag=True), expected[..., :6, :].diagonal(dim1=-2, dim2=-1)
    )
    variables = (x, z, *kernel.parameters())
    first = torch.autograd.grad(
        actual.square().sum(), variables, create_graph=True, retain_graph=True
    )
    oracle = torch.autograd.grad(
        expected.square().sum(), variables, create_graph=True, retain_graph=True
    )
    for a, b in zip(first, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=3e-11, rtol=3e-9)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), variables, retain_graph=True)
    refsecond = torch.autograd.grad(sum(g.square().sum() for g in oracle), variables)
    for a, b in zip(second, refsecond, strict=True):
        torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-8)
