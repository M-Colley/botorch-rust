"""Contextual Tensor models: official laws and full independent graphs."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.models.contextual import LCEAGP as ReferenceLCEA
from botorch.models.contextual import SACGP as ReferenceSAC
from botorch.models.kernels.contextual_lcea import LCEAKernel as ReferenceLCEAKernel
from botorch.models.kernels.contextual_sac import SACKernel as ReferenceSACKernel

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models._torch_backend_modules import build_kernel
from botorch_rs.models.contextual import LCEAGP, SACGP
from botorch_rs.models.kernels.contextual_lcea import LCEAKernel
from botorch_rs.models.kernels.contextual_sac import SACKernel

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]
DECOMP = {"first": [0, 1], "second": [2, 3]}


def _sync(reference, port):
    actual = dict(port.named_parameters())
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name not in actual:
                raise AssertionError(name)
            parameter.copy_(actual[name])


def _rbf(a, b, lengthscale):
    return (
        -0.5 * ((a / lengthscale).unsqueeze(-2) - (b / lengthscale).unsqueeze(-3)).square().sum(-1)
    ).exp()


def _manual(kernel, x, z):
    if isinstance(kernel, SACKernel):
        return sum(
            child.outputscale[..., None, None]
            * _rbf(x[..., dims], z[..., dims], kernel.base_kernel.lengthscale)
            for dims, child in zip(
                kernel.decomposition.values(), kernel.kernel_dict.values(), strict=True
            )
        )
    # The formula is independent of the module's context-covariance and
    # reshaping paths, so it detects ordering and cross-context errors.
    if kernel.train_embedding:
        emb = torch.cat(
            [
                layer.weight.index_select(-2, kernel.context_cat_feature[:, i])
                for i, layer in enumerate(kernel.emb_layers)
            ],
            -1,
        )
        if kernel.context_emb_feature.shape[-1]:
            emb = torch.cat(
                [
                    emb,
                    kernel.context_emb_feature.expand(
                        *kernel.batch_shape, *kernel.context_emb_feature.shape
                    ),
                ],
                -1,
            )
    else:
        emb = kernel.context_emb_feature.expand(
            *kernel.batch_shape, *kernel.context_emb_feature.shape
        )
    context = _rbf(emb, emb, kernel.task_covar_module.lengthscale)
    scales = kernel.outputscale_list
    if kernel.context_weight is not None:
        scales = scales * kernel.context_weight
    result = 0
    for i, left in enumerate(kernel.decomposition.values()):
        for j, right in enumerate(kernel.decomposition.values()):
            weight = context[..., i, j] * scales[..., i] * scales[..., j]
            result = result + weight[..., None, None] * _rbf(
                x[..., left], z[..., right], kernel.base_kernel.lengthscale
            )
    return result


def _gradients(value, leaves):
    gradients = torch.autograd.grad(
        value, leaves, create_graph=True, retain_graph=True, allow_unused=True
    )
    return [
        torch.zeros_like(leaf) if gradient is None else gradient
        for leaf, gradient in zip(leaves, gradients, strict=True)
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["sac", "learned", "fixed", "combined", "weighted"])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_kernel_official_values_first_derivatives_and_manual_hessians(device, family, batch):
    torch.manual_seed(713)
    like = torch.empty((), dtype=torch.double, device=device)
    kwargs = {}
    if family in ("fixed", "combined", "weighted"):
        kwargs["embs_feature_dict"] = {"first": [0.1, -0.2], "second": [0.4, 0.7]}
    if family == "fixed":
        kwargs["train_embedding"] = False
    if family == "combined":
        kwargs.update(cat_feature_dict={"first": [0, 1], "second": [1, 0]}, embs_dim_list=[1, 2])
    if family == "weighted":
        kwargs["context_weight_dict"] = {"first": 0.3, "second": 0.8}
    cls, ref_cls = (
        (SACKernel, ReferenceSACKernel) if family == "sac" else (LCEAKernel, ReferenceLCEAKernel)
    )
    kernel = cls(DECOMP, batch, backend="torch", **kwargs).to(like)
    # Keep embedding max-norm projection inactive for derivative oracles.
    with torch.no_grad():
        for name, parameter in kernel.named_parameters():
            if "weight" in name:
                parameter.fill_(0.15)
                parameter[..., -1, :].fill_(-0.2)
    x = torch.rand(*batch, 3, 4, dtype=torch.double, device=device, requires_grad=True)
    z = torch.rand(
        *((1,) if batch else ()), 4, 4, dtype=torch.double, device=device, requires_grad=True
    )
    leaves = (x, z, *kernel.parameters())
    actual = kernel(x, z)
    expected = _manual(kernel, x, z)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)
    first = _gradients(actual.square().sum(), leaves)
    first_manual = _gradients(expected.square().sum(), leaves)
    for a, b in zip(first, first_manual, strict=True):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=2e-10)
    second = _gradients(sum(value.sum() for value in first), leaves)
    second_manual = _gradients(sum(value.sum() for value in first_manual), leaves)
    for a, b in zip(second, second_manual, strict=True):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(kernel(x, x, diag=True), kernel(x, x).diagonal(dim1=-2, dim2=-1))
    # Official batched learned LCEA has separate sampled-weight tables and
    # broken multiple-feature lookup. Its corrected unbatched slice defines
    # each mathematical batch, without delegating runtime computations.
    if batch and family != "sac":
        for b in range(batch[0]):
            reference = ref_cls(DECOMP, torch.Size(), device=like.device, **kwargs).to(like)
            with torch.no_grad():
                for name, p in reference.named_parameters():
                    p.copy_(dict(kernel.named_parameters())[name][b])
            torch.testing.assert_close(
                actual[b], reference(x[b], z[0]).to_dense(), atol=2e-12, rtol=2e-11
            )
    else:
        reference = ref_cls(DECOMP, torch.Size(batch), device=like.device, **kwargs).to(like)
        _sync(reference, kernel)
        ref = reference(x, z).to_dense()
        torch.testing.assert_close(actual, ref, atol=2e-12, rtol=2e-11)
        reference_parameters = dict(reference.named_parameters())
        reference_leaves = (
            x,
            z,
            *(reference_parameters[name] for name, _ in kernel.named_parameters()),
        )
        ref_first = _gradients(ref.square().sum(), reference_leaves)
        for a, b in zip(first, ref_first, strict=True):
            torch.testing.assert_close(a, b, atol=3e-11, rtol=2e-9)
    assert all(parameter.device.type == device for parameter in kernel.parameters())
    with pytest.raises(UnsupportedError, match="last_dim_is_batch"):
        kernel(x, last_dim_is_batch=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["sac", "lcea"])
@pytest.mark.parametrize("fixed", [False, True])
def test_models_official_moments_query_gradient_and_full_training_graph(device, family, fixed):
    torch.manual_seed(38)
    x = torch.rand(6, 4, dtype=torch.double, device=device, requires_grad=True)
    y = torch.tensor(
        [[-0.8], [0.2], [0.9], [-0.3], [0.6], [-0.6]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    var = torch.full_like(y, 0.03, requires_grad=True) if fixed else None
    model_cls, ref_cls = (SACGP, ReferenceSAC) if family == "sac" else (LCEAGP, ReferenceLCEA)
    model = model_cls(x, y, var, DECOMP, backend="torch")
    assert isinstance(model, model_cls)
    if family == "lcea":
        with torch.no_grad():
            model.covar_module.emb_layers[0].weight.copy_(x.new_tensor([[0.1], [-0.3]]))
    reference = ref_cls(x.detach(), y.detach(), None if var is None else var.detach(), DECOMP).to(x)
    _sync(reference, model)
    query = torch.rand(3, 4, dtype=torch.double, device=device, requires_grad=True)
    posterior = model.posterior(query)
    ref = reference.posterior(query)
    torch.testing.assert_close(posterior.mean, ref.mean, atol=5e-10, rtol=2e-9)
    torch.testing.assert_close(
        posterior.distribution.covariance_matrix,
        ref.distribution.covariance_matrix,
        atol=5e-10,
        rtol=2e-9,
    )
    loss = posterior.mean.square().sum() + posterior.variance.sum()
    ref_loss = ref.mean.square().sum() + ref.variance.sum()
    torch.testing.assert_close(
        torch.autograd.grad(loss, query, retain_graph=True)[0],
        torch.autograd.grad(ref_loss, query)[0],
        atol=3e-9,
        rtol=2e-8,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # Repeated evaluations rebuild data and parameter graphs (no stale
    # context-covariance cache or constructor-time embedding views).
    leaves = (x, y, query, *(() if var is None else (var,)), *model.parameters())
    for _ in range(2):
        p = model.posterior(query)
        objective = (
            p.mean.square().sum()
            + p.variance.sum()
            + mll(model(*model.train_inputs), model.train_targets).sum()
        )
        first = _gradients(objective, leaves)
        second = _gradients(sum(g.square().sum() for g in first), leaves)
        assert all(torch.isfinite(g).all() for g in (*first, *second))
        assert first[0].abs().sum() > 0 and first[1].abs().sum() > 0
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    optimizer.zero_grad()
    (-mll(model(*model.train_inputs), model.train_targets).sum()).backward()
    optimizer.step()
    assert any(
        not torch.equal(parameter, original[name]) for name, parameter in model.named_parameters()
    )
    conditioned = model.condition_on_observations(
        query.detach()[:1], y.detach()[:1], noise=None if var is None else var.detach()[:1]
    )
    assert conditioned.posterior(query.detach()).mean.shape == (3, 1)
    assert conditioned._train_X.shape[-2] == model._train_X.shape[-2] + 1
    assert model._train_X.shape[-2] == 6
    clone = model_cls(
        x.detach().clone(),
        y.detach().clone(),
        None if var is None else var.detach().clone(),
        DECOMP,
        backend="torch",
    )
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone.posterior(query).mean, model.posterior(query).mean)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch,outputs", [((2,), 1), ((), 2), ((2,), 2)])
def test_sac_model_batch_and_output_augmentation(device, batch, outputs):
    torch.manual_seed(30)
    x = torch.rand(*batch, 5, 4, dtype=torch.double, device=device)
    y = torch.randn(*batch, 5, outputs, dtype=torch.double, device=device)
    model = SACGP(x, y, None, DECOMP, backend="torch")
    reference = ReferenceSAC(x, y, None, DECOMP).to(x)
    _sync(reference, model)
    query = torch.rand(*batch, 2, 4, dtype=torch.double, device=device, requires_grad=True)
    actual, expected = model.posterior(query), reference.posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-10, rtol=2e-9)
    torch.testing.assert_close(actual.variance, expected.variance, atol=3e-10, rtol=2e-9)
    gradient = torch.autograd.grad(actual.mean.sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_descriptor_conversion_preserves_shared_leaf_and_context_order(device):
    like = torch.empty((), dtype=torch.double, device=device)
    source = SACKernel(DECOMP, ())
    source.base_kernel.lengthscale = [0.3, 0.7]
    converted = build_kernel(source, 4, (), like)
    assert all(
        child.base_kernel is converted.base_kernel for child in converted.kernel_dict.values()
    )
    assert sum(name.endswith("raw_lengthscale") for name, _ in converted.named_parameters()) == 1
    x = torch.rand(3, 4, dtype=torch.double, device=device)
    torch.testing.assert_close(converted(x), _manual(converted, x, x))
    reordered = {"first": [1, 0], "second": [2, 3]}
    kernel = LCEAKernel(reordered, (), backend="torch").to(like)
    torch.testing.assert_close(kernel(x), _manual(kernel, x, x))
    with pytest.raises(ValueError, match="feature index"):
        converted(x[..., :2])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["sac", "lcea"])
def test_float32_migration_embedding_projection_and_auto_factory(device, family):
    cls = SACKernel if family == "sac" else LCEAKernel
    kernel = cls(DECOMP, (), backend="torch").to(dtype=torch.float32, device=device)
    x = torch.rand(3, 4, dtype=torch.float32, device=device, requires_grad=True)
    if family == "lcea":
        with torch.no_grad():
            kernel.emb_layers[0].weight.fill_(3)
    for _ in range(2):
        value = kernel(x).square().sum()
        first = torch.autograd.grad(value, (x, *kernel.parameters()), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), x)[0]
        assert torch.isfinite(second).all()
    if family == "lcea":
        assert bool((kernel.emb_layers[0].weight.norm(dim=-1) <= 1).all())
    assert all(buffer.device.type == device for buffer in kernel.buffers())
    # CUDA auto-selects this module; on CPU an actual differentiable
    # observation-noise argument must also select the Tensor model.
    y = torch.rand(3, 1, dtype=x.dtype, device=device)
    variance = torch.full_like(y, 0.04, requires_grad=True)
    model_cls = SACGP if family == "sac" else LCEAGP
    model = model_cls(x.detach(), y, train_Yvar=variance, decomposition=DECOMP)
    assert model._tensor_backend == "torch"
    gradient = torch.autograd.grad(model.posterior(x).mean.sum(), variance)[0]
    assert torch.isfinite(gradient).all()
