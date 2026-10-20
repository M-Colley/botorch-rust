"""Exact native posterior hyperparameter adjoints against official autograd."""

from __future__ import annotations

import pytest

from helpers import as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models import SingleTaskGP as BtGP  # noqa: E402
from botorch.models.transforms.input import Warp as BtWarp  # noqa: E402
from botorch.models.transforms.outcome import Standardize as BtStandardize  # noqa: E402
from botorch.settings import propagate_grads  # noqa: E402
from gpytorch.kernels import LinearKernel as BtLinear  # noqa: E402
from gpytorch.kernels import MaternKernel as BtMatern  # noqa: E402
from gpytorch.kernels import ScaleKernel as BtScale  # noqa: E402
from gpytorch.settings import fast_pred_var  # noqa: E402

from botorch_rs.kernels import LinearKernel, MaternKernel, ScaleKernel  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.models.transforms.input import Warp  # noqa: E402
from botorch_rs.models.transforms.outcome import Standardize  # noqa: E402


def model_pair(outputs=1, batch=(), tree=False, warp=False, fixed=False):
    generator = torch.Generator().manual_seed(3124)
    x = torch.rand(*batch, 7, 2, generator=generator, dtype=torch.double) * 0.8 + 0.1
    y = torch.rand(*batch, 7, outputs, generator=generator, dtype=torch.double)
    noise = torch.full_like(y, 0.03) if fixed else None
    aug = (*batch, *((outputs,) if outputs > 1 else ()))
    kwargs, other = {}, {}
    if tree:
        kwargs["covar_module"] = ScaleKernel(
            MaternKernel(nu=1.5, ard_num_dims=2, batch_shape=aug)
            + LinearKernel(ard_num_dims=2, batch_shape=aug),
            batch_shape=aug,
        )
        other["covar_module"] = BtScale(
            BtMatern(nu=1.5, ard_num_dims=2, batch_shape=torch.Size(aug))
            + BtLinear(ard_num_dims=2, batch_shape=torch.Size(aug)),
            batch_shape=torch.Size(aug),
        ).double()
    if warp:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        kwargs["input_transform"] = Warp(2, [0, 1], bounds=bounds)
        other["input_transform"] = BtWarp(2, [0, 1], bounds=bounds)
    actual = SingleTaskGP(
        x, y, noise, outcome_transform=Standardize(outputs, batch_shape=batch), **kwargs
    )
    reference = BtGP(
        x,
        y,
        noise,
        outcome_transform=BtStandardize(outputs, batch_shape=torch.Size(batch)),
        **other,
    )
    assert [name for name, _ in actual.named_parameters()] == [
        name for name, _ in reference.named_parameters()
    ]
    for (_, native), (_, value) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(native, value)
    return actual, reference


@pytest.mark.parametrize(
    "outputs,batch,tree,warp,fixed",
    [
        (1, (), False, False, False),
        (2, (), False, False, True),
        (2, (2,), False, False, False),
        (1, (), True, False, False),
        (2, (2,), True, False, True),
        (1, (), True, True, False),
    ],
)
@pytest.mark.parametrize("noisy", [False, True])
def test_posterior_raw_parameter_vjp_matches_official(outputs, batch, tree, warp, fixed, noisy):
    actual, reference = model_pair(outputs, batch, tree, warp, fixed)
    query = torch.tensor([[[[0.2, 0.6], [0.7, 0.3]]]], dtype=torch.double).expand(3, 1, 2, 2)
    # BoTorch normally detaches its training caches. The explicit native
    # pullback differentiates the full conditional, including the solve.
    with propagate_grads(True), fast_pred_var(False):
        if warp:
            # eval() stores transformed training inputs under no_grad. Rebuild
            # that graph explicitly to compare the full Warp sensitivity.
            reference.eval()
            reference.set_train_data(
                inputs=reference.input_transform(reference._original_train_inputs), strict=False
            )
        post = reference.posterior(query, observation_noise=noisy)
    covariance = post.distribution.covariance_matrix
    mean_bar = torch.linspace(-0.3, 0.7, post.mean.numel(), dtype=torch.double).reshape(
        post.mean.shape
    )
    cov_bar = torch.linspace(0.2, 0.9, covariance.numel(), dtype=torch.double).reshape(
        covariance.shape
    )
    objective = (post.mean * mean_bar).sum() + (covariance * cov_bar).sum()
    expected = torch.cat(
        [g.reshape(-1) for g in torch.autograd.grad(objective, tuple(reference.parameters()))]
    )
    gradient = actual.posterior_parameter_vjp(query, mean_bar, cov_bar, observation_noise=noisy)
    torch.testing.assert_close(gradient, expected, rtol=3e-7, atol=3e-8)


def test_parameter_vjp_preserves_array_kind(kind):
    actual, _ = model_pair()
    x = [[0.25, 0.65], [0.75, 0.35]]
    mean_bar, cov_bar = [[0.3], [0.7]], [[0.5, -0.2], [0.1, 0.6]]
    expected = actual.posterior_parameter_vjp(x, mean_bar, cov_bar)
    # The model's kind determines parameter output dtype, as for named_parameters.
    value = actual.posterior_parameter_vjp(
        as_kind(x, kind), as_kind(mean_bar, kind), as_kind(cov_bar, kind)
    )
    assert_close(value, expected, rtol=1e-12, atol=1e-12)


def test_parameter_vjp_validates_adjoint_shapes():
    actual, _ = model_pair()
    with pytest.raises(ValueError, match="adjoints must have shapes"):
        actual.posterior_parameter_vjp([[0.2, 0.6]], [0.2], [[0.3]])
