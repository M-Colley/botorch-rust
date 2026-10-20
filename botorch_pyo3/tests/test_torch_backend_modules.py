"""Device-native modules: independent GPyTorch values and higher derivatives."""

# ruff: noqa: E402

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch_rs import _botorch_rs as native
from botorch_rs.constraints import GreaterThan, Interval, LessThan, Positive
from botorch_rs.kernels import AdditiveKernel, MaternKernel, RBFKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean, ZeroMean
from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    TorchPrior,
    build_input_transform,
    build_kernel,
    build_likelihood,
    build_mean,
    build_outcome_transform,
)
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.priors import (
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    LogNormalPrior,
    NormalPrior,
    SmoothedBoxPrior,
    UniformPrior,
)
from botorch_rs.utils.types import DEFAULT

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "constraint",
    [
        Positive(),
        GreaterThan(0.1),
        LessThan(2.0),
        Interval(-0.5, 1.5),
        Interval(-0.5, 1.5, transform=None),
    ],
)
def test_constraint_values_higher_derivatives_and_inverse(device, constraint):
    raw = torch.tensor([-0.4, 0.7], dtype=torch.double, device=device, requires_grad=True)
    module = TorchConstraint(constraint, raw)
    reference = getattr(gpytorch.constraints, type(constraint).__name__)
    if type(constraint) is Positive:
        ref = reference()
    elif type(constraint) is Interval:
        ref = reference(
            constraint.lower_bound,
            constraint.upper_bound,
            transform=None if not constraint.enforced else torch.sigmoid,
        )
    else:
        ref = reference(
            constraint.upper_bound if type(constraint) is LessThan else constraint.lower_bound
        )
    ref = ref.to(raw)
    torch.testing.assert_close(module.transform(raw), ref.transform(raw))
    torch.testing.assert_close(module.inverse_transform(module.transform(raw)), raw)
    assert torch.autograd.gradcheck(module.transform, (raw,))
    assert torch.autograd.gradgradcheck(module.transform, (raw,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name,args",
    [
        ("Normal", (0.2, 0.7)),
        ("LogNormal", (-0.2, 0.8)),
        ("Gamma", (2.0, 0.7)),
        ("Uniform", (0.1, 2.0)),
        ("HalfNormal", (0.8,)),
        ("HalfCauchy", (0.8,)),
        ("SmoothedBox", (0.2, 0.8, 0.1)),
    ],
)
def test_prior_matches_reference_with_first_and_second_derivatives(device, name, args):
    classes = {
        "Normal": NormalPrior,
        "LogNormal": LogNormalPrior,
        "Gamma": GammaPrior,
        "Uniform": UniformPrior,
        "HalfNormal": HalfNormalPrior,
        "HalfCauchy": HalfCauchyPrior,
        "SmoothedBox": SmoothedBoxPrior,
    }
    x = torch.tensor([0.15, 0.45, 1.1], dtype=torch.double, device=device, requires_grad=True)
    module = TorchPrior(classes[name](*args), x)
    # Construct with tensor hyperparameters: several GPyTorch transformed
    # priors retain CPU/f32 cached distribution state after Module.to().
    reference = getattr(gpytorch.priors, f"{name}Prior")(*module.hyperparameters.unbind())
    value = module.log_prob(x)
    expected = reference.log_prob(x.unsqueeze(-1) if name == "SmoothedBox" else x)
    torch.testing.assert_close(value, expected)
    for output in (value, expected):
        # Uniform's constant log-density has no graph in the reference either.
        if output.requires_grad:
            grad = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
            hessian = torch.autograd.grad(grad.sum(), x)[0]
            if output is value:
                actual = (grad, hessian)
            else:
                for a, b in zip(actual, (grad, hessian), strict=True):
                    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_mean_and_noise_parameters_are_persistent_real_modules(device, batch_shape):
    x = torch.randn(*batch_shape, 5, 3, dtype=torch.double, device=device)
    mean = build_mean(
        ConstantMean(batch_shape=batch_shape, constant_prior=NormalPrior(0, 1)), batch_shape, x
    )
    likelihood = build_likelihood(
        GaussianLikelihood(batch_shape=batch_shape, noise_prior=GammaPrior(2, 3)),
        None,
        batch_shape,
        x,
    )
    held_mean, held_noise = mean.raw_constant, likelihood.raw_noise
    mean.constant = 0.3
    likelihood.noise = 0.2
    assert mean.raw_constant is held_mean and likelihood.raw_noise is held_noise
    assert all(isinstance(p, torch.nn.Parameter) for p in [held_mean, held_noise])
    torch.testing.assert_close(mean(x), x.new_full((*batch_shape, 5), 0.3))
    torch.testing.assert_close(likelihood.noise_for(5), x.new_full((*batch_shape, 5), 0.2))
    assert mean.log_prior().shape == likelihood.log_prior().shape == batch_shape
    torch.autograd.grad(
        (mean(x).square().sum() + likelihood.noise.square().sum()),
        [held_mean, held_noise],
        create_graph=True,
    )
    optimizer = torch.optim.Adam([*mean.parameters(), *likelihood.parameters()], lr=0.01)
    optimizer.zero_grad()
    loss = (
        mean(x).square().sum()
        + likelihood.noise.square().sum()
        - mean.log_prior().sum()
        - likelihood.log_prior().sum()
    )
    loss.backward()
    optimizer.step()
    assert mean.raw_constant is held_mean and likelihood.raw_noise is held_noise
    zero = build_mean(ZeroMean(batch_shape=batch_shape), batch_shape, x)
    assert list(zero.parameters()) == []
    assert not zero(x).any()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [1, 2])
def test_default_modules_and_fixed_noise_layout(device, m):
    x = torch.randn(2, 5, 3, dtype=torch.double, device=device)
    aug = (2,) if m == 1 else (2, m)
    kernel = build_kernel(None, 3, aug, x)
    likelihood = build_likelihood(None, None, aug, x)
    assert kernel.raw_lengthscale.shape == (*aug, 1, 3)
    assert likelihood.raw_noise.shape == (*aug, 1)
    # The defaults start at the modes of BoTorch's dimension-scaled priors.
    torch.testing.assert_close(
        kernel.lengthscale,
        kernel.lengthscale.new_full((*aug, 1, 3), math.exp(math.sqrt(2) + 0.5 * math.log(3) - 3)),
        rtol=1e-6,
        atol=1e-8,
    )
    assert likelihood.log_prior().shape == aug
    noise = torch.full((2, 5, m), 0.03, dtype=torch.double, device=device, requires_grad=True)
    expected = noise.squeeze(-1) if m == 1 else noise.movedim(-1, -2)
    fixed = build_likelihood(None, expected, aug, x)
    torch.testing.assert_close(fixed.noise, expected)
    assert not fixed.noise.requires_grad
    with pytest.raises(ValueError, match="explicit observation noise"):
        fixed.noise_for(3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [1, 2])
def test_standardize_training_graph_and_independent_covariance_untransform(device, m):
    from botorch.models.transforms.outcome import Standardize as ReferenceStandardize

    y = (
        torch.tensor(
            [[0.2, 0.8], [0.7, -0.2], [1.2, 0.1], [-0.1, -0.5]], dtype=torch.double, device=device
        )[:, :m]
        .clone()
        .requires_grad_()
    )
    module = build_outcome_transform(DEFAULT, m, y)
    reference = ReferenceStandardize(m).to(y)
    module.train()
    actual, _ = module(y)
    expected, _ = reference(y)
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(actual.numel(), dtype=y.dtype, device=y.device).reshape_as(actual)
    grads = [
        torch.autograd.grad((value * weights).sum(), y, create_graph=True, retain_graph=True)[0]
        for value in (actual, expected)
    ]
    torch.testing.assert_close(*grads)
    second = [torch.autograd.grad(grad.square().sum(), y)[0] for grad in grads]
    torch.testing.assert_close(*second)
    module.fit(y)
    module.eval()
    mean = y.new_zeros(3, m)
    covariance = torch.eye(3, dtype=y.dtype, device=y.device).expand(m, 3, 3)
    unmean, uncov = module.untransform_moments(mean, covariance)
    torch.testing.assert_close(unmean, module.means.expand(3, m))
    torch.testing.assert_close(
        uncov.diagonal(dim1=-2, dim2=-1), module.stdvs.square().movedim(-1, -2).expand(m, 3)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("indices", [None, [0, 2]])
def test_normalize_training_graph_and_frozen_query_bounds(device, indices):
    from botorch.models.transforms.input import Normalize as ReferenceNormalize

    x = torch.tensor(
        [[0.2, 0.8, 0.1], [0.7, -0.2, 0.9], [1.2, 0.1, 0.4], [-0.1, -0.5, 0.5]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    module = build_input_transform(Normalize(3, indices=indices), 3, x)
    reference = ReferenceNormalize(3, indices=indices).to(x)
    module.train()
    actual, expected = module(x), reference(x)
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(x.numel(), dtype=x.dtype, device=x.device).reshape_as(x)
    grads = [
        torch.autograd.grad((value * weights).sum(), x, create_graph=True, retain_graph=True)[0]
        for value in (actual, expected)
    ]
    torch.testing.assert_close(*grads)
    second = [torch.autograd.grad(grad.square().sum(), x)[0] for grad in grads]
    torch.testing.assert_close(*second)
    module.fit(x)
    module.eval()
    before = module.offset.clone()
    query = x.detach() + 5
    torch.testing.assert_close(module.untransform(module(query)), query)
    torch.testing.assert_close(before, module.offset)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["rbf", "matern12", "matern32", "matern52", "tree"])
@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("feature_batch", [False, True])
def test_native_kernel_graph_reference_first_and_second_derivatives(
    device, kind, batch_shape, feature_batch
):
    if not hasattr(native, "tensor_kernel"):
        pytest.skip("native Tensor dispatcher not built in this artifact")
    kwargs = {"ard_num_dims": 2, "active_dims": [0, 2], "batch_shape": batch_shape}
    if kind == "tree":
        source = ScaleKernel(
            AdditiveKernel(RBFKernel(**kwargs), MaternKernel(nu=2.5, **kwargs)),
            batch_shape=batch_shape,
        )
        ref = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.AdditiveKernel(
                gpytorch.kernels.RBFKernel(**kwargs),
                gpytorch.kernels.MaternKernel(nu=2.5, **kwargs),
            ),
            batch_shape=torch.Size(batch_shape),
        )
    elif kind == "rbf":
        source, ref = RBFKernel(**kwargs), gpytorch.kernels.RBFKernel(**kwargs)
    else:
        nu = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}[kind]
        source, ref = MaternKernel(nu=nu, **kwargs), gpytorch.kernels.MaternKernel(nu=nu, **kwargs)
    x = torch.rand(*batch_shape, 4, 3, dtype=torch.double, device=device, requires_grad=True)
    z = (torch.rand(*batch_shape, 3, 3, dtype=torch.double, device=device) + 0.4).requires_grad_()
    module = build_kernel(source, 3, batch_shape, x)
    ref = ref.to(x)
    actual = module(x, z, last_dim_is_batch=feature_batch)
    # Eager covariance avoids GPyTorch's known composite feature-batch lazy
    # shape bug, without changing the reference kernel calculation.
    with gpytorch.settings.lazily_evaluate_kernels(False):
        expected = ref(x, z, last_dim_is_batch=feature_batch).to_dense()
    torch.testing.assert_close(actual, expected)
    weight = torch.linspace(-0.7, 0.9, actual.numel(), dtype=x.dtype, device=device).reshape_as(
        actual
    )
    actual_inputs = [x, z, *module.parameters()]
    expected_inputs = [x, z, *ref.parameters()]
    first = [
        torch.autograd.grad((value * weight).sum(), inputs, create_graph=True)
        for value, inputs in [(actual, actual_inputs), (expected, expected_inputs)]
    ]
    for a, b in zip(*first, strict=True):
        torch.testing.assert_close(a, b)
    directions = [
        torch.ones_like(value) * (0.1 + i * 0.03) for i, value in enumerate(actual_inputs)
    ]
    second = torch.autograd.grad(
        sum((g * direction).sum() for g, direction in zip(first[0], directions, strict=True)),
        actual_inputs,
    )
    # Torch cdist's backward used by GPyTorch's Matérn kernels has no second
    # derivative. A central difference of its genuine first gradients supplies
    # an independent Hessian-vector oracle rather than a second copy of the
    # port's covariance formula.
    epsilon = 1e-5
    finite = []
    for sign in (-1, 1):
        shifted = [
            value.detach().clone().add(direction, alpha=sign * epsilon).requires_grad_()
            for value, direction in zip(expected_inputs, directions, strict=True)
        ]
        parameters = dict(zip(dict(ref.named_parameters()), shifted[2:], strict=True))
        with gpytorch.settings.lazily_evaluate_kernels(False):
            shifted_output = torch.func.functional_call(
                ref, parameters, (shifted[0], shifted[1]), {"last_dim_is_batch": feature_batch}
            ).to_dense()
        finite.append(torch.autograd.grad((shifted_output * weight).sum(), shifted))
    for actual_hvp, low, high in zip(second, *finite, strict=True):
        torch.testing.assert_close(actual_hvp, (high - low) / (2 * epsilon), atol=2e-6, rtol=2e-5)
