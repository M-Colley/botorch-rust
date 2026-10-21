"""Device-native modules: independent GPyTorch values and higher derivatives."""

# ruff: noqa: E402

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch_rs import _botorch_rs as native
from botorch_rs.constraints import GreaterThan, Interval, LessThan, Positive
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import NumericsWarning
from botorch_rs.kernels import (
    AdditiveKernel,
    CosineKernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PolynomialKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
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
from botorch_rs.models.transforms.input import ChainedInputTransform, Normalize, Warp
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
    assert list(dict(likelihood.named_parameters())) == ["noise_covar.raw_noise"]
    assert "noise_covar.raw_noise_constraint.lower_bound" in likelihood.state_dict()
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
@pytest.mark.parametrize("fixed", [False, True])
def test_likelihood_conditional_and_lazy_singular_latent_marginal(device, fixed):
    from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

    x = torch.zeros(3, 2, dtype=torch.double, device=device)
    observed_noise = x.new_full((3,), 0.2) if fixed else None
    module = build_likelihood(None, observed_noise, (), x)
    if not fixed:
        module.noise = 0.2
    mean = x.new_tensor([0.2, -0.1, 0.4]).requires_grad_()
    # This PSD latent law is singular: constructing the likelihood must not
    # attempt a Cholesky before observation noise makes it positive definite.
    covariance = x.new_ones(3, 3).requires_grad_()
    latent = TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution
    marginal = module(latent)
    expected = torch.distributions.MultivariateNormal(
        mean, covariance_matrix=covariance + torch.eye(3, dtype=x.dtype, device=device) * 0.2
    )
    target = x.new_tensor([0.1, 0.5, -0.2])
    torch.testing.assert_close(marginal.log_prob(target), expected.log_prob(target))
    actual_grad = torch.autograd.grad(
        marginal.log_prob(target), (mean, covariance), create_graph=True
    )
    reference_grad = torch.autograd.grad(
        expected.log_prob(target), (mean, covariance), create_graph=True
    )
    for a, b in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(a, b)
    conditional = module(mean)
    assert isinstance(conditional, torch.distributions.Normal)
    torch.testing.assert_close(conditional.mean, mean)
    torch.testing.assert_close(conditional.variance, x.new_full((3,), 0.2))
    with pytest.raises(UnsupportedError, match="Unsupported.*likelihood"):
        module(mean, invented=True)


@pytest.mark.parametrize("device", DEVICES)
def test_default_lognormal_prior_preserves_reference_factory_constants(device):
    from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior

    x = torch.zeros(4, 3, dtype=torch.double, device=device)
    actual = build_kernel(None, 3, (), x)
    reference = get_covar_module_with_dim_scaled_prior(3).to(x)
    expected = reference.lengthscale_prior.log_prob(reference.lengthscale).sum()
    torch.testing.assert_close(actual.log_prior(), expected, atol=2e-14, rtol=2e-14)
    grads = [
        torch.autograd.grad(value, parameter)[0]
        for value, parameter in [
            (actual.log_prior(), actual.raw_lengthscale),
            (expected, reference.raw_lengthscale),
        ]
    ]
    torch.testing.assert_close(*grads, atol=2e-14, rtol=2e-14)


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
@pytest.mark.parametrize("dtype,minimum", [(torch.float32, 1e-4), (torch.float64, 1e-6)])
def test_fixed_noise_inspection_floor_and_construction_warning(device, dtype, minimum):
    x = torch.zeros(3, 2, dtype=dtype, device=device)
    noise = x.new_tensor([0.0, minimum * 0.5, 0.1])
    with pytest.warns(NumericsWarning, match="Rounding small noise") as recorded:
        module = build_likelihood(None, noise, (), x)
    assert len(recorded) == 1
    torch.testing.assert_close(module.noise, noise.clamp_min(minimum))


@pytest.mark.parametrize("device", DEVICES)
def test_kernel_shared_aliases_prior_reduction_bounds_and_leaf_updates(device):
    x = torch.randn(4, 2, dtype=torch.double, device=device)
    source = RBFKernel(ard_num_dims=2, lengthscale_prior=LogNormalPrior(0, 1))
    module = build_kernel(source + source, 2, (), x)
    left, right = module.kernels
    assert left is right
    assert list(dict(module.named_parameters())) == ["kernels.0.raw_lengthscale"]
    torch.testing.assert_close(
        module.log_prior(), left.lengthscale_prior.log_prob(left.lengthscale).sum()
    )
    held = left.raw_lengthscale
    left.lengthscale = 0.7
    assert left.raw_lengthscale is right.raw_lengthscale is held
    torch.testing.assert_close(right.lengthscale, x.new_full((1, 2), 0.7))
    bounds = module.raw_parameter_bounds()
    assert bounds["kernels.0.raw_lengthscale"] == (-math.inf, math.inf)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("train_flag,eval_flag", [(False, True), (True, False), (False, False)])
def test_transform_flags_distinguish_stored_training_from_queries(device, train_flag, eval_flag):
    x = torch.tensor([[0.2, 2.0], [0.7, 3.0]], dtype=torch.double, device=device)
    source = Normalize(
        2,
        bounds=[[0.0, 1.0], [1.0, 5.0]],
        transform_on_train=train_flag,
        transform_on_eval=eval_flag,
    )
    module = build_input_transform(source, 2, x)
    expected = (x - x.new_tensor([0.0, 1.0])) / x.new_tensor([1.0, 4.0])
    module.eval()
    torch.testing.assert_close(module(x), expected if eval_flag else x)
    torch.testing.assert_close(module.preprocess_transform(x), expected if train_flag else x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["Normalize", "Warp"])
@pytest.mark.parametrize("train_flag,reverse", [(False, False), (True, True), (False, True)])
def test_input_transform_initial_fit_honors_disabled_and_reverse_maps(
    device, family, train_flag, reverse
):
    from botorch.models.transforms import input as reference_input

    x = torch.tensor([[0.2, 0.3], [0.6, 0.8], [0.4, 0.7]], dtype=torch.double, device=device)
    kwargs = {"d": 2, "transform_on_train": train_flag, "reverse": reverse}
    if family == "Warp":
        kwargs["indices"] = [0, 1]
    source = {"Normalize": Normalize, "Warp": Warp}[family](**kwargs)
    module = build_input_transform(source, 2, x)
    reference = getattr(reference_input, family)(**kwargs).to(x)
    with torch.no_grad():
        reference(x)
    reference.eval()
    query = x * 0.5 + 0.2
    torch.testing.assert_close(module(query), reference(query))


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
@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("chain", [False, True])
def test_warp_and_chain_full_training_parameter_graphs(device, batch_shape, chain):
    from botorch.models.transforms.input import ChainedInputTransform as ReferenceChain
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from botorch.models.transforms.input import Warp as ReferenceWarp

    x = (
        torch.tensor(
            [[0.1, 0.4, 0.2], [0.4, 0.9, 0.8], [0.8, 0.2, 0.4], [0.7, 0.5, 0.6]],
            dtype=torch.double,
            device=device,
        )
        .expand(*batch_shape, 4, 3)
        .clone()
        .requires_grad_()
    )
    source_warp = Warp(
        3, [0, 2], batch_shape=batch_shape, concentration0_prior=LogNormalPrior(0, 0.7)
    ).to(x)
    source_warp.initialize(concentration0=0.9, concentration1=1.3)
    reference_warp = ReferenceWarp(3, [0, 2], batch_shape=torch.Size(batch_shape)).to(x)
    with torch.no_grad():
        reference_warp.concentration0.fill_(0.9)
        reference_warp.concentration1.fill_(1.3)
    source = (
        ChainedInputTransform(normalize=Normalize(3), warp=source_warp) if chain else source_warp
    )
    reference = (
        ReferenceChain(normalize=ReferenceNormalize(3), warp=reference_warp).to(x)
        if chain
        else reference_warp
    )
    actual = build_input_transform(source, 3, x)
    actual.train()
    expected = reference(x)
    observed = actual(x)
    torch.testing.assert_close(observed, expected)
    weight = torch.linspace(-0.4, 0.9, x.numel(), dtype=x.dtype, device=device).reshape_as(x)
    first = [
        torch.autograd.grad((value * weight).sum(), [x, *module.parameters()], create_graph=True)
        for value, module in [(observed, actual), (expected, reference)]
    ]
    for a, b in zip(*first, strict=True):
        torch.testing.assert_close(a, b)
    second = [
        torch.autograd.grad(sum(g.square().sum() for g in grads), [x, *module.parameters()])
        for grads, module in zip(first, [actual, reference], strict=True)
    ]
    for a, b in zip(*second, strict=True):
        torch.testing.assert_close(a, b)
    actual.fit(x)
    actual.eval()
    expected_prior_shape = batch_shape
    assert actual.log_prior().shape == expected_prior_shape
    warp = actual.warp if chain else actual
    assert warp.raw_parameter_bounds()["concentration0"][0] == pytest.approx(1e-4)
    query = x.detach() * 0.6 + 0.2
    torch.testing.assert_close(actual.untransform(actual(query)), query, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kind",
    [
        "rbf",
        "matern12",
        "matern32",
        "matern52",
        "tree",
        "product",
        "rq",
        "periodic",
        "cosine",
        "linear",
        "polynomial",
        "polynomial_fractional",
    ],
)
@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("feature_batch", [False, True])
def test_native_kernel_graph_reference_first_and_second_derivatives(
    device, kind, batch_shape, feature_batch
):
    if not hasattr(native, "tensor_kernel"):
        pytest.skip("native Tensor dispatcher not built in this artifact")
    kwargs = {"ard_num_dims": 2, "active_dims": [0, 2], "batch_shape": batch_shape}
    if kind in ("tree", "product"):
        left, right = RBFKernel(**kwargs), MaternKernel(nu=2.5, **kwargs)
        source = ScaleKernel(
            AdditiveKernel(left, right) if kind == "tree" else left * right,
            batch_shape=batch_shape,
        )
        ref_left, ref_right = (
            gpytorch.kernels.RBFKernel(**kwargs),
            gpytorch.kernels.MaternKernel(nu=2.5, **kwargs),
        )
        ref = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.AdditiveKernel(ref_left, ref_right)
            if kind == "tree"
            else ref_left * ref_right,
            batch_shape=torch.Size(batch_shape),
        )
    elif kind == "rbf":
        source, ref = RBFKernel(**kwargs), gpytorch.kernels.RBFKernel(**kwargs)
    elif kind in ("rq", "periodic", "cosine", "linear", "polynomial", "polynomial_fractional"):
        classes = {
            "rq": RQKernel,
            "periodic": PeriodicKernel,
            "cosine": CosineKernel,
            "linear": LinearKernel,
            "polynomial": PolynomialKernel,
            "polynomial_fractional": PolynomialKernel,
        }
        extra = (
            {"power": 2.5 if kind == "polynomial_fractional" else 3}
            if kind.startswith("polynomial")
            else {}
        )
        source = classes[kind](**extra, **kwargs)
        ref = getattr(gpytorch.kernels, classes[kind].__name__)(**extra, **kwargs)
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
    def reference_covariance(left, right, parameters=None):
        # PolynomialKernel's model-batch offset misses its extra feature axis
        # in GPyTorch. Evaluate each individual feature through its unchanged
        # forward formula, then stack according to feature-batch semantics.
        if kind.startswith("polynomial") and feature_batch:

            class PerFeature(torch.nn.Module):
                def __init__(self, kernel):
                    super().__init__()
                    self.kernel = kernel

                def forward(self, a, b):
                    return torch.stack(
                        [self.kernel.forward(a[..., j : j + 1], b[..., j : j + 1]) for j in [0, 2]],
                        dim=-3,
                    )

            wrapper = PerFeature(ref)
            if parameters is None:
                return wrapper(left, right)
            return torch.func.functional_call(
                wrapper,
                {f"kernel.{name}": value for name, value in parameters.items()},
                (left, right),
            )
        with gpytorch.settings.lazily_evaluate_kernels(False):
            if parameters is None:
                return ref(left, right, last_dim_is_batch=feature_batch).to_dense()
            return torch.func.functional_call(
                ref, parameters, (left, right), {"last_dim_is_batch": feature_batch}
            ).to_dense()

    expected = reference_covariance(x, z)
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
        shifted_output = reference_covariance(shifted[0], shifted[1], parameters)
        finite.append(torch.autograd.grad((shifted_output * weight).sum(), shifted))
    for actual_hvp, low, high in zip(second, *finite, strict=True):
        torch.testing.assert_close(actual_hvp, (high - low) / (2 * epsilon), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "family", ["rbf", "matern", "rq", "periodic", "cosine", "linear", "polynomial"]
)
@pytest.mark.parametrize("feature_batch", [False, True])
def test_kernel_diagonal_matches_full_cross_covariance_and_self_gradients(
    device, dtype, family, feature_batch
):
    classes = {
        "rbf": RBFKernel,
        "matern": MaternKernel,
        "rq": RQKernel,
        "periodic": PeriodicKernel,
        "cosine": CosineKernel,
        "linear": LinearKernel,
        "polynomial": PolynomialKernel,
    }
    kwargs = {"ard_num_dims": 2, "active_dims": [0, 2], "batch_shape": (2,)}
    if family == "polynomial":
        kwargs["power"] = 3
    x = torch.rand(2, 4, 3, dtype=dtype, device=device, requires_grad=True)
    z = torch.rand_like(x).requires_grad_()
    module = build_kernel(classes[family](**kwargs), 3, (2,), x)
    with torch.no_grad():
        # Different per-batch values expose accidental alignment of a model
        # batch axis with an equal-sized feature axis.
        for parameter in module.parameters():
            parameter.copy_(
                torch.linspace(-0.2, 0.5, parameter.numel(), dtype=dtype, device=device).reshape_as(
                    parameter
                )
            )
    actual = module(x, z, diag=True, last_dim_is_batch=feature_batch)
    expected = module(x, z, last_dim_is_batch=feature_batch).diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(actual, expected)
    inputs = [x, z, *module.parameters()]
    first = [
        torch.autograd.grad(value.sum(), inputs, create_graph=True) for value in (actual, expected)
    ]
    for a, b in zip(*first, strict=True):
        torch.testing.assert_close(a, b)
    self_covariance = module(x, last_dim_is_batch=feature_batch)
    self_diagonal = module(x, diag=True, last_dim_is_batch=feature_batch)
    torch.testing.assert_close(self_diagonal, self_covariance.diagonal(dim1=-2, dim2=-1))
    gradient = torch.autograd.grad(
        self_covariance.sum() + self_diagonal.sum(), x, create_graph=True
    )[0]
    hessian = torch.autograd.grad(gradient.sum(), x)[0]
    assert torch.isfinite(gradient).all() and torch.isfinite(hessian).all()
    with pytest.raises(ValueError, match="diag|same|equal"):
        module(x, z[..., :3, :], diag=True, last_dim_is_batch=feature_batch)
