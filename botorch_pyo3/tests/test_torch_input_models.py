"""Public port transforms inside models preserve actual state and graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import io

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.kernels import RBFKernel
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.input import (
    AffineInputTransform,
    AppendFeatures,
    BatchBroadcastedInputTransform,
    ChainedInputTransform,
    FilterFeatures,
    InputPerturbation,
    InputStandardize,
    InteractionFeatures,
    LearnedFeatureImputation,
    Log10,
    Normalize,
    NumericToCategoricalEncoding,
    OneHotToNumeric,
    Round,
    Warp,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device):
    X = torch.tensor(
        [[0.21, 0.71, 0.31], [0.43, 0.27, 0.62], [0.64, 0.83, 0.48], [0.82, 0.41, 0.79]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    Y = X.new_tensor([[-0.4], [0.8], [0.1], [-0.2]]).requires_grad_()
    Q = X.new_tensor([[0.33, 0.54, 0.44], [0.71, 0.65, 0.58]]).requires_grad_()
    return X, Y, Q


def _case(family, X, Q):
    if family == "normalize":
        transform = Normalize(3)

        def mapping(value):
            low = X.amin(-2, keepdim=True)
            return (value - low) / (X.amax(-2, keepdim=True) - low)
    elif family == "standardize":
        transform = InputStandardize(3)

        def mapping(value):
            return (value - X.mean(-2, keepdim=True)) / X.std(-2, keepdim=True)
    elif family == "affine":
        transform = AffineInputTransform(
            3, X.new_tensor([[0.7, 1.3, 0.9]]), X.new_tensor([[0.1, -0.2, 0.3]])
        )

        def mapping(value):
            return (value - X.new_tensor([[0.1, -0.2, 0.3]])) / X.new_tensor([[0.7, 1.3, 0.9]])
    elif family == "warp":
        transform = Warp(3, [2, 0], bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        transform.concentration0 = torch.nn.Parameter(X.new_tensor([1.3, 1.1]))
        transform.concentration1 = torch.nn.Parameter(X.new_tensor([0.8, 1.2]))

        def mapping(value):
            selected = value[..., [2, 0]] * (1 - 2 * transform._eps) + transform._eps
            result = value.clone()
            result[..., [2, 0]] = 1 - (1 - selected.pow(transform.concentration1)).pow(
                transform.concentration0
            )
            return result
    elif family == "round":
        transform = Round([0], approximate=True, tau=0.3)

        def mapping(value):
            result = value.clone()
            base = value[..., 0].floor()
            result[..., 0] = base + (1 + ((value[..., 0] - base - 0.5) / 0.3).tanh()) / 2
            return result
    elif family == "log":
        transform = Log10([2, 0])

        def mapping(value):
            result = value.clone()
            result[..., [2, 0]] = value[..., [2, 0]].log10()
            return result
    elif family == "filter":
        transform = FilterFeatures([2, 0])

        def mapping(value):
            return value[..., [2, 0]]
    elif family == "interaction":
        transform = InteractionFeatures([2, 0])

        def mapping(value):
            return torch.cat((value, (value[..., 2] * value[..., 0]).unsqueeze(-1)), -1)
    elif family == "append":
        # A single appended feature is a row-preserving training transform.
        feature = X.new_tensor([[0.23]]).requires_grad_()
        transform = AppendFeatures(feature_set=feature, transform_on_train=True)

        def mapping(value):
            return torch.cat((value, feature.expand(*value.shape[:-1], 1)), -1)
    elif family == "perturb":
        delta = X.new_tensor([[0.02, -0.03, 0.01], [-0.01, 0.04, -0.02]]).requires_grad_()
        transform = InputPerturbation(delta)

        def mapping(value):
            return value if value is X else (value.unsqueeze(-2) + delta).flatten(-3, -2)
    elif family == "impute":
        X = torch.cat((X, X.new_tensor([[0.0], [1.0], [0.0], [1.0]])), -1).detach().requires_grad_()
        Q = torch.cat((Q, Q.new_tensor([[0.0], [1.0]])), -1).detach().requires_grad_()
        transform = LearnedFeatureImputation(
            {0: [0, 2], 1: [1, 2]}, 3, bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )

        def mapping(value):
            result = value.clone()
            raw = transform.raw_imputation_values.reshape(2, 4).sigmoid()
            result[..., 1] = torch.where(value[..., -1] == 0, raw[0, 1], value[..., 1])
            result[..., 0] = torch.where(value[..., -1] == 1, raw[1, 0], value[..., 0])
            return result
    elif family == "onehot":
        X = (
            torch.cat(
                (X[..., :1], X.new_tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])), -1
            )
            .detach()
            .requires_grad_()
        )
        Q = (
            torch.cat((Q[..., :1], Q.new_tensor([[0.0, 1.0], [1.0, 0.0]])), -1)
            .detach()
            .requires_grad_()
        )
        transform = OneHotToNumeric(3, {1: 2})

        def mapping(value):
            return torch.cat(
                (value[..., :1], value[..., 1:].argmax(-1, keepdim=True).to(value)), -1
            )
    elif family == "categorical":
        X = (
            torch.cat((X[..., :1], X.new_tensor([[0.0], [1.0], [0.0], [1.0]])), -1)
            .detach()
            .requires_grad_()
        )
        Q = torch.cat((Q[..., :1], Q.new_tensor([[1.0], [0.0]])), -1).detach().requires_grad_()

        def encoder(value):
            return torch.nn.functional.one_hot(value, 2)

        transform = NumericToCategoricalEncoding(2, {1: 2}, {1: encoder})

        def mapping(value):
            return torch.cat((value[..., :1], encoder(value[..., 1].long()).to(value)), -1)
    elif family == "chain":
        transform = ChainedInputTransform(
            keep=FilterFeatures([2, 0]), interactions=InteractionFeatures()
        )

        def mapping(value):
            selected = value[..., [2, 0]]
            return torch.cat((selected, selected.prod(-1, keepdim=True)), -1)
    elif family == "broadcast":
        X = torch.stack((X, X * 0.9)).detach().requires_grad_()
        Q = torch.stack((Q, Q * 1.1)).detach().requires_grad_()
        transform = BatchBroadcastedInputTransform([Normalize(3), InputStandardize(3)])

        def mapping(value):
            a = (value[0] - X[0].amin(-2, keepdim=True)) / (
                X[0].amax(-2, keepdim=True) - X[0].amin(-2, keepdim=True)
            )
            b = (value[1] - X[1].mean(-2, keepdim=True)) / X[1].std(-2, keepdim=True)
            return torch.stack((a, b))
    else:
        raise AssertionError(family)
    return transform, X, Q, mapping


def _manual(model, X, Y, Q):
    def kernel(A, B):
        difference = (A.unsqueeze(-2) - B.unsqueeze(-3)) / model.covar_module.lengthscale.unsqueeze(
            -2
        )
        return (-0.5 * difference.square().sum(-1)).exp()

    K = kernel(X, X) + torch.diag_embed(model.likelihood.noise.expand(*X.shape[:-2], X.shape[-2]))
    cross = kernel(X, Q)
    mean = model.mean_module.constant
    centered = Y.squeeze(-1) - mean.unsqueeze(-1)
    prediction = mean.unsqueeze(-1) + (
        cross.transpose(-1, -2) @ torch.linalg.solve(K, centered.unsqueeze(-1))
    ).squeeze(-1)
    covariance = kernel(Q, Q) - cross.transpose(-1, -2) @ torch.linalg.solve(K, cross)
    return prediction.unsqueeze(-1), covariance


def _derivatives(mean, covariance, leaves):
    loss = mean.sin().sum() + 0.2 * covariance.square().sum()
    first = torch.autograd.grad(loss, leaves, create_graph=True, retain_graph=True)
    direction = sum((i + 1) * value.square().sum() for i, value in enumerate(first))
    second = torch.autograd.grad(direction, leaves, retain_graph=True)
    return first, second


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family",
    [
        "normalize",
        "standardize",
        "affine",
        "warp",
        "round",
        "log",
        "filter",
        "interaction",
        "append",
        "perturb",
        "impute",
        "onehot",
        "categorical",
        "chain",
        "broadcast",
    ],
)
def test_model_transform_gaussian_law_and_full_graphs(device, family, monkeypatch):
    X, Y, Q = _data(device)
    transform, X, Q, mapping = _case(family, X, Q)
    if X.ndim == 3:
        Y = Y.expand(*X.shape[:-2], *Y.shape).clone().detach().requires_grad_()
    width = mapping(X).shape[-1] if family != "impute" else 4

    def forbidden(*args, **kwargs):
        raise AssertionError("Model input transforms must remain on device")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        model = SingleTaskGP(
            X,
            Y,
            input_transform=transform,
            covar_module=RBFKernel(ard_num_dims=width),
            outcome_transform=None,
            backend="torch",
        )
        posterior = model.posterior(Q)
        leaves = (X, Y, Q, *model.parameters())
        actual = _derivatives(posterior.mean, posterior.distribution.covariance_matrix, leaves)
    expected = _manual(model, mapping(X), Y, mapping(Q))
    torch.testing.assert_close(posterior.mean, expected[0], atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(
        posterior.distribution.covariance_matrix, expected[1], atol=2e-12, rtol=2e-11
    )
    reference = _derivatives(*expected, leaves)
    for actual_order, expected_order in zip(actual, reference, strict=True):
        for result, oracle in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(result, oracle, atol=5e-9, rtol=5e-8)
    assert model.input_transform._source is transform
    if family in ("warp", "impute"):
        for name, parameter in transform.named_parameters():
            assert dict(model.input_transform.named_parameters())[name] is parameter


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["filter", "interaction", "chain"])
def test_default_kernel_uses_actual_transformed_training_width(device, family):
    X, Y, Q = _data(device)
    transform, X, Q, mapping = _case(family, X, Q)
    model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None, backend="torch")
    assert model.covar_module.lengthscale.shape[-1] == mapping(X).shape[-1]
    torch.testing.assert_close(
        model.posterior(Q).mean, _manual(model, mapping(X), Y, mapping(Q))[0]
    )


@pytest.mark.parametrize("device", DEVICES)
def test_append_query_width_differs_from_already_augmented_training_width(device):
    X, Y, Q = _data(device)
    features = X.new_tensor([[0.12], [0.36]]).requires_grad_()
    augmented = torch.cat((X, X.new_full((4, 1), 0.23)), -1)
    transform = AppendFeatures(feature_set=features)
    model = SingleTaskGP(
        augmented, Y, input_transform=transform, outcome_transform=None, backend="torch"
    )
    posterior = model.posterior(Q)
    expanded = torch.cat((Q.unsqueeze(-2).expand(2, 2, 3), features.expand(2, 2, 1)), -1).flatten(
        -3, -2
    )
    mean, covariance = _manual(model, augmented, Y, expanded)
    torch.testing.assert_close(posterior.mean, mean)
    torch.testing.assert_close(posterior.distribution.covariance_matrix, covariance)
    _derivatives(
        posterior.mean,
        posterior.distribution.covariance_matrix,
        (X, Y, Q, features, *model.parameters()),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_actual_subclass_hooks_frozen_statistics_and_serialized_parameter_identity(device):
    X, Y, Q = _data(device)

    class ShiftedNormalize(Normalize):
        def transform(self, value):
            return super().transform(value) + 0.17

    source = ShiftedNormalize(3)
    source(X.detach() * 0.8)
    source.eval()
    old = source.coefficient.clone()
    transform = ChainedInputTransform(
        normalize=source,
        warp=Warp(3, [0, 2], bounds=X.new_tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])),
    )
    model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None, backend="torch")
    torch.testing.assert_close(source.coefficient, old)
    assert model.input_transform.normalize._source is source
    torch.testing.assert_close(model.input_transform.normalize(Q), source(Q))
    leaf = transform.warp.concentration0
    assert model.input_transform.warp.concentration0 is leaf
    state = copy.deepcopy(model.state_dict())
    cloned = copy.deepcopy(model)
    cloned.load_state_dict(state)
    assert (
        cloned.input_transform.warp.concentration0
        is cloned.input_transform._source.warp.concentration0
    )
    torch.testing.assert_close(cloned.posterior(Q).mean, model.posterior(Q).mean)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    model.load_state_dict(torch.load(stream, weights_only=True))
    assert model.input_transform.warp.concentration0 is leaf
    model.to(dtype=torch.float32)
    assert transform.warp.concentration0 is model.input_transform.warp.concentration0
    assert transform.warp.concentration0.dtype == torch.float32
    assert source._coefficient.dtype == torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_live_public_transform_parameter_selects_tensor_model_on_cpu(device):
    X, Y, Q = (value.detach() for value in _data(device))
    transform = Warp(3, [0, 2])
    transform(Q)
    leaf = transform.concentration0
    model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None)
    assert model._tensor_backend == "torch"
    assert model.input_transform.concentration0 is leaf


@pytest.mark.parametrize("device", DEVICES)
def test_training_row_expansion_rejected_before_kernel_construction(device):
    X, Y, _ = _data(device)
    with pytest.raises(ValueError, match="target rows"):
        SingleTaskGP(
            X,
            Y,
            input_transform=InputPerturbation(X.new_zeros(2, 3), transform_on_train=True),
            backend="torch",
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family",
    ["normalize", "standardize", "filter", "interaction", "chain", "append", "perturb", "impute"],
)
def test_source_model_values_and_query_derivatives(device, family):
    source_input = pytest.importorskip("botorch.models.transforms.input")
    source_gp = pytest.importorskip("botorch.models.gp_regression").SingleTaskGP
    source_kernel = pytest.importorskip("gpytorch.kernels").RBFKernel
    X, Y, Q = _data(device)
    transform, X, Q, mapping = _case(family, X, Q)
    if family == "normalize":
        reference_transform = source_input.Normalize(3)
    elif family == "standardize":
        reference_transform = source_input.InputStandardize(3)
    elif family == "filter":
        reference_transform = source_input.FilterFeatures(torch.tensor([2, 0], device=device))
    elif family == "interaction":
        reference_transform = source_input.InteractionFeatures([2, 0])
    elif family == "chain":
        reference_transform = source_input.ChainedInputTransform(
            keep=source_input.FilterFeatures(torch.tensor([2, 0], device=device)),
            interactions=source_input.InteractionFeatures(),
        )
    elif family == "append":
        reference_transform = source_input.AppendFeatures(
            feature_set=transform.feature_set, transform_on_train=True
        )
    elif family == "perturb":
        reference_transform = source_input.InputPerturbation(transform.perturbation_set)
    else:
        reference_transform = source_input.LearnedFeatureImputation(
            {0: [0, 2], 1: [1, 2]},
            3,
            bounds=transform.bounds,
            dtype=X.dtype,
            device=X.device,
        )
    width = mapping(X).shape[-1]
    actual = SingleTaskGP(
        X,
        Y,
        input_transform=transform,
        covar_module=RBFKernel(ard_num_dims=width),
        outcome_transform=None,
        backend="torch",
    )
    reference = source_gp(
        X,
        Y,
        train_Yvar=actual.likelihood.noise.detach().expand_as(Y),
        input_transform=reference_transform,
        covar_module=source_kernel(ard_num_dims=width),
        outcome_transform=None,
    ).to(X)
    reference.covar_module.lengthscale = actual.covar_module.lengthscale.detach()
    reference.mean_module.constant = actual.mean_module.constant.detach()
    result, expected = actual.posterior(Q), reference.posterior(Q)
    torch.testing.assert_close(result.mean, expected.mean, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=2e-11,
        rtol=2e-10,
    )
    for a, e in zip(
        _derivatives(result.mean, result.distribution.covariance_matrix, (Q,)),
        _derivatives(expected.mean, expected.distribution.covariance_matrix, (Q,)),
        strict=True,
    ):
        torch.testing.assert_close(a[0], e[0], atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_feature_callback_parameters_and_warp_priors_are_registered_once(device):
    from botorch_rs.priors import LogNormalPrior

    X, Y, Q = _data(device)
    feature = torch.nn.Linear(3, 1, dtype=X.dtype, device=X.device)

    class AppendModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = feature

        def forward(self, value):
            return self.feature(value).unsqueeze(-2)

    source = ChainedInputTransform(
        append=AppendFeatures(f=AppendModule(), transform_on_train=True),
        warp=Warp(
            4,
            [0, 2],
            concentration0_prior=LogNormalPrior(0.0, 0.7),
            concentration1_prior=LogNormalPrior(0.1, 0.8),
        ),
    )
    model = SingleTaskGP(X, Y, input_transform=source, outcome_transform=None, backend="torch")
    assert (
        dict(model.named_parameters())["input_transform.append._f.feature.weight"] is feature.weight
    )
    priors = list(model.input_transform.named_priors())
    assert len(priors) == 2
    expected = sum(prior.log_prob(get(owner)).sum() for _, owner, prior, get, _ in priors)
    torch.testing.assert_close(model.input_transform.log_prior(), expected)
    posterior = model.posterior(Q)
    first, second = _derivatives(
        posterior.mean, posterior.distribution.covariance_matrix, (feature.weight, feature.bias)
    )
    assert all(torch.isfinite(value).all() for value in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_no_fit_converter_needs_neither_training_data_nor_invented_width(device):
    from botorch_rs.models._torch_backend_modules import build_input_transform

    like = torch.empty((), dtype=torch.double, device=device)
    source = ChainedInputTransform(keep=FilterFeatures([2, 0]), normalize=Normalize(2))
    module = build_input_transform(source, None, like, fit=False)
    assert module.training and source.training
    assert "_coefficient" not in source.normalize.__dict__
    X, _, _ = _data(device)
    result = module(X)
    selected = X[..., [2, 0]]
    expected = (selected - selected.amin(-2, keepdim=True)) / (
        selected.amax(-2, keepdim=True) - selected.amin(-2, keepdim=True)
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_feature_module_and_live_feature_tensor_select_tensor_factory(device):
    X, Y, Q = (value.detach() for value in _data(device))
    for features in (
        AppendFeatures(feature_set=X.new_tensor([[0.3]]).requires_grad_(), transform_on_train=True),
        AppendFeatures(
            f=torch.nn.Linear(3, 1, dtype=X.dtype, device=X.device), transform_on_train=True
        ),
    ):
        # A callback's declared q x n_f x d_f shape is its own public contract.
        if features._f is not None:

            class Feature(torch.nn.Module):
                def __init__(self, linear):
                    super().__init__()
                    self.linear = linear

                def forward(self, value):
                    return self.linear(value).unsqueeze(-2)

            features._f = Feature(features._f)
        model = SingleTaskGP(X, Y, input_transform=features, outcome_transform=None)
        assert model._tensor_backend == "torch"
        assert model.posterior(Q).mean.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_shared_transform_alias_preserves_one_parameter_and_prior_owner(device):
    from botorch_rs.priors import LogNormalPrior

    X, Y, Q = _data(device)
    warp = Warp(
        3,
        [0, 2],
        bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        concentration0_prior=LogNormalPrior(0.0, 0.7),
        concentration1_prior=LogNormalPrior(0.0, 0.8),
    )
    source = ChainedInputTransform(first=warp, second=warp)
    model = SingleTaskGP(X, Y, input_transform=source, outcome_transform=None, backend="torch")
    assert model.input_transform.first is model.input_transform.second
    assert len(list(model.input_transform.named_priors())) == 2
    assert len(list(model.input_transform.parameters())) == 2
    posterior = model.posterior(Q)
    _derivatives(
        posterior.mean, posterior.distribution.covariance_matrix, (X, Y, Q, *model.parameters())
    )
