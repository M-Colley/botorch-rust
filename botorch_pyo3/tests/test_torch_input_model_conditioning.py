"""Source-valid transformed features and fantasy flags retain Gaussian laws."""

# ruff: noqa: E402
from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")

from test_torch_input_models import DEVICES, _case, _data, _derivatives, _manual

from botorch_rs.kernels import RBFKernel
from botorch_rs.models import SingleTaskGP


class MeanSampler:
    sample_shape = torch.Size([2])

    def __call__(self, posterior):
        self.shape = posterior.mean.shape
        return torch.stack((posterior.mean - 0.05, posterior.mean + 0.05))


def _models(device, family):
    source_inputs = pytest.importorskip("botorch.models.transforms.input")
    source_gp = pytest.importorskip("botorch.models.gp_regression").SingleTaskGP
    source_kernel = pytest.importorskip("gpytorch.kernels").RBFKernel
    X, Y, Q = _data(device)
    transform, X, Q, mapping = _case(family, X, Q)
    options = {}
    name = {
        "normalize": "Normalize",
        "standardize": "InputStandardize",
        "filter": "FilterFeatures",
        "interaction": "InteractionFeatures",
        "append": "AppendFeatures",
        "impute": "LearnedFeatureImputation",
        "onehot": "OneHotToNumeric",
        "categorical": "NumericToCategoricalEncoding",
        "perturb": "InputPerturbation",
    }[family]
    if family in ("normalize", "standardize"):
        options = {"d": 3}
    elif family == "filter":
        options = {"feature_indices": torch.tensor([2, 0], device=device)}
    elif family == "interaction":
        options = {"indices": [2, 0]}
    elif family == "append":
        options = {"feature_set": transform.feature_set, "transform_on_train": True}
    elif family == "impute":
        options = {
            "feature_indices": transform.feature_indices,
            "d": 3,
            "bounds": transform.bounds,
            "dtype": X.dtype,
            "device": X.device,
        }
    elif family == "onehot":
        options = {"dim": 3, "categorical_features": {1: 2}}
    elif family == "categorical":
        options = {
            "dim": 2,
            "categorical_features": {1: 2},
            "encoders": {1: functools.partial(torch.nn.functional.one_hot, num_classes=2)},
        }
    else:
        options = {"perturbation_set": transform.perturbation_set}
    actual = SingleTaskGP(
        X,
        Y,
        input_transform=transform,
        covar_module=RBFKernel(ard_num_dims=mapping(X).shape[-1]),
        outcome_transform=None,
        backend="torch",
    )
    reference = source_gp(
        X,
        Y,
        train_Yvar=actual.likelihood.noise.detach().expand_as(Y),
        input_transform=getattr(source_inputs, name)(**options),
        covar_module=source_kernel(ard_num_dims=mapping(X).shape[-1]),
        outcome_transform=None,
    ).to(X)
    reference.covar_module.lengthscale = actual.covar_module.lengthscale.detach()
    reference.mean_module.constant = actual.mean_module.constant.detach()
    return actual, reference, X, Y, Q, mapping


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family",
    [
        "normalize",
        "standardize",
        "filter",
        "interaction",
        "append",
        "impute",
        "onehot",
        "categorical",
    ],
)
def test_conditioning_transformed_width_matches_source_and_dense_law(device, family):
    actual, source, X, Y, Q, mapping = _models(device, family)
    extra = Q.detach().clone().requires_grad_()
    observations = Y.new_tensor([[0.14], [-0.22]]).requires_grad_()
    # Official ExactGP needs a populated prediction cache for conditioning.
    source.posterior(Q)
    expected = source.condition_on_observations(
        extra, observations, noise=actual.likelihood.noise.detach().expand_as(observations)
    ).posterior(Q)
    conditioned = actual.condition_on_observations(extra, observations)
    posterior = conditioned.posterior(Q)
    training_mapping = mapping
    if family == "impute":
        parameter = conditioned.input_transform.raw_imputation_values

        def mapping(value):
            result = value.clone()
            raw = parameter.reshape(2, 4).sigmoid()
            result[..., 1] = torch.where(value[..., -1] == 0, raw[0, 1], value[..., 1])
            result[..., 0] = torch.where(value[..., -1] == 1, raw[1, 0], value[..., 0])
            return result

    independent = _manual(
        conditioned,
        torch.cat((training_mapping(X), training_mapping(extra)), -2),
        torch.cat((Y, observations), -2),
        mapping(Q),
    )
    torch.testing.assert_close(posterior.mean, expected.mean, atol=4e-10, rtol=4e-9)
    torch.testing.assert_close(
        posterior.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=4e-10,
        rtol=4e-9,
    )
    torch.testing.assert_close(posterior.mean, independent[0], atol=3e-11, rtol=3e-10)
    torch.testing.assert_close(
        posterior.distribution.covariance_matrix, independent[1], atol=3e-11, rtol=3e-10
    )
    leaves = (X, Y, extra, observations, Q, *conditioned.parameters())
    for values, oracle in zip(
        _derivatives(posterior.mean, posterior.distribution.covariance_matrix, leaves),
        _derivatives(*independent, leaves),
        strict=True,
    ):
        for value, reference in zip(values, oracle, strict=True):
            torch.testing.assert_close(value, reference, atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("propagate", [False, True])
def test_fantasize_skips_default_environmental_expansion(device, propagate):
    actual, source, X, Y, Q, mapping = _models(device, "perturb")
    left, right = MeanSampler(), MeanSampler()
    fantasy = actual.fantasize(Q, left, propagate_grads=propagate)
    reference = source.fantasize(Q, right, propagate_grads=propagate)
    assert left.shape == right.shape == torch.Size([2, 1])
    result, expected = fantasy.posterior(Q), reference.posterior(Q)
    torch.testing.assert_close(result.mean, expected.mean, atol=3e-10, rtol=3e-9)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-10,
        rtol=3e-9,
    )
    assert not actual.input_transform.transform_on_fantasize
    assert fantasy._train_X.shape[-2] == X.shape[-2] + Q.shape[-2]
    query = (Q.detach() + 0.03).requires_grad_()
    from botorch.settings import propagate_grads

    reference.prediction_strategy = None
    with propagate_grads(True):
        result, expected = fantasy.posterior(query), reference.posterior(query)
    actual_gradient = torch.autograd.grad(
        result.mean.sin().sum() + result.distribution.covariance_matrix.square().sum(),
        (X, Y, Q, query, actual.covar_module.raw_lengthscale, fantasy.covar_module.raw_lengthscale),
        retain_graph=True,
    )
    source_gradient = torch.autograd.grad(
        expected.mean.sin().sum() + expected.distribution.covariance_matrix.square().sum(),
        (
            X,
            Y,
            Q,
            query,
            source.covar_module.raw_lengthscale,
            reference.covar_module.raw_lengthscale,
        ),
        retain_graph=True,
    )
    for value, oracle in zip(actual_gradient, source_gradient, strict=True):
        torch.testing.assert_close(value, oracle, atol=3e-8, rtol=3e-7)
    # A dense independent law also checks mixed higher derivatives. The
    # unmodified source has a known predictive-update double-backward defect.
    lengthscale = actual.covar_module.lengthscale
    constant = actual.mean_module.constant

    def kernel(A, B):
        difference = (A.unsqueeze(-2) - B.unsqueeze(-3)) / lengthscale.unsqueeze(-2)
        return (-0.5 * difference.square().sum(-1)).exp()

    matrix = kernel(X, X) + torch.diag_embed(actual.likelihood.noise.expand(X.shape[-2]))
    alpha = torch.linalg.solve(matrix, Y - constant)
    if not propagate:
        alpha = alpha.detach()
    samples = constant + kernel(X, Q).transpose(-1, -2) @ alpha
    samples = torch.stack((samples - 0.05, samples + 0.05))
    independent = _manual(
        fantasy,
        torch.cat((X, Q), -2),
        torch.cat((Y.expand(2, *Y.shape), samples), -2),
        mapping(query),
    )
    independent = (independent[0], independent[1].expand_as(result.distribution.covariance_matrix))
    torch.testing.assert_close(result.mean, independent[0])
    torch.testing.assert_close(result.distribution.covariance_matrix, independent[1])
    leaves = (
        X,
        Y,
        Q,
        query,
        actual.mean_module.raw_constant,
        actual.covar_module.raw_lengthscale,
        *fantasy.parameters(),
    )
    if propagate:
        leaves += (actual.likelihood.raw_noise,)
    for values, oracle in zip(
        _derivatives(result.mean, result.distribution.covariance_matrix, leaves),
        _derivatives(*independent, leaves),
        strict=True,
    ):
        for value, reference_value in zip(values, oracle, strict=True):
            torch.testing.assert_close(value, reference_value, atol=3e-7, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_source_different_raw_width_conditioning_boundary(device):
    source_inputs = pytest.importorskip("botorch.models.transforms.input")
    source_gp = pytest.importorskip("botorch.models.gp_regression").SingleTaskGP
    X, Y, Q = _data(device)
    features = X.new_tensor([[0.2], [0.4]])
    augmented = torch.cat((X, X.new_full((4, 1), 0.3)), -1).detach()
    model = source_gp(
        augmented,
        Y.detach(),
        input_transform=source_inputs.AppendFeatures(feature_set=features),
        outcome_transform=None,
    ).to(X)
    model.posterior(Q.detach())
    # The source preserves original raw training inputs and concatenates them
    # with new raw observations; differing widths are unsupported there too.
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        model.condition_on_observations(Q.detach(), Y.new_zeros(4, 1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["normalize", "filter", "impute", "categorical"])
def test_conditioned_parameters_and_modes_are_independent_like_source(device, family):
    actual, source, _, Y, Q, _ = _models(device, family)
    values = Y.new_tensor([[0.17], [-0.21]])
    original = actual.posterior(Q).mean.detach().clone()
    source_original = source.posterior(Q).mean.detach().clone()
    conditioned = actual.condition_on_observations(Q, values)
    reference = source.condition_on_observations(
        Q, values, noise=actual.likelihood.noise.detach().expand_as(values)
    )
    for (name, parameter), (copy_name, copied) in zip(
        actual.named_parameters(), conditioned.named_parameters(), strict=True
    ):
        assert copy_name == name
        assert copied is not parameter and copied.data_ptr() != parameter.data_ptr()
        torch.testing.assert_close(copied, parameter)
    assert reference.covar_module.raw_lengthscale is not source.covar_module.raw_lengthscale
    with torch.no_grad():
        conditioned.covar_module.raw_lengthscale.add_(0.07)
        reference.covar_module.raw_lengthscale.add_(0.07)
    torch.testing.assert_close(actual.posterior(Q).mean, original)
    torch.testing.assert_close(source.posterior(Q).mean, source_original)
    conditioned.train()
    assert actual.input_transform.training is False
    assert conditioned.input_transform.training is True
    conditioned.to(dtype=torch.float32)
    assert actual.covar_module.raw_lengthscale.dtype == torch.float64
    assert conditioned.covar_module.raw_lengthscale.dtype == torch.float32
