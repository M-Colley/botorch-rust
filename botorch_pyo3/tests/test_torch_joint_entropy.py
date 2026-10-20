"""JES device graphs and independently corrected conditional-Gaussian oracle."""

import math

import pytest
import torch

from botorch_rs.acquisition.joint_entropy_search import qJointEntropySearch
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class GaussianConditioningOracle(torch.nn.Module):
    """Condition the official initial Gaussian by its Schur complement.

    This bypasses the known explicit-fantasy-noise cache inconsistency,
    without using the port's conditioned model or inference implementation.
    """

    def __init__(self, model, inputs, outputs, noise):
        super().__init__()
        self.model, self.inputs, self.outputs, self.noise = model, inputs, outputs, noise

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        batch = torch.broadcast_shapes(X.shape[:-2], self.inputs.shape[:-2])
        X = X.expand(*batch, *X.shape[-2:])
        inputs = self.inputs.expand(*batch, *self.inputs.shape[-2:])
        joint = self.model.posterior(torch.cat((X, inputs), -2))
        q = X.shape[-2]
        covariance = joint.mvn.covariance_matrix
        cross = covariance[..., :q, q:]
        weight = cross / (covariance[..., q:, q:] + self.noise)
        mean = joint.mean[..., :q, :] + weight @ (self.outputs - joint.mean[..., q:, :])
        conditional = covariance[..., :q, :q] - weight @ cross.transpose(-1, -2)
        if observation_noise:
            noisy = self.model.posterior(X, observation_noise=True)
            latent = self.model.posterior(X)
            conditional = conditional + noisy.mvn.covariance_matrix - latent.mvn.covariance_matrix
        result = GPyTorchPosterior(MultivariateNormal(mean.squeeze(-1), conditional))
        return result if posterior_transform is None else posterior_transform(result)


def data(device):
    X = torch.tensor([[0.05], [0.32], [0.61], [0.86]], dtype=torch.double, device=device)
    return X, (5 * X).sin()


def make_pair(
    device, estimation_type, condition_noiseless, q, pending, normalized=False, dtype=torch.double
):
    from botorch.acquisition.joint_entropy_search import qJointEntropySearch as ReferenceJES
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
    from hvkg_helpers import use_fantasy_mean_cache
    from so_acquisition_helpers import correct_jes_monte_carlo, jes_optimal_inputs

    from botorch_rs.models.transforms.input import Normalize

    X, Y = data(device)
    X, Y = X.to(dtype), Y.to(dtype)
    if normalized:
        X = X * 3 + 2
    model = SingleTaskGP(
        X,
        Y,
        backend="torch",
        outcome_transform=None,
        input_transform=Normalize(1) if normalized else None,
    )
    reference_model = ReferenceGP(
        X,
        Y,
        outcome_transform=None,
        input_transform=ReferenceNormalize(1) if normalized else None,
    )
    optima, outputs = X.new_tensor([[0.24], [0.4]]), Y.new_tensor([[1.05], [1.1]])
    query = X.new_tensor([[[0.17], [0.7]], [[0.46], [0.77]]])[:, :q]
    points_pending = X.new_tensor([[0.91]]) if pending else None
    if normalized:
        optima, query = optima * 3 + 2, query * 3 + 2
        if points_pending is not None:
            points_pending = points_pending * 3 + 2
    actual = qJointEntropySearch(
        model,
        optima,
        outputs,
        condition_noiseless=condition_noiseless,
        estimation_type=estimation_type,
        X_pending=points_pending,
        sampler=SobolQMCNormalSampler((24,), seed=29),
    )
    reference = ReferenceJES(
        reference_model,
        jes_optimal_inputs(reference_model, optima),
        outputs,
        condition_noiseless=condition_noiseless,
        estimation_type=estimation_type,
        X_pending=points_pending,
        num_samples=24,
    )
    reference.sampler = ReferenceSampler(torch.Size((24,)), seed=29)
    use_fantasy_mean_cache(reference.conditional_model)
    if condition_noiseless:
        reference.conditional_model = GaussianConditioningOracle(
            reference_model, optima.unsqueeze(-2), outputs.unsqueeze(-2), 1e-4
        )
    if estimation_type == "MC":
        correct_jes_monte_carlo(reference)
    return actual, reference, query


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("estimation_type", ["LB", "MC"])
@pytest.mark.parametrize("condition_noiseless", [False, True])
@pytest.mark.parametrize("q,pending", [(1, False), (2, False), (1, True)])
def test_values_and_query_gradients_match_corrected_reference(
    device,
    estimation_type,
    condition_noiseless,
    q,
    pending,
):
    from so_acquisition_helpers import no_fast_pred_var

    actual, reference, query = make_pair(device, estimation_type, condition_noiseless, q, pending)
    query.requires_grad_()
    value = actual(query)
    with no_fast_pred_var():
        expected = reference(query)
    torch.testing.assert_close(value, expected, rtol=2e-7, atol=2e-8)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    assert value.device == query.device and isinstance(actual, qJointEntropySearch)
    assert actual._tensor_backend == "torch"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("estimation_type", ["LB", "MC"])
def test_normalize_applied_once_and_arbitrary_t_batches(device, estimation_type):
    from so_acquisition_helpers import no_fast_pred_var

    actual, reference, query = make_pair(device, estimation_type, False, 2, False, normalized=True)
    query = query[:, None].expand(2, 3, 2, 1).clone().requires_grad_()
    with no_fast_pred_var():
        expected = reference(query)
    value = actual(query)
    torch.testing.assert_close(value, expected, rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(
        torch.autograd.grad(value.sum(), query)[0],
        torch.autograd.grad(expected.sum(), query)[0],
        rtol=3e-5,
        atol=3e-6,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("estimation_type", ["LB", "MC"])
def test_training_optima_and_query_gradgradcheck(device, estimation_type):
    X, Y = data(device)
    X, Y = X[:3].requires_grad_(), Y[:3].requires_grad_()
    query = X.new_tensor([[0.43]], requires_grad=True)
    optima = X.new_tensor([[0.26], [0.37]], requires_grad=True)
    outputs = X.new_tensor([[1.05], [1.1]], requires_grad=True)

    def evaluate(train_X, train_Y, points, best_X, best_Y):
        model = SingleTaskGP(train_X, train_Y, backend="torch", outcome_transform=None)
        acquisition = qJointEntropySearch(
            model,
            best_X,
            best_Y,
            condition_noiseless=False,
            estimation_type=estimation_type,
            sampler=SobolQMCNormalSampler((8,), seed=41),
        )
        return acquisition(points)

    arguments = (X, Y, query, optima, outputs)
    assert torch.autograd.gradcheck(evaluate, arguments, fast_mode=True, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(evaluate, arguments, fast_mode=True, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_conditional_normal_scaling_reproduces_official_error(device):
    from scipy.stats import norm

    # F variance .3, observation Y=F+E variance 1. At Y=1,
    # F|Y ~ N(.3,.21), so its threshold-zero CDF has this closed form.
    correct = norm.cdf(-math.sqrt(3 / 7))
    wrong = norm.cdf(-math.sqrt(0.3) / 0.7)
    assert abs(correct - wrong) > 0.03
    actual, reference, query = make_pair(device, "MC", False, 1, False)
    value = actual(query)
    corrected = reference(query)
    del reference.__dict__["_compute_monte_carlo_information_gain"]
    original = reference(query)
    torch.testing.assert_close(value, corrected, rtol=2e-7, atol=2e-8)
    assert (value - original).abs().amax() > 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("estimation_type", ["LB", "MC"])
def test_float32_values_and_second_query_derivatives(device, estimation_type):
    actual, reference, query = make_pair(
        device, estimation_type, False, 2, False, dtype=torch.float32
    )
    query.requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=2e-4, atol=2e-5)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert value.dtype == torch.float32
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("estimation_type", ["LB", "MC"])
def test_all_parameter_hvps_and_no_host_data_conversion(device, estimation_type, monkeypatch):
    X, Y = data(device)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    acquisition = qJointEntropySearch(
        model,
        X.new_tensor([[0.27], [0.42]]),
        Y.new_tensor([[1.05], [1.1]]),
        estimation_type=estimation_type,
        sampler=SobolQMCNormalSampler((16,), seed=43),
    )
    query = X.new_tensor([[[0.17]], [[0.48]]], requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference tensors were transferred to host Python/NumPy.")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    value = acquisition(query)
    variables = (query, *model.parameters())
    gradient = torch.autograd.grad(value.sum(), variables, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), variables)
    assert all(torch.isfinite(g).all() for g in (*gradient, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("estimation_type", ["LB", "MC"])
@pytest.mark.parametrize("model_list", [False, True])
def test_scalarization_and_independent_model_list(device, estimation_type, model_list):
    from botorch.acquisition.joint_entropy_search import qJointEntropySearch as ReferenceJES
    from botorch.acquisition.objective import ScalarizedPosteriorTransform as ReferenceTransform
    from botorch.models import ModelListGP as ReferenceList
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
    from jes_reference import correct_jes_monte_carlo
    from so_acquisition_helpers import no_fast_pred_var

    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
    from botorch_rs.models import ModelListGP

    X, Y = data(device)
    Y = torch.cat((Y, Y.cos()), -1)
    if model_list:
        model = ModelListGP(
            *[
                SingleTaskGP(X, Y[..., i : i + 1], backend="torch", outcome_transform=None)
                for i in range(2)
            ]
        )
        reference_model = ReferenceList(
            *[ReferenceGP(X, Y[..., i : i + 1], outcome_transform=None) for i in range(2)]
        )
    else:
        model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
        reference_model = ReferenceGP(X, Y, outcome_transform=None)
    weights = X.new_tensor([0.8, -0.3], requires_grad=True)
    optima, outputs = X.new_tensor([[0.26], [0.37]]), Y.new_tensor([[1.1, 0.1], [1.2, 0.2]])
    actual = qJointEntropySearch(
        model,
        optima,
        outputs,
        condition_noiseless=False,
        estimation_type=estimation_type,
        posterior_transform=ScalarizedPosteriorTransform(weights, offset=0.2),
        sampler=SobolQMCNormalSampler((16,), seed=47),
    )
    reference = ReferenceJES(
        reference_model,
        optima,
        outputs,
        condition_noiseless=False,
        estimation_type=estimation_type,
        posterior_transform=ReferenceTransform(weights, offset=0.2),
        num_samples=16,
    )
    reference.sampler = ReferenceSampler(torch.Size((16,)), seed=47)
    if estimation_type == "MC":
        correct_jes_monte_carlo(reference)
    query = X.new_tensor([[[0.17], [0.7]], [[0.46], [0.77]]], requires_grad=True)
    value = actual(query)
    with no_fast_pred_var():
        expected = reference(query)
    torch.testing.assert_close(value, expected, rtol=3e-7, atol=3e-8)
    gradient = torch.autograd.grad(value.sum(), (query, weights), create_graph=True)
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient[0], expected_gradient, rtol=3e-5, atol=3e-6)
    assert all(torch.isfinite(g).all() for g in gradient)
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(sum(g.square().sum() for g in gradient), (query, weights))
    )


@pytest.mark.parametrize("device", DEVICES)
def test_joint_entropy_dispatches_device_candidate_optimizer(device, monkeypatch):
    from botorch_rs.generation.gen import gen_candidates_torch

    acquisition, _, query = make_pair(device, "LB", False, 1, False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Entropy optimizer transferred inference tensors to host.")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    candidates, values = gen_candidates_torch(
        query,
        acquisition,
        lower_bounds=0.0,
        upper_bounds=1.0,
        options={"stopping_criterion_options": {"maxiter": 2}},
    )
    assert candidates.device == query.device and values.device == query.device
    assert torch.isfinite(values).all()
    assert all(parameter.grad is None for parameter in acquisition.model.parameters())
