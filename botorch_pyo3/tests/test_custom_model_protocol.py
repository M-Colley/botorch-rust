# ruff: noqa: E402
"""Custom public Model classes retain Torch module and acquisition protocols."""

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.acquisition.analytic import LogExpectedImprovement
from botorch_rs.acquisition.logei import qLogExpectedImprovement
from botorch_rs.models.model import Model
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def initialize(model, anchor):
    torch.nn.Module.__init__(model)
    model.register_buffer("anchor", anchor)
    model.offset = torch.nn.Parameter(anchor.new_tensor(0.15))
    model.raw_noise = torch.nn.Parameter(anchor.new_tensor(-2.0))
    model.child = torch.nn.Linear(1, 1, bias=False).to(anchor)
    with torch.no_grad():
        model.child.weight.fill_(0.7)


def gaussian(model, X, observation_noise):
    mean = model.child(X).sin() + model.offset + model.anchor.mean()
    distance = (X.unsqueeze(-2) - X.unsqueeze(-3)).square().sum(-1)
    eye = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    covariance = 0.2 * (-distance).exp() + model.raw_noise.exp() * eye
    if observation_noise:
        covariance = covariance + 0.03 * eye
    return mean, covariance


class DeviceModel(Model):
    def __init__(self, anchor):
        super().__init__()
        initialize(self, anchor)

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size()

    def posterior(self, X, observation_noise=False, posterior_transform=None, **kwargs):
        mean, covariance = gaussian(self, X, observation_noise)
        posterior = TorchGPyTorchPosterior(mean, covariance)
        return posterior if posterior_transform is None else posterior_transform(posterior)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_model_parameters_buffers_modes_migration_and_serialization(device):
    anchor = torch.tensor([[0.1], [0.4]], device=device, dtype=torch.double)
    model = DeviceModel(anchor)
    assert isinstance(model, DeviceModel) and isinstance(model, torch.nn.Module)
    assert model._tensor_backend == "torch"
    assert set(dict(model.named_parameters())) == {"offset", "raw_noise", "child.weight"}
    assert dict(model.named_buffers())["anchor"] is anchor
    assert model.eval() is model and not model.child.training
    assert model.train() is model and model.child.training
    query = anchor.new_tensor([[0.2], [0.7]])
    expected = model.posterior(query).mean
    for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert isinstance(restored, DeviceModel) and isinstance(restored, torch.nn.Module)
        torch.testing.assert_close(restored.posterior(query).mean, expected)
        assert restored.offset is not model.offset
    model.to(dtype=torch.float)
    assert model.anchor.dtype == torch.float and model.offset.dtype == torch.float
    assert set(model.state_dict()) == {"offset", "raw_noise", "anchor", "child.weight"}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mc", [False, True])
def test_custom_model_acquisitions_match_source_full_higher_graphs(device, mc):
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    from botorch.acquisition.analytic import LogExpectedImprovement as ReferenceEI
    from botorch.acquisition.logei import qLogExpectedImprovement as ReferenceQEI
    from botorch.models.model import Model as ReferenceModel
    from botorch.posteriors.gpytorch import GPyTorchPosterior as ReferencePosterior
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
    from gpytorch.distributions import MultivariateNormal

    class GaussianReference(ReferenceModel):
        def __init__(self, anchor):
            initialize(self, anchor)

        num_outputs = DeviceModel.num_outputs
        batch_shape = DeviceModel.batch_shape

        def posterior(self, X, observation_noise=False, posterior_transform=None, **kwargs):
            mean, covariance = gaussian(self, X, observation_noise)
            posterior = ReferencePosterior(MultivariateNormal(mean.squeeze(-1), covariance))
            return posterior if posterior_transform is None else posterior_transform(posterior)

    anchor = torch.tensor([[0.1], [0.4]], device=device, dtype=torch.double, requires_grad=True)
    actual_model, source_model = DeviceModel(anchor), GaussianReference(anchor)
    query = anchor.new_tensor([[0.2], [0.7]][: 2 if mc else 1], requires_grad=True)
    if mc:
        actual = qLogExpectedImprovement(
            actual_model, 0.3, sampler=SobolQMCNormalSampler((12,), seed=91)
        )(query)
        expected = ReferenceQEI(
            source_model, 0.3, sampler=ReferenceSampler(torch.Size([12]), seed=91)
        )(query)
    else:
        actual = LogExpectedImprovement(actual_model, 0.3)(query)
        expected = ReferenceEI(source_model, 0.3)(query)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    actual_leaves = (query, anchor, *actual_model.parameters())
    source_leaves = (query, anchor, *source_model.parameters())
    first = torch.autograd.grad(actual.sum(), actual_leaves, create_graph=True, retain_graph=True)
    source_first = torch.autograd.grad(
        expected.sum(), source_leaves, create_graph=True, retain_graph=True
    )
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), actual_leaves, retain_graph=True
    )
    for a, b in zip(first, source_first, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-8)
    if not mc:
        source_second = torch.autograd.grad(
            sum(g.square().sum() for g in source_first), source_leaves
        )
        for a, b in zip(second, source_second, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-8)
    else:
        # The source's TriangularLinearOperator sampling backward drops
        # matrix-parameter terms in double backward (linear_operator #152).
        # Differentiate its unmodified first derivatives independently.
        state = tuple(leaf.detach().clone() for leaf in source_leaves)
        direction = tuple(2 * gradient.detach() for gradient in source_first)

        def reference_gradient(step):
            candidate = (state[0] + step * direction[0]).requires_grad_()
            location = (state[1] + step * direction[1]).requires_grad_()
            reference = GaussianReference(location)
            with torch.no_grad():
                for parameter, value, tangent in zip(
                    reference.parameters(), state[2:], direction[2:], strict=True
                ):
                    parameter.copy_(value + step * tangent)
            value = ReferenceQEI(
                reference, 0.3, sampler=ReferenceSampler(torch.Size([12]), seed=91)
            )(candidate)
            return torch.autograd.grad(value.sum(), (candidate, location, *reference.parameters()))

        estimates = []
        for step in (2e-5, 1e-5):
            plus, minus = reference_gradient(step), reference_gradient(-step)
            estimates.append(tuple((a - b) / (2 * step) for a, b in zip(plus, minus, strict=True)))
        for actual_higher, larger, smaller in zip(second, *estimates, strict=True):
            torch.testing.assert_close(larger, smaller, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual_higher, smaller, rtol=2e-5, atol=2e-6)
