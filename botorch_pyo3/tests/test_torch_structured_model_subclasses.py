# ruff: noqa: E402
"""Specialized Tensor factories preserve cooperative subclasses and virtual priors."""

import copy
import pickle
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


from botorch_rs.models import KroneckerMultiTaskGP, LatentKroneckerGP
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _compare_graphs(actual, expected, leaves):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    first = torch.autograd.grad(actual, leaves, create_graph=True, retain_graph=True)
    reference = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    reference_second = torch.autograd.grad(
        sum(g.square().sum() for g in reference), leaves, retain_graph=True
    )
    for a, b in zip((*first, *second), (*reference, *reference_second), strict=True):
        assert bool(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=3e-8, atol=3e-9)


class _Tagged:
    def __init__(self, *args, shift=0.17, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter("shift", torch.nn.Parameter(self._train_X.new_tensor(shift)))
        self.register_buffer("marker", self._train_X.new_tensor([3.0]))
        self.posterior_calls = 0

    def posterior(self, *args, **kwargs):
        self.posterior_calls += 1
        return super().posterior(*args, **kwargs)


class _VirtualPrior:
    def forward(self, x):
        base = super().forward(x)
        transformed = self.transform_inputs(x) if self.training else x
        if self._structured_kind == "kronecker":
            weights = torch.arange(1, self.num_outputs + 1, device=x.device, dtype=x.dtype)
            feature = (0.4 + transformed.sum(-1, keepdim=True)) * weights
            flat = feature.transpose(-1, -2).flatten(-2)
            mean = base.mean + self.shift * feature
        else:
            weights = torch.arange(1, self._grid_size + 1, device=x.device, dtype=x.dtype)
            feature = (0.4 + transformed.sum(-1, keepdim=True)) * weights
            flat = feature.flatten(-2)
            mean = (base.mean + self.shift * flat).unsqueeze(-1)
        covariance = base.covariance_matrix + self.shift.square() * (
            flat.unsqueeze(-1) * flat.unsqueeze(-2)
        )
        gaussian = getattr(self, "gaussian_type", "port")
        if gaussian != "port":
            import gpytorch

            if self._structured_kind == "kronecker":
                interleaved = gaussian == "point_major"
                if interleaved:
                    q, m = mean.shape[-2:]
                    index = torch.arange(q * m, device=x.device).reshape(m, q).T.flatten()
                    covariance = covariance.index_select(-2, index).index_select(-1, index)
                return gpytorch.distributions.MultitaskMultivariateNormal(
                    mean, covariance, interleaved=interleaved
                )
            return gpytorch.distributions.MultivariateNormal(mean.squeeze(-1), covariance)
        return TorchGPyTorchPosterior(mean, covariance).distribution


class TaggedKronecker(_Tagged, _VirtualPrior, KroneckerMultiTaskGP):
    _structured_kind = "kronecker"


class TaggedHigherOrder(_Tagged, _VirtualPrior, HigherOrderGP):
    _structured_kind = "higher_order"


class TaggedLatent(_Tagged, LatentKroneckerGP):
    pass


def data(device, batch=False, gradients=False):
    x = torch.tensor([[0.2], [0.6], [1.0], [1.4]], device=device, dtype=torch.double)
    y = torch.cat(((x * 2).sin() + 0.5, (x * 3).cos() + 0.2), -1)
    time = x.new_tensor([[0.0], [1.0]])
    query = x.new_tensor([[0.43], [1.13]])
    if batch:
        x, y = x.expand(2, 4, 1).clone(), y.expand(2, 4, 2).clone()
        query = query.expand(3, 1, 2, 1).clone()
        y[1] += 0.2
    if gradients:
        x.requires_grad_()
        y.requires_grad_()
        query.requires_grad_()
    return x, y, time, query


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kronecker", "higher_order", "latent"])
@pytest.mark.parametrize("backend", ["torch", "auto"])
def test_structured_subclass_factories_preserve_constructors_methods_state_and_serialization(
    device, family, backend
):
    x, y, time, query = data(device)
    options = {"outcome_transform": None, "backend": backend, "shift": 0.21}
    # An actual supplied module requests Tensor construction even for ordinary
    # CPU data, without changing the native/auto selection for plain descriptors.
    if family == "kronecker":
        from botorch_rs.models._torch_backend_modules import build_kernel

        options["data_covar_module"] = build_kernel(None, 1, (), x)
        cls, args = TaggedKronecker, (x, y)
    elif family == "higher_order":
        from botorch_rs.models._torch_backend_modules import build_kernel

        options["covar_modules"] = [build_kernel(None, 1, (), x) for _ in range(2)]
        cls, args = TaggedHigherOrder, (x, y)
    else:
        from botorch_rs.models._torch_backend_modules import TorchZeroMean

        options["mean_module_X"] = TorchZeroMean((), x)
        cls, args = TaggedLatent, (x, time, y)
    model = cls(*args, **options).eval()
    assert isinstance(model, cls) and isinstance(model, torch.nn.Module)
    assert any(model.shift is value for value in model.parameters())
    assert dict(model.named_buffers())["marker"] is model.marker
    posterior = model.posterior(query)
    assert model.posterior_calls == 1
    for clone in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert isinstance(clone, cls)
        torch.testing.assert_close(clone.shift, x.new_tensor(0.21))
        torch.testing.assert_close(clone.posterior(query).mean, posterior.mean)
        assert clone.shift is not model.shift
    model.float().cpu()
    assert model.shift.dtype == torch.float and model.marker.device.type == "cpu"


def point_major(model, distribution, family):
    mean, covariance = distribution.mean, distribution.covariance_matrix
    if family == "kronecker":
        if not getattr(distribution, "_interleaved", False):
            n, m = mean.shape[-2:]
            indices = torch.arange(n * m, device=mean.device).reshape(m, n).T.flatten()
            covariance = covariance.index_select(-2, indices).index_select(-1, indices)
        return mean.flatten(-2), covariance
    return mean, covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family,gaussian",
    [
        ("kronecker", "port"),
        ("kronecker", "point_major"),
        ("kronecker", "output_major"),
        ("higher_order", "port"),
        ("higher_order", "univariate"),
    ],
)
@pytest.mark.parametrize("batch", [False, True])
def test_structured_conditionals_mll_and_all_leaf_hessians_honor_the_actual_virtual_prior(
    device, family, gaussian, batch
):
    if gaussian != "port":
        pytest.importorskip("gpytorch")
    x, y, _, query = data(device, batch=batch, gradients=True)
    options = {
        "backend": "torch",
        "outcome_transform": None,
        "input_transform": Normalize(1, bounds=x.new_tensor([[0.0], [2.0]])),
    }
    model = (TaggedKronecker if family == "kronecker" else TaggedHigherOrder)(
        x, y, **options
    ).eval()
    model.gaussian_type = gaussian
    train, targets = model._training_values()
    transformed = model.transform_inputs(query)
    shape = torch.broadcast_shapes(train.shape[:-2], transformed.shape[:-2])
    train, transformed = [v.expand(*shape, *v.shape[-2:]) for v in (train, transformed)]
    full = torch.cat((train, transformed), -2)
    train_mean, covariance = point_major(model, model.forward(train), family)
    full_mean, joint = point_major(model, model.forward(full), family)
    count = train_mean.shape[-1]
    cross, test = joint[..., :count, count:], joint[..., count:, count:]
    if family == "kronecker":
        from botorch_rs.models._torch_backend_kronecker import tensor_kron

        noise = tensor_kron(
            torch.eye(train.shape[-2], device=x.device, dtype=x.dtype), model._noise_matrix()
        )
    else:
        noise = torch.diag_embed(model.likelihood.noise.expand_as(train_mean))
    observed = covariance + noise
    residual = targets.flatten(-2).expand_as(train_mean) - train_mean
    solved = torch.linalg.solve(observed, residual.unsqueeze(-1))
    expected_mean = full_mean[..., count:] + (cross.transpose(-1, -2) @ solved).squeeze(-1)
    expected_covariance = test - cross.transpose(-1, -2) @ torch.linalg.solve(observed, cross)
    with patch("botorch_rs._arrays.to_flat", side_effect=AssertionError("host copy")):
        result = model(transformed)
    actual_mean, actual_covariance = point_major(model, result, family)
    leaves = (x, y, query, *model.parameters())
    _compare_graphs(
        actual_mean.square().sum() + actual_covariance.square().sum(),
        expected_mean.square().sum() + expected_covariance.square().sum(),
        leaves,
    )
    posterior = model.posterior(query)
    torch.testing.assert_close(posterior.mean.flatten(-2), actual_mean)
    # Public model MLLs normalize by every observed grid cell; the prior callback
    # is evaluated independently from the solve, rather than reusing port bindings.
    train, targets = model._training_values()
    train_mean, covariance = point_major(model, model.forward(train), family)
    if family == "kronecker":
        noise = tensor_kron(
            torch.eye(train.shape[-2], device=x.device, dtype=x.dtype), model._noise_matrix()
        )
    else:
        noise = torch.diag_embed(model.likelihood.noise.expand_as(train_mean))
    expected_mll = torch.distributions.MultivariateNormal(
        train_mean, covariance_matrix=covariance + noise
    ).log_prob(targets.flatten(-2))
    expected_mll = (expected_mll + model._log_prior()) / train_mean.shape[-1]
    torch.testing.assert_close(model._exact_mll(), expected_mll, atol=2e-9, rtol=2e-9)
    model.train()
    _compare_graphs(model._exact_mll().sum(), expected_mll.sum(), (x, y, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kronecker", "higher_order"])
def test_structured_custom_prior_conditional_values_and_query_shift_gradients_match_source(
    device, family
):
    gpytorch = pytest.importorskip("gpytorch")
    pytest.importorskip("botorch")
    from botorch.models import KroneckerMultiTaskGP as ReferenceKronecker
    from botorch.models.higher_order_gp import HigherOrderGP as ReferenceHigher
    from botorch.models.transforms.input import Normalize as ReferenceNormalize

    class ReferencePrior:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_parameter(
                "shift", torch.nn.Parameter(self.train_inputs[0].new_tensor(0.17))
            )

        def forward(self, x):
            base = super().forward(x)
            transformed = self.transform_inputs(x) if self.training else x
            m = (
                self.num_outputs
                if family == "kronecker"
                else int(torch.Size(self.target_shape).numel())
            )
            feature = (0.4 + transformed.sum(-1, keepdim=True)) * torch.arange(
                1, m + 1, dtype=x.dtype, device=x.device
            )
            flat = feature.flatten(-2)
            mean = base.mean + self.shift * (feature if family == "kronecker" else flat)
            covariance = base.covariance_matrix + self.shift.square() * (
                flat.unsqueeze(-1) * flat.unsqueeze(-2)
            )
            cls = (
                gpytorch.distributions.MultitaskMultivariateNormal
                if family == "kronecker"
                else gpytorch.distributions.MultivariateNormal
            )
            return cls(mean, covariance)

    class ReferenceKron(ReferencePrior, ReferenceKronecker):
        pass

    class ReferenceHOGP(ReferencePrior, ReferenceHigher):
        pass

    x, y, _, query = data(device, gradients=True)
    bounds = x.new_tensor([[0.0], [2.0]])
    actual = (TaggedKronecker if family == "kronecker" else TaggedHigherOrder)(
        x,
        y,
        backend="torch",
        outcome_transform=None,
        input_transform=Normalize(1, bounds=bounds),
    ).eval()
    source = (ReferenceKron if family == "kronecker" else ReferenceHOGP)(
        x, y, outcome_transform=None, input_transform=ReferenceNormalize(1, bounds=bounds)
    ).to(x)
    aa, bb = dict(actual.named_parameters()), dict(source.named_parameters())
    assert aa.keys() == bb.keys()
    with torch.no_grad():
        for name in aa:
            bb[name].copy_(aa[name])
    source.eval()
    result = actual(actual.transform_inputs(query))
    with gpytorch.settings.fast_pred_var(False), gpytorch.settings.detach_test_caches(False):
        expected = source(source.transform_inputs(query))
        mean, covariance = expected.mean, expected.covariance_matrix
        if family == "kronecker":
            q, m = mean.shape[-2:]
            index = torch.arange(q * m, device=x.device).reshape(q, m).T.flatten()
            covariance = covariance.index_select(-2, index).index_select(-1, index)
        torch.testing.assert_close(result.mean, mean, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(result.covariance_matrix, covariance, atol=2e-8, rtol=2e-8)
        a = torch.autograd.grad(
            result.mean.square().sum() + result.covariance_matrix.square().sum(),
            (query, actual.shift),
        )
        b = torch.autograd.grad(
            mean.square().sum() + covariance.square().sum(), (query, source.shift)
        )
        for left, right in zip(a, b, strict=True):
            torch.testing.assert_close(left, right, atol=3e-7, rtol=3e-7)
