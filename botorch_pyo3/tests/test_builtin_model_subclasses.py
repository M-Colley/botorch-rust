"""Caller model subclasses preserve construction and define the actual GP prior."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.contextual_multioutput import LCEMGP
from botorch_rs.models.multitask import MultiTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _CallerConstructor:
    def __new__(cls, *args, **kwargs):
        model = super().__new__(cls, *args, **kwargs)
        model.caller_new_executed = True
        return model

    def __init__(self, *args, label="caller subclass", **kwargs):
        self.before_super = label
        super().__init__(*args, **kwargs)
        self.constructor_label = label
        if getattr(self, "_tensor_backend", None) == "torch":
            self.register_buffer("caller_buffer", self._train_X.new_tensor(0.7))

    def caller_method(self):
        return self.constructor_label


class TaggedSingleGP(_CallerConstructor, SingleTaskGP):
    pass


class TaggedMultiTaskGP(_CallerConstructor, MultiTaskGP):
    pass


class TaggedLCEMGP(_CallerConstructor, LCEMGP):
    pass


class _TwistedPrior:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_twist = torch.nn.Parameter(self.train_inputs[0].new_tensor(0.2))

    def forward(self, X):
        prior = super().forward(X)
        feature = X[..., 0].square() + 0.3
        mean = prior.mean + self.raw_twist * (1.3 * X[..., 0]).sin()
        covariance = prior.covariance_matrix + self.raw_twist.square() * feature.unsqueeze(
            -1
        ) * feature.unsqueeze(-2)
        return self.make_gaussian(mean, covariance)


class TwistedGP(_TwistedPrior, SingleTaskGP):
    @staticmethod
    def make_gaussian(mean, covariance):
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution


def _inputs(family, device, *, grad=False):
    x = torch.tensor([[0.1], [0.35], [0.65], [0.9]], device=device, dtype=torch.double)
    if family != "single":
        x = torch.cat((x, x.new_tensor([[0], [1], [0], [1]])), dim=-1)
    x.requires_grad_(grad)
    y = torch.sin(4 * x[..., :1])
    options = {"outcome_transform": None}
    if family != "single":
        options["task_feature"] = -1
        options["output_tasks"] = [1, 0]
    if family == "lcem":
        options["context_emb_feature"] = x.new_tensor([[0.2], [0.8]])
        options["seed"] = 13
    return x, y, options


def _independent_twisted_posterior_loss(model, x, y, query):
    """Dense Gaussian conditioning without port bindings or source operators."""
    if y.shape[-1] > 1:
        outputs = y.shape[-1]
        x = x.unsqueeze(-3).expand(*x.shape[:-2], outputs, *x.shape[-2:])
        query = query.unsqueeze(-3).expand(*query.shape[:-2], outputs, *query.shape[-2:])
        targets = y.transpose(-1, -2)
    else:
        targets = y.squeeze(-1)

    def mean(z):
        return model.mean_module.constant.unsqueeze(-1) + model.raw_twist * (1.3 * z[..., 0]).sin()

    def kernel(z, w):
        distance = (z.unsqueeze(-2) - w.unsqueeze(-3)) / model.covar_module.lengthscale.unsqueeze(
            -2
        )
        rbf = (-0.5 * distance.square().sum(-1)).exp()
        return rbf + model.raw_twist.square() * (z[..., 0].square() + 0.3).unsqueeze(-1) * (
            w[..., 0].square() + 0.3
        ).unsqueeze(-2)

    train_covariance = kernel(x, x) + torch.diag_embed(model.likelihood.noise.expand_as(targets))
    cross_covariance = kernel(x, query)
    posterior_mean = mean(query) + (
        cross_covariance.transpose(-1, -2)
        @ torch.linalg.solve(train_covariance, (targets - mean(x)).unsqueeze(-1))
    ).squeeze(-1)
    covariance = kernel(query, query) - cross_covariance.transpose(-1, -2) @ torch.linalg.solve(
        train_covariance, cross_covariance
    )
    return (
        posterior_mean.square().sum()
        + 0.7 * covariance.diagonal(dim1=-1, dim2=-2).sum()
        + 0.13 * covariance.square().sum()
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["single", "multitask", "lcem"])
@pytest.mark.parametrize("selection", ["explicit", "automatic"])
def test_subclass_constructor_methods_posterior_graph_and_serialization(device, family, selection):
    cls, canonical = {
        "single": (TaggedSingleGP, SingleTaskGP),
        "multitask": (TaggedMultiTaskGP, MultiTaskGP),
        "lcem": (TaggedLCEMGP, LCEMGP),
    }[family]
    x, y, options = _inputs(family, device, grad=selection == "automatic")
    torch.manual_seed(19)
    model = cls(
        x, y, **options, backend="torch" if selection == "explicit" else "auto", label="kept"
    )
    assert isinstance(model, cls) and isinstance(model, torch.nn.Module)
    assert model._tensor_backend == "torch"
    assert model.before_super == model.caller_method() == "kept"
    assert model.caller_new_executed
    torch.manual_seed(19)
    reference = canonical(x, y, **options, backend="torch")
    reference.load_state_dict(model.state_dict(), strict=False)
    query = x.new_tensor([[0.24], [0.56]]).requires_grad_()
    actual, expected = model.posterior(query), reference.posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected.distribution.covariance_matrix
    )
    gradient = torch.autograd.grad(
        actual.mean.sum() + actual.variance.sum(), query, create_graph=True
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    for duplicate in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert isinstance(duplicate, cls)
        assert duplicate.caller_method() == "kept" and duplicate.caller_new_executed
        assert duplicate.caller_buffer.device == x.device
        torch.testing.assert_close(duplicate.posterior(query).mean, expected.mean)


@pytest.mark.parametrize("family", ["single", "multitask", "lcem"])
def test_explicit_native_subclasses_keep_existing_native_family(family):
    cls = {"single": TaggedSingleGP, "multitask": TaggedMultiTaskGP, "lcem": TaggedLCEMGP}[family]
    x, y, options = _inputs(family, "cpu")
    model = cls(x, y, **options, backend="native", label="native")
    assert type(model) is cls
    assert not isinstance(model, torch.nn.Module)
    assert model.caller_method() == "native" and model.caller_new_executed
    assert torch.isfinite(model.posterior(x.new_tensor([[0.3]])).mean).all()


@pytest.mark.parametrize("cls", [LCEMGP, TaggedLCEMGP])
@pytest.mark.parametrize("option", ["train_Yvar", "mean_module", "likelihood", "input_transform"])
def test_lcem_cpu_tensor_configuration_selects_supported_backend_automatically(cls, option):
    from botorch_rs.exceptions.errors import UnsupportedError
    from botorch_rs.likelihoods import GaussianLikelihood
    from botorch_rs.means import ConstantMean
    from botorch_rs.models.transforms.input import Normalize

    x, y, options = _inputs("lcem", "cpu")
    options[option] = {
        "train_Yvar": torch.full_like(y, 0.02),
        "mean_module": ConstantMean(),
        "likelihood": GaussianLikelihood(),
        "input_transform": Normalize(d=2, indices=[0]),
    }[option]
    model = cls(x, y, **options)
    assert isinstance(model, cls) and model._tensor_backend == "torch"
    query = x.new_tensor([[0.2], [0.6]]).requires_grad_()
    posterior = model.posterior(query)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(torch.autograd.grad(posterior.mean.sum(), query)[0]).all()
    with pytest.raises(UnsupportedError):
        cls(x, y, **options, backend="native")


def test_lcem_positional_fixed_noise_configuration_selects_tensor_backend():
    x, y, _ = _inputs("lcem", "cpu")
    model = TaggedLCEMGP(x, y, -1, torch.full_like(y, 0.02), outcome_transform=None)
    assert isinstance(model, TaggedLCEMGP) and model._tensor_backend == "torch"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch", [(1, ()), (2, ()), (1, (2,)), (2, (2,))])
def test_single_task_virtual_forward_defines_joint_prior_values_and_full_derivatives(
    device, outputs, batch
):
    pytest.importorskip("botorch")
    gpytorch = pytest.importorskip("gpytorch")
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.settings import propagate_grads

    class ReferenceTwistedGP(_TwistedPrior, ReferenceGP):
        @staticmethod
        def make_gaussian(mean, covariance):
            return gpytorch.distributions.MultivariateNormal(mean, covariance)

    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    x = x.expand(*batch, *x.shape).clone().requires_grad_()
    y = (
        torch.cat([(4 * x).sin() + 0.1 * index for index in range(outputs)], dim=-1)
        .detach()
        .requires_grad_()
    )
    rx, ry = x.detach().clone().requires_grad_(), y.detach().clone().requires_grad_()
    model, reference = (
        TwistedGP(x, y, backend="torch", outcome_transform=None),
        ReferenceTwistedGP(rx, ry, outcome_transform=None),
    )
    for current in (model, reference):
        current.mean_module.constant = x.new_tensor(0.15)
        current.covar_module.lengthscale = x.new_tensor(0.42)
        current.likelihood.noise = x.new_tensor(0.04)
    query = x.new_tensor([[0.2], [0.63]]).requires_grad_()
    ref_query = query.detach().clone().requires_grad_()
    with propagate_grads(True), gpytorch.settings.fast_pred_var(False):
        actual, expected = model.posterior(query), reference.posterior(ref_query)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        rtol=1e-8,
        atol=1e-9,
    )
    losses = [
        posterior.mean.square().sum()
        + 0.7 * posterior.variance.sum()
        + 0.13 * posterior.distribution.covariance_matrix.square().sum()
        for posterior in (actual, expected)
    ]
    leaves = [
        (
            tx,
            ty,
            q,
            gp.raw_twist,
            gp.mean_module.raw_constant,
            gp.covar_module.raw_lengthscale,
            gp.likelihood.raw_noise,
        )
        for gp, tx, ty, q in ((model, x, y, query), (reference, rx, ry, ref_query))
    ]
    gradients = [
        torch.autograd.grad(loss, params, create_graph=True)
        for loss, params in zip(losses, leaves, strict=True)
    ]
    hessians = [
        torch.autograd.grad(sum(gradient.square().sum() for gradient in row), params)
        for row, params in zip(gradients, leaves, strict=True)
    ]
    for first, second in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(first, second, rtol=3e-6, atol=3e-7)
    independent_gradient = torch.autograd.grad(
        _independent_twisted_posterior_loss(model, x, y, query), leaves[0], create_graph=True
    )
    independent_hessian = torch.autograd.grad(
        sum(g.square().sum() for g in independent_gradient), leaves[0]
    )
    for actual_row, expected_row in (
        (gradients[0], independent_gradient),
        (hessians[0], independent_hessian),
    ):
        for first, second in zip(actual_row, expected_row, strict=True):
            torch.testing.assert_close(first, second, rtol=3e-6, atol=3e-7)
    if outputs == 1:
        for first, second in zip(hessians[0], hessians[1], strict=True):
            torch.testing.assert_close(first, second, rtol=3e-6, atol=3e-7)
    else:
        # The original source's independent-MVN operator has an incorrect
        # double backward here. Check its unmodified first gradients by finite
        # differences, independently of the dense Gaussian oracle above.
        start = [p.detach().clone() for p in leaves[1]]
        scale = max(1.0, *(float((2 * g.detach()).abs().max()) for g in gradients[1]))
        direction = [2 * g.detach() / scale for g in gradients[1]]
        errors = []
        try:
            for step in (2e-5, 1e-5):
                differences = []
                for sign in (-1, 1):
                    with torch.no_grad():
                        for p, original, d in zip(leaves[1], start, direction, strict=True):
                            p.copy_(original + sign * step * d)
                    reference.prediction_strategy = None
                    with propagate_grads(True), gpytorch.settings.fast_pred_var(False):
                        posterior = reference.posterior(ref_query)
                        loss = (
                            posterior.mean.square().sum()
                            + 0.7 * posterior.variance.sum()
                            + 0.13 * posterior.distribution.covariance_matrix.square().sum()
                        )
                    differences.append(torch.autograd.grad(loss, leaves[1]))
                finite_difference = [
                    (positive - negative) / (2 * step)
                    for negative, positive in zip(*differences, strict=True)
                ]
                errors.append(
                    max(
                        float((fd - h / scale).abs().max())
                        for fd, h in zip(finite_difference, hessians[0], strict=True)
                    )
                )
            assert errors[-1] < 1e-6
            assert errors[-1] < errors[0]
        finally:
            with torch.no_grad():
                for p, original in zip(leaves[1], start, strict=True):
                    p.copy_(original)
            reference.prediction_strategy = None


@pytest.mark.parametrize("device", DEVICES)
def test_single_task_forward_is_prior_while_call_is_conditional_and_mll_uses_override(device):
    x, y, _ = _inputs("single", device)
    model = TwistedGP(x, y, backend="torch", outcome_transform=None)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    prior = model.forward(model.train_inputs[0])
    expected = torch.distributions.MultivariateNormal(
        prior.mean,
        prior.covariance_matrix + torch.diag_embed(model.likelihood.noise.expand(len(x))),
    ).log_prob(y.squeeze(-1))
    expected = (expected + model._log_prior()) / len(x)
    torch.testing.assert_close(mll(), expected)
    actual_gradient = torch.autograd.grad(mll(), model.raw_twist)[0]
    expected_gradient = torch.autograd.grad(expected, model.raw_twist)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    model.eval()
    query = x.new_tensor([[0.2], [0.63]])
    prior = model.forward(query)
    conditional = model(query)
    torch.testing.assert_close(conditional.mean, model.posterior(query).mean.squeeze(-1))
    assert not torch.allclose(prior.mean, conditional.mean)
    with torch.no_grad():
        before = model.posterior(query).mean.clone()
        model.raw_twist.add_(0.2)
        after = model.posterior(query).mean
    assert not torch.allclose(before, after)
