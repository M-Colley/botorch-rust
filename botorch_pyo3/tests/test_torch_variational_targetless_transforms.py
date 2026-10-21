"""Optional observations do not prevent attaching an outcome transform."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models import SingleTaskVariationalGP
from botorch_rs.models.transforms.outcome import (
    Bilog,
    ChainedOutcomeTransform,
    Log,
    Power,
    Standardize,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def outcome(kind, like, reference):
    module = __import__(
        "botorch.models.transforms.outcome"
        if reference
        else "botorch_rs.models.transforms.outcome",
        fromlist=["Log"],
    )
    if kind == "standardize":
        result = module.Standardize(1)
        result(like.new_tensor([[0.4], [0.9], [1.4], [1.9]]))
        return result.eval()
    if kind == "chain":
        return module.ChainedOutcomeTransform(log=module.Log(), power=module.Power(0.5))
    return getattr(module, kind)(**({"power": 0.5} if kind == "Power" else {}))


def model_pair(device, kind, factory=False):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskVariationalGP as Source

    if factory:
        from test_torch_variational_factories import PlainStrategy

    results = []
    for reference, cls in enumerate((SingleTaskVariationalGP, Source)):
        x = torch.linspace(0.1, 0.9, 7, device=device, dtype=torch.double).unsqueeze(-1)
        model = cls(
            x,
            inducing_points=x[[0, 3, 6]],
            outcome_transform=outcome(kind, x, reference),
            **({"variational_strategy": PlainStrategy} if factory else {}),
            **({} if reference else {"backend": "torch"}),
        ).to(x)
        strategy = model.model.variational_strategy
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.copy_(
                x.new_tensor([-0.2, 0.1, 0.3])
            )
            strategy._variational_distribution.chol_variational_covar.copy_(
                x.new_tensor([[0.8, 0, 0], [0.1, 0.9, 0], [0.02, -0.1, 0.7]])
            )
        results.append(model)
    return results


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["Log", "Bilog", "Power", "standardize", "chain"])
@pytest.mark.parametrize("factory", [False, True])
def test_targetless_outcome_values_and_all_first_derivatives(device, kind, factory):
    values, gradients = [], []
    for model in model_pair(device, kind, factory):
        query = torch.tensor(
            [[0.23], [0.61]], device=device, dtype=torch.double, requires_grad=True
        )
        posterior = model.posterior(query, observation_noise=True)
        shape = torch.Size([3])
        base = torch.linspace(
            -0.3, 0.5, 3 * posterior.base_sample_shape.numel(), device=device, dtype=torch.double
        ).reshape(shape + posterior.base_sample_shape)
        samples = posterior.rsample_from_base_samples(shape, base)
        values.append(samples)
        parameters = dict(model.named_parameters())
        gradients.append(
            torch.autograd.grad(
                samples.sin().sum() + 0.3 * samples.square().sum(),
                (query, *(parameters[n] for n in sorted(parameters))),
            )
        )
    torch.testing.assert_close(*values, atol=3e-9, rtol=3e-9)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_unfitted_targetless_standardization_preserves_the_source_error(device):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskVariationalGP as Source
    from botorch.models.transforms.outcome import Standardize as SourceStandardize

    x = torch.linspace(0.1, 0.9, 7, device=device, dtype=torch.double).unsqueeze(-1)
    for cls, transform in (
        (SingleTaskVariationalGP, Standardize(1)),
        (Source, SourceStandardize(1)),
    ):
        model = cls(
            x,
            inducing_points=x[:3],
            outcome_transform=transform,
            **({"backend": "torch"} if cls is SingleTaskVariationalGP else {}),
        )
        with pytest.raises(RuntimeError, match="before"):
            model.posterior(x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transform", [Log, Bilog, Power, ChainedOutcomeTransform])
def test_targetless_port_transform_attaches_without_fabricated_targets(device, transform):
    x = torch.linspace(0.1, 0.9, 7, device=device, dtype=torch.double).unsqueeze(-1)
    specification = (
        transform(log=Log(), power=Power(0.5))
        if transform is ChainedOutcomeTransform
        else transform(0.5)
        if transform is Power
        else transform()
    )
    model = SingleTaskVariationalGP(
        x, inducing_points=x[:3], outcome_transform=specification, backend="torch"
    )
    assert model._train_Y is None
    model.posterior(x)
    for cloned in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        actual, expected = cloned.posterior(x), model.posterior(x)
        shape = torch.Size([2])
        base = x.new_full(shape + actual.base_sample_shape, 0.2)
        torch.testing.assert_close(
            actual.rsample_from_base_samples(shape, base),
            expected.rsample_from_base_samples(shape, base),
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("factory", [False, True])
def test_stratified_variational_training_and_query_contexts_preserve_full_graphs(
    device, chain, factory
):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskVariationalGP as Source
    from test_torch_variational_factories import PlainStrategy
    from test_torch_variational_shared_distribution import DenseKernel, DenseMean

    records, firsts, seconds = [], [], []
    for reference, cls in enumerate((SingleTaskVariationalGP, Source)):
        module = __import__(
            "botorch.models.transforms.outcome"
            if reference
            else "botorch_rs.models.transforms.outcome",
            fromlist=["StratifiedStandardize"],
        )
        x = torch.tensor(
            [[0.1, 0], [0.2, 1], [0.4, 0], [0.5, 1], [0.8, 0], [0.9, 1]],
            dtype=torch.double,
            device=device,
        )
        y = x.new_tensor([[0.4], [1.2], [0.8], [2.1], [1.3], [1.7]], requires_grad=True)
        transform = module.StratifiedStandardize(-1, x.new_tensor([0, 1]))
        if chain:
            transform = module.ChainedOutcomeTransform(log=module.Log(), strata=transform)
        model = cls(
            x,
            y,
            inducing_points=x[[0, 3, 4]],
            covar_module=DenseKernel((), x),
            mean_module=DenseMean((), x),
            outcome_transform=transform,
            **({"variational_strategy": PlainStrategy} if factory else {}),
            **({} if reference else {"backend": "torch"}),
        ).to(x)
        strategy = model.model.variational_strategy
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.copy_(
                x.new_tensor([-0.2, 0.1, 0.3])
            )
            strategy._variational_distribution.chol_variational_covar.copy_(
                torch.eye(3, dtype=x.dtype, device=x.device) * 0.8
            )
        query = x.new_tensor([[0.31, 1], [0.62, 0]], requires_grad=True)
        posterior = model.posterior(query, observation_noise=True)
        records.append((posterior.mean, posterior.variance, model.model.train_targets))
        leaves = (query, y, *(p for _, p in sorted(model.named_parameters())))
        loss = posterior.mean.sin().sum() + 0.03 * posterior.variance.square().sum()
        first = torch.autograd.grad(loss, leaves, create_graph=True)
        firsts.append(first)
        if not reference:
            latent = model.model(query)
            values = y.log() if chain else y
            groups = [values[x[:, -1] == task] for task in (0, 1)]
            means = torch.stack([group.mean() for group in groups])
            stds = torch.stack([group.std() for group in groups])
            indices = query[:, -1].long()
            gaussian_mean = latent.mean * stds[indices] + means[indices]
            gaussian_var = (latent.variance + model.likelihood.noise) * stds[indices].square()
            if chain:
                manual_mean = (gaussian_mean + gaussian_var / 2).exp()
                manual_variance = gaussian_var.expm1() * (2 * gaussian_mean + gaussian_var).exp()
            else:
                manual_mean, manual_variance = gaussian_mean, gaussian_var
            manual_loss = manual_mean.sin().sum() + 0.03 * manual_variance.square().sum()
            manual_first = torch.autograd.grad(manual_loss, leaves, create_graph=True)
            for actual, expected in zip(first, manual_first, strict=True):
                torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
            for derivative in (first, manual_first):
                seconds.append(
                    torch.autograd.grad(
                        sum(g.square().sum() / 2 for g in derivative), leaves, retain_graph=True
                    )
                )
    for actual, expected in zip(*records, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    # The source's masked nanstd calibration has NaN Y derivatives (#3520).
    # Its values and other first derivatives remain useful oracle controls;
    # the independent per-stratum formula above verifies every port leaf/HVP.
    assert torch.isnan(firsts[1][1]).all()
    for actual, expected in zip(
        (firsts[0][0], *firsts[0][2:]), (firsts[1][0], *firsts[1][2:]), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    for actual, expected in zip(*seconds, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)
