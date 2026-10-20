"""Preference acquisition values and complete first/second device graphs."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_pairwise_gp import DEVICES, _data, _pair

from botorch_rs.acquisition import preference as own
from botorch_rs.acquisition.objective import GenericMCObjective
from botorch_rs.sampling.normal import IIDNormalSampler

NAMES = [
    "AnalyticExpectedUtilityOfBestOption",
    "qExpectedUtilityOfBestOption",
    "PairwiseBayesianActiveLearningByDisagreement",
]


def _kwargs(name, like, *, reference=False, winner=False, pending=False):
    if name == NAMES[0]:
        return (
            {"previous_winner": like.new_tensor([[0.2, 0.5]], requires_grad=True)} if winner else {}
        )
    if name == NAMES[1]:
        from botorch.sampling.normal import IIDNormalSampler as ReferenceSampler

        sampler = (ReferenceSampler if reference else IIDNormalSampler)(
            sample_shape=torch.Size([32]), seed=21
        )
        return {
            "sampler": sampler,
            **({"X_pending": like.new_tensor([[0.15, 0.3]])} if pending else {}),
        }
    return {"num_samples": 32, "std_noise": like.new_tensor(0.12, requires_grad=True)}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("batch", [False, True])
def test_independent_official_preference_values_and_query_gradients(device, name, batch):
    from botorch.acquisition import preference as reference

    models, _, inputs = _pair(device, "probit", batch=batch)
    results = []
    for i, model in enumerate(models):
        query = inputs[i].new_tensor([[[0.25, 0.35], [0.6, 0.4]]], requires_grad=True)
        kwargs = _kwargs(name, query, reference=bool(i), pending=True)
        acquisition = getattr(reference if i else own, name)(pref_model=model, **kwargs)
        torch.manual_seed(13)
        evaluated = query.expand(2, -1, -1) if i and batch else query
        value = acquisition(evaluated)
        gradient = torch.autograd.grad(value.sum(), query)[0]
        results.append((value, gradient))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-7)


class _ReferenceOnPort(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model
        self.num_outputs = 1

    @property
    def datapoints(self):
        return self.base.datapoints

    @property
    def batch_shape(self):
        return self.base.batch_shape

    def posterior(self, X, posterior_transform=None):
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        posterior = self.base.posterior(X, posterior_transform=posterior_transform)
        return GPyTorchPosterior(
            MultivariateNormal(posterior.mean.squeeze(-1), posterior.distribution.covariance_matrix)
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("outcome", [False, True])
def test_all_data_query_parameter_second_graphs_against_unmodified_official_formula(
    device, name, outcome
):
    from botorch.acquisition import preference as reference
    from botorch.acquisition.objective import GenericMCObjective as ReferenceObjective

    from botorch_rs.models.pairwise_gp import PairwiseGP

    x, comparisons = _data(device, gradients=True)
    model = PairwiseGP(x, comparisons, backend="torch")
    query = x.new_tensor([[0.25, 0.35], [0.6, 0.4]], requires_grad=True)
    weight = x.new_tensor([1.1, 0.9], requires_grad=True)
    bias = x.new_tensor([0.03, -0.02], requires_grad=True)

    class Outcome(torch.nn.Module):
        def forward(self, X):
            return X * weight + bias

    results = []
    for i in range(2):
        winner = name == NAMES[0]
        kwargs = _kwargs(name, query, reference=bool(i), winner=winner, pending=True)
        if name == NAMES[1]:

            def objective(samples, X):
                return samples.squeeze(-1).square() + 0.1 * X[..., 0]

            kwargs["objective"] = (ReferenceObjective if i else GenericMCObjective)(objective)
        if outcome:
            kwargs["outcome_model"] = Outcome()
        acquisition = getattr(reference if i else own, name)(
            pref_model=_ReferenceOnPort(model) if i else model, **kwargs
        )
        evaluated = query[:1] if winner else query
        targets = (x, query, *(p for p in model.parameters() if p.requires_grad))
        if outcome:
            targets = (*targets, weight, bias)
        if winner:
            targets = (*targets, kwargs["previous_winner"])
        if name == NAMES[2]:
            targets = (*targets, kwargs["std_noise"])
        torch.manual_seed(13)
        value = acquisition(evaluated)
        first = torch.autograd.grad(value.sum(), targets, create_graph=True)
        direction = tuple(
            torch.linspace(0.07, 0.19, t.numel(), dtype=t.dtype, device=t.device).reshape(t.shape)
            for t in targets
        )
        second = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(first, direction, strict=True)), targets
        )
        if i and name == NAMES[1]:
            # TriangularLinearOperator's generic bilinear derivative drops
            # higher-order graphs. Its first derivatives remain independent
            # reference oracles, so differentiate those by finite changes.
            originals = tuple(target.detach().clone() for target in targets)
            epsilon = 2e-5
            parts = []
            for sign in (-1, 1):
                with torch.no_grad():
                    for target, original, step in zip(targets, originals, direction, strict=True):
                        target.copy_(original + sign * epsilon * step)
                changed = acquisition(evaluated)
                parts.append(torch.autograd.grad(changed.sum(), targets))
            with torch.no_grad():
                for target, original in zip(targets, originals, strict=True):
                    target.copy_(original)
            second = tuple(
                (right - left) / (2 * epsilon) for left, right in zip(*parts, strict=True)
            )
        results.append((value, *first, *second))
    for index, (actual, expected) in enumerate(zip(*results, strict=True)):
        is_fd = name == NAMES[1] and index > len(targets)
        torch.testing.assert_close(
            actual, expected, atol=3e-6 if is_fd else 3e-8, rtol=2e-5 if is_fd else 3e-7
        )


@pytest.mark.parametrize("device", DEVICES)
def test_public_preference_query_optimizer_and_value_grad(device):
    from botorch_rs.models.pairwise_gp import PairwiseGP
    from botorch_rs.optim.optimize import optimize_acqf

    x, comparisons = _data(device)
    model = PairwiseGP(x, comparisons, backend="torch")
    acquisition = own.AnalyticExpectedUtilityOfBestOption(pref_model=model)
    query = x.new_tensor([[0.25, 0.35], [0.6, 0.4]], requires_grad=True)
    value, gradient = acquisition.value_and_grad(query)
    assert value.device == x.device
    assert gradient.shape == query.shape
    assert bool(torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all())
    candidates, values = optimize_acqf(
        acquisition,
        x.new_tensor([[0.0, 0.0], [1.0, 1.0]]),
        q=2,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 3},
    )
    assert candidates.device == x.device
    assert candidates.shape == (2, 2)
    assert bool(((candidates >= 0) & (candidates <= 1)).all())
    assert bool(torch.isfinite(values).all())
