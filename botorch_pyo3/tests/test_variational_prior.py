"""Inner SVGP prior semantics, distinct from its variational strategy output.

BoTorch b0dd01be's _SingleTaskVariationalGP.forward constructs the mean/kernel
prior. Installed BoTorch 0.18.1 has the same implementation; its GPyTorch
strategy provides the independent-task conversion for prior=True.
"""

from __future__ import annotations

import copy

import pytest

from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape

importorskip_botorch()
torch = pytest.importorskip("torch")

from approximate_helpers import (  # noqa: E402
    CASES,
    Case,
    build_pair,
    elbo_pair,
    set_random_parameters,
    strategy,
)

from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.models import SingleTaskVariationalGP  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

PRIOR_CASES = [*CASES[:1], CASES[3], CASES[4], CASES[5], CASES[7], CASES[8]]
PRIOR_CASES.append(Case("zero_mean_two_outputs", m=2, mean="zero"))


def _prior(model, x, method):
    return model.model.forward(x) if method == "forward" else model.model(x, prior=True)


@pytest.mark.parametrize("case", [CASES[0], CASES[5], CASES[6]], ids=lambda case: case.name)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("training", [False, True])
def test_variational_location_matches_official_flattened_task_layout(case, kind, training):
    reference, model, x, _ = build_pair(case)
    set_random_parameters(reference, model, 67)
    reference.train(training)
    model.train(training)
    actual = model.model(as_kind(x[:5].tolist(), kind))
    expected = reference.model(x[:5])
    assert shape(actual.mean) == ((5, case.m) if case.m > 1 else (5,))
    assert shape(actual.loc) == (5 * case.m,)
    for name in ("mean", "loc"):
        check_kind(getattr(actual, name), kind)
        assert_close(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("case", PRIOR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("method", ["forward", "prior"])
@pytest.mark.parametrize("batched", [False, True])
def test_inner_prior_matches_official_shapes_and_values(case, method, batched):
    reference, model, x, _ = build_pair(case)
    set_random_parameters(reference, model, 34)
    if case.mean == "zero" and case.m > 1 and method == "prior" and not batched:
        from gpytorch.means import ZeroMean

        # GPyTorch's unbatched ZeroMean produces a rank-one prior mean, which
        # from_batch_mvn cannot convert. Use the dimension-correct reference.
        reference.model.mean_module = ZeroMean(batch_shape=torch.Size([case.m]))
    query = x[:5].clone()
    if batched:
        # The final batch axis broadcasts against the independent-output axis.
        query = torch.stack([query, query + 0.07, query - 0.11]).unsqueeze(1)
    actual, expected = _prior(model, query, method), _prior(reference, query, method)
    assert actual.batch_shape == tuple(expected.batch_shape)
    assert actual.event_shape == tuple(expected.event_shape)
    for name in ("mean", "loc", "variance", "covariance_matrix"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), rtol=2e-9, atol=2e-10
        )
    # Changing train/eval state must not introduce the variational correction.
    model.eval()
    for name in ("mean", "variance", "covariance_matrix"):
        torch.testing.assert_close(
            getattr(_prior(model, query, method), name), getattr(expected, name)
        )


@pytest.mark.parametrize("method", ["forward", "prior"])
def test_task_specific_input_batches_keep_the_correct_output_covariances(method):
    reference, model, x, _ = build_pair(CASES[5])
    set_random_parameters(reference, model, 42)
    query = torch.stack([x[:4], x[4:8]])
    actual, expected = _prior(model, query, method), _prior(reference, query, method)
    assert actual.batch_shape == tuple(expected.batch_shape)
    assert actual.event_shape == tuple(expected.event_shape)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix)


@pytest.mark.parametrize("case", [CASES[0], CASES[5]], ids=lambda case: case.name)
def test_prior_calls_bypass_variational_initialization_and_random_draws(case):
    reference, model, x, _ = build_pair(case)
    rng_state = torch.get_rng_state().clone()
    for current in (reference, model):
        assert int(strategy(current).variational_params_initialized) == 0
        for method in ("forward", "prior"):
            output = _prior(current, x[:4], method)
            _ = output.mean, output.variance, output.covariance_matrix
        assert int(strategy(current).variational_params_initialized) == 0
    assert torch.equal(torch.get_rng_state(), rng_state)
    # A normal variational call still initializes q(u).
    assert model.model(x[:4]).event_shape == ((4, case.m) if case.m > 1 else (4,))
    assert int(strategy(model).variational_params_initialized) == 1


def test_prior_true_normalizes_one_dimensional_inputs_like_gpytorch():
    reference, model, x, _ = build_pair(Case("one_feature", d=1))
    query = x[:4, 0]
    actual, expected = model.model(query, prior=True), reference.model(query, prior=True)
    assert actual.event_shape == (4,)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix)


@pytest.mark.parametrize("case", [CASES[7], CASES[8]], ids=lambda case: case.name)
def test_inner_prior_uses_supplied_transformed_space_and_outer_forward_stays_variational(case):
    reference, model, x, _ = build_pair(case)
    set_random_parameters(reference, model, 25)
    transformed = reference.transform_inputs(x[:4])
    torch.testing.assert_close(model.transform_inputs(x[:4]), transformed)
    raw_prior = model.model.forward(x[:4])
    scaled_prior = model.model.forward(transformed)
    torch.testing.assert_close(
        raw_prior.covariance_matrix, reference.model.forward(x[:4]).covariance_matrix
    )
    torch.testing.assert_close(
        scaled_prior.covariance_matrix, reference.model.forward(transformed).covariance_matrix
    )
    assert not torch.allclose(raw_prior.covariance_matrix, scaled_prior.covariance_matrix)
    # Outer forward applies the static input transform in train mode and q(u).
    actual, expected = model(x[:4]), reference(x[:4])
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix)
    assert not torch.allclose(actual.mean, _prior(model, transformed, "prior").mean)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("method", ["forward", "prior"])
@pytest.mark.parametrize("materialize", [False, True])
def test_prior_laws_snapshot_inputs_and_native_parameters(kind, outputs, method, materialize):
    x = [[0.1, 0.2], [0.3, 0.8], [0.7, 0.5], [0.9, 0.4]]
    y = [[row[0] - 0.2 * t for t in range(outputs)] for row in x]
    model = SingleTaskVariationalGP(as_kind(x, kind), as_kind(y, kind), inducing_points=2)
    query = as_kind(copy.deepcopy(x), kind)
    output = _prior(model, query, method)
    expected = _prior(copy.deepcopy(model), as_kind(x, kind), method)
    if materialize:
        _ = output.mean, output.covariance_matrix
    model.model.mean_module.constant = 0.6
    model.model.covar_module.lengthscale = 0.9
    with torch.no_grad():
        query[0][0] = 0.95
    for name in ("mean", "variance", "covariance_matrix"):
        check_kind(getattr(output, name), kind)
        assert_close(getattr(output, name), getattr(expected, name), rtol=0, atol=0)
    assert int(strategy(model).variational_params_initialized) == 0


def test_prior_flushes_pending_torch_leaves_before_snapshot():
    reference, model, x, _ = build_pair(CASES[5])
    with torch.no_grad():
        for current in (model, reference):
            current.model.mean_module.raw_constant.copy_(torch.tensor([0.8, -0.2]))
            current.model.covar_module.raw_lengthscale.add_(0.31)
    actual = model.model(x[:3], prior=True)
    expected = reference.model(x[:3], prior=True)
    # Subsequent edits cannot affect either eagerly or lazily read moments.
    with torch.no_grad():
        model.model.mean_module.raw_constant.add_(0.4)
        model.model.covar_module.raw_lengthscale.add_(0.7)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix)


def test_zero_mean_multitask_prior_fixes_reference_unbroadcast_mean_conversion():
    from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
    from gpytorch.means import ZeroMean

    reference, model, x, _ = build_pair(Case("zero_two", m=2, mean="zero"))
    with pytest.raises(RuntimeError, match="mean should be a matrix"):
        reference.model(x[:4], prior=True)
    actual = model.model(x[:4], prior=True)
    assert actual.event_shape == (4, 2)
    assert torch.count_nonzero(actual.mean) == 0
    # Upstream solution: expand the base prior mean to batch_shape + event_shape
    # before from_batch_mvn, or use this explicit output-batched mean module.
    base = reference.model.forward(x[:4])
    repaired = MultitaskMultivariateNormal.from_batch_mvn(
        MultivariateNormal(
            base.mean.expand(base.batch_shape + base.event_shape), base.lazy_covariance_matrix
        )
    )
    torch.testing.assert_close(actual.mean, repaired.mean)
    torch.testing.assert_close(actual.covariance_matrix, repaired.covariance_matrix)
    reference.model.mean_module = ZeroMean(batch_shape=torch.Size([2]))
    expected = reference.model(x[:4], prior=True)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix)


@pytest.mark.parametrize("case", [CASES[0], CASES[5]], ids=lambda case: case.name)
@pytest.mark.parametrize("method", ["forward", "prior"])
@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("minibatch", [False, True])
def test_elbo_rejects_prior_laws_instead_of_recomputing_variational_outputs(
    case, method, combine, minibatch
):
    reference, model, _, _ = build_pair(case)
    _, mll = elbo_pair(reference, model, case.n, combine=combine)
    x, y = model.model.train_inputs[0], model.model.train_targets
    if minibatch:
        x, y = x[::3], y[::3]
    output = _prior(model, x, method)
    for evaluate in (mll, mll._terms):
        with pytest.raises(UnsupportedError, match=r"requires q\(f\)"):
            evaluate(output, y)
    assert int(strategy(model).variational_params_initialized) == 0
    # The supported q(f) path remains trainable and uses the supplied data.
    loss = mll(model.model(x), y)
    loss = loss if combine else loss[0] - loss[1] + loss[2]
    assert loss.requires_grad
    assert all(g is not None for g in torch.autograd.grad(loss, tuple(mll.parameters())))


@pytest.mark.parametrize("method", ["forward", "prior"])
def test_prior_rejects_incompatible_input_and_task_batches(method):
    _, model, x, _ = build_pair(CASES[5])
    query = x[:4].expand(3, 4, 2)
    with pytest.raises(RuntimeError, match="batch shapes do not broadcast"):
        _prior(model, query, method)
