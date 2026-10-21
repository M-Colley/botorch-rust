"""HigherOrderGP transformed fantasies and supplied-factor variances."""

from __future__ import annotations

import math

import pytest

from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models.higher_order_gp import HigherOrderGP as Reference  # noqa: E402
from botorch.models.transforms.input import Normalize as ReferenceNormalize  # noqa: E402
from linear_operator.operators import KroneckerProductLinearOperator  # noqa: E402
from test_10c_higher_order import data  # noqa: E402

from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.models._higher_order_variance import posterior_variances  # noqa: E402
from botorch_rs.models.higher_order_gp import HigherOrderGP  # noqa: E402
from botorch_rs.models.transforms.input import Normalize  # noqa: E402


def paired(standardize=True):
    x, y = (torch.tensor(a, dtype=torch.float64) for a in data())
    x = 2 + 3 * x
    bounds = torch.stack((x.amin(0), x.amax(0)))
    kwargs = {} if standardize else {"outcome_transform": None}
    torch.manual_seed(47)
    reference = Reference(x, y, input_transform=ReferenceNormalize(2, bounds=bounds), **kwargs)
    torch.manual_seed(47)
    native = HigherOrderGP(x, y, input_transform=Normalize(2, bounds=bounds), **kwargs)
    return native, reference


def dense_reference_mean(model, query):
    train = model.train_inputs[0]
    latent = torch.ones(1, 1, dtype=query.dtype)
    for kernel, coordinates in zip(model.covar_modules[1:], model.latent_parameters, strict=True):
        latent = torch.kron(latent, kernel(coordinates).to_dense())
    data = model.covar_modules[0](train).to_dense()
    cross = model.covar_modules[0](model.transform_inputs(query), train).to_dense()
    batch = data.shape[:-2]
    count = math.prod(batch)
    targets = model.train_targets.reshape(count, -1)
    data = data.reshape(count, *data.shape[-2:])
    cross = cross.reshape(count, *cross.shape[-2:])
    means = []
    for k, c, y in zip(data, cross, targets, strict=True):
        full = torch.kron(k.contiguous(), latent.contiguous())
        full = (
            full
            + torch.eye(full.shape[-1], dtype=query.dtype) * model.likelihood.noise.reshape(-1)[0]
        )
        means.append(torch.kron(c.contiguous(), latent.contiguous()) @ torch.linalg.solve(full, y))
    result = torch.stack(means).reshape(*batch, query.shape[-2], *model.target_shape)
    if hasattr(model, "outcome_transform"):
        result = result * model.outcome_transform.stdvs.reshape(
            ()
        ) + model.outcome_transform.means.reshape(())
    return result


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("fantasies", [False, True])
def test_get_fantasy_model_already_transformed_targets_matches_official(standardize, fantasies):
    model, reference = paired(standardize)
    query = torch.tensor([[2.4, 4.2], [4.4, 3.3]], dtype=torch.float64)
    target = torch.linspace(1.2, 3.2, 12, dtype=torch.float64).reshape(2, 2, 3)
    if fantasies:
        target = torch.stack((target, target + 0.7))
    reference.posterior(query)
    model.eval()
    transformed = model.outcome_transform(target)[0] if standardize else target
    expected_target = reference.outcome_transform(target)[0] if standardize else target
    fantasy = model.get_fantasy_model(model.transform_inputs(query), transformed)
    expected = reference.get_fantasy_model(reference.transform_inputs(query), expected_target)
    assert_close(fantasy.train_targets, expected.train_targets, rtol=2e-8, atol=2e-9)
    assert_close(fantasy.train_inputs[0], expected.train_inputs[0], rtol=2e-8, atol=2e-9)
    if fantasies:
        # Official fantasy caches have generic AddedDiag/BatchRepeat wrappers
        # rather than the exact Kronecker wrapper assumed by HOGP.posterior.
        with pytest.raises(AttributeError, match="linear_op"):
            expected.posterior(query)
        expected_mean = dense_reference_mean(expected, query)
    else:
        expected_mean = expected.posterior(query).mean
    assert_close(fantasy.posterior(query).mean, expected_mean, rtol=2e-7, atol=2e-8)
    raw_conditioned = model.condition_on_observations(query, target)
    assert_close(fantasy.train_targets, raw_conditioned.train_targets, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_fantasy_flattening_broadcast_and_kwargs(kind):
    x, y = data()
    model = HigherOrderGP(as_kind(x, kind), as_kind(y, kind), outcome_transform=None)
    query = as_kind([[[0.2, 0.4], [0.6, 0.7]], [[0.3, 0.5], [0.7, 0.6]]], kind)
    targets = torch.linspace(0.1, 1.1, 3 * 1 * 2 * 2 * 3, dtype=torch.float64).reshape(
        3, 1, 2, 2, 3
    )
    fantasy = model.get_fantasy_model(query, as_kind(targets.tolist(), kind))
    assert fantasy.batch_shape == (3, 2)
    assert shape(fantasy.train_inputs[0]) == (3, 2, 7, 2)
    assert shape(fantasy.train_targets) == (3, 2, 42)
    flat = model.get_fantasy_model(
        as_kind([[0.2, 0.4], [0.6, 0.7]], kind), as_kind(targets[0, 0].reshape(-1).tolist(), kind)
    )
    assert shape(flat.train_targets) == (42,)
    for fn in (model.condition_on_observations, model.get_fantasy_model):
        with pytest.raises(UnsupportedError):
            fn(as_kind([[0.2, 0.4]], kind), as_kind([[[0.2] * 3] * 2], kind), mystery=True)
        with pytest.raises(UnsupportedError):
            fn(as_kind([[0.2, 0.4]], kind), as_kind([[[0.2] * 3] * 2], kind), noise=0.1)


def factors(batch=False, requires_grad=False, dtype=torch.float64):
    generator = torch.Generator().manual_seed(56)
    shapes = [(7, 7), (2, 2), (3, 3)]
    if batch:
        shapes = [(2, 1, 7, 7), (1, 3, 2, 2), (3, 3)]
    out = []
    for dimensions in shapes:
        a = torch.randn(dimensions, generator=generator, dtype=dtype)
        out.append(
            (a @ a.transpose(-2, -1) + torch.eye(dimensions[-1], dtype=dtype) * 0.5).requires_grad_(
                requires_grad
            )
        )
    return out


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", [False, True])
def test_supplied_factors_honored_with_broadcast_batches(kind, batch):
    model, reference = paired()
    supplied = factors(batch)
    native_factors = [as_kind(a.tolist(), kind) for a in supplied]
    if kind != "torch":
        x, y = data()
        model = HigherOrderGP(as_kind(x, kind), as_kind(y, kind))
        model.likelihood.noise = float(reference.likelihood.noise[0])
    expected = reference.make_posterior_variances(KroneckerProductLinearOperator(*supplied))
    actual = model.make_posterior_variances(native_factors)
    check_kind(actual, kind)
    assert_close(actual, expected, rtol=2e-7, atol=2e-8)
    changed = [a.clone() for a in supplied]
    changed[0][..., 5:, :5] *= 0.4
    changed[0][..., :5, 5:] *= 0.4
    changed_actual = model.make_posterior_variances([as_kind(a.tolist(), kind) for a in changed])
    changed_expected = reference.make_posterior_variances(KroneckerProductLinearOperator(*changed))
    assert_close(changed_actual, changed_expected, rtol=2e-7, atol=2e-8)
    assert not torch.allclose(torch.as_tensor(actual), torch.as_tensor(changed_actual))


@pytest.mark.parametrize("batch", [False, True])
def test_supplied_operator_factor_graphs_against_official_and_dense_noise_gradient(batch):
    model, reference = paired()
    supplied = factors(batch, requires_grad=True)
    other = [a.detach().clone().requires_grad_() for a in supplied]
    av = model.make_posterior_variances(KroneckerProductLinearOperator(*supplied))
    bv = reference.make_posterior_variances(KroneckerProductLinearOperator(*other))
    ag = torch.autograd.grad(av.square().sum(), supplied)
    bg = torch.autograd.grad(bv.square().sum(), other)
    for a, b in zip(ag, bg, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)
    # Native covariance-factor adjoints also include noise broadcast reduction.
    noise = (
        torch.tensor([[0.13], [0.21]], dtype=torch.float64, requires_grad=True)
        if batch
        else torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
    )
    if batch:
        noise = noise.unsqueeze(-1)
    value = posterior_variances(supplied, 5, noise)
    gradient = torch.autograd.grad(value.sum(), noise)[0]
    h = 1e-5
    fd = (
        posterior_variances([a.detach() for a in supplied], 5, noise.detach() + h).sum()
        - posterior_variances([a.detach() for a in supplied], 5, noise.detach() - h).sum()
    ) / (2 * h)
    torch.testing.assert_close(gradient.sum(), fd, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_variance_dtype_no_grad_and_shape_errors(dtype):
    model, _ = paired()
    supplied = factors(requires_grad=True, dtype=dtype)
    result = model.make_posterior_variances(supplied)
    assert result.dtype == dtype and result.requires_grad
    with torch.no_grad():
        assert not model.make_posterior_variances(supplied).requires_grad
    with pytest.raises(UnsupportedError, match="Kronecker operator"):
        model.make_posterior_variances(torch.eye(7))
    with pytest.raises(ValueError):
        model.make_posterior_variances([torch.ones(3, 2)])
