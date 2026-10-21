"""Native prediction snapshots retain their Gaussian law across model updates."""

import pytest

from botorch_rs._arrays import to_flat
from botorch_rs.models import SingleTaskGP


def array(values, kind):
    if kind == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(values, dtype=torch.double)
    if kind == "numpy":
        numpy = pytest.importorskip("numpy")
        return numpy.asarray(values, dtype=numpy.float64)
    return values


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("outputs,batch", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_lazy_and_materialized_posterior_snapshots_and_model_copies_are_independent(
    kind, outputs, batch, fixed_noise
):
    x = [[i / 9, ((i * 3) % 10) / 10] for i in range(9)]
    y = [[row[0] ** 2 - row[1] + 0.1 * t for t in range(outputs)] for row in x]
    noise = [[0.04] * outputs for _ in x]
    query = [[0.21, 0.63], [0.78, 0.42]]
    if batch:
        x = [x, [[a + 0.03, b - 0.02] for a, b in x]]
        y = [y, [[v + 0.1 for v in row] for row in y]]
        noise = [noise, noise]
    model = SingleTaskGP(
        array(x, kind),
        array(y, kind),
        train_Yvar=array(noise, kind) if fixed_noise else None,
        outcome_transform=None,
        backend="native",
    )
    if kind == "torch":
        model.requires_grad_(False)
    q = array(query, kind)
    old = model.posterior(q)
    expected_mean = to_flat(old.mean)[0]
    expected_variance = to_flat(old.variance)[0]
    lazy_old = model.posterior(q)
    copied = model._rs.copy()
    model.mean_module.constant = 0.7
    model.covar_module.lengthscale = [[0.4, 0.8]]
    assert to_flat(old.mean)[0] == pytest.approx(expected_mean, abs=1e-12)
    assert to_flat(lazy_old.mean)[0] == pytest.approx(expected_mean, abs=1e-12)
    assert to_flat(lazy_old.variance)[0] == pytest.approx(expected_variance, abs=1e-12)
    native_mean, native_variance = copied.posterior_marginals(to_flat(q))
    assert native_mean[0] == pytest.approx(expected_mean, abs=1e-12)
    assert native_variance[0] == pytest.approx(expected_variance, abs=1e-12)
    updated = to_flat(model.posterior(q).mean)[0]
    assert max(abs(a - b) for a, b in zip(updated, expected_mean, strict=True)) > 1e-6
    raw = copied.raw_parameters()
    raw[0 if fixed_noise else 1] += 0.4
    copied.set_raw_parameters(raw)
    assert to_flat(model.posterior(q).mean)[0] == pytest.approx(updated, abs=1e-12)
    assert to_flat(lazy_old.mean)[0] == pytest.approx(expected_mean, abs=1e-12)
