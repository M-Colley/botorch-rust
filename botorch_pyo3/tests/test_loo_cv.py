"""LOO predictions, array types, sampling and batched outcome untransforms."""

from __future__ import annotations

import pytest

from botorch_rs.cross_validation import efficient_loo_cv, ensemble_loo_cv, loo_cv
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms import Standardize
from helpers import as_kind, assert_close, check_kind, make_data, shape


def test_loo_array_types(kind: str) -> None:
    x, y = make_data(n=7)
    model = SingleTaskGP(as_kind(x, kind), as_kind(y, kind))
    result = loo_cv(model)
    assert result.model is model
    assert not model.training
    for value in (result.posterior.mean, result.posterior.variance, result.observed_Y):
        check_kind(value, kind)
        assert shape(value) == (7, 1, 1)
    assert result.observed_Yvar is None
    assert shape(result.posterior.rsample((2, 3))) == (2, 3, 7, 1, 1)
    assert_close(result.observed_Y, [[v] for v in y])
    assert_close(efficient_loo_cv(model).posterior.mean, result.posterior.mean)


def test_ensemble_requires_ensemble() -> None:
    x, y = make_data(n=4)
    with pytest.raises(NotImplementedError, match="ensemble model"):
        ensemble_loo_cv(SingleTaskGP(x, y))


def test_batched_standardize_posterior(kind: str) -> None:
    from botorch_rs.cross_validation import _DiagonalPosterior

    y = as_kind([[[0.0], [2.0]], [[10.0], [16.0]]], kind)
    transform = Standardize(m=1, batch_shape=(2,))
    transform(y)
    mean = as_kind([[[0.0]], [[0.0]]], kind)
    var = as_kind([[[1.0]], [[1.0]]], kind)
    posterior = transform.untransform_posterior(_DiagonalPosterior(mean, var, model_kind(mean)))
    assert_close(posterior.mean, [[[1.0]], [[13.0]]])
    assert_close(posterior.variance, [[[2.0]], [[18.0]]])
    # Batched statistics must broadcast across a leading sample dimension.
    base = as_kind([[[0.0], [0.0]], [[1.0], [1.0]]], kind)
    samples = posterior.rsample_from_base_samples((2,), base)
    assert_close(samples[0], [[[1.0]], [[13.0]]])


def test_stratified_loo_untransform(kind: str) -> None:
    import math

    from botorch_rs.models.transforms.outcome import StratifiedStandardize
    from helpers import flatten

    x = as_kind([[0.1, 0.0], [0.3, 0.0], [0.7, 1.0], [0.9, 1.0]], kind)
    y = as_kind([[1.0], [3.0], [10.0], [16.0]], kind)
    transform = StratifiedStandardize(1, as_kind([0.0, 1.0], kind))
    model = SingleTaskGP(x, y, outcome_transform=transform)
    raw = loo_cv(model, untransform=False)
    actual = loo_cv(model)
    offsets = [2.0, 2.0, 13.0, 13.0]
    scales = [math.sqrt(2.0), math.sqrt(2.0), math.sqrt(18.0), math.sqrt(18.0)]
    assert_close(
        flatten(actual.posterior.mean),
        [o + s * m for o, s, m in zip(offsets, scales, flatten(raw.posterior.mean), strict=True)],
    )
    assert_close(
        flatten(actual.posterior.variance),
        [s * s * v for s, v in zip(scales, flatten(raw.posterior.variance), strict=True)],
    )
    assert_close(flatten(actual.observed_Y), flatten(y))


def model_kind(value):
    from botorch_rs._arrays import kind_of

    return kind_of(value)
