"""Cross-validation utilities using batch evaluation mode
(``botorch.cross_validation``)."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


class CVFolds(NamedTuple):
    """Leave-one-out folds (BoTorch's ``CVFolds``)."""

    train_X: Any
    test_X: Any
    train_Y: Any
    test_Y: Any
    train_Yvar: Any = None
    test_Yvar: Any = None


class CVResults(NamedTuple):
    """Results from cross-validation (BoTorch's ``CVResults``): the batched
    model, its posterior at the held-out points (mean of shape ``n x 1 x
    m``), the observed values and noise variances."""

    model: Any
    posterior: Any
    observed_Y: Any
    observed_Yvar: Any = None


def _folds(x: Any, n_dim: int) -> tuple[Any, Any]:
    """``(train, test)`` folds of ``x`` (``batch x n x k``): ``batch x n x
    (n-1) x k`` without point ``i`` and ``batch x n x 1 x k`` with it."""
    data, shape = to_flat(x, "x")
    *batch, n, k = shape
    nb = 1
    for b in batch:
        nb *= b
    train: list[float] = []
    test: list[float] = []
    for bi in range(nb):
        base = bi * n * k
        for i in range(n):
            for j in range(n):
                row = data[base + j * k : base + (j + 1) * k]
                (test if j == i else train).extend(row)
    kind = kind_of(x)
    return (
        from_flat(train, (*batch, n, n - 1, k), kind),
        from_flat(test, (*batch, n, 1, k), kind),
    )


def gen_loo_cv_folds(train_X: Any, train_Y: Any, train_Yvar: Any = None) -> CVFolds:
    r"""Generate LOO CV folds w.r.t. to ``n`` (BoTorch's
    ``gen_loo_cv_folds``).

    Args:
        train_X: A ``n x d`` or ``batch_shape x n x d`` (batch mode) array of
            training features.
        train_Y: A ``n x (m)`` or ``batch_shape x n x (m)`` (batch mode)
            array of training observations.
        train_Yvar: An ``n x (m)`` or ``batch_shape x n x (m)`` (batch mode)
            array of observed measurement noise.

    Returns:
        CVFolds with ``train_X`` of shape ``n x (n-1) x d`` (``batch_shape x n
        x (n-1) x d``), ``test_X`` of shape ``n x 1 x d``, ``train_Y`` /
        ``train_Yvar`` of shape ``n x (n-1) x m`` and ``test_Y`` /
        ``test_Yvar`` of shape ``n x 1 x m``.

    Example:
        >>> train_X = torch.rand(10, 1)
        >>> train_Y = torch.rand_like(train_X)
        >>> cv_folds = gen_loo_cv_folds(train_X, train_Y)
        >>> cv_folds.train_X.shape
        torch.Size([10, 9, 1])
    """
    xshape = to_flat(train_X, "train_X")[1]
    ydata, yshape = to_flat(train_Y, "train_Y")
    if len(yshape) < len(xshape):
        train_Y = from_flat(ydata, (*yshape, 1), kind_of(train_Y))
        if train_Yvar is not None:
            vdata, vshape = to_flat(train_Yvar, "train_Yvar")
            train_Yvar = from_flat(vdata, (*vshape, 1), kind_of(train_Yvar))
    train_X_cv, test_X_cv = _folds(train_X, len(xshape))
    train_Y_cv, test_Y_cv = _folds(train_Y, len(xshape))
    if train_Yvar is None:
        train_Yvar_cv = test_Yvar_cv = None
    else:
        train_Yvar_cv, test_Yvar_cv = _folds(train_Yvar, len(xshape))
    return CVFolds(
        train_X=train_X_cv,
        test_X=test_X_cv,
        train_Y=train_Y_cv,
        test_Y=test_Y_cv,
        train_Yvar=train_Yvar_cv,
        test_Yvar=test_Yvar_cv,
    )


def batch_cross_validation(
    model_cls: Any,
    mll_cls: Any,
    cv_folds: CVFolds,
    fit_args: dict[str, Any] | None = None,
    observation_noise: bool = False,
    model_init_kwargs: dict[str, Any] | None = None,
) -> CVResults:
    r"""Perform cross validation by using batch mode (BoTorch's
    ``batch_cross_validation``): one batched model holds a GP per fold, its
    hyperparameters are fitted per fold (in parallel) and its posterior is
    evaluated at the held-out points.

    Args:
        model_cls: A botorch_rs model class (e.g. ``SingleTaskGP``) that
            initializes the likelihood internally.
        mll_cls: A marginal log likelihood class.
        cv_folds: A CVFolds tuple, e.g. from :func:`gen_loo_cv_folds`.
        fit_args: Arguments passed along to ``fit_gpytorch_mll``.
        observation_noise: Evaluate the posterior with observation noise.
        model_init_kwargs: Keyword arguments passed to the model constructor.

    Returns:
        A CVResults tuple with the batched model, the posterior at the
        held-out points (mean ``n x 1 x m``), the observed values and noise
        variances.
    """
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.models.multitask import MultiTaskGP

    if isinstance(model_cls, type) and issubclass(model_cls, MultiTaskGP):
        raise UnsupportedError(
            "Multi-task GPs are not currently supported by `batch_cross_validation`."
        )
    model_init_kws = dict(model_init_kwargs) if model_init_kwargs is not None else {}
    if cv_folds.train_Yvar is not None:
        model_init_kws["train_Yvar"] = cv_folds.train_Yvar
    model_cv = model_cls(train_X=cv_folds.train_X, train_Y=cv_folds.train_Y, **model_init_kws)
    mll_cv = mll_cls(model_cv.likelihood, model_cv)
    mll_cv = fit_gpytorch_mll(mll_cv, **(fit_args or {}))
    posterior = model_cv.posterior(cv_folds.test_X, observation_noise=observation_noise)
    return CVResults(
        model=model_cv,
        posterior=posterior,
        observed_Y=cv_folds.test_Y,
        observed_Yvar=cv_folds.test_Yvar,
    )


class _DiagonalPosterior(GPyTorchPosterior):
    """Independent Gaussian marginals with the normal posterior sampling API."""

    def __init__(self, mean: Any, variance: Any, kind: Any) -> None:
        mu, shape = to_flat(mean, "mean")
        var, vshape = to_flat(variance, "variance")
        if shape != vshape:
            raise ValueError("mean and variance must have the same shape")
        super().__init__(None, ([], (*shape[:-2], shape[-2], 0)), kind, False, shape[-1])
        self._mean_data = list(mu)
        self._variance_data = list(var)
        self._marginals = (list(mu), [max(v, 1e-10) for v in var])

    def _samples(self, z: Any) -> Any:
        return _rs.SingleTaskGP.diagonal_samples(
            self._mean_data, self._variance_data, z, self._q * self._m
        )

    def _get_joint(self) -> Any:
        if self._joint is None:
            q, m = self._q, self._m
            qm = q * m
            covariance = [0.0] * (math.prod(self._batch) * qm * qm)
            for b in range(math.prod(self._batch)):
                for t in range(m):
                    for i in range(q):
                        j = t * q + i
                        covariance[b * qm * qm + j * qm + j] = self._variance_data[
                            b * qm + i * m + t
                        ]
            self._joint = (
                (self._mean_data, (*self._batch, q, m)),
                (covariance, (*self._batch, qm, qm)),
            )
        return self._joint


def _reshape(value: Any, shape: tuple[int, ...]) -> Any:
    return from_flat(to_flat(value, "value")[0], shape, kind_of(value))


def efficient_loo_cv(
    model: Any, observation_noise: bool = True, untransform: bool = True
) -> CVResults:
    r"""Leave-one-out predictions without refitting, using one cached training
    factorization per output and batch element. Results have shape
    ``batch_shape x n x 1 x m``, as in BoTorch. Supports exact independent
    output GPs, including fixed noise and input and outcome transforms.
    """
    from botorch_rs.posteriors.transformed import TransformedPosterior

    if not isinstance(getattr(model, "_rs", None), _rs.SingleTaskGP):
        raise UnsupportedError("efficient_loo_cv requires an exact SingleTaskGP model.")
    model.eval()
    kind = model._kind
    mean, variance, targets, noise = model._rs.loo_predictions(bool(observation_noise))
    shape = tuple(mean[1])
    folds = (*shape[:-1], 1, shape[-1])
    mu = from_flat(mean[0], shape, kind)
    var = from_flat(variance[0], shape, kind)
    observed = from_flat(targets[0], shape, kind)
    observed_noise = from_flat(noise[0], shape, kind) if noise is not None else None
    posterior: Any = _DiagonalPosterior(_reshape(mu, folds), _reshape(var, folds), kind)
    transform = getattr(model, "outcome_transform", None) if untransform else None
    if transform is not None:
        observed, observed_noise = transform.untransform(observed, observed_noise)
        if transform._is_linear:
            mu, var = transform.untransform(mu, var)
            posterior = _DiagonalPosterior(_reshape(mu, folds), _reshape(var, folds), kind)
        else:
            # Lay folds out as q while untransforming, so transform batch
            # statistics broadcast against the model batch dimensions.
            transformed = transform.untransform_posterior(_DiagonalPosterior(mu, var, kind))

            def sample_transform(samples: Any) -> Any:
                data, sshape = to_flat(samples, "samples")
                squeezed = from_flat(data, (*sshape[:-2], sshape[-1]), kind)
                output, _ = transform.untransform(squeezed)
                return _reshape(output, tuple(sshape))

            posterior = TransformedPosterior(
                posterior,
                sample_transform=sample_transform,
                mean_transform=lambda m, v: _reshape(transformed.mean, folds),
                variance_transform=lambda m, v: _reshape(transformed.variance, folds),
            )
    return CVResults(
        model,
        posterior,
        _reshape(observed, folds),
        _reshape(observed_noise, folds) if observed_noise is not None else None,
    )


def ensemble_loo_cv(
    model: Any, observation_noise: bool = True, untransform: bool = True
) -> CVResults:
    """Leave-one-out predictions for exact GP ensembles, with the component
    axis at ``MCMC_DIM=-3`` and a ``GaussianMixturePosterior``.
    """
    if not getattr(model, "_is_ensemble", False):
        raise UnsupportedError("ensemble_loo_cv requires an ensemble model.")
    if not isinstance(getattr(model, "_rs", None), _rs.SingleTaskGP):
        raise UnsupportedError("ensemble_loo_cv requires an ensemble of exact SingleTaskGPs.")
    model.eval()
    kind = model._kind
    mean, variance, targets, noise = model._rs.loo_predictions(bool(observation_noise))
    native_shape = tuple(mean[1])
    if len(native_shape) < 3 or native_shape[-3] == 0:
        raise UnsupportedError("Expected an ensemble batch dimension for LOO predictions.")

    def check_shared(value: Any, name: str) -> None:
        data, shape = value
        count, n, m = shape[-3:]
        size = n * m
        for b in range(math.prod(shape[:-3])):
            start = b * count * size
            first = list(data[start : start + size])
            if any(
                list(data[start + k * size : start + (k + 1) * size]) != first
                for k in range(1, count)
            ):
                raise UnsupportedError(f"Ensemble members have different {name}.")

    check_shared(targets, "train_Y")
    if noise is not None:
        check_shared(noise, "observation noise")

    def as_folds(value: Any) -> Any:
        data, shape = to_flat(value, "value")
        count, n, m = shape[-3:]
        return from_flat(
            [
                data[((b * count + k) * n + i) * m + t]
                for b in range(math.prod(shape[:-3]))
                for i in range(n)
                for k in range(count)
                for t in range(m)
            ],
            (*shape[:-3], n, count, 1, m),
            kind,
        )

    def as_native(value: Any) -> Any:
        data, shape = to_flat(value, "value")
        n, count, _, m = shape[-4:]
        return from_flat(
            [
                data[((b * n + i) * count + k) * m + t]
                for b in range(math.prod(shape[:-4]))
                for k in range(count)
                for i in range(n)
                for t in range(m)
            ],
            (*shape[:-4], count, n, m),
            kind,
        )

    mu = from_flat(mean[0], native_shape, kind)
    var = from_flat(variance[0], native_shape, kind)
    observed = from_flat(targets[0], native_shape, kind)
    observed_noise = from_flat(noise[0], native_shape, kind) if noise is not None else None
    transform = getattr(model, "outcome_transform", None) if untransform else None
    if transform is not None:
        observed, observed_noise = transform.untransform(observed, observed_noise)
    if transform is not None and transform._is_linear:
        mu, var = transform.untransform(mu, var)
    posterior: Any = _DiagonalMixturePosterior(as_folds(mu), as_folds(var), kind)
    if transform is not None and not transform._is_linear:
        from botorch_rs.posteriors.transformed import TransformedPosterior

        transformed = transform.untransform_posterior(_DiagonalPosterior(mu, var, kind))
        posterior = TransformedPosterior(
            posterior,
            sample_transform=lambda s: as_folds(transform.untransform(as_native(s))[0]),
            mean_transform=lambda m, v: as_folds(transformed.mean),
            variance_transform=lambda m, v: as_folds(transformed.variance),
        )
    return CVResults(
        model,
        posterior,
        as_folds(observed),
        as_folds(observed_noise) if observed_noise is not None else None,
    )


class _DiagonalMixturePosterior(_DiagonalPosterior, GaussianMixturePosterior):
    pass


def loo_cv(model: Any, observation_noise: bool = True, untransform: bool = True) -> CVResults:
    """Compute efficient LOO predictions for an exact GP or GP ensemble."""
    function = ensemble_loo_cv if getattr(model, "_is_ensemble", False) else efficient_loo_cv
    return function(model, observation_noise=observation_noise, untransform=untransform)


__all__ = [
    "CVFolds",
    "CVResults",
    "batch_cross_validation",
    "gen_loo_cv_folds",
    "efficient_loo_cv",
    "ensemble_loo_cv",
    "loo_cv",
]
