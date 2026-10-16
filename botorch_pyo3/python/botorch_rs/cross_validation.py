"""Cross-validation utilities using batch evaluation mode
(``botorch.cross_validation``)."""

from __future__ import annotations

from typing import Any, NamedTuple

from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError


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


__all__ = ["CVFolds", "CVResults", "batch_cross_validation", "gen_loo_cv_folds"]
