r"""Active learning acquisition functions (``botorch.acquisition.active_learning``),
computed in Rust with exact gradients.

``qNegIntegratedPosteriorVariance`` is the negative posterior variance
(without observation noise) of the model after observing the candidates (and
pending points), averaged over a set of Monte-Carlo integration points. For
exact GPs the variance after conditioning does not depend on the observed
values, so no fantasies are needed: with the noisy covariance ``A`` of the
candidates, ``var_new(z) = var(z) - Sigma(z, X) A^-1 Sigma(X, z)``.

References:
    D. Seo, M. Wallat, T. Graepel and K. Obermayer. Gaussian Process
    Regression: Active Data Selection and Test Point Rejection. IJCNN 2000.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, pending_arg
from botorch_rs.acquisition.objective import transform_arg
from botorch_rs.exceptions.errors import UnsupportedError


class qNegIntegratedPosteriorVariance(AcquisitionFunction):
    r"""Batch Integrated Negative Posterior Variance for Active Learning.

    This acquisition function quantifies the (negative) integrated posterior
    variance (excluding observation noise, computed using MC integration) of
    the model. In that, it is a proxy for global model uncertainty, and thus
    purely focused on "exploration", rather the "exploitation" of many of
    the classic Bayesian Optimization acquisition functions.

    ``sampler`` is accepted for compatibility: the value of an exact GP does
    not depend on the fantasized observations.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> mc_points = draw_sobol_samples(bounds=bounds, n=100, q=1).squeeze(-2)
        >>> qNIPV = qNegIntegratedPosteriorVariance(model, mc_points)
        >>> qnipv = qNIPV(test_X)
    """

    def __init__(
        self,
        model: Any,
        mc_points: Any,
        sampler: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
    ) -> None:
        super().__init__(model)
        rs_model = model._acquisition_model(None)
        flat = to_flat(mc_points, "mc_points")
        if len(flat[1]) != 2:
            raise UnsupportedError("`mc_points` must be an `N x d` tensor.")
        if posterior_transform is None and model.num_outputs != 1:
            raise UnsupportedError(
                "Must specify a posterior transform when using a multi-output model."
            )
        self.mc_points = mc_points
        self.sampler = sampler
        self.posterior_transform = posterior_transform
        pending = pending_arg(X_pending, int(rs_model.dim))
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self._rs = _rs.q_neg_integrated_posterior_variance(
            rs_model,
            flat,
            transform_arg(posterior_transform, model.num_outputs),
            pending,
        )


__all__ = ["qNegIntegratedPosteriorVariance"]
