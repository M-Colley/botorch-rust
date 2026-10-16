r"""Prior-guided acquisition functions (``botorch.acquisition.prior_guided``),
evaluated in Rust with the prior called through torch.

The base acquisition function is weighted by ``prior(x)^e`` (``log``: ``e
log prior(x)`` is added). For the sample-reducing Monte-Carlo acquisition
functions (``qExpectedImprovement``, ``qNoisyExpectedImprovement``,
``qLogExpectedImprovement``, ...) every point's per-sample utilities are
weighted before the reductions over the points and the samples, which
supports ``q > 1``; otherwise ``q = 1``.

References:
    C. Hvarfner, D. Stoll, A. Souza, M. Lindauer, F. Hutter and L. Nardi.
    piBO: Augmenting Acquisition Functions with User Beliefs for Bayesian
    Optimization. ICLR 2022.

Difference to BoTorch: for a sample-reducing acquisition function, ``log``
must match the scale of its utilities (``True`` for the log acquisition
functions such as ``qLogExpectedImprovement``, ``False`` otherwise); BoTorch
also allows multiplying log-utilities by a prior or adding a log prior to
utilities, which has no probabilistic meaning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.acquisition import AcquisitionFunction, pending_arg
from botorch_rs.exceptions.errors import BotorchRsError, UnsupportedError


class PriorGuidedAcquisitionFunction(AcquisitionFunction):
    r"""Class for weighting acquisition functions by a prior distribution.

    Supports MC and batch acquisition functions via the sample-reducing MC
    acquisition functions. ``prior_module`` is a callable of torch tensors
    (e.g. a ``torch.nn.Module``) mapping ``batch_shape x q x d`` inputs to
    ``batch_shape x q`` (log) probabilities; it is called with ``1 x n x d``
    tensors, with gradients from ``torch.autograd``.

    Example:
        >>> prior = MyPrior()  # log density of the inputs
        >>> qLogEI = qLogExpectedImprovement(model, best_f=0.0)
        >>> piBO = PriorGuidedAcquisitionFunction(qLogEI, prior, log=True)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        prior_module: Callable[[Any], Any],
        log: bool = False,
        prior_exponent: float = 1.0,
        X_pending: Any = None,
    ) -> None:
        if not isinstance(acq_function, AcquisitionFunction):
            raise UnsupportedError("`acq_function` must be a botorch_rs acquisition function.")
        super().__init__(acq_function.model)
        if acq_function.X_pending is not None:
            raise BotorchRsError(
                "X_pending is set on acq_function, but should be set on "
                "`PriorGuidedAcquisitionFunction`."
            )
        from botorch_rs._torch_bridge import require_torch

        require_torch()
        self.acq_func = acq_function
        self.prior_module = prior_module
        self._log = bool(log)
        self._prior_exponent = float(prior_exponent)
        d = int(acq_function._rs.dim)
        self._rs = _rs.prior_guided_acquisition_function(
            acq_function._rs,
            prior_module,
            bool(log),
            float(prior_exponent),
            pending_arg(X_pending, d),
        )
        if X_pending is not None:
            from botorch_rs._arrays import kind_of

            self._pending_kind = kind_of(X_pending)

    def _after_evaluation(self) -> None:
        self.acq_func._after_evaluation()


__all__ = ["PriorGuidedAcquisitionFunction"]
