r"""Multi-objective predictive entropy search
(``botorch.acquisition.multi_objective.predictive_entropy_search``), computed
in Rust.

PES approximates the posterior conditioned on the sampled Pareto sets being
optimal with expectation propagation (EP), run at construction over the
training inputs and the Pareto sets with BoTorch's damping schedule, and
updated once at the candidates. The value is the expected reduction of the
log-determinant of the predictive covariance. The gradients are exact for
the single EP update at the candidates (BoTorch: autograd through the same
update).

Deviation from BoTorch main: the EP factors use the model's raw training
inputs (BoTorch main passes the transformed ``train_inputs`` of a model in
eval mode to the model, which transforms them again when the model has an
input transform), and each candidate set's fallback and damping follow its
own factors (BoTorch decides them for the whole batch of candidates).

References:
    E. Garrido-Merchán and D. Hernández-Lobato. Predictive Entropy Search for
    Multi-objective Bayesian Optimization with Constraints. Neurocomputing
    2019; D. Hernández-Lobato, J. Hernández-Lobato, A. Shah and R. Adams.
    Predictive Entropy Search for Multi-objective Bayesian Optimization. ICML
    2016.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, pending_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model_list_gp_regression import ModelListGP


class qMultiObjectivePredictiveEntropySearch(AcquisitionFunction):
    r"""The acquisition function for predictive entropy search for
    multi-objective Bayesian optimization, with ``q >= 1`` (the joint
    predictive covariance of the batch).

    The value can be negative. EP is not always differentiable in the
    candidates (the damping changes with them); BoTorch recommends
    finite-difference optimization.

    Example:
        >>> ps, _ = sample_optimal_points(model, bounds, num_samples=8, num_points=10)
        >>> acq = qMultiObjectivePredictiveEntropySearch(model, ps)
        >>> value = acq(test_X)

    botorch_rs extensions: ``ep_iterations``, ``ep_damping`` and
    ``cached_factors`` describe the result of EP.
    """

    def __init__(
        self,
        model: Any,
        pareto_sets: Any,
        maximize: bool = True,
        X_pending: Any = None,
        max_ep_iterations: int = 250,
        ep_jitter: float = 1e-4,
        test_jitter: float = 1e-4,
        threshold: float = 1e-2,
    ) -> None:
        if getattr(model, "batch_shape", ()):
            raise NotImplementedError("Batch GP models (e.g. fantasized models) are not supported.")
        super().__init__(model)
        self.maximize = maximize
        models = model.models if isinstance(model, ModelListGP) else [model]
        d = int(models[0]._rs.dim)
        flat = to_flat(pareto_sets, "pareto_sets")
        if len(flat[1]) != 3 or flat[1][-1] != d:
            raise UnsupportedError(
                "The Pareto set should have a shape of "
                "`num_pareto_samples x num_pareto_points x input_dim`."
            )
        self.pareto_sets = pareto_sets
        self.num_pareto_samples = flat[1][0]
        self.max_ep_iterations = max_ep_iterations
        self.ep_jitter = ep_jitter
        self.test_jitter = test_jitter
        self.threshold = threshold
        pending = pending_arg(X_pending, d)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self._rs, self._ep = _rs.q_mo_predictive_entropy_search(
            self._model_handle(),
            flat,
            bool(maximize),
            pending,
            int(max_ep_iterations),
            float(ep_jitter),
            float(test_jitter),
            float(threshold),
        )

    @property
    def ep_iterations(self) -> int:
        """The number of EP iterations at construction (a botorch_rs
        extension)."""
        return int(self._ep.ep_iterations)

    @property
    def ep_damping(self) -> Any:
        """The final EP damping factors (``num_pareto_samples x M``; zero once
        converged; a botorch_rs extension)."""
        return from_pair(self._ep.ep_damping(), self.model._kind)

    @property
    def cached_factors(self) -> tuple[Any, Any]:
        """The cached EP factors of the Pareto points (a botorch_rs
        extension): natural means (``num_pareto_samples x M x P x P x 2``) and
        covariances (``num_pareto_samples x M x P x P x 2 x 2``), BoTorch's
        ``_omega_f_nat_mean[..., N:, :, :]`` and
        ``_omega_f_nat_cov[..., N:, :, :, :]``."""
        means, covs = self._ep.cached_factors()
        return from_pair(means, self.model._kind), from_pair(covs, self.model._kind)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(maximize={self.maximize})"


def log_cdf_robust(x: Any) -> Any:
    r"""The logarithm of the standard normal CDF, computed robustly:
    ``log Phi(x)`` for ``x < 5`` and ``-Phi(-x)`` otherwise, clamped to
    ``[min, -eps]``. Device tensors and tensors requiring gradients use the
    differentiable Torch algebra; array inputs retain the native CPU path."""
    if is_torch_tensor(x) and (x.requires_grad or x.is_cuda):
        from botorch_rs.acquisition._torch_backend_predictive_entropy import (
            log_cdf_robust as tensor_log_cdf,
        )

        return tensor_log_cdf(x)
    return from_pair(_rs.log_cdf_robust(to_flat(x, "x")), kind_of(x))


__all__ = ["log_cdf_robust", "qMultiObjectivePredictiveEntropySearch"]
