r"""
Acquisition functions for Bayesian active learning
(``botorch.acquisition.bayesian_active_learning``). This includes:
BALD [Houlsby2011bald]_ and its batch version [kirsch2019batchbald]_.

References

.. [kirsch2019batchbald]
    Andreas Kirsch, Joost van Amersfoort, Yarin Gal.
    BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
    Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2019.

"""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin, pending_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.utils.transforms import is_fully_bayesian

FULLY_BAYESIAN_ERROR_MSG = (
    "Fully Bayesian acquisition functions require a SaasFullyBayesianSingleTaskGP "
    "or of ModelList of SaasFullyBayesianSingleTaskGPs to run."
)

NEGATIVE_INFOGAIN_WARNING = (
    "Information gain is negative. This is likely due to a poor Monte Carlo "
    "estimation of the entropies, extremely high or extremely low correlation "
    "in the data."  # because both of those cases result in no information gain
)


def check_negative_info_gain(info_gain: Any) -> None:
    r"""Check if the (expected) information gain is negative, raise a warning if so."""
    data, _ = to_flat(info_gain, "info_gain")
    if any(float(v) < 0 for v in data):
        warnings.warn(NEGATIVE_INFOGAIN_WARNING, RuntimeWarning, stacklevel=2)


class FullyBayesianAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Any) -> None:
        """Base class for acquisition functions which require a Fully Bayesian
        model treatment.

        Args:
            model: A fully bayesian single-outcome model.
        """
        if not is_fully_bayesian(model):
            raise RuntimeError(FULLY_BAYESIAN_ERROR_MSG)
        self.model = model
        self._pending_kind = getattr(model, "_kind", kind_of(None))

    # The acquisition functions of fully Bayesian models are not averages
    # over the MCMC samples (see `AcquisitionFunction.__init_subclass__`).
    __init__._ensemble_aware = True  # type: ignore[attr-defined]


class qBayesianActiveLearningByDisagreement(FullyBayesianAcquisitionFunction, MCSamplerMixin):
    def __init__(
        self,
        model: Any,
        sampler: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
    ) -> None:
        """
        Batch implementation [kirsch2019batchbald]_ of BALD [Houlsby2011bald]_,
        which maximizes the mutual information between the next observation and the
        hyperparameters of the model. Computed by Monte Carlo integration.

        Args:
            model: A fully bayesian model (SaasFullyBayesianSingleTaskGP).
            sampler: The sampler used for drawing samples to approximate the entropy
                of the Gaussian Mixture posterior.
            posterior_transform: An optional PosteriorTransform. Without scalarization,
                multi-output models use their full joint Gaussian-mixture law.
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design points

        botorch_rs evaluates BALD in Rust with exact gradients on the MCMC
        samples of fully Bayesian GPs, including joint multi-output or scalarized
        fully Bayesian ``ModelList`` models. Without a ``sampler``, a
        ``SobolQMCNormalSampler`` of 512 samples (BoTorch's default) with a
        random seed is used.
        """
        from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

        super().__init__(model=model)
        members_of = getattr(model, "_ensemble_members", None)
        if members_of is None:
            raise UnsupportedError(
                "Native BALD requires access to the Gaussian MCMC component models."
            )
        members = members_of()
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(512,))
        if not isinstance(sampler, (IIDNormalSampler, SobolQMCNormalSampler)):
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        if getattr(sampler.base_samples, "requires_grad", False):
            raise UnsupportedError(
                "Gradients through native BALD base samples require backend='torch'."
            )
        self.sampler = sampler
        self.posterior_transform = posterior_transform
        handles = [m._acquisition_model(posterior_transform) for m in members]
        self._event_outputs = handles[0].num_outputs
        d = handles[0].dim
        self._native_sampler = sampler._rs
        self._rs = _rs.q_bayesian_active_learning_by_disagreement(
            handles, self._native_sampler, pending_arg(X_pending, d)
        )
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)

    def forward(self, X: Any) -> Any:
        r"""Evaluate qBayesianActiveLearningByDisagreement on the candidate set ``X``.
        A monte carlo-estimated information gain is computed over a Gaussian Mixture
        marginal posterior, and the Gaussian conditional posterior to obtain the
        qBayesianActiveLearningByDisagreement on the candidate set ``X``.

        Args:
            X: ``batch_shape x q x D``-dim Tensor of input points.

        Returns:
            A ``batch_shape``-dim Tensor of BALD values (BoTorch: averaged
            over the trailing ensemble dimension of size one).
        """
        self._prepare_iid_samples(X)
        out = super().forward(X)
        check_negative_info_gain(out.detach() if hasattr(out, "detach") else out)
        return out

    def _prepare_iid_samples(self, X: Any) -> None:
        """Keep the public IID normal cache and the native acquisition in sync."""
        import math

        from botorch_rs._arrays import from_flat
        from botorch_rs.sampling.normal import IIDNormalSampler

        if getattr(self.sampler.base_samples, "requires_grad", False):
            raise UnsupportedError(
                "Gradients through native BALD base samples require backend='torch'."
            )
        if not isinstance(self.sampler, IIDNormalSampler):
            return
        _, shape = to_flat(X, "X")
        if len(shape) < 2:
            return  # Candidate validation supplies the public shape error.
        pending = self._rs.x_pending()
        points = shape[-2] + (0 if pending is None else pending[1][-2])
        # Collapse query batches and the final MCMC batch, retaining every
        # sample axis. As in the public sampler, a changed event shape redraws.
        target = (
            *self.sampler.sample_shape,
            *(1 for _ in range(max(1, len(shape) - 2) + 1)),
            *((points,) if self._event_outputs == 1 else (points, self._event_outputs)),
        )
        current = self.sampler.base_samples
        if current is None or to_flat(current)[1] != target:
            values = self.sampler._draw(target, kind_of(X))
            self.sampler.base_samples = from_flat(values, target, kind_of(X))
            self.sampler._base_shape = target
        flat_values, _ = to_flat(self.sampler.base_samples, "base_samples")
        self._native_sampler.set_base_samples(
            (
                flat_values,
                (self.sampler.num_samples, math.prod(target[len(self.sampler.sample_shape) :])),
            )
        )

    def value_and_grad(self, X: Any) -> tuple[Any, Any]:
        self._prepare_iid_samples(X)
        return super().value_and_grad(X)

    __init__._ensemble_aware = True  # type: ignore[attr-defined]


__all__ = [
    "FULLY_BAYESIAN_ERROR_MSG",
    "NEGATIVE_INFOGAIN_WARNING",
    "FullyBayesianAcquisitionFunction",
    "check_negative_info_gain",
    "qBayesianActiveLearningByDisagreement",
]
