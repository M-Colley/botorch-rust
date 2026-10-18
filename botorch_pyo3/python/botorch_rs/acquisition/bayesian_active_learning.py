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
from botorch_rs.acquisition.acquisition import AcquisitionFunction, pending_arg
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


class qBayesianActiveLearningByDisagreement(FullyBayesianAcquisitionFunction):
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
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design points

        botorch_rs evaluates BALD in Rust with exact gradients on the MCMC
        samples of a single-output fully Bayesian GP (``ModelList``\\s of
        them raise ``UnsupportedError``). Without a ``sampler``, a
        ``SobolQMCNormalSampler`` of 512 samples (BoTorch's default) with a
        random seed is used.
        """
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        super().__init__(model=model)
        members_of = getattr(model, "_ensemble_members", None)
        if members_of is None:
            raise UnsupportedError(
                "botorch_rs's qBayesianActiveLearningByDisagreement supports single-output fully "
                "Bayesian GPs (e.g. SaasFullyBayesianSingleTaskGP), not ModelLists of them."
            )
        members = members_of()
        if any(m.num_outputs != 1 for m in members) or getattr(model, "num_outputs", 1) != 1:
            raise UnsupportedError(
                "botorch_rs's qBayesianActiveLearningByDisagreement supports single-output fully "
                "Bayesian GPs."
            )
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(512,))
        if not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        self.sampler = sampler
        self.posterior_transform = posterior_transform
        handles = [m._acquisition_model(posterior_transform) for m in members]
        d = to_flat(model.train_inputs[0], "train_X")[1][-1]
        self._rs = _rs.q_bayesian_active_learning_by_disagreement(
            handles, sampler._rs, pending_arg(X_pending, d)
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
        out = super().forward(X)
        check_negative_info_gain(out.detach() if hasattr(out, "detach") else out)
        return out

    __init__._ensemble_aware = True  # type: ignore[attr-defined]


__all__ = [
    "FULLY_BAYESIAN_ERROR_MSG",
    "NEGATIVE_INFOGAIN_WARNING",
    "FullyBayesianAcquisitionFunction",
    "check_negative_info_gain",
    "qBayesianActiveLearningByDisagreement",
]
