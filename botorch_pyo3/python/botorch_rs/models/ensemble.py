"""Ensemble models (``botorch.models.ensemble``): simple wrappers that allow
the usage of ensembles via the BoTorch Model and Posterior APIs.

An ``EnsembleModel``'s ``forward`` (a torch function) returns the values of
its ``s`` members. In botorch_rs Monte-Carlo acquisition functions, the
members are deterministic models evaluated through ``forward`` (with
gradients from ``torch.autograd``), and the acquisition value is the
average of the members' values weighted by the number of Monte-Carlo
samples that select each member (BoTorch's ``IndexSampler`` draws one
member per sample); see ``ENSEMBLE_ACQUISITION_FUNCTIONS``. Analytic
acquisition functions use the Gaussian moments of the ``EnsemblePosterior``
(its ``mean`` and ``variance``), as BoTorch's.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, ArrayKind, is_torch_tensor, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model import Model
from botorch_rs.posteriors.ensemble import EnsemblePosterior


class EnsembleModel(Model):
    """Abstract base class for ensemble models."""

    _num_outputs: int

    def __init__(self, weights: Any = None) -> None:
        """Initialize the ensemble model.

        Args:
            weights: Optional weights for the ensemble members.
                If None, the model weights will default to uniform in the
                corresponding mixture posterior.
        """
        # buffer ``weights`` is generally a name occupied by another module,
        # so we have to be more specific here
        self.ensemble_weights = weights

    def forward(self, X: Any) -> Any:  # pragma: no cover - abstract
        r"""Compute the (ensemble) model output at X.

        Args:
            X: A ``batch_shape x n x d``-dim input tensor ``X``.

        Returns:
            A ``batch_shape x s x n x m``-dimensional output tensor where
            ``s`` is the size of the ensemble.
        """
        raise NotImplementedError

    def _forward(self, X: Any) -> Any:
        return self.forward(X=X)

    def __call__(self, X: Any) -> Any:
        return self.forward(X)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        """Applies the model's ``input_transform`` (if any) in eval mode."""
        tf = (
            input_transform
            if input_transform is not None
            else getattr(self, "input_transform", None)
        )
        if tf is None:
            return X
        return tf(X)

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> EnsemblePosterior:
        r"""Compute the ensemble posterior at X.

        Args:
            X: A ``batch_shape x q x d``-dim input tensor ``X``.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior. If omitted, computes the posterior
                over all model outputs.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            An ``EnsemblePosterior`` object, representing ``batch_shape`` joint
            posteriors over ``n`` points and the outputs selected by ``output_indices``.
        """
        self.eval()
        X = self.transform_inputs(X)
        if is_torch_tensor(kwargs.get("observation_noise")):
            raise UnsupportedError("Ensemble models do not support observation noise.")
        values = self._forward(X)
        if hasattr(self, "outcome_transform"):
            values, _ = self.outcome_transform.untransform(values, X=X)
        if output_indices is not None:
            values = values[..., output_indices]
        posterior = EnsemblePosterior(values=values, weights=self.ensemble_weights)
        if posterior_transform is not None:
            return posterior_transform(posterior=posterior, X=X)
        return posterior

    # -- acquisition functions ------------------------------------------

    def _acquisition_dims(self) -> tuple[int, int]:
        """``(d, s)``: the input dimension (the attribute ``dim``, a botorch_rs
        extension) and the ensemble size of a model that acquisition
        functions support (no input or outcome transforms)."""
        from botorch_rs._torch_bridge import require_torch

        for name in ("input_transform", "outcome_transform"):
            if getattr(self, name, None) is not None:
                raise UnsupportedError(
                    f"botorch_rs acquisition functions do not support an EnsembleModel with an "
                    f"{name}."
                )
        dim = getattr(self, "dim", None)
        if dim is None:
            raise UnsupportedError(
                "botorch_rs acquisition functions need the input dimension of an EnsembleModel: "
                "set the attribute `dim` (a botorch_rs extension)."
            )
        require_torch()
        import torch

        size = int(self.forward(torch.zeros(1, 1, int(dim), dtype=torch.float64)).shape[-3])
        return int(dim), size

    def _ensemble_acquisition(self, sampler: Any) -> tuple[list[Any], list[float] | None, Any]:
        """The members (deterministic models evaluating ``forward``), their
        weights (the number of Monte-Carlo samples of ``sampler``, an
        ``IndexSampler``, that select each member for one t-batch) and the
        sampler of the members' acquisition functions."""
        from botorch_rs.models.deterministic import GenericDeterministicModel
        from botorch_rs.sampling.index_sampler import IndexSampler
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        dim, size = self._acquisition_dims()
        if sampler is None:
            sampler = IndexSampler(sample_shape=(512,))
        if not isinstance(sampler, IndexSampler):
            raise UnsupportedError("The sampler of an EnsembleModel must be an IndexSampler.")
        weights = sampler._member_counts(size, self.ensemble_weights)

        def member(k: int) -> Any:
            def f(X: Any) -> Any:
                return self.forward(X)[..., k, :, :]

            return GenericDeterministicModel(f, num_outputs=self.num_outputs, dim=int(dim))

        members = [member(k) for k in range(size)]
        return members, weights, SobolQMCNormalSampler(sample_shape=(1,))


class _EnsembleMoments(Model):
    """The Gaussian moments of the ``EnsemblePosterior`` of an
    ``EnsembleModel`` (its ``mean`` and unbiased weighted ``variance`` at
    every point, independent points and outputs), as BoTorch's analytic
    acquisition functions use them."""

    def __init__(self, model: EnsembleModel) -> None:
        self.ensemble = model
        self._kind: ArrayKind = LIST

    @property
    def num_outputs(self) -> int:
        return self.ensemble.num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The moments model handle (``forward`` called from Rust)."""
        if posterior_transform is not None:
            # BoTorch: ``scalarize_posterior`` raises NotImplementedError for
            # an ``EnsemblePosterior``.
            raise UnsupportedError(
                "Posterior transforms are not supported for ensemble models (BoTorch's "
                "scalarize_posterior only works with a posterior that has a `distribution`)."
            )
        dim, size = self.ensemble._acquisition_dims()
        m = self.ensemble.num_outputs
        forward = self.ensemble.forward

        def values(X: Any) -> Any:
            # ``batch x s x q x m`` -> ``batch x q x (s * m)`` (member ``k``,
            # output ``t`` at output ``k * m + t``).
            v = forward(X)
            return v.transpose(-3, -2).reshape(*v.shape[:-3], v.shape[-2], size * m)

        weights = self.ensemble.ensemble_weights
        w = None if weights is None else [float(v) for v in to_flat(weights, "weights")[0]]
        members = _rs.generic_deterministic_model(values, dim, size * m)
        return _rs.ensemble_moments_model(members, size, w)


__all__ = ["EnsembleModel"]
