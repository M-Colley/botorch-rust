r"""Joint entropy search (``botorch.acquisition.joint_entropy_search``),
computed in Rust with exact gradients.

qJES is the mutual information between the observations at the candidates
and the optimal input-output pair, estimated with the model conditioned on
sampled optima (``get_optimal_samples``): the lower bound ``"LB"`` or the
Monte-Carlo estimate ``"MC"`` of the conditional entropy.

References:
    C. Hvarfner, F. Hutter and L. Nardi. Joint Entropy Search for Maximally-
    Informed Bayesian Optimization. NeurIPS 2022; B. Tu, A. Gandy, N. Kantas
    and B. Shafei. Joint Entropy Search for Multi-Objective Bayesian
    Optimization. NeurIPS 2022.

Differences to BoTorch main:

* The optimal inputs are input-transformed once when the model is conditioned
  on them (meta-pytorch/botorch#3399; BoTorch main transforms them twice).
* The ``"MC"`` estimate compares the posterior with the posterior-transformed
  optimal values (#3399).
* The ``"MC"`` estimate of ``q > 1`` points (meta-pytorch/botorch#3428): the
  conditional entropy of every point uses that point's marginal density, and
  the entropy of the observations includes the constant ``0.5 (1 + log 2 pi)``
  of every point; BoTorch pairs every point's truncation with the joint
  density of the batch and adds the constant once. Both agree for ``q = 1``.
* The noise of the noiseless conditioning also enters the conditioned
  models' posterior mean (meta-pytorch/botorch#3396).
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin, pending_arg
from botorch_rs.acquisition.objective import transform_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler

#: BoTorch's noise variance of the noiseless conditioning on the optima.
MIN_INFERRED_NOISE_LEVEL = 1e-4


def _condition_noise(model: Any) -> list[float] | None:
    """The noise variance of every output's noiseless conditioning in that
    output's outcome-transformed space: ``1e-4`` for a ``SingleTaskGP``
    (BoTorch passes the noise as transformed), ``1e-4 / std^2`` for the
    sub-models of a ``ModelListGP`` (BoTorch transforms it with their
    ``Standardize``)."""
    if not isinstance(model, ModelListGP):
        return None
    out: list[float] = []
    for sub in model.models:
        tf = getattr(sub, "outcome_transform", None)
        if tf is None:
            out.extend([MIN_INFERRED_NOISE_LEVEL] * int(sub.num_outputs))
            continue
        stdvs, _ = to_flat(tf.stdvs, "stdvs")
        out.extend(MIN_INFERRED_NOISE_LEVEL / (float(s) ** 2) for s in stdvs)
    return out


def _matrix(value: Any, name: str, cols: int | None = None) -> Any:
    """``value`` as an ``n x c`` flat array (BoTorch's ``num_optima x c``)."""
    data, shape = to_flat(value, name)
    if len(shape) == 1:
        shape = (shape[0], 1)
    if len(shape) != 2:
        raise UnsupportedError(f"`{name}` must be a `num_optima x {cols or 'c'}` tensor.")
    return data, tuple(shape)


#: BoTorch's error for fully Bayesian GPs.
FULLY_BAYESIAN_ERROR_MSG = (
    "JES is not yet available with Fully Bayesian GPs. Track the issue, "
    "which regards conditioning on a number of optima on a collection "
    "of models, in detail at https://github.com/meta-pytorch/botorch/issues/1680"
)


class qJointEntropySearch(AcquisitionFunction, MCSamplerMixin):
    r"""The acquisition function for the Joint Entropy Search, where the
    batches ``q > 1`` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the
    observation at a candidate point ``X`` and the optimal input-output pair.

    Example:
        >>> optimal_inputs, optimal_outputs = get_optimal_samples(
        ...     model, bounds, num_optima=12
        ... )
        >>> qJES = qJointEntropySearch(model, optimal_inputs, optimal_outputs)
        >>> qjes = qJES(test_X)

    botorch_rs extension (keyword-only): the ``sampler`` of the ``"MC"``
    estimate (default: ``num_samples`` Sobol samples).
    """

    #: Raised for fully Bayesian GPs (BoTorch's ``NotImplementedError``).
    _FULLY_BAYESIAN_ERROR = FULLY_BAYESIAN_ERROR_MSG

    def __init__(
        self,
        model: Any,
        optimal_inputs: Any,
        optimal_outputs: Any,
        condition_noiseless: bool = True,
        posterior_transform: Any = None,
        X_pending: Any = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
        *,
        sampler: SobolQMCNormalSampler | None = None,
    ) -> None:
        if getattr(model, "batch_shape", ()):
            raise NotImplementedError("Batch GP models (e.g. fantasized models) are not supported.")
        super().__init__(model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(num_samples,))
        elif not isinstance(sampler, NormalMCSampler):
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        self.sampler = sampler
        self.optimal_inputs = optimal_inputs
        self.optimal_outputs = optimal_outputs
        self.posterior_transform = posterior_transform
        self.condition_noiseless = bool(condition_noiseless)
        self.estimation_type = estimation_type
        rs_model = model._acquisition_model(None)
        xs = _matrix(optimal_inputs, "optimal_inputs")
        self.num_samples = xs[1][0]
        pending = pending_arg(X_pending, int(rs_model.dim))
        if X_pending is not None:
            from botorch_rs._arrays import kind_of

            self._pending_kind = kind_of(X_pending)
        self._rs = _rs.q_joint_entropy_search(
            rs_model,
            xs,
            _matrix(optimal_outputs, "optimal_outputs"),
            self.condition_noiseless,
            _condition_noise(model) if self.condition_noiseless else None,
            transform_arg(posterior_transform, model.num_outputs),
            pending,
            str(estimation_type),
            sampler._rs,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_samples={self.num_samples}, "
            f"estimation_type={self.estimation_type!r})"
        )


__all__ = ["MIN_INFERRED_NOISE_LEVEL", "qJointEntropySearch"]
