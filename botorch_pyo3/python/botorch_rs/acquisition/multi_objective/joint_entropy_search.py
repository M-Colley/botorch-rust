r"""Lower bound multi-objective entropy search
(``botorch.acquisition.multi_objective.joint_entropy_search``):
``qLowerBoundMultiObjectiveJointEntropySearch`` and the entropy estimators
``_compute_entropy_noiseless``, ``_compute_entropy_upper_bound`` and
``_compute_entropy_monte_carlo``, computed in Rust with exact gradients.

JES estimates the mutual information between the observations at the
candidates and the sampled Pareto sets and fronts: the entropy of the
Gaussian posterior minus the expected entropy of the posterior conditioned
on each sampled Pareto set (observed at its front) and truncated to the
region its front dominates. The conditioning is an exact rank update of the
posterior on the Pareto set with the model's observation noise (BoTorch's
``condition_on_observations``), on the raw Pareto sets
(meta-pytorch/botorch#3399; BoTorch main transforms them twice).

References:
    B. Tu, A. Gandy, N. Kantas and B. Shafei. Joint Entropy Search for
    Multi-Objective Bayesian Optimization. NeurIPS 2022.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin, pending_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler

#: The entropy estimates.
ESTIMATION_TYPES = ("0", "LB", "LB2", "MC")


def _check_ndim(x: Any, ndim: int) -> bool:
    return x is None or len(to_flat(x)[1]) == ndim


class LowerBoundMultiObjectiveEntropySearch(AcquisitionFunction, MCSamplerMixin):
    r"""Base class of the lower bound multi-objective entropy search
    acquisition functions (MES and JES).

    ``estimation_type`` selects the entropy estimate: ``"0"`` (noiseless),
    ``"LB"`` (the moments of the truncated posterior, the default), ``"LB2"``
    (their diagonal) or ``"MC"`` (Monte Carlo with ``num_samples`` samples of
    ``sampler``, a botorch_rs ``SobolQMCNormalSampler``; a keyword-only
    botorch_rs extension to set the base samples).
    """

    def __init__(
        self,
        model: Any,
        pareto_sets: Any,
        pareto_fronts: Any,
        hypercell_bounds: Any,
        X_pending: Any = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
        *,
        sampler: SobolQMCNormalSampler | None = None,
    ) -> None:
        batch = getattr(model, "batch_shape", ())
        if batch:
            raise NotImplementedError("Batch GP models (e.g. fantasized models) are not supported.")
        super().__init__(model)
        self.initial_model = model
        if not (_check_ndim(pareto_sets, 3) and _check_ndim(pareto_fronts, 3)):
            raise UnsupportedError(
                "The Pareto set and front should have a shape of "
                "`num_pareto_samples x num_pareto_points x input_dim` and "
                "`num_pareto_samples x num_pareto_points x num_objectives`, "
                "respectively"
            )
        self.pareto_sets = pareto_sets
        self.pareto_fronts = pareto_fronts
        cells = to_flat(hypercell_bounds, "hypercell_bounds")
        if len(cells[1]) != 4:
            raise UnsupportedError(
                "The hypercell_bounds should have a shape of "
                "`num_pareto_samples x 2 x num_boxes x num_objectives`."
            )
        self.hypercell_bounds = hypercell_bounds
        self.num_pareto_samples = cells[1][0]
        if estimation_type not in ESTIMATION_TYPES:
            raise NotImplementedError(
                "Currently the only supported estimation type are: "
                + ", ".join(f'"{h}"' for h in ESTIMATION_TYPES)
                + "."
            )
        self.estimation_type = estimation_type
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=[num_samples])
        elif not isinstance(sampler, NormalMCSampler):
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        self.sampler = sampler
        self._cells = cells
        self._pending = pending_arg(X_pending, self._model_dim())
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)

    def _model_dim(self) -> int:
        models = self.model.models if isinstance(self.model, ModelListGP) else [self.model]
        return int(models[0]._rs.dim)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(estimation_type={self.estimation_type!r})"


class qLowerBoundMultiObjectiveJointEntropySearch(LowerBoundMultiObjectiveEntropySearch):
    r"""The acquisition function for the multi-objective joint entropy search,
    where the batches ``q > 1`` are supported through the lower bound
    formulation.

    This acquisition function computes the mutual information between the
    observation at a candidate point ``X`` and the Pareto optimal input-output
    pairs. The estimated value can be negative, and the lower bound is not
    monotone in ``q``.

    Example:
        >>> ps, pf = sample_optimal_points(model, bounds, num_samples=8, num_points=10)
        >>> cells = compute_sample_box_decomposition(pf)
        >>> acq = qLowerBoundMultiObjectiveJointEntropySearch(model, ps, pf, cells)
        >>> value = acq(test_X)

    Fixed-noise models are not supported (as in BoTorch, whose
    ``condition_on_observations`` requires the noise of the Pareto points).
    """

    def __init__(
        self,
        model: Any,
        pareto_sets: Any,
        pareto_fronts: Any,
        hypercell_bounds: Any,
        X_pending: Any = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
        *,
        sampler: SobolQMCNormalSampler | None = None,
    ) -> None:
        super().__init__(
            model,
            pareto_sets,
            pareto_fronts,
            hypercell_bounds,
            X_pending=X_pending,
            estimation_type=estimation_type,
            num_samples=num_samples,
            sampler=sampler,
        )
        self._rs = _rs.q_lower_bound_mo_joint_entropy_search(
            self._model_handle(),
            to_flat(pareto_sets, "pareto_sets"),
            to_flat(pareto_fronts, "pareto_fronts"),
            self._cells,
            self._pending,
            estimation_type,
            self.sampler._rs,
        )


def _entropy(
    estimation_type: str,
    hypercell_bounds: Any,
    mean: Any,
    variance: Any,
    observation_noise: Any,
    samples: Any = None,
    samples_log_prob: Any = None,
) -> Any:
    inputs = (hypercell_bounds, mean, variance, observation_noise, samples, samples_log_prob)
    if any(is_torch_tensor(value) and (value.requires_grad or value.is_cuda) for value in inputs):
        import torch

        from botorch_rs.acquisition import _torch_backend_mo_entropy as tensor_entropy

        like = (
            mean
            if is_torch_tensor(mean)
            else next(value for value in inputs if is_torch_tensor(value))
        )
        cells, mu, var, noise = (
            torch.as_tensor(value, dtype=like.dtype, device=like.device) for value in inputs[:4]
        )
        arguments = dict(hypercell_bounds=cells, mean=mu, variance=var, observation_noise=noise)
        if estimation_type == "0":
            return tensor_entropy._compute_entropy_noiseless(**arguments)
        if estimation_type in ("LB", "LB2"):
            return tensor_entropy._compute_entropy_upper_bound(
                **arguments, only_diagonal=estimation_type == "LB2"
            )
        if samples is None or samples_log_prob is None:
            raise ValueError("Monte Carlo entropy requires samples and their log probabilities.")
        return tensor_entropy._compute_entropy_monte_carlo(
            **arguments,
            samples=torch.as_tensor(samples, dtype=like.dtype, device=like.device),
            samples_log_prob=torch.as_tensor(
                samples_log_prob, dtype=like.dtype, device=like.device
            ),
        )
    values = _rs.compute_entropy(
        estimation_type,
        to_flat(hypercell_bounds, "hypercell_bounds"),
        to_flat(mean, "mean"),
        to_flat(variance, "variance"),
        to_flat(observation_noise, "observation_noise"),
        None if samples is None else to_flat(samples, "samples"),
        None if samples_log_prob is None else to_flat(samples_log_prob, "samples_log_prob"),
    )
    return from_pair(values, kind_of(mean))


def _compute_entropy_noiseless(
    hypercell_bounds: Any, mean: Any, variance: Any, observation_noise: Any
) -> Any:
    r"""The entropy estimate assuming noiseless observations (JES-0 and
    MES-0): ``hypercell_bounds`` is ``num_pareto_samples x 2 x J x M``, the
    statistics ``batch_shape x num_pareto_samples x q x 1 x M``; returns
    ``batch_shape x q`` values."""
    return _entropy("0", hypercell_bounds, mean, variance, observation_noise)


def _compute_entropy_upper_bound(
    hypercell_bounds: Any,
    mean: Any,
    variance: Any,
    observation_noise: Any,
    only_diagonal: bool = False,
) -> Any:
    r"""The entropy upper bound (JES-LB and MES-LB; JES-LB2 and MES-LB2 with
    ``only_diagonal``); shapes as in ``_compute_entropy_noiseless``."""
    return _entropy(
        "LB2" if only_diagonal else "LB", hypercell_bounds, mean, variance, observation_noise
    )


def _compute_entropy_monte_carlo(
    hypercell_bounds: Any,
    mean: Any,
    variance: Any,
    observation_noise: Any,
    samples: Any,
    samples_log_prob: Any,
) -> Any:
    r"""The Monte Carlo entropy estimate (JES-MC and MES-MC) with noisy
    ``samples`` (``num_mc_samples x batch_shape x num_pareto_samples x q x 1
    x M``) of the conditioned posterior and their log-densities
    (``num_mc_samples x batch_shape x num_pareto_samples x q``); other shapes
    as in ``_compute_entropy_noiseless``."""
    return _entropy(
        "MC",
        hypercell_bounds,
        mean,
        variance,
        observation_noise,
        samples,
        samples_log_prob,
    )


__all__ = [
    "LowerBoundMultiObjectiveEntropySearch",
    "qLowerBoundMultiObjectiveJointEntropySearch",
]
