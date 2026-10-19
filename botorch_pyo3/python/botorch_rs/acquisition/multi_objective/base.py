"""Base classes of the multi-objective acquisition functions
(``botorch.acquisition.multi_objective.base``)."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import Flat, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin, pending_arg
from botorch_rs.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    is_mo_objective,
    mo_objective_arg,
)
from botorch_rs.acquisition.objective import PosteriorTransform, constraints_arg, fat_arg
from botorch_rs.exceptions.errors import BotorchError, UnsupportedError
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.sampling.stochastic_samplers import mc_sampler_arg

#: BoTorch's default number of MC samples of the multi-objective MC
#: acquisition functions.
DEFAULT_MO_MC_SAMPLES = 128


def ref_point_arg(ref_point: Any) -> list[float]:
    """The reference point as a list of floats."""
    data, shape = to_flat(ref_point, "ref_point")
    if len(shape) != 1:
        raise UnsupportedError(f"The reference point must be one-dimensional, got shape {shape}.")
    return [float(v) for v in data]


def fat_bool(fat: Any) -> bool:
    """``fat`` as a bool (a per-constraint list is not supported: BoTorch's
    multi-objective acquisition functions also use its truth value for the
    smooth approximations)."""
    value, per_constraint = fat_arg(fat)
    if per_constraint is not None:
        raise UnsupportedError(
            "A per-constraint `fat` list is not supported by the multi-objective acquisition "
            "functions of botorch_rs."
        )
    return value


def partitioning_cells(partitioning: Any) -> Flat:
    """The hypercell bounds (``2 x num_cells x m``) of an unbatched box
    decomposition of botorch_rs or BoTorch."""
    try:
        bounds = partitioning.get_hypercell_bounds()
    except AttributeError as e:
        raise UnsupportedError(
            "`partitioning` must be a box decomposition with `get_hypercell_bounds()`, got "
            f"{type(partitioning).__name__}."
        ) from e
    if bounds is None:
        raise BotorchError("The partitioning has no hypercell bounds.")
    data, shape = to_flat(bounds, "hypercell bounds")
    if len(shape) != 3 or shape[0] != 2:
        raise UnsupportedError(
            "botorch_rs supports unbatched box decompositions (hypercell bounds of shape "
            f"`2 x num_cells x m`), got hypercell bounds of shape {tuple(shape)}."
        )
    return data, tuple(shape)


class MultiObjectiveAnalyticAcquisitionFunction(AcquisitionFunction):
    r"""Abstract base class of the multi-objective analytic acquisition
    functions."""

    def __init__(self, model: Any, posterior_transform: Any = None) -> None:
        super().__init__(model)
        if posterior_transform is not None and not isinstance(
            posterior_transform, PosteriorTransform
        ):
            raise UnsupportedError(
                "Only a posterior_transform of type PosteriorTransform is "
                "supported for Multi-Objective analytic acquisition functions."
            )
        if posterior_transform is not None:
            raise UnsupportedError(
                "Posterior transforms are not supported by botorch_rs's multi-objective "
                "analytic acquisition functions."
            )
        self.posterior_transform = posterior_transform

    def set_X_pending(self, X_pending: Any = None) -> None:
        raise UnsupportedError("Analytic acquisition functions do not account for X_pending yet.")


class MultiObjectiveMCAcquisitionFunction(AcquisitionFunction, MCSamplerMixin):
    r"""Abstract base class of the multi-objective MC acquisition functions.

    Posterior samples of all model outputs are mapped by a multi-output
    ``objective`` (default: the identity) to the objectives; outcome
    ``constraints`` (callables ``samples -> slack``, feasible iff ``<= 0``)
    weight the hypervolume contributions with their smoothed feasibility
    (temperature ``eta``). The default sampler draws 128 Sobol samples.
    """

    _default_sample_shape = (DEFAULT_MO_MC_SAMPLES,)

    def __init__(
        self,
        model: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        constraints: Any = None,
        eta: Any = 1e-3,
        X_pending: Any = None,
    ) -> None:
        super().__init__(model)
        self._rs_model = self._model_handle(None)
        self.sampler = mc_sampler_arg(sampler, self._rs_model, self._default_sample_shape)
        if objective is None:
            objective = IdentityMCMultiOutputObjective()
            self._mo_objective: tuple[Any, ...] | None = None
        elif not is_mo_objective(objective):
            raise UnsupportedError(
                "Only objectives of type MCMultiOutputObjective are supported for "
                "Multi-Objective MC acquisition functions."
            )
        else:
            self._mo_objective = mo_objective_arg(objective)
        self.objective = objective
        cons, etas = constraints_arg(constraints, eta)
        self.constraints = None if constraints is None else list(constraints)
        self._constraint_args: dict[str, Any] = {"constraints": cons, "eta": etas}
        if constraints:
            self.eta = eta
        self._pending = pending_arg(X_pending, self._rs_model.dim)
        if X_pending is not None:
            from botorch_rs._arrays import kind_of

            self._pending_kind = kind_of(X_pending)


__all__ = [
    "MultiObjectiveAnalyticAcquisitionFunction",
    "MultiObjectiveMCAcquisitionFunction",
]
