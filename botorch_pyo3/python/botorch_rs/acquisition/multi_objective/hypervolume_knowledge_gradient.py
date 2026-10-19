"""The hypervolume knowledge gradient
(``botorch.acquisition.multi_objective.hypervolume_knowledge_gradient``;
Daulton, Balandat and Bakshy, *Hypervolume Knowledge Gradient: A Lookahead
Approach for Multi-Objective Bayesian Optimization with Partial Information*,
ICML 2023), computed in Rust with exact gradients through the fantasy models.
"""

from __future__ import annotations

import copy
import functools
import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch_rs.acquisition.cost_aware import CostAwareUtility, _mask_arg
from botorch_rs.acquisition.multi_objective.base import ref_point_arg
from botorch_rs.acquisition.multi_objective.objective import is_mo_objective, mo_objective_arg
from botorch_rs.exceptions.errors import (
    BotorchError,
    BotorchTensorDimensionError,
    UnsupportedError,
)
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.sampling.list_sampler import ListSampler
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler

#: BoTorch's default number of samples of the inner sampler.
DEFAULT_INNER_SAMPLES = 32


def _scalar(value: Any, name: str) -> float:
    data, _ = to_flat(value, name)
    if len(data) != 1:
        raise UnsupportedError(f"`{name}` must be a scalar.")
    return float(data[0])


def _project_arg(project: Any) -> Any:
    """The projection specification of ``project``: ``None`` for the
    identity, ``("target", ...)`` for a ``functools.partial`` of
    ``project_to_target_fidelity`` (of botorch_rs or BoTorch; evaluated in
    Rust), ``("callable", project)`` otherwise."""
    if project is None or project is _identity:
        return None
    if isinstance(project, functools.partial) and not project.args:
        func = project.func
        kw = project.keywords
        if getattr(func, "__name__", "") == "project_to_target_fidelity" and set(kw) == {
            "target_fidelities",
            "d",
        }:
            return ("target", [(int(k), float(v)) for k, v in kw["target_fidelities"].items()])
    return ("callable", project)


def _identity(X: Any) -> Any:
    return X


class qHypervolumeKnowledgeGradient(OneShotAcquisitionFunction):
    r"""Batch hypervolume knowledge gradient using one-shot optimization.

    The input ``X`` (``batch_shape x (q + num_fantasies * num_pareto) x d``)
    holds the ``q`` candidates followed by the fantasy designs: the
    ``num_pareto`` designs of fantasy ``i`` of t-batch ``b`` are
    ``X[b, q + i * num_pareto : q + (i + 1) * num_pareto]``. For every fantasy,
    the model (a ``ModelListGP``) is conditioned on observations fantasized at
    the candidates and pending points (on the observed outputs only with
    ``X_evaluation_mask`` / ``X_pending_evaluation_mask``), and the value
    function -- the hypervolume of the posterior mean of the fantasy designs
    (``use_posterior_mean=True``) or their expected hypervolume over
    posterior samples (``inner_sampler``) -- is averaged over the fantasies.
    ``current_value`` is subtracted (log: ``logdiffexp``), and a
    ``cost_aware_utility`` weights the improvements by the cost of the
    candidates.

    Differences to BoTorch (see the migration notes): every t-batch is
    evaluated on its own fantasy designs (BoTorch mixes the designs of
    different t-batches); with posterior samples the gradient with respect
    to the candidates is exact (BoTorch's misses the dependence of the fantasy
    models' covariances); ``GenericCostAwareUtility`` and
    ``InverseCostWeightedUtility(use_mean=False)`` with deterministic cost
    models work (both fail in BoTorch).
    """

    _name = "qHypervolumeKnowledgeGradient"

    def __init__(
        self,
        model: Any,
        ref_point: Any,
        num_fantasies: int = 8,
        num_pareto: int = 10,
        sampler: Any = None,
        objective: Any = None,
        inner_sampler: Any = None,
        X_evaluation_mask: Any = None,
        X_pending: Any = None,
        X_pending_evaluation_mask: Any = None,
        current_value: Any = None,
        use_posterior_mean: bool = True,
        cost_aware_utility: Any = None,
        log: bool = False,
    ) -> None:
        self._init(
            model,
            ref_point,
            num_fantasies,
            num_pareto,
            sampler,
            objective,
            inner_sampler,
            X_evaluation_mask,
            X_pending,
            X_pending_evaluation_mask,
            current_value,
            use_posterior_mean,
            cost_aware_utility,
            log,
            None,
            None,
        )

    def _init(
        self,
        model: Any,
        ref_point: Any,
        num_fantasies: int,
        num_pareto: int,
        sampler: Any,
        objective: Any,
        inner_sampler: Any,
        X_evaluation_mask: Any,
        X_pending: Any,
        X_pending_evaluation_mask: Any,
        current_value: Any,
        use_posterior_mean: bool,
        cost_aware_utility: Any,
        log: bool,
        target_fidelities: dict[int, float] | None,
        project: Any,
    ) -> None:
        if sampler is None:
            # Base samples should be fixed for the joint optimization over X
            # and the fantasy designs.
            n_out = getattr(model, "num_outputs", 1)
            sampler = ListSampler(
                *[SobolQMCNormalSampler(sample_shape=(num_fantasies,)) for _ in range(n_out)]
            )
        else:
            if not isinstance(sampler, ListSampler):
                raise UnsupportedError(
                    "`sampler` must be a botorch_rs ListSampler of SobolQMCNormalSamplers."
                )
            if tuple(sampler.samplers[0].sample_shape) != (num_fantasies,):
                raise ValueError(f"The sampler shape must match num_fantasies={num_fantasies}.")
        if not isinstance(model, ModelListGP):
            raise ValueError(f"{type(self).__name__} requires using a ModelList.")
        if any(m.num_outputs != 1 for m in model.models):
            raise UnsupportedError(
                f"{type(self).__name__} requires a ModelListGP of single-output models "
                "(BoTorch's ModelList.fantasize iterates over the outputs)."
            )
        super().__init__(model)
        self.num_outputs = model.num_outputs
        if len(sampler.samplers) != self.num_outputs:
            raise ValueError(
                f"The ListSampler must have one sampler per output ({self.num_outputs})."
            )
        if inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_INNER_SAMPLES,))
        elif not isinstance(inner_sampler, NormalMCSampler):
            raise UnsupportedError(
                "`inner_sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        if current_value is None and cost_aware_utility is not None:
            raise UnsupportedError("Cost-aware HVKG requires current_value to be specified.")
        if cost_aware_utility is not None and not isinstance(cost_aware_utility, CostAwareUtility):
            raise UnsupportedError(
                "`cost_aware_utility` must be a botorch_rs InverseCostWeightedUtility or "
                "GenericCostAwareUtility."
            )
        if objective is not None and not is_mo_objective(objective):
            raise UnsupportedError(
                "Only objectives of type MCMultiOutputObjective are supported for "
                "Multi-Objective MC acquisition functions."
            )
        mask = _mask_arg(X_evaluation_mask)
        if mask is not None and (len(mask[1]) != 2 or mask[1][-1] != self.num_outputs):
            raise BotorchTensorDimensionError(
                f"Expected X_evaluation_mask to be `q x m`, but got shape {tuple(mask[1])}."
            )
        ref = ref_point_arg(ref_point)
        self._kind = first_kind(ref_point, current_value)
        self.ref_point = from_flat(ref, (len(ref),), self._kind)
        self.sampler = sampler
        self.objective = objective
        self.inner_sampler = inner_sampler
        self.num_fantasies = int(num_fantasies)
        self.num_pareto = int(num_pareto)
        self.num_pseudo_points = self.num_fantasies * self.num_pareto
        self.current_value = current_value
        self.use_posterior_mean = bool(use_posterior_mean)
        self.cost_aware_utility = cost_aware_utility
        self._cost_sampler: Any = None
        self._log = bool(log)
        self._X_evaluation_mask = X_evaluation_mask
        pending = None if X_pending is None else to_flat(X_pending, "X_pending")
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self._rs, self._state = _rs.q_hypervolume_knowledge_gradient(
            model._acquisition_model(None),
            ref,
            self.num_fantasies,
            self.num_pareto,
            samplers=[s._rs for s in sampler.samplers],
            objective=None if objective is None else mo_objective_arg(objective),
            inner_sampler=inner_sampler._rs,
            x_evaluation_mask=mask,
            x_pending=pending,
            x_pending_evaluation_mask=_mask_arg(X_pending_evaluation_mask),
            current_value=None
            if current_value is None
            else _scalar(current_value, "current_value"),
            use_posterior_mean=self.use_posterior_mean,
            cost_aware_utility=None
            if cost_aware_utility is None
            else cost_aware_utility._rs_utility,
            log=self._log,
            target_fidelities=None
            if target_fidelities is None
            else [(int(k), float(v)) for k, v in target_fidelities.items()],
            project=_project_arg(project),
        )
        self._X_pending_evaluation_mask = X_pending_evaluation_mask

    # -- decoupled evaluations ----------------------------------------------

    @property
    def X_evaluation_mask(self) -> Any:
        """Which outputs are evaluated at which candidate (``q x m``), or ``None``."""
        return self._X_evaluation_mask

    @property
    def X_pending_evaluation_mask(self) -> Any:
        """Which outputs are evaluated at which pending point, or ``None``."""
        return self._X_pending_evaluation_mask

    def set_X_pending(self, X_pending: Any = None, X_pending_evaluation_mask: Any = None) -> None:
        """Informs the acquisition function about pending points (``n' x d``)
        and the outputs they are evaluated on (``n' x m``; ``None``: all;
        setting pending points without a mask clears the previous mask, as
        with meta-pytorch/botorch#3391)."""
        from botorch_rs.acquisition.acquisition import pending_arg

        flat = pending_arg(X_pending, self._rs.dim)
        self._state.set_x_pending(flat, _mask_arg(X_pending_evaluation_mask))
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self._X_pending_evaluation_mask = X_pending_evaluation_mask

    def construct_evaluation_mask(self, X: Any) -> Any:
        """The evaluation mask of ``X`` (``batch_shape x q x d``) and the
        pending points (``(q + n') x m``), or ``None``."""
        q = to_flat(X, "X")[1][-2]
        out = self._state.construct_evaluation_mask(q)
        if out is None:
            return None
        from botorch_rs._arrays import bool_array

        data, shape = out
        return bool_array(data, shape, first_kind(self._X_evaluation_mask, X))

    # -- one-shot ---------------------------------------------------------

    @property
    def cost_sampler(self) -> Any:
        """A copy of the fantasy sampler (BoTorch's ``cost_sampler``)."""
        if self._cost_sampler is None:
            self._cost_sampler = copy.deepcopy(self.sampler)
        return self._cost_sampler

    def get_augmented_q_batch_size(self, q: int) -> int:
        """``q + num_fantasies * num_pareto``, the number of points of the
        augmented candidate sets of the one-shot optimization."""
        return q + self.num_pseudo_points

    def extract_candidates(self, X_full: Any) -> Any:
        """The candidates (``... x q x d``) of augmented candidate sets."""
        if hasattr(X_full, "shape"):
            return X_full[..., : -self.num_pseudo_points, :]
        return _split_hvkg_fantasy_points(X_full, self.num_fantasies, self.num_pareto)[0]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_fantasies={self.num_fantasies}, "
            f"num_pareto={self.num_pareto}, use_posterior_mean={self.use_posterior_mean}, "
            f"log={self._log})"
        )


class qMultiFidelityHypervolumeKnowledgeGradient(qHypervolumeKnowledgeGradient):
    r"""Batch hypervolume knowledge gradient for multi-fidelity optimization:
    the value function is evaluated at the projected fantasy designs
    (``project``, e.g. ``project_to_target_fidelity``; by default the
    identity), typically with a ``cost_aware_utility``. ``target_fidelities``
    are used by the one-shot initial conditions. ``valfunc_cls`` and
    ``valfunc_argfac`` are stored but not used, as in BoTorch; trace
    observations (``expand``) are not supported, as in BoTorch.

    A ``project`` callable operates on torch tensors (``1 x num_pareto x d``)
    and is evaluated from Rust with gradients from ``torch.autograd``.
    """

    _name = "qMultiFidelityHypervolumeKnowledgeGradient"

    def __init__(
        self,
        model: Any,
        ref_point: Any,
        target_fidelities: dict[int, float],
        num_fantasies: int = 8,
        num_pareto: int = 10,
        sampler: Any = None,
        objective: Any = None,
        inner_sampler: Any = None,
        X_pending: Any = None,
        X_evaluation_mask: Any = None,
        X_pending_evaluation_mask: Any = None,
        current_value: Any = None,
        cost_aware_utility: Any = None,
        project: Any = _identity,
        valfunc_cls: Any = None,
        valfunc_argfac: Any = None,
        use_posterior_mean: bool = True,
        log: bool = False,
        **kwargs: Any,
    ) -> None:
        self._init(
            model,
            ref_point,
            num_fantasies,
            num_pareto,
            sampler,
            objective,
            inner_sampler,
            X_evaluation_mask,
            X_pending,
            X_pending_evaluation_mask,
            current_value,
            use_posterior_mean,
            cost_aware_utility,
            log,
            dict(target_fidelities),
            project,
        )
        self.project = project
        if kwargs.get("expand") is not None:
            raise NotImplementedError(
                "Trace observations are not currently supported "
                "by `qMultiFidelityHypervolumeKnowledgeGradient`."
            )
        self.expand = _identity
        self.valfunc_cls = valfunc_cls
        self.valfunc_argfac = valfunc_argfac
        self.target_fidelities = target_fidelities


class _HypervolumeValueFunction(AcquisitionFunction):
    """The value function of the hypervolume knowledge gradient (BoTorch's
    ``_get_hv_value_function``): qEHVI (qLogEHVI with ``log``) of the
    posterior mean or of posterior samples over an empty Pareto front, at
    projected designs if ``project`` is given."""

    def __init__(
        self,
        model: Any,
        ref_point: Any,
        objective: Any,
        sampler: Any,
        project: Any,
        use_posterior_mean: bool,
        log: bool,
    ) -> None:
        if not isinstance(model, ModelListGP) or any(m.num_outputs != 1 for m in model.models):
            raise UnsupportedError(
                "The hypervolume value function requires a botorch_rs ModelListGP of "
                "single-output models."
            )
        super().__init__(model)
        if sampler is not None and not isinstance(sampler, NormalMCSampler):
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        ref = ref_point_arg(ref_point)
        self.ref_point = from_flat(ref, (len(ref),), first_kind(ref_point))
        self.objective = objective
        self.sampler = sampler
        self.project = project
        self.posterior_transform = None
        self._log = bool(log)
        self._rs = _rs.hv_value_function(
            model._acquisition_model(None),
            ref,
            objective=None if objective is None else mo_objective_arg(objective),
            sampler=None if sampler is None else sampler._rs,
            use_posterior_mean=bool(use_posterior_mean),
            log=bool(log),
            project=_project_arg(project),
        )


def _get_hv_value_function(
    model: Any,
    ref_point: Any,
    objective: Any = None,
    sampler: Any = None,
    project: Any = None,
    valfunc_cls: Any = None,
    valfunc_argfac: Any = None,
    use_posterior_mean: bool = False,
    log: bool = False,
) -> AcquisitionFunction:
    r"""The value function of the hypervolume knowledge gradient: the
    hypervolume (qEHVI over an empty Pareto front; qLogEHVI with ``log``) of
    the posterior mean (``use_posterior_mean``) or of posterior samples
    (``sampler``, default: 128 Sobol samples) of a set of designs, at
    ``project(X)`` if ``project`` is given. ``valfunc_cls`` and
    ``valfunc_argfac`` are not used (as in BoTorch)."""
    return _HypervolumeValueFunction(
        model, ref_point, objective, sampler, project, use_posterior_mean, log
    )


def _split_hvkg_fantasy_points(X: Any, n_f: int, num_pareto: int) -> tuple[Any, Any]:
    r"""Splits an augmented candidate set ``X`` (``batch_shape x (q +
    n_f * num_pareto) x d``) into the candidates (``batch_shape x q x d``) and
    the fantasy designs (``n_f x batch_shape x num_pareto x d``), where
    ``X_fantasies[i, b]`` are the designs of fantasy ``i`` of t-batch ``b``
    (rows ``q + i * num_pareto ..`` of ``X[b]``). For a single t-batch this is
    BoTorch's reshape; for several, BoTorch's reshape mixes the designs of
    different t-batches (see the migration notes)."""
    data, shape = to_flat(X, "X")
    rows, d = shape[-2], shape[-1]
    if n_f * num_pareto > rows:
        raise ValueError(
            f"`n_f*num_pareto` ({n_f * num_pareto}) must be less than"
            f" the `q`-batch dimension of `X` ({rows})."
        )
    batch = tuple(shape[:-2])
    nb = math.prod(batch)
    q = rows - n_f * num_pareto
    if is_torch_tensor(X):
        # Views of X (differentiable).
        fantasies = X[..., q:, :].reshape(*batch, n_f, num_pareto, d)
        return X[..., :q, :], fantasies.movedim(-3, 0)
    kind = kind_of(X)
    actual: list[float] = []
    fant: list[float] = [0.0] * (n_f * nb * num_pareto * d)
    for b in range(nb):
        block = data[b * rows * d : (b + 1) * rows * d]
        actual.extend(block[: q * d])
        for i in range(n_f):
            start = (q + i * num_pareto) * d
            dst = (i * nb + b) * num_pareto * d
            fant[dst : dst + num_pareto * d] = block[start : start + num_pareto * d]
    return (
        from_flat(actual, (*batch, q, d), kind),
        from_flat(fant, (n_f, *batch, num_pareto, d), kind),
    )


def _check_log_utilities(cost_aware_utility: Any) -> None:
    if not getattr(cost_aware_utility, "_log", False):
        raise BotorchError(
            "Cost-aware HVKG has _log=True and requires cost_aware_utility to output log utilities."
        )


__all__ = [
    "qHypervolumeKnowledgeGradient",
    "qMultiFidelityHypervolumeKnowledgeGradient",
]
