"""Batch Knowledge Gradient (KG) via one-shot optimization
(``botorch.acquisition.knowledge_gradient``; Balandat et al., *BoTorch: A
Framework for Efficient Monte-Carlo Bayesian Optimization*, NeurIPS 2020),
computed in Rust with exact gradients through the fantasy models.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
    pending_arg,
)
from botorch_rs.acquisition.cost_aware import CostAwareUtility
from botorch_rs.acquisition.monte_carlo import MCAcquisitionFunction
from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _identity,
    _project_arg,
)
from botorch_rs.acquisition.objective import (
    MCAcquisitionObjective,
    _botorch_class_names,
    objective_arg,
    transform_arg,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler

#: BoTorch's default number of samples of the inner sampler with an objective.
DEFAULT_INNER_SAMPLES = 128

#: The value functions (``valfunc_cls``) evaluated natively, by class name.
NATIVE_VALUE_FUNCTIONS = (
    "PosteriorMean",
    "PosteriorStandardDeviation",
    "UpperConfidenceBound",
    "ExpectedImprovement",
    "LogExpectedImprovement",
    "ProbabilityOfImprovement",
    "LogProbabilityOfImprovement",
    "ScalarizedPosteriorMean",
    "qSimpleRegret",
    "qExpectedImprovement",
    "qProbabilityOfImprovement",
    "qUpperConfidenceBound",
    "qLowerConfidenceBound",
    "qPosteriorStandardDeviation",
)


def _scalar(value: Any) -> Any:
    """A float (or a list of floats for vectors) from a tensor, array or
    number."""
    if isinstance(value, bool):
        return value
    data, shape = to_flat(value, "argument")
    if len(data) == 1 and len(shape) <= 1:
        return float(data[0])
    return [float(v) for v in data]


def _value_function_arg(
    model: Any, valfunc_cls: Any, valfunc_argfac: Callable[..., dict[str, Any]] | None
) -> tuple[str, dict[str, Any]] | None:
    """``(name, kwargs)`` of a native value function. The argument factory
    is called once, with the current model (BoTorch calls it with every
    fantasy model)."""
    if valfunc_cls is None:
        return None
    name = getattr(valfunc_cls, "__name__", str(valfunc_cls))
    if name not in NATIVE_VALUE_FUNCTIONS:
        raise UnsupportedError(
            f"`valfunc_cls` {name} is not supported by botorch_rs (supported: "
            f"{', '.join(NATIVE_VALUE_FUNCTIONS)})."
        )
    kwargs = dict(valfunc_argfac(model=model)) if valfunc_argfac is not None else {}
    allowed = {"best_f", "beta", "tau", "maximize", "weights"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise UnsupportedError(f"Unsupported arguments of the value function {name}: {unknown}.")
    return name, {k: _scalar(v) for k, v in kwargs.items()}


def _is_mc_objective(objective: Any) -> bool:
    """Whether ``objective`` is an MC objective of botorch_rs or BoTorch."""
    if objective is None:
        return False
    if isinstance(objective, MCAcquisitionObjective):
        return True
    return "MCAcquisitionObjective" in _botorch_class_names(objective)


def _expand_arg(expand: Any) -> Any:
    """The expansion specification of ``expand``: ``None`` for the identity,
    ``("trace", fidelity_dims, num_trace_obs)`` for a ``functools.partial``
    of ``expand_trace_observations`` (evaluated in Rust), ``("callable",
    expand)`` otherwise."""
    if expand is None or expand is _identity:
        return None
    if isinstance(expand, functools.partial) and not expand.args:
        func = expand.func
        kw = expand.keywords
        if getattr(func, "__name__", "") == "expand_trace_observations" and set(kw) <= {
            "fidelity_dims",
            "num_trace_obs",
        }:
            dims = kw.get("fidelity_dims")
            return (
                "trace",
                None if dims is None else [int(k) for k in dims],
                int(kw.get("num_trace_obs", 0)),
            )
    return ("callable", expand)


class qKnowledgeGradient(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""Batch Knowledge Gradient using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the
    inner expectation.

    In addition to the design variables, the input ``X`` also includes
    variables for the optimal designs for each of the fantasy models: ``X``
    (``batch_shape x (q + num_fantasies) x d``) holds the ``q`` candidates
    followed by ``num_fantasies`` fantasy points, the last ones being the
    current solutions of the inner optimization problems. For a fixed number
    of fantasies, all parts of ``X`` can be optimized in a "one-shot"
    fashion.

    The model is fantasized at the candidates and pending points with the
    base samples of ``sampler`` (a ``SingleTaskGP``, single- or multi-output,
    or a ``ModelListGP``), and the value function -- the posterior mean
    (through ``posterior_transform``), or with an MC ``objective`` the simple
    regret of ``inner_sampler``'s posterior samples -- is evaluated at every
    fantasy point. With a ``current_value`` the result is the actual KG
    value.

    Gradients are exact. With an MC objective, BoTorch's gradient with
    respect to the candidates misses the dependence of the fantasy
    posteriors' covariances on them (meta-pytorch/botorch#3427); the values
    agree.
    """

    def __init__(
        self,
        model: Any,
        num_fantasies: int | None = 64,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: Any = None,
        inner_sampler: SobolQMCNormalSampler | None = None,
        X_pending: Any = None,
        current_value: Any = None,
    ) -> None:
        self._init(
            model,
            num_fantasies,
            sampler,
            objective,
            posterior_transform,
            inner_sampler,
            X_pending,
            current_value,
        )

    def _init(
        self,
        model: Any,
        num_fantasies: int | None,
        sampler: Any,
        objective: Any,
        posterior_transform: Any,
        inner_sampler: Any,
        X_pending: Any,
        current_value: Any,
        multi_fidelity: dict[str, Any] | None = None,
    ) -> None:
        if sampler is None:
            if num_fantasies is None:
                raise ValueError("Must specify `num_fantasies` if no `sampler` is provided.")
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(sample_shape=(num_fantasies,))
        elif not isinstance(sampler, NormalMCSampler):
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        elif num_fantasies is not None:
            if tuple(sampler.sample_shape) != (num_fantasies,):
                raise ValueError(f"The sampler shape must match num_fantasies={num_fantasies}.")
        else:
            num_fantasies = sampler.sample_shape[0]
        AcquisitionFunction.__init__(self, model)
        # if not explicitly specified, we use the posterior mean for linear objs
        if _is_mc_objective(objective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_INNER_SAMPLES,))
        elif objective is not None and not _is_mc_objective(objective):
            raise UnsupportedError(
                "Objectives that are not an `MCAcquisitionObjective` are not supported."
            )
        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            elif not getattr(posterior_transform, "scalarize", False):
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        if inner_sampler is not None and not isinstance(inner_sampler, NormalMCSampler):
            raise UnsupportedError(
                "`inner_sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        self.sampler = sampler
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.inner_sampler = inner_sampler
        self.num_fantasies: int = int(num_fantasies)
        self.current_value = current_value
        self._constraints = None
        pending = pending_arg(X_pending, model._acquisition_model(None).dim)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        mf = multi_fidelity or {}
        self._rs, self._state = _rs.q_knowledge_gradient(
            model._acquisition_model(None),
            self.num_fantasies,
            sampler=sampler._rs,
            objective=objective_arg(objective),
            transform=transform_arg(posterior_transform, model.num_outputs),
            inner_sampler=None if inner_sampler is None else inner_sampler._rs,
            x_pending=pending,
            current_value=None if current_value is None else _scalar(current_value),
            multi_fidelity=multi_fidelity is not None,
            cost_aware_utility=mf.get("cost_aware_utility"),
            project=mf.get("project"),
            expand=mf.get("expand"),
            value_function=mf.get("value_function"),
        )

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization: ``q +
        num_fantasies``."""
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Any) -> Any:
        r"""The candidates (``... x q x d``) of augmented candidate sets
        (``... x (q + num_fantasies) x d``)."""
        if hasattr(X_full, "shape"):
            return X_full[..., : -self.num_fantasies, :]
        data, shape = to_flat(X_full, "X_full")
        rows, d = shape[-2], shape[-1]
        q = rows - self.num_fantasies
        out: list[float] = []
        for b in range(len(data) // (rows * d)):
            out.extend(data[b * rows * d : b * rows * d + q * d])
        return from_flat(out, (*shape[:-2], q, d), kind_of(X_full))

    def evaluate(self, X: Any, bounds: Any, **kwargs: Any) -> Any:
        r"""Evaluate qKnowledgeGradient on the candidate set ``X`` (``batch_shape
        x q x d``, without fantasy points) by solving the inner optimization
        problems: the value function of every fantasy model is maximized over
        ``bounds`` from ``num_restarts`` (20) initial conditions of
        ``gen_value_function_initial_conditions`` (``raw_samples`` (1024) raw
        points, ``options``: ``frac_random`` (0.6), ``num_inner_restarts``,
        ``raw_inner_samples``, ``eta`` (2), ``seed``) by one joint L-BFGS-B
        problem (``scipy_options``: ``maxiter``, ``maxfun``, ``ftol``,
        ``gtol``).

        The random choices use botorch_rs's generator seeded by ``seed``.
        A cost-aware utility is applied to the candidates without pending
        points and expansion, as in ``forward`` (BoTorch's ``evaluate``
        passes the expanded candidates with the pending points).
        """
        unknown = sorted(set(kwargs) - {"num_restarts", "raw_samples", "options", "scipy_options"})
        if unknown:
            raise UnsupportedError(f"Unsupported keyword arguments: {unknown}.")
        options = dict(kwargs.get("options") or {})
        scipy_options = dict(kwargs.get("scipy_options") or {})
        options.update(scipy_options)
        known = {
            "frac_random",
            "num_inner_restarts",
            "raw_inner_samples",
            "eta",
            "seed",
            "maxiter",
            "maxfun",
            "ftol",
            "gtol",
        }
        extra = sorted(set(options) - known - {"disp", "iprint", "batch_limit", "init_batch_limit"})
        if extra:
            raise UnsupportedError(f"Unsupported options of `evaluate`: {extra}.")
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) == 2:
            flat = (flat[0], (1, *flat[1]))
        values, batch = self._state.evaluate(
            flat,
            to_flat(bounds, "bounds"),
            int(kwargs.get("num_restarts", 20)),
            int(kwargs.get("raw_samples", 1024)),
            frac_random=None if "frac_random" not in options else float(options["frac_random"]),
            num_inner_restarts=None
            if "num_inner_restarts" not in options
            else int(options["num_inner_restarts"]),
            raw_inner_samples=None
            if "raw_inner_samples" not in options
            else int(options["raw_inner_samples"]),
            eta=None if "eta" not in options else float(options["eta"]),
            seed=None if options.get("seed") is None else int(options["seed"]),
            maxiter=None if "maxiter" not in options else int(options["maxiter"]),
            maxfun=None if "maxfun" not in options else int(options["maxfun"]),
            ftol=None if "ftol" not in options else float(options["ftol"]),
            gtol=None if "gtol" not in options else float(options["gtol"]),
        )
        return from_flat(values, tuple(batch), kind)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_fantasies={self.num_fantasies})"


class qMultiFidelityKnowledgeGradient(qKnowledgeGradient):
    r"""Batch Knowledge Gradient for multi-fidelity optimization.

    A version of ``qKnowledgeGradient`` that supports multi-fidelity
    optimization via a ``CostAwareUtility`` and the ``project`` and
    ``expand`` operators. If none of these are set, this acquisition function
    reduces to ``qKnowledgeGradient``. Through ``valfunc_cls`` and
    ``valfunc_argfac``, this can be changed into a custom multi-fidelity
    acquisition function (it is only KG if the terminal value is computed
    using a posterior mean).

    ``project`` and ``expand`` are evaluated in Rust for ``functools.partial``
    objects of ``project_to_target_fidelity`` / ``expand_trace_observations``
    (of botorch_rs or BoTorch), and otherwise as callables on torch tensors
    (with gradients from ``torch.autograd``). ``valfunc_cls`` may be one of
    the classes in ``NATIVE_VALUE_FUNCTIONS`` (of botorch_rs or BoTorch);
    ``valfunc_argfac`` is called once with the current model (BoTorch calls it
    with every fantasy model), so its arguments must not depend on the
    fantasized observations.
    """

    def __init__(
        self,
        model: Any,
        num_fantasies: int | None = 64,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: Any = None,
        inner_sampler: SobolQMCNormalSampler | None = None,
        X_pending: Any = None,
        current_value: Any = None,
        cost_aware_utility: CostAwareUtility | None = None,
        project: Callable[[Any], Any] = _identity,
        expand: Callable[[Any], Any] = _identity,
        valfunc_cls: Any = None,
        valfunc_argfac: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if current_value is None and cost_aware_utility is not None:
            raise UnsupportedError("Cost-aware KG requires current_value to be specified.")
        if cost_aware_utility is not None and not isinstance(cost_aware_utility, CostAwareUtility):
            raise UnsupportedError(
                "`cost_aware_utility` must be a botorch_rs InverseCostWeightedUtility or "
                "GenericCostAwareUtility."
            )
        self._init(
            model,
            num_fantasies,
            sampler,
            objective,
            posterior_transform,
            inner_sampler,
            X_pending,
            current_value,
            multi_fidelity={
                "cost_aware_utility": None
                if cost_aware_utility is None
                else cost_aware_utility._rs_utility,
                "project": _project_arg(project),
                "expand": _expand_arg(expand),
                "value_function": _value_function_arg(model, valfunc_cls, valfunc_argfac),
            },
        )
        self.cost_aware_utility = cost_aware_utility
        self.project = project
        self.expand = expand
        self._cost_sampler: Any = None
        self.valfunc_cls = valfunc_cls
        self.valfunc_argfac = valfunc_argfac

    @property
    def cost_sampler(self) -> Any:
        """A copy of the fantasy sampler (BoTorch's ``cost_sampler``)."""
        if self._cost_sampler is None:
            self._cost_sampler = copy.deepcopy(self.sampler)
        return self._cost_sampler


class ProjectedAcquisitionFunction(AcquisitionFunction):
    r"""Defines a wrapper around an ``AcquisitionFunction`` that incorporates
    the project operator. Typically used to handle value functions in
    look-ahead methods."""

    def __init__(
        self, base_value_function: AcquisitionFunction, project: Callable[[Any], Any]
    ) -> None:
        if not isinstance(base_value_function, AcquisitionFunction):
            raise UnsupportedError(
                "`base_value_function` must be a botorch_rs acquisition function."
            )
        super().__init__(base_value_function.model)
        self.base_value_function = base_value_function
        self.project = project
        self.objective = getattr(base_value_function, "objective", None)
        self.posterior_transform = getattr(base_value_function, "posterior_transform", None)
        self.sampler = getattr(base_value_function, "sampler", None)
        self._log = getattr(base_value_function, "_log", False)
        spec = _project_arg(project)
        if spec is None:
            spec = ("callable", project)
        self._rs = _rs.projected_acquisition_function(base_value_function._rs, spec)


def _get_value_function(
    model: Any,
    objective: Any = None,
    posterior_transform: Any = None,
    sampler: Any = None,
    project: Callable[[Any], Any] | None = None,
    valfunc_cls: Any = None,
    valfunc_argfac: Callable[..., dict[str, Any]] | None = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    from botorch_rs.acquisition.analytic import PosteriorMean
    from botorch_rs.acquisition.monte_carlo import qSimpleRegret

    if valfunc_cls is not None:
        common_kwargs: dict[str, Any] = {
            "model": model,
            "posterior_transform": posterior_transform,
        }
        if issubclass(valfunc_cls, MCAcquisitionFunction):
            common_kwargs["sampler"] = sampler
            common_kwargs["objective"] = objective
        kwargs = valfunc_argfac(model=model) if valfunc_argfac is not None else {}
        base_value_function = valfunc_cls(**common_kwargs, **kwargs)
    else:
        if objective is not None:
            base_value_function = qSimpleRegret(
                model=model,
                sampler=sampler,
                objective=objective,
                posterior_transform=posterior_transform,
            )
        else:
            base_value_function = PosteriorMean(
                model=model, posterior_transform=posterior_transform
            )

    if project is None:
        return base_value_function
    return ProjectedAcquisitionFunction(base_value_function=base_value_function, project=project)


def _split_fantasy_points(X: Any, n_f: int) -> tuple[Any, Any]:
    r"""Split a one-shot optimization input into actual and fantasy points.

    Args:
        X: A ``batch_shape x (q + n_f) x d``-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A ``batch_shape x q x d``-dim tensor ``X_actual`` of input candidates.
        - A ``n_f x batch_shape x 1 x d``-dim tensor ``X_fantasies`` of fantasy
            points, where ``X_fantasies[i, batch_idx]`` is the i-th fantasy point
            associated with the batch indexed by ``batch_idx``.
    """
    data, shape = to_flat(X, "X")
    if n_f > shape[-2]:
        raise ValueError(f"n_f ({n_f}) must be less than the q-batch dimension of X ({shape[-2]})")
    if is_torch_tensor(X):
        X_actual, X_fantasies = X.split([X.size(-2) - n_f, n_f], dim=-2)
        X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
        return X_actual, X_fantasies.unsqueeze(dim=-2)
    rows, d = shape[-2], shape[-1]
    batch = tuple(shape[:-2])
    nb = len(data) // (rows * d) if rows * d else 0
    q = rows - n_f
    actual: list[float] = []
    fant: list[float] = [0.0] * (n_f * nb * d)
    for b in range(nb):
        block = data[b * rows * d : (b + 1) * rows * d]
        actual.extend(block[: q * d])
        for i in range(n_f):
            dst = (i * nb + b) * d
            fant[dst : dst + d] = block[(q + i) * d : (q + i + 1) * d]
    kind = kind_of(X)
    return (
        from_flat(actual, (*batch, q, d), kind),
        from_flat(fant, (n_f, *batch, 1, d), kind),
    )


__all__ = [
    "ProjectedAcquisitionFunction",
    "qKnowledgeGradient",
    "qMultiFidelityKnowledgeGradient",
]
