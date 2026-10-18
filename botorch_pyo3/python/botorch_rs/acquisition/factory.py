r"""
Utilities for acquisition functions (``botorch.acquisition.factory``):
:func:`get_acquisition_function` builds botorch_rs acquisition functions by
name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition import logei, monte_carlo
from botorch_rs.acquisition.multi_objective import logei as moo_logei
from botorch_rs.acquisition.multi_objective import monte_carlo as moo_monte_carlo
from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch_rs.acquisition.utils import compute_best_feasible_objective


def _sampler(model: Any, mc_samples: int, seed: int | None) -> Any:
    """BoTorch's ``get_sampler`` for the posteriors of botorch_rs models: an
    ``IndexSampler`` for an ``EnsembleModel``, else a
    ``SobolQMCNormalSampler``."""
    if hasattr(model, "_ensemble_acquisition") and not hasattr(model, "_ensemble_members"):
        from botorch_rs.sampling.index_sampler import IndexSampler

        return IndexSampler(sample_shape=(mc_samples,), seed=seed)
    from botorch_rs.sampling.normal import SobolQMCNormalSampler

    return SobolQMCNormalSampler(sample_shape=(mc_samples,), seed=seed)


def _feasible(Y: Any, constraints: list[Callable[[Any], Any]]) -> Any:
    """``Y[all(c(Y) <= 0)]`` (torch for the constraint callables)."""
    from botorch_rs.acquisition.input_constructors import _feasible_rows

    return _feasible_rows(Y, constraints)


def _as_ref_point(ref_point: Any, Y: Any) -> Any:
    """``torch.as_tensor(ref_point, dtype=Y.dtype, device=Y.device)``."""
    if is_torch_tensor(Y):
        import torch

        return torch.as_tensor(ref_point, dtype=Y.dtype, device=Y.device)
    data, shape = to_flat(ref_point, "ref_point")
    return from_flat(list(data), tuple(shape), kind_of(Y))


def get_acquisition_function(
    acquisition_function_name: str,
    model: Any,
    objective: Any,
    X_observed: Any,
    posterior_transform: Any = None,
    X_pending: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    eta: Any = 1e-3,
    mc_samples: int = 512,
    seed: int | None = None,
    *,
    # optional parameters that are only needed for certain acquisition functions
    tau: float = 1e-3,
    prune_baseline: bool = True,
    marginalize_dim: int | None = None,
    cache_root: bool | None = None,
    beta: float | None = None,
    ref_point: Any = None,
    Y: Any = None,
    alpha: float = 0.0,
) -> Any:
    r"""Convenience function for initializing botorch_rs acquisition functions.

    Args:
        acquisition_function_name: Name of the acquisition function.
        model: A fitted model.
        objective: A MCAcquisitionObjective.
        X_observed: A ``m1 x d``-dim Tensor of ``m1`` design points that have
            already been observed.
        posterior_transform: A PosteriorTransform (optional).
        X_pending: A ``m2 x d``-dim Tensor of ``m2`` design points whose evaluation
            is pending.
        constraints: A list of callables, each mapping a Tensor of dimension
            ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
            ``sample_shape x batch-shape x q``, where negative values imply
            feasibility. Used for all acquisition functions except qSR and qUCB.
        eta: The temperature parameter for the sigmoid function used for the
            differentiable approximation of the constraints. In case of a float the
            same eta is used for every constraint in constraints. In case of a
            tensor the length of the tensor must match the number of provided
            constraints. The i-th constraint is then estimated with the i-th
            eta value. Used for all acquisition functions except qSR and qUCB.
        mc_samples: The number of samples to use for (q)MC evaluation of the
            acquisition function.
        seed: If provided, perform deterministic optimization (i.e. the
            function to optimize is fixed and not stochastic).

    Returns:
        The requested acquisition function.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> obj = LinearMCObjective(weights=torch.tensor([1.0, 2.0]))
        >>> acqf = get_acquisition_function("qEI", model, obj, train_X)
    """
    # thompson sampling does not need a sampler, so we do this first
    if acquisition_function_name == "pTS":
        return PathwiseThompsonSampling(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
        )
    # initialize the sampler
    sampler = _sampler(model, mc_samples, seed)
    if posterior_transform is not None and acquisition_function_name in [
        "qEHVI",
        "qNEHVI",
        "qLogEHVI",
        "qLogNEHVI",
    ]:
        raise NotImplementedError(
            "PosteriorTransforms are not yet implemented for multi-objective "
            "acquisition functions."
        )
    # instantiate and return the requested acquisition function
    if acquisition_function_name in ("qEI", "qLogEI", "qPI"):
        # Since these are the non-noisy variants, use the posterior mean at the observed
        # inputs directly to compute the best feasible value without sampling.
        Y = model.posterior(X_observed, posterior_transform=posterior_transform).mean
        obj = objective(samples=Y, X=X_observed)
        best_f = compute_best_feasible_objective(
            samples=Y,
            obj=obj,
            constraints=constraints,
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            X_baseline=X_observed,
        )
    if acquisition_function_name in ["qEI", "qLogEI"]:
        acqf_class = (
            monte_carlo.qExpectedImprovement
            if acquisition_function_name == "qEI"
            else logei.qLogExpectedImprovement
        )
        return acqf_class(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
    elif acquisition_function_name == "qPI":
        return monte_carlo.qProbabilityOfImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            tau=tau,
            constraints=constraints,
            eta=eta,
        )
    elif acquisition_function_name == "qLogPF":
        return logei.qLogProbabilityOfFeasibility(
            model=model,
            constraints=constraints,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            eta=eta,
        )
    elif acquisition_function_name in ["qNEI", "qLogNEI"]:
        acqf_class = (
            monte_carlo.qNoisyExpectedImprovement
            if acquisition_function_name == "qNEI"
            else logei.qLogNoisyExpectedImprovement
        )
        return acqf_class(
            model=model,
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        )
    elif acquisition_function_name == "qSR":
        return monte_carlo.qSimpleRegret(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
    elif acquisition_function_name == "qUCB":
        if beta is None:
            raise ValueError("`beta` must be not be None for qUCB.")
        return monte_carlo.qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
    elif acquisition_function_name in ["qEHVI", "qLogEHVI"]:
        if Y is None:
            raise ValueError(f"`Y` must not be None for {acquisition_function_name}")
        if ref_point is None:
            raise ValueError(f"`ref_point` must not be None for {acquisition_function_name}")
        # get feasible points
        if constraints is not None:
            Y = _feasible(Y, constraints)
        obj = objective(Y)
        from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
            FastNondominatedPartitioning,
            NondominatedPartitioning,
        )

        partitioning: Any
        if alpha > 0:
            partitioning = NondominatedPartitioning(
                ref_point=_as_ref_point(ref_point, Y),
                Y=obj,
                alpha=alpha,
            )
        else:
            partitioning = FastNondominatedPartitioning(
                ref_point=_as_ref_point(ref_point, Y),
                Y=obj,
            )
        acqf_class = (
            moo_monte_carlo.qExpectedHypervolumeImprovement
            if acquisition_function_name == "qEHVI"
            else moo_logei.qLogExpectedHypervolumeImprovement
        )
        return acqf_class(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
    elif acquisition_function_name in ["qNEHVI", "qLogNEHVI"]:
        if ref_point is None:
            raise ValueError(f"`ref_point` must not be None for {acquisition_function_name}")
        acqf_class = (
            moo_monte_carlo.qNoisyExpectedHypervolumeImprovement
            if acquisition_function_name == "qNEHVI"
            else moo_logei.qLogNoisyExpectedHypervolumeImprovement
        )
        return acqf_class(
            model=model,
            ref_point=ref_point,
            X_baseline=X_observed,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            prune_baseline=prune_baseline,
            alpha=alpha,
            X_pending=X_pending,
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
        )
    raise NotImplementedError(f"Unknown acquisition function {acquisition_function_name}")


__all__ = ["get_acquisition_function"]
