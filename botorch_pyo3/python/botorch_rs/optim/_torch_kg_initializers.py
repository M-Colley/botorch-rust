"""Device one-shot and fantasy-value initialization heuristics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from botorch_rs.optim._torch_acq_initializers import (
    acquisition_batch_shape,
    gen_initial_conditions,
    initialize_q_batch,
)


def _stream(bounds: Tensor, options: Mapping[str, Any]) -> Any:
    seed = options.get("seed")
    return None if seed is None else torch.Generator(device=bounds.device).manual_seed(seed)


def _resample(
    candidates: Tensor, values: Tensor, n: int, batch: tuple[int, ...], generator: Any, eta: float
) -> Tensor:
    """Boltzmann sampling across optimized restart points, independently per batch."""
    old_batch = candidates.shape[1:-2]
    missing = len(batch) - len(old_batch)
    if missing < 0:
        raise ValueError("Current optima cannot broadcast to the fantasy batch shape.")
    candidates = candidates.reshape(
        candidates.shape[0], *((1,) * missing), *candidates.shape[1:]
    ).expand(candidates.shape[0], *batch, *candidates.shape[-2:])
    values = values.reshape(values.shape[0], *((1,) * missing), *values.shape[1:]).expand(
        values.shape[0], *batch
    )
    std = values.std(0) if values.shape[0] > 1 else torch.zeros_like(values[0])
    std = torch.where(std > 0, std, torch.ones_like(std))
    logits = eta * (values - values.mean(0)) / std
    weights = (logits - logits.amax(0)).exp().movedim(0, -1)
    indices = torch.multinomial(
        weights.reshape(-1, values.shape[0]), n, replacement=True, generator=generator
    )
    indices = indices.reshape(*batch, n).movedim(-1, 0)
    return candidates.gather(0, indices[..., None, None].expand(n, *batch, *candidates.shape[-2:]))


def _inner(
    value: Any,
    bounds: Tensor,
    q: int,
    fixed: Any,
    options: Mapping[str, Any],
    inequality: Any = None,
    equality: Any = None,
) -> tuple[Tensor, Tensor]:
    from botorch_rs.generation.gen import gen_candidates_scipy, gen_candidates_torch
    from botorch_rs.optim.optimize import _INIT_KEYS, optimize_acqf

    forbidden = {"frac_random", "num_inner_restarts", "raw_inner_samples", "eta"}
    opts = {key: entry for key, entry in options.items() if key not in forbidden}
    solver_keys = set(opts) - _INIT_KEYS
    scipy_keys = {
        "method",
        "ftol",
        "gtol",
        "maxcor",
        "max_cor",
        "maxls",
        "pgtol",
        "factr",
        "tol",
        "maxfun",
        "disp",
        "iprint",
    }
    scipy = bool(inequality or equality or solver_keys & scipy_keys)
    return optimize_acqf(
        value,
        bounds,
        q,
        num_restarts=int(options.get("num_inner_restarts", 20)),
        raw_samples=int(options.get("raw_inner_samples", 1024)),
        fixed_features=fixed,
        return_best_only=False,
        options=opts,
        inequality_constraints=inequality,
        equality_constraints=equality,
        gen_candidates=gen_candidates_scipy if scipy else gen_candidates_torch,
    )


def _validate(options: Mapping[str, Any], default: float) -> float:
    fraction = float(options.get("frac_random", default))
    if not 0 < fraction < 1:
        raise ValueError(f"frac_random must take on values in (0,1). Value: {fraction}")
    return fraction


def one_shot_kg(arguments: Mapping[str, Any]) -> Tensor:
    from botorch_rs.acquisition.knowledge_gradient import _get_value_function

    args = dict(arguments)
    acquisition, bounds = args["acq_function"], args["bounds"]
    options = dict(args["options"] or {})
    fraction = _validate(options, 0.1)
    for name in ("inequality_constraints", "equality_constraints"):
        if any(torch.as_tensor(index).ndim != 1 for index, _, _ in args[name] or []):
            raise NotImplementedError(
                "Indices must be one-dimensional in gen_one_shot_kg_initial_conditions."
            )
    augmented = acquisition.get_augmented_q_batch_size(args["q"])
    initial = gen_initial_conditions(
        acquisition,
        bounds,
        augmented,
        args["num_restarts"],
        args["raw_samples"],
        args["fixed_features"],
        options,
        inequality_constraints=args["inequality_constraints"],
        equality_constraints=args["equality_constraints"],
    )
    count = int((1 - fraction) * (augmented - args["q"]))
    if count == 0:
        return initial
    value = _get_value_function(
        acquisition.model,
        objective=acquisition.objective,
        posterior_transform=acquisition.posterior_transform,
        sampler=acquisition.inner_sampler,
        project=getattr(acquisition, "project", None),
    )
    candidates, values = _inner(
        value,
        bounds,
        1,
        args["fixed_features"],
        options,
        args["inequality_constraints"],
        args["equality_constraints"],
    )
    batch = initial.shape[1:-2]
    selected = _resample(
        candidates,
        values,
        args["num_restarts"] * count,
        batch,
        _stream(bounds, options),
        float(options.get("eta", 2.0)),
    )
    selected = (
        selected.squeeze(-2)
        .reshape(args["num_restarts"], count, *batch, bounds.shape[-1])
        .movedim(1, -2)
    )
    return torch.cat((initial[..., :-count, :], selected), -2)


def one_shot_hvkg(arguments: Mapping[str, Any]) -> Tensor:
    from botorch_rs.acquisition._torch_backend_hvkg import tensor_hv_value_function

    args = dict(arguments)
    acquisition, bounds = args["acq_function"], args["bounds"]
    options = dict(args["options"] or {})
    fraction = _validate(options, 0.1)
    for name in ("inequality_constraints", "equality_constraints"):
        if any(torch.as_tensor(index).ndim != 1 for index, _, _ in args[name] or []):
            raise NotImplementedError("HVKG initialization requires intra-point constraints.")
    restarts = args["num_restarts"]
    optimized = round(restarts * (1 - fraction))
    augmented = acquisition.get_augmented_q_batch_size(args["q"])
    if optimized < restarts:
        initial = gen_initial_conditions(
            acquisition,
            bounds,
            augmented,
            restarts,
            args["raw_samples"],
            args["fixed_features"],
            options,
            inequality_constraints=args["inequality_constraints"],
            equality_constraints=args["equality_constraints"],
        )
        if optimized == 0:
            return initial
    else:
        initial = bounds.new_zeros(
            restarts, *acquisition_batch_shape(acquisition), augmented, bounds.shape[-1]
        )
    value = tensor_hv_value_function(
        acquisition.model,
        acquisition.ref_point,
        objective=acquisition.objective,
        sampler=acquisition.inner_sampler,
        use_posterior_mean=acquisition.use_posterior_mean,
    )
    fixed = dict(args["fixed_features"] or {})
    # Optimize in full physical coordinates with target fidelities fixed. This
    # avoids remapping parameter constraints to a reduced feature space.
    fixed.update(getattr(acquisition, "target_fidelities", None) or {})
    candidates, values = _inner(
        value,
        bounds,
        acquisition.num_pareto,
        fixed,
        options,
        args["inequality_constraints"],
        args["equality_constraints"],
    )
    batch = initial.shape[1:-2]
    generator = _stream(bounds, options)
    selected = _resample(
        candidates,
        values,
        optimized * acquisition.num_fantasies,
        batch,
        generator,
        float(options.get("eta", 2.0)),
    )
    selected = (
        selected.reshape(
            optimized, acquisition.num_fantasies, *batch, acquisition.num_pareto, bounds.shape[-1]
        )
        .movedim(1, -3)
        .flatten(-3, -2)
    )
    rows = (
        torch.randperm(restarts, device=bounds.device, generator=generator)[:optimized]
        if optimized < restarts
        else torch.arange(restarts, device=bounds.device)
    )
    initial = initial.clone()
    initial[rows, ..., args["q"] :, :] = selected
    for row in rows.unbind(0):
        conditional = gen_initial_conditions(
            acquisition,
            bounds,
            args["q"],
            1,
            args["raw_samples"],
            args["fixed_features"],
            options,
            inequality_constraints=args["inequality_constraints"],
            equality_constraints=args["equality_constraints"],
            fixed_X_fantasies=initial[row, ..., args["q"] :, :],
        )
        initial[row] = conditional[0]
    return initial


def value_function(arguments: Mapping[str, Any]) -> Tensor:
    from botorch_rs.acquisition.knowledge_gradient import _get_value_function
    from botorch_rs.optim._torch_acq_optim import _fixed_values, fix_features

    args = dict(arguments)
    acquisition, bounds = args["acq_function"], args["bounds"]
    options = dict(args["options"] or {})
    fraction = _validate(options, 0.6)
    value = _get_value_function(
        args["current_model"],
        objective=getattr(acquisition, "objective", None),
        posterior_transform=acquisition.posterior_transform,
        sampler=getattr(acquisition, "sampler", None),
        project=getattr(acquisition, "project", None),
    )
    candidates, values = _inner(value, bounds, 1, args["fixed_features"], options)
    batch = acquisition_batch_shape(acquisition)
    count = int((1 - fraction) * args["raw_samples"])
    selected = (
        _resample(
            candidates,
            values,
            count,
            batch,
            _stream(bounds, options),
            float(options.get("eta", 2.0)),
        )
        if count
        else bounds.new_empty(0, *batch, 1, bounds.shape[-1])
    )
    random_count = args["raw_samples"] - count
    dimension = bounds.shape[-1]
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=options.get("seed"))
    import math

    uniform = engine.draw(random_count * math.prod(batch), dtype=bounds.dtype).to(bounds.device)
    uniform = uniform.reshape(*batch, random_count, 1, dimension).movedim(len(batch), 0)
    random = bounds[0] + (bounds[1] - bounds[0]) * uniform
    raw = torch.cat((selected, random))
    raw = fix_features(raw, _fixed_values(raw, args["fixed_features"], bounds[0], bounds[1]))
    with torch.no_grad():
        scores = acquisition(raw)
        return initialize_q_batch(
            raw,
            scores,
            args["num_restarts"],
            eta=float(options.get("eta", 2.0)),
            generator=_stream(bounds, options),
        )[0]
