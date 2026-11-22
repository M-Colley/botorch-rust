"""Independent native L-BFGS-B states with batched Tensor acquisition calls."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.warnings import OptimizationWarning

_OPTIONS = {"maxcor", "ftol", "gtol", "maxiter", "maxls"}


def eligible(
    initial: Tensor,
    lo: Tensor,
    hi: Tensor,
    fixed: dict[int, Tensor],
    method: Any,
    with_grad: bool,
    constrained: bool,
    timeout: Any,
    options: dict[str, Any],
) -> bool:
    if (
        initial.ndim not in (2, 3)
        or method != "L-BFGS-B"
        or not with_grad
        or constrained
        or timeout is not None
        or set(options) - _OPTIONS
    ):
        return False
    free = [i for i in range(initial.shape[-1]) if i not in fixed]
    return (
        initial.ndim == 2
        or not free
        or (
            torch.equal(lo[..., free], lo[:1, ..., free].expand_as(lo[..., free]))
            and torch.equal(hi[..., free], hi[:1, ..., free].expand_as(hi[..., free]))
        )
    )


def gen_parallel(
    initial: Tensor,
    acquisition: Any,
    lo: Tensor,
    hi: Tensor,
    fixed: dict[int, Tensor],
    options: dict[str, Any],
    callback: Any,
) -> tuple[Tensor, Tensor]:
    import numpy as np

    from botorch_rs.exceptions.errors import OptimizationGradientError
    from botorch_rs.optim._torch_acq_optim import clamp, fix_features
    from botorch_rs.optim.batched_lbfgs_b import fmin_l_bfgs_b_batched

    free = [i for i in range(initial.shape[-1]) if i not in fixed]
    if not free:
        with torch.no_grad():
            return initial, acquisition(initial)
    shape = initial.shape
    batch = initial.unsqueeze(0) if initial.ndim == 2 else initial
    batch_lo = lo.unsqueeze(0) if initial.ndim == 2 else lo
    batch_hi = hi.unsqueeze(0) if initial.ndim == 2 else hi
    n = len(batch)
    free_shape = (batch.shape[-2], len(free))
    current = batch.detach().clone()
    initial_done = [False] * n
    last_finite_loss = [0.0] * n
    x0 = batch[..., free].double().cpu().numpy().reshape(n, -1)
    bounds = list(
        zip(
            batch_lo[0][..., free].double().cpu().numpy().ravel(),
            batch_hi[0][..., free].double().cpu().numpy().ravel(),
            strict=True,
        )
    )

    def objective(vectors: Any, batch_indices: list[int]) -> tuple[Any, Any]:
        nonlocal current
        index = torch.as_tensor(batch_indices, device=batch.device)
        values = torch.as_tensor(vectors, dtype=batch.dtype, device=batch.device).reshape(
            len(batch_indices), *free_shape
        )
        updated = current.index_select(0, index).clone()
        updated[..., free] = values
        trial = current.index_copy(0, index, updated).detach()
        variable = trial.clone().requires_grad_(True)
        full = variable.squeeze(0) if len(shape) == 2 else variable
        losses = -acquisition(full).reshape(-1)
        if losses.shape != (n,):
            raise ValueError("Parallel acquisition values must have one value per restart.")
        loss = losses.index_select(0, index)
        gradient = torch.autograd.grad(loss.sum(), variable)[0].index_select(0, index)[..., free]
        loss_np = loss.detach().double().cpu().numpy().copy()
        grad_np = gradient.detach().double().cpu().numpy().reshape(len(batch_indices), -1).copy()
        for j, i in enumerate(batch_indices):
            if np.isfinite(loss_np[j]) and np.all(np.isfinite(grad_np[j])):
                initial_done[i] = True
                last_finite_loss[i] = float(loss_np[j])
                current[i] = trial[i]
            elif not initial_done[i]:
                nan_count = int((~np.isfinite(grad_np[j])).sum())
                msg = (
                    f"{nan_count} elements of the {grad_np[j].size} element "
                    "gradient array `gradf` are non-finite (or objective is non-finite). "
                    "This often indicates numerical issues."
                )
                if batch.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise OptimizationGradientError(msg, current_x=vectors[j].copy())
            else:
                # Keep `current[i]` at its last valid iterate and supply a steep
                # finite penalty so L-BFGS-B line search backtracks cleanly.
                loss_np[j] = max(last_finite_loss[i], 0.0) + 1e30
                grad_np[j].fill(0.0)
        return loss_np, grad_np

    settings = dict(options)
    ftol = settings.pop("ftol", None)
    if "gtol" in settings:
        settings["pgtol"] = settings.pop("gtol")
    xs, _, results = fmin_l_bfgs_b_batched(
        objective,
        x0,
        bounds=bounds,
        callback=callback,
        pass_batch_indices=True,
        **({"ftol": ftol, "factr": None} if ftol is not None else {}),
        **settings,
    )
    for result in results:
        if not result.success:
            warnings.warn(
                f"Native L-BFGS-B candidate generation: {result.message}",
                OptimizationWarning,
                stacklevel=3,
            )
    final = current.clone()
    final[..., free] = torch.as_tensor(
        np.asarray(xs), dtype=batch.dtype, device=batch.device
    ).reshape(n, *free_shape)
    final = final.reshape(shape)
    final = clamp(fix_features(final, fixed), lo, hi)
    with torch.no_grad():
        return final.detach(), acquisition(final)
