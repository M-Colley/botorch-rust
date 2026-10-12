"""``project_to_target_fidelity`` and ``expand_trace_observations``
(``botorch.acquisition.utils``): computed in Rust, or with torch operations
(as in BoTorch) for torch tensors, so that gradients flow through them."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, is_torch_tensor, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError


def project_to_target_fidelity(X: Any, target_fidelities: dict[int, float], d: int) -> Any:
    r"""``X`` (``batch_shape x q x d``, or ``x (d - d_f)``) with the fidelity
    columns set to their target values (inserted if ``X`` lacks them);
    negative indices count from the end."""
    tfs = {int(k) if k >= 0 else d + int(k): float(v) for k, v in target_fidelities.items()}
    out_of_bounds = sorted(k for k in tfs if k < 0 or k >= d)
    if out_of_bounds:
        raise BotorchTensorDimensionError(
            f"Target fidelity indices {out_of_bounds} are out of bounds for `d={d}`."
        )
    if is_torch_tensor(X):
        import torch

        ones = torch.ones(*X.shape[:-1], device=X.device, dtype=X.dtype)
        if X.shape[-1] == d:
            return torch.stack(
                [X[..., i] if i not in tfs else tfs[i] * ones for i in range(d)], dim=-1
            )
        if X.shape[-1] == d - len(target_fidelities):
            cols, j = [], 0
            for i in range(d):
                if i not in tfs:
                    cols.append(X[..., j])
                    j += 1
                else:
                    cols.append(tfs[i] * ones)
            return torch.stack(cols, dim=-1)
        raise BotorchTensorDimensionError(
            f"X must have a last dimension with size `d` or `d-d_f`, but got {X.shape[-1]}."
        )
    flat = to_flat(X, "X")
    if len(flat[1]) == 1:
        flat = (flat[0], (1, flat[1][0]))
        out = _rs.project_to_target_fidelity(flat, [(k, v) for k, v in tfs.items()], int(d))
        data, _ = out
        return from_pair((data, [d]), kind_of(X))
    out = _rs.project_to_target_fidelity(flat, [(k, v) for k, v in tfs.items()], int(d))
    return from_pair(out, kind_of(X))


def expand_trace_observations(
    X: Any, fidelity_dims: list[int] | None = None, num_trace_obs: int = 0
) -> Any:
    r"""``X`` (``batch_shape x q x d``) followed by ``num_trace_obs`` copies
    whose fidelity columns (default: the last one) are scaled by
    ``(K - j) / (K + 1)`` (``K = num_trace_obs``): the trace observations
    obtained while evaluating ``X``. Returns ``batch_shape x (q + K q) x d``."""
    if num_trace_obs == 0:
        return X
    if is_torch_tensor(X):
        import torch

        dims = [-1] if fidelity_dims is None else list(fidelity_dims)
        reps = [1] * (X.ndim - 2) + [1 + num_trace_obs, 1]
        X_expanded = X.repeat(*reps)
        scale_fac = torch.ones_like(X_expanded)
        s_pad = 1 / (num_trace_obs + 1)
        sf = torch.linspace(1 - s_pad, s_pad, num_trace_obs, device=X.device, dtype=X.dtype)
        q = X.size(-2)
        sf = torch.repeat_interleave(sf, q)
        sf = sf.unsqueeze(-1).expand(X_expanded.size(-2) - q, len(dims))
        scale_fac[..., q:, dims] = sf
        return scale_fac * X_expanded
    out = _rs.expand_trace_observations(
        to_flat(X, "X"),
        None if fidelity_dims is None else [int(k) for k in fidelity_dims],
        int(num_trace_obs),
    )
    return from_pair(out, kind_of(X))


__all__ = ["expand_trace_observations", "project_to_target_fidelity"]
