"""Pathwise updates (``botorch.sampling.pathwise.update_strategies``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.paths import GeneralizedLinearPath
from botorch_rs.sampling.pathwise.prior_samplers import seed_arg
from botorch_rs.sampling.pathwise.utils import model_outputs


class _Default:
    def __repr__(self) -> str:
        return "DEFAULT"


#: The default ``likelihood`` (the model's own).
DEFAULT = _Default()


def _per_output(values: Any, m: int, n: int, name: str) -> tuple[list[Any], tuple[int, ...]]:
    """``sample_shape x [m x] n`` values as one ``S x n`` flat matrix per
    output, and ``sample_shape``."""
    data, shape = to_flat(values, name)
    lead = 2 if m > 1 else 1
    if len(shape) < lead or shape[-1] != n or (m > 1 and shape[-2] != m):
        expected = f"sample_shape x {m} x {n}" if m > 1 else f"sample_shape x {n}"
        raise ValueError(f"Expected `{name}` of shape `{expected}`, got {shape}.")
    sample_shape = tuple(shape[:-lead])
    s = 1
    for v in sample_shape:
        s *= v
    values_list = list(data)
    out = []
    for t in range(m):
        rows: list[float] = []
        for k in range(s):
            start = (k * m + t) * n
            rows.extend(values_list[start : start + n])
        out.append((rows, (s, n)))
    return out, sample_shape


def gaussian_update(
    model: Any,
    sample_values: Any,
    likelihood: Any = DEFAULT,
    *,
    target_values: Any = None,
    noise: Any = None,
    seed: int | None = None,
) -> GeneralizedLinearPath:
    r"""The Gaussian pathwise update of an exact GP (BoTorch's
    ``gaussian_update``)::

        update(·) = k(·, X) (K + Σ)^{-1} (y - f(X) - ε),   ε ~ N(0, Σ),

    where ``X`` and ``y`` are the model's transformed training inputs and
    targets, ``Σ`` its observation noise and ``sample_values`` the values
    ``f(X)`` of prior paths at ``X`` (``sample_shape x [m x] n``, in the
    transformed outcome space). The returned path applies the model's input
    transform and returns values in the transformed outcome space.

    botorch_rs extensions (keyword-only): ``noise``, the standard normal
    draws of ``ε / sqrt(Σ)`` (shape of ``sample_values``; BoTorch's
    ``torch.randn_like``), or ``seed`` (the noise of ``draw_matheron_paths``
    with the same seed).
    """
    from botorch_rs.models.gp_regression import SingleTaskGP

    if not isinstance(model, SingleTaskGP):
        raise UnsupportedError("botorch_rs gaussian_update requires a botorch_rs SingleTaskGP.")
    if likelihood not in (DEFAULT, None) and likelihood is not getattr(model, "likelihood", None):
        raise UnsupportedError("gaussian_update only supports the model's own likelihood.")
    if target_values is not None:
        raise UnsupportedError(
            "botorch_rs gaussian_update always uses the model's training targets."
        )
    rs_model, sizes = model_outputs(model)
    m = sizes[0]
    n = int(model._rs.num_train)
    values, sample_shape = _per_output(sample_values, m, n, "sample_values")
    noises = None if noise is None else _per_output(noise, m, n, "noise")[0]
    handle = _rs.gaussian_update(rs_model, values, noises, seed_arg(seed))
    return GeneralizedLinearPath._from_native(handle, sample_shape, range(m), model._kind)


__all__ = ["gaussian_update"]
