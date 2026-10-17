"""A posterior represented by a list of independent posteriors
(``botorch.posteriors.posterior_list``)."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import first_kind, from_flat, to_flat
from botorch_rs.posteriors.gpytorch import Posterior


def _cat_last(arrays: list[Any]) -> Any:
    """Concatenates arrays of equal leading shape along the last dimension."""
    flats = [to_flat(a, "x") for a in arrays]
    lead = tuple(flats[0][1][:-1])
    for _, shape in flats:
        if tuple(shape[:-1]) != lead:
            raise NotImplementedError(
                "`PosteriorList` is only supported if the constituent posteriors all have the "
                f"same `batch_shape`. Batch shapes: {[tuple(s[:-1]) for _, s in flats]}."
            )
    widths = [shape[-1] for _, shape in flats]
    rows = math.prod(lead)
    data: list[float] = []
    for r in range(rows):
        for (vals, _), w in zip(flats, widths, strict=True):
            data.extend(list(vals)[r * w : (r + 1) * w])
    return from_flat(data, (*lead, sum(widths)), first_kind(*arrays))


class PosteriorList(Posterior):
    r"""A Posterior represented by a list of independent posteriors (what a
    ``ModelListGP`` returns when a sub-model has a nonlinear outcome
    transform). ``mean``, ``variance`` and ``rsample`` concatenate those of
    the posteriors along the output dimension."""

    def __init__(self, *posteriors: Any) -> None:
        self.posteriors: list[Any] = list(posteriors)

    @property
    def device(self) -> Any:
        return getattr(self.posteriors[0], "device", None)

    @property
    def dtype(self) -> Any:
        return getattr(self.posteriors[0], "dtype", None)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        shapes = [p._extended_shape(sample_shape) for p in self.posteriors]
        batch = {tuple(s[:-1]) for s in shapes}
        if len(batch) > 1:
            raise NotImplementedError(
                "`PosteriorList` is only supported if the constituent posteriors all have the "
                f"same `batch_shape`. Batch shapes: {sorted(batch)}."
            )
        return (*shapes[0][:-1], sum(s[-1] for s in shapes))

    @property
    def mean(self) -> Any:
        r"""The mean of the posterior (only if all posteriors provide one)."""
        return _cat_last([p.mean for p in self.posteriors])

    @property
    def variance(self) -> Any:
        r"""The variance of the posterior (only if all posteriors provide one)."""
        return _cat_last([p.variance for p in self.posteriors])

    def rsample(self, sample_shape: Any = None) -> Any:
        r"""Samples of every posterior (with independent base samples),
        concatenated along the output dimension."""
        return _cat_last([p.rsample(sample_shape) for p in self.posteriors])

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"`PosteriorList` does not define the attribute {name}. "
            "Consider accessing the attributes of the individual posteriors instead."
        )

    def __repr__(self) -> str:
        return f"PosteriorList({', '.join(repr(p) for p in self.posteriors)})"


__all__ = ["PosteriorList"]
