"""A posterior represented by a list of independent posteriors
(``botorch.posteriors.posterior_list``)."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import first_kind, from_flat, to_flat
from botorch_rs.posteriors.gpytorch import Posterior


def _cat_last(arrays: list[Any]) -> Any:
    """Concatenates arrays along the last dimension (``torch.cat(..., -1)``)."""
    flats = [to_flat(a, "values") for a in arrays]
    lead = tuple(flats[0][1][:-1])
    if any(tuple(f[1][:-1]) != lead for f in flats):
        raise NotImplementedError(
            "`PosteriorList` is only supported if the constituent posteriors all have the "
            f"same `batch_shape`. Batch shapes: {[tuple(f[1][:-1]) for f in flats]}."
        )
    widths = [f[1][-1] for f in flats]
    rows = math.prod(lead)
    out: list[float] = []
    for r in range(rows):
        for (data, _), w in zip(flats, widths, strict=True):
            out.extend(data[r * w : (r + 1) * w])
    return from_flat(out, (*lead, sum(widths)), first_kind(*arrays))


class PosteriorList(Posterior):
    r"""A Posterior represented by a list of independent Posteriors (what a
    ``ModelList`` returns, and a ``ModelListGP`` when a sub-model has a
    nonlinear outcome transform). ``mean``, ``variance`` and ``rsample``
    concatenate those of the posteriors along the output dimension."""

    def __init__(self, *posteriors: Any) -> None:
        r"""A Posterior represented by a list of independent Posteriors.

        Args:
            *posteriors: A variable number of single-outcome posteriors.

        Example:
            >>> p_1 = model_1.posterior(test_X)
            >>> p_2 = model_2.posterior(test_X)
            >>> p_12 = PosteriorList(p_1, p_2)

        Note: This is typically produced automatically in ``ModelList``; it should
        generally not be necessary for the end user to invoke it manually.
        """
        self.posteriors: list[Any] = list(posteriors)

    @property
    def device(self) -> Any:
        r"""The torch device of the posterior."""
        return getattr(self.posteriors[0], "device", None)

    @property
    def dtype(self) -> Any:
        r"""The torch dtype of the posterior."""
        return getattr(self.posteriors[0], "dtype", None)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        shapes = [tuple(p._extended_shape(sample_shape)) for p in self.posteriors]
        batch_shapes = [s[:-1] for s in shapes]
        if len(set(batch_shapes)) > 1:
            raise NotImplementedError(
                "`PosteriorList` is only supported if the constituent posteriors "
                f"all have the same `batch_shape`. Batch shapes: {batch_shapes}."
            )
        return (*batch_shapes[0], sum(s[-1] for s in shapes))

    @property
    def mean(self) -> Any:
        r"""The mean of the posterior as a ``(b) x n x m``-dim Tensor."""
        return _cat_last([p.mean for p in self.posteriors])

    @property
    def variance(self) -> Any:
        r"""The variance of the posterior as a ``(b) x n x m``-dim Tensor."""
        return _cat_last([p.variance for p in self.posteriors])

    def rsample(self, sample_shape: Any = None) -> Any:
        r"""Sample from the posterior: the samples of the posteriors,
        concatenated along the output dimension."""
        return _cat_last([p.rsample(sample_shape=sample_shape) for p in self.posteriors])

    def __getattr__(self, name: str) -> Any:
        r"""A catch-all for attributes not defined on the posterior level.

        Raises an attribute error.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(
            f"`PosteriorList` does not define the attribute {name}. "
            "Consider accessing the attributes of the individual posteriors instead."
        )

    def __repr__(self) -> str:
        return f"PosteriorList({', '.join(repr(p) for p in self.posteriors)})"


__all__ = ["PosteriorList"]
