"""Stopping criteria (``botorch.optim.stopping``)."""

from __future__ import annotations

from typing import Any, Protocol

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat


class StoppingCriterion(Protocol):
    r"""Protocol for evaluating optimization convergence.

    Stopping criteria are implemented as objects rather than functions, so
    that they can keep track of past function values between optimization
    steps.
    """

    def __call__(self, fvals: Any) -> bool:  # pragma: no cover - protocol
        r"""Evaluate the stopping criterion (``True`` to stop)."""
        ...

    def reset(self) -> None:  # pragma: no cover - protocol
        r"""Reset the stopping criterion to its initial state."""
        ...


class ExpMAStoppingCriterion:
    r"""Exponential moving average stopping criterion (BoTorch's
    ``ExpMAStoppingCriterion``).

    Computes an exponentially weighted moving average over window length
    ``n_window`` and checks whether the relative decrease in this moving
    average between steps is less than a provided tolerance level. That is,
    in iteration ``i``, it computes

        v[i,j] := fvals[i - n_window + j] * w[j]

    for all ``j = 0, ..., n_window``, where ``w[j] = exp(-eta * (1 - j /
    n_window))``. Letting ``ma[i] := sum_j(v[i,j])``, the criterion evaluates
    to ``True`` whenever

        (ma[i-1] - ma[i]) / abs(ma[i-1]) < rel_tol (if minimize=True)
        (ma[i] - ma[i-1]) / abs(ma[i-1]) < rel_tol (if minimize=False)

    The weights are torch's float32 values (BoTorch computes them in float32);
    the native optimizers of botorch_rs evaluate the same criterion in Rust.
    """

    def __init__(
        self,
        maxiter: int = 10000,
        minimize: bool = True,
        n_window: int = 10,
        eta: float = 1.0,
        rel_tol: float = 1e-5,
    ) -> None:
        r"""Exponential moving average stopping criterion.

        Args:
            maxiter: Maximum number of iterations.
            minimize: If True, assume minimization.
            n_window: The size of the exponential moving average window.
            eta: The exponential decay factor in the weights.
            rel_tol: Relative tolerance for termination.
        """
        self.maxiter = int(maxiter)
        self.minimize = bool(minimize)
        self.n_window = int(n_window)
        self.eta = float(eta)
        self.rel_tol = float(rel_tol)
        self.iter = 0
        self.weights = list(_rs.expma_weights_py(self.eta, self.n_window))
        self._prev_fvals: list[list[float]] | None = None

    def _native_args(self) -> tuple[int, bool, int, float, float]:
        """The arguments of the native criterion."""
        return (self.maxiter, self.minimize, self.n_window, self.eta, self.rel_tol)

    def __call__(self, fvals: Any) -> bool:
        r"""Evaluate the stopping criterion.

        Args:
            fvals: The function values of the current iteration (a scalar or
                an array). If ``fvals`` contains more than one element, the
                stopping criterion is evaluated element-wise and True is
                returned if it is true for all elements.

        Returns:
            Stopping indicator (if True, stop the optimization).
        """
        self.iter += 1
        if self.iter == self.maxiter:
            return True
        values = [float(v) for v in to_flat(fvals, "fvals")[0]]
        if self._prev_fvals is None:
            self._prev_fvals = [values]
        else:
            self._prev_fvals = self._prev_fvals[-self.n_window :] + [values]
        if len(self._prev_fvals) < self.n_window + 1:
            return False
        max_delta = -float("inf")
        for e in range(len(values)):
            prev_ma = 0.0
            ma = 0.0
            for j, w in enumerate(self.weights):
                prev_ma += self._prev_fvals[j][e] * w
                ma += self._prev_fvals[j + 1][e] * w
            rel_delta = (
                (prev_ma - ma) / abs(prev_ma)
                if prev_ma != 0
                else (float("nan") if prev_ma == ma else float("inf") * (1 if prev_ma > ma else -1))
            )
            if not self.minimize:
                rel_delta = -rel_delta
            if rel_delta != rel_delta or max_delta != max_delta:
                max_delta = float("nan")
            elif rel_delta > max_delta:
                max_delta = rel_delta
        return max_delta < self.rel_tol

    def reset(self) -> None:
        r"""Reset the stopping criterion to its initial state."""
        self.iter = 0
        self._prev_fvals = None


__all__ = ["ExpMAStoppingCriterion", "StoppingCriterion"]
