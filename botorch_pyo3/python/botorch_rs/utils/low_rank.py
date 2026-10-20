"""Low-rank updates of cached Cholesky factors (``botorch.utils.low_rank``).

``extract_batch_covar`` returns the per-output covariance blocks of a
multi-output posterior's distribution as a dense array (BoTorch returns the
blocks of a ``BlockDiagLinearOperator``); ``sample_cached_cholesky`` samples
the new points of a joint posterior over ``(X_baseline, X)`` by a low-rank
update of the cached baseline factor, in Rust, with BoTorch's use of the
base samples.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_to, from_flat, to_flat
from botorch_rs.exceptions.errors import BotorchError, NanError, UnsupportedError
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior, MultitaskMultivariateNormal


def _blocks(posterior: GPyTorchPosterior) -> tuple[list[float], tuple[int, ...]]:
    """The covariance blocks of the outputs (``batch x m x Q x Q``) of a
    botorch_rs posterior; ``BotorchError`` unless the outputs are
    independent (BoTorch's ``BlockDiagLinearOperator``)."""
    _, (cov, cshape) = posterior._get_joint()
    batch = tuple(cshape[:-2])
    m = posterior._m
    Q = posterior._q
    N = Q * m
    nb = math.prod(batch)
    out: list[float] = []
    for b in range(nb):
        base = b * N * N
        for o in range(m):
            for r in range(Q):
                row = base + (o * Q + r) * N
                out.extend(float(cov[row + o * Q + c]) for c in range(Q))
        # Off-diagonal blocks must vanish.
        for o in range(m):
            for p in range(m):
                if o == p:
                    continue
                for r in range(Q):
                    row = base + (o * Q + r) * N + p * Q
                    if any(float(cov[row + c]) != 0.0 for c in range(Q)):
                        raise BotorchError(
                            "Expected BlockDiagLinearOperator, but got a dense covariance "
                            "matrix (the outputs are correlated)."
                        )
    return out, (*batch, m, Q, Q)


def extract_batch_covar(mt_mvn: Any) -> Any:
    r"""Extract a batched independent covariance matrix from an MTMVN.

    Args:
        mt_mvn: A multi-task multivariate normal with a block diagonal
            covariance matrix.

    Returns:
        A lazy covariance matrix consisting of a batch of the blocks of
            the diagonal of the MultitaskMultivariateNormal.

    For a botorch_rs ``MultitaskMultivariateNormal`` the blocks are a dense
    array of shape ``batch_shape x m x q x q``; GPyTorch's distributions are
    handled as in BoTorch (the ``base_linear_op`` of the block-diagonal
    operator).
    """
    if isinstance(mt_mvn, MultitaskMultivariateNormal):
        data, shape = _blocks(mt_mvn._p)
        return from_flat(data, shape, mt_mvn._p._kind)
    lazy_covar = mt_mvn.lazy_covariance_matrix
    if type(lazy_covar).__name__ != "BlockDiagLinearOperator":
        raise BotorchError(f"Expected BlockDiagLinearOperator, but got {type(lazy_covar)}.")
    return lazy_covar.base_linear_op


def sample_cached_cholesky(
    posterior: Any,
    baseline_L: Any,
    q: int,
    base_samples: Any,
    sample_shape: Any,
    max_tries: int = 6,
) -> Any:
    r"""Get posterior samples at the ``q`` new points from the joint multi-output
    posterior.

    Args:
        posterior: The joint posterior is over (X_baseline, X).
        baseline_L: The baseline lower triangular cholesky factor.
        q: The number of new points in X.
        base_samples: The base samples.
        sample_shape: The sample shape.
        max_tries: The number of tries for computing the Cholesky
            decomposition with increasing jitter.


    Returns:
        A ``sample_shape x batch_shape x q x m``-dim tensor of posterior
            samples at the new points.

    ``posterior`` is a botorch_rs posterior; ``baseline_L`` has shape
    ``batch_shape x [m x] n x n`` and ``base_samples`` the ``sample_shape x
    [1 x ...] (n + q) x m`` base samples of the joint posterior (shared by
    the t-batches, as BoTorch views them).
    """
    if not isinstance(posterior, GPyTorchPosterior):
        raise UnsupportedError("sample_cached_cholesky requires a botorch_rs GPyTorchPosterior.")
    sample_shape = tuple(int(v) for v in sample_shape)
    S = math.prod(sample_shape)
    m = posterior._m
    Q = posterior._q
    n = Q - q
    batch = tuple(posterior._batch)
    nb = math.prod(batch)
    if posterior._is_mt:
        blocks, _ = _blocks(posterior)
    else:
        _, (cov, _) = posterior._get_joint()
        blocks = [float(v) for v in cov]
    (mean, _), _ = posterior._get_joint()
    rows: list[float] = []
    mean_new: list[float] = []
    for b in range(nb):
        for o in range(m):
            base = (b * m + o) * Q * Q
            for r in range(n, Q):
                rows.extend(blocks[base + r * Q : base + (r + 1) * Q])
            for r in range(n, Q):
                mean_new.append(float(mean[(b * Q + r) * m + o]))
    L, lshape = to_flat(baseline_L, "baseline_L")
    target_L_shape = (*batch, m, n, n) if posterior._is_mt else (*batch, n, n)
    try:
        L = broadcast_to(L, lshape, target_L_shape)
    except ValueError as error:
        raise BotorchError(
            f"baseline_L of shape {tuple(lshape)} does not match the posterior "
            f"(batch {batch}, {m} outputs, {n} baseline points)."
        ) from error
    z, zshape = to_flat(base_samples, "base_samples")
    if len(z) != S * Q * m:
        raise RuntimeError(
            f"shape {tuple(zshape)} of the base samples is invalid for the sample shape "
            f"{sample_shape} and {Q} x {m} points and outputs."
        )
    zz: list[float] = []
    for _b in range(nb):
        for o in range(m):
            for k in range(Q):
                zz.extend(float(z[(s * Q + k) * m + o]) for s in range(S))
    data, _ = _rs.sample_cached_cholesky(
        rows, list(L), mean_new, zz, nb * m, n, q, S, int(max_tries)
    )
    out = [0.0] * (S * nb * q * m)
    for b in range(nb):
        for o in range(m):
            for j in range(q):
                for s in range(S):
                    out[((s * nb + b) * q + j) * m + o] = float(data[((b * m + o) * q + j) * S + s])
    bad = [v for v in out if not math.isfinite(v)]
    if bad:
        suffix = []
        if any(math.isnan(v) for v in bad):
            suffix.append("nans")
        if any(math.isinf(v) for v in bad):
            suffix.append("infs")
        raise NanError(f"Samples contain {' and '.join(suffix)}.")
    return from_flat(out, (*sample_shape, *batch, q, m), posterior._kind)


__all__ = ["extract_batch_covar", "sample_cached_cholesky"]
