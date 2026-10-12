"""Quasi-random and polytope sampling (``botorch.utils.sampling``).

The Sobol points come from an Owen-scrambled Sobol sequence (Burley's
hash-based scrambling, Joe-Kuo direction numbers): for a given seed they
differ from torch's ``SobolEngine``, with the same distribution.

The polytope samplers (``get_polytope_samples``,
``HitAndRunPolytopeSampler``, ``sample_polytope``) implement BoTorch's
hit-and-run algorithm (unit-cube normalization, equality constraints via
their null space, an interior point from a linear program); their random
streams differ from torch's for the same seed, with the same distribution.

``sample_simplex`` and ``sample_hypersphere`` apply BoTorch's transformations
(sorted spacings of uniform numbers, normalized normal vectors) to i.i.d.
numbers from a seeded Xoshiro256++ generator or, with ``qmc=True``, to the
scrambled Sobol sequence above; for the same seed the samples differ from
BoTorch's, with the same distribution.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_flat, from_pair, kind_from_dtype, to_flat


def draw_sobol_samples(
    bounds: Any,
    n: int,
    q: int,
    batch_shape: Any = None,
    seed: int | None = None,
) -> Any:
    r"""``n`` scrambled Sobol ``q``-batches in the box ``bounds`` (``2 x d``):
    an array of shape ``n x batch_shape x q x d`` of the type of ``bounds``."""
    batch = tuple(int(b) for b in batch_shape) if batch_shape is not None else ()
    nb = math.prod(batch)
    data, shape = _rs.draw_sobol_samples(
        to_flat(bounds, "bounds"), n * nb, int(q), None if seed is None else int(seed)
    )
    if batch:
        # Like BoTorch, consecutive draws fill `batch_shape x n` and are then
        # permuted to `n x batch_shape`.
        row = math.prod(shape[1:])
        out = [0.0] * len(data)
        for b in range(nb):
            for i in range(n):
                src, dst = (b * n + i) * row, (i * nb + b) * row
                out[dst : dst + row] = data[src : src + row]
        data = out
    return from_flat(data, (n, *batch, *shape[1:]), first_kind(bounds))


def draw_sobol_normal_samples(
    d: int,
    n: int,
    device: Any = None,
    dtype: Any = None,
    seed: int | None = None,
) -> Any:
    r"""``n x d`` quasi-random standard-normal samples (inverse transform of a
    scrambled Sobol sequence)."""
    return from_pair(
        _rs.draw_sobol_normal_samples(int(d), int(n), None if seed is None else int(seed)),
        kind_from_dtype(dtype, device),
    )


def _seed(seed: int | None) -> int | None:
    return None if seed is None else int(seed)


def _linear_constraints_arg(constraints: Any, name: str) -> Any:
    # Imported lazily: ``botorch_rs.optim`` imports the acquisition functions.
    from botorch_rs.optim.parameter_constraints import linear_constraints_arg as convert

    return convert(constraints, name)


def get_polytope_samples(
    n: int,
    bounds: Any,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    seed: int | None = None,
    n_burnin: int = 10000,
    n_thinning: int = 32,
) -> Any:
    r"""``n`` samples (``n x d``) from the polytope defined by ``bounds``
    (``2 x d``) and the linear constraints (BoTorch's ``(indices,
    coefficients, rhs)`` tuples, ``sum_i x[indices[i]] * coefficients[i] >=
    rhs`` or ``= rhs``) with a hit-and-run sampler (``n_burnin`` burn-in
    steps, every ``n_thinning``-th sample)."""
    out = _rs.get_polytope_samples(
        int(n),
        to_flat(bounds, "bounds"),
        _linear_constraints_arg(inequality_constraints, "inequality_constraints"),
        _linear_constraints_arg(equality_constraints, "equality_constraints"),
        _seed(seed),
        int(n_burnin),
        int(n_thinning),
    )
    return from_pair(out, first_kind(bounds))


def sparse_to_dense_constraints(
    d: int, constraints: list[tuple[Any, Any, float]]
) -> tuple[Any, Any]:
    r"""Dense ``(A, b)`` (``n_constraints x d`` and ``n_constraints x 1``)
    with ``A[i, indices_i] = coefficients_i`` and ``b[i] = rhs_i``."""
    kind = first_kind(*(c[1] for c in constraints))
    a, b = _rs.sparse_to_dense_constraints(
        int(d), _linear_constraints_arg(constraints, "constraints")
    )
    return from_pair(a, kind), from_pair(b, kind)


def find_interior_point(A: Any, b: Any, A_eq: Any = None, b_eq: Any = None) -> Any:
    r"""A point of ``{x : A x <= b, A_eq x = b_eq}`` maximizing the slack of
    the inequalities (the linear program ``min -s`` subject to ``A x + 2 s <=
    b``, ``s >= 0``, ``A_eq (x, s) = b_eq``, with ``s <= 1`` if unbounded).
    ``A_eq`` may include BoTorch's zero column of the slack variable.

    Raises:
        InfeasibilityError: If the polytope is empty.
    """
    x = _rs.find_interior_point(
        to_flat(A, "A"),
        to_flat(b, "b"),
        None if A_eq is None else to_flat(A_eq, "A_eq"),
        None if b_eq is None else to_flat(b_eq, "b_eq"),
    )
    return from_flat(x, (len(x),), first_kind(A, b))


def sample_polytope(
    A: Any,
    b: Any,
    x0: Any,
    n: int = 10000,
    n0: int = 100,
    n_thinning: int = 1,
    seed: int | None = None,
) -> Any:
    r"""``n`` hit-and-run samples (``n x d``) of ``{x : A x <= b}`` started
    from the interior point ``x0`` (``d x 1``) after ``n0`` burn-in steps,
    keeping every ``n_thinning``-th sample."""
    out = _rs.sample_polytope(
        to_flat(A, "A"),
        to_flat(b, "b"),
        to_flat(x0, "x0"),
        int(n),
        int(n0),
        int(n_thinning),
        _seed(seed),
    )
    return from_pair(out, first_kind(A, b, x0))


class HitAndRunPolytopeSampler:
    r"""A hit-and-run sampler of the polytope ``{x : A x <= b, C x = e}``
    within ``bounds`` (BoTorch's ``HitAndRunPolytopeSampler``).

    Args:
        inequality_constraints: ``(A, b)`` with ``A x <= b`` (``A`` of shape
            ``n_ineq x d``, ``b`` of shape ``n_ineq x 1``).
        equality_constraints: ``(C, e)`` with ``C x = e``.
        bounds: A ``2 x d`` tensor of bounds.
        interior_point: A strictly feasible point (``d x 1``); found by a
            linear program if not given.
        n_burnin: The number of burn-in steps (on the first draw).
        n_thinning: The number of steps between consecutive samples.
        seed: The random seed.
    """

    def __init__(
        self,
        inequality_constraints: tuple[Any, Any] | None = None,
        equality_constraints: tuple[Any, Any] | None = None,
        bounds: Any = None,
        interior_point: Any = None,
        n_burnin: int = 200,
        n_thinning: int = 20,
        seed: int | None = None,
    ) -> None:
        def dense(c: tuple[Any, Any] | None, name: str) -> Any:
            if c is None:
                return None
            return (to_flat(c[0], name), to_flat(c[1], name))

        arrays = [c[0] for c in (inequality_constraints, equality_constraints) if c is not None]
        self._kind = first_kind(*arrays, bounds, interior_point)
        self.n_burnin = n_burnin
        self.n_thinning = n_thinning
        self._rs = _rs.HitAndRunPolytopeSampler(
            dense(inequality_constraints, "inequality_constraints"),
            dense(equality_constraints, "equality_constraints"),
            None if bounds is None else to_flat(bounds, "bounds"),
            None if interior_point is None else to_flat(interior_point, "interior_point"),
            int(n_burnin),
            int(n_thinning),
            _seed(seed),
        )

    def draw(self, n: int = 1) -> Any:
        """``n`` samples (``n x d``); consecutive draws continue the chain."""
        return from_pair(self._rs.draw(int(n)), self._kind)


def sample_hypersphere(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: int | None = None,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    r"""Sample uniformly from a unit d-sphere.

    Args:
        d: The dimension of the hypersphere.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. normal
            samples).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype: The torch (or NumPy) dtype; float64 torch tensors by default
            if torch has been imported, else lists.

    Returns:
        An ``n x d`` array of uniform samples from the d-hypersphere (``+-1``
        for ``d = 1``).
    """
    return from_pair(
        _rs.sample_hypersphere(int(d), int(n), bool(qmc), _seed(seed)),
        kind_from_dtype(dtype, device),
    )


def sample_simplex(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: int | None = None,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    r"""Sample uniformly from a d-simplex.

    Args:
        d: The dimension of the simplex.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype: The torch (or NumPy) dtype; float64 torch tensors by default
            if torch has been imported, else lists.

    Returns:
        An ``n x d`` array of uniform samples from the d-simplex.
    """
    return from_pair(
        _rs.sample_simplex(int(d), int(n), bool(qmc), _seed(seed)),
        kind_from_dtype(dtype, device),
    )


__all__ = [
    "HitAndRunPolytopeSampler",
    "draw_sobol_normal_samples",
    "draw_sobol_samples",
    "find_interior_point",
    "get_polytope_samples",
    "sample_polytope",
    "sparse_to_dense_constraints",
    "sample_hypersphere",
    "sample_simplex",
]
