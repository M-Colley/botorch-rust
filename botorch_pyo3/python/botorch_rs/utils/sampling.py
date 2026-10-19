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

import importlib
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    first_kind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_from_dtype,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import BotorchError, InfeasibilityError


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
        out = data[:0]
        for i in range(n):
            for b in range(nb):
                src = (b * n + i) * row
                out.extend(data[src : src + row])
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


@contextmanager
def manual_seed(seed: int | None = None) -> Generator[None, None, None]:
    r"""Contextmanager for manual setting the torch.random seed (NumPy's
    global seed if torch has not been imported).

    Args:
        seed: The seed to set the random number generator to.

    Returns:
        Generator

    Example:
        >>> with manual_seed(1234):
        >>>     X = torch.rand(3)
    """
    torch = sys.modules.get("torch")
    if torch is None:
        np = sys.modules.get("numpy")
        old_np = np.random.get_state() if (np is not None and seed is not None) else None
        try:
            if seed is not None and np is not None:
                np.random.seed(seed)
            yield
        finally:
            if old_np is not None:
                np.random.set_state(old_np)
        return
    old_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.random.manual_seed(seed)
        yield
    finally:
        if seed is not None:
            torch.random.set_rng_state(old_state)


def _elementwise_sub(a: Any, b: Any) -> Any:
    return _ops.arith("sub", a, b)


def _elementwise_div(a: Any, b: Any) -> Any:
    return _ops.arith("div", a, b)


def _column(v: Any) -> Any:
    """A vector as a ``k x 1`` column."""
    if kind_of(v).lib == "list":
        data, shape = to_flat(v)
        return from_flat(data, (math.prod(shape), 1), kind_of(v))
    return v.reshape(-1, 1)


def _convert_bounds_to_inequality_constraints(bounds: Any) -> tuple[Any, Any]:
    r"""Convert bounds into inequality constraints of the form Ax <= b.

    Args:
        bounds: A ``2 x d``-dim tensor of bounds

    Returns:
        A two-element tuple containing
            - A: A ``2d x d``-dim tensor of coefficients
            - b: A ``2d x 1``-dim tensor containing the right hand side
    """
    data, shape = to_flat(bounds, "bounds")
    d = shape[-1]
    lower, upper = list(data[:d]), list(data[d:])
    rows: list[list[float]] = []
    rhs: list[float] = []
    for k in range(d):
        if math.isfinite(lower[k]):
            rows.append([-1.0 if j == k else 0.0 for j in range(d)])
            rhs.append(-lower[k])
    for k in range(d):
        if math.isfinite(upper[k]):
            rows.append([1.0 if j == k else 0.0 for j in range(d)])
            rhs.append(upper[k])
    kind = kind_of(bounds)
    A = from_flat([v for r in rows for v in r], (len(rows), d), kind)
    b = from_flat(rhs, (len(rhs), 1), kind)
    return A, b


def _null_space(C: Any) -> Any:
    """BoTorch's orthonormal null space ``Vh[rank:].T`` of ``C``: torch's
    SVD for torch tensors (so that the basis, which is not unique, is
    BoTorch's), Rust's SVD with a relative rank tolerance otherwise."""
    if is_torch_tensor(C):
        torch = sys.modules["torch"]
        _, S, Vh = torch.linalg.svd(C)
        r = torch.nonzero(S).size(0)
        return Vh[r:, :].transpose(-1, -2)
    return from_pair(_rs.null_space(to_flat(C, "C")), kind_of(C))


class PolytopeSampler(ABC):
    """Base class for samplers that sample points from a polytope."""

    def __init__(
        self,
        inequality_constraints: tuple[Any, Any] | None = None,
        equality_constraints: tuple[Any, Any] | None = None,
        bounds: Any = None,
        interior_point: Any = None,
    ) -> None:
        r"""
        Args:
            inequality_constraints: Tensors ``(A, b)`` describing inequality
                constraints ``A @ x <= b``, where ``A`` is a ``n_ineq_con x d``-dim
                Tensor and ``b`` is a ``n_ineq_con x 1``-dim Tensor, with ``n_ineq_con``
                the number of inequalities and ``d`` the dimension of the sample space.
            equality_constraints: Tensors ``(C, d)`` describing the equality
                constraints ``C @ x = d``, where ``C`` is a ``n_eq_con x d``-dim Tensor
                and ``d`` is a ``n_eq_con x 1``-dim Tensor with ``n_eq_con`` the number
                of equalities.
            bounds: A ``2 x d``-dim tensor of box bounds, where ``inf`` (``-inf``) means
                that the respective dimension is unbounded above (below).
            interior_point: A ``d x 1``-dim Tensor presenting a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.
        """
        self._setup(inequality_constraints, equality_constraints, bounds, interior_point)

    def _setup(
        self,
        inequality_constraints: tuple[Any, Any] | None,
        equality_constraints: tuple[Any, Any] | None,
        bounds: Any,
        interior_point: Any,
        find_point: bool = True,
    ) -> None:
        if inequality_constraints is None:
            if bounds is None:
                raise BotorchError(
                    "PolytopeSampler requires either inequality constraints or bounds."
                )
            d = _ops.shape_of(bounds)[-1]
            kind = kind_of(bounds)
            A = from_flat([], (0, d), kind)
            b = from_flat([], (0, 1), kind)
        else:
            A, b = inequality_constraints
        if bounds is not None:
            A2, b2 = _convert_bounds_to_inequality_constraints(bounds=bounds)
            kind = first_kind(A, b, bounds)
            A = _ops.concat([_ops._as_array(A, kind), _ops._as_array(A2, kind)], dim=0)
            b = _ops.concat([_ops._as_array(b, kind), _ops._as_array(b2, kind)], dim=0)
        self.A = A
        self.b = b
        self.equality_constraints = equality_constraints
        if equality_constraints is not None:
            self.C, self.d = equality_constraints
            self.nullC = _null_space(self.C)
        else:
            self.C = None
            self.d = None
            self.nullC = _ops.eye(_ops.shape_of(self.A)[-1], first_kind(self.A, self.b))
        self.new_A = _ops.matmul(self.A, self.nullC)
        if interior_point is not None:
            if self.feasible(interior_point):
                self.x0 = interior_point
            else:
                raise InfeasibilityError("The given input point is not feasible.")
        elif find_point:
            self.x0 = self.find_interior_point()

    def feasible(self, x: Any) -> Any:
        r"""Check whether a point is contained in the polytope.

        Args:
            x: A ``d x 1``-dim Tensor.

        Returns:
            True if ``x`` is contained inside the polytope (incl. its boundary),
            False otherwise (a boolean tensor for torch inputs).
        """
        ineq = _ops.arith("sub", _ops.matmul(self.A, x), self.b)
        ok = all(float(v) <= 0 for v in to_flat(ineq)[0])
        if self.equality_constraints is not None:
            eq = _ops.arith("sub", _ops.matmul(self.C, x), self.d)
            ok = ok and all(float(v) == 0 for v in to_flat(eq)[0])
        if is_torch_tensor(x):
            return sys.modules["torch"].tensor(ok)
        return ok

    def find_interior_point(self) -> Any:
        r"""Find an interior point of the polytope.

        Returns:
            A ``d x 1``-dim Tensor representing a point contained in the polytope.
            This function will raise a ValueError if there is no such point.
        """
        if self.equality_constraints:
            cdata, cshape = to_flat(self.C)
            d = cshape[-1]
            A_eq = from_flat(
                [
                    float(cdata[r * d + c]) if c < d else 0.0
                    for r in range(cshape[0])
                    for c in range(d + 1)
                ],
                (cshape[0], d + 1),
                kind_of(self.C),
            )
            b_eq = self.d
        else:
            A_eq = None
            b_eq = None
        x0 = find_interior_point(A=self.A, b=self.b, A_eq=A_eq, b_eq=b_eq)
        data, shape = to_flat(x0)
        return from_flat(data, (shape[0], 1), first_kind(self.A, self.b))

    # -------- Abstract methods to be implemented by subclasses -------- #

    @abstractmethod
    def draw(self, n: int = 1) -> Any:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.

        Returns:
            A ``n x d`` Tensor of samples from the polytope.
        """
        pass  # pragma: no cover


class _GenericPolytope(PolytopeSampler):
    """A concrete ``PolytopeSampler`` used internally for its constraint
    system."""

    def draw(self, n: int = 1) -> Any:  # pragma: no cover - not used
        raise NotImplementedError


class HitAndRunPolytopeSampler(PolytopeSampler):
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
        self._py_args = (inequality_constraints, equality_constraints, bounds, interior_point)
        self._py_polytope: PolytopeSampler | None = None

    def _polytope(self) -> PolytopeSampler:
        """BoTorch's normalized constraint system (``A``, ``b``, ``nullC``,
        ...; the unit cube if ``bounds`` were given), built on demand."""
        if self._py_polytope is None:
            ineq, eq, bounds, interior = self._py_args
            offset = None
            if (ineq or eq) and bounds is not None:
                if ineq:
                    ineq = normalize_dense_linear_constraints(bounds, ineq)
                if eq:
                    eq = normalize_dense_linear_constraints(bounds, eq)
                lower, upper = _ops.getitem(bounds, 0), _ops.getitem(bounds, 1)
                offset = lower
                if interior is not None:
                    interior = _elementwise_div(
                        _elementwise_sub(interior, _column(offset)),
                        _column(_elementwise_sub(upper, lower)),
                    )
                d = _ops.shape_of(bounds)[-1]
                kind = kind_of(bounds)
                bounds = from_flat([0.0] * d + [1.0] * d, (2, d), kind)
            sampler = _GenericPolytope.__new__(_GenericPolytope)
            sampler._setup(ineq, eq, bounds, interior, find_point=False)
            self._py_polytope = sampler
        return self._py_polytope

    def feasible(self, x: Any) -> Any:
        r"""Whether ``x`` (``d x 1``, in the normalized space of the sampler,
        as in BoTorch) is contained in the polytope (incl. its boundary)."""
        return self._polytope().feasible(x)

    def find_interior_point(self) -> Any:
        r"""An interior point (``d x 1``) of the (normalized) polytope from
        a linear program."""
        return self._polytope().find_interior_point()

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


def batched_multinomial(
    weights: Any,
    num_samples: int,
    replacement: bool = False,
    generator: Any = None,
    out: Any = None,
) -> Any:
    r"""Sample from multinomials with an arbitrary number of batch dimensions
    (BoTorch's ``batched_multinomial``): for every row of ``weights``
    (``batch_shape x num_categories``, non-negative, finite, not all zero),
    ``num_samples`` category indices. Torch tensors are sampled with
    ``torch.multinomial`` (and ``generator``), other arrays with Python's
    ``random``.

    Returns:
        A ``batch_shape x num_samples`` integer array.
    """
    from botorch_rs.generation.sampling import batched_multinomial_indices

    kind = first_kind(weights)
    if kind.lib == "torch":
        import torch

        w = weights.reshape(-1, weights.shape[-1])
        res = torch.multinomial(w, num_samples, replacement=replacement, generator=generator)
        res = res.view(*weights.shape[:-1], num_samples)
        if out is not None:
            out.copy_(res)
            return out
        return res
    data, shape = to_flat(weights, "weights")
    n = shape[-1]
    rows = len(data) // n if n else 0
    idcs: list[int] = []
    for r in range(rows):
        idcs.extend(
            batched_multinomial_indices(
                [float(v) for v in data[r * n : (r + 1) * n]], num_samples, replacement, kind
            )
        )
    out_shape = (*shape[:-1], num_samples)
    if kind.lib == "numpy":
        import numpy as np

        return np.asarray(idcs, dtype=np.int64).reshape(out_shape)
    values: Any = idcs
    for size in reversed(out_shape[1:]):
        values = [values[i : i + size] for i in range(0, len(values), size)]
    return values


def boltzmann_sample(
    function_values: Any,
    num_samples: int,
    eta: float,
    replacement: bool = False,
    temp_decrease: float = 0.5,
) -> Any:
    r"""Boltzmann sampling from a set of function values (BoTorch's
    ``boltzmann_sample``): indices drawn with weights ``exp(eta *
    standardize(values))``, ``eta`` multiplied by ``temp_decrease`` while a
    weight is infinite (one-dimensional ``function_values``).

    Returns:
        A ``num_samples`` integer array of sampled positions.
    """
    data, shape = to_flat(function_values, "function_values")
    if len(shape) != 1:
        raise NotImplementedError(
            "botorch_rs's boltzmann_sample supports one-dimensional function values."
        )
    weights, _ = _rs.boltzmann_weights_py(
        [float(v) for v in data], float(eta), float(temp_decrease)
    )
    kind = first_kind(function_values)
    w = from_flat(list(weights), (len(weights),), kind)
    return batched_multinomial(w, int(num_samples), replacement=replacement)


def sample_truncated_normal_perturbations(
    X: Any,
    n_discrete_points: int,
    sigma: float,
    bounds: Any,
    qmc: bool = True,
    *,
    seed: int | None = None,
) -> Any:
    r"""Sample points around ``X`` (BoTorch's
    ``sample_truncated_normal_perturbations``): randomly chosen points of
    ``X`` (``n x d``) perturbed by ``N(0, sigma^2 I)`` noise relative to the
    bounds, truncated to stay within the bounds (inverse-transform sampling
    of Sobol points if ``qmc``). ``seed`` (botorch_rs only) seeds the draws.

    Returns:
        An ``n_discrete_points x d`` array.
    """
    out = _rs.sample_truncated_normal_perturbations_py(
        to_flat(X, "X"),
        int(n_discrete_points),
        float(sigma),
        to_flat(bounds, "bounds"),
        bool(qmc),
        _seed(seed),
    )
    return from_pair(out, first_kind(X, bounds))


def sample_perturbed_subset_dims(
    X: Any,
    bounds: Any,
    n_discrete_points: int,
    sigma: float = 1e-1,
    qmc: bool = True,
    prob_perturb: float | None = None,
    *,
    seed: int | None = None,
) -> Any:
    r"""Sample around ``X`` by perturbing a subset of the dimensions
    (BoTorch's ``sample_perturbed_subset_dims``): every dimension of a
    randomly chosen point of ``X`` is replaced by a truncated normal
    perturbation of itself with probability ``prob_perturb`` (default
    ``min(20 / d, 1)``); points without a perturbed dimension get ``ceil(d *
    prob_perturb)`` random dimensions perturbed.

    Unlike BoTorch, every point is perturbed around itself (BoTorch draws the
    perturbations around a second, independently chosen point of ``X``).

    Returns:
        An ``n_discrete_points x d`` array.
    """
    bshape = to_flat(bounds, "bounds")[1]
    xshape = to_flat(X, "X")[1]
    if len(bshape) != 2:
        from botorch_rs.exceptions.errors import BotorchTensorDimensionError

        raise BotorchTensorDimensionError("bounds must be a `2 x d`-dim tensor.")
    if len(xshape) != 2:
        from botorch_rs.exceptions.errors import BotorchTensorDimensionError

        raise BotorchTensorDimensionError("X must be a `n x d`-dim tensor.")
    out = _rs.sample_perturbed_subset_dims_py(
        to_flat(X, "X"),
        to_flat(bounds, "bounds"),
        int(n_discrete_points),
        float(sigma),
        bool(qmc),
        None if prob_perturb is None else float(prob_perturb),
        _seed(seed),
    )
    return from_pair(out, first_kind(X, bounds))


def normalize_sparse_linear_constraints(
    bounds: Any, constraints: list[tuple[Any, Any, float]]
) -> list[tuple[Any, Any, float]]:
    r"""Normalize sparse linear constraints to the unit cube.

    Args:
        bounds: A ``2 x d``-dim tensor containing the box bounds.
        constraints: A list of tuples (``indices``, ``coefficients``, ``rhs``), with
            ``indices`` and ``coefficients`` one-dimensional tensors and ``rhs`` a
            scalar, where each tuple encodes an inequality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) >= rhs`` or
            ``\sum_i (X[indices[i]] * coefficients[i]) = rhs``.
    """
    bdata, bshape = to_flat(bounds, "bounds")
    d = bshape[-1]
    new_constraints = []
    for index, coefficient, rhs in constraints:
        if len(_ops.shape_of(index)) != 1:
            raise ValueError(
                "`indices` must be a one-dimensional tensor. This method does not "
                "support the kind of 'inter-point constraints' that are supported by "
                "`optimize_acqf()`. To achieve this behavior, you need define the "
                "problem on the joint space over `q` points and impose use constraints,"
                "see https://github.com/meta-pytorch/botorch/issues/2468#issuecomment-2287706461"  # noqa: E501
            )
        idx = _ops.ints(index)
        coef = [float(v) for v in to_flat(coefficient, "coefficients")[0]]
        lower = [float(bdata[i]) for i in idx]
        upper = [float(bdata[d + i]) for i in idx]
        s = [u - lo for u, lo in zip(upper, lower, strict=True)]
        new_coef = [si * ci for si, ci in zip(s, coef, strict=True)]
        dot = math.fsum(c * lo for c, lo in zip(coef, lower, strict=True))
        kind = first_kind(coefficient, bounds)
        new_constraints.append(
            (index, from_flat(new_coef, (len(new_coef),), kind), float(rhs) - dot)
        )
    return new_constraints


def normalize_dense_linear_constraints(
    bounds: Any,
    constraints: tuple[Any, Any],
) -> tuple[Any, Any]:
    r"""Normalize dense linear constraints to the unit cube.

    Args:
        bounds: A ``2 x d``-dim tensor containing the box bounds.
        constraints: A tensor tuple ``(A, b)`` describing constraints
            ``A @ x (<)= b``, where ``A`` is a ``n_con x d``-dim Tensor and
            ``b`` is a ``n_con x 1``-dim Tensor, with ``n_con`` the number of
            constraints and ``d`` the dimension of the sample space.

    Returns:
        A tensor tuple ``(A_nlz, b_nlz)`` of normalized constraints.
    """
    lower, upper = _ops.getitem(bounds, 0), _ops.getitem(bounds, 1)
    A, b = constraints
    A_nlz = _ops.arith("mul", _ops.arith("sub", upper, lower), A)
    Al = _ops.matmul(A, _column(lower))
    b_nlz = _ops.arith("sub", b, Al)
    return A_nlz, b_nlz


class DelaunayPolytopeSampler(PolytopeSampler):
    r"""A polytope sampler using Delaunay triangulation.

    This sampler first enumerates the vertices of the constraint polytope and
    then uses a Delaunay triangulation to tesselate its convex hull.

    The sampling happens in two stages:
    1. First, we sample from the set of hypertriangles generated by the
    Delaunay triangulation (i.e. which hyper-triangle to draw the sample
    from) with probabilities proportional to the triangle volumes.
    2. Then, we sample uniformly from the chosen hypertriangle by sampling
    uniformly from the unit simplex of the appropriate dimension, and
    then computing the convex combination of the vertices of the
    hypertriangle according to that draw from the simplex.

    The best reference (not exactly the same, but functionally equivalent) is
    [Trikalinos2014polytope]_. A simple R implementation is available at
    https://github.com/gertvv/tesselample.

    The vertex enumeration and the triangulation use SciPy's Qhull
    (``HalfspaceIntersection``, ``Delaunay``), as in BoTorch, so SciPy is
    required. Draws use torch's generator for torch tensors (BoTorch's
    draws for the same seed: ``torch.multinomial`` and ``sample_simplex``)
    and NumPy's (or Python's) otherwise.
    """

    def __init__(
        self,
        inequality_constraints: tuple[Any, Any] | None = None,
        equality_constraints: tuple[Any, Any] | None = None,
        bounds: Any = None,
        interior_point: Any = None,
    ) -> None:
        r"""Initialize DelaunayPolytopeSampler.

        Args:
            inequality_constraints: Tensors ``(A, b)`` describing inequality
                constraints ``A @ x <= b``, where ``A`` is a ``n_ineq_con x d``-dim
                Tensor and ``b`` is a ``n_ineq_con x 1``-dim Tensor, with ``n_ineq_con``
                the number of inequalities and ``d`` the dimension of the sample space.
            equality_constraints: Tensors ``(C, d)`` describing the equality
                constraints ``C @ x = d``, where ``C`` is a ``n_eq_con x d``-dim Tensor
                and ``d`` is a ``n_eq_con x 1``-dim Tensor with ``n_eq_con`` the number
                of equalities.
            bounds: A ``2 x d``-dim tensor of box bounds, where ``inf`` (``-inf``) means
                that the respective dimension is unbounded from above (below).
            interior_point: A ``d x 1``-dim Tensor representing a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.

        Warning: The vertex enumeration performed in this algorithm can become
        extremely costly if there are a large number of inequalities. Similarly,
        the triangulation can get very expensive in high dimensions. Only use
        this algorithm for moderate dimensions / moderately complex constraint sets.
        An alternative is the ``HitAndRunPolytopeSampler``.
        """
        super().__init__(
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            bounds=bounds,
            interior_point=interior_point,
        )
        np = importlib.import_module("numpy")
        new_b = _ops.arith("sub", self.b, _ops.matmul(self.A, self.x0))
        A_np = np.asarray(to_flat(self.new_A)[0], dtype=np.float64).reshape(
            _ops.shape_of(self.new_A)
        )
        b_np = np.asarray(to_flat(new_b)[0], dtype=np.float64).reshape(-1, 1)
        kind = kind_of(self.A)
        if A_np.shape[-1] < 2:
            # if the polytope is in dim 1 (i.e. a line segment) Qhull won't work
            tshlds = b_np / A_np
            neg = A_np < 0
            self.y_min = float(tshlds[neg].max())
            self.y_max = float(tshlds[~neg].min())
            self.dim = 1
        else:
            spatial = importlib.import_module("scipy.spatial")
            # Qhull expects inputs of the form A @ x + b <= 0, so we need to negate here
            halfspaces = np.concatenate([A_np, -b_np], axis=-1)
            vertices = spatial.HalfspaceIntersection(
                halfspaces=halfspaces, interior_point=np.zeros(A_np.shape[-1])
            ).intersections
            self.dim = vertices.shape[-1]
            try:
                delaunay = spatial.Delaunay(vertices)
            except ValueError as e:
                if "Points cannot contain NaN" in str(e):
                    raise ValueError("Polytope is unbounded.") from None
                raise e  # pragma: no cover
            polytopes = np.array([delaunay.points[s] for s in delaunay.simplices])
            volumes = np.array([abs(np.linalg.det(p[1:] - p[0])) for p in polytopes])
            self._polytopes = from_flat(
                polytopes.reshape(-1).tolist(), polytopes.shape, kind
            )
            self._p = from_flat((volumes / volumes.sum()).tolist(), volumes.shape, kind)
            self._polytopes_np = polytopes
            self._p_np = volumes / volumes.sum()

    def draw(self, n: int = 1, seed: int | None = None) -> Any:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.
            seed: The random seed.

        Returns:
            A ``n x d`` Tensor of samples from the polytope.
        """
        kind = kind_of(self.A)
        if kind.lib == "torch":
            torch = sys.modules["torch"]
            if self.dim == 1:
                with manual_seed(seed):
                    e = torch.rand(n, 1, device=self.new_A.device, dtype=self.new_A.dtype)
                transformed = self.y_min + (self.y_max - self.y_min) * e
            else:
                if seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=self.A.device)
                    generator.manual_seed(seed)
                index_rvs = torch.multinomial(
                    self._p, num_samples=n, replacement=True, generator=generator
                )
                with manual_seed(seed):
                    rnd = torch.rand(n, self.dim, dtype=self.A.dtype)
                srnd, _ = torch.sort(rnd, dim=-1)
                zeros = torch.zeros(n, 1, dtype=self.A.dtype)
                ones = torch.ones(n, 1, dtype=self.A.dtype)
                srnd = torch.cat([zeros, srnd, ones], dim=-1).to(self.A.device)
                simplex_rvs = srnd[..., 1:] - srnd[..., :-1]
                transformed = torch.stack(
                    [rv @ self._polytopes[idx] for rv, idx in zip(simplex_rvs, index_rvs)]
                )
            init_shift = self.x0.transpose(-1, -2)
            return init_shift + transformed @ self.nullC.transpose(-1, -2)
        import random

        np = sys.modules.get("numpy")
        rng = np.random.default_rng(seed) if np is not None else random.Random(seed)
        if self.dim == 1:
            e = [[float(rng.random())] for _ in range(n)]
            rows = [[self.y_min + (self.y_max - self.y_min) * v[0]] for v in e]
        else:
            p = list(self._p_np)
            cum = [math.fsum(p[: i + 1]) for i in range(len(p))]
            rows = []
            for _ in range(n):
                u = float(rng.random()) * cum[-1]
                idx = next(i for i, c in enumerate(cum) if c >= u)
                cuts = sorted(float(rng.random()) for _ in range(self.dim))
                cuts = [0.0, *cuts, 1.0]
                w = [cuts[k + 1] - cuts[k] for k in range(self.dim + 1)]
                verts = self._polytopes_np[idx]
                rows.append(
                    [
                        math.fsum(w[k] * float(verts[k][j]) for k in range(self.dim + 1))
                        for j in range(self.dim)
                    ]
                )
        transformed = from_flat([v for r in rows for v in r], (n, self.dim), kind)
        shifted = _ops.matmul(transformed, _ops.transpose(self.nullC))
        return _ops.arith("add", _ops.transpose(self.x0), shifted)


def _joint_lbfgsb(X0: Any, func: Callable[[Any], Any], bounds: Any) -> tuple[Any, Any]:
    """Maximizes ``func(X).sum()`` over ``X`` (``batch x d``, within
    ``bounds``) with SciPy's L-BFGS-B and torch gradients; returns ``X`` and
    ``func(X)``."""
    import numpy as np
    from scipy.optimize import minimize

    torch = sys.modules["torch"]
    shape = X0.shape
    lower = bounds[0].expand(shape).reshape(-1).detach().cpu().double().numpy()
    upper = bounds[1].expand(shape).reshape(-1).detach().cpu().double().numpy()
    x0 = X0.detach().cpu().double().reshape(-1).numpy().clip(lower, upper)

    def f_np(x: Any) -> tuple[float, Any]:
        X = torch.from_numpy(x).to(X0).view(shape).contiguous().requires_grad_(True)
        loss = -func(X).sum()
        (grad,) = torch.autograd.grad(loss, X)
        return float(loss.item()), grad.reshape(-1).detach().cpu().double().numpy()

    res = minimize(
        f_np,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": 2000},
    )
    X = torch.from_numpy(np.asarray(res.x)).to(X0).view(shape)
    X = torch.minimum(torch.maximum(X, bounds[0]), bounds[1])
    with torch.no_grad():
        values = func(X)
    return X, values


def optimize_posterior_samples(
    paths: Any,
    bounds: Any,
    raw_samples: int = 1024,
    num_restarts: int = 20,
    sample_transform: Callable[[Any], Any] | None = None,
    return_transformed: bool = False,
) -> tuple[Any, Any]:
    r"""Cheaply maximizes posterior samples by random querying followed by
    gradient-based optimization using SciPy's L-BFGS-B routine.

    Args:
        paths: Random Fourier Feature-based sample paths from the GP
        bounds: The bounds on the search space.
        raw_samples: The number of samples with which to query the samples initially.
        num_restarts: The number of points selected for gradient-based optimization.
        sample_transform: A callable transform of the sample outputs (e.g.
            MCAcquisitionObjective or ScalarizedPosteriorTransform.evaluate) used to
            negate the objective or otherwise transform the output.
        return_transformed: A boolean indicating whether to return the transformed
            or non-transformed samples.

    Returns:
        A two-element tuple containing:
            - X_opt: A ``num_optima x [batch_size] x d``-dim tensor of optimal
              inputs x*.
            - f_opt: A ``num_optima x [batch_size] x m``-dim, optionally
                ``num_optima x [batch_size] x 1``-dim,  tensor of optimal outputs f*.

    The candidates are scrambled Sobol points of torch's ``SobolEngine``
    (seeded from torch's global generator, as in BoTorch) for torch bounds,
    else of botorch_rs's Sobol sequence; the restarts are optimized with
    SciPy's L-BFGS-B as one joint problem (BoTorch's ``gen_candidates_scipy``
    with ``use_parallel_mode=False``: ``maxiter=2000``, the summed values).
    ``paths`` may be any callable (botorch_rs or BoTorch sample paths, a
    deterministic model).
    """
    def path_func(x: Any) -> Any:
        res = paths(x)
        if sample_transform:
            res = sample_transform(res)
        return res.squeeze(-1)

    kind = kind_of(bounds)
    if kind.lib != "torch":
        raise NotImplementedError(
            "optimize_posterior_samples operates on torch tensors (the sample paths are "
            "evaluated and differentiated through torch)."
        )
    torch = sys.modules["torch"]
    d = bounds.shape[1]
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True).draw(n=raw_samples)
    lower, upper = bounds[0], bounds[1]
    candidate_set = sobol.to(bounds) * (upper - lower) + lower
    candidate_queries = path_func(candidate_set)
    argtop_k = torch.topk(candidate_queries, num_restarts, dim=-1).indices
    X_top_k = candidate_set[argtop_k, :]
    # One joint L-BFGS-B problem over all restarts (the leading dimensions
    # of the restarts pair them with the sample paths, so the paths are
    # evaluated on all restarts at once, as BoTorch's ``gen_candidates_scipy``
    # does without the parallel mode).
    X_top_k, f_top_k = _joint_lbfgsb(X_top_k, path_func, bounds)
    f_opt, arg_opt = f_top_k.max(dim=-1, keepdim=True)
    final_shape = candidate_queries.shape[:-1]
    X_opt = X_top_k.reshape(final_shape.numel(), num_restarts, -1)[
        torch.arange(final_shape.numel()), arg_opt.flatten()
    ].reshape(*final_shape, -1)
    if return_transformed:
        return X_opt, f_opt
    f_opt = paths(X_opt.unsqueeze(-2)).squeeze(-2)
    return X_opt, f_opt


__all__ = [
    "DelaunayPolytopeSampler",
    "HitAndRunPolytopeSampler",
    "PolytopeSampler",
    "manual_seed",
    "normalize_dense_linear_constraints",
    "normalize_sparse_linear_constraints",
    "optimize_posterior_samples",
    "batched_multinomial",
    "boltzmann_sample",
    "sample_perturbed_subset_dims",
    "sample_truncated_normal_perturbations",
    "draw_sobol_normal_samples",
    "draw_sobol_samples",
    "find_interior_point",
    "get_polytope_samples",
    "sample_polytope",
    "sparse_to_dense_constraints",
    "sample_hypersphere",
    "sample_simplex",
]
