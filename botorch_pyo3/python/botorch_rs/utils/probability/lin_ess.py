r"""Linear Elliptical Slice Sampler
(``botorch.utils.probability.lin_ess``).

References

.. [Gessner2020]
    A. Gessner, O. Kanjilal, and P. Hennig. Integrals over gaussians under
    linear domain constraints. AISTATS 2020.

.. [Wu2024]
    K. Wu, and J. Gardner. A Fast, Robust Elliptical Slice Sampling Implementation for
    Linearly Truncated Multivariate Normal Distributions. arXiv:2407.10449. 2024.

The Markov chains run in Rust. The random numbers are drawn in Python in
BoTorch's order, one step at a time (``torch.randn_like(z)`` for the
direction, then ``torch.rand(num_chains)`` for the angle): with torch
tensors from torch's global generator, so that ``torch.manual_seed`` gives
BoTorch's chains (given the same interior point); with NumPy arrays from
NumPy's global generator and with nested lists from Python's ``random``.
The interior point, if not given, comes from botorch_rs's linear program
(BoTorch uses SciPy's HiGHS, whose solution can differ when the optimal
slack is attained by several points).
"""

from __future__ import annotations

import math
import random
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.utils.sampling import PolytopeSampler

_twopi = 2.0 * math.pi


def _is_diag_operator(x: Any) -> bool:
    return type(x).__name__ == "DiagLinearOperator"


class LinearEllipticalSliceSampler(PolytopeSampler):
    r"""Linear Elliptical Slice Sampler.

    Ideas:
    - Optimize computations if possible, potentially with torch.compile.
    - Extend fixed features constraint to general linear equality constraints.
    """

    def __init__(
        self,
        inequality_constraints: tuple[Any, Any] | None = None,
        bounds: Any = None,
        interior_point: Any = None,
        fixed_indices: Any = None,
        mean: Any = None,
        covariance_matrix: Any = None,
        covariance_root: Any = None,
        check_feasibility: bool = False,
        burnin: int = 0,
        thinning: int = 0,
        num_chains: int = 1,
    ) -> None:
        r"""Initialize LinearEllipticalSliceSampler.

        Args:
            inequality_constraints: Tensors ``(A, b)`` describing inequality
                 constraints ``A @ x <= b``, where ``A`` is an
                 ``n_ineq_con x d``-dim Tensor and ``b`` is an
                 ``n_ineq_con x 1``-dim Tensor, with ``n_ineq_con`` the number of
                 inequalities and ``d`` the dimension of the sample space. If omitted,
                 must provide ``bounds`` instead.
            bounds: A ``2 x d``-dim tensor of box bounds. If omitted, must provide
                ``inequality_constraints`` instead.
            interior_point: A ``d x 1``-dim Tensor presenting a point in the (relative)
                interior of the polytope. If omitted, an interior point is determined
                automatically by solving a Linear Program. Note: It is crucial that
                the point lie in the interior of the feasible set (rather than on the
                boundary), otherwise the sampler will produce invalid samples.
            fixed_indices: Integer list or ``d``-dim Tensor representing the indices
                of dimensions that are constrained to be fixed to the values specified
                in the ``interior_point``, which is required to be passed in
                conjunction with ``fixed_indices``.
            mean: The ``d x 1``-dim mean of the MVN distribution (if omitted, use zero).
            covariance_matrix: The ``d x d``-dim covariance matrix of the MVN
                distribution (if omitted, use the identity).
            covariance_root: A ``d x d``-dim root of the covariance matrix such that
                covariance_root @ covariance_root.T = covariance_matrix. NOTE: This
                matrix is assumed to be lower triangular. covariance_root can only be
                passed in conjunction with fixed_indices if covariance_root is a
                DiagLinearOperator. Otherwise the factorization would need to be re-
                computed, as we need to solve in ``standardize``.
            check_feasibility: If True, raise an error if the sampling results in an
                infeasible sample. This creates some overhead and so is switched off
                by default.
            burnin: Number of samples to generate upon initialization to warm up the
                sampler.
            thinning: Number of samples to skip before returning a sample in ``draw``.
            num_chains: Number of Markov chains to run in parallel.

        This sampler samples from a multivariate Normal ``N(mean, covariance_matrix)``
        subject to linear domain constraints ``A x <= b`` (intersected with box bounds,
        if provided).
        """
        if interior_point is not None and len(_ops.shape_of(interior_point)) == 1:
            interior_point = _column(interior_point)
        if mean is not None and len(_ops.shape_of(mean)) == 1:
            mean = _column(mean)
        super().__init__(
            inequality_constraints=inequality_constraints,
            interior_point=interior_point,
            bounds=bounds,
        )
        if covariance_matrix is not None and covariance_root is not None:
            raise ValueError("Provide either covariance_matrix or covariance_root, not both.")
        if fixed_indices is not None and interior_point is None:
            raise ValueError(
                "If `fixed_indices` are provided, an interior point must also be "
                "provided in order to infer feasible values of the fixed features."
            )
        self._kind = first_kind(self.A, self.b, self.x0)
        d = _ops.shape_of(self.A)[-1]
        root, root_diag = None, None
        if covariance_root is not None:
            if _is_diag_operator(covariance_root):
                root_diag = to_flat(covariance_root.diagonal())[0]
            else:
                root = to_flat(covariance_root, "covariance_root")
        fixed = None
        if fixed_indices is not None:
            fixed = sorted({int(i) % d for i in _ops.ints(fixed_indices)})
            if root is not None:
                rdata, rshape = root
                n = rshape[0]
                is_diag = all(
                    rdata[r * n + c] == 0 for r in range(n) for c in range(n) if r != c
                )
                if not is_diag:
                    raise ValueError("Provide either covariance_root or fixed_indices, not both.")
                root_diag = [rdata[r * n + r] for r in range(n)]
                root = None
        self._fixed_indices = fixed
        self._is_fixed, self._not_fixed = (
            get_index_tensors(fixed_indices=fixed, d=d) if fixed is not None else (None, None)
        )
        self._mean = mean
        single = _ops.is_single_precision(self._kind)
        self._rs = _rs.LinearEllipticalSliceSampler(
            to_flat(self.A, "A"),
            to_flat(self.b, "b")[0],
            to_flat(self.x0, "interior_point")[0],
            fixed,
            None if mean is None else to_flat(mean, "mean")[0],
            None if covariance_matrix is None else to_flat(covariance_matrix, "covariance_matrix"),
            root,
            root_diag,
            int(num_chains),
            single,
        )
        self._covariance_root = covariance_root
        self.num_chains = int(num_chains)
        self.check_feasibility = check_feasibility
        self.thinning = 0
        if burnin > 0:
            self.draw(burnin)
        self.thinning = thinning

    # -- state ---------------------------------------------------------------

    @property
    def lifetime_samples(self) -> int:
        """The total number of samples generated by the sampler during its lifetime."""
        return self._rs.lifetime_samples

    @property
    def _x(self) -> Any:
        """The current points (``d x num_chains``)."""
        return from_flat(*self._rs.x, self._kind)

    @property
    def _z(self) -> Any:
        """The current standardized points (``k x num_chains``)."""
        return from_flat(*self._rs.z, self._kind)

    @property
    def _Az(self) -> Any:
        return from_flat(*self._rs.standardized[0], self._kind)

    @property
    def _bz(self) -> Any:
        return from_flat(*self._rs.standardized[1], self._kind)

    # -- sampling ------------------------------------------------------------

    def _draws(self, steps: int) -> tuple[list[float], list[float]]:
        """The random numbers of ``steps`` steps in BoTorch's order."""
        k, c = self._rs.draw_shape
        nu: list[float] = []
        u: list[float] = []
        kind = self._kind
        if kind.lib == "torch":
            torch = sys.modules["torch"]
            dtype = kind.dtype or torch.float64
            for _ in range(steps):
                nu.extend(torch.randn(k, c, dtype=dtype).reshape(-1).tolist())
                u.extend(torch.rand(c, dtype=dtype).tolist())
            return nu, u
        np = sys.modules.get("numpy") if kind.lib == "numpy" else None
        for _ in range(steps):
            if np is not None:
                nu.extend(np.random.standard_normal(k * c).tolist())
                u.extend(np.random.random_sample(c).tolist())
            else:
                nu.extend(random.gauss(0.0, 1.0) for _ in range(k * c))
                u.extend(random.random() for _ in range(c))
        return nu, u

    def draw(self, n: int = 1) -> Any:
        r"""Draw samples.

        Args:
            n: The number of samples.

        Returns:
            A ``(n * num_chains) x d``-dim tensor of ``n * num_chains`` samples.
        """
        keep = self.thinning + 1
        steps = n * keep
        nu, u = self._draws(steps)
        try:
            data, (kept, d, c) = self._rs.run(nu, u, steps, keep, bool(self.check_feasibility))
        except RuntimeError as e:
            raise RuntimeError(str(e)) from None
        # (kept, d, c) -> cat(dim=-1) -> d x (kept c) -> transpose.
        out = [data[(s * d + j) * c + ch] for s in range(kept) for ch in range(c) for j in range(d)]
        return from_flat(out, (kept * c, d), self._kind)

    def step(self) -> Any:
        r"""Take a step, return the new sample, update the internal state.

        Returns:
            A ``d x num_chains``-dim tensor, where each column is a sample from a Markov
            chain.
        """
        nu, u = self._draws(1)
        try:
            data, (_, d, c) = self._rs.run(nu, u, 1, 1, bool(self.check_feasibility))
        except RuntimeError as e:
            raise RuntimeError(str(e)) from None
        return from_flat(data, (d, c), self._kind)

    def _is_feasible(self, points: Any, transformed: bool = False) -> Any:
        r"""Returns a Boolean tensor indicating whether the ``points`` are feasible,
        i.e. they satisfy ``A @ points <= b``, where ``(A, b)`` are the tensors passed
        as the ``inequality_constraints`` to the constructor of the sampler.

        Args:
            points: A ``d x M``-dim tensor of points.
            transformed: Wether points are assumed to be transformed by a change of
                basis, which means feasibility should be computed based on the
                transformed constraint system (_Az, _bz), instead of (A, b).

        Returns:
            An ``M``-dim binary tensor where ``True`` indicates that the associated
            point is feasible.
        """
        from botorch_rs._arrays import bool_array

        flags = self._rs.is_feasible(to_flat(points, "points"), transformed)
        return bool_array(flags, (len(flags),), kind_of(points))


def _column(v: Any) -> Any:
    if is_torch_tensor(v) or kind_of(v).lib == "numpy":
        return v.reshape(-1, 1)
    data, shape = to_flat(v)
    return from_flat(data, (math.prod(shape), 1), kind_of(v))


def get_index_tensors(fixed_indices: Any, d: int) -> tuple[Any, Any]:
    """Converts ``fixed_indices`` to a ``d``-dim integral Tensor that is True at indices
    that are contained in ``fixed_indices`` and False otherwise.

    Args:
        fixed_indices: A list or Tensor of integer indices to fix.
        d: The dimensionality of the Tensors to be indexed.

    Returns:
        A Tuple of integral Tensors partitioning [1, d] into indices that are fixed
        (first tensor) and non-fixed (second tensor).

    The results are torch ``long`` tensors for tensor inputs or if torch has
    been imported (as in BoTorch), NumPy arrays for NumPy inputs, else lists.
    """
    is_fixed = _ops.ints(fixed_indices)
    not_fixed = [i for i in range(d) if i not in is_fixed]
    kind = kind_of(fixed_indices)
    if kind.lib == "list" and sys.modules.get("torch") is not None:
        from botorch_rs._arrays import kind_from_dtype

        kind = kind_from_dtype(None)
    return (
        _ops.int_array(is_fixed, (len(is_fixed),), kind),
        _ops.int_array(not_fixed, (len(not_fixed),), kind),
    )


__all__ = ["LinearEllipticalSliceSampler", "get_index_tensors"]
