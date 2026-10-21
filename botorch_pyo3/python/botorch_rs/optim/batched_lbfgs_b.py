"""Independent native L-BFGS-B states with batched objective evaluations.

NumPy is loaded only when this NumPy-facing API is used. Optimizer steps,
line searches, curvature updates, and inverse-Hessian products run in Rust.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from botorch_rs import _botorch_rs as _rs

status_messages = {
    0: "START",
    1: "NEW_X",
    2: "RESTART",
    3: "FG",
    4: "CONVERGENCE",
    5: "STOP",
    6: "WARNING",
    7: "ERROR",
    8: "ABNORMAL",
}
task_messages = {
    0: "",
    401: "NORM OF PROJECTED GRADIENT <= PGTOL",
    402: "RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    502: "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    504: "TOTAL NO. OF ITERATIONS REACHED LIMIT",
    505: "CALLBACK REQUESTED HALT",
}


class OptimState:
    """A suspended native optimization with independent line-search memory.

    ``bounds`` uses the reference state's ``2 x n`` lower/upper layout.
    Supply objective results with :meth:`provide_evaluation`, and resume
    accepted iterates with :meth:`resume`. ``copy`` preserves a suspended
    line search and all curvature pairs. SciPy's private work arrays are not
    exposed: the native state owns the actual solver workspace.
    """

    def __init__(self, bounds: Any, maxls: int, x0: Any, n: int, m: int) -> None:
        self._initialize(bounds, maxls, x0, n, m, {})

    def _initialize(
        self, bounds: Any, maxls: int, x0: Any, n: int, m: int, options: dict[str, Any]
    ) -> None:
        import numpy as np

        x = np.asarray(x0, dtype=np.float64)
        if x.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},), got {x.shape}")
        bound_array = (
            np.asarray(bounds, dtype=np.float64)
            if bounds is not None
            else np.stack((np.full(n, -np.inf), np.full(n, np.inf)))
        )
        if bound_array.shape != (2, n):
            raise ValueError(f"bounds must have shape (2, {n})")
        self._native = _rs._LbfgsbState(
            x.tolist(),
            bound_array[0].tolist(),
            bound_array[1].tolist(),
            maxcor=m,
            maxls=maxls,
            **options,
        )
        self._bounds = bound_array.copy()
        self._n, self._m = n, m
        self.state_str = None

    @property
    def x(self) -> Any:
        import numpy as np

        return np.asarray(self._native.x, dtype=np.float64)

    @property
    def f(self) -> Any:
        import numpy as np

        return np.asarray(self._native.f, dtype=np.float64)

    @property
    def g(self) -> Any:
        import numpy as np

        return np.asarray(self._native.g, dtype=np.float64)

    @property
    def task(self) -> Any:
        import numpy as np

        return np.asarray(self._native.task, dtype=np.int32)

    @property
    def n_iterations(self) -> int:
        return int(self._native.iterations)

    @property
    def fun_calls(self) -> int:
        return int(self._native.evaluations)

    @property
    def nbd(self) -> Any:
        import numpy as np

        lo, hi = np.isfinite(self._bounds)
        return np.where(lo, np.where(hi, 2, 1), np.where(hi, 3, 0)).astype(np.int32)

    @property
    def low_bnd(self) -> Any:
        import numpy as np

        return np.where(np.isfinite(self._bounds[0]), self._bounds[0], 0.0)

    @property
    def upper_bnd(self) -> Any:
        import numpy as np

        return np.where(np.isfinite(self._bounds[1]), self._bounds[1], 0.0)

    def provide_evaluation(self, value: float, gradient: Any) -> None:
        import numpy as np

        g = np.asarray(gradient, dtype=np.float64)
        if g.shape != (self._n,):
            raise ValueError(f"gradient must have shape ({self._n},)")
        self._native.provide_evaluation(float(value), g.tolist())

    def resume(self, stop: bool = False) -> None:
        self._native.resume(bool(stop))

    def copy(self) -> OptimState:
        result = object.__new__(OptimState)
        result._native = self._native.copy()
        result._bounds = self._bounds.copy()
        result._n, result._m = self._n, self._m
        result.state_str = self.state_str
        return result


class _InverseHessian:
    """Native two-loop inverse-Hessian operator, with the SciPy result API."""

    def __init__(self, state: OptimState) -> None:
        import numpy as np

        self._native = state._native.copy()
        self.shape = (state._n, state._n)
        self.dtype = np.dtype(np.float64)
        self.sk, self.yk = (
            np.asarray(x, dtype=np.float64).reshape(-1, state._n)
            for x in self._native.corrections()
        )
        self.n_corrs = len(self.sk)
        self.rho = 1.0 / np.einsum("ij,ij->i", self.sk, self.yk)

    def matvec(self, vector: Any) -> Any:
        import numpy as np

        v = np.asarray(vector, dtype=np.float64)
        if v.shape not in ((self.shape[0],), (self.shape[0], 1)):
            raise ValueError("inverse Hessian vector has the wrong shape")
        out = np.asarray(self._native.inverse_hessian_product(v.reshape(-1).tolist()))
        return out.reshape(v.shape)

    rmatvec = matvec

    def matmat(self, matrix: Any) -> Any:
        import numpy as np

        v = np.asarray(matrix, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != self.shape[1]:
            raise ValueError("inverse Hessian matrix has the wrong shape")
        return (
            np.stack([self.matvec(column) for column in v.T], axis=-1)
            if v.shape[1]
            else np.empty(v.shape)
        )

    def dot(self, value: Any) -> Any:
        import numpy as np

        return self.matvec(value) if np.asarray(value).ndim == 1 else self.matmat(value)

    __matmul__ = dot

    def todense(self) -> Any:
        import numpy as np

        return self.matmat(np.eye(self.shape[0]))

    @property
    def T(self) -> _InverseHessian:
        return self


def _inverse_hessian(state: OptimState) -> Any:
    from scipy.sparse.linalg import LinearOperator

    class NativeInverseHessian(LinearOperator):
        def __init__(self) -> None:
            self._operator = _InverseHessian(state)
            super().__init__(dtype=self._operator.dtype, shape=self._operator.shape)
            for name in ("sk", "yk", "n_corrs", "rho"):
                setattr(self, name, getattr(self._operator, name))

        def _matvec(self, vector: Any) -> Any:
            return self._operator.matvec(vector)

        def _rmatvec(self, vector: Any) -> Any:
            return self._operator.matvec(vector)

        def _matmat(self, matrix: Any) -> Any:
            return self._operator.matmat(matrix)

        def todense(self) -> Any:
            return self._operator.todense()

    return NativeInverseHessian()


def fmin_l_bfgs_b_batched(
    func: Any,
    x0: Any,
    bounds: Any = None,
    maxcor: int = 10,
    factr: float | None = 1e7,
    ftol: float | None = None,
    pgtol: float = 1e-5,
    tol: float | None = None,
    maxiter: int = 15000,
    disp: Any = None,
    callback: Any = None,
    maxls: int = 20,
    pass_batch_indices: bool = False,
) -> tuple[Any, Any, list[Any]]:
    """Minimize independent rows, grouping pending native objective requests.

    ``func`` returns values of shape ``(active_batch,)`` and gradients of
    shape ``(active_batch, n)``. With ``pass_batch_indices=True`` it also
    receives the original row indices. Callback ``StopIteration`` stops the
    corresponding problem. Results include native curvature-based Hessians.
    """
    import inspect

    import numpy as np
    from scipy.optimize import OptimizeResult

    if disp is not None:
        print("The option `disp` is deprecated and will be removed in a future.")
    if ftol is None:
        if factr is None:
            raise ValueError("factr must be specified when ftol is None")
        ftol = factr * np.finfo(float).eps
    elif factr is not None:
        raise AssertionError(
            "ftol and factr cannot be used together, set factr explicitly to None."
        )
    x = np.asarray(x0, dtype=np.float64)
    if x.ndim != 2:
        raise AssertionError("x0 must be a 2-D array")
    batch, n = x.shape
    if not batch or not n:
        raise ValueError("x0 must contain at least one nonempty problem")
    bound_array = None
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"length of x0 != length of bounds, {n} != len({bounds})")
        bound_array = np.asarray(
            [
                [
                    (-np.inf if pair[0] is None else pair[0]),
                    (np.inf if pair[1] is None else pair[1]),
                ]
                for pair in bounds
            ],
            dtype=np.float64,
        ).T
    options = {"ftol": ftol, "gtol": pgtol if tol is None else tol, "maxiter": maxiter}
    states = []
    for row in x:
        state = object.__new__(OptimState)
        state._initialize(bound_array, maxls, row, n, maxcor, options)
        states.append(state)
    intermediate_callback = callback is not None and set(
        inspect.signature(callback).parameters
    ) == {"intermediate_result"}
    while True:
        for state in states:
            if state.task[0] == 1:
                stop = False
                if callback is not None:
                    try:
                        if intermediate_callback:
                            callback(intermediate_result=OptimizeResult(x=state.x, fun=state.f))
                        else:
                            callback(state.x)
                    except StopIteration:
                        stop = True
                state.resume(stop)
        active = [i for i, state in enumerate(states) if state.task[0] == 3]
        if not active:
            break
        points = np.stack([states[i].x for i in active])
        result = func(points, batch_indices=active) if pass_batch_indices else func(points)
        values, gradients = (np.asarray(v, dtype=np.float64) for v in result)
        if values.shape != (len(active),) or gradients.shape != points.shape:
            raise ValueError(
                f"batched objective must return shapes {(len(active),)} and {points.shape}, got {values.shape} and {gradients.shape}"
            )
        for j, i in enumerate(active):
            states[i].provide_evaluation(values[j], gradients[j])
    results = []
    for state in states:
        code, detail = state.task
        status = 0 if code == 4 else (1 if detail in (502, 504) else 2)
        results.append(
            OptimizeResult(
                fun=float(state.f),
                jac=state.g,
                nfev=state.fun_calls,
                njev=None,
                nit=state.n_iterations,
                status=status,
                message=status_messages[code] + ": " + task_messages.get(detail, ""),
                x=state.x,
                success=status == 0,
                hess_inv=_inverse_hessian(state),
            )
        )
    return (
        np.stack([result.x for result in results]),
        np.asarray([result.fun for result in results]),
        results,
    )


def translate_bounds_for_lbfgsb(
    lower_bounds: Any, upper_bounds: Any, num_features: int, q: int
) -> list[tuple[Any, Any]]:
    """Expand scalar/per-feature bounds to the flattened ``q x d`` layout."""
    expanded = []
    for bounds in (lower_bounds, upper_bounds):
        if bounds is None:
            values = [None] * num_features
        elif isinstance(bounds, Iterable):
            values = list(bounds)
        else:
            values = [bounds] * num_features
        if len(values) == num_features:
            values *= q
        if len(values) != num_features * q:
            raise AssertionError(f"Instead got {len(values)} != {num_features} * {q}.")
        expanded.append(values)
    return list(zip(*expanded, strict=True))


__all__ = ["OptimState", "fmin_l_bfgs_b_batched", "translate_bounds_for_lbfgsb"]
