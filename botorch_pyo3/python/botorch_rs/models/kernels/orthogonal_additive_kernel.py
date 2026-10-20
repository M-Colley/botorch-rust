"""``OrthogonalAdditiveKernel``
(``botorch.models.kernels.orthogonal_additive_kernel``)."""

from __future__ import annotations

import struct
import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_shapes, from_flat, kind_from_dtype, kind_of, to_flat
from botorch_rs.constraints import Interval, Positive
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import (
    Kernel,
    MaternKernel,
    RBFKernel,
    _batch_matrix,
    _check_batch_shape,
    _check_prior,
    _flat_index,
    _indices,
    _properties,
)
from botorch_rs.priors import Prior

_positivity_constraint = Positive()
SECOND_ORDER_PRIOR_ERROR_MSG = (
    "Second order interactions are disabled, but there is a prior on the second order "
    "coefficients. Please remove the second order prior or enable second order terms."
)


def _f32(x: float) -> float:
    return float(struct.unpack("f", struct.pack("f", x))[0])


class OrthogonalAdditiveKernel(Kernel):
    r"""Orthogonal additive kernel (Lu et al., 2022) with a stationary base
    kernel orthogonalized by Gauss-Legendre quadrature on ``[0, 1]``:

    ``K = offset + sum_i c1_i K_i + sum_{i < j} c2_ij K_i K_j``, with ``K_i``
    the orthogonalized base kernel of input ``i``.

    The base kernel must be an ``RBFKernel`` or ``MaternKernel`` with one
    lengthscale, with ``batch_shape = batch_shape + [dim]`` (one lengthscale
    per input) or ``batch_shape`` (shared). The coefficients are transformed
    by ``coeff_constraint``, which BoTorch does not register (no optimizer
    bounds).

    Deviation: a prior on the second-order coefficients applies to the
    ``dim * (dim - 1) / 2`` coefficients; BoTorch evaluates it on the full
    ``dim x dim`` matrix including the zeros below and on its diagonal (a
    ``-inf`` log density for many priors).
    """

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        from botorch_rs.models.map_saas import _use_torch_backend

        device = kwargs.get("device", args[7] if len(args) > 7 else None)
        if cls is OrthogonalAdditiveKernel and (
            _use_torch_backend(backend)
            or (backend == "auto" and device is not None and str(device) != "cpu")
        ):
            from botorch_rs.models._torch_backend_additive import TorchOrthogonalAdditiveKernel

            return object.__new__(TorchOrthogonalAdditiveKernel)
        return object.__new__(cls)

    def __init__(
        self,
        dim: int,
        base_kernel: Kernel | None = None,
        per_dim_lengthscales: bool = True,
        quad_deg: int = 32,
        second_order: bool = False,
        batch_shape: Any = None,
        dtype: Any = None,
        device: Any = None,
        coeff_constraint: Interval = _positivity_constraint,
        offset_prior: Prior | None = None,
        coeffs_1_prior: Prior | None = None,
        coeffs_2_prior: Prior | None = None,
        check_hypercube: bool = True,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__(batch_shape=batch_shape)
        self.dim = int(dim)
        self.check_hypercube = bool(check_hypercube)
        self.per_dim_lengthscales = per_dim_lengthscales
        batch = self._batch_shape
        expected = (*batch, self.dim) if per_dim_lengthscales else batch
        if base_kernel is None:
            base_kernel = RBFKernel(batch_shape=expected)
        else:
            if tuple(base_kernel.batch_shape) != expected:
                warnings.warn(
                    f"The provided base_kernel has batch_shape={base_kernel.batch_shape}, but "
                    f"the expected batch_shape is {expected} ({per_dim_lengthscales=}, "
                    f"batch_shape={batch}, {dim=}). Consider adjusting the base kernel's "
                    "batch_shape or setting per_dim_lengthscales=False.",
                    stacklevel=2,
                )
        if not isinstance(base_kernel, (RBFKernel, MaternKernel)):
            raise UnsupportedError(
                "botorch_rs supports an RBFKernel or MaternKernel base kernel of an "
                "OrthogonalAdditiveKernel."
            )
        base_batch = tuple(base_kernel.batch_shape)
        if base_batch == (*batch, self.dim):
            self._per_dim = True
        elif base_batch in (batch, (*batch, 1)):
            self._per_dim = False
        else:
            raise UnsupportedError(
                f"The base kernel's batch_shape {base_batch} must be {(*batch, self.dim)} (one "
                f"lengthscale per input) or {batch} (a shared lengthscale)."
            )
        self.base_kernel = base_kernel
        if not second_order and coeffs_2_prior is not None:
            raise AttributeError(SECOND_ORDER_PRIOR_ERROR_MSG)
        for prior, name in (
            (offset_prior, "offset_prior"),
            (coeffs_1_prior, "coeffs_1_prior"),
            (coeffs_2_prior, "coeffs_2_prior"),
        ):
            _check_prior(prior, name)
        self.offset_prior = offset_prior
        self.coeffs_1_prior = coeffs_1_prior
        self.coeffs_2_prior = coeffs_2_prior
        self.coeff_constraint = coeff_constraint
        self.second_order = bool(second_order)
        self.quad_deg = int(quad_deg)
        # GPyTorch creates the initial values in `dtype` (float32 by default).
        exact = dtype is not None and "64" in str(dtype)
        self._dtype = dtype
        self._round = (lambda v: v) if exact else _f32
        self._local_kind = kind_from_dtype(dtype, device)
        z, w = leggauss(self.quad_deg, a=0, b=1, dtype=dtype, device=device)
        self.z = z
        self.w = w

    @property
    def batch_shape(self) -> tuple[int, ...]:  # type: ignore[override]
        """The explicitly set batch shape (without the base kernel's)."""
        return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._batch_shape = _check_batch_shape(value)

    def _own_params(self) -> list[tuple[str, Interval | None, Prior | None]]:
        out: list[tuple[str, Interval | None, Prior | None]] = [
            ("raw_offset", self.coeff_constraint, self.offset_prior),
            ("raw_coeffs_1", self.coeff_constraint, self.coeffs_1_prior),
        ]
        if self.second_order:
            out.append(("raw_coeffs_2", self.coeff_constraint, self.coeffs_2_prior))
        return out

    def _children(self) -> list[tuple[str, Kernel]]:
        return [("base_kernel", self.base_kernel)]

    def _base_node(self) -> _rs.KernelNode:
        import math

        node = _rs.KernelNode.orthogonal_additive(
            self.dim,
            self.base_kernel._node(),
            self._per_dim,
            self.quad_deg,
            self.second_order,
            self.check_hypercube,
            self.coeff_constraint._rs,
        )
        log_d = math.log(self.dim)
        node = node.with_param("raw_coeffs_1", default_raw=self._round(-log_d))
        if self.second_order:
            node = node.with_param("raw_coeffs_2", default_raw=self._round(-2 * log_d))
        return node

    def _configure(self, node: _rs.KernelNode) -> _rs.KernelNode:
        # The coefficient constraint is part of the node already.
        for name, _, prior in self._own_params():
            if prior is not None:
                node = node.with_param(name, prior=prior._rs)
        return node

    raw_offset, offset = _properties("raw_offset")
    raw_coeffs_1, coeffs_1 = _properties("raw_coeffs_1")

    @property
    def raw_coeffs_2(self) -> Any:
        if not self.second_order:
            return None
        return self._get_raw("raw_coeffs_2")

    @raw_coeffs_2.setter
    def raw_coeffs_2(self, value: Any) -> None:
        self._set_raw("raw_coeffs_2", value)

    @property
    def coeffs_2(self) -> Any:
        """The ``batch_shape x dim x dim`` upper-triangular matrix of
        second-order coefficients (``None`` without them)."""
        if not self.second_order:
            return None
        d = self.dim
        raw = self._raw_values("raw_coeffs_2")
        values = [self.coeff_constraint.transform(v) for v in raw]
        m = d * (d - 1) // 2
        nb = len(values) // max(m, 1) if m else 1
        out: list[float] = []
        for e in range(nb):
            idx = 0
            mat = [[0.0] * d for _ in range(d)]
            for i in range(d):
                for j in range(i + 1, d):
                    mat[i][j] = values[e * m + idx]
                    idx += 1
            out.extend(v for row in mat for v in row)
        return from_flat(out, (*self._param_shape("raw_coeffs_2")[:-1], d, d), self._kind_out())

    @coeffs_2.setter
    def coeffs_2(self, value: Any) -> None:
        from botorch_rs._arrays import broadcast_to, to_flat

        d = self.dim
        data, shape = to_flat(value, "coeffs_2")
        batch = self._param_shape("raw_coeffs_2")[:-1]
        full = broadcast_to(data, shape, (*batch, d, d))
        raw: list[float] = []
        for e in range(max(len(full) // (d * d), 1)):
            for i in range(d):
                for j in range(i + 1, d):
                    raw.append(self.coeff_constraint.inverse_transform(full[e * d * d + i * d + j]))
        from botorch_rs._arrays import LIST

        self._set_raw("raw_coeffs_2", from_flat(raw, self._param_shape("raw_coeffs_2"), LIST))

    @property
    def num_components(self) -> int:
        n = 1 + self.dim
        if self.second_order:
            n += self.dim * (self.dim - 1) // 2
        return n

    @property
    def component_indices(self) -> dict[str, Any]:
        """Input dimension indices in the native component enumeration."""
        kind = self._kind_out()

        def indices(values: list[int], shape: tuple[int, ...]) -> Any:
            if kind.lib == "torch":
                import torch

                return torch.tensor(values, dtype=torch.long, device=kind.device).reshape(shape)
            if kind.lib == "numpy":
                import numpy as np

                return np.asarray(values, dtype=np.int64).reshape(shape)
            return from_flat(values, shape, kind)

        result = {
            "bias": indices([0], (1,)),
            "first_order": indices(list(range(self.dim)), (self.dim,)),
        }
        if self.second_order:
            pairs = [v for i in range(self.dim) for j in range(i + 1, self.dim) for v in (i, j)]
            result["second_order"] = indices(pairs, (self.dim * (self.dim - 1) // 2, 2))
        return result

    def get_component_index(
        self, component_type: str, dim_index: int | tuple[int, int] | None = None
    ) -> int:
        """Map a component type and input dimensions to its batch index."""
        if component_type == "bias":
            return 0
        if component_type == "first_order":
            if not isinstance(dim_index, int):
                raise ValueError("dim_index must be an int for first_order")
            if not 0 <= dim_index < self.dim:
                raise IndexError(f"dim_index {dim_index} out of range [0, {self.dim - 1}]")
            return 1 + dim_index
        if component_type == "second_order":
            if not self.second_order:
                raise ValueError("second_order components not enabled for this kernel")
            if not isinstance(dim_index, tuple) or len(dim_index) != 2:
                raise ValueError("dim_index must be a tuple (i, j) for second_order")
            i, j = dim_index
            if i >= j:
                raise ValueError(f"For second_order, require i < j, got ({i}, {j})")
            if i < 0 or j >= self.dim:
                raise IndexError(f"Invalid second_order index ({i}, {j}) for dim={self.dim}")
            return 1 + self.dim + sum(self.dim - k - 1 for k in range(i)) + j - i - 1
        raise ValueError(f"Unknown component_type: {component_type}")

    def _parts(
        self, x1: Any = None, x2: Any = None, *, mode: str, diag: bool = False, eps: float = 1e-6
    ) -> Any:
        batch = self._aug if self._model is not None else self.batch_shape
        a = None if x1 is None else to_flat(x1, "x1")
        b = None if x2 is None else to_flat(x2, "x2")
        shape = broadcast_shapes(
            batch, () if a is None else a[1][:-2], () if b is None else b[1][:-2]
        )
        if shape is None:
            raise RuntimeError("The kernel and input batch shapes do not broadcast.")
        raws = self._raw_per_element(len(batch))
        data: list[float] = []
        tail: tuple[int, ...] = ()
        for index in _indices(shape):
            values, out_shape = _rs.additive_kernel_parts(
                self._node(),
                raws[_flat_index(index, batch)],
                self.dim,
                None if a is None else _batch_matrix(a, index),
                None if b is None else _batch_matrix(b, index),
                mode,
                diag,
                eps,
            )
            data.extend(values)
            tail = tuple(out_shape)
        return from_flat(data, (*shape, *tail), self._kind_out() if a is None else kind_of(x1))

    def k(self, x1: Any, x2: Any, diag: bool = False) -> Any:
        """Unorthogonalized base covariance for each input dimension."""
        return self._parts(x1, x2, mode="base", diag=diag)

    def normalizer(self, eps: float = 1e-6) -> Any:
        """Native quadrature normalizers, recomputed from current parameters."""
        return self._parts(mode="normalizer", eps=eps)

    def _non_reduced_forward(
        self, x1: Any, x2: Any, diag: bool = False, last_dim_is_batch: bool = False
    ) -> Any:
        if last_dim_is_batch:
            raise UnsupportedError("OrthogonalAdditiveKernel does not support `last_dim_is_batch`.")
        return self._parts(x1, x2, mode="components", diag=diag)

    def __repr__(self) -> str:
        return f"OrthogonalAdditiveKernel(dim={self.dim}, base_kernel={self.base_kernel!r})"


def leggauss(
    deg: int,
    a: float = -1.0,
    b: float = 1.0,
    dtype: Any = None,
    device: Any = None,
) -> tuple[Any, Any]:
    """Gauss-Legendre quadrature nodes and weights of degree ``deg`` on
    ``[a, b]`` (NumPy's ``leggauss`` rescaled), as arrays of the type given
    by ``dtype`` / ``device`` (torch tensors if given and torch is installed,
    else lists)."""
    x, w = _rs.leggauss(int(deg))
    if not (a == -1 and b == 1):
        x = [(b - a) * (v + 1) / 2 + a for v in x]
        w = [v * ((b - a) / 2) for v in w]
    kind = kind_from_dtype(dtype, device)
    return from_flat(list(x), (len(x),), kind), from_flat(list(w), (len(w),), kind)


__all__ = ["OrthogonalAdditiveKernel", "leggauss"]
