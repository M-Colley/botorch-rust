"""Linear algebra of the bivariate conditioning algorithm
(``botorch.utils.probability.linalg``): ``block_matrix_concat``,
``augment_cholesky`` and the partial pivoted Cholesky factorization
``PivotedCholesky``.

The decomposition steps (``update_``, ``pivot_``) and ``augment_cholesky``
run in Rust on the flattened batch; the arrays may be torch tensors, NumPy
arrays or nested lists. Unlike BoTorch, ``update_`` and ``pivot_`` replace
``tril``, ``perm`` and ``diag`` by new arrays (BoTorch modifies them in
place) and are not differentiable; ``augment_cholesky`` of torch tensors
that require gradients is computed with torch's linear algebra (BoTorch's
expressions) so that gradients flow.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from itertools import chain
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    broadcast_shapes,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import NotPSDError


def _broadcast_batch(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    out = broadcast_shapes(*shapes)
    if out is None:
        raise RuntimeError(f"Shapes {shapes} are not broadcastable.")
    return out


def block_matrix_concat(blocks: Sequence[Sequence[Any]]) -> Any:
    """Concatenates a block matrix: ``blocks`` is a sequence of rows, each a
    sequence of matrices (with broadcastable batch shapes)."""
    shapes = [_ops.shape_of(x) for x in chain.from_iterable(blocks)]
    shape = _broadcast_batch(*(s[:-2] for s in shapes))
    rows = []
    for tensors in blocks:
        parts = [_ops.expand(x, (*shape, *_ops.shape_of(x)[-2:])) for x in tensors]
        if len(parts) > 1:
            rows.append(_ops.concat(parts, dim=-1))
        else:
            rows.extend(parts)
    return _ops.concat(rows, dim=-2)


def _augment_cholesky_torch(
    Laa: Any, Kbb: Any, Kba: Any, Lba: Any, jitter: float | None
) -> Any:
    torch = importlib.import_module("torch")
    if jitter is not None:
        Kbb = Kbb.clone()
        Kbb.diagonal(dim1=-2, dim2=-1).add_(jitter)
    if Lba is None:
        Lba = torch.linalg.solve_triangular(Laa.transpose(-2, -1), Kba, left=False, upper=True)
    Lbb, info = torch.linalg.cholesky_ex(Kbb - Lba @ Lba.transpose(-2, -1))
    if info.any():
        raise NotPSDError(
            "Schur complement of `K` with respect to `Kaa` not PSD for the given "
            "Cholesky factor `Laa`"
            f"{'.' if jitter is None else f' and nugget jitter={jitter}.'}"
        )
    n = Lbb.shape[-1]
    return block_matrix_concat(blocks=([torch.nn.functional.pad(Laa, (0, n))], [Lba, Lbb]))


def augment_cholesky(
    Laa: Any,
    Kbb: Any,
    Kba: Any = None,
    Lba: Any = None,
    jitter: float | None = None,
) -> Any:
    r"""Computes the Cholesky factor of a block matrix ``K = [[Kaa, Kab], [Kba, Kbb]]``
    based on a precomputed Cholesky factor ``Kaa = Laa Laa^T``.

    Args:
        Laa: Cholesky factor of K's upper left block.
        Kbb: Lower-right block of K.
        Kba: Lower-left block of K.
        Lba: Precomputed solve ``Kba Laa^{-T}``.
        jitter: Optional nugget to be added to the diagonal of Kbb.
    """
    if not (Kba is None) ^ (Lba is None):
        raise ValueError("One and only one of `Kba` or `Lba` must be provided.")
    other = Kba if Kba is not None else Lba
    kind = first_kind(Laa, Kbb, other)
    if kind.lib == "torch" and _ops._requires_grad([Laa, Kbb, other]):
        return _augment_cholesky_torch(Laa, Kbb, Kba, Lba, jitter)
    sa, sb, so = _ops.shape_of(Laa), _ops.shape_of(Kbb), _ops.shape_of(other)
    na, nb = sa[-1], sb[-1]
    batch = _broadcast_batch(sa[:-2], sb[:-2], so[:-2])
    nbatch = math.prod(batch)

    def flat(x: Any, r: int, c: int) -> Any:
        return to_flat(_ops.expand(x, (*batch, r, c)))[0]

    try:
        data, _ = _rs.augment_cholesky(
            flat(Laa, na, na),
            flat(Kbb, nb, nb),
            flat(Kba, nb, na) if Kba is not None else None,
            flat(Lba, nb, na) if Lba is not None else None,
            None if jitter is None else float(jitter),
            nbatch,
            na,
            nb,
        )
    except NotPSDError:
        raise NotPSDError(
            "Schur complement of `K` with respect to `Kaa` not PSD for the given "
            "Cholesky factor `Laa`"
            f"{'.' if jitter is None else f' and nugget jitter={jitter}.'}"
        ) from None
    return from_flat(data, (*batch, na + nb, na + nb), kind)


class PivotedCholesky:
    """A partial pivoted Cholesky factorization (BoTorch's dataclass): the
    decomposition ``step``, the factor ``tril`` (``batch_shape x n x n``; the
    first ``step`` columns are final, the lower triangle of the rest holds the
    Schur complement), the permutation ``perm`` and optionally the standard
    deviations ``diag`` of the original variables."""

    __slots__ = ("step", "tril", "perm", "diag")

    def __init__(
        self,
        step: int,
        tril: Any,
        perm: Any,
        diag: Any = None,
        validate_init: bool = True,
    ) -> None:
        self.step = step
        self.tril = tril
        self.perm = perm
        self.diag = diag
        self.__post_init__(validate_init)

    def __post_init__(self, validate_init: bool = True) -> None:
        if not validate_init:
            return
        tshape = _ops.shape_of(self.tril)
        if tshape[-2] != tshape[-1]:
            raise ValueError(
                f"Expected square matrices but `matrix` has shape `{_size(tshape)}`."
            )
        if _ops.shape_of(self.perm) != tshape[:-1]:
            raise ValueError(
                f"`perm` of shape `{_size(_ops.shape_of(self.perm))}` incompatible with "
                f"`matrix` of shape `{_size(tshape)}`."
            )
        if self.diag is not None and _ops.shape_of(self.diag) != tshape[:-1]:
            raise ValueError(
                f"`diag` of shape `{_size(_ops.shape_of(self.diag))}` incompatible with "
                f"`matrix` of shape `{_size(tshape)}`."
            )

    def __repr__(self) -> str:
        return (
            f"PivotedCholesky(step={self.step}, tril={self.tril!r}, perm={self.perm!r}, "
            f"diag={self.diag!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (
            self.step == other.step
            and _equal(self.tril, other.tril)
            and _equal(self.perm, other.perm)
            and (
                (self.diag is None and other.diag is None)
                or (
                    self.diag is not None
                    and other.diag is not None
                    and _equal(self.diag, other.diag)
                )
            )
        )

    __hash__ = None  # type: ignore[assignment]

    def __getitem__(self, key: Any) -> PivotedCholesky:
        return PivotedCholesky(
            step=self.step,
            tril=_ops.getitem(self.tril, key),
            perm=_ops.getitem(self.perm, key),
            diag=None if self.diag is None else _ops.getitem(self.diag, key),
        )

    @property
    def _batch(self) -> tuple[tuple[int, ...], int]:
        shape = _ops.shape_of(self.tril)
        return shape[:-2], shape[-1]

    def update_(self, eps: float = 1e-10) -> None:
        r"""Performs a single matrix decomposition step."""
        batch, n = self._batch
        data, _ = _rs.pivoted_cholesky_update(
            to_flat(self.tril)[0], math.prod(batch), n, self.step, float(eps)
        )
        self.tril = from_flat(data, (*batch, n, n), kind_of(self.tril))
        self.step += 1

    def pivot_(self, pivot: Any) -> None:
        """Swaps the variables at ``step`` and ``pivot`` (one per member of
        the batch)."""
        batch, n = self._batch
        if _ops.shape_of(pivot) != tuple(batch):
            raise ValueError("Argument `pivot` does to match with batch shape`.")
        kind = kind_of(self.tril)
        tril, perm, diag = _rs.pivoted_cholesky_pivot(
            to_flat(self.tril)[0],
            _ops.ints(self.perm),
            None if self.diag is None else to_flat(self.diag)[0],
            math.prod(batch),
            n,
            self.step,
            _ops.ints(pivot),
        )
        self.tril = from_flat(tril, (*batch, n, n), kind)
        self.perm = _ops.int_array(perm, (*batch, n), kind_of(self.perm))
        if self.diag is not None:
            self.diag = from_flat(diag, (*batch, n), kind_of(self.diag))

    def expand(self, *sizes: int) -> PivotedCholesky:
        fields = {}
        for name, ndim in {"perm": 1, "diag": 1, "tril": 2}.items():
            src = getattr(self, name)
            if src is not None:
                fields[name] = _ops.expand(src, (*sizes, *_ops.shape_of(src)[-ndim:]))
        return type(self)(step=self.step, **fields)

    def concat(self, other: PivotedCholesky, dim: int = 0) -> PivotedCholesky:
        if self.step != other.step:
            raise ValueError("Cannot conncatenate decompositions at different steps.")
        fields = {}
        for name in ("tril", "perm", "diag"):
            a = getattr(self, name)
            b = getattr(other, name)
            if (a is None) != (b is None) or (
                a is not None and kind_of(a).lib != kind_of(b).lib
            ):
                raise NotImplementedError(f"Types of field {name} do not match.")
            if a is not None:
                fields[name] = _ops.concat((a, b), dim=dim)
        return type(self)(step=self.step, **fields)

    def detach(self) -> PivotedCholesky:
        fields = {}
        for name in ("tril", "perm", "diag"):
            obj = getattr(self, name)
            if obj is not None:
                fields[name] = _ops.detach(obj)
        return type(self)(step=self.step, **fields)

    def clone(self) -> PivotedCholesky:
        fields = {}
        for name in ("tril", "perm", "diag"):
            obj = getattr(self, name)
            if obj is not None:
                fields[name] = _ops.clone(obj)
        return type(self)(step=self.step, **fields)


def _size(shape: tuple[int, ...]) -> str:
    return f"torch.Size({list(shape)})"


def _equal(a: Any, b: Any) -> bool:
    if is_torch_tensor(a) and is_torch_tensor(b):
        return bool(a.shape == b.shape and a.equal(b))
    fa, fb = to_flat(a), to_flat(b)
    return tuple(fa[1]) == tuple(fb[1]) and list(fa[0]) == list(fb[0])


__all__ = ["PivotedCholesky", "augment_cholesky", "block_matrix_concat"]
