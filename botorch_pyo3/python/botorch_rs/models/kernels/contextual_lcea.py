"""Native latent-context embedding additive covariance."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, from_flat, kind_of, to_flat
from botorch_rs.constraints import Interval, Positive
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel, _KernelList, _properties, _randn
from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch_rs.priors import GammaPrior, Prior


def get_order(indices: list[int]) -> list[int]:
    return [i % len(indices) for i in indices]


def is_contiguous(indices: list[int]) -> bool:
    first = min(indices)
    return set(indices) == set(range(first, first + len(indices)))


def _create_new_permutation(decomposition: dict[str, list[int]]) -> list[int]:
    return [i for indices in decomposition.values() for i in sorted(indices)]


def get_permutation(decomposition: dict[str, list[int]]) -> list[int] | None:
    """BoTorch's public ordering helper (retained for compatibility).

    Kernel evaluation uses the actual listed decomposition directly, fixing
    the helper's loss of feature alignment for differently ordered contexts.
    """
    order = get_order(next(iter(decomposition.values())))
    if not all(is_contiguous(v) and get_order(v) == order for v in decomposition.values()):
        return _create_new_permutation(decomposition)
    return None


class _Embedding:
    def __init__(self, kernel: LCEAKernel, index: int, rows: int, dim: int) -> None:
        self._kernel = kernel
        self._index = index
        self.num_embeddings, self.embedding_dim = rows, dim
        self.max_norm = 1.0
        self.norm_type = 2.0

    @property
    def weight(self) -> Any:
        return self._kernel._get_raw(f"emb_layers.{self._index}.weight")

    @weight.setter
    def weight(self, value: Any) -> None:
        self._kernel._set_raw(f"emb_layers.{self._index}.weight", value)

    def __call__(self, indices: Any) -> Any:
        self._kernel._forward_pass()
        data, _ = to_flat(self.weight)
        rows, shape = to_flat(indices, "indices")
        if any(r != int(r) or not 0 <= r < self.num_embeddings for r in rows):
            raise RuntimeError("Embedding index is out of range.")
        return from_flat(
            [
                data[int(r) * self.embedding_dim + j]
                for r in rows
                for j in range(self.embedding_dim)
            ],
            (*shape, self.embedding_dim),
            kind_of(indices),
        )


class LCEAKernel(Kernel):
    """Sum of context-pair spatial RBFs weighted by a learned context RBF.

    One spatial lengthscale vector and one embedding lengthscale vector are
    shared across all context pairs. Learned embedding tables use LCEMGP's
    native max-norm convention. Optional fixed features and known context
    weights are supported. All covariance and model derivatives run in Rust.
    """

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        from botorch_rs.models.map_saas import _use_torch_backend

        device = kwargs.get("device", args[8] if len(args) > 8 else None)
        if cls is LCEAKernel and (
            _use_torch_backend(backend)
            or (backend == "auto" and device is not None and str(device) != "cpu")
        ):
            from botorch_rs.models._torch_backend_contextual import TorchLCEAKernel

            return object.__new__(TorchLCEAKernel)
        return object.__new__(cls)

    def __init__(
        self,
        decomposition: dict[str, list[int]],
        batch_shape: Any,
        train_embedding: bool = True,
        cat_feature_dict: dict | None = None,
        embs_feature_dict: dict | None = None,
        embs_dim_list: list[int] | None = None,
        context_weight_dict: dict | None = None,
        device: Any = None,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__(batch_shape=batch_shape)
        if self._batch_shape and train_embedding:
            raise UnsupportedError(
                "Batched learned LCEA embeddings are not yet supported; use fixed features or independent LCEAGP models."
            )
        if not decomposition:
            raise ValueError("LCEAKernel requires a nonempty context decomposition.")
        self.decomposition = {name: list(indices) for name, indices in decomposition.items()}
        self.num_param = len(next(iter(decomposition.values())))
        if self.num_param == 0 or any(len(v) != self.num_param for v in decomposition.values()):
            raise ValueError("The number of parameters needs to be same across all contexts.")
        if any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0
            for v in decomposition.values()
            for i in v
        ):
            raise ValueError("Context indices must be nonnegative integers.")
        self.context_list = list(decomposition)
        self.num_contexts = len(decomposition)
        self.permutation = get_permutation(decomposition)
        self.train_embedding = bool(train_embedding)
        self._device = device
        categories = (
            [[i] for i in range(self.num_contexts)]
            if cat_feature_dict is None
            else [list(cat_feature_dict[c]) for c in self.context_list]
        )
        nc = len(categories[0])
        if any(len(row) != nc for row in categories):
            raise ValueError("All contexts must have the same number of categorical features.")
        dims = [1] * nc if embs_dim_list is None else list(embs_dim_list)
        if len(dims) != nc or any(not isinstance(d, int) or d <= 0 for d in dims):
            raise ValueError("embs_dim_list needs a positive dimension per categorical feature.")
        mappings = [
            {v: i for i, v in enumerate(sorted({row[j] for row in categories}))} for j in range(nc)
        ]
        self.context_cat_feature = categories
        self.emb_dims = [(len(mappings[j]), dims[j]) for j in range(nc)]
        self._embedding_shapes = self.emb_dims if train_embedding else []
        self._categories = (
            [[mappings[j][row[j]] for j in range(nc)] for row in categories]
            if train_embedding
            else [[] for _ in categories]
        )
        fixed = (
            [[float(v) for v in embs_feature_dict[c]] for c in self.context_list]
            if embs_feature_dict is not None
            else [[] for _ in self.context_list]
        )
        self._fixed = to_flat(fixed, "embs_feature_dict")
        self.context_emb_feature = fixed if embs_feature_dict is not None else None
        self.n_embs = sum(d for _, d in self._embedding_shapes) + self._fixed[1][-1]
        if self.n_embs == 0:
            raise ValueError("LCEAKernel requires learned or fixed embedding features.")
        self.context_weight = (
            None
            if context_weight_dict is None
            else [float(context_weight_dict[c]) for c in self.context_list]
        )
        self.task_covar_module = get_covar_module_with_dim_scaled_prior(
            self.n_embs, self._batch_shape
        )
        self.base_kernel = get_covar_module_with_dim_scaled_prior(self.num_param, self._batch_shape)
        self.raw_outputscale_list_constraint = Positive()
        self.outputscale_list_prior = GammaPrior(2.0, 15.0)
        self.emb_layers = _KernelList()
        for i, (rows, dim) in enumerate(self._embedding_shapes):
            self.emb_layers.append(_Embedding(self, i, rows, dim))
            self._local[f"emb_layers.{i}.weight"] = _randn((*self._batch_shape, rows, dim))
        self.emb_weight_matrix_list = None

    @property
    def device(self) -> Any:
        return self._device

    def _own_params(self) -> list[tuple[str, Interval | None, Prior | None]]:
        out: list[tuple[str, Interval | None, Prior | None]] = [
            (
                "raw_outputscale_list",
                self.raw_outputscale_list_constraint,
                self.outputscale_list_prior,
            )
        ]
        out.extend(
            (f"emb_layers.{i}.weight", None, None) for i in range(len(self._embedding_shapes))
        )
        for prefix, child in [
            ("task_covar_module", self.task_covar_module),
            ("base_kernel", self.base_kernel),
        ]:
            out.extend((f"{prefix}.{name}", c, p) for name, c, p in child._own_params())
        return out

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.contextual_lcea(
            list(self.decomposition.values()),
            self._categories,
            self._embedding_shapes,
            self._fixed,
            self.context_weight,
            self.task_covar_module._node(),
            self.base_kernel._node(),
        )

    def _raw_values(self, name: str) -> list[float]:
        if self._model is None:
            for prefix, child in [
                ("task_covar_module.", self.task_covar_module),
                ("base_kernel.", self.base_kernel),
            ]:
                if name.startswith(prefix):
                    return child._raw_values(name.removeprefix(prefix))
        return super()._raw_values(name)

    def _all_kernels(self) -> Iterator[Kernel]:
        yield self
        yield self.task_covar_module
        yield self.base_kernel

    def named_sub_kernels(self) -> Iterator[tuple[str, Kernel]]:
        yield "task_covar_module", self.task_covar_module
        yield "base_kernel", self.base_kernel

    def _bind(self, model: Any, prefix: str = "") -> None:
        super()._bind(model, prefix)
        self.task_covar_module._bind(model, prefix + "task_covar_module.")
        self.base_kernel._bind(model, prefix + "base_kernel.")

    raw_outputscale_list, outputscale_list = _properties("raw_outputscale_list")

    def _set_outputscale_list(self, value: Any) -> None:
        self.outputscale_list = value

    def _forward_pass(self) -> None:
        if not self._embedding_shapes:
            return
        batch = self._aug if self._model is not None else self.batch_shape
        node = self._node()
        raw = self._raw_per_element(len(batch))
        normalized = [node.contextual_lcea_normalize(values) for values in raw]
        offset = 0
        for name, shape, _ in node.own_params():
            size = math.prod(shape)
            if name.startswith("emb_layers."):
                values = [v for element in normalized for v in element[offset : offset + size]]
                before = self._raw_values(name)
                if values != before:
                    self._set_raw(name, from_flat(values, (*batch, *shape), LIST))
            offset += size

    def _task_embeddings(self) -> Any:
        self._forward_pass()
        batch = self._aug if self._model is not None else self.batch_shape
        node = self._node()
        values = [
            v
            for raw in self._raw_per_element(len(batch))
            for v in node.contextual_lcea_embeddings(raw)[0]
        ]
        return from_flat(values, (*batch, self.num_contexts, self.n_embs), self._kind_out())

    _task_embeddings_batch = _task_embeddings

    def _eval_context_covar(self) -> Any:
        self._forward_pass()
        batch = self._aug if self._model is not None else self.batch_shape
        node = self._node()
        values = [
            v
            for raw in self._raw_per_element(len(batch))
            for v in node.contextual_lcea_covariance(raw)[0]
        ]
        return from_flat(values, (*batch, self.num_contexts, self.num_contexts), self._kind_out())

    def _eval_base_covar_perm(self, x1: Any, x2: Any) -> Any:
        """Spatial covariance with context axes, evaluated by native RBF."""
        expanded = []
        counts = []
        for value in (x1, x2):
            data, shape = to_flat(value)
            n, d = shape[-2:]
            counts.append(n)
            values = [
                data[(b * n + i) * d + j]
                for b in range(math.prod(shape[:-2]))
                for i in range(n)
                for dims in self.decomposition.values()
                for j in dims
            ]
            expanded.append(
                from_flat(
                    values, (*shape[:-2], n * self.num_contexts, self.num_param), kind_of(value)
                )
            )
        data, shape = to_flat(self.base_kernel(*expanded))
        n, nz = counts
        c = self.num_contexts
        values = [
            data[(b * n * c + i * c + a) * nz * c + j * c + e]
            for b in range(math.prod(shape[:-2]))
            for i in range(n)
            for j in range(nz)
            for a in range(c)
            for e in range(c)
        ]
        return from_flat(values, (*shape[:-2], n, nz, c, c), kind_of(x1))

    def train(self, mode: bool = True) -> LCEAKernel:
        self.training = bool(mode)
        if not mode:
            self._context_covar = self._eval_context_covar()
        return self

    def eval(self) -> LCEAKernel:
        return self.train(False)

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        self._forward_pass()
        return super().__call__(x1, x2, diag, last_dim_is_batch, **params)

    forward = __call__


__all__ = ["LCEAKernel", "_create_new_permutation", "get_order", "get_permutation", "is_contiguous"]
