"""Device-resident contextual covariance modules and exact GP models."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    TorchKernel,
    TorchScaleKernel,
    _tensor,
    build_kernel,
)
from botorch_rs.models.contextual import LCEAGP, SACGP
from botorch_rs.models.kernels.contextual_lcea import LCEAKernel
from botorch_rs.models.kernels.contextual_sac import SACKernel


class _ContextKernel(TorchKernel):
    _tensor_backend = "torch"
    decomposition: dict[str, list[int]]

    @property
    def device(self) -> Any:
        return self._anchor.device

    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)

    def _context_inputs(self, x1: Tensor, x2: Tensor | None) -> tuple[Tensor, Tensor]:
        x1, x2 = self._inputs(x1, x2)
        maximum = max(index for indices in self.decomposition.values() for index in indices)
        if maximum >= x1.shape[-1]:
            raise ValueError("A context feature index exceeds the input dimension.")
        return x1, x2


class TorchSACKernel(_ContextKernel, SACKernel):  # type: ignore[misc]
    """Shared spatial RBF, one independent scale per context."""

    base_kernel: Any

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        kwargs.pop("backend", None)
        source = SACKernel(*args, backend="native", **kwargs) if _source is None else _source
        like = torch.empty((), device=source.device) if _like is None else _like
        TorchKernel.__init__(self, like)
        self.batch_shape = source.batch_shape
        self.decomposition = source.decomposition
        self._device = like.device
        self.base_kernel = build_kernel(source.base_kernel, 0, self.batch_shape, like)
        self.kernel_dict = nn.ModuleDict(  # type: ignore[assignment]
            {
                name: TorchScaleKernel(kernel, self.base_kernel, like)
                for name, kernel in source.kernel_dict.items()
            }
        )
        for index, indices in enumerate(self.decomposition.values()):
            self.register_buffer(f"_indices_{index}", torch.tensor(indices, device=like.device))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        x1, x2 = self._context_inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("Contextual kernels do not define last_dim_is_batch.")
        result: Tensor | None = None
        for i, kernel in enumerate(self.kernel_dict.values()):
            indices = getattr(self, f"_indices_{i}")
            value = kernel(x1.index_select(-1, indices), x2.index_select(-1, indices), diag=diag)
            result = value if result is None else result + value
        assert result is not None
        return result


class _ContextEmbedding(nn.Module):
    """Embedding lookup with reference max-norm parameter projection."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.num_embeddings, self.embedding_dim = weight.shape[-2:]
        self.max_norm, self.norm_type = 1.0, 2.0

    def forward(self, indices: Tensor) -> Tensor:
        # As in nn.Embedding(max_norm=1), projection changes the parameter
        # before the differentiable lookup. Avoid gratuitous version changes.
        with torch.no_grad():
            norms = self.weight.norm(dim=-1, keepdim=True)
            if bool((norms > self.max_norm).any()):
                self.weight.mul_(torch.where(norms > 1, 1 / (norms + 1e-7), 1.0))
        return self.weight.index_select(-2, indices)


class TorchLCEAKernel(_ContextKernel, LCEAKernel):  # type: ignore[misc]
    """Sum context-pair spatial kernels weighted by learned context covariance."""

    context_emb_feature: Any
    context_cat_feature: Any
    context_weight: Any
    base_kernel: Any
    task_covar_module: Any
    emb_layers: Any
    raw_outputscale_list_constraint: Any
    _context_indices: Tensor

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        kwargs.pop("backend", None)
        requested_batch = None
        if _source is None:
            values = inspect.signature(LCEAKernel.__init__).bind(None, *args, **kwargs)
            values.apply_defaults()
            arguments = dict(values.arguments)
            arguments.pop("self")
            requested_batch = torch.Size(arguments["batch_shape"])
            # The native CPU descriptor deliberately rejects this layout.
            # Construct its unbatched initialization, then allocate independent
            # Tensor leaves in every requested batch below.
            if arguments["train_embedding"]:
                arguments["batch_shape"] = ()
            arguments["backend"] = "native"
            source = LCEAKernel(**arguments)
        else:
            source = _source
        like = torch.empty((), device=source.device) if _like is None else _like
        TorchKernel.__init__(self, like)
        self.batch_shape = source.batch_shape if requested_batch is None else requested_batch
        self.decomposition = source.decomposition
        self.num_param, self.num_contexts = source.num_param, source.num_contexts
        self.context_list, self.permutation = source.context_list, source.permutation
        self.train_embedding, self.n_embs = source.train_embedding, source.n_embs
        self.emb_dims, self._device = source.emb_dims, like.device
        self.register_buffer(
            "_context_indices", torch.tensor(list(self.decomposition.values()), device=like.device)
        )
        self.register_buffer(
            "context_cat_feature",
            torch.tensor(source._categories, device=like.device, dtype=torch.long),
        )
        self.register_buffer(
            "context_emb_feature", _tensor(source._fixed[0], like).reshape(source._fixed[1])
        )
        self.register_buffer(
            "context_weight",
            None if source.context_weight is None else _tensor(source.context_weight, like),
        )
        self.emb_layers = nn.ModuleList(  # type: ignore[assignment]
            [
                _ContextEmbedding(
                    _tensor(source._get_raw(f"emb_layers.{i}.weight"), like).expand(
                        *self.batch_shape, *source._embedding_shapes[i]
                    )
                )
                for i in range(len(source._embedding_shapes))
            ]
        )
        self.emb_weight_matrix_list = None
        self.base_kernel = build_kernel(source.base_kernel, self.num_param, self.batch_shape, like)
        self.task_covar_module = build_kernel(
            source.task_covar_module, self.n_embs, self.batch_shape, like
        )
        for kernel in (self.base_kernel, self.task_covar_module):
            if kernel.batch_shape != self.batch_shape:
                kernel.raw_lengthscale = nn.Parameter(
                    kernel.raw_lengthscale.expand(
                        *self.batch_shape, *kernel.raw_lengthscale.shape[-2:]
                    ).clone()
                )
                kernel.batch_shape = self.batch_shape
        self.raw_outputscale_list_constraint = TorchConstraint(
            source.raw_outputscale_list_constraint, like
        )
        self.raw_outputscale_list = nn.Parameter(
            _tensor(source.raw_outputscale_list, like).expand(*self.batch_shape, -1).clone()
        )
        self._register_prior(
            "outputscale_list_prior", source.outputscale_list_prior, "outputscale_list", 1
        )

    @property
    def outputscale_list(self) -> Tensor:
        return self.raw_outputscale_list_constraint.transform(self.raw_outputscale_list)

    @outputscale_list.setter
    def outputscale_list(self, value: Any) -> None:
        self._assign("raw_outputscale_list", value, self.raw_outputscale_list_constraint)

    def _task_embeddings(self) -> Tensor:
        learned = [layer(self.context_cat_feature[:, i]) for i, layer in enumerate(self.emb_layers)]
        fixed = self.context_emb_feature.expand(*self.batch_shape, *self.context_emb_feature.shape)
        return torch.cat([*learned, fixed], dim=-1) if learned else fixed

    _task_embeddings_batch = _task_embeddings

    def _eval_context_covar(self) -> Tensor:
        embeddings = self._task_embeddings()
        cov = self.task_covar_module(embeddings)
        scales = self.outputscale_list
        if self.context_weight is not None:
            scales = scales * self.context_weight
        return cov * scales.unsqueeze(-1) * scales.unsqueeze(-2)

    def _eval_base_covar_perm(self, x1: Tensor, x2: Tensor) -> Tensor:
        c, p = self.num_contexts, self.num_param
        flat = self._context_indices.flatten()
        a = x1.index_select(-1, flat).reshape(*x1.shape[:-2], -1, p)
        b = x2.index_select(-1, flat).reshape(*x2.shape[:-2], -1, p)
        cov = self.base_kernel(a, b)
        return cov.reshape(*cov.shape[:-2], x1.shape[-2], c, x2.shape[-2], c).transpose(-3, -2)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        x1, x2 = self._context_inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("Contextual kernels do not define last_dim_is_batch.")
        context = self._eval_context_covar()
        spatial = self._eval_base_covar_perm(x1, x2)
        value = (spatial * context[..., None, None, :, :]).sum((-2, -1))
        if diag:
            if x1.shape[-2] != x2.shape[-2]:
                raise ValueError("diag=True requires equal numbers of input points.")
            return value.diagonal(dim1=-2, dim2=-1)
        return value

    def _forward_pass(self) -> None:
        self._task_embeddings()


class TorchSACGP(TorchSingleTaskGP, SACGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None,
        decomposition: dict[str, list[int]],
        *,
        backend: str = "torch",
    ) -> None:
        batch = train_Y.shape[:-2] + ((train_Y.shape[-1],) if train_Y.shape[-1] > 1 else ())
        kernel = TorchSACKernel(decomposition, batch, _like=train_X)
        TorchSingleTaskGP.__init__(self, train_X, train_Y, train_Yvar, covar_module=kernel)
        self.decomposition = kernel.decomposition


class TorchLCEAGP(TorchSingleTaskGP, LCEAGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None,
        decomposition: dict[str, list[int]],
        train_embedding: bool = True,
        cat_feature_dict: dict | None = None,
        embs_feature_dict: dict | None = None,
        embs_dim_list: list[int] | None = None,
        context_weight_dict: dict | None = None,
        *,
        backend: str = "torch",
    ) -> None:
        if train_X.ndim != 2 or train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("LCEAGP requires unbatched n x d inputs and n x 1 outcomes.")
        kernel = TorchLCEAKernel(
            decomposition,
            (),
            train_embedding,
            cat_feature_dict,
            embs_feature_dict,
            embs_dim_list,
            context_weight_dict,
            _like=train_X,
        )
        TorchSingleTaskGP.__init__(
            self, train_X, train_Y, train_Yvar, covar_module=kernel, outcome_transform=None
        )
        self.decomposition = kernel.decomposition
