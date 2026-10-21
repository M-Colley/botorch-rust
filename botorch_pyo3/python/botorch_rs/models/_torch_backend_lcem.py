"""Latent context embedding multitask GPs with device-resident full graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.constraints import Interval
from botorch_rs.kernels import RBFKernel
from botorch_rs.models._torch_backend_contextual import _ContextEmbedding
from botorch_rs.models._torch_backend_modules import build_kernel, module_log_prior
from botorch_rs.models._torch_backend_multitask import TorchMultiTaskGP
from botorch_rs.models.contextual_multioutput import LCEMGP
from botorch_rs.utils.types import DEFAULT


class TorchLCEMGP(TorchMultiTaskGP, LCEMGP):
    """Exact long-format multitask GP with learned contextual covariance.

    The context kernel sees all learned and supplied embedding dimensions.
    Categorical labels are remapped independently per feature. Tensor-valued
    fixed embeddings remain differentiable model inputs on their own device.
    """

    context_cat_feature: Any
    context_emb_feature: Any
    _context_category_indices: Tensor
    emb_layers: Any
    emb_dims: list[tuple[int, int]]

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        mean_module: Any = None,
        covar_module: Any = None,
        likelihood: Any = None,
        context_cat_feature: Tensor | None = None,
        context_emb_feature: Tensor | None = None,
        embs_dim_list: list[int] | None = None,
        output_tasks: list[int] | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        seed: int | None = None,
        backend: str = "torch",
    ) -> None:
        TorchMultiTaskGP.__init__(
            self,
            train_X,
            train_Y,
            task_feature,
            train_Yvar,
            mean_module,
            None,
            likelihood,
            task_covar_prior=None,
            output_tasks=output_tasks,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.covar_module = build_kernel(covar_module, self.num_non_task_features, (), train_X)
        categorical = (
            self._task_labels.unsqueeze(-1)
            if context_cat_feature is None
            else torch.as_tensor(context_cat_feature, device=train_X.device)
        )
        if (
            categorical.ndim != 2
            or categorical.shape[0] != self.num_tasks
            or not categorical.shape[-1]
        ):
            raise ValueError("context_cat_feature must have shape (num_tasks, num_categories).")
        if not bool(torch.isfinite(categorical).all() & (categorical == categorical.long()).all()):
            raise ValueError("Context category labels must be finite integers.")
        categories = []
        sizes = []
        for column in categorical.unbind(-1):
            values, indices = column.unique(sorted=True, return_inverse=True)
            categories.append(indices)
            sizes.append(values.numel())
        dims = [1] * len(categories) if embs_dim_list is None else list(embs_dim_list)
        if len(dims) != len(categories) or any(
            not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in dims
        ):
            raise ValueError("embs_dim_list needs one positive dimension per category.")
        self.register_buffer("context_cat_feature", categorical)
        self.register_buffer("_context_category_indices", torch.stack(categories, -1))
        fixed = (
            None
            if context_emb_feature is None
            else torch.as_tensor(context_emb_feature, device=train_X.device, dtype=train_X.dtype)
        )
        if fixed is not None and (
            fixed.ndim != 2
            or fixed.shape[0] != self.num_tasks
            or not bool(torch.isfinite(fixed).all())
        ):
            raise ValueError(
                "context_emb_feature must be a finite (num_tasks, embedding_dim) Tensor."
            )
        self.register_buffer("context_emb_feature", fixed)
        self.emb_dims = list(zip(sizes, dims, strict=True))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=train_X.device).manual_seed(seed)
        self.emb_layers = nn.ModuleList(
            [
                _ContextEmbedding(
                    torch.randn(
                        rows, width, device=train_X.device, dtype=train_X.dtype, generator=generator
                    )
                )
                for rows, width in self.emb_dims
            ]
        )
        embedding_dim = sum(dims) + (0 if fixed is None else fixed.shape[-1])
        task_kernel = RBFKernel(
            ard_num_dims=embedding_dim,
            lengthscale_constraint=Interval(0, 2, transform=None, initial_value=1),
        )
        self.task_covar_module_base = build_kernel(task_kernel, embedding_dim, (), train_X)

    @property
    def device(self) -> Any:
        return self._train_X.device

    @property
    def task_covar_module_base(self) -> Any:
        return self._modules["task_covar_module_base"]

    @task_covar_module_base.setter
    def task_covar_module_base(self, kernel: Any) -> None:
        self._modules["task_covar_module_base"] = build_kernel(
            kernel, self._task_embeddings().shape[-1], (), self._train_X
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "task_covar_module_base"
            and "_modules" in self.__dict__
            and not isinstance(value, nn.Module)
        ):
            value = build_kernel(value, self._task_embeddings().shape[-1], (), self._train_X)
        super().__setattr__(name, value)

    def _task_embeddings(self) -> Tensor:
        values = [
            layer(self._context_category_indices[:, i]) for i, layer in enumerate(self.emb_layers)
        ]
        if self.context_emb_feature is not None:
            values.append(self.context_emb_feature)
        return torch.cat(values, dim=-1)

    def _eval_context_covar(self) -> Tensor:
        embeddings = self._task_embeddings()
        covariance = self.task_covar_module_base(embeddings, embeddings)
        return covariance if isinstance(covariance, Tensor) else covariance.to_dense()

    def _forward_pass(self) -> None:
        self._task_embeddings()

    def task_covar_module(self, task_idcs: Tensor) -> Tensor:
        if task_idcs.ndim < 2 or task_idcs.shape[-1] != 1:
            raise ValueError("task_idcs must have shape (..., n, 1).")
        indices = task_idcs[..., 0].long()
        if not bool(
            ((task_idcs[..., 0] == indices) & (indices >= 0) & (indices < self.num_tasks)).all()
        ):
            raise IndexError("Context covariance indices must be valid mapped task integers.")
        covariance = self._eval_context_covar()
        return covariance[indices.unsqueeze(-1), indices.unsqueeze(-2)]

    def _covariance(self, x1: Tensor, x2: Tensor) -> Tensor:
        left, right = x1[..., self._task_feature].long(), x2[..., self._task_feature].long()
        context = self._eval_context_covar()[left.unsqueeze(-1), right.unsqueeze(-2)]
        spatial = self.covar_module(
            x1.index_select(-1, self._base_idxr), x2.index_select(-1, self._base_idxr)
        )
        if not isinstance(spatial, Tensor):
            spatial = spatial.to_dense()
        return spatial * context

    def _log_prior(self) -> Any:
        return sum(
            module_log_prior(module, ())
            for module in (
                self.likelihood,
                self.mean_module,
                self.covar_module,
                self.task_covar_module_base,
                self.input_transform,
            )
            if module is not None
        )

    def eval(self) -> Any:
        # Unobserved contexts retain their explicitly initialized embeddings;
        # there is no low-rank IndexKernel factor to overwrite.
        return self.train(False)
