"""The references of botorch_rs's ``LCEMGP`` and ``ProvidedEmbeddingLCEMGP``:
the contextual models of the Unity project BOforUnity on BoTorch.

The classes and ``build_contextual_model`` are copied from the Unity
project's ``context_support.py`` (``_lcemgp_class``,
``_provided_embedding_lcemgp_class`` and ``build_contextual_model`` at
origin/main, October 2026); ``build_contextual_model`` takes the context
configuration as arguments instead of a ``ContextSetup``. The Unity project
runs two variants:

* **learned** embeddings: BoTorch's ``LCEMGP`` with ``embs_dim_list=[1]``,
  whose ``eval()`` skips ``MultiTaskGP.eval()`` (which raises an
  ``AttributeError`` for contexts without data) and sets the constant means
  of such contexts to the average of the observed ones;
* **provided** embeddings: no learned embedding, the task kernel on the
  provided embeddings only and a learned context-specific weight ``w``:
  ``(1 - w) * RBF(E) + w * I``.

``stock`` is BoTorch's ``LCEMGP`` with learned *and* fixed embeddings and the
task kernel resized to both (the Unity project's earlier workaround for
BoTorch's undersized task kernel), which botorch_rs's ``LCEMGP`` also
supports.

Import only when BoTorch is importable.
"""

from __future__ import annotations

from typing import Any

import torch

DT = torch.float64

_LCEMGP_CLASS: Any = None
_PROVIDED_LCEMGP_CLASS: Any = None


def lcemgp_class() -> Any:
    """The Unity project's ``_UnobservedContextLCEMGP``."""
    global _LCEMGP_CLASS
    if _LCEMGP_CLASS is None:
        from botorch.models.contextual_multioutput import LCEMGP
        from botorch.models.multitask import MultiTaskGP
        from gpytorch.means import MultitaskMean

        class _UnobservedContextLCEMGP(LCEMGP):
            def eval(self):  # type: ignore[no-untyped-def]
                unobserved = self._unobserved_task_indices.tolist()
                if unobserved and isinstance(self.mean_module, MultitaskMean):
                    means = self.mean_module.base_means
                    with torch.no_grad():
                        level = torch.stack(
                            [means[i].constant for i in self._observed_task_indices.tolist()]
                        ).mean(dim=0)
                        for i in unobserved:
                            means[i].constant = level
                return super(MultiTaskGP, self).eval()

        _LCEMGP_CLASS = _UnobservedContextLCEMGP
    return _LCEMGP_CLASS


def provided_embedding_lcemgp_class() -> Any:
    """The Unity project's ``_ProvidedEmbeddingLCEMGP``."""
    global _PROVIDED_LCEMGP_CLASS
    if _PROVIDED_LCEMGP_CLASS is None:
        from gpytorch.constraints import Interval
        from gpytorch.kernels.rbf_kernel import RBFKernel

        class _ProvidedEmbeddingLCEMGP(lcemgp_class()):  # type: ignore[misc]
            def __init__(self, train_X, *args, context_emb_feature, **kwargs):  # type: ignore[no-untyped-def]
                super().__init__(train_X, *args, context_emb_feature=context_emb_feature, **kwargs)
                self.emb_layers = torch.nn.ModuleList()
                self.task_covar_module_base = RBFKernel(
                    ard_num_dims=context_emb_feature.shape[-1],
                    lengthscale_constraint=Interval(0.0, 2.0, transform=None, initial_value=1.0),
                ).to(train_X)
                self.register_parameter(
                    "raw_context_specific_weight",
                    torch.nn.Parameter(torch.zeros(1, dtype=train_X.dtype, device=train_X.device)),
                )
                self.register_constraint(
                    "raw_context_specific_weight",
                    Interval(0.0, 1.0, transform=None, initial_value=0.0).to(train_X),
                )

            @property
            def context_specific_weight(self):  # type: ignore[no-untyped-def]
                return self.raw_context_specific_weight_constraint.transform(
                    self.raw_context_specific_weight
                )

            def _task_embeddings(self):  # type: ignore[no-untyped-def]
                return self.context_emb_feature.to(self.device)

            def _eval_context_covar(self):  # type: ignore[no-untyped-def]
                base = self.task_covar_module_base(self._task_embeddings()).to_dense()
                w = self.context_specific_weight
                eye = torch.eye(base.shape[-1], dtype=base.dtype, device=base.device)
                return (1.0 - w) * base + w * eye

        _PROVIDED_LCEMGP_CLASS = _ProvidedEmbeddingLCEMGP
    return _PROVIDED_LCEMGP_CLASS


def build_model(
    lib: str,
    variant: str,
    train_x_with_task: Any,
    train_y: Any,
    num_contexts: int,
    embeddings: Any,
    current_index: int,
) -> Any:
    """One LCEMGP as the Unity project's ``build_contextual_model`` builds it
    (``lib``: ``"bt"`` or ``"rs"``; ``variant``: ``"learned"``,
    ``"provided"`` or ``"stock"``)."""
    context_cat_feature = torch.arange(num_contexts, dtype=DT).unsqueeze(-1)
    kw: dict[str, Any] = {
        "train_X": train_x_with_task,
        "train_Y": train_y,
        "task_feature": train_x_with_task.shape[-1] - 1,
        "context_cat_feature": context_cat_feature,
        "output_tasks": [current_index],
        "all_tasks": list(range(num_contexts)),
    }
    if variant == "provided":
        kw["context_emb_feature"] = torch.as_tensor(embeddings, dtype=DT)
    elif variant == "learned":
        kw["embs_dim_list"] = [1]
    else:
        kw["context_emb_feature"] = torch.as_tensor(embeddings, dtype=DT)
        kw["embs_dim_list"] = [1]
    if lib == "rs":
        from botorch_rs.models.contextual_multioutput import LCEMGP, ProvidedEmbeddingLCEMGP

        return (ProvidedEmbeddingLCEMGP if variant == "provided" else LCEMGP)(**kw)
    if variant == "provided":
        return provided_embedding_lcemgp_class()(**kw)
    model = lcemgp_class()(**kw)
    if variant == "stock":
        from gpytorch.constraints import Interval
        from gpytorch.kernels.rbf_kernel import RBFKernel

        model.task_covar_module_base = RBFKernel(
            ard_num_dims=1 + kw["context_emb_feature"].shape[-1],
            lengthscale_constraint=Interval(0.0, 2.0, transform=None, initial_value=1.0),
        ).to(train_x_with_task)
    return model


def build_contextual_model(
    lib: str,
    train_x_with_task: Any,
    train_y: Any,
    num_contexts: int,
    embeddings: Any,
    current_index: int,
) -> tuple[Any, Any]:
    """The Unity project's ``build_contextual_model``: one LCEMGP per
    objective (provided embeddings if ``embeddings`` is given, else learned),
    a ``ModelListGP`` with a ``SumMarginalLogLikelihood`` for several
    objectives. Returns ``(mll, model)``."""
    if lib == "rs":
        from botorch_rs.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
        from botorch_rs.models import ModelListGP
    else:
        from botorch.models import ModelListGP
        from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
    variant = "provided" if embeddings is not None else "learned"
    models = [
        build_model(
            lib,
            variant,
            train_x_with_task,
            train_y[:, j : j + 1],
            num_contexts,
            embeddings,
            current_index,
        )
        for j in range(train_y.shape[-1])
    ]
    if len(models) == 1:
        return ExactMarginalLogLikelihood(models[0].likelihood, models[0]), models[0]
    model = ModelListGP(*models)
    return SumMarginalLogLikelihood(model.likelihood, model), model


def mll_parameter_names(model: Any) -> list[str]:
    """The names of BoTorch's optimizer variables, in the order of
    ``get_parameters_and_bounds(mll)`` (the likelihood first), without the
    ``model.`` prefix: botorch_rs's flat raw-parameter order."""
    from botorch.optim.utils import get_parameters_and_bounds
    from gpytorch.mlls import ExactMarginalLogLikelihood

    params, _ = get_parameters_and_bounds(ExactMarginalLogLikelihood(model.likelihood, model))
    return [n[len("model.") :] if n.startswith("model.") else n for n in params]


__all__ = [
    "DT",
    "build_contextual_model",
    "build_model",
    "lcemgp_class",
    "mll_parameter_names",
    "provided_embedding_lcemgp_class",
]
