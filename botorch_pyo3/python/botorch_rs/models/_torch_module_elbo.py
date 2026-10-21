"""Variational ELBO for an actual caller-owned approximate model and likelihood."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models._torch_backend_modules import tensor_prior_records


class TorchModuleVariationalELBO(nn.Module, VariationalELBO):
    """Use the caller's q(f), expected log probability, KL and regularizers."""

    _tensor_backend = "torch"
    _external_model_protocol = True
    likelihood: Any
    model: Any

    def __init__(
        self,
        likelihood: nn.Module,
        model: nn.Module,
        num_data: int,
        beta: float = 1,
        combine_terms: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(model, nn.Module) or not callable(
            getattr(getattr(model, "variational_strategy", None), "kl_divergence", None)
        ):
            raise TypeError(
                "model must be a Torch module with variational_strategy.kl_divergence()."
            )
        if not isinstance(likelihood, nn.Module) or not callable(
            getattr(likelihood, "expected_log_prob", None)
        ):
            raise TypeError("likelihood must be a Torch module with expected_log_prob().")
        if not math.isfinite(num_data) or num_data <= 0 or not math.isfinite(beta) or beta < 0:
            raise ValueError("num_data must be positive and beta nonnegative, both finite.")
        self.likelihood, self.model = likelihood, model
        self.num_data, self.beta, self.combine_terms = num_data, beta, combine_terms

    def named_priors(self) -> Any:
        seen = set()
        for prefix, root in (("likelihood", self.likelihood), ("model", self.model)):
            for name, owner, prior, closure, setting in tensor_prior_records(root):
                key = (id(owner), name.rsplit(".", 1)[-1], id(prior))
                if key not in seen:
                    seen.add(key)
                    yield f"{prefix}.{name}", owner, prior, closure, setting

    def _log_likelihood_term(self, distribution: Any, target: Tensor, **kwargs: Any) -> Tensor:
        return self.likelihood.expected_log_prob(target, distribution, **kwargs).sum(-1)

    def forward(self, approximate_dist_f: Any = None, target: Any = None, **kwargs: Any) -> Any:
        if approximate_dist_f is None:
            inputs = getattr(self.model, "train_inputs", None)
            if inputs is None:
                raise TypeError(
                    "Provide q(f) and targets, or a training closure/data_loader for this supplied model."
                )
            approximate_dist_f = self.model(*inputs)
            target = getattr(self.model, "train_targets", None) if target is None else target
        if target is None:
            raise TypeError("missing target values")
        n = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(approximate_dist_f, target, **kwargs) / n
        kl = self.model.variational_strategy.kl_divergence() * (self.beta / self.num_data)
        added = torch.zeros_like(log_likelihood)
        losses = getattr(self.model, "added_loss_terms", lambda: ())()
        had_added_losses = False
        for term in losses:
            added = added + term.loss()
            had_added_losses = True
        prior = torch.zeros_like(log_likelihood)
        for _, owner, distribution, closure, _ in self.named_priors():
            prior = prior + distribution.log_prob(closure(owner)).sum() / self.num_data
        if self.combine_terms:
            return log_likelihood - kl + prior - added
        terms = (log_likelihood, kl, prior)
        return (*terms, added) if had_added_losses else terms

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        if raw_parameters is not None or not self.combine_terms:
            raise NotImplementedError(
                "Tensor value_and_grad uses current Parameters and requires combine_terms=True."
            )
        parameters = tuple(self.parameters())
        trainable = tuple(p for p in parameters if p.requires_grad)
        value = self().sum()
        gradients = torch.autograd.grad(value, trainable, allow_unused=True) if trainable else ()
        by_identity = dict(zip(map(id, trainable), gradients, strict=True))
        parts = [
            (torch.zeros_like(p) if by_identity.get(id(p)) is None else by_identity[id(p)]).reshape(
                -1
            )
            for p in parameters
        ]
        return float(value.detach()), torch.cat(parts).detach().cpu().tolist() if parts else []
