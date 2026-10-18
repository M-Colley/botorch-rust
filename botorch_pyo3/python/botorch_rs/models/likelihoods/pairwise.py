"""Pairwise likelihoods for preference models (``botorch.models.likelihoods.pairwise``).

The likelihoods of a ``PairwiseGP``: the probability that the first item of
a comparison is preferred, as a function of the utilities of the items.
Comparisons are given by BoTorch's matrix ``D`` (``(batch_size x) m x n``,
``D[k, i] = 1`` for the preferred and ``-1`` for the other item of comparison
``k``). The methods are evaluated in Rust with BoTorch's expressions (probit
z-scores clamped at +-3, logits at +-8); they return arrays of the type of
``utility`` and are not differentiable with torch's autograd (botorch_rs
differentiates the ``PairwiseGP`` marginal log likelihood natively).
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, kind_of, to_flat
from botorch_rs.likelihoods import Likelihood


class PairwiseLikelihood(Likelihood):
    """
    Pairwise likelihood base class for pairwise preference GP (e.g., PairwiseGP).
    """

    #: The likelihood's name in the extension.
    _rs_name: str = ""

    def __init__(self, max_plate_nesting: int = 1) -> None:
        """
        Initialized like a ``gpytorch.likelihoods.Likelihood``.

        Args:
            max_plate_nesting: Defaults to 1.
        """
        self.max_plate_nesting = max_plate_nesting

    def _eval(self, what: str, utility: Any, D: Any) -> Any:
        u = to_flat(utility, "utility")
        d = to_flat(D, "D")
        return from_pair(_rs.pairwise_likelihood_eval(self._rs_name, what, u, d), kind_of(utility))

    def forward(self, utility: Any, D: Any) -> Any:
        """Given the difference in (estimated) utility util_diff = f(v) - f(u),
        return a Bernoulli distribution object representing the likelihood of
        the user prefer v over u (a ``torch.distributions.Bernoulli``; torch
        is required).

        Note that this is not used by the ``PairwiseGP`` model.
        """
        import importlib

        torch = importlib.import_module("torch")
        p = self.p(utility=utility, D=D)
        return torch.distributions.Bernoulli(probs=torch.as_tensor(p))

    __call__ = forward

    def p(self, utility: Any, D: Any) -> Any:
        """Given the difference in (estimated) utility util_diff = f(v) - f(u),
        return the probability of the user prefer v over u.

        Args:
            utility: A Tensor of shape ``(batch_size) x n``, the utility at MAP point
            D: D is ``(batch_size x) m x n`` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.
        """
        return self._eval("p", utility, D)

    def log_p(self, utility: Any, D: Any) -> Any:
        """return the log of p"""
        return self._eval("log_p", utility, D)

    def negative_log_gradient_sum(self, utility: Any, D: Any) -> Any:
        """Calculate the sum of negative log gradient with respect to each item's latent
            utility values. Useful for models using laplace approximation.

        Args:
            utility: A Tensor of shape ``(batch_size x) n``, the utility at MAP point
            D: D is ``(batch_size x) m x n`` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.

        Returns:
            A ``(batch_size x) n`` Tensor representing the sum of negative log gradient
            values of the likelihood over all comparisons (i.e., the m dimension)
            with respect to each item.
        """
        return self._eval("negative_log_gradient_sum", utility, D)

    def negative_log_hessian_sum(self, utility: Any, D: Any) -> Any:
        """Calculate the sum of negative log hessian with respect to each item's latent
            utility values. Useful for models using laplace approximation.

        Args:
            utility: A Tensor of shape ``(batch_size) x n``, the utility at MAP point
            D: D is ``(batch_size x) m x n`` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.

        Returns:
            A ``(batch_size x) n x n`` Tensor representing the sum of
            negative log hessian values of the likelihood over all
            comparisons (i.e., the m dimension) with respect to each item.
        """
        return self._eval("negative_log_hessian_sum", utility, D)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PairwiseProbitLikelihood(PairwiseLikelihood):
    """Pairwise likelihood using probit function

    Given two items v and u with utilities f(v) and f(u), the probability that we
    prefer v over u with probability std_normal_cdf((f(v) - f(u))/sqrt(2)). Note
    that this formulation implicitly assume the noise term is fixed at 1.
    """

    # Clamping z values for better numerical stability. See self._calc_z for detail
    # norm_cdf(z=3) ~= 0.999, top 0.1% percent
    _zlim = 3
    _rs_name = "probit"


class PairwiseLogitLikelihood(PairwiseLikelihood):
    """Pairwise likelihood using logistic (i.e., sigmoid) function

    Given two items v and u with utilities f(v) and f(u), the probability that we
    prefer v over u with probability sigmoid(f(v) - f(u)). Note
    that this formulation implicitly assume the beta term in logistic function is
    fixed at 1.
    """

    # Clamping logit values for better numerical stability.
    # See self._calc_logit for detail logistic(8) ~= 0.9997, top 0.03% percent
    _logit_lim = 8
    _rs_name = "logit"


__all__ = ["PairwiseLikelihood", "PairwiseLogitLikelihood", "PairwiseProbitLikelihood"]
