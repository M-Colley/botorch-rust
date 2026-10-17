"""Additive GPs (``botorch.models.additive_gp``): ``OrthogonalAdditiveGP``.

The model is a ``SingleTaskGP`` with an ``OrthogonalAdditiveKernel``
(evaluated in Rust). The posteriors of the individual additive components
(``posterior(..., infer_all_components=True)``,
``evaluate_first_order_on_grid``) are not ported.
"""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import Likelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch_rs.models.transforms.input import InputTransform
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.utils.types import DEFAULT


class OrthogonalAdditiveGP(SingleTaskGP):
    """A Gaussian Process with Orthogonal Additive Kernel for interpretable modeling.

    This GP model uses an OrthogonalAdditiveKernel which decomposes the function into
    interpretable additive components: a bias term, first-order effects for each input
    dimension, and optionally second-order interaction terms.
    """

    _infer_all_components: bool = False

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        covar_module: OrthogonalAdditiveKernel | None = None,
        second_order: bool = False,
        likelihood: Likelihood | None = None,
        mean_module: ConstantMean | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the OrthogonalAdditiveGP.

        Args:
            train_X: Training inputs (batch_shape x n x d) in [0, 1]^d.
            train_Y: Training outputs (batch_shape x n x 1).
            covar_module: An OrthogonalAdditiveKernel instance. If None, creates a
                default OrthogonalAdditiveKernel with dim inferred from train_X.
            second_order: If True and covar_module is None, enables second-order
                interactions in the default kernel. Ignored if covar_module is provided.
            likelihood: Optional likelihood (defaults to GaussianLikelihood).
            mean_module: Optional mean module (defaults to ConstantMean).
            outcome_transform: Optional outcome transform.
            input_transform: Optional input transform.

        Raises:
            TypeError: If covar_module is provided but is not an
                OrthogonalAdditiveKernel.
        """
        if covar_module is None:
            covar_module = OrthogonalAdditiveKernel(
                dim=to_flat(train_X, "train_X")[1][-1],
                second_order=second_order,
                dtype=getattr(train_X, "dtype", None) or "float64",
            )
        elif not isinstance(covar_module, OrthogonalAdditiveKernel):
            raise TypeError(
                "covar_module must be an OrthogonalAdditiveKernel, "
                f"got {type(covar_module).__name__}"
            )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        infer_all_components: bool = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """Posterior inference of the additive Gaussian process.

        Args:
            X: The input tensor of shape (batch_shape x n x d).
            output_indices: Not supported for this model.
            observation_noise: Whether to add observation noise to the posterior.
            posterior_transform: Optional posterior transform.
            infer_all_components: The posteriors of the individual components
                are not supported by botorch_rs.
            kwargs: Passed on to ``SingleTaskGP.posterior``.

        Returns:
            The posterior distribution at X.
        """
        if infer_all_components:
            raise UnsupportedError(
                "The posteriors of the individual additive components "
                "(infer_all_components=True) are not supported by botorch_rs."
            )
        return super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )


__all__ = ["OrthogonalAdditiveGP"]
