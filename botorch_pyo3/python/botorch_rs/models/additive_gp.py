"""Additive GPs (``botorch.models.additive_gp``): ``OrthogonalAdditiveGP``.

Full and component posteriors condition on the same native training solve.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import broadcast_shapes, kind_of, to_flat
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
                batch_shape=self.get_batch_dimensions(train_X, train_Y)[1],
                dtype=getattr(train_X, "dtype", None) or "float64",
                device=getattr(train_X, "device", None),
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
            infer_all_components: Return bias, first-order and interaction
                marginals as an extra leading batch dimension.
            kwargs: Passed on to ``SingleTaskGP.posterior``.

        Returns:
            The posterior distribution at X.
        """
        if infer_all_components:
            if kwargs:
                raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
            if not isinstance(observation_noise, bool):
                raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
            from botorch_rs.acquisition.objective import ExpectationPosteriorTransform

            self.eval()
            shape = to_flat(X, "X")[1]
            if len(shape) < 2 or shape[-1] != self._rs.dim:
                raise ValueError(f"Expected X with shape (..., q, {self._rs.dim})")
            batch = broadcast_shapes(shape[:-2], self.batch_shape)
            if batch is None:
                raise ValueError("The model and query batch shapes do not broadcast.")
            source = _rs.additive_component_model(self._rs, len(batch) - len(self.batch_shape))
            query = _ops.getitem(X, (None, Ellipsis))
            oi, transform = self._posterior_args(
                output_indices,
                None
                if isinstance(posterior_transform, ExpectationPosteriorTransform)
                else posterior_transform,
            )
            posterior = GPyTorchPosterior(
                source,
                to_flat(query, "X"),
                kind_of(X),
                observation_noise,
                model_batch_shape=tuple(source.batch_shape),
                output_indices=oi,
                transform=transform,
                points_per_input=source.points_per_input,
                query_inputs=query,
            )
            return (
                posterior_transform(posterior, X=X)
                if isinstance(posterior_transform, ExpectationPosteriorTransform)
                else posterior
            )
        return super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    @property
    def component_indices(self) -> dict[str, Any]:
        return self.covar_module.component_indices  # type: ignore[attr-defined,no-any-return]

    @property
    def num_components(self) -> int:
        return int(self.covar_module.num_components)  # type: ignore[attr-defined]

    def get_component_index(
        self, component_type: str, dim_index: int | tuple[int, int] | None = None
    ) -> int:
        return int(self.covar_module.get_component_index(component_type, dim_index))  # type: ignore[attr-defined]

    def evaluate_first_order_on_grid(self, grid_1d: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        """Bias and first-order posterior moments on a shared one-dimensional grid."""
        shape = to_flat(grid_1d, "grid_1d")[1]
        if len(shape) != 1 or not shape[0]:
            raise ValueError("grid_1d must be a nonempty one-dimensional grid")
        grid = _ops.expand(
            _ops.getitem(grid_1d, (slice(None), None)),
            (shape[0], to_flat(self.train_inputs[0])[1][-1]),
        )
        posterior = self.posterior(grid, infer_all_components=True)
        mean = _ops.getitem(posterior.mean, (Ellipsis, 0))
        variance = _ops.getitem(posterior.variance, (Ellipsis, 0))
        bias = (
            _ops.reduce("mean", _ops.getitem(mean, 0), -1),
            _ops.reduce("mean", _ops.getitem(variance, 0), -1),
        )
        first = (
            _ops.getitem(mean, slice(1, 1 + len(self.component_indices["first_order"]))),
            _ops.getitem(variance, slice(1, 1 + len(self.component_indices["first_order"]))),
        )
        return bias, first


__all__ = ["OrthogonalAdditiveGP"]
