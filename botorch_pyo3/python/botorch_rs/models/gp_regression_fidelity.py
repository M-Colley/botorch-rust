r"""Multi-fidelity Gaussian process regression
(``botorch.models.gp_regression_fidelity``).

``SingleTaskMultiFidelityGP`` is a ``SingleTaskGP`` whose covariance module
is built from BoTorch's fidelity kernels: a product of
``LinearTruncatedFidelityKernel``\ s in a ``ScaleKernel`` (the default,
``linear_truncated=True``), or the product of a kernel on the non-fidelity
inputs, an ``ExponentialDecayKernel`` on the iteration fidelity and a
``DownsamplingKernel`` per data fidelity. The kernels are evaluated as kernel
trees in Rust, so the model supports everything a ``SingleTaskGP`` does
(posteriors, the MLL and its gradient, ``fit_gpytorch_mll``, conditioning,
fantasies, all acquisition functions, e.g. ``qMultiFidelityKnowledgeGradient``
and ``qMultiFidelityMaxValueEntropy``).

.. [Wu2019mf]
    J. Wu, S. Toscano-Palmerin, P. I. Frazier, and A. G. Wilson. Practical
    multi-fidelity bayesian optimization for hyperparameter tuning. ArXiv 2019.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel, ProductKernel, ScaleKernel
from botorch_rs.likelihoods import Likelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models._backend_selection import requires_tensor_backend
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.kernels.downsampling import DownsamplingKernel
from botorch_rs.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch_rs.models.kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
from botorch_rs.models.transforms.input import InputTransform
from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch_rs.priors import GammaPrior
from botorch_rs.utils.types import DEFAULT


class SingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A SingleTaskGP model using a DownsamplingKernel for the data fidelity
    parameter (if present) and an ExponentialDecayKernel for the iteration
    fidelity parameter (if present).

    This kernel is described in [Wu2019mf]_.

    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[3])
    """

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if cls is SingleTaskMultiFidelityGP:
            if backend not in {"auto", "native", "torch"}:
                raise ValueError("backend must be 'auto', 'native', or 'torch'.")
            torch = sys.modules.get("torch")
            values = (train_X, train_Y, kwargs.get("train_Yvar"), *args[:1])
            requested = any(
                getattr(v, "requires_grad", False)
                or getattr(getattr(v, "device", None), "type", "cpu") != "cpu"
                for v in values
            )
            modules = (
                any(
                    requires_tensor_backend(kwargs.get(k))
                    for k in (
                        "covar_module",
                        "likelihood",
                        "mean_module",
                        "input_transform",
                        "outcome_transform",
                    )
                )
                if torch
                else False
            )
            if backend == "torch" or (backend == "auto" and (requested or modules)):
                from botorch_rs.models._torch_backend_specialized import (
                    TorchSingleTaskMultiFidelityGP,
                )

                return object.__new__(TorchSingleTaskMultiFidelityGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        covar_module: Kernel | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: InputTransform | None = None,
        mean_module: ConstantMean | None = None,
        *,
        backend: str = "auto",
    ) -> None:
        r"""
        Args:
            train_X: A ``batch_shape x n x (d + s)`` tensor of training features,
                where ``s`` is the dimension of the fidelity parameters (either one
                or two).
            train_Y: A ``batch_shape x n x m`` tensor of training observations.
            train_Yvar: An optional ``batch_shape x n x m`` tensor of observed
                measurement noise.
            iteration_fidelity: The column index for the training iteration fidelity
                parameter (optional).
            data_fidelities: The column indices for the downsampling fidelity parameter.
                If a list/tuple of indices is provided, a kernel will be constructed for
                each index (optional).
            linear_truncated: If True, use a ``LinearTruncatedFidelityKernel`` instead
                of the default kernel.
            nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
                5/2. Only used when ``linear_truncated=True``.
            covar_module: The module for computing the covariance matrix between
                the non-fidelity features. Defaults to ``RBFKernel``.
            likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
                with inferred noise level.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale). We use a
                ``Standardize`` transform if no ``outcome_transform`` is specified.
                Pass down ``None`` to use no outcome transform.
            input_transform: An input transform that is applied in the model's
                    forward pass.
            mean_module: The mean function to be used. If omitted, use a
                ``ConstantMean``.
        """
        self._init_args = {
            "iteration_fidelity": iteration_fidelity,
            "data_fidelities": data_fidelities,
            "linear_truncated": linear_truncated,
            "nu": nu,
            "outcome_transform": outcome_transform,
            "mean_module": mean_module,
        }
        if iteration_fidelity is None and (data_fidelities is None or len(data_fidelities) == 0):
            raise UnsupportedError(
                f"{self.__class__.__name__} requires at least one fidelity parameter."
            )
        xshape = tuple(to_flat(train_X, "train_X")[1])
        yshape = tuple(to_flat(train_Y, "train_Y")[1])
        # The input transforms of botorch_rs keep the input dimension.
        dim = xshape[-1] if xshape else 0
        m = yshape[-1] if yshape else 1
        aug = tuple(xshape[:-2]) + ((m,) if m > 1 else ())
        covar, subset_batch_dict = _setup_multifidelity_covar_module(
            dim=dim,
            aug_batch_shape=aug,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            data_covar_module=covar_module,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        # Used for subsetting along the output dimension. See Model.subset_output.
        self._subset_batch_dict = {**subset_batch_dict}
        if tuple(self.mean_module.batch_shape) == self._aug_batch_shape:
            self._subset_batch_dict["mean_module.raw_constant"] = -1
        if linear_truncated:
            self._subset_batch_dict["covar_module.raw_outputscale"] = -1
        if train_Yvar is None:
            self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2

    @classmethod
    def construct_inputs(  # type: ignore[override]
        cls,
        training_data: Any,
        fidelity_features: list[int],
    ) -> dict[str, Any]:
        r"""Construct ``Model`` keyword arguments from a dict of ``SupervisedDataset``.

        Args:
            training_data: Dictionary of ``SupervisedDataset``.
            fidelity_features: Index of fidelity parameter as input columns.
        """
        inputs = super().construct_inputs(training_data=training_data)
        inputs["data_fidelities"] = fidelity_features
        return inputs

    def __repr__(self) -> str:
        return (
            f"SingleTaskMultiFidelityGP(n={self._rs.num_train}, d={self._rs.dim}, "
            f"covar_module={self.covar_module!r})"
        )


def _setup_multifidelity_covar_module(
    dim: int,
    aug_batch_shape: Any,
    iteration_fidelity: int | None,
    data_fidelities: Sequence[int] | None,
    linear_truncated: bool,
    nu: float,
    data_covar_module: Kernel | None = None,
) -> tuple[Kernel, dict[str, int]]:
    """Helper function to get the covariance module and associated subset_batch_dict
    for the multifidelity setting.

    Args:
        dim: The dimensionality of the training data.
        aug_batch_shape: The output-augmented batch shape as defined in
            ``BatchedMultiOutputGPyTorchModel``.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelities: The column indices for the downsampling fidelity parameters
            (optional).
        linear_truncated: If True, use a ``LinearTruncatedFidelityKernel`` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when ``linear_truncated=True``.
        data_covar_module: The module for computing the covariance matrix between
                the non-fidelity features. Defaults to ``RBFKernel``.

    Returns:
        The covariance module and subset_batch_dict.

    An empty list of ``data_fidelities`` is treated like ``None`` (BoTorch
    builds an empty ``ProductKernel`` and fails with an ``IndexError`` for
    ``linear_truncated=True``).
    """
    aug = tuple(aug_batch_shape)
    if iteration_fidelity is not None and iteration_fidelity < 0:
        iteration_fidelity = dim + iteration_fidelity
    if data_fidelities is not None:
        data_fidelities = [dim + i if i < 0 else i for i in data_fidelities]
        if len(data_fidelities) == 0:
            data_fidelities = None

    kernels: list[Kernel] = []

    if linear_truncated and data_covar_module is not None:
        raise ValueError(
            "Non-fidelity covariance module cannot be specified when using a linear "
            "truncated kernel."
        )

    if linear_truncated:
        leading_dims = [iteration_fidelity] if iteration_fidelity is not None else []
        trailing_dims = [[i] for i in data_fidelities] if data_fidelities is not None else [[]]
        for tdims in trailing_dims:
            kernels.append(
                LinearTruncatedFidelityKernel(
                    fidelity_dims=leading_dims + tdims,
                    dimension=dim,
                    nu=nu,
                    batch_shape=aug,
                    power_prior=GammaPrior(3.0, 3.0),
                )
            )
    else:
        non_active_dims = set(data_fidelities or [])
        if iteration_fidelity is not None:
            non_active_dims.add(iteration_fidelity)
        active_dimsX = sorted(set(range(dim)) - non_active_dims)

        if data_covar_module is None:
            kernels.append(
                get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=len(active_dimsX),
                    batch_shape=aug,
                    active_dims=active_dimsX,
                )
            )
        else:
            kernels.append(data_covar_module)

        if iteration_fidelity is not None:
            kernels.append(
                ExponentialDecayKernel(
                    batch_shape=aug,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                    offset_prior=GammaPrior(3.0, 6.0),
                    power_prior=GammaPrior(3.0, 6.0),
                    active_dims=[iteration_fidelity],
                )
            )
        if data_fidelities is not None:
            for data_fidelity in data_fidelities:
                kernels.append(
                    DownsamplingKernel(
                        batch_shape=aug,
                        offset_prior=GammaPrior(3.0, 6.0),
                        power_prior=GammaPrior(3.0, 6.0),
                        active_dims=[data_fidelity],
                    )
                )

    kernel = ProductKernel(*kernels)

    covar_module: Kernel
    if linear_truncated:
        covar_module = ScaleKernel(kernel, batch_shape=aug, outputscale_prior=GammaPrior(2.0, 0.15))
        key_prefix = "covar_module.base_kernel.kernels"
    else:
        covar_module = kernel
        key_prefix = "covar_module.kernels"

    subset_batch_dict: dict[str, int] = {}
    if linear_truncated:
        for i in range(len(kernels)):
            subset_batch_dict.update(
                {
                    f"{key_prefix}.{i}.raw_power": -2,
                    f"{key_prefix}.{i}.covar_module_unbiased.raw_lengthscale": -3,
                    f"{key_prefix}.{i}.covar_module_biased.raw_lengthscale": -3,
                }
            )
    else:
        # Only set the subset_batch_dict if using the default kernel. See SingleTaskGP.
        if data_covar_module is None:
            subset_batch_dict[f"{key_prefix}.0.raw_lengthscale"] = -3
        if iteration_fidelity is not None:
            subset_batch_dict.update(
                {
                    f"{key_prefix}.1.raw_power": -2,
                    f"{key_prefix}.1.raw_offset": -2,
                    f"{key_prefix}.1.raw_lengthscale": -3,
                }
            )
        if data_fidelities is not None:
            start_idx = 2 if iteration_fidelity is not None else 1
            for i in range(start_idx, len(data_fidelities) + start_idx):
                subset_batch_dict.update(
                    {
                        f"{key_prefix}.{i}.raw_power": -2,
                        f"{key_prefix}.{i}.raw_offset": -2,
                    }
                )

    return covar_module, subset_batch_dict


__all__ = ["SingleTaskMultiFidelityGP"]
