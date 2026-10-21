r"""
Functionality for allocating the inducing points of sparse Gaussian
process models (``botorch.models.utils.inducing_point_allocators``).

References

.. [chen2018dpp]
    Laming Chen and Guoxin Zhang and Hanning Zhou, Fast greedy MAP inference
    for determinantal point process to improve recommendation diversity,
    Proceedings of the 32nd International Conference on Neural Information
    Processing Systems, 2018, https://arxiv.org/abs/1709.05135.

The greedy pivoted-Cholesky selection (``_pivoted_cholesky_init``) and the
expected-improvement quality scores are computed in Rust; kernel matrices
come from botorch_rs kernels.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


def _shape_of(x: Any) -> tuple[int, ...]:
    return tuple(to_flat(x, "inputs")[1])


class InducingPointAllocator(ABC):
    r"""
    This class provides functionality to initialize the inducing point locations
    of an inducing point-based model, e.g. a ``SingleTaskVariationalGP``.
    """

    @abstractmethod
    def _get_quality_function(
        self,
    ) -> QualityFunction:
        """
        Build the quality function required for this inducing point allocation strategy.

        Returns:
            A quality function.
        """

    def allocate_inducing_points(
        self,
        inputs: Any,
        covar_module: Any,
        num_inducing: int,
        input_batch_shape: Any,
    ) -> Any:
        r"""
        Initialize the ``num_inducing`` inducing point locations according to a
        specific initialization strategy. todo say something about quality

        Args:
            inputs: A (\*batch_shape, n, d)-dim input data tensor.
            covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
            num_inducing: The maximun number (m) of inducing points (m <= n).
            input_batch_shape: The non-task-related batch shape.

        Returns:
            A (\*batch_shape, m, d)-dim tensor of inducing point locations.
        """
        quality_function = self._get_quality_function()
        if is_torch_tensor(inputs):
            return _tensor_allocate(
                inputs, covar_module, num_inducing, input_batch_shape, quality_function
            )
        input_batch_shape = tuple(int(b) for b in (input_batch_shape or ()))
        kind = kind_of(inputs)
        x = to_flat(inputs, "inputs")
        kernel = to_flat(covar_module(inputs), "kernel")
        kdim = len(kernel[1])
        base_case = kdim == 2
        multi_task_case = kdim == 3 and len(input_batch_shape) == 0
        if base_case:
            return _pivoted_cholesky_init(
                train_inputs=inputs,
                kernel_matrix=from_flat(kernel[0], tuple(kernel[1]), kind),
                max_length=num_inducing,
                quality_scores=quality_function(inputs),
            )
        if multi_task_case:
            n = kernel[1][-1]
            if len(x[1]) == 3:
                d = x[1][-1]
                element = from_flat(x[0][: x[1][-2] * d], tuple(x[1][-2:]), kind)
            else:
                element = inputs
            kernel_element = from_flat(kernel[0][: n * n], (n, n), kind)
            return _pivoted_cholesky_init(
                train_inputs=element,
                kernel_matrix=kernel_element,
                max_length=num_inducing,
                quality_scores=quality_function(element),
            )
        # Batched input cases.
        shape = tuple(x[1])
        n, d = shape[-2], shape[-1]
        if len(shape) == 2:
            nb = math.prod(input_batch_shape)
            data = list(x[0]) * nb
        else:
            nb = math.prod(shape[:-2])
            data = list(x[0])
        out: list[float] = []
        for b in range(nb):
            element = from_flat(data[b * n * d : (b + 1) * n * d], (n, d), kind)
            k = to_flat(covar_module(element), "kernel")
            kern = from_flat(k[0][: n * n], (n, n), kind)
            pts = _pivoted_cholesky_init(
                train_inputs=element,
                kernel_matrix=kern,
                max_length=num_inducing,
                quality_scores=quality_function(element),
            )
            pdata, pshape = to_flat(pts, "inducing_points")
            if pshape[0] != num_inducing:
                # torch.stack(...).view(*input_batch_shape, num_inducing, -1)
                raise RuntimeError(
                    "stack expects each tensor to be equal size, or the inducing points of "
                    f"a batch element are {tuple(pshape)} instead of ({num_inducing}, {d})"
                )
            out.extend(pdata)
        return from_flat(out, (*input_batch_shape, num_inducing, d), kind)


class QualityFunction(ABC):
    """A function that scores inputs with respect
    to a specific criterion."""

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:  # [n, d] -> [n]
        """
        Args:
            inputs: inputs (of shape n x d)

        Returns:
            A tensor of quality scores for each input, of shape [n]
        """


class UnitQualityFunction(QualityFunction):
    """
    A function returning ones for each element. Using this quality function
    for inducing point allocation corresponds to allocating inducing points
    with the sole aim of minimizing predictive variance, i.e. the approach
    of [burt2020svgp]_.
    """

    def __call__(self, inputs: Any) -> Any:  # [n, d]-> [n]
        """
        Args:
            inputs: inputs (of shape n x d)

        Returns:
            A tensor of ones for each input, of shape [n]
        """
        if is_torch_tensor(inputs):
            return inputs.new_ones(inputs.shape[0])
        shape = _shape_of(inputs)
        return from_flat([1.0] * shape[0], (shape[0],), kind_of(inputs))


class ExpectedImprovementQualityFunction(QualityFunction):
    """
    A function measuring the quality of input points as their expected
    improvement with respect to a conservative baseline. Expectations
    are according to the model from the previous BO step. See [moss2023ipa]_
    for details and justification.
    """

    def __init__(self, model: Any, maximize: bool):
        r"""
        Args:
            model: The model fitted during the previous BO step. For now, this
                must be a single task model (i.e. num_outputs=1).
            maximize: Set True if we are performing function maximization, else
                set False.
        """
        if model.num_outputs != 1:
            raise NotImplementedError("Multi-output models are currently not supported. ")
        self._model = model
        self._maximize = maximize

    def __call__(self, inputs: Any) -> Any:  # [n, d] -> [n]
        """
        Args:
            inputs: inputs (of shape n x d)

        Returns:
            A tensor of quality scores for each input, of shape [n]
        """
        if is_torch_tensor(inputs):
            import torch

            with torch.no_grad():
                posterior = self._model.posterior(inputs)
                mean = posterior.mean.squeeze(-2).squeeze(-1)
                sigma = posterior.variance.clamp_min(1e-12).sqrt().reshape(mean.shape)
                best = mean.max() if self._maximize else mean.min()
                u = (mean - best) / sigma if self._maximize else (best - mean) / sigma
                return sigma * (
                    (-0.5 * u.square()).exp() / math.sqrt(2 * math.pi) + u * torch.special.ndtr(u)
                )
        posterior = self._model.posterior(inputs)
        mean = to_flat(posterior.mean, "mean")[0]
        variance = to_flat(posterior.variance, "variance")[0]
        scores = _rs.expected_improvement_quality_py(
            [float(v) for v in mean], [float(v) for v in variance], bool(self._maximize)
        )
        return from_flat(scores, (len(scores),), kind_of(inputs))


class GreedyVarianceReduction(InducingPointAllocator):
    r"""
    The inducing point allocator proposed by [burt2020svgp]_, that
    greedily chooses inducing point locations with maximal (conditional)
    predictive variance.
    """

    def _get_quality_function(
        self,
    ) -> QualityFunction:
        """
        Build the unit quality function required for the greedy variance
        reduction inducing point allocation strategy.

        Returns:
            A quality function.
        """

        return UnitQualityFunction()


class GreedyImprovementReduction(InducingPointAllocator):
    r"""
    An inducing point allocator that greedily chooses inducing points with large
    predictive variance and that are in promising regions of the search
    space (according to the model form the previous BO step), see [moss2023ipa]_.
    """

    def __init__(self, model: Any, maximize: bool):
        r"""

        Args:
            model: The model fitted during the previous BO step.
            maximize: Set True if we are performing function maximization, else
                set False.
        """
        self._model = model
        self._maximize = maximize

    def _get_quality_function(
        self,
    ) -> QualityFunction:
        """
        Build the improvement-based quality function required for the greedy
        improvement reduction inducing point allocation strategy.

        Returns:
            A quality function.
        """

        return ExpectedImprovementQualityFunction(self._model, self._maximize)


def _pivoted_cholesky_init(
    train_inputs: Any,
    kernel_matrix: Any,
    max_length: int,
    quality_scores: Any,
    epsilon: float = 1e-6,
) -> Any:
    r"""
    A pivoted Cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. This method returns a greedy approximation of the MAP
    estimate of the specified DPP, i.e. its returns a set of points that are
    highly diverse (according to the provided kernel_matrix) and have high
    quality (according to the provided quality_scores).

    Args:
        train_inputs: training inputs (of shape n x d)
        kernel_matrix: kernel matrix on the training inputs
        max_length: number of inducing points to initialize
        quality_scores: scores representing the quality of each candidate
            input (of shape [n])
        epsilon: numerical jitter for stability.

    Returns:
        max_length x d tensor of the training inputs corresponding to the top
        max_length pivots of the training kernel matrix
    """
    if is_torch_tensor(train_inputs):
        return _tensor_pivoted_cholesky(
            train_inputs, kernel_matrix, max_length, quality_scores, epsilon
        )
    x_data, x_shape = to_flat(train_inputs, "train_inputs")
    q_data, q_shape = to_flat(quality_scores, "quality_scores")
    if (q_shape[0] if q_shape else 1) != x_shape[0]:
        raise ValueError("_pivoted_cholesky_init requires a quality score for each of train_inputs")
    if getattr(kernel_matrix, "requires_grad", False):
        raise UnsupportedError(
            "`_pivoted_cholesky_init` does not support using a `kernel_matrix` "
            "with `requires_grad=True`."
        )
    k = to_flat(kernel_matrix, "kernel_matrix")
    if len(k[1]) != 2 or k[1][0] != k[1][1]:
        raise BotorchTensorDimensionError(
            f"kernel_matrix must be a square matrix, got shape {tuple(k[1])}."
        )
    idx = _rs.pivoted_cholesky_init_py(k, int(max_length), [float(v) for v in q_data], epsilon)
    d = x_shape[-1] if len(x_shape) > 1 else 1
    out: list[float] = []
    for i in idx:
        out.extend(x_data[i * d : (i + 1) * d])
    return from_flat(out, (len(idx), *x_shape[1:]), kind_of(train_inputs))


def _tensor_pivoted_cholesky(
    inputs: Any, covariance: Any, count: int, quality: Any, epsilon: float = 1e-6
) -> Any:
    """Discrete pivot selection on the Tensor device; selected inputs stay live."""
    import torch

    covariance = covariance if isinstance(covariance, torch.Tensor) else covariance.to_dense()
    if inputs.ndim != 2 or covariance.shape != (inputs.shape[0], inputs.shape[0]):
        raise BotorchTensorDimensionError("Expected n x d inputs and an n x n kernel matrix.")
    if quality.shape != (inputs.shape[0],):
        raise ValueError("_pivoted_cholesky_init requires a quality score for each of train_inputs")
    if covariance.requires_grad:
        raise UnsupportedError(
            "`_pivoted_cholesky_init` does not support using a `kernel_matrix` "
            "with `requires_grad=True`."
        )
    if type(count) is not int or count < 0:
        raise ValueError("max_length must be a nonnegative integer.")
    count = min(count, inputs.shape[0])
    if not inputs.shape[0] or not count:
        return inputs[:0]
    if covariance.device != inputs.device or quality.device != inputs.device:
        raise ValueError("Inputs, kernel matrix and quality scores must share a device.")
    if not bool(torch.isfinite(covariance).all() & torch.isfinite(quality).all()):
        raise ValueError("Kernel matrix and quality scores must be finite.")
    factors = covariance.new_zeros(count, inputs.shape[0])
    residual = covariance.diagonal().clone()
    scores = residual * quality.square()
    selected = [scores.argmax()]
    while len(selected) < count:
        k = len(selected) - 1
        pivot = selected[-1]
        previous = factors[:k, pivot]
        row = (covariance[pivot] - previous @ factors[:k]) / residual[pivot].sqrt()
        factors[k] = row
        residual = residual - row.square()
        residual[pivot] = -torch.inf
        pivot = (residual * quality.square()).argmax()
        if bool(residual[pivot] < epsilon):
            break
        selected.append(pivot)
    return inputs.index_select(0, torch.stack(selected))


def _tensor_allocate(
    inputs: Any, kernel: Any, count: int, input_batch_shape: Any, quality_function: Any
) -> Any:
    import torch

    from botorch_rs.kernels import Kernel

    batch = torch.Size(input_batch_shape or ())
    if inputs.ndim < 2 or tuple(torch.broadcast_shapes(inputs.shape[:-2], batch)) != tuple(batch):
        raise BotorchTensorDimensionError(
            "Input batch dimensions must broadcast to input_batch_shape."
        )
    if type(count) is not int or count < 0:
        raise ValueError("num_inducing must be a nonnegative integer.")
    count = min(count, inputs.shape[-2])
    if isinstance(kernel, Kernel):
        from botorch_rs.models._torch_backend_modules import build_kernel

        kernel = build_kernel(kernel, inputs.shape[-1], kernel.batch_shape, inputs)
    expanded = inputs.expand(*batch, *inputs.shape[-2:])
    kernel_batch = tuple(getattr(kernel, "batch_shape", ()))
    evaluated = expanded.unsqueeze(-3) if len(kernel_batch) > len(batch) else expanded
    with torch.no_grad():
        covariance = kernel(evaluated, evaluated)
        covariance = covariance if isinstance(covariance, torch.Tensor) else covariance.to_dense()
        # Independent output batches share the locations selected for the first output.
        if covariance.ndim > len(batch) + 2:
            covariance = covariance.select(len(batch), 0)
        covariance = covariance.expand(*batch, inputs.shape[-2], inputs.shape[-2])
    if not batch:
        return _tensor_pivoted_cholesky(expanded, covariance, count, quality_function(expanded))
    points = [
        _tensor_pivoted_cholesky(x, k, count, quality_function(x))
        for x, k in zip(
            expanded.reshape(-1, *expanded.shape[-2:]),
            covariance.reshape(-1, *covariance.shape[-2:]),
            strict=True,
        )
    ]
    if not points:
        return expanded.new_empty(*batch, count, expanded.shape[-1])
    if any(point.shape[-2] != count for point in points):
        raise RuntimeError("Batched inducing allocation selected fewer than num_inducing points.")
    return torch.stack(points).reshape(*batch, count, expanded.shape[-1])


__all__ = [
    "ExpectedImprovementQualityFunction",
    "GreedyImprovementReduction",
    "GreedyVarianceReduction",
    "InducingPointAllocator",
    "QualityFunction",
    "UnitQualityFunction",
]
