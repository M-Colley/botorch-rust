"""``SingleTaskGP`` (``botorch.models.gp_regression``)."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Pair, from_flat, from_pair, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.kernels import Kernel
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, Likelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models.model import Model
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.utils.types import DEFAULT


def _is_default(value: Any) -> bool:
    return value is DEFAULT or type(value).__name__ == "_DefaultType"


def _validate_tensor_args(
    xshape: tuple[int, ...], yshape: tuple[int, ...], vshape: tuple[int, ...] | None
) -> None:
    if len(xshape) != len(yshape):
        if len(xshape) - len(yshape) == 1 and xshape[:-1] == yshape:
            msg = (
                "An explicit output dimension is required for targets."
                f" Expected Y with dimension {len(xshape)} (got Y.dim()={len(yshape)})."
            )
        else:
            msg = (
                "Expected X and Y to have the same number of dimensions"
                f" (got X with dimension {len(xshape)} and Y with dimension {len(yshape)})."
            )
        raise BotorchTensorDimensionError(msg)
    if len(xshape) > 2:
        raise UnsupportedError(
            "Batched training data (`batch_shape x n x d`) are not supported by botorch_rs."
        )
    if len(xshape) < 2:
        raise BotorchTensorDimensionError(
            f"Expected train_X to be an `n x d` tensor, got shape {xshape}."
        )
    if vshape is not None and tuple(yshape[len(yshape) - len(vshape) :]) != tuple(vshape):
        raise BotorchTensorDimensionError(
            "An explicit output dimension is required for observation noise."
            f" Expected Yvar with shape: {yshape[len(yshape) - len(vshape) :]} (got {vshape})."
        )


def _validate_input_scaling(x: Pair, y: Pair) -> None:
    """BoTorch's ``validate_input_scaling`` warnings for the transformed
    training data (unit cube inputs, standardized outcomes)."""
    atol = 1e-2
    if any(v < -atol or v > 1 + atol for v in x[0]):
        warnings.warn(
            "Data (input features) is not contained to the unit cube. "
            "Please consider min-max scaling the input data.",
            InputDataWarning,
            stacklevel=3,
        )
    values = y[0]
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        if abs(mean) > atol:
            warnings.warn(
                f"Data (outcome observations) is not standardized (mean = {mean}). "
                "Please consider scaling the input to zero mean and unit variance.",
                InputDataWarning,
                stacklevel=3,
            )
        return
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    if abs(mean) > atol or abs(std - 1) > atol:
        warnings.warn(
            f"Data (outcome observations) is not standardized (std = {std}, mean = {mean})."
            "Please consider scaling the input to zero mean and unit variance.",
            InputDataWarning,
            stacklevel=3,
        )


# Shapes of the raw parameters in BoTorch (for d inputs).
def _param_shape(name: str, size: int) -> tuple[int, ...]:
    if name.endswith("raw_noise"):
        return (size,)
    if name.endswith("raw_lengthscale"):
        return (1, size)
    return ()


class SingleTaskGP(Model):
    r"""A single-task exact GP with BoTorch's defaults (single output).

    Without ``train_Yvar`` the noise level is inferred (``GaussianLikelihood``
    with a ``LogNormal(-4, 1)`` noise prior); with ``train_Yvar`` the given
    observation noise is used. The default covariance module is an ARD RBF
    kernel with the dimension-scaled lengthscale prior of Hvarfner et al.
    (2024), the default outcome transform is ``Standardize(m=1)`` and inputs
    are not transformed unless ``input_transform`` is given.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=2))
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
        >>> posterior = model.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        likelihood: Likelihood | None = None,
        covar_module: Kernel | None = None,
        mean_module: ConstantMean | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Normalize | None = None,
    ) -> None:
        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        yvar = to_flat(train_Yvar, "train_Yvar") if train_Yvar is not None else None
        _validate_tensor_args(x[1], y[1], yvar[1] if yvar is not None else None)
        if yvar is not None and yvar[1] != y[1]:  # broadcast trailing dimensions
            reps = len(y[0]) // max(len(yvar[0]), 1)
            yvar = (yvar[0] * reps, y[1])
        d = x[1][1]
        m = y[1][1]
        if m != 1:
            raise UnsupportedError(
                f"SingleTaskGP in botorch_rs supports a single output (m = 1), got m = {m}."
            )
        if likelihood is not None:
            if not isinstance(likelihood, GaussianLikelihood):
                raise UnsupportedError(
                    "`likelihood` must be a botorch_rs GaussianLikelihood "
                    "(e.g. get_gaussian_likelihood_with_gamma_prior())."
                )
            if yvar is not None:
                raise UnsupportedError(
                    "A custom likelihood cannot be combined with `train_Yvar` in botorch_rs."
                )
        if covar_module is not None and not isinstance(covar_module, Kernel):
            raise UnsupportedError(
                "`covar_module` must be a botorch_rs kernel (RBFKernel, MaternKernel(nu=2.5) "
                "or ScaleKernel), e.g. from botorch_rs.models.utils.gpytorch_modules."
            )
        if mean_module is not None and not isinstance(mean_module, ConstantMean):
            raise UnsupportedError("Only a ConstantMean `mean_module` is supported.")
        if _is_default(outcome_transform):
            ot_obj: Standardize | None = Standardize(m=1)
            ot_rs, std_default = None, True
        elif outcome_transform is None:
            ot_obj, ot_rs, std_default = None, None, False
        elif isinstance(outcome_transform, Standardize):
            if outcome_transform._m != m:
                raise BotorchTensorDimensionError(
                    f"The outcome transform is for m = {outcome_transform._m} outputs, got m = {m}."
                )
            ot_obj, ot_rs, std_default = outcome_transform, outcome_transform._rs, False
        else:
            raise UnsupportedError("`outcome_transform` must be a Standardize, DEFAULT or None.")
        if input_transform is not None and not isinstance(input_transform, Normalize):
            raise UnsupportedError("`input_transform` must be a Normalize.")
        covar = (
            covar_module
            if covar_module is not None
            else get_covar_module_with_dim_scaled_prior(ard_num_dims=d)
        )
        covar_spec = covar._spec(d) if covar_module is not None else None
        self._rs = _rs.SingleTaskGP(
            x,
            y,
            yvar,
            input_transform._rs if input_transform is not None else None,
            ot_rs,
            std_default,
            covar_spec,
            likelihood._spec if likelihood is not None else None,
        )
        self._kind = kind
        # GP modules bound to the model's hyperparameters.
        if likelihood is None:
            likelihood = (
                FixedNoiseGaussianLikelihood._create()
                if yvar is not None
                else get_gaussian_likelihood_with_lognormal_prior()
            )
        likelihood._bind(self)
        covar._bind(self)
        mean = mean_module if mean_module is not None else ConstantMean()
        mean._bind(self)
        self.likelihood: Likelihood = likelihood
        self.covar_module: Kernel = covar
        self.mean_module: ConstantMean = mean
        if input_transform is not None:
            fitted_input = self._rs.input_transform()
            assert fitted_input is not None
            input_transform._set_fitted(fitted_input, kind)
            self.input_transform = input_transform
        if ot_obj is not None:
            fitted_outcome = self._rs.outcome_transform()
            assert fitted_outcome is not None
            ot_obj._set_fitted(fitted_outcome, kind_of(train_Y))
            self.outcome_transform = ot_obj
        if covar_module is None:
            _validate_input_scaling(
                self._rs.transformed_train_inputs(), self._rs.transformed_train_targets()
            )
        self.training = True

    # -- data --------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def train_inputs(self) -> tuple[Any]:
        """``(train_X,)`` in the original input space."""
        return (from_pair(self._rs.train_inputs(), self._kind),)

    @property
    def train_targets(self) -> Any:
        """The training targets after the outcome transform, shape ``(n,)``."""
        return from_pair(self._rs.transformed_train_targets(), self._kind)

    # -- modes -------------------------------------------------------------

    def train(self, mode: bool = True) -> SingleTaskGP:
        self.training = mode
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    # -- posterior ---------------------------------------------------------

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """The posterior at ``X`` (``batch_shape x q x d``), in the original
        outcome space; with ``observation_noise=True`` the likelihood noise is
        added (the mean training noise for fixed-noise models)."""
        if output_indices is not None and list(output_indices) != [0]:
            raise UnsupportedError("SingleTaskGP has a single output (output_indices=[0]).")
        if posterior_transform is not None:
            raise UnsupportedError("`posterior_transform` is not supported by botorch_rs.")
        if not isinstance(observation_noise, bool):
            raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}."
            )
        if flat[1][-1] != self._rs.dim:
            raise BotorchTensorDimensionError(
                f"Expected X with {self._rs.dim} columns (d), got {flat[1][-1]}."
            )
        self.eval()
        return GPyTorchPosterior(self, flat, kind, observation_noise)

    # -- parameters --------------------------------------------------------

    def _raw_param(self, suffix: str) -> list[float]:
        values = self._rs.raw_parameters()
        offset = 0
        for name, size in self._rs.parameter_names():
            if name.endswith(suffix):
                return values[offset : offset + size]
            offset += size
        raise AttributeError(f"The model has no parameter {suffix!r}.")

    def _set_raw_param(self, suffix: str, new: list[float]) -> None:
        values = self._rs.raw_parameters()
        offset = 0
        for name, size in self._rs.parameter_names():
            if name.endswith(suffix):
                if len(new) == 1 and size > 1:
                    new = new * size
                if len(new) != size:
                    raise BotorchTensorDimensionError(f"{name} has {size} values, got {len(new)}.")
                values[offset : offset + size] = new
                self._rs.set_raw_parameters(values)
                return
            offset += size
        raise AttributeError(f"The model has no parameter {suffix!r}.")

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every hyperparameter, with BoTorch's names,
        order and shapes."""
        values = self._rs.raw_parameters()
        offset = 0
        for name, size in self._rs.parameter_names():
            data = values[offset : offset + size]
            offset += size
            yield name, from_flat(data, _param_shape(name, size), self._kind)

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    def initialize(self, **kwargs: Any) -> SingleTaskGP:
        """Sets hyperparameters by (dotted) name, like GPyTorch's
        ``Module.initialize``, e.g. ``model.initialize(**{"likelihood.noise": 1e-3,
        "covar_module.lengthscale": 0.5})``."""
        for name, value in kwargs.items():
            obj: Any = self
            *path, attr = name.split(".")
            for part in path:
                obj = getattr(obj, part)
            if not isinstance(getattr(type(obj), attr, None), property):
                raise AttributeError(f"Unknown hyperparameter {name!r}.")
            setattr(obj, attr, value)
        return self

    # -- copying -----------------------------------------------------------

    def __deepcopy__(self, memo: dict[int, Any]) -> SingleTaskGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_rs":
                new.__dict__[key] = value.copy()
            else:
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        return (
            f"SingleTaskGP(n={self._rs.num_train}, d={self._rs.dim}, "
            f"covar_module={self.covar_module!r})"
        )


__all__ = ["SingleTaskGP"]
