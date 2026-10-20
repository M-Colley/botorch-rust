"""Native exact GP inference on feature/time grids with missing observations."""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, broadcast_to, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.kernels import Kernel, MaternKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean, ZeroMean
from botorch_rs.models.gpytorch import GPyTorchModel
from botorch_rs.models.higher_order_gp import _initial_raw_noise
from botorch_rs.models.model import FantasizeMixin
from botorch_rs.models.transforms.input import InputTransform, chain_spec
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.posteriors.latent_kronecker import LatentKroneckerGPPosterior
from botorch_rs.utils.types import DEFAULT


class _MeanBinding:
    """Bind one factor mean to its own native scalar parameter."""

    def __init__(self, owner: Any, name: str) -> None:
        self.owner, self.name = owner, name

    @property
    def _rs(self) -> _MeanBinding:
        return self

    @property
    def _kind(self) -> Any:
        return self.owner._kind

    @property
    def aug_batch_shape(self) -> tuple[()]:
        return ()

    @property
    def constant(self) -> list[float]:
        return self.owner._raw_param(self.name)

    def set_constant(self, value: Any) -> None:
        self.owner._set_raw_param(self.name, from_flat(value[0], tuple(value[1]), LIST))


class LatentKroneckerGP(GPyTorchModel, FantasizeMixin):
    """GP on X x T, with arbitrary missing finite observations in train_Y.

    Complete grids use native Kronecker eigensolves. Missing-data inference
    uses exact dense covariance solves. Static input transforms, Gaussian
    noise, constant/zero factor means and scalar Standardize are supported.
    Posterior moments and samples consistently have shape batch x q x t and
    are returned in original outcome units.
    """

    def __init__(
        self,
        train_X: Any,
        train_T: Any,
        train_Y: Any,
        likelihood: Any = None,
        mean_module_X: Any = None,
        mean_module_T: Any = None,
        covar_module_X: Any = None,
        covar_module_T: Any = None,
        input_transform: Any = None,
        outcome_transform: Any = DEFAULT,
    ) -> None:
        x, t, y = (
            to_flat(train_X, "train_X"),
            to_flat(train_T, "train_T"),
            to_flat(train_Y, "train_Y"),
        )
        if len(x[1]) != 2:
            raise UnsupportedError(
                "LatentKroneckerGP currently supports unbatched n x d training features."
            )
        n, d = x[1]
        if len(t[1]) != 2 or t[1][-1] != 1 or tuple(y[1]) != (n, t[1][0]):
            raise BotorchTensorDimensionError("Expected train_T t x 1 and train_Y n x t.")
        self._kind = kind_of(train_X)
        if input_transform is not None and not isinstance(input_transform, InputTransform):
            raise UnsupportedError("input_transform must be a botorch_rs InputTransform.")
        transformed_X = train_X if input_transform is None else input_transform(train_X)
        self._validate_tensor_args(transformed_X, train_Y)
        transformed = to_flat(transformed_X, "train_X")
        dt = transformed[1][-1]
        spec = chain_spec(input_transform, d, ()) if input_transform is not None else None
        if spec is not None and any(
            list(getattr(step.rs, "params", []) or []) for step in spec.chains[0]
        ):
            raise UnsupportedError(
                "LatentKroneckerGP does not yet optimize learned input transforms such as Warp."
            )
        observed = [i for i, value in enumerate(y[0]) if math.isfinite(value)]
        if not observed:
            raise ValueError("At least one finite observation is required.")
        if outcome_transform is DEFAULT or type(outcome_transform).__name__ == "_DefaultType":
            outcome_transform = Standardize(m=1)
        targets = list(y[0])
        outcome = None
        if outcome_transform is not None:
            if (
                type(outcome_transform) is not Standardize
                or outcome_transform.batch_shape
                or outcome_transform._m != 1
            ):
                raise UnsupportedError(
                    "LatentKroneckerGP supports an unbatched Standardize(m=1) outcome transform."
                )
            outcome_transform.train()
            finite = from_flat([targets[i] for i in observed], (len(observed), 1), self._kind)
            standardized, _ = outcome_transform(finite)
            for i, value in zip(observed, to_flat(standardized)[0], strict=True):
                targets[i] = value
            outcome = (
                float(outcome_transform._rs.means[0][0]),
                float(outcome_transform._rs.stdvs[0][0]),
            )
        likelihood = GaussianLikelihood() if likelihood is None else likelihood
        if type(likelihood) is not GaussianLikelihood or likelihood.batch_shape:
            raise UnsupportedError("LatentKroneckerGP supports an unbatched GaussianLikelihood.")
        mean_X = ZeroMean() if mean_module_X is None else mean_module_X
        mean_T = ZeroMean() if mean_module_T is None else mean_module_T
        kernel_X = (
            MaternKernel(nu=2.5, ard_num_dims=dt) if covar_module_X is None else covar_module_X
        )
        kernel_T = (
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=1))
            if covar_module_T is None
            else covar_module_T
        )
        for mean in (mean_X, mean_T):
            if not isinstance(mean, (ConstantMean, ZeroMean)):
                raise UnsupportedError(
                    "LatentKroneckerGP supports ConstantMean or ZeroMean factor means."
                )
        if not isinstance(kernel_X, Kernel) or not isinstance(kernel_T, Kernel):
            raise UnsupportedError("Covariance modules must be botorch_rs kernels.")
        for module in (likelihood, mean_X, mean_T, kernel_X, kernel_T):
            if module._model is not None:
                raise UnsupportedError("A supplied module is already bound to another model.")
            if tuple(module.batch_shape):
                raise UnsupportedError(
                    "Batched modules in LatentKroneckerGP are not yet supported."
                )
        raw = [_initial_raw_noise(likelihood)]
        mean_options: dict[str, Any] = {}
        for name, mean in (("x", mean_X), ("t", mean_T)):
            constant = isinstance(mean, ConstantMean)
            mean_options[f"mean_{name}"] = "constant" if constant else "zero"
            if isinstance(mean, ConstantMean):
                constraint = mean.raw_constant_constraint
                raw.append(
                    float(constraint._rs.initial_raw_value or 0.0)
                    if constraint is not None
                    else 0.0
                )
                mean_options[f"mean_{name}_prior"] = (
                    None if mean.mean_prior is None else mean.mean_prior._rs
                )
                mean_options[f"mean_{name}_constraint"] = (
                    None if constraint is None else constraint._rs
                )
        raw.extend(kernel_X._raw_flat())
        raw.extend(kernel_T._raw_flat())
        self._rs = _rs.LatentKroneckerGP(
            transformed,
            t,
            (targets, y[1]),
            d,
            kernel_X._node(),
            kernel_T._node(),
            likelihood._spec,
            raw,
            **mean_options,
            input_chain=None if spec is None else [step.rs for step in spec.chains[0]],
            input_chain_training=None
            if spec is None
            else [step.training for step in spec.chains[0]],
            input_chain_prefixes=None if spec is None else [step.prefix for step in spec.chains[0]],
            raw_train_x=None if spec is None else x,
            outcome=outcome,
        )
        self._num_outputs, self._aug_batch_shape = t[1][0], ()
        self._param_shapes: dict[str, tuple[int, ...]] = {"likelihood.noise_covar.raw_noise": (1,)}
        for name, mean in (("X", mean_X), ("T", mean_T)):
            if isinstance(mean, ConstantMean):
                self._param_shapes[f"model.mean_module_{name}.raw_constant"] = ()
                mean._bind(_MeanBinding(self, f"model.mean_module_{name}.raw_constant"))
        for name, kernel in (("X", kernel_X), ("T", kernel_T)):
            self._param_shapes.update(
                {
                    key: tuple(value)
                    for key, value in kernel._node().param_shapes(f"model.covar_module_{name}.")
                }
            )
            kernel._bind(self, f"covar_module_{name}.")
        likelihood._bind(self)
        self.likelihood, self.mean_module_X, self.mean_module_T = likelihood, mean_X, mean_T
        self.covar_module_X, self.covar_module_T = kernel_X, kernel_T
        self._train_X, self._train_T = train_X, train_T
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.training = True

    def _offsets(self) -> dict[str, tuple[int, int]]:
        offsets, position = {}, 0
        for name, size in self._rs.parameter_names():
            offsets[name] = position, size
            position += size
        return offsets

    def _param_index(self, suffix: str) -> tuple[str, int, int]:
        for name, (offset, size) in self._offsets().items():
            if name == suffix or name.endswith(suffix):
                return name, offset, size
        raise AttributeError(f"Unknown parameter {suffix!r}.")

    def _raw_param(self, suffix: str) -> list[float]:
        _, offset, size = self._param_index(suffix)
        return list(self._rs.raw_parameters()[offset : offset + size])

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        return self._param_shapes[self._param_index(suffix)[0]]

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        name, offset, size = self._param_index(suffix)
        data, value_shape = to_flat(value, name)
        values = broadcast_to(data, tuple(value_shape), self._param_shapes[name])
        raw = self._rs.raw_parameters()
        raw[offset : offset + size] = values
        self._rs.set_raw_parameters(raw)

    def named_parameters(self) -> Any:
        raw = self._rs.raw_parameters()
        for name, (offset, size) in self._offsets().items():
            short = name.removeprefix("model.")
            yield (
                short,
                from_flat(raw[offset : offset + size], self._param_shapes[name], self._kind),
            )

    def parameters(self) -> Any:
        for _, value in self.named_parameters():
            yield value

    def _param_entries(self) -> list[Any]:
        return [
            (name, value, offset, size)
            for (name, value), (offset, size) in zip(
                self.named_parameters(), self._offsets().values(), strict=True
            )
        ]

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> tuple[()]:
        return ()

    @property
    def train_inputs(self) -> tuple[Any, Any]:
        return self._train_X, self.train_T

    @property
    def train_T(self) -> Any:
        return self._train_T

    @property
    def train_targets(self) -> Any:
        data, shape = self._rs.train_targets()
        return from_flat(data, tuple(shape), self._kind)

    @property
    def mask_valid(self) -> Any:
        observed = set(self._rs.observed_indices())
        n = self._rs.num_train * self.num_outputs
        if self._kind.lib == "torch":
            import torch

            return torch.tensor(
                [i in observed for i in range(n)], dtype=torch.bool, device=self._kind.device
            )
        if self._kind.lib == "numpy":
            import numpy as np

            return np.asarray([i in observed for i in range(n)], dtype=bool)
        return [i in observed for i in range(n)]

    def train(self, mode: bool = True) -> LatentKroneckerGP:
        self.training = mode
        for name in ("input_transform", "outcome_transform"):
            transform = getattr(self, name, None)
            if transform is not None:
                transform.train(mode)
        return self

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        # Time inputs share the ExactGP input tuple, but only X is transformed.
        if hasattr(self, "_train_T") and X is self._train_T:
            return X
        return super().transform_inputs(X, input_transform)

    @classmethod
    def construct_inputs(cls, training_data: Any) -> dict[str, Any]:
        values = super().construct_inputs(training_data)
        if "train_Yvar" in values:
            warnings.warn(
                "Ignoring Yvar values in provided training data, because they are currently "
                "not supported by LatentKroneckerGP.",
                InputDataWarning,
                stacklevel=2,
            )
        step = training_data.feature_names.index("step")
        x, xs = to_flat(values["train_X"])
        y, ys = to_flat(values["train_Y"])
        if len(xs) != 2 or len(ys) != 2 or ys != (xs[0], 1) or xs[1] < 2:
            raise UnsupportedError("construct_inputs requires unbatched scalar observations.")
        features = [
            tuple(x[i * xs[1] + j] for j in range(xs[1]) if j != step) for i in range(xs[0])
        ]
        times = [x[i * xs[1] + step] for i in range(xs[0])]
        unique_x, unique_t = sorted(set(features)), sorted(set(times))
        x_index, t_index = (
            {v: i for i, v in enumerate(unique_x)},
            {v: i for i, v in enumerate(unique_t)},
        )
        targets = [math.nan] * (len(unique_x) * len(unique_t))
        for row, time, target in zip(features, times, y, strict=True):
            targets[x_index[row] * len(unique_t) + t_index[time]] = target
        kind = kind_of(values["train_X"])
        return {
            "train_X": from_flat(
                [v for row in unique_x for v in row], (len(unique_x), xs[1] - 1), kind
            ),
            "train_T": from_flat(unique_t, (len(unique_t), 1), kind),
            "train_Y": from_flat(
                targets, (len(unique_x), len(unique_t)), kind_of(values["train_Y"])
            ),
        }

    def posterior(
        self,
        X: Any,
        T: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> LatentKroneckerGPPosterior:
        if observation_noise is not False:
            raise NotImplementedError(
                "Observation noise currently not supported for LatentKroneckerGP."
            )
        if posterior_transform is not None:
            raise NotImplementedError(
                "Posterior transforms currently not supported for LatentKroneckerGP."
            )
        if kwargs:
            raise UnsupportedError(f"Unsupported arguments: {sorted(kwargs)}.")
        self.eval()
        return LatentKroneckerGPPosterior(self, X=X, T=self.train_T if T is None else T)

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError("Conditioning currently not supported for LatentKroneckerGP.")

    def use_iterative_methods(self, *args: Any, **kwargs: Any) -> Any:
        raise UnsupportedError(
            "LatentKroneckerGP uses native exact solves; iterative solver settings are not yet supported."
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> LatentKroneckerGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            new.__dict__[name] = value.copy() if name == "_rs" else copy.deepcopy(value, memo)
        return new


__all__ = ["LatentKroneckerGP"]
