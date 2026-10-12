"""``SingleTaskGP`` (``botorch.models.gp_regression``)."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Pair, broadcast_to, from_flat, from_pair, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.kernels import Kernel, ScaleKernel
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, Likelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models.model import Model
from botorch_rs.models.transforms.input import (
    ChainedInputTransform,
    InputPerturbation,
    InputTransform,
    Normalize,
    apply_chain_state,
    chain_steps,
)
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
    if len(xshape) < 2:
        raise BotorchTensorDimensionError(
            f"Expected train_X to be an `n x d` tensor, got shape {xshape}."
        )
    if vshape is not None and tuple(yshape[len(yshape) - len(vshape) :]) != tuple(vshape):
        raise BotorchTensorDimensionError(
            "An explicit output dimension is required for observation noise."
            f" Expected Yvar with shape: {yshape[len(yshape) - len(vshape) :]} (got {vshape})."
        )


def _validate_input_scaling(x: Pair, y: Pair, n: int) -> None:
    """BoTorch's ``validate_input_scaling`` warnings for the transformed
    training data (unit cube inputs, standardized outcomes of every output
    and batch element; ``y`` holds the targets of every element, ``n`` each)."""
    atol = 1e-2
    if any(v < -atol or v > 1 + atol for v in x[0]):
        warnings.warn(
            "Data (input features) is not contained to the unit cube. "
            "Please consider min-max scaling the input data.",
            InputDataWarning,
            stacklevel=3,
        )
    values = list(y[0])
    k = len(values) // max(n, 1)
    means, stds = [], []
    for e in range(k):
        col = values[e * n : (e + 1) * n]
        mean = sum(col) / n
        means.append(mean)
        if n > 1:
            stds.append(math.sqrt(sum((v - mean) ** 2 for v in col) / (n - 1)))

    def fmt(v: list[float]) -> Any:
        return v[0] if len(v) == 1 else v

    if n <= 1:
        if max(abs(v) for v in means) > atol:
            warnings.warn(
                f"Data (outcome observations) is not standardized (mean = {fmt(means)}). "
                "Please consider scaling the input to zero mean and unit variance.",
                InputDataWarning,
                stacklevel=3,
            )
        return
    if any(abs(v) > atol for v in means) or any(abs(s - 1) > atol for s in stds):
        warnings.warn(
            f"Data (outcome observations) is not standardized (std = {fmt(stds)}, "
            f"mean = {fmt(means)}).Please consider scaling the input to zero mean and unit "
            "variance.",
            InputDataWarning,
            stacklevel=3,
        )


def _check_module_batch_shape(module: Any, aug: tuple[int, ...], what: str) -> None:
    """A module of a multi-output or batched model must have
    ``batch_shape = _aug_batch_shape`` (independent hyperparameters per output
    and batch element, like BoTorch's default modules)."""
    modules = [module]
    if isinstance(module, ScaleKernel):
        modules.append(module.base_kernel)
    for mod in modules:
        shape = tuple(getattr(mod, "batch_shape", ()))
        if shape == aug:
            continue
        if aug and not shape:
            raise UnsupportedError(
                f"`{what}` must have batch_shape={aug} (the model's _aug_batch_shape): "
                "botorch_rs fits independent hyperparameters for every output and batch "
                "element, as BoTorch does with its default modules. Modules shared by all "
                "outputs (batch_shape=()) cannot be fitted by BoTorch's batched fitting "
                "either."
            )
        raise BotorchTensorDimensionError(
            f"`{what}` has batch_shape={shape}, expected the model's _aug_batch_shape {aug}."
        )


class SingleTaskGP(Model):
    r"""A single-task exact GP with BoTorch's defaults.

    Without ``train_Yvar`` the noise level is inferred (``GaussianLikelihood``
    with a ``LogNormal(-4, 1)`` noise prior); with ``train_Yvar`` the given
    observation noise is used. The default covariance module is an ARD RBF
    kernel with the dimension-scaled lengthscale prior of Hvarfner et al.
    (2024), the default outcome transform is ``Standardize(m, batch_shape)``
    and inputs are not transformed unless ``input_transform`` is given
    (``Normalize``, ``InputPerturbation`` or a ``ChainedInputTransform`` of
    them). With an ``InputPerturbation`` (``n_w`` perturbations), the
    posterior of ``q`` inputs is over the ``q * n_w`` perturbed points, as in
    BoTorch.

    Training data of shape ``batch_shape x n x d`` / ``batch_shape x n x m``
    give ``batch_shape x m`` independent GPs (BoTorch's ``_aug_batch_shape``),
    each with its own hyperparameters, standardization and noise; the posterior
    of ``batch_shape x q x d`` inputs is over ``q`` points and ``m`` outputs.

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
        input_transform: InputTransform | None = None,
    ) -> None:
        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        yvar = to_flat(train_Yvar, "train_Yvar") if train_Yvar is not None else None
        _validate_tensor_args(x[1], y[1], yvar[1] if yvar is not None else None)
        if yvar is not None and yvar[1] != y[1]:  # broadcast trailing dimensions
            reps = len(y[0]) // max(len(yvar[0]), 1)
            yvar = (yvar[0] * reps, y[1])
        batch = tuple(x[1][:-2])
        if tuple(y[1][:-2]) != batch:
            raise BotorchTensorDimensionError(
                f"Expected train_X and train_Y to have the same batch shape, got "
                f"{batch} and {tuple(y[1][:-2])}."
            )
        d = x[1][-1]
        m = y[1][-1]
        aug = batch + ((m,) if m > 1 else ())
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
            _check_module_batch_shape(likelihood, aug, "likelihood")
        if covar_module is not None:
            if not isinstance(covar_module, Kernel):
                raise UnsupportedError(
                    "`covar_module` must be a botorch_rs kernel (RBFKernel, MaternKernel(nu=2.5) "
                    "or ScaleKernel), e.g. from botorch_rs.models.utils.gpytorch_modules."
                )
            _check_module_batch_shape(covar_module, aug, "covar_module")
        if mean_module is not None:
            if not isinstance(mean_module, ConstantMean):
                raise UnsupportedError("Only a ConstantMean `mean_module` is supported.")
            _check_module_batch_shape(mean_module, aug, "mean_module")
        if _is_default(outcome_transform):
            ot_obj: Standardize | None = Standardize(m=m, batch_shape=batch)
            ot_rs, std_default = None, True
        elif outcome_transform is None:
            ot_obj, ot_rs, std_default = None, None, False
        elif isinstance(outcome_transform, Standardize):
            if outcome_transform._m != m:
                raise BotorchTensorDimensionError(
                    f"The outcome transform is for m = {outcome_transform._m} outputs, got m = {m}."
                )
            if outcome_transform.batch_shape != batch:
                raise RuntimeError(
                    f"Expected Y.shape[:-2] to be {outcome_transform.batch_shape}, matching the "
                    f"`batch_shape` argument to `Standardize`, but got Y.shape[:-2]={batch}."
                )
            ot_obj, ot_rs, std_default = outcome_transform, outcome_transform._rs, False
        else:
            raise UnsupportedError("`outcome_transform` must be a Standardize, DEFAULT or None.")
        chain: list[Any] | None = None
        if isinstance(input_transform, (InputPerturbation, ChainedInputTransform)):
            chain = chain_steps(input_transform, d)
        elif input_transform is not None:
            if not isinstance(input_transform, Normalize):
                raise UnsupportedError(
                    "`input_transform` must be a Normalize, an InputPerturbation or a "
                    "ChainedInputTransform of them."
                )
            tf_batch = input_transform.batch_shape
            if tf_batch and tf_batch != batch:
                raise BotorchTensorDimensionError(
                    f"The input transform has batch shape {tf_batch} but the training data "
                    f"have batch shape {batch}."
                )
            if tf_batch and m > 1:
                raise UnsupportedError(
                    "A Normalize with a batch shape is not supported for batched models with "
                    "several outputs: BoTorch broadcasts its per-batch bounds against the "
                    "output dimension of such models. Use Normalize(d) (bounds shared by all "
                    "batch elements) or one output per model."
                )
        covar = (
            covar_module
            if covar_module is not None
            else get_covar_module_with_dim_scaled_prior(ard_num_dims=d, batch_shape=aug)
        )
        covar_spec = covar._spec(d) if covar_module is not None else None
        self._rs = _rs.SingleTaskGP(
            x,
            y,
            yvar,
            input_transform._rs if isinstance(input_transform, Normalize) else None,
            ot_rs,
            std_default,
            covar_spec,
            likelihood._spec if likelihood is not None else None,
            input_chain=chain,
        )
        self._kind = kind
        # GP modules bound to the model's hyperparameters.
        if likelihood is None:
            likelihood = (
                FixedNoiseGaussianLikelihood._create(aug)
                if yvar is not None
                else get_gaussian_likelihood_with_lognormal_prior(batch_shape=aug)
            )
        likelihood._bind(self)
        covar._bind(self)
        mean = mean_module if mean_module is not None else ConstantMean(batch_shape=aug)
        mean._bind(self)
        self.likelihood: Likelihood = likelihood
        self.covar_module: Kernel = covar
        self.mean_module: ConstantMean = mean
        if isinstance(input_transform, Normalize):
            fitted_input = self._rs.input_transform()
            assert fitted_input is not None
            input_transform._set_fitted(fitted_input, kind)
            self.input_transform: InputTransform = input_transform
        elif input_transform is not None:
            state = self._rs.input_chain_state()
            assert state is not None
            apply_chain_state(input_transform, state, kind)
            self.input_transform = input_transform
        if ot_obj is not None:
            fitted_outcome = self._rs.outcome_transform()
            assert fitted_outcome is not None
            ot_obj._set_fitted(fitted_outcome, kind_of(train_Y))
            self.outcome_transform = ot_obj
        if covar_module is None:
            _validate_input_scaling(
                self._rs.transformed_train_inputs(),
                self._rs.transformed_train_targets(),
                self._rs.num_train,
            )
        self.training = True

    # -- data --------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return self._rs.num_outputs

    @property
    def _num_outputs(self) -> int:
        return self._rs.num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the training data (``train_X.shape[:-2]``)."""
        return tuple(self._rs.batch_shape)

    @property
    def _input_batch_shape(self) -> tuple[int, ...]:
        return self.batch_shape

    @property
    def _aug_batch_shape(self) -> tuple[int, ...]:
        """The batch shape followed by the number of outputs if it exceeds one."""
        return tuple(self._rs.aug_batch_shape)

    @property
    def train_inputs(self) -> tuple[Any]:
        """``(train_X,)`` in the original input space; with several outputs
        expanded to ``batch_shape x m x n x d`` like BoTorch's."""
        data, shape = self._rs.train_inputs()
        m = self.num_outputs
        if m == 1:
            return (from_flat(data, shape, self._kind),)
        n, d = shape[-2], shape[-1]
        block = n * d
        expanded: list[float] = []
        for b in range(len(data) // block):
            expanded.extend(data[b * block : (b + 1) * block] * m)
        return (from_flat(expanded, (*shape[:-2], m, n, d), self._kind),)

    @property
    def train_targets(self) -> Any:
        """The training targets after the outcome transform, shape
        ``_aug_batch_shape x n``."""
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

    def _posterior_args(
        self, output_indices: Any, posterior_transform: Any
    ) -> tuple[list[int] | None, tuple[list[float], float] | None]:
        from botorch_rs.acquisition.objective import transform_arg

        m = self.num_outputs
        oi = None
        if output_indices is not None:
            oi = [int(t) for t in output_indices]
            if not oi or any(not -m <= t < m for t in oi):
                raise BotorchTensorDimensionError(
                    f"output_indices {list(output_indices)} are invalid for a model with "
                    f"{m} outputs."
                )
            oi = [t % m for t in oi]
            if oi == list(range(m)):
                oi = None
        return oi, transform_arg(posterior_transform)

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """The posterior at ``X`` (``batch_shape x q x d``), in the original
        outcome space, over the outputs ``output_indices`` (all by default),
        scalarized by ``posterior_transform`` if given; with
        ``observation_noise=True`` the likelihood noise is added (the mean
        training noise for fixed-noise models)."""
        if not isinstance(observation_noise, bool):
            raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
        oi, transform = self._posterior_args(output_indices, posterior_transform)
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
        return GPyTorchPosterior(
            self._rs,
            flat,
            kind,
            observation_noise,
            num_outputs=self.num_outputs,
            model_batch_shape=self.batch_shape,
            output_indices=oi,
            transform=transform,
            points_per_input=self._rs.points_per_input,
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle that acquisition functions are built on."""
        from botorch_rs.acquisition.objective import transform_arg

        return _rs.Model.from_single_task_gp(
            self._rs, transform_arg(posterior_transform, self.num_outputs)
        )

    # -- conditioning and fantasies ----------------------------------------

    def _with_rs(self, rs: _rs.SingleTaskGP) -> SingleTaskGP:
        """A copy of this model (with its own modules) around the Rust model
        ``rs``, in eval mode (e.g. a conditioned or fantasy model)."""
        new = self.__class__.__new__(self.__class__)
        memo: dict[int, Any] = {id(self): new}
        for key, value in self.__dict__.items():
            new.__dict__[key] = rs if key == "_rs" else copy.deepcopy(value, memo)
        new.training = False
        return new

    def condition_on_observations(
        self, X: Any, Y: Any, noise: Any = None, **kwargs: Any
    ) -> SingleTaskGP:
        r"""The model conditioned on new observations (BoTorch's
        ``condition_on_observations``): ``X`` (``batch_shape x n' x d``),
        ``Y`` (``batch_shape' x n' x m``, original outcome space) and optional
        observation-noise variances ``noise`` (shape of ``Y``, in the
        *outcome-transformed* space, as BoTorch assumes). The result has the
        new points appended to the training data, the same hyperparameters
        and the fitted transforms (applied in eval mode), and the broadcast
        batch shape of the model, ``X`` and ``Y``. Explicit noise of a model
        that infers its noise level enters the posterior mean and covariance
        (meta-pytorch/botorch#3396; BoTorch main uses it for the covariance
        only).

        As in BoTorch, the model must be in eval mode (e.g. after a call of
        ``posterior``).
        """
        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        if self.training:
            raise RuntimeError(
                "Fantasy observations can only be added after making predictions with a model "
                "so that all test independent caches exist. Call the model on some data first!"
            )
        x = to_flat(X, "X")
        y = to_flat(Y, "Y")
        tb, ib = len(y[1]) - 2, len(x[1]) - 2
        if not (tb == ib or tb == ib + 1):
            raise RuntimeError(
                f"Unsupported batch shapes: The target batch shape ({tuple(y[1][:-2])}) must "
                "have either the same dimension as or one more dimension than the input batch "
                f"shape ({tuple(x[1][:-2])})"
            )
        nz = None if noise is None else to_flat(noise, "noise")
        return self._with_rs(_rs.condition_on_observations(self._rs, x, y, nz))

    def fantasize(
        self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> SingleTaskGP:
        r"""A fantasy model (BoTorch's ``fantasize``): ``n_f`` sets of
        observations at ``X`` (``batch_shape x n' x d``) are sampled from the
        posterior with observation noise using ``sampler`` (a
        ``SobolQMCNormalSampler`` with ``sample_shape=(n_f,)``; its base samples
        are shared by the batch elements), and the model is conditioned on
        each of them. The fantasy model has the batch shape ``n_f x
        broadcast(batch_shape, model.batch_shape)``. ``observation_noise``
        (``model_batch x 1 x m`` or ``model_batch x n' x m``, outcome-transformed
        space) defaults to the inferred noise level, or the mean training
        noise of a fixed-noise model."""
        from botorch_rs.exceptions.errors import DeprecationError
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        if isinstance(observation_noise, bool):
            raise DeprecationError(
                "`fantasize` no longer accepts a boolean for `observation_noise`."
            )
        if not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        self.eval()
        x = to_flat(X, "X")
        nz = None if observation_noise is None else to_flat(observation_noise, "observation_noise")
        return self._with_rs(_rs.fantasize(self._rs, x, sampler._rs, nz))

    # -- parameters --------------------------------------------------------

    def _raw_param(self, suffix: str) -> list[float]:
        values = self._rs.raw_parameters()
        offset = 0
        for name, size in self._rs.parameter_names():
            if name.endswith(suffix):
                return values[offset : offset + size]
            offset += size
        raise AttributeError(f"The model has no parameter {suffix!r}.")

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        data, shape = to_flat(value, suffix)
        values = self._rs.raw_parameters()
        offset = 0
        shapes = dict(self._rs.parameter_shapes())
        for name, size in self._rs.parameter_names():
            if name.endswith(suffix):
                target = tuple(shapes[name])
                try:
                    new = broadcast_to(data, shape, target)
                except ValueError:
                    raise BotorchTensorDimensionError(
                        f"{name} has shape {target}, got a value of shape {tuple(shape)}."
                    ) from None
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
        for name, shape in self._rs.parameter_shapes():
            size = math.prod(shape)
            data = values[offset : offset + size]
            offset += size
            yield name, from_flat(data, tuple(shape), self._kind)

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
        extra = ""
        if self.num_outputs > 1:
            extra += f", m={self.num_outputs}"
        if self.batch_shape:
            extra += f", batch_shape={self.batch_shape}"
        return (
            f"SingleTaskGP(n={self._rs.num_train}, d={self._rs.dim}{extra}, "
            f"covar_module={self.covar_module!r})"
        )


__all__ = ["SingleTaskGP"]
