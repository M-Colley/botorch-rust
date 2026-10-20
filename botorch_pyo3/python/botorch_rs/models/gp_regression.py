"""``SingleTaskGP`` (``botorch.models.gp_regression``)."""

from __future__ import annotations

import copy
import math
import sys
import warnings
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Pair, broadcast_to, from_flat, from_pair, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.kernels import Kernel
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, Likelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch_rs.models.transforms.input import (
    InputTransform,
    Normalize,
    apply_chain_states,
    chain_spec,
)
from botorch_rs.models.transforms.outcome import (
    OutcomeTransform,
    Standardize,
    apply_outcome_states,
    is_plain_standardize,
    outcome_chains,
)
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


def _kernels_with_parameters(kernel: Kernel) -> list[Kernel]:
    """The kernels of a tree that have parameters of their own (not the base
    kernel of an ``OrthogonalAdditiveKernel``, whose batch shape has an extra
    input dimension)."""
    out = []
    if kernel._own_params():
        out.append(kernel)
    if type(kernel).__name__ == "OrthogonalAdditiveKernel":
        return out
    for _, child in kernel._children():
        out.extend(_kernels_with_parameters(child))
    return out


def _check_module_batch_shape(module: Any, aug: tuple[int, ...], what: str) -> None:
    """A module of a multi-output or batched model must have
    ``batch_shape = _aug_batch_shape`` (independent hyperparameters per output
    and batch element, like BoTorch's default modules); for a kernel tree,
    every kernel with parameters."""
    modules: list[Any] = (
        list(_kernels_with_parameters(module)) if isinstance(module, Kernel) else [module]
    )
    seen: set[int] = set()
    for mod in modules:
        if id(mod) in seen:
            raise UnsupportedError(
                f"The same kernel object appears twice in `{what}`; botorch_rs does not share "
                "parameters between the parts of a kernel tree."
            )
        seen.add(id(mod))
        shape = tuple(
            mod._batch_shape if isinstance(mod, Kernel) else getattr(mod, "batch_shape", ())
        )
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


def _rs_of(obj: Any) -> Any:
    """The extension object of a prior or constraint (``None`` for ``None``)."""
    return None if obj is None else obj._rs


def _element_saas_taus(covar: Kernel, num_elements: int) -> list[list[float]] | None:
    """The fixed SAAS ``tau`` of every SAAS kernel of the tree (pre-order) for
    every model element (``None`` without SAAS kernels with a fixed ``tau``)."""
    columns = []
    for k in covar._all_kernels():
        saas = getattr(k, "_saas", None)
        if saas is not None and saas.fixed_taus is not None:
            values = saas.fixed_taus
            if len(values) == 1:
                values = values * num_elements
            if len(values) != num_elements:
                raise BotorchTensorDimensionError(
                    f"Expected one SAAS tau per model element ({num_elements}), got {len(values)}."
                )
            columns.append(values)
    if not columns:
        return None
    return [[col[e] for col in columns] for e in range(num_elements)]


def _rebatch(model: Any, old_aug: tuple[int, ...]) -> None:
    """Updates the batch shapes of the modules of ``model`` (a subset or a
    batch element of a model with ``_aug_batch_shape`` ``old_aug``) and the
    fitted outcome transform."""
    new_aug = model._aug_batch_shape
    for module in _modules_of(model):
        if isinstance(module, Kernel):
            if tuple(module._batch_shape) == old_aug:
                module._batch_shape = new_aug
        elif tuple(getattr(module, "batch_shape", ())) == old_aug:
            module.batch_shape = new_aug
    ot = getattr(model, "outcome_transform", None)
    if ot is not None:
        states = model._rs.outcome_chain_states()
        if states is not None:
            # A general outcome transform (phase 6b): the fitted steps of the
            # remaining elements.
            apply_outcome_states(
                ot, states, model.num_outputs, tuple(model._rs.batch_shape), model._kind
            )
            return
        fitted = model._rs.outcome_transform()
        assert fitted is not None
        ot._set_fitted(fitted, ot._kind)


def _modules_of(model: Any) -> list[Any]:
    """The GP modules of a model (likelihood, mean and every kernel)."""
    out: list[Any] = []
    for name in ("likelihood", "mean_module"):
        mod = getattr(model, name, None)
        if mod is not None:
            out.append(mod)
    covar = getattr(model, "covar_module", None)
    if isinstance(covar, Kernel):
        out.extend(covar._all_kernels())
    return out


class SingleTaskGP(BatchedMultiOutputGPyTorchModel):
    r"""A single-task exact GP with BoTorch's defaults.

    Without ``train_Yvar`` the noise level is inferred (``GaussianLikelihood``
    with a ``LogNormal(-4, 1)`` noise prior); with ``train_Yvar`` the given
    observation noise is used. The default covariance module is an ARD RBF
    kernel with the dimension-scaled lengthscale prior of Hvarfner et al.
    (2024), the default outcome transform is ``Standardize(m, batch_shape)``
    and inputs are not transformed unless ``input_transform`` is given (any
    of :mod:`botorch_rs.models.transforms.input`, e.g. ``Normalize``,
    ``Warp`` with concentrations fitted with the GP, ``InputPerturbation``,
    or a ``ChainedInputTransform`` of them). With a one-to-many input
    transform (``InputPerturbation`` with ``n_w`` perturbations), the
    posterior of ``q`` inputs is over the ``q * n_w`` perturbed points, as in
    BoTorch. With a nonlinear outcome transform (``Log``, ``Power``,
    ``Bilog``), the posterior is a ``TransformedPosterior``.

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

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if cls is SingleTaskGP:
            if backend not in ("auto", "native", "torch"):
                raise ValueError("backend must be 'auto', 'native', or 'torch'.")
            values = (train_X, train_Y, kwargs.get("train_Yvar"), *args[:1])
            torch = sys.modules.get("torch")
            modules = (
                *args[1:],
                *(
                    kwargs.get(name)
                    for name in (
                        "likelihood",
                        "covar_module",
                        "mean_module",
                        "input_transform",
                        "outcome_transform",
                    )
                ),
            )
            tensor_backend = backend == "torch" or (
                backend == "auto"
                and (
                    any(
                        getattr(value, "requires_grad", False)
                        or getattr(getattr(value, "device", None), "type", "cpu") != "cpu"
                        for value in values
                    )
                    or (
                        torch is not None
                        and any(isinstance(module, torch.nn.Module) for module in modules)
                    )
                )
            )
            if tensor_backend:
                from botorch_rs.models._torch_backend import TorchSingleTaskGP
                from botorch_rs.models.likelihoods.sparse_outlier_noise import (
                    SparseOutlierGaussianLikelihood,
                )

                likelihood = kwargs.get("likelihood", args[1] if len(args) > 1 else None)
                if isinstance(likelihood, SparseOutlierGaussianLikelihood):
                    from botorch_rs.models._torch_backend_relevance import TorchSparseSingleTaskGP

                    return object.__new__(TorchSparseSingleTaskGP)
                return object.__new__(TorchSingleTaskGP)
        return object.__new__(cls)

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
        *,
        backend: str = "auto",
    ) -> None:
        from botorch_rs.models.likelihoods.sparse_outlier_noise import (
            SparseOutlierGaussianLikelihood,
        )

        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        yvar = to_flat(train_Yvar, "train_Yvar") if train_Yvar is not None else None
        _validate_tensor_args(x[1], y[1], yvar[1] if yvar is not None else None)
        if yvar is not None and yvar[1] != y[1]:  # broadcast trailing dimensions
            reps = len(y[0]) // max(len(yvar[0]), 1)
            yvar = (yvar[0] * reps, y[1])
        batch, aug = self.get_batch_dimensions(train_X, train_Y)
        if tuple(y[1][:-2]) != batch:
            raise BotorchTensorDimensionError(
                f"Expected train_X and train_Y to have the same batch shape, got "
                f"{batch} and {tuple(y[1][:-2])}."
            )
        d = x[1][-1]
        m = y[1][-1]
        if likelihood is not None:
            if not isinstance(likelihood, (GaussianLikelihood, SparseOutlierGaussianLikelihood)):
                raise UnsupportedError(
                    "`likelihood` must be a botorch_rs GaussianLikelihood "
                    "(e.g. get_gaussian_likelihood_with_gamma_prior())."
                )
            if isinstance(likelihood, SparseOutlierGaussianLikelihood) and not isinstance(
                likelihood.noise_covar.base_noise, GaussianLikelihood
            ):
                raise UnsupportedError(
                    "A custom sparse likelihood requires native Gaussian base noise; for fixed noise use RobustRelevancePursuitSingleTaskGP(train_Yvar=...)."
                )
            if yvar is not None:
                raise UnsupportedError(
                    "A custom likelihood cannot be combined with `train_Yvar` in botorch_rs."
                )
            _check_module_batch_shape(likelihood, aug, "likelihood")
        if covar_module is not None:
            if not isinstance(covar_module, Kernel):
                raise UnsupportedError(
                    "`covar_module` must be a botorch_rs kernel (from botorch_rs.kernels, "
                    "botorch_rs.models.kernels or botorch_rs.models.utils.gpytorch_modules)."
                )
            if covar_module._model is not None:
                raise UnsupportedError(
                    "This kernel is already part of another model; create a new one for each model."
                )
            _check_module_batch_shape(covar_module, aug, "covar_module")
        if mean_module is not None:
            if not isinstance(mean_module, ConstantMean):
                raise UnsupportedError("Only a ConstantMean `mean_module` is supported.")
            _check_module_batch_shape(mean_module, aug, "mean_module")
        ochains: list[list[Any]] | None = None
        if _is_default(outcome_transform):
            ot_obj: OutcomeTransform | None = Standardize(m=m, batch_shape=batch)
            ot_rs, std_default = None, True
        elif outcome_transform is None:
            ot_obj, ot_rs, std_default = None, None, False
        elif isinstance(outcome_transform, OutcomeTransform) and not is_plain_standardize(
            outcome_transform
        ):
            if isinstance(outcome_transform, Standardize):
                if outcome_transform._m != m:
                    raise RuntimeError(
                        f"Wrong output dimension. Y.size(-1) is {m}; expected "
                        f"{outcome_transform._m}."
                    )
            outcome_transform.train()
            ochains = outcome_chains(outcome_transform, m, batch)
            ot_obj, ot_rs, std_default = outcome_transform, None, False
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
            raise UnsupportedError(
                "`outcome_transform` must be a botorch_rs outcome transform, DEFAULT or None."
            )
        chain: list[Any] | None = None
        chains: list[list[Any]] | None = None
        spec = None
        plain_normalize = isinstance(input_transform, Normalize) and input_transform._plain
        if input_transform is not None and not plain_normalize:
            if not isinstance(input_transform, InputTransform):
                raise UnsupportedError(
                    "`input_transform` must be a botorch_rs input transform (from "
                    "botorch_rs.models.transforms.input)."
                )
            spec = chain_spec(input_transform, d, batch)
            steps = [[st.rs for st in c] for c in spec.chains]
            if spec.per_batch:
                chains = steps
            else:
                chain = steps[0]
        elif input_transform is not None:
            assert isinstance(input_transform, Normalize)
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
        covar_spec = None
        covar_tree = None
        if covar_module is not None:
            covar_spec = covar._standard_spec(d)
            if covar_spec is None:
                covar_tree = covar._node()
        training = prefixes = None
        if spec is not None:
            training = [st.training for st in spec.chains[0]]
            prefixes = [st.prefix for st in spec.chains[0]]
        native_normalize = (
            input_transform._rs
            if isinstance(input_transform, Normalize) and plain_normalize
            else None
        )
        self._rs = _rs.SingleTaskGP(
            x,
            y,
            yvar,
            native_normalize,
            ot_rs,
            std_default,
            covar_spec,
            likelihood._spec if likelihood is not None else None,
            input_chain=chain,
            covar_tree=covar_tree,
            input_chain_training=training,
            input_chain_prefixes=prefixes,
            input_chains=chains,
            outcome_chains=ochains,
            input_chain_params=spec.batch_params if spec is not None else None,
            mean_prior=_rs_of(getattr(mean_module, "mean_prior", None)),
            mean_constraint=_rs_of(getattr(mean_module, "raw_constant_constraint", None)),
        )
        if covar_tree is not None:
            taus = _element_saas_taus(covar, math.prod(aug))
            if taus is not None:
                self._rs.set_saas_taus(taus)
        self._kind = kind
        # GP modules bound to the model's hyperparameters.
        if likelihood is None:
            likelihood = (
                FixedNoiseGaussianLikelihood._create(aug)
                if yvar is not None
                else get_gaussian_likelihood_with_lognormal_prior(batch_shape=aug)
            )
        likelihood._bind(self)
        covar._bind(self, "covar_module.")
        mean = mean_module if mean_module is not None else ConstantMean(batch_shape=aug)
        mean._bind(self)
        self.likelihood: Likelihood = likelihood
        self.covar_module: Kernel = covar
        self.mean_module: ConstantMean = mean
        if plain_normalize:
            assert isinstance(input_transform, Normalize)
            fitted_input = self._rs.input_transform()
            assert fitted_input is not None
            input_transform._set_fitted(fitted_input, kind)
            self.input_transform: InputTransform = input_transform
        elif input_transform is not None:
            assert spec is not None
            states = self._rs.input_chain_states()
            assert states is not None
            apply_chain_states(spec, states, kind, batch)
            input_transform._bind(self, "input_transform.")
            self.input_transform = input_transform
        if ot_obj is not None:
            if ochains is not None:
                states = self._rs.outcome_chain_states()
                assert states is not None
                apply_outcome_states(ot_obj, states, m, batch, kind_of(train_Y))
            else:
                assert isinstance(ot_obj, Standardize)
                fitted_outcome = self._rs.outcome_transform()
                assert fitted_outcome is not None
                ot_obj._set_fitted(fitted_outcome, kind_of(train_Y))
            self.outcome_transform = ot_obj
        from botorch_rs.settings import validate_input_scaling

        if covar_module is None:
            if validate_input_scaling.on():
                _validate_input_scaling(
                    self._rs.construction_train_inputs(),
                    self._rs.transformed_train_targets(),
                    self._rs.num_train,
                )
            # Used for subsetting along the output dimension (BoTorch's
            # `Model.subset_output`; botorch_rs subsets any covariance module).
            self._subset_batch_dict = {
                "mean_module.raw_constant": -1,
                "covar_module.raw_lengthscale": -3,
            }
            if yvar is None:
                self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        self.training = True

        from botorch_rs._buffers import adopt_buffer_dtype

        adopt_buffer_dtype(self, kind)
        self._torch_parameters_ready = True

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

    def _replace_transformed_train_data(
        self, targets: Any, noise: Any = None, strict: bool = True
    ) -> None:
        """Atomically replace internal targets and optional fixed noise.

        Fitted transforms and hyperparameters are preserved. The number of
        observations must continue to match the stored training inputs.
        """
        self._rs.replace_transformed_train_data(
            to_flat(targets, "targets"), None if noise is None else to_flat(noise, "noise"), strict
        )

    def set_train_data(self, inputs: Any = None, targets: Any = None, strict: bool = True) -> None:
        """Replace transformed targets while keeping existing training inputs."""
        if inputs is not None:
            raise UnsupportedError(
                "Replacing training inputs is not supported yet; construct a new model."
            )
        if targets is not None:
            self._replace_transformed_train_data(targets, strict=strict)

    # -- modes -------------------------------------------------------------

    def train(self, mode: bool = True) -> SingleTaskGP:
        self._sync_torch_parameters()
        self.training = mode
        self.likelihood.train(mode)
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    # -- posterior ---------------------------------------------------------

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle that acquisition functions are built on."""
        from botorch_rs.acquisition.objective import transform_arg

        self._sync_torch_parameters()
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
            if (
                key == "_torch_parameter_requires_grad"
                and "_torch_parameter_state" in self.__dict__
            ):
                continue
            if key == "_torch_parameter_state":
                new.__dict__["_torch_parameter_requires_grad"] = {
                    name: p.requires_grad for name, p in value.entries
                }
                continue
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
        self._sync_torch_parameters()
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
        from botorch_rs.sampling.normal import NormalMCSampler

        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        if isinstance(observation_noise, bool):
            raise DeprecationError(
                "`fantasize` no longer accepts a boolean for `observation_noise`."
            )
        if not isinstance(sampler, NormalMCSampler):
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
        self.eval()
        x = to_flat(X, "X")
        nz = None if observation_noise is None else to_flat(observation_noise, "observation_noise")
        return self._with_rs(_rs.fantasize(self._rs, x, sampler._rs, nz))

    def subset_output(self, idcs: list[int]) -> SingleTaskGP:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices (with
            the hyperparameters, noise and outcome transform of these
            outputs).

        BoTorch subsets only the parameters listed in ``_subset_batch_dict``
        and raises ``NotImplementedError`` for models without it (e.g. a
        ``SingleTaskGP`` with a custom ``covar_module``); botorch_rs subsets
        every module, since all of them have independent parameters per
        output.
        """
        self._sync_torch_parameters()
        idcs = [int(i) for i in idcs]
        m = self.num_outputs
        if m == len(idcs) and idcs == list(range(m)):
            return copy.deepcopy(self)
        if not idcs or any(not 0 <= i < m for i in idcs):
            raise IndexError(f"index out of range in self (output indices {idcs}, m = {m})")
        old_aug = self._aug_batch_shape
        new = self._with_rs(self._rs.subset_output(idcs))
        new.training = self.training
        ot = getattr(new, "outcome_transform", None)
        if ot is not None and not is_plain_standardize(ot):
            # BoTorch's `outcome_transform.subset_output(idcs)` (re-indexed
            # `outputs`, meta-pytorch/botorch#3380).
            ot = ot.subset_output(idcs)
            new.outcome_transform = ot
        _rebatch(new, old_aug)
        if ot is not None and is_plain_standardize(ot):
            ot._m = len(idcs)
        return new

    def _member(self, k: int) -> SingleTaskGP:
        """Batch element ``k`` of a batched model (row-major) as an
        unbatched ``SingleTaskGP`` with its own modules, e.g. a member of an
        ensemble of GPs."""
        self._sync_torch_parameters()
        old_aug = self._aug_batch_shape
        new = SingleTaskGP.__new__(SingleTaskGP)
        memo: dict[int, Any] = {id(self): new}
        for key, value in self.__dict__.items():
            if (
                key == "_torch_parameter_requires_grad"
                and "_torch_parameter_state" in self.__dict__
            ):
                continue
            if key == "_rs":
                new.__dict__[key] = value.batch_element(k)
            elif key in ("taus", "_ensemble_cache", "_torch_parameter_state"):
                if key == "_torch_parameter_state":
                    new.__dict__["_torch_parameter_requires_grad"] = {
                        name: p.requires_grad for name, p in value.entries
                    }
                continue
            else:
                new.__dict__[key] = copy.deepcopy(value, memo)
        new.training = self.training
        ot = getattr(new, "outcome_transform", None)
        if ot is not None and not is_plain_standardize(ot):
            raise UnsupportedError(
                "The members of a batched model with an outcome transform other than "
                "Standardize are not supported."
            )
        _rebatch(new, old_aug)
        if ot is not None:
            ot._batch_shape = ()
        return new

    # -- parameters --------------------------------------------------------

    def _sync_torch_parameters(self) -> Any:
        """Synchronize the persistent leaves of ordinary Torch SingleTaskGPs."""
        if (
            type(self) is not SingleTaskGP
            or self._kind.lib != "torch"
            or not self.__dict__.get("_torch_parameters_ready", False)
        ):
            return None
        state = self.__dict__.get("_torch_parameter_state")
        if state is None:
            from botorch_rs.models._torch_parameters import TorchParameterState

            state = self._torch_parameter_state = TorchParameterState(self)
        else:
            state.sync(self)
        return state

    def _param_index(self, suffix: str) -> tuple[str, int, int]:
        """The full name, offset and size of the parameter named ``suffix``
        (a full name, or the end of one)."""
        names = self._rs.parameter_names()
        for exact in (True, False):
            offset = 0
            for name, size in names:
                if name == suffix if exact else name.endswith(suffix):
                    return name, offset, size
                offset += size
        raise AttributeError(f"The model has no parameter {suffix!r}.")

    def _raw_param(self, suffix: str) -> list[float]:
        self._sync_torch_parameters()
        _, offset, size = self._param_index(suffix)
        return self._rs.raw_parameters()[offset : offset + size]

    def _raw_param_array(self, suffix: str) -> Any:
        """The raw parameter named ``suffix`` as an array of its shape."""
        state = self._sync_torch_parameters()
        if state is not None:
            return state.parameter(self._param_index(suffix)[0])
        return from_flat(self._raw_param(suffix), self._raw_param_shape(suffix), self._kind)

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        name, _, _ = self._param_index(suffix)
        return tuple(dict(self._rs.parameter_shapes())[name])

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        self._sync_torch_parameters()
        data, shape = to_flat(value, suffix)
        name, offset, size = self._param_index(suffix)
        values = self._rs.raw_parameters()
        target = tuple(dict(self._rs.parameter_shapes())[name])
        try:
            new = broadcast_to(data, shape, target)
        except ValueError:
            # GPyTorch's `Module.initialize`: `expand_as`, else `view_as`.
            if len(data) != size:
                raise BotorchTensorDimensionError(
                    f"{name} has shape {target}, got a value of shape {tuple(shape)}."
                ) from None
            new = [float(v) for v in data]
        values[offset : offset + size] = new
        self._rs.set_raw_parameters(values)
        self._sync_torch_parameters()

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every hyperparameter, with BoTorch's names,
        order and shapes."""
        state = self._sync_torch_parameters()
        if state is not None:
            yield from state.entries
            return
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

    def requires_grad_(self, requires_grad: bool = True) -> SingleTaskGP:
        """Set trainability of persistent CPU Torch parameters in place."""
        state = self._sync_torch_parameters()
        if state is None:
            raise UnsupportedError(
                "Persistent parameter trainability requires a CPU Torch SingleTaskGP."
            )
        for _, parameter in state.entries:
            parameter.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear accumulated gradients on the persistent Torch parameters."""
        state = self._sync_torch_parameters()
        if state is not None:
            for _, parameter in state.entries:
                if parameter.grad is not None:
                    if set_to_none:
                        parameter.grad = None
                    else:
                        if parameter.grad.grad_fn is not None:
                            parameter.grad.detach_()
                        else:
                            parameter.grad.requires_grad_(False)
                        parameter.grad.zero_()

    def posterior_parameter_vjp(
        self,
        X: Any,
        mean_adjoint: Any,
        covariance_adjoint: Any,
        observation_noise: bool = False,
    ) -> Any:
        """Native posterior derivative in flat ``named_parameters`` order.

        ``mean_adjoint`` has the posterior mean's ``... x q x m`` layout.
        ``covariance_adjoint`` uses the joint normal's output-major
        ``... x (q*m) x (q*m)`` layout. Model and query batch broadcasting
        is reduced into the original model parameters. The result is an
        explicit first-order gradient; no prior-density terms are added.
        """
        self._sync_torch_parameters()
        return from_pair(
            self._rs.posterior_parameter_vjp(
                to_flat(X, "X"),
                to_flat(mean_adjoint, "mean_adjoint"),
                to_flat(covariance_adjoint, "covariance_adjoint"),
                observation_noise,
            ),
            self._kind,
        )

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
        self._sync_torch_parameters()
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if (
                key == "_torch_parameter_requires_grad"
                and "_torch_parameter_state" in self.__dict__
            ):
                continue
            if key == "_torch_parameter_state":
                new.__dict__["_torch_parameter_requires_grad"] = {
                    name: p.requires_grad for name, p in value.entries
                }
                continue
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


def _model_posterior(
    source: Any,
    flat: Any,
    kind: Any,
    observation_noise: bool,
    num_outputs: int,
    model_batch_shape: tuple[int, ...],
    output_indices: list[int] | None,
    transform: Any,
    points_per_input: int,
    transformed: list[bool],
    is_list: bool,
    query_inputs: Any = None,
    parameter_owner: Any = None,
) -> Any:
    """The posterior of a model: a ``GPyTorchPosterior``, or a
    ``TransformedPosterior`` if a selected output has a nonlinear outcome
    transform (BoTorch's ``untransform_posterior``)."""
    from botorch_rs.posteriors.transformed import _ModelTransformedPosterior

    selected = transformed if output_indices is None else [transformed[t] for t in output_indices]
    args = (source, flat, kind, observation_noise)
    kwargs: dict[str, Any] = {
        "num_outputs": num_outputs,
        "model_batch_shape": model_batch_shape,
        "output_indices": output_indices,
        "transform": transform,
        "points_per_input": points_per_input,
        "query_inputs": query_inputs,
        "parameter_owner": parameter_owner,
    }
    native = GPyTorchPosterior(*args, **kwargs)
    if not any(selected):
        return native
    if transform is not None:
        if not is_list:
            raise UnsupportedError(
                "scalarize_posterior only works with a posterior that has an attribute "
                "`distribution`, such as a GPyTorchPosterior, or a posterior that contains "
                "sub-posteriors in an attribute `posteriors`, as in a PosteriorList."
            )
        return native
    base = GPyTorchPosterior(*args, base=True, **kwargs)
    return _ModelTransformedPosterior(native, base, moments=True)


__all__ = ["SingleTaskGP"]
