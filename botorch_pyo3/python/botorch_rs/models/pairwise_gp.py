r"""
Preference Learning with Gaussian Process (``botorch.models.pairwise_gp``).

.. [Chu2005preference]
    Wei Chu, and Zoubin Ghahramani. Preference learning with Gaussian processes.
    Proceedings of the 22nd international conference on Machine learning. 2005.

.. [Brochu2010tutorial]
    Eric Brochu, Vlad M. Cora, and Nando De Freitas.
    A tutorial on Bayesian optimization of expensive cost functions,
    with application to active user modeling and hierarchical reinforcement learning.
    arXiv preprint arXiv:1012.2599 (2010).

The Laplace approximation (the MAP utility from MINPACK's ``hybrj``, ported
to Rust, followed by BoTorch's two Newton steps), the posterior, the
Laplace marginal log likelihood and its gradient, and the fit are computed
in Rust. The random perturbation of the initial utility for new data is
drawn from numpy's global generator (``np.random.standard_normal``), as in
BoTorch, so that ``np.random.seed`` gives BoTorch's starting points.
"""

from __future__ import annotations

import copy
import importlib
import math
import random
import struct
import warnings
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    LIST,
    ArrayKind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.kernels import Kernel, RBFKernel, ScaleKernel, _kernel_from_spec
from botorch_rs.means import ConstantMean
from botorch_rs.mlls import MarginalLogLikelihood
from botorch_rs.models.likelihoods.pairwise import (
    PairwiseLikelihood,
    PairwiseProbitLikelihood,
)
from botorch_rs.models.model import FantasizeMixin, Model
from botorch_rs.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
    apply_chain_states,
    chain_spec,
)
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


def _float32_default() -> bool:
    """Whether torch's default dtype is float32 (or torch is not imported)."""
    import sys

    torch = sys.modules.get("torch")
    return torch is None or bool(torch.get_default_dtype() == torch.float32)


def _f32(x: float) -> float:
    return float(struct.unpack("f", struct.pack("f", x))[0])


def _standard_normal(shape: tuple[int, ...]) -> list[float]:
    """``np.random.standard_normal(shape)`` (BoTorch's perturbation of the
    initial utility), flattened; Python's ``random`` without numpy."""
    n = math.prod(shape)
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        return [random.gauss(0.0, 1.0) for _ in range(n)]
    return [float(v) for v in np.random.standard_normal(shape).reshape(-1)]


def _long_array(values: list[int], shape: tuple[int, ...], kind: ArrayKind) -> Any:
    """Indices as a long tensor, an int64 array or nested lists."""
    if kind.lib == "torch":
        torch = importlib.import_module("torch")
        t = torch.tensor(values, dtype=torch.long).reshape(shape)
        return t if kind.device is None or kind.device.type == "cpu" else t.to(kind.device)
    if kind.lib == "numpy":
        np = importlib.import_module("numpy")
        return np.asarray(values, dtype=np.int64).reshape(shape)
    data: Any = values
    for dim in reversed(shape[1:]):
        data = [data[i : i + dim] for i in range(0, len(data), dim)]
    return data if shape else values[0]


def _is_float_array(x: Any) -> bool:
    """Whether ``x`` has a floating dtype (BoTorch rejects float comparisons
    in ``condition_on_observations``)."""
    if is_torch_tensor(x):
        return bool(x.is_floating_point())
    if hasattr(x, "dtype") and hasattr(x.dtype, "kind"):
        return x.dtype.kind == "f"
    flat, _ = to_flat(x, "Y")
    return any(float(v) != int(v) for v in flat) if flat else False


def _check_comparisons(c: Any) -> Any:
    """torch's errors of BoTorch's ``comparisons.view(-1, m, 2)`` for
    comparisons that are not ``... x m x 2``."""
    data, shape = c
    if len(shape) < 2:
        raise IndexError("tuple index out of range")
    if shape[-1] != 2:
        raise RuntimeError(f"shape '[-1, {shape[-2]}, 2]' is invalid for input of size {len(data)}")
    return c


def _default_covar_module(dim: int | None, batch_shape: tuple[int, ...]) -> Kernel:
    """BoTorch's default covariance module of ``PairwiseGP``."""
    if dim is not None:
        spec = _rs.PairwiseGP.default_kernel_spec(dim, _float32_default())
        preset = lambda d: _rs.PairwiseGP.default_kernel_spec(d, _float32_default())  # noqa: E731
        # GPyTorch's ScaleKernel inherits the base kernel's batch shape,
        # including one independent outputscale per training batch member.
        return _kernel_from_spec(spec, preset, batch_shape)
    # Without data (no dimension): one shared lengthscale, as in BoTorch.
    from botorch_rs.constraints import GreaterThan, Interval
    from botorch_rs.priors import GammaPrior, SmoothedBoxPrior

    os_lb, os_ub = 1e-2, 1e2
    ls_prior = GammaPrior(concentration=2.4, rate=2.7)
    mode = (2.4 - 1) / 2.7
    if _float32_default():
        mode = _f32((_f32(2.4) - 1) / _f32(2.7))
    return ScaleKernel(
        RBFKernel(
            lengthscale_prior=ls_prior,
            lengthscale_constraint=GreaterThan(1e-4, transform=None, initial_value=mode),
        ),
        outputscale_prior=SmoothedBoxPrior(a=os_lb, b=os_ub),
        outputscale_constraint=Interval(os_lb * 0.5, os_ub * 2.0, initial_value=1.0),
    )


class _TrainingOutput:
    """The output of a training-mode forward pass: the posterior centered at
    the MAP utility of the (consolidated) training data."""

    def __init__(self, model: PairwiseGP) -> None:
        self._model = model

    @property
    def mean(self) -> Any:
        return self._model.utility

    @property
    def loc(self) -> Any:
        return self.mean


class PairwiseGP(Model, FantasizeMixin):
    r"""Probit GP for preference learning with Laplace approximation

    A probit-likelihood GP that learns via pairwise comparison data, using a
    Laplace approximation of the posterior of the estimated utility values. By
    default it uses a scaled RBF kernel.

    Implementation is based on [Chu2005preference]_.
    Also see [Brochu2010tutorial]_ for additional reference.

    Note that in [Chu2005preference]_ the likelihood of a pairwise comparison
    is :math:``\left(\frac{f(x_1) - f(x_2)}{\sqrt{2}\sigma}\right)``, i.e. a scale is
    used in the denominator. To maintain consistency with usage of kernels
    elsewhere in BoTorch, we instead do not include :math:``\sigma`` in the code
    (implicitly setting it to 1) and use ScaleKernel to scale the function.

    In the example below, the user/decision maker has stated that they prefer
    the first item over the second item and the third item over the second item,
    generating comparisons [0, 1] and [2, 1].
    Example:
        >>> from botorch_rs.models import PairwiseGP
        >>> import torch
        >>> datapoints = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> comparisons = torch.Tensor([[0, 1], [2, 1]])
        >>> model = PairwiseGP(datapoints, comparisons)
    """

    _buffer_names = [
        "consolidated_datapoints",
        "consolidated_comparisons",
        "D",
        "DT",
        "utility",
        "covar_chol",
        "likelihood_hess",
        "hlcov_eye",
        "covar",
        "covar_inv",
        "unconsolidated_datapoints",
        "unconsolidated_comparisons",
        "consolidated_indices",
    ]

    def __init__(
        self,
        datapoints: Any,
        comparisons: Any,
        likelihood: PairwiseLikelihood | None = None,
        covar_module: ScaleKernel | None = None,
        input_transform: InputTransform | None = None,
        *,
        jitter: float = 1e-6,
        xtol: float | None = None,
        consolidate_rtol: float = 0.0,
        consolidate_atol: float = 1e-4,
        maxfev: int | None = None,
    ) -> None:
        r"""
        Args:
            datapoints: Either ``None`` or a ``batch_shape x n x d`` tensor of
                training features. If either ``datapoints`` or ``comparisons`` is
                ``None``, construct a prior-only model.
            comparisons: Either ``None`` or a ``batch_shape x m x 2`` tensor of
                training comparisons; comparisons[i] is a noisy indicator
                suggesting the utility value of comparisons[i, 0]-th is greater
                than comparisons[i, 1]-th. If either ``comparisons`` or
                ``datapoints`` is ``None``, construct a prior-only model.
            likelihood: A PairwiseLikelihood.
            covar_module: Covariance module.
            input_transform: An input transform that is applied in the model's
                forward pass.
            jitter: Value added to diagonal for numerical stability in
                ``psd_safe_cholesky``.
            xtol: Stopping criteria in scipy.optimize.fsolve used to find f_map
                in ``PairwiseGP._update``. If None, default behavior is handled by
                ``PairwiseGP._update``.
            consolidate_rtol: ``rtol`` passed to ``consolidate_duplicates``.
            consolidate_atol: ``atol`` passed to ``consolidate_duplicates``.
            maxfev: The maximum number of calls to the function in
                scipy.optimize.fsolve. If None, default behavior is handled by
                ``PairwiseGP._update``.
        """
        if datapoints is not None and is_torch_tensor(datapoints):
            import torch

            if datapoints.dtype == torch.float32:
                warnings.warn(
                    "The model inputs are of type torch.float32. It is strongly "
                    "recommended to use double precision in BoTorch, as this "
                    "improves both precision and stability and can help avoid "
                    "numerical errors. See "
                    "https://github.com/meta-pytorch/botorch/discussions/1444",
                    InputDataWarning,
                    stacklevel=2,
                )
        self._jitter = jitter
        self._xtol = xtol
        self._consolidate_rtol = consolidate_rtol
        self._consolidate_atol = consolidate_atol
        self._maxfev = maxfev
        if likelihood is None:
            likelihood = PairwiseProbitLikelihood()
        if not isinstance(likelihood, PairwiseLikelihood):
            raise UnsupportedError(
                "`likelihood` must be a botorch_rs PairwiseProbitLikelihood or "
                "PairwiseLogitLikelihood."
            )
        self.likelihood = likelihood
        self.mean_module = ConstantMean()
        self.training = True
        self._kind: ArrayKind = kind_of(datapoints) if datapoints is not None else LIST
        self._has_data = datapoints is not None and comparisons is not None
        self._dim: int | None = None
        self._batch: tuple[int, ...] = ()
        if self._has_data:
            x = to_flat(datapoints, "datapoints")
            shape = x[1]
            if len(shape) == 1:
                x = (x[0], (shape[0], 1))
                shape = x[1]
            self._dim = int(shape[-1])
            self._batch = tuple(shape[:-2])
        self._parameter_batch = self._batch
        covar: Kernel = (
            covar_module
            if covar_module is not None
            else _default_covar_module(self._dim, self._batch)
        )
        if not isinstance(covar, ScaleKernel):
            raise UnsupportedError("PairwiseGP must be used with a ScaleKernel.")
        covar_module = covar
        if covar_module._model is not None:
            raise UnsupportedError(
                "This kernel is already part of another model; create a new one for each model."
            )
        self.covar_module: ScaleKernel = covar_module
        self.input_transform: InputTransform | None = input_transform
        self._spec: Any = None
        self._rs: Any = None
        self._train_inputs: tuple[Any, ...] = ()
        self._train_targets: Any = None
        if self._has_data:
            self._build(datapoints, comparisons)
        from botorch_rs._buffers import adopt_buffer_dtype

        adopt_buffer_dtype(self, self._kind)

    # -- construction --------------------------------------------------------

    def _covar_args(self, d: int) -> dict[str, Any]:
        spec = self.covar_module._standard_spec(d)
        if spec is not None:
            return {"covar_spec": spec}
        return {"covar_tree": self.covar_module._node()}

    def _transform_args(self, d: int, batch: tuple[int, ...] = ()) -> dict[str, Any]:
        if self.input_transform is None:
            return {}
        if not isinstance(self.input_transform, InputTransform):
            raise UnsupportedError(
                "`input_transform` must be a botorch_rs input transform (from "
                "botorch_rs.models.transforms.input)."
            )
        if self.input_transform._per_batch() is not None:
            raise UnsupportedError(
                "PairwiseGP supports shared transform coefficients and shared or "
                "per-member Warp concentrations; per-member coefficient chains "
                "(such as Normalize(batch_shape=...)) are not supported."
            )
        self._spec = chain_spec(self.input_transform, d, batch)
        for step in self._spec.chains[0]:
            parameter_batch = tuple(getattr(step.owner, "batch_shape", ()))
            if getattr(step.rs, "params", []) and parameter_batch not in ((), batch):
                raise UnsupportedError(
                    f"A PairwiseGP Warp must have batch_shape=() or {batch}, got {parameter_batch}."
                )
        steps = self._spec.chains[0]
        native_steps = []
        for step in steps:
            owner = step.owner
            native = step.rs
            if isinstance(owner, Normalize) and owner._plain and not step.training:
                # The legacy Normalize.for_model creates fresh learnable
                # bounds. An eval-mode replacement must retain its fitted
                # coefficients, together with the ability to learn on train().
                native = _rs.InputStep.affine(
                    owner._d,
                    None,
                    list(owner._rs.offset[0]),
                    list(owner._rs.coefficient[0]),
                    "normalize",
                    owner.learn_bounds,
                    False,
                    owner._flags(),
                    min_range=owner.min_range,
                    center=owner.center,
                )
            native_steps.append(native)
        return {
            "input_chain": native_steps,
            "input_chain_training": [st.training for st in steps],
            "input_chain_prefixes": [st.prefix for st in steps],
        }

    def _settings(self) -> dict[str, Any]:
        return {
            "jitter": float(self._jitter),
            "xtol": None if self._xtol is None else float(self._xtol),
            "consolidate_rtol": float(self._consolidate_rtol),
            "consolidate_atol": float(self._consolidate_atol),
            "maxfev": None if self._maxfev is None else int(self._maxfev),
            "float32": _float32_default(),
        }

    def _build(
        self,
        datapoints: Any,
        comparisons: Any,
        parameter_batch: tuple[int, ...] | None = None,
    ) -> None:
        x = to_flat(datapoints, "datapoints")
        if len(x[1]) == 1:
            x = (x[0], (x[1][0], 1))
        c = _check_comparisons(to_flat(comparisons, "comparisons"))
        d = int(x[1][-1])
        batch = tuple(x[1][:-2])
        parameter_batch = batch if parameter_batch is None else parameter_batch
        if parameter_batch and (
            len(batch) < len(parameter_batch) or batch[-len(parameter_batch) :] != parameter_batch
        ):
            raise UnsupportedError(
                f"Replacement data batch {batch} must retain the parameter batch "
                f"{parameter_batch} in its trailing dimensions."
            )
        transform_args = self._transform_args(d, parameter_batch)
        param_map = None
        if batch:
            raw, param_map = self._batched_params(d, batch, parameter_batch)
        else:
            raw = self.covar_module._raw_flat()
        # The values are passed to the extension; the kernel then reads them
        # from the model.
        for k in self.covar_module._all_kernels():
            k._local = {}
        args = {**self._covar_args(d), **transform_args, **self._settings()}
        draws = None
        if batch:
            n = int(x[1][-2])
            draws = _standard_normal((*batch, n))
        self._rs = _rs.PairwiseGP(
            x,
            c,
            d,
            likelihood=self.likelihood._rs_name,
            raw=raw,
            param_map=param_map,
            draws=draws,
            update=bool(batch),
            **args,
        )
        self._dim = d
        self._batch = batch
        self._parameter_batch = parameter_batch
        if not self.training:
            self._rs.training = False
        self._has_data = True
        self._train_inputs = (datapoints,)
        self._train_targets = comparisons
        self._bind_modules()
        if not batch:
            self._update()
        self._sync_transform()

    def _batched_params(
        self, d: int, batch: tuple[int, ...], parameter_batch: tuple[int, ...]
    ) -> tuple[list[float], list[list[int]]]:
        """The global raw parameters of a batched model and every element's
        indices into them (the parameters of kernels with the model's batch
        shape are per element; others are shared)."""
        nb = math.prod(batch)
        parameter_members = math.prod(parameter_batch)
        layout: list[tuple[int, bool]] = []  # (size per element, batched)
        values: list[list[float]] = []
        kernels = [k for k in self.covar_module._all_kernels() if k._own_params()]
        for k in kernels:
            batched = bool(k._batch_shape) and tuple(k._batch_shape) == parameter_batch
            if k._batch_shape and not batched:
                raise UnsupportedError(
                    f"A kernel of a batched PairwiseGP must have batch_shape=() or {parameter_batch}, "
                    f"got {tuple(k._batch_shape)}."
                )
            for name, _, _ in k._own_params():
                shape = k._element_shapes()[name]
                size = math.prod(shape)
                vals = k._raw_values(name)
                if not batched:
                    vals = vals[:size]
                layout.append((size, batched))
                values.append(vals)
        if self._spec is not None:
            local_offset = 0
            for step in self._spec.chains[0]:
                params = list(getattr(step.rs, "params", []))
                batched = bool(getattr(step.owner, "batch_shape", ()))
                for _, shape in getattr(step.rs, "param_shapes", []):
                    size = math.prod(shape)
                    if batched:
                        assert self._spec.batch_params is not None
                        vals = [
                            value
                            for member in self._spec.batch_params
                            for value in member[local_offset : local_offset + size]
                        ]
                    else:
                        vals = params[:size]
                    layout.append((size, batched))
                    values.append(vals)
                    local_offset += size
                    params = params[size:]
        raw: list[float] = []
        offsets = []
        for (size, batched), vals in zip(layout, values, strict=True):
            offsets.append(len(raw))
            raw.extend(vals if batched else vals[:size])
        maps = []
        for e in range(nb):
            m: list[int] = []
            for (size, batched), off in zip(layout, offsets, strict=True):
                start = off + ((e % parameter_members) * size if batched else 0)
                m.extend(range(start, start + size))
            maps.append(m)
        self._param_layout = [(size, batched) for size, batched in layout]
        return raw, maps

    def _bind_modules(self) -> None:
        self.covar_module._bind(self, "covar_module.")
        self.mean_module._bind(self)
        if self.input_transform is not None:
            self.input_transform._bind(self, "input_transform.")

    def _sync_transform(self) -> None:
        if self._spec is not None and self._rs is not None:
            states = self._rs.input_chain_states()
            if states is not None:
                for i, (step, state) in enumerate(zip(self._spec.chains[0], states, strict=True)):
                    owner = step.owner
                    if (
                        isinstance(owner, Normalize)
                        and owner._plain
                        and hasattr(state, "affine_state")
                    ):
                        offset, coefficient = state.affine_state
                        bounds = list(offset) + [
                            o + c for o, c in zip(offset, coefficient, strict=True)
                        ]
                        states[i] = _rs.Normalize(
                            owner._d,
                            (bounds, (2, owner._d)),
                            learn_bounds=owner.learn_bounds,
                            min_range=owner.min_range,
                        )
                apply_chain_states(self._spec, [states], self._kind, ())

    def _update(self) -> None:
        """BoTorch's ``_update``: recomputes the Laplace approximation
        (warm-started from the previous solution, or from the win counts
        with numpy's standard normal perturbation for new data)."""
        k = self._rs.needs_draws
        if k:
            n = self._rs.n
            self._rs.update(
                _standard_normal((*self._batch, n)) if self._batch else _standard_normal((n,))
            )
        else:
            self._rs.update(None)
        self._sync_transform()

    # -- parameters (bound modules) -------------------------------------------

    def _param_names(self) -> list[tuple[str, tuple[int, ...]]]:
        out = []
        for name, value in self.covar_module.named_parameters():
            out.append((f"covar_module.{name}", tuple(_shape_of(value))))
        return out + self._input_param_shapes()

    def _input_param_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        if self._spec is None:
            return []
        return [
            (
                step.prefix + name,
                tuple(step.owner._param_shape())
                if hasattr(step.owner, "_param_shape")
                else tuple(shape),
            )
            for step in self._spec.chains[0]
            for name, shape in getattr(step.rs, "param_shapes", [])
        ]

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        if suffix.startswith("input_transform."):
            return dict(self._input_param_shapes())[suffix]
        k, name = self._kernel_param(suffix)
        shape = k._element_shapes()[name]
        return (*k._batch_shape, *shape)

    def _kernel_param(self, suffix: str) -> tuple[Kernel, str]:
        full = suffix[len("covar_module.") :] if suffix.startswith("covar_module.") else suffix
        *path, name = full.split(".")
        k: Any = self.covar_module
        for part in path:
            k = getattr(k, part)
        return k, name

    def _offsets(self) -> dict[str, tuple[int, int]]:
        """The offset and size of every raw parameter in the global vector."""
        out: dict[str, tuple[int, int]] = {}
        off = 0
        for k in self.covar_module._all_kernels():
            for name, _, _ in k._own_params():
                size = math.prod(self._raw_param_shape(k._prefix + name))
                out[k._prefix + name] = (off, size)
                off += size
        for name, shape in self._input_param_shapes():
            size = math.prod(shape)
            out[name] = (off, size)
            off += size
        return out

    def _raw_param(self, suffix: str) -> list[float]:
        off, size = self._offsets()[suffix]
        return list(self._rs.raw_parameters())[off : off + size]

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        from botorch_rs._arrays import broadcast_to

        data, shape = to_flat(value, suffix)
        off, size = self._offsets()[suffix]
        target = self._raw_param_shape(suffix)
        try:
            new = broadcast_to(data, shape, target)
        except ValueError:
            if len(data) != size:
                from botorch_rs.exceptions.errors import BotorchTensorDimensionError

                raise BotorchTensorDimensionError(
                    f"{suffix} has shape {target}, got a value of shape {tuple(shape)}."
                ) from None
            new = [float(v) for v in data]
        raw = list(self._rs.raw_parameters())
        raw[off : off + size] = new
        self._rs.set_raw_parameters(raw)

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every parameter (the mean module's constant,
        which is not fitted, the input transform, and the covariance module)."""
        if self.input_transform is not None:
            yield from _input_named_parameters(self.input_transform, "input_transform.")
        yield (
            "mean_module.raw_constant",
            from_flat([self.mean_constant], (), self._kind if self._kind.lib != "list" else LIST),
        )
        for name, value in self.covar_module.named_parameters():
            yield f"covar_module.{name}", value

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    @property
    def mean_constant(self) -> float:
        return float(self._rs.mean_constant) if self._rs is not None else 0.0

    def state_dict(self) -> dict[str, Any]:
        """The parameters (BoTorch's state dict also holds the buffers of the
        Laplace approximation, which ``load_state_dict`` ignores)."""
        return dict(self.named_parameters())

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = False) -> Any:
        r"""Load kernel hyperparameters and recompute Laplace approximation.

        Only the parameters are loaded (the buffers of the Laplace
        approximation are ignored); the Laplace approximation is then
        recomputed on the current training data.

        Args:
            state_dict: The state dict.
            strict: Must be ``False`` (as in BoTorch).
        """
        if strict:
            raise UnsupportedError("Passing strict=True is not supported.")
        names = dict(self.named_parameters())
        for key, value in state_dict.items():
            if key in self._buffer_names or key not in names:
                continue
            if key == "mean_module.raw_constant":
                self._rs.mean_constant = float(to_flat(value)[0][0])
                continue
            self._set_raw_param(key, value)
        if self._has_data:
            self._update()

    # -- data ----------------------------------------------------------------

    def _out(self, pair: Any) -> Any:
        return None if pair is None else from_pair(pair, self._kind)

    def _comparisons_out(self, pair: Any) -> Any:
        if pair is None:
            return None
        data, shape = pair
        return _long_array([int(v) for v in data], tuple(shape), self._kind)

    @property
    def datapoints(self) -> Any:
        r"""Alias for consolidated datapoints"""
        return self._out(self._rs.datapoints()) if self._has_data else None

    @property
    def comparisons(self) -> Any:
        r"""Alias for consolidated comparisons"""
        return self._comparisons_out(self._rs.comparisons()) if self._has_data else None

    @property
    def consolidated_datapoints(self) -> Any:
        return self.datapoints

    @property
    def consolidated_comparisons(self) -> Any:
        return self.comparisons

    @property
    def unconsolidated_datapoints(self) -> Any:
        """The training inputs as given (the object passed to the model)."""
        return self._train_inputs[0] if self._has_data else None

    @property
    def unconsolidated_comparisons(self) -> Any:
        """The comparisons as given (the object passed to the model)."""
        return self._train_targets if self._has_data else None

    @property
    def consolidated_indices(self) -> Any:
        if not self._has_data:
            return None
        idx = self._rs.consolidated_indices()
        return None if idx is None else _long_array(idx, (len(idx),), self._kind)

    @property
    def train_inputs(self) -> tuple[Any, ...]:
        return self._train_inputs

    @property
    def train_targets(self) -> Any:
        return self._train_targets

    @property
    def D(self) -> Any:
        return self._out(self._rs.D()) if self._has_data else None

    @property
    def DT(self) -> Any:
        D = self.D
        if D is None:
            return None
        if is_torch_tensor(D):
            return D.transpose(-1, -2)
        data, shape = to_flat(D, "D")
        *b, m, n = shape
        out = []
        nb = math.prod(b)
        for e in range(nb):
            for j in range(n):
                for k in range(m):
                    out.append(data[e * m * n + k * n + j])
        return from_flat(out, (*b, n, m), self._kind)

    @property
    def utility(self) -> Any:
        return self._out(self._rs.utility()) if self._has_data else None

    @property
    def unconsolidated_utility(self) -> Any:
        r"""Utility of the unconsolidated datapoints"""
        u = self.utility
        idx = self._rs.consolidated_indices() if self._has_data else None
        if idx is None:
            return u
        data, _ = to_flat(u, "utility")
        return from_flat([data[i] for i in idx], (len(idx),), self._kind)

    @property
    def likelihood_hess(self) -> Any:
        return self._out(self._rs.likelihood_hess()) if self._has_data else None

    @property
    def covar(self) -> Any:
        return self._out(self._rs.covar()) if self._has_data else None

    @property
    def covar_chol(self) -> Any:
        return self._out(self._rs.covar_chol()) if self._has_data else None

    @property
    def covar_inv(self) -> Any:
        return self._out(self._rs.covar_inv()) if self._has_data else None

    @property
    def hlcov_eye(self) -> Any:
        return self._out(self._rs.hlcov_eye()) if self._has_data else None

    @property
    def _x0(self) -> Any:
        """The fsolve solution (the warm start of the next update), a NumPy
        array as in BoTorch (a list without NumPy)."""
        if not self._has_data:
            return None
        pair = self._rs.x0()
        if pair is None:
            return None
        try:
            np = importlib.import_module("numpy")
        except ImportError:
            return list(pair[0])
        return np.asarray(pair[0], dtype=np.float64).reshape(pair[1])

    @_x0.setter
    def _x0(self, value: Any) -> None:
        if self._rs is not None:
            self._rs.set_x0(None if value is None else [float(v) for v in to_flat(value)[0]])

    @property
    def n(self) -> int:
        return int(self._rs.n) if self._has_data else 0

    @property
    def m(self) -> int:
        return int(self._rs.m) if self._has_data else 0

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1

    @property
    def _num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""The batch shape of the model (that of the training data)."""
        return self._batch if self._has_data else ()

    def _has_no_data(self) -> bool:
        r"""Return true if the model does not have both datapoints and comparisons"""
        return not self._has_data

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        """Transforms ``X`` with the model's input transform (if any)."""
        tf = input_transform if input_transform is not None else self.input_transform
        return X if tf is None else tf(X)

    @classmethod
    def construct_inputs(cls, training_data: Any) -> dict[str, Any]:
        r"""
        Construct ``Model`` keyword arguments from a ``RankingDataset``.

        Args:
            training_data: A ``RankingDataset`` (BoTorch's, holding a
                ``SliceContainer`` of the datapoints and the ranking vectors).

        Returns:
            A dict of keyword arguments that can be used to initialize a
            ``PairwiseGP``, including ``datapoints`` and ``comparisons``.
        """
        if type(training_data).__name__ != "RankingDataset":
            raise UnsupportedError(
                "Only `RankingDataset` is supported for `PairwiseGP`. Received "
                f"{type(training_data)}."
            )
        import torch

        datapoints = training_data._X.values
        comparisons = training_data._X.indices
        comp_order = training_data.Y
        comparisons = torch.gather(input=comparisons, dim=-1, index=comp_order)
        return {"datapoints": datapoints, "comparisons": comparisons}

    def set_train_data(
        self,
        datapoints: Any = None,
        comparisons: Any = None,
        strict: bool = False,
        update_model: bool = True,
    ) -> None:
        r"""Set datapoints and comparisons and update model properties if needed

        Args:
            datapoints: Either ``None`` or a ``batch_shape x n x d`` dimension
                tensor X. If there are input transformations, assume the
                datapoints are not transformed. If either ``datapoints`` or
                ``comparisons`` is ``None``, nothing is changed.
            comparisons: Either ``None`` or a tensor of size `batch_shape x m x
                2`. (i, j) means f_i is preferred over f_j.
            strict: ``strict`` argument as in gpytorch.models.exact_gp: the
                shape, dtype and device of the data must not change.
            update_model: True if we want to refit the model (see _update) after
                re-setting the data.
        """
        if datapoints is None or comparisons is None:
            return
        x = to_flat(datapoints, "datapoints")
        if len(x[1]) == 1:
            datapoints = _unsqueeze_last(datapoints)
        if strict:
            for name, new, old in (
                ("inputs", datapoints, self._train_inputs[0] if self._train_inputs else None),
                ("targets", comparisons, self._train_targets),
            ):
                for attr in ("shape", "dtype", "device"):
                    expected = getattr(old, attr, None)
                    found = getattr(new, attr, None)
                    if expected != found:
                        raise RuntimeError(
                            f"Cannot modify {attr} of {name} (expected {expected}, found {found})."
                        )
        # Build and validate on a separate owner so a failed replacement does
        # not invalidate the original native model or its bound module views.
        candidate = copy.deepcopy(self)
        if candidate._rs is None or candidate._batch or len(to_flat(datapoints)[1]) > 2:
            candidate._kind = kind_of(datapoints)
            if candidate._rs is not None:
                # Preserve all member parameters before replacing the native
                # model that owns their bound module views.
                for kernel in candidate.covar_module._all_kernels():
                    local = {name: kernel._raw_values(name) for name, _, _ in kernel._own_params()}
                    kernel._model = None
                    kernel._local = local
                if candidate._spec is not None:
                    for step in candidate._spec.chains[0]:
                        if hasattr(step.owner, "_current_steps"):
                            step.owner._steps = step.owner._current_steps()
                            step.owner._model = None
            candidate._rs = None
            candidate._build(datapoints, comparisons, candidate._parameter_batch)
        else:
            candidate._rs.set_train_data(
                to_flat(datapoints, "datapoints"),
                _check_comparisons(to_flat(comparisons, "comparisons")),
                False,
            )
            candidate._train_inputs = (datapoints,)
            candidate._train_targets = comparisons
            if update_model:
                candidate._update()
        covar, mean, transform = self.covar_module, self.mean_module, self.input_transform
        self.__dict__.update(candidate.__dict__)
        # Keep caller-held module objects live, as torch Module data
        # replacement does. Their views now point at the validated snapshot.
        self.covar_module, self.mean_module, self.input_transform = covar, mean, transform
        self._bind_modules()
        assert self._dim is not None
        self._spec = (
            None if transform is None else chain_spec(transform, self._dim, self._parameter_batch)
        )
        self._sync_transform()

    # -- modes ---------------------------------------------------------------

    def train(self, mode: bool = True) -> PairwiseGP:
        self.training = mode
        if self._rs is not None:
            self._rs.training = mode
        if self.input_transform is not None:
            self.input_transform.train(mode)
        return self

    def eval(self) -> PairwiseGP:
        return self.train(False)

    # -- posterior -----------------------------------------------------------

    def _prior_handle(self, d: int) -> Any:
        """A prior-only model of dimension ``d`` (without data)."""
        args = {**self._covar_args(d), **self._transform_args(d), **self._settings()}
        raw = self.covar_module._raw_flat()
        m = _rs.PairwiseGP(None, None, d, likelihood=self.likelihood._rs_name, raw=raw, **args)
        m.training = self.training
        return m

    def _handle(self, d: int | None = None) -> Any:
        if self._has_data:
            return self._rs.model_handle()
        if d is None:
            raise UnsupportedError("A PairwiseGP without data needs inputs of known dimension.")
        return self._prior_handle(d).model_handle()

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor, where ``d`` is the
                dimension of the feature space and ``q`` is the number of points
                considered jointly.
            output_indices: As defined in parent Model class, not used for this model.
            observation_noise: Ignored (since noise is not identifiable from scale
                in probit models).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A ``Posterior`` object, representing joint
                distributions over ``q`` points.
        """
        from botorch_rs.acquisition.objective import transform_arg
        from botorch_rs.exceptions.errors import BotorchTensorDimensionError

        self.eval()
        if output_indices is not None:
            raise RuntimeError(
                "output_indices is not None. PairwiseGP should not be amulti-output model."
            )
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}."
            )
        d = int(flat[1][-1])
        if self._dim is not None and d != self._dim:
            raise BotorchTensorDimensionError(f"Expected X with {self._dim} columns (d), got {d}.")
        handle = self._handle(d)
        return GPyTorchPosterior(
            handle,
            flat,
            kind,
            False,
            num_outputs=1,
            model_batch_shape=self.batch_shape,
            transform=transform_arg(posterior_transform, 1),
            query_inputs=X,
        )

    def __call__(self, X: Any) -> Any:
        """BoTorch's ``forward``: in train mode, the posterior of the training
        data at the MAP utility (``X`` must be the training datapoints object;
        the Laplace approximation is recomputed, warm-started); in eval mode,
        the posterior distribution at ``X``."""
        if self.training:
            if self._has_no_data():
                raise RuntimeError(
                    "datapoints and comparisons cannot be None in training mode. "
                    "Call .eval() for prior predictions, "
                    "or call .set_train_data() to add training data."
                )
            if X is not self.unconsolidated_datapoints:
                raise RuntimeError("Must train on training data")
            self._update()
            return _TrainingOutput(self)
        return self.posterior(X).distribution

    forward = __call__

    def _acquisition_model(self, posterior_transform: Any = None) -> Any:
        """The model handle that acquisition functions are built on."""
        from botorch_rs.acquisition.objective import transform_arg

        handle = self._handle(self._dim)
        t = transform_arg(posterior_transform, 1)
        if t is None:
            return handle
        return _rs.Model.from_models([handle], t)

    # -- conditioning ----------------------------------------------------------

    def _with_rs(self, rs: Any, X: Any, Y: Any) -> PairwiseGP:
        new = self.__class__.__new__(self.__class__)
        memo: dict[int, Any] = {id(self): new}
        for key, value in self.__dict__.items():
            if key in ("_rs", "_train_inputs", "_train_targets"):
                continue
            new.__dict__[key] = copy.deepcopy(value, memo)
        new._rs = rs
        new._has_data = True
        new._batch = tuple(rs.batch_shape)
        new._dim = rs.dim
        new._kind = self._kind if self._has_data else kind_of(X)
        new._train_inputs = (new._out(rs.unconsolidated_datapoints()),)
        new._train_targets = new._comparisons_out(rs.comparisons(True))
        return new

    def condition_on_observations(self, X: Any, Y: Any, **kwargs: Any) -> PairwiseGP:
        r"""Condition the model on new observations.

        Note that unlike other BoTorch models, PairwiseGP requires Y to be
        pairwise comparisons.

        Args:
            X: A ``batch_shape x n x d`` dimension tensor X
            Y: A tensor of size ``batch_shape x m x 2``. (i, j) means
                f_i is preferred over f_j
            kwargs: Not used.

        Returns:
            A (deepcopied) ``Model`` object of the same type, representing the
            original model conditioned on the new observations ``(X, Y)``.
        """
        if self._has_no_data():
            new = copy.deepcopy(self)
            new.covar_module._model = None
            new.set_train_data(X, Y, update_model=True)
            return new
        y = to_flat(Y, "Y")
        if _is_float_array(Y) or not y[1] or y[1][-1] != 2:
            raise RuntimeError("Conditioning on non-pairwise comparison observations.")
        x = to_flat(X, "X")
        if len(x[1]) == 1:
            x = (x[0], (x[1][0], 1))
        rs = self._rs.condition_on_observations(x, y, update=False)
        new = self._with_rs(rs, X, Y)
        new._update()
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> PairwiseGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_rs":
                new.__dict__[key] = None if value is None else value.copy()
            elif key in ("_train_inputs", "_train_targets"):
                new.__dict__[key] = value
            else:
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        if not self._has_data:
            return "PairwiseGP(prior-only)"
        extra = f", batch_shape={self._batch}" if self._batch else ""
        return f"PairwiseGP(n={self.n}, m={self.m}, d={self._dim}{extra})"


def _shape_of(value: Any) -> tuple[int, ...]:
    return tuple(to_flat(value)[1])


def _input_named_parameters(transform: InputTransform, prefix: str) -> Iterator[tuple[str, Any]]:
    if isinstance(transform, ChainedInputTransform):
        for name, child in transform.items():
            yield from _input_named_parameters(child, prefix + name + ".")
    elif hasattr(transform, "named_parameters"):
        for name, value in transform.named_parameters():
            yield prefix + name, value


def _unsqueeze_last(x: Any) -> Any:
    if is_torch_tensor(x):
        return x.unsqueeze(-1)
    if hasattr(x, "reshape") and hasattr(x, "shape"):
        return x.reshape(*x.shape, 1)
    return [[v] for v in x]


class PairwiseLaplaceMarginalLogLikelihood(MarginalLogLikelihood):
    r"""Laplace-approximated marginal log likelihood/evidence for PairwiseGP

    See (12) from [Chu2005preference]_.

    ``mll(model(datapoints), comparisons)`` (with the training datapoints and
    comparisons objects, after a training-mode forward pass) and ``mll()``
    evaluate it at the current Laplace approximation; ``value_and_grad``
    also returns its gradient with respect to the raw hyperparameters
    (BoTorch's autograd gradient, computed natively). ``fit_gpytorch_mll``
    maximizes it.
    """

    #: ``fit_gpytorch_mll`` fits this MLL natively.
    _rs_fit = True

    def __init__(self, likelihood: Any, model: PairwiseGP) -> None:
        """
        Args:
            likelihood: Used as in args to GPyTorch MarginalLogLikelihood
            model: Used as in args to GPyTorch MarginalLogLikelihood
        """
        if not isinstance(model, PairwiseGP):
            raise UnsupportedError(
                "PairwiseLaplaceMarginalLogLikelihood requires a botorch_rs PairwiseGP."
            )
        super().__init__(likelihood, model)

    def __call__(self, post: Any = None, comp: Any = None, *args: Any, **kwargs: Any) -> Any:
        r"""Calculate approximated log evidence, i.e., log(P(D|theta))

        Args:
            post: training posterior distribution from self.model (after
                consolidation)
            comp: Comparisons pairs (before consolidation)

        Returns:
            The approximated evidence, i.e., the marginal log likelihood
        """
        model = self.model
        if post is not None or comp is not None:
            if comp is not model.unconsolidated_comparisons:
                raise RuntimeError("Must train on training data")
        if model._has_no_data():
            raise RuntimeError(
                "datapoints and comparisons cannot be None in training mode. "
                "Call .eval() for prior predictions, "
                "or call .set_train_data() to add training data."
            )
        value = model._rs.mll()
        return from_flat([value], (), model._kind)

    forward = __call__

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        """A training forward pass at the current or the given flat raw
        hyperparameters (kernel parameters followed by input-transform parameters,
        in ``model._param_names()`` order; the fixed mean is excluded):
        the Laplace approximation is recomputed (warm-started) and installed,
        and the MLL and its gradient with respect to the raw hyperparameters
        are returned (a botorch_rs extension)."""
        self.model.train()
        if self.model._batch and self.model.input_transform is not None:
            # A conditioned eval-mode batch may carry bounds frozen before
            # its new observations. A training forward relearns those bounds
            # jointly before evaluating any member's parameter pullback.
            self.model._update()
        params = None if raw_parameters is None else [float(v) for v in to_flat(raw_parameters)[0]]
        value, grad = self.model._rs.forward_mll(params)
        return value, list(grad)


__all__ = ["PairwiseGP", "PairwiseLaplaceMarginalLogLikelihood"]
