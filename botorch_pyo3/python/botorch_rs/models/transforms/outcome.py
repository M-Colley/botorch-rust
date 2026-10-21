"""Outcome transforms (``botorch.models.transforms.outcome``): ``Standardize``,
``StratifiedStandardize``, ``Log``, ``Power``, ``Bilog`` and
``ChainedOutcomeTransform``.

A model applies its outcome transform to the training targets when it is
built (train mode: the statistics of the standardizations are fitted) and
un-transforms its posteriors: affine transforms keep them Gaussian
(``GPyTorchPosterior``), nonlinear ones (``Log``, ``Power``, ``Bilog``) make
them a :class:`~botorch_rs.posteriors.transformed.TransformedPosterior`,
whose samples are the transformed Gaussian samples and whose mean and
variance are the log-normal moments for ``Log`` (``NotImplementedError`` for
``Power`` and ``Bilog``), as in BoTorch. Models evaluate every output's
transform natively; transforms of some outputs (``outputs``) also un-transform
posteriors there (BoTorch raises ``NotImplementedError``).
"""

from __future__ import annotations

import builtins
import math
from collections import OrderedDict
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, ArrayKind, first_kind, from_flat, from_pair, to_flat
from botorch_rs._botorch_rs import Standardize as _NativeStandardize
from botorch_rs._buffers import NativeBufferModule
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


def _normalize_outputs(outputs: Any, m: int) -> list[int] | None:
    if outputs is None:
        return None
    out = []
    for i in outputs:
        i = int(i)
        if i < -m or i >= m:
            raise ValueError(f"Index {i} out of bounds for dimension of size {m}.")
        out.append(i % m)
    return out


def _columns(data: Any, shape: tuple[int, ...]) -> tuple[list[float], int, int]:
    """The values of ``batch_shape x n x m`` data, ``n`` rows per batch and
    ``m`` columns."""
    if len(shape) < 2:
        raise BotorchTensorDimensionError(
            f"Expected Y with shape `batch_shape x n x m`, got shape {shape}."
        )
    return list(data), shape[-2], shape[-1]


def _get_col(values: list[float], m: int, t: int) -> list[float]:
    return values[t::m]


def _set_col(values: list[float], m: int, t: int, col: list[float]) -> None:
    values[t::m] = col


class OutcomeTransform(NativeBufferModule):
    """Abstract base class for outcome transforms."""

    training: bool = True

    def train(self, mode: bool = True) -> OutcomeTransform:
        self.training = mode
        return self

    def eval(self) -> OutcomeTransform:
        return self.train(False)

    def forward(
        self, Y: Any, Yvar: Any = None, X: Any = None
    ) -> tuple[Any, Any]:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        return self.forward(Y, Yvar, X=X)

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `subset_output` method"
        )

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method"
        )

    @property
    def _is_linear(self) -> bool:
        """Whether the transform is affine (posteriors stay Gaussian)."""
        return False

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform_posterior` method"
        )

    # -- models ----------------------------------------------------------

    def _output_steps(self, m: int, t: int, batch: tuple[int, ...]) -> list[Any]:
        """The native steps of output ``t`` (of ``m``) for a model."""
        raise UnsupportedError(
            f"{type(self).__name__} cannot be used as the outcome transform of a botorch_rs model."
        )

    def _set_states(
        self, states: list[list[Any]], m: int, batch: tuple[int, ...], kind: ArrayKind
    ) -> None:
        """Receives the fitted steps of every element (``batch x m``,
        row-major) and consumes its own (the first of every list)."""

    def _num_steps(self, t: int, m: int) -> int:
        return len(self._output_steps(m, t, ()))


# -----------------------------------------------------------------------------
# Elementwise transforms
# -----------------------------------------------------------------------------


class _Elementwise(OutcomeTransform):
    """Shared implementation of ``Log``, ``Power`` and ``Bilog``."""

    _name = ""
    _noise_supported = False

    def __init__(self, outputs: list[int] | None = None) -> None:
        self._outputs = None if outputs is None else [int(o) for o in outputs]
        self.training = True

    def _step(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _map(self, Y: Any, Yvar: Any, inverse: bool) -> tuple[Any, Any]:
        kind = first_kind(Y)
        if kind.lib == "torch":
            if Yvar is not None and not self._noise_supported:
                raise NotImplementedError(
                    f"{self._name} does not yet support transforming observation noise"
                )
            from botorch_rs.models._torch_backend_modules import TorchElementwiseOutcome

            return TorchElementwiseOutcome(self, Y.shape[-1])._map(Y, Yvar, inverse)
        data, shape = to_flat(Y, "Y")
        values, n, m = _columns(data, tuple(shape))
        outputs = _normalize_outputs(self._outputs, m)
        selected = list(range(m)) if outputs is None else outputs
        var: list[float] | None = None
        if Yvar is not None:
            if not self._noise_supported:
                raise NotImplementedError(
                    f"{self._name} does not yet support transforming observation noise"
                )
            vdata, vshape = to_flat(Yvar, "Yvar")
            if tuple(vshape) != tuple(shape):
                raise BotorchTensorDimensionError(
                    f"Yvar of shape {tuple(vshape)} does not match Y of shape {tuple(shape)}."
                )
            var = list(vdata)
        step = self._step()
        for t in sorted(set(selected)):
            col = _get_col(values, m, t)
            vcol = _get_col(var, m, t) if var is not None else None
            out, vout = step.apply(col, vcol, None, inverse)
            _set_col(values, m, t, list(out))
            if var is not None and vout is not None:
                _set_col(var, m, t, list(vout))
        y_out = from_flat(values, shape, kind)
        v_out = from_flat(var, shape, first_kind(Yvar)) if var is not None else None
        return y_out, v_out

    def forward(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        return self._map(Y, Yvar, inverse=False)

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        return self._map(Y, Yvar, inverse=True)

    def _new(self, outputs: list[int] | None) -> _Elementwise:
        return self.__class__(outputs=outputs)

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension: the transformed
        outputs are re-indexed to their positions in ``idcs`` (BoTorch main
        since meta-pytorch/botorch#3380)."""
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + list(idcs)) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [j for j, i in enumerate(idcs) if i in self._outputs]
        new_tf = self._new(new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        r"""Un-transform a posterior: a ``TransformedPosterior``."""
        if self._outputs is not None:
            raise NotImplementedError(
                f"{self._name} does not yet support output selection for untransform_posterior"
            )
        from botorch_rs.posteriors.transformed import TransformedPosterior

        return TransformedPosterior(
            posterior=posterior,
            sample_transform=self._sample_fn(),
            mean_transform=self._mean_fn(),
            variance_transform=self._variance_fn(),
        )

    def _sample_fn(self) -> Any:
        step = self._step()
        kind, power = self._name, float(getattr(self, "power", 1))

        def sample_transform(s: Any) -> Any:
            if first_kind(s).lib == "torch":
                from botorch_rs.models._torch_backend_modules import _elementwise_outcome

                return _elementwise_outcome(s, kind, power, True)
            data, shape = to_flat(s, "samples")
            out, _ = step.apply(list(data), None, None, True)
            return from_flat(list(out), shape, first_kind(s))

        return sample_transform

    def _mean_fn(self) -> Any:
        return None

    def _variance_fn(self) -> Any:
        return None

    def _output_steps(self, m: int, t: int, batch: tuple[int, ...]) -> list[Any]:
        outputs = _normalize_outputs(self._outputs, m)
        if outputs is not None and t not in outputs:
            return []
        return [self._step()]

    def _set_states(
        self, states: list[list[Any]], m: int, batch: tuple[int, ...], kind: ArrayKind
    ) -> None:
        outputs = _normalize_outputs(self._outputs, m)
        for e, st in enumerate(states):
            if outputs is None or (e % m) in outputs:
                st.pop(0)


class Log(_Elementwise):
    r"""Log-transform outcomes (of the outputs ``outputs``, all by default).

    Observation noise is transformed by the delta method,
    ``Var[log Y] ≈ Var[Y] / max(Y, 1e-8)²``. The un-transformed posterior is a
    ``TransformedPosterior`` with samples ``exp(s)`` and the log-normal mean
    and variance."""

    _name = "Log"
    _noise_supported = True

    def _step(self) -> Any:
        return _rs.OutcomeStep.log()

    def _mean_fn(self) -> Any:
        from botorch_rs.models.transforms.utils import norm_to_lognorm_mean

        return norm_to_lognorm_mean

    def _variance_fn(self) -> Any:
        from botorch_rs.models.transforms.utils import norm_to_lognorm_variance

        return norm_to_lognorm_variance

    def __repr__(self) -> str:
        return "Log()" if self._outputs is None else f"Log(outputs={self._outputs})"


class Power(_Elementwise):
    r"""Power-transform outcomes, ``Y ** power`` (of the outputs ``outputs``).
    Observation noise is not supported (``NotImplementedError``); the
    un-transformed posterior has samples ``s ** (1 / power)`` and no mean or
    variance."""

    _name = "Power"

    def __init__(self, power: float, outputs: list[int] | None = None) -> None:
        super().__init__(outputs=outputs)
        self.power = power

    def _new(self, outputs: list[int] | None) -> _Elementwise:
        return self.__class__(power=self.power, outputs=outputs)

    def _step(self) -> Any:
        return _rs.OutcomeStep.power(float(self.power))

    def __repr__(self) -> str:
        return f"Power(power={self.power})"


class Bilog(_Elementwise):
    r"""Bilog-transform outcomes, ``sign(Y) log(|Y| + 1)`` (Eriksson and
    Poloczek, 2021), which magnifies values near zero and flattens extreme
    values. Observation noise is not supported; the un-transformed posterior
    has samples ``sign(s) expm1(|s|)`` and no mean or variance."""

    _name = "Bilog"

    def _step(self) -> Any:
        return _rs.OutcomeStep.bilog()

    def __repr__(self) -> str:
        return "Bilog()"


# -----------------------------------------------------------------------------
# Standardize
# -----------------------------------------------------------------------------


class Standardize(OutcomeTransform):
    r"""Standardize outcomes (zero mean, unit variance) of every output and
    batch element (of the outputs ``outputs`` only, if given: the other
    outputs keep means 0 and standard deviations 1).

    Fitted in train mode (and by the model that uses it) with the column means
    and unbiased standard deviations; standard deviations below ``min_stdv``
    are replaced by 1.
    """

    def __init__(
        self,
        m: int,
        outputs: Any = None,
        batch_shape: Any = (),
        min_stdv: float = 1e-8,
    ) -> None:
        self._m = int(m)
        self._outputs = _normalize_outputs(outputs, self._m)
        self._batch_shape = tuple(int(b) for b in (batch_shape or ()))
        self._min_stdv = float(min_stdv)
        self._rs = _rs.Standardize(min_stdv, list(self._batch_shape))
        self._kind: ArrayKind = LIST
        self.training = True

    @property
    def _rs(self) -> _NativeStandardize:
        if "_native_standardize" not in self.__dict__:
            # Compatibility with descriptor state saved before the lazy
            # Tensor/native boundary used a private backing handle.
            self._native_standardize = self.__dict__.pop("_rs")
        self._sync_native_statistics()
        return self._native_standardize

    @_rs.setter
    def _rs(self, value: _NativeStandardize) -> None:
        self._native_standardize = value
        # Native model loaders can replace this handle and its batch shape.
        # Their explicit replacement is authoritative over an earlier Tensor
        # cache; exporting that cache would overwrite the new native state.
        self.__dict__.pop("_tensor_standardize", None)

    def _set_fitted(self, fitted: _NativeStandardize, kind: ArrayKind) -> None:
        self._rs = fitted
        self._kind = kind
        self.__dict__.pop("_tensor_standardize", None)

    def _tensor_state(self, like: Any = None) -> Any:
        """Create actual Tensor buffers only when a Tensor protocol is used."""
        import torch

        from botorch_rs.models.transforms._torch_standardize import TensorStandardizeState

        state = self.__dict__.get("_tensor_standardize")
        if state is None:
            if like is None:
                dtype = self.__dict__.get("_buffer_factory_dtype", "float32")
                like = torch.empty(0, dtype=getattr(torch, dtype))
            state = TensorStandardizeState(self, like)
            if isinstance(self, StratifiedStandardize):
                if self._step.is_trained:
                    stats = self._step.stats
                    assert stats is not None
                    means, stdvs = stats
                    count = len(self._task_values)
                    state.means[..., :count, :] = like.new_tensor(means).unsqueeze(-1)
                    state.stdvs[..., :count, :] = like.new_tensor(stdvs).unsqueeze(-1)
                    state._stdvs_sq = state.stdvs.square()
                    state._is_trained.fill_(True)
            elif self._rs.is_trained:
                for name in ("means", "stdvs"):
                    data, shape = getattr(self._rs, name)
                    setattr(state, name, like.new_tensor(data).reshape(shape))
                state._stdvs_sq = state.stdvs.square()
                state._is_trained.fill_(True)
            self._tensor_standardize = state
        return state.train(self.training)

    def _sync_native_statistics(self) -> None:
        """An explicit return to an array API transfers fitted statistics."""
        state = self.__dict__.get("_tensor_standardize")
        if state is None or not bool(state._is_trained):
            return
        if isinstance(self, StratifiedStandardize):
            if self._batch_shape:
                raise UnsupportedError("Batched StratifiedStandardize requires Tensor inputs.")
            count = len(self._task_values)
            means = list(to_flat(state.means[..., :count, :])[0])
            stdvs = list(to_flat(state.stdvs[..., :count, :])[0])
            self._step = _rs.OutcomeStep.stratified(
                max(self._stratification_idx, 0), self._task_values, self._min_stdv, (means, stdvs)
            )
        else:
            means = list(to_flat(state.means)[0])
            stdvs = list(to_flat(state.stdvs)[0])
            self._native_standardize = _rs.Standardize.fitted(
                [means[i : i + self._m] for i in range(0, len(means), self._m)],
                [stdvs[i : i + self._m] for i in range(0, len(stdvs), self._m)],
                self._min_stdv,
                list(self._batch_shape),
            )

    def to(self, *args: Any, **kwargs: Any) -> Standardize:
        self._tensor_state().to(*args, **kwargs)
        return self

    def cpu(self) -> Standardize:
        return self.to(device="cpu")

    def cuda(self, device: Any = None) -> Standardize:
        import torch

        if device is None:
            target = torch.device("cuda")
        elif isinstance(device, int):
            target = torch.device("cuda", device)
        else:
            target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("cuda() requires a CUDA device.")
        return self.to(device=target)

    def float(self) -> Standardize:
        import torch

        return self.to(dtype=torch.float32)

    def double(self) -> Standardize:
        import torch

        return self.to(dtype=torch.float64)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._tensor_state().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        return self._tensor_state().load_state_dict(state_dict, strict=strict, assign=assign)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Any:
        state = self.__dict__.get("_tensor_standardize")
        if state is None:
            yield from super().named_buffers(prefix, recurse, remove_duplicate)
        else:
            yield from state.named_buffers(prefix, recurse, remove_duplicate)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Any:
        yield from ()

    def parameters(self, recurse: bool = True) -> Any:
        yield from ()

    @property
    def _is_trained(self) -> bool:
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return bool(state._is_trained)
        return self._rs.is_trained

    @property
    def min_stdv(self) -> builtins.float:
        return self._min_stdv

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def _is_linear(self) -> bool:
        return True

    def _stats(self, pair: Any, fill: builtins.float) -> Any:
        if not self._rs.is_trained:
            shape = (*self._batch_shape, 1, self._m)
            return from_flat([fill] * math.prod(shape), shape, self._kind)
        return from_pair(pair, self._kind)

    @property
    def means(self) -> Any:
        """The output means, shape ``batch_shape x 1 x m`` (zeros until fitted)."""
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return state.means
        return self._stats(self._rs.means, 0.0)

    @means.setter
    def means(self, value: Any) -> None:
        self._tensor_state(value).means = value

    @property
    def stdvs(self) -> Any:
        """The output standard deviations, shape ``batch_shape x 1 x m``."""
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return state.stdvs
        return self._stats(self._rs.stdvs, 1.0)

    @stdvs.setter
    def stdvs(self, value: Any) -> None:
        self._tensor_state(value).stdvs = value

    @property
    def _stdvs_sq(self) -> Any:
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return state._stdvs_sq
        if not self._rs.is_trained:
            return self._stats(None, 1.0)
        data, shape = self._rs.stdvs
        return from_flat([s * s for s in data], tuple(shape), self._kind)

    @_stdvs_sq.setter
    def _stdvs_sq(self, value: Any) -> None:
        self._tensor_state(value)._stdvs_sq = value

    def _check(self, shape: tuple[int, ...], train: bool) -> None:
        if len(shape) < 2:
            raise BotorchTensorDimensionError(
                f"Expected Y with shape `batch_shape x n x m`, got shape {shape}."
            )
        if train and tuple(shape[:-2]) != self._batch_shape:
            raise RuntimeError(
                f"Expected Y.shape[:-2] to be {self._batch_shape}, matching the "
                f"`batch_shape` argument to `Standardize`, but got "
                f"Y.shape[:-2]={tuple(shape[:-2])}."
            )
        if train and shape[-2] < 1:
            raise ValueError(f"Can't standardize with no observations. Y.shape={tuple(shape)}.")
        if shape[-1] != self._m:
            raise RuntimeError(
                f"Wrong output dimension. Y.size(-1) is {shape[-1]}; expected {self._m}."
            )

    def _apply_outputs(self, fitted: _NativeStandardize) -> _NativeStandardize:
        """Resets the statistics of the outputs that are not standardized."""
        if self._outputs is None:
            return fitted
        mdata, _ = fitted.means
        sdata, _ = fitted.stdvs
        nb = math.prod(self._batch_shape)
        means = [list(mdata[b * self._m : (b + 1) * self._m]) for b in range(nb)]
        stdvs = [list(sdata[b * self._m : (b + 1) * self._m]) for b in range(nb)]
        for b in range(nb):
            for t in range(self._m):
                if t not in self._outputs:
                    means[b][t] = 0.0
                    stdvs[b][t] = 1.0
        return _rs.Standardize.fitted(means, stdvs, self._min_stdv, list(self._batch_shape))

    def forward(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        """Standardizes ``Y`` (``batch_shape x n x m``) and ``Yvar``; in train
        mode the statistics are first fitted to ``Y``."""
        kind = first_kind(Y)
        if kind.lib == "torch":
            self._check(tuple(Y.shape), train=self.training)
            return self._tensor_state(Y)(Y, Yvar, X=X)
        self._sync_native_statistics()
        y = to_flat(Y, "Y")
        self._check(tuple(y[1]), train=self.training)
        v = to_flat(Yvar, "Yvar") if Yvar is not None else None
        if self.training:
            fitted, _, _ = self._rs.fit_transform(y, v)
            self._set_fitted(self._apply_outputs(fitted), kind)
        y_tf, v_tf = self._rs.transform(y, v)
        return from_pair(y_tf, kind), (from_pair(v_tf, kind) if v_tf is not None else None)

    __call__ = forward  # type: ignore[assignment]

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        """Un-standardizes ``Y`` (``... x n x m``) and ``Yvar``."""
        if first_kind(Y).lib == "torch":
            self._check(tuple(Y.shape), train=False)
            return self._tensor_state(Y).untransform(Y, Yvar, X=X)
        self._sync_native_statistics()
        if not self._rs.is_trained:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform`, since "
                "means and standard deviations need to be computed."
            )
        kind = first_kind(Y)
        y = to_flat(Y, "Y")
        self._check(tuple(y[1]), train=False)
        v = to_flat(Yvar, "Yvar") if Yvar is not None else None
        y_out, v_out = self._rs.untransform(y, v)
        return from_pair(y_out, kind), (from_pair(v_out, kind) if v_out is not None else None)

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension (the transformed
        outputs re-indexed to their positions in ``idcs``, as BoTorch main
        does since meta-pytorch/botorch#3380)."""
        new_m = len(idcs)
        if new_m > self._m:
            raise RuntimeError(
                "Trying to subset a transform have more outputs than  the original transform."
            )
        nlzd = _normalize_outputs(idcs, self._m) or []
        new_outputs = None
        if self._outputs is not None:
            new_outputs = [j for j, i in enumerate(nlzd) if i in self._outputs]
        new_tf = self.__class__(
            m=new_m, outputs=new_outputs, batch_shape=self._batch_shape, min_stdv=self._min_stdv
        )
        if (state := self.__dict__.get("_tensor_standardize")) is not None:
            result = new_tf._tensor_state(state.means)
            result.means = state.means[..., nlzd].clone()
            result.stdvs = state.stdvs[..., nlzd].clone()
            result._stdvs_sq = state._stdvs_sq[..., nlzd].clone()
            result._is_trained = state._is_trained.clone()
            return new_tf.train(self.training)
        if self._rs.is_trained:
            mdata, _ = self._rs.means
            sdata, _ = self._rs.stdvs
            nb = math.prod(self._batch_shape)
            means = [[mdata[b * self._m + i] for i in nlzd] for b in range(nb)]
            stdvs = [[sdata[b * self._m + i] for i in nlzd] for b in range(nb)]
            new_tf._rs = _rs.Standardize.fitted(
                means, stdvs, self._min_stdv, list(self._batch_shape)
            )
            new_tf._kind = self._kind
        if not self.training:
            new_tf.eval()
        return new_tf

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        r"""Un-standardize a posterior: a ``TransformedPosterior`` with the
        affine maps of the samples, mean and variance (BoTorch returns a
        rescaled ``GPyTorchPosterior`` for Gaussian posteriors; botorch_rs's
        models un-standardize natively)."""
        if first_kind(posterior.mean).lib == "torch":
            return self._tensor_state(posterior.mean).untransform_posterior(posterior, X=X)
        if self._outputs is not None:
            raise NotImplementedError(
                "Standardize does not yet support output selection for untransform_posterior"
            )
        self._sync_native_statistics()
        if not self._rs.is_trained:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data (e.g. `transform(Y)`) "
                "before calling `untransform_posterior`, since means and standard deviations "
                "need to be computed."
            )
        from botorch_rs.posteriors.transformed import TransformedPosterior, _affine

        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda s: _affine(s, self._rs.means, self._rs.stdvs, 1),
            mean_transform=lambda m, v: _affine(m, self._rs.means, self._rs.stdvs, 1),
            variance_transform=lambda m, v: _affine(v, None, self._rs.stdvs, 2),
        )

    # -- models ----------------------------------------------------------

    def _output_steps(self, m: int, t: int, batch: tuple[int, ...]) -> list[Any]:
        if self._outputs is not None and t not in self._outputs:
            return []
        return [_rs.OutcomeStep.standardize(self._min_stdv)]

    def _set_states(
        self, states: list[list[Any]], m: int, batch: tuple[int, ...], kind: ArrayKind
    ) -> None:
        nb = max(len(states) // m, 1)
        means = [[0.0] * m for _ in range(nb)]
        stdvs = [[1.0] * m for _ in range(nb)]
        for e, st in enumerate(states):
            t = e % m
            if self._outputs is not None and t not in self._outputs:
                continue
            step = st.pop(0)
            mu, sd = step.stats
            means[e // m][t] = mu[0]
            stdvs[e // m][t] = sd[0]
        if self._batch_shape == tuple(batch):
            self._set_fitted(
                _rs.Standardize.fitted(means, stdvs, self._min_stdv, list(self._batch_shape)), kind
            )

    def __repr__(self) -> str:
        return f"Standardize(m={self._m})"


class StratifiedStandardize(Standardize):
    r"""Standardize outcomes (zero mean, unit variance) per stratum, the
    stratum of a point being the (integer) value of its input
    ``stratification_idx``. Single output; strata without training data and
    unknown values are not standardized."""

    def __init__(
        self,
        stratification_idx: int,
        all_task_values: Any,
        batch_shape: Any = (),
        min_stdv: float = 1e-8,
        dtype: Any = None,
    ) -> None:
        data, _ = to_flat(all_task_values, "all_task_values")
        values = sorted({int(v) for v in data})
        self._stratification_idx = int(stratification_idx)
        self._task_values = values
        self._task_values_dtype = (
            getattr(all_task_values, "dtype", None)
            if first_kind(all_task_values).lib == "torch"
            else None
        )
        self._strata_dtype = dtype
        self._m = 1
        self._outputs = None
        self._batch_shape = tuple(int(b) for b in (batch_shape or ()))
        self._min_stdv = float(min_stdv)
        self._rs = _rs.Standardize(self._min_stdv, [])
        self._kind = LIST
        self.training = True
        self._step = _rs.OutcomeStep.stratified(
            self._stratification_idx if self._stratification_idx >= 0 else 0,
            values,
            self._min_stdv,
        )

    def _resolved_step(self, d: int) -> Any:
        idx = self._stratification_idx
        if idx < 0:
            idx += d
        stats = self._step.stats if self._step.is_trained else None
        return _rs.OutcomeStep.stratified(idx, self._task_values, self._min_stdv, stats)

    @property
    def _is_trained(self) -> bool:
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return bool(state._is_trained)
        return self._step.is_trained

    @property
    def strata_mapping(self) -> Any:
        return self._tensor_state().strata_mapping

    @property
    def means(self) -> Any:
        """The means per stratum (``n_strata x 1``)."""
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return state.means
        mu, _ = self._stratum_stats()
        return from_flat(list(mu), (len(mu), 1), self._kind)

    @means.setter
    def means(self, value: Any) -> None:
        self._tensor_state(value).means = value

    @property
    def stdvs(self) -> Any:
        """The standard deviations per stratum (``n_strata x 1``)."""
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return state.stdvs
        _, sd = self._stratum_stats()
        return from_flat(list(sd), (len(sd), 1), self._kind)

    @stdvs.setter
    def stdvs(self, value: Any) -> None:
        self._tensor_state(value).stdvs = value

    def _stratum_stats(self) -> tuple[list[float], list[float]]:
        stats = self._step.stats
        if stats is None:
            raise RuntimeError("StratifiedStandardize has not been fitted yet.")
        return stats

    @property
    def _stdvs_sq(self) -> Any:
        state = self.__dict__.get("_tensor_standardize")
        if state is not None:
            return state._stdvs_sq
        _, sd = self._stratum_stats()
        return from_flat([s * s for s in sd], (len(sd), 1), self._kind)

    @_stdvs_sq.setter
    def _stdvs_sq(self, value: Any) -> None:
        self._tensor_state(value)._stdvs_sq = value

    def _xy(
        self, Y: Any, Yvar: Any, X: Any
    ) -> tuple[list[float], list[float] | None, Any, tuple[int, ...]]:
        if self._batch_shape:
            raise UnsupportedError("Batched StratifiedStandardize requires Tensor inputs.")
        if X is None:
            raise ValueError("X is required for StratifiedStandardize.")
        ydata, yshape = to_flat(Y, "Y")
        if len(yshape) != 2:
            raise UnsupportedError("StratifiedStandardize supports `n x 1` outcomes only.")
        if yshape[-1] != 1:
            raise RuntimeError(f"Wrong output dimension. Y.size(-1) is {yshape[-1]}; expected 1.")
        vdata = list(to_flat(Yvar, "Yvar")[0]) if Yvar is not None else None
        x = to_flat(X, "X")
        return list(ydata), vdata, x, tuple(yshape)

    def forward(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        if first_kind(Y).lib == "torch":
            return self._tensor_state(Y)(Y, Yvar, X=X)
        self._sync_native_statistics()
        ydata, vdata, x, shape = self._xy(Y, Yvar, X)
        kind = first_kind(Y)
        step = self._resolved_step(x[1][-1])
        if self.training:
            if shape[-2] < 1:
                raise ValueError(f"Can't standardize with no observations. Y.shape={shape}.")
            step, _, _ = step.fit_transform(ydata, vdata, x)
            self._step = step
            self._kind = kind
            self.__dict__.pop("_tensor_standardize", None)
        y_tf, v_tf = step.apply(ydata, vdata, x, False)
        return (
            from_flat(list(y_tf), shape, kind),
            from_flat(list(v_tf), shape, kind) if v_tf is not None else None,
        )

    __call__ = forward  # type: ignore[assignment]

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        if first_kind(Y).lib == "torch":
            return self._tensor_state(Y).untransform(Y, Yvar, X=X)
        self._sync_native_statistics()
        ydata, vdata, x, shape = self._xy(Y, Yvar, X)
        kind = first_kind(Y)
        y_u, v_u = self._resolved_step(x[1][-1]).apply(ydata, vdata, x, True)
        return (
            from_flat(list(y_u), shape, kind),
            from_flat(list(v_u), shape, kind) if v_u is not None else None,
        )

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        raise NotImplementedError

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        if X is None:
            raise ValueError("X is required for StratifiedStandardize.")
        if first_kind(posterior.mean).lib == "torch":
            return self._tensor_state(posterior.mean).untransform_posterior(posterior, X=X)
        raise UnsupportedError(
            "StratifiedStandardize.untransform_posterior is only supported inside botorch_rs "
            "models."
        )

    def _output_steps(self, m: int, t: int, batch: tuple[int, ...]) -> list[Any]:
        if self._batch_shape:
            raise UnsupportedError("Batched StratifiedStandardize requires Tensor inputs.")
        if m != 1:
            raise RuntimeError(f"Wrong output dimension. Y.size(-1) is {m}; expected 1.")
        if self._stratification_idx < 0:
            raise UnsupportedError(
                "a negative stratification_idx is not supported in models; use the index of the "
                "transformed input column"
            )
        return [self._resolved_step(self._stratification_idx + 1)]

    def _set_states(
        self, states: list[list[Any]], m: int, batch: tuple[int, ...], kind: ArrayKind
    ) -> None:
        for st in states:
            self._step = st.pop(0)
        self._kind = kind
        self.__dict__.pop("_tensor_standardize", None)

    def __repr__(self) -> str:
        return f"StratifiedStandardize(stratification_idx={self._stratification_idx})"


class ChainedOutcomeTransform(OutcomeTransform):
    r"""An outcome transform representing the chaining of individual
    transforms (applied in order; un-transformed in reverse order)."""

    def __init__(self, **transforms: OutcomeTransform) -> None:
        for name, tf in transforms.items():
            if not isinstance(tf, OutcomeTransform):
                raise UnsupportedError(
                    f"botorch_rs chains its own outcome transforms, got {type(tf).__name__} for "
                    f"{name!r}."
                )
        self._transforms: dict[str, OutcomeTransform] = OrderedDict(transforms)
        self.training = True

    def __getitem__(self, key: str) -> OutcomeTransform:
        return self._transforms[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._transforms)

    def __len__(self) -> int:
        return len(self._transforms)

    def keys(self) -> KeysView[str]:
        return self._transforms.keys()

    def values(self) -> ValuesView[OutcomeTransform]:
        return self._transforms.values()

    def items(self) -> ItemsView[str, OutcomeTransform]:
        return self._transforms.items()

    def train(self, mode: bool = True) -> ChainedOutcomeTransform:
        self.training = mode
        for tf in self._transforms.values():
            tf.train(mode)
        return self

    def to(self, *args: Any, **kwargs: Any) -> ChainedOutcomeTransform:
        seen: set[int] = set()
        for transform in self.values():
            if id(transform) in seen:
                continue
            seen.add(id(transform))
            migrate = getattr(transform, "to", None)
            if migrate is not None:
                migrate(*args, **kwargs)
            elif not isinstance(transform, _Elementwise) or transform.__dict__.get(
                "_registered_buffers"
            ):
                raise UnsupportedError(
                    f"{type(transform).__name__} must implement to() for chained migration."
                )
        return self

    def cpu(self) -> ChainedOutcomeTransform:
        return self.to(device="cpu")

    def cuda(self, device: Any = None) -> ChainedOutcomeTransform:
        import torch

        target = (
            torch.device("cuda")
            if device is None
            else torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if target.type != "cuda":
            raise ValueError("cuda() requires a CUDA device.")
        return self.to(device=target)

    def float(self) -> ChainedOutcomeTransform:
        import torch

        return self.to(dtype=torch.float32)

    def double(self) -> ChainedOutcomeTransform:
        import torch

        return self.to(dtype=torch.float64)

    def state_dict(self, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
        if destination is None:
            destination = OrderedDict()
        for name, transform in self.items():
            save = getattr(transform, "state_dict", None)
            if save is not None:
                save(destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
            elif not isinstance(transform, _Elementwise) or transform.__dict__.get(
                "_registered_buffers"
            ):
                raise UnsupportedError(
                    f"{type(transform).__name__} must implement state_dict() for chained state."
                )
        return destination

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        import torch

        missing: list[str] = []
        unexpected: list[str] = []
        for name, transform in self.items():
            prefix = name + "."
            state = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            load = getattr(transform, "load_state_dict", None)
            if load is not None:
                result = load(state, strict=False, assign=assign)
                missing.extend(prefix + key for key in result.missing_keys)
                unexpected.extend(prefix + key for key in result.unexpected_keys)
            elif isinstance(transform, _Elementwise):
                unexpected.extend(prefix + key for key in state)
            else:
                raise UnsupportedError(
                    f"{type(transform).__name__} must implement load_state_dict() for chained state."
                )
        unexpected.extend(key for key in state_dict if key.split(".", 1)[0] not in self._transforms)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict: missing keys {missing}, unexpected keys {unexpected}."
            )
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Any:
        if not recurse:
            return
        seen: set[int] = set()
        for name, transform in self.items():
            if remove_duplicate and id(transform) in seen:
                continue
            seen.add(id(transform))
            path = f"{prefix}.{name}" if prefix else name
            yield from transform.named_buffers(path, recurse, remove_duplicate)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Any:
        for transform in self.values():
            if not isinstance(transform, (Standardize, _Elementwise, ChainedOutcomeTransform)):
                raise UnsupportedError(
                    "Custom chained outcomes must supply their parameter protocol."
                )
        yield from ()

    def parameters(self, recurse: bool = True) -> Any:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def forward(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        for tf in self._transforms.values():
            Y, Yvar = tf.forward(Y, Yvar, X=X)
        return Y, Yvar

    __call__ = forward  # type: ignore[assignment]

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        for tf in reversed(list(self._transforms.values())):
            Y, Yvar = tf.untransform(Y, Yvar, X=X)
        return Y, Yvar

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        return self.__class__(**{name: tf.subset_output(idcs) for name, tf in self.items()})

    @property
    def _is_linear(self) -> bool:
        return all(tf._is_linear for tf in self._transforms.values())

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        for tf in reversed(list(self._transforms.values())):
            posterior = tf.untransform_posterior(posterior, X=X)
        return posterior

    def _output_steps(self, m: int, t: int, batch: tuple[int, ...]) -> list[Any]:
        steps: list[Any] = []
        for tf in self._transforms.values():
            steps.extend(tf._output_steps(m, t, batch))
        return steps

    def _set_states(
        self, states: list[list[Any]], m: int, batch: tuple[int, ...], kind: ArrayKind
    ) -> None:
        for tf in self._transforms.values():
            tf._set_states(states, m, batch, kind)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._transforms.items())
        return f"ChainedOutcomeTransform({inner})"


def is_plain_standardize(tf: Any) -> bool:
    """Whether an outcome transform is a ``Standardize`` of all outputs (the
    transform the models have always evaluated)."""
    return type(tf) is Standardize and tf._outputs is None


def outcome_chains(tf: OutcomeTransform, m: int, batch: tuple[int, ...]) -> list[list[Any]]:
    """The native steps of every output of a model's outcome transform."""
    return [tf._output_steps(m, t, batch) for t in range(m)]


def apply_outcome_states(
    tf: OutcomeTransform,
    states: list[list[Any]],
    m: int,
    batch: tuple[int, ...],
    kind: ArrayKind,
) -> None:
    """Stores the fitted steps of a model's elements in the Python
    transforms."""
    tf._set_states([list(s) for s in states], m, batch, kind)


__all__ = [
    "Bilog",
    "ChainedOutcomeTransform",
    "Log",
    "OutcomeTransform",
    "Power",
    "Standardize",
    "StratifiedStandardize",
]
