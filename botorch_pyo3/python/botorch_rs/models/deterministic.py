"""Deterministic models (``botorch.models.deterministic``):
``PosteriorMeanModel`` (the posterior mean of a model),
``GenericDeterministicModel`` (a callable on torch tensors) and the
``MatheronPathModel`` of a posterior sample path.

``PosteriorMeanModel`` and ``GenericDeterministicModel`` can be the model of
any botorch_rs acquisition function, as in BoTorch: their posteriors have zero
variance, Monte-Carlo samples equal the mean (so the sampler does not matter;
BoTorch's ``StochasticSampler`` is accepted), and gradients flow through the
mean only. ``PosteriorMeanModel`` is evaluated in Rust;
``GenericDeterministicModel`` calls ``f`` on torch tensors and differentiates
it with ``torch.autograd``. ``AffineDeterministicModel`` (``b + X a``)
and ``FixedSingleSampleModel`` (``mean + sqrt(variance + jitter) * w`` of
another model's posterior) are evaluated in Rust.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, ArrayKind, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model import Model


class DeterministicPosterior:
    """The posterior of a deterministic model at ``X``: ``mean`` holds the
    model's values (``batch_shape x n x m``) and ``variance`` is zero (the
    ``mean`` of BoTorch's ``EnsemblePosterior`` of one member)."""

    def __init__(self, values: Any) -> None:
        self.values = values

    @property
    def mean(self) -> Any:
        return self.values

    @property
    def variance(self) -> Any:
        return _zeros_like(self.values)

    def rsample(self, sample_shape: Any = None) -> Any:
        """Samples (``sample_shape x batch_shape x n x m``): copies of the
        values (BoTorch's ``EnsemblePosterior`` of one member)."""
        shape = () if sample_shape is None else tuple(int(s) for s in sample_shape)
        if not shape:
            return self.values
        if isinstance(self.values, list):
            out: Any = self.values
            for s in reversed(shape):
                out = [out for _ in range(s)]
            return out
        if hasattr(self.values, "expand"):  # torch
            return self.values.expand(*shape, *self.values.shape)
        data, vshape = to_flat(self.values, "values")
        return from_flat(list(data) * math.prod(shape), (*shape, *vshape), kind_of(self.values))


def _zeros_like(v: Any) -> Any:
    if isinstance(v, list):
        return [_zeros_like(x) for x in v]
    return v * 0.0


class DeterministicModel(Model):
    """Base class of the deterministic models: ``posterior(X).mean`` is
    ``forward(X)``."""

    def forward(self, X: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, X: Any) -> Any:
        return self.forward(X)

    def posterior(
        self,
        X: Any,
        output_indices: Sequence[int] | None = None,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> DeterministicPosterior:
        """The (deterministic) posterior at ``X`` (``batch_shape x n x d``)."""
        if posterior_transform is not None:
            raise UnsupportedError(
                "posterior_transform is not supported by botorch_rs deterministic models."
            )
        values = self.forward(X)
        if output_indices is not None:
            values = _select_outputs(values, list(output_indices))
        return DeterministicPosterior(values)


def _select_outputs(values: Any, idx: list[int]) -> Any:
    if isinstance(values, list):
        if values and isinstance(values[0], list):
            return [_select_outputs(v, idx) for v in values]
        return [values[i] for i in idx]
    return values[..., idx]


class PosteriorMeanModel(DeterministicModel):
    r"""A deterministic model that always returns the posterior mean of
    ``model`` (a botorch_rs ``SingleTaskGP`` or ``ModelListGP``, or another
    deterministic model): ``forward(X) = model.posterior(X).mean``.

    In acquisition functions it is evaluated in Rust, with exact gradients
    through the mean. The model is used as it is when the acquisition
    function is created (see the migration notes on model snapshots).

    Example:
        >>> model_list = ModelListGP(gp1, gp2)
        >>> acqf = qLogExpectedHypervolumeImprovement(
        ...     PosteriorMeanModel(model_list), ref_point, partitioning,
        ...     sampler=StochasticSampler(torch.Size([1])),
        ... )
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def forward(self, X: Any) -> Any:
        """The posterior mean of ``model`` at ``X`` (``batch_shape x n x d``):
        ``batch_shape x n x m``."""
        return self.model.posterior(X).mean

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return int(self.model.num_outputs)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the model."""
        return tuple(self.model.batch_shape)

    @property
    def _kind(self) -> ArrayKind:
        return getattr(self.model, "_kind", LIST)

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle of acquisition functions (in Rust)."""
        _reject_transform(posterior_transform)
        inner = getattr(self.model, "_acquisition_model", None)
        if inner is None:
            raise UnsupportedError(
                "PosteriorMeanModel requires a botorch_rs model (SingleTaskGP, ModelListGP or "
                f"a deterministic model) in acquisition functions, got {type(self.model).__name__}."
            )
        return _rs.posterior_mean_model(inner(None))


class GenericDeterministicModel(DeterministicModel):
    r"""A generic deterministic model constructed from a callable ``f`` that
    maps ``batch_shape x n x d`` torch tensors to ``batch_shape x n x m``
    tensors (the outcome dimension must be explicit, even if ``m = 1``).

    In acquisition functions, ``f`` is called from Rust on float64 torch
    tensors (``1 x q x d`` for a candidate set) and differentiated with
    ``torch.autograd``; ``f`` must therefore be differentiable with torch
    (botorch_rs models are not: use ``PosteriorMeanModel`` for their mean).
    botorch_rs needs the input dimension there: pass the keyword-only ``dim``
    (a botorch_rs extension).

    Example:
        >>> f = lambda x: x.sum(dim=-1, keepdim=True)
        >>> model = GenericDeterministicModel(f, dim=3)
    """

    def __init__(
        self,
        f: Callable[[Any], Any],
        num_outputs: int = 1,
        batch_shape: Any = None,
        *,
        dim: int | None = None,
    ) -> None:
        self._f = f
        self._num_outputs = int(num_outputs)
        self._batch_shape = None if batch_shape is None else tuple(int(s) for s in batch_shape)
        self.dim = None if dim is None else int(dim)
        self._kind: ArrayKind = LIST

    @property
    def num_outputs(self) -> int:
        """The number of outputs ``m``."""
        return self._num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...] | None:
        """The batch shape of the model (``None`` if not given)."""
        return self._batch_shape

    def subset_output(self, idcs: list[int]) -> GenericDeterministicModel:
        """The model of the outputs ``idcs``."""

        def f_subset(X: Any) -> Any:
            return self._f(X)[..., idcs]

        return self.__class__(f=f_subset, num_outputs=len(idcs), dim=self.dim)

    def forward(self, X: Any) -> Any:
        """``f(X)`` (``batch_shape x n x m``)."""
        Y = self._f(X)
        if self._batch_shape is not None and hasattr(Y, "shape"):
            from botorch_rs._arrays import broadcast_shapes

            batch = tuple(Y.shape[:-2])
            if broadcast_shapes(self._batch_shape, batch) is None:
                raise ValueError(
                    "GenericDeterministicModel was initialized with batch_shape="
                    f"self.batch_shape={self._batch_shape} but the output of f has a "
                    f"batch_shape=batch_shape={batch} that is not broadcastable with it."
                )
        return Y

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle of acquisition functions (``f`` called from Rust)."""
        from botorch_rs._torch_bridge import require_torch

        _reject_transform(posterior_transform)
        if self.dim is None:
            raise UnsupportedError(
                "botorch_rs acquisition functions need the input dimension of a "
                "GenericDeterministicModel: create it with `dim=d`."
            )
        if self._batch_shape:
            raise UnsupportedError(
                "botorch_rs acquisition functions require unbatched models, got a "
                f"GenericDeterministicModel with batch shape {self._batch_shape}."
            )
        require_torch()
        return _rs.generic_deterministic_model(self._f, self.dim, self._num_outputs)


def _randn(n: int) -> list[float]:
    """``torch.randn(n)`` (float32 values, torch's global generator) or
    Python's ``random`` without torch."""
    import importlib
    import random

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return [random.gauss(0.0, 1.0) for _ in range(n)]
    return [float(v) for v in torch.randn(n).tolist()]


class AffineDeterministicModel(DeterministicModel):
    r"""An affine deterministic model."""

    def __init__(self, a: Any, b: Any = 0.01) -> None:
        r"""Affine deterministic model from weights and offset terms.

        A simple model of the form

            y[..., m] = b[m] + sum_{i=1}^d a[i, m] * X[..., i]

        Args:
            a: A ``d x m``-dim tensor of linear weights, where ``m`` is the number
                of outputs (must be explicit if ``m=1``)
            b: The affine (offset) term. Either a float (for single-output
                models or if the offset is shared), or a ``m``-dim tensor (with
                different offset values for the ``m`` different outputs).
        """
        from botorch_rs._buffers import _default_dtype
        from botorch_rs.priors import _float32

        adata, ashape = to_flat(a, "a")
        if len(ashape) != 2:
            raise ValueError("a must be two-dimensional")
        if isinstance(b, (int, float)):
            value = float(b)
            if _float32():
                # BoTorch: ``torch.tensor([b])`` in torch's default dtype.
                import struct

                value = float(struct.unpack("f", struct.pack("f", value))[0])
            bdata: list[float] = [value]
            bshape: tuple[int, ...] = (1,)
        else:
            bflat = to_flat(b, "b")
            bdata, bshape = [float(v) for v in bflat[0]], tuple(bflat[1])
        if len(bshape) != 1:
            raise ValueError("b must be one-dimensional")
        m = ashape[-1]
        if len(bdata) not in (1, m):
            raise RuntimeError(
                f"The expanded size of the tensor ({m}) must match the existing size "
                f"({len(bdata)}) at non-singleton dimension 0."
            )
        self.a = a
        self._a = (list(adata), tuple(ashape))
        self._b = [float(v) for v in bdata] * (m if len(bdata) == 1 else 1)
        if isinstance(b, (int, float)):
            a_kind = kind_of(a)
            dtype: Any = _default_dtype()
            if a_kind.lib == "torch":
                import torch

                dtype = getattr(torch, dtype)
            b_kind = ArrayKind(a_kind.lib, dtype, a_kind.device)
        else:
            b_kind = kind_of(b)
        self.b = from_flat(self._b, (m,), b_kind)
        # Integer/bool tensor offsets are buffers too; ArrayKind intentionally
        # restores floating result types only, so retain the original dtype.
        if hasattr(b, "dtype") and b_kind.dtype is None:
            self.b = b.expand(m) if hasattr(b, "expand") else self.b.astype(b.dtype)
        self._num_outputs = m
        self._kind = kind_of(a)

    @property
    def num_outputs(self) -> int:
        """The number of outputs ``m``."""
        return self._num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the model (empty)."""
        return ()

    def subset_output(self, idcs: list[int]) -> AffineDeterministicModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        adata, (d, m) = self._a
        a_sub = [adata[i * m + t] for i in range(d) for t in idcs]
        b_sub = [self._b[t] for t in idcs]
        kind = kind_of(self.a)
        return self.__class__(
            a=from_flat(a_sub, (d, len(idcs)), kind),
            b=self.b[idcs] if hasattr(self.b, "shape") else b_sub,
        )

    def forward(self, X: Any) -> Any:
        """``b + X a`` (``batch_shape x n x m``)."""
        if hasattr(X, "shape") and hasattr(self.a, "shape") and type(X) is type(self.a):
            try:
                import torch

                if isinstance(X, torch.Tensor):
                    b = torch.as_tensor(self._b, dtype=X.dtype)
                    return b + torch.einsum("...d,dm", X, self.a.to(X))
            except ImportError:  # pragma: no cover
                pass
        data, shape = to_flat(X, "X")
        adata, (d, m) = self._a
        if shape[-1] != d:
            raise RuntimeError(f"einsum(): the input has {shape[-1]} columns, a has {d} rows.")
        rows = len(data) // d
        out = [
            self._b[t] + sum(data[r * d + i] * adata[i * m + t] for i in range(d))
            for r in range(rows)
            for t in range(m)
        ]
        return from_flat(out, (*shape[:-1], m), kind_of(X))

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle of acquisition functions (in Rust)."""
        _reject_transform(posterior_transform)
        return _rs.affine_deterministic_model(self._a, self._b)


class FixedSingleSampleModel(DeterministicModel):
    r"""
    A deterministic model defined by a single sample ``w``.

    Given a base model ``f`` and a fixed sample ``w``, the model always outputs

        y = f_mean(x) + f_stddev(x) * w

    We assume the outcomes are uncorrelated here.
    """

    def __init__(
        self,
        model: Any,
        w: Any = None,
        dim: int | None = None,
        jitter: float | None = 1e-8,
        dtype: Any = None,
        device: Any = None,
    ) -> None:
        r"""
        Args:
            model: The base model.
            w: A 1-d tensor with length model.num_outputs.
                If None, draw it from a standard normal distribution.
            dim: dimensionality of w.
                If None and w is not provided, draw w samples of size model.num_outputs.
            jitter: jitter value to be added for numerical stability, 1e-8 by default.
            dtype: dtype for w if specified
            device: device for w if specified
        """
        self.model = model
        self._num_outputs = model.num_outputs
        self.jitter = jitter
        if w is None:
            n = model.num_outputs if dim is None else int(dim)
            self.w = from_flat(_randn(n), (n,), getattr(model, "_kind", LIST))
        else:
            self.w = w

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return self._num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the model."""
        return tuple(self.model.batch_shape)

    @property
    def _kind(self) -> ArrayKind:
        return getattr(self.model, "_kind", LIST)

    def forward(self, X: Any) -> Any:
        """``mean + sqrt(variance + jitter) * w`` of the model's posterior at
        ``X`` (``batch_shape x n x m``)."""
        post = self.model.posterior(X)
        mean, shape = to_flat(post.mean, "mean")
        var, _ = to_flat(post.variance, "variance")
        w, _ = to_flat(self.w, "w")
        m = shape[-1]
        jitter = 0.0 if self.jitter is None else float(self.jitter)
        out = [
            mu + math.sqrt(v + jitter) * w[i % m if len(w) > 1 else 0]
            for i, (mu, v) in enumerate(zip(mean, var, strict=True))
        ]
        return from_flat(out, tuple(shape), kind_of(X))

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle of acquisition functions (in Rust)."""
        _reject_transform(posterior_transform)
        inner = getattr(self.model, "_acquisition_model", None)
        if inner is None:
            raise UnsupportedError(
                "FixedSingleSampleModel requires a botorch_rs model in acquisition functions, "
                f"got {type(self.model).__name__}."
            )
        w = [float(v) for v in to_flat(self.w, "w")[0]]
        jitter = 0.0 if self.jitter is None else float(self.jitter)
        return _rs.fixed_single_sample_model(inner(None), w, jitter)


def _reject_transform(posterior_transform: Any) -> None:
    if posterior_transform is not None:
        # BoTorch: ``scalarize_posterior`` raises NotImplementedError for the
        # ``EnsemblePosterior`` of a deterministic model.
        raise UnsupportedError(
            "Posterior transforms are not supported for deterministic models (BoTorch's "
            "scalarize_posterior only works with a posterior that has a `distribution`)."
        )


class MatheronPathModel(DeterministicModel):
    r"""A deterministic model that returns a Matheron path sample of a model's
    posterior (``draw_matheron_paths`` of all outputs): ``forward(X)`` returns
    ``[sample_shape x] batch_shape x n x m`` values.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> matheron_model = MatheronPathModel(model)
        >>> output = matheron_model(test_X)

    ``seed`` seeds botorch_rs's random numbers (which differ from torch's for
    the same seed). botorch_rs extensions (keyword-only): ``num_features`` of
    the prior paths (BoTorch's default 1024) and explicit ``draws`` (see
    ``draw_matheron_paths``).
    """

    def __new__(cls, model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if cls is MatheronPathModel and getattr(model, "_tensor_backend", None) == "torch":
            from botorch_rs.sampling.pathwise._torch_backend import TorchMatheronPathModel

            return object.__new__(TorchMatheronPathModel)
        return object.__new__(cls)

    def __init__(
        self,
        model: Any,
        sample_shape: Any = None,
        ensemble_as_batch: bool = False,
        seed: int | None = None,
        *,
        num_features: int = 1024,
        draws: Sequence[Sequence[Any]] | None = None,
    ) -> None:
        from botorch_rs.models.model_list_gp_regression import ModelListGP
        from botorch_rs.sampling.pathwise import draw_kernel_feature_paths, draw_matheron_paths
        from botorch_rs.sampling.pathwise.utils import sample_shape_arg

        self.model = model
        if isinstance(model, ModelListGP) and len(model.models) != model.num_outputs:
            raise UnsupportedError("A model-list of multi-output models is not supported.")
        self.sample_shape = sample_shape_arg(sample_shape)
        self.ensemble_as_batch = ensemble_as_batch
        self._path = draw_matheron_paths(
            model,
            sample_shape=self.sample_shape,
            prior_sampler=functools.partial(draw_kernel_feature_paths, num_features=num_features),
            seed=seed,
            draws=draws,
        )
        self._path.set_ensemble_as_batch(ensemble_as_batch)
        # The handle of the paths of all outputs (shared by the paths of the
        # models of a ModelListGP).
        path: Any = self._path
        first = path.paths[0] if isinstance(model, ModelListGP) else path
        self._handle = getattr(first, "_handle", None)
        if isinstance(model, ModelListGP) and any(
            getattr(child, "_handle", None) is not self._handle for child in path.paths
        ):
            self._handle = None
        from botorch_rs.utils.transforms import is_ensemble

        self._is_ensemble = is_ensemble(model) or bool(self.sample_shape)

    @classmethod
    def _from_handle(cls, model: Any, handle: Any) -> MatheronPathModel:
        """The model of the single Matheron path of every output held by an
        extension handle (e.g. one of ``sample_optimal_points``)."""
        from botorch_rs.sampling.pathwise.posterior_samplers import MatheronPath
        from botorch_rs.sampling.pathwise.prior_samplers import as_paths
        from botorch_rs.sampling.pathwise.utils import model_outputs

        _, sizes = model_outputs(model)
        self = cls.__new__(cls)
        self.model = model
        self.sample_shape = ()
        self.ensemble_as_batch = False
        self._path = as_paths(model, handle, sizes, (), MatheronPath)
        self._handle = handle
        self._is_ensemble = False
        return self

    def forward(self, X: Any) -> Any:
        r"""The path values at ``X`` (``batch_shape x n x d``): a
        ``[sample_shape x] batch_shape x n x m`` array."""
        from botorch_rs.sampling.pathwise.utils import arrange, input_shape, path_values

        handle = self._handle
        if handle is None:
            from botorch_rs._ops import concat, getitem, transpose

            if hasattr(self.model, "models"):
                return concat([getitem(v, (..., None)) for v in self._path(X)], -1)
            if self.num_outputs == 1:
                return getitem(self._path(X), (..., None))
            return transpose(self._path(getitem(X, (..., None, slice(None), slice(None)))))
        shape = input_shape(X, handle.dim)
        pv = path_values(handle, X)
        return arrange(
            pv,
            handle.num_samples,
            math.prod(shape[:-1]),
            shape,
            self.sample_shape,
            list(range(handle.num_outputs)),
            multi_output=False,
            outputs_last=True,
        )

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return int(self.model.num_outputs)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape of the model: ``sample_shape``."""
        return self.sample_shape + tuple(self.model.batch_shape)


__all__ = [
    "AffineDeterministicModel",
    "DeterministicModel",
    "FixedSingleSampleModel",
    "GenericDeterministicModel",
    "MatheronPathModel",
    "PosteriorMeanModel",
]
