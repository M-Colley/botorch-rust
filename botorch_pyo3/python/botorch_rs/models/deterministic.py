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
it with ``torch.autograd``. ``AffineDeterministicModel`` and
``FixedSingleSampleModel`` are not ported.
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
        self._handle = path.paths[0]._handle if isinstance(model, ModelListGP) else path._handle
        self._is_ensemble = len(self.sample_shape) > 0

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
    "DeterministicModel",
    "GenericDeterministicModel",
    "MatheronPathModel",
    "PosteriorMeanModel",
]
