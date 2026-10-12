"""Deterministic models (``botorch.models.deterministic``): the
``MatheronPathModel`` of a posterior sample path.

The other deterministic models of BoTorch (``GenericDeterministicModel``,
``AffineDeterministicModel``, ...) are not ported.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any

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


__all__ = ["DeterministicModel", "MatheronPathModel"]
