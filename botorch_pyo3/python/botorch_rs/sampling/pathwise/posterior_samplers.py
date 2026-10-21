r"""Posterior sample paths by Matheron's rule
(``botorch.sampling.pathwise.posterior_samplers``)::

                  "Prior path"
                       v
        (f | y)(·) = f(·) + Cov(f(·), y) Cov(y, y)^{-1} (y - f(X) - ε),
                            \_______________________________________/
                                                v
                                          "Update path"

The prior paths are random Fourier feature approximations of the GP prior
(``draw_kernel_feature_paths``) and the updates are exact
(``gaussian_update``). The native backend computes both in Rust. The Tensor
backend retains device graphs for exact and scalar whitened variational GPs.

References:
    J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky and M. Deisenroth.
    Efficiently sampling functions from Gaussian process posteriors. ICML
    2020; Pathwise Conditioning of Gaussian Processes. JMLR 22, 2021.
"""

from __future__ import annotations

import copy
import functools
import math
from collections.abc import Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, from_flat, is_torch_tensor
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.paths import GeneralizedLinearPath, PathDict, PathList, _RsPaths
from botorch_rs.sampling.pathwise.prior_samplers import (
    DEFAULT_NUM_FEATURES,
    as_paths,
    draw_kernel_feature_paths,
    draws_arg,
    seed_arg,
)
from botorch_rs.sampling.pathwise.update_strategies import gaussian_update
from botorch_rs.sampling.pathwise.utils import model_outputs, sample_shape_arg


class MatheronPath(_RsPaths, PathDict):
    r"""Function draws from a GP posterior via Matheron's rule: a ``PathDict``
    of the ``"prior_paths"`` and ``"update_paths"`` joined by their sum, in
    the model's original outcome space (its outcome transform is applied to
    the sum, as in BoTorch).

    Evaluated at ``X`` (``batch_shape x n x d``), the paths return
    ``sample_shape x batch_shape x [m x] n`` values (for a multi-output
    model, the outputs form the last batch dimension, which ``batch_shape``
    broadcasts with). The sub-paths return values in the transformed outcome
    space, like BoTorch's.
    """

    _native_components: dict[str, Any]

    def __init__(
        self,
        prior_paths: Any,
        update_paths: Any,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> None:
        PathDict.__init__(
            self,
            paths={"prior_paths": prior_paths, "update_paths": update_paths},
            join=_sum_paths,
            input_transform=input_transform,
            output_transform=output_transform,
        )

    @classmethod
    def _from_native(
        cls,
        handle: _rs.SamplePaths,
        sample_shape: tuple[int, ...],
        outputs: Sequence[int],
        kind: ArrayKind,
    ) -> MatheronPath:
        self = cls.__new__(cls)
        _RsPaths.__init__(self, handle, sample_shape, outputs, kind)
        paths = {}
        if handle.has_components:
            for name in ("prior_paths", "update_paths"):
                paths[name] = GeneralizedLinearPath._from_native(
                    handle, sample_shape, outputs, kind, name
                )
        PathDict.__init__(self, paths=paths, join=_sum_paths)
        self._native_components = dict(paths)
        self._native_output_transform = None
        return self

    def _set_model_transform(self, model: Any) -> None:
        from botorch_rs.sampling.pathwise.utils import get_output_transform

        self.output_transform = copy.deepcopy(get_output_transform(model))
        self._native_output_transform = self.output_transform

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        if (
            "_handle" in self.__dict__
            and self.input_transform is None
            and self.output_transform is self._native_output_transform
            and self.join is _sum_paths
            and self.paths.keys() == self._native_components.keys()
            and all(path is self._native_components[name] for name, path in self.paths.items())
        ):
            return self._values(x)
        return super().__call__(x, **kwargs)

    def forward(self, x: Any, **kwargs: Any) -> Any:
        return PathDict.forward(self, x, **kwargs)

    @property
    def draws(self) -> list[tuple[Any, Any, Any]]:
        """The random numbers of the paths of every output (a botorch_rs
        extension): ``(frequencies, weights, noise)`` of shapes
        ``num_features / 2 x d``, ``prod(sample_shape) x num_features`` and
        ``prod(sample_shape) x n``; passing them as ``draws`` to
        ``draw_matheron_paths`` reproduces the paths."""
        if "_handle" not in self.__dict__:
            raise UnsupportedError(
                "Explicitly composed paths do not have a joint native draw record."
            )
        draws = self._handle.draws()
        return [
            tuple(from_flat(a[0], a[1], self._kind) for a in draws[t])  # type: ignore[misc]
            for t in self._outputs
        ]

    def __repr__(self) -> str:
        if "_handle" not in self.__dict__:
            return "MatheronPath(prior_paths, update_paths)"
        return f"MatheronPath(sample_shape={self._sample_shape}, num_outputs={len(self._outputs)})"


def _sum_paths(paths: Any) -> Any:
    from botorch_rs._ops import apply

    if not paths:
        return 0
    total = paths[0]
    for path in paths[1:]:
        total = total + path if is_torch_tensor(total) else apply("add_raw", total, path)
    return total


def _num_features(prior_sampler: Any) -> int:
    """The number of features of a ``draw_kernel_feature_paths`` prior sampler
    (possibly a ``functools.partial`` with ``num_features``)."""
    func, kwargs = prior_sampler, {}
    if isinstance(prior_sampler, functools.partial):
        if prior_sampler.args:
            raise UnsupportedError("Positional arguments of the prior sampler are not supported.")
        func, kwargs = prior_sampler.func, dict(prior_sampler.keywords)
    if func is not draw_kernel_feature_paths:
        raise UnsupportedError(
            "botorch_rs only supports draw_kernel_feature_paths as prior_sampler."
        )
    num_features = int(kwargs.pop("num_features", DEFAULT_NUM_FEATURES))
    if kwargs:
        raise UnsupportedError(
            f"Unsupported arguments of draw_kernel_feature_paths: {sorted(kwargs)}."
        )
    return num_features


def draw_matheron_paths(
    model: Any,
    sample_shape: Any,
    prior_sampler: Any = draw_kernel_feature_paths,
    update_strategy: Any = gaussian_update,
    *,
    seed: int | None = None,
    draws: Sequence[Sequence[Any]] | None = None,
) -> MatheronPath | PathList:
    r"""Generates function draws from a GP posterior with Matheron's rule.

    Evaluated at ``X`` (``batch_shape x n x d``), the paths return
    ``sample_shape x batch_shape x [m x] n`` values; the paths of a
    ``ModelListGP`` form a ``PathList`` of the ``MatheronPath`` of each model.

    ``prior_sampler`` is called with ``model`` and ``sample_shape``;
    ``update_strategy`` receives ``model``, the resulting ``sample_values``
    and the exact model's transformed ``target_values``. Defaults are
    ``draw_kernel_feature_paths`` (1024 features) and ``gaussian_update``.
    Outcome untransformation is applied once, after joining both components.

    botorch_rs extensions (keyword-only): ``seed`` (the random numbers of
    botorch_rs, which differ from torch's for the same seed; ``None`` draws a
    random seed) or ``draws``, one ``(frequencies, weights, noise)`` tuple per
    output (``num_features / 2 x d``, ``prod(sample_shape) x num_features`` and
    ``prod(sample_shape) x n``), e.g. BoTorch's ``feature_map.weight`` and
    ``weight`` of its prior paths and the standard normal noise of its update.
    Given BoTorch's draws, the paths equal BoTorch's to rounding.

    For device scalar ``SingleTaskVariationalGP`` models, the third draw is
    the whitened variational standard normal and its last dimension is the
    number of inducing points. The current variational distribution supplies
    latent targets; no observation noise is added. Multi-output variational
    prior paths are supported, but their Matheron update is not registered
    by the reference's ``IndependentMultitaskVariationalStrategy`` dispatcher.
    """
    shape = sample_shape_arg(sample_shape)
    try:
        num_features = _num_features(prior_sampler)
    except UnsupportedError:
        num_features = None
    if (
        num_features is None
        or update_strategy is not gaussian_update
        or (getattr(model, "_tensor_backend", None) != "torch" and bool(model.batch_shape))
    ):
        return _compose_matheron_paths(model, shape, prior_sampler, update_strategy, seed, draws)
    if getattr(model, "_tensor_backend", None) == "torch":
        from botorch_rs.sampling.pathwise._torch_backend import tensor_matheron_paths

        return tensor_matheron_paths(model, shape, num_features, seed, draws)
    rs_model, sizes = model_outputs(model)
    handle = _rs.draw_matheron_paths(
        rs_model,
        math.prod(shape),
        num_features,
        seed_arg(seed),
        draws_arg(draws, ("frequencies", "weights", "noise")),
    )
    return as_paths(model, handle, sizes, shape, MatheronPath)


def _compose_matheron_paths(
    model: Any,
    shape: tuple[int, ...],
    prior_sampler: Any,
    update_strategy: Any,
    seed: Any,
    draws: Any,
) -> Any:
    from botorch_rs.models._pathwise_context import suppress_path_outcome
    from botorch_rs.sampling.pathwise.utils import (
        get_output_transform,
        get_train_inputs,
        get_train_targets,
    )

    if not callable(prior_sampler) or not callable(update_strategy):
        raise TypeError("prior_sampler and update_strategy must be callable.")
    if draws is not None:
        raise ValueError("Custom Matheron samplers cannot be combined with explicit native draws.")
    if hasattr(model, "models"):
        return PathList(
            [
                draw_matheron_paths(
                    child,
                    shape,
                    prior_sampler,
                    update_strategy,
                    seed=None if seed is None else seed + i,
                )
                for i, child in enumerate(model.models)
            ]
        )
    model.eval()
    inner = getattr(model, "model", None)
    variational = hasattr(inner, "variational_strategy")
    if variational:
        assert inner is not None
        strategy = inner.variational_strategy
        if hasattr(strategy, "base_variational_strategy"):
            raise NotImplementedError(
                "The reference Matheron dispatcher does not support independent multitask variational updates."
            )
        points, targets = strategy.inducing_points, None
    else:
        (points,) = get_train_inputs(model, transformed=True)
        targets = get_train_targets(model, transformed=True)
    outcome = get_output_transform(model)
    prior_options: dict[str, Any] = {}
    if seed is not None:
        try:
            _num_features(prior_sampler)
        except UnsupportedError:
            pass
        else:
            prior_options["seed"] = seed
    with suppress_path_outcome(model):
        prior = prior_sampler(
            model=model,
            sample_shape=shape,
            **prior_options,
        )
        values = prior.forward(points)
        options = {} if variational else {"target_values": targets}
        if update_strategy is gaussian_update and seed is not None:
            options["seed"] = seed
        update = update_strategy(model=model, sample_values=values, **options)
    cls: Any = MatheronPath
    if getattr(model, "_tensor_backend", None) == "torch":
        from botorch_rs.sampling.pathwise._torch_backend import TorchMatheronPath

        cls = TorchMatheronPath
    return cls(prior_paths=prior, update_paths=update, output_transform=outcome)


def get_matheron_path_model(
    model: Any,
    sample_shape: Any = None,
    ensemble_as_batch: bool = False,
    *,
    seed: int | None = None,
    num_features: int = DEFAULT_NUM_FEATURES,
    draws: Sequence[Sequence[Any]] | None = None,
) -> Any:
    r"""A deterministic model that evaluates a Matheron path drawn from the
    model's posterior (``MatheronPathModel``)."""
    from botorch_rs.models.deterministic import MatheronPathModel

    return MatheronPathModel(
        model,
        sample_shape=sample_shape,
        ensemble_as_batch=ensemble_as_batch,
        seed=seed,
        num_features=num_features,
        draws=draws,
    )


__all__ = ["MatheronPath", "draw_matheron_paths", "get_matheron_path_model"]
