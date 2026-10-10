"""``ModelListGP`` (``botorch.models.model_list_gp_regression``)."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.likelihoods import LikelihoodList
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model import Model
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


class ModelListGP(Model):
    r"""A multi-output model of independent GPs, possibly with different
    training data.

    The outputs of the list are the outputs of its sub-models in order (a
    sub-model may itself have several outputs). The posterior is the
    block-diagonal joint Gaussian of the sub-model posteriors in BoTorch's
    output-major layout (a ``MultitaskMultivariateNormal`` with
    ``interleaved=False``). Fit it with a ``SumMarginalLogLikelihood``.

    Example:
        >>> model1 = SingleTaskGP(train_X1, train_Y1)
        >>> model2 = SingleTaskGP(train_X2, train_Y2)
        >>> model = ModelListGP(model1, model2)
        >>> mll = SumMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
    """

    def __init__(self, *gp_models: SingleTaskGP) -> None:
        if not gp_models:
            raise ValueError("ModelListGP requires at least one model.")
        for m in gp_models:
            if not isinstance(m, SingleTaskGP):
                raise UnsupportedError(
                    "The sub-models of a botorch_rs ModelListGP must be botorch_rs SingleTaskGPs "
                    f"(got {type(m).__name__})."
                )
        dims = {m._rs.dim for m in gp_models}
        if len(dims) > 1:
            raise BotorchTensorDimensionError(
                f"All sub-models must have the same input dimension, got {sorted(dims)}."
            )
        shapes = {m.batch_shape for m in gp_models}
        if len(shapes) > 1:
            raise NotImplementedError(
                "`ModelListGP.batch_shape` is only supported if all constituent models have "
                "the same `batch_shape`."
            )
        self.models: list[SingleTaskGP] = list(gp_models)
        self.likelihood = LikelihoodList(*[m.likelihood for m in gp_models])
        kinds = [m._kind for m in gp_models if m._kind.lib != "list"]
        self._kind = kinds[0] if kinds else LIST
        self.training = True

    def _rs_list(self) -> _rs.ModelListGP:
        """A list of the current snapshots of the sub-models."""
        return _rs.ModelListGP([m._rs for m in self.models])

    # -- data --------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return sum(m.num_outputs for m in self.models)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.models[0].batch_shape

    @property
    def train_inputs(self) -> list[tuple[Any]]:
        return [m.train_inputs for m in self.models]

    @property
    def train_targets(self) -> list[Any]:
        return [m.train_targets for m in self.models]

    # -- modes -------------------------------------------------------------

    def train(self, mode: bool = True) -> ModelListGP:
        self.training = mode
        for m in self.models:
            m.train(mode)
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
        """The posterior at ``X`` (``batch_shape x q x d``) over the outputs
        ``output_indices`` (all by default), scalarized by
        ``posterior_transform`` if given, with the observation noise of every
        sub-model if ``observation_noise=True``."""
        from botorch_rs.acquisition.objective import transform_arg

        if not isinstance(observation_noise, bool):
            raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
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
        kind = kind_of(X)
        flat = to_flat(X, "X")
        d = self.models[0]._rs.dim
        if len(flat[1]) < 2 or flat[1][-1] != d:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d` with d = {d}, got shape {flat[1]}."
            )
        self.eval()
        return GPyTorchPosterior(
            self._rs_list(),
            flat,
            kind,
            observation_noise,
            num_outputs=m,
            model_batch_shape=self.batch_shape,
            output_indices=oi,
            transform=transform_arg(posterior_transform),
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle that acquisition functions are built on."""
        from botorch_rs.acquisition.objective import transform_arg

        return _rs.Model.from_model_list(
            self._rs_list(), transform_arg(posterior_transform, self.num_outputs)
        )

    # -- parameters --------------------------------------------------------

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every hyperparameter (``models.<i>.<name>``)."""
        for i, m in enumerate(self.models):
            for name, value in m.named_parameters():
                yield f"models.{i}.{name}", value

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    def initialize(self, **kwargs: Any) -> ModelListGP:
        """Sets hyperparameters by dotted name, e.g.
        ``model.initialize(**{"models.0.likelihood.noise": 1e-3})``."""
        for name, value in kwargs.items():
            head, _, rest = name.partition(".")
            index, _, rest = rest.partition(".")
            if head != "models" or not index.isdigit() or not rest:
                raise AttributeError(f"Unknown hyperparameter {name!r}.")
            self.models[int(index)].initialize(**{rest: value})
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ModelListGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo)
        new.likelihood = LikelihoodList(*[m.likelihood for m in new.models])
        return new

    def __repr__(self) -> str:
        return f"ModelListGP({', '.join(repr(m) for m in self.models)})"


__all__ = ["ModelListGP"]
