"""``ModelListGP`` (``botorch.models.model_list_gp_regression``)."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.likelihoods import LikelihoodList
from botorch_rs.models.contextual_multioutput import LCEMGP
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
    ``interleaved=False``). Fit it with a ``SumMarginalLogLikelihood``. The
    sub-models are ``SingleTaskGP``s or ``LCEMGP``s (conditioning and
    fantasies require ``SingleTaskGP``s).

    Example:
        >>> model1 = SingleTaskGP(train_X1, train_Y1)
        >>> model2 = SingleTaskGP(train_X2, train_Y2)
        >>> model = ModelListGP(model1, model2)
        >>> mll = SumMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
    """

    def __init__(self, *gp_models: SingleTaskGP | LCEMGP) -> None:
        if not gp_models:
            raise ValueError("ModelListGP requires at least one model.")
        for m in gp_models:
            if not isinstance(m, (SingleTaskGP, LCEMGP)):
                raise UnsupportedError(
                    "The sub-models of a botorch_rs ModelListGP must be botorch_rs SingleTaskGPs "
                    f"or LCEMGPs (got {type(m).__name__})."
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
        self.models: list[Any] = list(gp_models)
        self.likelihood = LikelihoodList(*[m.likelihood for m in gp_models])
        kinds = [m._kind for m in gp_models if m._kind.lib != "list"]
        self._kind = kinds[0] if kinds else LIST
        self.training = True

    @property
    def _all_single_task(self) -> bool:
        return all(isinstance(m, SingleTaskGP) for m in self.models)

    def _require_single_task(self, what: str) -> None:
        if not self._all_single_task:
            raise UnsupportedError(f"{what} requires SingleTaskGP sub-models in botorch_rs.")

    def _rs_list(self) -> Any:
        """A list of the current snapshots of the sub-models (a generic list
        of model handles if a sub-model is not a ``SingleTaskGP``)."""
        if self._all_single_task:
            return _rs.ModelListGP([m._rs for m in self.models])
        return _rs.Model.from_models([m._acquisition_model(None) for m in self.models])

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
        rs_list = self._rs_list()
        return GPyTorchPosterior(
            rs_list,
            flat,
            kind,
            observation_noise,
            num_outputs=m,
            model_batch_shape=self.batch_shape,
            output_indices=oi,
            transform=transform_arg(posterior_transform),
            points_per_input=rs_list.points_per_input,
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle that acquisition functions are built on."""
        from botorch_rs.acquisition.objective import transform_arg

        transform = transform_arg(posterior_transform, self.num_outputs)
        if not self._all_single_task:
            return _rs.Model.from_models(
                [m._acquisition_model(None) for m in self.models], transform
            )
        return _rs.Model.from_model_list(self._rs_list(), transform)

    # -- conditioning and fantasies ----------------------------------------

    def condition_on_observations(
        self, X: list[Any], Y: Any, noise: Any = None, **kwargs: Any
    ) -> ModelListGP:
        r"""The list conditioned on new observations (BoTorch's
        ``ModelListGP.condition_on_observations`` with
        meta-pytorch/botorch#3399): ``X`` holds the new inputs of every output
        (``batch_shape x n' x d``), ``Y`` the observations (``batch_shape' x n'
        x m``) and ``noise`` their noise variances in the *original* outcome
        space; every (single-output) sub-model is conditioned with its own
        input and outcome transforms. BoTorch main passes untransformed
        inputs and outcome-transformed targets to GPyTorch instead."""
        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        self._require_single_task("condition_on_observations")
        if self.training:
            raise RuntimeError(
                "Fantasy observations can only be added after making predictions with a model "
                "so that all test independent caches exist. Call the model on some data first!"
            )
        if not isinstance(X, (list, tuple)):
            raise BotorchTensorDimensionError(
                "ModelListGP.condition_on_observations expects a list of inputs, one per output."
            )
        y = to_flat(Y, "Y")
        if noise is not None:
            nz = to_flat(noise, "noise")
            if tuple(nz[1]) != tuple(y[1][len(y[1]) - len(nz[1]) :]):
                raise BotorchTensorDimensionError(
                    "The shape of observation noise does not agree with the outcomes. "
                    f"Received {tuple(nz[1])} noise with {tuple(y[1])} outcomes."
                )
        else:
            nz = None
        out = _rs.model_list_condition_on_observations(
            [m._rs for m in self.models], [to_flat(x, "X") for x in X], y, nz
        )
        return self._with_models(out)

    def fantasize(
        self,
        X: Any,
        sampler: Any,
        observation_noise: Any = None,
        evaluation_mask: Any = None,
        **kwargs: Any,
    ) -> ModelListGP:
        r"""A fantasy model (BoTorch's ``ModelList.fantasize``): every
        (single-output) sub-model is fantasized at ``X`` with its sampler of a
        ``ListSampler`` (or with a shared ``SobolQMCNormalSampler``). With an
        ``evaluation_mask`` (``n' x m``, which outputs are observed at which
        point; requires a ``ListSampler``), output ``i`` is fantasized only at
        the points ``X[..., mask[:, i], :]``. ``observation_noise``
        (``model_batch x 1 x m`` or ``x n' x m``, outcome-transformed space)
        defaults to each sub-model's noise."""
        from botorch_rs.sampling.list_sampler import ListSampler
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        self._require_single_task("fantasize")
        if isinstance(sampler, ListSampler):
            samplers, shared = [s._rs for s in sampler.samplers], False
        elif isinstance(sampler, SobolQMCNormalSampler):
            samplers, shared = [sampler._rs], True
        else:
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or ListSampler."
            )
        mask = None
        if evaluation_mask is not None:
            from botorch_rs.acquisition.cost_aware import _mask_arg

            mask = _mask_arg(evaluation_mask)
            x_shape = to_flat(X, "X")[1]
            if (
                mask is None
                or len(mask[1]) != 2
                or tuple(mask[1]) != (x_shape[-2], self.num_outputs)
            ):
                raise BotorchTensorDimensionError(
                    f"Expected evaluation_mask of shape `{x_shape[0]} x {self.num_outputs}`, "
                    f"but got {tuple(mask[1]) if mask is not None else None}."
                )
            if shared:
                raise ValueError("Decoupled fantasization requires a list of samplers.")
        self.eval()
        nz = None if observation_noise is None else to_flat(observation_noise, "observation_noise")
        out = _rs.model_list_fantasize(
            [m._rs for m in self.models], to_flat(X, "X"), samplers, shared, nz, mask
        )
        return self._with_models(out)

    def _get_group_subset_indices(self, idcs: list[int] | None) -> dict[int, list[int] | None]:
        """See ``ModelList._get_group_subset_indices``."""
        from botorch_rs.models.model import ModelList

        return ModelList._get_group_subset_indices(self, idcs)  # type: ignore[arg-type]

    def subset_output(self, idcs: list[int]) -> Model:
        r"""Subset the model along the output dimension: a ``ModelListGP`` of
        the (subset) sub-models, or the single (subset) sub-model.

        Args:
            idcs: The output indices to subset the model to. Relative to the
                overall number of outputs of the model.
        """
        from botorch_rs.models.model import ModelList

        return ModelList.subset_output(self, idcs)  # type: ignore[arg-type]

    def _with_models(self, handles: list[Any]) -> ModelListGP:
        """A list of copies of the sub-models around new Rust models."""
        new = self.__class__(*[m._with_rs(h) for m, h in zip(self.models, handles, strict=True)])
        new.training = False
        return new

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
