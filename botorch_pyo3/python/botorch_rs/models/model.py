"""Base class of the models (``botorch.models.model``)."""

from __future__ import annotations

from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError


class Model:
    """Abstract base class of botorch_rs models."""

    training: bool = True

    def train(self, mode: bool = True) -> Model:
        """Sets the training flag (for API compatibility; it has no effect)."""
        self.training = mode
        return self

    def eval(self) -> Model:
        return self.train(False)

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        """Apply an explicit input transform, or the model's transform.

        Transforms preserve the caller's array type and run their numerical
        operations in Rust. Without a transform, return ``X`` unchanged.
        """
        tf = (
            input_transform
            if input_transform is not None
            else getattr(self, "input_transform", None)
        )
        return X if tf is None else tf(X)

    def posterior(self, X: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def subset_output(self, idcs: list[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            A ``Model`` object of the same type and with the same parameters as
            the current model, subset to the specified output indices.
        """
        raise NotImplementedError

    def condition_on_observations(self, X: Any, Y: Any, **kwargs: Any) -> Model:
        r"""Condition the model on new observations (not implemented by every
        model)."""
        raise NotImplementedError(
            f"`condition_on_observations` not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def construct_inputs(cls, training_data: Any) -> dict[str, Any]:
        """
        Construct ``Model`` keyword arguments from a ``SupervisedDataset``.

        Args:
            training_data: A ``SupervisedDataset``, with attributes ``train_X``,
                ``train_Y``, and, optionally, ``train_Yvar``.

        Returns:
            A dict of keyword arguments that can be used to initialize a ``Model``,
            with keys ``train_X``, ``train_Y``, and, optionally, ``train_Yvar``.
        """
        from botorch_rs.utils.datasets import is_supervised_dataset

        if not is_supervised_dataset(training_data):
            raise TypeError(
                "Expected `training_data` to be a `SupervisedDataset`, but got "
                f"{type(training_data)}."
            )
        parsed_data = {"train_X": training_data.X, "train_Y": training_data.Y}
        if training_data.Yvar is not None:
            parsed_data["train_Yvar"] = training_data.Yvar
        return parsed_data


def _check_unsupported_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs:
        raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")


class FantasizeMixin:
    """
    Mixin to add a ``fantasize`` method to a ``Model``.

    Classes that inherit from ``FantasizeMixin`` must implement
    ``condition_on_observations`` and ``posterior`` (returning a botorch_rs
    posterior that a ``SobolQMCNormalSampler`` can sample from). botorch_rs's
    ``SingleTaskGP`` and ``ModelListGP`` implement ``fantasize`` themselves
    (in Rust).

    Example:
        class ModelThatCanFantasize(BaseModel, FantasizeMixin):
            ...

        model = ModelThatCanFantasize(...)
        model.fantasize(X, sampler)
    """

    def fantasize(
        self,
        X: Any,
        sampler: Any,
        observation_noise: Any = None,
        **kwargs: Any,
    ) -> Any:
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at ``X``, including observation noise.
        If ``observation_noise`` is a Tensor, use it directly as the observation
        noise to add.
        (2) sample from this posterior (using ``sampler``) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A ``batch_shape x n' x d``-dim Tensor.
            sampler: The sampler used for sampling from the posterior at ``X``.
            observation_noise: A ``model_batch_shape x 1 x m``-dim tensor or
                a ``model_batch_shape x n' x m``-dim tensor containing the average
                noise for each batch and output (in the outcome-transformed
                space), or None for the model's noise.
            kwargs: Will be passed to ``model.condition_on_observations``

        Returns:
            The constructed fantasy model.
        """
        from botorch_rs.exceptions.errors import DeprecationError

        if isinstance(observation_noise, bool):
            raise DeprecationError(
                "`fantasize` no longer accepts a boolean for `observation_noise`."
            )
        post_X = self.posterior(  # type: ignore[attr-defined]
            X, observation_noise=True if observation_noise is None else observation_noise
        )
        Y_fantasized = sampler(post_X)
        if observation_noise is not None:
            kwargs["noise"] = observation_noise
        return self.condition_on_observations(X=X, Y=Y_fantasized, **kwargs)  # type: ignore[attr-defined]


class ModelList(Model):
    r"""A multi-output Model represented by a list of independent models.

    All botorch_rs models are acceptable as inputs. The cost of this
    flexibility is that ``ModelList`` does not support all methods that may be
    implemented by its component models. One use case for ``ModelList`` is
    combining a regression model and a deterministic model in one
    multi-output container model, e.g. for cost-aware or multi-objective
    optimization where one of the outcomes is a deterministic function of the
    inputs.

    In acquisition functions the list is evaluated in Rust (a block-diagonal
    joint posterior over the outputs of the sub-models); every sub-model
    must be one that acquisition functions support.
    """

    def __init__(self, *models: Any) -> None:
        r"""
        Args:
            *models: A variable number of models.

        Example:
            >>> m_1 = SingleTaskGP(train_X, train_Y)
            >>> m_2 = GenericDeterministicModel(lambda x: x.sum(dim=-1), dim=2)
            >>> m_12 = ModelList(m_1, m_2)
            >>> m_12.posterior(test_X)
        """
        self.models: list[Any] = list(models)
        self.training = True

    @property
    def _ensemble_members(self) -> Any:
        """With ensemble sub-models (e.g. fully Bayesian GPs): the lists of
        their members of the same index (``AttributeError`` otherwise)."""
        from botorch_rs.models._ensemble_lists import members_method

        return members_method(self)

    @property
    def _ensemble_acquisition(self) -> Any:
        from botorch_rs.models._ensemble_lists import acquisition_method

        return acquisition_method(self)

    def _get_group_subset_indices(self, idcs: list[int] | None) -> dict[int, list[int] | None]:
        r"""Convert global subset indices to indices for the individual models.

        Args:
            idcs: A list of indices to which the ``ModelList`` model is to be
                subset to.

        Returns:
            A dictionary mapping model indices to subset indices of the
                respective model in the ``ModelList``.
        """
        if idcs is None:
            return dict.fromkeys(range(len(self.models)))
        output_sizes = [model.num_outputs for model in self.models]
        cum: list[int] = []
        total = 0
        for size in output_sizes:
            total += size
            cum.append(total)
        idcs = [idx % cum[-1] for idx in idcs]
        group_indices: dict[int, list[int] | None] = {}
        for idx in idcs:
            grp_idx = next(i for i, c in enumerate(cum) if idx < c)
            sub_idx = idx - sum(output_sizes[:grp_idx])
            group = group_indices.setdefault(grp_idx, [])
            assert group is not None
            group.append(sub_idx)
        return group_indices

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        r"""Computes the posterior over model outputs at the provided points:
        a ``PosteriorList`` of the posteriors of the sub-models.

        Args:
            X: A ``b x q x d``-dim Tensor.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior. If omitted, computes the posterior
                over all model outputs.
            observation_noise: If True, add the observation noise from the
                respective likelihoods to the posterior.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A ``Posterior`` object, representing a batch of ``b`` joint distributions
            over ``q`` points and ``m`` outputs each.
        """
        from botorch_rs.posteriors.posterior_list import PosteriorList

        group_indices = self._get_group_subset_indices(idcs=output_indices)
        posteriors = []
        for i, idcs in group_indices.items():
            model = self.models[i]
            kw: dict[str, Any] = {}
            if idcs is not None:
                kw["output_indices"] = idcs
            if observation_noise is not False:
                kw["observation_noise"] = observation_noise
            posteriors.append(model.posterior(X, **kw))
        posterior = PosteriorList(*posteriors)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior=posterior, X=X)
        return posterior

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""The batch shape of the model (the common batch shape of the
        sub-models)."""
        batch_shape = tuple(self.models[0].batch_shape or ())
        if all(batch_shape == tuple(m.batch_shape or ()) for m in self.models[1:]):
            return batch_shape
        raise NotImplementedError(
            f"`{self.__class__.__name__}.batch_shape` is only supported if all "
            "constituent models have the same `batch_shape`."
        )

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model: the sum of the numbers of
        outputs of the sub-models."""
        return sum(model.num_outputs for model in self.models)

    @property
    def _kind(self) -> Any:
        from botorch_rs._arrays import LIST

        kinds = [getattr(m, "_kind", LIST) for m in self.models]
        return next((k for k in kinds if k.lib != "list"), LIST)

    def subset_output(self, idcs: list[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to. Relative to the
                overall number of outputs of the model.

        Returns:
            A ``Model`` (either a ``ModelList`` or one of the submodels) with
            the outputs subset to the indices in ``idcs``.
        """
        group_indices = self._get_group_subset_indices(idcs=idcs)
        subset_models = []
        for grp_idx, sub_idcs in group_indices.items():
            subset_model = self.models[grp_idx]
            if sub_idcs is not None and subset_model.num_outputs != len(sub_idcs):
                subset_model = subset_model.subset_output(idcs=sub_idcs)
            subset_models.append(subset_model)
        if len(subset_models) == 1:
            return subset_models[0]
        return self.__class__(*subset_models)

    def transform_inputs(self, X: Any) -> list[Any]:
        r"""Individually transform the inputs for each model.

        Args:
            X: A tensor of inputs.

        Returns:
            A list of tensors of transformed inputs.
        """
        out = []
        for model in self.models:
            tf = getattr(model, "input_transform", None)
            out.append(X if tf is None else tf(X))
        return out

    def fantasize(
        self,
        X: Any,
        sampler: Any,
        observation_noise: Any = None,
        evaluation_mask: Any = None,
        **kwargs: Any,
    ) -> ModelList:
        r"""Construct a fantasy model: every sub-model is fantasized at ``X``
        (with its sampler of a ``ListSampler``, or the shared sampler).

        BoTorch iterates over the outputs and indexes the sub-models with the
        output index, which fails for sub-models with several outputs;
        botorch_rs fantasizes every sub-model with its outputs (its slice of
        ``observation_noise`` and ``evaluation_mask``).

        Args:
            X: A ``batch_shape x n' x d``-dim Tensor.
            sampler: The sampler used for sampling from the posterior at ``X``. If
                evaluation_mask is not None, this must be a ``ListSampler``.
            observation_noise: A ``model_batch_shape x 1 x m``-dim tensor or
                a ``model_batch_shape x n' x m``-dim tensor containing the average
                noise for each batch and output (outcome-transformed space).
            evaluation_mask: A ``n' x m``-dim tensor of booleans indicating which
                outputs should be fantasized for a given design.

        Returns:
            The constructed fantasy model.
        """
        from botorch_rs._arrays import to_flat
        from botorch_rs.exceptions.errors import BotorchTensorDimensionError
        from botorch_rs.sampling.list_sampler import ListSampler

        if evaluation_mask is not None:
            mdata, mshape = to_flat(evaluation_mask, "evaluation_mask")
            xshape = to_flat(X, "X")[1]
            if len(mshape) != 2 or tuple(mshape) != (xshape[-2], self.num_outputs):
                raise BotorchTensorDimensionError(
                    f"Expected evaluation_mask of shape `{xshape[0]} x {self.num_outputs}`, "
                    f"but got {tuple(mshape)}."
                )
            if not isinstance(sampler, ListSampler):
                raise ValueError("Decoupled fantasization requires a list of samplers.")
        fant_models = []
        start = 0
        for i, model in enumerate(self.models):
            m = model.num_outputs
            outs = list(range(start, start + m))
            start += m
            X_i = X
            noise_i = None
            if observation_noise is not None:
                noise_i = _select_last(observation_noise, outs)
            if evaluation_mask is not None:
                xdata, xshape = to_flat(X, "X")
                cols = [bool(mdata[r * self.num_outputs + outs[0]]) for r in range(xshape[-2])]
                if any(
                    [bool(mdata[r * self.num_outputs + t]) for r in range(xshape[-2])] != cols
                    for t in outs
                ):
                    raise UnsupportedError(
                        "The outputs of a multi-output sub-model must have the same "
                        "evaluation mask."
                    )
                X_i = _select_rows(X, cols)
                if noise_i is not None:
                    noise_i = _select_rows(noise_i, cols)
            sampler_i = sampler.samplers[i] if isinstance(sampler, ListSampler) else sampler
            kw = dict(kwargs)
            if noise_i is not None:
                kw["observation_noise"] = noise_i
            fant_models.append(model.fantasize(X=X_i, sampler=sampler_i, **kw))
        return self.__class__(*fant_models)

    def _acquisition_model(self, posterior_transform: Any = None) -> Any:
        """The model handle of acquisition functions: a list of the
        sub-models' handles (in Rust)."""
        from botorch_rs import _botorch_rs as _rs
        from botorch_rs.acquisition.objective import transform_arg

        handles = []
        for m in self.models:
            inner = getattr(m, "_acquisition_model", None)
            if inner is None:
                raise UnsupportedError(
                    "The sub-models of a ModelList in botorch_rs acquisition functions must be "
                    f"botorch_rs models, got {type(m).__name__}."
                )
            handles.append(inner(None))
        return _rs.Model.from_models(handles, transform_arg(posterior_transform, self.num_outputs))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(repr(m) for m in self.models)})"


def _select_last(values: Any, idcs: list[int]) -> Any:
    """``values[..., idcs]``."""
    from botorch_rs._arrays import from_flat, kind_of, to_flat

    data, shape = to_flat(values, "values")
    m = shape[-1]
    rows = len(data) // max(m, 1)
    out = [data[r * m + t] for r in range(rows) for t in idcs]
    return from_flat(out, (*shape[:-1], len(idcs)), kind_of(values))


def _select_rows(values: Any, keep: list[bool]) -> Any:
    """``values[..., keep, :]``."""
    from botorch_rs._arrays import from_flat, kind_of, to_flat

    data, shape = to_flat(values, "values")
    n, c = shape[-2], shape[-1]
    blocks = len(data) // max(n * c, 1)
    out: list[float] = []
    for b in range(blocks):
        for r in range(n):
            if keep[r]:
                out.extend(data[(b * n + r) * c : (b * n + r + 1) * c])
    return from_flat(out, (*shape[:-2], sum(keep), c), kind_of(values))


class ModelDict(dict):  # type: ignore[type-arg]
    r"""A lightweight container mapping model names to models."""

    def __init__(self, **models: Any) -> None:
        r"""Initialize a ``ModelDict``.

        Args:
            models: An arbitrary number of models. Each model can be any type
                of BoTorch ``Model``, including multi-output models and ``ModelList``.
        """
        from botorch_rs.exceptions.errors import InputDataError

        if any(not isinstance(m, Model) for m in models.values()):
            raise InputDataError(f"Expected all models to be a BoTorch `Model`. Got {models}.")
        super().__init__(**models)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


__all__ = ["FantasizeMixin", "Model", "ModelDict", "ModelList"]
