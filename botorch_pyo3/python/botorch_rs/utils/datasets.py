"""Representations for different kinds of datasets (``botorch.utils.datasets``):
``SupervisedDataset`` (the input of the models' ``construct_inputs``),
``RankingDataset``, ``MultiTaskDataset`` and ``ContextualDataset``.

The data are arrays (torch tensors, NumPy arrays or nested lists) or
BoTorch-style containers (``DenseContainer``, ``SliceContainer`` of
botorch_rs or BoTorch)."""

from __future__ import annotations

import copy
import math
import sys
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.exceptions.errors import InputDataError, UnsupportedError


def _shape(x: Any) -> tuple[int, ...]:
    return tuple(to_flat(x, "data")[1])


def _equal(a: Any, b: Any) -> bool:
    fa, fb = to_flat(a, "data"), to_flat(b, "data")
    return tuple(fa[1]) == tuple(fb[1]) and list(fa[0]) == list(fb[0])


class SupervisedDataset:
    r"""Base class for datasets consisting of labelled pairs ``(X, Y)``
    and an optional ``Yvar`` that stipulates observations variances so
    that ``Y[i] ~ N(f(X[i]), Yvar[i])``.

    ``X``, ``Y`` and ``Yvar`` are arrays (torch tensors, NumPy arrays or
    nested lists); BoTorch's ``BotorchContainer``\ s are not supported.

    Example:

    .. code-block:: python

        X = torch.rand(16, 2)
        Y = torch.rand(16, 1)
        feature_names = ["learning_rate", "embedding_dim"]
        outcome_names = ["neg training loss"]
        A = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
    """

    def __init__(
        self,
        X: Any,
        Y: Any,
        *,
        feature_names: list[str],
        outcome_names: list[str],
        Yvar: Any = None,
        validate_init: bool = True,
        group_indices: Any = None,
    ) -> None:
        r"""Constructs a ``SupervisedDataset``.

        Args:
            X: A ``Tensor`` representing the input features.
            Y: A ``Tensor`` representing the outcomes.
            feature_names: A list of names of the features in ``X``.
            outcome_names: A list of names of the outcomes in ``Y``.
            Yvar: An optional ``Tensor`` representing the observation noise.
            validate_init: If ``True``, validates the input shapes.
            group_indices: A ``Tensor`` representing the which rows of X and Y are
                grouped together. If provided, its shape must be compatible
                with X and Y.
        """
        self._X = X
        self._Y = Y
        self._Yvar = Yvar
        self.feature_names = list(feature_names).copy()
        self.outcome_names = list(outcome_names).copy()
        self.group_indices = group_indices
        self.validate_init = validate_init
        if validate_init:
            self._validate()

    @property
    def X(self) -> Any:
        return _unwrap(self._X)

    @property
    def Y(self) -> Any:
        return _unwrap(self._Y)

    @property
    def Yvar(self) -> Any:
        return None if self._Yvar is None else _unwrap(self._Yvar)

    def _validate(
        self,
        validate_feature_names: bool = True,
        validate_outcome_names: bool = True,
        validate_group_indices: bool = True,
    ) -> None:
        r"""Checks that the shapes of the inputs are compatible with each other."""
        full_X = _shape(self.X)
        full_Y = _shape(self.Y)
        shape_X = full_X[: len(full_X) - _event_ndim(self._X)]
        shape_Y = full_Y[: len(full_Y) - _event_ndim(self._Y)]
        if shape_X != shape_Y:
            raise ValueError("Batch dimensions of `X` and `Y` are incompatible.")
        shape_Y = full_Y
        shape_X = full_X
        if self.Yvar is not None and _shape(self.Yvar) != shape_Y:
            raise ValueError("Shapes of `Y` and `Yvar` are incompatible.")
        if validate_feature_names and len(self.feature_names) != shape_X[-1]:
            raise ValueError(
                "`X` must have the same number of columns as the number of "
                "features in `feature_names`."
            )
        if validate_outcome_names and len(self.outcome_names) != shape_Y[-1]:
            raise ValueError(
                "`Y` must have the same number of columns as the number of "
                "outcomes in `outcome_names`."
            )
        if validate_group_indices and self.group_indices is not None:
            if _shape(self.group_indices) != shape_X[: len(shape_X) - _event_ndim(self._X)]:
                raise ValueError(
                    f"shape_X ({shape_X[:-1]}) must have the same shape as "
                    f"group_indices ({_shape(self.group_indices)})."
                )

    def __eq__(self, other: Any) -> bool:
        if self.group_indices is None and other.group_indices is None:
            group_indices_equal = True
        elif self.group_indices is None or other.group_indices is None:
            group_indices_equal = False
        else:
            group_indices_equal = _equal(self.group_indices, other.group_indices)
        return (
            type(other) is type(self)
            and _equal(self.X, other.X)
            and _equal(self.Y, other.Y)
            and (
                other.Yvar is None
                if self.Yvar is None
                else other.Yvar is not None and _equal(self.Yvar, other.Yvar)
            )
            and self.feature_names == other.feature_names
            and self.outcome_names == other.outcome_names
            and group_indices_equal
        )

    __hash__ = None  # type: ignore[assignment]

    def clone(self, deepcopy: bool = False, mask: Any = None) -> SupervisedDataset:
        """Return a copy of the dataset.

        Args:
            deepcopy: If True, perform a deep copy. Otherwise, use the same
                tensors/lists.
            mask: A ``n``-dim boolean mask indicating which rows to keep. This is used
                along the -2 dimension.

        Returns:
            The new dataset.
        """
        new_X, new_Y, new_Yvar = self._X, self._Y, self._Yvar
        feature_names, outcome_names = self.feature_names, self.outcome_names
        if mask is not None:
            if any(_is_container(x) for x in [new_X, new_Y, new_Yvar]):
                raise NotImplementedError("Masking is not supported for BotorchContainers.")
            new_X = _mask_rows(new_X, mask)
            new_Y = _mask_rows(new_Y, mask)
            if new_Yvar is not None:
                new_Yvar = _mask_rows(new_Yvar, mask)
        if deepcopy:
            new_X, new_Y, new_Yvar = (copy.deepcopy(v) for v in (new_X, new_Y, new_Yvar))
            feature_names = copy.copy(self.feature_names)
            outcome_names = copy.copy(self.outcome_names)
        kwargs = {}
        if new_Yvar is not None:
            kwargs = {"Yvar": new_Yvar}
        return type(self)(
            X=new_X,
            Y=new_Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
            validate_init=self.validate_init,
            **kwargs,
        )


def _is_container(x: Any) -> bool:
    return any(c.__name__ == "BotorchContainer" for c in type(x).__mro__)


def _unwrap(x: Any) -> Any:
    """The array of a container (``container()``) or the array itself."""
    return x() if _is_container(x) else x


def _event_ndim(x: Any) -> int:
    return len(tuple(x.event_shape)) if _is_container(x) else 1


def _mask_rows(x: Any, mask: Any) -> Any:
    if isinstance(x, list):
        keep = [bool(v) for v in to_flat(mask, "mask")[0]]
        if x and isinstance(x[0], list) and x[0] and isinstance(x[0][0], list):
            return [_mask_rows(v, mask) for v in x]
        return [row for row, k in zip(x, keep, strict=True) if k]
    return x[..., mask, :]


def is_supervised_dataset(data: Any) -> bool:
    """Whether ``data`` is a ``SupervisedDataset`` of botorch_rs or BoTorch."""
    if isinstance(data, SupervisedDataset):
        return True
    return any(
        c.__name__ == "SupervisedDataset" and c.__module__.startswith("botorch.")
        for c in type(data).__mro__
    )


class RankingDataset(SupervisedDataset):
    r"""A SupervisedDataset whose labelled pairs ``(x, y)`` consist of m-ary
    combinations ``x ∈ Z^{m}`` of elements from a ground set ``Z = (z_1, ...)``
    and ranking vectors ``y {0, ..., m - 1}^{m}`` with properties:

        a) Ranks start at zero, i.e. min(y) = 0.
        b) Sorted ranks are contiguous unless one or more ties are present.
        c) ``k`` ranks are skipped after a ``k``-way tie.

    Example:

    .. code-block:: python

        X = SliceContainer(
            values=torch.rand(16, 2),
            indices=torch.stack([torch.randperm(16)[:3] for _ in range(8)]),
            event_shape=torch.Size([3 * 2]),
        )
        Y = DenseContainer(
            torch.stack([torch.randperm(3) for _ in range(8)]),
            event_shape=torch.Size([3])
        )
        feature_names = ["item_0", "item_1"]
        outcome_names = ["ranking outcome"]
        dataset = RankingDataset(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
        )
    """

    def __init__(
        self,
        X: Any,
        Y: Any,
        feature_names: list[str],
        outcome_names: list[str],
        validate_init: bool = True,
    ) -> None:
        r"""Construct a ``RankingDataset``.

        Args:
            X: A ``SliceContainer`` representing the input features being ranked.
            Y: A ``Tensor`` or ``BotorchContainer`` representing the rankings.
            feature_names: A list of names of the features in X.
            outcome_names: A list of names of the outcomes in Y.
            validate_init: If ``True``, validates the input shapes.
        """
        super().__init__(
            X=X,
            Y=Y,
            feature_names=feature_names,
            outcome_names=outcome_names,
            Yvar=None,
            validate_init=validate_init,
        )

    def _validate(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        super()._validate(
            validate_feature_names=False,
            validate_outcome_names=False,
            validate_group_indices=False,
        )
        if len(self.feature_names) != _shape(self._X.values)[-1]:
            raise ValueError(
                "The `values` field of `X` must have the same number of columns as "
                "the number of features in `feature_names`."
            )

        data, shape = to_flat(self.Y, "Y")
        Y = [float(v) for v in data]
        arity = _shape(self._X.indices)[-1]
        if min(Y) < 0 or max(Y) >= arity:
            raise ValueError("Invalid ranking(s): out-of-bounds ranks detected.")

        # Ensure that rankings are well-defined
        k = shape[-1]
        rows = [sorted(Y[i : i + k]) for i in range(0, len(Y), k)]
        nrows = len(rows)
        y_incr = [1.0] * nrows
        y_prev: list[float] = []
        for i in range(k):
            y = [r[i] for r in rows]
            if i == 0:
                if any(v != 0 for v in y):
                    raise ValueError("Invalid ranking(s): missing zero-th rank.")
                y_prev = y
                continue
            y_diff = [a - b for a, b in zip(y, y_prev, strict=True)]
            y_prev = y
            # Either a tie or next ranking when accounting for previous ties
            if not all(d == 0 or d == inc for d, inc in zip(y_diff, y_incr, strict=True)):
                raise ValueError("Invalid ranking(s): ranks not skipped after ties.")
            # Same as: torch.where(y_diff == 0, y_incr + 1, 1)
            y_incr = [inc - d + 1 for inc, d in zip(y_incr, y_diff, strict=True)]


def _concat_rows(xs: list[Any], dim: int) -> Any:
    return _ops.concat(xs, dim=dim)


def _pad_column(X: Any, value: float) -> Any:
    """``X`` (``n x d``) with a column of ``value`` appended."""
    if is_torch_tensor(X):
        return sys.modules["torch"].nn.functional.pad(X, (0, 1), value=value)
    data, shape = to_flat(X, "X")
    d = shape[-1]
    n = len(data) // d if d else 0
    out: list[float] = []
    for i in range(n):
        out.extend(float(v) for v in data[i * d : (i + 1) * d])
        out.append(float(value))
    return from_flat(out, (*shape[:-1], d + 1), kind_of(X))


def _select_rows(X: Any, keep: list[bool]) -> Any:
    if is_torch_tensor(X):
        torch = sys.modules["torch"]
        return X[torch.tensor(keep, dtype=torch.bool, device=X.device)]
    np = sys.modules.get("numpy")
    if np is not None and isinstance(X, np.ndarray):
        return X[np.asarray(keep, dtype=bool)]
    data, shape = to_flat(X, "X")
    row = math.prod(shape[1:])
    rows = [i for i, k in enumerate(keep) if k]
    out = [float(v) for i in rows for v in data[i * row : (i + 1) * row]]
    return from_flat(out, (len(rows), *shape[1:]), kind_of(X))


def _select_columns(X: Any, indices: list[int]) -> Any:
    if is_torch_tensor(X) or kind_of(X).lib == "numpy":
        return X[..., indices]
    data, shape = to_flat(X, "X")
    d = shape[-1]
    out = [float(data[i + j]) for i in range(0, len(data), d) for j in indices]
    return from_flat(out, (*shape[:-1], len(indices)), kind_of(X))


class MultiTaskDataset(SupervisedDataset):
    """This is a multi-task dataset that is constructed from the datasets of
    individual tasks. It offers functionality to combine parts of individual
    datasets to construct the inputs necessary for the ``MultiTaskGP`` models.

    The datasets of individual tasks are allowed to represent different sets
    of features. When there are heterogeneous feature sets, calling
    ``MultiTaskDataset.X`` will result in an error.
    """

    def __init__(
        self,
        datasets: list[SupervisedDataset],
        target_outcome_name: str,
        task_feature_index: int | None = None,
    ) -> None:
        """Construct a ``MultiTaskDataset``.

        Args:
            datasets: A list of the datasets of individual tasks. Each dataset
                is expected to contain data for only one outcome.
            target_outcome_name: Name of the target outcome to be modeled.
            task_feature_index: If the task feature is included in the Xs of the
                individual datasets, this should be used to specify its index.
                If omitted, the task feature will be appended while concatenating Xs.
                If given, we sanity-check that the names of the task features
                match between all datasets.
        """
        self.datasets: dict[str, SupervisedDataset] = {ds.outcome_names[0]: ds for ds in datasets}
        self.target_outcome_name = target_outcome_name
        self.task_feature_index = task_feature_index
        self._validate_datasets(datasets=datasets)
        self.feature_names = self.datasets[target_outcome_name].feature_names.copy()
        self.outcome_names = [target_outcome_name]

        # Check if the datasets have identical feature sets.
        self.has_heterogeneous_features = any(
            datasets[0].feature_names != ds.feature_names for ds in datasets[1:]
        )
        self.group_indices = None
        self.validate_init = True

    @classmethod
    def from_joint_dataset(
        cls,
        dataset: SupervisedDataset,
        task_feature_index: int,
        target_task_value: int,
        outcome_names_per_task: dict[int, str] | None = None,
    ) -> MultiTaskDataset:
        r"""Construct a ``MultiTaskDataset`` from a joint dataset that includes the
        data for all tasks with the task feature index.

        This will break down the joint dataset into individual datasets by the value
        of the task feature. Each resulting dataset will have its outcome name set
        based on ``outcome_names_per_task``, with the missing values defaulting to
        ``task_<task_feature>`` (except for the target task, which will retain the
        original outcome name from the dataset).

        Args:
            dataset: The joint dataset.
            task_feature_index: The column index of the task feature in ``dataset.X``.
            target_task_value: The value of the task feature for the target task
                in the dataset. The data for the target task is filtered according to
                ``dataset.X[task_feature_index] == target_task_value``.
            outcome_names_per_task: Optional dictionary mapping task feature values
                to the outcome names for each task. If not provided, the auxiliary
                tasks will be named ``task_<task_feature>`` and the target task will
                retain the outcome name from the dataset.

        Returns:
            A ``MultiTaskDataset`` instance.
        """
        if len(dataset.outcome_names) > 1:
            raise UnsupportedError(
                "Dataset containing more than one outcome is not supported. "
                f"Got {dataset.outcome_names=}."
            )
        outcome_names_per_task = outcome_names_per_task or {}
        # Split datasets by task feature.
        datasets = []
        data, shape = to_flat(dataset.X, "X")
        d = shape[-1]
        tf = task_feature_index % d
        all_task_features = [float(data[i * d + tf]) for i in range(len(data) // d)]
        for task_value in sorted({int(v) for v in all_task_features}):
            default_name = (
                dataset.outcome_names[0] if task_value == target_task_value else f"task_{task_value}"
            )
            outcome_name = outcome_names_per_task.get(task_value, default_name)
            keep = [v == task_value for v in all_task_features]
            new_dataset = SupervisedDataset(
                X=_select_rows(dataset.X, keep),
                Y=_select_rows(dataset.Y, keep),
                Yvar=_select_rows(dataset.Yvar, keep) if dataset.Yvar is not None else None,
                feature_names=dataset.feature_names,
                outcome_names=[outcome_name],
            )
            datasets.append(new_dataset)
        # Return the new dataset
        return cls(
            datasets=datasets,
            target_outcome_name=outcome_names_per_task.get(
                target_task_value, dataset.outcome_names[0]
            ),
            task_feature_index=task_feature_index,
        )

    def _validate_datasets(self, datasets: list[SupervisedDataset]) -> None:
        """Validates that:
        * Each dataset models only one outcome;
        * Each outcome is modeled by only one dataset;
        * The target outcome is included in the datasets;
        * The datasets do not model batched inputs;
        * The task feature names of the datasets all match;
        * Either all or none of the datasets specify Yvar.
        """
        if any(len(ds.outcome_names) > 1 for ds in datasets):
            raise UnsupportedError("Datasets containing more than one outcome are not supported.")
        if len(self.datasets) != len(datasets):
            raise UnsupportedError(
                "Received multiple datasets for the same outcome. Each dataset "
                "must contain data for a unique outcome. Got datasets with "
                f"outcome names: {[ds.outcome_names for ds in datasets]}."
            )
        if self.target_outcome_name not in self.datasets:
            raise InputDataError(
                "Target outcome is not present in the datasets. "
                f"Got self.target_outcome_name={self.target_outcome_name!r} and datasets for "
                f"outcomes {list(self.datasets.keys())}."
            )
        if any(len(_shape(ds.X)) > 2 for ds in datasets):
            raise UnsupportedError("Datasets modeling batched inputs are not supported.")
        if self.task_feature_index is not None:
            tf_names = [ds.feature_names[self.task_feature_index] for ds in datasets]
            if any(name != tf_names[0] for name in tf_names[1:]):
                raise InputDataError(
                    "Expected the names of the task features to match across all "
                    f"datasets. Got {tf_names}."
                )
        all_Yvars = [ds.Yvar for ds in datasets]
        is_none = [yvar is None for yvar in all_Yvars]
        # Check that either all or None of the Yvars exist.
        if not all(is_none) and any(is_none):
            raise UnsupportedError(
                "Expected either all or none of the datasets to have a Yvar. "
                "Only subset of datasets define Yvar, which is unsupported."
            )

    @property
    def X(self) -> Any:
        """Appends task features, if needed, and concatenates the Xs of datasets to
        produce the ``train_X`` expected by ``MultiTaskGP`` and subclasses.

        If appending the task features, 0 is reserved for the target task and the
        remaining tasks are populated with 1, 2, ..., len(datasets) - 1.
        """
        if self.has_heterogeneous_features:
            raise UnsupportedError(
                "Concatenating `X`s from datasets with heterogeneous feature sets "
                "is not supported."
            )
        all_Xs = []
        next_task = 1
        for outcome, ds in self.datasets.items():
            if self.task_feature_index is None:
                # Append the task feature index.
                if outcome == self.target_outcome_name:
                    task_feature = 0
                else:
                    task_feature = next_task
                    next_task = next_task + 1
                all_Xs.append(_pad_column(ds.X, task_feature))
            else:
                all_Xs.append(ds.X)
        return _concat_rows(all_Xs, dim=0)

    @property
    def Y(self) -> Any:
        """Concatenates Ys of the datasets."""
        return _concat_rows([ds.Y for ds in self.datasets.values()], dim=0)

    @property
    def Yvar(self) -> Any:
        """Concatenates Yvars of the datasets if they exist."""
        all_Yvars = [ds.Yvar for ds in self.datasets.values()]
        return None if all_Yvars[0] is None else _concat_rows(all_Yvars, dim=0)

    def get_dataset_without_task_feature(self, outcome_name: str) -> SupervisedDataset:
        """A helper for extracting the child datasets with their task features removed.

        If the task feature index is ``None``, the dataset will be returned as is.

        Args:
            outcome_name: The outcome name for the dataset to extract.

        Returns:
            The dataset without the task feature.
        """
        dataset = self.datasets[outcome_name]
        if self.task_feature_index is None:
            return dataset
        indices = list(range(len(self.feature_names)))
        indices.pop(self.task_feature_index)
        return SupervisedDataset(
            X=_select_columns(dataset.X, indices),
            Y=dataset.Y,
            Yvar=dataset.Yvar,
            feature_names=[fn for i, fn in enumerate(dataset.feature_names) if i in indices],
            outcome_names=[outcome_name],
        )

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and self.datasets == other.datasets
            and self.target_outcome_name == other.target_outcome_name
            and self.task_feature_index == other.task_feature_index
        )

    __hash__ = None  # type: ignore[assignment]

    def get_heterogeneous_feature_mapping(
        self,
    ) -> tuple[list[SupervisedDataset], list[list[int]], int]:
        """Compute canonical feature ordering for heterogeneous datasets.

        Target features come first (preserving order), then source-only
        features are appended. The task column (at ``task_feature_index``)
        is excluded from the mapping.

        Returns:
            A 3-tuple of:
            - Ordered datasets (target first, then sources).
            - Feature indices mapping each dataset's non-task features
              to the canonical ordering.
            - Full feature dimensionality (number of unique non-task features).

        Raises:
            NotImplementedError: If ``task_feature_index`` is not ``-1``.
        """
        if self.task_feature_index != -1:
            raise NotImplementedError(
                "Heterogeneous feature mapping requires `task_feature_index` to be -1."
            )
        child_datasets = self.datasets.copy()
        target_dataset = child_datasets.pop(self.target_outcome_name)
        all_datasets = [target_dataset] + list(child_datasets.values())

        # Target's feature order is canonical; source-only features appended.
        all_features: list[str] = list(target_dataset.feature_names[:-1])
        for ds in all_datasets[1:]:
            for fn in ds.feature_names[:-1]:
                if fn not in all_features:
                    all_features.append(fn)

        feature_indices = [
            [all_features.index(fn) for fn in ds.feature_names[:-1]] for ds in all_datasets
        ]
        return all_datasets, feature_indices, len(all_features)

    def clone(self, deepcopy: bool = False, mask: Any = None) -> MultiTaskDataset:
        """Return a copy of the dataset.

        Args:
            deepcopy: If True, perform a deep copy. Otherwise, use the same
                tensors/lists/datasets.
            mask: A ``n``-dim boolean mask indicating which rows to keep from the target
                dataset. This is used along the -2 dimension.

        Returns:
            The new dataset.
        """
        datasets = list(self.datasets.values())
        if mask is not None or deepcopy:
            new_datasets = []
            for outcome, ds in self.datasets.items():
                new_datasets.append(
                    ds.clone(
                        deepcopy=deepcopy,
                        mask=mask if outcome == self.target_outcome_name else None,
                    )
                )
            datasets = new_datasets
        return MultiTaskDataset(
            datasets=datasets,
            target_outcome_name=self.target_outcome_name,
            task_feature_index=self.task_feature_index,
        )


class ContextualDataset(SupervisedDataset):
    """This is a contextual dataset that is constructed from either a single
    dateset containing overall outcome or a list of datasets that each corresponds
    to a context breakdown.
    """

    def __init__(
        self,
        datasets: list[SupervisedDataset],
        parameter_decomposition: dict[str, list[str]],
        metric_decomposition: dict[str, list[str]] | None = None,
    ) -> None:
        """Construct a ``ContextualDataset``.

        Args:
            datasets: A list of the datasets of individual tasks. Each dataset
                is expected to contain data for only one outcome.
            parameter_decomposition: Dict from context name to list of feature
                names corresponding to that context.
            metric_decomposition: Context breakdown metrics. Keys are context names.
                Values are the lists of metric names belonging to the context:
                {'context1': ['m1_c1'], 'context2': ['m1_c2'],}.
        """
        self.datasets: dict[str, SupervisedDataset] = {ds.outcome_names[0]: ds for ds in datasets}
        self.parameter_decomposition = parameter_decomposition
        self.metric_decomposition = metric_decomposition
        self._validate_datasets()
        self._validate_decompositions()
        self.context_buckets = self._extract_context_buckets()
        self.parameter_index_decomp = {
            c: [self.feature_names.index(i) for i in parameter_decomposition[c]]
            for c in self.context_buckets
        }
        self.group_indices = None
        self.validate_init = True

    @property  # type: ignore[override]
    def feature_names(self) -> list[str]:
        return self.datasets[self.outcome_names[0]].feature_names

    @feature_names.setter
    def feature_names(self, value: list[str]) -> None:  # pragma: no cover - read-only
        raise AttributeError("can't set attribute 'feature_names'")

    @property  # type: ignore[override]
    def outcome_names(self) -> list[str]:
        return list(self.datasets.keys())

    @outcome_names.setter
    def outcome_names(self, value: list[str]) -> None:  # pragma: no cover - read-only
        raise AttributeError("can't set attribute 'outcome_names'")

    @property
    def X(self) -> Any:
        return self.datasets[self.outcome_names[0]].X

    @property
    def Y(self) -> Any:
        """Concatenates the Ys from the child datasets to create the Y expected
        by LCEM model if there are multiple datasets; Or return the Y expected
        by LCEA model if there is only one dataset.
        """
        Ys = [ds.Y for ds in self.datasets.values()]
        if len(Ys) == 1:
            return Ys[0]
        return _concat_rows(Ys, dim=-1)

    @property
    def Yvar(self) -> Any:
        """Concatenates the Yvars from the child datasets to create the Y expected
        by LCEM model if there are multiple datasets; Or return the Yvar expected
        by LCEA model if there is only one dataset.
        """
        Yvars = [ds.Yvar for ds in self.datasets.values()]
        if Yvars[0] is None:
            return None
        elif len(Yvars) == 1:
            return Yvars[0]
        return _concat_rows(Yvars, dim=-1)

    def _extract_context_buckets(self) -> list[str]:
        """Determines the context buckets from the data, and sets the
        context_buckets attribute.

        If we have an outcome for each context, we will lists the contexts
        in the same order as the outcomes (i.e., the order of datasets).

        If there is a single outcome (aggregated across contexts), the context
        buckets are taken from the parameter decomposition.
        """
        if len(self.outcome_names) > 1:
            assert self.metric_decomposition is not None
            assert len(self.outcome_names) == len(self.metric_decomposition), (
                "Expected a single dataset, or one for each context bucket."
            )
            context_buckets = []
            for outcome_name in self.outcome_names:
                for k, v in self.metric_decomposition.items():
                    if outcome_name in v:
                        context_buckets.append(k)
                        break
        else:
            context_buckets = list(self.parameter_decomposition.keys())
        return context_buckets

    def _validate_datasets(self) -> None:
        """Validation of given datasets.
        1. each dataset has same X.
        2. metric_decomposition is not None if there are multiple datasets.
        3. metric_decomposition contains all the outcomes in datasets.
        4. value keys of parameter decomposition and the keys of
        metric_decomposition match context buckets.
        5. Yvar is None for all, or not for all.
        """
        datasets = list(self.datasets.values())
        X = datasets[0].X
        Yvar_is_none = datasets[0].Yvar is None
        for dataset in datasets:
            if not _equal(X, dataset.X):
                raise InputDataError("Require same X for context buckets")
            if (dataset.Yvar is None) != Yvar_is_none:
                raise InputDataError("Require Yvar to be specified for all buckets, or for none")

        if len(datasets) > 1:
            if self.metric_decomposition is None:
                raise InputDataError(
                    "metric_decomposition must be provided when there are" + " multiple datasets."
                )
        else:
            if self.metric_decomposition is not None:
                raise InputDataError(
                    "metric_decomposition is redundant when there is one "
                    + "dataset for overall outcome."
                )

    def _validate_decompositions(self) -> None:
        """Checks that the decompositions are valid.

        Raises:
            InputDataError: If any of the decompositions are invalid.
        """
        if self.metric_decomposition is not None:
            m = len(list(self.metric_decomposition.values())[0])
            existing_metrics: set[str] = set()
            for v in self.metric_decomposition.values():
                if existing_metrics.intersection(list(v)):
                    raise InputDataError(
                        "metric_decomposition has same metric for multiple contexts."
                    )
                if len(v) != m or len(set(v)) != m:
                    raise InputDataError(
                        "All values in metric_decomposition must have the same length."
                    )
                existing_metrics.update(list(v))

            if set(self.metric_decomposition.keys()) != set(self.parameter_decomposition.keys()):
                raise InputDataError("Keys of metric and parameter decompositions do not match.")

            all_metrics = []
            for mm in self.metric_decomposition.values():
                all_metrics.extend(mm)
            for outcome in self.outcome_names:
                if outcome not in all_metrics:
                    raise InputDataError(f"{outcome} is missing in metric_decomposition.")

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and self.datasets == other.datasets
            and self.parameter_decomposition == other.parameter_decomposition
            and self.metric_decomposition == other.metric_decomposition
        )

    __hash__ = None  # type: ignore[assignment]

    def clone(self, deepcopy: bool = False, mask: Any = None) -> ContextualDataset:
        """Return a copy of the dataset.

        Args:
            deepcopy: If True, perform a deep copy. Otherwise, use the same
                tensors/lists/datasets.
            mask: A ``n``-dim boolean mask indicating which rows to keep. This is used
                along the -2 dimension. ``n`` here corresponds to the number of rows in
                an individual dataset.

        Returns:
            The new dataset.
        """
        datasets = list(self.datasets.values())
        if mask is not None or deepcopy:
            datasets = [ds.clone(deepcopy=deepcopy, mask=mask) for ds in datasets]
        if deepcopy:
            parameter_decomposition = copy.deepcopy(self.parameter_decomposition)
            metric_decomposition = copy.deepcopy(self.metric_decomposition)
        else:
            parameter_decomposition = self.parameter_decomposition
            metric_decomposition = self.metric_decomposition
        return ContextualDataset(
            datasets=datasets,
            parameter_decomposition=parameter_decomposition,
            metric_decomposition=metric_decomposition,
        )


__all__ = ["ContextualDataset", "MultiTaskDataset", "RankingDataset", "SupervisedDataset"]
