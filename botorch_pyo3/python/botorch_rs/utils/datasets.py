"""Representations for different kinds of datasets (``botorch.utils.datasets``):
``SupervisedDataset``, the input of the models' ``construct_inputs``."""

from __future__ import annotations

import copy
from typing import Any

from botorch_rs._arrays import to_flat


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
        return self._X

    @property
    def Y(self) -> Any:
        return self._Y

    @property
    def Yvar(self) -> Any:
        return self._Yvar

    def _validate(
        self,
        validate_feature_names: bool = True,
        validate_outcome_names: bool = True,
        validate_group_indices: bool = True,
    ) -> None:
        r"""Checks that the shapes of the inputs are compatible with each other."""
        shape_X = _shape(self.X)
        shape_Y = _shape(self.Y)
        if shape_X[:-1] != shape_Y[:-1]:
            raise ValueError("Batch dimensions of `X` and `Y` are incompatible.")
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
            if _shape(self.group_indices) != shape_X[:-1]:
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


__all__ = ["SupervisedDataset"]
