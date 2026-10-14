"""Multi-task GPs (``botorch.models.multitask``).

Only the task handling of BoTorch's ``MultiTaskGP`` is ported, as the base
class of :class:`~botorch_rs.models.contextual_multioutput.LCEMGP`: the task
feature, the tasks inferred from the data or given as ``all_tasks`` (also
tasks without observations), the output tasks and the validation of task
values. ``MultiTaskGP`` itself (the ICM kernel with an
``IndexKernel`` / ``PositiveIndexKernel`` task covariance and a ``BetaPrior``)
is not ported: creating one raises ``UnsupportedError``.
"""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model import Model


class MultiTaskGP(Model):
    r"""Base class of the multi-task GPs (BoTorch's ``MultiTaskGP``).

    ``MultiTaskGP.eval()`` only sets the evaluation flag. BoTorch's version
    also initializes the ``IndexKernel`` factors of tasks without data, which
    fails for ``LCEMGP`` (it has no ``IndexKernel``); see the deviations of
    ``LCEMGP``.
    """

    _supports_batched_models = False
    _supports_cache_root = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise UnsupportedError(
            "MultiTaskGP (the ICM kernel with an IndexKernel / PositiveIndexKernel task "
            "covariance) is not ported to botorch_rs; LCEMGP (botorch_rs.models."
            "contextual_multioutput) is."
        )

    @classmethod
    def get_all_tasks(
        cls,
        train_X: Any,
        task_feature: int,
        output_tasks: list[int] | None = None,
    ) -> tuple[list[int], int, int]:
        """The sorted task values of ``train_X``, the (non-negative) task
        feature and the number of non-task features (BoTorch's
        ``get_all_tasks``)."""
        data, shape = to_flat(train_X, "train_X")
        if len(shape) != 2:
            raise ValueError(f"Unsupported shape {tuple(shape)} for train_X.")
        d = shape[-1] - 1
        if not (-d <= task_feature <= d):
            raise ValueError(f"Must have that -{d} <= task_feature <= {d}")
        task_feature = task_feature % (d + 1)
        n, c = shape
        all_tasks = sorted({int(data[i * c + task_feature]) for i in range(n)})
        return all_tasks, task_feature, d


__all__ = ["MultiTaskGP"]
