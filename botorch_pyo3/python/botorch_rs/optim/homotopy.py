"""Homotopy schedules and parameters (``botorch.optim.homotopy``).

A homotopy parameter is any object with ``item()`` and ``data.fill_(value)``
(a torch tensor or parameter used by a torch-callable acquisition function
or model, as in BoTorch, or a :class:`HomotopyValue`). Callbacks run after
every change, e.g. to rebuild a botorch_rs acquisition function that reads
the parameter.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _linspace(start: float, end: float, steps: int) -> list[float]:
    """``torch.linspace(start, end, steps, dtype=torch.double).tolist()``
    (torch's symmetric formula)."""
    if steps <= 0:
        return []
    if steps == 1:
        return [float(start)]
    step = (end - start) / (steps - 1)
    half = steps // 2
    return [start + step * i if i < half else end - step * (steps - 1 - i) for i in range(steps)]


class FixedHomotopySchedule:
    """Homotopy schedule with a fixed list of values."""

    def __init__(self, values: list[float]) -> None:
        r"""Initialize FixedHomotopySchedule.

        Args:
            values: A list of values used in homotopy.
        """
        self._values = values
        self.idx = 0

    @property
    def num_steps(self) -> int:
        return len(self._values)

    @property
    def value(self) -> float:
        return self._values[self.idx]

    @property
    def should_stop(self) -> bool:
        return self.idx == len(self._values)

    def restart(self) -> None:
        self.idx = 0

    def step(self) -> None:
        self.idx += 1


class LinearHomotopySchedule(FixedHomotopySchedule):
    """Linear homotopy schedule."""

    def __init__(self, start: float, end: float, num_steps: int) -> None:
        r"""Initialize LinearHomotopySchedule.

        Args:
            start: start value of homotopy
            end: end value of homotopy
            num_steps: number of steps in the homotopy schedule.
        """
        super().__init__(values=_linspace(float(start), float(end), int(num_steps)))


class LogLinearHomotopySchedule(FixedHomotopySchedule):
    """Log-linear homotopy schedule."""

    def __init__(self, start: float, end: float, num_steps: int) -> None:
        r"""Initialize LogLinearHomotopySchedule.

        Args:
            start: start value of homotopy
            end: end value of homotopy
            num_steps: number of steps in the homotopy schedule.
        """
        exponents = _linspace(math.log10(start), math.log10(end), int(num_steps))
        super().__init__(values=[10.0**e for e in exponents])


class _Data:
    def __init__(self, owner: HomotopyValue) -> None:
        self._owner = owner

    def fill_(self, value: float) -> None:
        self._owner.value = float(value)


class HomotopyValue:
    """A scalar homotopy parameter without torch: ``item()`` and
    ``data.fill_(value)`` read and set :attr:`value` (a botorch_rs
    extension; read it in a ``Homotopy`` callback)."""

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.data = _Data(self)

    def item(self) -> float:
        return self.value

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"HomotopyValue({self.value})"


@dataclass(slots=True)
class HomotopyParameter:
    r"""Homotopy parameter.

    The parameter is expected to either be a torch parameter or a torch tensor
    which may correspond to a buffer of a module (or a :class:`HomotopyValue`).
    The parameter has a corresponding schedule.
    """

    parameter: Any
    schedule: FixedHomotopySchedule


class Homotopy:
    """Generic homotopy class (BoTorch's ``Homotopy``).

    This class is designed to be used in ``optimize_acqf_homotopy``. Given a
    set of homotopy parameters and corresponding schedules we step through
    the homotopies until we have solved the final problem. We additionally
    support passing in a list of callbacks that will be executed each time
    ``step``, ``reset``, and ``restart`` are called.
    """

    def __init__(
        self,
        homotopy_parameters: list[HomotopyParameter],
        callbacks: list[Callable[[], Any]] | None = None,
    ) -> None:
        r"""Initialize the homotopy.

        Args:
            homotopy_parameters: List of homotopy parameters
            callbacks: Optional list of callbacks that are executed each time
                ``restart``, ``reset``, or ``step`` are called. These may be
                used to, e.g., reinitialize the acquisition function.
        """
        self._homotopy_parameters = homotopy_parameters
        self._callbacks = callbacks or []
        self._original_values = [hp.parameter.item() for hp in self._homotopy_parameters]
        assert all(
            hasattr(hp.parameter, "item") and hasattr(hp.parameter, "data")
            for hp in self._homotopy_parameters
        )
        # Assume the same number of steps for now
        assert len({h.schedule.num_steps for h in self._homotopy_parameters}) == 1
        # Initialize the homotopy parameters
        self.restart()

    def _execute_callbacks(self) -> None:
        """Execute the callbacks."""
        for callback in self._callbacks:
            callback()

    @property
    def should_stop(self) -> bool:
        """Returns true if all schedules have reached the end."""
        return all(h.schedule.should_stop for h in self._homotopy_parameters)

    def restart(self) -> None:
        """Restart the homotopy to use the initial value in the schedule."""
        for hp in self._homotopy_parameters:
            hp.schedule.restart()
            hp.parameter.data.fill_(hp.schedule.value)
        self._execute_callbacks()

    def reset(self) -> None:
        """Reset the homotopy parameter to their original values."""
        for hp, val in zip(self._homotopy_parameters, self._original_values, strict=True):
            hp.parameter.data.fill_(val)
        self._execute_callbacks()

    def step(self) -> None:
        """Take a step according to the schedules."""
        for hp in self._homotopy_parameters:
            hp.schedule.step()
            if not hp.schedule.should_stop:
                hp.parameter.data.fill_(hp.schedule.value)
        self._execute_callbacks()


__all__ = [
    "FixedHomotopySchedule",
    "Homotopy",
    "HomotopyParameter",
    "HomotopyValue",
    "LinearHomotopySchedule",
    "LogLinearHomotopySchedule",
]
