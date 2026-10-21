"""Iteration reporting shared by native and Tensor NUTS without optional imports."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

Progress = Callable[[int, int, bool, float, int, bool], None]


class _NutsProgress:
    def __init__(self) -> None:
        self.stream = sys.stderr
        self.started = False
        self.divergences = 0

    def __call__(
        self, completed: int, total: int, warmup: bool, accept: float, steps: int, diverging: bool
    ) -> None:
        self.started = True
        self.divergences += int(diverging)
        stage = "Warmup" if warmup else "Sample"
        self.stream.write(
            f"\r{stage}: {completed}/{total} [accept={accept:.2f}, "
            f"steps={steps}, divergences={self.divergences}]"
        )
        self.stream.flush()

    def close(self) -> None:
        if self.started:
            self.stream.write("\n")
            self.stream.flush()


@contextmanager
def nuts_progress(disabled: bool) -> Iterator[Progress | None]:
    if disabled:
        yield None
        return
    progress = _NutsProgress()
    try:
        yield progress
    finally:
        progress.close()
