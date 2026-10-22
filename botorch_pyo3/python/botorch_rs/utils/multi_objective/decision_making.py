"""Multiobjective decision extensions; see :mod:`botorch_rs.decision_making`."""

from botorch_rs.decision_making import knee_region, select_preferred


def __getattr__(name):
    from botorch_rs import decision_making

    return getattr(decision_making, name)


__all__ = ["knee_region", "select_preferred"]
