"""Multi-objective multi-fidelity test problems
(``botorch.test_functions.multi_objective_multi_fidelity``).

Both problems of Irshad et al. (2021) take the fidelity ``s`` as their last
input; see :mod:`botorch_rs.test_functions.multi_objective` for the shared
behaviour.
"""

from __future__ import annotations

from botorch_rs.test_functions.base import MultiObjectiveTestProblem


class MOMFBraninCurrin(MultiObjectiveTestProblem):
    """The Branin and Currin functions with fidelity-dependent coefficients
    on ``[0, 1]^3`` (the last input is the fidelity), negated and rescaled;
    both objectives are minimized."""

    _name = "MOMFBraninCurrin"


class MOMFPark(MultiObjectiveTestProblem):
    """Two modified Park functions on ``[0, 1]^5`` (the last input is the
    fidelity), negated; both objectives are minimized."""

    _name = "MOMFPark"


__all__ = ["MOMFBraninCurrin", "MOMFPark"]
