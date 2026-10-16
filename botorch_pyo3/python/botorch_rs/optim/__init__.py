"""Optimization (``botorch.optim``)."""

from botorch_rs.optim.closures import (
    ForwardBackwardClosure,
    get_loss_closure,
    get_loss_closure_with_grads,
)
from botorch_rs.optim.core import (
    OptimizationResult,
    OptimizationStatus,
    scipy_minimize,
    torch_minimize,
)
from botorch_rs.optim.homotopy import (
    FixedHomotopySchedule,
    Homotopy,
    HomotopyParameter,
    LinearHomotopySchedule,
    LogLinearHomotopySchedule,
)
from botorch_rs.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_hvkg_initial_conditions,
    initialize_q_batch,
    initialize_q_batch_nonneg,
    initialize_q_batch_topn,
    sample_q_batches_from_polytope,
    transform_constraints,
)
from botorch_rs.optim.optimize import (
    optimize_acqf,
    optimize_acqf_cyclic,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_list,
    optimize_acqf_mixed,
)
from botorch_rs.optim.optimize_homotopy import optimize_acqf_homotopy
from botorch_rs.optim.optimize_mixed import optimize_acqf_mixed_alternating
from botorch_rs.optim.parameter_constraints import evaluate_feasibility
from botorch_rs.optim.stopping import ExpMAStoppingCriterion

__all__ = [
    "ExpMAStoppingCriterion",
    "FixedHomotopySchedule",
    "ForwardBackwardClosure",
    "Homotopy",
    "HomotopyParameter",
    "LinearHomotopySchedule",
    "LogLinearHomotopySchedule",
    "OptimizationResult",
    "OptimizationStatus",
    "evaluate_feasibility",
    "gen_batch_initial_conditions",
    "gen_one_shot_hvkg_initial_conditions",
    "get_loss_closure",
    "get_loss_closure_with_grads",
    "initialize_q_batch",
    "initialize_q_batch_nonneg",
    "initialize_q_batch_topn",
    "optimize_acqf",
    "optimize_acqf_cyclic",
    "optimize_acqf_discrete",
    "optimize_acqf_discrete_local_search",
    "optimize_acqf_homotopy",
    "optimize_acqf_list",
    "optimize_acqf_mixed",
    "optimize_acqf_mixed_alternating",
    "sample_q_batches_from_polytope",
    "scipy_minimize",
    "torch_minimize",
    "transform_constraints",
]
