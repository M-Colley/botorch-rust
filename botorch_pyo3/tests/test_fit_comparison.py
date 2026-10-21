"""The two-stage comparison of fits (``parity_helpers.compare_twin_fits``,
``docs/fitting.md``, "Comparing fits") on recorded and degraded outcomes:

* the outcomes of CI's Linux run that failed the one-sample bound of the
  former test (6b ``warp_priors_bounds``: botorch_rs's fit of the model
  itself landed in the tail of early stops that both libraries share) fail
  stage 1 and pass stage 2;
* fits of botorch_rs stopped after 5 L-BFGS-B iterations (a systematic
  regression) fail both stages.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from test_parity import CONFIGS, build, config_seed, make_data, twin_inputs  # noqa: E402

from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from parity_helpers import (  # noqa: E402
    FIT_TWINS,
    FIT_Z,
    FitOutcome,
    rank_bound_k,
    rel_err,
    two_stage_fit_comparisons,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.mark.parametrize("left,right", [(math.nan, math.nan), (math.nan, 1.0), (1.0, math.nan)])
def test_numeric_equivalence_rejects_nan_even_when_both_implementations_return_it(left, right):
    assert math.isinf(rel_err([left], [right]))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("port_invalid", [False, True])
def test_fit_equivalence_rejects_nonfinite_successful_first_attempts(value, port_invalid):
    def fit_twin(t):
        valid, invalid = ([0.1, 0.2], 1), ([0.1, value], 1)
        return (valid, invalid) if port_invalid else (invalid, valid)

    with pytest.raises(AssertionError, match="Nonfinite loss"):
        two_stage_fit_comparisons("invalid first-attempt success", fit_twin, 0)


#: First-attempt losses of BoTorch and botorch_rs on the model (twin 0) and
#: 64 twins of the 6b case ``warp_priors_bounds`` on CI's Linux runner (torch
#: 2.14.1, run 39038608252), printed with 10 decimals: twins 0..7 in order,
#: the others sorted (their order does not matter in stage 2).
LINUX_BT = """
-0.9761285871 -0.9761285871 -0.9761285848 -0.9761285868 -0.9761285867 -0.9761285536
-0.9761285593 -0.9761285776 -0.9761285872 -0.9761285871 -0.9761285871 -0.9761285871
-0.9761285871 -0.9761285871 -0.9761285871 -0.9761285871 -0.9761285871 -0.9761285871
-0.9761285870 -0.9761285870 -0.9761285870 -0.9761285870 -0.9761285869 -0.9761285868
-0.9761285868 -0.9761285868 -0.9761285868 -0.9761285866 -0.9761285862 -0.9761285859
-0.9761285859 -0.9761285849 -0.9761285848 -0.9761285846 -0.9761285846 -0.9761285843
-0.9761285842 -0.9761285830 -0.9761285820 -0.9761285816 -0.9761285816 -0.9761285792
-0.9761285759 -0.9761285757 -0.9761285739 -0.9761285639 -0.9761285598 -0.9761285598
-0.9761285596 -0.9761285537 -0.9761285503 -0.9761285493 -0.9761284267 -0.9761283722
-0.9761282954 -0.9761281886 -0.9761275755 -0.9761263592 -0.9761256537 -0.9759206499
-0.9759188063 -0.9759168316 -0.9759164862 -0.9759160175 -0.9759034842
"""
LINUX_RS = """
-0.9759152272 -0.9761285870 -0.9761285800 -0.9761285871 -0.9761285870 -0.9761285614
-0.9761285871 -0.9759339203 -0.9761285871 -0.9761285871 -0.9761285871 -0.9761285871
-0.9761285871 -0.9761285871 -0.9761285871 -0.9761285871 -0.9761285871 -0.9761285871
-0.9761285871 -0.9761285871 -0.9761285870 -0.9761285870 -0.9761285870 -0.9761285870
-0.9761285870 -0.9761285869 -0.9761285869 -0.9761285869 -0.9761285867 -0.9761285866
-0.9761285866 -0.9761285865 -0.9761285864 -0.9761285860 -0.9761285857 -0.9761285855
-0.9761285854 -0.9761285853 -0.9761285851 -0.9761285848 -0.9761285848 -0.9761285847
-0.9761285846 -0.9761285832 -0.9761285825 -0.9761285779 -0.9761285752 -0.9761285744
-0.9761285718 -0.9761285717 -0.9761285716 -0.9761285671 -0.9761285468 -0.9761285246
-0.9761284781 -0.9761284770 -0.9761284680 -0.9761284211 -0.9761283147 -0.9761279780
-0.9761276173 -0.9761269717 -0.9761263555 -0.9760761263 -0.9759204257
"""


def test_linux_warp_outcomes_fail_stage_one_and_pass_stage_two() -> None:
    bt = [float(v) for v in LINUX_BT.split()]
    rs = [float(v) for v in LINUX_RS.split()]
    assert len(bt) == len(rs) == FIT_TWINS + 1

    def fit_twin(t: int) -> tuple[FitOutcome, FitOutcome]:
        return (bt[t], 1), (rs[t], 1)

    # Stage 1 with 4 twins, as test_parity_6b_transforms.py.
    (c,) = two_stage_fit_comparisons("warp_priors_bounds (Linux)", fit_twin, 4)
    print(c.diagnostics())
    assert c.stage == 2 and c.passed
    # botorch_rs's fit of the model itself stopped in the tail 2.1e-4 above
    # BoTorch's worst of 5 (the former one-sample bound failed with 213 x tol).
    assert c.worst_excess > 200
    assert c.k == rank_bound_k(65, 65) == 10
    assert -0.46 < c.z < -0.44
    assert c.kth is not None and min(c.rs) <= c.kth


def test_degraded_fits_fail_both_stages() -> None:
    """botorch_rs's fits stopped after 5 iterations (a regression that leaves
    every fit far above BoTorch's optimum) fail stage 1 and stage 2."""
    config, d, n, xrange = CONFIGS[0]
    X, Y, Yvar = make_data(n, d, seed=7 + config_seed(config), low=xrange[0], high=xrange[1])

    def fit_twin(t: int) -> tuple[FitOutcome, FitOutcome]:
        import botorch.optim.fit as bt_optim_fit
        from botorch.fit import fit_gpytorch_mll as bt_fit
        from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

        bt, rs = build(config, twin_inputs(X, t), Y, Yvar)
        record: dict[str, Any] = {"attempts": 0}
        orig = bt_optim_fit.fit_gpytorch_mll_scipy

        def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
            record["attempts"] += 1
            record["r"] = orig(mll, *args, **kwargs)
            return record["r"]

        bt_fit(BtMLL(bt.likelihood, bt), optimizer=recorder)
        rs_mll = RsMLL(rs.likelihood, rs)
        rs_fit(rs_mll, optimizer_kwargs={"options": {"maxiter": 5}})
        rs_out = (float(rs_mll._fit_result["fval"]), int(rs_mll._fit_result["attempts"]))
        return (float(record["r"].fval), record["attempts"]), rs_out

    (c,) = two_stage_fit_comparisons(f"degraded {config}", fit_twin, 4)
    print(c.diagnostics())
    assert c.stage == 2 and not c.passed
    assert c.z is not None and c.z > FIT_Z
    assert c.kth is not None and min(c.rs) - c.kth > 1e-6
