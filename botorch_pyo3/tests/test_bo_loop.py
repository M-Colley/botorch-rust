"""A complete Bayesian optimization loop on Hartmann-6: the Python version of
``botorch_core/examples/bo_loop.rs`` (14 Sobol points, then 30 iterations of
fitting a default SingleTaskGP and maximizing LogEI with 10 restarts and 512
raw samples)."""

from __future__ import annotations

import statistics
import warnings
from typing import Any

import pytest

from botorch_rs.acquisition import LogExpectedImprovement
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim import optimize_acqf
from botorch_rs.test_functions import Hartmann
from botorch_rs.utils.sampling import draw_sobol_samples
from helpers import KINDS, as_kind, flatten, importorskip_botorch, to_list

D = 6
N_INIT = 2 * (D + 1)
ITERATIONS = 30
BOUNDS = [[0.0] * D, [1.0] * D]


def concat(a: Any, b: Any, kind: str) -> Any:
    if kind == "list":
        return a + b
    if kind == "numpy":
        import numpy as np

        return np.concatenate([a, b])
    import torch

    return torch.cat([a, b])


def bo_loop(
    seed: int, kind: str = "list", iterations: int = ITERATIONS, callback: Any = None
) -> tuple:
    problem = Hartmann(dim=D, negate=True)
    bounds = as_kind(BOUNDS, kind)
    X = as_kind([x[0] for x in to_list(draw_sobol_samples(bounds, n=N_INIT, q=1, seed=seed))], kind)
    Y = as_kind([[y] for y in flatten(problem(X, noise=False))], kind)
    for it in range(iterations):
        model = SingleTaskGP(X, Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acqf = LogExpectedImprovement(model, best_f=max(flatten(Y)))
        candidate, value = optimize_acqf(
            acqf, bounds, q=1, num_restarts=10, raw_samples=512, options={"seed": seed * 1000 + it}
        )
        if callback is not None:
            callback(it, X, Y, model, candidate, value)
        X = concat(X, candidate, kind)
        Y = concat(Y, as_kind([[y] for y in flatten(problem(candidate, noise=False))], kind), kind)
    return X, Y


def test_bo_loop_finds_the_optimum_region() -> None:
    """Over 30 initial designs, the median best value is below -3.0 (the
    optimum is -3.32237; BoTorch's loop: median -3.159 over 30 designs, see the
    README). About one run in six ends in a local optimum above -3.0, and the
    platforms' math libraries differ in the last bit, which changes individual
    runs, so a median over fewer designs would fail by chance."""
    finals = []
    for seed in range(30):
        _, Y = bo_loop(seed)
        finals.append(-max(flatten(Y)))
    assert statistics.median(finals) < -3.0, finals
    assert min(finals) > -3.32237 - 1e-6


def test_bo_loop_is_reproducible_and_independent_of_the_array_type() -> None:
    results = [bo_loop(0, kind, iterations=8) for kind in KINDS]
    reference = flatten(results[0][0])
    for X, _ in results[1:]:
        assert flatten(X) == reference
    assert flatten(bo_loop(0, "list", iterations=8)[0]) == reference


def test_bo_loop_steps_match_botorch() -> None:
    """Along the trajectory, every fifth step is repeated with BoTorch on the
    same data.

    * Fitting: the loss is no worse than BoTorch's (up to 1e-6, or else
      BoTorch's own optimizer started from our solution cannot improve it: on
      flat likelihood surfaces L-BFGS-B's relative-reduction stopping rule ends
      at round-off-dependent points).
    * Acquisition: with BoTorch's fitted hyperparameters, botorch_rs' LogEI
      equals BoTorch's, and the best of eight `optimize_acqf` runs is at least
      as good as BoTorch's median of eight runs. LogEI is multimodal late in
      the loop, so single runs of either library can end in different local
      maxima. If both libraries had the same distribution of outcomes, all
      eight runs would fall below BoTorch's median with probability 1/256 per
      check. With four runs it was 1/16, which made the trajectory's last bits
      (the platform) decide.
    """
    importorskip_botorch()
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from botorch.acquisition import LogExpectedImprovement as BtLogEI
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.models import SingleTaskGP as BtSingleTaskGP
    from botorch.optim import optimize_acqf as bt_optimize_acqf

    pytest.importorskip("gpytorch")
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    def bt_loss_of(model: Any) -> float:
        mll = BtMLL(model.likelihood, model)
        model.train()
        with torch.no_grad():
            loss = -mll(model(*model.train_inputs), model.train_targets).item()
        model.eval()
        return loss

    bounds = torch.tensor(BOUNDS, dtype=torch.float64)
    checks: list[tuple[float, float]] = []

    def compare(it: int, X: Any, Y: Any, model: Any, candidate: Any, value: Any) -> None:
        if it % 5 != 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt = BtSingleTaskGP(X, Y)
            bt_fit(BtMLL(bt.likelihood, bt))
            bt_loss = bt_loss_of(bt)
            rs_loss = -float(ExactMarginalLogLikelihood(model.likelihood, model)())
            if rs_loss > bt_loss + 1e-6:
                restarted = BtSingleTaskGP(X, Y)
                restarted.initialize(**dict(model.named_parameters()))
                bt_fit(BtMLL(restarted.likelihood, restarted))
                refit = bt_loss_of(restarted)
                assert refit >= rs_loss - 1e-6 * max(1.0, abs(rs_loss)), (rs_loss, bt_loss, refit)
            same = SingleTaskGP(X, Y)
            same.initialize(**{k: v.detach() for k, v in bt.named_parameters()})
            a_bt = BtLogEI(bt, best_f=Y.max())
            a_rs = LogExpectedImprovement(same, best_f=Y.max())
            Xt = torch.rand(
                16, 1, D, dtype=torch.float64, generator=torch.Generator().manual_seed(it)
            )
            with torch.no_grad():
                assert torch.allclose(a_rs(Xt), a_bt(Xt), rtol=1e-9, atol=1e-12)
            bt_values = [
                bt_optimize_acqf(
                    a_bt, bounds, q=1, num_restarts=10, raw_samples=512, options={"seed": s}
                )[1].item()
                for s in range(8)
            ]
            rs_values = []
            for s in range(8):
                x, _ = optimize_acqf(
                    a_rs, bounds, q=1, num_restarts=10, raw_samples=512, options={"seed": s}
                )
                with torch.no_grad():
                    rs_values.append(a_bt(x.unsqueeze(0)).item())
        checks.append((max(rs_values), statistics.median(bt_values)))

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bo_loop(1, "torch", callback=compare)
    finally:
        torch.set_num_threads(threads)
    assert len(checks) == ITERATIONS // 5
    for best_rs, median_bt in checks:
        assert best_rs >= median_bt - 1e-6 * max(1.0, abs(median_bt)), checks
