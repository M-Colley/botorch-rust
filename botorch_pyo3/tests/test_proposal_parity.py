"""Guided BO / MOBO proposal parity against BoTorch.

Verifies that ``botorch_rs`` reproduces BoTorch's quasi-random Sobol sequences,
Boltzmann initial-condition selection, baseline pruning, contextual LCEMGP
embedding initialization, single-step guided proposals and multi-step
closed-loop MOBO trajectories both under ``torch.manual_seed`` and with
explicit ``seed`` options.
"""

from __future__ import annotations

import statistics
import warnings
from typing import Any

import pytest

from botorch_rs.acquisition import (
    qExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch_rs.acquisition.multi_objective import (
    qLogNoisyExpectedHypervolumeImprovement,
    qLogNParEGO,
)
from botorch_rs.acquisition.multi_objective.utils import prune_inferior_points_multi_objective
from botorch_rs.acquisition.utils import prune_inferior_points
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.optim import gen_batch_initial_conditions, optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.sampling.qmc import NormalQMCEngine, _RustSobol
from botorch_rs.utils.sampling import (
    draw_sobol_normal_samples,
    draw_sobol_samples,
    sample_hypersphere,
    sample_simplex,
)
from helpers import importorskip_botorch


def _eval_mobo_objectives(X: Any) -> Any:
    import torch

    y1 = torch.sin(3.0 * X[..., 0]) + (X[..., 1:] - 0.3).pow(2).sum(-1)
    y2 = -((X[..., 0] - 0.6) ** 2) - 0.5 * (X[..., 1:] - 0.7).pow(2).sum(-1)
    return torch.stack([y1, y2], dim=-1)


def test_sobol_and_normal_qmc_bit_exact_parity() -> None:
    """``draw_sobol_samples``, ``draw_sobol_normal_samples``, ``_RustSobol`` and
    ``SobolQMCNormalSampler`` match BoTorch / PyTorch's ``SobolEngine``."""
    importorskip_botorch()
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
    from botorch.utils.sampling import (
        draw_sobol_normal_samples as bt_draw_sobol_normal,
    )
    from botorch.utils.sampling import (
        draw_sobol_samples as bt_draw_sobol,
    )
    from botorch.utils.sampling import (
        sample_hypersphere as bt_sample_hypersphere,
    )
    from botorch.utils.sampling import (
        sample_simplex as bt_sample_simplex,
    )

    dt = torch.float64
    bounds = torch.tensor([[-1.5, 0.0, 2.0, -0.5], [2.5, 1.0, 5.0, 3.5]], dtype=dt)

    # 1. Explicit seeds (unbatched and batched).
    for seed in (0, 7, 42, 999):
        bt_pts = bt_draw_sobol(bounds, n=32, q=2, seed=seed)
        rs_pts = draw_sobol_samples(bounds, n=32, q=2, seed=seed)
        assert torch.equal(bt_pts, rs_pts)

        bt_batched = bt_draw_sobol(bounds, n=8, q=2, batch_shape=torch.Size([3]), seed=seed)
        rs_batched = draw_sobol_samples(bounds, n=8, q=2, batch_shape=torch.Size([3]), seed=seed)
        assert torch.equal(bt_batched, rs_batched)

        bt_norm = bt_draw_sobol_normal(d=5, n=64, dtype=dt, seed=seed)
        rs_norm = draw_sobol_normal_samples(d=5, n=64, dtype=dt, seed=seed)
        assert float((bt_norm - rs_norm).abs().max()) <= 1e-14

        # Rust's built-in SobolEngine matches torch.quasirandom.SobolEngine bit-for-bit.
        rust_raw = torch.tensor(_RustSobol(dimension=8, seed=seed).draw(32), dtype=dt).view(32, 8)
        torch_raw = torch.quasirandom.SobolEngine(8, scramble=True, seed=seed).draw(32, dtype=dt)
        assert torch.equal(rust_raw, torch_raw)

    # 2. Unseeded calls under torch.manual_seed.
    torch.manual_seed(2026)
    bt_u1 = bt_draw_sobol(bounds, n=16, q=2)
    bt_u2 = bt_draw_sobol_normal(d=4, n=32, dtype=dt)
    bt_s = bt_sample_simplex(d=4, n=5, dtype=dt)
    bt_h = bt_sample_hypersphere(d=4, n=5, dtype=dt)

    torch.manual_seed(2026)
    rs_u1 = draw_sobol_samples(bounds, n=16, q=2)
    rs_u2 = draw_sobol_normal_samples(d=4, n=32, dtype=dt)
    rs_s = sample_simplex(d=4, n=5, dtype=dt)
    rs_h = sample_hypersphere(d=4, n=5, dtype=dt)

    assert torch.equal(bt_u1, rs_u1)
    assert float((bt_u2 - rs_u2).abs().max()) <= 1e-14
    assert torch.equal(bt_s, rs_s)
    assert torch.equal(bt_h, rs_h)

    # 3. NormalQMCEngine without torch engine fallback matches torch engine.
    eng_rs = NormalQMCEngine(d=4, seed=123, inv_transform=True)
    eng_rs._sobol_engine = _RustSobol(4, 123)
    eng_bt = NormalQMCEngine(d=4, seed=123, inv_transform=True)
    assert torch.equal(eng_rs.draw(16, dtype=dt), eng_bt.draw(16, dtype=dt))

    # 4. SobolQMCNormalSampler posterior samples match BoTorch closely.
    X = bt_draw_sobol(bounds[:, :2], n=8, q=1, seed=3).squeeze(1)
    Y = torch.sin(X[:, :1]) + torch.cos(X[:, 1:])
    from botorch.models import SingleTaskGP as BtSingleTaskGP

    m_bt = BtSingleTaskGP(X, Y)
    m_rs = SingleTaskGP(X, Y)
    X_test = bt_draw_sobol(bounds[:, :2], n=5, q=1, seed=4).squeeze(1)

    torch.manual_seed(88)
    s_bt = BtSampler(sample_shape=torch.Size([64]))(m_bt.posterior(X_test))
    torch.manual_seed(88)
    s_rs = SobolQMCNormalSampler(sample_shape=torch.Size([64]))(m_rs.posterior(X_test))
    assert float((s_bt.detach() - s_rs.detach()).abs().max()) <= 1e-12


def test_prune_baseline_and_batch_initial_conditions_exact_parity() -> None:
    """Baseline pruning and Boltzmann batch initial conditions select the exact
    same points as BoTorch under both ``torch.manual_seed`` and explicit ``seed``."""
    importorskip_botorch()
    import torch
    from botorch.acquisition import (
        qExpectedImprovement as BtQEI,
    )
    from botorch.acquisition import (
        qUpperConfidenceBound as BtQUCB,
    )
    from botorch.acquisition.multi_objective.utils import (
        prune_inferior_points_multi_objective as bt_prune_mo,
    )
    from botorch.acquisition.utils import prune_inferior_points as bt_prune_so
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.models import SingleTaskGP as BtSingleTaskGP
    from botorch.optim.initializers import gen_batch_initial_conditions as bt_gen_ics
    from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    dt = torch.float64
    d = 3
    bounds = torch.stack([torch.zeros(d, dtype=dt), torch.ones(d, dtype=dt)])
    X = draw_sobol_samples(bounds, n=14, q=1, seed=11).squeeze(1)
    Y_mo = _eval_mobo_objectives(X)
    ref_point = Y_mo.min(0).values - 0.1

    m_bt = BtSingleTaskGP(X, Y_mo)
    bt_fit(BtMLL(m_bt.likelihood, m_bt))
    m_rs = SingleTaskGP(X, Y_mo)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m_rs.likelihood, m_rs))

    # Multi-objective baseline pruning under torch.manual_seed.
    for seed in (0, 1, 2, 3):
        torch.manual_seed(400 + seed)
        p_bt = bt_prune_mo(m_bt, X=X, ref_point=ref_point)
        torch.manual_seed(400 + seed)
        p_rs = prune_inferior_points_multi_objective(m_rs, X=X, ref_point=ref_point)
        assert torch.equal(p_bt, p_rs)

    # Single-objective baseline pruning + gen_batch_initial_conditions (general & nonnegative).
    m1_bt = BtSingleTaskGP(X, Y_mo[:, :1])
    bt_fit(BtMLL(m1_bt.likelihood, m1_bt))
    m1_rs = SingleTaskGP(X, Y_mo[:, :1])
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m1_rs.likelihood, m1_rs))

    torch.manual_seed(555)
    ps_bt = bt_prune_so(m1_bt, X=X)
    torch.manual_seed(555)
    ps_rs = prune_inferior_points(m1_rs, X=X)
    assert torch.equal(ps_bt, ps_rs)

    # General acquisition (Boltzmann sampling) under both manual_seed and explicit seed.
    ucb_bt = BtQUCB(m1_bt, beta=0.5, sampler=BtSampler(sample_shape=torch.Size([64]), seed=9))
    ucb_rs = qUpperConfidenceBound(
        m1_rs, beta=0.5, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=9)
    )
    torch.manual_seed(777)
    ic_bt = bt_gen_ics(ucb_bt, bounds, q=2, num_restarts=8, raw_samples=128)
    torch.manual_seed(777)
    ic_rs = gen_batch_initial_conditions(ucb_rs, bounds, q=2, num_restarts=8, raw_samples=128)
    assert float((ic_bt - ic_rs).abs().max()) <= 1e-15

    torch.manual_seed(888)
    ic_bt_seeded = bt_gen_ics(
        ucb_bt, bounds, q=1, num_restarts=8, raw_samples=128, options={"seed": 888}
    )
    ic_rs_seeded = gen_batch_initial_conditions(
        ucb_rs, bounds, q=1, num_restarts=8, raw_samples=128, options={"seed": 888}
    )
    assert float((ic_bt_seeded - ic_rs_seeded).abs().max()) <= 1e-15

    # Nonnegative acquisition (initialize_q_batch_nonneg) under manual_seed.
    best_f = float(Y_mo[:, 0].max().item())
    ei_bt = BtQEI(m1_bt, best_f=best_f, sampler=BtSampler(sample_shape=torch.Size([64]), seed=5))
    ei_rs = qExpectedImprovement(
        m1_rs,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=5),
    )
    torch.manual_seed(999)
    ic_ei_bt = bt_gen_ics(ei_bt, bounds, q=1, num_restarts=6, raw_samples=128)
    torch.manual_seed(999)
    ic_ei_rs = gen_batch_initial_conditions(ei_rs, bounds, q=1, num_restarts=6, raw_samples=128)
    assert float((ic_ei_bt - ic_ei_rs).abs().max()) <= 1e-15


def test_lcemgp_torch_rng_embedding_parity() -> None:
    """Unseeded ``LCEMGP`` and ``ProvidedEmbeddingLCEMGP`` on torch tensors
    match BoTorch's initial parameters and global RNG consumption."""
    importorskip_botorch()
    import torch
    from lcemgp_reference import build_contextual_model

    dt = torch.float64
    g = torch.Generator().manual_seed(3100)
    rows = [0] * 5 + [1] * 5 + [2] * 4
    X_raw = torch.rand(len(rows), 3, generator=g, dtype=dt)
    t = torch.tensor(rows, dtype=dt).unsqueeze(-1)
    X = torch.cat([X_raw, t], dim=-1)
    Y = torch.randn(len(rows), 2, generator=g, dtype=dt)
    emb = torch.rand(3, 2, generator=g, dtype=dt)

    for variant_emb in (None, emb):
        torch.manual_seed(42)
        _, mod_bt = build_contextual_model("bt", X, Y, 3, variant_emb, 2)
        draw_after_bt = torch.rand(4)

        torch.manual_seed(42)
        _, mod_rs = build_contextual_model("rs", X, Y, 3, variant_emb, 2)
        draw_after_rs = torch.rand(4)

        assert torch.equal(draw_after_bt, draw_after_rs)
        for mb, mr in zip(mod_bt.models, mod_rs.models, strict=True):
            pb = dict(mb.named_parameters())
            pr = dict(mr.named_parameters())
            for name, val_r in pr.items():
                val_b = pb[name].detach()
                assert torch.equal(val_b, torch.as_tensor(val_r, dtype=val_b.dtype))


def test_first_guided_mobo_proposal_parity_across_repeats() -> None:
    """First guided MOBO proposal on identical data agrees to ~1e-9 median
    (was ~4e-3 in v0.4.0) across 20 repeats."""
    importorskip_botorch()
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement as BtQLogNEHVI,
    )
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.models import ModelListGP as BtModelListGP
    from botorch.models import SingleTaskGP as BtSingleTaskGP
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.optim import optimize_acqf as bt_optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
    from botorch.utils.sampling import draw_sobol_samples as bt_draw_sobol
    from gpytorch.mlls import SumMarginalLogLikelihood as BtSumMLL

    warnings.filterwarnings("ignore")
    dt = torch.float64
    d = 4
    bounds = torch.stack([torch.zeros(d, dtype=dt), torch.ones(d, dtype=dt)])
    diffs: list[float] = []

    for rep in range(20):
        torch.manual_seed(6000 + rep)
        X_bt = bt_draw_sobol(bounds, n=8, q=1).squeeze(1)
        torch.manual_seed(6000 + rep)
        X_rs = draw_sobol_samples(bounds, n=8, q=1).squeeze(1)
        assert torch.equal(X_bt, X_rs)

        Y = _eval_mobo_objectives(X_bt)
        ref_point = (Y.min(0).values - 0.1).tolist()

        torch.manual_seed(rep)
        m_bt = BtModelListGP(
            BtSingleTaskGP(X_bt, Y[:, :1], input_transform=BtNormalize(d=d, bounds=bounds)),
            BtSingleTaskGP(X_bt, Y[:, 1:], input_transform=BtNormalize(d=d, bounds=bounds)),
        )
        bt_fit(BtSumMLL(m_bt.likelihood, m_bt))
        acqf_bt = BtQLogNEHVI(
            m_bt,
            ref_point=ref_point,
            X_baseline=X_bt,
            prune_baseline=True,
            sampler=BtSampler(sample_shape=torch.Size([128])),
        )
        cand_bt, _ = bt_optimize_acqf(acqf_bt, bounds, q=1, num_restarts=8, raw_samples=256)

        torch.manual_seed(rep)
        m_rs = ModelListGP(
            SingleTaskGP(X_rs, Y[:, :1], input_transform=Normalize(d=d, bounds=bounds)),
            SingleTaskGP(X_rs, Y[:, 1:], input_transform=Normalize(d=d, bounds=bounds)),
        )
        fit_gpytorch_mll(SumMarginalLogLikelihood(m_rs.likelihood, m_rs))
        acqf_rs = qLogNoisyExpectedHypervolumeImprovement(
            m_rs,
            ref_point=ref_point,
            X_baseline=X_rs,
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )
        cand_rs, _ = optimize_acqf(acqf_rs, bounds, q=1, num_restarts=8, raw_samples=256)

        diffs.append(float((cand_bt - cand_rs).abs().max()))

    s = sorted(diffs)
    median_diff = statistics.median(s)
    p95_diff = s[int(0.95 * (len(s) - 1))]
    assert median_diff < 1e-6, f"median Proposal 1 diff {median_diff:.3e} >= 1e-6"
    assert p95_diff < 1e-4, f"p95 Proposal 1 diff {p95_diff:.3e} >= 1e-4"
    assert max(s) < 1e-2, f"max Proposal 1 diff {max(s):.3e} >= 1e-2"


def test_explicit_seed_guided_mobo_proposal_parity() -> None:
    """Explicitly seeded ``SobolQMCNormalSampler(seed=...)`` and
    ``optimize_acqf(options={'seed': ...})`` match BoTorch on Proposal 1."""
    importorskip_botorch()
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement as BtQLogNEHVI,
    )
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.models import SingleTaskGP as BtSingleTaskGP
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.optim import optimize_acqf as bt_optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    warnings.filterwarnings("ignore")
    dt = torch.float64
    d = 4
    bounds = torch.stack([torch.zeros(d, dtype=dt), torch.ones(d, dtype=dt)])
    diffs: list[float] = []

    for rep in range(15):
        X = draw_sobol_samples(bounds, n=8, q=1, seed=200 + rep).squeeze(1)
        Y = _eval_mobo_objectives(X)
        ref_point = (Y.min(0).values - 0.1).tolist()

        m_bt = BtSingleTaskGP(X, Y, input_transform=BtNormalize(d=d, bounds=bounds))
        bt_fit(BtMLL(m_bt.likelihood, m_bt))
        torch.manual_seed(rep)
        acqf_bt = BtQLogNEHVI(
            m_bt,
            ref_point=ref_point,
            X_baseline=X,
            prune_baseline=True,
            sampler=BtSampler(sample_shape=torch.Size([128]), seed=rep),
        )
        torch.manual_seed(rep)
        cand_bt, _ = bt_optimize_acqf(
            acqf_bt, bounds, q=1, num_restarts=8, raw_samples=256, options={"seed": rep}
        )

        m_rs = SingleTaskGP(X, Y, input_transform=Normalize(d=d, bounds=bounds))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m_rs.likelihood, m_rs))
        acqf_rs = qLogNoisyExpectedHypervolumeImprovement(
            m_rs,
            ref_point=ref_point,
            X_baseline=X,
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=rep),
        )
        cand_rs, _ = optimize_acqf(
            acqf_rs, bounds, q=1, num_restarts=8, raw_samples=256, options={"seed": rep}
        )
        diffs.append(float((cand_bt - cand_rs).abs().max()))

    s = sorted(diffs)
    assert statistics.median(s) < 1e-6
    assert s[int(0.95 * (len(s) - 1))] < 5e-3


def test_closed_loop_multi_step_mobo_proposal_parity() -> None:
    """Closed-loop 5-proposal MOBO trajectories (each library continuing from
    its own proposals) stay tightly aligned across steps 1..5."""
    importorskip_botorch()
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement as BtQLogNEHVI,
    )
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.models import ModelListGP as BtModelListGP
    from botorch.models import SingleTaskGP as BtSingleTaskGP
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.optim import optimize_acqf as bt_optimize_acqf
    from gpytorch.mlls import SumMarginalLogLikelihood as BtSumMLL

    warnings.filterwarnings("ignore")
    dt = torch.float64
    d = 4
    bounds = torch.stack([torch.zeros(d, dtype=dt), torch.ones(d, dtype=dt)])
    n_steps = 5
    per_step: list[list[float]] = [[] for _ in range(n_steps)]

    for rep in range(10):
        torch.manual_seed(5000 + rep)
        X_bt = draw_sobol_samples(bounds, n=8, q=1).squeeze(1)
        Y_bt = _eval_mobo_objectives(X_bt)

        torch.manual_seed(5000 + rep)
        X_rs = draw_sobol_samples(bounds, n=8, q=1).squeeze(1)
        Y_rs = _eval_mobo_objectives(X_rs)

        for step in range(n_steps):
            ref_bt = (Y_bt.min(0).values - 0.1).tolist()
            ref_rs = (Y_rs.min(0).values - 0.1).tolist()

            # Also exercises `sampler=None` default sampler + prune_baseline RNG order.
            torch.manual_seed(rep * 100 + step)
            m_bt = BtModelListGP(
                BtSingleTaskGP(X_bt, Y_bt[:, :1], input_transform=BtNormalize(d=d, bounds=bounds)),
                BtSingleTaskGP(X_bt, Y_bt[:, 1:], input_transform=BtNormalize(d=d, bounds=bounds)),
            )
            bt_fit(BtSumMLL(m_bt.likelihood, m_bt))
            acqf_bt = BtQLogNEHVI(m_bt, ref_point=ref_bt, X_baseline=X_bt, prune_baseline=True)
            c_bt, _ = bt_optimize_acqf(acqf_bt, bounds, q=1, num_restarts=6, raw_samples=256)

            torch.manual_seed(rep * 100 + step)
            m_rs = ModelListGP(
                SingleTaskGP(X_rs, Y_rs[:, :1], input_transform=Normalize(d=d, bounds=bounds)),
                SingleTaskGP(X_rs, Y_rs[:, 1:], input_transform=Normalize(d=d, bounds=bounds)),
            )
            fit_gpytorch_mll(SumMarginalLogLikelihood(m_rs.likelihood, m_rs))
            acqf_rs = qLogNoisyExpectedHypervolumeImprovement(
                m_rs, ref_point=ref_rs, X_baseline=X_rs, prune_baseline=True
            )
            c_rs, _ = optimize_acqf(acqf_rs, bounds, q=1, num_restarts=6, raw_samples=256)

            per_step[step].append(float((c_bt - c_rs).abs().max()))
            X_bt = torch.cat([X_bt, c_bt], dim=0)
            Y_bt = torch.cat([Y_bt, _eval_mobo_objectives(c_bt)], dim=0)
            X_rs = torch.cat([X_rs, c_rs], dim=0)
            Y_rs = torch.cat([Y_rs, _eval_mobo_objectives(c_rs)], dim=0)

    assert statistics.median(per_step[0]) < 1e-6
    for step in range(n_steps):
        med = statistics.median(per_step[step])
        assert med < 1e-3, f"Proposal {step + 1} median diff {med:.3e} >= 1e-3"


@pytest.mark.parametrize("variant", ["provided", "learned"])
def test_unity_contextual_lcemgp_guided_proposal_parity(variant: str) -> None:
    """Unity contextual MOBO (``ProvidedEmbeddingLCEMGP`` and learned ``LCEMGP``)
    guided proposals agree closely with BoTorch across repeats."""
    importorskip_botorch()
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement as BtQLogNEHVI,
    )
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.optim import optimize_acqf as bt_optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
    from lcemgp_reference import build_contextual_model

    warnings.filterwarnings("ignore")
    dt = torch.float64
    d, num_contexts = 3, 3
    bounds = torch.stack([torch.zeros(d, dtype=dt), torch.ones(d, dtype=dt)])
    cand_diffs: list[float] = []

    for rep in range(10):
        g = torch.Generator().manual_seed(3000 + rep)
        rows = [0] * 5 + [1] * 5 + [2] * 4
        X_raw = torch.rand(len(rows), d, generator=g, dtype=dt)
        t = torch.tensor(rows, dtype=dt).unsqueeze(-1)
        X = torch.cat([X_raw, t], dim=-1)
        emb = torch.rand(num_contexts, 2, generator=g, dtype=dt) if variant == "provided" else None
        y1 = (
            torch.sin(3.0 * X_raw[:, 0]) + (X_raw[:, 1:] - 0.3).pow(2).sum(-1) + 0.2 * t[:, 0]
        ).unsqueeze(-1)
        y2 = (
            -((X_raw[:, 0] - 0.6) ** 2) - 0.5 * (X_raw[:, 1:] - 0.7).pow(2).sum(-1) - 0.1 * t[:, 0]
        ).unsqueeze(-1)
        Y = torch.cat([y1, y2], dim=-1)
        X_base = X_raw[t[:, 0] == 2]
        ref_point = (Y.min(0).values - 0.1).tolist()

        torch.manual_seed(rep)
        mll_b, model_b = build_contextual_model("bt", X, Y, num_contexts, emb, 2)
        bt_fit(mll_b)
        model_b.eval()
        acqf_b = BtQLogNEHVI(
            model_b,
            ref_point=ref_point,
            X_baseline=X_base,
            prune_baseline=True,
            sampler=BtSampler(sample_shape=torch.Size([128])),
        )
        c_b, _ = bt_optimize_acqf(acqf_b, bounds, q=1, num_restarts=6, raw_samples=256)

        torch.manual_seed(rep)
        mll_r, model_r = build_contextual_model("rs", X, Y, num_contexts, emb, 2)
        fit_gpytorch_mll(mll_r)
        model_r.eval()
        acqf_r = qLogNoisyExpectedHypervolumeImprovement(
            model_r,
            ref_point=ref_point,
            X_baseline=X_base,
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )
        c_r, _ = optimize_acqf(acqf_r, bounds, q=1, num_restarts=6, raw_samples=256)
        cand_diffs.append(float((c_b - c_r).abs().max()))

    assert statistics.median(cand_diffs) < 1e-3
    if variant == "provided":
        assert max(cand_diffs) < 1e-2
    else:
        assert sum(v <= 0.05 for v in cand_diffs) >= 9


def test_single_objective_qlognei_and_qlognparego_proposal_parity() -> None:
    """``qLogNoisyExpectedImprovement`` and ``qLogNParEGO`` with
    ``prune_baseline=True`` match BoTorch under ``torch.manual_seed``."""
    importorskip_botorch()
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement as BtQLogNEI
    from botorch.acquisition.multi_objective.parego import qLogNParEGO as BtQLogNParEGO
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.models import SingleTaskGP as BtSingleTaskGP
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.optim import optimize_acqf as bt_optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    warnings.filterwarnings("ignore")
    dt = torch.float64
    d = 3
    bounds = torch.stack([torch.zeros(d, dtype=dt), torch.ones(d, dtype=dt)])
    X = draw_sobol_samples(bounds, n=10, q=1, seed=33).squeeze(1)
    Y = _eval_mobo_objectives(X)

    # 1. qLogNoisyExpectedImprovement (prune_baseline=True, default sampler).
    torch.manual_seed(101)
    m1_bt = BtSingleTaskGP(X, Y[:, :1], input_transform=BtNormalize(d=d, bounds=bounds))
    bt_fit(BtMLL(m1_bt.likelihood, m1_bt))
    nei_bt = BtQLogNEI(m1_bt, X_baseline=X, prune_baseline=True)
    c_nei_bt, _ = bt_optimize_acqf(nei_bt, bounds, q=1, num_restarts=6, raw_samples=128)

    torch.manual_seed(101)
    m1_rs = SingleTaskGP(X, Y[:, :1], input_transform=Normalize(d=d, bounds=bounds))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m1_rs.likelihood, m1_rs))
    nei_rs = qLogNoisyExpectedImprovement(m1_rs, X_baseline=X, prune_baseline=True)
    c_nei_rs, _ = optimize_acqf(nei_rs, bounds, q=1, num_restarts=6, raw_samples=128)

    assert float((c_nei_bt - c_nei_rs).abs().max()) < 1e-6

    # 2. qLogNParEGO (default random simplex weights + prune_baseline=True + default sampler).
    torch.manual_seed(202)
    m2_bt = BtSingleTaskGP(X, Y, input_transform=BtNormalize(d=d, bounds=bounds))
    bt_fit(BtMLL(m2_bt.likelihood, m2_bt))
    parego_bt = BtQLogNParEGO(m2_bt, X_baseline=X, prune_baseline=True)
    c_par_bt, _ = bt_optimize_acqf(parego_bt, bounds, q=1, num_restarts=6, raw_samples=128)

    torch.manual_seed(202)
    m2_rs = SingleTaskGP(X, Y, input_transform=Normalize(d=d, bounds=bounds))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m2_rs.likelihood, m2_rs))
    parego_rs = qLogNParEGO(m2_rs, X_baseline=X, prune_baseline=True)
    c_par_rs, _ = optimize_acqf(parego_rs, bounds, q=1, num_restarts=6, raw_samples=128)

    assert torch.allclose(parego_bt.scalarization_weights, parego_rs.scalarization_weights)
    assert float((c_par_bt - c_par_rs).abs().max()) < 1e-6
