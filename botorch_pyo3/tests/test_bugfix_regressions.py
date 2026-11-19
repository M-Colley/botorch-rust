"""Regression tests for bug fixes and performance invariants across the PyO3
bindings, array interoperability, and Python / Torch backends."""

from __future__ import annotations

from typing import Any

import pytest

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, bool_array, from_flat, kind_from_dtype, shape_of
from botorch_rs.acquisition import qExpectedImprovement, qNegIntegratedPosteriorVariance
from botorch_rs.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.acquisition.penalized import GroupLassoPenalty, L0PenaltyApprox, L2Penalty
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.models import PairwiseGP, SingleTaskGP
from botorch_rs.models.additive_gp import OrthogonalAdditiveGP
from botorch_rs.models.deterministic import AffineDeterministicModel
from botorch_rs.models.ensemble import EnsembleModel
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.optim.initializers import is_nonnegative
from botorch_rs.optim.stopping import ExpMAStoppingCriterion
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import HAS_NUMPY, HAS_TORCH

needs_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is not installed")
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")


@needs_torch
def test_array_kind_string_device_and_shape_of() -> None:
    import torch

    kind = kind_from_dtype("float64", "cpu")
    assert kind.lib == "torch"
    assert kind.dtype == torch.float64
    assert kind.device == torch.device("cpu")

    t = from_flat([1.0, 2.0, 3.0, 4.0], (2, 2), ArrayKind("torch", torch.float32, "cpu"))
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert shape_of(t) == (2, 2)

    b = bool_array([True, False], (2,), ArrayKind("torch", None, "cpu"))
    assert b.dtype == torch.bool
    assert b.tolist() == [True, False]


@needs_numpy
def test_strided_numpy_buffer_and_standardize_preserves_array_kind() -> None:
    import numpy as np

    # Strided 1D NumPy view (non-contiguous) passed directly to float_data.
    base = np.arange(10, dtype=np.float64)
    strided = base[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    vals, _ = _rs.elementwise("exp", [strided])
    assert len(vals[0]) == 5
    assert abs(vals[0][1] - float(np.exp(2.0))) < 1e-12

    # Standardize fitted on NumPy / nested lists preserves array kind on transform outputs.
    tf_np = Standardize(m=1)
    y_np = np.array([[1.0], [3.0], [5.0]], dtype=np.float64)
    out_np, _ = tf_np(y_np)
    assert isinstance(out_np, np.ndarray)

    tf_list = Standardize(m=1)
    out_list, _ = tf_list([[1.0], [3.0], [5.0]])
    assert isinstance(out_list, list)

    # Mismatched Yvar batch shape in native fit_transform raises cleanly instead of panicking.
    raw_st = _rs.Standardize(1e-8, [2])
    with pytest.raises(BotorchTensorDimensionError):
        raw_st.fit_transform(
            ([1.0, 2.0, 3.0, 4.0], [2, 2, 1]),
            ([0.1, 0.1], [1, 2, 1]),
        )


def test_orthogonal_additive_gp_accepts_python_lists() -> None:
    train_x = [[0.1, 0.2], [0.4, 0.5], [0.8, 0.9]]
    train_y = [[1.0], [2.0], [1.5]]
    model = OrthogonalAdditiveGP(train_x, train_y)
    post = model.posterior([[0.3, 0.6]])
    assert shape_of(post.mean) == (1, 1)


@needs_torch
def test_ensemble_model_with_none_outcome_transform() -> None:
    import torch

    class DummyEnsemble(EnsembleModel):
        def __init__(self) -> None:
            super().__init__()
            self._num_outputs = 1
            self.outcome_transform = None

        def forward(self, X: Any) -> Any:
            return torch.stack([X.sum(dim=-1, keepdim=True), 2.0 * X.sum(dim=-1, keepdim=True)], -3)

    model = DummyEnsemble()
    post = model.posterior(torch.tensor([[0.2, 0.5]], dtype=torch.float64))
    assert post.mean.shape == (1, 1)


@needs_torch
def test_expma_stopping_clones_inplace_mutated_tensor() -> None:
    import torch

    crit = ExpMAStoppingCriterion(maxiter=100, n_window=3, rel_tol=1e-5)
    fval = torch.tensor(100.0, dtype=torch.float64)
    stopped = False
    for step in range(5):
        fval.fill_(100.0 - 20.0 * step)
        if crit(fval):
            stopped = True
            break
    assert not stopped, "in-place updates to fval must not alias history and trigger early stop"


@needs_torch
def test_torch_reductions_penalties_and_active_learning_multi_fantasy() -> None:
    import torch

    from botorch_rs.acquisition._torch_reductions import compute_smoothed_feasibility_indicator

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        samples = torch.tensor([[[1.0, -1.0], [-0.5, 0.2]]], dtype=torch.float64, device=dev)
        ind = compute_smoothed_feasibility_indicator(
            constraints=[lambda s: s[..., 0], lambda s: s[..., 1]],
            samples=samples,
            eta=[0.1, 0.2],
        )
        assert ind.device.type == dev.type
        assert ind.shape == (1, 2)

        x = torch.tensor([[[0.2, 0.7]]], dtype=torch.float64, device=dev)
        assert L2Penalty([0.0, 0.0])(x).device.type == dev.type
        assert L0PenaltyApprox([0.0, 0.0], a=0.5)(x).device.type == dev.type
        assert GroupLassoPenalty([0.0, 0.0], [[0], [1]])(x).device.type == dev.type

        aff = AffineDeterministicModel(
            torch.tensor([[1.0], [2.0]], dtype=torch.float64, device=dev),
            b=0.5,
        )
        out = aff(torch.tensor([[0.3, 0.4]], dtype=torch.float64, device=dev))
        assert out.device.type == dev.type

    # Mixed scalar and 2D tensor fixed-feature values.
    train_x = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.8, 0.7, 0.9]], dtype=torch.float64)
    train_y = torch.tensor([[1.0], [1.5], [0.8]], dtype=torch.float64)
    gp_native = SingleTaskGP(train_x, train_y)
    qei_native = qExpectedImprovement(gp_native, best_f=1.0)
    ff = FixedFeatureAcquisitionFunction(
        qei_native,
        d=3,
        columns=[0, 2],
        values=[0.25, torch.tensor([[0.5], [0.7]], dtype=torch.float64, device=devices[-1])],
    )
    val = ff(torch.tensor([[[0.3], [0.6]]], dtype=torch.float64))
    assert val.shape == (1,)

    # Torch-backend qNIPV with >1 fantasy samples on single- and multi-output models,
    # plus is_nonnegative recognition on Torch-backend acquisitions.
    gp_torch = SingleTaskGP(train_x[:, :2], train_y, backend="torch")
    qei_torch = qExpectedImprovement(gp_torch, best_f=1.0)
    assert is_nonnegative(qei_torch)

    mc_pts = torch.tensor([[0.2, 0.3], [0.5, 0.5], [0.8, 0.1]], dtype=torch.float64)
    sampler4 = SobolQMCNormalSampler(sample_shape=torch.Size([4]))
    qnipv_1 = qNegIntegratedPosteriorVariance(gp_torch, mc_points=mc_pts, sampler=sampler4)
    res_1 = qnipv_1(torch.tensor([[[0.25, 0.75]], [[0.6, 0.4]]], dtype=torch.float64))
    assert res_1.shape == (2,)

    train_y2 = torch.cat([train_y, 2.0 * train_y], dim=-1)
    gp_torch_mo = SingleTaskGP(train_x[:, :2], train_y2, backend="torch")
    pt = ScalarizedPosteriorTransform(weights=torch.tensor([0.5, 0.5], dtype=torch.float64))
    qnipv_mo = qNegIntegratedPosteriorVariance(
        gp_torch_mo, mc_points=mc_pts, sampler=sampler4, posterior_transform=pt
    )
    res_mo = qnipv_mo(torch.tensor([[[0.25, 0.75]], [[0.6, 0.4]]], dtype=torch.float64))
    assert res_mo.shape == (2,)


@needs_torch
def test_polytope_and_pairwise_validation_guards() -> None:
    import torch

    with pytest.raises(ValueError, match="A_eq and b_eq"):
        _rs.find_interior_point(
            ([1.0, 0.0, -1.0, 0.0], [2, 2]),
            ([1.0, 0.0], [2]),
            a_eq=([1.0, 1.0], [1, 2]),
            b_eq=None,
        )

    dp = torch.tensor(
        [
            [[0.1, 0.2], [0.5, 0.6], [0.8, 0.9]],
            [[0.2, 0.3], [0.4, 0.7], [0.9, 0.8]],
        ],
        dtype=torch.float64,
    )
    comp = torch.tensor([[[0, 1]], [[1, 2]]], dtype=torch.long)
    pw = PairwiseGP(dp, comp)
    # Batch shape [3] of new X cannot broadcast with model batch shape [2].
    bad_x = torch.zeros((3, 2, 2), dtype=torch.float64)
    bad_y = torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long)
    with pytest.raises(BotorchTensorDimensionError):
        pw.condition_on_observations(bad_x, bad_y)
