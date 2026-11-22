"""Tests for ``get_covar_module_with_dim_scaled_prior`` (``nu`` and ``ard``),
``HierarchicalGateTransform`` across array backends and native GP models, and
non-finite line-search recovery in ``_torch_acq_parallel``."""

from __future__ import annotations

import copy
import math
import pickle
import warnings

import pytest

from botorch_rs.acquisition import LogExpectedImprovement
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    OptimizationGradientError,
)
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms import (
    ChainedInputTransform,
    HierarchicalGateTransform,
    Normalize,
    Standardize,
)
from botorch_rs.models.transforms.input import _gates_vjp_flat
from botorch_rs.models.utils import fantasize
from botorch_rs.models.utils.gpytorch_modules import (
    SQRT2,
    SQRT3,
    get_covar_module_with_dim_scaled_prior,
)
from botorch_rs.optim import optimize_acqf
from helpers import as_kind, assert_close, check_kind, shape, to_list


def test_dim_scaled_prior_nu_and_ard(kind: str) -> None:
    for batch_shape in (None, (2,)):
        expected_batch = () if batch_shape is None else batch_shape
        for nu in (0.5, 1.5, 2.5):
            k = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=3,
                batch_shape=batch_shape,
                use_rbf_kernel=False,
                nu=nu,
            )
            assert isinstance(k, MaternKernel)
            assert k.nu == nu
            assert k.ard_num_dims == 3
            assert k.batch_shape == expected_batch
            assert k.lengthscale_prior is not None
            assert math.isclose(k.lengthscale_prior.loc, SQRT2 + 0.5 * math.log(3.0), rel_tol=1e-6)
            assert math.isclose(k.lengthscale_prior.scale, SQRT3, rel_tol=1e-6)

            X = as_kind(
                [[0.1, 0.2, 0.3], [0.5, 0.4, 0.8], [0.9, 0.1, 0.6]],
                kind,
            )
            K_mat = k(X)
            check_kind(K_mat, kind)
            assert shape(K_mat) == (*expected_batch, 3, 3)

        for use_rbf in (True, False):
            iso = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=4,
                batch_shape=batch_shape,
                use_rbf_kernel=use_rbf,
                nu=1.5,
                ard=False,
            )
            assert isinstance(iso, RBFKernel if use_rbf else MaternKernel)
            assert iso.ard_num_dims is None
            assert iso.batch_shape == expected_batch
            assert shape(iso.raw_lengthscale) == (*expected_batch, 1, 1)
            assert iso.lengthscale_prior is not None
            assert math.isclose(
                iso.lengthscale_prior.loc, SQRT2 + 0.5 * math.log(4.0), rel_tol=1e-6
            )

    with pytest.raises(ValueError, match="Expected `nu` to be 0.5, 1.5, or 2.5"):
        get_covar_module_with_dim_scaled_prior(ard_num_dims=2, use_rbf_kernel=False, nu=3.5)
    # Ignored when use_rbf_kernel=True
    rbf = get_covar_module_with_dim_scaled_prior(ard_num_dims=2, use_rbf_kernel=True, nu=3.5)
    assert isinstance(rbf, RBFKernel)


def test_dim_scaled_matern_single_task_gp_subset_output(kind: str) -> None:
    X = as_kind(
        [
            [0.1, 0.2],
            [0.4, 0.8],
            [0.7, 0.3],
            [0.9, 0.6],
        ],
        kind,
    )
    Y = as_kind(
        [
            [1.0, 2.0, -0.5],
            [1.5, 1.0, 0.2],
            [0.8, 2.5, 1.1],
            [2.0, 0.5, 0.7],
        ],
        kind,
    )
    covar = ScaleKernel(
        get_covar_module_with_dim_scaled_prior(
            ard_num_dims=2,
            batch_shape=(3,),
            use_rbf_kernel=False,
            nu=1.5,
        ),
        batch_shape=(3,),
    )
    model = SingleTaskGP(X, Y, covar_module=covar, outcome_transform=Standardize(m=3))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    X_test = as_kind([[0.25, 0.5], [0.75, 0.4]], kind)
    full_post = model.posterior(X_test)
    sub_model = model.subset_output([1])
    sub_post = sub_model.posterior(X_test)
    check_kind(sub_post.mean, kind)
    full_mean = [[row[1]] for row in to_list(full_post.mean)]
    assert_close(sub_post.mean, full_mean, rtol=1e-5, atol=1e-5)


def test_hierarchical_gate_transform_arrays_and_dag(kind: str) -> None:
    tf = HierarchicalGateTransform(
        d=2,
        gates={1: 0},
        thresholds=0.2,
        neutral_values=0.5,
    )
    X = as_kind(
        [[0.0, 0.9], [0.2, 0.1], [0.6, 0.9], [1.0, 0.9]],
        kind,
    )
    out = tf(X)
    check_kind(out, kind)
    assert_close(
        out,
        [[0.0, 0.5], [0.2, 0.5], [0.6, 0.7], [1.0, 0.9]],
        rtol=1e-6,
        atol=1e-6,
    )

    # Multi-level DAG declared out of order: 0 -> 1 -> 2
    tf_dag = HierarchicalGateTransform(
        d=3,
        gates={2: 1, 1: 0},
        thresholds={1: 0.0, 2: 0.5},
        neutral_values={1: 0.0, 2: 1.0},
    )
    X_dag = as_kind(
        [
            [0.0, 0.9, 0.2],
            [0.5, 1.0, 0.2],
            [1.0, 1.0, 0.2],
        ],
        kind,
    )
    out_dag = tf_dag(X_dag)
    check_kind(out_dag, kind)
    assert_close(
        out_dag,
        [
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 1.0],
            [1.0, 1.0, 0.2],
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    # Batched 3D input + pickle/deepcopy/equals
    X_batch = as_kind(
        [
            [[0.0, 0.9, 0.2], [1.0, 1.0, 0.2]],
            [[0.5, 1.0, 0.2], [1.0, 1.0, 0.2]],
        ],
        kind,
    )
    out_batch = tf_dag(X_batch)
    check_kind(out_batch, kind)
    assert shape(out_batch) == (2, 2, 3)

    tf_same = HierarchicalGateTransform(
        d=3,
        gates={1: 0, 2: 1},
        thresholds={1: 0.0, 2: 0.5},
        neutral_values={1: 0.0, 2: 1.0},
    )
    assert tf_dag.equals(tf_same)
    assert tf_dag.equals(copy.deepcopy(tf_dag))
    assert tf_dag.equals(pickle.loads(pickle.dumps(tf_dag)))
    assert not tf_dag.equals(tf)


def test_hierarchical_gate_modes_and_native_gp_pipeline(kind: str) -> None:
    tf_eval_only = HierarchicalGateTransform(
        d=2,
        gates={1: 0},
        transform_on_train=False,
        transform_on_eval=True,
        transform_on_fantasize=False,
    )
    X_pt = as_kind([[0.0, 0.8]], kind)
    tf_eval_only.train()
    assert_close(tf_eval_only(X_pt), [[0.0, 0.8]])
    tf_eval_only.eval()
    assert_close(tf_eval_only(X_pt), [[0.0, 0.0]])
    with fantasize():
        assert_close(tf_eval_only(X_pt), [[0.0, 0.8]])

    bounds = as_kind([[0.0, -5.0, 10.0], [1.0, 5.0, 20.0]], kind)
    gate = HierarchicalGateTransform(
        d=3,
        gates={1: 0, 2: 1},
        thresholds=0.1,
        neutral_values={1: 0.0, 2: 0.5},
        sharpness=12.0,
    )
    chained = ChainedInputTransform(
        normalize=Normalize(d=3, bounds=bounds),
        gate=gate,
    )
    train_X = as_kind(
        [
            [0.0, -4.0, 12.0],
            [0.0, 4.0, 18.0],
            [0.6, 1.0, 15.0],
            [1.0, 3.0, 19.0],
        ],
        kind,
    )
    train_Y = as_kind([[1.0], [1.0], [2.2], [3.5]], kind)
    model = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=chained,
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    post = model.posterior(as_kind([[0.0, -4.0, 12.0], [0.0, 4.0, 18.0]], kind))
    post_means = to_list(post.mean)
    assert math.isclose(post_means[0][0], post_means[1][0], rel_tol=1e-6, abs_tol=1e-6)

    cand, val = optimize_acqf(
        LogExpectedImprovement(model, best_f=3.5),
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
    )
    check_kind(cand, kind)
    assert shape(cand) == (1, 3)
    assert val is not None


def test_hierarchical_gate_validation_errors() -> None:
    with pytest.raises(ValueError, match="`d` must be an integer >= 2"):
        HierarchicalGateTransform(d=1, gates={0: 0})
    with pytest.raises(ValueError, match="`d` must be an integer >= 2"):
        HierarchicalGateTransform(d=True, gates={1: 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`gates` must be a non-empty"):
        HierarchicalGateTransform(d=2, gates={})
    with pytest.raises(ValueError, match="must be integers"):
        HierarchicalGateTransform(d=2, gates={True: 0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        HierarchicalGateTransform(d=2, gates={2: 0})
    with pytest.raises(ValueError, match="cannot be gated by itself"):
        HierarchicalGateTransform(d=2, gates={1: 1})
    with pytest.raises(ValueError, match="contains a cycle"):
        HierarchicalGateTransform(d=2, gates={0: 1, 1: 0})
    with pytest.raises(ValueError, match="contains a cycle"):
        HierarchicalGateTransform(d=3, gates={0: 1, 1: 2, 2: 0})
    with pytest.raises(ValueError, match=r"`thresholds` .* must lie in \[0, 1\)"):
        HierarchicalGateTransform(d=2, gates={1: 0}, thresholds=1.0)
    with pytest.raises(ValueError, match="must be finite"):
        HierarchicalGateTransform(d=2, gates={1: 0}, thresholds=float("nan"))
    with pytest.raises(ValueError, match=r"`neutral_values` .* must lie in \[0, 1\]"):
        HierarchicalGateTransform(d=2, gates={1: 0}, neutral_values=1.5)
    with pytest.raises(ValueError, match="`sharpness` .* must be >= 0"):
        HierarchicalGateTransform(d=2, gates={1: 0}, sharpness=-1.0)

    tf = HierarchicalGateTransform(d=2, gates={1: 0})
    with pytest.raises(BotorchTensorDimensionError):
        tf([0.2, 0.8])
    with pytest.raises(BotorchTensorDimensionError):
        tf([[0.1, 0.2, 0.3]])


def test_hierarchical_gate_vjp_matches_torch_autograd() -> None:
    torch = pytest.importorskip("torch")
    for sharpness in (0.0, 15.0):
        tf = HierarchicalGateTransform(
            d=3,
            gates={2: 1, 1: 0},
            thresholds={1: 0.15, 2: 0.35},
            neutral_values={1: 0.2, 2: 0.7},
            sharpness=sharpness,
        )
        X = torch.tensor(
            [
                [0.05, 0.8, 0.3],
                [0.45, 0.9, 0.2],
                [0.85, 0.75, 0.9],
            ],
            dtype=torch.double,
            requires_grad=True,
        )
        adj = torch.tensor(
            [
                [0.3, -1.1, 0.7],
                [-0.4, 0.9, -0.5],
                [0.8, 0.2, -0.6],
            ],
            dtype=torch.double,
        )
        Y = tf(X)
        (expected_grad,) = torch.autograd.grad(Y, X, grad_outputs=adj)
        actual_flat = _gates_vjp_flat(
            X.detach().reshape(-1).tolist(),
            3,
            3,
            adj.reshape(-1).tolist(),
            tf._ordered_gates,
        )
        actual_grad = torch.tensor(actual_flat, dtype=torch.double).reshape(3, 3)
        assert torch.allclose(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_parallel_lbfgsb_recovers_from_nonfinite_trial_and_flags_bad_x0() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from botorch_rs.generation.gen import gen_candidates_scipy

    class TrialSpikeAcq(torch.nn.Module):
        """Finite near initial starts, returns NaN past x > 0.75 on aggressive steps."""

        _tensor_backend = "torch"

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            x = X.squeeze(-2)
            # Steep slope toward +x[0] pushes L-BFGS-B trial step past 0.75 from 0.65
            base = 4.0 * x[..., 0] - ((x - 0.5) ** 2).sum(dim=-1)
            bad = x[..., 0] > 0.75
            if bool(bad.any()):
                out = base.clone()
                out[bad] = torch.log(-x[bad, 0].abs() - 1.0)
                return out
            return base

    starts = torch.tensor(
        [[[0.1, 0.2]], [[0.65, 0.4]]],
        dtype=torch.double,
    )
    lo = torch.zeros(2, dtype=torch.double)
    hi = torch.ones(2, dtype=torch.double)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizationWarning)
        cands, vals = gen_candidates_scipy(
            starts,
            TrialSpikeAcq(),
            lower_bounds=lo,
            upper_bounds=hi,
            use_parallel_mode=True,
        )
    assert torch.isfinite(cands).all()
    assert torch.isfinite(vals).all()

    # Non-finite gradient at initial x0 raises OptimizationGradientError
    bad_starts = torch.tensor([[[0.9, 0.2]]], dtype=torch.float32)
    with pytest.raises(OptimizationGradientError, match="Consider using `dtype=torch.double`"):
        gen_candidates_scipy(
            bad_starts,
            TrialSpikeAcq(),
            lower_bounds=lo.float(),
            upper_bounds=hi.float(),
            use_parallel_mode=True,
        )
