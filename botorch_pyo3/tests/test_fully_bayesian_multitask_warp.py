"""Native fully Bayesian multitask warping, matched to corrected official GP."""

# ruff: noqa: E402

from __future__ import annotations

import copy

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models.fully_bayesian_multitask import (  # noqa: E402
    MultitaskSaasPyroModel as ReferencePyro,
)
from botorch.models.fully_bayesian_multitask import (
    SaasFullyBayesianMultiTaskGP as Reference,
)
from botorch.models.transforms.input import (  # noqa: E402
    ChainedInputTransform as ReferenceChain,
)
from botorch.models.transforms.input import (
    Normalize as ReferenceNormalize,
)
from botorch.models.transforms.input import (
    Warp as ReferenceWarp,
)
from botorch.models.transforms.outcome import Standardize as ReferenceStandardize  # noqa: E402
from fully_bayesian_helpers import jax_x64  # noqa: E402

from botorch_rs.fit import fit_fully_bayesian_model_nuts  # noqa: E402
from botorch_rs.models.fully_bayesian_multitask import (  # noqa: E402
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)
from botorch_rs.models.transforms.input import Normalize  # noqa: E402
from botorch_rs.models.transforms.outcome import Standardize  # noqa: E402


def build(kind="torch", task_feature=1, subset=False, transforms=False, fixed=False):
    features = torch.tensor(
        [[0.1, 0.3], [0.3, 0.8], [0.7, 0.2], [0.8, 0.6], [0.4, 0.7], [0.9, 0.4]],
        dtype=torch.float64,
    )
    labels = torch.tensor([3.0, 7.0, 3.0, 7.0, 3.0, 7.0], dtype=torch.float64).unsqueeze(-1)
    cols = [features[:, :task_feature], labels, features[:, task_feature:]]
    x = torch.cat(cols, -1)
    y = (features[:, 0].sin() + features[:, 1].square() + labels.reshape(-1) * 0.1).unsqueeze(-1)
    base = [c for c in range(3) if c != task_feature]
    indices = [base[-1]] if subset else base
    kws, reference_kws = {}, {}
    if transforms:
        x[:, base] = 2 + 3 * x[:, base]
        bounds = torch.stack((x.amin(0), x.amax(0)))
        kws = {
            "input_transform": Normalize(3, indices=base, bounds=as_kind(bounds.tolist(), kind)),
            "outcome_transform": Standardize(1),
        }
        reference_kws = {
            "input_transform": ReferenceNormalize(3, indices=base, bounds=bounds),
            "outcome_transform": ReferenceStandardize(1),
        }
    noise = torch.linspace(0.02, 0.05, len(x), dtype=torch.float64).unsqueeze(-1) if fixed else None
    warp_indices = indices if subset else None
    model = SaasFullyBayesianMultiTaskGP(
        as_kind(x.tolist(), kind),
        as_kind(y.tolist(), kind),
        task_feature=task_feature,
        train_Yvar=None if noise is None else as_kind(noise.tolist(), kind),
        rank=2,
        pyro_model=MultitaskSaasPyroModel(
            use_input_warping=True, indices_to_warp=warp_indices, eps=0.03
        ),
        **kws,
    )
    with jax_x64():
        reference = Reference(
            x,
            y,
            task_feature=task_feature,
            train_Yvar=noise,
            rank=2,
            pyro_model=ReferencePyro(
                use_input_warping=True, indices_to_warp=warp_indices, eps=0.03
            ),
            **reference_kws,
        )
    samples = {
        "mean": torch.tensor([[0.2, 0.3], [-0.2, 0.1], [0.4, -0.1]], dtype=torch.float64),
        "lengthscale": torch.tensor([[0.6, 0.8], [0.9, 0.5], [0.7, 1.1]], dtype=torch.float64),
        "outputscale": torch.tensor([1.2, 0.9, 1.4], dtype=torch.float64),
        "latent_features": torch.tensor(
            [[[0.1, 0.2], [0.9, 0.7]], [[0.3, 0.1], [0.8, 0.6]], [[0.2, 0.5], [1.1, 0.3]]],
            dtype=torch.float64,
        ),
        "task_lengthscale": torch.tensor([[1.3, 0.9], [0.8, 1.1], [1.0, 0.6]], dtype=torch.float64),
        "c0": torch.tensor([[0.6, 1.5], [1.4, 0.7], [0.9, 1.7]], dtype=torch.float64)[
            ..., : len(indices)
        ],
        "c1": torch.tensor([[1.3, 0.8], [0.7, 1.6], [1.4, 0.6]], dtype=torch.float64)[
            ..., : len(indices)
        ],
    }
    if not fixed:
        samples["noise"] = torch.tensor([0.03, 0.06, 0.04], dtype=torch.float64)
    model.load_mcmc_samples({key: as_kind(value.tolist(), kind) for key, value in samples.items()})
    # Its discarded Warp is also built with ard_num_dims rather than the
    # full input width, so an explicit last physical feature index raises.
    # Bypass just that broken construction, then attach the corrected Warp.
    reference.pyro_model.use_input_warping = False
    reference.load_mcmc_samples(samples)
    reference.pyro_model.use_input_warping = True
    # Official b0dd discards the returned warp in multitask load_mcmc_samples.
    # Install the very same canonical Torch Warp used by its single-task GPs,
    # with the task column excluded and its sampled concentrations preserved.
    warp = ReferenceWarp(
        3,
        indices=indices,
        batch_shape=torch.Size((3,)),
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64),
        eps=0.03,
    )
    warp.concentration0.data = samples["c0"].unsqueeze(-2).clone()
    warp.concentration1.data = samples["c1"].unsqueeze(-2).clone()
    common = getattr(reference, "input_transform", None)
    reference.input_transform = (
        warp if common is None else ReferenceChain(normalize=common, warp=warp)
    )
    return model, reference, x, y, base


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("task_feature", [0, 1, 2])
@pytest.mark.parametrize("subset", [False, True])
def test_native_warped_posteriors_match_corrected_official(kind, task_feature, subset):
    model, reference, _, _, _ = build(kind, task_feature, subset)
    query = torch.tensor([[0.2, 0.6], [0.6, 0.4]], dtype=torch.float64)
    actual, expected = model.posterior(as_kind(query.tolist(), kind)), reference.posterior(query)
    assert_close(actual.mean, expected.mean, rtol=2e-7, atol=2e-8)
    assert_close(actual.variance, expected.variance, rtol=2e-7, atol=2e-8)
    # Task labels stay external values, including an interior task column.
    full = torch.cat(
        (
            query[:, :task_feature],
            torch.tensor([[7.0], [3.0]], dtype=torch.float64),
            query[:, task_feature:],
        ),
        -1,
    )
    assert_close(
        model.posterior(as_kind(full.tolist(), kind)).mean,
        reference.posterior(full).mean,
        rtol=2e-7,
        atol=2e-8,
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_common_normalization_standardization_and_native_query_gradients(fixed):
    model, reference, _, _, _ = build(transforms=True, fixed=fixed)
    query = torch.tensor([[2.6, 3.7], [3.9, 3.2]], dtype=torch.float64, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    actual, expected = model.posterior(query), reference.posterior(other)
    assert_close(actual.mean, expected.mean, rtol=3e-7, atol=3e-8)
    assert_close(actual.variance, expected.variance, rtol=3e-7, atol=3e-8)
    ag = torch.autograd.grad(actual.mean.square().sum() + actual.variance.sum(), query)[0]
    bg = torch.autograd.grad(expected.mean.square().sum() + expected.variance.sum(), other)[0]
    torch.testing.assert_close(ag, bg, rtol=3e-6, atol=3e-7)


def test_sample_warp_mutation_snapshots_copy_and_conditioning():
    model, _, _, _, _ = build()
    query = torch.tensor([[0.2, 0.6]], dtype=torch.float64)
    posterior = model.posterior(query)
    before = posterior.mean.clone()
    copied = copy.deepcopy(model)
    warp = model._samples[0].input_transform
    warp.concentration0 = torch.tensor([1.9, 0.7], dtype=torch.float64)
    assert not torch.allclose(before, model.posterior(query).mean)
    torch.testing.assert_close(posterior.mean, before)
    torch.testing.assert_close(copied.posterior(query).mean, before)
    conditioned = model.condition_on_observations(
        torch.tensor([[0.25, 7.0, 0.5]], dtype=torch.float64),
        torch.tensor([[0.9]], dtype=torch.float64),
    )
    assert torch.isfinite(conditioned.posterior(query).mean).all()
    with pytest.raises(ValueError):
        warp.concentration0 = -1
    assert torch.isfinite(model.posterior(query).mean).all()


@pytest.mark.parametrize("kind", KINDS)
def test_native_nuts_actually_fits_warped_multitask_model(kind):
    model, _, x, y, _ = build(kind)
    fresh = SaasFullyBayesianMultiTaskGP(
        as_kind(x.tolist(), kind),
        as_kind(y.tolist(), kind),
        task_feature=1,
        rank=2,
        pyro_model=MultitaskSaasPyroModel(use_input_warping=True, eps=0.03),
    )
    fit_fully_bayesian_model_nuts(
        fresh, warmup_steps=6, num_samples=4, thinning=1, max_tree_depth=2, seed=40
    )
    assert fresh.num_mcmc_samples == 4
    assert fresh.pyro_model._warp_indices() == [0, 2]
    assert torch.isfinite(torch.as_tensor(fresh.posterior(as_kind([[0.3, 0.7]], kind)).mean)).all()


@pytest.mark.parametrize("indices", [[1], [0, 0], [3], [], [-4]])
def test_invalid_warp_feature_selection_is_rejected(indices):
    with pytest.raises(ValueError, match="non-task input columns"):
        SaasFullyBayesianMultiTaskGP(
            [[0.2, 3.0, 0.4], [0.8, 7.0, 0.6]],
            [[0.5], [0.9]],
            task_feature=1,
            pyro_model=MultitaskSaasPyroModel(use_input_warping=True, indices_to_warp=indices),
        )


@pytest.mark.parametrize("eps", [0, 0.5, float("nan")])
def test_invalid_warp_margin_is_rejected(eps):
    with pytest.raises(ValueError, match="eps"):
        SaasFullyBayesianMultiTaskGP(
            [[0.2, 3.0, 0.4], [0.8, 7.0, 0.6]],
            [[0.5], [0.9]],
            task_feature=1,
            pyro_model=MultitaskSaasPyroModel(use_input_warping=True, eps=eps),
        )


def test_native_invalid_warp_updates_preserve_fitted_state():
    model, _, _, _, _ = build()
    query = torch.tensor([[0.2, 0.6]], dtype=torch.float64)
    before = model.posterior(query).mean.clone()
    member = model._samples[0]
    for indices, c0, c1, eps in [
        ([1], [1.0], [1.0], 0.03),
        ([0, 0], [1.0, 1.0], [1.0, 1.0], 0.03),
        ([0, 2], [1.0], [1.0], 0.03),
        ([0, 2], [1.0, -1.0], [1.0, 1.0], 0.03),
        ([0, 2], [1.0, 1.0], [1.0, 1.0], 0.5),
    ]:
        with pytest.raises(ValueError):
            member._rs.set_input_warp(indices, c0, c1, eps)
        torch.testing.assert_close(model.posterior(query).mean, before)
