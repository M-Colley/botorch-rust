"""API and error tests of the models of phase 7b (multi-fidelity, mixed and
MAP SAAS GPs, deterministic models, model lists and dicts, ensembles,
``subset_output``, ``construct_inputs``, ``SupervisedDataset``,
``LogTransformedInterval``), without BoTorch.

The exceptions and messages are BoTorch's.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from botorch_rs.constraints import GreaterThan, Interval
from botorch_rs.exceptions.errors import InputDataError, UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import LinearKernel, MaternKernel, RBFKernel
from botorch_rs.means import ConstantMean
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import (
    AdditiveMapSaasSingleTaskGP,
    AffineDeterministicModel,
    EnsembleMapSaasSingleTaskGP,
    MixedSingleTaskGP,
    ModelList,
    SingleTaskGP,
    SingleTaskMultiFidelityGP,
)
from botorch_rs.models.deterministic import FixedSingleSampleModel
from botorch_rs.models.ensemble import EnsembleModel
from botorch_rs.models.map_saas import (
    SaasPriorHelper,
    add_saas_prior,
    get_additive_map_saas_covar_module,
    get_gaussian_likelihood_with_gamma_prior,
    get_map_saas_model,
    get_mean_module_with_normal_prior,
)
from botorch_rs.models.model import ModelDict
from botorch_rs.posteriors.ensemble import EnsemblePosterior
from botorch_rs.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch_rs.priors import GammaPrior, NormalPrior
from botorch_rs.utils.constraints import LogTransformedInterval, NonTransformedInterval
from botorch_rs.utils.datasets import SupervisedDataset
from helpers import HAS_TORCH


def _data(n: int = 10, d: int = 3, m: int = 1, seed: int = 0) -> tuple[list, list]:
    import random

    r = random.Random(seed)
    X = [[r.random() for _ in range(d)] for _ in range(n)]
    Y = [
        [
            math.sin(3 * x[0]) + sum(v * v for v in x[1:]) + 0.3 * t + 0.01 * r.random()
            for t in range(m)
        ]
        for x in X
    ]
    return X, Y


def _mixed_data(n: int = 10, seed: int = 0) -> tuple[list, list]:
    import random

    r = random.Random(seed)
    X = [[r.random(), r.random(), float(r.randrange(3))] for _ in range(n)]
    Y = [[math.sin(3 * x[0]) + x[1] + 0.5 * x[2]] for x in X]
    return X, Y


# ---------------------------------------------------------------------------
# SingleTaskMultiFidelityGP
# ---------------------------------------------------------------------------


def test_multi_fidelity_errors() -> None:
    X, Y = _data(d=3)
    with pytest.raises(UnsupportedError, match="requires at least one fidelity parameter"):
        SingleTaskMultiFidelityGP(X, Y)
    with pytest.raises(UnsupportedError, match="requires at least one fidelity parameter"):
        SingleTaskMultiFidelityGP(X, Y, data_fidelities=[])
    with pytest.raises(ValueError, match="Non-fidelity covariance module cannot be specified"):
        SingleTaskMultiFidelityGP(X, Y, data_fidelities=[2], covar_module=RBFKernel())
    with pytest.raises(ValueError, match="nu must be one of 0.5, 1.5, or 2.5"):
        SingleTaskMultiFidelityGP(X, Y, data_fidelities=[2], nu=3.5)


def test_multi_fidelity_structure_and_attributes() -> None:
    X, Y = _data(d=4, m=2)
    m = SingleTaskMultiFidelityGP(X, Y, iteration_fidelity=2, data_fidelities=[3])
    assert m._init_args["iteration_fidelity"] == 2
    assert m.num_outputs == 2 and m._aug_batch_shape == (2,)
    names = [n for n, _ in m.named_parameters()]
    assert names[:3] == [
        "likelihood.noise_covar.raw_noise",
        "mean_module.raw_constant",
        "covar_module.raw_outputscale",
    ]
    assert m._subset_batch_dict["covar_module.base_kernel.kernels.0.raw_power"] == -2
    assert m._subset_batch_dict["likelihood.noise_covar.raw_noise"] == -2
    ds = SingleTaskMultiFidelityGP(X, Y, data_fidelities=[3], linear_truncated=False)
    assert "covar_module.kernels.1.raw_offset" in ds._subset_batch_dict
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    fit_gpytorch_mll(mll)
    post = m.posterior([[0.1, 0.2, 0.3, 1.0]])
    assert len(post.mean) == 1 and len(post.mean[0]) == 2
    sub = m.subset_output([1])
    assert sub.num_outputs == 1 and isinstance(sub, SingleTaskMultiFidelityGP)
    assert sub.covar_module.batch_shape == ()
    assert "SingleTaskMultiFidelityGP" in repr(m)


def test_subset_output_errors_and_identity() -> None:
    X, Y = _data(m=2)
    m = SingleTaskGP(X, Y)
    with pytest.raises(IndexError):
        m.subset_output([2])
    same = m.subset_output([0, 1])
    assert same is not m and same.num_outputs == 2
    one = m.subset_output([0])
    assert one.num_outputs == 1 and one.batch_shape == ()


# ---------------------------------------------------------------------------
# MixedSingleTaskGP
# ---------------------------------------------------------------------------


def test_mixed_errors_and_gradients() -> None:
    X, Y = _mixed_data()
    with pytest.raises(ValueError, match="Must specify categorical dimensions"):
        MixedSingleTaskGP(X, Y, cat_dims=[])
    with pytest.raises(ValueError, match="out of bounds"):
        MixedSingleTaskGP(X, Y, cat_dims=[5])
    m = MixedSingleTaskGP(X, Y, cat_dims=[-1])
    names = [n for n, _ in m.named_parameters()]
    assert names[2] == "covar_module.kernels.0.raw_outputscale"
    assert m._ignore_X_dims_scaling_check == [-1]
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    from botorch_rs.acquisition import PosteriorMean

    value, grad = PosteriorMean(m).value_and_grad([[[0.3, 0.4, 1.0]]])
    assert grad[0][0][2] == 0.0  # no gradient with respect to a category
    assert math.isfinite(value[0])
    factory_calls = []

    def factory(**kw: Any) -> Any:
        factory_calls.append(kw)
        return MaternKernel(nu=1.5, **kw)

    MixedSingleTaskGP(X, Y, cat_dims=[2], cont_kernel_factory=factory)
    assert factory_calls[0] == {"batch_shape": (), "ard_num_dims": 2, "active_dims": [0, 1]}


def test_mixed_construct_inputs() -> None:
    X, Y = _mixed_data()
    ds = SupervisedDataset(X, Y, feature_names=["a", "b", "c"], outcome_names=["y"])
    kw = MixedSingleTaskGP.construct_inputs(ds, categorical_features=[2])
    assert kw["cat_dims"] == [2] and kw["likelihood"] is None
    assert MixedSingleTaskGP(**kw).num_outputs == 1
    with pytest.raises(TypeError, match="SupervisedDataset"):
        SingleTaskGP.construct_inputs({"X": X})


# ---------------------------------------------------------------------------
# MAP SAAS
# ---------------------------------------------------------------------------


def test_add_saas_prior_errors() -> None:
    with pytest.raises(UnsupportedError, match="must have lengthscale"):
        add_saas_prior(LinearKernel())
    with pytest.raises(UnsupportedError, match="must not specify a lengthscale prior"):
        add_saas_prior(MaternKernel(lengthscale_prior=GammaPrior(3.0, 6.0)))
    with pytest.raises(ValueError, match="Expected tau to have shape"):
        add_saas_prior(MaternKernel(ard_num_dims=2, batch_shape=[3]), tau=[0.1, 0.2])
    k = add_saas_prior(MaternKernel(ard_num_dims=2), tau=None)
    assert [n for n, _ in k.named_parameters()] == ["raw_lengthscale", "raw_tau"]
    assert isinstance(k.raw_lengthscale_constraint, LogTransformedInterval)
    X, Y = _data(d=2)
    model = SingleTaskGP(X, Y, covar_module=k)
    with pytest.raises(UnsupportedError, match="before the kernel is part of a model"):
        add_saas_prior(k)
    assert abs(model.covar_module.raw_tau - math.log(0.1)) < 1e-6
    helper = SaasPriorHelper()
    tau = helper.tau_prior_param_or_closure(model.covar_module)
    assert abs(float(tau) - 0.1) < 1e-6
    inv = helper.inv_lengthscale_prior_param_or_closure(model.covar_module)
    assert abs(inv[0][0] - float(tau)) < 1e-6  # lengthscale 1
    helper.inv_lengthscale_prior_setting_closure(model.covar_module, [[0.4, 0.025]])
    ls = model.covar_module.lengthscale[0]
    assert abs(ls[0] - math.sqrt(float(tau) / 0.4)) < 1e-9
    helper.tau_prior_setting_closure(model.covar_module, 100.0)  # clamped to 10 - EPS
    assert abs(float(helper.tau_prior_param_or_closure(model.covar_module)) - 10.0) < 1e-6


def test_map_saas_models() -> None:
    X, Y = _data(d=3)
    m = get_map_saas_model(X, Y)
    names = [n for n, _ in m.named_parameters()]
    assert names == [
        "likelihood.noise_covar.raw_noise",
        "mean_module.raw_constant",
        "covar_module.raw_outputscale",
        "covar_module.base_kernel.raw_lengthscale",
        "covar_module.base_kernel.raw_tau",
    ]
    bounds = m._rs.raw_parameter_bounds()
    assert bounds[1] == (-10.0, 10.0)
    assert abs(bounds[0][0] - math.log(1e-4)) < 1e-6 and bounds[0][1] == 0.0
    loss = fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    assert loss is not None
    fixed = get_map_saas_model(X, Y, tau=0.2)
    assert len(list(fixed.named_parameters())) == 4
    add = AdditiveMapSaasSingleTaskGP(X, Y, num_taus=2)
    assert len(add.covar_module.kernels) == 2
    k = get_additive_map_saas_covar_module(ard_num_dims=3, num_taus=3, batch_shape=[2])
    assert len(k.kernels) == 3 and k.batch_shape == (2,)
    assert get_mean_module_with_normal_prior().raw_constant_constraint.upper_bound == 10.0
    lik = get_gaussian_likelihood_with_gamma_prior()
    assert isinstance(lik.noise_constraint, LogTransformedInterval)


def test_ensemble_map_saas_errors_and_posterior() -> None:
    X, Y = _data(d=3)
    with pytest.raises(ValueError, match="Expected taus to be of shape"):
        EnsembleMapSaasSingleTaskGP(X, Y, num_taus=3, taus=[0.1, 0.2])
    with pytest.raises(UnsupportedError, match="only supports single-output"):
        EnsembleMapSaasSingleTaskGP(X, [y + y for y in Y])
    with pytest.raises(UnsupportedError, match="only supports 2D inputs"):
        EnsembleMapSaasSingleTaskGP([X, X], [Y, Y])
    m = EnsembleMapSaasSingleTaskGP(X, Y, num_taus=3, taus=[0.1, 0.2, 0.3])
    assert m._is_ensemble and m.batch_shape == (3,)
    post = m.posterior([[0.2, 0.3, 0.4], [0.5, 0.5, 0.5]])
    assert isinstance(post, GaussianMixturePosterior)
    assert len(post.mean) == 3  # num_taus x q x m
    mix = post.mixture_mean
    assert len(mix) == 2 and len(mix[0]) == 1
    q = post.quantile([0.5])
    assert len(q) == 2
    members = m._ensemble_members()
    assert len(members) == 3 and members[0].batch_shape == ()
    from botorch_rs.acquisition import LogExpectedImprovement, qKnowledgeGradient

    acqf = LogExpectedImprovement(m, best_f=1.0)
    assert len(acqf._ensemble_acqfs) == 3
    with pytest.raises(UnsupportedError, match="does not support ensemble models"):
        qKnowledgeGradient(m, num_fantasies=2)


def test_log_transformed_and_non_transformed_intervals() -> None:
    c = LogTransformedInterval(0.01, 1e4, initial_value=10)
    assert c.enforced
    assert abs(c.transform(0.5) - math.exp(0.5)) < 1e-15
    assert abs(c.inverse_transform(2.0) - math.log(2.0)) < 1e-15
    assert c._rs.raw_bounds[0] == math.log(c.lower_bound)
    assert abs(c.initial_value - math.log(10.0)) < 1e-6
    with pytest.raises(ValueError):
        LogTransformedInterval(1.0, 0.5)
    n = NonTransformedInterval(-1.0, 2.0, initial_value=0.5)
    assert not n.enforced and n.transform(0.3) == 0.3


def test_constant_mean_prior_and_constraint() -> None:
    with pytest.raises(UnsupportedError, match="not enforced"):
        ConstantMean(constant_constraint=Interval(-1.0, 1.0))
    with pytest.raises(TypeError, match="Prior"):
        ConstantMean(constant_prior=1.0)
    mean = ConstantMean(
        constant_prior=NormalPrior(0.0, 1.0),
        constant_constraint=GreaterThan(-5.0, transform=None),
    )
    X, Y = _data()
    m = SingleTaskGP(X, Y, mean_module=mean)
    assert m._rs.raw_parameter_bounds()[1] == (-5.0, math.inf)


# ---------------------------------------------------------------------------
# Deterministic models, ModelList, ModelDict
# ---------------------------------------------------------------------------


def test_affine_deterministic_model() -> None:
    with pytest.raises(ValueError, match="a must be two-dimensional"):
        AffineDeterministicModel([1.0, 2.0])
    with pytest.raises(ValueError, match="b must be one-dimensional"):
        AffineDeterministicModel([[1.0], [2.0]], b=[[0.1]])
    with pytest.raises(RuntimeError):
        AffineDeterministicModel([[1.0, 2.0]], b=[0.1, 0.2, 0.3])
    m = AffineDeterministicModel([[1.0, -1.0], [2.0, 0.5]], b=[0.5, 0.25])
    assert m.num_outputs == 2
    assert m.forward([[1.0, 1.0]]) == [[3.5, -0.25]]
    assert m.subset_output([1]).forward([[1.0, 1.0]]) == [[-0.25]]
    from botorch_rs.acquisition import PosteriorMean
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform

    with pytest.raises(UnsupportedError):
        PosteriorMean(m, posterior_transform=ScalarizedPosteriorTransform([1.0, 0.0]))


def test_fixed_single_sample_model() -> None:
    X, Y = _data()
    gp = SingleTaskGP(X, Y)
    m = FixedSingleSampleModel(gp, w=[0.5])
    out = m.forward([[0.1, 0.2, 0.3]])
    post = gp.posterior([[0.1, 0.2, 0.3]])
    expected = post.mean[0][0] + math.sqrt(post.variance[0][0] + 1e-8) * 0.5
    assert abs(out[0][0] - expected) < 1e-12
    assert len(FixedSingleSampleModel(gp, dim=1).w) == 1


def test_model_list_and_dict() -> None:
    X, Y = _data(m=2)
    a = SingleTaskGP(X, [y[:1] for y in Y])
    b = SingleTaskGP(X, Y)
    det = AffineDeterministicModel([[1.0], [0.0], [0.0]], b=0.0)
    lst = ModelList(a, b, det)
    assert lst.num_outputs == 4
    assert lst._get_group_subset_indices([0, 2, 3]) == {0: [0], 1: [1], 2: [0]}
    assert lst.subset_output([3]) is det
    sub = lst.subset_output([0, 2])
    assert isinstance(sub, ModelList) and sub.num_outputs == 2
    post = lst.posterior([[0.1, 0.2, 0.3]])
    assert len(post.mean[0]) == 4
    with pytest.raises(AttributeError, match="does not define the attribute"):
        post.distribution  # noqa: B018
    batched = SingleTaskGP([X, X], [Y, Y])
    with pytest.raises(NotImplementedError, match="only supported if all"):
        ModelList(a, batched).batch_shape  # noqa: B018
    with pytest.raises(InputDataError, match="Expected all models"):
        ModelDict(a=a, b=1)
    d = ModelDict(a=a, b=b)
    assert d.a is a and d["b"] is b


# ---------------------------------------------------------------------------
# Ensembles, datasets
# ---------------------------------------------------------------------------


def test_ensemble_posterior() -> None:
    values = [[[[1.0], [2.0]], [[3.0], [4.0]]]]  # 1 x s=2 x q=2 x m=1
    p = EnsemblePosterior(values)
    assert p.ensemble_size == 2 and p.batch_shape == (1,)
    assert p.mean == [[[2.0], [3.0]]]
    assert p.variance == [[[2.0], [2.0]]]
    assert p.rsample_from_base_samples((2,), [[1.0], [0.0]]) == [
        [[[3.0], [4.0]]],
        [[[1.0], [2.0]]],
    ]
    with pytest.raises(ValueError, match="does not match"):
        p.rsample_from_base_samples((3,), [[1.0], [0.0]])
    with pytest.raises(ValueError, match="at least three-dimensional"):
        EnsemblePosterior([[1.0]])
    w = EnsemblePosterior(values, weights=[1.0, 3.0])
    assert w.weights == [0.25, 0.75]


@pytest.mark.skipif(not HAS_TORCH, reason="EnsembleModel acquisition needs torch")
def test_ensemble_model_errors() -> None:
    import torch

    from botorch_rs.acquisition import qSimpleRegret
    from botorch_rs.sampling import SobolQMCNormalSampler

    class Ens(EnsembleModel):
        _num_outputs = 1

        def forward(self, X: Any) -> Any:
            return torch.stack([X.sum(-1, keepdim=True), X.prod(-1, keepdim=True)], dim=-3)

    m = Ens()
    # These checks exercise the native callback adapter's dimension and sampler
    # validation. Caller-defined Torch Model modules also have a device route.
    m._tensor_backend = "native"
    with pytest.raises(UnsupportedError, match="do not support observation noise"):
        m.posterior(torch.rand(2, 3, dtype=torch.float64), observation_noise=torch.ones(2, 1))
    with pytest.raises(UnsupportedError, match="input dimension"):
        qSimpleRegret(m)
    m.dim = 3
    with pytest.raises(UnsupportedError, match="IndexSampler"):
        qSimpleRegret(m, sampler=SobolQMCNormalSampler((4,)))
    a = qSimpleRegret(m)
    v = a(torch.rand(1, 2, 3, dtype=torch.float64))
    assert v.shape == (1,)

    from botorch_rs.acquisition import (
        ExpectedImprovement,
        NoisyExpectedImprovement,
        ScalarizedPosteriorTransform,
    )

    # Analytic: the Gaussian moments of the EnsemblePosterior.
    ei = ExpectedImprovement(m, best_f=0.5)
    assert ei.model is m
    X = torch.rand(4, 1, 3, dtype=torch.float64)
    p = m.posterior(X)
    mu, sigma = p.mean.squeeze(-1).squeeze(-1), p.variance.squeeze(-1).squeeze(-1).sqrt()
    u = (mu - 0.5) / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    ref = sigma * (normal.log_prob(u).exp() + u * normal.cdf(u))
    assert torch.allclose(ei(X), ref, rtol=1e-12, atol=0.0)
    w = torch.ones(1, dtype=torch.float64)
    with pytest.raises(UnsupportedError, match="Posterior transforms are not supported"):
        ExpectedImprovement(m, best_f=0.5, posterior_transform=ScalarizedPosteriorTransform(w))
    with pytest.raises(UnsupportedError, match="does not support ensemble models"):
        NoisyExpectedImprovement(m, X_observed=torch.rand(3, 3, dtype=torch.float64))


def test_supervised_dataset() -> None:
    X, Y = _data(n=4, d=2)
    ds = SupervisedDataset(X, Y, feature_names=["a", "b"], outcome_names=["y"])
    assert ds.Yvar is None and ds == ds.clone()
    with pytest.raises(ValueError, match="Batch dimensions of `X` and `Y` are incompatible"):
        SupervisedDataset(X, Y[:3], feature_names=["a", "b"], outcome_names=["y"])
    with pytest.raises(ValueError, match="Shapes of `Y` and `Yvar` are incompatible"):
        SupervisedDataset(X, Y, Yvar=[[0.1]], feature_names=["a", "b"], outcome_names=["y"])
    with pytest.raises(ValueError, match="number of features"):
        SupervisedDataset(X, Y, feature_names=["a"], outcome_names=["y"])
    with pytest.raises(ValueError, match="number of outcomes"):
        SupervisedDataset(X, Y, feature_names=["a", "b"], outcome_names=["y", "z"])
    kw = SingleTaskGP.construct_inputs(ds)
    assert set(kw) == {"train_X", "train_Y"}
    masked = ds.clone(mask=[True, False, True, True])
    assert len(masked.X) == 3
