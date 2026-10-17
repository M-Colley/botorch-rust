"""Input and outcome transforms (phase 6b): the transform utilities against
closed forms, the stand-alone transforms, ``subset_output`` (BoTorch main's
re-indexing since meta-pytorch/botorch#3380), ``get_rounding_input_transform``,
the moments of transformed posteriors, and the errors (BoTorch's exception
types and messages). Runs without torch and BoTorch;
``test_parity_6b_transforms.py`` compares with BoTorch."""

from __future__ import annotations

import math

import pytest

from botorch_rs.acquisition import ExpectedImprovement
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.transforms import (
    AppendFeatures,
    Bilog,
    ChainedInputTransform,
    ChainedOutcomeTransform,
    FilterFeatures,
    InputPerturbation,
    InputStandardize,
    InteractionFeatures,
    Log,
    Log10,
    Normalize,
    OneHotToNumeric,
    Power,
    Round,
    Standardize,
    StratifiedStandardize,
    Warp,
    get_rounding_input_transform,
)
from botorch_rs.models.transforms.utils import (
    expand_and_copy_tensor,
    interaction_features,
    inv_kumaraswamy_warp,
    kumaraswamy_warp,
    lognorm_to_norm,
    nanstd,
    norm_to_lognorm,
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
    subset_transform,
)
from botorch_rs.priors import LogNormalPrior
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import as_kind, assert_close, check_kind, flatten, make_data, shape, to_list

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_lognormal_moments(kind: str) -> None:
    mu, var = [0.3, -1.2], [0.5, 2.0]
    mean = norm_to_lognorm_mean(as_kind(mu, kind), as_kind(var, kind))
    check_kind(mean, kind)
    assert_close(mean, [math.exp(m + v / 2) for m, v in zip(mu, var, strict=True)], rtol=1e-15)
    assert_close(
        norm_to_lognorm_variance(as_kind(mu, kind), as_kind(var, kind)),
        [math.expm1(v) * math.exp(2 * m + v) for m, v in zip(mu, var, strict=True)],
        rtol=1e-14,
    )
    cov = [[0.5, 0.2], [0.2, 2.0]]
    m_ln, c_ln = norm_to_lognorm(as_kind(mu, kind), as_kind(cov, kind))
    expected = [
        [
            math.exp(mu[i] + mu[j] + (cov[i][i] + cov[j][j]) / 2) * math.expm1(cov[i][j])
            for j in range(2)
        ]
        for i in range(2)
    ]
    assert_close(c_ln, expected, rtol=1e-14)
    # lognorm_to_norm inverts norm_to_lognorm.
    m_n, c_n = lognorm_to_norm(m_ln, c_ln)
    assert_close(m_n, mu, rtol=1e-13)
    assert_close(c_n, cov, rtol=1e-12)


def test_tensor_utilities(kind: str) -> None:
    X = as_kind([[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]], kind)  # 1 x 2 x 3
    feats = interaction_features(X)
    assert shape(feats) == (1, 2, 1, 3)
    assert_close(flatten(feats), [2.0, 3.0, 6.0, -0.5, 1.0, -2.0], rtol=1e-15)
    expanded = expand_and_copy_tensor(X, (4,))
    assert shape(expanded) == (4, 2, 3)
    with pytest.raises(RuntimeError, match="not broadcastable"):
        expand_and_copy_tensor(as_kind([[[0.0]], [[1.0]]], kind), (3,))
    nan = float("nan")
    s = nanstd(as_kind([[1.0, nan, 3.0], [2.0, 5.0, nan]], kind), dim=1)
    assert_close(s, [math.sqrt(2.0), math.sqrt(4.5)], rtol=1e-15)
    with pytest.raises(IndexError):
        nanstd(as_kind([1.0, 2.0], kind), dim=2)


def test_kumaraswamy_warp(kind: str) -> None:
    x = [[0.1, 0.5], [0.9, 0.3]]
    c0, c1, eps = [0.7, 1.8], [2.0, 0.5], 1e-7
    out = kumaraswamy_warp(as_kind(x, kind), as_kind(c0, kind), as_kind(c1, kind), eps=eps)
    check_kind(out, kind)
    expected = []
    for row in x:
        r = []
        for j, v in enumerate(row):
            v = v * (1 - 2 * eps) + eps
            r.append(1 - (1 - v ** c1[j]) ** c0[j])
        expected.append(r)
    assert_close(out, expected, rtol=1e-14)
    back = inv_kumaraswamy_warp(out, as_kind(c0, kind), as_kind(c1, kind), eps=eps)
    assert_close(back, x, rtol=1e-12)


def test_subset_transform_decorator() -> None:
    class Doubler:
        indices = [0, 2]

        @subset_transform
        def transform(self, X: list) -> list:
            return [[2 * v for v in row] for row in X]

    assert Doubler().transform([[1.0, 2.0, 3.0]]) == [[2.0, 2.0, 6.0]]


# ---------------------------------------------------------------------------
# Stand-alone input transforms
# ---------------------------------------------------------------------------


def test_input_transforms_stand_alone(kind: str) -> None:
    X = as_kind([[1.0, 10.0, 0.2], [3.0, 100.0, 0.4], [2.0, 1000.0, 0.9]], kind)
    st = InputStandardize(d=3, indices=[0])
    out = to_list(st(X))
    assert_close([r[0] for r in out], [-1.0, 1.0, 0.0], rtol=1e-15)
    assert [r[1:] for r in out] == [[10.0, 0.2], [100.0, 0.4], [1000.0, 0.9]]
    assert_close([r[1] for r in to_list(Log10(indices=[1])(X))], [1.0, 2.0, 3.0], rtol=1e-15)
    assert to_list(FilterFeatures(feature_indices=[2, 0])(X)) == [
        [0.2, 1.0],
        [0.4, 3.0],
        [0.9, 2.0],
    ]
    feats = to_list(InteractionFeatures(indices=[0, 2])(X))
    assert_close([r[-1] for r in feats], [0.2, 1.2, 1.8], rtol=1e-15)
    rd = Round(integer_indices=[0], transform_on_train=False)
    rd.eval()
    assert to_list(rd(as_kind([[1.5, 0.3], [2.5, 0.7], [-0.6, 0.1]], kind))) == [
        [2.0, 0.3],
        [2.0, 0.7],
        [-1.0, 0.1],
    ]
    oh = OneHotToNumeric(dim=4, categorical_features={1: 3})
    oh.eval()
    assert to_list(oh(as_kind([[0.5, 0.0, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]], kind))) == [
        [0.5, 1.0],
        [0.1, 2.0],
    ]
    app = AppendFeatures(feature_set=as_kind([[7.0], [8.0]], kind))
    app.eval()
    assert shape(app(as_kind([[0.1, 0.2]], kind))) == (2, 3)


def test_warp_learns_its_concentrations_with_priors() -> None:
    X, Y = make_data(n=12, d=2, seed=3)
    tf = Warp(
        d=2,
        indices=[0, 1],
        concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
        concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
    )
    model = SingleTaskGP(X, Y, input_transform=tf)
    names = [n for n, _ in model.named_parameters()]
    assert "input_transform.concentration0" in names and "input_transform.concentration1" in names
    assert flatten(tf.concentration0) == [1.0, 1.0]
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    assert flatten(tf.concentration0) != [1.0, 1.0]
    assert all(c >= 1e-4 for c in flatten(tf.concentration0) + flatten(tf.concentration1))


# ---------------------------------------------------------------------------
# Outcome transforms
# ---------------------------------------------------------------------------


def test_elementwise_outcome_transforms(kind: str) -> None:
    Y = as_kind([[1.0, -2.0], [4.0, 0.5]], kind)
    y, _ = Log(outputs=[0])(Y)
    assert_close(y, [[0.0, -2.0], [math.log(4.0), 0.5]], rtol=1e-15)
    back, _ = Log(outputs=[0]).untransform(y)
    assert_close(back, to_list(Y), rtol=1e-15)
    # Observation noise: Log by the delta method; Power and Bilog do not
    # support it.
    _, yvar = Log()(as_kind([[2.0]], kind), as_kind([[0.4]], kind))
    assert_close(yvar, [[0.1]], rtol=1e-15)
    for tf, name in ((Power(2.0), "Power"), (Bilog(), "Bilog")):
        with pytest.raises(NotImplementedError, match=f"{name} does not yet support"):
            tf(Y, as_kind([[0.1, 0.1], [0.1, 0.1]], kind))
    y, _ = Bilog()(Y)
    assert_close(y, [[math.log(2.0), -math.log(3.0)], [math.log(5.0), math.log(1.5)]], rtol=1e-15)
    y, _ = Power(0.5)(as_kind([[4.0, 9.0]], kind))
    assert_close(y, [[2.0, 3.0]], rtol=1e-15)


def test_subset_output_reindexes_outputs() -> None:
    """BoTorch main (meta-pytorch/botorch#3380): the outputs of a subset
    transform are the positions of the transformed outputs in ``idcs``."""
    sub = Log(outputs=[1, 3]).subset_output([3, 0, 1])
    assert sub._outputs == [0, 2]
    st = Standardize(m=3, outputs=[0, 2])
    st(as_kind([[1.0, 2.0, 3.0], [3.0, 6.0, 7.0]], "list"))
    sub_st = st.subset_output([2, 1])
    assert sub_st._outputs == [0]
    assert_close(sub_st.means, [[5.0, 0.0]], rtol=1e-15)
    with pytest.raises(NotImplementedError, match="Negative indexing"):
        Log(outputs=[1]).subset_output([-1])
    chain = ChainedOutcomeTransform(log=Log(outputs=[2]), std=Standardize(m=3))
    sub_chain = chain.subset_output([2])
    assert sub_chain["log"]._outputs == [0]


def test_stratified_standardize(kind: str) -> None:
    X = as_kind([[0.1, 0.0], [0.2, 1.0], [0.3, 0.0], [0.4, 1.0], [0.5, 2.0]], kind)
    Y = as_kind([[1.0], [10.0], [3.0], [30.0], [7.0]], kind)
    tf = StratifiedStandardize(stratification_idx=1, all_task_values=[0, 1, 2, 3])
    y, _ = tf(Y, X=X)
    # Strata 0 and 1 are standardized; stratum 2 has one point (zero spread:
    # standard deviation 1); stratum 3 has no data.
    s = math.sqrt(2.0)
    assert_close(flatten(y), [-1 / s, -1 / s, 1 / s, 1 / s, 0.0], atol=1e-15)


def test_transformed_posterior_moments(kind: str) -> None:
    X, Y = make_data(n=10, d=2, seed=4)
    Ypos = [[math.exp(r[0])] for r in Y]
    model = SingleTaskGP(as_kind(X, kind), as_kind(Ypos, kind), outcome_transform=Log())
    base = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), outcome_transform=None)
    base.initialize(**dict(model.named_parameters()))
    Xq = as_kind([[0.2, 0.4], [0.7, 0.1]], kind)
    post, pb = model.posterior(Xq), base.posterior(Xq)
    assert type(post).__name__ == "TransformedPosterior" or "Transformed" in type(post).__name__
    assert_close(post.mean, norm_to_lognorm_mean(pb.mean, pb.variance), rtol=1e-13)
    assert_close(post.variance, norm_to_lognorm_variance(pb.mean, pb.variance), rtol=1e-13)
    # Power and Bilog have no mean transform.
    pmodel = SingleTaskGP(as_kind(X, kind), as_kind(Ypos, kind), outcome_transform=Power(2.0))
    with pytest.raises(NotImplementedError, match="No mean transform provided."):
        pmodel.posterior(Xq).mean  # noqa: B018
    # Scalarizing a transformed posterior is unsupported (BoTorch scalarizes
    # the Gaussian base posterior of the transformed outcomes).
    with pytest.raises(UnsupportedError):
        model.posterior(Xq, posterior_transform=ScalarizedPosteriorTransform([1.0]))
    # Analytic acquisition functions use the moments.
    ei = ExpectedImprovement(model, best_f=1.0)
    assert shape(ei(as_kind([[[0.2, 0.4]]], kind))) == (1,)


def test_model_list_of_transformed_models_is_a_posterior_list() -> None:
    X, Y = make_data(n=8, d=2, seed=5)
    Ypos = [[math.exp(r[0])] for r in Y]
    ml = ModelListGP(SingleTaskGP(X, Ypos, outcome_transform=Log()), SingleTaskGP(X, Y))
    post = ml.posterior([[0.2, 0.4], [0.5, 0.5]])
    assert type(post).__name__ == "PosteriorList"
    assert shape(post.mean) == (2, 2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_rounding_input_transform(kind: str) -> None:
    bounds = as_kind([[0.0, 0.0, 0.0, 0.0], [4.0, 1.0, 1.0, 1.0]], kind)
    tf = get_rounding_input_transform(
        bounds, integer_indices=[0], categorical_features={1: 3}, return_numeric=True
    )
    out = to_list(tf(as_kind([[0.3, 0.2, 0.7, 0.1], [0.9, 0.6, 0.1, 0.65]], kind)))
    # x0 in [0, 4]: 1.2 -> 1 -> 0.25; 3.6 -> 4 -> 1.0; one-hot -> index.
    assert_close(out, [[0.25, 1.0], [1.0, 2.0]], rtol=1e-15)
    with pytest.raises(ValueError, match="no-op"):
        get_rounding_input_transform(bounds)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unsupported_configurations_raise() -> None:
    X, Y = make_data(n=8, d=2, seed=6)
    with pytest.raises(UnsupportedError, match="reverse=True"):
        SingleTaskGP(X, Y, input_transform=Warp(d=2, indices=[0], reverse=True))
    with pytest.raises(UnsupportedError, match="transform_on_train"):
        SingleTaskGP(
            X,
            Y,
            input_transform=AppendFeatures(feature_set=[[0.1], [0.2]], transform_on_train=True),
        )
    with pytest.raises(UnsupportedError, match="learns its coefficients after"):
        SingleTaskGP(
            X,
            Y,
            input_transform=ChainedInputTransform(
                warp=Warp(d=2, indices=[0]), normalize=Normalize(d=2)
            ),
        )
    with pytest.raises(UnsupportedError, match="chains its own input transforms"):
        ChainedInputTransform(a=Log())  # type: ignore[arg-type]


def test_append_features_with_features_in_the_training_inputs() -> None:
    """BoTorch's usual ``AppendFeatures`` setup: the training inputs include
    the feature columns, the points of a posterior do not."""
    X, Y = make_data(n=10, d=3, seed=7)
    model = SingleTaskGP(X, Y, input_transform=AppendFeatures(feature_set=[[0.1], [0.5], [0.9]]))
    post = model.posterior([[0.2, 0.3]])
    assert shape(post.mean) == (3, 1)
    with pytest.raises(UnsupportedError, match="condition_on_observations is not supported"):
        model.condition_on_observations([[0.2, 0.3]], [[1.0]])


def test_fantasies_skip_input_perturbations() -> None:
    X, Y = make_data(n=8, d=2, seed=8)
    tf = InputPerturbation([[0.05, 0.0], [-0.05, 0.02]])
    model = SingleTaskGP(X, Y, input_transform=tf)
    model.posterior([[0.5, 0.5]])
    fm = model.fantasize([[0.2, 0.3]], SobolQMCNormalSampler(sample_shape=[2], seed=0))
    assert fm.batch_shape == (2,)


def test_input_perturbation_batch_shape(kind: str) -> None:
    """BoTorch's shape of the cached perturbations without ``n x d``."""
    tf = InputPerturbation(as_kind([[0.1, 0.0], [0.0, 0.1], [0.2, 0.2]], kind))
    with pytest.raises(AttributeError):
        tf.batch_shape  # noqa: B018
    tf.transform(as_kind([[[0.2, 0.3]], [[0.4, 0.5]]], kind))
    assert tf.batch_shape == (3,)
