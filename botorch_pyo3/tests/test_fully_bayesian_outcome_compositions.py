"""Native fully Bayesian Gaussian outcome compositions preserve their law."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_fully_bayesian_multitask import _data, _samples

from botorch_rs.acquisition.analytic import LogExpectedImprovement
from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch_rs.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Standardize


class InheritedStandardize(Standardize):
    pass


def _outcome(kind):
    if kind == "selected":
        return Standardize(1, outputs=[0])
    if kind == "identity":
        return Standardize(1, outputs=[])
    if kind == "inherited":
        return InheritedStandardize(1)
    if kind == "mean_only":
        return ChainedOutcomeTransform(
            first=Standardize(1, min_stdv=100.0), second=Standardize(1, min_stdv=100.0)
        )
    return ChainedOutcomeTransform(
        first=Standardize(1), second=Standardize(1), third=Standardize(1, outputs=[])
    )


@pytest.mark.parametrize("multitask", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("outcome", ["selected", "identity", "inherited", "chain", "mean_only"])
def test_native_gaussian_outcome_compositions_match_tensor_law_and_conditioning(
    multitask, fixed, outcome
):
    x, y, noise = _data("cpu")
    y = 2.3 * y + 1.7
    noise = noise * 2.3**2
    if not multitask:
        x = x[:, [0, 2]]
    samples = _samples(x)
    if not multitask:
        samples = {
            key: value
            for key, value in samples.items()
            if key not in {"latent_features", "task_lengthscale", "c0", "c1"}
        }
        samples["mean"] = samples["mean"][:, 0]
    cls = SaasFullyBayesianMultiTaskGP if multitask else SaasFullyBayesianSingleTaskGP
    models = []
    for backend in ("native", "torch"):
        model = cls(
            x,
            y,
            train_Yvar=noise if fixed else None,
            outcome_transform=_outcome(outcome),
            backend=backend,
            **({"task_feature": 1, "rank": 2} if multitask else {}),
        )
        model.load_mcmc_samples(samples)
        models.append(model.eval())
    query = x.new_tensor([[0.2, 0.7], [0.6, 0.4]])

    def compare(a, b):
        actual, expected = (
            a.posterior(query, observation_noise=True),
            b.posterior(query, observation_noise=True),
        )
        torch.testing.assert_close(actual.mean, expected.mean, atol=3e-10, rtol=2e-9)
        torch.testing.assert_close(
            actual.distribution.covariance_matrix,
            expected.distribution.covariance_matrix,
            atol=3e-10,
            rtol=2e-9,
        )

    compare(*models)
    if multitask:
        members = models[0]._samples
        for member in members:
            actual_raw, shape = member._rs.train_targets()
            torch.testing.assert_close(
                torch.tensor(actual_raw, dtype=y.dtype).reshape(shape), y, atol=2e-14, rtol=2e-14
            )
            torch.testing.assert_close(
                member.train_targets,
                models[0].pyro_model.train_Y.squeeze(-1),
                atol=2e-14,
                rtol=2e-14,
            )
    added_x = x.new_tensor([[0.25, 7.0, 0.45]] if multitask else [[0.25, 0.45]])
    added_y = x.new_tensor([[2.8]])
    transformed_noise = x.new_tensor([[0.04]])
    conditioned = [
        m.condition_on_observations(added_x, added_y, noise=transformed_noise) for m in models
    ]
    compare(*conditioned)
    if not multitask:
        # Native acquisition handles must observe the same affine law, not a
        # Python-only rescaling of posterior moment attributes.
        acqs = [LogExpectedImprovement(m, best_f=x.new_tensor(2.8)) for m in models]
        queries = [query[:1].clone().requires_grad_() for _ in models]
        results = [
            (a(z), torch.autograd.grad(a(z).sum(), z)[0])
            for a, z in zip(acqs, queries, strict=True)
        ]
        for actual, expected in zip(results[0], results[1], strict=True):
            torch.testing.assert_close(actual, expected, atol=3e-10, rtol=2e-8)


@pytest.mark.parametrize("multitask", [False, True])
def test_native_outcome_chain_matches_official_supported_gaussian_chain(multitask):
    pytest.importorskip("botorch")
    pytest.importorskip("jax")
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP as Reference
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as ReferenceMT
    from botorch.models.transforms.outcome import ChainedOutcomeTransform as ReferenceChain
    from botorch.models.transforms.outcome import Standardize as ReferenceStandardize

    x, y, noise = _data("cpu")
    x, y = x if multitask else x[:, [0, 2]], y * 2.3 + 1.7
    samples = _samples(x)
    if not multitask:
        samples = {
            k: v for k, v in samples.items() if k in {"mean", "lengthscale", "outputscale", "noise"}
        }
        samples["mean"] = samples["mean"][:, 0]
    kwargs = {"task_feature": 1, "rank": 2} if multitask else {}
    own = (SaasFullyBayesianMultiTaskGP if multitask else SaasFullyBayesianSingleTaskGP)(
        x,
        y,
        train_Yvar=noise,
        outcome_transform=ChainedOutcomeTransform(first=Standardize(1), second=Standardize(1)),
        backend="native",
        **kwargs,
    )
    reference = (ReferenceMT if multitask else Reference)(
        x,
        y,
        train_Yvar=noise,
        outcome_transform=ReferenceChain(
            first=ReferenceStandardize(1), second=ReferenceStandardize(1)
        ),
        **kwargs,
    )
    for m in (own, reference):
        m.load_mcmc_samples(samples)
        m.eval()
    query = x.new_tensor([[0.2, 0.7], [0.6, 0.4]])
    actual, expected = (
        own.posterior(query, observation_noise=True),
        reference.posterior(query, observation_noise=True),
    )
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-10, rtol=2e-9)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-10,
        rtol=2e-9,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_explicit_standardize_like_module_preserves_custom_fit_identity_and_graph(device):
    from botorch_rs.models._torch_backend_modules import TorchStandardize

    class CustomFit(TorchStandardize):
        def __init__(self, like):
            super().__init__(1, (), like)
            self.custom_scale = torch.nn.Parameter(like.new_tensor(1.7))
            self.fit_calls = 0

        def fit(self, Y):
            self.fit_calls += 1
            self.means = Y.new_full((1, 1), 0.25)
            self.stdvs = self.custom_scale.expand(1, 1)
            self._is_trained = True
            return self

    x, y, noise = _data(device)
    x = x[:, [0, 2]]
    transform = CustomFit(y)
    model = SingleTaskGP(x, y, noise, outcome_transform=transform, backend="torch")
    assert model.outcome_transform is transform and transform.fit_calls > 0
    baseline = SingleTaskGP(
        x,
        (y - 0.25) / transform.custom_scale,
        noise / transform.custom_scale.square(),
        outcome_transform=None,
        backend="torch",
    )
    query = x.new_tensor([[0.2, 0.7], [0.6, 0.4]], requires_grad=True)
    actual, expected = model.posterior(query), baseline.posterior(query)
    expected_mean = expected.mean * transform.custom_scale + 0.25
    expected_cov = expected.distribution.covariance_matrix * transform.custom_scale.square()
    torch.testing.assert_close(actual.mean, expected_mean, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected_cov, atol=2e-12, rtol=2e-11
    )
    targets = (query, transform.custom_scale)
    gradients = []
    for mean, covariance in (
        (actual.mean, actual.distribution.covariance_matrix),
        (expected_mean, expected_cov),
    ):
        first = torch.autograd.grad(
            mean.square().sum() + covariance.square().sum(), targets, create_graph=True
        )
        second = torch.autograd.grad(sum(t.square().sum() for t in first), targets)
        gradients.append((*first, *second))
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)


def test_custom_external_standardize_overrides_are_not_silently_canonicalized():
    pytest.importorskip("botorch")
    from botorch.models.transforms.outcome import Standardize as ReferenceStandardize

    from botorch_rs.exceptions.errors import UnsupportedError

    class Changed(ReferenceStandardize):
        def forward(self, Y, Yvar=None, X=None):
            raise AssertionError("An override must not be replaced with canonical statistics.")

    x, y, noise = _data("cpu")
    with pytest.raises(UnsupportedError, match="overridden behavior is not canonicalized"):
        SingleTaskGP(x[:, [0, 2]], y, noise, outcome_transform=Changed(1), backend="torch")
