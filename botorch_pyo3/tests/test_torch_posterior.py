"""Explicit PyTorch distribution adapters, including nested metadata."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys

import pytest

from botorch_rs.posteriors import TorchPosterior


def test_importing_torch_posterior_keeps_optional_libraries_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from botorch_rs.posteriors import TorchPosterior; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
        ],
        check=True,
    )


def test_distribution_forwarding_shapes_and_no_grad() -> None:
    torch = pytest.importorskip("torch")
    loc = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    distribution = torch.distributions.Normal(loc, torch.ones(2, dtype=torch.float64))
    posterior = TorchPosterior(distribution)
    assert posterior.distribution is distribution
    assert posterior.dtype == torch.float64 and posterior.device == loc.device
    torch.testing.assert_close(posterior.mean, distribution.mean)
    torch.testing.assert_close(posterior.variance, distribution.variance)
    torch.testing.assert_close(posterior.cdf(loc), distribution.cdf(loc))
    assert posterior._extended_shape((2, 3)) == torch.Size([2, 3, 2])
    assert posterior.rsample().shape == torch.Size([2])
    assert posterior.rsample((2, 3)).shape == torch.Size([2, 3, 2])
    assert posterior.rsample().requires_grad
    assert not posterior.sample().requires_grad
    assert not posterior.sample((2,)).requires_grad
    torch.manual_seed(19)
    actual = posterior.rsample((3,))
    torch.manual_seed(19)
    torch.testing.assert_close(actual, distribution.rsample((3,)), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("shape", [(), (1,), (3,), (2, 3)])
def test_quantile_and_density_match_distribution(shape: tuple[int, ...]) -> None:
    torch = pytest.importorskip("torch")
    distribution = torch.distributions.Normal(
        torch.tensor([0.0, 1.0], dtype=torch.float64), torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    posterior = TorchPosterior(distribution)
    value = torch.full(shape, 0.3, dtype=torch.float64, requires_grad=True)
    quantile = posterior.quantile(value)
    density = posterior.density(value)
    expected_shape = (*shape, 2) if value.numel() > 1 else (2,)
    assert tuple(quantile.shape) == expected_shape
    assert tuple(density.shape) == expected_shape
    expected_q = distribution.icdf(torch.tensor(0.3, dtype=torch.float64))
    expected_d = distribution.log_prob(torch.tensor(0.3, dtype=torch.float64)).exp()
    torch.testing.assert_close(quantile, expected_q.expand(expected_shape))
    torch.testing.assert_close(density, expected_d.expand(expected_shape))
    assert torch.isfinite(torch.autograd.grad(quantile.sum(), value, retain_graph=True)[0]).all()
    assert torch.isfinite(torch.autograd.grad(density.sum(), value)[0]).all()


@pytest.mark.parametrize("wrapper", ["independent", "transformed"])
def test_nested_distribution_metadata(wrapper: str) -> None:
    torch = pytest.importorskip("torch")
    base = torch.distributions.Normal(
        torch.zeros(2, 3, dtype=torch.float64), torch.ones(2, 3, dtype=torch.float64)
    )
    if wrapper == "independent":
        distribution = torch.distributions.Independent(base, 1)
    else:
        distribution = torch.distributions.TransformedDistribution(
            base, [torch.distributions.ExpTransform()]
        )
    posterior = TorchPosterior(distribution)
    assert posterior.dtype == torch.float64
    assert posterior.device == torch.device("cpu")
    assert posterior.rsample((4,)).shape == (4, 2, 3)


def test_discrete_density_is_probability_mass() -> None:
    torch = pytest.importorskip("torch")
    posterior = TorchPosterior(
        torch.distributions.Bernoulli(probs=torch.tensor(0.3, dtype=torch.float64))
    )
    torch.testing.assert_close(
        posterior.density(torch.tensor([0.0, 1.0], dtype=torch.float64)),
        torch.tensor([0.7, 0.3], dtype=torch.float64),
    )
    with pytest.raises(NotImplementedError):
        posterior.rsample()


def test_pickle_and_deepcopy() -> None:
    torch = pytest.importorskip("torch")
    original = TorchPosterior(
        torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
            ),
            1,
        )
    )
    for posterior in [pickle.loads(pickle.dumps(original)), copy.deepcopy(original)]:
        assert posterior.distribution is not original.distribution
        torch.testing.assert_close(posterior.mean, original.mean)
        assert posterior.dtype == original.dtype and posterior.device == original.device
        assert posterior.rsample((3,)).shape == (3, 2)
    uninitialized = TorchPosterior.__new__(TorchPosterior)
    with pytest.raises(AttributeError):
        _ = uninitialized.distribution


def test_nested_metadata_containers_and_cycles() -> None:
    torch = pytest.importorskip("torch")

    class NestedDistribution(torch.distributions.Distribution):
        arg_constraints: dict = {}

        def __init__(self) -> None:
            self.metadata = {"cycle": self, "nested": [torch.tensor(0.0, dtype=torch.float64)]}
            super().__init__()

    posterior = TorchPosterior(NestedDistribution())
    assert posterior.dtype == torch.float64 and posterior.device == torch.device("cpu")
    with pytest.raises(TypeError, match="no tensor parameters"):
        TorchPosterior(torch.distributions.Distribution(validate_args=False))


def test_sample_detaches_when_custom_posterior_imports_torch_lazily() -> None:
    pytest.importorskip("torch")
    script = """
import sys
from botorch_rs.posteriors import Posterior
assert 'torch' not in sys.modules
class LazyPosterior(Posterior):
    def rsample(self, sample_shape=None):
        import torch
        return 2 * torch.tensor(1.0, requires_grad=True)
posterior = LazyPosterior()
assert not posterior.sample().requires_grad
assert posterior.rsample().requires_grad
"""
    subprocess.run([sys.executable, "-c", script], check=True)
