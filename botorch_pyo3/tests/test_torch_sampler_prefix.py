# ruff: noqa: E402
"""Device-only common normals for baseline/joint posterior sampling."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
def test_tensor_sampler_reuses_prefix_without_host_copy(
    device, outputs, sampler_class, monkeypatch
):
    X = torch.tensor([[0.05], [0.35], [0.72], [0.95]], device=device, dtype=torch.double)
    Y = torch.tensor([[0.12], [0.62], [-0.17], [0.31]], device=device, dtype=torch.double)
    X = X.repeat(2, 1, 1)
    Y = torch.cat([Y + 0.2 * j for j in range(outputs)], -1).repeat(2, 1, 1)
    model = SingleTaskGP(X, Y, backend="torch")
    baseline = X[:, ::2]
    query = torch.tensor(
        [[[0.21], [0.56]], [[0.18], [0.81]], [[0.44], [0.88]]], device=device, dtype=torch.double
    ).unsqueeze(-3)
    baseline_sampler = sampler_class((3, 4), seed=11)
    joint_sampler = sampler_class((3, 4), seed=19)

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampler prefix update copied data to host")

    for method in ("cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    base_posterior = model.posterior(baseline)
    baseline_sampler._construct_base_samples(base_posterior)
    base_samples = baseline_sampler(base_posterior)
    for count in (1, 2):
        candidate = query[..., :count, :]
        full = torch.cat((baseline.expand(3, 2, 2, 1), candidate.expand(3, 2, count, 1)), dim=-2)
        posterior = model.posterior(full)
        joint_sampler._update_base_samples(posterior, baseline_sampler)
        result = joint_sampler(posterior)
        expected = base_samples.unsqueeze(2).expand(3, 4, 3, 2, 2, outputs)
        torch.testing.assert_close(result[..., :2, :], expected, rtol=1e-10, atol=1e-11)
        assert result.device == X.device
