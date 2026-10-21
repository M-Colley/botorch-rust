# ruff: noqa: E402
"""Structured fantasy ownership works without the upstream GP runtime."""

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models import HigherOrderGP, KroneckerMultiTaskGP, LatentKroneckerGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher", "latent"])
def test_default_structured_conditioning_owns_parameters_and_fantasy_query_graph(device, family):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device, requires_grad=True)
    T = X.new_tensor([[0.0], [0.6]])
    Y = torch.sin(4 * X.detach() + T.squeeze(-1)).requires_grad_()
    cls = {"kron": KroneckerMultiTaskGP, "higher": HigherOrderGP, "latent": LatentKroneckerGP}[
        family
    ]
    args = (X, T, Y) if family == "latent" else (X, Y)
    model = cls(*args, backend="torch", outcome_transform=None)
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    original_mean = model.posterior(Q).mean.detach().clone()
    conditioned = model.condition_on_observations(Q, Y[:2])
    original, updated = dict(model.named_parameters()), dict(conditioned.named_parameters())
    assert original.keys() == updated.keys()
    assert all(original[name] is not updated[name] and updated[name].is_leaf for name in original)
    with torch.no_grad():
        for value in updated.values():
            value.add_(0.15)
    torch.testing.assert_close(model.posterior(Q).mean, original_mean)
    for propagate in (False, True):
        future = model.fantasize(
            Q, lambda posterior: posterior.mean.unsqueeze(0), propagate_grads=propagate
        )
        values = future._train_Y[..., -Q.shape[-2] :, :]
        assert values.requires_grad
        gradient = torch.autograd.grad(values.sum(), Q, create_graph=True, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        hessian = torch.autograd.grad(gradient.sum(), Q, retain_graph=True)[0]
        assert torch.isfinite(hessian).all()
