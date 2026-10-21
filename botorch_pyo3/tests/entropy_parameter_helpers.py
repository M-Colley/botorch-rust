"""Independent source models for conditioned entropy parameter derivatives."""

from types import MethodType

import torch


def fresh_source_model(model):
    """Recompute an augmented exact GP with separate, shape-matched leaves.

    GPyTorch's incremental fantasy caches retain the original model's graph.
    A fresh exact GP gives the partial derivatives of the actual child's
    parameters independently of that disposable cache graph.
    """
    from botorch.models import SingleTaskGP

    result = SingleTaskGP(model._train_X, model._train_Y, outcome_transform=None)
    for name, parameter in model.named_parameters():
        module_name, attribute = name.rsplit(".", 1)
        module = result.get_submodule(module_name)
        setattr(module, attribute, torch.nn.Parameter(parameter.detach().clone()))
    result.posterior = MethodType(dense_source_posterior, result)
    return result.eval()


def dense_source_samples(self, sample_shape, base_samples):
    """Use the dense Cholesky root, avoiding the known generic operator VJP."""
    mean = self.mean
    root = torch.linalg.cholesky(self.distribution.covariance_matrix)
    flat = base_samples.reshape(*sample_shape, *mean.shape[:-2], -1)
    samples = mean.flatten(-2) + (root @ flat.unsqueeze(-1)).squeeze(-1)
    return samples.reshape(*sample_shape, *mean.shape)


def dense_source_posterior(model, X, observation_noise=False, posterior_transform=None):
    """Exact Gaussian law from the source prior, without operator VJP shortcuts.

    The pinned source RBF cache and concatenated independent-output operators
    have separately reproduced higher-backward defects (#2821 / #153). Source
    prior evaluation and dense Cholesky algebra isolate the entropy derivatives
    from those known defects while retaining independent source parameter leaves.
    """
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

    inputs, targets = model.train_inputs[0], model.train_targets
    outputs = model.num_outputs
    points = X if outputs == 1 else X.unsqueeze(-3)
    batch = torch.broadcast_shapes(inputs.shape[:-2], points.shape[:-2])
    inputs = inputs.expand(*batch, *inputs.shape[-2:])
    points = points.expand(*batch, *points.shape[-2:])
    prior = model.forward(torch.cat((inputs, points), -2))
    covariance, n = prior.covariance_matrix, inputs.shape[-2]
    noise = model.likelihood.noise
    root = torch.linalg.cholesky(
        covariance[..., :n, :n] + torch.diag_embed(noise.expand(*batch, n))
    )
    cross = covariance[..., n:, :n]
    mean = prior.mean[..., n:] + (
        cross @ torch.cholesky_solve((targets - prior.mean[..., :n]).unsqueeze(-1), root)
    ).squeeze(-1)
    conditional = covariance[..., n:, n:] - cross @ torch.cholesky_solve(
        cross.transpose(-1, -2), root
    )
    if observation_noise:
        conditional = conditional + torch.diag_embed(noise.expand(*batch, X.shape[-2]))
    if outputs == 1:
        distribution = MultivariateNormal(mean, conditional)
    else:
        q = X.shape[-2]
        dense = torch.einsum(
            "...mij,mn->...imjn", conditional, torch.eye(outputs, dtype=X.dtype, device=X.device)
        ).reshape(*conditional.shape[:-3], q * outputs, q * outputs)
        distribution = MultitaskMultivariateNormal(mean.transpose(-1, -2), dense)
    posterior = GPyTorchPosterior(distribution)
    posterior.rsample_from_base_samples = MethodType(dense_source_samples, posterior)
    return posterior if posterior_transform is None else posterior_transform(posterior)


def entropy_leaves(acquisition, query, joint):
    """Name the sole unused initial mean and strictly retain all active leaves."""
    model = getattr(acquisition, "initial_model", acquisition.model)
    parameters = dict(sorted(model.named_parameters()))
    if joint:
        mean = parameters.pop("mean_module.raw_constant")
        child = dict(sorted(acquisition.conditional_model.named_parameters()))
        assert child.keys() == dict(model.named_parameters()).keys()
        assert all(child[name] is not value for name, value in model.named_parameters())
        return (query, *parameters.values(), *child.values()), mean
    return (query, *parameters.values()), None


def assert_initial_mean_unused(value, mean):
    if mean is not None:
        # Deliberately permit None only for this named Gaussian-entropy mean.
        assert torch.autograd.grad(value.sum(), mean, allow_unused=True, retain_graph=True) == (
            None,
        )


def gradients_and_mixed_hvp(value, variables):
    first = torch.autograd.grad(value.sum(), variables, create_graph=True)
    directions = tuple(
        torch.linspace(0.3, 0.9, item.numel(), dtype=item.dtype, device=item.device).reshape_as(
            item
        )
        for item in variables
    )
    second = torch.autograd.grad(
        sum(
            (gradient * direction).sum()
            for gradient, direction in zip(first, directions, strict=False)
        ),
        variables,
    )
    assert all(torch.isfinite(item).all() for item in (*first, *second))
    return first, second
