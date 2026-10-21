"""Task-aware public loader hooks retain actual device modules and joint laws."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("jax")
pytest.importorskip("numpyro")
from botorch.models.fully_bayesian_multitask import (
    MultitaskSaasPyroModel as ReferenceProgram,
)
from botorch.models.fully_bayesian_multitask import (
    SaasFullyBayesianMultiTaskGP as ReferenceModel,
)
from botorch.models.transforms.input import Warp as ReferenceWarp
from fully_bayesian_helpers import jax_x64
from test_torch_fully_bayesian_multitask import DEVICES, _data, _samples

from botorch_rs.models.fully_bayesian_multitask import (
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)


class RbfHook:
    def _get_covar_module(self, use_scale_kernel, batch_shape, *, dtype, device):
        assert dtype == self.train_X.dtype and device == self.train_X.device
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.ard_num_dims, batch_shape=batch_shape)
        if use_scale_kernel:
            kernel = gpytorch.kernels.ScaleKernel(kernel, batch_shape=batch_shape)
        self.data_kernel = kernel.to(dtype=dtype, device=device)
        return self.data_kernel

    def load_mcmc_samples(self, *, mcmc_samples):
        self.loaded = super().load_mcmc_samples(mcmc_samples=mcmc_samples)
        return self.loaded


class Program(RbfHook, MultitaskSaasPyroModel):
    def get_dummy_mcmc_samples(self, num_mcmc_samples, **kwargs):
        self.dummy_calls = getattr(self, "dummy_calls", 0) + 1
        return super().get_dummy_mcmc_samples(num_mcmc_samples, **kwargs)


class SourceProgram(RbfHook, ReferenceProgram):
    pass


def _models(device, feature, fixed=False, warp=False, gradients=False):
    X, Y, noise = _data(device, feature, gradients=gradients)
    program = Program(use_input_warping=warp, eps=0.03)
    model = SaasFullyBayesianMultiTaskGP(
        X, Y, feature, noise if fixed else None, pyro_model=program, backend="torch"
    )
    samples = _samples(X)
    model.load_mcmc_samples(samples)
    source_program = SourceProgram(use_input_warping=False, eps=0.03)
    with jax_x64():
        source = ReferenceModel(
            X.detach(),
            Y.detach(),
            feature,
            noise.detach() if fixed else None,
            pyro_model=source_program,
        )
        source.load_mcmc_samples(samples)
    if warp:
        # The source wrapper discards the sampled Warp (and builds it with
        # the wrong physical width). Correct only that previously verified defect.
        indices = [i for i in range(3) if i != feature]
        transform = ReferenceWarp(
            3,
            indices,
            batch_shape=torch.Size([3]),
            bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            eps=0.03,
        )
        transform.concentration0.data = samples["c0"].unsqueeze(-2).clone()
        transform.concentration1.data = samples["c1"].unsqueeze(-2).clone()
        source.input_transform = transform
    return model.eval(), source.eval(), (X, Y, noise), program, samples


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("feature", [0, 1, 2])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("warp", [False, True])
def test_task_loader_hooks_match_joint_reference_and_keep_full_graphs(device, feature, fixed, warp):
    model, source, data, program, _ = _models(device, feature, fixed, warp, gradients=True)
    assert model.mean_module is program.loaded[0]
    assert model.covar_module is program.loaded[1] and model.likelihood is program.loaded[2]
    assert model.covar_module.kernels[0].base_kernel is program.data_kernel
    X, Y, noise = data
    query = X.new_tensor(
        [[[0.22, 0.43], [0.57, 0.71]], [[0.31, 0.5], [0.69, 0.6]]], requires_grad=True
    )
    actual = model.posterior(query, observation_noise=True)
    with gpytorch.settings.detach_test_caches(False):
        expected = source.posterior(query, observation_noise=True)
        expected_gradient = torch.autograd.grad(
            expected.mean.sum() + expected.variance.sum(), query
        )[0]
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-12,
        rtol=3e-11,
    )
    gradient = torch.autograd.grad(
        actual.mean.sum() + actual.variance.sum(), query, create_graph=True
    )[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=3e-10, rtol=3e-9)
    targets = (X, Y, query, *((noise,) if fixed else ()), *model.parameters())
    posterior = model.posterior(query)
    first = torch.autograd.grad(
        posterior.mean.sin().sum() + posterior.distribution.covariance_matrix.square().sum(),
        targets,
        create_graph=True,
    )
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(torch.isfinite(g).all()) for g in (*first, *second))
    assert actual.mean.device == X.device


@pytest.mark.parametrize("device", DEVICES)
def test_task_loader_no_host_copy_and_failed_noise_load_preserve_original_modules(
    device, monkeypatch
):
    model, _, data, program, samples = _models(device, 1, fixed=True, gradients=True)
    X, _, _ = data
    expected = model.posterior(X[:2]).mean.detach().clone()
    original = (model.mean_module, model.covar_module, model.likelihood)
    state = copy.deepcopy(model.state_dict())
    broken = {**samples, "lengthscale": samples["lengthscale"][:, :1]}
    with pytest.raises((ValueError, RuntimeError)):
        model.load_mcmc_samples(broken)
    assert original == (model.mean_module, model.covar_module, model.likelihood)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(model.posterior(X[:2]).mean, expected, atol=0, rtol=0)

    # Reloading a task-aware Tensor hook does not flatten training data or
    # sampled parameter arrays through a host numerical representation.
    def reject(*args, **kwargs):
        raise AssertionError("unexpected host data conversion")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, reject)
        model.load_mcmc_samples(samples)
        query = X.new_tensor([[0.2, 0.4], [0.6, 0.5]], requires_grad=True)
        posterior = model.posterior(query)
        (first,) = torch.autograd.grad(posterior.mean.sum(), query, create_graph=True)
        (second,) = torch.autograd.grad(first.sum(), query)
    assert bool(torch.isfinite(second).all())


@pytest.mark.parametrize("device", DEVICES)
def test_task_loader_state_uses_actual_dummy_hook_and_median_data_lengthscale(device):
    model, _, data, program, samples = _models(device, 1)
    X, _, _ = data
    expected = model.posterior(X[:2]).mean.detach().clone()
    torch.testing.assert_close(
        model.median_lengthscale, samples["lengthscale"].median(0).values, atol=1e-15, rtol=1e-15
    )
    model.load_state_dict(copy.deepcopy(model.state_dict()))
    assert program.dummy_calls == 1
    torch.testing.assert_close(model.posterior(X[:2]).mean, expected, atol=0, rtol=0)


def _dense_task_oracle(model, query, observation_noise=False):
    """Independent RBF × categorical-factor Gaussian conditioning."""
    train = model._train_X
    query = query.unsqueeze(-3)
    feature = model._task_feature
    columns = [i for i in range(train.shape[-1]) if i != feature]
    train_tasks = model._map_tasks(train[..., feature]).long()
    query_tasks = model._map_tasks(query[..., feature]).long()
    data = model.covar_module.kernels[0].base_kernel
    base = data.base_kernel
    task = model.covar_module.kernels[1]
    task_cov = task.covar_factor @ task.covar_factor.transpose(-1, -2) + torch.diag_embed(task.var)
    means = torch.stack([m.constant for m in model.mean_module.base_means], -1)

    def blocks(left, right, left_tasks, right_tasks):
        delta = left[..., columns].unsqueeze(-2) - right[..., columns].unsqueeze(-3)
        covariance = (
            data.outputscale[..., None, None]
            * (-0.5 * (delta / base.lengthscale.unsqueeze(-2)).square().sum(-1)).exp()
        )
        # Advanced indexing is simpler after explicit broadcast of the task
        # labels, and retains the independent factor/variance parameter graph.
        batch = torch.broadcast_shapes(covariance.shape[:-2], task_cov.shape[:-2])
        tc = task_cov.expand(*batch, 2, 2)
        lt = left_tasks.expand(*batch, left.shape[-2])
        rt = right_tasks.expand(*batch, right.shape[-2])
        selected = tc.gather(-2, lt.unsqueeze(-1).expand(*batch, left.shape[-2], 2))
        selected = selected.gather(
            -1, rt.unsqueeze(-2).expand(*batch, left.shape[-2], right.shape[-2])
        )
        return covariance * selected

    def mean_for(tasks):
        batch = torch.broadcast_shapes(tasks.shape[:-1], means.shape[:-1])
        return means.expand(*batch, 2).gather(-1, tasks.expand(*batch, tasks.shape[-1]))

    K = blocks(train, query, train_tasks, query_tasks)
    observed = blocks(train, train, train_tasks, train_tasks)
    noise = model.likelihood.noise
    noise_module = getattr(model.likelihood, "noise_covar", None)
    correlated = getattr(noise_module, "covariance", None)
    observed = observed + (
        correlated(train)
        if correlated is not None
        else torch.diag_embed(noise.expand(*observed.shape[:-1]))
    )
    residual = model._train_Y.squeeze(-1) - mean_for(train_tasks)
    mean = mean_for(query_tasks) + (
        K.transpose(-1, -2) @ torch.linalg.solve(observed, residual.unsqueeze(-1))
    ).squeeze(-1)
    covariance = blocks(query, query, query_tasks, query_tasks) - K.transpose(
        -1, -2
    ) @ torch.linalg.solve(observed, K)
    if observation_noise:
        covariance = covariance + (
            correlated(query)
            if correlated is not None
            else torch.diag_embed(noise.expand(*covariance.shape[:-1]))
        )
    return mean.unsqueeze(-1), covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_task_loader_conditioning_and_all_graphs_match_independent_gaussian(device, fixed):
    model, _, (X, Y, noise), _, _ = _models(device, 1, fixed=fixed, gradients=True)
    query = X.new_tensor([[0.21, 3.0, 0.41], [0.65, 7.0, 0.63]], requires_grad=True)
    extra_X = X.new_tensor([[0.4, 3.0, 0.5]], requires_grad=True)
    extra_Y = Y.new_tensor([[0.7]], requires_grad=True)
    extra_noise = Y.new_tensor([[0.04]], requires_grad=True) if fixed else None
    conditioned = model.condition_on_observations(extra_X, extra_Y, noise=extra_noise)
    assert conditioned.pyro_model is not model.pyro_model
    assert conditioned.pyro_model.data_kernel is conditioned.covar_module.kernels[0].base_kernel
    assert model._train_X.shape[-2] == 6 and conditioned._train_X.shape[-2] == 7
    # No second MCMC axis is introduced by a conditioning batch.
    actual = conditioned.posterior(query)
    expected = _dense_task_oracle(conditioned, query)
    torch.testing.assert_close(actual.mean, expected[0], atol=4e-13, rtol=4e-12)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected[1], atol=4e-13, rtol=4e-12
    )
    targets = (
        X,
        Y,
        query,
        extra_X,
        extra_Y,
        *((noise, extra_noise) if fixed else ()),
        *conditioned.parameters(),
    )

    def derivatives(mean, covariance):
        first = torch.autograd.grad(
            mean.sin().sum() + covariance.square().sum(), targets, create_graph=True
        )
        return (
            *first,
            *torch.autograd.grad(sum(g.square().sum() for g in first), targets, retain_graph=True),
        )

    for actual_grad, expected_grad in zip(
        derivatives(actual.mean, actual.distribution.covariance_matrix),
        derivatives(*expected),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_source_conditioning_copies_program_and_keeps_copied_module_aliases(device):
    model, source, (X, Y, _), _, _ = _models(device, 1)
    query = X.new_tensor([[0.21, 3.0, 0.41], [0.65, 7.0, 0.63]])
    added_X, added_Y = X.new_tensor([[0.4, 3.0, 0.5]]), Y.new_tensor([[0.7]])
    source.posterior(query)
    for parent, index in ((source, False), (model, True)):
        conditioned = parent.condition_on_observations(added_X, added_Y)
        assert conditioned.pyro_model is not parent.pyro_model
        kernel = conditioned.covar_module.kernels[0]
        if index:
            kernel = kernel.base_kernel
        assert conditioned.pyro_model.data_kernel is kernel
        assert {id(p) for p in conditioned.parameters()}.isdisjoint(
            id(p) for p in parent.parameters()
        )


@pytest.mark.parametrize("device", DEVICES)
def test_task_loader_preserves_full_correlated_noise_and_all_gaussian_graphs(device):
    from test_torch_fully_bayesian_likelihood import CorrelatedLikelihood, InputNoise

    class TaskNoise(InputNoise):
        def __init__(self, like, count):
            super().__init__(like)
            self.raw_diagonal = torch.nn.Parameter(like.new_full((count, 1), 0.2))
            self.amplitude = torch.nn.Parameter(like.new_full((count, 1), 0.3))

    class NoiseProgram(Program):
        def _likelihood(self, mcmc_samples, batch_shape):
            self.actual_likelihood = CorrelatedLikelihood(TaskNoise(self.train_X, batch_shape[0]))
            return self.actual_likelihood

    X, Y, _ = _data(device, 1, gradients=True)
    program = NoiseProgram()
    model = SaasFullyBayesianMultiTaskGP(X, Y, 1, pyro_model=program, backend="torch")
    model.load_mcmc_samples(_samples(X))
    assert model.likelihood is program.actual_likelihood
    query = X.new_tensor([[0.22, 3.0, 0.41], [0.63, 7.0, 0.71]], requires_grad=True)
    actual = model.posterior(query, observation_noise=True)
    expected = _dense_task_oracle(model, query, observation_noise=True)
    torch.testing.assert_close(actual.mean, expected[0], atol=4e-13, rtol=4e-12)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected[1], atol=4e-13, rtol=4e-12
    )
    targets = (X, Y, query, *model.parameters())

    def derivatives(mean, covariance):
        first = torch.autograd.grad(
            mean.sin().sum() + covariance.square().sum(), targets, create_graph=True
        )
        return (*first, *torch.autograd.grad(sum(g.square().sum() for g in first), targets))

    for actual_grad, expected_grad in zip(
        derivatives(actual.mean, actual.distribution.covariance_matrix),
        derivatives(*expected),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_real_custom_task_program_fits_and_adopts_actual_returned_modules(device):
    from botorch_rs.fit import fit_fully_bayesian_model_nuts

    class CustomPrior(Program):
        def sample_task_lengthscale(self, concentration=6.0, rate=3.0):
            return super().sample_task_lengthscale(concentration=7.0, rate=rate)

    X, Y, noise = _data(device, 1, gradients=True)
    program = CustomPrior()
    model = SaasFullyBayesianMultiTaskGP(X, Y, 1, noise, pyro_model=program, backend="torch")
    with jax_x64():
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=1,
            num_samples=2,
            thinning=1,
            max_tree_depth=1,
            seed=14,
            disable_progbar=True,
        )
    assert model.num_mcmc_samples == 2 and model._numpyro_result["backend"] == "numpyro"
    assert model.covar_module is program.loaded[1] and model.likelihood is program.loaded[2]
    query = X.new_tensor([[0.22, 3.0, 0.41], [0.63, 7.0, 0.71]], requires_grad=True)
    posterior = model.posterior(query)
    targets = (X, Y, noise, query, *model.parameters())
    first = torch.autograd.grad(
        posterior.mean.sin().sum() + posterior.variance.sum(), targets, create_graph=True
    )
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_actual_task_mean_covariance_and_likelihood_hook_objects_are_adopted(device):
    class ActualHooks:
        def _build_mean_module(self, mcmc_samples, batch_shape, **kwargs):
            self.actual_mean = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(batch_shape=batch_shape), self.num_tasks
            ).to(**kwargs)
            for i, mean in enumerate(self.actual_mean.base_means):
                mean.constant = mcmc_samples["mean"][:, i].detach()
            return self.actual_mean

        def _build_multitask_covariance(self, mcmc_samples, covar_module, batch_shape, **kwargs):
            columns = [i for i in range(self.train_X.shape[-1]) if i != self.task_feature]
            covar_module.active_dims = torch.tensor(columns, device=kwargs["device"])
            task = gpytorch.kernels.IndexKernel(
                self.num_tasks,
                rank=self.num_tasks,
                batch_shape=batch_shape,
                active_dims=[self.task_feature],
            ).to(**kwargs)
            # A caller may return a different task factor, not only the
            # builtin sampled latent-feature covariance.
            factor = mcmc_samples["latent_features"].detach()
            task.covar_factor = torch.nn.Parameter(factor)
            task.var = torch.full_like(task.var, 0.2)
            self.actual_covariance = covar_module * task
            return self.actual_covariance

        def _likelihood(self, mcmc_samples, batch_shape):
            self.actual_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=batch_shape
            ).to(self.train_X)
            self.actual_likelihood.noise = mcmc_samples["noise"].unsqueeze(-1)
            return self.actual_likelihood

    class Port(ActualHooks, Program):
        pass

    class Source(ActualHooks, SourceProgram):
        pass

    X, Y, _ = _data(device, 1, gradients=True)
    program = Port()
    model = SaasFullyBayesianMultiTaskGP(X, Y, 1, pyro_model=program, backend="torch")
    samples = _samples(X)
    model.load_mcmc_samples(samples)
    with jax_x64():
        reference = ReferenceModel(X.detach(), Y.detach(), 1, pyro_model=Source())
        reference.load_mcmc_samples(samples)
    assert model.mean_module is program.actual_mean
    assert model.covar_module is program.actual_covariance
    assert model.likelihood is program.actual_likelihood
    query = X.new_tensor([[0.22, 0.41], [0.63, 0.71]], requires_grad=True)
    actual = model.posterior(query, observation_noise=True)
    expected = reference.posterior(query, observation_noise=True)
    torch.testing.assert_close(actual.mean, expected.mean, atol=5e-13, rtol=5e-12)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=5e-13,
        rtol=5e-12,
    )
    targets = (X, Y, query, *model.parameters())
    first = torch.autograd.grad(
        actual.mean.sin().sum() + actual.variance.sum(), targets, create_graph=True
    )
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second))
