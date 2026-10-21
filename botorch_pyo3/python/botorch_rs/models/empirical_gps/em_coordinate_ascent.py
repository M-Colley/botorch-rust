"""Native Gaussian KL and kernel fitting for empirical-Bayes priors."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel
from botorch_rs.means import ConstantMean, ZeroMean
from botorch_rs.mlls import MarginalLogLikelihood


def kl_divergence_mvn(mu_p: Any, Sigma_p: Any, mu_q: Any, Sigma_q: Any) -> Any:
    """KL(p || q), with device-local full Torch graphs or native array evaluation."""
    kind = kind_of(mu_p)
    if kind.lib == "torch":
        from botorch_rs.models.empirical_gps._torch_em_helpers import gaussian_kl

        return gaussian_kl(mu_p, Sigma_p, mu_q, Sigma_q)
    a, b, c, d = (to_flat(x) for x in (mu_p, Sigma_p, mu_q, Sigma_q))
    value, _ = _rs.gaussian_kl(a, b, c, d)
    return from_flat([value], (), kind)


class _KernelFit:
    def __init__(
        self, mean_module: Any, covar_module: Kernel, X_ref: Any, mu_em: Any, Sigma_em: Any
    ) -> None:
        if not isinstance(mean_module, (ConstantMean, ZeroMean)) or not isinstance(
            covar_module, Kernel
        ):
            raise UnsupportedError(
                "KL prior fitting requires a native ConstantMean/ZeroMean and covariance kernel."
            )
        if tuple(mean_module.batch_shape) or tuple(covar_module.batch_shape):
            raise UnsupportedError(
                "KL prior fitting currently requires unbatched mean and kernel modules."
            )
        if _ops._requires_grad([X_ref]):
            raise UnsupportedError(
                "KL kernel fitting currently treats reference locations as fixed; query-location gradients are not implemented."
            )
        self.mean_module, self.covar_module = mean_module, covar_module
        self.kind = kind_of(X_ref)
        self.learn_mean = isinstance(mean_module, ConstantMean)
        constraint = getattr(mean_module, "raw_constant_constraint", None)
        self.native = _rs.KernelPriorKl(
            covar_module._node(),
            to_flat(X_ref, "X_ref"),
            to_flat(mu_em, "mu_em"),
            to_flat(Sigma_em, "Sigma_em"),
            self.learn_mean,
            None if constraint is None else constraint._rs,
        )

    def raw(self) -> list[float]:
        return (
            []
            if not isinstance(self.mean_module, ConstantMean)
            else list(to_flat(self.mean_module.raw_constant)[0])
        ) + self.covar_module._raw_flat()

    def install(self, raw: list[float]) -> None:
        offset = 0
        if isinstance(self.mean_module, ConstantMean):
            self.mean_module.raw_constant = from_flat(raw[:1], (), self.kind)
            offset = 1
        for name, value in self.covar_module.named_parameters():
            shape = to_flat(value)[1]
            size = math.prod(shape)
            module: Any = self.covar_module
            *parents, attribute = name.split(".")
            for parent in parents:
                module = getattr(module, parent)
            setattr(module, attribute, from_flat(raw[offset : offset + size], shape, self.kind))
            offset += size
        if offset != len(raw):
            raise RuntimeError("Native KL fit returned an inconsistent parameter vector.")

    def fit(self, optimizer_kwargs: Any = None, **kwargs: Any) -> dict[str, Any]:
        optimizer_kwargs = dict(optimizer_kwargs or {})
        if optimizer_kwargs.pop("method", "L-BFGS-B") != "L-BFGS-B":
            raise UnsupportedError("KL prior fitting supports native L-BFGS-B.")
        options = dict(optimizer_kwargs.pop("options", None) or {})
        unknown = set(options) - {
            "maxiter",
            "maxfun",
            "gtol",
            "ftol",
            "maxcor",
            "maxls",
            "disp",
            "iprint",
        }
        if unknown or optimizer_kwargs:
            raise UnsupportedError(
                f"Unsupported KL fit options: {sorted(unknown | set(optimizer_kwargs))}."
            )
        raw, loss, iterations, evaluations, status, message, attempts = self.native.fit(
            self.raw(),
            maxcor=options.get("maxcor"),
            pgtol=options.get("gtol"),
            ftol=options.get("ftol"),
            maxiter=options.get("maxiter"),
            maxfun=options.get("maxfun"),
            maxls=options.get("maxls"),
            max_attempts=kwargs.pop("max_attempts", 5),
            seed=kwargs.pop("seed", None),
        )
        if kwargs:
            raise UnsupportedError(f"Unsupported KL fit arguments: {sorted(kwargs)}.")
        self.install(raw)
        return {
            "loss": loss,
            "iterations": iterations,
            "evaluations": evaluations,
            "status": status,
            "message": message,
            "attempts": attempts,
        }


class KLPriorFitMLL(MarginalLogLikelihood):
    """Negative Gaussian KL at reference points; no likelihood or prior terms.

    The raw gradient follows the model's full parameter order. Likelihood noise
    and input-transform parameters have zero derivatives, as they do not enter
    this objective. Native fitting updates the mean and kernel in place.
    """

    _rs_custom_objective = True

    def __init__(self, likelihood: Any, model: Any, X_ref: Any, mu_em: Any, Sigma_em: Any) -> None:
        if getattr(model, "num_outputs", 1) != 1 or tuple(getattr(model, "batch_shape", ())):
            raise UnsupportedError(
                "KLPriorFitMLL currently requires an unbatched single-output model."
            )
        if likelihood is not model.likelihood:
            raise ValueError("The likelihood must be the model's likelihood.")
        super().__init__(likelihood, model)
        if getattr(model, "_tensor_backend", None) == "torch":
            self._X_ref = X_ref
            self._mu_em = mu_em.detach().clone()
            self._Sigma_em = Sigma_em.detach().clone()
            return
        self._X_ref = _ops.clone(X_ref)
        self._mu_em = _ops.clone(mu_em)
        self._Sigma_em = _ops.clone(Sigma_em)
        self._objective = _KernelFit(
            model.mean_module, model.covar_module, self._X_ref, self._mu_em, self._Sigma_em
        )

    def forward(self, function_dist: Any = None, target: Any = None) -> Any:
        if getattr(self.model, "_tensor_backend", None) == "torch":
            from botorch_rs.models.empirical_gps._torch_em import dense, mean_at

            return -kl_divergence_mvn(
                self._mu_em,
                self._Sigma_em,
                mean_at(self.model.mean_module, self._X_ref),
                dense(self.model.covar_module, self._X_ref, self._X_ref),
            )
        value, _ = self._objective.native.value_and_grad(self._objective.raw())
        return from_flat([value], (), self._objective.kind)

    __call__ = forward

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        if getattr(self.model, "_tensor_backend", None) == "torch":
            raise UnsupportedError(
                "The flat native KL protocol requires backend='native'; use mll() with torch.autograd.grad for Tensor models."
            )
        shapes = list(self.model._rs.parameter_shapes())
        parameters = (
            self.model._rs.raw_parameters()
            if raw_parameters is None
            else list(to_flat(raw_parameters)[0])
        )
        if len(parameters) != sum(math.prod(shape) for _, shape in shapes):
            raise ValueError("Wrong number of raw model parameters.")
        offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, shape in shapes:
            count = math.prod(shape)
            offsets[name] = (offset, count)
            offset += count
        names = (["mean_module.raw_constant"] if self._objective.learn_mean else []) + [
            "covar_module." + name for name, _ in self.model.covar_module.named_parameters()
        ]
        selected = [
            index for name in names for index in range(offsets[name][0], sum(offsets[name]))
        ]
        value, values = self._objective.native.value_and_grad(
            [parameters[index] for index in selected]
        )
        gradient = [0.0] * len(parameters)
        for index, derivative in zip(selected, values, strict=True):
            gradient[index] = derivative
        return float(value), gradient

    def _fit(
        self,
        closure: Any = None,
        optimizer: Any = None,
        closure_kwargs: Any = None,
        optimizer_kwargs: Any = None,
        **kwargs: Any,
    ) -> KLPriorFitMLL:
        from botorch_rs.fit import _fit_generic, _requires_generic_fit

        if (
            getattr(self.model, "_tensor_backend", None) == "torch"
            or _requires_generic_fit(
                self, closure, optimizer, closure_kwargs, optimizer_kwargs, kwargs
            )
            or kwargs.get("pick_best_of_all_attempts", False)
            or getattr(optimizer, "__name__", None) == "fit_gpytorch_mll_torch"
        ):
            _fit_generic(self, closure, optimizer, closure_kwargs, optimizer_kwargs, kwargs)
            return self
        self._fit_result = self._objective.fit(optimizer_kwargs, **kwargs)
        self.eval()
        return self


def fit_kernel_to_em_prior(
    mean_module: Any,
    covar_module: Kernel,
    X_ref: Any,
    mu_em: Any,
    Sigma_em: Any,
    *,
    backend: str = "auto",
) -> None:
    """Fit caller-owned mean/kernel modules by native Gaussian KL minimization."""
    from botorch_rs.models.empirical_gps.em_empirical_gp import _torch_em_requested

    if _torch_em_requested(backend, (X_ref, mu_em, Sigma_em), (mean_module, covar_module)):
        from botorch_rs.models.empirical_gps._torch_em_helpers import fit_kernel

        fit_kernel(mean_module, covar_module, X_ref, mu_em, Sigma_em)
        return
    _KernelFit(mean_module, covar_module, X_ref, mu_em, Sigma_em).fit()


def coordinate_ascent_em(
    datasets: list[Any],
    mean_module: Any,
    covar_module: Kernel,
    likelihood_noise: Any = None,
    num_em_iterations: int = 16,
    n_coord_ascent_iterations: int = 1,
    inducing_points: Any = None,
    optimize_inducing_points: bool = False,
    use_mean_prior: bool = False,
    use_covar_prior: bool = False,
    iw_nu: float | None = None,
    em_convergence_tol: float | None = 1e-6,
    enable_interpolation: bool = True,
    init_mode: str = "kernel",
    mll_max_iter: int = 50,
    mll_dataset_subsample: int | None = None,
    mll_use_adam: bool | None = None,
    mll_adam_lr: float = 0.01,
    mll_adam_steps: int = 200,
    container_history: list[Any] | None = None,
    *,
    backend: str = "auto",
) -> Any:
    """Alternate exact native EM with native fixed-moment kernel fitting."""
    import random
    from dataclasses import replace

    from botorch_rs.models.empirical_gps.em_empirical_gp import _torch_em_requested

    if _torch_em_requested(
        backend,
        [value for data in datasets for value in (data.X, data.Y)]
        + [inducing_points, likelihood_noise],
        (mean_module, covar_module),
    ):
        from botorch_rs.models.empirical_gps._torch_em_helpers import coordinate_ascent

        return coordinate_ascent(
            datasets,
            mean_module,
            covar_module,
            likelihood_noise,
            num_em_iterations,
            n_coord_ascent_iterations,
            inducing_points,
            optimize_inducing_points,
            use_mean_prior,
            use_covar_prior,
            iw_nu,
            em_convergence_tol,
            enable_interpolation,
            init_mode,
            mll_max_iter,
            mll_dataset_subsample,
            mll_use_adam,
            mll_adam_lr,
            mll_adam_steps,
            container_history,
        )

    from botorch_rs.constraints import GreaterThan
    from botorch_rs.likelihoods import GaussianLikelihood
    from botorch_rs.models.empirical_gps.em_empirical_gp import (
        EMEmpiricalGaussianProcess,
        _copy_modules,
        pretrain_em_prior,
    )

    if optimize_inducing_points:
        raise UnsupportedError("Coordinate ascent currently requires fixed inducing locations.")
    if not isinstance(n_coord_ascent_iterations, int) or n_coord_ascent_iterations < 0:
        raise ValueError("n_coord_ascent_iterations must be a nonnegative integer.")
    if mll_dataset_subsample is not None and (
        not isinstance(mll_dataset_subsample, int) or mll_dataset_subsample <= 0
    ):
        raise ValueError("mll_dataset_subsample must be a positive integer.")
    if (
        not isinstance(mll_max_iter, int)
        or mll_max_iter < 0
        or not isinstance(mll_adam_steps, int)
        or mll_adam_steps < 1
        or not math.isfinite(mll_adam_lr)
        or mll_adam_lr <= 0
    ):
        raise ValueError("Coordinate fit iteration counts and learning rate are invalid.")
    options = dict(
        likelihood_noise=likelihood_noise,
        inducing_points=inducing_points,
        num_em_iterations=num_em_iterations,
        use_mean_prior=use_mean_prior,
        use_covar_prior=use_covar_prior,
        iw_nu=iw_nu,
        em_convergence_tol=em_convergence_tol,
        enable_interpolation=enable_interpolation,
        init_mode=init_mode,
    )
    prior = pretrain_em_prior(datasets, mean_module, covar_module, **options)
    if container_history is not None:
        container_history.append(_copy_modules(prior)[0])
    for _ in range(n_coord_ascent_iterations):
        kind = kind_of(prior.X_inducing)
        noise = (
            None if likelihood_noise is None else from_flat(to_flat(likelihood_noise)[0], (), kind)
        )
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-16, transform=None, initial_value=noise)
        )
        selected = datasets
        if mll_dataset_subsample is not None and mll_dataset_subsample < len(datasets):
            try:
                import torch
            except ImportError:
                indices = random.sample(range(len(datasets)), mll_dataset_subsample)
            else:
                indices = torch.randperm(len(datasets))[:mll_dataset_subsample].tolist()
            selected = [datasets[i] for i in indices]
        fitting_prior = replace(prior, datasets=selected)
        X = _ops.getitem(prior.X_inducing, slice(0, 1))
        Y = from_flat([0.0], (1, 1), kind)
        model = EMEmpiricalGaussianProcess.from_pretrained(
            fitting_prior, X, Y, likelihood=likelihood, freeze_pretrained=False
        )
        mask = [False] + [True] * (len(model._rs.raw_parameters()) - 1)
        if mll_use_adam:
            model._rs.fit_torch(mask=mask, step_limit=mll_adam_steps, stopping=None, lr=mll_adam_lr)
        else:
            model._rs.fit(mask=mask, maxiter=mll_max_iter)
        if isinstance(mean_module, ConstantMean):
            mean_module.raw_constant = model.initial_mean_module.raw_constant
        for name, value in model.initial_covar_module.named_parameters():
            target: Any = covar_module
            *parents, attribute = name.split(".")
            for parent in parents:
                target = getattr(target, parent)
            setattr(target, attribute, value)
        prior = pretrain_em_prior(datasets, mean_module, covar_module, **options)
        if container_history is not None:
            container_history.append(_copy_modules(prior)[0])
    return prior


__all__ = ["KLPriorFitMLL", "coordinate_ascent_em", "fit_kernel_to_em_prior", "kl_divergence_mvn"]
