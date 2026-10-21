"""Device-resident exact GPs using the Rust Torch-operation dispatcher."""

from __future__ import annotations

import copy
from typing import Any

import torch

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior, independent_covariance
from botorch_rs.utils.types import DEFAULT


def _mutation_version(tensor: Any) -> Any:
    try:
        return tensor._version
    except RuntimeError:
        return None


class TorchSingleTaskGP(torch.nn.Module, SingleTaskGP):
    """An exact GP with ordinary Torch autograd through every Tensor operation.

    Rust dispatches kernels and Gaussian conditioning to the installed Torch
    runtime. No input, intermediate moment, or gradient is copied to a CPU
    numerical backend. ``SingleTaskGP(..., backend="torch")`` selects this
    implementation; CUDA inputs select it automatically.
    """

    _tensor_backend = "torch"
    _train_X: Any
    _train_Y: Any
    _train_Yvar: Any
    _noise_override: Any
    _transform_fit_X: Any
    _transform_fit_Y: Any
    input_transform: Any
    outcome_transform: Any
    likelihood: Any
    covar_module: Any
    mean_module: Any

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        likelihood: Any = None,
        covar_module: Any = None,
        mean_module: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        from botorch_rs.models._torch_backend_modules import (
            build_input_transform,
            build_kernel,
            build_likelihood,
            build_mean,
            build_outcome_transform,
        )

        torch.nn.Module.__init__(self)
        self._prediction_cache: Any = None
        self._transform_fit_versions: dict[int, Any] = {}
        self._validate_data(train_X, train_Y, train_Yvar)
        self._output_count = train_Y.shape[-1]
        self._is_custom_likelihood = likelihood is not None
        self.register_buffer("_train_X", train_X)
        self.register_buffer("_train_Y", train_Y)
        self.register_buffer("_train_Yvar", train_Yvar)
        self.register_buffer("_noise_override", None)
        # Sources for fitted transform statistics stay unchanged when adding
        # fantasies. Recompute their graph on every evaluation, allowing
        # independent backward calls without retaining a stale fit graph.
        self.register_buffer("_transform_fit_X", train_X)
        self.register_buffer("_transform_fit_Y", train_Y)
        self.input_transform = build_input_transform(input_transform, train_X.shape[-1], train_X)
        self.outcome_transform = build_outcome_transform(
            outcome_transform, self.num_outputs, train_Y
        )
        _, _, noise = self._training_values(clamp_noise=False)
        self.covar_module = build_kernel(
            covar_module, train_X.shape[-1], self._aug_batch_shape, train_X
        )
        self.mean_module = build_mean(mean_module, self._aug_batch_shape, train_X)
        self.likelihood = build_likelihood(likelihood, noise, self._aug_batch_shape, train_X)

    @staticmethod
    def _validate_data(X: Any, Y: Any, Yvar: Any = None) -> None:
        if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
            raise TypeError("The Torch backend requires Tensor training inputs and targets.")
        if X.ndim < 2 or Y.ndim != X.ndim or X.shape[:-1] != Y.shape[:-1]:
            raise ValueError("Expected train_X (..., n, d) and train_Y (..., n, m).")
        if not X.shape[-2] or not X.shape[-1] or not Y.shape[-1]:
            raise ValueError("Training data dimensions must be nonempty.")
        if X.dtype not in (torch.float32, torch.float64) or Y.dtype != X.dtype:
            raise TypeError("Training inputs and targets must share float32 or float64 dtype.")
        if X.device != Y.device:
            raise ValueError("Training inputs and targets must share a device.")
        if Yvar is not None:
            if (
                not isinstance(Yvar, torch.Tensor)
                or Yvar.device != X.device
                or Yvar.dtype != X.dtype
            ):
                raise TypeError("train_Yvar must be a Tensor with the training dtype and device.")
            if torch.broadcast_shapes(Y.shape, Yvar.shape) != Y.shape:
                raise ValueError("train_Yvar must broadcast to train_Y.shape.")
            if bool((Yvar < 0).any()):
                raise ValueError("Observation variances must be nonnegative.")
        if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(Y).all()):
            raise ValueError("Training inputs and targets must be finite.")
        if Yvar is not None and not bool(torch.isfinite(Yvar).all()):
            raise ValueError("Observation variances must be finite.")

    @property
    def num_outputs(self) -> int:
        return self._output_count

    @property
    def _num_outputs(self) -> int:
        return self.num_outputs

    @property
    def batch_shape(self) -> Any:
        return self._train_Y.shape[:-2]

    @property
    def _input_batch_shape(self) -> Any:
        return self.batch_shape

    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape + ((self.num_outputs,) if self.num_outputs > 1 else ())

    def _augment_X(self, X: Any) -> Any:
        return (
            X.unsqueeze(-3).expand(*X.shape[:-2], self.num_outputs, *X.shape[-2:])
            if self.num_outputs > 1
            else X
        )

    def _augment_Y(self, Y: Any) -> Any:
        return Y.transpose(-1, -2) if self.num_outputs > 1 else Y.squeeze(-1)

    def _fit_transform_graphs(self) -> None:
        from botorch_rs.models._pathwise_context import training_outcome_transform

        for transform, values in (
            (self.input_transform, self._transform_fit_X),
            (training_outcome_transform(self), self._transform_fit_Y),
        ):
            if transform is None:
                continue
            if getattr(transform, "_tensor_backend", None) == "torch":
                parameters = tuple(transform.parameters())
                versions = tuple(_mutation_version(tensor) for tensor in (values, *parameters))
                key = (
                    id(values),
                    versions[0],
                    values.shape,
                    values.dtype,
                    values.device,
                    tuple(
                        (id(parameter), version)
                        for parameter, version in zip(parameters, versions[1:], strict=True)
                    ),
                )
                needs_graph = torch.is_grad_enabled() and (
                    values.requires_grad or any(parameter.requires_grad for parameter in parameters)
                )
                if (
                    needs_graph
                    or None in versions
                    or self._transform_fit_versions.get(id(transform)) != key
                ):
                    if getattr(transform, "transform_on_train", True):
                        fit = getattr(transform, "fit", None)
                        if fit is not None:
                            fit(values)
                    self._transform_fit_versions[id(transform)] = key
            transform.eval()

    def _prediction_key(self) -> Any:
        if type(self).forward is not TorchSingleTaskGP.forward:
            # A virtual prior may depend on unregistered caller state. Preserve
            # its full graph rather than reusing a stale training factor.
            return None
        tensors = (*self.parameters(), *self.buffers())
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
            return None
        versions = []
        for tensor in tensors:
            try:
                version = tensor._version
            except RuntimeError:
                # Inference-mode tensors do not expose mutation counters.
                return None
            versions.append(
                (id(tensor), tensor.data_ptr(), version, tensor.shape, tensor.dtype, tensor.device)
            )
        configuration = []
        for name, module in self.named_modules():
            if (
                module is not self
                and not isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict))
                and type(module).__module__ != "botorch_rs.models._torch_backend_modules"
            ):
                # User modules may depend on unregistered tensors, Python
                # containers, or external state. Recompute their training
                # factor rather than assume parameter counters cover it.
                return None
            values = tuple(
                (key, value)
                for key, value in vars(module).items()
                if (
                    key != "_prediction_cache"
                    and isinstance(value, (str, int, float, bool, type(None)))
                )
            )
            configuration.append((name, id(module), values))
        return torch.is_inference_mode_enabled(), tuple(versions), tuple(configuration)

    def train(self, mode: bool = True) -> Any:
        if mode:
            self._prediction_cache = None
        return torch.nn.Module.train(self, mode)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        self._prediction_cache = None
        self._transform_fit_versions = {}
        return torch.nn.Module._apply(self, fn, recurse)

    def _training_values(self, clamp_noise: bool = True) -> tuple[Any, Any, Any]:
        from botorch_rs.models._pathwise_context import training_outcome_transform

        self._fit_transform_graphs()
        X = self._train_X
        if self.input_transform is not None:
            transform = getattr(self.input_transform, "preprocess_transform", self.input_transform)
            X = transform(X)
        Y, noise = self._train_Y, self._train_Yvar
        outcome = training_outcome_transform(self)
        if outcome is not None:
            Y, noise = outcome(Y, noise, X=self._train_X)
        if noise is not None and clamp_noise:
            noise = noise.clamp_min(1e-4 if noise.dtype == torch.float32 else 1e-6)
        if getattr(self, "_is_custom_likelihood", False):
            # As in SingleTaskGP, a supplied likelihood owns observation
            # noise even when constructor Yvar was also supplied.
            noise = None
        return (
            self._augment_X(X),
            self._augment_Y(Y),
            None if noise is None else self._augment_Y(noise.expand_as(Y)),
        )

    @property
    def train_inputs(self) -> tuple[Any]:
        return (self._augment_X(self._train_X),)

    @property
    def train_targets(self) -> Any:
        return self._training_values()[1]

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        transform = self.input_transform if input_transform is None else input_transform
        return X if transform is None else transform(X)

    def _training_noise(self, Y: Any, fixed_noise: Any) -> Any:
        noise = fixed_noise if fixed_noise is not None else self._noise_for(Y.shape[-1])
        if self._noise_override is not None:
            noise = torch.where(self._noise_override.isnan(), noise, self._noise_override)
        return noise

    def _noise_for(self, count: int) -> Any:
        method = getattr(self.likelihood, "noise_for", None)
        if method is not None:
            return method(count)
        noise = self.likelihood.noise
        return noise.expand(*noise.shape[:-1], count)

    def _covariance(self, X1: Any, X2: Any) -> Any:
        value = self.covar_module(X1, X2)
        # GPyTorch kernel modules and user kernels may expose lazy covariance
        # objects. Densifying them retains their Torch graph and device.
        return value if isinstance(value, torch.Tensor) else value.to_dense()

    def _prior_blocks(
        self, train_X: Any, query_X: Any, *, need_training: bool = True
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Prior Gaussian blocks on already transformed independent-output inputs.

        The default path keeps blockwise kernel evaluation. A caller's forward
        override defines the actual prior, including cross-covariances, so use
        its joint Gaussian just as ExactGP does.
        """
        if type(self).forward is TorchSingleTaskGP.forward:
            return (
                self.mean_module(train_X) if need_training else None,
                self._covariance(train_X, train_X) if need_training else None,
                self._covariance(train_X, query_X),
                self.mean_module(query_X),
                self._covariance(query_X, query_X),
            )
        train_prior: Any = torch.nn.Module.__call__(self, train_X) if need_training else None
        batch = torch.broadcast_shapes(train_X.shape[:-2], query_X.shape[:-2])
        joint_X = torch.cat(
            (
                train_X.expand(*batch, *train_X.shape[-2:]),
                query_X.expand(*batch, *query_X.shape[-2:]),
            ),
            dim=-2,
        )
        prior = torch.nn.Module.__call__(self, joint_X)
        if any(
            not isinstance(getattr(distribution, attribute, None), torch.Tensor)
            for distribution in ((train_prior, prior) if need_training else (prior,))
            for attribute in ("mean", "covariance_matrix")
        ):
            raise TypeError(
                "forward must return a Gaussian with Tensor mean and covariance_matrix."
            )
        n = train_X.shape[-2]
        covariance = prior.covariance_matrix
        return (
            train_prior.mean if need_training else None,
            train_prior.covariance_matrix if need_training else None,
            covariance[..., :n, n:],
            prior.mean[..., n:],
            covariance[..., n:, n:],
        )

    def _posterior_tensors(
        self,
        X: Any,
        observation_noise: Any = False,
        *,
        inputs_transformed: bool = False,
    ) -> Any:
        from botorch_rs.models._torch_gaussian_likelihood import is_fixed_noise, uses_marginal

        if uses_marginal(self.likelihood) or (
            getattr(self, "_is_custom_likelihood", False) and is_fixed_noise(self.likelihood)
        ):
            from botorch_rs.models._torch_fully_bayesian_likelihood import posterior_tensors

            return posterior_tensors(
                self, X, observation_noise, inputs_transformed=inputs_transformed
            )
        if (
            not isinstance(X, torch.Tensor)
            or X.ndim < 2
            or (not inputs_transformed and X.shape[-1] != self._train_X.shape[-1])
        ):
            raise ValueError("X must be a Tensor with shape (..., q, d).")
        if X.dtype != self._train_X.dtype or X.device != self._train_X.device:
            raise ValueError("X must share the model's dtype and device; use model.to(X).")
        self.eval()
        train_X, Y, fixed_noise = self._training_values()
        if inputs_transformed and X.shape[-1] != train_X.shape[-1]:
            raise ValueError("Evaluation inputs must use the transformed feature dimension.")
        query_X = X if inputs_transformed else self._augment_X(self.transform_inputs(X))
        train_noise = self._training_noise(Y, fixed_noise)
        query_noise = None
        if isinstance(observation_noise, torch.Tensor):
            query_noise = self._augment_Y(observation_noise)
        elif observation_noise is True:
            query_noise = (
                fixed_noise.mean(-1, keepdim=True)
                if fixed_noise is not None
                else self._noise_for(query_X.shape[-2])
            )
        elif observation_noise is not False:
            raise TypeError("observation_noise must be bool or a Tensor.")
        key = self._prediction_key()
        cache = self._prediction_cache
        cached = key is not None and cache is not None and cache[0] == key
        train_mean, train_covar, cross_covar, query_mean, query_covar = self._prior_blocks(
            train_X, query_X, need_training=not cached
        )
        if cached:
            factor, alpha = cache[1:]
        else:
            factor, alpha = _rs.tensor_exact_gp_factor(Y, train_covar, train_mean, train_noise)
            if key is not None:
                self._prediction_cache = (key, factor, alpha)
        mean, covariance = _rs.tensor_exact_gp_predict(
            factor,
            alpha,
            cross_covar,
            query_covar,
            query_mean,
            observation_noise=query_noise,
        )
        return mean, covariance

    def posterior(
        self,
        X: Any,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported posterior arguments: {sorted(kwargs)}.")
        mean, covariance = self._posterior_tensors(X, observation_noise)
        if self.num_outputs == 1:
            mean, blocks = mean.unsqueeze(-1), covariance.unsqueeze(-3)
        else:
            mean, blocks = mean.transpose(-1, -2), covariance
        posterior = TorchGPyTorchPosterior(mean, independent_covariance(blocks))
        outcome = self.outcome_transform
        if output_indices is not None:
            indices = torch.as_tensor(output_indices, device=mean.device, dtype=torch.long)
            if indices.ndim != 1 or not indices.numel():
                raise ValueError("output_indices must be a nonempty one-dimensional sequence.")
            if bool(((indices < -self.num_outputs) | (indices >= self.num_outputs)).any()):
                raise IndexError("output_indices contains an out-of-range output.")
            indices = indices.remainder(self.num_outputs)
            if outcome is not None:
                subset = getattr(outcome, "subset_output", None)
                if subset is None:
                    method = getattr(outcome, "untransform_posterior", None)
                    if method is not None:
                        posterior = method(posterior, X=X)
                    else:
                        mean, blocks = outcome.untransform_moments(mean, blocks)
                        posterior = TorchGPyTorchPosterior(mean, independent_covariance(blocks))
                    if not isinstance(posterior, TorchGPyTorchPosterior):
                        raise UnsupportedError(
                            "This outcome transform must implement subset_output for selecting nonlinear outputs."
                        )
                    outcome = None
                else:
                    outcome = subset(indices)
            posterior = posterior.subset_output(indices)
        if outcome is not None:
            method = getattr(outcome, "untransform_posterior", None)
            if method is not None:
                posterior = method(posterior, X=X)
            else:
                mean, blocks = outcome.untransform_moments(mean, blocks)
                posterior = TorchGPyTorchPosterior(mean, independent_covariance(blocks))
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def __call__(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        if self.training:
            return torch.nn.Module.__call__(self, X, *args, **kwargs)
        if args or kwargs:
            raise UnsupportedError(
                "Evaluation calls accept one input Tensor; use posterior for posterior options."
            )
        mean, covariance = self._posterior_tensors(X, inputs_transformed=True)
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution

    def forward(self, X: Any) -> Any:
        """The latent prior; evaluation-mode calls condition through __call__."""
        if self.training:
            self._fit_transform_graphs()
        if self.training and self.input_transform is not None:
            transform = getattr(self.input_transform, "preprocess_transform", self.input_transform)
            original = (
                X.select(-3, 0) if self.num_outputs > 1 and X.ndim > self._train_X.ndim else X
            )
            X = self._augment_X(transform(original))
        # The latent training law is not factored until noise is added by
        # the MLL or likelihood. Noiseless RBF matrices can be singular.
        return TorchGPyTorchPosterior(
            self.mean_module(X).unsqueeze(-1), self._covariance(X, X)
        ).distribution

    def _log_prior(self) -> Any:
        from botorch_rs.models._torch_backend_modules import module_log_prior

        result = self._train_X.new_zeros(())
        seen: set[int] = set()
        for module in (self.likelihood, self.mean_module, self.covar_module, self.input_transform):
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                result = result + module_log_prior(module, self._aug_batch_shape)
        return result

    def _exact_mll(
        self, output: Any = None, target: Any = None, *params: Any, **kwargs: Any
    ) -> Any:
        from botorch_rs.models._torch_gaussian_likelihood import gaussian_marginal, uses_marginal

        if output is None:
            X, Y, fixed_noise = self._training_values()
            if type(self).forward is TorchSingleTaskGP.forward:
                mean, covariance = self.mean_module(X), self._covariance(X, X)
            else:
                prior = torch.nn.Module.__call__(self, self.train_inputs[0] if self.training else X)
                mean, covariance = prior.mean, prior.covariance_matrix
        else:
            X, _, fixed_noise = self._training_values()
            Y = self.train_targets if target is None else target
            mean, covariance = output.mean, output.covariance_matrix
        if uses_marginal(self.likelihood):
            mean, covariance = gaussian_marginal(
                self.likelihood,
                mean,
                covariance,
                X,
                training=True,
                params=params or None,
                kwargs=kwargs,
            )
            noise = torch.zeros_like(mean)
        else:
            if params or kwargs:
                raise TypeError(
                    "Additional likelihood arguments require a supplied Gaussian marginal."
                )
            noise = self._training_noise(Y, fixed_noise)
        value = _rs.tensor_exact_mll(Y, covariance, mean, noise, normalize=False)
        return (value + self._log_prior()) / Y.shape[-1]

    def set_train_data(self, inputs: Any = None, targets: Any = None, strict: bool = True) -> None:
        X = self._train_X if inputs is None else inputs[0] if isinstance(inputs, tuple) else inputs
        Y = self._train_Y if targets is None else targets
        if Y.ndim == X.ndim - 1 and self.num_outputs == 1:
            Y = Y.unsqueeze(-1)
        self._validate_data(X, Y, self._train_Yvar)
        if Y.shape[-1] != self.num_outputs:
            raise ValueError("set_train_data cannot change the number of outputs.")
        if strict and (X.shape != self._train_X.shape or Y.shape != self._train_Y.shape):
            raise RuntimeError("Cannot modify the training data shape with strict=True.")
        self._train_X, self._train_Y = X, Y
        self._transform_fit_X, self._transform_fit_Y = X, Y

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        if getattr(self, "_is_custom_likelihood", False):
            from botorch_rs.models._torch_gaussian_likelihood import (
                condition_fixed_likelihood,
                is_fixed_noise,
            )

            if is_fixed_noise(self.likelihood):
                return condition_fixed_likelihood(self, X, Y, noise, **kwargs)
        if kwargs:
            raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
        if X.shape[-2] != Y.shape[-2] or Y.shape[-1] != self.num_outputs:
            raise ValueError("Conditioning X and Y dimensions must match the model.")
        batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2], Y.shape[:-2])
        model = copy.copy(self)
        model._buffers = self._buffers.copy()
        model._modules = self._modules.copy()
        model._train_X = torch.cat(
            (
                self._train_X.expand(*batch, *self._train_X.shape[-2:]),
                X.expand(*batch, *X.shape[-2:]),
            ),
            dim=-2,
        )
        model._train_Y = torch.cat(
            (
                self._train_Y.expand(*batch, *self._train_Y.shape[-2:]),
                Y.expand(*batch, *Y.shape[-2:]),
            ),
            dim=-2,
        )
        if self._train_Yvar is not None:
            if noise is None:
                noise = self._train_Yvar.mean(-2, keepdim=True).expand_as(Y)
            elif self.outcome_transform is not None:
                # Conditioning noise follows BoTorch's transformed-space API.
                noise = self.outcome_transform.untransform(torch.zeros_like(Y), noise)[1]
            model._train_Yvar = torch.cat(
                (
                    self._train_Yvar.expand(*batch, *self._train_Y.shape[-2:]),
                    noise.expand(*batch, *Y.shape[-2:]),
                ),
                dim=-2,
            )
        elif noise is not None or self._noise_override is not None:
            previous = self._noise_override
            if previous is None:
                previous = torch.full_like(self._augment_Y(self._train_Y), torch.nan)
            extra = (
                torch.full_like(self._augment_Y(Y), torch.nan)
                if noise is None
                else self._augment_Y(noise)
            )
            augmented_batch = batch + ((self.num_outputs,) if self.num_outputs > 1 else ())
            model._noise_override = torch.cat(
                (
                    previous.expand(*augmented_batch, previous.shape[-1]),
                    extra.expand(*augmented_batch, Y.shape[-2]),
                ),
                dim=-1,
            )
        return model.eval()

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
        propagate = kwargs.pop("propagate_grads", False)
        if kwargs:
            raise UnsupportedError(f"Unsupported fantasy arguments: {sorted(kwargs)}.")
        if isinstance(observation_noise, bool):
            raise TypeError("fantasize observation_noise must be None or a Tensor.")
        if observation_noise is None and getattr(self, "_is_custom_likelihood", False):
            from botorch_rs.models._torch_gaussian_likelihood import is_fixed_noise

            if is_fixed_noise(self.likelihood):
                observation_noise = self.likelihood.noise
                observation_noise = (
                    observation_noise.transpose(-1, -2)
                    if self.num_outputs > 1
                    else observation_noise.unsqueeze(-1)
                ).mean(-2, keepdim=True)
        posterior = self.posterior(
            X, observation_noise=True if observation_noise is None else observation_noise
        )
        Y = sampler(posterior)
        if not propagate:
            Y = Y.detach()
        return self.condition_on_observations(X, Y, noise=observation_noise)

    def _sync_torch_parameters(self) -> Any:
        # Parameters already are the tensors consumed by the inference graph.
        return None

    def subset_output(self, idcs: Any) -> Any:
        indices = torch.as_tensor(idcs, dtype=torch.long, device=self._train_Y.device)
        if indices.ndim != 1 or not indices.numel():
            raise ValueError("Output indices must be a nonempty vector.")
        if bool(((indices < -self.num_outputs) | (indices >= self.num_outputs)).any()):
            raise IndexError("Output index out of range.")
        indices = indices.remainder(self.num_outputs)
        new = copy.deepcopy(self)
        if self.num_outputs == 1:
            if indices.numel() != 1:
                raise UnsupportedError("Repeating a single output requires a model list.")
            return new
        old_aug = self._aug_batch_shape
        axis = len(self.batch_shape)
        count = indices.numel()
        new_aug = self.batch_shape + ((count,) if count > 1 else ())
        replacements: dict[int, Any] = {}
        module: Any
        for module in new.modules():
            if module is new or tuple(getattr(module, "batch_shape", ())) != tuple(old_aug):
                continue
            for name, parameter in list(module.named_parameters(recurse=False)):
                if tuple(parameter.shape[: axis + 1]) == tuple(old_aug):
                    if id(parameter) not in replacements:
                        selected = parameter.index_select(axis, indices)
                        if count == 1:
                            selected = selected.squeeze(axis)
                        replacements[id(parameter)] = torch.nn.Parameter(
                            selected.detach().clone(), requires_grad=parameter.requires_grad
                        )
                    setattr(module, name, replacements[id(parameter)])
            if type(module).__name__ == "TorchFixedNoiseLikelihood":
                selected = module.noise.index_select(axis, indices)
                module.noise = selected.squeeze(axis) if count == 1 else selected
            module.batch_shape = torch.Size(new_aug)
        new._output_count = count
        new._train_Y = self._train_Y.index_select(-1, indices)
        new._transform_fit_Y = self._transform_fit_Y.index_select(-1, indices)
        if self._train_Yvar is not None:
            new._train_Yvar = self._train_Yvar.expand_as(self._train_Y).index_select(-1, indices)
        if self._noise_override is not None:
            selected = self._noise_override.index_select(axis, indices)
            new._noise_override = selected.squeeze(axis) if count == 1 else selected
        outcome = new.outcome_transform
        if outcome is not None:
            subset = getattr(outcome, "subset_output", None)
            if subset is None:
                raise UnsupportedError("This outcome transform does not implement subset_output.")
            new.outcome_transform = subset(indices)
        return new

    def posterior_parameter_vjp(
        self,
        X: Any,
        mean_adjoint: Any,
        covariance_adjoint: Any,
        observation_noise: bool = False,
    ) -> Any:
        posterior = self.posterior(X, observation_noise=observation_noise)
        loss = (posterior.mean * mean_adjoint).sum()
        loss = loss + (posterior.distribution.covariance_matrix * covariance_adjoint).sum()
        parameters = tuple(self.parameters())
        active = [parameter for parameter in parameters if parameter.requires_grad]
        if not active:
            return torch.cat([torch.zeros_like(parameter).reshape(-1) for parameter in parameters])
        gradients = torch.autograd.grad(
            loss, active, create_graph=torch.is_grad_enabled(), allow_unused=True
        )
        by_identity = {
            id(parameter): gradient for parameter, gradient in zip(active, gradients, strict=True)
        }
        return torch.cat(
            [
                (
                    torch.zeros_like(parameter)
                    if by_identity.get(id(parameter)) is None
                    else by_identity[id(parameter)]
                ).reshape(-1)
                for parameter in parameters
            ]
        )

    def initialize(self, **kwargs: Any) -> Any:
        for name, value in kwargs.items():
            path, _, attribute = name.rpartition(".")
            module = self.get_submodule(path) if path else self
            current = getattr(module, attribute)
            if isinstance(current, torch.nn.Parameter):
                with torch.no_grad():
                    current.copy_(
                        torch.as_tensor(value, dtype=current.dtype, device=current.device)
                    )
            else:
                setattr(module, attribute, value)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        for module in self.modules():
            for buffer in module.buffers(recurse=False):
                if not buffer.is_leaf and id(buffer) not in memo:
                    memo[id(buffer)] = buffer.clone()
        new = object.__new__(type(self))
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            if name == "_buffers":
                # Keep graph connectivity for training tensors and fantasies.
                new.__dict__[name] = value.copy()
            else:
                new.__dict__[name] = copy.deepcopy(value, memo)
        return new
