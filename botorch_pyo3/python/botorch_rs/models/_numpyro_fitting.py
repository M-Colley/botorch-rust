"""Optional inference for explicit NumPyro probabilistic extensions.

Standard model families retain their native or Tensor NUTS implementation.
Only caller-defined probabilistic hooks and prior-mode programs use this
adapter, which evaluates the port's own program with NumPyro/JAX.
"""

from __future__ import annotations

import copy
import sys
from importlib import import_module
from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError

_BUILTIN_MODULES = {
    "botorch_rs.models.fully_bayesian",
    "botorch_rs.models.fully_bayesian_multitask",
}
_PROGRAM_HOOKS = (
    "set_inputs",
    "sample",
    "sample_mean",
    "sample_noise",
    "sample_concentrations",
    "sample_lengthscale",
    "sample_outputscale",
    "sample_weight_variance",
    "sample_observations",
    "sample_latent_features",
    "sample_task_lengthscale",
    "_prepare_features",
    "_maybe_input_warp",
    "_warp_jax",
    "_maybe_multitask_transform",
    "_build_task_covar",
    "_get_task_indices_and_base_idxr_jax",
    "postprocess_mcmc_samples",
)


def requires_numpyro_inference(model: Any) -> bool:
    """Recognize explicit custom programs without importing optional dependencies."""
    pyro = getattr(model, "pyro_model", None)
    return pyro is not None and (
        bool(getattr(pyro, "_prior_mode", False))
        or any(
            (method := getattr(pyro, name, None)) is not None
            and getattr(method, "__module__", "") not in _BUILTIN_MODULES
            for name in _PROGRAM_HOOKS
        )
    )


def _copy_module_shell(module: Any) -> Any:
    """Isolate Python state without invoking caller pickle/copy hooks."""
    result = object.__new__(type(module))
    object.__setattr__(result, "__dict__", module.__dict__.copy())
    for base in type(module).__mro__:
        slots = getattr(base, "__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in ("__dict__", "__weakref__") and hasattr(module, name):
                object.__setattr__(result, name, getattr(module, name))
    return result


def _copy_transform(module: Any) -> Any:
    """Copy mutable buffers while retaining explicit user parameter identities."""
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(module, torch.nn.Module):
        return copy.deepcopy(module)
    result = _copy_module_shell(module)
    result._parameters = module._parameters.copy()
    result._buffers = {
        name: None if value is None else value.detach().clone()
        for name, value in module._buffers.items()
    }
    result._modules = {
        name: None if child is None else _copy_transform(child)
        for name, child in module._modules.items()
    }
    return result


def _candidate(model: Any) -> Any:
    candidate = (
        _copy_module_shell(model)
        if getattr(model, "_tensor_backend", None) == "torch"
        else copy.copy(model)
    )
    if getattr(model, "_tensor_backend", None) == "torch":
        candidate._modules = model._modules.copy()
    candidate.pyro_model = _copy_module_shell(model.pyro_model)
    candidate.pyro_model._trace_inputs = {}
    candidate.pyro_model._rs = None
    if getattr(model, "_tensor_backend", None) == "torch":
        candidate._parameters = model._parameters.copy()
        candidate._buffers = model._buffers.copy()
        user_transform = _copy_transform(model._fb_user_input_transform)
        object.__setattr__(candidate, "_fb_user_input_transform", user_transform)
        candidate.input_transform = user_transform
        candidate.outcome_transform = _copy_transform(model.outcome_transform)
        candidate._transform_fit_versions = {}
        candidate._prediction_cache = None
        candidate._mcmc_count, candidate._initializing = None, True
        X, Y, noise = candidate._training_values()
        if X.ndim != 2 or Y.ndim != 1:
            raise UnsupportedError("Custom NumPyro fitting requires unbatched training data.")
        kwargs: dict[str, Any] = {}
        if hasattr(candidate.pyro_model, "task_feature"):
            X = candidate._mapped_inputs(X)
            kwargs.update(
                task_feature=candidate._task_feature,
                task_rank=candidate._rank,
                all_tasks=list(range(candidate.num_tasks)),
            )
        candidate.pyro_model.set_inputs(
            train_X=X,
            train_Y=Y.unsqueeze(-1),
            train_Yvar=None if noise is None else noise.unsqueeze(-1),
            **kwargs,
        )
        candidate._initializing = False
        from botorch_rs.models._torch_fully_bayesian_loader import has_custom_loader

        custom_loader = has_custom_loader(candidate.pyro_model)
        if (
            candidate.pyro_model._rs_kernel not in ("matern", "saas", "linear")
            and not custom_loader
        ):
            raise UnsupportedError(
                "Custom Tensor NumPyro loading requires a declared Matern, SAAS or Linear family."
            )
    return candidate


def _rebind(value: Any, previous: Any, current: Any, seen: set[int]) -> None:
    from botorch_rs._module import GPModule

    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, GPModule):
        if getattr(value, "_model", None) is previous:
            value._model = current
        for name, child in vars(value).items():
            if name != "_model":
                _rebind(child, previous, current, seen)
    elif isinstance(value, dict):
        for child in value.values():
            _rebind(child, previous, current, seen)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _rebind(child, previous, current, seen)


def _adopt_transform(original: Any, staged: Any) -> None:
    """Commit fitted buffers while preserving caller-held module identities."""
    children = {
        name: None if child is None else original._modules[name]
        for name, child in staged._modules.items()
    }
    for name, child in staged._modules.items():
        if child is not None:
            _adopt_transform(children[name], child)
    state = staged.__dict__.copy()
    state["_modules"] = children
    original.__dict__.clear()
    original.__dict__.update(state)


def _commit_transform_identities(model: Any, candidate: Any) -> None:
    if getattr(model, "_tensor_backend", None) != "torch":
        return
    for original, staged in (
        (model._fb_user_input_transform, candidate._fb_user_input_transform),
        (model.outcome_transform, candidate.outcome_transform),
    ):
        if original is None:
            continue
        _adopt_transform(original, staged)
        for parent in candidate.modules():
            for name, child in parent._modules.items():
                if child is staged:
                    parent._modules[name] = original
        if candidate._fb_user_input_transform is staged:
            object.__setattr__(candidate, "_fb_user_input_transform", original)


def fit_numpyro_model_nuts(
    model: Any,
    max_tree_depth: int,
    warmup_steps: int,
    num_samples: int,
    thinning: int,
    disable_progbar: bool,
    jit_compile: bool,
    seed: int,
    draws: Any = None,
    init_params: Any = None,
) -> None:
    """Fit the port's custom program, committing a successfully loaded law.

    JAX controls the inference device and precision, including its usual
    ``jax_enable_x64`` setting. Data cross the framework boundary explicitly;
    MCMC inference is not a Torch differentiation operation. Supported loaded
    Tensor models retain their ordinary differentiable prediction graphs.
    """
    if (
        any(
            type(value) is not int
            for value in (max_tree_depth, warmup_steps, num_samples, thinning)
        )
        or max_tree_depth < 1
        or warmup_steps < 0
        or num_samples < 1
        or thinning < 1
    ):
        raise ValueError("NUTS needs positive depth, samples, thinning and nonnegative warmup.")
    if draws is not None or init_params is not None:
        raise UnsupportedError("Native/Tensor recorded-draw replay is not a NumPyro bridge option.")
    import numpy as np

    from botorch_rs._ops import dtype_name

    jax, jnp = import_module("jax"), import_module("jax.numpy")
    inference = import_module("numpyro.infer")
    candidate = _candidate(model)
    pyro = candidate.pyro_model
    for name in ("train_X", "train_Y", "train_Yvar"):
        value = getattr(pyro, name)
        if value is None:
            pyro._trace_inputs[name] = None
        else:
            data, shape = to_flat(value, name)
            array = np.asarray(data, dtype=dtype_name(value)).reshape(shape)
            pyro._trace_inputs[name] = jnp.asarray(array)
    # The pinned public fitter accepts jit_compile but does not consume it.
    sampler = inference.MCMC(
        inference.NUTS(pyro.sample, dense_mass=True, max_tree_depth=max_tree_depth),
        num_warmup=warmup_steps,
        num_samples=num_samples,
        progress_bar=not disable_progbar,
    )
    sampler.run(jax.random.PRNGKey(seed))
    raw = sampler.get_samples()
    processed = pyro.postprocess_mcmc_samples(mcmc_samples=raw)
    if not isinstance(processed, dict) or not processed:
        raise TypeError("postprocess_mcmc_samples must return a nonempty sample dictionary.")
    kind = kind_of(pyro.train_X)
    samples = {}
    for name, value in processed.items():
        if type(value).__module__.startswith(("jax.", "jaxlib.")):
            array = np.asarray(value)
            data, shape = array.reshape(-1).tolist(), tuple(array.shape)
        else:
            data, shape = to_flat(value, name)
        if not shape or shape[0] != num_samples:
            raise ValueError(f"Postprocessed sample {name!r} must have {num_samples} leading rows.")
        array = np.asarray(data).reshape(shape)[::thinning]
        samples[name] = from_flat(array.reshape(-1).tolist(), array.shape, kind)
    candidate.load_mcmc_samples(samples)
    candidate.eval()
    candidate.__dict__.pop("_nuts_result", None)
    candidate._numpyro_result = {
        "backend": "numpyro",
        "raw_samples": raw,
        "jax_devices": sorted(
            {str(device) for value in raw.values() for device in value.devices()}
        ),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "num_samples": num_samples,
        "thinning": thinning,
    }
    _commit_transform_identities(model, candidate)
    original_pyro = model.pyro_model
    original_pyro.__dict__.clear()
    original_pyro.__dict__.update(pyro.__dict__)
    candidate.pyro_model = original_pyro
    _rebind(vars(candidate), candidate, model, set())
    model.__dict__.clear()
    model.__dict__.update(candidate.__dict__)


__all__ = ["requires_numpyro_inference", "fit_numpyro_model_nuts"]
