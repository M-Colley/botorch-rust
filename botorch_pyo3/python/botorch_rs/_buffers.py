"""Read native logical buffers without treating parameters or data as buffers.

Native entries are tensor snapshots, like the port's other state getters.
Explicitly registered user tensors and Torch child modules retain identity.
Torch is imported only when this tensor inspection protocol is requested.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import sys
from collections.abc import Iterator
from typing import Any, cast

from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError


def _default_dtype() -> str:
    torch = sys.modules.get("torch")
    return str(torch.get_default_dtype()).removeprefix("torch.") if torch else "float32"


def _dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype).removeprefix("torch.") if dtype is not None else None


class BufferModule:
    """Buffer inspection and explicit buffer registration for Python modules."""

    def register_buffer(self, name: str, tensor: Any, persistent: bool = True) -> None:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(self, torch.nn.Module):
            torch.nn.Module.register_buffer(self, name, tensor, persistent=persistent)
            return
        if not isinstance(name, str):
            raise TypeError("buffer name must be a string")
        if not name or "." in name:
            raise KeyError("buffer names must be nonempty and cannot contain a dot")
        registered = self.__dict__.setdefault("_registered_buffers", {})
        if hasattr(self, name) and name not in registered:
            raise KeyError(f"attribute {name!r} already exists")
        if tensor is not None and not hasattr(tensor, "dtype"):
            raise TypeError("buffers must be tensors, arrays, or None")
        if not isinstance(persistent, bool):
            raise TypeError("persistent must be a boolean")
        registered[name] = persistent
        setattr(self, name, tensor)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Any]]:
        """Yield registered tensors and supported native buffer snapshots.

        Training arrays and parameters are deliberately absent. Native values
        are read at iteration time; changing a returned snapshot does not
        replace the Rust state. Use the component's setter for that operation.
        """
        torch = importlib.import_module("torch")
        seen_modules: set[int] = set()
        seen_tensors: set[int] = set()
        seen_native: set[tuple[int, str]] = set()
        active: set[int] = set()

        def visit(module: Any, path: str) -> Iterator[tuple[str, Any]]:
            if remove_duplicate and id(module) in seen_modules:
                return
            if id(module) in active:
                raise ValueError("a cycle in the module graph prevents buffer traversal")
            seen_modules.add(id(module))
            active.add(id(module))
            if isinstance(module, torch.nn.Module):
                entries = torch.nn.Module.named_buffers(
                    module, recurse=recurse, remove_duplicate=remove_duplicate
                )
                for name, value in entries:
                    if remove_duplicate and id(value) in seen_tensors:
                        continue
                    seen_tensors.add(id(value))
                    yield _join(path, name), value
            else:
                for name in module.__dict__.get("_registered_buffers", {}):
                    value = getattr(module, name)
                    if value is None or (remove_duplicate and id(value) in seen_tensors):
                        continue
                    seen_tensors.add(id(value))
                    yield _join(path, name), value
                for name, value, dtype in _native_entries(module):
                    identity = _native_identity(module, name)
                    if remove_duplicate and identity in seen_native:
                        continue
                    seen_native.add(identity)
                    flat, shape = to_flat(value, name)
                    tensor = torch.tensor(list(flat), dtype=getattr(torch, dtype)).reshape(shape)
                    yield _join(path, name), tensor
                if recurse:
                    for name, child in _children(module):
                        yield from visit(child, _join(path, name))
            active.remove(id(module))

        yield from visit(self, prefix)

    def buffers(self, recurse: bool = True) -> Iterator[Any]:
        """Yield actual user buffers and supported native buffer snapshots."""
        for _, value in self.named_buffers(recurse=recurse):
            yield value


class NativeBufferModule(BufferModule):
    """Capture construction dtypes before native scalar adapters erase them."""

    _buffer_factory_dtype: str
    _buffer_cast_dtype: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        constructor = cls.__dict__.get("__init__")
        if constructor is None or getattr(constructor, "_buffer_aware", False):
            return
        signature = inspect.signature(constructor)

        @functools.wraps(constructor)
        def initialize(self: Any, *args: Any, **kw: Any) -> None:
            if "_buffer_factory_dtype" not in self.__dict__:
                self._buffer_factory_dtype = _default_dtype()
                bound = signature.bind_partial(self, *args, **kw)
                self._buffer_argument_dtypes = {
                    name: dtype
                    for name, value in bound.arguments.items()
                    if (dtype := _dtype(value)) is not None
                }
            constructor(self, *args, **kw)

        initialize._buffer_aware = True  # type: ignore[attr-defined]
        cls.__init__ = initialize  # type: ignore[method-assign]


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _native_identity(module: Any, name: str) -> tuple[int, str]:
    # GPyTorch ScaleKernel registers its base kernel's same active_dims tensor.
    from botorch_rs.kernels import ScaleKernel

    if name == "active_dims" and isinstance(module, ScaleKernel):
        while isinstance(module, ScaleKernel):
            module = module.base_kernel
        return id(module), name
    if type(module).__name__ == "PairwiseGP" and name.startswith("unconsolidated_"):
        key = name.removeprefix("un")
        # With no duplicate observations, official consolidation keeps the
        # original objects. With duplicates it creates new data tensors.
        original = getattr(module, name)
        consolidated = getattr(module, key)
        if to_flat(original)[1] == to_flat(consolidated)[1]:
            return id(module), key
    return id(module), name


def _children(module: Any) -> Iterator[tuple[str, Any]]:
    from botorch_rs.likelihoods import GaussianLikelihood, HadamardGaussianLikelihood

    remapped: set[str] = set()
    if isinstance(module, (GaussianLikelihood, HadamardGaussianLikelihood)):
        for attribute, name in (
            ("noise_prior", "noise_covar.noise_prior"),
            ("noise_constraint", "noise_covar.raw_noise_constraint"),
        ):
            child = getattr(module, attribute, None)
            remapped.add(attribute)
            if child is not None:
                yield name, child
    torch = sys.modules.get("torch")
    containers: set[int] = set()

    def children(value: Any, name: str) -> Iterator[tuple[str, Any]]:
        if isinstance(value, BufferModule) or (torch and isinstance(value, torch.nn.Module)):
            yield name, value
        elif isinstance(value, (tuple, list, dict)):
            if id(value) in containers:
                return
            containers.add(id(value))
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in items:
                yield from children(child, _join(name, str(key)))
            containers.remove(id(value))

    for name, value in module.__dict__.items():
        if name in remapped or name in ("_model", "_owner", "_registered_buffers"):
            continue
        # Native chains represent a ModuleDict, whose keys are the child names.
        if isinstance(value, (list, tuple, dict)) and name not in (
            "models",
            "kernels",
            "_transforms",
            "transforms",
            "_context_kernels",
        ):
            continue
        yield from children(value, "" if name == "_transforms" else name)


def adopt_buffer_dtype(module: Any, kind: Any) -> None:
    """Record the constructor's Module.to(training_tensor) buffer conversion."""
    dtype = str(kind.dtype).removeprefix("torch.") if kind.dtype is not None else None
    if dtype is None:
        return
    seen: set[int] = set()

    def visit(current: Any) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        if isinstance(current, NativeBufferModule):
            current._buffer_cast_dtype = dtype
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(current, torch.nn.Module):
            dormant_affine = [
                child
                for child in current.modules()
                if child.__dict__.get("_native_affine_dormant", False)
            ]
            current.to(dtype=getattr(torch, dtype))
            # Native model dtype adoption updates inspection buffers without
            # changing the authoritative array state or its public array kind.
            # A caller's explicit Module.to still activates live Tensor state.
            for child in dormant_affine:
                child.__dict__["_native_affine_dormant"] = True
        else:
            for name in current.__dict__.get("_registered_buffers", {}):
                value = getattr(current, name)
                if (
                    torch is not None
                    and isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                ):
                    setattr(current, name, value.to(dtype=getattr(torch, dtype)))
                elif getattr(getattr(value, "dtype", None), "kind", None) == "f":
                    setattr(current, name, value.astype(dtype))
        for _, child in _children(current):
            visit(child)

    visit(module)


def _logical_dtype(module: Any, argument: str | None = None) -> str:
    cast = module.__dict__.get("_buffer_cast_dtype")
    if cast is not None:
        return str(cast)
    arguments = module.__dict__.get("_buffer_argument_dtypes", {})
    if argument in arguments:
        return str(arguments[argument])
    kind = module.__dict__.get("_kind")
    if kind is not None and kind.dtype is not None:
        return str(kind.dtype).removeprefix("torch.")
    return str(module.__dict__.get("_buffer_factory_dtype", _default_dtype()))


def _native_entries(module: Any) -> list[tuple[str, Any, str]]:
    from botorch_rs._module import GPModule
    from botorch_rs.constraints import Interval
    from botorch_rs.kernels import Kernel
    from botorch_rs.models.model import Model
    from botorch_rs.models.transforms.input import InputTransform
    from botorch_rs.models.transforms.outcome import OutcomeTransform, Standardize
    from botorch_rs.priors import LKJCholeskyFactorPrior, Prior

    dtype = _logical_dtype(module)
    if isinstance(module, Interval):
        return [(name, getattr(module, name), dtype) for name in ("lower_bound", "upper_bound")]
    if isinstance(module, Prior):
        if isinstance(module, LKJCholeskyFactorPrior):
            return []  # GPyTorch does not register LKJ concentration tensors.
        if module._rs.kind == "Uniform":
            return []  # GPyTorch UniformPrior does not register low/high.
        names = {
            "Normal": (("loc", "loc"), ("scale", "scale")),
            "LogNormal": (("_buffered_loc", "loc"), ("_buffered_scale", "scale")),
            "Gamma": (("concentration", "concentration"), ("rate", "rate")),
            "Uniform": (("low", "a"), ("high", "b")),
            "HalfCauchy": (("_buffered_scale", "scale"),),
            "HalfNormal": (("_buffered_scale", "scale"),),
            "Beta": (
                ("_buffered_concentration1", "concentration1"),
                ("_buffered_concentration0", "concentration0"),
            ),
            "SmoothedBox": (("a", "a"), ("b", "b"), ("sigma", "sigma")),
        }[module._rs.kind]
        values = list(module._rs.parameters)
        if module._rs.kind == "Beta":
            # Torch's Beta stacks the two concentrations, promoting both to
            # the same dtype before BoTorch registers their cloned buffers.
            arguments = module.__dict__.get("_buffer_argument_dtypes", {})
            beta_dtype = "float64" if "float64" in arguments.values() else dtype
            shape = tuple(cast(Any, module).batch_shape)
            return [
                (
                    name,
                    _shaped_constant(value, shape),
                    module.__dict__.get("_buffer_cast_dtype", beta_dtype),
                )
                for (name, _), value in zip(names, values, strict=True)
            ]
        result = [
            (
                name,
                [value] if module._rs.kind == "SmoothedBox" else value,
                _logical_dtype(module, argument),
            )
            for (name, argument), value in zip(names, values, strict=True)
        ]
        if module._rs.kind == "SmoothedBox":
            result.extend((("tails.loc", [0.0], dtype), ("tails.scale", [values[2]], dtype)))
        return result
    if isinstance(module, Kernel):
        supported = {
            "CategoricalKernel",
            "DownsamplingKernel",
            "ExponentialDecayKernel",
            "InfiniteWidthBNNKernel",
            "SACKernel",
            "DeltaKernel",
            "CombinatorialCovarModule",
            "MultiTaskConditionalKernel",
            "HierarchicalConditionalKernel",
            "_ActivatedKernel",
            "PositiveIndexKernel",
        }
        if (
            type(module).__module__ != "botorch_rs.kernels"
            and type(module).__name__ not in supported
        ):
            raise UnsupportedError(
                f"Native buffer descriptors for {type(module).__name__} are not available."
            )
        dims = module.active_dims
        return [] if dims is None else [("active_dims", dims, "int64")]
    if isinstance(module, GPModule):
        return []  # Hyperparameters are parameters; children hold prior/constraint buffers.
    if isinstance(module, Standardize):
        if type(module).__name__ == "StratifiedStandardize" and not module._is_trained:
            shape = (*module._batch_shape, len(cast(Any, module)._task_values), 1)
            entries = [
                (name, _shaped_constant(fill, shape), dtype)
                for name, fill in (("means", 0.0), ("stdvs", 1.0), ("_stdvs_sq", 1.0))
            ]
            return entries + [("_is_trained", False, "bool")]
        return [
            (name, getattr(module, name), dtype) for name in ("means", "stdvs", "_stdvs_sq")
        ] + [("_is_trained", module._is_trained, "bool")]
    if isinstance(module, (InputTransform, OutcomeTransform)):
        return _transform_entries(module)
    if isinstance(module, Model):
        return _model_entries(module)
    return []


def _shaped_constant(value: float, shape: tuple[int, ...]) -> Any:
    result: Any = value
    for size in reversed(shape):
        result = [result for _ in range(size)]
    return result


def _transform_entries(module: Any) -> list[tuple[str, Any, str]]:
    from botorch_rs.models.transforms.input import AffineInputTransform

    dtype = _logical_dtype(module)
    name = type(module).__name__
    result: list[tuple[str, Any, str]] = []
    if isinstance(module, AffineInputTransform):
        if module.indices is not None:
            result.append(("indices", module.indices, "int64"))
        result.extend(
            (private, getattr(module, public), _logical_dtype(module, public))
            for private, public in (("_coefficient", "coefficient"), ("_offset", "offset"))
        )
    elif name in ("Round", "Log10", "Warp", "FilterFeatures"):
        attribute = {"Round": "integer_indices", "FilterFeatures": "feature_indices"}.get(
            name, "indices"
        )
        result.append((attribute, getattr(module, attribute), "int64"))
        if name == "Warp":
            from botorch_rs import _botorch_rs as _rs

            # These are the immutable native WarpStep::constraint bounds.
            constraint = _rs.Constraint.greater_than(1e-4, None, 1.0, True)
            for index in (0, 1):
                for bound in ("lower_bound", "upper_bound"):
                    result.append(
                        (
                            f"concentration{index}_constraint.{bound}",
                            getattr(constraint, bound),
                            dtype,
                        )
                    )
    elif name in ("AppendFeatures", "InteractionFeatures"):
        value = getattr(module, "feature_set", None)
        if value is not None:
            result.append(("feature_set", value, _logical_dtype(module, "feature_set")))
    elif name == "InputPerturbation":
        raise UnsupportedError(
            "InputPerturbation buffer inspection needs the native cached perturbations; that cache is not exposed yet."
        )
    elif name not in (
        "ChainedInputTransform",
        "ChainedOutcomeTransform",
        "BatchBroadcastedInputTransform",
        "Log",
        "Power",
        "Bilog",
        "OneHotToNumeric",
        "NumericToCategoricalEncoding",
    ):
        raise UnsupportedError(f"Native buffer descriptors for {name} are not available.")
    return result


def _model_entries(module: Any) -> list[tuple[str, Any, str]]:
    from botorch_rs.models.cost import DeterministicCostModel
    from botorch_rs.models.deterministic import DeterministicModel
    from botorch_rs.models.ensemble import EnsembleModel
    from botorch_rs.models.gp_regression import SingleTaskGP
    from botorch_rs.models.gpytorch import GPyTorchModel
    from botorch_rs.models.model import Model, ModelList

    name = type(module).__name__
    if name == "PairwiseGP":
        integer = {"consolidated_comparisons", "unconsolidated_comparisons", "consolidated_indices"}
        return [
            (key, value, "int64" if key in integer else _logical_dtype(module))
            for key in module._buffer_names
            if (value := getattr(module, key)) is not None
        ]
    if name == "AffineDeterministicModel":
        return [
            (key, getattr(module, key), _dtype(getattr(module, key)) or _default_dtype())
            for key in ("a", "b")
        ]
    if name in ("AffineFidelityCostModel", "FixedCostModel"):
        key = "weights" if name == "AffineFidelityCostModel" else "fixed_cost"
        return [(key, getattr(module, key), _logical_dtype(module, key))]
    if name == "EnsembleMapSaasSingleTaskGP":
        return [("taus", module.taus, _dtype(module.taus) or _logical_dtype(module))]
    if (
        type(module) is Model
        or isinstance(
            module,
            (
                SingleTaskGP,
                GPyTorchModel,
                ModelList,
                DeterministicModel,
                DeterministicCostModel,
                EnsembleModel,
            ),
        )
        or name == "ModelListGP"
    ):
        return []
    if not type(module).__module__.startswith("botorch_rs.models"):
        return []  # User models explicitly register tensors or provide Torch child modules.
    raise UnsupportedError(f"Native buffer descriptors for {name} are not available.")
