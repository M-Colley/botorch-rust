"""Torch module registration for caller-owned port input transforms.

The public object supplies all numerical methods and flags. This shell only
registers its actual Tensor state, parameters and children with a model.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn

from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    _TorchModule,
    tensor_prior_records,
)
from botorch_rs.models.transforms import _torch_input


class TorchPublicInputTransform(_TorchModule):
    _tensor_backend = "torch"

    def __init__(
        self,
        source: Any,
        like: Tensor,
        *,
        _memo: dict[int, TorchPublicInputTransform] | None = None,
    ) -> None:
        from botorch_rs.constraints import GreaterThan
        from botorch_rs.models.transforms.input import LearnedFeatureImputation, Warp

        super().__init__(like)
        memo = {} if _memo is None else _memo
        memo[id(source)] = self
        object.__setattr__(self, "_source", source)
        self._fit_on_model_data = source.training
        self._source_buffers: set[str] = set()
        self._source_parameters: set[str] = set()
        if isinstance(source, Warp):
            _torch_input.warp_parameters(source, like)
        elif isinstance(source, LearnedFeatureImputation):
            _torch_input.imputation_parameter(source, like)
        children = source.__dict__.get("_transforms", source.__dict__.get("transforms", {}))
        items = children.items() if isinstance(children, dict) else enumerate(children)
        for name, child in items:
            key = str(name)
            module = memo.get(id(child))
            if module is None:
                module = TorchPublicInputTransform(child, like, _memo=memo)
            self.add_module(key, module)
        if isinstance(source, Warp):
            normalize = memo.get(id(source._normalize))
            if normalize is None:
                normalize = TorchPublicInputTransform(source._normalize, like, _memo=memo)
            self.add_module("_normalize", normalize)
        # Caller-supplied feature maps are genuine registered modules too.
        source_children = dict(source.__dict__.items())
        source_children.update(source.__dict__.get("_modules", {}))
        for name, value in source_children.items():
            if isinstance(value, nn.Module) and name not in self._modules:
                self.add_module(name, value.to(like))
        encoders = getattr(source, "encoders", {})
        if isinstance(encoders, dict):
            modules = {
                str(index): encoder.to(like)
                for index, encoder in encoders.items()
                if isinstance(encoder, nn.Module)
            }
            if modules:
                self.add_module("_encoder_modules", nn.ModuleDict(modules))
        self._sync_from_source()
        if isinstance(source, Warp):
            for i in (0, 1):
                name = f"concentration{i}"
                self.add_module(
                    f"{name}_constraint",
                    TorchConstraint(
                        GreaterThan(source._min_concentration_level, transform=None), like
                    ),
                )
                prior_name, prior = f"{name}_prior", getattr(source, f"{name}_prior")
                if isinstance(prior, nn.Module):
                    self._modules[prior_name] = prior.to(like)
                    self._prior_parameters[prior_name] = name, 2 if source.batch_shape else 1
                else:
                    self._register_prior(prior_name, prior, name, 2 if source.batch_shape else 1)
        self.training = source.training

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            source = self.__dict__.get("_source")
            if (
                source is not None
                and not name.startswith("__")
                and not self.__dict__.get("_registering_source_module", False)
            ):
                return getattr(source, name)
            raise

    def add_module(self, name: str, module: nn.Module | None) -> None:
        self.__dict__["_registering_source_module"] = True
        try:
            super().add_module(name, module)
        finally:
            self.__dict__.pop("_registering_source_module", None)

    def __setattr__(self, name: str, value: Any) -> None:
        source = self.__dict__.get("_source")
        if (
            source is not None
            and not name.startswith("_")
            and name != "training"
            and hasattr(source, name)
        ):
            setattr(source, name, value)
            self._sync_from_source()
        else:
            super().__setattr__(name, value)

    def __getitem__(self, name: str) -> Any:
        return self._modules[name]

    def _sync_from_source(self) -> None:
        source = self._source
        parameters = dict(source.__dict__.get("_direct_parameters", {}))
        parameters.update(source.__dict__.get("_parameters", {}))
        parameters.update(
            (name, value)
            for name, value in source.__dict__.items()
            if isinstance(value, nn.Parameter)
        )
        for name, parameter in parameters.items():
            self._parameters[name] = parameter
            self._source_parameters.add(name)
        buffers = dict(source.__dict__.items())
        buffers.update(source.__dict__.get("_buffers", {}))
        for name, value in buffers.items():
            if isinstance(value, Tensor) and not isinstance(value, nn.Parameter):
                self._buffers[name] = value
                self._source_buffers.add(name)
        for child in self.children():
            if isinstance(child, TorchPublicInputTransform):
                child._sync_from_source()

    def _sync_to_source(self) -> None:
        for name in self._source_buffers:
            setattr(self._source, name, self._buffers[name])
        direct = self._source.__dict__.get("_direct_parameters", {})
        for name in self._source_parameters:
            if name in direct and not isinstance(self._source, nn.Module):
                direct[name] = self._parameters[name]
            else:
                setattr(self._source, name, self._parameters[name])
        for child in self.children():
            if isinstance(child, TorchPublicInputTransform):
                child._sync_to_source()

    @contextmanager
    def _graphs(self) -> Iterator[None]:
        ids = frozenset(
            id(module._source)
            for module in self.modules()
            if isinstance(module, TorchPublicInputTransform)
            and module._fit_on_model_data
            and getattr(module._source, "learn_coefficients", False)
        )
        token = _torch_input._model_affine_graphs.set(_torch_input._model_affine_graphs.get() | ids)
        try:
            yield
        finally:
            _torch_input._model_affine_graphs.reset(token)
            self._sync_from_source()

    def train(self, mode: bool = True) -> TorchPublicInputTransform:
        super().train(mode)
        self._source.train(mode)
        if mode:
            self._fit_on_model_data = True
        return self

    def fit(self, X: Tensor) -> TorchPublicInputTransform:
        from botorch_rs.models.transforms.input import (
            BatchBroadcastedInputTransform,
            ChainedInputTransform,
        )

        if not self._fit_on_model_data:
            return self
        with self._graphs():
            if isinstance(self._source, ChainedInputTransform) and (
                type(self._source).transform is ChainedInputTransform.transform
            ):
                for name in self._source.keys():
                    child = self[name]
                    child.fit(X)
                    X = child.preprocess_transform(X)
            elif isinstance(self._source, BatchBroadcastedInputTransform) and (
                type(self._source).transform is BatchBroadcastedInputTransform.transform
            ):
                for i, part in enumerate(X.unbind(self._source.broadcast_index)):
                    self[str(i)].fit(part)
            else:
                previous = self._source.training
                # Do not recursively train an intentionally frozen chain child.
                self._source.training = True
                try:
                    self._source(X)
                finally:
                    self._source.training = previous
        return self

    def forward(self, X: Tensor) -> Tensor:
        with self._graphs():
            return self._source(X)

    def transform(self, X: Tensor) -> Tensor:
        with self._graphs():
            return self._source.transform(X)

    def untransform(self, X: Tensor) -> Tensor:
        with self._graphs():
            return self._source.untransform(X)

    def preprocess_transform(self, X: Tensor) -> Tensor:
        with self._graphs():
            return self._source.preprocess_transform(X)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        result = super()._apply(fn, recurse)
        self._sync_to_source()
        return result

    def _save_to_state_dict(self, destination: Any, prefix: str, keep_vars: bool) -> None:
        self._sync_from_source()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._sync_to_source()

    def log_prior(self) -> Tensor:
        # InputPerturbation.batch_shape describes environmental draws, not
        # parameter batches; a stateless transform has a scalar zero prior.
        result = self._anchor.new_zeros(())
        for name, owner, prior, closure, _ in tensor_prior_records(self):
            value = prior.log_prob(closure(owner))
            local = name.rsplit(".", 1)[-1]
            if isinstance(owner, _TorchModule) and local in owner._prior_parameters:
                event_ndim = owner._prior_parameters[local][1]
            else:
                event_ndim = max(0, value.ndim - len(getattr(owner, "batch_shape", ())))
            if event_ndim:
                value = value.sum(tuple(range(-event_ndim, 0)))
            result = result + value
        return result
