r"""
References

.. [Zhe2019hogp]
    S. Zhe, W. Xing, and R. M. Kirby. Scalable high-order gaussian process regression.
    Proceedings of Machine Learning Research, volume 89, Apr 2019.

The Kronecker-structured MLL with its gradient, posteriors and Matheron
samples run in Rust (``botorch_rs._botorch_rs.HigherOrderGP``). The initial
latent coordinates are drawn from torch's global generator when torch is
installed (as BoTorch draws them), else from Python's ``random``.
"""

from __future__ import annotations

import copy
import math
import random
import warnings
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_to, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
    UnsupportedError,
)
from botorch_rs.kernels import Kernel
from botorch_rs.likelihoods import GaussianLikelihood, Likelihood
from botorch_rs.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch_rs.models.model import FantasizeMixin
from botorch_rs.models.transforms.input import InputTransform, chain_spec
from botorch_rs.models.transforms.outcome import OutcomeTransform, Standardize
from botorch_rs.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch_rs.posteriors.gpytorch import Posterior
from botorch_rs.posteriors.higher_order import HigherOrderGPPosterior
from botorch_rs.posteriors.transformed import TransformedPosterior
from botorch_rs.utils.types import DEFAULT


def _is_default(value: Any) -> bool:
    return value is DEFAULT or type(value).__name__ == "_DefaultType"


class FlattenedStandardize(Standardize):
    r"""
    Standardize outcomes in a structured multi-output settings by reshaping the
    batched output dimensions to be a vector. Specifically, an output dimension
    of [a x b x c] will be squeezed to be a vector of [a * b * c].
    """

    def __init__(
        self,
        output_shape: Any,
        batch_shape: Any = None,
        min_stdv: float = 1e-8,
    ) -> None:
        r"""
        Args:
            output_shape: A ``n x output_shape``-dim tensor of training targets.
            batch_shape: The batch_shape of the training targets.
            min_stdv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        batch_shape = tuple(int(b) for b in (batch_shape or ()))
        super().__init__(m=1, outputs=None, batch_shape=batch_shape, min_stdv=min_stdv)
        self.output_shape = tuple(int(s) for s in output_shape)

    def _squeeze_to_single_output(self, tsr: Any) -> Any:
        data, shape = to_flat(tsr, "Y")
        dim_ct = len(shape) - len(self.output_shape) - 1
        lead = tuple(shape[:dim_ct])
        return from_flat(list(data), (*lead, len(data) // max(math.prod(lead), 1), 1), kind_of(tsr))

    def _return_to_output_shape(self, tsr: Any) -> Any:
        data, shape = to_flat(tsr, "Y")
        lead = tuple(shape[:-2])
        n = len(data) // max(math.prod(lead) * math.prod(self.output_shape), 1)
        return from_flat(list(data), (*lead, n, *self.output_shape), kind_of(tsr))

    def forward(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        Y = self._squeeze_to_single_output(Y)
        if Yvar is not None:
            Yvar = self._squeeze_to_single_output(Yvar)
        Y, Yvar = super().forward(Y, Yvar)
        Y_out = self._return_to_output_shape(Y)
        Yvar_out = self._return_to_output_shape(Yvar) if Yvar is not None else None
        return Y_out, Yvar_out

    __call__ = forward  # type: ignore[assignment]

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        Y = self._squeeze_to_single_output(Y)
        if Yvar is not None:
            Yvar = self._squeeze_to_single_output(Yvar)
        Y, Yvar = super().untransform(Y=Y, Yvar=Yvar, X=X)
        Y = self._return_to_output_shape(Y)
        if Yvar is not None:
            Yvar = self._return_to_output_shape(Yvar)
        return Y, Yvar

    def _scalar_stats(self) -> tuple[float, float]:
        if not self._rs.is_trained:
            return 0.0, 1.0
        return float(self._rs.means[0][0]), float(self._rs.stdvs[0][0])

    def untransform_posterior(self, posterior: Any, X: Any = None) -> TransformedPosterior:  # type: ignore[override]
        mu, sd = self._scalar_stats()

        def affine(t: Any, scale: float, offset: float) -> Any:
            data, shape = to_flat(t, "t")
            return from_flat([offset + scale * v for v in data], tuple(shape), kind_of(t))

        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda s: affine(s, sd, mu),
            mean_transform=lambda m, v: affine(m, sd, mu),
            variance_transform=lambda m, v: affine(v, sd * sd, 0.0),
        )

    def __repr__(self) -> str:
        return f"FlattenedStandardize(output_shape={self.output_shape})"


class _LatentParameters:
    """The latent coordinates of a ``HigherOrderGP`` (``ParameterList``):
    ``latent_parameters[i]`` is ``target_shape[i] x num_latent_dims[i]``."""

    def __init__(self, model: HigherOrderGP) -> None:
        self._model = model

    def __len__(self) -> int:
        return len(self._model._latent_dims)

    def __getitem__(self, i: int) -> Any:
        if not -len(self) <= i < len(self):
            raise IndexError(f"index {i} is out of range")
        i %= len(self)
        return self._model._raw_param_array(f"model.latent_parameters.{i}")

    def __setitem__(self, i: int, value: Any) -> None:
        self._model._set_raw_param(f"model.latent_parameters.{i % len(self)}", value)

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]


def _rand(shape: tuple[int, ...]) -> list[float]:
    """``torch.rand(shape, dtype=float64)`` draws (Python's ``random``
    without torch)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised without torch
        return [random.random() for _ in range(math.prod(shape))]
    return [float(v) for v in torch.rand(shape, dtype=torch.float64).reshape(-1).tolist()]


def _gp_latent_sample(K: list[float], s: int, r: int) -> tuple[list[float], list[float]]:
    """BoTorch's ``latent_init="gp"`` draw: ``MultivariateNormal(0, K + 1e-4
    I).sample((r,))`` reshaped to ``s x r`` (torch's ``randn`` of shape ``s x
    r`` through the Cholesky factor), and the prior covariance."""
    cov = [K[i * s + j] + (1e-4 if i == j else 0.0) for i in range(s) for j in range(s)]
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised without torch
        torch = None  # type: ignore[assignment]
    if torch is not None:
        normal = torch.randn(s, r, dtype=torch.float64).reshape(-1).tolist()
    else:
        normal = [random.gauss(0.0, 1.0) for _ in range(s * r)]
    sample, _ = _rs.higher_order_latent_sample_py((cov, (s, s)), (normal, (s, r)))
    return list(sample), cov


class HigherOrderGP(BatchedMultiOutputGPyTorchModel, FantasizeMixin):
    r"""
    A model for high-dimensional output regression.

    As described in [Zhe2019hogp]_. "Higher-order" means that the predictions
    are matrices (tensors) with at least two dimensions, such as images or
    grids of images, or measurements taken from a region of at least two
    dimensions.
    The posterior uses Matheron's rule [Doucet2010sampl]_
    as described in [Maddox2021bohdo]_.

    ``HigherOrderGP`` differs from a "vector" multi-output model in that it uses
    Kronecker algebra to obtain parsimonious covariance matrices for these
    outputs (see ``KroneckerMultiTaskGP`` for more information). For example,
    imagine a 10 x 20 x 30 grid of images. If we were to vectorize the
    resulting 6,000 data points in order to use them in a non-higher-order GP,
    they would have a 6,000 x 6,000 covariance matrix, with 36 million entries.
    The Kronecker structure allows representing this as a product of 10x10,
    20x20, and 30x30 covariance matrices, with only 1,400 entries.

    botorch_rs computes the MLL and the posterior exactly with Kronecker
    algebra (no approximate solves), so ``fit_gpytorch_mll`` (L-BFGS-B) and
    ``fit_gpytorch_mll_torch`` both work; unbatched training data only.

    Example:
        >>> model = HigherOrderGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
        >>> samples = model.posterior(test_X).rsample()
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        likelihood: Likelihood | None = None,
        covar_modules: list[Kernel] | None = None,
        num_latent_dims: list[int] | None = None,
        learn_latent_pars: bool = True,
        latent_init: str = "default",
        outcome_transform: OutcomeTransform | Any = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""
        Args:
            train_X: A ``batch_shape x n x d``-dim tensor of training inputs.
            train_Y: A ``batch_shape x n x output_shape``-dim tensor of
                training targets.
            likelihood: Gaussian likelihood for the model.
            covar_modules: List of kernels for each output structure.
            num_latent_dims: Sizes for the latent dimensions.
            learn_latent_pars: If true, learn the latent parameters.
            latent_init: [default or gp] how to initialize the latent parameters.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale). We use a
                ``Standardize`` transform if no ``outcome_transform`` is specified.
                Pass down ``None`` to use no outcome transform. Note that
                ``.train()`` will be called on the outcome transform during
                instantiation of the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        xs, ys = tuple(x[1]), tuple(y[1])
        if len(xs) < 2 or not ys:
            raise BotorchTensorDimensionError(
                f"Expected train_X of shape `n x d` and train_Y of shape `n x output_shape`, got {xs} and {ys}."
            )
        num_output_dims = len(ys) - len(xs) + 1
        batch_shape = xs[:-2]
        if len(batch_shape) > 1:
            raise NotImplementedError("HigherOrderGP currently only supports 1-dim `batch_shape`.")
        if batch_shape:
            raise UnsupportedError(
                "botorch_rs's HigherOrderGP supports unbatched training data (`n x d`) only."
            )
        if num_output_dims < 1 or ys[0] != xs[0]:
            raise BotorchTensorDimensionError(
                f"Expected train_Y of shape `n x output_shape` for train_X of shape {xs}, got {ys}."
            )
        if any(math.isnan(v) for v in x[0]) or any(math.isnan(v) for v in y[0]):
            raise InputDataError("Input data contains NaN values.")
        target_shape = ys[-num_output_dims:]
        n, d = xs[-2], xs[-1]
        if _is_default(outcome_transform):
            outcome_transform = FlattenedStandardize(output_shape=target_shape, batch_shape=())
        if outcome_transform is not None:
            if isinstance(outcome_transform, Standardize) and not isinstance(
                outcome_transform, FlattenedStandardize
            ):
                warnings.warn(
                    "HigherOrderGP does not support the outcome_transform "
                    "`Standardize`! Using `FlattenedStandardize` with `output_shape="
                    f"{target_shape} and batch_shape=() instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                outcome_transform = FlattenedStandardize(output_shape=target_shape, batch_shape=())
            if not isinstance(outcome_transform, FlattenedStandardize):
                raise UnsupportedError(
                    "botorch_rs's HigherOrderGP supports the FlattenedStandardize outcome transform "
                    "(or None)."
                )
            outcome_transform.train()
            train_Y, _ = outcome_transform(train_Y, X=train_X)
        if input_transform is not None and not isinstance(input_transform, InputTransform):
            raise UnsupportedError(
                "`input_transform` must be a botorch_rs input transform (from "
                "botorch_rs.models.transforms.input)."
            )
        transformed_X = train_X if input_transform is None else input_transform(train_X)
        transformed_dim = int(to_flat(transformed_X, "train_X")[1][-1])
        self._kind = kind
        self._num_dimensions = num_output_dims + 1
        self._num_outputs = 1
        self.target_shape = tuple(target_shape)
        self._input_batch_shape: tuple[int, ...] = ()
        self._aug_batch_shape: tuple[int, ...] = ()
        if likelihood is None:
            likelihood = get_gaussian_likelihood_with_lognormal_prior(batch_shape=())
        else:
            if type(likelihood) is not GaussianLikelihood:
                raise UnsupportedError("HigherOrderGP supports a GaussianLikelihood.")
            self._is_custom_likelihood = True
        if num_latent_dims is None:
            num_latent_dims = [1] * (self._num_dimensions - 1)
        num_latent_dims = [int(r) for r in num_latent_dims]
        if any(r <= 0 for r in num_latent_dims):
            raise ValueError("num_latent_dims must contain positive integers.")
        if len(num_latent_dims) != self._num_dimensions - 1:
            raise BotorchTensorDimensionError(
                f"Expected {self._num_dimensions - 1} latent dimension sizes, got "
                f"{len(num_latent_dims)}."
            )
        if covar_modules is None:
            # BoTorch uses `ard_num_dims=1` for every latent kernel, which
            # fails for `num_latent_dims > 1`; botorch_rs uses the number of
            # latent coordinates (the same kernels for one coordinate).
            covar_modules = [
                get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=num_latent_dims[i - 1] if i > 0 else transformed_dim,
                    batch_shape=(),
                )
                for i in range(self._num_dimensions)
            ]
        covar_modules = list(covar_modules)
        if len(covar_modules) != self._num_dimensions:
            raise BotorchTensorDimensionError(
                f"Expected {self._num_dimensions} covariance modules, got {len(covar_modules)}."
            )
        for k in covar_modules:
            if not isinstance(k, Kernel):
                raise UnsupportedError("`covar_modules` must be botorch_rs kernels.")
            if k._model is not None:
                raise UnsupportedError(
                    "This kernel is already part of another model; create a new one for each model."
                )
            if tuple(k._batch_shape):
                raise UnsupportedError("Kernels with a batch shape are not supported.")
        if latent_init not in ("default", "gp"):
            raise UnsupportedError(f"Unknown latent_init {latent_init!r}.")
        # Latent coordinates.
        latents: list[float] = []
        priors: list[Any] = []
        for i in range(self._num_dimensions - 1):
            s, r = self.target_shape[i], num_latent_dims[i]
            if latent_init == "default":
                latents.extend(_rand((s, r)))
                priors.append(None)
            else:
                if r != 1:
                    raise UnsupportedError(
                        "latent_init='gp' with num_latent_dims > 1 (BoTorch's latent kernels "
                        "are evaluated on one coordinate and its prior on the first one)."
                    )
                lin = [[j / (s - 1) if s > 1 else 0.0] for j in range(s)]
                Kf, _ = to_flat(covar_modules[i + 1](lin), "K")
                sample, cov = _gp_latent_sample(list(Kf), s, r)
                latents.extend(sample)
                priors.append((cov, [s, s]))
        raw = [_initial_raw_noise(likelihood)]
        for k in covar_modules:
            raw.extend(k._raw_flat())
        raw.extend(latents)
        input_steps = prefixes = training = None
        if input_transform is not None:
            spec = chain_spec(input_transform, d, ())
            if any(list(getattr(st.rs, "params", []) or []) for st in spec.chains[0]):
                raise UnsupportedError(
                    "HigherOrderGP does not yet optimize learned input transforms "
                    "(such as Warp); pre-transform the inputs instead."
                )
            input_steps = [st.rs for st in spec.chains[0]]
            prefixes = [st.prefix for st in spec.chains[0]]
            training = [st.training for st in spec.chains[0]]
        S = math.prod(self.target_shape)
        yflat = to_flat(train_Y, "train_Y")[0]
        outcome = None
        if outcome_transform is not None:
            outcome = outcome_transform._scalar_stats()
        self._rs = _rs.HigherOrderGP(
            to_flat(transformed_X, "train_X"),
            (list(yflat), [n, S]),
            list(self.target_shape),
            num_latent_dims,
            d,
            [k._node() for k in covar_modules],
            likelihood._spec,
            raw,
            latent_priors=priors if latent_init == "gp" else None,
            learn_latent_pars=bool(learn_latent_pars),
            input_chain=input_steps,
            input_chain_training=training,
            input_chain_prefixes=prefixes,
            raw_train_x=None if input_steps is None else x,
            outcome=outcome,
        )
        self._latent_dims = num_latent_dims
        self._learn_latent_pars = bool(learn_latent_pars)
        self._latent_init = latent_init
        self._train_X = train_X
        self._param_shapes = self._compute_shapes(covar_modules)
        self.likelihood: Likelihood = likelihood
        self.covar_modules: list[Kernel] = covar_modules
        likelihood._bind(self)
        for i, k in enumerate(covar_modules):
            k._bind(self, f"covar_modules.{i}.")
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.training = True
        self._fantasy_batch: list[Any] | None = None
        self._fantasy_shape: tuple[int, ...] = ()

    # -- parameters ------------------------------------------------------------

    def _compute_shapes(self, covar_modules: list[Kernel]) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {"likelihood.noise_covar.raw_noise": (1,)}
        for i, k in enumerate(covar_modules):
            for name, shape in k._node().param_shapes(f"model.covar_modules.{i}."):
                shapes[name] = tuple(shape)
        for i, r in enumerate(self._latent_dims):
            shapes[f"model.latent_parameters.{i}"] = (self.target_shape[i], r)
        return shapes

    def _offsets(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        off = 0
        for name, size in self._rs.parameter_names():
            out[name] = (off, size)
            off += size
        return out

    def _param_index(self, suffix: str) -> tuple[str, int, int]:
        offsets = self._offsets()
        if suffix in offsets:
            return suffix, *offsets[suffix]
        for name, (off, size) in offsets.items():
            if name.endswith(suffix):
                return name, off, size
        raise AttributeError(f"The model has no parameter {suffix!r}.")

    def _raw_param(self, suffix: str) -> list[float]:
        _, off, size = self._param_index(suffix)
        return list(self._rs.raw_parameters()[off : off + size])

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        name, _, _ = self._param_index(suffix)
        return self._param_shapes[name]

    def _raw_param_array(self, suffix: str) -> Any:
        return from_flat(self._raw_param(suffix), self._raw_param_shape(suffix), self._kind)

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        name, off, size = self._param_index(suffix)
        data, shape = to_flat(value, suffix)
        target = self._param_shapes[name]
        try:
            new = broadcast_to(data, tuple(shape), target)
        except ValueError:
            if len(data) != size:
                raise BotorchTensorDimensionError(
                    f"{name} has shape {target}, got a value of shape {tuple(shape)}."
                ) from None
            new = [float(v) for v in data]
        values = list(self._rs.raw_parameters())
        values[off : off + size] = new
        self._rs.set_raw_parameters(values)

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every parameter (BoTorch's names and order;
        the latent coordinates are parameters even without
        ``learn_latent_pars``, as in BoTorch)."""
        values = self._rs.raw_parameters()
        off = 0
        for name, size in self._rs.parameter_names():
            short = name[len("model.") :] if name.startswith("model.") else name
            yield short, from_flat(values[off : off + size], self._param_shapes[name], self._kind)
            off += size

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    def _param_entries(self) -> list[tuple[str, Any, int, int]]:
        offsets = self._offsets()
        out = []
        for (full, (off, size)), (short, value) in zip(
            offsets.items(), self.named_parameters(), strict=True
        ):
            if not self._learn_latent_pars and "latent_parameters" in full:
                continue
            out.append((short, value, off, size))
        return out

    @property
    def latent_parameters(self) -> _LatentParameters:
        """The latent coordinates of every output dimension."""
        return _LatentParameters(self)

    # -- Model API -------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        if self._fantasy_batch is not None:
            return self._fantasy_shape
        return self._input_batch_shape

    @property
    def train_inputs(self) -> tuple[Any]:
        data, shape = self._rs.train_inputs()
        return (from_flat(data, tuple(shape), self._kind),)

    @property
    def train_targets(self) -> Any:
        """The standardized training targets, flattened (``n·S``)."""
        data, _ = self._rs.train_targets()
        return from_flat(data, (len(data),), self._kind)

    def train(self, mode: bool = True) -> HigherOrderGP:
        self.training = mode
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        tf = (
            input_transform
            if input_transform is not None
            else getattr(self, "input_transform", None)
        )
        return X if tf is None else tf(X)

    def _posterior_of(self, source: Any, flat: Any, kind: Any, noise: bool) -> Any:
        has_ot = getattr(self, "outcome_transform", None) is not None
        post = HigherOrderGPPosterior(
            source,
            flat,
            kind,
            self.target_shape,
            int(source.num_train),
            observation_noise=noise,
            original=not has_ot,
            num_outputs=self._num_outputs,
        )
        if has_ot:
            return self.outcome_transform.untransform_posterior(post)
        return post

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        """The posterior at ``X`` (``batch_shape x q x d``): mean and variance
        of shape ``batch_shape x q x target_shape``, samples by Matheron's
        rule (a ``TransformedPosterior`` of a ``HigherOrderGPPosterior`` with
        ``FlattenedStandardize``). With ``observation_noise``, the variance
        includes the likelihood noise."""
        self.eval()
        if output_indices is not None:
            raise UnsupportedError("HigherOrderGP does not support output_indices.")
        if kwargs:
            raise UnsupportedError(f"Unsupported arguments: {sorted(kwargs)}.")
        if posterior_transform is not None:
            raise NotImplementedError(
                f"Posterior transforms currently not supported for {self.__class__.__name__}"
            )
        if not isinstance(observation_noise, bool):
            raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
        kind = kind_of(X)
        from botorch_rs.posteriors._query_autograd import requires_query_grad

        if requires_query_grad(X):
            raise UnsupportedError(
                "Direct HigherOrderGP posterior query gradients require a structured "
                "Matheron adjoint path, which is not implemented; native acquisition "
                "functions provide query gradients through the model handle."
            )
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}."
            )
        if flat[1][-1] != self._rs.dim:
            raise BotorchTensorDimensionError(
                f"Expected X with {self._rs.dim} columns (d), got {flat[1][-1]}."
            )
        if self._fantasy_batch is not None:
            if len(flat[1]) != 2:
                raise UnsupportedError(
                    "Batched test inputs on a batch of HigherOrderGP fantasy models."
                )
            return _BatchPosterior(
                [self._posterior_of(m, flat, kind, observation_noise) for m in self._fantasy_batch],
                kind,
                self._fantasy_shape,
            )
        return self._posterior_of(
            self._rs, (list(flat[0]), tuple(flat[1])), kind, observation_noise
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> Any:
        if posterior_transform is not None:
            raise NotImplementedError(
                f"Posterior transforms currently not supported for {self.__class__.__name__}"
            )
        if self._fantasy_batch is not None:
            raise UnsupportedError("Acquisition functions on a batch of fantasy models.")
        return self._rs.model_handle()

    # -- conditioning ----------------------------------------------------------

    def _with_rs(self, rs: Any) -> HigherOrderGP:
        new = self.__class__.__new__(self.__class__)
        memo: dict[int, Any] = {id(self): new}
        for key, value in self.__dict__.items():
            if key in ("_rs", "_fantasy_batch"):
                continue
            new.__dict__[key] = copy.deepcopy(value, memo)
        new._rs = rs
        new._fantasy_batch = None
        new._fantasy_shape = ()
        return new

    def _standardized(self, Y: Any) -> tuple[list[float], tuple[int, ...]]:
        ot = getattr(self, "outcome_transform", None)
        if ot is not None:
            ot.eval()
            Y, _ = ot(Y)
        data, shape = to_flat(Y, "Y")
        return list(data), tuple(shape)

    def condition_on_observations(
        self, X: Any, Y: Any, noise: Any = None, **kwargs: Any
    ) -> HigherOrderGP:
        r"""Condition the model on new observations (``X``: transformed
        inputs ``n' x d``, ``Y``: ``n' x target_shape``, or ``batch x n' x
        target_shape`` for a batch of conditioned models with shared
        hyperparameters).

        Returns:
            A ``HigherOrderGP`` with ``n + n'`` training examples.
        """
        if noise is not None:
            raise UnsupportedError(
                "condition_on_observations with explicit noise is not supported by "
                "HigherOrderGP (BoTorch: 'we don't support fixed noise here yet')."
            )
        if self._fantasy_batch is not None:
            raise UnsupportedError(
                "Conditioning an existing batch of HigherOrderGP fantasy models."
            )
        S = math.prod(self.target_shape)
        x = to_flat(X, "X")
        ydata, yshape = self._standardized(Y)
        k = len(self.target_shape)
        ybatch = tuple(yshape[: len(yshape) - k - 1])
        nprime = yshape[-k - 1]
        xshape = tuple(x[1])
        if xshape[-2] != nprime:
            raise BotorchTensorDimensionError(f"X has {xshape[-2]} points but Y has {nprime}.")
        nb = math.prod(ybatch)
        xb = math.prod(xshape[:-2])
        if xb not in (1, nb):
            raise BotorchTensorDimensionError(
                f"The batch shapes of X {xshape[:-2]} and Y {ybatch} do not match."
            )
        models = []
        for b in range(nb):
            xs = x[0][(b if xb > 1 else 0) * nprime * xshape[-1] :][: nprime * xshape[-1]]
            ys = ydata[b * nprime * S : (b + 1) * nprime * S]
            models.append(
                self._rs.condition_on_observations(
                    (list(xs), [nprime, xshape[-1]]), (list(ys), [nprime, S])
                )
            )
        new = self._with_rs(models[0])
        if ybatch:
            new._fantasy_batch = models
            new._fantasy_shape = ybatch
        return new

    def fantasize(
        self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> HigherOrderGP:
        r"""A batch of fantasy models conditioned on posterior samples at ``X``
        (``q x d``) drawn by ``sampler`` (with ``num_fantasies`` samples):
        batch shape ``num_fantasies``."""
        if observation_noise is not None:
            raise UnsupportedError("fantasize with explicit observation noise is not supported.")
        post = self.posterior(X)
        samples = sampler(post)
        return self.condition_on_observations(self.transform_inputs(X), samples)

    def __deepcopy__(self, memo: dict[int, Any]) -> HigherOrderGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_rs":
                new.__dict__[key] = value.copy()
            elif key == "_fantasy_batch":
                new.__dict__[key] = None if value is None else [m.copy() for m in value]
            else:
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        return f"HigherOrderGP(n={self._rs.num_train}, target_shape={self.target_shape})"


class _BatchPosterior(Posterior):
    """The posteriors of a batch of fantasy models, stacked in front."""

    def __init__(self, posteriors: list[Any], kind: Any, batch_shape: tuple[int, ...]) -> None:
        self._posteriors = posteriors
        self._kind = kind
        self.batch_shape = batch_shape

    @property
    def dtype(self) -> Any:
        return self._kind.dtype

    @property
    def device(self) -> Any:
        return self._kind.device

    def _stack(self, attr: str) -> Any:
        parts = [to_flat(getattr(p, attr), attr) for p in self._posteriors]
        data = [v for d, _ in parts for v in d]
        return from_flat(data, (*self.batch_shape, *parts[0][1]), self._kind)

    @property
    def mean(self) -> Any:
        return self._stack("mean")

    @property
    def variance(self) -> Any:
        return self._stack("variance")

    @property
    def base_sample_shape(self) -> tuple[int, ...]:
        return (*self.batch_shape, *self._posteriors[0].base_sample_shape)

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -1)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        return (*tuple(sample_shape), *self.batch_shape, *self._posteriors[0]._extended_shape())

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        sample_shape = tuple(sample_shape)
        data, shape = to_flat(base_samples, "base_samples")
        full = broadcast_to(data, tuple(shape), (*sample_shape, *self.base_sample_shape))
        nb, ns = len(self._posteriors), math.prod(sample_shape)
        inner = math.prod(self._posteriors[0].base_sample_shape)
        results = []
        for b, post in enumerate(self._posteriors):
            base = [
                v for s in range(ns) for v in full[(s * nb + b) * inner : (s * nb + b + 1) * inner]
            ]
            values = post.rsample_from_base_samples(
                sample_shape, from_flat(base, (*sample_shape, *post.base_sample_shape), self._kind)
            )
            results.append(to_flat(values, "samples")[0])
        block = len(results[0]) // max(ns, 1)
        output = [
            v for s in range(ns) for result in results for v in result[s * block : (s + 1) * block]
        ]
        return from_flat(output, self._extended_shape(sample_shape), self._kind)

    def rsample(self, sample_shape: Any = None) -> Any:
        from botorch_rs.posteriors.higher_order import _randn

        sample_shape = (1,) if sample_shape is None else tuple(sample_shape)
        shape = (*sample_shape, *self.base_sample_shape)
        return self.rsample_from_base_samples(
            sample_shape, from_flat(_randn(math.prod(shape), self._kind), shape, self._kind)
        )


def _initial_raw_noise(likelihood: Any) -> float:
    """The raw noise of a (not yet bound) Gaussian likelihood."""
    init = likelihood.noise_constraint._rs.initial_raw_value
    return float(init) if init is not None else 0.0


__all__ = ["FlattenedStandardize", "HigherOrderGP"]
