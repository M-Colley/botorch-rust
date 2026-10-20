r"""Gaussian Process Regression models with fully Bayesian inference
(``botorch.models.fully_bayesian``).

Fully Bayesian models use Bayesian inference over model hyperparameters, such
as lengthscales and noise variance, learning a posterior distribution for the
hyperparameters using the No-U-Turn-Sampler (NUTS). This is followed by
sampling a small set of hyperparameters (often ~16) from the posterior
that we will use for model predictions and for computing acquisition function
values. By contrast, our “standard” models (e.g.
``SingleTaskGP``) learn only a single best value for each hyperparameter using
MAP. The fully Bayesian method generally results in a better and more
well-calibrated model, but is more computationally intensive. For a full
description, see [Eriksson2021saasbo].

BoTorch evaluates its models with JAX / NumPyro during NUTS; botorch_rs
evaluates the same potential energies (with analytic gradients) and runs a
port of NumPyro's NUTS in Rust (``botorch_core::fully_bayesian``). The
samples are loaded into a batched GP with one batch element per sample, as
BoTorch's ``load_mcmc_samples`` does.

References:

.. [Eriksson2021saasbo]
    D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-
    Seventh Conference on Uncertainty in Artificial Intelligence, 2021.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    from_flat,
    kind_from_dtype,
    kind_of,
    to_flat,
)
from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.errors import InputDataError, UnsupportedError
from botorch_rs.kernels import LinearKernel, MaternKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models import _numpyro
from botorch_rs.models.gp_regression import SingleTaskGP, _validate_input_scaling
from botorch_rs.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
    Warp,
)
from botorch_rs.models.transforms.outcome import OutcomeTransform, Standardize
from botorch_rs.models.transforms.utils import kumaraswamy_warp, subset_transform
from botorch_rs.posteriors.fully_bayesian import MCMC_DIM, GaussianMixturePosterior

#: BoTorch's ``MIN_INFERRED_NOISE_LEVEL`` (``botorch.models.utils.gpytorch_modules``).
MIN_INFERRED_NOISE_LEVEL = 1e-4

# ---------------------------------------------------------------------------
# Kernel helpers (BoTorch evaluates them with JAX)
# ---------------------------------------------------------------------------


def _kernel_helper(X: Any, scale: Any, kernel: str) -> Any:
    if _numpyro.is_jax(X) or _numpyro.is_jax(scale):
        return _numpyro.kernel(X, scale, kernel)
    data, shape = _rs.fully_bayesian_kernel(kernel, to_flat(X, "X"), to_flat(scale, "scale"))
    return from_flat(data, tuple(shape), kind_of(X))


def compute_dists(X: Any, lengthscale: Any) -> Any:
    """Compute kernel distances.

    ``sqrt(max(||(x_i - x_j) / lengthscale||^2, 1e-30))`` of the rows of
    ``X`` (``... x n x d``). ``X / lengthscale`` broadcasts as in NumPy;
    the pairwise distances of each batch are computed in Rust."""
    return _kernel_helper(X, lengthscale, "distances")


def matern52_kernel(X: Any, lengthscale: Any) -> Any:
    """Matern-5/2 kernel, with batched inputs and broadcast lengthscales."""
    return _kernel_helper(X, lengthscale, "matern52")


def linear_kernel(X: Any, weight_variance: Any) -> Any:
    """Linear kernel.

    ``(X sqrt(w)) (X sqrt(w))^T`` of ``X`` (``... x n x d``), with
    broadcast weight variances and one kernel matrix per batch."""
    return _kernel_helper(X, weight_variance, "linear")


def reshape_and_detach(target: Any, new_value: Any) -> Any:
    """Detach and reshape ``new_value`` to match ``target``."""
    data, _ = to_flat(new_value, "new_value")
    _, shape = to_flat(target, "target")
    if len(data) != math.prod(shape):
        raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {len(data)}")
    return from_flat(list(data), tuple(shape), kind_of(target))


# ---------------------------------------------------------------------------
# The probabilistic models
# ---------------------------------------------------------------------------


def _column(data: Any, rows: int) -> list[float]:
    values = [float(v) for v in data]
    if len(values) != rows:
        raise RuntimeError(f"Expected {rows} values, got {len(values)}.")
    return values


def _values(samples: dict[str, Any], name: str) -> tuple[list[float], tuple[int, ...]]:
    data, shape = to_flat(samples[name], name)
    return [float(v) for v in data], tuple(shape)


def _sample_kind(samples: dict[str, Any]) -> ArrayKind:
    for v in samples.values():
        return kind_of(v)
    return kind_of(None)


class PyroModel:
    r"""
    Base class for a Pyro model; used to assist in learning hyperparameters.

    This class and its subclasses are not a standard BoTorch models; instead
    the subclasses are used as inputs to a ``SaasFullyBayesianSingleTaskGP``,
    which should then have its hyperparameters fit with
    ``fit_fully_bayesian_model_nuts``. (By default, its subclass ``SaasPyroModel``
    is used).

    The default native fitter evaluates its potential energy
    (the negative log joint density of the unconstrained hyperparameters,
    NumPyro's parameterization) and its gradient in Rust.
    ``fit_fully_bayesian_model_nuts`` samples it with a port of NumPyro's
    NUTS. The public ``sample`` / ``sample_*`` methods additionally implement
    optional NumPyro named-site tracing. Custom trace overrides can be used
    with NumPyro directly; the native fitter rejects such overrides.
    """

    _prior_mode: bool = False
    _noiseless_eps_for_sampleability: float = 1e-7
    #: The kernel of the Rust model ("saas", "matern" or "linear").
    _rs_kernel: str = ""
    _outputscale_prior_concentration: float | None = None
    _outputscale_prior_rate: float | None = None

    def __init__(
        self,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
        eps: float = 1e-7,
    ) -> None:
        r"""Initialize the PyroModel.

        Args:
            use_input_warping: A boolean indicating whether to use input warping.
            indices_to_warp: An optional list of indices to warp. The default
                is to warp all inputs.
            eps: A small value that is used to ensure inputs are not 0 or 1,
                when using input warping.
        """
        self.use_input_warping = use_input_warping
        self.indices = indices_to_warp
        self._eps = eps
        self._rs: Any = None

    @subset_transform
    def warp(self, X: Any, c0: Any, c1: Any) -> Any:
        r"""Warp the input through a Kumaraswamy CDF."""
        return kumaraswamy_warp(X=X, c0=c0, c1=c1, eps=self._eps)

    def set_inputs(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        task_feature: int | None = None,
        task_rank: int | None = None,
    ) -> None:
        """Set the training data.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            task_feature: The index of the task feature column. Used by
                multi-task mixins.
            task_rank: The number of learned task embeddings. Used by
                multi-task mixins.
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.ard_num_dims = to_flat(train_X, "train_X")[1][-1]
        self._rs = None
        self._trace_inputs: dict[str, Any] = {}

    def _trace_input(self, name: str) -> Any:
        if name not in self._trace_inputs:
            value = getattr(self, name)
            self._trace_inputs[name] = None if value is None else _numpyro.array(value)
        return self._trace_inputs[name]

    @property
    def train_X_jax(self) -> Any:
        return self._trace_input("train_X")

    @property
    def train_Y_jax(self) -> Any:
        return self._trace_input("train_Y")

    @property
    def train_Yvar_jax(self) -> Any:
        return self._trace_input("train_Yvar")

    def _task_structure(self) -> tuple[int, int, int] | None:
        """``(task_feature, num_tasks, rank)`` of a multi-task model."""
        return None

    def _warp_indices(self) -> list[int] | None:
        if not self.use_input_warping:
            return None
        return list(range(self.ard_num_dims)) if self.indices is None else list(self.indices)

    def _potential(self) -> _rs.PyroModel:
        """The Rust model (potential energy) of the training data."""
        if self._prior_mode:
            raise UnsupportedError(
                "The native NUTS fitter conditions on training observations. "
                "Use NumPyro directly for the optional prior-mode trace."
            )
        for name in (
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
        ):
            method = getattr(self, name, None)
            if method is not None and getattr(method, "__module__", "") not in (
                __name__,
                "botorch_rs.models.fully_bayesian_multitask",
            ):
                raise UnsupportedError(
                    f"The native NUTS potential does not implement the custom {name} override. "
                    "Use the optional NumPyro trace interface with NumPyro inference directly."
                )
        if self._rs is None:
            if not self._rs_kernel:
                raise UnsupportedError(
                    f"{type(self).__name__} has no botorch_rs implementation: botorch_rs samples "
                    "SaasPyroModel, MaternPyroModel, LinearPyroModel and MultitaskSaasPyroModel "
                    "(custom NumPyro models are not supported)."
                )
            x = to_flat(self.train_X, "train_X")
            y, _ = to_flat(self.train_Y, "train_Y")
            yvar = None
            if self.train_Yvar is not None:
                yvar = [float(v) for v in to_flat(self.train_Yvar, "train_Yvar")[0]]
            prior = None
            if (
                self._outputscale_prior_concentration is not None
                and self._outputscale_prior_rate is not None
            ):
                prior = (
                    float(self._outputscale_prior_concentration),
                    float(self._outputscale_prior_rate),
                )
            self._rs = _rs.PyroModel(
                self._rs_kernel,
                (list(x[0]), list(x[1])),
                [float(v) for v in y],
                train_yvar=yvar,
                outputscale_prior=prior,
                warp_indices=self._warp_indices(),
                warp_eps=float(self._eps),
                task=self._task_structure(),
            )
        return self._rs

    def sample(self) -> None:
        """Define a NumPyro program in a concrete subclass."""
        raise NotImplementedError

    def sample_noise(self) -> Any:
        return _numpyro.noise(self)

    def sample_mean(self) -> Any:
        return _numpyro.mean(self)

    def sample_concentrations(self) -> tuple[Any, Any]:
        return _numpyro.concentrations(self)

    def _warp_jax(self, X: Any, c0: Any, c1: Any) -> Any:
        return _numpyro.warp(self, X, c0, c1)

    def _maybe_input_warp(self, X_jax: Any) -> Any:
        if not self.use_input_warping:
            return X_jax
        c0, c1 = self.sample_concentrations()
        return self._warp_jax(X_jax, c0, c1)

    def _prepare_features(self, X: Any) -> Any:
        return X

    def _maybe_multitask_transform(self, K_noiseless: Any, mean: Any) -> tuple[Any, Any]:
        return K_noiseless, mean

    def sample_observations(self, mean: Any, K_noiseless: Any, noise: Any) -> None:
        _numpyro.observations(self, mean, K_noiseless, noise)

    def postprocess_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> dict[str, Any]:
        """Post-process the final MCMC samples.

        ``mcmc_samples`` are the (constrained) values of the latent sites, as
        NumPyro's ``MCMC.get_samples()`` returns them. Unlike BoTorch, the
        ``noise`` samples are converted to the noise variance of the
        likelihood they were sampled with, ``MIN_INFERRED_NOISE_LEVEL +
        noise`` (BoTorch's models use ``max(noise, MIN_INFERRED_NOISE_LEVEL)``).
        """
        out = dict(mcmc_samples)
        if "noise" in out:
            data, shape = to_flat(out["noise"], "noise")
            out["noise"] = from_flat(
                [MIN_INFERRED_NOISE_LEVEL + float(v) for v in data],
                tuple(shape),
                kind_of(out["noise"]),
            )
        return out

    def get_dummy_mcmc_samples(self, num_mcmc_samples: int, **tkwargs: Any) -> dict[str, Any]:
        """Return dummy MCMC samples for state dict loading.

        Each subclass returns a dict of ones with the keys and shapes that
        ``load_mcmc_samples`` expects.
        """
        raise NotImplementedError

    def _ones(self, shape: tuple[int, ...], tkwargs: dict[str, Any]) -> Any:
        kind = kind_from_dtype(tkwargs.get("dtype"), tkwargs.get("device"))
        return from_flat([1.0] * math.prod(shape), shape, kind)

    def _common_dummy_samples(
        self, mcmc_samples: dict[str, Any], num_mcmc_samples: int, **tkwargs: Any
    ) -> dict[str, Any]:
        """Add noise and warping entries to ``mcmc_samples`` in-place."""
        dim = self.ard_num_dims if self.indices is None else len(self.indices)
        if self.train_Yvar is None:
            mcmc_samples["noise"] = self._ones((num_mcmc_samples,), tkwargs)
        if self.use_input_warping:
            mcmc_samples["c0"] = self._ones((num_mcmc_samples, dim), tkwargs)
            mcmc_samples["c1"] = self._ones((num_mcmc_samples, dim), tkwargs)
        return mcmc_samples

    def load_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        raise NotImplementedError

    # -- modules -----------------------------------------------------------

    def _mean_module(self, mcmc_samples: dict[str, Any], batch_shape: tuple[int, ...]) -> Any:
        mean = ConstantMean(batch_shape=batch_shape)
        mean._pending = {"constant": _values(mcmc_samples, "mean")[0]}  # type: ignore[attr-defined]
        return mean

    def _likelihood(self, mcmc_samples: dict[str, Any], batch_shape: tuple[int, ...]) -> Any:
        if self.train_Yvar is not None:
            return None  # fixed noise: train_Yvar of every sample
        likelihood = GaussianLikelihood(
            batch_shape=batch_shape,
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        )
        noise = _values(mcmc_samples, "noise")[0]
        likelihood._pending = {  # type: ignore[attr-defined]
            "noise": [max(v, MIN_INFERRED_NOISE_LEVEL) for v in noise]
        }
        return likelihood

    def _warping_function(
        self, mcmc_samples: dict[str, Any], batch_shape: tuple[int, ...]
    ) -> Warp | None:
        if not self.use_input_warping:
            return None
        d = self.ard_num_dims
        indices = list(range(d)) if self.indices is None else list(self.indices)
        warp = Warp(
            d=d,
            batch_shape=batch_shape,
            indices=indices,
            bounds=[[0.0] * d, [1.0] * d],
            eps=self._eps,
        )
        num = batch_shape[0]
        k = len(indices)
        for attr, name in (("concentration0", "c0"), ("concentration1", "c1")):
            values, _ = _values(mcmc_samples, name)
            setattr(warp, attr, from_flat(values, (num, 1, k), kind_of(None)))
        return warp


class MaternPyroModel(PyroModel):
    r"""Implementation of a fully Bayesian model with a dimension-scaling prior.

    ``MaternPyroModel`` is not a standard BoTorch model; instead, it is used as
    an input to ``FullyBayesianSingleTaskGP``.
    """

    _rs_kernel = "matern"
    _outputscale_prior_concentration: float | None = None
    _outputscale_prior_rate: float | None = None

    def sample(self) -> None:
        _numpyro.matern_sample(self)

    def sample_lengthscale(self, dim: int) -> Any:
        return _numpyro.lengthscale(dim)

    def sample_outputscale(
        self, concentration: float | None = None, rate: float | None = None
    ) -> Any:
        return _numpyro.outputscale(concentration, rate)

    def postprocess_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> dict[str, Any]:
        r"""Post-process the MCMC samples (and convert ``noise`` to the noise
        variance of the likelihood, see ``PyroModel``)."""
        return super().postprocess_mcmc_samples(mcmc_samples)

    def get_dummy_mcmc_samples(self, num_mcmc_samples: int, **tkwargs: Any) -> dict[str, Any]:
        """Return dummy MCMC samples for state dict loading."""
        mcmc_samples = {
            "mean": self._ones((num_mcmc_samples,), tkwargs),
            "lengthscale": self._ones((num_mcmc_samples, self.ard_num_dims), tkwargs),
        }
        return self._common_dummy_samples(mcmc_samples, num_mcmc_samples, **tkwargs)

    def _get_covar_module(
        self,
        use_scale_kernel: bool,
        batch_shape: tuple[int, ...],
        dtype: Any = None,
        device: Any = None,
    ) -> Any:
        """Get the covar module to load samples into.

        Args:
            use_scale_kernel: A boolean indicating whether to use an outputscale other
                than 1.
            batch_shape: The batch shape (number of mcmc samples).
            dtype: The dtype.
            device: The device.

        Returns:
            The covar module.
        """
        covar_module: Any = MaternKernel(ard_num_dims=self.ard_num_dims, batch_shape=batch_shape)
        if use_scale_kernel:
            covar_module = ScaleKernel(covar_module, batch_shape=batch_shape)
        return covar_module

    def load_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood.

        Returns botorch_rs modules (the mean and the likelihood receive their
        values when the model is built from them) and the warping function.
        """
        mean_values, _ = _values(mcmc_samples, "mean")
        num = len(mean_values)
        batch_shape = (num,)
        mean_module = self._mean_module(mcmc_samples, batch_shape)
        outputscale = mcmc_samples.get("outputscale")
        covar_module = self._get_covar_module(
            use_scale_kernel=outputscale is not None, batch_shape=batch_shape
        )
        likelihood = self._likelihood(mcmc_samples, batch_shape)
        if isinstance(covar_module, ScaleKernel):
            os_values, _ = _values(mcmc_samples, "outputscale")
            covar_module.outputscale = from_flat(os_values, batch_shape, kind_of(None))
            base_kernel = covar_module.base_kernel
        else:
            base_kernel = covar_module
        ls, _ = _values(mcmc_samples, "lengthscale")
        base_kernel.lengthscale = from_flat(ls, (num, 1, self.ard_num_dims), kind_of(None))
        warping_function = self._warping_function(mcmc_samples, batch_shape)
        covar_module = self._build_multitask_covariance(mcmc_samples, covar_module, batch_shape)
        return mean_module, covar_module, likelihood, warping_function

    def _build_multitask_covariance(
        self, mcmc_samples: dict[str, Any], covar_module: Any, batch_shape: tuple[int, ...]
    ) -> Any:
        """Optionally wrap covar_module with task covariance (multi-task mixins)."""
        return covar_module


class SaasPyroModel(MaternPyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    ``SaasPyroModel`` is not a standard BoTorch model; instead, it is used as
    an input to ``SaasFullyBayesianSingleTaskGP``. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    ``SaasPyroModel`` unless they want to customize its attributes (such as
    ``covar_module``).
    """

    _rs_kernel = "saas"
    _outputscale_prior_concentration: float | None = 2.0
    _outputscale_prior_rate: float | None = 0.15

    def sample_lengthscale(self, dim: int, alpha: float = 0.1) -> Any:
        return _numpyro.saas_lengthscale(dim, alpha)

    def postprocess_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> dict[str, Any]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales, removes the inverse lengthscales and
        tausq (global shrinkage), and converts ``noise`` to the noise variance
        of the likelihood (see ``PyroModel``).
        """
        out = dict(mcmc_samples)
        tausq, _ = _values(out, "kernel_tausq")
        ils, shape = _values(out, "_kernel_inv_length_sq")
        d = shape[-1]
        inv = [tausq[i // d] * v for i, v in enumerate(ils)]
        out["lengthscale"] = from_flat(
            [1.0 / math.sqrt(v) for v in inv], shape, kind_of(mcmc_samples["_kernel_inv_length_sq"])
        )
        del out["kernel_tausq"], out["_kernel_inv_length_sq"]
        return super().postprocess_mcmc_samples(out)

    def get_dummy_mcmc_samples(self, num_mcmc_samples: int, **tkwargs: Any) -> dict[str, Any]:
        """Return dummy MCMC samples for state dict loading."""
        mcmc_samples = super().get_dummy_mcmc_samples(num_mcmc_samples=num_mcmc_samples, **tkwargs)
        mcmc_samples["outputscale"] = self._ones((num_mcmc_samples,), tkwargs)
        return mcmc_samples


class LinearPyroModel(PyroModel):
    r"""Implementation of a Bayesian Linear pyro model.

    ``LinearPyroModel`` is not a standard BoTorch model; instead, it is used as
    an input to ``FullyBayesianLinearSingleTaskGP``. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    ``LinearPyroModel`` unless they want to customize its attributes (such as
    ``covar_module``).
    """

    _rs_kernel = "linear"

    def sample(self) -> None:
        _numpyro.linear_sample(self)

    def sample_weight_variance(self, alpha: float = 0.1) -> Any:
        return _numpyro.weight_variance(self.ard_num_dims, alpha)

    def postprocess_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> dict[str, Any]:
        r"""Post-process the MCMC samples.

        This computes the true weight variance, removes tausq (global shrinkage),
        and converts ``noise`` to the noise variance of the likelihood (see
        ``PyroModel``).
        """
        out = dict(mcmc_samples)
        tau, _ = _values(out, "tau_sq")
        w, shape = _values(out, "_weight_variance_sq")
        d = shape[-1]
        out["weight_variance"] = from_flat(
            [math.sqrt(tau[i // d] * v) for i, v in enumerate(w)],
            shape,
            kind_of(mcmc_samples["_weight_variance_sq"]),
        )
        del out["tau_sq"], out["_weight_variance_sq"]
        return super().postprocess_mcmc_samples(out)

    def get_dummy_mcmc_samples(self, num_mcmc_samples: int, **tkwargs: Any) -> dict[str, Any]:
        """Return dummy MCMC samples for state dict loading."""
        mcmc_samples = {
            "mean": self._ones((num_mcmc_samples,), tkwargs),
            "weight_variance": self._ones((num_mcmc_samples, self.ard_num_dims), tkwargs),
        }
        return self._common_dummy_samples(mcmc_samples, num_mcmc_samples, **tkwargs)

    def load_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        r"""Load the MCMC samples into their corresponding modules."""
        wv, shape = _values(mcmc_samples, "weight_variance")
        num = shape[0]
        batch_shape = (num,)
        d = self.ard_num_dims
        mean_module = self._mean_module(mcmc_samples, batch_shape)
        covar_module = LinearKernel(batch_shape=batch_shape, ard_num_dims=d)
        covar_module.variance = from_flat(wv, (num, 1, d), kind_of(None))
        bounds = [[0.0] * d, [1.0] * d]
        input_tf: Any = Normalize(
            d=d,
            bounds=bounds,
            center=0.0,
            batch_shape=batch_shape if self.use_input_warping else (),
        )
        warping_function = self._warping_function(mcmc_samples, batch_shape)
        if warping_function is not None:
            input_tf = ChainedInputTransform(warp=warping_function, normalize=input_tf)
        likelihood = self._likelihood(mcmc_samples, batch_shape)
        return mean_module, covar_module, likelihood, input_tf


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------


def _repeat(data: Any, shape: tuple[int, ...], num: int) -> tuple[list[float], tuple[int, ...]]:
    values = list(data)
    return values * num, (num, *shape)


def _apply_pending(module: Any) -> None:
    pending = getattr(module, "_pending", None)
    if pending is None:
        return
    for name, values in pending.items():
        setattr(module, name, values)
    del module._pending


class AbstractFullyBayesianSingleTaskGP(SingleTaskGP):
    r"""An abstract fully Bayesian single-task GP model.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.

    Once fitted, the model is a batched ``SingleTaskGP`` with one batch
    element per MCMC sample (BoTorch's batched modules on unbatched
    training data), whose posterior is a ``GaussianMixturePosterior``.
    """

    _is_fully_bayesian = True
    _is_ensemble = True
    _supports_batched_models = False
    _pyro_model_class: type[PyroModel] = PyroModel

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale).
                Note that ``.train()`` will be called on the outcome transform during
                instantiation of the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
            use_input_warping: A boolean indicating whether to use input warping.
            indices_to_warp: An optional list of indices to warp. The default
                is to warp all inputs.

        botorch_rs supports ``Standardize`` (single-output) as the outcome
        transform: BoTorch's posteriors of these models need a linear one.
        """
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        if not (len(x[1]) == len(y[1]) == 2 and x[1][0] == y[1][0] and y[1][-1] == 1):
            raise ValueError("Expected train_X to have shape n x d and train_Y to have shape n x 1")
        if train_Yvar is not None:
            if tuple(to_flat(train_Yvar, "train_Yvar")[1]) != tuple(y[1]):
                raise ValueError("Expected train_Yvar to be None or have the same shape as train_Y")
        if outcome_transform is not None and not (
            type(outcome_transform) is Standardize and outcome_transform._outputs is None
        ):
            raise UnsupportedError(
                "botorch_rs supports the Standardize outcome transform on fully Bayesian models "
                "(BoTorch's GaussianMixturePosterior needs a linear outcome transform)."
            )
        kind = kind_of(train_X)
        transformed_X = train_X
        if input_transform is not None:
            transformed_X = input_transform(train_X)
        Yvar = train_Yvar
        if outcome_transform is not None:
            outcome_transform.train()
            train_Y, Yvar = outcome_transform(Y=train_Y, Yvar=train_Yvar, X=transformed_X)
        self._fb_check_data(transformed_X, train_Y, Yvar)
        if Yvar is not None:  # Clamp after transforming
            vdata, vshape = to_flat(Yvar, "train_Yvar")
            Yvar = from_flat(
                [max(float(v), MIN_INFERRED_NOISE_LEVEL) for v in vdata], tuple(vshape), kind
            )
        self._kind = kind
        self._fb_train_X = (list(x[0]), tuple(x[1]))
        self._fb_train_Y = (list(to_flat(train_Y, "train_Y")[0]), tuple(y[1]))
        self._fb_raw_Y = (list(y[0]), tuple(y[1]))
        self._fb_input_transform = input_transform
        self.mean_module: Any = None  # type: ignore[assignment]
        self.covar_module: Any = None  # type: ignore[assignment]
        self.likelihood: Any = None  # type: ignore[assignment]
        self.pyro_model = self._pyro_model_class(
            use_input_warping=use_input_warping,
            indices_to_warp=indices_to_warp,
        )
        self.pyro_model.set_inputs(train_X=transformed_X, train_Y=train_Y, train_Yvar=Yvar)
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.training = True

    @staticmethod
    def _fb_check_data(X: Any, Y: Any, Yvar: Any) -> None:
        """BoTorch's ``validate_input_scaling`` of the transformed data."""
        x = to_flat(X, "train_X")
        y = to_flat(Y, "train_Y")
        values = list(x[0]) + list(y[0])
        if any(math.isnan(v) for v in values):
            raise InputDataError("Input data contains NaN values.")
        if Yvar is not None:
            v = to_flat(Yvar, "train_Yvar")
            if any(float(t) < 0 for t in v[0]):
                raise InputDataError("Input data contains negative variances.")
        n = y[1][-2]
        if n > 0:
            _validate_input_scaling((x[0], x[1]), (y[0], y[1]), n)

    # -- fitted state --------------------------------------------------------

    def _check_if_fitted(self) -> None:
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return int(self._rs.batch_shape[0])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that ``SaasFullyBayesianSingleTaskGP`` does not support batching
        over input data at this point."""
        return (self.num_mcmc_samples,)

    @property
    def _aug_batch_shape(self) -> tuple[int, ...]:
        r"""The batch shape of the model, augmented to include the output dim."""
        return self.batch_shape

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def _num_outputs(self) -> int:
        return 1

    @property
    def train_inputs(self) -> tuple[Any]:
        """``(train_X,)`` (the original training inputs, ``n x d``)."""
        data, shape = self._fb_train_X
        return (from_flat(data, shape, self._kind),)

    @property
    def train_targets(self) -> Any:
        """The (outcome-transformed) training targets, shape ``n``."""
        data, shape = self._fb_train_Y
        return from_flat(data, (shape[0],), self._kind)

    def train(self, mode: bool = True, reset: bool = True) -> AbstractFullyBayesianSingleTaskGP:  # type: ignore[override]
        r"""Puts the model in ``train`` mode.

        Args:
            mode: A boolean indicating whether to put the model in training mode.
            reset: A boolean indicating whether to reset the model to its initial
                state if mode is True. If ``mode`` is False, this argument is ignored.

        Returns:
            The model itself.
        """
        self.training = mode
        if mode and reset:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None
            self.__dict__.pop("_rs", None)
            if self._fb_input_transform is not None:
                self.input_transform = self._fb_input_transform
            else:
                self.__dict__.pop("input_transform", None)
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    def load_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by ``fit_fully_bayesian_model_nuts`` when the model
        has been fitted in order to create a batched SingleTaskGP model.

        The input transform of the model is the original one followed by the
        warping (and normalization) of the samples; loading samples again
        replaces them (BoTorch chains them to the previous ones, a bug fixed
        on the M-Colley fork).
        """
        mean_module, covar_module, likelihood, input_tf = self.pyro_model.load_mcmc_samples(
            mcmc_samples=mcmc_samples
        )
        num = int(mean_module.batch_shape[0])
        user_tf = self._fb_input_transform
        chain: Any = input_tf
        if user_tf is not None:
            chain = copy.deepcopy(user_tf)
            if input_tf is not None:
                tfs = [chain]
                if isinstance(input_tf, ChainedInputTransform):
                    tfs.extend(list(input_tf.values()))
                else:
                    tfs.append(input_tf)
                chain = ChainedInputTransform(**{f"tf{i}": tf for i, tf in enumerate(tfs)})
        if chain is not None:
            chain.train()
        xdata, xshape = self._fb_train_X
        ydata, yshape = self._fb_raw_Y
        user_ot = getattr(self, "outcome_transform", None)
        ot: Any = None
        if user_ot is not None:
            ot = Standardize(m=1, batch_shape=(num,), min_stdv=user_ot.min_stdv)
        yvar = None
        if self.pyro_model.train_Yvar is not None:
            # The (outcome-transformed, clamped) noise variances of the
            # samples, in the outcome space of the batched model.
            vdata = [float(v) for v in to_flat(self.pyro_model.train_Yvar, "train_Yvar")[0]]
            if user_ot is not None:
                sd = float(to_flat(user_ot.stdvs, "stdvs")[0][0])
                vdata = [v * sd * sd for v in vdata]
            yvar = _repeat(vdata, yshape, num)
        SingleTaskGP.__init__(
            self,
            train_X=from_flat(*_repeat(xdata, xshape, num), self._kind),
            train_Y=from_flat(*_repeat(ydata, yshape, num), self._kind),
            train_Yvar=None if yvar is None else from_flat(*yvar, self._kind),
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=ot,
            input_transform=chain,
        )
        _apply_pending(self.mean_module)
        _apply_pending(self.likelihood)
        if user_ot is not None:
            self.outcome_transform = user_ot
        if chain is None:
            self.__dict__.pop("input_transform", None)

    @property
    def fantasize(self) -> Any:  # type: ignore[override]
        """BoTorch's fully Bayesian single-task GPs have no ``fantasize``
        (they are not a ``FantasizeMixin``), so that the acquisition
        functions that fantasize (e.g. the knowledge gradient) do not support
        them."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'fantasize'")

    def forward(self, X: Any) -> Any:
        """The prior of the batched GP at ``X`` (BoTorch's ``forward``; not
        supported by botorch_rs, use ``posterior``)."""
        self._check_if_fitted()
        raise UnsupportedError(
            "The forward pass of a fully Bayesian model is not supported by botorch_rs; use "
            "`posterior`."
        )

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GaussianMixturePosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A ``(batch_shape) x q x d``-dim Tensor, where ``d`` is the dimension
                of the feature space and ``q`` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A ``GaussianMixturePosterior`` object. Includes observation noise
                if specified.
        """
        self._check_if_fitted()
        data, shape = to_flat(X, "X")
        shape = tuple(shape)
        if len(shape) < 2:
            raise ValueError(f"Expected X to be `batch_shape x q x d`, got shape {shape}.")
        from botorch_rs import _ops

        Xu = _ops.getitem(X, (Ellipsis, None, slice(None), slice(None)))
        posterior = SingleTaskGP.posterior(
            self,
            X=Xu,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        return GaussianMixturePosterior._from_posterior(posterior)

    def condition_on_observations(  # type: ignore[override]
        self, X: Any, Y: Any, **kwargs: Any
    ) -> AbstractFullyBayesianSingleTaskGP:
        """Conditions on additional observations for a Fully Bayesian model (either
        identical across models or unique per-model).

        Args:
            X: A ``batch_shape x num_samples x d``-dim Tensor, where ``d`` is
                the dimension of the feature space and ``batch_shape`` is the number of
                sampled models.
            Y: A ``batch_shape x num_samples x 1``-dim Tensor, where ``d`` is
                the dimension of the feature space and ``batch_shape`` is the number of
                sampled models.

        Returns:
            A fully bayesian model conditioned on given observations. The
            returned model has ``batch_shape`` copies of the training data in
            case of identical observations (and ``batch_shape`` training
            datasets otherwise).

        As in BoTorch, ``X`` is in the model's transformed input space (e.g.
        ``model.transform_inputs(X)``) and ``Y`` in the original outcome
        space; observation noise ``noise`` (outcome-transformed) of
        identical observations is repeated for every sample like ``X`` and
        ``Y`` (BoTorch fails, see the draft).
        """
        self._check_if_fitted()
        x = to_flat(X, "X")
        y = to_flat(Y, "Y")
        xs, ys = tuple(x[1]), tuple(y[1])
        noise = kwargs.get("noise")
        if len(xs) == 2 and len(ys) == 2:
            num = self.num_mcmc_samples
            X = from_flat(*_repeat(x[0], xs, num), kind_of(X))
            Y = from_flat(*_repeat(y[0], ys, num), kind_of(Y))
            if noise is not None:
                # BoTorch repeats X and Y but not the noise (and fails, see
                # the draft).
                nz = to_flat(noise, "noise")
                if len(nz[1]) == 2:
                    kwargs["noise"] = from_flat(*_repeat(nz[0], tuple(nz[1]), num), kind_of(noise))
        elif len(xs) < len(ys):
            reps = math.prod(ys[:-2])
            X = from_flat(list(x[0]) * reps, (*ys[:-2], *xs[-2:]), kind_of(X))
        return SingleTaskGP.condition_on_observations(self, X, Y, **kwargs)  # type: ignore[return-value]

    # -- ensembles -----------------------------------------------------------

    def _ensemble_members(self) -> list[SingleTaskGP]:
        """The MCMC samples as unbatched ``SingleTaskGP`` views."""
        self._check_if_fitted()
        return [self._member(k) for k in range(self.num_mcmc_samples)]

    def _member(self, k: int) -> SingleTaskGP:
        saved = {
            key: self.__dict__.pop(key)
            for key in list(self.__dict__)
            if key.startswith("_fb_") or key == "pyro_model"
        }
        try:
            return SingleTaskGP._member(self, k)
        finally:
            self.__dict__.update(saved)

    def _ensemble_acquisition(self, sampler: Any) -> tuple[list[Any], list[float] | None, Any]:
        """The members, their (equal) weights and the sampler shared by the
        members' acquisition functions."""
        from botorch_rs.acquisition.monte_carlo import DEFAULT_MC_SAMPLES
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_MC_SAMPLES,))
        return self._ensemble_members(), None, sampler

    @classmethod
    def construct_inputs(  # type: ignore[override]
        cls,
        training_data: Any,
        *,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
    ) -> dict[str, Any]:
        r"""Construct ``SingleTaskGP`` keyword arguments from a ``SupervisedDataset``.

        Args:
            training_data: A ``SupervisedDataset``, with attributes ``train_X``,
                ``train_Y``, and, optionally, ``train_Yvar``.
            use_input_warping: A boolean indicating whether to use input warping.
            indices_to_warp: An optional list of indices to warp. The default
                is to warp all inputs.

        Returns:
            A dict of keyword arguments that can be used to initialize a
            ``FullyBayesianLinearSingleTaskGP``, with keys ``train_X``, ``train_Y``,
            ``use_input_warping``, ``indices_to_warp``, and, optionally, ``train_Yvar``.
        """
        return {
            **SingleTaskGP.construct_inputs(training_data=training_data),
            "use_input_warping": use_input_warping,
            "indices_to_warp": indices_to_warp,
        }

    def __repr__(self) -> str:
        fitted = "unfitted" if self.covar_module is None else f"{self.num_mcmc_samples} samples"
        n, d = self._fb_train_X[1]
        return f"{type(self).__name__}(n={n}, d={d}, {fitted})"


def _median(values: list[float]) -> float:
    """``torch.median``: the lower of the two middle values."""
    s = sorted(values)
    return s[(len(s) - 1) // 2]


class FullyBayesianSingleTaskGP(AbstractFullyBayesianSingleTaskGP):
    r"""A fully Bayesian single-task GP model.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``. A model with a Matern-5/2 kernel
    and dimension-scaled priors on the hyperparameters from [Hvarfner2024vanilla]_
    is used by default.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> fully_bayesian_gp = FullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(fully_bayesian_gp)
        >>> posterior = fully_bayesian_gp.posterior(test_X)
    """

    _pyro_model_class: type[PyroModel] = MaternPyroModel

    @property
    def median_lengthscale(self) -> Any:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        covar = self.covar_module
        base_kernel = covar.base_kernel if isinstance(covar, ScaleKernel) else covar
        data, shape = to_flat(base_kernel.lengthscale, "lengthscale")
        num, d = shape[0], shape[-1]
        out = [_median([float(data[s * d + k]) for s in range(num)]) for k in range(d)]
        return from_flat(out, (d,), self._kind)


class SaasFullyBayesianSingleTaskGP(FullyBayesianSingleTaskGP):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``. The SAAS model
    [Eriksson2021saasbo]_ with a Matern-5/2 kernel is used by default.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model
    as it isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    _pyro_model_class: type[PyroModel] = SaasPyroModel


class FullyBayesianLinearSingleTaskGP(AbstractFullyBayesianSingleTaskGP):
    r"""A fully Bayesian single-task GP model with a linear kernel.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> gp = FullyBayesianLinearSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
        >>> posterior = gp.posterior(test_X)
    """

    _pyro_model_class: type[PyroModel] = LinearPyroModel

    @property
    def median_weight_variance(self) -> Any:
        r"""Median weight variance across the MCMC samples."""
        self._check_if_fitted()
        data, shape = to_flat(self.covar_module.variance, "variance")
        num, d = shape[0], shape[-1]
        out = [_median([float(data[s * d + k]) for s in range(num)]) for k in range(d)]
        return from_flat(out, (d,), self._kind)


__all__ = [
    "MCMC_DIM",
    "AbstractFullyBayesianSingleTaskGP",
    "FullyBayesianLinearSingleTaskGP",
    "FullyBayesianSingleTaskGP",
    "LinearPyroModel",
    "MaternPyroModel",
    "PyroModel",
    "SaasFullyBayesianSingleTaskGP",
    "SaasPyroModel",
    "compute_dists",
    "linear_kernel",
    "matern52_kernel",
    "reshape_and_detach",
]
