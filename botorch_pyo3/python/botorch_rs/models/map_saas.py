"""MAP SAAS models (``botorch.models.map_saas``): the sparse axis-aligned
subspace (SAAS) prior on the inverse lengthscales of a kernel, fitted by
maximum a posteriori estimation instead of NUTS.

The SAAS prior (a half-Cauchy prior ``HC(1)`` on ``tau / lengthscale**2``,
with the global shrinkage ``tau`` fixed or learned under a ``HC(0.1)``
prior), the log-scale constraints (``LogTransformedInterval``) and the
priors of the mean and noise are evaluated in Rust as part of the kernel
tree and the model, with exact gradients; fits run BoTorch's L-BFGS-B
procedure (one problem per ensemble member for
``EnsembleMapSaasSingleTaskGP``).

References

.. [Daulton2026bonsai]
    S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. BONSAI: Bayesian
    Optimization with Natural Simplicity and Interpretability. Advances in Neural
    Information Processing Systems 39 (NeurIPS), 2026.
"""

from __future__ import annotations

import importlib
import math
import random
import struct
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.constraints import Interval
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel, _check_batch_shape
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models._backend_selection import requires_tensor_backend
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.transforms.input import InputTransform
from botorch_rs.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch_rs.posteriors.fully_bayesian import MCMC_DIM, GaussianMixturePosterior
from botorch_rs.priors import GammaPrior, HalfCauchyPrior, NormalPrior, _float32
from botorch_rs.utils.constraints import LogTransformedInterval
from botorch_rs.utils.types import DEFAULT

EPS = 1e-8


def _use_torch_backend(backend: str, *values: Any) -> bool:
    if backend not in ("auto", "native", "torch"):
        raise ValueError("backend must be 'auto', 'native', or 'torch'.")
    torch = sys.modules.get("torch")
    return backend == "torch" or (
        backend == "auto"
        and torch is not None
        and any(
            requires_tensor_backend(value)
            or (
                isinstance(value, torch.Tensor)
                and (value.device.type != "cpu" or value.requires_grad)
            )
            for value in values
        )
    )


def _double_half_cauchy(scale: float) -> HalfCauchyPrior:
    """``HalfCauchyPrior(torch.tensor(scale, dtype=torch.float64))``: the
    priors of ``add_saas_prior`` are created in the kernel's dtype, which is
    double precision in BoTorch's MAP SAAS models (and in botorch_rs)."""
    prior = HalfCauchyPrior.__new__(HalfCauchyPrior)
    prior.scale = float(scale)
    prior._rs = _rs.Prior.half_cauchy(float(scale), False)
    return prior


def _half_cauchy_samples(scale: float, shape: tuple[int, ...]) -> list[float]:
    """``HalfCauchy(scale).sample(shape)`` (float32 values) from torch's global
    generator, as BoTorch draws its taus; Python's ``random`` without torch."""
    n = math.prod(shape)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return [abs(scale * math.tan(math.pi * (random.random() - 0.5))) for _ in range(n)]
    dist = torch.distributions.HalfCauchy(torch.tensor(scale))
    return [float(v) for v in dist.sample(torch.Size(shape)).reshape(-1).tolist()]


class _SaasSpec:
    """The SAAS prior of a kernel (see :func:`add_saas_prior`)."""

    def __init__(self, prior: HalfCauchyPrior, fixed_taus: list[float] | None) -> None:
        self.prior = prior
        self.fixed_taus = fixed_taus

    def apply(self, kernel: Kernel, node: _rs.KernelNode) -> _rs.KernelNode:
        if self.fixed_taus is not None:
            return node.with_saas_prior(self.prior._rs, tau=self.fixed_taus[0])
        constraint = kernel.raw_tau_constraint  # type: ignore[attr-defined]
        tau_prior = kernel.tau_prior  # type: ignore[attr-defined]
        return node.with_saas_prior(
            self.prior._rs, tau_constraint=constraint._rs, tau_prior=tau_prior._rs
        )


class SaasPriorHelper:
    """Helper class for specifying parameter and setting closures."""

    def __init__(self, tau: Any = None) -> None:
        """Instantiates a new helper object.

        Args:
            tau: Value of the global shrinkage parameter. If ``None``, the tau will be
                a free parameter and inferred from the data.
                Tau can be a tensor for batched models, like
                ``EnsembleMapSaasSingleTaskGP``, where each batch has a different
                sparsity prior. If tau is a tensor, it must have shape ``batch_shape``.
        """
        self._tau = tau

    def _tau_values(self, m: Kernel) -> tuple[list[float], tuple[int, ...]]:
        if self._tau is not None:
            data, shape = to_flat(self._tau, "tau")
            return list(data), tuple(shape)
        data, shape = to_flat(self.tau_prior_param_or_closure(m), "tau")
        return list(data), tuple(shape)

    def tau(self, m: Any) -> Any:
        """The global shrinkage parameter ``tau``.

        Args:
            m: A kernel object equipped with a lengthscale.

        Returns:
            The global shrinkage parameter of the SAAS prior.
        """
        if getattr(m, "_tensor_backend", None) == "torch":
            if self._tau is None:
                return m.tau
            torch = importlib.import_module("torch")
            return torch.as_tensor(self._tau).to(m.lengthscale)
        data, shape = self._tau_values(m)
        return from_flat(data, shape, kind_of(m.lengthscale))

    def inv_lengthscale_prior_param_or_closure(self, m: Any) -> Any:
        """Closure to compute the scaled inverse lengthscale parameter (``tau / l^2``)
        to which the SAAS prior is applied.

        Args:
            m: A kernel object equipped with a lengthscale.

        Returns:
            The scaled inverse lengthscale parameter.
        """
        if getattr(m, "_tensor_backend", None) == "torch":
            return self.tau(m)[..., None, None] / m.lengthscale.square()
        ls, shape = to_flat(m.lengthscale, "lengthscale")
        taus, _ = self._tau_values(m)
        d = shape[-1]
        out = [taus[(i // d) % len(taus)] / (v * v) for i, v in enumerate(ls)]
        return from_flat(out, tuple(shape), kind_of(m.lengthscale))

    def inv_lengthscale_prior_setting_closure(self, m: Any, value: Any) -> None:
        """Closure to set the inverse lengthscale prior parameter.

        Args:
            m: A kernel object equipped with a lengthscale.
            value: The value of the scaled inverse lengthscale parameter,
                (``tau / l^2``), used to recover and set the lengthscale of
                the kernel.
        """
        if getattr(m, "_tensor_backend", None) == "torch":
            torch = importlib.import_module("torch")
            c = m.raw_lengthscale_constraint
            m.lengthscale = (
                (self.tau(m)[..., None, None] / torch.as_tensor(value).to(m.lengthscale))
                .sqrt()
                .clamp(c.lower_bound + EPS, c.upper_bound - EPS)
            )
            return
        values, _ = to_flat(value, "value")
        _, shape = to_flat(m.lengthscale, "lengthscale")
        taus, _ = self._tau_values(m)
        c = m.raw_lengthscale_constraint
        lb, ub = c.lower_bound, c.upper_bound
        d = shape[-1]
        new = [
            min(max(math.sqrt(taus[(i // d) % len(taus)] / v), lb + EPS), ub - EPS)
            for i, v in enumerate(values)
        ]
        m.lengthscale = from_flat(new, tuple(shape), kind_of(m.lengthscale))

    def tau_prior_param_or_closure(self, m: Any) -> Any:
        """Closure to compute the global shrinkage parameter ``tau``.

        Args:
            m: A kernel object equipped with a ``raw_tau`` parameter.

        Returns:
            The transformed global shrinkage parameter ``tau``.
        """
        if getattr(m, "_tensor_backend", None) == "torch":
            return m.raw_tau_constraint.transform(m.raw_tau)
        raw, shape = to_flat(m.raw_tau, "raw_tau")
        c = m.raw_tau_constraint  # type: ignore[attr-defined]
        return from_flat([c.transform(r) for r in raw], tuple(shape), kind_of(m.raw_tau))

    def tau_prior_setting_closure(self, m: Any, value: Any) -> None:
        """Closure to set the global shrinkage parameter ``tau``.

        Args:
            m: A kernel object equipped with a ``raw_tau`` parameter.
            value: The value of the global shrinkage parameter.
        """
        if getattr(m, "_tensor_backend", None) == "torch":
            m.tau = value
            return
        c = m.raw_tau_constraint  # type: ignore[attr-defined]
        values, shape = to_flat(value, "value")
        lb, ub = c.lower_bound, c.upper_bound
        raw = [c.inverse_transform(min(max(v, lb + EPS), ub - EPS)) for v in values]
        m.raw_tau = from_flat(raw, tuple(shape), kind_of(value))


def add_saas_prior(
    base_kernel: Any,
    tau: Any = None,
    log_scale: bool = True,
) -> Any:
    """Add a SAAS prior to a given base_kernel.

    The SAAS prior is given by tau / lengthscale^2 ~ HC(1.0). If tau is None,
    we place an additional HC(0.1) prior on tau similar to the original SAAS prior
    that relies on inference with NUTS.

    Example:
        >>> matern_kernel = MaternKernel(...)
        >>> add_saas_prior(matern_kernel, tau=None)  # Add a SAAS prior

    Args:
        base_kernel: Base kernel that has a lengthscale and uses ARD.
            Note that this function modifies the kernel object in place.
        tau: Value of the global shrinkage. If ``None``, infer the global
            shrinkage parameter. Can be a tensor for batched models (e.g.,
            ensembles) where each batch has a different sparsity prior.
        log_scale: Set to ``True`` if the lengthscale and tau should be optimized on
            a log-scale without any domain rescaling. That is, we will learn
            ``raw_lengthscale := log(lengthscale)`` and this hyperparameter needs to
            satisfy the corresponding bound constraints. Setting this to ``True`` will
            generally improve the numerical stability, but requires an optimizer that
            can handle bound constraints, e.g., L-BFGS-B.

    Returns:
        Base kernel with SAAS priors added.

    botorch_rs supports the SAAS prior on ``RBFKernel`` and ``MaternKernel``
    (BoTorch: any kernel with lengthscales), before the kernel is part of a
    model. The priors are created in double precision (BoTorch: in the
    kernel's dtype, double precision in its MAP SAAS models).
    """
    if getattr(base_kernel, "_tensor_backend", None) == "torch":
        return base_kernel.configure_saas_prior(tau, log_scale)
    if not getattr(base_kernel, "has_lengthscale", False):
        raise UnsupportedError("base_kernel must have lengthscale(s)")
    if getattr(base_kernel, "lengthscale_prior", None) is not None:
        raise UnsupportedError("base_kernel must not specify a lengthscale prior")
    if not isinstance(base_kernel, MaternKernel) and type(base_kernel).__name__ != "RBFKernel":
        raise UnsupportedError(
            "botorch_rs supports the SAAS prior on RBFKernel and MaternKernel, got "
            f"{type(base_kernel).__name__}."
        )
    if base_kernel._model is not None:
        raise UnsupportedError(
            "add_saas_prior must be called before the kernel is part of a model in botorch_rs."
        )
    batch_shape = tuple(base_kernel._param_shape("raw_lengthscale")[:-2])
    fixed: list[float] | None = None
    if tau is not None:
        data, shape = to_flat(tau, "tau")
        if not isinstance(tau, (int, float)) and tuple(shape) != batch_shape:
            raise ValueError(
                f"Expected tau to have shape {batch_shape} matching the batch shape "
                f"of the base kernel. Got {tuple(shape)}."
            )
        fixed = [float(v) for v in data]
        if isinstance(tau, (int, float)) and _float32():
            # BoTorch: `torch.as_tensor(tau)` (torch's default dtype).
            fixed = [float(struct.unpack("f", struct.pack("f", v))[0]) for v in fixed]
    interval_class: Any = LogTransformedInterval if log_scale else Interval
    base_kernel.raw_lengthscale_constraint = interval_class(0.01, 1e4, initial_value=1)
    # GPyTorch re-initializes the parameter to the new constraint's initial value.
    base_kernel._local.pop("raw_lengthscale", None)
    if tau is None:  # Place a HC(0.1) prior on tau
        base_kernel.raw_tau_constraint = interval_class(1e-3, 10, initial_value=0.1)  # type: ignore[attr-defined]
        base_kernel.tau_prior = _double_half_cauchy(0.1)  # type: ignore[attr-defined]
        base_kernel._extra_params = (("raw_tau", "raw_tau_constraint", "tau_prior"),)  # type: ignore[attr-defined]
        base_kernel._local.pop("raw_tau", None)
    # Place a HC(1) prior on tau / lengthscale^2
    base_kernel.inv_lengthscale_prior = _double_half_cauchy(1.0)  # type: ignore[attr-defined]
    base_kernel._saas = _SaasSpec(base_kernel.inv_lengthscale_prior, fixed)  # type: ignore[attr-defined]
    base_kernel._saas_helper = SaasPriorHelper(tau=tau)  # type: ignore[attr-defined]
    return base_kernel


def get_mean_module_with_normal_prior(batch_shape: Any = None) -> ConstantMean:
    """Return constant mean with a N(0, 1) prior constrained to [-10, 10].

    This prior assumes the outputs (targets) have been standardized to have zero mean
    and unit variance.

    Args:
        batch_shape: Optional batch shape for the constant-mean module.

    Returns:
        ConstantMean module.
    """
    return ConstantMean(
        constant_prior=NormalPrior(loc=0.0, scale=1.0),
        constant_constraint=Interval(-10, 10, initial_value=0, transform=None),
        batch_shape=_check_batch_shape(batch_shape),
    )


def get_gaussian_likelihood_with_gamma_prior(batch_shape: Any = None) -> GaussianLikelihood:
    """Return Gaussian likelihood with a Gamma(0.9, 10) prior.

    This prior prefers small noise, but also has heavy tails.

    Args:
        batch_shape: Batch shape for the likelihood.

    Returns:
        GaussianLikelihood with Gamma(0.9, 10) prior constrained to [1e-4, 0.1].
    """
    return GaussianLikelihood(
        noise_prior=GammaPrior(0.9, 10.0),
        noise_constraint=LogTransformedInterval(1e-4, 1, initial_value=1e-2),
        batch_shape=_check_batch_shape(batch_shape),
    )


def _ard_num_dims(train_X: Any, input_transform: InputTransform | None) -> int:
    # The input transforms of botorch_rs keep the input dimension.
    return to_flat(train_X, "train_X")[1][-1]


def get_map_saas_model(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any = None,
    input_transform: InputTransform | None = None,
    outcome_transform: Any = None,
    tau: Any = None,
    *,
    backend: str = "auto",
) -> SingleTaskGP:
    """Helper method for creating an unfitted MAP SAAS model.

    Args:
        train_X: Tensor of shape ``n x d`` with training inputs.
        train_Y: Tensor of shape ``n x 1`` with training targets.
        train_Yvar: Optional tensor of shape ``n x 1`` with observed noise,
            inferred if None.
        input_transform: An optional input transform.
        outcome_transform: An optional outcome transform.
        tau: Fixed value of the global shrinkage tau. If None, the model
            places a HC(0.1) prior on tau and infers it. Can be a tensor
            for batched models where each batch has a different sparsity prior.

    Returns:
        A SingleTaskGP with a Matern kernel and a SAAS prior.
    """
    if _use_torch_backend(
        backend, train_X, train_Y, train_Yvar, tau, input_transform, outcome_transform
    ):
        from botorch_rs.models._torch_backend_map_saas import get_torch_map_saas_model

        return get_torch_map_saas_model(
            train_X, train_Y, train_Yvar, input_transform, outcome_transform, tau
        )
    _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(train_X=train_X, train_Y=train_Y)
    mean_module = get_mean_module_with_normal_prior(batch_shape=aug_batch_shape)
    ard_num_dims = _ard_num_dims(train_X, input_transform)
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, batch_shape=aug_batch_shape)
    add_saas_prior(base_kernel=base_kernel, tau=tau)
    covar_module = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
        batch_shape=aug_batch_shape,
    )
    likelihood = (
        get_gaussian_likelihood_with_gamma_prior(batch_shape=aug_batch_shape)
        if train_Yvar is None
        else None
    )
    return SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        mean_module=mean_module,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )


def get_additive_map_saas_covar_module(
    ard_num_dims: int,
    num_taus: int = 4,
    active_dims: tuple[int, ...] | None = None,
    batch_shape: Any = None,
    dtype: Any = None,
    device: Any = None,
    *,
    backend: str = "auto",
) -> Any:
    """Return an additive map SAAS covar module.

    The constructed kernel is an additive kernel with ``num_taus`` terms. Each term is a
    scaled Matern kernel with a SAAS prior and a tau sampled from a HalfCauchy(0, 1)
    distribution.

    Args:
        ard_num_dims: The number of inputs dimensions.
        num_taus: The number of taus to use (4 if omitted).
        active_dims: Active dims for the covar module. The kernel will be evaluated
            only using these columns of the input tensor.
        batch_shape: Batch shape for the covar module.

    Returns:
        An additive MAP SAAS covar module.

    The taus are drawn from torch's global generator (float32 samples of
    ``HalfCauchy(0.1)``, as BoTorch's), or from Python's ``random`` module
    without torch.
    """
    if backend == "torch" or (backend == "auto" and (dtype is not None or device is not None)):
        torch = importlib.import_module("torch")
        from botorch_rs.models._torch_backend_map_saas import build_additive_saas_kernel

        return build_additive_saas_kernel(
            ard_num_dims,
            num_taus,
            torch.Size(batch_shape or ()),
            torch.empty(0, dtype=dtype, device=device),
            active_dims,
        )
    _use_torch_backend(backend)
    shape = _check_batch_shape(batch_shape)
    kernels: list[Kernel] = []
    for _ in range(num_taus):
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=shape,
            active_dims=active_dims,
        )
        taus = _half_cauchy_samples(0.1, shape)
        add_saas_prior(base_kernel=base_kernel, tau=from_flat(taus, shape, kind_of(0.0)))
        scaled_kernel = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
            batch_shape=shape,
        )
        kernels.append(scaled_kernel)
    return AdditiveKernel(*kernels)


class AdditiveMapSaasSingleTaskGP(SingleTaskGP):
    """An additive MAP SAAS single-task GP.

    This is a maximum-a-posteriori (MAP) version of sparse axis-aligned subspace BO
    (SAASBO). We create an additive kernel where each kernel in the sum is a
    Matern-5/2 kernel with a SAAS prior and a separate outputscale. The sparsity
    level for each kernel is sampled from an HC(0.1) distribution leading to a mix
    of sparsity levels. We learn all the hyperparameters using MAP inference.
    """

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if cls is AdditiveMapSaasSingleTaskGP and _use_torch_backend(
            backend,
            train_X,
            train_Y,
            *args[:1],
            *(kwargs.get(name) for name in ("train_Yvar", "input_transform", "outcome_transform")),
        ):
            from botorch_rs.models._torch_backend_map_saas import TorchAdditiveMapSaasSingleTaskGP

            return object.__new__(TorchAdditiveMapSaasSingleTaskGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: InputTransform | None = None,
        num_taus: int = 4,
        *,
        backend: str = "auto",
    ) -> None:
        """Instantiates an AdditiveMapSaasSingleTaskGP.

        Args:
            train_X: A ``batch_shape x n x d`` tensor of training features.
            train_Y: A ``batch_shape x n x m`` tensor of training observations.
            train_Yvar: A ``batch_shape x n x m`` tensor of observed noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale). We use a
                ``Standardize`` transform if no ``outcome_transform`` is specified.
                Pass down ``None`` to use no outcome transform.
            input_transform: An optional input transform.
            num_taus: The number of taus to use (4 if omitted).
        """
        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)
        mean_module = get_mean_module_with_normal_prior(batch_shape=aug_batch_shape)
        likelihood = (
            get_gaussian_likelihood_with_lognormal_prior(batch_shape=aug_batch_shape)
            if train_Yvar is None
            else None
        )
        covar_module = get_additive_map_saas_covar_module(
            ard_num_dims=_ard_num_dims(train_X, input_transform),
            num_taus=num_taus,
            batch_shape=aug_batch_shape,
        )
        SingleTaskGP.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )


class EnsembleMapSaasSingleTaskGP(SingleTaskGP):
    """An ensemble of ``num_taus`` MAP SAAS GPs with different sparsity levels
    (``taus``) [Daulton2026bonsai]_: a batched ``SingleTaskGP`` whose
    posterior is a ``GaussianMixturePosterior`` over its members, so that
    acquisition functions average over the ensemble.
    """

    _is_ensemble = True
    _supports_batched_models = False

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if cls is EnsembleMapSaasSingleTaskGP and _use_torch_backend(
            backend,
            train_X,
            train_Y,
            *args[:1],
            *(
                kwargs.get(name)
                for name in ("train_Yvar", "taus", "input_transform", "outcome_transform")
            ),
        ):
            from botorch_rs.models._torch_backend_map_saas import TorchEnsembleMapSaasSingleTaskGP

            return object.__new__(TorchEnsembleMapSaasSingleTaskGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        num_taus: int = 4,
        taus: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: InputTransform | None = None,
        *,
        backend: str = "auto",
    ) -> None:
        """Instantiates an ``EnsembleMapSaasSingleTaskGP``, which is a batched
        ensemble of ``SingleTaskGP``s with the Matern-5/2 kernel and a SAAS prior.
        The model is intended to be trained with ``ExactMarginalLogLikelihood`` and
        ``fit_gpytorch_mll``. Under the hood, the model is equivalent to a
        multi-output ``BatchedMultiOutputGPyTorchModel``, but it produces a
        ``GaussianMixturePosterior``, which leads to ensembling of the model outputs.

        Args:
            train_X: An ``n x d`` tensor of training features.
            train_Y: An ``n x 1`` tensor of training observations.
            train_Yvar: An optional ``n x 1`` tensor of observed measurement noise.
            num_taus: The number of taus to use (4 if omitted). Each tau is
                a sparsity parameter for the corresponding kernel in the ensemble.
            taus: An optional tensor of shape ``num_taus`` containing the taus to use.
                If omitted, the taus are sampled from a HalfCauchy(0.1) distribution.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale). We use a
                ``Standardize`` transform if no ``outcome_transform`` is specified.
                Pass down ``None`` to use no outcome transform.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        if taus is None:
            tau_values = _half_cauchy_samples(0.1, (num_taus,))
            taus = from_flat(tau_values, (num_taus,), kind_of(train_X))
        else:
            tflat = to_flat(taus, "taus")
            tau_values, tshape = [float(v) for v in tflat[0]], tflat[1]
            if tuple(tshape) != (num_taus,):
                raise ValueError(
                    f"Expected taus to be of shape {[num_taus]}. Got taus.shape={tuple(tshape)}."
                )
        xdata, xshape = to_flat(train_X, "train_X")
        ydata, yshape = to_flat(train_Y, "train_Y")
        if yshape[-1] != 1:
            raise UnsupportedError(
                f"EnsembleMapSAASGP only supports single-output. Got train_Y.shape={tuple(yshape)}."
            )
        if len(xshape) != 2:
            raise UnsupportedError(
                f"EnsembleMapSAASGP only supports 2D inputs. Got train_X.ndim={len(xshape)}."
            )
        kind = kind_of(train_X)
        # Add batch dimension for ensemble.
        rep_X = from_flat(list(xdata) * num_taus, (num_taus, *xshape), kind)
        rep_Y = from_flat(list(ydata) * num_taus, (num_taus, *yshape), kind_of(train_Y))
        rep_Yvar = None
        if train_Yvar is not None:
            vdata, vshape = to_flat(train_Yvar, "train_Yvar")
            rep_Yvar = from_flat(list(vdata) * num_taus, (num_taus, *vshape), kind_of(train_Yvar))
        batch_shape = (num_taus,)
        mean_module = get_mean_module_with_normal_prior(batch_shape=batch_shape)
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=xshape[-1], batch_shape=batch_shape)
        add_saas_prior(base_kernel=base_kernel, tau=taus)
        covar_module = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
            batch_shape=batch_shape,
        )
        likelihood = (
            get_gaussian_likelihood_with_gamma_prior(batch_shape=batch_shape)
            if train_Yvar is None
            else None
        )
        super().__init__(
            train_X=rep_X,
            train_Y=rep_Y,
            train_Yvar=rep_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.taus = taus

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
            A ``GaussianMixturePosterior`` object (``batch_shape x num_taus x q
            x m``). Includes observation noise if specified.
        """
        if any(self._rs.transformed_outputs):
            # BoTorch fails here too (a `TransformedPosterior` has no
            # `distribution` for the `GaussianMixturePosterior`).
            raise UnsupportedError(
                "EnsembleMapSaasSingleTaskGP supports the Standardize outcome transform only: "
                "a nonlinear outcome transform makes the posterior a TransformedPosterior, which "
                "has no Gaussian mixture."
            )
        data, shape = to_flat(X, "X")
        shape = tuple(shape)
        Xu = from_flat(data, (*shape[:-2], 1, *shape[-2:]), kind_of(X))
        posterior = super().posterior(
            X=Xu,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        return GaussianMixturePosterior._from_posterior(posterior)

    @classmethod
    def construct_inputs(cls, training_data: Any, *, num_taus: int = 4) -> dict[str, Any]:
        r"""Construct ``Model`` keyword arguments from a dict of ``SupervisedDataset``.

        Args:
            training_data: A ``SupervisedDataset`` containing the training data.
            num_taus: Number of taus to use in the ensemble (4 if omitted).
        """
        base_inputs = super().construct_inputs(training_data=training_data)
        return {**base_inputs, "num_taus": num_taus}

    def _ensemble_members(self) -> list[SingleTaskGP]:
        """The members of the ensemble as unbatched ``SingleTaskGP`` views."""
        return [self._member(k) for k in range(self.batch_shape[0])]

    def _ensemble_acquisition(self, sampler: Any) -> tuple[list[Any], list[float] | None, Any]:
        """The members, their (equal) weights and the sampler shared by the
        members' acquisition functions (BoTorch's base samples are shared by
        the batch of member posteriors)."""
        from botorch_rs.acquisition.monte_carlo import DEFAULT_MC_SAMPLES
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_MC_SAMPLES,))
        return self._ensemble_members(), None, sampler


__all__ = [
    "MCMC_DIM",
    "AdditiveMapSaasSingleTaskGP",
    "EnsembleMapSaasSingleTaskGP",
    "SaasPriorHelper",
    "add_saas_prior",
    "get_additive_map_saas_covar_module",
    "get_gaussian_likelihood_with_gamma_prior",
    "get_map_saas_model",
    "get_mean_module_with_normal_prior",
]
