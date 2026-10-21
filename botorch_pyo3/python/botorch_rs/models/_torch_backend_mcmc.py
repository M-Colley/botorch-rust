"""Device-resident fully Bayesian potentials and multinomial No-U-Turn sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch import Tensor

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError

_NOISE_FLOOR = 1e-4


@dataclass(frozen=True)
class _Site:
    name: str
    shape: tuple[int, ...]
    prior: str
    parameters: tuple[float, ...]
    positive: bool = True

    @property
    def size(self) -> int:
        return math.prod(self.shape)


class TorchMCMCPotential:
    """Negative joint density in NumPyro's sorted unconstrained site layout.

    GP algebra and all continuous states stay on the input device. Positive
    sites use exp with its log-Jacobian; the parameter ``noise`` excludes the
    additive likelihood floor until prediction samples are postprocessed.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        kernel: str = "saas",
        outputscale_prior: Any = "default",
        warp_indices: list[int] | None = None,
        warp_eps: float = 1e-7,
        input_transform: Any = None,
        outcome_transform: Any = None,
        task: tuple[int, int, int] | None = None,
    ) -> None:
        if kernel not in ("matern", "saas", "linear"):
            raise UnsupportedError(f"Unknown Torch MCMC kernel {kernel!r}.")
        if not isinstance(train_X, Tensor) or not isinstance(train_Y, Tensor):
            raise TypeError("Torch MCMC training data must be Tensors.")
        if train_X.ndim != 2 or train_Y.shape != (train_X.shape[0], 1) or train_X.shape[-1] == 0:
            raise ValueError("Expected train_X n x d and train_Y n x 1 with nonempty features.")
        for name, value in (("train_X", train_X), ("train_Y", train_Y), ("train_Yvar", train_Yvar)):
            if value is None:
                continue
            if (
                not isinstance(value, Tensor)
                or value.dtype != train_X.dtype
                or value.device != train_X.device
            ):
                raise TypeError(f"{name} must share the training dtype and device.")
            if value.dtype not in (torch.float32, torch.float64) or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must contain finite float32/float64 values.")
        if train_Yvar is not None and (
            train_Yvar.shape != train_Y.shape or bool((train_Yvar < 0).any())
        ):
            raise ValueError("train_Yvar must be nonnegative and match train_Y.shape.")
        columns = train_X.shape[-1]
        self.task = task
        if task is not None:
            feature, tasks, rank = task
            if (
                type(feature) is not int
                or not 0 <= feature < columns
                or type(tasks) is not int
                or tasks < 1
                or type(rank) is not int
                or not 1 <= rank <= tasks
            ):
                raise ValueError("Invalid task feature, count or latent rank.")
            labels = train_X[..., feature]
            if not bool(((labels == labels.long()) & (labels >= 0) & (labels < tasks)).all()):
                raise ValueError("Potential task labels must be mapped to [0, num_tasks).")
            self.task_indices = labels.long()
            self.base_indices = torch.tensor(
                [i for i in range(columns) if i != feature], device=train_X.device, dtype=torch.long
            )
        d = columns - int(task is not None)
        if d < 1:
            raise ValueError("MCMC needs at least one continuous feature.")
        if warp_indices is not None and (
            not warp_indices
            or len(set(warp_indices)) != len(warp_indices)
            or any(
                type(index) is not int
                or not 0 <= index < columns
                or (task is not None and index == task[0])
                for index in warp_indices
            )
            or not math.isfinite(warp_eps)
            or not 0 < warp_eps < 0.5
        ):
            raise ValueError("Warp columns must be unique in-range features with 0 < eps < .5.")
        self.train_X, self.train_Y, self.train_Yvar = train_X, train_Y, train_Yvar
        self.input_transform, self.outcome_transform = input_transform, outcome_transform
        self.kernel, self.warp_eps = kernel, warp_eps
        self.warp_indices = (
            None
            if warp_indices is None
            else torch.tensor(warp_indices, device=train_X.device, dtype=torch.long)
        )
        if outputscale_prior == "default":
            outputscale_prior = (2.0, 0.15) if kernel == "saas" else None
        if outputscale_prior is not None and (
            len(outputscale_prior) != 2
            or any(not math.isfinite(v) or v <= 0 for v in outputscale_prior)
        ):
            raise ValueError(
                "Outputscale Gamma concentration and rate must be finite positive scalars."
            )
        sites = [
            _Site("mean", () if task is None else (task[1],), "normal", (0.0, 1.0), positive=False)
        ]
        if task is not None:
            sites.extend(
                (
                    _Site("latent_features", (task[1], task[2]), "normal", (0.0, 1.0), False),
                    _Site("task_lengthscale", (task[2],), "gamma", (6.0, 3.0)),
                )
            )
        if kernel == "matern":
            sites.append(
                _Site(
                    "lengthscale",
                    (d,),
                    "lognormal",
                    (math.sqrt(2) + 0.5 * math.log(d), math.sqrt(3)),
                )
            )
        else:
            sites.extend(
                [
                    _Site(
                        "_kernel_inv_length_sq" if kernel == "saas" else "_weight_variance_sq",
                        (d,),
                        "half_cauchy",
                        (1.0,),
                    ),
                    _Site(
                        "kernel_tausq" if kernel == "saas" else "tau_sq", (), "half_cauchy", (0.1,)
                    ),
                ]
            )
        if train_Yvar is None:
            sites.append(_Site("noise", (), "gamma", (0.9, 10.0)))
        if kernel != "linear" and outputscale_prior is not None:
            sites.append(_Site("outputscale", (), "gamma", tuple(outputscale_prior)))
        if warp_indices is not None:
            sites.extend(
                _Site(f"c{i}", (len(warp_indices),), "lognormal", (0.0, math.sqrt(0.1)))
                for i in (0, 1)
            )
        self._sites = sorted(sites, key=lambda site: site.name)
        self.dim = sum(site.size for site in self._sites)

    @classmethod
    def from_model(cls, model: Any) -> TorchMCMCPotential:
        from botorch_rs.models.fully_bayesian import LinearPyroModel, MaternPyroModel, SaasPyroModel
        from botorch_rs.models.fully_bayesian_multitask import LatentFeatureMultiTaskPyroMixin

        pyro = model.pyro_model
        multitask = isinstance(pyro, LatentFeatureMultiTaskPyroMixin)
        if (
            not isinstance(pyro, (MaternPyroModel, SaasPyroModel, LinearPyroModel))
            or pyro._prior_mode
        ):
            raise UnsupportedError(
                "Torch NUTS implements the observed-data scalar Matern, SAAS and Linear models."
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
            "_prepare_features",
            "_maybe_input_warp",
            "_warp_jax",
            "_maybe_multitask_transform",
            "sample_latent_features",
            "sample_task_lengthscale",
            "_build_task_covar",
            "_get_task_indices_and_base_idxr_jax",
            "_task_structure",
            "_warp_indices",
            "_member_modules",
            "load_mcmc_samples",
            "_build_covar_module",
            "_build_mean_module",
            "_build_multitask_covariance",
            "_build_input_transform",
        ):
            method = getattr(pyro, name, None)
            if method is not None and getattr(method, "__module__", "") not in (
                "botorch_rs.models.fully_bayesian",
                "botorch_rs.models.fully_bayesian_multitask",
            ):
                raise UnsupportedError(f"Torch NUTS does not implement custom {name} overrides.")
        prior = None
        if (
            pyro._outputscale_prior_concentration is not None
            and pyro._outputscale_prior_rate is not None
        ):
            prior = (pyro._outputscale_prior_concentration, pyro._outputscale_prior_rate)
        indices = None
        if pyro.use_input_warping:
            indices = (
                pyro._warp_indices()
                if multitask
                else (
                    list(range(model._train_X.shape[-1]))
                    if pyro.indices is None
                    else list(pyro.indices)
                )
            )
        task_spec = None
        if multitask:
            task_model = cast(LatentFeatureMultiTaskPyroMixin, pyro)
            task_spec = (task_model.task_feature, task_model.num_tasks, task_model.task_rank)
        result = cls(
            model._mapped_inputs(model._train_X) if multitask else model._train_X,
            model._train_Y,
            model._train_Yvar,
            kernel=pyro._rs_kernel,
            outputscale_prior=prior,
            warp_indices=indices,
            warp_eps=pyro._eps,
            input_transform=model._fb_user_input_transform,
            outcome_transform=model.outcome_transform,
            task=task_spec,
        )
        if multitask:
            # Task indices are discrete metadata, while common transforms
            # must still receive original task labels and original data graphs.
            result.train_X = model._train_X
        return result

    def sites(self) -> list[tuple[str, list[int], bool]]:
        return [(site.name, list(site.shape), site.positive) for site in self._sites]

    def _data(self) -> tuple[Tensor, Tensor, Tensor | None]:
        x, y, noise = self.train_X, self.train_Y, self.train_Yvar
        for transform, value in ((self.input_transform, x), (self.outcome_transform, y)):
            if transform is not None:
                if getattr(transform, "_tensor_backend", None) == "torch" and getattr(
                    transform, "transform_on_train", True
                ):
                    fit = getattr(transform, "fit", None)
                    if fit is not None:
                        fit(value)
                transform.eval()
        if self.input_transform is not None:
            transform = getattr(self.input_transform, "preprocess_transform", self.input_transform)
            x = transform(x)
        if self.task is not None and not torch.equal(
            x[..., self.task[0]], self.train_X[..., self.task[0]]
        ):
            raise UnsupportedError("Input transforms must preserve mapped categorical task labels.")
        if self.outcome_transform is not None:
            y, noise = self.outcome_transform(y, noise, X=self.train_X)
        if x.ndim != 2 or y.shape != (x.shape[-2], 1):
            raise UnsupportedError("Torch NUTS requires unbatched transformed training data.")
        return (
            x,
            y.squeeze(-1),
            None if noise is None else noise.squeeze(-1).clamp_min(_NOISE_FLOOR),
        )

    def constrain(self, z: Tensor) -> dict[str, Tensor]:
        if not isinstance(z, Tensor) or z.ndim < 1 or z.shape[-1] != self.dim:
            raise ValueError(f"Expected unconstrained Tensor ending in {self.dim} parameters.")
        if z.dtype != self.train_X.dtype or z.device != self.train_X.device:
            raise ValueError("MCMC states must share the training dtype and device.")
        values, offset = {}, 0
        for site in self._sites:
            raw = (
                z[..., offset : offset + site.size].reshape(*z.shape[:-1], *site.shape)
                if site.shape or z.ndim > 1
                else z[offset]
            )
            values[site.name] = raw.exp() if site.positive else raw
            offset += site.size
        return values

    def __call__(self, z: Tensor) -> Tensor:
        values = self.constrain(z)
        joint, offset = z.new_zeros(z.shape[:-1]), 0
        for site in self._sites:
            raw = z[..., offset : offset + site.size]
            p = site.parameters
            if site.prior in ("normal", "lognormal"):
                term = (
                    -0.5 * ((raw - p[0]) / p[1]).square()
                    - math.log(p[1])
                    - 0.5 * math.log(2 * math.pi)
                )
            elif site.prior == "half_cauchy":
                term = (
                    math.log(2 / math.pi)
                    - math.log(p[0])
                    + raw
                    - torch.nn.functional.softplus(2 * (raw - math.log(p[0])))
                )
            else:
                term = p[0] * math.log(p[1]) - math.lgamma(p[0]) + p[0] * raw - p[1] * raw.exp()
            joint = joint + term.sum(-1)
            offset += site.size
        x, y, fixed_noise = self._data()
        if not x.shape[-2]:
            return -joint
        if self.warp_indices is not None:
            selected = x.index_select(-1, self.warp_indices)
            selected = (selected * (1 - 2 * self.warp_eps) + self.warp_eps).clamp(
                self.warp_eps, 1 - self.warp_eps
            )
            warped = 1 - (1 - selected.pow(values["c1"].unsqueeze(-2))).pow(
                values["c0"].unsqueeze(-2)
            )
            shape = torch.broadcast_shapes(x.shape[:-2], z.shape[:-1])
            x = x.expand(*shape, *x.shape[-2:]).index_copy(-1, self.warp_indices, warped)
        if self.task is not None:
            x = x.index_select(-1, self.base_indices)
        if self.kernel == "linear":
            variance = (values["tau_sq"].unsqueeze(-1) * values["_weight_variance_sq"]).sqrt()
            covariance = _rs.tensor_kernel(
                "linear", x - 0.5, x - 0.5, parameters={"variance": variance.unsqueeze(-2)}
            )
        else:
            lengthscale = (
                values["lengthscale"]
                if self.kernel == "matern"
                else (
                    values["kernel_tausq"].unsqueeze(-1) * values["_kernel_inv_length_sq"]
                ).rsqrt()
            )
            covariance = _rs.tensor_kernel("matern", x, x, lengthscale.unsqueeze(-2), nu=2.5)
            if "outputscale" in values:
                covariance = covariance * values["outputscale"][..., None, None]
        noise = (
            fixed_noise
            if fixed_noise is not None
            else (values["noise"] + _NOISE_FLOOR).unsqueeze(-1)
        )
        mean = values["mean"].unsqueeze(-1)
        if self.task is not None:
            features = values["latent_features"].index_select(-2, self.task_indices)
            covariance = covariance * _rs.tensor_kernel(
                "matern", features, features, values["task_lengthscale"].unsqueeze(-2), nu=2.5
            )
            mean = values["mean"].index_select(-1, self.task_indices)
        likelihood = _rs.tensor_exact_mll(y, covariance, mean, noise, normalize=False)
        return -joint - likelihood

    def value_and_grad(self, z: Tensor, create_graph: bool = False) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            point = z if create_graph and z.requires_grad else z.detach().requires_grad_()
            try:
                value = self(point)
                (gradient,) = torch.autograd.grad(value.sum(), point, create_graph=create_graph)
            except torch.linalg.LinAlgError:
                return z.new_full(z.shape[:-1], math.inf), torch.full_like(z, math.nan)
        return (value, gradient) if create_graph else (value.detach(), gradient.detach())

    def postprocess(self, samples: Tensor) -> dict[str, Tensor]:
        values = self.constrain(samples)
        if self.kernel == "saas":
            inverse = values.pop("kernel_tausq").unsqueeze(-1) * values.pop("_kernel_inv_length_sq")
            values.update(lengthscale=inverse.rsqrt(), kernel_inv_length_sq=inverse)
        elif self.kernel == "linear":
            values["weight_variance"] = (
                values.pop("tau_sq").unsqueeze(-1) * values.pop("_weight_variance_sq")
            ).sqrt()
        if "noise" in values:
            values["noise"] = values["noise"] + _NOISE_FLOOR
        return values


class _TensorDraws:
    def __init__(self, like: Tensor, seed: int, draws: Any = None) -> None:
        self.like = like
        self.generator = torch.Generator(device=like.device).manual_seed(seed)
        self.recorded = None if draws is None else list(draws)
        self.position = 0
        self.max_probability_error = 0.0
        self.first_flip: int | None = None

    def _next(self, kind: str) -> Any:
        if self.recorded is None or self.position >= len(self.recorded):
            raise ValueError(f"Recorded NUTS draws exhausted at draw {self.position}.")
        row = self.recorded[self.position]
        self.position += 1
        if not row or row[0] != kind:
            raise ValueError(f"Recorded NUTS draw {self.position - 1} must have kind {kind!r}.")
        return row

    def normal(self, dim: int) -> Tensor:
        if self.recorded is None:
            return torch.randn(
                dim, generator=self.generator, dtype=self.like.dtype, device=self.like.device
            )
        value = torch.as_tensor(self._next("n")[1], dtype=self.like.dtype, device=self.like.device)
        if value.shape != (dim,):
            raise ValueError(f"Recorded momentum must contain {dim} standard normals.")
        return value

    def bernoulli(self, probability: Tensor) -> bool:
        if self.recorded is None:
            uniform = torch.rand(
                (), generator=self.generator, dtype=self.like.dtype, device=self.like.device
            )
        else:
            row = self._next("u")
            uniform = self.like.new_tensor(row[1])
            actual = float(probability)
            expected = float(row[2])
            if math.isfinite(actual) and math.isfinite(expected):
                self.max_probability_error = max(self.max_probability_error, abs(actual - expected))
            if (float(row[1]) < expected) != bool(
                uniform < probability
            ) and self.first_flip is None:
                self.first_flip = self.position - 1
        return bool(uniform < probability)


@dataclass(frozen=True)
class _Tree:
    left: tuple[Tensor, Tensor, Tensor]
    right: tuple[Tensor, Tensor, Tensor]
    z: Tensor
    pe: Tensor
    grad: Tensor
    energy: Tensor
    depth: int
    weight: Tensor
    r_sum: Tensor
    turning: bool
    diverging: bool
    accept_sum: Tensor
    count: int


def _mass_apply(matrix: Tensor, value: Tensor) -> Tensor:
    return matrix @ value if matrix.ndim == 2 else matrix * value


def _kinetic(inverse: Tensor, momentum: Tensor) -> Tensor:
    return 0.5 * torch.dot(_mass_apply(inverse, momentum), momentum)


class _Integrator:
    def __init__(
        self, potential: Any, inverse: Tensor, step: float, energy: Tensor, max_delta: float
    ) -> None:
        self.potential, self.inverse, self.step = potential, inverse, step
        self.energy, self.max_delta = energy, max_delta

    def leaf(self, edge: tuple[Tensor, Tensor, Tensor], right: bool) -> _Tree:
        z, momentum, gradient = edge
        step = self.step if right else -self.step
        half = momentum - (0.5 * step) * gradient
        point = z + step * _mass_apply(self.inverse, half)
        pe, grad = self.potential.value_and_grad(point)
        final_momentum = half - (0.5 * step) * grad
        energy = pe + _kinetic(self.inverse, final_momentum)
        delta = torch.nan_to_num(
            energy - self.energy, nan=math.inf, posinf=math.inf, neginf=-math.inf
        )
        edge = (point, final_momentum, grad)
        return _Tree(
            edge,
            edge,
            point,
            pe,
            grad,
            energy,
            0,
            -delta,
            final_momentum,
            False,
            bool(delta > self.max_delta),
            (-delta).exp().clamp_max(1),
            1,
        )

    def turning(self, left: Tensor, right: Tensor, total: Tensor) -> bool:
        adjusted = total - 0.5 * (left + right)
        return bool(torch.dot(_mass_apply(self.inverse, left), adjusted) <= 0) or bool(
            torch.dot(_mass_apply(self.inverse, right), adjusted) <= 0
        )

    def combine(
        self, current: _Tree, new: _Tree, right: bool, biased: bool, rng: _TensorDraws
    ) -> _Tree:
        left_tree, right_tree = (current, new) if right else (new, current)
        total = current.r_sum + new.r_sum
        if biased:
            probability = (
                current.weight.new_zeros(())
                if new.turning or new.diverging
                else (new.weight - current.weight).exp().clamp_max(1)
            )
            turning = new.turning or self.turning(left_tree.left[1], right_tree.right[1], total)
        else:
            probability = (new.weight - current.weight).sigmoid()
            turning = current.turning
        proposal = new if rng.bernoulli(probability) else current
        return _Tree(
            left_tree.left,
            right_tree.right,
            proposal.z,
            proposal.pe,
            proposal.grad,
            proposal.energy,
            current.depth + 1,
            torch.logaddexp(current.weight, new.weight),
            total,
            turning,
            new.diverging,
            current.accept_sum + new.accept_sum,
            current.count + new.count,
        )

    def subtree(self, prototype: _Tree, right: bool, checkpoints: int, rng: _TensorDraws) -> _Tree:
        tree, turning = replace(prototype, count=0), False
        momenta = [torch.zeros_like(tree.r_sum) for _ in range(checkpoints)]
        sums = [torch.zeros_like(tree.r_sum) for _ in range(checkpoints)]
        while tree.count < (1 << prototype.depth) and not turning and not tree.diverging:
            leaf = self.leaf(tree.right if right else tree.left, right)
            index = tree.count
            updated = leaf if index == 0 else self.combine(tree, leaf, right, False, rng)
            maximum = (index >> 1).bit_count()
            minimum = maximum - (((~index & (index + 1)) - 1).bit_count()) + 1
            if index % 2 == 0:
                momenta[maximum], sums[maximum] = leaf.right[1], updated.r_sum
            turning = False
            for checkpoint in range(maximum, minimum - 1, -1):
                turning = self.turning(
                    momenta[checkpoint],
                    leaf.right[1],
                    updated.r_sum - sums[checkpoint] + momenta[checkpoint],
                )
                if turning:
                    break
            tree = updated
        return replace(tree, depth=prototype.depth, turning=turning)


def _adaptation_windows(steps: int) -> list[tuple[int, int]]:
    if steps < 20:
        return [(0, steps - 1)]
    start, end, size = 75, 50, 25
    if start + end + size > steps:
        start, end = int(0.15 * steps), int(0.1 * steps)
        size = steps - start - end
    result = [(0, start - 1)]
    next_start, final_start = start, steps - end
    while next_start < final_start:
        width = size
        if 3 * width <= final_start - next_start:
            size *= 2
        else:
            width = final_start - next_start
        result.append((next_start, next_start + width - 1))
        next_start += width
    return [*result, (final_start, steps - 1)]


class _DualAveraging:
    def __init__(self, step: float) -> None:
        self.center, self.x, self.average, self.gradient, self.count = (
            math.log(10 * step),
            0.0,
            0.0,
            0.0,
            0,
        )

    def update(self, error: float, last: bool) -> float:
        self.count += 1
        t = self.count
        self.gradient = (1 - 1 / (t + 10)) * self.gradient + error / (t + 10)
        self.x = self.center - math.sqrt(t) / 0.05 * self.gradient
        weight = t**-0.75
        self.average = (1 - weight) * self.average + weight * self.x
        bounds = torch.finfo(torch.float64)
        return math.exp(
            min(max(self.average if last else self.x, math.log(bounds.tiny)), math.log(bounds.max))
        )


class _Welford:
    def __init__(self, point: Tensor, dense: bool) -> None:
        self.mean = torch.zeros_like(point)
        self.m2 = point.new_zeros((point.numel(), point.numel()) if dense else point.shape)
        self.count, self.dense = 0, dense

    def update(self, point: Tensor) -> None:
        self.count += 1
        before = point - self.mean
        self.mean = self.mean + before / self.count
        after = point - self.mean
        self.m2 = self.m2 + (torch.outer(after, before) if self.dense else before * after)

    def finalize(self, regularize: bool) -> tuple[Tensor, Tensor]:
        n = self.count
        covariance = self.m2 / (n - 1)
        if regularize:
            diagonal = (
                torch.eye(self.mean.numel(), device=self.mean.device, dtype=self.mean.dtype)
                if self.dense
                else torch.ones_like(self.mean)
            )
            covariance = n / (n + 5) * covariance + (1e-3 * 5 / (n + 5)) * diagonal
        if not self.dense:
            return covariance, covariance.rsqrt()
        symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
        reverse = torch.linalg.cholesky(symmetric.flip((-1, -2)))
        inverse_root = reverse.flip((-1, -2)).transpose(-1, -2)
        root = torch.linalg.solve_triangular(
            inverse_root,
            torch.eye(self.mean.numel(), device=self.mean.device, dtype=self.mean.dtype),
            upper=False,
        )
        return covariance, root


def sample_torch_nuts(
    potential: Any,
    initial: Tensor,
    *,
    num_warmup: int = 512,
    num_samples: int = 256,
    max_tree_depth: int = 6,
    seed: int = 0,
    draws: Any = None,
    dense_mass: bool = True,
    target_accept_prob: float = 0.8,
    step_size: float = 1.0,
    adapt_step_size: bool = True,
    adapt_mass_matrix: bool = True,
    regularize_mass_matrix: bool = True,
    max_delta_energy: float = 1000.0,
    progress: Any = None,
) -> dict[str, Any]:
    """NumPyro/native-port multinomial NUTS with Tensor states and GPU algebra.

    Python controls tree decisions using scalar synchronization. State vectors,
    momenta, covariance estimates and mass-matrix solves never flatten to CPU.
    Markov transitions and random/discrete tree decisions are detached, as in
    ordinary MCMC; the supplied potential remains fully differentiable.
    """
    if torch.is_inference_mode_enabled():
        raise UnsupportedError("NUTS needs autograd and cannot run under torch.inference_mode.")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    if (
        not isinstance(initial, Tensor)
        or initial.ndim != 1
        or not initial.numel()
        or initial.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("initial must be a nonempty float32/float64 state vector.")
    if (
        any(type(value) is not int for value in (num_warmup, num_samples, max_tree_depth))
        or num_warmup < 0
        or num_samples < 1
        or max_tree_depth < 1
    ):
        raise ValueError("NUTS needs nonnegative warmup, positive samples and positive tree depth.")
    if (
        not 0 < target_accept_prob < 1
        or not math.isfinite(step_size)
        or step_size <= 0
        or not math.isfinite(max_delta_energy)
        or max_delta_energy <= 0
    ):
        raise ValueError("Invalid NUTS acceptance, step-size or divergence settings.")
    point = initial.detach().clone()
    pe, gradient = potential.value_and_grad(point)
    if not bool(
        torch.isfinite(point).all() & torch.isfinite(pe).all() & torch.isfinite(gradient).all()
    ):
        raise ValueError("NUTS initial energy and gradient must be finite.")
    dimension = point.numel()
    inverse = (
        torch.eye(dimension, dtype=point.dtype, device=point.device)
        if dense_mass
        else torch.ones_like(point)
    )
    mass_root = inverse
    rng = _TensorDraws(point, seed, draws)
    rng.normal(dimension)  # NumPyro init_kernel's initial momentum draw.
    schedule, window = _adaptation_windows(num_warmup), 0
    dual, welford = _DualAveraging(step_size), _Welford(point, dense_mass)
    samples, trace = [], []
    for iteration in range(num_warmup + num_samples):
        momentum = _mass_apply(mass_root, rng.normal(dimension))
        current_energy = pe + _kinetic(inverse, momentum)
        integrator = _Integrator(potential, inverse, step_size, current_energy, max_delta_energy)
        edge = (point, momentum, gradient)
        tree = _Tree(
            edge,
            edge,
            point,
            pe,
            gradient,
            current_energy,
            0,
            point.new_zeros(()),
            momentum,
            False,
            False,
            point.new_zeros(()),
            0,
        )
        while tree.depth < max_tree_depth and not tree.turning and not tree.diverging:
            right = rng.bernoulli(point.new_tensor(0.5))
            subtree = integrator.subtree(tree, right, max_tree_depth, rng)
            tree = integrator.combine(tree, subtree, right, True, rng)
        point, pe, gradient = tree.z, tree.pe, tree.grad
        acceptance = tree.accept_sum / tree.count
        if iteration < num_warmup:
            if adapt_step_size:
                step_size = dual.update(
                    target_accept_prob - float(acceptance), iteration + 1 == num_warmup
                )
            middle = 0 < window < len(schedule) - 1
            if adapt_mass_matrix and middle:
                welford.update(point)
            at_end = iteration == schedule[min(window, len(schedule) - 1)][1]
            if at_end:
                window += 1
            if at_end and middle:
                if adapt_mass_matrix:
                    inverse, mass_root = welford.finalize(regularize_mass_matrix)
                    welford = _Welford(point, dense_mass)
                if adapt_step_size:
                    dual = _DualAveraging(step_size)
        trace.append(
            dict(
                z=point,
                potential_energy=pe,
                step_size=step_size,
                accept_prob=acceptance,
                num_steps=tree.count,
                diverging=tree.diverging,
            )
        )
        if iteration >= num_warmup:
            samples.append(point)
        if progress is not None:
            progress(
                iteration + 1,
                num_warmup + num_samples,
                iteration < num_warmup,
                float(acceptance),
                tree.count,
                tree.diverging,
            )
    return dict(
        samples=torch.stack(samples),
        trace=trace,
        step_size=step_size,
        inverse_mass_matrix=inverse,
        stopped_at=None,
        replay_draws_consumed=rng.position,
        replay_draws_total=None if rng.recorded is None else len(rng.recorded),
        replay_max_probability_error=rng.max_probability_error,
        replay_first_flip=rng.first_flip,
    )


def fit_torch_fully_bayesian_nuts(
    model: Any,
    max_tree_depth: int,
    warmup_steps: int,
    num_samples: int,
    thinning: int,
    jit_compile: bool,
    seed: int,
    draws: Any,
    init_params: Any,
    disable_progbar: bool = False,
) -> None:
    """Fit a Tensor model without resetting previously loaded draws on failure."""
    if torch.is_inference_mode_enabled():
        raise UnsupportedError("NUTS needs autograd and cannot run under torch.inference_mode.")
    # Pinned BoTorch accepts this legacy flag but does not pass it to its
    # NumPyro implementation. Preserve that accepted no-op on this backend.
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
        raise ValueError(
            "NUTS needs positive tree depth, samples, thinning and nonnegative warmup."
        )
    if (draws is None) != (init_params is None):
        raise ValueError("`draws` and `init_params` must be given together.")
    potential = TorchMCMCPotential.from_model(model)
    like = model._train_X
    if init_params is None:
        rng = torch.Generator(device=like.device).manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        for _ in range(100):
            initial = torch.rand(potential.dim, device=like.device, dtype=like.dtype, generator=rng)
            initial = initial * 4 - 2
            value, gradient = potential.value_and_grad(initial)
            if bool(torch.isfinite(value).all() & torch.isfinite(gradient).all()):
                break
        else:
            raise ValueError("Failed to find finite NUTS initial parameters in 100 attempts.")
    else:
        initial = torch.as_tensor(init_params, device=like.device, dtype=like.dtype)
        if initial.shape != (potential.dim,):
            raise ValueError(f"Expected {potential.dim} initial unconstrained parameters.")
    from botorch_rs.models._nuts_progress import nuts_progress

    with nuts_progress(disable_progbar) as progress:
        result = sample_torch_nuts(
            potential,
            initial,
            num_warmup=warmup_steps,
            num_samples=num_samples,
            max_tree_depth=max_tree_depth,
            seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
            draws=draws,
            progress=progress,
        )
    samples = potential.postprocess(result["samples"])
    model.load_mcmc_samples({name: values[::thinning] for name, values in samples.items()})
    model._nuts_result = result
    model.eval()
