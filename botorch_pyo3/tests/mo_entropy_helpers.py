"""Shared helpers of the live comparisons of pathwise sampling and the
information-theoretic multi-objective acquisition functions with BoTorch
(``test_parity_mo_entropy.py``, ``test_equivalence_mo_entropy.py``).

Import only after ``importorskip_botorch()``: this module requires torch and
BoTorch.

Random numbers. BoTorch draws the random numbers of its paths from torch's
global generator: the Fourier frequencies (``feature_map.weight`` of the
prior paths), the prior weights (``weight``) and the noise of the update
(``torch.randn_like``, recorded by :func:`recorded_randn`). botorch_rs takes
them as explicit ``draws`` (one ``(frequencies, weights, noise)`` tuple per
output, :func:`bt_path_draws`), and the candidate sets of
``random_search_optimizer`` (BoTorch's ``draw_sobol_samples``) as explicit
``candidates`` (:func:`recorded_optima`).
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterator
from typing import Any

import torch
from botorch.sampling.pathwise import PathList as BtPathList

DT = torch.float64


@contextlib.contextmanager
def recorded_randn(record: list[torch.Tensor]) -> Iterator[None]:
    """Records the draws of ``torch.randn_like`` (the noise of BoTorch's
    ``gaussian_update``)."""
    orig = torch.randn_like

    def randn_like(t: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        z = orig(t, *args, **kwargs)
        record.append(z.detach().clone())
        return z

    torch.randn_like = randn_like  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.randn_like = orig  # type: ignore[assignment]


def matheron_models(paths: Any) -> list[Any]:
    """The ``MatheronPath`` (or prior path) of every model of BoTorch's paths."""
    return list(paths) if isinstance(paths, BtPathList) else [paths]


def bt_path_draws(paths: Any, noises: list[torch.Tensor] | None) -> list[tuple[Any, ...]]:
    """BoTorch's random numbers of Matheron paths (``noises``: the recorded
    update noise of every model) or of prior paths (``noises=None``), one
    tuple per output: frequencies (``L/2 x d``), weights (``S x L``) and,
    for Matheron paths, the standard normal noise (``S x n``)."""
    out: list[tuple[Any, ...]] = []
    subs = matheron_models(paths)
    for i, sub in enumerate(subs):
        prior = sub.paths["prior_paths"] if noises is not None else sub
        freq, weight = prior.feature_map.weight.detach(), prior.weight.detach()
        noise = None if noises is None else noises[i]
        if freq.ndim == 2:  # a single output: no output dimension
            freq, weight = freq.unsqueeze(0), weight.unsqueeze(-2)
            noise = None if noise is None else noise.unsqueeze(-2)
        for t in range(freq.shape[0]):
            w = weight[..., t, :].reshape(-1, weight.shape[-1])
            if noise is None:
                out.append((freq[t], w))
            else:
                out.append((freq[t], w, noise[..., t, :].reshape(-1, noise.shape[-1])))
    return out


@contextlib.contextmanager
def recorded_optima(draws: list[list[Any]], candidates: list[list[Any]]) -> Iterator[None]:
    """Records BoTorch's random numbers in ``sample_optimal_points``: the path
    draws of every sample's ``get_matheron_path_model`` and the candidate
    sets of its ``random_search_optimizer`` (``draw_sobol_samples``)."""
    import botorch.acquisition.multi_objective.utils as bt_utils

    noises: list[torch.Tensor] = []
    orig_get = bt_utils.get_matheron_path_model
    orig_draw = bt_utils.draw_sobol_samples

    def get_path_model(model: Any, *args: Any, **kwargs: Any) -> Any:
        start = len(noises)
        pm = orig_get(model, *args, **kwargs)
        draws.append(bt_path_draws(pm._path, noises[start:]))
        candidates.append([])
        return pm

    def draw_sobol(*args: Any, **kwargs: Any) -> torch.Tensor:
        X = orig_draw(*args, **kwargs)
        if candidates:
            candidates[-1].append(X.squeeze(-2).clone())
        return X

    bt_utils.get_matheron_path_model = get_path_model
    bt_utils.draw_sobol_samples = draw_sobol
    try:
        with recorded_randn(noises):
            yield
    finally:
        bt_utils.get_matheron_path_model = orig_get
        bt_utils.draw_sobol_samples = orig_draw


@contextlib.contextmanager
def recorded_candidates(candidates: list[Any]) -> Iterator[None]:
    """Records the candidate sets of BoTorch's ``random_search_optimizer``."""
    import botorch.acquisition.multi_objective.utils as bt_utils

    orig = bt_utils.draw_sobol_samples

    def draw_sobol(*args: Any, **kwargs: Any) -> torch.Tensor:
        X = orig(*args, **kwargs)
        candidates.append(X.squeeze(-2).clone())
        return X

    bt_utils.draw_sobol_samples = draw_sobol
    try:
        yield
    finally:
        bt_utils.draw_sobol_samples = orig


def botorch_has_3399() -> bool:
    """Whether BoTorch's JES conditions on the raw Pareto sets
    (meta-pytorch/botorch#3399; BoTorch main transforms them twice)."""
    from botorch.acquisition.multi_objective import joint_entropy_search as jes

    src = inspect.getsource(jes.qLowerBoundMultiObjectiveJointEntropySearch.__init__)
    return "transform_inputs(self.pareto_sets)" not in src


def jes_pareto_sets(model: Any, pareto_sets: torch.Tensor) -> torch.Tensor:
    """The Pareto sets to pass to BoTorch's JES so that it conditions on the
    raw sets (#3399, which botorch_rs implements): without the fix, the
    untransformed sets, which BoTorch's double transform maps back."""
    models = getattr(model, "models", [model])
    tf = getattr(models[0], "input_transform", None)
    if botorch_has_3399() or tf is None:
        return pareto_sets
    return tf.untransform(pareto_sets)


@contextlib.contextmanager
def raw_train_inputs(model: Any) -> Iterator[None]:
    """BoTorch's PES with the model's raw training inputs: BoTorch main reads
    ``model.train_inputs`` at construction, which a model with an input
    transform holds transformed in eval mode and transforms again (botorch_rs
    uses the raw inputs); in train mode they are raw."""
    model.train()
    try:
        yield
    finally:
        model.eval()


@contextlib.contextmanager
def ulp_noise(seed: int) -> Iterator[None]:
    """Perturbs every result of ``torch.exp`` and ``torch.erf`` by about one
    unit in the last place (a factor ``1 +- 2^-52``): BoTorch with a math
    library whose last bits differ. PES evaluates ``log1p(-exp(S))`` for
    ``S`` near zero, where such differences change EP's factors by up to a
    few percent."""
    g = torch.Generator().manual_seed(seed)
    orig_exp, orig_erf = torch.exp, torch.erf

    def factors(y: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        sign = torch.randint(0, 2, y.shape, generator=g).to(y.dtype) * 2 - 1
        return torch.where(keep, 1 + sign * 2.0**-52, torch.ones_like(y))

    def exp(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        y = orig_exp(x, *args, **kwargs)
        return y * factors(y, torch.ones_like(y, dtype=torch.bool))

    def erf(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        y = orig_erf(x, *args, **kwargs)
        return y * factors(y, y.abs() <= 1 - 2.0**-50)

    torch.exp = exp  # type: ignore[assignment]
    torch.erf = erf  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.exp = orig_exp  # type: ignore[assignment]
        torch.erf = orig_erf  # type: ignore[assignment]


def single_output_list(model_list: Any) -> Any:
    """The equivalent BoTorch ``ModelListGP`` of single-output models: every
    output of a multi-output ``SingleTaskGP`` (RBF or Matern kernel without
    ``ScaleKernel``, no input transform) becomes a ``SingleTaskGP`` with that
    output's data and hyperparameters (``Standardize`` of one column has the
    same statistics as that column of ``Standardize(m)``)."""
    import gpytorch
    from botorch.models import ModelListGP, SingleTaskGP

    out = []
    for sub in model_list.models:
        if sub.num_outputs == 1:
            out.append(sub)
            continue
        X = sub.train_inputs[0][0]
        Y = sub.outcome_transform.untransform(sub.train_targets.transpose(-1, -2))[0]
        base = sub.covar_module
        for t in range(sub.num_outputs):
            if isinstance(base, gpytorch.kernels.MaternKernel):
                kernel = gpytorch.kernels.MaternKernel(nu=base.nu, ard_num_dims=X.shape[-1])
            else:
                kernel = gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[-1])
            gp = SingleTaskGP(X, Y[:, t : t + 1], covar_module=kernel)
            gp.initialize(
                **{
                    "covar_module.lengthscale": base.lengthscale[t].detach(),
                    "mean_module.constant": sub.mean_module.constant[t].detach(),
                    "likelihood.noise": sub.likelihood.noise[t].detach(),
                }
            )
            gp.eval()
            out.append(gp)
    equiv = ModelListGP(*out)
    equiv.eval()
    return equiv
