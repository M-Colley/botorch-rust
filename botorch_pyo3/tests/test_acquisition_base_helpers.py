"""Custom acquisition base classes and Gaussian constraint preprocessing."""

from __future__ import annotations

import pytest

from botorch_rs.acquisition import (
    ConstrainedAnalyticAcquisitionFunctionMixin,
    MultiModelAcquisitionFunction,
)
from botorch_rs.acquisition.cached_cholesky import supports_cache_root
from helpers import KINDS, as_kind, to_list


def test_multimodel_acquisition_preserves_model_dictionary_and_custom_forward() -> None:
    with pytest.raises(TypeError, match="abstract"):
        MultiModelAcquisitionFunction({})

    class Composite(MultiModelAcquisitionFunction):
        def forward(self, X):
            return self.model_dict["score"](X) * self.model_dict["feasibility"](X)

    models = {"score": lambda X: X + 1, "feasibility": lambda X: 0.5}
    acquisition = Composite(models)
    assert acquisition.model_dict is models
    assert acquisition(3.0) == 2.0


@pytest.mark.parametrize("kind", KINDS)
def test_constrained_mixin_probabilities_and_buffers(kind: str) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.acquisition.analytic")

    class Native(ConstrainedAnalyticAcquisitionFunctionMixin):
        def register_buffer(self, name, tensor, persistent=True):
            setattr(self, name, tensor)

    class Reference(torch.nn.Module, bt.ConstrainedAnalyticAcquisitionFunctionMixin):
        def __init__(self, constraints):
            torch.nn.Module.__init__(self)
            bt.ConstrainedAnalyticAcquisitionFunctionMixin.__init__(self, constraints)

    constraints = {0: (0.2, None), 1: (None, 1.1), 2: (-0.3, 0.7), 3: (None, None)}
    actual, expected = Native(constraints), Reference(constraints)
    for name in (
        "con_lower_inds",
        "con_upper_inds",
        "con_both_inds",
        "con_lower",
        "con_upper",
        "con_both",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    means = [[0.1, 1.5, 0.2, 1.0], [-8.0, 9.0, 12.0, 0.0]]
    sigmas = [[1.1, 0.7, 0.4, 1.0], [1.0, 2.0, 1.5, 1.0]]
    values = actual._compute_log_prob_feas(as_kind(means, kind), as_kind(sigmas, kind))
    reference = expected._compute_log_prob_feas(
        torch.tensor(means, dtype=torch.double), torch.tensor(sigmas, dtype=torch.double)
    )
    torch.testing.assert_close(
        torch.tensor(to_list(values), dtype=torch.double), reference, atol=1e-10, rtol=1e-10
    )
    if kind == "torch":
        X = torch.tensor(means, dtype=torch.double, requires_grad=True)
        S = torch.tensor(sigmas, dtype=torch.double, requires_grad=True)
        actual_grad = torch.autograd.grad(actual._compute_log_prob_feas(X, S).sum(), (X, S))
        expected_grad = torch.autograd.grad(expected._compute_log_prob_feas(X, S).sum(), (X, S))
        for a, e in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-10)
    for invalid, message in (({}, "at least one"), ({0: (1.0, 0.5)}, "Upper bound")):
        with pytest.raises(ValueError, match=message):
            Native(invalid)
    Native.objective_index = 0
    with pytest.raises(ValueError, match="objective"):
        Native(constraints)


def test_cache_root_support_respects_gp_protocol_and_opt_out() -> None:
    from botorch_rs.models.gpytorch import GPyTorchModel
    from botorch_rs.models.model_list_gp_regression import ModelListGP

    class GP(GPyTorchModel):
        pass

    model = GP()
    assert supports_cache_root(model)
    model._supports_cache_root = False
    assert not supports_cache_root(model)
    del model._supports_cache_root
    assert not supports_cache_root(object())
    from types import SimpleNamespace

    model.outcome_transform = SimpleNamespace(_is_linear=False)
    assert not supports_cache_root(model)
    model.outcome_transform._is_linear = True
    assert supports_cache_root(model)
    model_list = object.__new__(ModelListGP)
    model_list.models = [model, object()]
    assert not supports_cache_root(model_list)
    model_list.models = [model, model]
    assert supports_cache_root(model_list)
