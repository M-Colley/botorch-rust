"""The explicit torch DataLoader adapter, independent of native fitting."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models import SingleTaskGP as BtGP  # noqa: E402
from botorch.optim.utils.model_utils import get_data_loader as bt_loader  # noqa: E402

from botorch_rs.models import SingleTaskGP as RsGP  # noqa: E402
from botorch_rs.optim.utils import get_data_loader  # noqa: E402


@pytest.mark.parametrize("batch_size", [1, 3, 1024])
@pytest.mark.parametrize("drop_last", [False, True])
def test_data_loader_parity(batch_size, drop_last) -> None:
    x = torch.linspace(0, 1, 7, dtype=torch.float64).reshape(7, 1)
    y = (6 * x).sin()
    expected_model = BtGP(x, y)
    actual_model = RsGP(x, y)
    expected = list(bt_loader(expected_model, batch_size=batch_size, drop_last=drop_last))
    actual = list(get_data_loader(actual_model, batch_size=batch_size, drop_last=drop_last))
    assert len(actual) == len(expected)
    for rows, reference in zip(actual, expected, strict=True):
        for value, target in zip(rows, reference, strict=True):
            torch.testing.assert_close(value, target)


def test_data_loader_for_variational_model() -> None:
    from botorch_rs.models.approximate_gp import SingleTaskVariationalGP

    x = torch.linspace(0, 1, 7, dtype=torch.float64).reshape(7, 1)
    y = (6 * x).sin()
    model = SingleTaskVariationalGP(x, y, inducing_points=3)
    loader = get_data_loader(model.model, batch_size=3)
    assert len(loader) == 3
    torch.testing.assert_close(torch.cat([row[-1] for row in loader]), model.model.train_targets)
