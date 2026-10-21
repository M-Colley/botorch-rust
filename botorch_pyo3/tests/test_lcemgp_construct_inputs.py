"""Context datasets construct usable native/device LCEM models."""

import pytest

from botorch_rs.models import LCEMGP
from botorch_rs.utils.datasets import MultiTaskDataset, SupervisedDataset
from helpers import KINDS, as_kind, assert_close, shape


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("context", ["default", "categorical", "continuous", "both"])
def test_contextual_dataset_constructs_and_predicts(kind, context):
    X = as_kind([[0.1, 0.0], [0.4, 1.0], [0.7, 0.0], [0.9, 1.0]], kind)
    Y = as_kind([[0.2], [0.6], [-0.1], [0.3]], kind)
    dataset = SupervisedDataset(X, Y, feature_names=["x", "task"], outcome_names=["y"])
    extra = {}
    if context in ("categorical", "both"):
        extra.update(context_cat_feature=as_kind([[0, 0], [1, 0]], kind), embs_dim_list=[1, 2])
    if context in ("continuous", "both"):
        extra["context_emb_feature"] = as_kind([[0.2], [0.5]], kind)
    inputs = LCEMGP.construct_inputs(dataset, task_feature=-1, output_tasks=[1], **extra)
    assert inputs["train_X"] is X and inputs["train_Y"] is Y
    assert "train_Yvar" not in inputs
    for name, value in extra.items():
        assert inputs[name] is value
    model = LCEMGP(**inputs, seed=19)
    posterior = model.posterior(as_kind([[0.3], [0.6]], kind))
    assert shape(posterior.mean) == (2, 1) and shape(posterior.variance) == (2, 1)


@pytest.mark.parametrize("kind", KINDS)
def test_multitask_dataset_preserves_inferred_labels_and_optional_noise(kind):
    children = [
        SupervisedDataset(
            as_kind([[0.1], [0.4]], kind),
            as_kind([[0.2], [0.6]], kind),
            Yvar=as_kind([[0.03], [0.04]], kind),
            feature_names=["x"],
            outcome_names=[name],
        )
        for name in ("first", "second")
    ]
    dataset = MultiTaskDataset(children, target_outcome_name="second")
    inputs = LCEMGP.construct_inputs(dataset, task_feature=-1)
    assert inputs["all_tasks"] == [0, 1]
    assert_close(inputs["train_Yvar"], dataset.Yvar)
    assert shape(inputs["train_X"]) == (4, 2)
    assert inputs["output_tasks"] is None


def test_construct_inputs_matches_official_and_keeps_cuda_leaf_graphs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from botorch.models.contextual_multioutput import LCEMGP as ReferenceGP
    from botorch.utils.datasets import SupervisedDataset as ReferenceDataset

    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        X = torch.tensor(
            [[0.1, 0], [0.4, 1], [0.7, 0], [0.9, 1]], device=device, dtype=torch.double
        )
        Y = torch.tensor([[0.2], [0.6], [-0.1], [0.3]], device=device, dtype=torch.double)
        X.requires_grad_()
        Y.requires_grad_()
        embedding = torch.tensor(
            [[0.2], [0.5]], device=device, dtype=torch.double, requires_grad=True
        )
        dataset = ReferenceDataset(X, Y, feature_names=["x", "task"], outcome_names=["y"])
        kwargs = dict(task_feature=-1, output_tasks=[1], context_emb_feature=embedding)
        actual = LCEMGP.construct_inputs(dataset, **kwargs)
        expected = ReferenceGP.construct_inputs(dataset, **kwargs)
        assert actual.keys() == expected.keys()
        for name in actual:
            if isinstance(actual[name], torch.Tensor):
                assert actual[name] is expected[name]
            else:
                assert actual[name] == expected[name]
        model = LCEMGP(**actual, backend="torch", seed=19)
        query = X.new_tensor([[0.3], [0.6]]).requires_grad_()
        prediction = model.posterior(query)
        first = torch.autograd.grad(
            prediction.mean.square().sum() + prediction.variance.sum(),
            (query, X, Y, embedding),
            create_graph=True,
        )
        second = torch.autograd.grad(sum(g.square().sum() for g in first), (query, X, Y, embedding))
        assert all(g.device == X.device and g.isfinite().all() for g in (*first, *second))
