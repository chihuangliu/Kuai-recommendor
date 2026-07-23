"""Unit tests for MultiTaskModel.

The model prepends learned embeddings of the hash-bucketed categorical features
onto the continuous feature vector, then runs a shared trunk and one linear head
per task. The properties under test are the ones that are easy to get subtly
wrong:

  * the shared trunk's first Linear is widened to input_dim + sum(embedding_dim),
    so the concatenated vector actually fits,
  * each categorical *column* of x_cat ([B, C]) is routed through its own
    embedding -- column i -> embeddings[i] -- and the results are concatenated in
    order after the continuous block (regression: indexing x_cat by batch row
    instead of by column, which crashes whenever batch size != num categoricals),
  * forward returns one [B, output_dim] tensor per task, keyed by task name, for
    an arbitrary batch size, and
  * the embeddings receive gradient (they are genuinely part of the graph).
"""

import torch

from kuai_recommender.nn.multitask import MultiTaskModel


def _make_model(
    input_dim: int = 3,
    hidden_dim: int = 5,
    embedding_dims: list[tuple[int, int]] | None = None,
    output_dims: dict[str, int] | None = None,
) -> MultiTaskModel:
    torch.manual_seed(0)
    model = MultiTaskModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dims=[(10, 2), (7, 4)] if embedding_dims is None else embedding_dims,
        output_dims={"is_click": 1, "dwell_log": 1} if output_dims is None else output_dims,
    )
    model.eval()
    return model


def test_first_linear_is_widened_by_embedding_width():
    """shared_layers[0].in_features == input_dim + sum(embedding_dim)."""
    model = _make_model(input_dim=3, embedding_dims=[(10, 2), (7, 4)])
    first_linear = model.shared_layers[0]
    assert first_linear.in_features == 3 + (2 + 4)


def test_forward_returns_one_tensor_per_task_with_batch_rows():
    """outputs is keyed by task name; each is [B, output_dim] for arbitrary B."""
    output_dims = {"is_click": 1, "long_view": 1, "dwell_log": 1}
    model = _make_model(input_dim=3, output_dims=output_dims)
    B = 4  # deliberately != number of categorical features (2)
    x = torch.randn(B, 3)
    x_cat = torch.tensor([[1, 2], [3, 4], [5, 6], [0, 1]], dtype=torch.long)

    outputs = model(x, x_cat)

    assert set(outputs) == set(output_dims)
    for task, dim in output_dims.items():
        assert outputs[task].shape == (B, dim)


def test_forward_concatenates_embeddings_by_column():
    """Column i of x_cat is embedded by embeddings[i] and concatenated after the
    continuous block. Reconstructing that concat by hand and pushing it through
    the trunk/heads must reproduce forward() exactly -- this pins both the
    per-column routing and the concat order."""
    model = _make_model(input_dim=3, embedding_dims=[(10, 2), (7, 4)])
    B = 4
    x = torch.randn(B, 3)
    x_cat = torch.tensor([[1, 2], [3, 4], [5, 6], [0, 1]], dtype=torch.long)

    emb_user = model.embeddings[0](x_cat[:, 0])  # [B, 2]
    emb_author = model.embeddings[1](x_cat[:, 1])  # [B, 4]
    expected_in = torch.cat([x, emb_user, emb_author], dim=1)  # [B, 3+2+4]
    shared = model.shared_layers(expected_in)
    expected = {t: layer(shared) for t, layer in model.task_layers.items()}

    outputs = model(x, x_cat)

    for task in expected:
        assert torch.allclose(outputs[task], expected[task], atol=1e-6)


def test_swapping_a_categorical_column_changes_the_output():
    """Distinct bucket ids in a column flow through the model (guards against the
    categorical block being dropped or a column being ignored)."""
    model = _make_model(input_dim=2, output_dims={"is_click": 1})
    x = torch.zeros(1, 2)
    base = model(x, torch.tensor([[1, 1]], dtype=torch.long))["is_click"]
    changed_user = model(x, torch.tensor([[2, 1]], dtype=torch.long))["is_click"]
    changed_author = model(x, torch.tensor([[1, 2]], dtype=torch.long))["is_click"]

    assert not torch.allclose(base, changed_user)
    assert not torch.allclose(base, changed_author)


def test_embeddings_receive_gradient():
    """Backprop reaches the embedding tables -- they are trainable parameters,
    not a detached lookup."""
    model = _make_model(input_dim=2, output_dims={"is_click": 1})
    model.train()
    x = torch.randn(3, 2)
    x_cat = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)

    loss = model(x, x_cat)["is_click"].sum()
    loss.backward()

    for embedding in model.embeddings:
        assert embedding.weight.grad is not None
        assert embedding.weight.grad.abs().sum() > 0
