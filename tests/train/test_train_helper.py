"""Unit tests for the train_helper building blocks.

The training loop was split into small, independently testable helpers:

  * ``tensors_to_device`` -- move a batch onto the compute device,
  * ``inference_batch``   -- one forward pass of the MultiTaskModel,
  * ``calc_loss``         -- combine the two task losses into one scalar,
  * ``score2label``       -- threshold probabilities into 0/1 predictions,
  * ``prepare_binary_per_batch`` -- detach/sigmoid a batch into numpy arrays,
  * ``prepare_continuous_per_batch`` -- detach a regression batch into numpy
                             arrays (no sigmoid -- the head outputs values),
  * ``BinaryScores``      -- accumulate per-batch predictions and dump
                             per-feature classification metrics,
  * ``ContinuousScores``  -- accumulate per-batch predictions and dump
                             per-feature RMSE.

The properties pinned down here are the ones easy to get subtly wrong:

  * ``calc_loss`` is exactly ``masked_bce(...).mean() + masked_huber(...)`` on
    the model's "binary"/"continuous" heads, with ``pos_weights`` threaded in,
    masked/NaN positions ignored, and gradient flowing back through the model,
  * ``prepare_binary_per_batch`` applies the sigmoid (scores are probabilities,
    not logits) and returns a boolean numpy mask, while
    ``prepare_continuous_per_batch`` leaves the raw head output untouched,
  * ``BinaryScores`` selects valid rows *per column*, matches sklearn on the
    kept rows, and emits ``None`` for empty / single-class columns instead of
    crashing,
  * ``ContinuousScores`` selects valid rows *per column* (across many columns,
    without cross-column leakage), matches sklearn RMSE on the kept rows, and
    emits ``None`` for a fully-masked column instead of crashing.

The per-element BCE / Huber and the sklearn metric maths are not re-derived --
the references come from the project's own losses and from sklearn directly.
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from kuai_recommender.nn.loss import masked_bce, masked_huber
from kuai_recommender.nn.multitask import MultiTaskModel
from kuai_recommender.train.train_helper import (
    BinaryScores,
    ContinuousScores,
    calc_loss,
    inference_batch,
    prepare_binary_per_batch,
    prepare_continuous_per_batch,
    score2label,
    tensors_to_device,
)

_N_BINARY = 4
_N_CONTINUOUS = 1
_INPUT_DIM = 3
_EMBEDDING_DIMS = [(10, 2), (7, 4)]


def _make_model(hidden_dim: int = 5) -> MultiTaskModel:
    """A MultiTaskModel with the "binary"/"continuous" heads the helpers read."""
    torch.manual_seed(0)
    model = MultiTaskModel(
        input_dim=_INPUT_DIM,
        hidden_dim=hidden_dim,
        embedding_dims=_EMBEDDING_DIMS,
        output_dims={"binary": _N_BINARY, "continuous": _N_CONTINUOUS},
    )
    model.eval()
    return model


def _make_batch(
    batch_size: int = 4,
    y_binary: torch.Tensor | None = None,
    mask_binary: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """One collate-shaped batch: (x, x_cat, y_bin, y_cont, m_bin, m_cont)."""
    torch.manual_seed(1)
    x = torch.randn(batch_size, _INPUT_DIM)
    # bucket ids must stay < num_embeddings for both columns (min is 7)
    x_cat = torch.randint(0, 7, (batch_size, len(_EMBEDDING_DIMS)), dtype=torch.long)
    if y_binary is None:
        y_binary = torch.randint(0, 2, (batch_size, _N_BINARY)).float()
    y_continuous = torch.randn(batch_size, _N_CONTINUOUS)
    if mask_binary is None:
        mask_binary = torch.ones(batch_size, _N_BINARY, dtype=torch.bool)
    mask_continuous = torch.ones(batch_size, _N_CONTINUOUS, dtype=torch.bool)
    return x, x_cat, y_binary, y_continuous, mask_binary, mask_continuous


def _forward_and_loss(model, batch, pos_weights):
    """Reproduce the training-loop flow: to-device -> forward -> loss."""
    x, x_cat, y_bin, y_cont, m_bin, m_cont = tensors_to_device(*batch, device="cpu")
    outputs = inference_batch(model, x, x_cat)
    return calc_loss(outputs, pos_weights, y_bin, y_cont, m_bin, m_cont)


# --------------------------------------------------------------------------- #
# tensors_to_device
# --------------------------------------------------------------------------- #
def test_tensors_to_device_preserves_order_and_moves_each_tensor():
    a = torch.randn(2, 2)
    b = torch.arange(3)
    moved = tensors_to_device(a, b, device="cpu")

    assert len(moved) == 2
    assert torch.equal(moved[0], a) and torch.equal(moved[1], b)
    assert all(t.device.type == "cpu" for t in moved)


# --------------------------------------------------------------------------- #
# inference_batch
# --------------------------------------------------------------------------- #
def test_inference_batch_matches_direct_model_call():
    model = _make_model()
    x, x_cat, *_ = _make_batch()

    outputs = inference_batch(model, x, x_cat)
    expected = model(x, x_cat)

    assert set(outputs) == {"binary", "continuous"}
    assert torch.allclose(outputs["binary"], expected["binary"])
    assert torch.allclose(outputs["continuous"], expected["continuous"])


# --------------------------------------------------------------------------- #
# calc_loss
# --------------------------------------------------------------------------- #
def test_calc_loss_equals_masked_bce_mean_plus_masked_huber():
    """The scalar equals masked_bce(binary).mean() + masked_huber(continuous),
    reading the "binary" head for BCE and the "continuous" head for Huber."""
    model = _make_model()
    batch = _make_batch()
    x, x_cat, y_bin, y_cont, m_bin, m_cont = batch
    pos_weights = torch.tensor([1.0, 2.0, 1.0, 3.0])

    out = _forward_and_loss(model, batch, pos_weights)

    outputs = model(x, x_cat)  # deterministic under eval()
    expected = (
        masked_bce(outputs["binary"], y_bin, m_bin, pos_weights=pos_weights).mean()
        + masked_huber(outputs["continuous"], y_cont, m_cont)
    )

    assert out.shape == ()
    assert out.device.type == "cpu"
    assert torch.allclose(out, expected)


def test_pos_weight_upweights_positive_binary_class():
    """A larger pos_weight raises the loss when the binary labels are positive
    (regression: pos_weights dropped or not forwarded to masked_bce)."""
    model = _make_model()
    y_bin = torch.ones(4, _N_BINARY)  # all-positive -> pos_weight bites
    batch = _make_batch(y_binary=y_bin)

    base = _forward_and_loss(model, batch, torch.ones(_N_BINARY))
    weighted = _forward_and_loss(model, batch, torch.full((_N_BINARY,), 5.0))

    assert weighted > base


def test_masked_positions_with_nan_labels_stay_finite():
    """A fully-masked binary column carrying NaN labels must not leak NaN into
    the scalar (the real collate stores NaN at 'no signal' positions)."""
    model = _make_model()
    y_bin = torch.randint(0, 2, (4, _N_BINARY)).float()
    m_bin = torch.ones(4, _N_BINARY, dtype=torch.bool)
    y_bin[:, 1] = float("nan")  # column 1 has no signal...
    m_bin[:, 1] = False  # ...and is masked out
    x, x_cat, _, y_cont, _, m_cont = _make_batch()
    batch = (x, x_cat, y_bin, y_cont, m_bin, m_cont)

    out = _forward_and_loss(model, batch, torch.ones(_N_BINARY))

    assert torch.isfinite(out)


def test_loss_is_differentiable_through_the_model():
    """backward() from the returned scalar reaches the embeddings and the shared
    trunk -- the loop can train on it."""
    model = _make_model()
    model.train()
    batch = _make_batch()

    loss = _forward_and_loss(model, batch, torch.ones(_N_BINARY))
    loss.backward()

    for embedding in model.embeddings:
        assert embedding.weight.grad is not None
        assert embedding.weight.grad.abs().sum() > 0
    first_linear = model.shared_layers[0]
    assert first_linear.weight.grad is not None
    assert first_linear.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# score2label
# --------------------------------------------------------------------------- #
def test_score2label_thresholds_at_default_half():
    scores = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    labels = score2label(scores)

    assert labels.dtype == int
    # >= 0.5 counts as positive
    assert labels.tolist() == [0, 0, 1, 1, 1]


def test_score2label_custom_threshold():
    scores = np.array([0.6, 0.7, 0.8, 0.9])
    labels = score2label(scores, thres=0.85)

    assert labels.tolist() == [0, 0, 0, 1]


# --------------------------------------------------------------------------- #
# prepare_binary_per_batch
# --------------------------------------------------------------------------- #
def test_prepare_binary_per_batch_applies_sigmoid_and_returns_numpy():
    logits = torch.tensor([[0.0, 2.0], [-2.0, 0.0]])
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # float mask, as collate emits

    y_true, y_score, out_mask = prepare_binary_per_batch(logits, y, mask)

    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_score, np.ndarray)
    assert out_mask.dtype == bool
    # scores are sigmoid(logits), NOT raw logits
    np.testing.assert_allclose(y_score, torch.sigmoid(logits).numpy())
    np.testing.assert_array_equal(out_mask, np.array([[True, False], [True, True]]))
    np.testing.assert_array_equal(y_true, y.numpy())


def test_prepare_binary_per_batch_detaches_from_graph():
    logits = torch.randn(3, 2, requires_grad=True)
    y = torch.zeros(3, 2)
    mask = torch.ones(3, 2)

    # numpy conversion would raise if the tensors were still attached to the graph
    y_true, y_score, out_mask = prepare_binary_per_batch(logits, y, mask)

    assert y_score.shape == (3, 2)


# --------------------------------------------------------------------------- #
# BinaryScores
# --------------------------------------------------------------------------- #
def _perfect_column(name="a"):
    """A single-feature batch where score ranks the labels perfectly."""
    y_true = np.array([[0.0], [0.0], [1.0], [1.0]])
    y_score = np.array([[0.1], [0.2], [0.8], [0.9]])
    mask = np.array([[True], [True], [True], [True]])
    return y_true, y_score, mask


def test_dump_metrics_perfect_separation_scores_one():
    scores = BinaryScores(["a"])
    scores.append(*_perfect_column())

    metrics = scores.dump_metrics()["a"]

    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall_precision_auc"] == 1.0


def test_dump_metrics_matches_sklearn_on_kept_rows():
    """Per-feature metrics equal sklearn computed on exactly the masked-in rows."""
    rng = np.random.default_rng(0)
    n, f = 50, 3
    y_true = rng.integers(0, 2, size=(n, f)).astype(float)
    y_score = rng.random((n, f))
    mask = rng.random((n, f)) > 0.3  # ~70% valid, varies per column

    scores = BinaryScores([f"feat{i}" for i in range(f)])
    # split into two batches to also exercise append/concat
    scores.append(y_true[:20], y_score[:20], mask[:20])
    scores.append(y_true[20:], y_score[20:], mask[20:])

    metrics = scores.dump_metrics()

    for i in range(f):
        col = mask[:, i]
        yt = y_true[col, i]
        ys = y_score[col, i]
        if yt.sum() == 0 or yt.all():
            continue  # single-class columns are asserted separately
        yp = (ys >= 0.5).astype(int)
        m = metrics[f"feat{i}"]
        assert m["roc_auc"] == roc_auc_score(yt, ys)
        assert m["recall"] == recall_score(yt, yp, zero_division=np.nan)
        assert m["precision"] == precision_score(yt, yp, zero_division=np.nan)
        assert m["recall_precision_auc"] == average_precision_score(yt, ys)


def test_dump_metrics_masked_rows_are_excluded():
    """Rows masked out for a column must not affect that column's metrics: a
    poisoned-but-masked row leaves the perfect-separation score untouched."""
    y_true = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])
    y_score = np.array([[0.1], [0.2], [0.8], [0.9], [0.0]])  # last row contradicts
    mask = np.array([[True], [True], [True], [True], [False]])  # ...but is masked

    scores = BinaryScores(["a"])
    scores.append(y_true, y_score, mask)

    assert scores.dump_metrics()["a"]["roc_auc"] == 1.0


def test_dump_metrics_single_class_column_returns_none():
    y_true = np.array([[1.0], [1.0], [1.0]])  # only positives
    y_score = np.array([[0.2], [0.6], [0.9]])
    mask = np.ones((3, 1), dtype=bool)

    scores = BinaryScores(["a"])
    scores.append(y_true, y_score, mask)
    metrics = scores.dump_metrics()["a"]

    assert metrics == {
        "roc_auc": None,
        "recall": None,
        "precision": None,
        "recall_precision_auc": None,
    }


def test_dump_metrics_fully_masked_column_returns_none():
    y_true = np.array([[0.0], [1.0]])
    y_score = np.array([[0.3], [0.7]])
    mask = np.zeros((2, 1), dtype=bool)  # no valid rows

    scores = BinaryScores(["a"])
    scores.append(y_true, y_score, mask)

    assert scores.dump_metrics()["a"]["roc_auc"] is None


def test_dump_metrics_is_json_serializable():
    import json

    scores = BinaryScores(["a", "b"])
    # 2-feature batch; "b" is single-class (all ones) -> its metrics are None
    y_true = np.array([[0.0, 1.0], [1.0, 1.0]])
    y_score = np.array([[0.2, 0.9], [0.8, 0.7]])
    mask = np.ones((2, 2), dtype=bool)
    scores.append(y_true, y_score, mask)

    # must serialize even though "b"'s metrics are None
    json.dumps(scores.dump_metrics())


# --------------------------------------------------------------------------- #
# prepare_continuous_per_batch
# --------------------------------------------------------------------------- #
def test_prepare_continuous_per_batch_keeps_raw_output_and_returns_numpy():
    """Regression predictions must NOT be squashed through sigmoid (unlike the
    binary head) and the mask comes back as boolean numpy."""
    preds = torch.tensor([[0.0, 3.5], [-2.0, 10.0]])
    y = torch.tensor([[0.1, 3.4], [-1.9, 9.8]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # float mask, as collate emits

    y_true, y_score, out_mask = prepare_continuous_per_batch(preds, y, mask)

    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_score, np.ndarray)
    assert out_mask.dtype == bool
    # raw head output, NOT sigmoid(output)
    np.testing.assert_array_equal(y_score, preds.numpy())
    np.testing.assert_array_equal(out_mask, np.array([[True, False], [True, True]]))
    np.testing.assert_array_equal(y_true, y.numpy())


def test_prepare_continuous_per_batch_detaches_from_graph():
    preds = torch.randn(3, 2, requires_grad=True)
    y = torch.zeros(3, 2)
    mask = torch.ones(3, 2)

    # numpy conversion would raise if the tensors were still attached to the graph
    y_true, y_score, out_mask = prepare_continuous_per_batch(preds, y, mask)

    assert y_score.shape == (3, 2)


# --------------------------------------------------------------------------- #
# ContinuousScores
# --------------------------------------------------------------------------- #
def test_continuous_dump_metrics_perfect_prediction_scores_zero():
    y_true = np.array([[1.0], [2.0], [3.0]])
    scores = ContinuousScores(["a"])
    scores.append(y_true, y_true.copy(), np.ones((3, 1), dtype=bool))

    assert scores.dump_metrics()["a"]["rmse"] == 0.0


def test_continuous_dump_metrics_matches_sklearn_on_kept_rows():
    """Per-feature RMSE equals sklearn on exactly the masked-in rows, and each
    column is scored independently (guards cross-column slice leakage when
    there is more than one continuous head)."""
    rng = np.random.default_rng(0)
    n, f = 50, 3
    y_true = rng.standard_normal((n, f))
    y_score = y_true + rng.standard_normal((n, f))  # noisy predictions
    mask = rng.random((n, f)) > 0.3  # ~70% valid, varies per column

    scores = ContinuousScores([f"feat{i}" for i in range(f)])
    # split into two batches to also exercise append/concat
    scores.append(y_true[:20], y_score[:20], mask[:20])
    scores.append(y_true[20:], y_score[20:], mask[20:])

    metrics = scores.dump_metrics()

    for i in range(f):
        col = mask[:, i]
        expected = root_mean_squared_error(y_true[col, i], y_score[col, i])
        assert metrics[f"feat{i}"]["rmse"] == float(expected)


def test_continuous_dump_metrics_masked_rows_are_excluded():
    """A masked-out row must not affect the column's RMSE -- a perfect
    prediction stays 0.0 even with a poisoned-but-masked last row."""
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_score = np.array([[1.0], [2.0], [99.0]])  # last row contradicts...
    mask = np.array([[True], [True], [False]])  # ...but is masked out

    scores = ContinuousScores(["a"])
    scores.append(y_true, y_score, mask)

    assert scores.dump_metrics()["a"]["rmse"] == 0.0


def test_continuous_dump_metrics_fully_masked_column_returns_none():
    y_true = np.array([[1.0], [2.0]])
    y_score = np.array([[1.5], [2.5]])
    mask = np.zeros((2, 1), dtype=bool)  # no valid rows -> RMSE undefined

    scores = ContinuousScores(["a"])
    scores.append(y_true, y_score, mask)

    assert scores.dump_metrics()["a"]["rmse"] is None


def test_continuous_dump_metrics_is_json_serializable():
    import json

    scores = ContinuousScores(["a", "b"])
    # "b" is fully masked -> its rmse is None
    y_true = np.array([[1.0, 5.0], [2.0, 6.0]])
    y_score = np.array([[1.1, 4.0], [1.9, 7.0]])
    mask = np.array([[True, False], [True, False]])
    scores.append(y_true, y_score, mask)

    metrics = scores.dump_metrics()
    assert metrics["b"]["rmse"] is None
    # rmse must be a plain float, not np.float64, so json can serialize it
    json.dumps(metrics)
