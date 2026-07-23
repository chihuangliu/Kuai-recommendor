"""Unit tests for inference_and_calc_loss.

The helper runs one forward pass of a MultiTaskModel and combines the two
task losses into a single scalar the training loop can call ``.backward()`` on.
The properties that are easy to get subtly wrong -- and that these tests pin
down:

  * the returned scalar is exactly ``masked_bce(...).mean() + masked_huber(...)``
    on the model's "binary"/"continuous" heads (regression: dropping the
    ``.mean()`` over binary heads, or swapping which head each loss reads),
  * ``pos_weights`` is threaded into the binary loss and actually up-weights the
    positive class,
  * masked-out positions never contribute -- even when their label is NaN (the
    real collate stores NaN for "no signal"),
  * the result is a single 0-dim tensor on the requested device, and
  * gradient flows back through the model (embeddings + trunk), so the loop can
    train on it.

The per-element BCE / Huber maths is not re-derived here -- the reference comes
from the project's own masked_bce / masked_huber, so these tests exercise only
the forward + combine wrapper.
"""

import torch

from kuai_recommender.nn.loss import masked_bce, masked_huber
from kuai_recommender.nn.multitask import MultiTaskModel
from kuai_recommender.train.train_helper import inference_and_calc_loss

_N_BINARY = 4
_N_CONTINUOUS = 1
_INPUT_DIM = 3
_EMBEDDING_DIMS = [(10, 2), (7, 4)]


def _make_model(hidden_dim: int = 5) -> MultiTaskModel:
    """A MultiTaskModel with the "binary"/"continuous" heads the helper reads."""
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


def test_returns_sum_of_masked_bce_mean_and_masked_huber():
    """The scalar equals masked_bce(binary).mean() + masked_huber(continuous),
    reading the "binary" head for BCE and the "continuous" head for Huber."""
    model = _make_model()
    x, x_cat, y_bin, y_cont, m_bin, m_cont = _make_batch()
    pos_weights = torch.tensor([1.0, 2.0, 1.0, 3.0])

    out = inference_and_calc_loss(
        model, "cpu", x, x_cat, y_bin, y_cont, m_bin, m_cont, pos_weights
    )

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

    base = inference_and_calc_loss(model, "cpu", *batch, torch.ones(_N_BINARY))
    weighted = inference_and_calc_loss(model, "cpu", *batch, torch.full((_N_BINARY,), 5.0))

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

    out = inference_and_calc_loss(
        model, "cpu", x, x_cat, y_bin, y_cont, m_bin, m_cont, torch.ones(_N_BINARY)
    )

    assert torch.isfinite(out)


def test_loss_is_differentiable_through_the_model():
    """backward() from the returned scalar reaches the embeddings and the shared
    trunk -- the loop can train on it."""
    model = _make_model()
    model.train()
    batch = _make_batch()

    loss = inference_and_calc_loss(model, "cpu", *batch, torch.ones(_N_BINARY))
    loss.backward()

    for embedding in model.embeddings:
        assert embedding.weight.grad is not None
        assert embedding.weight.grad.abs().sum() > 0
    first_linear = model.shared_layers[0]
    assert first_linear.weight.grad is not None
    assert first_linear.weight.grad.abs().sum() > 0
