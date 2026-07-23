"""Unit tests for the masked multi-task losses.

Both losses take a per-element mask (in practice ``~isnan(labels)``) and must
average only over the *valid* entries. The properties that are easy to get
subtly wrong -- and that these tests pin down:

  * masked_bce reduces along the batch dim and returns one loss *per column*
    ([C]); masked_huber reduces globally and returns a scalar,
  * the denominator is the number of unmasked entries, not the batch size, so
    each output is a genuine mean over valid rows,
  * masked-out positions never contribute -- even when their label is NaN
    (the real caller stores NaN for "no signal" and relies on this),
  * a fully-masked column/tensor returns 0, not NaN (the clamp(min=1) guard
    against divide-by-zero), and
  * pos_weight is threaded through and actually up-weights the positive class.

The per-element BCE / Huber maths is not re-derived here: the reference values
come from torch's own functional losses with ``reduction="none"``, so these
tests exercise only the masking + reduction wrapper.
"""

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, huber_loss

from kuai_recommender.nn.loss import masked_bce, masked_huber


def _bce_reference(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    pos_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-column masked-mean BCE computed straight from torch's element-wise loss."""
    elementwise = binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(labels), pos_weight=pos_weights, reduction="none"
    )
    elementwise = elementwise * masks
    return elementwise.sum(0) / masks.sum(0).clamp(min=1)


def _huber_reference(
    logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Global masked-mean Huber computed straight from torch's element-wise loss."""
    elementwise = huber_loss(logits, torch.nan_to_num(labels), reduction="none")
    elementwise = elementwise * masks
    return elementwise.sum() / masks.sum().clamp(min=1)


# --------------------------------------------------------------------------- #
# masked_bce
# --------------------------------------------------------------------------- #


def test_masked_bce_returns_per_column_mean_over_valid_rows():
    """Output is [C]; each column is the mean loss over that column's unmasked
    rows (regression: reducing over the wrong axis, or dividing by batch size)."""
    logits = torch.tensor([[1.5, -0.5], [-2.0, 0.3], [0.7, 2.1]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    masks = torch.tensor([[True, True], [True, False], [False, True]])
    pos_weights = torch.tensor([1.0, 1.0])

    out = masked_bce(logits, labels, masks, pos_weights)

    assert out.shape == (2,)
    assert torch.allclose(out, _bce_reference(logits, labels, masks, pos_weights))


def test_masked_bce_denominator_is_valid_count_not_batch_size():
    """Column 1 has a single valid row, so its output equals that one row's loss
    -- not the sum spread over the full batch of 3."""
    logits = torch.tensor([[0.4], [1.1], [-0.9]])
    labels = torch.tensor([[1.0], [0.0], [1.0]])
    masks = torch.tensor([[False], [True], [False]])
    pos_weights = torch.tensor([1.0])

    out = masked_bce(logits, labels, masks, pos_weights)

    single_row = binary_cross_entropy_with_logits(
        logits[1], labels[1], pos_weight=pos_weights, reduction="none"
    )
    assert torch.allclose(out, single_row)


def test_masked_bce_ignores_masked_positions_even_with_nan_labels():
    """NaN at a masked position must not leak into the result: replacing those
    labels with NaN gives the same loss as replacing them with any valid value."""
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.5], [-0.3, 2.0]])
    masks = torch.tensor([[True, False], [False, True], [True, True]])
    pos_weights = torch.tensor([1.0, 1.0])

    labels_nan = torch.tensor([[1.0, float("nan")], [float("nan"), 1.0], [0.0, 0.0]])
    labels_valid = labels_nan.clone()
    labels_valid[~masks] = 0.5  # arbitrary in-range garbage at masked spots

    out_nan = masked_bce(logits, labels_nan, masks, pos_weights)
    out_valid = masked_bce(logits, labels_valid, masks, pos_weights)

    assert torch.isfinite(out_nan).all()
    assert torch.allclose(out_nan, out_valid)


def test_masked_bce_fully_masked_column_is_zero_not_nan():
    """A column with no valid rows (all masked, label NaN) returns 0, not NaN."""
    logits = torch.tensor([[0.2, 1.0], [-0.4, -1.0]])
    labels = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])
    masks = torch.tensor([[True, False], [True, False]])
    pos_weights = torch.tensor([1.0, 1.0])

    out = masked_bce(logits, labels, masks, pos_weights)

    assert torch.isfinite(out).all()
    assert out[1] == 0.0


def test_masked_bce_pos_weight_upweights_positive_class():
    """A pos_weight > 1 increases the loss on a positive-labelled entry, and the
    weighted value matches torch's reference with the same weight."""
    logits = torch.tensor([[-2.0]])  # confidently wrong on a positive label
    labels = torch.tensor([[1.0]])
    masks = torch.tensor([[True]])

    baseline = masked_bce(logits, labels, masks, torch.tensor([1.0]))
    weighted = masked_bce(logits, labels, masks, torch.tensor([5.0]))

    assert weighted > baseline
    assert torch.allclose(
        weighted, _bce_reference(logits, labels, masks, torch.tensor([5.0]))
    )


# --------------------------------------------------------------------------- #
# masked_huber
# --------------------------------------------------------------------------- #


def test_masked_huber_returns_scalar_mean_over_valid_entries():
    """Output is a scalar equal to the mean Huber over unmasked entries."""
    logits = torch.tensor([[0.0, 3.0], [1.0, -1.0], [2.5, 0.5]])
    labels = torch.tensor([[0.5, 2.0], [1.0, 0.0], [0.0, 0.5]])
    masks = torch.tensor([[True, True], [True, False], [False, True]])

    out = masked_huber(logits, labels, masks)

    assert out.shape == ()
    assert torch.allclose(out, _huber_reference(logits, labels, masks))


def test_masked_huber_ignores_masked_positions_even_with_nan_labels():
    """NaN labels at masked positions must not corrupt the scalar loss."""
    logits = torch.tensor([[0.2, 4.0], [1.0, 1.0]])
    masks = torch.tensor([[True, False], [True, True]])

    labels_nan = torch.tensor([[0.0, float("nan")], [2.0, 1.5]])
    labels_valid = labels_nan.clone()
    labels_valid[~masks] = 9.0

    out_nan = masked_huber(logits, labels_nan, masks)
    out_valid = masked_huber(logits, labels_valid, masks)

    assert torch.isfinite(out_nan)
    assert torch.allclose(out_nan, out_valid)


def test_masked_huber_fully_masked_is_zero_not_nan():
    """No valid entries at all -> 0, not a divide-by-zero NaN."""
    logits = torch.tensor([[1.0], [2.0]])
    labels = torch.tensor([[float("nan")], [float("nan")]])
    masks = torch.tensor([[False], [False]])

    out = masked_huber(logits, labels, masks)

    assert torch.isfinite(out)
    assert out == 0.0
