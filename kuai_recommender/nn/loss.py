from torch import Tensor, nan_to_num
from torch.nn.functional import binary_cross_entropy_with_logits, huber_loss


def masked_bce(
    logits: Tensor, labels: Tensor, masks: Tensor, pos_weights: Tensor
) -> Tensor:
    labels = nan_to_num(labels)
    loss = binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weights, reduction="none"
    )
    loss *= masks
    return loss.sum(0) / masks.sum(0).clamp(min=1)


def masked_huber(logits: Tensor, labels: Tensor, masks: Tensor) -> Tensor:
    labels = nan_to_num(labels)
    loss = huber_loss(logits, labels, reduction="none")
    loss *= masks
    return loss.sum() / masks.sum().clamp(min=1)
