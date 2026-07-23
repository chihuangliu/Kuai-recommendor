from kuai_recommender.nn.loss import masked_bce, masked_huber
from kuai_recommender.nn.multitask import MultiTaskModel
import kuai_recommender
import torch
import uuid
from pathlib import Path


def inference_and_calc_loss(
    model: MultiTaskModel,
    device: str,
    x_batch: torch.Tensor,
    x_cat_batch: torch.Tensor,
    y_binary_batch: torch.Tensor,
    y_continuous_batch: torch.Tensor,
    mask_binary_batch: torch.Tensor,
    mask_continuous_batch: torch.Tensor,
    pos_weights: torch.Tensor,
) -> torch.Tensor:
    x_batch = x_batch.to(device)
    x_cat_batch = x_cat_batch.to(device)
    y_binary_batch = y_binary_batch.to(device)
    y_continuous_batch = y_continuous_batch.to(device)
    mask_binary_batch = mask_binary_batch.to(device)
    mask_continuous_batch = mask_continuous_batch.to(device)

    outputs = model(x_batch, x_cat_batch)
    binary_loss = masked_bce(
        outputs["binary"],
        y_binary_batch,
        mask_binary_batch,
        pos_weights=pos_weights,
    )
    cont_loss = masked_huber(
        outputs["continuous"], y_continuous_batch, mask_continuous_batch
    )
    return binary_loss.mean() + cont_loss  # TODO: weight the loss in different tasks


def get_run_id():
    return uuid.uuid4().hex[:8]


def get_run_dir(run_id: str) -> Path:
    root = Path(kuai_recommender.__file__).resolve().parents[1]
    return root / "run" / run_id
