from kuai_recommender.nn.loss import masked_bce, masked_huber
from kuai_recommender.nn.multitask import MultiTaskModel
import kuai_recommender
import torch
import uuid
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    root_mean_squared_error,
)
import numpy as np


def tensors_to_device(*tensors: torch.Tensor, device: str) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in tensors)


def inference_batch(
    model: MultiTaskModel,
    x_batch: torch.Tensor,
    x_cat_batch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return model(x_batch, x_cat_batch)


def calc_loss(
    outputs: torch.Tensor,
    pos_weights: torch.Tensor,
    y_binary_batch: torch.Tensor,
    y_continuous_batch: torch.Tensor,
    mask_binary_batch: torch.Tensor,
    mask_continuous_batch: torch.Tensor,
) -> torch.Tensor:
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


class Scores:
    def __init__(self, cols: list[str]):
        self.cols = cols
        self.y_score: list[np.ndarray] = []
        self.y_true: list[np.ndarray] = []
        self.mask: list[np.ndarray] = []

    def append(self, y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray) -> None:
        self.y_true.append(y_true)
        self.y_score.append(y_score)
        self.mask.append(mask)

    def concat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(self.y_true),
            np.concatenate(self.y_score),
            np.concatenate(self.mask),
        )

    def dump_metrics(self) -> dict[str, dict[str, float | None]]:
        raise NotImplementedError


class BinaryScores(Scores):
    def dump_metrics(self) -> dict[str, dict[str, float | None]]:
        y_true, y_score, mask = self.concat()
        res = {}
        for i, name in enumerate(self.cols):
            valid_i = mask[:, i]
            y_true_i = y_true[valid_i, i]
            if valid_i.sum() == 0 or y_true_i.sum() == 0 or y_true_i.all():
                res[name] = {
                    "roc_auc": None,
                    "recall": None,
                    "precision": None,
                    "recall_precision_auc": None,
                }
                continue
            y_score_i = y_score[valid_i, i]
            y_label_pred = score2label(y_score_i)
            roc_auc = roc_auc_score(
                y_true_i,
                y_score_i,
            )
            recall = recall_score(y_true_i, y_label_pred, zero_division=np.nan)
            precision = precision_score(y_true_i, y_label_pred, zero_division=np.nan)
            recall_precision_auc = average_precision_score(y_true_i, y_score_i)
            res[name] = {
                "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
                "recall": float(recall) if not np.isnan(recall) else None,
                "precision": float(precision) if not np.isnan(precision) else None,
                "recall_precision_auc": float(recall_precision_auc)
                if not np.isnan(recall_precision_auc)
                else None,
            }
        return res


class ContinuousScores(Scores):
    def dump_metrics(self) -> dict[str, dict[str, float | None]]:
        y_true, y_score, mask = self.concat()

        res = {}
        for i, name in enumerate(self.cols):
            valid = mask[:, i]
            if valid.sum() == 0:
                res[name] = {"rmse": None}
                continue
            y_true_i = y_true[valid, i]
            y_score_i = y_score[valid, i]
            res[name] = {"rmse": float(root_mean_squared_error(y_true_i, y_score_i))}
        return res


def score2label(score: np.ndarray, thres: float = 0.5) -> np.ndarray:
    return (score >= thres).astype(int)


def prepare_binary_per_batch(
    outputs: torch.Tensor, y_binary_batch: torch.Tensor, mask_binary_batch: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = mask_binary_batch.bool().detach().cpu().numpy()
    y_true = y_binary_batch.detach().cpu().numpy()
    y_score = torch.sigmoid(outputs).detach().cpu().numpy()

    return y_true, y_score, mask


def prepare_continuous_per_batch(
    outputs: torch.Tensor,
    y_continuous_batch: torch.Tensor,
    mask_continuous_batch: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = mask_continuous_batch.bool().detach().cpu().numpy()
    y_true = y_continuous_batch.detach().cpu().numpy()
    y_score = outputs.detach().cpu().numpy()

    return y_true, y_score, mask


def get_run_id():
    return uuid.uuid4().hex[:8]


def get_run_dir(run_id: str) -> Path:
    root = Path(kuai_recommender.__file__).resolve().parents[1]
    return root / "run" / run_id
