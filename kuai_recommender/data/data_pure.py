from datetime import datetime
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kuai_recommender.data.utils import rng
from kuai_recommender.features.registry import (
    BINARY_SIGNALS,
    BINARY_TARGETS,
    CONTINUOUS_TARGETS,
    MODEL_CATEGORICAL,
    MODEL_CONTINUOUS,
)


class KuaiPureDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        continuous_features: list[str] = MODEL_CONTINUOUS,
        categorical_features: list[str] = MODEL_CATEGORICAL,
        neg_keep_frac: float = 1.0,
    ):
        self.df = df
        self.features = continuous_features
        self.labels = BINARY_TARGETS + CONTINUOUS_TARGETS
        self.cat_features = categorical_features
        self.rng = rng
        self._neg_sampling(neg_keep_frac)

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.features].to_numpy(dtype="float32"))
        x = torch.nan_to_num(x, nan=0.0)  # replace NaN for the first impression

        x_cat = torch.tensor(
            row[self.cat_features].to_numpy(dtype="int64"), dtype=torch.long
        )

        y = {c: torch.tensor(row[c], dtype=torch.float32) for c in self.labels}
        return x, x_cat, y

    def _neg_sampling(self, neg_keep_frac: float) -> None:
        if isclose(neg_keep_frac, 1.0):
            return

        is_pure_neg = (self.df[BINARY_SIGNALS].sum(axis=1) == 0).to_numpy()

        neg_pos = np.flatnonzero(is_pure_neg)
        keep_neg = self.rng.choice(
            neg_pos, size=int(len(neg_pos) * neg_keep_frac), replace=False
        )

        keep_mask = ~is_pure_neg
        keep_mask[keep_neg] = True
        self.df = self.df[keep_mask].reset_index(drop=True)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        continuous_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        neg_keep_frac: float = 1.0,
        start_dt: datetime | None = None,
        end_dt: datetime | None = None,
    ) -> "KuaiPureDataset":
        if continuous_features is None:
            continuous_features = MODEL_CONTINUOUS
        if categorical_features is None:
            categorical_features = MODEL_CATEGORICAL
        obj = cls.__new__(cls)
        obj.df = pd.read_parquet(path).reset_index(drop=True)
        if start_dt is not None:
            obj.df = obj.df[obj.df["event_timestamp"] >= start_dt]
        if end_dt is not None:
            obj.df = obj.df[obj.df["event_timestamp"] < end_dt]
        obj.features = continuous_features
        obj.labels = BINARY_TARGETS + CONTINUOUS_TARGETS
        obj.cat_features = categorical_features
        obj.rng = rng
        obj._neg_sampling(neg_keep_frac)
        return obj


def collate_with_masks(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    xs, x_cats, ys = zip(*batch)
    x_batch = torch.stack(xs)  # [B, F]
    x_cat_batch = torch.stack(x_cats)  # [B, C]

    def stack_cols(cols: list[str]) -> torch.Tensor:
        return torch.stack(
            [torch.stack([y[c] for y in ys]) for c in cols], dim=1
        )  # [B, len(cols)]

    y_binary_batch = stack_cols(BINARY_TARGETS)
    y_continuous_batch = stack_cols(CONTINUOUS_TARGETS)
    mask_binary_batch = ~torch.isnan(y_binary_batch)
    mask_continuous_batch = ~torch.isnan(y_continuous_batch)
    return (
        x_batch,
        x_cat_batch,
        y_binary_batch,
        y_continuous_batch,
        mask_binary_batch,
        mask_continuous_batch,
    )
