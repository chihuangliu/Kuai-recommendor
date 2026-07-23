import numpy as np
from torch.utils.data import Dataset
from kuai_recommender.data.utils import (
    DATA_DIR,
    VIDEO_FEATURES_BASIC_PATH,
    KuaiPureDatasetSplits,
    rng,
    get_bucket_size,
)
import pandas as pd
import torch
from math import isclose
from sklearn.utils import murmurhash3_32
from kuai_recommender.config import FEATURES


class KuaiPureData:
    BINARY_COLUMNS_ORIGINAL = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
    ]
    BINARY_COLUMNS_PREPROCESSED = BINARY_COLUMNS_ORIGINAL + ["is_skip"]
    CONTINUOUS_COLUMNS_PREPROCESSED = ["dwell_log"]
    CATEGORICAL_COLUMNS_PREPROCESSED = ["user_id_bucket", "author_id_bucket"]
    FEATURE_COLUMNS = FEATURES

    def __init__(
        self,
        name: KuaiPureDatasetSplits,
        history: tuple[KuaiPureDatasetSplits, ...] = (),
    ):
        self.df: pd.DataFrame = self._read_csv(name).assign(is_target=True)
        history_dfs = [self._read_csv(h).assign(is_target=False) for h in history]
        self.df = pd.concat([self.df, *history_dfs], ignore_index=True)

        self.df["dt"] = pd.to_datetime(
            self.df["time_ms"], unit="ms", utc=True
        ).dt.tz_convert("Asia/Shanghai")

        self._join_video_features()
        self._set_user_rolling()
        self._set_user_author_rolling()
        self._set_video_rolling()
        self._set_video_cumulative()
        self._set_engagement_targets()
        self._set_user_buckets(get_bucket_size()["user_id"])
        self._set_author_buckets(get_bucket_size()["author_id"])
        self.df = (
            self.df[self.df["is_target"]]
            .drop(columns="is_target")
            .reset_index(drop=True)
        )

    def _read_csv(self, name: KuaiPureDatasetSplits) -> pd.DataFrame:
        return pd.read_csv(DATA_DIR / name)

    def _join_video_features(self) -> None:
        video_features_basic = pd.read_csv(VIDEO_FEATURES_BASIC_PATH)
        self.df = self.df.merge(
            video_features_basic[["video_id", "author_id"]], on="video_id", how="left"
        )

    def _set_rolling_columns(
        self, group_by: list[str] | str, window: str = "7D"
    ) -> None:
        sort_columns = ([group_by] if isinstance(group_by, str) else group_by) + ["dt"]
        self.df = self.df.sort_values(sort_columns)
        grouped = self.df.groupby(group_by, dropna=False)

        suffix = "_".join(group_by if isinstance(group_by, list) else [group_by])
        for col in self.BINARY_COLUMNS_ORIGINAL:
            self.df[f"{col}_rolling_{suffix}"] = (
                grouped.rolling(window=window, closed="left", on="dt")[col]
                .mean()
                .values
            )

    def _set_cumulative_columns(self, group_by: list[str] | str) -> None:
        sort_columns = ([group_by] if isinstance(group_by, str) else group_by) + ["dt"]
        self.df = self.df.sort_values(sort_columns)
        grouped = self.df.groupby(group_by, dropna=False)

        suffix = "_".join(group_by if isinstance(group_by, list) else [group_by])
        for col in self.BINARY_COLUMNS_ORIGINAL:
            cumulative = grouped[col].cumsum() - self.df[col]
            self.df[f"{col}_cumulative_{suffix}"] = cumulative

    def _set_user_rolling(self, window: str = "7D") -> None:
        self._set_rolling_columns(group_by="user_id", window=window)

    def _set_user_author_rolling(self, window: str = "7D") -> None:
        self._set_rolling_columns(group_by=["user_id", "author_id"], window=window)

    def _set_video_rolling(self, window: str = "7D") -> None:
        self._set_rolling_columns(group_by="video_id", window=window)

    def _set_video_cumulative(self) -> None:
        self._set_cumulative_columns(group_by="video_id")

    def _set_engagement_targets(self) -> None:
        dur = self.df["duration_ms"]
        play = self.df["play_time_ms"]
        valid = dur > 0
        completion = (play / dur.where(valid)).clip(upper=1.0)
        dwell = np.where(valid, np.minimum(play, 2 * dur), np.nan).astype("float32")
        self.df["is_skip"] = np.where(
            valid, (completion < 0.5) & (play < 5000), np.nan
        ).astype("float32")
        self.df["dwell_log"] = np.log1p(dwell).astype("float32")

    @staticmethod
    def _hash_to_bucket(value: str | int | float, n_buckets: int) -> int:
        if pd.isna(value):
            return 0
        n_valid_buckets = n_buckets - 1
        return murmurhash3_32(str(value), positive=True) % n_valid_buckets + 1

    def _set_hash_bucket(self, column: str, n_buckets: int) -> None:
        self.df[f"{column}_bucket"] = self.df[column].apply(
            lambda x: self._hash_to_bucket(x, n_buckets)
        )

    def _set_user_buckets(self, n_buckets: int) -> None:
        self._set_hash_bucket("user_id", n_buckets)

    def _set_author_buckets(self, n_buckets: int) -> None:
        self._set_hash_bucket("author_id", n_buckets)


class KuaiPureDataset(Dataset):
    def __init__(
        self,
        kuai_pure_data: KuaiPureData,
        continuous_features: list[str] = KuaiPureData.FEATURE_COLUMNS,
        categorical_features: list[str] = KuaiPureData.CATEGORICAL_COLUMNS_PREPROCESSED,
        neg_keep_frac: float = 1.0,
    ):
        self.df = kuai_pure_data.df
        self.features = continuous_features
        self.labels = (
            KuaiPureData.BINARY_COLUMNS_PREPROCESSED
            + KuaiPureData.CONTINUOUS_COLUMNS_PREPROCESSED
        )
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

        is_pure_neg = (
            self.df[KuaiPureData.BINARY_COLUMNS_ORIGINAL].sum(axis=1) == 0
        ).to_numpy()

        neg_pos = np.flatnonzero(is_pure_neg)
        keep_neg = self.rng.choice(
            neg_pos, size=int(len(neg_pos) * neg_keep_frac), replace=False
        )

        keep_mask = ~is_pure_neg
        keep_mask[keep_neg] = True
        self.df = self.df[keep_mask].reset_index(drop=True)


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

    y_binary_batch = stack_cols(KuaiPureData.BINARY_COLUMNS_PREPROCESSED)
    y_continuous_batch = stack_cols(KuaiPureData.CONTINUOUS_COLUMNS_PREPROCESSED)
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
