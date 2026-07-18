from torch.utils.data import Dataset
from kuai_recommender.data.utils import (
    DATA_DIR,
    VIDEO_FEATURES_BASIC_PATH,
    KuaiPureDatasetSplits,
)
import pandas as pd
import torch


class KuaiPureData:
    BINARY_COLUMNS = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
    ]

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
        # dropna=False keeps NaN-key rows (e.g. a video missing from the basic
        # feature file -> NaN author_id) as their own group, so .mean().values
        # stays the same length as self.df and the positional assign can't misalign.
        grouped = self.df.groupby(group_by, dropna=False)

        suffix = "_".join(group_by if isinstance(group_by, list) else [group_by])
        for col in self.BINARY_COLUMNS:
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
        for col in self.BINARY_COLUMNS:
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


class KuaiPureDataset(Dataset):
    def __init__(self, kuai_pure_data: KuaiPureData, features):
        self.df = kuai_pure_data.df
        self.features = features
        self.labels = KuaiPureData.BINARY_COLUMNS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.features].to_numpy(dtype="float32"))
        x = torch.nan_to_num(x, nan=0.0)  # replace NaN for the first impression
        y = {c: torch.tensor(row[c], dtype=torch.float32) for c in self.labels}
        return x, y
