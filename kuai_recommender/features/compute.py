import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32

from kuai_recommender.data.utils import (
    DATA_DIR,
    VIDEO_FEATURES_BASIC_PATH,
    KuaiPureDatasetSplits,
)
from kuai_recommender.features.schema import (
    BINARY_FEATURES,
    USER_AUTHOR_FEATURES,
    USER_FEATURES,
    VIDEO_FEATURES,
)


def build_base_frame() -> pd.DataFrame:
    df_train = pd.read_csv(DATA_DIR / KuaiPureDatasetSplits.TRAIN)
    df_val = pd.read_csv(DATA_DIR / KuaiPureDatasetSplits.VAL)
    df = pd.concat([df_train, df_val], ignore_index=True)
    df["dt"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    df_video = pd.read_csv(VIDEO_FEATURES_BASIC_PATH, usecols=["author_id", "video_id"])
    df = df.merge(df_video, on="video_id", how="left")

    return df


def build_user_source(base_frame: pd.DataFrame) -> pd.DataFrame:
    df = _set_rolling_columns(base_frame, "user_id")
    cols = ["dt", "user_id"] + USER_FEATURES
    return df[cols]


def build_user_author_source(base_frame: pd.DataFrame) -> pd.DataFrame:
    base_frame = base_frame.dropna(subset=["author_id"])
    df = _set_rolling_columns(base_frame, ["user_id", "author_id"])
    df["author_id"] = df["author_id"].astype("int64")
    cols = ["dt", "user_id", "author_id"] + USER_AUTHOR_FEATURES
    return df[cols]


def build_video_source(base_frame: pd.DataFrame) -> pd.DataFrame:
    df = _set_rolling_columns(base_frame, "video_id")
    df = _set_cumulative_columns(df, "video_id", ["is_click"])
    cols = ["dt", "video_id"] + VIDEO_FEATURES
    return df[cols]


def _set_cumulative_columns(
    df, group_by: list[str] | str, cols: list[str]
) -> pd.DataFrame:
    sort_columns = ([group_by] if isinstance(group_by, str) else group_by) + ["dt"]
    df = df.sort_values(sort_columns)
    grouped = df.groupby(group_by, dropna=False)

    suffix = "_".join(group_by if isinstance(group_by, list) else [group_by])
    for col in cols:
        cumulative = grouped[col].cumsum() - df[col]
        df[f"{col}_cumulative_{suffix}"] = cumulative
    return df


def _set_rolling_columns(
    df, group_by: list[str] | str, window: str = "7D"
) -> pd.DataFrame:
    sort_columns = ([group_by] if isinstance(group_by, str) else group_by) + ["dt"]
    df = df.sort_values(sort_columns)
    grouped = df.groupby(group_by, dropna=False)

    suffix = "_".join(group_by if isinstance(group_by, list) else [group_by])
    for col in BINARY_FEATURES:
        df[f"{col}_rolling_{suffix}"] = (
            grouped.rolling(window=window, closed="left", on="dt")[col].mean().values
        )
    return df


def set_engagement_targets(df: pd.DataFrame) -> pd.DataFrame:
    dur = df["duration_ms"]
    play = df["play_time_ms"]
    valid = dur > 0
    completion = (play / dur.where(valid)).clip(upper=1.0)
    dwell = np.where(valid, np.minimum(play, 2 * dur), np.nan).astype("float32")
    df["is_skip"] = np.where(valid, (completion < 0.5) & (play < 5000), np.nan).astype(
        "float32"
    )
    df["dwell_log"] = np.log1p(dwell).astype("float32")
    return df


def set_hash_bucket(df: pd.DataFrame, column: str, n_buckets: int) -> pd.DataFrame:
    df[f"{column}_bucket"] = df[column].apply(lambda x: _hash_to_bucket(x, n_buckets))
    return df


def _hash_to_bucket(value: str | float, n_buckets: int) -> int:
    if pd.isna(value):
        return 0
    n_valid_buckets = n_buckets - 1
    return murmurhash3_32(str(value), positive=True) % n_valid_buckets + 1
