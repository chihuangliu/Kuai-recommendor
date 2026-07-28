import pandas as pd

from kuai_recommender.data.utils import (
    DATA_DIR,
    VIDEO_FEATURES_BASIC_PATH,
    KuaiPureDatasetSplits,
)

from .schema import BINARY_FEATURES, ENGAGEMENT_INPUT_COLUMNS


def build_impression_frame(split: KuaiPureDatasetSplits) -> pd.DataFrame:
    csv_path = DATA_DIR / split.value
    df = pd.read_csv(
        csv_path,
        usecols=[
            "user_id",
            "video_id",
            "time_ms",
            *ENGAGEMENT_INPUT_COLUMNS,
            *BINARY_FEATURES,
        ],
    )
    video_features_df = pd.read_csv(
        VIDEO_FEATURES_BASIC_PATH, usecols=["video_id", "author_id"]
    )
    df = df.merge(video_features_df, on="video_id", how="left")
    df["author_id"] = df["author_id"].astype("Int64")
    df["event_timestamp"] = pd.to_datetime(
        df["time_ms"], unit="ms", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    return df
