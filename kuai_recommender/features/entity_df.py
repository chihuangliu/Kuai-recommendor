import pandas as pd

from kuai_recommender.data.utils import (
    DATA_DIR,
    KuaiPureDatasetSplits,
    attach_author_id,
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
    df = attach_author_id(df)
    df["event_timestamp"] = pd.to_datetime(
        df["time_ms"], unit="ms", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    return df
