from enum import StrEnum
from pathlib import Path
import os
import pandas as pd
import numpy as np
from functools import cache

_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "KuaiRand-Pure" / "data"
DATA_DIR = Path(os.environ.get("KUAI_DATA_DIR", _DEFAULT))

VIDEO_FEATURES_BASIC_PATH = DATA_DIR / "video_features_basic_pure.csv"
VIDEO_FEATURES_STATISTIC_PATH = DATA_DIR / "video_features_statistic_pure.csv"


class KuaiPureDatasetSplits(StrEnum):
    TRAIN = "log_standard_4_08_to_4_21_pure.csv"
    VAL = "log_standard_4_22_to_5_08_pure.csv"
    TEST = "log_random_4_22_to_5_08_pure.csv"


def build_splits():
    return {
        "train": {"name": KuaiPureDatasetSplits.TRAIN, "history": ()},
        "val": {
            "name": KuaiPureDatasetSplits.VAL,
            "history": (KuaiPureDatasetSplits.TRAIN,),
        },
        "test": {
            "name": KuaiPureDatasetSplits.TEST,
            "history": (KuaiPureDatasetSplits.TRAIN,),
        },
    }


SEED = 43
rng = np.random.default_rng(SEED)


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@cache
def get_bucket_size() -> dict[str, int]:
    df = pd.read_csv(DATA_DIR / KuaiPureDatasetSplits.TRAIN)[["user_id", "video_id"]]
    video_features_basic = pd.read_csv(VIDEO_FEATURES_BASIC_PATH)
    df = df.merge(
        video_features_basic[["video_id", "author_id"]], on="video_id", how="left"
    )
    return {
        "user_id": next_pow2(4 * df["user_id"].nunique()),
        "author_id": next_pow2(4 * df["author_id"].nunique()),
    }
