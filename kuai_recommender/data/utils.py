import os
from enum import StrEnum
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "KuaiRand-Pure" / "data"
DATA_DIR = Path(os.environ.get("KUAI_DATA_DIR", _DEFAULT))

VIDEO_FEATURES_BASIC_PATH = DATA_DIR / "video_features_basic_pure.csv"
VIDEO_FEATURES_STATISTIC_PATH = DATA_DIR / "video_features_statistic_pure.csv"


class KuaiPureDatasetSplits(StrEnum):
    TRAIN = "log_standard_4_08_to_4_21_pure.csv"
    TEST_STANDARD = "log_standard_4_22_to_5_08_pure.csv"
    TEST_RANDOM = "log_random_4_22_to_5_08_pure.csv"


def build_splits():
    return {
        "train": {"name": KuaiPureDatasetSplits.TRAIN, "history": ()},
        "val": {
            "name": KuaiPureDatasetSplits.TEST_STANDARD,
            "history": (KuaiPureDatasetSplits.TRAIN,),
        },
        "test": {
            "name": KuaiPureDatasetSplits.TEST_RANDOM,
            "history": (KuaiPureDatasetSplits.TRAIN,),
        },
    }


SEED = 43
rng = np.random.default_rng(SEED)


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@cache
def _author_lookup() -> pd.DataFrame:
    return pd.read_csv(VIDEO_FEATURES_BASIC_PATH, usecols=["video_id", "author_id"])


def attach_author_id(df: pd.DataFrame) -> pd.DataFrame:
    """Join ``author_id`` onto an impression frame as a non-null ``int64`` column.

    This is the single place the video -> author join happens, so ``author_id`` is
    total everywhere downstream: no NaN checks, no float/int round trips. A missing
    author means ``video_features_basic`` is out of sync with the log, which is a
    data bug rather than a row to silently drop.
    """
    out = df.merge(_author_lookup(), on="video_id", how="left", validate="many_to_one")
    missing = out["author_id"].isna()
    if missing.any():
        unknown = sorted(out.loc[missing, "video_id"].unique())
        raise ValueError(
            f"{missing.sum()} impressions over {len(unknown)} video_id(s) have no "
            f"author_id; video_features_basic is out of sync with the log. "
            f"First unknown video_ids: {unknown[:10]}"
        )
    out["author_id"] = out["author_id"].astype("int64")
    return out


@cache
def get_bucket_size() -> dict[str, int]:
    df = pd.read_csv(DATA_DIR / KuaiPureDatasetSplits.TRAIN)[["user_id", "video_id"]]
    df = attach_author_id(df)
    return {
        "user_id": next_pow2(4 * df["user_id"].nunique()),
        "author_id": next_pow2(4 * df["author_id"].nunique()),
    }
