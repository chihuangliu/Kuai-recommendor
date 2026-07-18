from enum import StrEnum
from pathlib import Path
import os

import numpy as np

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
