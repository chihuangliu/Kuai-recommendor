from pathlib import Path

from kuai_recommender.data.utils import KuaiPureDatasetSplits
from kuai_recommender.utils.data import DATA_ROOT

FEATURE_SOURCES_DIR = DATA_ROOT / "feature_sources"
TRAINING_DIR = DATA_ROOT / "training"


def get_training_file_path(split: KuaiPureDatasetSplits) -> Path:
    return TRAINING_DIR / f"{split.name.lower()}.parquet"
