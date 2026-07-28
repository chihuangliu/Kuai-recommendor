import pandas as pd

from kuai_recommender.data.utils import KuaiPureDatasetSplits, get_bucket_size

from .common import TRAINING_DIR, get_training_file_path
from .compute import set_engagement_targets, set_hash_bucket
from .entity_df import build_impression_frame
from .feature_repo.store import store
from .schema import USER_AUTHOR_FEATURES, USER_FEATURES, VIDEO_FEATURES


def build_history_frame(impression_df: pd.DataFrame) -> pd.DataFrame:
    df = set_engagement_targets(impression_df)
    df = set_hash_bucket(df, "user_id", get_bucket_size()["user_id"])
    df = set_hash_bucket(df, "author_id", get_bucket_size()["author_id"])
    return df


CHUNK = 20_000


def get_features(
    impressions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_feats = store.get_historical_features(
        entity_df=impressions[["user_id", "event_timestamp"]].drop_duplicates(),
        features=[f"user_features:{f}" for f in USER_FEATURES],
    ).to_df()
    video_feats = store.get_historical_features(
        entity_df=impressions[["video_id", "event_timestamp"]].drop_duplicates(),
        features=[f"video_features:{f}" for f in VIDEO_FEATURES],
    ).to_df()
    ua_feats = store.get_historical_features(
        entity_df=impressions[
            ["user_id", "author_id", "event_timestamp"]
        ].drop_duplicates(),
        features=[f"user_author_features:{f}" for f in USER_AUTHOR_FEATURES],
    ).to_df()
    return user_feats, video_feats, ua_feats


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--split",
        type=str,
        choices=[split.name.lower() for split in KuaiPureDatasetSplits],
    )
    args = argument_parser.parse_args()
    split = KuaiPureDatasetSplits[args.split.upper()]
    entity_df = build_impression_frame(split)
    impressions = build_history_frame(entity_df)
    merged_chunks = []
    for start in range(0, len(impressions), CHUNK):
        chunked = impressions.iloc[start : start + CHUNK]
        user_feats, video_feats, ua_feats = get_features(chunked)
        merged_chunks.append(
            chunked.merge(user_feats, on=["user_id", "event_timestamp"], how="left")
            .merge(video_feats, on=["video_id", "event_timestamp"], how="left")
            .merge(ua_feats, on=["user_id", "author_id", "event_timestamp"], how="left")
        )
    training = pd.concat(merged_chunks, ignore_index=True)
    assert len(training) == len(impressions), "Training data length mismatch"
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    training.to_parquet(get_training_file_path(split), index=False)
    print(f"Write {len(training)} rows to {get_training_file_path(split)}")
