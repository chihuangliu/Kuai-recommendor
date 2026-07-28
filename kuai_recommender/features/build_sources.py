from kuai_recommender.features.common import FEATURE_SOURCES_DIR
from kuai_recommender.features.compute import (
    build_base_frame,
    build_user_author_source,
    build_user_source,
    build_video_source,
)

if __name__ == "__main__":
    base = build_base_frame()
    user_source = build_user_source(base)
    video_source = build_video_source(base)
    user_author_source = build_user_author_source(base)

    FEATURE_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    user_source.to_parquet(FEATURE_SOURCES_DIR / "user.parquet")
    print(
        f"User source with {len(user_source)} rows saved to {FEATURE_SOURCES_DIR / 'user.parquet'}"
    )
    video_source.to_parquet(FEATURE_SOURCES_DIR / "video.parquet")
    print(
        f"Video source with {len(video_source)} rows saved to {FEATURE_SOURCES_DIR / 'video.parquet'}"
    )
    user_author_source.to_parquet(FEATURE_SOURCES_DIR / "user_author.parquet")
    print(
        f"User author source with {len(user_author_source)} rows saved to {FEATURE_SOURCES_DIR / 'user_author.parquet'}"
    )
