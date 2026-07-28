from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int32, ValueType

from kuai_recommender.features.common import FEATURE_SOURCES_DIR
from kuai_recommender.features.schema import (
    USER_AUTHOR_FEATURES,
    USER_FEATURES,
    VIDEO_FEATURES_FLOAT,
    VIDEO_FEATURES_INT,
)

user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.INT64)
author = Entity(name="author", join_keys=["author_id"], value_type=ValueType.INT64)
video = Entity(name="video", join_keys=["video_id"], value_type=ValueType.INT64)

user_features = FeatureView(
    name="user_features",
    entities=[user],
    source=FileSource(
        path=str(FEATURE_SOURCES_DIR / "user.parquet"), timestamp_field="dt"
    ),
    schema=[Field(name=f, dtype=Float64) for f in USER_FEATURES],
    online=True,
    ttl=timedelta(days=365),
)

user_author_features = FeatureView(
    name="user_author_features",
    entities=[user, author],
    source=FileSource(
        path=str(FEATURE_SOURCES_DIR / "user_author.parquet"), timestamp_field="dt"
    ),
    schema=[Field(name=f, dtype=Float64) for f in USER_AUTHOR_FEATURES],
    online=True,
    ttl=timedelta(days=365),
)

video_features = FeatureView(
    name="video_features",
    entities=[video],
    source=FileSource(
        path=str(FEATURE_SOURCES_DIR / "video.parquet"), timestamp_field="dt"
    ),
    schema=[Field(name=f, dtype=Float64) for f in VIDEO_FEATURES_FLOAT]
    + [Field(name=f, dtype=Int32) for f in VIDEO_FEATURES_INT],
    online=True,
    ttl=timedelta(days=365),
)
