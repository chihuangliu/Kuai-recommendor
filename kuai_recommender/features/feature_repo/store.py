from feast import FeatureStore
from feast.repo_config import RepoConfig

from kuai_recommender.utils.data import DATA_ROOT

from .definitions import (
    author,
    user,
    user_author_features,
    user_features,
    video,
    video_features,
)

FEAST_DIR = DATA_ROOT / "feast"
FEAST_DIR.mkdir(parents=True, exist_ok=True)

config = RepoConfig(
    project="kuai_recommender",
    provider="local",
    registry=str(FEAST_DIR / "registry.db"),
    offline_store={"type": "file"},
    online_store={"type": "redis", "connection_string": "localhost:6379"},
    entity_key_serialization_version=3,
)

store = FeatureStore(config=config)
if __name__ == "__main__":
    store.apply(
        [user, author, video, user_features, user_author_features, video_features]
    )
