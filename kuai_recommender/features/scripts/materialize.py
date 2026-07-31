from datetime import datetime
from zoneinfo import ZoneInfo

from ..feature_repo.store import store

if __name__ == "__main__":
    start_date = datetime(2022, 4, 8, tzinfo=ZoneInfo("Asia/Shanghai"))
    end_date = datetime(2022, 5, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    store.materialize(
        start_date,
        end_date,
        feature_views=["user_features", "user_author_features", "video_features"],
    )
