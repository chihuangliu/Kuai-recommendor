import numpy as np
import pandas as pd
from feast import FeatureStore

from ...data.utils import KuaiPureDatasetSplits
from ..compute import build_base_frame
from ..feature_repo.store import store
from ..schema import (
    BINARY_FEATURES,
    USER_AUTHOR_FEATURES,
    USER_FEATURES,
    VIDEO_FEATURES_FLOAT,
    VIDEO_FEATURES_INT,
)
from ..streaming import WindowFeatureAgg


def warmup() -> WindowFeatureAgg:
    base_df = build_base_frame().sort_values("dt")
    user_ids = base_df["user_id"].to_numpy(dtype=np.int64)
    author_ids = base_df["author_id"].to_numpy(dtype=np.int64)
    video_ids = base_df["video_id"].to_numpy(dtype=np.int64)
    is_clicks = base_df["is_click"].to_numpy(dtype=np.int8)
    dts = list(base_df["dt"])
    signals = base_df[BINARY_FEATURES].to_numpy(dtype=np.float32)
    wf_agg = WindowFeatureAgg(pd.Timedelta("7D"))

    for i in range(len(base_df)):
        wf_agg.step(
            user_id=user_ids[i],
            author_id=author_ids[i],
            video_id=video_ids[i],
            ts=dts[i],
            is_click=is_clicks[i],
            signal=signals[i],
        )
    return wf_agg


def replay(
    wf_agg: WindowFeatureAgg, split: KuaiPureDatasetSplits, store: FeatureStore
) -> None:
    df = build_base_frame(split).sort_values("dt")
    df["hour_floor"] = df["dt"].dt.floor("h")
    for _, batch_df in df.groupby("hour_floor"):
        latest = {
            "user": {},
            "ua": {},
            "video": {},
        }
        for event in batch_df.itertuples(index=False):
            user_id = event.user_id
            author_id = event.author_id
            video_id = event.video_id
            dt = event.dt
            served = wf_agg.step(
                user_id=user_id,
                author_id=author_id,
                video_id=video_id,
                ts=dt,
                is_click=event.is_click,
                signal=np.array(
                    [getattr(event, f) for f in BINARY_FEATURES], dtype=np.float32
                ),
            )
            latest["user"][user_id] = (dt, served.user)
            latest["video"][video_id] = (dt, served.video, served.video_cum)
            latest["ua"][(user_id, author_id)] = (dt, served.user_author)

        def _feat2dict(
            signal: np.ndarray, feature_names: list[str]
        ) -> dict[str, float | int]:
            return {name: value for name, value in zip(feature_names, signal)}

        store.write_to_online_store(
            "user_features",
            pd.DataFrame(
                [
                    {
                        "user_id": user_id,
                        "dt": dt,
                        **_feat2dict(features, USER_FEATURES),
                    }
                    for user_id, (dt, features) in latest["user"].items()
                ]
            ),
        )
        store.write_to_online_store(
            "user_author_features",
            pd.DataFrame(
                [
                    {
                        "user_id": user_id,
                        "author_id": author_id,
                        "dt": dt,
                        **_feat2dict(features, USER_AUTHOR_FEATURES),
                    }
                    for (user_id, author_id), (dt, features) in latest["ua"].items()
                ]
            ),
        )
        store.write_to_online_store(
            "video_features",
            pd.DataFrame(
                [
                    {
                        "video_id": video_id,
                        "dt": dt,
                        **_feat2dict(features, VIDEO_FEATURES_FLOAT),
                        **_feat2dict(video_cum, VIDEO_FEATURES_INT),
                    }
                    for video_id, (dt, features, video_cum) in latest["video"].items()
                ]
            ),
        )


if __name__ == "__main__":
    wf_agg = warmup()
    replay(wf_agg, KuaiPureDatasetSplits.TEST_STANDARD, store)
