"""Test the warm-up driver in scripts/stream_replay.py.

warmup() folds the whole standard-train stream through WindowFeatureAgg to
*seed per-key state* (it writes nothing and keeps no served values -- state, not
emission). The per-aggregator maths is already pinned by tests/features/
test_streaming.py; this test targets the **wiring** warmup adds on top:

  * the raw ``BINARY_FEATURES`` signal is what reaches the aggregators (not the
    precomputed rolling columns),
  * a row whose video has no author (NaN join key) is *excluded* from the
    user-author state -- never keyed under a bogus author,
  * the user-author key is the integer pair ``(user_id, author_id)``, matching
    the offline source / Feast entity,
  * every entity that appeared is present, and the video click counter equals the
    per-video sum of ``is_click``.

build_base_frame is monkeypatched to a tiny hand-built frame so we don't read the
real multi-million-row CSV.
"""

import numpy as np
import pandas as pd

from kuai_recommender.data.utils import KuaiPureDatasetSplits
from kuai_recommender.features.scripts import stream_replay
from kuai_recommender.features.schema import (
    BINARY_FEATURES,
    USER_AUTHOR_FEATURES,
    USER_FEATURES,
    VIDEO_FEATURES_FLOAT,
    VIDEO_FEATURES_INT,
)
from kuai_recommender.features.streaming import WindowFeatureAgg

TZ = "Asia/Shanghai"


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A stand-in for build_base_frame(): raw log columns with dt attached and
    author_id already merged (NaN where the video has no author)."""
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(TZ)
    for col in BINARY_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df


def _keys(store) -> set:
    return {k for k, _ in store.items()}


def test_warmup_seeds_state_with_correct_wiring(monkeypatch):
    frame = _frame(
        [
            {"user_id": 1, "author_id": 10, "video_id": 1, "dt": "2022-04-08", "is_click": 1},
            {"user_id": 1, "author_id": 10, "video_id": 2, "dt": "2022-04-09", "is_click": 0},
            {"user_id": 1, "author_id": np.nan, "video_id": 3, "dt": "2022-04-10", "is_click": 1},
            {"user_id": 2, "author_id": 20, "video_id": 1, "dt": "2022-04-08", "is_click": 1},
            {"user_id": 2, "author_id": 20, "video_id": 4, "dt": "2022-04-11", "is_click": 0},
        ]
    )
    monkeypatch.setattr(stream_replay, "build_base_frame", lambda: frame)

    wf = stream_replay.warmup()

    # every user / video that appeared is seeded
    assert _keys(wf.user_avg.store) == {1, 2}
    assert _keys(wf.video_avg.store) == {1, 2, 3, 4}

    # user-author: NaN-author row (u1,v3) excluded; keys are integer pairs
    assert _keys(wf.ua_avg.store) == {(1, 10), (2, 20)}

    # cumulative click counter == per-video sum of is_click (v1 seen twice, both clicks)
    cum = {k: v.count for k, v in wf.video_cum.store.items()}
    assert cum == {1: 2, 2: 0, 3: 1, 4: 0}


# --- replay(): sink A -- fold test stream, push per-hour to the online store ---


class _FakeStore:
    """Records write_to_online_store calls instead of touching Redis. Keyed by
    feature-view name -> list of the per-batch DataFrames pushed for it."""

    def __init__(self):
        self.pushed: dict[str, list[pd.DataFrame]] = {}

    def write_to_online_store(self, name: str, df: pd.DataFrame) -> None:
        self.pushed.setdefault(name, []).append(df)


def _replay_frame() -> pd.DataFrame:
    """Two hour-batches. Hour 00: (u1,a10,v1) twice -> keep-latest picks the
    00:30 row. Hour 01: (u2,a20,v2) plus a NaN-author row (u1,v3) that must be
    dropped from the user-author sink."""
    return _frame(
        [
            {"user_id": 1, "author_id": 10, "video_id": 1, "dt": "2022-04-22 00:00", "is_click": 1},
            {"user_id": 1, "author_id": 10, "video_id": 1, "dt": "2022-04-22 00:30", "is_click": 0},
            {"user_id": 2, "author_id": 20, "video_id": 2, "dt": "2022-04-22 01:00", "is_click": 1},
            {"user_id": 1, "author_id": np.nan, "video_id": 3, "dt": "2022-04-22 01:30", "is_click": 1},
        ]
    )


def test_replay_pushes_correct_schemas(monkeypatch):
    frame = _replay_frame()
    monkeypatch.setattr(stream_replay, "build_base_frame", lambda split=None: frame)

    agg = WindowFeatureAgg(pd.Timedelta("7D"))
    fake = _FakeStore()
    stream_replay.replay(agg, KuaiPureDatasetSplits.TEST_STANDARD, fake)

    # all three feature views pushed
    assert set(fake.pushed) == {
        "user_features",
        "user_author_features",
        "video_features",
    }

    def cols(name: str) -> set:
        return set().union(*(df.columns for df in fake.pushed[name]))

    def combined(name: str) -> pd.DataFrame:
        return pd.concat(fake.pushed[name], ignore_index=True)

    # --- user_features: entity key + dt timestamp + 8 rolling floats ----------
    assert cols("user_features") == {"user_id", "dt", *USER_FEATURES}
    u = combined("user_features")
    assert pd.api.types.is_datetime64_any_dtype(u["dt"])  # dt stays a Timestamp

    # keep-latest within a batch: hour-00 pushed one u1 row, the 00:30 one
    batch0 = fake.pushed["user_features"][0]
    assert list(batch0["user_id"]) == [1]
    assert batch0["dt"].iloc[0] == pd.Timestamp("2022-04-22 00:30", tz=TZ)

    # --- user_author_features: NaN-author row excluded, keys never null -------
    assert cols("user_author_features") == {
        "user_id",
        "author_id",
        "dt",
        *USER_AUTHOR_FEATURES,
    }
    ua = combined("user_author_features")
    assert ua["author_id"].notna().all()
    assert set(ua["author_id"]) == {10, 20}  # (u1, v3/NaN) never keyed

    # --- video_features: floats + the int cumulative counter ------------------
    assert cols("video_features") == {
        "video_id",
        "dt",
        *VIDEO_FEATURES_FLOAT,
        *VIDEO_FEATURES_INT,
    }
    v = combined("video_features")
    assert pd.api.types.is_datetime64_any_dtype(v["dt"])
