"""Test the warm-up driver in scripts/stream_replay.py.

warmup() folds the whole standard-train stream through WindowFeatureAgg to
*seed per-key state* (it writes nothing and keeps no served values -- state, not
emission). The per-aggregator maths is already pinned by tests/features/
test_streaming.py; this test targets the **wiring** warmup adds on top:

  * the raw ``BINARY_FEATURES`` signal is what reaches the aggregators (not the
    precomputed rolling columns),
  * the user-author key is the integer pair ``(user_id, author_id)``, matching
    the offline source / Feast entity -- ``attach_author_id`` upstream guarantees
    author_id is a non-null int64, so nothing here handles a missing author,
  * every entity that appeared is present, and the video click counter equals the
    per-video sum of ``is_click``.

build_base_frame is monkeypatched to a tiny hand-built frame so we don't read the
real multi-million-row CSV.
"""

import json

import pandas as pd
import torch

from kuai_recommender.data.data_pure import KuaiPureData
from kuai_recommender.data.utils import KuaiPureDatasetSplits
from kuai_recommender.features.schema import (
    BINARY_FEATURES,
    USER_AUTHOR_FEATURES,
    USER_FEATURES,
    VIDEO_FEATURES_FLOAT,
    VIDEO_FEATURES_INT,
)
from kuai_recommender.scripts import stream_replay
from kuai_recommender.features.streaming import WindowFeatureAgg

TZ = "Asia/Shanghai"


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A stand-in for build_base_frame(): raw log columns with dt attached and
    author_id already merged as a non-null int64."""
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(TZ)
    for col in BINARY_FEATURES:
        if col not in df.columns:
            df[col] = 0
    assert df["author_id"].dtype == "int64"  # the attach_author_id postcondition
    return df


def _keys(store) -> set:
    return {k for k, _ in store.items()}


def test_warmup_seeds_state_with_correct_wiring(monkeypatch):
    frame = _frame(
        [
            {
                "user_id": 1,
                "author_id": 10,
                "video_id": 1,
                "dt": "2022-04-08",
                "is_click": 1,
            },
            {
                "user_id": 1,
                "author_id": 10,
                "video_id": 2,
                "dt": "2022-04-09",
                "is_click": 0,
            },
            {
                "user_id": 1,
                "author_id": 30,
                "video_id": 3,
                "dt": "2022-04-10",
                "is_click": 1,
            },
            {
                "user_id": 2,
                "author_id": 20,
                "video_id": 1,
                "dt": "2022-04-08",
                "is_click": 1,
            },
            {
                "user_id": 2,
                "author_id": 20,
                "video_id": 4,
                "dt": "2022-04-11",
                "is_click": 0,
            },
        ]
    )
    monkeypatch.setattr(stream_replay, "build_base_frame", lambda: frame)

    wf = stream_replay.warmup()

    # every user / video that appeared is seeded
    assert _keys(wf.user_avg.store) == {1, 2}
    assert _keys(wf.video_avg.store) == {1, 2, 3, 4}

    # user-author keys are integer pairs, one per distinct (user, author)
    assert _keys(wf.ua_avg.store) == {(1, 10), (1, 30), (2, 20)}

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
    00:30 row. Hour 01: (u2,a20,v2) plus a second author for u1."""
    return _frame(
        [
            {
                "user_id": 1,
                "author_id": 10,
                "video_id": 1,
                "dt": "2022-04-22 00:00",
                "is_click": 1,
            },
            {
                "user_id": 1,
                "author_id": 10,
                "video_id": 1,
                "dt": "2022-04-22 00:30",
                "is_click": 0,
            },
            {
                "user_id": 2,
                "author_id": 20,
                "video_id": 2,
                "dt": "2022-04-22 01:00",
                "is_click": 1,
            },
            {
                "user_id": 1,
                "author_id": 30,
                "video_id": 3,
                "dt": "2022-04-22 01:30",
                "is_click": 1,
            },
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

    # --- user_author_features: every (user, author) pair keyed, never null ----
    assert cols("user_author_features") == {
        "user_id",
        "author_id",
        "dt",
        *USER_AUTHOR_FEATURES,
    }
    ua = combined("user_author_features")
    assert ua["author_id"].notna().all()
    assert set(ua["author_id"]) == {10, 20, 30}

    # --- video_features: floats + the int cumulative counter ------------------
    assert cols("video_features") == {
        "video_id",
        "dt",
        *VIDEO_FEATURES_FLOAT,
        *VIDEO_FEATURES_INT,
    }
    v = combined("video_features")
    assert pd.api.types.is_datetime64_any_dtype(v["dt"])


# --- replay(): sink B (E1) -- score the fold output, no online store ----------


def _fake_predict_recording(calls: list):
    """Stand-in for serving.predictor.predict: records every call and returns
    zero logits sized to the batch, so the eval loop exercises the real
    prepare_*/BinaryScores/ContinuousScores path without a trained model."""

    def fake_predict(
        model, user_ids, author_ids, video_ids, features=None, from_online=True
    ):
        calls.append({"from_online": from_online, "n": len(user_ids)})
        n = len(user_ids)
        return {
            "binary": torch.zeros((n, len(KuaiPureData.BINARY_TARGETS))),
            "continuous": torch.zeros((n, len(KuaiPureData.CONTINUOUS_TARGETS))),
        }

    return fake_predict


def _eval_frame() -> pd.DataFrame:
    """Two hour-batches with the engagement columns set_engagement_targets needs
    (duration_ms/play_time_ms). The last row has duration_ms=0 -> invalid ->
    is_skip/dwell_log NaN, exercising the target-mask path."""
    return _frame(
        [
            {
                "user_id": 1,
                "author_id": 10,
                "video_id": 1,
                "dt": "2022-04-22 00:00",
                "is_click": 1,
                "duration_ms": 10000,
                "play_time_ms": 9000,
            },
            {
                "user_id": 2,
                "author_id": 20,
                "video_id": 2,
                "dt": "2022-04-22 00:30",
                "is_click": 0,
                "duration_ms": 8000,
                "play_time_ms": 1000,
            },
            {
                "user_id": 1,
                "author_id": 10,
                "video_id": 1,
                "dt": "2022-04-22 01:00",
                "is_click": 1,
                "duration_ms": 5000,
                "play_time_ms": 6000,
            },
            {
                "user_id": 3,
                "author_id": 30,
                "video_id": 3,
                "dt": "2022-04-22 01:30",
                "is_click": 0,
                "duration_ms": 0,
                "play_time_ms": 0,
            },
        ]
    )


def test_replay_e1_scores_fold_output_and_skips_online(monkeypatch, tmp_path):
    frame = _eval_frame()
    monkeypatch.setattr(stream_replay, "build_base_frame", lambda split=None: frame)
    calls: list = []
    monkeypatch.setattr(stream_replay, "predict", _fake_predict_recording(calls))

    # EvalConfig.__post_init__ derives output_dir from <repo_root>/eval_res/<id>,
    # where repo_root = Path(kuai_recommender.__file__).parents[1]. Redirect that
    # anchor into tmp so metrics.json never lands in the source tree.
    monkeypatch.setattr(
        stream_replay.kuai_recommender,
        "__file__",
        str(tmp_path / "kuai_recommender" / "__init__.py"),
    )
    expected_dir = tmp_path / "eval_res" / "e1test"

    agg = WindowFeatureAgg(pd.Timedelta("7D"))
    fake = _FakeStore()
    cfg = stream_replay.EvalConfig(
        id="e1test",
        model=object(),  # unused: predict is monkeypatched
        eval_batch_size=32,
    )

    stream_replay.replay(
        agg,
        KuaiPureDatasetSplits.TEST_STANDARD,
        fake,
        write_to_online=False,
        eval=True,
        eval_config=cfg,
    )

    # E1 never touches the online store
    assert fake.pushed == {}

    # scored the in-process served features (offline path), once per hour-batch
    assert calls, "predict was never called"
    assert all(c["from_online"] is False for c in calls)
    assert len(calls) == 2  # two hour buckets

    # metrics.json lands in the derived (redirected) dir with the score structure
    metrics = json.loads((expected_dir / "metrics.json").read_text())
    assert set(metrics) == {"binary", "continuous"}
    assert set(metrics["binary"]) == set(KuaiPureData.BINARY_TARGETS)
    assert set(metrics["continuous"]) == set(KuaiPureData.CONTINUOUS_TARGETS)
    assert all(isinstance(v, dict) for v in metrics["binary"].values())
