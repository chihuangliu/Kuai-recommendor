"""Tests for the sink-decoupled replay refactor.

The current ``test_stream_replay.py`` stays as the live net for the pre-refactor
code; these target the new seams:

    prepare_frame(split, start_dt=None, end_dt=None, *, with_targets=True) -> DataFrame
        select rows, filter to the half-open window [start, end), attach targets,
        add hour_floor.

    replay(wf_agg, df, sinks) -> None
        thin producer: for each hour-bucket, on_hour_start -> (step + on_event per
        event) -> on_hour_end, then on_finish. Calls wf_agg.step ONCE per event.
        df must already carry hour_floor (prepare_frame's job).

    class ReplaySink(Protocol): on_event / on_hour_start / on_hour_end / on_finish
    class OnlineStoreSink(store)                     # sink A
    class ServedFeaturesSource(FeatureSource):       # E1 strategy
        from_online = False; get_features(event, served) -> np.ndarray
    class EvalSink(model, feature_source, eval_config)   # sink B; EvalConfig kept
"""

import json

import numpy as np
import pandas as pd
import torch

import kuai_recommender
from kuai_recommender.data.utils import KuaiPureDatasetSplits
from kuai_recommender.features.compute import set_engagement_targets
from kuai_recommender.features.registry import (
    UA,
    USER,
    VIDEO,
    BINARY_SIGNALS,
    BINARY_TARGETS,
    CONTINUOUS_TARGETS,
    MODEL_CONTINUOUS,
    Kind,
    get_cols,
)
from kuai_recommender.scripts import stream_replay
from kuai_recommender.features.streaming import ServedFeatures, WindowFeatureAgg

# The per-view slices these sinks zip positionally against the served arrays.
USER_FEATURES = [c.name for c in get_cols(USER, Kind.ROLLING)]
USER_AUTHOR_FEATURES = [c.name for c in get_cols(UA, Kind.ROLLING)]
VIDEO_FEATURES_FLOAT = [c.name for c in get_cols(VIDEO, Kind.ROLLING)]
VIDEO_FEATURES_INT = [c.name for c in get_cols(VIDEO, Kind.CUMULATIVE)]

TZ = "Asia/Shanghai"
SPLIT = KuaiPureDatasetSplits.TEST_STANDARD


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _frame(rows: list[dict]) -> pd.DataFrame:
    """build_base_frame() stand-in: raw log columns, dt tz-aware, author_id int64,
    every BINARY_SIGNALS column present (0 by default)."""
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(TZ)
    for col in BINARY_SIGNALS:
        if col not in df.columns:
            df[col] = 0
    assert df["author_id"].dtype == "int64"  # the attach_author_id postcondition
    return df


def _prepared(df: pd.DataFrame) -> pd.DataFrame:
    """What prepare_frame hands to replay: sorted by dt, hour_floor attached."""
    df = df.sort_values("dt").reset_index(drop=True)
    df["hour_floor"] = df["dt"].dt.floor("h")
    return df


def _eval_config(monkeypatch, tmp_path, id_: str = "e1test"):
    """Build an EvalConfig whose derived output_dir lands under tmp.

    EvalConfig.__post_init__ derives output_dir from
    Path(kuai_recommender.__file__).parents[1] / "eval_res" / id, so redirect that
    anchor into tmp. Returns (cfg, expected_metrics_dir)."""
    monkeypatch.setattr(
        stream_replay.kuai_recommender,
        "__file__",
        str(tmp_path / "kuai_recommender" / "__init__.py"),
    )
    cfg = stream_replay.EvalConfig(id=id_, model=object(), eval_batch_size=32)
    return cfg, tmp_path / "eval_res" / id_


class _FakeStore:
    """Records write_to_online_store calls instead of touching Redis."""

    def __init__(self):
        self.pushed: dict[str, list[pd.DataFrame]] = {}

    def write_to_online_store(self, name: str, df: pd.DataFrame) -> None:
        self.pushed.setdefault(name, []).append(df)


def _fake_predict_recording(calls: list):
    """serving.predictor.predict stand-in: records each call, returns zero logits
    sized to the batch so the real prepare_*/Scores path runs without a model."""

    def fake_predict(
        model, user_ids, author_ids, video_ids, features=None, from_online=True
    ):
        calls.append({"from_online": from_online, "n": len(user_ids)})
        n = len(user_ids)
        return {
            "binary": torch.zeros((n, len(BINARY_TARGETS))),
            "continuous": torch.zeros((n, len(CONTINUOUS_TARGETS))),
        }

    return fake_predict


def _replay_frame() -> pd.DataFrame:
    """Two hour-batches. Hour 00: (u1,a10,v1) twice -> keep-latest picks the 00:30
    row. Hour 01: (u2,a20,v2) plus a second author for u1."""
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


def _eval_frame() -> pd.DataFrame:
    """Two hour-batches with duration_ms/play_time_ms so set_engagement_targets can
    derive is_skip/dwell_log. Last row has duration_ms=0 -> targets NaN -> exercises
    the mask path."""
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


# --------------------------------------------------------------------------- #
# 1. prepare_frame -- pins the eval_end_dt window bug (half-open [start, end))  #
# --------------------------------------------------------------------------- #
def test_prepare_frame_filters_half_open_window_and_adds_targets(monkeypatch):
    frame = _frame(
        [
            {
                "user_id": 1,
                "author_id": 10,
                "video_id": 1,
                "dt": "2022-04-21 12:00",
                "is_click": 1,
                "duration_ms": 10000,
                "play_time_ms": 9000,
            },
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
                "dt": "2022-04-23 12:00",
                "is_click": 0,
                "duration_ms": 8000,
                "play_time_ms": 1000,
            },
            {
                "user_id": 3,
                "author_id": 30,
                "video_id": 3,
                "dt": "2022-04-24 00:00",
                "is_click": 1,
                "duration_ms": 5000,
                "play_time_ms": 6000,
            },
        ]
    )
    monkeypatch.setattr(stream_replay, "build_base_frame", lambda split=None: frame)

    start = pd.Timestamp("2022-04-22", tz=TZ)
    end = pd.Timestamp("2022-04-24", tz=TZ)
    out = stream_replay.prepare_frame(SPLIT, start, end, with_targets=True)

    # half-open [start, end): keep 04-22 00:00 and 04-23; drop 04-21 and the 04-24
    # boundary row (this is exactly what the eval_end_dt copy-paste bug got wrong).
    assert out["dt"].min() == start
    assert out["dt"].max() < end
    assert list(out["user_id"]) == [1, 2]

    # engagement targets attached when asked
    assert "is_skip" in out.columns
    assert "dwell_log" in out.columns


# --------------------------------------------------------------------------- #
# 2. replay -- the producer dispatches ONE served stream to every sink         #
# --------------------------------------------------------------------------- #
class _RecordingSink:
    def __init__(self):
        self.starts: list = []
        self.events: list = []
        self.hours: list = []
        self.finished = 0

    def on_hour_start(self, hour):
        self.starts.append(hour)

    def on_event(self, event, served):
        self.events.append((event, served))

    def on_hour_end(self, hour):
        self.hours.append(hour)

    def on_finish(self):
        self.finished += 1


def test_replay_dispatches_one_served_stream_and_lifecycle():
    df = _prepared(_replay_frame())
    agg = WindowFeatureAgg(pd.Timedelta("7D"))
    sink = _RecordingSink()

    stream_replay.replay(agg, df, [sink])

    # one on_event per row, dt order preserved
    assert len(sink.events) == 4
    dts = [e.dt for e, _ in sink.events]
    assert dts == sorted(dts)

    # what a sink receives is the ServedFeatures namedtuple
    _, served0 = sink.events[0]
    assert isinstance(served0, ServedFeatures)

    # two hour buckets -> one start + one end each, one on_finish overall
    assert len(sink.starts) == 2
    assert len(sink.hours) == 2
    assert sink.finished == 1


# --------------------------------------------------------------------------- #
# 3. OnlineStoreSink -- sink A schemas                                          #
# --------------------------------------------------------------------------- #
def test_online_store_sink_pushes_correct_schemas():
    df = _prepared(_replay_frame())
    agg = WindowFeatureAgg(pd.Timedelta("7D"))
    fake = _FakeStore()

    stream_replay.replay(agg, df, [stream_replay.OnlineStoreSink(fake)])

    assert set(fake.pushed) == {
        "user_features",
        "user_author_features",
        "video_features",
    }

    def cols(name: str) -> set:
        return set().union(*(d.columns for d in fake.pushed[name]))

    def combined(name: str) -> pd.DataFrame:
        return pd.concat(fake.pushed[name], ignore_index=True)

    # user_features: entity key + dt + rolling floats
    assert cols("user_features") == {"user_id", "dt", *USER_FEATURES}
    u = combined("user_features")
    assert pd.api.types.is_datetime64_any_dtype(u["dt"])

    # keep-latest within hour 00: only the 00:30 u1 row
    batch0 = fake.pushed["user_features"][0]
    assert list(batch0["user_id"]) == [1]
    assert batch0["dt"].iloc[0] == pd.Timestamp("2022-04-22 00:30", tz=TZ)

    # user_author_features: (user, author) keyed, never null
    assert cols("user_author_features") == {
        "user_id",
        "author_id",
        "dt",
        *USER_AUTHOR_FEATURES,
    }
    ua = combined("user_author_features")
    assert ua["author_id"].notna().all()
    assert set(ua["author_id"]) == {10, 20, 30}

    # video_features: floats + int cumulative counter
    assert cols("video_features") == {
        "video_id",
        "dt",
        *VIDEO_FEATURES_FLOAT,
        *VIDEO_FEATURES_INT,
    }


# --------------------------------------------------------------------------- #
# 4. ServedFeaturesSource (E1 strategy) -- picks the right CONTINUOUS_FEATURES  #
# --------------------------------------------------------------------------- #
def test_served_features_source_selects_continuous_features_in_order():
    source = stream_replay.ServedFeaturesSource()
    assert source.from_online is False

    # give every one of the 25 served slots a globally-distinct value so a wrong
    # pick or wrong order is visible.
    user = np.arange(0, len(USER_FEATURES), dtype=np.float32)
    user_author = np.arange(100, 100 + len(USER_AUTHOR_FEATURES), dtype=np.float32)
    video = np.arange(200, 200 + len(VIDEO_FEATURES_FLOAT), dtype=np.float32)
    video_cum = np.arange(300, 300 + len(VIDEO_FEATURES_INT), dtype=np.int32)
    served = ServedFeatures(
        user=user, user_author=user_author, video=video, video_cum=video_cum
    )

    named = {
        **dict(zip(USER_FEATURES, user)),
        **dict(zip(USER_AUTHOR_FEATURES, user_author)),
        **dict(zip(VIDEO_FEATURES_FLOAT, video)),
        **dict(zip(VIDEO_FEATURES_INT, video_cum)),
    }
    expected = np.array(
        [named[f] for f in MODEL_CONTINUOUS], dtype=np.float32
    )

    x = source.get_features(event=None, served=served)
    assert x.shape == (len(MODEL_CONTINUOUS),)
    assert np.array_equal(x, expected)


# --------------------------------------------------------------------------- #
# 5. EvalSink -- scores the fold output, dumps metrics, touches no store        #
# --------------------------------------------------------------------------- #
def test_eval_sink_scores_and_dumps_metrics(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(stream_replay, "predict", _fake_predict_recording(calls))

    df = _prepared(set_engagement_targets(_eval_frame()))
    cfg, expected_dir = _eval_config(monkeypatch, tmp_path)
    agg = WindowFeatureAgg(pd.Timedelta("7D"))
    sink = stream_replay.EvalSink(
        model=object(),  # unused: predict is monkeypatched
        feature_source=stream_replay.ServedFeaturesSource(),
        eval_config=cfg,
    )

    stream_replay.replay(agg, df, [sink])

    # scored offline, once per hour bucket
    assert calls, "predict was never called"
    assert all(c["from_online"] is False for c in calls)
    assert len(calls) == 2

    metrics = json.loads((expected_dir / "metrics.json").read_text())
    assert set(metrics) == {"binary", "continuous"}
    assert set(metrics["binary"]) == set(BINARY_TARGETS)
    assert set(metrics["continuous"]) == set(CONTINUOUS_TARGETS)


# --------------------------------------------------------------------------- #
# 6. INTEGRATION -- both sinks over ONE pass; step() runs once per event        #
# --------------------------------------------------------------------------- #
def test_replay_runs_write_and_eval_in_one_pass(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(stream_replay, "predict", _fake_predict_recording(calls))

    df = _prepared(set_engagement_targets(_eval_frame()))
    cfg, expected_dir = _eval_config(monkeypatch, tmp_path)
    agg = WindowFeatureAgg(pd.Timedelta("7D"))

    # anti-double-consume guard: the stateful step() must fire once PER EVENT,
    # not once per sink, or the two sinks would see divergent state.
    n_events = len(df)
    counter = {"n": 0}
    orig_step = agg.step

    def counting_step(**kw):
        counter["n"] += 1
        return orig_step(**kw)

    agg.step = counting_step

    fake = _FakeStore()
    online = stream_replay.OnlineStoreSink(fake)
    ev = stream_replay.EvalSink(
        model=object(),
        feature_source=stream_replay.ServedFeaturesSource(),
        eval_config=cfg,
    )

    stream_replay.replay(agg, df, [online, ev])

    assert counter["n"] == n_events  # once per event, regardless of sink count

    # sink A wrote all three views ...
    assert set(fake.pushed) == {
        "user_features",
        "user_author_features",
        "video_features",
    }
    # ... and sink B scored + dumped, from the same pass
    assert calls and all(c["from_online"] is False for c in calls)
    assert (expected_dir / "metrics.json").exists()
