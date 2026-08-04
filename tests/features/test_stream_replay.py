"""Test the warm-up driver in scripts/stream_replay.py.

warmup() folds the whole standard-train stream through WindowFeatureAgg to
*seed per-key state* (it writes nothing and keeps no served values -- state, not
emission). The per-aggregator maths is already pinned by tests/features/
test_streaming.py; this test targets the **wiring** warmup adds on top:

  * the raw ``BINARY_SIGNALS`` signal is what reaches the aggregators (not the
    precomputed rolling columns),
  * the user-author key is the integer pair ``(user_id, author_id)``, matching
    the offline source / Feast entity -- ``attach_author_id`` upstream guarantees
    author_id is a non-null int64, so nothing here handles a missing author,
  * every entity that appeared is present, and the video click counter equals the
    per-video sum of ``is_click``.

build_base_frame is monkeypatched to a tiny hand-built frame so we don't read the
real multi-million-row CSV.

The replay()/sink behaviour that used to live here was written against the old
``replay(agg, split, store)`` signature and is superseded by
tests/features/test_replay_sinks.py, which covers the same ground against the
current ``replay(wf_agg, df, sinks)`` API (OnlineStoreSink schemas, EvalSink
metrics, and the combined one-pass run).
"""

import pandas as pd

from kuai_recommender.features.registry import BINARY_SIGNALS
from kuai_recommender.scripts import stream_replay

TZ = "Asia/Shanghai"


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A stand-in for build_base_frame(): raw log columns with dt attached and
    author_id already merged as a non-null int64."""
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(TZ)
    for col in BINARY_SIGNALS:
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
