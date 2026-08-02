"""Unit tests for the online streaming aggregator in features/streaming.py.

``WindowAvg`` is the online (serving-path) twin of the offline
``compute._set_rolling_columns``: it folds raw events one at a time into a
per-key sliding-window mean, so the value it serves must reproduce the exact
point-in-time value the model trained on -- **skew-free by construction**.

Two layers of tests:

  * hand-computed micro-cases pin the three easy-to-break invariants -- first
    event NaN, current row excluded (``closed="left"``), events older than the
    window evicted, empty window NaN (not 0), keys isolated;
  * an **oracle test** replays a multi-key frame event-by-event through
    ``WindowAvg`` and asserts it equals ``_set_rolling_columns`` value-for-value.
    That is the real contract: one definition, both paths.

All timestamps are kept globally distinct so no two events for the same key
share a ``dt`` -- equal-timestamp tie handling is a deliberately deferred gap
(sequential append-after-read disagrees with pandas' right-open window on ties),
so we test the window maths in isolation from it.
"""

import math

import numpy as np
import pandas as pd

from kuai_recommender.features.compute import (
    _set_cumulative_columns,
    _set_rolling_columns,
)
from kuai_recommender.features.schema import BINARY_FEATURES
from kuai_recommender.features.streaming import (
    DictStateStore,
    RunningCountAgg,
    WindowAvg,
)

TZ = "Asia/Shanghai"
WINDOW = pd.Timedelta("7D")
BASE = pd.Timestamp("2022-04-08", tz=TZ)


def _ts(day: float) -> pd.Timestamp:
    return BASE + pd.Timedelta(days=day)


def _sig(*vals: float) -> np.ndarray:
    return np.asarray(vals, dtype=float)


def _agg() -> WindowAvg:
    return WindowAvg(DictStateStore(), WINDOW)


def _run(events: list[tuple], key_default="a") -> list[np.ndarray]:
    """Feed (day, [signal...]) events (optionally (key, day, [signal...])) in the
    given order and collect the served vector of each step."""
    agg = _agg()
    out = []
    for ev in events:
        if len(ev) == 2:
            day, signal = ev
            key = key_default
        else:
            key, day, signal = ev
        out.append(agg.step(key, _ts(day), _sig(*signal)))
    return out


# --- DictStateStore ----------------------------------------------------------


def test_dict_state_store_roundtrips_and_lists_items():
    store = DictStateStore()
    assert store.get("missing") is None  # unseen key -> None, not KeyError
    store.put("k", 42)
    assert store.get("k") == 42
    store.put("k", 43)  # overwrite
    assert store.get("k") == 43
    assert list(store.items()) == [("k", 43)]


# --- WindowAvg micro-cases ---------------------------------------------------


def test_first_event_is_nan():
    """No prior history in the window -> mean of empty -> NaN (matches .mean())."""
    (res,) = _run([(0, [1])])
    assert math.isnan(res[0])


def test_excludes_current_row():
    """Read happens before append: the current event never influences its own
    served value (closed="left" / right-open)."""
    res = _run([(0, [1]), (1, [0])])
    assert math.isnan(res[0][0])
    assert res[1][0] == 1.0  # window is [day0] only -> 1.0, not (1+0)/2


def test_evicts_events_older_than_window():
    """An event that falls out of the 7-day window is dropped from the mean."""
    res = _run([(0, [1]), (3, [0]), (9, [1])])
    # day9 window [2, 9): day0 (t=0 < 2) evicted, day3 kept -> mean([0]) = 0.0
    # (if day0 were NOT evicted it would be mean([1, 0]) = 0.5)
    assert res[2][0] == 0.0


def test_empty_after_eviction_is_nan():
    """Every prior event evicted -> window empty again -> NaN, not 0."""
    res = _run([(0, [1]), (10, [0])])
    # day10 window [3, 10): day0 (t=0 < 3) evicted -> empty
    assert math.isnan(res[1][0])


def test_keys_are_isolated():
    """Per-key state: one key's events never leak into another's window."""
    res = _run([("a", 0, [1]), ("b", 1, [1]), ("a", 2, [0])])
    assert math.isnan(res[0][0])  # a's first
    assert math.isnan(res[1][0])  # b's first -- does NOT see a@day0
    assert res[2][0] == 1.0  # a@day2 sees only a@day0


# --- RunningCountAgg micro-cases ---------------------------------------------


def _run_count(events: list[tuple], key_default="v") -> list[int]:
    """Feed (x,) events (optionally (key, x)) and collect served counts. The
    cumulative count is window-free, so it takes no timestamp."""
    agg = RunningCountAgg(DictStateStore())
    out = []
    for ev in events:
        key, x = ev if len(ev) == 2 else (key_default, ev[0])
        out.append(agg.step(key, x))
    return out


def test_running_count_first_event_is_zero():
    """No prior history -> 0 (cumsum - current, first row = 0)."""
    assert _run_count([(1,)]) == [0]


def test_running_count_counts_signal_not_events():
    """Served value is the count of prior *clicks* (sum of is_click), not the
    number of prior events -- a non-click event must not bump the count."""
    # clicks = [1, 0, 1] -> prior-click counts [0, 1, 1]
    # (a bare event counter would give [0, 1, 2])
    assert _run_count([(1,), (0,), (1,)]) == [0, 1, 1]


def test_running_count_keys_isolated():
    res = _run_count([("a", 1), ("b", 1), ("a", 1)])
    assert res == [0, 0, 1]  # b does not see a's click; a's 2nd sees one


def test_running_count_matches_offline_cumulative_value_for_value():
    """Replay is_click through RunningCountAgg in event-time order and assert it
    reproduces _set_cumulative_columns exactly."""
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({"is_click": rng.integers(0, 2, size=n)})
    df["video_id"] = rng.integers(1, 5, size=n)  # 4 videos
    df["dt"] = [BASE + pd.Timedelta(hours=int(h)) for h in np.arange(n) * 5]
    df = df.sample(frac=1, random_state=0)

    offline = _set_cumulative_columns(df.copy(), "video_id", ["is_click"])

    agg = RunningCountAgg(DictStateStore())
    online: dict = {}
    for idx, row in df.sort_values("dt").iterrows():
        online[idx] = agg.step(row["video_id"], int(row["is_click"]))

    for idx in df.index:
        assert online[idx] == offline.loc[idx, "is_click_cumulative_video_id"], idx


# --- oracle: online WindowAvg == offline _set_rolling_columns -----------------


def test_matches_offline_rolling_value_for_value():
    """Replay a multi-key, 8-signal frame through WindowAvg in event-time order
    and assert it reproduces _set_rolling_columns exactly -- the skew-free
    guarantee. Timestamps are distinct (no ties, the deferred gap), the span
    (10 days) exceeds the 7-day window so eviction is exercised, and rows are
    shuffled so both paths' own sorts -- not input order -- decide the result."""
    rng = np.random.default_rng(0)
    n = 48
    signals = rng.integers(0, 2, size=(n, len(BINARY_FEATURES)))
    df = pd.DataFrame(signals, columns=BINARY_FEATURES)
    df["user_id"] = rng.integers(1, 5, size=n)  # 4 users
    # strictly increasing, distinct timestamps spanning 10 days (240h / 5h step)
    df["dt"] = [BASE + pd.Timedelta(hours=int(h)) for h in np.arange(n) * 5]
    df = df.sample(frac=1, random_state=0)  # shuffle rows, keep index labels

    # offline truth: same code the training pipeline ships
    offline = _set_rolling_columns(df.copy(), "user_id")
    rolling_cols = [f"{f}_rolling_user_id" for f in BINARY_FEATURES]

    # online: fold events in event-time (dt) order, keyed per user
    agg = _agg()
    online: dict = {}
    for idx, row in df.sort_values("dt").iterrows():
        signal = row[BINARY_FEATURES].to_numpy(dtype=float)
        online[idx] = agg.step(row["user_id"], row["dt"], signal)

    for idx in df.index:
        expected = offline.loc[idx, rolling_cols].to_numpy(dtype=float)
        np.testing.assert_allclose(
            online[idx], expected, equal_nan=True, err_msg=f"row {idx}"
        )
