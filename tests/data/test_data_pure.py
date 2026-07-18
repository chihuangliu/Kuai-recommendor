"""Unit tests for KuaiPureData._set_rolling_columns.

These exercise the point-in-time rolling-rate logic in isolation: we bypass
__init__ (which reads the real CSVs) and inject a tiny hand-built frame so the
expected rolling means can be computed by hand. The properties under test are
the ones that silently broke during development:

  * strictly-before-T semantics (closed="left") -> first event of a group is NaN,
    a row never sees its own label (no leakage),
  * correct *time* window (``"7D"``), so events older than the window are excluded,
  * correct row alignment after the internal sort (the values land on the right
    rows, not shuffled), and per-group isolation,
  * distinct, collision-free column names per group_by.
"""

import math

import pandas as pd
import pytest
import torch

from kuai_recommender.data.data_pure import KuaiPureData, KuaiPureDataset

TZ = "Asia/Shanghai"


def _make(rows: list[dict]) -> KuaiPureData:
    """Build a KuaiPureData with a synthetic df, skipping __init__/CSV reads.

    Each row dict must carry ``rid`` (unique row id used for lookup), the group
    keys it needs (``user_id``/``author_id``/``video_id``), ``dt`` (a date str),
    and optionally ``is_click``. Every other binary column is filled with 0 so
    the method's loop over BINARY_COLUMNS has real columns to read.
    """
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(TZ)
    for col in KuaiPureData.BINARY_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    # Deliberately shuffle so we prove the method's own sort/alignment, not the
    # input order, is what makes the result correct.
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    obj = KuaiPureData.__new__(KuaiPureData)
    obj.df = df
    return obj


def _rates(obj: KuaiPureData, column: str) -> dict:
    """Map rid -> value for a produced rolling column."""
    return obj.df.set_index("rid")[column].to_dict()


def _assert_rate(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for rid, exp in expected.items():
        got = actual[rid]
        if exp is None:
            assert got is None or (isinstance(got, float) and math.isnan(got)), (
                f"rid={rid}: expected NaN, got {got!r}"
            )
        else:
            assert got == pytest.approx(exp), f"rid={rid}: expected {exp}, got {got!r}"


def test_first_event_is_nan_and_no_self_leakage():
    """The first impression of a group has no prior -> NaN; a row never sees itself."""
    obj = _make(
        [
            {"rid": 1, "user_id": 1, "dt": "2022-04-08", "is_click": 1},
            {"rid": 2, "user_id": 1, "dt": "2022-04-09", "is_click": 0},
        ]
    )
    obj._set_user_rolling()
    rates = _rates(obj, "is_click_rolling_user_id")
    # rid1 has no prior -> NaN. rid2 sees only rid1 (=1), not its own 0.
    _assert_rate(rates, {1: None, 2: 1.0})


def test_rolling_mean_values_and_row_alignment():
    """Exact rolling means, with users interleaved, land on the correct rows."""
    obj = _make(
        [
            {"rid": 1, "user_id": 1, "dt": "2022-04-08", "is_click": 1},
            {"rid": 2, "user_id": 2, "dt": "2022-04-08", "is_click": 0},
            {"rid": 3, "user_id": 1, "dt": "2022-04-09", "is_click": 0},
            {"rid": 4, "user_id": 1, "dt": "2022-04-10", "is_click": 1},
            {"rid": 5, "user_id": 2, "dt": "2022-04-10", "is_click": 1},
        ]
    )
    obj._set_user_rolling()
    rates = _rates(obj, "is_click_rolling_user_id")
    _assert_rate(
        rates,
        {
            1: None,  # u1 first event
            3: 1.0,  # u1 prior = [1]
            4: 0.5,  # u1 prior = [1, 0]
            2: None,  # u2 first event
            5: 0.0,  # u2 prior = [0] -- unaffected by u1's clicks
        },
    )


def test_window_excludes_events_older_than_7d():
    """A 7-day window drops events that fall outside it (true time window, not row count)."""
    obj = _make(
        [
            {"rid": 1, "user_id": 1, "dt": "2022-04-08", "is_click": 1},
            {"rid": 2, "user_id": 1, "dt": "2022-04-13", "is_click": 0},  # 5d later
            {
                "rid": 3,
                "user_id": 1,
                "dt": "2022-04-18",
                "is_click": 1,
            },  # 10d after rid1
        ]
    )
    obj._set_user_rolling()
    rates = _rates(obj, "is_click_rolling_user_id")
    _assert_rate(
        rates,
        {
            1: None,  # no prior
            2: 1.0,  # prior within 7d = [rid1] = [1]
            3: 0.0,  # prior within 7d = [rid2] = [0]; rid1 is >7d old -> excluded
        },
    )


def test_user_author_grouping_is_per_pair():
    """user-author affinity aggregates per (user_id, author_id), not per user."""
    obj = _make(
        [
            {
                "rid": 1,
                "user_id": 1,
                "author_id": 10,
                "dt": "2022-04-08",
                "is_click": 1,
            },
            {
                "rid": 2,
                "user_id": 1,
                "author_id": 10,
                "dt": "2022-04-09",
                "is_click": 1,
            },
            {
                "rid": 3,
                "user_id": 1,
                "author_id": 20,
                "dt": "2022-04-09",
                "is_click": 0,
            },
            {
                "rid": 4,
                "user_id": 1,
                "author_id": 10,
                "dt": "2022-04-10",
                "is_click": 0,
            },
        ]
    )
    obj._set_user_author_rolling()
    rates = _rates(obj, "is_click_rolling_user_id_author_id")
    _assert_rate(
        rates,
        {
            1: None,  # (1,10) first
            2: 1.0,  # (1,10) prior = [1]
            4: 1.0,  # (1,10) prior = [1, 1]; the (1,20) row does not leak in
            3: None,  # (1,20) first
        },
    )


def test_column_names_are_distinct_per_group_by():
    """Each group_by writes its own suffix, so successive calls don't collide."""
    obj = _make(
        [
            {
                "rid": 1,
                "user_id": 1,
                "author_id": 10,
                "video_id": 100,
                "dt": "2022-04-08",
                "is_click": 1,
            },
            {
                "rid": 2,
                "user_id": 1,
                "author_id": 10,
                "video_id": 100,
                "dt": "2022-04-09",
                "is_click": 0,
            },
        ]
    )
    obj._set_user_rolling()
    obj._set_user_author_rolling()
    obj._set_video_rolling()

    for suffix in ("user_id", "user_id_author_id", "video_id"):
        assert f"is_click_rolling_{suffix}" in obj.df.columns
    # All three suffixes coexist -> no overwrite.
    assert obj.df["is_click_rolling_user_id"].notna().any()
    assert obj.df["is_click_rolling_video_id"].notna().any()


def test_all_binary_columns_get_a_rolling_feature():
    """The method produces one rolling column per binary label."""
    obj = _make(
        [
            {"rid": 1, "user_id": 1, "dt": "2022-04-08"},
            {"rid": 2, "user_id": 1, "dt": "2022-04-09"},
        ]
    )
    obj._set_user_rolling()
    for col in KuaiPureData.BINARY_COLUMNS:
        assert f"{col}_rolling_user_id" in obj.df.columns


def test_rates_are_bounded_in_unit_interval():
    """Rolling means of 0/1 labels stay within [0, 1] (ignoring NaN)."""
    obj = _make(
        [
            {
                "rid": i,
                "user_id": i % 3,
                "dt": f"2022-04-{8 + (i % 10):02d}",
                "is_click": i % 2,
            }
            for i in range(1, 31)
        ]
    )
    obj._set_user_rolling()
    vals = obj.df["is_click_rolling_user_id"].dropna()
    assert ((vals >= 0) & (vals <= 1)).all()


# --- _set_cumulative_columns -------------------------------------------------
# Cumulative features are *counts* of prior positives (all history < T), so the
# expected values are integers and the first event of a group is 0 (not NaN).


def _counts(obj: KuaiPureData, column: str) -> dict:
    """Map rid -> value for a produced cumulative column."""
    return obj.df.set_index("rid")[column].to_dict()


def test_cumulative_is_strictly_before_t_no_leakage():
    """Cumulative count excludes the current row: first event is 0, self never counted."""
    obj = _make(
        [
            {"rid": 1, "video_id": 9, "dt": "2022-04-08", "is_click": 1},
            {"rid": 2, "video_id": 9, "dt": "2022-04-09", "is_click": 1},
            {"rid": 3, "video_id": 9, "dt": "2022-04-10", "is_click": 0},
        ]
    )
    obj._set_video_cumulative()
    counts = _counts(obj, "is_click_cumulative_video_id")
    # rid1: no prior -> 0. rid2: prior [1] -> 1. rid3: prior [1,1] -> 2 (its own 0 excluded).
    assert counts == {1: 0, 2: 1, 3: 2}


def test_cumulative_alignment_and_per_group():
    """Counts land on the right rows (videos interleaved) and don't leak across groups."""
    obj = _make(
        [
            {"rid": 1, "video_id": 9, "dt": "2022-04-08", "is_click": 1},
            {"rid": 2, "video_id": 8, "dt": "2022-04-08", "is_click": 1},
            {"rid": 3, "video_id": 9, "dt": "2022-04-09", "is_click": 0},
            {"rid": 4, "video_id": 9, "dt": "2022-04-10", "is_click": 1},
            {"rid": 5, "video_id": 8, "dt": "2022-04-09", "is_click": 1},
        ]
    )
    obj._set_video_cumulative()
    counts = _counts(obj, "is_click_cumulative_video_id")
    assert counts == {
        1: 0,  # v9 first
        3: 1,  # v9 prior [1]
        4: 1,  # v9 prior [1, 0] -> 1 (video 8's clicks don't leak in)
        2: 0,  # v8 first
        5: 1,  # v8 prior [1]
    }


def test_cumulative_is_never_nan_and_monotonic_per_group():
    """Every row gets a finite count (unlike rolling), non-decreasing within a group."""
    obj = _make(
        [
            {
                "rid": i,
                "video_id": i % 2,
                "dt": f"2022-04-{8 + (i % 10):02d}",
                "is_click": i % 2,
            }
            for i in range(1, 21)
        ]
    )
    obj._set_video_cumulative()
    col = obj.df["is_click_cumulative_video_id"]
    assert col.notna().all()
    ordered = obj.df.sort_values(["video_id", "dt"])
    diffs = ordered.groupby("video_id")["is_click_cumulative_video_id"].diff().dropna()
    assert (diffs >= 0).all()  # counts only accumulate


def test_cumulative_creates_a_column_per_binary_label():
    obj = _make(
        [
            {"rid": 1, "video_id": 9, "dt": "2022-04-08"},
            {"rid": 2, "video_id": 9, "dt": "2022-04-09"},
        ]
    )
    obj._set_video_cumulative()
    for col in KuaiPureData.BINARY_COLUMNS:
        assert f"{col}_cumulative_video_id" in obj.df.columns


# --- _set_rolling_columns with NaN group keys (dropna=False fix) --------------
# A video absent from the basic feature file left-joins to a NaN author_id.
# groupby(dropna=False) must keep those rows so the .values assign stays aligned.


def test_rolling_survives_nan_group_key():
    """NaN author_id rows are kept (own group), not dropped -> no length mismatch."""
    obj = _make(
        [
            {"rid": 1, "user_id": 1, "author_id": 10, "dt": "2022-04-08", "is_click": 1},
            {"rid": 2, "user_id": 1, "author_id": 10, "dt": "2022-04-09", "is_click": 0},
            {"rid": 3, "user_id": 1, "author_id": float("nan"), "dt": "2022-04-08", "is_click": 1},
            {"rid": 4, "user_id": 1, "author_id": float("nan"), "dt": "2022-04-09", "is_click": 1},
        ]
    )
    obj._set_user_author_rolling()  # would ValueError on length mismatch if NaN rows were dropped
    rates = _rates(obj, "is_click_rolling_user_id_author_id")
    # Every input row still gets a value, and NaN-author rows form their own group.
    assert set(rates.keys()) == {1, 2, 3, 4}
    _assert_rate(rates, {1: None, 2: 1.0, 3: None, 4: 1.0})


# --- KuaiPureDataset ----------------------------------------------------------
# The Dataset wraps a prepared KuaiPureData.df and must hand the model finite
# float tensors: rolling features are NaN for a group's first impression, and
# that NaN must not reach training.


def _make_dataset(rows: list[dict], features: list[str]) -> KuaiPureDataset:
    """Wrap a hand-built df in a KuaiPureDataset, bypassing KuaiPureData.__init__."""
    df = pd.DataFrame(rows)
    for col in KuaiPureData.BINARY_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    data = KuaiPureData.__new__(KuaiPureData)
    data.df = df
    return KuaiPureDataset(data, features)


def test_getitem_returns_feature_tensor_and_label_dict():
    ds = _make_dataset(
        [{"rate": 0.25, "is_click": 1}, {"rate": 0.75, "is_click": 0}],
        features=["rate"],
    )
    assert len(ds) == 2
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.dtype == torch.float32
    assert x.shape == (1,)
    assert set(y.keys()) == set(KuaiPureData.BINARY_COLUMNS)
    assert y["is_click"].item() == 1.0


def test_no_nan_features_reach_training():
    """A first-impression NaN feature (as rolling produces) must be imputed, not emitted."""
    ds = _make_dataset(
        [
            {"rate": float("nan"), "is_click": 1},  # first event -> rolling NaN
            {"rate": 0.5, "is_click": 0},
        ],
        features=["rate"],
    )
    for i in range(len(ds)):
        x, _ = ds[i]
        assert not torch.isnan(x).any(), f"row {i} leaked NaN into features"
    # NaN is mapped to 0.0 (a 0% prior rate), not dropped.
    assert ds[0][0].item() == 0.0
