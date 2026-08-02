"""Unit tests for the offline feature-source builders in features/compute.py.

These target the *source-table layer* (Phase 0 of the Feast migration): the
functions take a base frame -- the full standard log with ``dt`` + ``author_id``
already attached -- and emit one tidy, entity-keyed, event-timestamped table per
FeatureView. We inject a tiny hand-built base frame (bypassing build_base_frame,
which reads the real CSVs) so every expected value is computable by hand.

The properties under test are the ones specific to this layer (the raw rolling /
cumulative maths is covered in tests/data/test_data_pure.py):

  * each source carries its Feast **join keys + event timestamp** -- user_id;
    (user_id, author_id); video_id -- plus exactly its own feature columns, and
    nothing extra (a stray column silently changes the FileSource schema),
  * point-in-time semantics survive the builder wrapper (closed="left" -> first
    impression NaN, no self-leakage),
  * the video source actually produces the cumulative column (regression: a str
    passed where a list of columns was expected iterates characters -> KeyError),
  * build_user_author_source carries an integer author_id straight through --
    ``attach_author_id`` already guarantees the key is total, so this layer does
    no NaN handling of its own (see tests/data/test_attach_author_id.py), and
  * the builders don't mutate the shared base frame (each takes its own sorted
    copy), so all three can run off one base_frame.
"""

import math

import pandas as pd
import pytest

from kuai_recommender.features.compute import (
    build_user_source,
    build_user_author_source,
    build_video_source,
)
from kuai_recommender.features.schema import (
    BINARY_FEATURES,
    USER_FEATURES,
    USER_AUTHOR_FEATURES,
    VIDEO_FEATURES,
)

TZ = "Asia/Shanghai"


def _base(rows: list[dict]) -> pd.DataFrame:
    """A synthetic base frame: the columns build_*_source read off base_frame.

    Each row supplies the ids it needs (user_id / author_id / video_id), a ``dt``
    date string, and optionally ``is_click``; every other binary column defaults
    to 0 so the builders' loop over BINARY_FEATURES has real columns to read. Rows
    are shuffled so the tests prove the builders' own sort/alignment, not the
    input order, produces the result.
    """
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(TZ)
    for col in BINARY_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def _seq(df: pd.DataFrame, sort_keys: list[str], col: str) -> list:
    """Values of ``col`` in (sort_keys, dt) order -- the natural per-group sequence."""
    return df.sort_values(sort_keys + ["dt"])[col].tolist()


def _assert_seq(actual: list, expected: list) -> None:
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        if e is None:
            assert isinstance(a, float) and math.isnan(a), f"expected NaN, got {a!r}"
        else:
            assert a == pytest.approx(e), f"expected {e}, got {a!r}"


# --- user source -------------------------------------------------------------


def test_user_source_has_join_key_timestamp_and_only_its_features():
    """Columns == dt + user_id + USER_FEATURES, in that shape, nothing extra."""
    out = build_user_source(
        _base(
            [
                {"user_id": 1, "video_id": 100, "author_id": 10, "dt": "2022-04-08"},
                {"user_id": 1, "video_id": 101, "author_id": 10, "dt": "2022-04-09"},
            ]
        )
    )
    assert list(out.columns) == ["dt", "user_id"] + USER_FEATURES
    # every binary label produced exactly one rolling feature column
    for col in BINARY_FEATURES:
        assert f"{col}_rolling_user_id" in out.columns


def test_user_source_is_point_in_time_and_per_user():
    """closed="left": first impression NaN, a row never sees itself, users isolated."""
    out = build_user_source(
        _base(
            [
                {"user_id": 1, "video_id": 1, "author_id": 10, "dt": "2022-04-08", "is_click": 1},
                {"user_id": 1, "video_id": 2, "author_id": 10, "dt": "2022-04-09", "is_click": 0},
                {"user_id": 1, "video_id": 3, "author_id": 10, "dt": "2022-04-10", "is_click": 1},
                {"user_id": 2, "video_id": 4, "author_id": 10, "dt": "2022-04-08", "is_click": 0},
                {"user_id": 2, "video_id": 5, "author_id": 10, "dt": "2022-04-10", "is_click": 1},
            ]
        )
    )
    seq = _seq(out, ["user_id"], "is_click_rolling_user_id")
    # u1: [NaN, prior=[1]->1.0, prior=[1,0]->0.5] ; u2: [NaN, prior=[0]->0.0]
    _assert_seq(seq, [None, 1.0, 0.5, None, 0.0])


# --- user-author source ------------------------------------------------------


def test_user_author_source_carries_both_join_keys():
    """The entity is the *pair* -- both user_id and author_id must be present."""
    out = build_user_author_source(
        _base(
            [
                {"user_id": 1, "video_id": 1, "author_id": 10, "dt": "2022-04-08"},
                {"user_id": 1, "video_id": 2, "author_id": 10, "dt": "2022-04-09"},
            ]
        )
    )
    assert list(out.columns) == ["dt", "user_id", "author_id"] + USER_AUTHOR_FEATURES


def test_user_author_source_rolls_per_author_pair():
    """Affinity is scoped to the (user, author) pair, not to the user alone."""
    out = build_user_author_source(
        _base(
            [
                {"user_id": 1, "video_id": 1, "author_id": 10, "dt": "2022-04-08", "is_click": 1},
                {"user_id": 1, "video_id": 2, "author_id": 10, "dt": "2022-04-09", "is_click": 0},
                {"user_id": 1, "video_id": 3, "author_id": 20, "dt": "2022-04-08", "is_click": 1},
                {"user_id": 1, "video_id": 4, "author_id": 20, "dt": "2022-04-09", "is_click": 1},
            ]
        )
    )
    assert len(out) == 4
    # per (user, author): [NaN first, prior=[1]->1.0] then [NaN first, prior=[1]->1.0]
    _assert_seq(
        _seq(out, ["user_id", "author_id"], "is_click_rolling_user_id_author_id"),
        [None, 1.0, None, 1.0],
    )


def test_user_author_source_author_key_is_integer():
    """The int64 key from attach_author_id survives the groupby/rolling round trip."""
    out = build_user_author_source(
        _base(
            [
                {"user_id": 1, "video_id": 1, "author_id": 10, "dt": "2022-04-08"},
                {"user_id": 1, "video_id": 2, "author_id": 10, "dt": "2022-04-09"},
            ]
        )
    )
    assert pd.api.types.is_integer_dtype(out["author_id"])
    assert out["author_id"].tolist() == [10, 10]


# --- video source ------------------------------------------------------------


def test_video_source_has_join_key_and_expected_columns():
    out = build_video_source(
        _base(
            [
                {"user_id": 1, "video_id": 9, "author_id": 10, "dt": "2022-04-08"},
                {"user_id": 2, "video_id": 9, "author_id": 10, "dt": "2022-04-09"},
            ]
        )
    )
    assert list(out.columns) == ["dt", "video_id"] + VIDEO_FEATURES


def test_video_source_produces_cumulative_column():
    """Regression: passing "is_click" (str) instead of ["is_click"] to
    _set_cumulative_columns iterates characters -> KeyError and no cumulative
    column. The cumulative count excludes the current row (first event = 0)."""
    out = build_video_source(
        _base(
            [
                {"user_id": 1, "video_id": 9, "author_id": 10, "dt": "2022-04-08", "is_click": 1},
                {"user_id": 2, "video_id": 9, "author_id": 10, "dt": "2022-04-09", "is_click": 1},
                {"user_id": 3, "video_id": 9, "author_id": 10, "dt": "2022-04-10", "is_click": 0},
            ]
        )
    )
    assert "is_click_cumulative_video_id" in out.columns
    # prior-positive counts, current row excluded: [0, 1, 2]
    assert _seq(out, ["video_id"], "is_click_cumulative_video_id") == [0, 1, 2]


def test_video_source_rolling_is_point_in_time():
    out = build_video_source(
        _base(
            [
                {"user_id": 1, "video_id": 9, "author_id": 10, "dt": "2022-04-08", "is_click": 1},
                {"user_id": 2, "video_id": 9, "author_id": 10, "dt": "2022-04-09", "is_click": 0},
            ]
        )
    )
    _assert_seq(_seq(out, ["video_id"], "is_click_rolling_video_id"), [None, 1.0])


# --- shared base frame -------------------------------------------------------


def test_builders_do_not_mutate_base_frame():
    """Each builder sorts into its own copy, so one base_frame feeds all three
    without accumulating feature columns across calls."""
    base = _base(
        [
            {"user_id": 1, "video_id": 9, "author_id": 10, "dt": "2022-04-08", "is_click": 1},
            {"user_id": 1, "video_id": 9, "author_id": 10, "dt": "2022-04-09", "is_click": 0},
        ]
    )
    before = list(base.columns)
    build_user_source(base)
    build_user_author_source(base)
    build_video_source(base)
    assert list(base.columns) == before
