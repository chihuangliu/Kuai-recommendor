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

import numpy as np
import pandas as pd
import pytest
import torch

from kuai_recommender.data.data_pure import (
    KuaiPureData,
    KuaiPureDataset,
    collate_with_masks,
)

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
    for col in KuaiPureData.BINARY_COLUMNS_ORIGINAL:
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
    for col in KuaiPureData.BINARY_COLUMNS_ORIGINAL:
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
    for col in KuaiPureData.BINARY_COLUMNS_ORIGINAL:
        assert f"{col}_cumulative_video_id" in obj.df.columns


# --- _set_rolling_columns with NaN group keys (dropna=False fix) --------------
# A video absent from the basic feature file left-joins to a NaN author_id.
# groupby(dropna=False) must keep those rows so the .values assign stays aligned.


def test_rolling_survives_nan_group_key():
    """NaN author_id rows are kept (own group), not dropped -> no length mismatch."""
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
                "is_click": 0,
            },
            {
                "rid": 3,
                "user_id": 1,
                "author_id": float("nan"),
                "dt": "2022-04-08",
                "is_click": 1,
            },
            {
                "rid": 4,
                "user_id": 1,
                "author_id": float("nan"),
                "dt": "2022-04-09",
                "is_click": 1,
            },
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


_ALL_LABELS = (
    KuaiPureData.BINARY_COLUMNS_PREPROCESSED
    + KuaiPureData.CONTINUOUS_COLUMNS_PREPROCESSED
)


def _make_dataset(rows: list[dict], features: list[str]) -> KuaiPureDataset:
    """Wrap a hand-built df in a KuaiPureDataset, bypassing KuaiPureData.__init__."""
    df = pd.DataFrame(rows)
    # Every label the Dataset emits (originals + is_skip + dwell_log) must exist
    # as a column, or __getitem__ KeyErrors reading the target dict.
    for col in _ALL_LABELS:
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
    assert set(y.keys()) == set(_ALL_LABELS)
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


def test_nan_targets_pass_through_untouched():
    """A NaN skip/dwell label (duration<=0) must reach y as NaN, NOT be zeroed.

    Masking happens in the loss (via isnan), so the Dataset must preserve the NaN;
    zeroing it here would silently teach the model 'not a skip' / dwell 0.
    """
    ds = _make_dataset(
        [{"rate": 0.5, "is_skip": float("nan"), "dwell_log": float("nan")}],
        features=["rate"],
    )
    _, y = ds[0]
    assert torch.isnan(y["is_skip"]), "is_skip NaN was zeroed instead of passed through"
    assert torch.isnan(y["dwell_log"]), "dwell_log NaN was zeroed instead of passed through"


# --- KuaiPureDataset negative sampling ---------------------------------------
# _neg_sampling downsamples only "pure negative" impressions (no engagement
# signal at all), to shrink the training set without starving the sparse
# positive heads. The mask is over BINARY_COLUMNS_ORIGINAL only: is_skip (a
# negative signal that fires on ~70% of rows) and dwell_log (continuous) must
# NOT count as positives, or the boring rows we mean to drop get protected.


def _dataset_with_sampling(
    rows: list[dict], features: list[str], neg_keep_frac: float
) -> KuaiPureDataset:
    df = pd.DataFrame(rows)
    for col in _ALL_LABELS:
        if col not in df.columns:
            df[col] = 0
    data = KuaiPureData.__new__(KuaiPureData)
    data.df = df
    return KuaiPureDataset(data, features, neg_keep_frac=neg_keep_frac)


def _pure_neg_row(rate: float = 0.5) -> dict:
    """No engagement, but skipped with positive dwell -- the exact shape the old
    mask wrongly protected (is_skip=1 and dwell_log>0 made sum(labels) > 0)."""
    return {"rate": rate, "is_skip": 1.0, "dwell_log": 2.0}


def test_neg_sampling_keeps_every_positive_and_downsamples_negatives():
    rows = [_pure_neg_row() for _ in range(10)] + [
        {"rate": 0.5, "is_click": 1} for _ in range(4)
    ]
    ds = _dataset_with_sampling(rows, ["rate"], neg_keep_frac=0.5)
    # 4 positives kept + int(10 * 0.5) = 5 negatives kept.
    assert len(ds) == 9
    n_click = sum(ds[i][1]["is_click"].item() == 1.0 for i in range(len(ds)))
    assert n_click == 4  # no positive was ever eligible for dropping


def test_skipped_no_engagement_rows_are_downsampled():
    """Regression for the is_skip/dwell_log bug: a skipped, positive-dwell row
    with no engagement is a pure negative and must be droppable."""
    rows = [_pure_neg_row() for _ in range(10)] + [
        {"rate": 0.5, "is_click": 1} for _ in range(3)
    ]
    ds = _dataset_with_sampling(rows, ["rate"], neg_keep_frac=0.0)
    # All pure negatives gone despite is_skip=1 / dwell_log>0; positives remain.
    assert len(ds) == 3


def test_any_engagement_signal_protects_a_row():
    """Every column in BINARY_COLUMNS_ORIGINAL counts -- not just is_click.
    A lone long_view or a lone (sparse) is_hate positive must survive."""
    rows = [_pure_neg_row() for _ in range(6)] + [
        {"rate": 0.5, "long_view": 1},
        {"rate": 0.5, "is_hate": 1},
    ]
    ds = _dataset_with_sampling(rows, ["rate"], neg_keep_frac=0.0)
    assert len(ds) == 2


def test_neg_keep_frac_one_is_a_noop():
    rows = [_pure_neg_row() for _ in range(5)] + [
        {"rate": 0.5, "is_click": 1} for _ in range(2)
    ]
    ds = _dataset_with_sampling(rows, ["rate"], neg_keep_frac=1.0)
    assert len(ds) == 7


def test_neg_sampling_is_reproducible_under_a_reseeded_rng(monkeypatch):
    """Sampling draws from the shared module rng, so re-seeding it to the same
    state must reproduce the same kept rows (the sampler is a pure function of
    the frame and the rng state)."""
    import kuai_recommender.data.data_pure as dp

    rows = [_pure_neg_row(rate=i / 20) for i in range(10)] + [
        {"rate": 1.0, "is_click": 1} for _ in range(3)
    ]

    def build_once() -> list[float]:
        monkeypatch.setattr(dp, "rng", np.random.default_rng(42))
        ds = _dataset_with_sampling(rows, ["rate"], neg_keep_frac=0.5)
        return sorted(ds[i][0].item() for i in range(len(ds)))

    assert build_once() == build_once()


# --- _set_engagement_targets -------------------------------------------------
# Skip / dwell targets derived row-wise from play_time_ms vs duration_ms.
#
#   is_skip   = (completion < 0.5) AND (play_time_ms < 5000)  -- both must be low
#   completion = clip(play/duration, upper=1)                 -- loops don't exceed 1
#   dwell_log  = log1p(min(play, 2*duration))                 -- cap loop replays
#
# When duration_ms <= 0 the denominator is unknown, so BOTH targets are NaN
# (we abstain rather than fabricate a label from raw watch time).


def _make_targets(rows: list[dict]) -> KuaiPureData:
    """Build a KuaiPureData carrying only duration_ms/play_time_ms, then derive targets."""
    df = pd.DataFrame(rows)
    obj = KuaiPureData.__new__(KuaiPureData)
    obj.df = df
    obj._set_engagement_targets()
    return obj


def _col(obj: KuaiPureData, column: str) -> dict:
    return obj.df.set_index("rid")[column].to_dict()


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def test_skip_needs_both_low_completion_and_low_time():
    """AND semantics: skip only when the user watched a small *fraction* AND little absolute time."""
    obj = _make_targets(
        [
            # low completion (0.1) + low time (1s)  -> skip
            {"rid": 1, "duration_ms": 10_000, "play_time_ms": 1_000},
            # low completion (0.2) but 40s watched of a long video -> NOT a skip (the rescue)
            {"rid": 2, "duration_ms": 200_000, "play_time_ms": 40_000},
            # short clip watched in full (completion 1.0), only 3s -> NOT a skip
            {"rid": 3, "duration_ms": 3_000, "play_time_ms": 3_000},
            # high completion + high time -> NOT a skip
            {"rid": 4, "duration_ms": 20_000, "play_time_ms": 18_000},
        ]
    )
    assert _col(obj, "is_skip") == {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}


def test_skip_thresholds_are_strict():
    """completion == 0.5 and play == 5000 sit on the 'not skip' side (strict `<`)."""
    obj = _make_targets(
        [
            # completion exactly 0.5 -> not < 0.5 -> not skip
            {"rid": 1, "duration_ms": 10_000, "play_time_ms": 5_000},
            # completion 0.4 (< 0.5) but play exactly 5000 -> not < 5000 -> not skip
            {"rid": 2, "duration_ms": 12_500, "play_time_ms": 5_000},
            # completion 0.4 and play 4999 -> both strictly low -> skip
            {"rid": 3, "duration_ms": 12_500, "play_time_ms": 4_999},
        ]
    )
    assert _col(obj, "is_skip") == {1: 0.0, 2: 0.0, 3: 1.0}


def test_completion_clipped_so_loops_are_not_skips():
    """A replayed short clip (play > duration) has completion clipped to 1.0 -> never a skip."""
    obj = _make_targets([{"rid": 1, "duration_ms": 2_000, "play_time_ms": 10_000}])
    assert _col(obj, "is_skip")[1] == 0.0


def test_dwell_log_is_log1p_of_watch_time():
    """Normal watch (play <= 2*duration): dwell_log = log1p(play_time_ms)."""
    obj = _make_targets(
        [
            {"rid": 1, "duration_ms": 10_000, "play_time_ms": 1_000},
            {"rid": 2, "duration_ms": 200_000, "play_time_ms": 40_000},
            {
                "rid": 3,
                "duration_ms": 10_000,
                "play_time_ms": 0,
            },  # instant skip -> log1p(0)=0
        ]
    )
    dwell = _col(obj, "dwell_log")
    assert dwell[1] == pytest.approx(math.log1p(1_000), rel=1e-5)
    assert dwell[2] == pytest.approx(math.log1p(40_000), rel=1e-5)
    assert dwell[3] == pytest.approx(0.0, abs=1e-6)


def test_dwell_caps_loop_replays_at_two_durations():
    """play_time far above the video length is capped at 2*duration before log1p."""
    obj = _make_targets([{"rid": 1, "duration_ms": 2_000, "play_time_ms": 10_000}])
    # min(10_000, 2*2_000) = 4_000
    assert _col(obj, "dwell_log")[1] == pytest.approx(math.log1p(4_000), rel=1e-5)


def test_invalid_duration_abstains_on_both_targets():
    """duration_ms <= 0 -> denominator unknown -> is_skip and dwell_log are NaN, not fabricated."""
    obj = _make_targets(
        [
            {"rid": 1, "duration_ms": 0, "play_time_ms": 1_000},
            {"rid": 2, "duration_ms": -5, "play_time_ms": 0},
            {"rid": 3, "duration_ms": 10_000, "play_time_ms": 1_000},  # valid control
        ]
    )
    skip = _col(obj, "is_skip")
    dwell = _col(obj, "dwell_log")
    assert _isnan(skip[1]) and _isnan(dwell[1])
    assert _isnan(skip[2]) and _isnan(dwell[2])
    # the valid row is unaffected and still labelled
    assert skip[3] == 1.0 and not _isnan(dwell[3])


# --- collate_with_masks -------------------------------------------------------
# Turns a list of per-sample (x, {label: scalar}) into batched tensors, laid out
# for a vectorised multi-task loss: binary labels stacked into [B, K] (column
# order == BINARY_COLUMNS_PREPROCESSED), continuous into [B, C], plus a validity
# mask per target group. The mask is the whole point: is_skip/dwell_log are NaN
# when duration<=0, and a NaN target must be *masked*, not silently zeroed.

_BIN_COLS = KuaiPureData.BINARY_COLUMNS_PREPROCESSED
_CONT_COLS = KuaiPureData.CONTINUOUS_COLUMNS_PREPROCESSED


def _sample(
    feature_vec: list[float], **labels: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build one (x, y) exactly as KuaiPureDataset.__getitem__ would: x is a
    feature vector, y maps every label to a scalar tensor (unset labels -> 0)."""
    x = torch.tensor(feature_vec, dtype=torch.float32)
    y = {
        c: torch.tensor(float(labels.get(c, 0.0)), dtype=torch.float32)
        for c in _ALL_LABELS
    }
    return x, y


def test_collate_shapes_and_dtypes():
    """x -> [B, F]; binary -> [B, K]; continuous -> [B, C]; masks match and are bool."""
    batch = [_sample([0.1, 0.2]), _sample([0.3, 0.4]), _sample([0.5, 0.6])]
    x, y_bin, y_cont, m_bin, m_cont = collate_with_masks(batch)

    assert x.shape == (3, 2) and x.dtype == torch.float32
    assert y_bin.shape == (3, len(_BIN_COLS))
    assert y_cont.shape == (3, len(_CONT_COLS))
    assert m_bin.shape == y_bin.shape and m_bin.dtype == torch.bool
    assert m_cont.shape == y_cont.shape and m_cont.dtype == torch.bool


def test_collate_binary_column_order_matches_preprocessed():
    """Column j of y_binary is BINARY_COLUMNS_PREPROCESSED[j] -- not insertion order."""
    batch = [_sample([0.0], is_click=1.0, is_hate=1.0)]
    _, y_bin, _, _, _ = collate_with_masks(batch)

    for j, col in enumerate(_BIN_COLS):
        expected = 1.0 if col in ("is_click", "is_hate") else 0.0
        assert y_bin[0, j].item() == expected, f"{col} landed in the wrong column"


def test_collate_preserves_row_order():
    """Row i of every batched tensor is sample i -- collate must not reorder."""
    batch = [
        _sample([1.0], is_click=1.0),
        _sample([2.0], is_like=1.0),
        _sample([3.0], long_view=1.0),
    ]
    x, y_bin, _, _, _ = collate_with_masks(batch)

    assert x[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert y_bin[0, _BIN_COLS.index("is_click")].item() == 1.0
    assert y_bin[1, _BIN_COLS.index("is_like")].item() == 1.0
    assert y_bin[2, _BIN_COLS.index("long_view")].item() == 1.0


def test_collate_masks_nan_targets_and_keeps_valid_ones():
    """NaN is_skip/dwell_log -> mask False at those cells, True everywhere else."""
    batch = [
        _sample([0.5], is_click=1.0),  # fully valid
        _sample([0.5], is_skip=float("nan"), dwell_log=float("nan")),  # duration<=0
    ]
    _, y_bin, y_cont, m_bin, m_cont = collate_with_masks(batch)

    skip_j = _BIN_COLS.index("is_skip")
    # Row 0 is entirely valid; row 1's is_skip is masked out, its other binaries stay valid.
    assert m_bin[0].all()
    assert not m_bin[1, skip_j]
    assert m_bin[1, [j for j in range(len(_BIN_COLS)) if j != skip_j]].all()
    # dwell_log: valid for row 0, masked for row 1.
    assert m_cont[0].all() and not m_cont[1].any()


def test_collate_does_not_zero_the_masked_target():
    """The masked cell must still carry NaN in y (the mask flags it; loss skips it).
    Zeroing here would teach the model a fake 'not a skip' / dwell 0."""
    batch = [_sample([0.5], is_skip=float("nan"), dwell_log=float("nan"))]
    _, y_bin, y_cont, _, _ = collate_with_masks(batch)

    assert torch.isnan(y_bin[0, _BIN_COLS.index("is_skip")])
    assert torch.isnan(y_cont[0, 0])
