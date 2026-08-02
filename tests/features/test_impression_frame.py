"""Contract for features.entity_df.build_impression_frame -- the per-impression
base frame that (a) feeds Feast retrieval via its [keys, event_timestamp] subset
and (b) becomes the training table once labels/buckets are attached downstream.

Because that frame is consumed by name in two places, a stray or missing column
silently breaks either retrieval or label building. This test pins the exact
column contract, plus the invariants of the two *derived* columns:

  * author_id is a plain non-null ``int64`` -- ``attach_author_id`` rejects an
    unmatched video rather than propagating <NA>, so the (user_id, author_id)
    retrieval key never drops on a dtype mismatch;
  * event_timestamp is tz-aware Asia/Shanghai, derived row-for-row from time_ms;

and that the video_features lookup is a join, not a fan-out (one row per
impression). Uses TEST_STANDARD (the smaller standard split) -- the code path is
split-independent, so the cheaper split exercises the same contract.
"""

import pandas as pd
import pytest

from kuai_recommender.data.utils import (
    DATA_DIR,
    VIDEO_FEATURES_BASIC_PATH,
    KuaiPureDatasetSplits,
)
from kuai_recommender.features.entity_df import build_impression_frame
from kuai_recommender.features.schema import BINARY_FEATURES, ENGAGEMENT_INPUT_COLUMNS

_SPLIT = KuaiPureDatasetSplits.TEST_STANDARD

pytestmark = pytest.mark.skipif(
    not (DATA_DIR.exists() and VIDEO_FEATURES_BASIC_PATH.exists()),
    reason="build_impression_frame contract needs the KuaiRand CSVs",
)

EXPECTED_COLUMNS = {
    "user_id",
    "video_id",
    "time_ms",
    *ENGAGEMENT_INPUT_COLUMNS,
    *BINARY_FEATURES,
    "author_id",  # merged from video_features_basic
    "event_timestamp",  # derived from time_ms
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_impression_frame(_SPLIT)


def test_columns_are_exactly_the_impression_contract(frame):
    """No missing column (breaks retrieval/label build) and no stray column."""
    assert set(frame.columns) == EXPECTED_COLUMNS


def test_author_id_is_non_null_int64(frame):
    """Total int64: an unmatched video raises upstream, so no <NA> reaches here."""
    assert frame["author_id"].dtype == "int64"
    assert frame["author_id"].notna().all()


def test_event_timestamp_is_tz_aware_shanghai_from_time_ms(frame):
    """tz-aware Asia/Shanghai, derived row-for-row from time_ms."""
    ts = frame["event_timestamp"]
    assert isinstance(ts.dtype, pd.DatetimeTZDtype)
    assert str(ts.dtype.tz) == "Asia/Shanghai"
    expected = pd.to_datetime(frame["time_ms"], unit="ms", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    pd.testing.assert_series_equal(ts, expected, check_names=False)


def test_one_row_per_impression(frame):
    """The video_features_basic merge is a lookup, not a fan-out."""
    raw_len = len(pd.read_csv(DATA_DIR / _SPLIT.value, usecols=["time_ms"]))
    assert len(frame) == raw_len
