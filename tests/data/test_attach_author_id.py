"""Contract for data.utils.attach_author_id -- the single video -> author join.

Every pipeline (offline sources, impression frame, streaming replay, the torch
dataset) gets author_id from this one function, so its postcondition is what lets
all of them treat author_id as a plain non-null int64: no isna checks, no
float/int round trips. The properties pinned here are exactly that postcondition
plus the two ways the join can go wrong:

  * on a full match, author_id comes back as int64 (a left join with any miss
    would silently upcast the column to float64),
  * an unmatched video_id raises -- video_features_basic being out of sync with
    the log is a data bug, not a row to drop behind the caller's back,
  * a duplicated video_id in the lookup raises instead of fanning out the frame
    (validate="many_to_one"), which would otherwise inflate every downstream
    rolling aggregate.
"""

import pandas as pd
import pytest

from kuai_recommender.data import utils
from kuai_recommender.data.utils import attach_author_id


@pytest.fixture
def lookup(monkeypatch):
    """Swap the cached video->author table for a hand-built one."""

    def _install(rows: list[dict]) -> None:
        monkeypatch.setattr(utils, "_author_lookup", lambda: pd.DataFrame(rows))

    return _install


def _impressions(video_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"user_id": range(len(video_ids)), "video_id": video_ids})


def test_full_match_yields_non_null_int64(lookup):
    """The postcondition every caller relies on: a total int64 key."""
    lookup([{"video_id": 1, "author_id": 10}, {"video_id": 2, "author_id": 20}])
    out = attach_author_id(_impressions([1, 2, 1]))
    assert out["author_id"].dtype == "int64"
    assert out["author_id"].tolist() == [10, 20, 10]


def test_unmatched_video_raises(lookup):
    """A video with no author is a sync bug -- loud, not a silent NaN or drop."""
    lookup([{"video_id": 1, "author_id": 10}])
    with pytest.raises(ValueError, match="out of sync"):
        attach_author_id(_impressions([1, 2]))


def test_duplicated_lookup_video_raises_instead_of_fanning_out(lookup):
    """A duplicate video_id would double rows and inflate every rolling feature."""
    lookup([{"video_id": 1, "author_id": 10}, {"video_id": 1, "author_id": 11}])
    with pytest.raises(pd.errors.MergeError):
        attach_author_id(_impressions([1]))


def test_row_count_and_order_are_preserved(lookup):
    """A lookup join: same rows, same order, one extra column."""
    lookup([{"video_id": 1, "author_id": 10}, {"video_id": 2, "author_id": 20}])
    frame = _impressions([2, 1, 2, 1])
    out = attach_author_id(frame)
    assert len(out) == len(frame)
    assert out["video_id"].tolist() == frame["video_id"].tolist()
    assert set(out.columns) == set(frame.columns) | {"author_id"}
