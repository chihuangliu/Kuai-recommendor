"""Phase-1 parity gate (README "Selm A"): Feast ``get_historical_features`` must
reproduce the inline ``KuaiPureData`` point-in-time features column-for-column on
the standard log (train + val). See the ``feast-offline-store`` memory for scope.

Why the assertion is split in two:

  * **rolling (24 feats)** -- ``closed="left"`` excludes every row at exactly the
    entity timestamp T, so same-``(video_id, time_ms)`` repeated impressions share
    identical rolling values. Alignment collapsed to one row per key is therefore
    unambiguous, and we require exact equality (within 1e-9).

  * **cumulative (``is_click_cumulative_video_id``)** -- ``cumsum() - self`` is
    per-impression order-dependent, but Feast keys the video view on
    ``(video_id, ts)`` alone and cannot disambiguate same-millisecond repeats.
    Those rows (~16.5k train / ~6.9k val) differ *by design*, so cumulative parity
    is asserted only on rows whose ``(video_id, time_ms)`` is unique in the split.

Scope: **train + val only.** The source is built from the standard log; ``test``
(random split) is served from that same source and intentionally diverges from the
old train+random-history code, so it is not a parity target.

Cost: the file offline store runs a full point-in-time join per granularity, so
these are slow integration tests, not micro-units. To keep them tractable the gate
runs on a deterministic sample of impressions (``PARITY_SAMPLE`` rows, default
5000); slicing the *entity* set does not affect point-in-time correctness because
each row is still joined against the full source. Set ``PARITY_SAMPLE=0`` to run the
full-split gate.
"""

import os

import numpy as np
import pandas as pd
import pytest

from kuai_recommender.data.data_pure import KuaiPureData
from kuai_recommender.data.utils import DATA_DIR, build_splits
from kuai_recommender.features.common import FEATURE_SOURCES_DIR
from kuai_recommender.features.entity_df import build_impression_frame
from kuai_recommender.features.feature_repo.store import FEAST_DIR, store
from kuai_recommender.features.schema import (
    USER_AUTHOR_FEATURES,
    USER_FEATURES,
    VIDEO_FEATURES_FLOAT,
    VIDEO_FEATURES_INT,
)

# --- what we compare --------------------------------------------------------
KEY = ["user_id", "video_id", "time_ms"]  # video_id -> author_id, so it is redundant
ROLLING_FEATURES = (
    list(USER_FEATURES) + list(USER_AUTHOR_FEATURES) + list(VIDEO_FEATURES_FLOAT)
)
CUMULATIVE_FEATURE = VIDEO_FEATURES_INT[0]  # is_click_cumulative_video_id

PARITY_SAMPLE = int(os.environ.get("PARITY_SAMPLE", "5000"))  # 0 -> full split
SAMPLE_SEED = 43
TOL = 1e-9

# --- guard: needs the real dataset + built sources + an applied registry ----
_SOURCES_READY = all(
    (FEATURE_SOURCES_DIR / f"{n}.parquet").exists()
    for n in ("user", "video", "user_author")
)
_REGISTRY_READY = (FEAST_DIR / "registry.db").exists()
_DATA_READY = DATA_DIR.exists()

pytestmark = pytest.mark.skipif(
    not (_SOURCES_READY and _REGISTRY_READY and _DATA_READY),
    reason=(
        "parity gate needs the KuaiRand CSVs, built feature_sources parquets, and "
        "an applied Feast registry -- run build_sources.py then "
        "`python -m kuai_recommender.features.feature_repo.store`"
    ),
)


def _retrieve_feast(impressions: pd.DataFrame) -> pd.DataFrame:
    """Three-way per-granularity point-in-time retrieval, joined back onto the
    impression spine -- structurally identical to ``features/history.py``.

    A single multi-granularity ``get_historical_features`` call collapses the
    output to the coarsest join key and silently drops rows; driving one call per
    granularity off a de-duplicated entity_df and left-joining back preserves
    one row per impression. (Documented in the feast-offline-store memory.)
    """
    user_feats = store.get_historical_features(
        entity_df=impressions[["user_id", "event_timestamp"]].drop_duplicates(),
        features=[f"user_features:{f}" for f in USER_FEATURES],
    ).to_df()
    video_feats = store.get_historical_features(
        entity_df=impressions[["video_id", "event_timestamp"]].drop_duplicates(),
        features=[
            f"video_features:{f}" for f in VIDEO_FEATURES_FLOAT + VIDEO_FEATURES_INT
        ],
    ).to_df()
    ua_feats = store.get_historical_features(
        entity_df=impressions[
            ["user_id", "author_id", "event_timestamp"]
        ].drop_duplicates(),
        features=[f"user_author_features:{f}" for f in USER_AUTHOR_FEATURES],
    ).to_df()
    return (
        impressions.merge(user_feats, on=["user_id", "event_timestamp"], how="left")
        .merge(video_feats, on=["video_id", "event_timestamp"], how="left")
        .merge(ua_feats, on=["user_id", "author_id", "event_timestamp"], how="left")
    )


def _tied_pairs(csv_name: str) -> pd.DataFrame:
    """Unique ``(video_id, time_ms)`` pairs that occur more than once in the split
    -- the same-millisecond repeats Feast cannot disambiguate for cumulative."""
    full = pd.read_csv(DATA_DIR / csv_name, usecols=["video_id", "time_ms"])
    dup = full[full.duplicated(["video_id", "time_ms"], keep=False)]
    return dup[["video_id", "time_ms"]].drop_duplicates()


@pytest.fixture(scope="module", params=["train", "val"])
def parity(request):
    """Build the aligned (feast vs reference) frame once per split.

    Reference = the inline pipeline with the split's own history (val needs train);
    Feast = the sampled three-way retrieval. Both are de-duplicated to one row per
    KEY before an inner merge (rolling values are identical across same-key ties,
    so the collapse is lossless for rolling; cumulative ties are excluded later).
    """
    cfg = build_splits()[request.param]
    name, history = cfg["name"], cfg["history"]

    impressions = build_impression_frame(name)
    if PARITY_SAMPLE and PARITY_SAMPLE < len(impressions):
        impressions = impressions.sample(n=PARITY_SAMPLE, random_state=SAMPLE_SEED)

    feast = _retrieve_feast(impressions)
    ref = KuaiPureData(name, history=history).df

    cols = KEY + ROLLING_FEATURES + [CUMULATIVE_FEATURE]
    feast = feast[cols].drop_duplicates(KEY)
    ref = ref[cols].drop_duplicates(KEY)

    merged = feast.merge(ref, on=KEY, suffixes=("_feast", "_ref"), validate="1:1")

    tied = _tied_pairs(name).assign(_tied=True)
    merged = merged.merge(tied, on=["video_id", "time_ms"], how="left")
    merged["_tied"] = merged["_tied"].notna()  # matched -> tied, else unique

    assert len(merged) > 0, "no impressions aligned between Feast and reference"
    return {"split": request.param, "merged": merged}


def _mismatch(a: pd.Series, b: pd.Series, tol: float) -> pd.Series:
    """NaN-aware inequality: rows where the two disagree (both-NaN counts as equal)."""
    a = a.astype("float64").to_numpy()
    b = b.astype("float64").to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=0.0, atol=tol, equal_nan=False)
    return pd.Series(~(both_nan | close))


def test_rolling_parity(parity):
    """Every rolling feature matches the inline pipeline exactly (within TOL)."""
    merged = parity["merged"]
    failures = {}
    for col in ROLLING_FEATURES:
        bad = _mismatch(merged[f"{col}_feast"], merged[f"{col}_ref"], TOL)
        if bad.any():
            failures[col] = int(bad.sum())

    # guard against a silent all-NaN "match": at least the dense user/video
    # granularities must have real values aligning, not just NaN==NaN.
    assert merged["is_click_rolling_user_id_feast"].notna().any()
    assert merged["is_click_rolling_video_id_feast"].notna().any()

    assert not failures, (
        f"[{parity['split']}] rolling parity broke on {failures} "
        f"(out of {len(merged)} aligned rows)"
    )


def test_cumulative_parity(parity):
    """is_click_cumulative_video_id matches on rows with a unique (video_id, time_ms);
    same-millisecond repeats are excluded because Feast keys on (video_id, ts) only."""
    merged = parity["merged"]
    unique = merged[~merged["_tied"]]
    assert len(unique) > 0, "no unique-key rows to check cumulative parity on"

    bad = _mismatch(
        unique[f"{CUMULATIVE_FEATURE}_feast"],
        unique[f"{CUMULATIVE_FEATURE}_ref"],
        TOL,
    )
    n_bad = int(bad.sum())
    assert n_bad == 0, (
        f"[{parity['split']}] cumulative parity broke on {n_bad}/{len(unique)} "
        f"unique-key rows (tied rows already excluded); "
        f"sample:\n{unique[bad][KEY + [f'{CUMULATIVE_FEATURE}_feast', f'{CUMULATIVE_FEATURE}_ref']].head()}"
    )
