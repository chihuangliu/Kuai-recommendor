"""Frozen contract for the model's feature vocabulary.

WHY THIS FILE EXISTS
--------------------
The column contracts -- which continuous features the ranker consumes and in what
ORDER, which raw signals get rolled into the store -- now live in a single
features/registry.py (they were split across KuaiPureData and features/schema.py,
both since deleted). Every other
test in the suite references these lists *symbolically* (``len(MODEL_CONTINUOUS)``,
``[x for f in CONTINUOUS_FEATURES]``, ``{*USER_FEATURES}``), so a refactor that
silently REORDERS or DROPS a feature stays green everywhere -- while permuting the
trained model's input vector and desynchronising the online-fetch order from the
x-assembly order. This file is the one place that pins the literal values, so that
drift is caught.

REFACTOR HAND-OFF
-----------------
During the registry extraction, change ONLY the imports in the "CONTRACT SOURCE"
block below to point at features.registry (MODEL_CONTINUOUS, MODEL_CATEGORICAL,
BINARY_TARGETS, ...). NEVER edit an EXPECTED_* literal to make a red test green:
a diff there means the input/label vector really changed, and the checkpoint must
be retrained on purpose. Green before the refactor AND green after == the registry
reproduced today's contract exactly.

HIGH-DIMENSIONAL TEXT FEATURES (Stage 3)
----------------------------------------
A caption / BERT embedding is a *static, video-keyed, N-dimensional* feature: it
is NOT a window aggregate of a binary signal and is NOT folded by the streaming
replay (it is materialised once). The current contract has no such feature, and
this file encodes the invariants that keep adding one honest:

  * ``test_store_vocab_is_raw_signals_rolled_per_entity`` pins the *windowed* store
    vocabulary by its derivation formula (signals x entities), so it needs no edit
    when text features arrive -- text columns live in a SEPARATE static vocab.
  * ``test_model_continuous_is_subset_of_store_vocab`` is the invariant Feast/
    serving relies on: every model input must be a stored column. Text columns must
    join the store schema before they can join MODEL_CONTINUOUS.
  * ``test_streaming_signal_surface_is_exactly_the_raw_signals`` pins that the only
    things the replay folds per event are the raw binary signals -- so a static
    text feature cannot silently get pulled into the windowed fold.

When text lands: extend EXPECTED_MODEL_CONTINUOUS with the embedding columns
(kept contiguous at the tail so the checkpoint slice stays predictable), and split
the store-vocab test into windowed (formula-pinned) + static (dim-pinned) halves.
See the STATIC-FEATURE marker at the bottom of this file.
"""

# ============================ CONTRACT SOURCE ================================ #
# The contract now lives entirely in features.registry (features/schema.py, which
# used to re-export it under different names, has been deleted). Only this block
# may be repointed; the EXPECTED_* literals below are the source of truth and
# must not move.
from kuai_recommender.features.registry import (
    UA,
    USER,
    VIDEO,
    BINARY_SIGNALS,
    BINARY_TARGETS,
    CONTINUOUS_TARGETS,
    MODEL_CATEGORICAL,
    MODEL_CONTINUOUS,
    Kind,
    get_cols,
)

# The per-view vocabularies, split by Kind -- the same slices the consumers take.
USER_FEATURES = [c.name for c in get_cols(USER, Kind.ROLLING)]
USER_AUTHOR_FEATURES = [c.name for c in get_cols(UA, Kind.ROLLING)]
VIDEO_FEATURES_FLOAT = [c.name for c in get_cols(VIDEO, Kind.ROLLING)]
VIDEO_FEATURES_INT = [c.name for c in get_cols(VIDEO, Kind.CUMULATIVE)]
# unfiltered: every kind stored on the video view
VIDEO_FEATURES = [c.name for c in get_cols(VIDEO)]

# The 8 raw per-impression signals -- the root the whole store vocabulary is
# derived from (rolled per entity) and the source of the binary target heads.
EXPECTED_RAW_SIGNALS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
]

# THE CHECKPOINT CONTRACT: the exact order of the ranker's continuous input vector.
# A curated 11-of-25 subset of the store vocabulary. NOTE the order does NOT follow
# EXPECTED_RAW_SIGNALS order (e.g. long_view precedes is_like here) -- which is
# exactly why it must be pinned as an explicit list and can never be derived by
# filtering the stored set.
EXPECTED_MODEL_CONTINUOUS = [
    "is_click_rolling_user_id",
    "long_view_rolling_user_id",
    "is_like_rolling_user_id",
    "is_profile_enter_rolling_user_id",
    "is_click_rolling_user_id_author_id",
    "long_view_rolling_user_id_author_id",
    "is_like_rolling_user_id_author_id",
    "is_click_rolling_video_id",
    "long_view_rolling_video_id",
    "is_like_rolling_video_id",
    "is_click_cumulative_video_id",
]

EXPECTED_MODEL_CATEGORICAL = ["user_id_bucket", "author_id_bucket"]
EXPECTED_BINARY_TARGETS = EXPECTED_RAW_SIGNALS + ["is_skip"]
EXPECTED_CONTINUOUS_TARGETS = ["dwell_log"]


# --- raw signals -------------------------------------------------------------


def test_raw_signals_frozen():
    """The 8 signals, in order. Guards the de-dup: after the refactor the single
    registry list must equal this (BINARY_FEATURES and the old
    BINARY_COLUMNS_ORIGINAL were two copies of it)."""
    assert list(BINARY_SIGNALS) == EXPECTED_RAW_SIGNALS


# --- model input vector (the checkpoint contract) ----------------------------


def test_model_continuous_contract_frozen():
    """Exact names AND order of the continuous input vector. A red here == the
    trained model's inputs were permuted/changed."""
    assert list(MODEL_CONTINUOUS) == EXPECTED_MODEL_CONTINUOUS


def test_model_categorical_contract_frozen():
    assert list(MODEL_CATEGORICAL) == EXPECTED_MODEL_CATEGORICAL


# --- target heads ------------------------------------------------------------


def test_binary_targets_are_raw_signals_plus_skip():
    """The binary heads == the 8 raw signals + is_skip, in that order (is_skip is
    the only derived binary target; order == output-head order)."""
    assert list(BINARY_TARGETS) == EXPECTED_BINARY_TARGETS


def test_continuous_targets_frozen():
    assert list(CONTINUOUS_TARGETS) == EXPECTED_CONTINUOUS_TARGETS


# --- store vocabulary <-> model-input coupling -------------------------------


def test_store_vocab_is_raw_signals_rolled_per_entity():
    """The store's WINDOWED vocabulary is the raw signals rolled per entity, plus
    the one cumulative video count. Pinned by formula so it stays compact and needs
    NO edit when static text features are added (those form a separate vocab)."""
    assert list(USER_FEATURES) == [f"{s}_rolling_user_id" for s in EXPECTED_RAW_SIGNALS]
    assert list(USER_AUTHOR_FEATURES) == [
        f"{s}_rolling_user_id_author_id" for s in EXPECTED_RAW_SIGNALS
    ]
    assert list(VIDEO_FEATURES_FLOAT) == [
        f"{s}_rolling_video_id" for s in EXPECTED_RAW_SIGNALS
    ]
    assert list(VIDEO_FEATURES_INT) == ["is_click_cumulative_video_id"]
    assert list(VIDEO_FEATURES) == list(VIDEO_FEATURES_FLOAT) + list(VIDEO_FEATURES_INT)


def test_model_continuous_is_subset_of_store_vocab():
    """Every continuous model input must be a stored column -- the invariant serving
    relies on (online fetch is keyed by these names). A text feature must enter the
    store schema BEFORE it can enter MODEL_CONTINUOUS."""
    store_vocab = {*USER_FEATURES, *USER_AUTHOR_FEATURES, *VIDEO_FEATURES}
    assert set(MODEL_CONTINUOUS) <= store_vocab


def test_streaming_signal_surface_is_exactly_the_raw_signals():
    """The replay folds ONLY the raw binary signals per event. Pinning this means a
    future STATIC text feature (materialised once, never folded) cannot silently be
    pulled into the windowed aggregation."""
    assert list(BINARY_SIGNALS) == EXPECTED_RAW_SIGNALS


# --- STATIC-FEATURE marker (Stage 3) -----------------------------------------
# When the caption embedding lands, add here:
#   EXPECTED_STATIC_VIDEO_FEATURES = [f"caption_emb_{i}" for i in range(N)]
#   def test_static_features_are_video_kind_static(): assert their registry kind is
#       STATIC and their view is the video FeatureView (so streaming skips them);
#   def test_static_features_dim_matches_embedding_config(): pin N == config dim;
# and append the same columns to EXPECTED_MODEL_CONTINUOUS above (tail-contiguous).
