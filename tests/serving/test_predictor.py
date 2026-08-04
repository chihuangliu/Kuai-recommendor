"""Contract for serving/predictor.predict -- the online read path.

predict() is where the registry contract turns into actual tensors, and it is the
one place train/serve skew can enter silently: the online store returns a *dict
keyed by feature name*, so the x-vector is assembled by name lookup, and x_cat is
assembled by stacking per-id bucket tensors. Nothing about either assembly is
checked by the type system -- a reordering produces a perfectly-shaped tensor that
feeds the model garbage.

What this file pins:
  * x column order == MODEL_CONTINUOUS (the checkpoint's input contract),
  * x_cat column order == BUCKET_COLS == MODEL_CATEGORICAL, and column i is
    bucketed with column i's OWN bucket size -- the invariant that keeps x_cat
    aligned with model_dims' embedding_dims (index i must be < that table's rows,
    or ids get looked up in the wrong embedding),
  * the online fetch asks for exactly the features the model consumes (no over-
    fetch, and nothing missing -- a missing ref KeyErrors during assembly),
  * missing online values (None) become 0.0, never NaN, never a crash,
  * from_online=False bypasses the store entirely and passes features through.

Everything is faked: no Redis, no Feast registry, no CSV reads. `store` and
`get_bucket_size` are monkeypatched in the predictor namespace, so this runs
anywhere.

NOTE ON NAMING: these tests read ``BucketCol.source`` (the raw id column -- the
hash input and the get_bucket_size key), as defined in features/registry.py.
``BucketCol.name`` is the *output* column (f"{source}_bucket") used in the
training parquet. Consumers that need a bucket SIZE or an id to hash want
``.source``; only the parquet column lookup wants ``.name``.
"""

import numpy as np
import pytest
import torch

from kuai_recommender.features.compute import hash_to_bucket
from kuai_recommender.features.registry import (
    BUCKET_COLS,
    MODEL_CATEGORICAL,
    MODEL_CONTINUOUS,
    get_online_refs,
)

# Distinct sizes per id, so a swapped x_cat column produces different bucket
# values and the ordering assertions actually bite.
BUCKET_SIZES = {"user_id": 16, "author_id": 64}

USER_IDS = [11, 22, 33]
AUTHOR_IDS = [101, 202, 303]
VIDEO_IDS = [7001, 7002, 7003]


def _online_value(row: int, col_index: int) -> float:
    """A value unique per (row, feature) so misordering is detectable."""
    return 100.0 * row + col_index + 0.5


class _FakeOnlineResponse:
    def __init__(self, payload: dict[str, list]):
        self._payload = payload

    def to_dict(self) -> dict[str, list]:
        return self._payload


class _FakeStore:
    """Records the fetch instead of touching Redis; serves per-name values."""

    def __init__(self, payload: dict[str, list] | None = None):
        self.payload = payload
        self.features: list[str] | None = None
        self.entity_rows: list[dict] | None = None
        self.call_count = 0

    def get_online_features(self, features, entity_rows):
        self.call_count += 1
        self.features = list(features)
        self.entity_rows = list(entity_rows)
        if self.payload is not None:
            return _FakeOnlineResponse(self.payload)
        payload = {
            name: [_online_value(r, j) for r in range(len(entity_rows))]
            for j, name in enumerate(MODEL_CONTINUOUS)
        }
        return _FakeOnlineResponse(payload)


class _RecordingModel:
    """Captures the tensors predict() hands the model."""

    def __init__(self):
        self.x: torch.Tensor | None = None
        self.x_cat: torch.Tensor | None = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def forward(self, x, x_cat):
        self.x, self.x_cat = x, x_cat
        return {"binary": torch.zeros(len(x), 1), "continuous": torch.zeros(len(x), 1)}


@pytest.fixture
def predictor(monkeypatch):
    """The predictor module with store + bucket sizes neutralised."""
    import kuai_recommender.serving.predictor as mod

    monkeypatch.setattr(mod, "get_bucket_size", lambda: BUCKET_SIZES)
    return mod


def _run(predictor, monkeypatch, *, store=None, **kwargs):
    store = store if store is not None else _FakeStore()
    monkeypatch.setattr(predictor, "store", store)
    model = _RecordingModel()
    out = predictor.predict(model, USER_IDS, AUTHOR_IDS, VIDEO_IDS, **kwargs)
    return model, store, out


# --- x: the continuous vector ------------------------------------------------


def test_x_columns_follow_model_continuous_order(predictor, monkeypatch):
    """Column j of x is MODEL_CONTINUOUS[j] for every row. The online store returns
    an unordered dict, so this ordering is entirely predict()'s responsibility --
    and it is the checkpoint's input contract."""
    model, _, _ = _run(predictor, monkeypatch)

    assert model.x.shape == (len(USER_IDS), len(MODEL_CONTINUOUS))
    assert model.x.dtype == torch.float32
    for r in range(len(USER_IDS)):
        for j in range(len(MODEL_CONTINUOUS)):
            assert model.x[r, j].item() == pytest.approx(_online_value(r, j)), (
                f"x[{r},{j}] should carry {MODEL_CONTINUOUS[j]}"
            )


def test_missing_online_values_become_zero_not_nan(predictor, monkeypatch):
    """A cold entity yields None from the store. That must impute to 0.0 (a 0%
    prior rate, matching the Dataset's nan_to_num), never reach the model as NaN."""
    payload = {
        name: [None for _ in USER_IDS] for name in MODEL_CONTINUOUS
    }
    model, _, _ = _run(predictor, monkeypatch, store=_FakeStore(payload))

    assert not torch.isnan(model.x).any()
    assert (model.x == 0.0).all()


# --- the online fetch --------------------------------------------------------


def test_fetch_asks_for_exactly_the_consumed_features(predictor, monkeypatch):
    """The fetch must cover every MODEL_CONTINUOUS name (a missing one KeyErrors
    during assembly) and nothing more (over-fetching costs a Redis round trip per
    unused feature). Refs are 'view_name:column'."""
    _, store, _ = _run(predictor, monkeypatch)

    assert store.features == get_online_refs()
    fetched_cols = [ref.split(":", 1)[1] for ref in store.features]
    assert fetched_cols == MODEL_CONTINUOUS
    assert all(":" in ref for ref in store.features)


def test_entity_rows_carry_one_row_per_request_with_join_keys(predictor, monkeypatch):
    _, store, _ = _run(predictor, monkeypatch)

    assert store.entity_rows == [
        {"user_id": u, "author_id": a, "video_id": v}
        for u, a, v in zip(USER_IDS, AUTHOR_IDS, VIDEO_IDS)
    ]


# --- x_cat: the categorical buckets ------------------------------------------


def test_x_cat_columns_follow_bucket_cols_order(predictor, monkeypatch):
    """Column i of x_cat is BUCKET_COLS[i], hashed with ITS OWN bucket size.

    This is the alignment that model_dims' embedding_dims depends on: embedding
    table i is built from BUCKET_COLS[i]'s size, so if x_cat's columns were
    swapped, author ids would index the user embedding table."""
    model, _, _ = _run(predictor, monkeypatch)
    ids_by_source = {"user_id": USER_IDS, "author_id": AUTHOR_IDS}

    assert model.x_cat.shape == (len(USER_IDS), len(BUCKET_COLS))
    assert model.x_cat.dtype == torch.long
    for i, col in enumerate(BUCKET_COLS):
        expected = [
            hash_to_bucket(v, BUCKET_SIZES[col.source]) for v in ids_by_source[col.source]
        ]
        assert model.x_cat[:, i].tolist() == expected, (
            f"x_cat column {i} should be {col.name}"
        )


def test_x_cat_columns_stay_inside_their_embedding_table(predictor, monkeypatch):
    """Every bucket index must be < that column's table size, else the embedding
    lookup is out of range. Catches a column bucketed with the wrong id's size."""
    model, _, _ = _run(predictor, monkeypatch)

    for i, col in enumerate(BUCKET_COLS):
        size = BUCKET_SIZES[col.source]
        column = model.x_cat[:, i]
        assert column.min().item() >= 0
        assert column.max().item() < size, (
            f"x_cat column {i} ({col.name}) exceeds its {size}-row embedding table"
        )


def test_x_cat_width_matches_the_categorical_contract(predictor, monkeypatch):
    model, _, _ = _run(predictor, monkeypatch)
    assert model.x_cat.shape[1] == len(MODEL_CATEGORICAL)


# --- offline / precomputed-feature path --------------------------------------


def test_from_online_false_passes_features_through_and_skips_the_store(
    predictor, monkeypatch
):
    """The replay's ServedFeaturesSource supplies x directly; predict must use it
    verbatim (same column order) and never call the online store."""
    features = np.arange(
        len(USER_IDS) * len(MODEL_CONTINUOUS), dtype=np.float32
    ).reshape(len(USER_IDS), len(MODEL_CONTINUOUS))
    model, store, _ = _run(
        predictor, monkeypatch, features=features, from_online=False
    )

    assert store.call_count == 0
    assert model.x.dtype == torch.float32
    assert model.x.tolist() == features.tolist()
    # x_cat is still hashed on the fly -- buckets are never read from the store.
    assert model.x_cat.shape == (len(USER_IDS), len(BUCKET_COLS))


def test_from_online_false_imputes_nan_features(predictor, monkeypatch):
    """A NaN in the supplied feature block (first impression, empty window) must
    be imputed to 0.0 exactly as the online path does."""
    features = np.full((len(USER_IDS), len(MODEL_CONTINUOUS)), np.nan, dtype=np.float32)
    model, _, _ = _run(predictor, monkeypatch, features=features, from_online=False)

    assert not torch.isnan(model.x).any()
    assert (model.x == 0.0).all()


def test_from_online_false_requires_features(predictor, monkeypatch):
    monkeypatch.setattr(predictor, "store", _FakeStore())
    with pytest.raises(AssertionError):
        predictor.predict(
            _RecordingModel(), USER_IDS, AUTHOR_IDS, VIDEO_IDS, from_online=False
        )


# --- inference hygiene -------------------------------------------------------


def test_model_is_put_in_eval_mode(predictor, monkeypatch):
    """Serving must not run dropout/batchnorm in training mode."""
    model, _, _ = _run(predictor, monkeypatch)
    assert model.eval_called
