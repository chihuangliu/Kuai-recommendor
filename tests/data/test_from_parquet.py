"""Contract for KuaiPureDataset.from_parquet -- the constructor that feeds the
Dataset from a Feast-built training parquet instead of an inline KuaiPureData.

Feature *values* are already covered by the parity gate (the parquet is the Feast
retrieval output, asserted == KuaiPureData). What from_parquet adds, and what this
locks, is the wiring:

  * it returns a usable KuaiPureDataset (regression: an early version set the
    attributes on ``obj.df`` and forgot ``return obj`` -> returned None / raised),
  * features / cat_features / labels default to the KuaiPureData column contract,
  * __getitem__ still zeroes feature NaNs but passes label NaNs through untouched
    (the mask, not a fake 0, is what a missing label must become), and
  * negative sampling drops pure-negative rows.

A tiny hand-built parquet is enough -- no real CSVs, no Feast.
"""

import numpy as np
import pandas as pd
import torch

from kuai_recommender.data.data_pure import KuaiPureData, KuaiPureDataset

_FEATURES = KuaiPureData.CONTINUOUS_FEATURES
_CATS = KuaiPureData.CATEGORICAL_FEATURES
_BINARIES = KuaiPureData.BINARY_COLUMNS_ORIGINAL  # neg-sampling + first 8 labels
_LABELS = KuaiPureData.BINARY_TARGETS + KuaiPureData.CONTINUOUS_TARGETS


def _write_parquet(path, *, feature_nan_row0: bool = False) -> pd.DataFrame:
    """A 6-row training frame carrying every column from_parquet consumes.

    Rows 0-1 are pure negatives (all binaries 0); rows 2-5 have a click, so
    negative sampling has both classes to act on.
    """
    n = 6
    df = pd.DataFrame({f: np.linspace(0.1, 0.9, n) for f in _FEATURES})
    for c in _CATS:
        df[c] = np.arange(1, n + 1, dtype="int64")
    for b in _BINARIES:
        df[b] = 0
    df.loc[2:, "is_click"] = 1  # rows 2..5 positive
    df["is_skip"] = np.array([np.nan, 0, 1, 0, 1, 0], dtype="float32")
    df["dwell_log"] = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
    if feature_nan_row0:
        df.loc[0, _FEATURES[0]] = np.nan
    df.to_parquet(path, index=False)
    return df


def test_from_parquet_returns_usable_dataset_with_default_contract(tmp_path):
    """Returns a real Dataset (not None) with the column contract wired on the
    object itself, indexable row-for-row."""
    path = tmp_path / "train.parquet"
    df = _write_parquet(path)

    ds = KuaiPureDataset.from_parquet(path, neg_keep_frac=1.0)

    assert isinstance(ds, KuaiPureDataset)  # regression: used to return None
    assert ds.features == _FEATURES
    assert ds.cat_features == _CATS
    assert ds.labels == _LABELS
    assert len(ds) == len(df)

    x, x_cat, y = ds[0]
    assert x.shape == (len(_FEATURES),) and x.dtype == torch.float32
    assert x_cat.shape == (len(_CATS),) and x_cat.dtype == torch.int64
    assert set(y) == set(_LABELS)


def test_from_parquet_zeroes_feature_nan_but_keeps_label_nan(tmp_path):
    """A NaN feature is zeroed for the model; a NaN label is preserved so collate
    can mask it (never silently zeroed into a real label)."""
    path = tmp_path / "train.parquet"
    _write_parquet(path, feature_nan_row0=True)

    ds = KuaiPureDataset.from_parquet(path, neg_keep_frac=1.0)
    x, _, y = ds[0]

    assert not torch.isnan(x).any(), "feature NaN should be zeroed in x"
    assert torch.isnan(y["is_skip"]), "label NaN must pass through, not be zeroed"


def test_from_parquet_negative_sampling_drops_pure_negatives(tmp_path):
    """neg_keep_frac=0 keeps no pure-negative rows; every survivor has a positive."""
    path = tmp_path / "train.parquet"
    _write_parquet(path)

    kept_all = KuaiPureDataset.from_parquet(path, neg_keep_frac=1.0)
    dropped = KuaiPureDataset.from_parquet(path, neg_keep_frac=0.0)

    assert len(dropped) < len(kept_all)
    survivors = dropped.df[KuaiPureData.BINARY_COLUMNS_ORIGINAL].sum(axis=1)
    assert (survivors > 0).all(), "a pure-negative row survived neg_keep_frac=0"
