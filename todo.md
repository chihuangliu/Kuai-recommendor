# TODO

## Code review follow-ups (deferred, low severity)

- [ ] **`KuaiPureDataset` uses mutable class lists as default args**
      (`kuai_recommender/data/data_pure.py:140-141`).
      `continuous_features` / `categorical_features` default to the shared
      `KuaiPureData.FEATURE_COLUMNS` / `CATEGORICAL_COLUMNS_PREPROCESSED` lists.
      Any in-place mutation of `self.features` / `self.cat_features` would leak
      across all instances and mutate the class constants (classic mutable-default
      footgun). Fix: default to `None` and assign
      `self.features = continuous_features or KuaiPureData.FEATURE_COLUMNS`.

- [ ] **`_hash_to_bucket` divides by zero for `n_buckets <= 1`**
      (`kuai_recommender/data/data_pure.py:121-122`).
      `n_valid_buckets = n_buckets - 1; murmurhash3_32(...) % n_valid_buckets`
      raises `ZeroDivisionError` when `n_buckets == 1`. Real callers always pass
      `next_pow2(4 * nunique) >= 4`, but the staticmethod is unit-tested directly
      with arbitrary sizes. Fix: add `assert n_buckets >= 2` to document/guard the
      contract.
