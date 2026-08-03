# Kuai-Recommendor

A learning project that re-implements the **Personalized News Feed** ML system design
(ByteByteGo ML System Design, Ch.10) end-to-end on real data, using the **KuaiRand**
short-video feed dataset (Kuaishou).

Two goals drive every scope decision:

1. **Understand the concepts in the notes and learn how to implement them.**
2. **Practice the everyday modules of `huggingface`, `pytorch`, and `sklearn`.**

Where the book and the dataset disagree, this project follows the dataset and records
the gap — see [Data notes](#data-notes-what-the-dataset-can-and-cant-do).

---

## Dataset

**KuaiRand-Pure** (`data/KuaiRand-Pure/`) — smallest version, laptop-friendly.

| File | Rows | What it is |
|---|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | 1.14M | Production-recommender impressions (biased), Apr 8–21 |
| `log_standard_4_22_to_5_08_pure.csv` | 295K | Production-recommender impressions (biased), Apr 22–May 8 |
| `log_random_4_22_to_5_08_pure.csv` | 1.19M | **Randomly exposed** impressions (unbiased), Apr 22–May 8 |
| `user_features_pure.csv` | 27K | User demographics/activity + 18 anonymized one-hot feats |
| `video_features_basic_pure.csv` | 7.6K | `author_id`, upload time, duration, tag, music |
| `video_features_statistic_pure.csv` | 7.6K | ~50 aggregate engagement counts per video |
| `kuairand_video_captions.csv` | 23M* | Caption text (Chinese) + cover OCR — *global file, filter to Pure's 7.6K |

Per-impression labels: `is_click, is_like, is_follow, is_comment, is_forward, is_hate,
long_view, is_profile_enter` + `play_time_ms` / `duration_ms` (dwell & skip).

### Train / validation / online-prediction split

The biased (standard) and unbiased (random) logs overlap in time for Apr 22–May 8 — this
parallel collection is by design and drives the split.

```
          4/08 ────── 4/19 │4/20–21│ 4/22 ─────────── 5/08
standard   [=== train ===]  │ [val] │ [== test:biased ==]
 random                             │ [== test:random ==]
```

| Role | Data | Why |
|---|---|---|
| **Training** | standard 4/08–4/19 (~1.10M, biased) | Fit model params on the earliest window; features computed point-in-time |
| **Validation** | standard 4/20–4/21 (~41K, biased) | Hyperparam / early-stop / loss-weight tuning. Train-tail, **time-disjoint from the eval window** → its selection optimism can't leak into either eval arm |
| **Test — biased arm** | standard 4/22–5/08 (295K, biased) | Biased arm of the bias gap; held out from selection. Same distribution as training → honest measure of the learned task |
| **Test — unbiased arm** | random 4/22–5/08 (1.19M, unbiased) | Headline + unbiased arm. Ranking quality of a *new* policy — NOT a production-traffic sim. Never touched during training/tuning |

Design rules:

- **Split by time, not randomly** — point-in-time features + evolving user behavior mean a
  random split leaks the future into the past. Splits run strictly train < selection < eval.
- **Validation is carved from the train tail** (4/20–4/21, ~3.6% — traffic is front-loaded so
  the tail is cheap), time-disjoint from the eval window so hyperparam/early-stop optimism
  can't couple into the bias gap or headline. Early-stop on the dense heads (is_click /
  long_view, ~13–18k positives); the sparse heads (hate/follow/forward) are too rare to steer
  selection at any split size — judge those on the large random arm instead.
- **The two eval arms share the same period** (Apr 22–May 8), differing only in exposure
  policy. So `test(biased)_AUC − test(random)_AUC` isolates selection/position bias with time held constant
  (Stage 5). The gap is large by construction: logged CTR is **46%** (biased) vs **17.6%**
  (random) — the production recommender only shows likely clicks, inflating its CTR.
- **All splits use point-in-time features** from full interaction history up to each
  impression's timestamp, regardless of which policy generated it — matching what an online
  store would serve, and keeping offline/online feature definitions consistent.
- **Negative sampling on training only.** The eval arms keep the natural class balance, or the
  metrics are meaningless.
- **Cold start is real:** the random log has 27,285 users vs training's 26,210, so the eval
  arms contain unseen users/videos → embedding cold-start. A realistic condition, not a bug.

---

## Scope

The project mirrors the book's ML-system pipeline. Each block below maps to a part of
Figure 10.14 and to specific modules being practiced.

### 1. Data preparation pipeline — *point-in-time features*
> Book: "Data preparation pipeline". The real concept here is **preventing feature
> leakage**, not infra. This is the highest-value concept block.

- [v] Build `<user, video>` training rows from impression logs (one row per impression)
- [v] **Point-in-time feature computation** with `pandas.merge_asof` — every feature at
      time T uses only data with `time_ms < T`
      - [v] User rolling reaction rates (7-day click/like/... rates)
      - [v] **User–author affinity** (historical like/click/comment rate per author)
      - [v] Video engagement, computed **from the logs** as-of T (NOT from the statistic
            file — see below): a cumulative popularity rate (`likes/shows` up to T) plus a
            short rolling rate (last N hours) for freshness/velocity
      - [x]~~Post-age bucketing from `upload_dt`; one-hot encode~~ — **dropped**:
            in KuaiRand-Pure all videos were uploaded within a 3-day window
            (`upload_dt` has only 3 values: Apr 9/10/11), so post-age ≈ impression
            date minus a constant → near-perfectly collinear with the impression day
            and no `<1d` fresh samples. The freshness signal is degenerate here; use
            the log-derived rolling video-engagement rate above instead.
- [v] Skip / dwell-time targets from `play_time_ms` vs `duration_ms`
- [v] Negative sampling to balance per-task positives (book Fig 10.11)
- [v] *(optional)* Wrap the time-varying features in a **Feast** feature store so **one
      definition serves both paths** — offline (point-in-time) for training, online (latest
      value) for serving. See [Feature store & serving](#feature-store--serving-feast) below.
- [v] *(optional)* Event-replay script to simulate **streaming** incremental updates to the
      online store (event-time, late data) — no real Kafka broker. See same section.

**Practices:** pandas `merge_asof`, groupby-rolling · sklearn `StandardScaler`,
`TfidfVectorizer`/`HashingVectorizer`, `train_test_split` (time-based)

### 2. Ranking model — *multi-task DNN*  ← core of the project
> Book: "Ranking service" + Fig 10.8/10.9/10.10. Shared backbone, N task heads.

- [v] Custom `Dataset` returning a multi-label target dict + `collate_fn`
- [v] Shared trunk + `nn.ModuleList` of heads (binary heads + dwell-time regression head)
- [v] `nn.Embedding` for `user_id` / `author_id` / `tag`
- [v] Combined loss: `BCEWithLogitsLoss(pos_weight=...)` per binary head + `HuberLoss`
      for dwell-time, weighted sum
- [ ] **Ablation A:** multi-task DNN vs N independent DNNs (esp. on sparse heads)
- [ ] **Ablation B:** add dwell-time + skip heads → measure effect on *passive* users
- [ ] Blend head probabilities into an engagement score (book Table 10.1) + weight
      sensitivity analysis
- [ ] Baseline first: sklearn / LightGBM single-task before the DNN

**Practices:** pytorch `Dataset`/`DataLoader`, `collate_fn`, `WeightedRandomSampler`,
`nn.Module`/`nn.Embedding`/`nn.ModuleList`, `BCEWithLogitsLoss`/`HuberLoss`, AdamW +
`CosineAnnealingLR`, checkpointing (best + partial load) · sklearn `roc_auc_score`

### 3. Text features — *caption via BERT*  (optional / high-value for HF practice)
> Book: BERT for textual content, TF-IDF/word2vec for hashtags.

- [ ] Filter global captions file down to Pure's 7.6K videos
- [ ] Tokenize captions with a **Chinese** BERT (`bert-base-chinese`); note: book's
      Viterbi hashtag-splitting doesn't apply to Chinese → use jieba for hashtags
- [ ] Caption embedding as a feature: frozen backbone → then fine-tune
- [ ] **Ablation C:** full-caption BERT vs lightweight TF-IDF on hashtags (test the
      book's claim that hashtags don't need a Transformer)
- [ ] CLS token vs mean-pooling comparison

**Practices:** HF `AutoTokenizer` (padding/truncation, attention_mask), `AutoModel`
(backbone, no head), pooling strategies, freeze/unfreeze, `Trainer` vs plain-pytorch,
custom loss via Trainer subclass

### 4. Retrieval — *two-tower + ANN*  (optional)
> Book: "Retrieval service". Book uses social-graph fan-out; KuaiRand has no friendship
> graph, so we do this as a **modelling** problem instead.

- [ ] User tower / item tower encoders, in-batch negatives
- [ ] `F.normalize` + cosine similarity, build a `faiss` index, retrieve top-K
- [ ] Feed top-K candidates into the Stage-2 ranker → real retrieval→ranking two-stage

**Practices:** pytorch two-tower, contrastive loss, `F.normalize`, faiss ANN

### 5. Evaluation — *offline + de-biased*
> Book: "Offline metrics" (ROC-AUC per reaction) + Other Talking Points (position bias).

- [ ] Per-task ROC-AUC on the biased log
- [ ] **Re-evaluate on the `log_random` slice** → quantify the biased/unbiased AUC gap
      (turns the book's hand-wave on position bias into a measured number)
- [ ] Passive-user segmented metrics (total dwell as engagement proxy)

**Practices:** sklearn `roc_auc_score`, calibration (`CalibratedClassifierCV`) before
score blending

---

## Feature store & serving (Feast)

Optional, high-value block. Goal: take the time-varying features already computed inline in
Stage 1 and put them behind a **feature store**, so the *same definition* serves training and
serving. This is where the project practices **train/serve consistency** and, later,
**streaming** — and it doubles as a rehearsal for *adding a new feature to a live store*
(text features, Stage 3).

### Two read paths — the core concept

| | **Offline store** | **Online store** |
|---|---|---|
| Serves | training / batch eval | live inference (serving) |
| Question | "feature value **as-of** time T?" (a different T per row) | "each entity's **latest** value **now**?" |
| Read shape | millions of rows, point-in-time join | few keys, millisecond KV lookup |
| Backend | `file` (parquet) | **Redis** |
| Feast API | `get_historical_features` | `get_online_features` |

Without a store, teams compute features one way for training and another for serving → the
formulas drift → **train/serve skew**. Feast's value is a single `FeatureView` feeding both.

### Scope — what goes in the store

Only the **11 time-varying rolling/cumulative features** (`config.FEATURES`), as three
FeatureViews by entity: `user_id`, `(user_id, author_id)`, `video_id`. The `*_id_bucket`
categoricals are a **pure hash of the id** (no state, time-invariant) → hashed on the fly at
serving, **not** stored. So the online store's job is exactly the 11 features.

### Materialization strategy

- **Offline source is event-driven**: one feature row per impression `(entity, event_ts,
  values)`, computed over the standard train window (4/08–4/21) in one pass — reproduces the
  `merge_asof`/rolling output column-for-column and replaces the per-split `history=` recompute
  with an `entity_df` point-in-time join.
- **The offline store is never queried past 4/21.** Val is a slice of the training parquet
  (4/20–4/21); the eval arms (4/22–5/08) are served from the online store via the streaming
  replay. `build_sources.py` builds the train source, `stream_replay.py` folds a test arm's raw
  events.
- **Leakage handling, two steps:**
  - **Selm A (done)** — store `f_pre` (window `[T−7D, T)`, current row excluded) with join
    `feature_ts <= entity_ts`, asserted equal to the old pipeline column-for-column. The parity
    gate.
  - **Selm B (later)** — store `f_post` (window includes the event) with a strict `<` as-of
    join, moving "exclude current row" into the join for a fresher online value. Needs a stable
    tiebreak for equal-timestamp events.
- **Online population:** batch `materialize_incremental` (staler) or the streaming replay
  (event-driven, fresher). Both recompute the same 7-day window, so the served value reproduces
  the offline point-in-time value — **train/serve consistency by construction**, pinned by an
  oracle test (`test_streaming.py`: `WindowAvg` == `_set_rolling_columns` value-for-value).
  Residual gap: the **freshness-expiry lag** — a pull-based online value only refreshes when a
  key sees a new event, a measurable Stage-5 number.

### Streaming replay

Serve the 7-day rolling window the model trained on, computed online incrementally from raw
events — one definition, both paths, **skew-free by construction**. (An EWMA / approximate
online value was rejected as the serving path: it would train on the rolling window but serve a
different distribution → train/serve skew. Kept only as a possible Stage-5 skew demo.)

- **What it replays**: the standard test arm (4/22–5/08) in event-time order, warmed up by
  folding the whole train window (4/08–4/21) to rebuild each key's `(count, Σsignal)` state
  (state can't be reconstructed from materialized rates). Late data is injected to exercise
  event-time vs processing-time: Feast's Redis store keeps the value with the max `event_ts`, so
  out-of-order writes don't clobber a fresher value.
- **How the online feature is computed**: an incremental sliding-window aggregate per key —
  running `(count, Σsignal)` with a deque (later: tiles) for eviction. Read-before-update yields
  `closed="left"` (current row excluded, leak-free), matching `compute.py`; equal-timestamp
  events must be read *before* any of them append. Redis holds only the served values; Feast is
  push-fed via `write_to_online_store` (Feast does not compute windows itself).
- **Why not replay the precomputed parquet**: that equals `materialize` (same numbers, different
  transport). Streaming computes the aggregate from raw events, so it can be fresher than a
  periodic batch *and* reproduce the point-in-time value exactly.
- The two eval arms overlap in time, so each replays from its own freshly warmed state — one
  shared aggregator would leak the standard arm's events into the random arm.

**Three roles — one script now, separate services in production:**

| Role | Job | Production | Here |
|---|---|---|---|
| Stream compute | fold events → window aggregate | Flink / Spark Structured Streaming / Kafka Streams | `WindowAgg` |
| State backend | keyed state (deque / tiles) | RocksDB (+ checkpoints) | `KeyedStateStore`: dict → RocksDB |
| Online store | serve computed values | Redis / DynamoDB, Feast-push-fed | Redis |

- **Repository pattern**: the aggregation depends only on a `KeyedStateStore` (`get`/`put`/
  `items`), so the backend swaps without touching the fold. An external backend serializes state
  per event, so a raw deque is O(window) I/O per event → the portable form is **tile
  partial-sums** (fixed small state per key). Tiles also bound memory and make the window
  queryable at any time, removing the freshness-expiry lag — so they are the prerequisite for an
  external state backend, not just an optimization.
- No Kafka broker — the point is event-time, state, and freshness, not infra. Feature platforms
  that do windowed aggregation natively, for reference: Chronon, Tecton, Materialize / RisingWave.

**Practices:** Feast `Entity`/`FeatureView`/`FileSource`, `get_historical_features` vs
`get_online_features`, `materialize`, Redis online store, point-in-time correctness · streaming
state: incremental sliding-window aggregation, tile partial-sums, `KeyedStateStore` (Repository
pattern), RocksDB-backed keyed state

---

## Suggested order (each stopping point leaves something runnable)

1. Stage 1 batch point-in-time features (highest concept value)
2. Stage 2 multi-task ranker (main goal)
3. Stage 5 evaluation incl. the `log_random` de-biased metric
4. Optional: Feast · BERT captions · two-tower retrieval · streaming replay

---

## Data notes — what the dataset can and can't do

Verified against the real files (2026-07-17):

- **All required signals present.** Positive rates on the standard log: `is_click` 46%,
  `long_view` 34%, `is_like` 1.9%, `is_profile_enter` 2.5%, `is_comment` 0.26%,
  `is_follow`/`is_forward` 0.10%, `is_hate` 0.04%. Skip proxy (`play < 0.5·duration`) ~70%.
- **`follow` / `forward` / `comment` / `hate` are very sparse** (hate ≈ 480 positives in
  1.1M). Too sparse to train standalone — which is exactly what makes the *multi-task vs
  N-independent-DNN* ablation meaningful.
- **Unbiased eval slice exists** (`log_random`) — rare and valuable; drives Stage 5.
- **Timestamps present** → point-in-time joins are real, not simulated.
- **`video_features_statistic_pure.csv` is a leakage trap.** It is one static snapshot per
  video (aggregate totals over the whole logging period, no timestamp), so it cannot be
  sliced to time T. Do **not** use it as a training feature — joining its totals onto a row
  at T leaks future engagement. Use it for EDA only; compute video engagement features from
  the interaction logs with `time_ms < T` instead.
- **No friendship table.** Book's close-friend/family affinity feature can't be built;
  user–author affinity (rates + follow) replaces it.
- **Captions are Chinese**, include hashtags (`#...`) and mentions (`@...`). Requires a
  Chinese BERT; book's English hashtag tokenization (Viterbi) does not apply.
- **No raw images/frames** — only `show_cover_text` (cover OCR). `torchvision` / CLIP
  is out of scope.

## Environment

Use `uv` for all Python (`uv run --with pandas,torch,transformers,scikit-learn ...`).

The Feast block additionally needs `feast` and a running **Redis** for the online store
(the offline store is `file`/parquet, no service). Serving/streaming steps assume Redis is up.
