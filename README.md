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
          4/08 ─────────── 4/21 │ 4/22 ─────────── 5/08
standard   [===== train =====]   │ [== val ==]
 random                          │ [===== test / online =====]
```

| Role | Data | Why |
|---|---|---|
| **Training** | standard 4/08–4/21 (1.14M, biased) | Fit model params on the earlier window; features computed point-in-time |
| **Validation** | standard 4/22–5/08 (295K, biased) | Model selection / early stopping / loss-weight tuning. Same biased distribution as training → honest measure of the learned task |
| **Test (unbiased)** | random 4/22–5/08 (1.19M, unbiased) | Headline metric. An unbiased offline eval set for estimating ranking quality of a *new* policy — NOT a simulation of production traffic. Never touched during training/tuning |

Design rules:

- **Split by time, not randomly** — point-in-time features + evolving user behavior mean a
  random split leaks the future into the past. Train = earlier, val = later.
- **Val and test share the same period** (Apr 22–May 8), differing only in exposure policy.
  So `val_AUC − test_AUC` isolates selection/position bias with time held constant
  (Stage 5). The gap is large by construction: logged CTR is **46%** (biased) vs **17.6%**
  (random) — the production recommender only shows likely clicks, inflating its CTR.
- **All three use point-in-time features** from full interaction history up to each
  impression's timestamp, regardless of which policy generated it — matching what an online
  store would serve, and keeping offline/online feature definitions consistent.
- **Negative sampling on training only.** Val/test keep the natural class balance, or the
  metrics are meaningless.
- **Cold start is real:** the random log has 27,285 users vs training's 26,210, so val/test
  contain unseen users/videos → embedding cold-start. A realistic condition, not a bug.

---

## Scope

The project mirrors the book's ML-system pipeline. Each block below maps to a part of
Figure 10.14 and to specific modules being practiced.

### 1. Data preparation pipeline — *point-in-time features*
> Book: "Data preparation pipeline". The real concept here is **preventing feature
> leakage**, not infra. This is the highest-value concept block.

- [ ] Build `<user, video>` training rows from impression logs (one row per impression)
- [ ] **Point-in-time feature computation** with `pandas.merge_asof` — every feature at
      time T uses only data with `time_ms < T`
      - [ ] User rolling reaction rates (7-day click/like/... rates)
      - [ ] **User–author affinity** (historical like/click/comment rate per author)
      - [ ] Video cumulative engagement (from statistic file, as-of T)
      - [ ] Post-age bucketing from `upload_dt`; one-hot encode
- [ ] Skip / dwell-time targets from `play_time_ms` vs `duration_ms`
- [ ] Negative sampling to balance per-task positives (book Fig 10.11)
- [ ] *(optional)* Wrap features in a **Feast** feature store — same definitions serve
      training (offline / point-in-time) and serving (online / latest value)
- [ ] *(optional)* Event-replay script to simulate **streaming** incremental updates
      (event-time, late data) — no real Kafka broker

**Practices:** pandas `merge_asof`, groupby-rolling · sklearn `StandardScaler`,
`TfidfVectorizer`/`HashingVectorizer`, `train_test_split` (time-based)

### 2. Ranking model — *multi-task DNN*  ← core of the project
> Book: "Ranking service" + Fig 10.8/10.9/10.10. Shared backbone, N task heads.

- [ ] Custom `Dataset` returning a multi-label target dict + `collate_fn`
- [ ] Shared trunk + `nn.ModuleList` of heads (binary heads + dwell-time regression head)
- [ ] `nn.Embedding` for `user_id` / `author_id` / `tag`
- [ ] Combined loss: `BCEWithLogitsLoss(pos_weight=...)` per binary head + `HuberLoss`
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
- **No friendship table.** Book's close-friend/family affinity feature can't be built;
  user–author affinity (rates + follow) replaces it.
- **Captions are Chinese**, include hashtags (`#...`) and mentions (`@...`). Requires a
  Chinese BERT; book's English hashtag tokenization (Viterbi) does not apply.
- **No raw images/frames** — only `show_cover_text` (cover OCR). `torchvision` / CLIP
  is out of scope.

## Environment

Use `uv` for all Python (`uv run --with pandas,torch,transformers,scikit-learn ...`).
