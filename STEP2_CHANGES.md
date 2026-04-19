# Step 2 — Model Improvement (drop-in diffs)

Addresses the items prioritized in the Step 1 audit. All paths are relative
to the repo root.

## Files added

| File | Purpose | Audit item |
|---|---|---|
| `src/training/samplers.py` | `ClusterBatchSampler` (SupCon-style, m×k per batch) + `positives_per_batch` helper | §1.2 HIGH |
| `src/data/source_norm.py`  | `normalize_source()` + alias table — collapse `NDTV.com / ndtv.com / www.ndtv.com / NDTV` | §1.1 MEDIUM |
| `tests/test_target_alignment.py` | Regression for Stage-B target ordering | §1.1 CRITICAL |
| `tests/test_contrastive_has_positives.py` | Sampler produces ≥ m·C(k,2) positives/batch | §1.2 HIGH |
| `tests/test_memory_kernel.py` | Shape + strict causality + monotone decay | §1.1 CRITICAL |
| `tests/test_source_normalization.py` | Alias collapse | §1.1 MEDIUM |
| `tests/test_cleavage_weak_labels.py` | No more `bjp ⊂ bjpa`, `obc ⊂ obcuring`, `aap ⊂ aapta` | §1.1 HIGH |
| `tests/test_nb_head.py` | NB NLL is finite; reduces to Poisson as r → ∞ | §1.2 MEDIUM |
| `STEP2_CHANGES.md` | this file | — |

## Files rewritten

### `src/training/train_stage_a.py`
- Drops the `train_stage_a._proj_to_hidden = nn.Linear(...)` monkeypatch.
- Uses the already-existing `DSFSArticleEncoder` — one `state_dict` for the
  backbone + every head. Cleavage/hostility heads consume the **raw CLS**,
  not zero-padded projections (so inference-time outputs match training).
- `DataLoader(batch_sampler=ClusterBatchSampler(...))` — every batch has
  guaranteed positives.
- `ClusterCentroidBank` (EMA μ̂_c with momentum 0.95) now drives the
  consensus-deviation loss.
- Adds `source_diversity_regularizer` to keep τ identifiable.
- Autocast is **bf16 when available**, fp32 otherwise — never fp16 (kills
  the logsumexp-NaN-clipped-to-0 silent failure).
- Cluster-level train/val split (no positive-pair leakage) + early stopping.
- Step-granular checkpointing with top-K retention and `--resume`.
- `log.info(...)` per-epoch summary with positives-per-batch, loss components.

### `src/training/train_stage_b.py`
- **Fixes the critical target/row alignment bug** (§1.1): a new
  `build_targets()` helper stacks labels in **(state outer, t_end inner)**
  order, matching what `build_windows` produces via its C-order `reshape`.
  The old `_targets()` had them swapped — silently training against
  scrambled labels.
- Time-mask is now `np.tile(mask[None, :], (S, 1))` (matches the new
  axis order). It was `np.repeat(..., axis=0)` before.
- Adds `--loss {poisson, nb}` flag; wires up the new
  `NegativeBinomialHead` with per-(horizon, target) learned dispersion.

### `src/models/forecasting_head.py`
- Adds `NegativeBinomialHead` (drop-in NB2 NLL with `softplus`-reparameterized
  dispersion) and standalone `negative_binomial_nll()`.
- `poisson_nll` clamps `log_rate ∈ [-20, 20]` for early-training stability.
- Module docstring updated to note both likelihoods.

### `src/models/cleavage_classifier.py`
- `weak_label()` now uses **word-boundary IGNORECASE regex** precompiled per
  cleavage — `_PATTERNS[c].search(text)` replaces `w.lower() in text.lower()`.
- Devanagari terms still work because `\w` boundaries split on script
  boundaries.

### `requirements.txt`
- Adds `statsmodels>=0.14` (was silently relying on Colab's preinstall).

## Files untouched but confirmed-correct

These were already in good shape when I audited the tree — they match what
Step 1 said they should be:
- `src/models/article_encoder.py` — `DSFSArticleEncoder` already owns
  backbone + all heads with a unified `state_dict`.
- `src/models/trust_learner.py` — `ClusterCentroidBank`,
  `agreement_contrastive_loss` (with hard-negative reweighting),
  `consensus_deviation_loss` (EMA + gate-clamp), and
  `source_diversity_regularizer` are all present.
- `src/models/temporal_kernel.py` — `ExponentialMemory` already has
  the correct `[S, K, T]` conv layout, left-only padding, and the
  `[..., :T]` trim for strict causality.

I wired the training loop to actually **use** these — the previous
`train_stage_a.py` imported `TrustEncoder` (the backward-compat alias) but
never touched the new EMA bank or sampler.

## How to run the tests

```bash
# From the repo root
pytest tests/ -q
```

The sandbox used to build these files had no torch available, so only
torch-free tests were executed here (source_norm, samplers,
cleavage weak_labels). Expect the torch-requiring tests to pass on your
local environment (CPU is fine — no GPU required).

## Severity-ranked punch list from Step 1 — status

| # | Item | Status |
|---|---|---|
| 1 | Fix target/row alignment in `train_stage_b._targets` | **done** (new `build_targets`) |
| 2 | Fix `ExponentialMemory` conv layout + off-by-one | already fixed in repo |
| 3 | Move `proj→hidden` into `TrustEncoder`; drop function-attribute hack | **done** (uses `DSFSArticleEncoder`) |
| 4 | Cluster-aware batch sampler for Stage A | **done** (`ClusterBatchSampler`) |
| 5 | Word-boundary lexicon for cleavage weak labels | **done** |
| 6 | Val split, early stopping, checkpointing in Stage A | **done** |
| 7 | Evaluation must honor train/val/test time split | **pending** — touches `evaluation/metrics.py`, out of Step 2 scope |
| 8 | Switch autocast to bf16 or move loss to fp32 | **done** (bf16 with fp32 loss fallback) |
| 9 | Source-domain canonicalization | **done** (`normalize_source`) |
| 9b | ACLED admin1 canonicalization | **pending** — see `src/data/acled_loader.py` |
| 10 | EMA centroids + min-distinct-sources filter | EMA: done in loop; min-distinct-sources: filter lives in `src/data/event_clusters.py`, pending |
| 11 | Vectorize `build_T_tensor` and `relational_strain_tensor` | **pending** (perf, not correctness) |
| 12 | FAISS-based clustering | **pending** (perf) |
| 13 | Negative Binomial head | **done** |
| 14 | Multi-label event cleavage | **pending** — `attach_cleavage_from_actors` |
| 15 | Monthly parquet partitioning; CI | **pending** |
