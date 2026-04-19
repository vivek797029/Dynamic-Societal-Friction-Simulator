# Dynamic Societal Friction Simulator (DSFS) — TRUST-CFE

Trust-weighted multi-signal estimator of societal friction across Indian states, 2015–2025.

## What this is

A research system that:
- Ingests GDELT 2.0 events + GKG + a multilingual news-article sample for India.
- Learns a **per-source trust weight τ_s endogenously** (no misinformation labels) via cross-source factual-agreement contrastive learning on MuRIL.
- Produces a **cleavage-decomposed friction field** F(state, week, cleavage) using trust-weighted discourse + Goldstein-weighted event intensity.
- Forecasts short-horizon (1/2/4-week) protest & fatality counts from ACLED as an independent validation target.
- Runs end-to-end on Google Colab (A100 or L4 for Stage A; CPU for Stage B).

## Project layout

```
dynamic-societal-friction-simulator/
├── config.yaml                 # all hyperparameters & paths
├── requirements.txt
├── notebooks/
│   └── dsfs_colab.ipynb        # one-click Colab runner
├── paper/
│   └── paper_outline.md        # IEEE-format section-by-section plan
├── src/
│   ├── data/                   # Tier 1: ingest, geolocate, cluster events, extract facts
│   ├── models/                 # Tier 2: trust learner (core), cleavage, hostility
│   ├── training/               # Stage A & B training loops
│   ├── evaluation/             # metrics, baselines, ablation
│   └── visualization/          # choropleth, timeline, trust bar chart
└── tests/
```

## Quick start (Colab)

1. Open `notebooks/dsfs_colab.ipynb` in Colab.
2. Mount Google Drive (for cache).
3. Run cells top-to-bottom. First run downloads ~4 GB of GDELT + a 20k-article sample and takes ~30 min on a standard runtime.
4. Stage A (trust + hostility) trains in ~4–6 h on A100. Stage B (aggregator + forecast) finishes in minutes.

## Pipeline (9 modules)

```
GDELT + Articles + ACLED
        │
        ▼
M1 Ingest ─► M2 Geolocate ─► M3 Event Clustering
                                   │
         ┌─────────────────────────┤
         ▼                         ▼
M4 Factual Signals       M6 Cleavage + Hostility (LoRA)
         │                         │
         ▼                         │
╔═══════════════════════╗          │
║ M5 TRUST LEARNING     ║          │
║ (contrastive + cons.) ║          │
╚══════════╦════════════╝          │
           └──────────┬────────────┘
                      ▼
           M7 Friction Aggregator
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
M8 Escalation Head      M9 Eval + Viz
 supervised by ACLED
```

## Core formula

```
E_k(r,t) = Σ |Goldstein(e)| · 1[Goldstein(e) < 0]           (events in state r, week t, cleavage k)
T_k(r,t) = Σᵢ τ_{s(i)} · h(xᵢ, k)  /  Σᵢ τ_{s(i)}            (trust-weighted hostility)
F_k(r,t) = α_k Ẽ_k(r,t) + β_k T̃_k(r,t) + λ Σ_{τ=1..W} exp(-τ/θ_k) · F_k(r,t-τ)
F(r,t)   = Σ_k ω_k F_k(r,t)
```

All of (α, β, ω, λ, θ) are learned against held-out ACLED outcomes — F is identified by predictive validity, not hand-tuned.

## Citation / license

Research code, MIT licensed. Not affiliated with GDELT, ACLED, or SATP.
