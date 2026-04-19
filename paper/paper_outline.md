# TRUST-CFE: Trust-Weighted Contrastive Friction Estimation for Sub-National Societal Tension in India

Target venue: **IEEE Access** (full paper) — alternatives: IEEE TCSS, IEEE BigData.

## Abstract (≤ 250 words)
Motivate the measurement problem (continuous, cleavage-decomposed, sub-national friction), the trust-reliability gap in existing news-based estimators, state TRUST-CFE in one sentence, headline result (FSA, lead-time AUC vs three baselines), and the ablation win from trust weighting.

## I. Introduction
Frame India as a natural laboratory: 36 states/UTs, six salient cleavages, rich multilingual press, ACLED coverage. Argue that *continuous* friction estimation — not binary onset prediction — is the underserved task. Thesis: (i) GDELT-style event pipelines lose semantic content; (ii) NLP pipelines assume source parity; (iii) we identify source reliability endogenously. Contribution bullets:
1. Definition of a cleavage-decomposed friction field F(r, t, k) identified by predictive validity on independent ACLED outcomes.
2. **TRUST-CFE**: an endogenous source-trust learning method via cross-source factual-agreement contrastive learning — no misinformation labels required.
3. A trust-weighted friction aggregator and escalation forecast head; empirical gain over event-count, Goldstein, sentiment-only, and ARIMA baselines at 1/2/4-week horizons.
4. Interpretability: per-state cleavage decomposition; qualitative alignment with known 2020–24 Indian episodes.
5. Fully reproducible Colab implementation; all artifacts released.

## II. Related Work
Four threads — event-based conflict forecasting (ViEWS, ICEWS, UCDP), NLP hate-speech / hostility detection (HS-BRC, IndicBERT evaluations), signed/temporal networks on political actors (TGN, TGL, STAGE), India-specific CSS (riot case studies, SATP analytics). Articulate the specific gap TRUST-CFE fills and cite ~30 papers.

## III. Problem Formulation
Define F_k(r, t), F(r, t). State estimation task and validation task formally. Define ACLED supervision protocol with zero data leakage into the estimator.

## IV. Method (TRUST-CFE)
### A. Architecture overview (module diagram → Fig. 1)
### B. Event clustering
Spatial-temporal + embedding + NER-overlap connected components.
### C. Factual signal extraction
Entities (XLM-R NER), numerics (regex+unit), simple OpenIE triples.
### D. Trust learning (core)
- Shared MuRIL+LoRA backbone producing projection z_i and trust logit s_i.
- Losses: (1) factual-Jaccard-reweighted supervised InfoNCE; (2) consensus-deviation penalty.
- Source-level aggregation τ_s = σ(mean logit).
- Identification argument: consensus loss gates the update by current trust logit, so a low-trust source's drift does not push itself high (avoids fixed-point collapse).
### E. Cleavage + hostility heads
Multi-label BCE with distant supervision from Indian cleavage lexicon; hostility MSE conditioned on cleavage embedding.
### F. Friction aggregator + temporal memory
Exponential-kernel memory with learnable per-cleavage half-life; per-state z-score normalization.
### G. Escalation forecasting head
Window-L MLP with Poisson NLL at horizons 1/2/4.

## V. Data
GDELT 2.0 events + GKG filtered to India, 2015-01-01 → 2025-12-31.
ACLED India (Battles, Protests, Riots, Violence against civilians, Explosions).
Source corpus: ~200–500k articles sampled via GDELT mention URLs, multilingual (en, hi, bn, ta, mr, te). Sampling protocol, pre-processing, and deduplication.

## VI. Experiments
### A. Setup
Train cutoff 2023-06-30; val 2024-06-30; test 2024-07-01 – 2025-12-31.
Hardware: single A100 (Stage A), CPU (Stage B).
Hyperparameters per config.yaml.
### B. Metrics
FSA (Spearman), EEP (F1 at 80th percentile), LTROC (AUC at 90th percentile + lead), Trust-Validation against MBFC.
### C. Main results — Table I
Full system vs EventCount, GoldsteinMean, SentimentOnly, ARIMA, at h ∈ {1, 2, 4}.
### D. Ablation — Table II
full vs no-trust vs no-text vs no-events vs no-graph vs no-memory.
### E. Trust-score validation — Fig. 2
τ_s correlation with MBFC English + human-labelled Indic subset; distribution plots by source-language group.
### F. Case studies — Figs. 3–5
2020 Delhi (communal), 2020–21 farmers' protests (economic + centre-state), 2023 Manipur (communal + linguistic + centre-state).
### G. Error analysis
Where does TRUST-CFE underperform? Low-press-coverage states; cleavages with brittle lexicon coverage.

## VII. Discussion
Why endogenous trust works: diversity of sources covering the same event creates a consensus signal that identifies systematic deviation; our gating prevents trivial solutions.
Limitations: no social-media layer; cleavage labelling relies on a seed lexicon; causal claims excluded by design.

## VIII. Ethical Considerations
Dual-use risk of tension estimation; our release policy; policy-maker guidance; bias audit against state-level press-freedom disparities; explicit declaration that TRUST-CFE is a monitoring tool, not an intervention planner.

## IX. Conclusion
TRUST-CFE delivers a continuous, interpretable, endogenously trust-aware friction estimator for India that outperforms established baselines. Future work: extend to hypergraph narrative contagion (STH-FRICTION) and to counterfactual simulation (CASCADE).

## References
~40 references across ViEWS, GDELT, MuRIL, LoRA, TGN, ACLED, contrastive learning (SimCLR/SupCon), and Indian CSS case studies.

## Figures planned
1. Architecture block diagram.
2. τ_s vs MBFC scatter.
3. Maharashtra friction timeline with cleavage stack (2020–24).
4. India-wide state choropleth for Manipur-crisis week.
5. Lead-time ROC at h=2.
6. Ablation bar chart (FSA delta from full).

## Tables planned
I — Main results (FSA / EEP / LTROC across baselines × horizons).
II — Ablation (delta from full across metrics).
III — Top-10 highest-τ and lowest-τ sources with language tag.
