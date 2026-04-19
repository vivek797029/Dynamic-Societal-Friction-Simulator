"""M3. Event clustering — group mentions / articles that describe the same incident.

Used by the trust learner: positive pairs are articles in the same cluster,
negatives are articles in different clusters.

Cluster construction:
  1. Pre-filter articles to (state, date-window) buckets.
  2. Compute MuRIL CLS embeddings (cached).
  3. Build within-bucket cosine-similarity graph with threshold τ_cos.
  4. Require at least `min_entity_overlap` shared named entities per edge.
  5. Connected components → event_cluster_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class ClusteringConfig:
    spatial_days: int = 2          # ± days window
    cosine_threshold: float = 0.75
    min_entity_overlap: int = 1
    min_cluster_size: int = 3
    max_cluster_size: int = 30


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _build_cluster_edges(emb: np.ndarray,
                         ents: list[set],
                         cfg: ClusteringConfig) -> list[tuple[int, int]]:
    """Within a single (state, date-window) bucket, emit edges passing both tests."""
    n = len(emb)
    if n < 2:
        return []
    # Normalize rows for cosine.
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    normed = emb / norms
    sims = normed @ normed.T  # [n,n]
    edges: list[tuple[int, int]] = []
    # Upper triangle only.
    ii, jj = np.triu_indices(n, k=1)
    mask = sims[ii, jj] >= cfg.cosine_threshold
    for i, j in zip(ii[mask], jj[mask]):
        if len(ents[i] & ents[j]) >= cfg.min_entity_overlap:
            edges.append((int(i), int(j)))
    return edges


def _connected_components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    # Re-index to 0..C-1.
    _, inv = np.unique(roots, return_inverse=True)
    return inv.astype(np.int64)


def cluster_events(articles: pd.DataFrame,
                   embeddings: np.ndarray,
                   entities: list[set],
                   cfg: ClusteringConfig | None = None) -> pd.Series:
    """Return a Series aligned with `articles` giving a global event_cluster_id.

    `articles` must contain columns: article_id, state, date.
    `embeddings[i]` is the MuRIL CLS for articles.iloc[i].
    `entities[i]` is the set of named entities for articles.iloc[i].
    """
    cfg = cfg or ClusteringConfig()
    assert len(articles) == len(embeddings) == len(entities)
    articles = articles.reset_index(drop=True)
    date = pd.to_datetime(articles["date"])
    day = (date - date.min()).dt.days.to_numpy()

    ids = np.full(len(articles), -1, dtype=np.int64)
    next_id = 0

    for state, idxs_state in tqdm(articles.groupby("state").groups.items(), desc="cluster by state"):
        idxs_state = np.array(list(idxs_state), dtype=np.int64)
        d_state = day[idxs_state]
        # Sort by day for efficient windowing.
        order = np.argsort(d_state)
        idxs_state = idxs_state[order]
        d_state = d_state[order]

        i = 0
        while i < len(idxs_state):
            j = i
            while j < len(idxs_state) and d_state[j] - d_state[i] <= cfg.spatial_days:
                j += 1
            bucket = idxs_state[i:j]
            if len(bucket) >= 2:
                emb_b = embeddings[bucket]
                ent_b = [entities[k] for k in bucket]
                edges = _build_cluster_edges(emb_b, ent_b, cfg)
                comps = _connected_components(len(bucket), edges)
                # Filter cluster sizes.
                _, counts = np.unique(comps, return_counts=True)
                size_of = counts[comps]
                keep = (size_of >= cfg.min_cluster_size) & (size_of <= cfg.max_cluster_size)
                for k_local, orig in enumerate(bucket):
                    if keep[k_local]:
                        ids[orig] = next_id + int(comps[k_local])
                next_id += int(comps.max()) + 1 if len(bucket) else 0
            i = j

    return pd.Series(ids, index=articles.index, name="event_cluster_id")
