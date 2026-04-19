"""Cluster-aware batch samplers.

The default random sampler over ~10^5 articles with ~10^3 event clusters
almost never puts two articles from the same cluster in the same batch of 32.
`agreement_contrastive_loss` then has no positives and silently returns 0
(see audit §1.2).

`ClusterBatchSampler` draws batches SupCon-style: m clusters x k articles per
cluster => guaranteed ~m * C(k, 2) positive pairs per batch.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, Sequence


class ClusterBatchSampler:
    """SupCon-style batch sampler.

    Args:
        cluster_ids:   per-article cluster id (int). Use -1 for singletons.
        m_per_batch:   number of clusters to sample per batch.
        k_per_cluster: number of articles to sample from each chosen cluster.
        n_batches:     batches per epoch (samplers are length-fixed; the
                       dataset itself is effectively infinite wrt combinations).
        seed:          optional RNG seed for deterministic sampling.
        drop_small:    if True, drop clusters with fewer than k_per_cluster
                       members (otherwise they'd be upsampled with replacement).
        allow_replacement_if_empty:
                       if the usable cluster pool is smaller than m_per_batch,
                       sample clusters with replacement instead of erroring.

    Yields:
        A flat list of article indices of length m_per_batch * k_per_cluster.
    """

    def __init__(
        self,
        cluster_ids: Sequence[int],
        m_per_batch: int = 8,
        k_per_cluster: int = 4,
        n_batches: int = 1000,
        seed: int | None = None,
        drop_small: bool = True,
        allow_replacement_if_empty: bool = True,
    ) -> None:
        if m_per_batch < 1 or k_per_cluster < 2:
            raise ValueError("need m>=1 and k>=2 for contrastive positives")
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(cluster_ids):
            if c is None or int(c) < 0:
                continue
            buckets[int(c)].append(i)
        if drop_small:
            buckets = {c: v for c, v in buckets.items() if len(v) >= k_per_cluster}
        if not buckets:
            raise ValueError(
                f"no cluster has >= {k_per_cluster} members; "
                f"cannot form SupCon batches"
            )
        self.buckets = buckets
        self._cluster_keys = list(buckets.keys())
        self.m = m_per_batch
        self.k = k_per_cluster
        self.n = n_batches
        self.allow_replacement = allow_replacement_if_empty
        self._rng = random.Random(seed)

    @property
    def batch_size(self) -> int:
        return self.m * self.k

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.n):
            yield self._draw_batch()

    def __len__(self) -> int:
        return self.n

    def _draw_batch(self) -> list[int]:
        keys = self._cluster_keys
        if len(keys) >= self.m:
            chosen = self._rng.sample(keys, self.m)
        else:
            if not self.allow_replacement:
                raise RuntimeError(
                    f"only {len(keys)} usable clusters; cannot draw m={self.m}"
                )
            chosen = [self._rng.choice(keys) for _ in range(self.m)]
        out: list[int] = []
        for c in chosen:
            pool = self.buckets[c]
            if len(pool) >= self.k:
                out.extend(self._rng.sample(pool, self.k))
            else:
                # shouldn't happen when drop_small=True, but guard anyway
                out.extend(self._rng.choices(pool, k=self.k))
        return out


def positives_per_batch(cluster_ids_in_batch: Sequence[int]) -> int:
    """Count (i, j) pairs with i<j that share a non-negative cluster id.

    Useful in tests / logs to verify the sampler actually produces positives.
    """
    counts: dict[int, int] = defaultdict(int)
    for c in cluster_ids_in_batch:
        if c is None or int(c) < 0:
            continue
        counts[int(c)] += 1
    total = 0
    for n in counts.values():
        total += n * (n - 1) // 2
    return total
