"""ClusterBatchSampler guarantees positive pairs.

Guards audit §1.2: at full scale, a random batch over ~10^5 articles and
~10^3 clusters produces ~0 positives per batch of 32. The SupCon sampler
must produce >= m * C(k, 2) positives per batch.
"""
from __future__ import annotations

import random

from src.training.samplers import ClusterBatchSampler, positives_per_batch


def test_sampler_always_has_positives():
    rng = random.Random(42)
    n_articles = 5_000
    n_clusters = 800
    cluster_ids = [rng.randrange(n_clusters) for _ in range(n_articles)]
    # Ensure every cluster has at least 4 members so the sampler has material.
    # (In the real dataset small clusters are dropped by drop_small=True.)
    for c in range(n_clusters):
        # pad with 4 dummy indices per cluster
        for _ in range(4):
            cluster_ids.append(c)

    m, k = 8, 4
    sampler = ClusterBatchSampler(
        cluster_ids, m_per_batch=m, k_per_cluster=k,
        n_batches=40, seed=0,
    )
    min_expected = m * (k * (k - 1) // 2)       # = 8 * 6 = 48
    for batch in sampler:
        assert len(batch) == m * k
        batch_clusters = [cluster_ids[i] for i in batch]
        pos = positives_per_batch(batch_clusters)
        assert pos >= min_expected, (
            f"expected >={min_expected} positives per batch, got {pos}"
        )


def test_sampler_drops_small_clusters():
    # Cluster 0 has 2 members, cluster 1 has 10. k=4, so only cluster 1 is usable.
    cluster_ids = [0, 0] + [1] * 10
    sampler = ClusterBatchSampler(
        cluster_ids, m_per_batch=1, k_per_cluster=4, n_batches=5,
        seed=0, allow_replacement_if_empty=True,
    )
    for batch in sampler:
        assert all(cluster_ids[i] == 1 for i in batch)
