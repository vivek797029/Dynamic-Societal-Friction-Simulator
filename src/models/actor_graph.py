"""DEPRECATED — removed in v1 cleanup.

Signed-actor-graph relational strain R_k was an optional ablation channel.
It was never required by the friction model, and integrating it cleanly on
GDELT actor codes turned out to be fragile (sparse per-week graphs, many
single-node components, slow eigensolves). The aggregator now operates on
(E, T) only; if we want R back for ablation it should be a separate v2 study.

This file is a tombstone so stale imports fail loudly.
"""
raise ImportError(
    "src.models.actor_graph was removed in v1. FrictionAggregator now "
    "operates on (E, T) only; relational strain is not part of the v1 scope."
)
