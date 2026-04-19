"""DEPRECATED — removed in v1 cleanup.

The standalone MuRIL CLS embedder lived here as a helper for event-clustering
notebooks. It is no longer part of the pipeline: `src/models/article_encoder.py`
is the trained encoder, and `src/data/event_clusters.py` handles its own
tokenization when needed. Keeping this module would just mean two copies of
the same backbone with no caller.

This file is intentionally left as a tombstone so stale imports fail loudly
rather than silently resurrecting dead code.
"""
raise ImportError(
    "src.models.embed was removed in v1. Use src.models.article_encoder for "
    "the trained MuRIL+LoRA encoder, or inline tokenization for event clustering."
)
