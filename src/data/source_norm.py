"""Source-domain canonicalization.

GDELT's `SourceCommonName` shows up in many surface forms that mean the same
outlet (`NDTV.com`, `ndtv.com`, `www.ndtv.com`, `NDTV`). Leaving these as
distinct source IDs dilutes every per-source τ estimate (audit §1.1).

Apply `normalize_source()` at ingest time wherever `articles["source_domain"]`
is populated.
"""
from __future__ import annotations

import re

__all__ = ["normalize_source", "SOURCE_ALIASES"]


# Known alias -> canonical mapping. Keep the key lowercased; runtime lookup
# is case-insensitive. Extend as you discover more collisions.
SOURCE_ALIASES: dict[str, str] = {
    "ndtv": "ndtv.com",
    "timesofindia": "timesofindia.indiatimes.com",
    "toi": "timesofindia.indiatimes.com",
    "thehindu": "thehindu.com",
    "indianexpress": "indianexpress.com",
    "indian express": "indianexpress.com",
    "hindustantimes": "hindustantimes.com",
    "ht": "hindustantimes.com",
    "the wire": "thewire.in",
    "wire": "thewire.in",
    "scroll": "scroll.in",
    "firstpost": "firstpost.com",
    "news18": "news18.com",
    "india today": "indiatoday.in",
    "indiatoday": "indiatoday.in",
    "the print": "theprint.in",
    "theprint": "theprint.in",
    "republic": "republicworld.com",
    "republicworld": "republicworld.com",
    "zee news": "zeenews.india.com",
    "zeenews": "zeenews.india.com",
}


_WWW = re.compile(r"^www\.", re.IGNORECASE)


def normalize_source(name: str | None) -> str:
    """Collapse `NDTV.com`, `www.NDTV.com`, `NDTV ` etc to a single id.

    Rules, in order:
      1. Strip whitespace and lowercase.
      2. Drop any leading `www.`.
      3. Look up in `SOURCE_ALIASES` — if the input matches an alias key
         (after lowercasing), return the canonical value.
      4. Otherwise return the lowercased, www-stripped string as-is.

    Returns an empty string for None / empty input so downstream groupbys
    don't treat NaN separately.
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    if not s:
        return ""
    s = _WWW.sub("", s)
    if s in SOURCE_ALIASES:
        return SOURCE_ALIASES[s]
    # "The Hindu" / "Times of India" style -- try a space-collapsed form.
    s_nospace = s.replace(" ", "")
    if s_nospace in SOURCE_ALIASES:
        return SOURCE_ALIASES[s_nospace]
    # Also look up the alias-key form without a TLD (`ndtv.com` -> `ndtv`).
    bare = s_nospace.split(".")[0]
    if bare in SOURCE_ALIASES:
        return SOURCE_ALIASES[bare]
    return s
