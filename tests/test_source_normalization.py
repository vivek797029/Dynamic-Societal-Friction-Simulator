"""Source-domain canonicalization collapses known surface forms.

Guards audit §1.1: `NDTV.com`, `ndtv.com`, `www.ndtv.com`, `NDTV` must all
map to a single domain id.
"""
from __future__ import annotations

from src.data.source_norm import normalize_source


def test_ndtv_aliases_collapse():
    variants = ["NDTV.com", "ndtv.com", "www.NDTV.com", "NDTV", "  ndtv  "]
    canon = {normalize_source(v) for v in variants}
    assert canon == {"ndtv.com"}, f"aliases didn't collapse: {canon}"


def test_other_alias_clusters():
    assert normalize_source("The Hindu") == normalize_source("thehindu.com")
    assert normalize_source("Times of India") == normalize_source(
        "www.timesofindia.indiatimes.com"
    )
    assert normalize_source("The Wire") == normalize_source("thewire.in")
    assert normalize_source("HT") == normalize_source("hindustantimes.com")


def test_unknown_source_passthrough_and_lowercased():
    assert normalize_source("SomeRegionalBlog.in") == "someregionalblog.in"
    assert normalize_source("www.obscuresite.co.in") == "obscuresite.co.in"


def test_empty_and_none():
    assert normalize_source(None) == ""
    assert normalize_source("") == ""
    assert normalize_source("   ") == ""
