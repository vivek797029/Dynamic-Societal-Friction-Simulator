"""Word-boundary weak labels (audit §1.1).

Substring-only matching produced false positives like `bjp` matching `bjpa`,
`aap` matching `aapta`, `obc` matching `obcuring`. Guard against those.
"""
from __future__ import annotations

from src.models.cleavage_classifier import CLEAVAGES, weak_label


def _idx(name):
    return CLEAVAGES.index(name)


def test_no_bjpa_false_positive():
    y = weak_label("the bjpa corporation is unrelated")
    assert y[_idx("political_party")] == 0.0


def test_bjp_true_positive():
    y = weak_label("the BJP won the election")
    assert y[_idx("political_party")] == 1.0


def test_no_obcuring_false_positive():
    # pretend-word with 'obc' inside; shouldn't trigger the caste cleavage.
    y = weak_label("the obcuring facts were unclear")
    assert y[_idx("caste")] == 0.0


def test_obc_true_positive():
    y = weak_label("the OBC reservation policy was debated")
    assert y[_idx("caste")] == 1.0


def test_no_aapta_false_positive():
    y = weak_label("aapta was a mythical character")
    assert y[_idx("political_party")] == 0.0


def test_empty_text():
    y = weak_label("")
    assert y.sum() == 0.0


def test_multi_label():
    y = weak_label("BJP leaders discussed OBC reservation at the mosque")
    assert y[_idx("political_party")] == 1.0
    assert y[_idx("caste")] == 1.0
    assert y[_idx("communal")] == 1.0
