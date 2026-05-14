"""Label + trigger lookup. No torch required."""

from __future__ import annotations

from backend.features.labels import FeatureLabel, load_labels, lookup
from backend.features.triggers import load_triggers
from backend.features.triggers import lookup as lookup_triggers


def test_bundled_labels_have_known_keys():
    labels = load_labels()
    assert "gpt2-small:6:12" in labels
    assert labels["gpt2-small:6:12"].tier in {"MEASURED", "SOURCED", "AUTO-LABEL"}


def test_label_lookup_returns_none_for_unknown():
    labels = load_labels()
    assert lookup(labels, "gpt2-small", 99, 99) is None


def test_label_lookup_returns_hit_for_known():
    labels = load_labels()
    hit = lookup(labels, "gpt2-small", 6, 12)
    assert isinstance(hit, FeatureLabel)
    # Placeholder label text — the bundled label is explicitly a placeholder,
    # and the source string must say so to comply with docs/HONESTY.md.
    assert hit.tier == "AUTO-LABEL"
    assert "placeholder" in hit.source.lower()


def test_triggers_sorted_by_activation():
    trigs = load_triggers()
    results = lookup_triggers(trigs, "gpt2-small", 6, 12, top_n=5)
    acts = [t.activation for t in results]
    assert acts == sorted(acts, reverse=True)


def test_triggers_empty_for_unknown():
    trigs = load_triggers()
    assert lookup_triggers(trigs, "gpt2-small", 99, 99) == []
