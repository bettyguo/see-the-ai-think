"""Feature labels with explicit trust tiers.

Three tiers (matches the UI badge system from PLANNING/00_think.md §5):

  MEASURED   — direct activation value, no interpretation. Never speculative.
  SOURCED    — a label backed by a paper or human-labeled artifact, with citation.
  AUTO-LABEL — automatic interpretation (e.g. Neuronpedia auto-interp). Visibly marked speculative.

We ship a tiny bundled snapshot for a handful of GPT-2 Small features as a
proof of concept. The real bundle is pulled from a release-attached JSON at
warm time; if absent, the snapshot here is the fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from backend.config import DATA_DIR, PROJECT_ROOT

Tier = Literal["MEASURED", "SOURCED", "AUTO-LABEL"]


@dataclass(frozen=True)
class FeatureLabel:
    text: str
    tier: Tier
    source: str  # short citation string, displayed verbatim in the UI


# Inline fallback. Keep short — the real snapshot lives in data/feature_labels.json.
_BUNDLED: dict[str, FeatureLabel] = {
    # gpt2-small/L6/F12: a well-known "first token of word after a period" feature
    # (we mark it auto-label because the canonical attribution is from
    # Neuronpedia's auto-interp, not a peer-reviewed source).
    "gpt2-small:6:12": FeatureLabel(
        text="activates on the first token of a word that follows a sentence-ending period",
        tier="AUTO-LABEL",
        source="Neuronpedia auto-interp (gpt2-small-res-jb)",
    ),
}


def load_labels() -> dict[str, FeatureLabel]:
    """Load labels from data/ if available, falling back to the bundled snapshot."""
    path = DATA_DIR / "feature_labels.json"
    if not path.exists():
        # Also check the in-package data directory (when installed from a wheel).
        alt = PROJECT_ROOT / "backend" / "data" / "feature_labels.json"
        if alt.exists():
            path = alt
        else:
            return dict(_BUNDLED)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_BUNDLED)

    out: dict[str, FeatureLabel] = dict(_BUNDLED)
    for key, entry in raw.items():
        tier = entry.get("tier")
        if tier not in ("MEASURED", "SOURCED", "AUTO-LABEL"):
            continue
        out[key] = FeatureLabel(
            text=str(entry.get("text", "")),
            tier=tier,
            source=str(entry.get("source", "")),
        )
    return out


def lookup(labels: dict[str, FeatureLabel], model: str, layer: int, feature: int) -> FeatureLabel | None:
    return labels.get(f"{model}:{layer}:{feature}")
