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


# Inline fallback. PLACEHOLDER — replaced at warm time by a real Neuronpedia
# snapshot pulled into data/feature_labels.json. Source string is honest about
# its origin per docs/HONESTY.md; we never ship "real-looking" labels we made up.
_BUNDLED: dict[str, FeatureLabel] = {
    "gpt2-small:6:12": FeatureLabel(
        text="placeholder label: in real use this would be a Neuronpedia auto-interp string for L6/F12",
        tier="AUTO-LABEL",
        source="placeholder — replaced at warm time by Neuronpedia snapshot",
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
        # Skip non-label keys like `_doc` (informational header at the top of the file).
        if not isinstance(entry, dict):
            continue
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
