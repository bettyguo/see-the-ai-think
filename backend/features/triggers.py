"""Triggering-examples corpus lookup.

A *very* small in-process index: for each known feature, we store a few
(text, activating_token_index, activation) triples. The "real" index is built
at warm time from a small permissively-licensed corpus; for tests and offline
use we ship a tiny inline fallback.

We deliberately keep the interface narrow so swapping in a parquet-backed
index later is a one-file change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from backend.config import DATA_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class TriggerExample:
    text: str
    activating_index: int
    activation: float


_BUNDLED: dict[str, list[TriggerExample]] = {
    "gpt2-small:6:12": [
        TriggerExample("The day was bright. Sunlight streamed through the window.", 5, 3.7),
        TriggerExample("She paused. Then she began to write.", 3, 3.4),
        TriggerExample("It rained heavily. Everyone ran for cover.", 4, 3.1),
    ],
}


def load_triggers() -> dict[str, list[TriggerExample]]:
    path = DATA_DIR / "feature_triggers.json"
    if not path.exists():
        alt = PROJECT_ROOT / "backend" / "data" / "feature_triggers.json"
        if alt.exists():
            path = alt
        else:
            return dict(_BUNDLED)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_BUNDLED)

    out: dict[str, list[TriggerExample]] = dict(_BUNDLED)
    for key, entries in raw.items():
        if not isinstance(entries, list):
            continue
        parsed: list[TriggerExample] = []
        for e in entries:
            try:
                parsed.append(
                    TriggerExample(
                        text=str(e["text"]),
                        activating_index=int(e["activating_index"]),
                        activation=float(e["activation"]),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        if parsed:
            out[key] = parsed
    return out


def lookup(
    triggers: dict[str, list[TriggerExample]],
    model: str,
    layer: int,
    feature: int,
    top_n: int = 10,
) -> list[TriggerExample]:
    found = triggers.get(f"{model}:{layer}:{feature}", [])
    return sorted(found, key=lambda t: -t.activation)[:top_n]
