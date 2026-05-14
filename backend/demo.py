"""Generate a deterministic demo-mode capture for the frontend.

When no model is loaded (Spaces cold-start, --lazy mode, before warm completes),
the UI loads this JSON and animates it so the visitor sees something within
~50ms. The data is *plausible*, not real — labels indicate that explicitly.

Run:
    python -m backend.demo > frontend/demo_capture.json
"""

from __future__ import annotations

import json
import random
import sys
from typing import Any

# A short, dramatic prompt + a deterministic completion that ends with " dog".
PROMPT_TOKENS = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy"]
GENERATED_TOKENS = [" dog", ".", " The", " dog", " barked", " loudly", " at", " the", " mailman", "."]
ALL_TOKENS = PROMPT_TOKENS + GENERATED_TOKENS

N_LAYERS = 12
N_FEATURES_BUNDLE = 24576  # gpt2-small-res-jb SAE size

# A handful of "themed" features we lean on so patterns are legible in the demo.
# These IDs are arbitrary; the bundled feature_labels.json contains real
# auto-labels for a subset of them where present.
THEMED_FEATURES = {
    # (layer, feature_id, base_activation, fires_on_predicate)
    "noun-start": (3, 8123, 3.4, lambda t: t.strip()[:1].isalpha() and t.strip()[:1].isupper()),
    "animal-noun": (8, 1024, 4.1, lambda t: t.strip().lower() in {"fox", "dog", "cat", "mailman"}),
    "post-period": (6, 12, 3.6, lambda t: t.startswith(" ") and t[1:2].isupper()),
    "verb-action": (7, 4501, 2.9, lambda t: t.strip().lower() in {"jumps", "barked"}),
    "adjective": (5, 17_300, 2.6, lambda t: t.strip().lower() in {"quick", "brown", "lazy", "loudly"}),
    "function-word": (2, 99, 2.2, lambda t: t.strip().lower() in {"the", "over", "at", "."}),
    "discourse-end": (11, 23_800, 3.2, lambda t: t.strip() in {".", ","}),
    "fox-specific": (9, 11_400, 5.1, lambda t: t.strip().lower() == "fox"),
    "dog-specific": (10, 22_111, 5.6, lambda t: t.strip().lower() == "dog"),
}


def feature_id_for(layer: int, slot: int) -> int:
    """Stable but synthetic id for non-themed features."""
    return (layer * 31337 + slot * 977) % N_FEATURES_BUNDLE


def top_features_for_token(text: str, rng: random.Random) -> list[dict[str, Any]]:
    hits: list[tuple[int, int, float]] = []
    for _name, (layer, fid, base, pred) in THEMED_FEATURES.items():
        if pred(text):
            jitter = rng.uniform(-0.2, 0.2)
            hits.append((layer, fid, max(0.1, base + jitter)))
    # Background features so the heatmap isn't sparse.
    for slot in range(28):
        layer = rng.randint(0, N_LAYERS - 1)
        fid = feature_id_for(layer, slot + rng.randint(0, 7))
        act = max(0.05, rng.gammavariate(2.0, 0.6))
        hits.append((layer, fid, act))
    hits.sort(key=lambda h: -h[2])
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for layer, fid, act in hits:
        key = (layer, fid)
        if key in seen:
            continue
        seen.add(key)
        out.append({"layer": layer, "feature": fid, "act": round(act, 3)})
        if len(out) >= 32:
            break
    return out


def logits_per_layer_for(position: int, text: str, ends_with_dog: bool, rng: random.Random) -> list[dict[str, Any]]:
    """Synthesize a logit-lens trajectory that 'develops' toward the final token."""
    # Last-position case: we should converge to " dog" at the final layer
    # when the prompt asks for the canonical completion.
    target = " dog" if ends_with_dog else text or " the"
    layers: list[dict[str, Any]] = []
    for L in range(N_LAYERS + 1):
        progress = L / N_LAYERS  # 0..1
        # Top-1 prob rises from ~0.05 at layer 0 to ~0.4 at the final layer.
        top_p = 0.05 + 0.35 * progress + rng.uniform(-0.02, 0.02)
        # Pick distractors that get displaced as we move up the layers.
        distractors = [" cat", " man", " kid", " thing", " sky", " day", " road", " house"]
        rng.shuffle(distractors)
        probs = [top_p, top_p * 0.5, top_p * 0.3, top_p * 0.2, top_p * 0.12]
        probs = [round(max(0.005, p), 4) for p in probs]
        tokens = [target] + distractors[:4]
        # Early layers: target is not yet on top.
        if progress < 0.4:
            tokens = distractors[:5]
            probs = [round(max(0.005, p * 0.7), 4) for p in probs]
        layers.append({"layer": L, "tokens": tokens, "probs": probs})
    return layers


def attn_top_per_layer(position: int, rng: random.Random) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for _L in range(N_LAYERS):
        entries = []
        for _ in range(3):
            head = rng.randint(0, 11)
            src = rng.randint(0, max(0, position))
            weight = round(rng.uniform(0.1, 0.9), 3)
            entries.append({"head": head, "src": src, "weight": weight})
        entries.sort(key=lambda e: -e["weight"])
        out.append(entries)
    return out


def build() -> dict[str, Any]:
    rng = random.Random(42)
    tokens_out: list[dict[str, Any]] = []
    for pos, text in enumerate(ALL_TOKENS):
        is_prompt_end = (pos == len(PROMPT_TOKENS) - 1)
        ends_with_dog = is_prompt_end or text == " dog"
        tokens_out.append(
            {
                "position": pos,
                "token_id": (pos * 173 + 7) % 50257,
                "text": text,
                "top_features": top_features_for_token(text, rng),
                "logits_per_layer": logits_per_layer_for(pos, text, ends_with_dog, rng),
                "attn_top_per_layer": attn_top_per_layer(pos, rng),
            }
        )
    return {
        "meta": {
            "model": "gpt2-small",
            "n_layers": N_LAYERS,
            "sae_loaded": True,
            "notes": [
                "Demo capture — deterministic synthetic activations for the static demo. "
                "Switch to live mode by running `make run` locally with a model loaded."
            ],
        },
        "tokens": tokens_out,
        "prompt_length": len(PROMPT_TOKENS),
    }


def main() -> int:
    json.dump(build(), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
