"""Public surface for model metadata used by routes/__init__."""

from __future__ import annotations

from backend.config import DEFAULT_MODEL, MODELS


def list_models() -> list[dict]:
    return [
        {
            "name": s.name,
            "hf_id": s.hf_id,
            "n_layers": s.n_layers,
            "d_model": s.d_model,
            "description": s.description,
            "has_sae": s.sae_release is not None,
            "is_default": s.name == DEFAULT_MODEL,
        }
        for s in MODELS.values()
    ]
