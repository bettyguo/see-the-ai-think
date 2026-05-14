"""Paths, defaults, and the model registry surface."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_DIR = Path(os.environ.get("STAT_DATA_DIR", PROJECT_ROOT / "data"))

CACHE_ROOT = Path(
    os.environ.get(
        "STAT_CACHE_DIR",
        Path.home() / ".cache" / "see-the-ai-think",
    )
)


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a supported model."""

    name: str
    hf_id: str
    n_layers: int
    d_model: int
    description: str
    sae_release: str | None = None
    sae_layer_ids: tuple[int, ...] = field(default_factory=tuple)


# The registry is intentionally tiny on launch — one solid default beats five flaky models.
MODELS: dict[str, ModelSpec] = {
    "gpt2-small": ModelSpec(
        name="gpt2-small",
        hf_id="gpt2",
        n_layers=12,
        d_model=768,
        description="GPT-2 Small (124M). The mech-interp canon. Ships with Joseph Bloom's residual-stream SAEs.",
        sae_release="gpt2-small-res-jb",
        sae_layer_ids=tuple(range(12)),
    ),
}

DEFAULT_MODEL = "gpt2-small"


@dataclass
class RuntimeConfig:
    """Runtime-only configuration (mutable, set at startup)."""

    model_name: str = DEFAULT_MODEL
    device: str = "cpu"
    max_new_tokens: int = 48
    top_k_features: int = 32
    top_k_logits: int = 5
    sae_enabled: bool = True


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"unknown model: {name}. known: {sorted(MODELS)}")
    return MODELS[name]
