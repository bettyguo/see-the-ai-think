"""SAE loading + top-K feature encoding.

We use SAELens when available, but the *interface* this module exposes is
deliberately minimal — a single `SAEBundle` opaque object plus an
`encode_topk(bundle, layer, residual, k)` function. This lets callers stay
agnostic to the SAE backend, and lets tests substitute a tiny in-memory bundle
without pulling sae-lens into the test deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.config import ModelSpec

if TYPE_CHECKING:
    pass


@dataclass
class SAEBundle:
    """A per-layer dict of SAE encoders. Loaded lazily."""

    saes: dict[int, Any] = field(default_factory=dict)
    is_loaded: bool = False
    source: str = ""

    @classmethod
    def empty(cls) -> SAEBundle:
        return cls(saes={}, is_loaded=False, source="")


def try_load_saes(spec: ModelSpec, device: str = "cpu") -> SAEBundle:
    """Best-effort SAE load. Returns an empty bundle if anything fails.

    Network errors, missing release names, optional-dep absence — all degrade
    gracefully. The caller checks `bundle.is_loaded` and shows a banner in the
    UI if False.
    """
    if spec.sae_release is None:
        return SAEBundle.empty()

    try:
        from sae_lens import SAE
    except Exception:
        return SAEBundle(saes={}, is_loaded=False, source="sae-lens-missing")

    saes: dict[int, Any] = {}
    for layer_idx in spec.sae_layer_ids:
        sae_id = f"blocks.{layer_idx}.hook_resid_pre"
        try:
            sae, _, _ = SAE.from_pretrained(
                release=spec.sae_release,
                sae_id=sae_id,
                device=device,
            )
            sae.eval()
            saes[layer_idx] = sae
        except Exception:
            # Skip layers that fail; the rest of the bundle still works.
            continue

    if not saes:
        return SAEBundle(saes={}, is_loaded=False, source="download-failed")
    return SAEBundle(saes=saes, is_loaded=True, source=spec.sae_release)


def encode_topk(bundle: SAEBundle, layer: int, residual: Any, k: int = 32) -> tuple[Any, Any]:
    """Encode `residual` ([T, D]) through the layer's SAE, return (top_ids, top_acts).

    Both returned tensors have shape [T, k]. Activations below zero are dropped
    by the caller — we return raw top-k here so the caller can decide thresholds.
    """
    import torch

    sae = bundle.saes.get(layer)
    if sae is None:
        # No SAE for this layer — return a zero-width slice the caller treats as empty.
        T = residual.shape[0]
        empty = torch.zeros((T, 0), dtype=torch.long, device=residual.device)
        return empty, empty.float()

    # SAELens SAE accepts [..., d_model] and returns [..., d_sae].
    with torch.no_grad():
        feats = sae.encode(residual)  # [T, F]
    k_eff = min(k, feats.shape[-1])
    topk = torch.topk(feats, k=k_eff, dim=-1)
    return topk.indices, topk.values
