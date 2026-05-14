"""Logit-lens: project each layer's residual through the model's unembed.

Reference: nostalgebraist, 'interpreting GPT: the logit lens' (2020).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.models.capture import LogitLensTop

if TYPE_CHECKING:
    pass


def logit_lens_topk(
    model: Any,
    residuals: list[Any],
    k: int = 5,
    tokenizer: Any | None = None,
) -> list[list[LogitLensTop]]:
    """For each layer's residual ([T, D]), return per-position top-k predicted tokens.

    Output shape: `result[layer][position] -> LogitLensTop(tokens, probs)`.

    We apply the model's final LayerNorm before the unembed (`lm_head`), matching
    the standard logit-lens recipe. The layer-0 entry uses the embedding output;
    the final-layer entry should match the actual model logits up to numeric drift.
    """
    import torch

    ln_f = _final_layernorm(model)
    lm_head = _unembed(model)

    out: list[list[LogitLensTop]] = []
    for layer_idx, resid in enumerate(residuals):
        if resid is None:
            out.append([])
            continue
        with torch.no_grad():
            h = ln_f(resid) if ln_f is not None else resid
            logits = lm_head(h)  # [T, V]
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=k, dim=-1)
        T = resid.shape[0]
        per_pos: list[LogitLensTop] = []
        for pos in range(T):
            ids = topk.indices[pos].tolist()
            vals = topk.values[pos].tolist()
            toks = [tokenizer.decode([int(i)]) if tokenizer else str(i) for i in ids]
            per_pos.append(LogitLensTop(layer=layer_idx, tokens=toks, probs=[float(v) for v in vals]))
        out.append(per_pos)
    return out


def _final_layernorm(model: Any) -> Any | None:
    """Locate the model's final LayerNorm (or RMSNorm) for the logit-lens projection."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


def _unembed(model: Any) -> Any:
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise NotImplementedError("no lm_head found on model")
